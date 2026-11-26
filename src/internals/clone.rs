//! Transactional cloning of entity trees whose components may contain
//! [`Entity`](crate::world::Entity) references.
//!
//! A regular component clone has no way of knowing which [`Entity`] references
//! point inside the structure being cloned and which point outside of it. This
//! module performs the clone in two phases so that the distinction is made
//! exactly once, from the set of entities which belong to the tree:
//!
//! 1. Every component of every tree entity is cloned into an off-world staging
//!    area. While cloning takes place, a mapping of tree entity IDs to their
//!    fresh clone IDs is installed as the thread-local context used by
//!    [`Entity`]'s [`Clone`] implementation. References to tree entities are
//!    therefore rewritten to the corresponding clones, while references to
//!    entities outside the tree are left untouched and keep pointing at their
//!    original targets.
//! 2. Only after every component has been cloned successfully are the clones
//!    committed to the destination world in a single step. If any component
//!    clone fails, the staging area is dropped and the destination world is not
//!    modified at all: no entities, archetypes or components are committed.
//!
//! Component types which are not registered with the [`CloneRegistry`] are
//! omitted from the clones, mirroring the behavior of the
//! [`Duplicate`](crate::world::Duplicate) merger.

use std::{any::Any, collections::HashMap, error::Error, marker::PhantomData};

use super::{
    entity::{Allocate, Entity, EntityHasher, ID_CLONE_MAPPINGS},
    insert::{ArchetypeSource, ArchetypeWriter},
    query::filter::{FilterResult, LayoutFilter},
    storage::{
        archetype::{Archetype, ArchetypeIndex, EntityLayout},
        component::{Component, ComponentTypeId},
        ComponentStorage, Components, UnknownComponentStorage,
    },
    world::World,
};

/// A boxed, thread-safe error produced by a fallible component clone function.
pub type BoxedCloneError = Box<dyn Error + Send + Sync + 'static>;

/// The error returned when an entity tree cannot be cloned.
#[derive(thiserror::Error, Debug)]
pub enum CloneError {
    /// One of the entities supplied as a tree root, or reached through an edge,
    /// does not exist in the source world.
    #[error("entity {0:?} does not exist in the source world")]
    EntityNotFound(Entity),

    /// A component clone function returned an error. Because the clone is
    /// transactional, no part of the tree has been committed when this is
    /// returned.
    #[error("failed to clone a component of type {component_type}: {source}")]
    ComponentCloneFailed {
        /// The type of the component whose clone failed.
        component_type: ComponentTypeId,
        /// The error produced by the clone function.
        #[source]
        source: BoxedCloneError,
    },
}

/// Describes how a single component type is copied or cloned, including how
/// [`Entity`] references found inside it are rewritten.
///
/// The cloner is fully typed; the surrounding registry performs the type
/// erasure.
trait Cloner: Send + Sync {
    /// Constructs an empty, type-erased staging `Vec<T>` for this cloner's
    /// component type.
    fn empty_buffer(&self) -> Box<dyn Any + Send + Sync>;

    /// Clones the component at `index` within `src_arch` and appends the clone
    /// to `staged`, which is a `Vec<T>` of the cloner's component type.
    ///
    /// # Safety
    /// `staged` must downcast to `Vec<T>` where `T` is the component type this
    /// cloner was registered for.
    unsafe fn stage(
        &mut self,
        src_arch: &Archetype,
        index: usize,
        src_components: &Components,
        staged: &mut dyn Any,
    ) -> Result<(), BoxedCloneError>;

    /// Commits every staged component into the destination archetype.
    ///
    /// # Safety
    /// `staged` must contain exactly the components produced by [`stage`], and
    /// the destination writer must claim the matching component type.
    ///
    /// [`stage`]: Cloner::stage
    unsafe fn write_all(&self, staged: &mut dyn Any, writer: &mut ArchetypeWriter<'_>);
}

struct CopyCloner<T: Component + Copy>(PhantomData<T>);
struct CloneCloner<T: Component + Clone>(PhantomData<T>);
struct FallibleCloner<T: Component>(Box<dyn FnMut(&T) -> Result<T, BoxedCloneError> + Send + Sync>);

/// A boxed, type-erased component cloner registered for one component type.
struct Registered {
    cloner: Box<dyn Cloner>,
    /// Constructs an empty storage of the registered component type. Retained
    /// as a function pointer so that destination layouts can be built without
    /// borrowing the cloner.
    storage_constructor: fn() -> Box<dyn UnknownComponentStorage>,
}

impl Registered {
    fn new<T: Component>(cloner: Box<dyn Cloner>) -> Self {
        Self {
            cloner,
            storage_constructor: || Box::new(T::Storage::default()),
        }
    }
}

/// Describes how each component type encountered while cloning an entity tree
/// should be duplicated.
///
/// Components of unregistered types are skipped. Component types must be
/// registered before calling [`World::clone_tree`].
///
/// # Examples
/// ```
/// # use legion::world::CloneRegistry;
/// # #[derive(Clone, Copy)]
/// # struct Position(f32, f32);
/// let mut registry = CloneRegistry::new();
/// registry.register_copy::<Position>();
/// ```
#[derive(Default)]
pub struct CloneRegistry {
    cloners: HashMap<ComponentTypeId, Registered>,
}

impl CloneRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a `Copy` component, which is duplicated by a raw memory copy.
    pub fn register_copy<T: Component + Copy>(&mut self) {
        self.insert::<T>(Box::new(CopyCloner::<T>(PhantomData)));
    }

    /// Registers a `Clone` component, which is duplicated with its [`Clone`]
    /// implementation.
    ///
    /// When the component contains [`Entity`] references (or is [`Entity`]
    /// itself), references to entities inside the cloned tree are rewritten to
    /// point at the clones, while references outside the tree are preserved.
    pub fn register_clone<T: Component + Clone>(&mut self) {
        self.insert::<T>(Box::new(CloneCloner::<T>(PhantomData)));
    }

    /// Registers a component with a custom, fallible clone function.
    ///
    /// Returning an error from `clone` aborts the whole tree clone: none of the
    /// cloned entities are committed to the destination world. The same entity
    /// reference rewriting rules as [`CloneRegistry::register_clone`] apply
    /// while `clone` runs.
    pub fn register_try_clone<T, E, F>(&mut self, mut clone: F)
    where
        T: Component,
        E: Error + Send + Sync + 'static,
        F: FnMut(&T) -> Result<T, E> + Send + Sync + 'static,
    {
        let clone: Box<dyn FnMut(&T) -> Result<T, BoxedCloneError> + Send + Sync> =
            Box::new(move |component| {
                clone(component).map_err(|error| Box::new(error) as BoxedCloneError)
            });
        self.insert::<T>(Box::new(FallibleCloner::<T>(clone)));
    }

    fn insert<T: Component>(&mut self, cloner: Box<dyn Cloner>) {
        self.cloners
            .insert(ComponentTypeId::of::<T>(), Registered::new::<T>(cloner));
    }
}

impl<T: Component + Copy> Cloner for CopyCloner<T> {
    fn empty_buffer(&self) -> Box<dyn Any + Send + Sync> {
        Box::new(Vec::<T>::new())
    }

    unsafe fn stage(
        &mut self,
        src_arch: &Archetype,
        index: usize,
        src_components: &Components,
        staged: &mut dyn Any,
    ) -> Result<(), BoxedCloneError> {
        let storage = src_components.get_downcast::<T>().unwrap();
        let value = storage.get(src_arch.index()).unwrap().into_slice()[index];
        staged.downcast_mut::<Vec<T>>().unwrap().push(value);
        Ok(())
    }

    unsafe fn write_all(&self, staged: &mut dyn Any, writer: &mut ArchetypeWriter<'_>) {
        write_staged::<T>(staged, writer);
    }
}

impl<T: Component + Clone> Cloner for CloneCloner<T> {
    fn empty_buffer(&self) -> Box<dyn Any + Send + Sync> {
        Box::new(Vec::<T>::new())
    }

    unsafe fn stage(
        &mut self,
        src_arch: &Archetype,
        index: usize,
        src_components: &Components,
        staged: &mut dyn Any,
    ) -> Result<(), BoxedCloneError> {
        let storage = src_components.get_downcast::<T>().unwrap();
        let value = &storage.get(src_arch.index()).unwrap().into_slice()[index];
        // `Entity::clone` consults ID_CLONE_MAPPINGS, rewriting references to
        // tree entities while leaving external references untouched.
        staged.downcast_mut::<Vec<T>>().unwrap().push(value.clone());
        Ok(())
    }

    unsafe fn write_all(&self, staged: &mut dyn Any, writer: &mut ArchetypeWriter<'_>) {
        write_staged::<T>(staged, writer);
    }
}

impl<T: Component> Cloner for FallibleCloner<T> {
    fn empty_buffer(&self) -> Box<dyn Any + Send + Sync> {
        Box::new(Vec::<T>::new())
    }

    unsafe fn stage(
        &mut self,
        src_arch: &Archetype,
        index: usize,
        src_components: &Components,
        staged: &mut dyn Any,
    ) -> Result<(), BoxedCloneError> {
        let storage = src_components.get_downcast::<T>().unwrap();
        let value = &storage.get(src_arch.index()).unwrap().into_slice()[index];
        let cloned = (self.0)(value)?;
        staged.downcast_mut::<Vec<T>>().unwrap().push(cloned);
        Ok(())
    }

    unsafe fn write_all(&self, staged: &mut dyn Any, writer: &mut ArchetypeWriter<'_>) {
        write_staged::<T>(staged, writer);
    }
}

/// Moves every staged component of type `T` into the destination archetype.
///
/// # Safety
/// The values are memcopied into the world storage and forgotten in the source
/// vector, so the vector's allocation is emptied without dropping them twice.
unsafe fn write_staged<T: Component>(staged: &mut dyn Any, writer: &mut ArchetypeWriter<'_>) {
    let staged = staged.downcast_mut::<Vec<T>>().unwrap();
    let mut destination = writer.claim_components::<T>();
    destination.ensure_capacity(staged.len());
    destination.extend_memcopy(staged.as_ptr(), staged.len());
    // ownership of the values has been transferred into the world storage
    staged.set_len(0);
}

/// A group of tree entities which share an identical destination layout and are
/// therefore committed into the same destination archetype.
struct CloneGroup {
    /// `(component type ID, storage constructor)` for every registered
    /// component the clones will carry.
    layout: Vec<(ComponentTypeId, fn() -> Box<dyn UnknownComponentStorage>)>,
    /// `(entity, source archetype, component index within the source archetype)`.
    members: Vec<(Entity, ArchetypeIndex, usize)>,
    /// Per-component-type staging buffers owned exclusively by this group, so
    /// groups never share or truncate each other's staged data. Each value is
    /// a `Vec<T>` for the component type of its key.
    staged: HashMap<ComponentTypeId, Box<dyn Any + Send + Sync>>,
}

impl CloneGroup {
    /// Returns the component type IDs of the destination layout.
    fn layout_ids(&self) -> Vec<ComponentTypeId> {
        self.layout.iter().map(|(type_id, _)| *type_id).collect()
    }
}

/// Restores the previous contents of the thread-local clone mapping when it
/// goes out of scope, including on unwind.
struct MappingGuard {
    previous: HashMap<Entity, Entity, EntityHasher>,
}

impl MappingGuard {
    fn install(mapping: HashMap<Entity, Entity, EntityHasher>) -> Self {
        let previous =
            ID_CLONE_MAPPINGS.with(|cell| std::mem::replace(&mut *cell.borrow_mut(), mapping));
        Self { previous }
    }
}

impl Drop for MappingGuard {
    fn drop(&mut self) {
        ID_CLONE_MAPPINGS.with(|cell| {
            *cell.borrow_mut() = std::mem::take(&mut self.previous);
        });
    }
}

/// A filter which matches, and can build, an archetype containing exactly the
/// given component types.
#[derive(Clone)]
struct ExactLayoutFilter {
    layout: Vec<(ComponentTypeId, fn() -> Box<dyn UnknownComponentStorage>)>,
}

impl LayoutFilter for ExactLayoutFilter {
    fn matches_layout(&self, components: &[ComponentTypeId]) -> FilterResult {
        FilterResult::Match(
            components.len() == self.layout.len()
                && self
                    .layout
                    .iter()
                    .all(|(type_id, _)| components.contains(type_id)),
        )
    }
}

impl ArchetypeSource for ExactLayoutFilter {
    type Filter = Self;

    fn filter(&self) -> Self::Filter {
        self.clone()
    }

    fn layout(&mut self) -> EntityLayout {
        let mut layout = EntityLayout::new();
        for (type_id, constructor) in &self.layout {
            // safety: each constructor builds the storage declared for the
            // matching component type
            unsafe { layout.register_component_raw(*type_id, *constructor) };
        }
        layout
    }
}

impl World {
    /// Clones a tree of entities, rewriting internal [`Entity`] references so
    /// that the clones form a self-contained copy of the tree.
    ///
    /// `roots` names the entities at which traversal starts and `edges` maps an
    /// entity to the other tree entities reachable from it (for example, the
    /// target of its parent or children components). The world is passed to the
    /// closure as a read-only view so that component data can be inspected
    /// while the clone is in progress. Traversal is breadth-first and
    /// cycle-safe; every reachable entity is cloned exactly once.
    ///
    /// `registry` describes how each component type is duplicated. While
    /// components are cloned, a mapping from every tree entity ID to its fresh
    /// clone ID is installed as the context used by [`Entity`]'s [`Clone`]
    /// implementation. References to tree entities are therefore rewritten to
    /// the corresponding clones, while references to entities outside the tree
    /// are not present in the mapping and keep pointing at their original
    /// targets.
    ///
    /// The clone is transactional: all component clones are produced and
    /// validated before anything is inserted into the world. If any component
    /// clone fails, the staging data is dropped and the world is left
    /// completely untouched.
    ///
    /// On success, the returned map maps every original tree entity ID to the
    /// ID of its clone.
    ///
    /// # Errors
    /// Returns [`CloneError::EntityNotFound`] if a root or an entity reached via
    /// `edges` does not exist in the world, and
    /// [`CloneError::ComponentCloneFailed`] if a registered fallible clone
    /// function fails. In both cases the world is unchanged.
    ///
    /// # Examples
    /// ```
    /// # use legion::*;
    /// # use legion::world::CloneRegistry;
    /// #[derive(Clone, Copy)]
    /// struct Label(&'static str);
    /// // components holding an Entity reference must be `Clone` (not `Copy`),
    /// // otherwise a bitwise copy would bypass Entity reference rewriting
    /// #[derive(Clone)]
    /// struct Parent(Entity);
    /// #[derive(Clone)]
    /// struct Child(Entity);
    ///
    /// let mut world = World::default();
    /// let root = world.push((Label("root"),));
    /// let child = world.push((Label("child"), Parent(root)));
    /// world.entry(root).unwrap().add_component(Child(child));
    ///
    /// let mut registry = CloneRegistry::new();
    /// registry.register_clone::<Label>();
    /// registry.register_clone::<Parent>();
    /// registry.register_clone::<Child>();
    ///
    /// let mapping = world
    ///     .clone_tree(
    ///         [root],
    ///         |world, entity| {
    ///             let entry = match world.entry_ref(entity) {
    ///                 Ok(entry) => entry,
    ///                 Err(_) => return Vec::new(),
    ///             };
    ///             let mut edges = Vec::new();
    ///             if let Ok(parent) = entry.get_component::<Parent>() {
    ///                 edges.push(parent.0);
    ///             }
    ///             if let Ok(c) = entry.get_component::<Child>() {
    ///                 edges.push(c.0);
    ///             }
    ///             edges
    ///         },
    ///         &mut registry,
    ///     )
    ///     .unwrap();
    ///
    /// let cloned_child = mapping[&child];
    /// let cloned_parent = world
    ///     .entry(cloned_child)
    ///     .unwrap()
    ///     .get_component::<Parent>()
    ///     .unwrap()
    ///     .0;
    /// assert_eq!(cloned_parent, mapping[&root]);
    /// ```
    pub fn clone_tree<I, E, F>(
        &mut self,
        roots: I,
        edges: F,
        registry: &mut CloneRegistry,
    ) -> Result<HashMap<Entity, Entity, EntityHasher>, CloneError>
    where
        I: IntoIterator<Item = Entity>,
        E: IntoIterator<Item = Entity>,
        F: Fn(&World, Entity) -> E,
    {
        // Phase 1: discover all entities which belong to the tree.
        let mut tree: Vec<Entity> = Vec::new();
        let mut seen: HashMap<Entity, (), EntityHasher> = HashMap::default();
        let mut queue: Vec<Entity> = roots.into_iter().collect();
        while let Some(entity) = queue.pop() {
            if seen.contains_key(&entity) {
                continue;
            }
            if !self.contains(entity) {
                return Err(CloneError::EntityNotFound(entity));
            }
            seen.insert(entity, ());
            tree.push(entity);
            queue.extend(edges(self, entity));
        }

        // Phase 2: allocate fresh IDs and group entities by destination layout.
        let mut allocator = Allocate::new();
        let mut mapping: HashMap<Entity, Entity, EntityHasher> = HashMap::default();
        let mut groups: Vec<CloneGroup> = Vec::new();
        for entity in &tree {
            let clone = allocator.next().unwrap();
            mapping.insert(*entity, clone);

            let location = self.entity_locations().get(*entity).unwrap();
            let src_arch = &self.archetypes()[location.archetype()];
            let mut layout: Vec<(ComponentTypeId, fn() -> Box<dyn UnknownComponentStorage>)> =
                src_arch
                    .layout()
                    .component_types()
                    .iter()
                    .filter_map(|type_id| {
                        registry
                            .cloners
                            .get(type_id)
                            .map(|registered| (*type_id, registered.storage_constructor))
                    })
                    .collect();
            layout.sort_by_key(|(type_id, _)| type_id.type_id());
            let layout_ids: Vec<ComponentTypeId> =
                layout.iter().map(|(type_id, _)| *type_id).collect();

            let group = match groups
                .iter_mut()
                .find(|group| group.layout_ids() == layout_ids)
            {
                Some(group) => group,
                None => {
                    groups.push(CloneGroup {
                        layout,
                        members: Vec::new(),
                        staged: HashMap::default(),
                    });
                    groups.last_mut().unwrap()
                }
            };
            group
                .members
                .push((*entity, location.archetype(), location.component().0));
        }

        // Phase 3: clone every component off-world. The mapping stays installed
        // for the whole phase so Entity::clone rewrites internal references and
        // leaves external references untouched. Nothing is written into the
        // world in this phase, so any failure leaves it untouched. Each group
        // owns its staging buffers, keeping the components aligned with the
        // entities that will be committed into the same destination archetype.
        let _guard = MappingGuard::install(mapping.clone());
        for group in &mut groups {
            for (_entity, arch_index, component_index) in &group.members {
                let src_arch = &self.archetypes()[*arch_index];
                for (type_id, _) in &group.layout {
                    let buffer = group
                        .staged
                        .entry(*type_id)
                        .or_insert_with(|| registry.cloners[type_id].cloner.empty_buffer());
                    let result = unsafe {
                        registry.cloners.get_mut(type_id).unwrap().cloner.stage(
                            src_arch,
                            *component_index,
                            self.components(),
                            buffer.as_mut(),
                        )
                    };
                    if let Err(source) = result {
                        // staged values are dropped with the group buffers;
                        // nothing has been written into the world yet
                        return Err(CloneError::ComponentCloneFailed {
                            component_type: *type_id,
                            source,
                        });
                    }
                }
            }
        }
        drop(_guard);

        // Phase 4: commit all clones in a single step. By this point every
        // component has been cloned successfully, so this phase cannot fail.
        for group in groups {
            let cloned_ids: Vec<Entity> = group
                .members
                .iter()
                .map(|(entity, _, _)| mapping[entity])
                .collect();

            let mut filter = ExactLayoutFilter {
                layout: group.layout.clone(),
            };
            let type_ids: Vec<ComponentTypeId> = filter.layout.iter().map(|(id, _)| *id).collect();
            let mut staged = group.staged;
            self.commit_clone_group(&mut filter, &cloned_ids, |writer| {
                for type_id in &type_ids {
                    let buffer = staged.get_mut(type_id).unwrap();
                    unsafe {
                        registry
                            .cloners
                            .get(type_id)
                            .unwrap()
                            .cloner
                            .write_all(buffer.as_mut(), writer);
                    }
                }
            });
        }

        Ok(mapping)
    }
}
