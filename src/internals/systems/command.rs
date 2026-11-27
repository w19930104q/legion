//! Contains types related to command buffers.
//!
//! Use command buffers to enqueue changes to a world from within a system.
//! For example, creating or destroying entities.
//! Command buffers are flushed at the end of the schedule, or by adding a
//! `flush_command_buffers` step to the schedule.

use std::{
    any::type_name,
    collections::{HashMap, VecDeque},
    fmt,
    iter::{Fuse, FusedIterator},
    marker::PhantomData,
    ops::Range,
    sync::Arc,
};

use smallvec::SmallVec;

use crate::{
    internals::{
        entity::{Entity, EntityHasher},
        insert::{
            ArchetypeSource, ArchetypeWriter, ComponentSource, IntoComponentSource, KnownLength,
        },
        storage::{archetype::EntityLayout, component::Component},
        systems::resources::Resources,
        world::{World, WorldId},
    },
    world::Allocate,
};

/// This trait can be used to implement custom world writer types that can be directly
/// inserted into the command buffer, for more custom and complex world operations. This is analogous
/// to the `CommandBuffer::exec_mut` function type, but does not perform explicit any/any archetype
/// access.
pub trait WorldWritable: Send + Sync {
    /// Destructs the writer and performs the write operations on the world.
    fn write(self: Arc<Self>, world: &mut World, cmd: &CommandBuffer, remap: &mut HashMap<Entity, Entity, EntityHasher>);
}

struct InsertBufferedCommand<T> {
    components: T,
    entities: Range<usize>,
}

impl<T> fmt::Debug for InsertBufferedCommand<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!(
            "InsertBufferedCommand<{}>({:?})",
            type_name::<T>(),
            self.entities
        ))
    }
}

impl<T> WorldWritable for InsertBufferedCommand<T>
where
    T: ComponentSource + Send + Sync,
{
    fn write(self: Arc<Self>, world: &mut World, cmd: &CommandBuffer, remap: &mut HashMap<Entity, Entity, EntityHasher>) {
        let consumed = Arc::try_unwrap(self).unwrap();

        let preallocated: Vec<Entity> = cmd.pending_insertion[consumed.entities].iter().copied().collect();
        let actual = world.extend(PreallocComponentSource::new(
            cmd.pending_insertion[consumed.entities].iter().copied(),
            consumed.components,
        ));

        // Build remapping from preallocated to actual world IDs.
        // Currently these are identity mappings since PreallocComponentSource
        // feeds preallocated IDs through ConcatIter, but the remap infrastructure
        // supports future non-identity remapping.
        for (pre, act) in preallocated.iter().zip(actual.iter()) {
            remap.insert(*pre, *act);
        }
    }
}

struct PreallocComponentSource<I: Iterator<Item = Entity> + FusedIterator, C: ComponentSource> {
    entities: I,
    components: C,
}

impl<I: Iterator<Item = Entity> + FusedIterator, C: ComponentSource> IntoComponentSource
    for PreallocComponentSource<I, C>
{
    type Source = Self;

    fn into(self) -> Self::Source {
        self
    }
}

impl<I: Iterator<Item = Entity>, C: ComponentSource> PreallocComponentSource<Fuse<I>, C> {
    pub fn new(entities: I, components: C) -> Self {
        Self {
            entities: entities.fuse(),
            components,
        }
    }
}

impl<I: Iterator<Item = Entity> + FusedIterator, C: ComponentSource> ArchetypeSource
    for PreallocComponentSource<I, C>
{
    type Filter = C::Filter;
    fn filter(&self) -> Self::Filter {
        self.components.filter()
    }
    fn layout(&mut self) -> EntityLayout {
        self.components.layout()
    }
}

impl<I: Iterator<Item = Entity> + FusedIterator, C: ComponentSource> ComponentSource
    for PreallocComponentSource<I, C>
{
    fn push_components<'a>(
        &mut self,
        writer: &mut ArchetypeWriter<'a>,
        mut entities: impl Iterator<Item = Entity>,
    ) {
        let iter = ConcatIter {
            a: &mut self.entities,
            b: &mut entities,
        };
        self.components.push_components(writer, iter)
    }
}

struct ConcatIter<'a, T, A: Iterator<Item = T> + FusedIterator, B: Iterator<Item = T>> {
    a: &'a mut A,
    b: &'a mut B,
}

impl<'a, T, A: Iterator<Item = T> + FusedIterator, B: Iterator<Item = T>> Iterator
    for ConcatIter<'a, T, A, B>
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.a.next().or_else(|| self.b.next())
    }
}

struct InsertCommand<T> {
    components: T,
}

impl<T> fmt::Debug for InsertCommand<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("InsertCommand<{}>", type_name::<T>()))
    }
}

impl<T> WorldWritable for InsertCommand<T>
where
    T: IntoComponentSource + Send + Sync,
{
    fn write(self: Arc<Self>, world: &mut World, _: &CommandBuffer, _remap: &mut HashMap<Entity, Entity, EntityHasher>) {
        let consumed = Arc::try_unwrap(self).unwrap();
        world.extend(consumed.components);
    }
}

struct DeleteEntityCommand(Entity);

impl fmt::Debug for DeleteEntityCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("DeleteEntityCommand({:?})", self.0))
    }
}

impl WorldWritable for DeleteEntityCommand {
    fn write(self: Arc<Self>, world: &mut World, _: &CommandBuffer, remap: &mut HashMap<Entity, Entity, EntityHasher>) {
        let entity = remap.get(&self.0).copied().unwrap_or(self.0);
        world.remove(entity);
    }
}

struct AddComponentCommand<C> {
    entity: Entity,
    component: C,
}

impl<T> fmt::Debug for AddComponentCommand<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!(
            "AddComponentCommand<{}>({:?})",
            type_name::<T>(),
            self.entity
        ))
    }
}

impl<C> WorldWritable for AddComponentCommand<C>
where
    C: Component,
{
    fn write(self: Arc<Self>, world: &mut World, _: &CommandBuffer, remap: &mut HashMap<Entity, Entity, EntityHasher>) {
        let consumed = Arc::try_unwrap(self).unwrap();
        let entity = remap.get(&consumed.entity).copied().unwrap_or(consumed.entity);
        world
            .entry(entity)
            .expect("entity not found")
            .add_component(consumed.component);
    }
}

struct RemoveComponentCommand<C> {
    entity: Entity,
    _marker: PhantomData<C>,
}

impl<T> fmt::Debug for RemoveComponentCommand<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!(
            "RemoveComponentCommand<{}>({:?})",
            type_name::<T>(),
            self.entity
        ))
    }
}

impl<C> WorldWritable for RemoveComponentCommand<C>
where
    C: Component,
{
    fn write(self: Arc<Self>, world: &mut World, _: &CommandBuffer, remap: &mut HashMap<Entity, Entity, EntityHasher>) {
        let entity = remap.get(&self.entity).copied().unwrap_or(self.entity);
        world
            .entry(entity)
            .expect("entity not found")
            .remove_component::<C>();
    }
}

#[derive(Clone)]
#[allow(clippy::enum_variant_names)]
enum Command {
    WriteWorld(Arc<dyn WorldWritable>),
    ExecMutWorld(Arc<dyn Fn(&mut World, &mut Resources) + Send + Sync>),
}

#[derive(Clone)]
/// A command buffer used to queue mutable changes to the world from a system. This buffer is automatically
/// flushed and refreshed at the beginning of every frame by `Schedule`. If `Schedule` is not used,
/// then the user needs to manually flush it by performing `CommandBuffer::flush`.
///
/// # Examples
///
/// Inserting an entity using the `CommandBuffer`:
///
/// ```
/// # use legion::*;
/// # use legion::systems::CommandBuffer;
/// # #[derive(Copy, Clone, Debug, PartialEq)]
/// # struct Position(f32);
/// # #[derive(Copy, Clone, Debug, PartialEq)]
/// # struct Rotation(f32);
/// # let mut world = World::default();
/// # let mut resources = Resources::default();
/// let mut command_buffer = CommandBuffer::new(&world);
/// let entity = command_buffer.push(());
///
/// command_buffer.add_component(entity, Position(123.0));
/// command_buffer.remove(entity);
///
/// command_buffer.flush(&mut world, &mut resources);
/// ```
pub struct CommandBuffer {
    world_id: WorldId,
    commands: VecDeque<Command>,
    entity_allocator: Allocate,
    pending_insertion: SmallVec<[Entity; 64]>,
}

impl CommandBuffer {
    /// Constructs an empty command buffer.
    pub fn new(world: &World) -> Self {
        Self {
            world_id: world.id(),
            commands: Default::default(),
            pending_insertion: SmallVec::new(),
            entity_allocator: Allocate::new(),
        }
    }

    /// Gets the ID of the world this command buffer belongs to.
    pub fn world(&self) -> WorldId {
        self.world_id
    }

    /// Flushes this command buffer, draining all stored commands and writing them to the world.
    ///
    /// Command flushes are performed in a FIFO manner, allowing for reliable, linear commands being
    /// executed in the order they were provided.
    pub fn flush(&mut self, world: &mut World, resources: &mut Resources) {
        if self.world_id != world.id() {
            panic!("command buffers may only write into their parent world");
        }

        let mut entity_remap: HashMap<Entity, Entity, EntityHasher> = HashMap::default();

        while let Some(command) = self.commands.pop_back() {
            match command {
                Command::WriteWorld(ptr) => ptr.write(world, self, &mut entity_remap),
                Command::ExecMutWorld(closure) => closure(world, resources),
            }
        }

        if !entity_remap.is_empty() {
            world.remap_entities(&entity_remap);
        }

        self.pending_insertion.clear();
    }

    /// Executes an arbitrary closure against the mutable world, allowing for queued exclusive
    /// access to the world.
    pub fn exec_mut<F>(&mut self, f: F)
    where
        F: 'static + Fn(&mut World, &mut Resources) + Send + Sync,
    {
        self.commands.push_front(Command::ExecMutWorld(Arc::new(f)));
    }

    /// Inserts an arbitrary implementor of the `WorldWritable` trait into the command queue.
    /// This can be leveraged for creating custom `WorldWritable` trait implementors, and is used
    /// internally for the default writers.
    fn insert_writer<W>(&mut self, writer: W)
    where
        W: 'static + WorldWritable,
    {
        self.commands
            .push_front(Command::WriteWorld(Arc::new(writer)));
    }

    /// Queues the insertion of a single entity into the world.
    pub fn push<T>(&mut self, components: T) -> Entity
    where
        Option<T>: 'static + IntoComponentSource,
        <Option<T> as IntoComponentSource>::Source: KnownLength + Send + Sync,
    {
        self.extend(Some(components))[0]
    }

    /// Queues the insertion of new entities into the world.
    pub fn extend<T>(&mut self, components: T) -> &[Entity]
    where
        T: 'static + IntoComponentSource,
        <T as IntoComponentSource>::Source: KnownLength + Send + Sync,
    {
        let components = components.into();
        let start = self.pending_insertion.len();
        let count = components.len();

        self.pending_insertion.reserve(count);
        for _ in 0..count {
            self.pending_insertion
                .push(self.entity_allocator.next().unwrap());
        }

        let range = start..self.pending_insertion.len();

        self.commands
            .push_front(Command::WriteWorld(Arc::new(InsertBufferedCommand {
                components,
                entities: range.clone(),
            })));

        &self.pending_insertion[range]
    }

    /// Queues the deletion of an entity in the command buffer.
    pub fn remove(&mut self, entity: Entity) {
        self.insert_writer(DeleteEntityCommand(entity));
    }

    /// Queues the addition of a component from an entity in the command buffer.
    pub fn add_component<C: Component>(&mut self, entity: Entity, component: C) {
        self.insert_writer(AddComponentCommand { entity, component });
    }

    /// Queues the removal of a component from an entity in the command buffer.
    pub fn remove_component<C: Component>(&mut self, entity: Entity) {
        self.insert_writer(RemoveComponentCommand {
            entity,
            _marker: PhantomData::<C>::default(),
        });
    }

    /// Returns the current number of commands already queued in this `CommandBuffer` instance.
    #[inline]
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Returns true if this `CommandBuffer` is currently empty and contains no writers.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internals::query::{view::read::Read, IntoQuery};

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Pos(f32, f32, f32);
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Vel(f32, f32, f32);
    #[derive(Default)]
    struct TestResource(pub i32);

    #[test]
    fn simple_write_test() {
        let mut world = World::default();
        let mut resources = Resources::default();

        let components = vec![
            (Pos(1., 2., 3.), Vel(0.1, 0.2, 0.3)),
            (Pos(4., 5., 6.), Vel(0.4, 0.5, 0.6)),
        ];
        let components_len = components.len();

        let mut command = CommandBuffer::new(&world);
        let _ = command.extend(components);

        // Assert writing checks
        // TODO:
        //assert_eq!(
        //    vec![ComponentTypeId::of::<Pos>(), ComponentTypeId::of::<Vel>()],
        //    command.write_components()
        //);

        command.flush(&mut world, &mut resources);

        let mut query = Read::<Pos>::query();

        let mut count = 0;
        for _ in query.iter(&world) {
            count += 1;
        }

        assert_eq!(components_len, count);
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Parent;
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Child(Entity);

    #[test]
    fn parent_child_entity_reference() {
        let mut world = World::default();
        let mut resources = Resources::default();
        let mut cmd = CommandBuffer::new(&world);

        // Create 10 parent entities
        let parents: Vec<Entity> = cmd.extend(vec![(Parent,); 10]).to_vec();

        // Create 10 child entities, each referencing its parent
        let children: Vec<Entity> = parents
            .iter()
            .map(|&p| cmd.push((Child(p),)))
            .collect();

        // Flush
        cmd.flush(&mut world, &mut resources);

        // Verify all parents exist with correct components
        for &parent in &parents {
            let entry = world.entry(parent).unwrap();
            assert!(entry.get_component::<Parent>().is_ok(),
                "Parent entity {:?} missing Parent component", parent);
        }

        // Verify all children exist and reference the correct parents
        for (i, &child) in children.iter().enumerate() {
            let entry = world.entry(child).unwrap();
            let child_comp = entry.get_component::<Child>().unwrap();
            assert_eq!(child_comp.0, parents[i],
                "Child {:?} references {:?} but should reference parent {:?}",
                child, child_comp.0, parents[i]);
            assert!(world.contains(parents[i]),
                "Parent {:?} referenced by child {:?} does not exist", parents[i], child);
        }

        // Verify parent and child counts match
        use crate::internals::query::{view::read::Read, IntoQuery};

        let mut parent_query = Read::<Parent>::query();
        assert_eq!(parent_query.iter(&world).count(), 10);

        let mut child_query = Read::<Child>::query();
        assert_eq!(child_query.iter(&world).count(), 10);
    }

    #[test]
    fn parent_child_entity_reference_single_push() {
        let mut world = World::default();
        let mut resources = Resources::default();
        let mut cmd = CommandBuffer::new(&world);

        // Create parent, then child referencing it
        let parent = cmd.push((Parent,));
        let child = cmd.push((Child(parent),));

        cmd.flush(&mut world, &mut resources);

        // Verify parent exists
        let parent_entry = world.entry(parent).unwrap();
        assert!(parent_entry.get_component::<Parent>().is_ok());

        // Verify child references correct parent
        let child_entry = world.entry(child).unwrap();
        let child_comp = child_entry.get_component::<Child>().unwrap();
        assert_eq!(child_comp.0, parent,
            "Child references {:?} but should reference parent {:?}",
            child_comp.0, parent);
    }

    #[test]
    fn entity_remap_after_flush() {
        let mut world = World::default();
        let mut resources = Resources::default();
        let mut cmd = CommandBuffer::new(&world);

        // Create entities that reference each other
        let parent = cmd.push((Parent,));
        let child = cmd.push((Child(parent),));

        // The preallocated IDs may differ from final IDs; flush should remap
        cmd.flush(&mut world, &mut resources);

        // After flush, world should have consistent references
        let child_entry = world.entry(child).unwrap();
        let child_comp = child_entry.get_component::<Child>().unwrap();
        assert_eq!(child_comp.0, parent);
        assert!(world.contains(parent));
    }
}
