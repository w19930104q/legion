//! Contains types related to the [`SubWorld`] which can split a world by component type access.

use std::borrow::Cow;
use std::fmt;

use bit_set::BitSet;

use super::{
    entity::Entity,
    entry::{EntryMut, EntryRef},
    permissions::Permissions,
    query::{
        filter::EntityFilter,
        view::{IntoView, View},
        Query,
    },
    storage::{archetype::ArchetypeIndex, component::ComponentTypeId},
    world::{EntityAccessError, EntityStore, StorageAccessor, World, WorldId},
};

#[derive(Clone)]
/// Describes which archetypes are available for access.
pub enum ArchetypeAccess {
    /// All archetypes.
    All,
    /// Some archetypes.
    Some(BitSet),
}

impl ArchetypeAccess {
    /// Determines if the given archetypes are disjoint from those allowed by this archetype access,
    pub fn is_disjoint(&self, other: &ArchetypeAccess) -> bool {
        match self {
            Self::All => false,
            Self::Some(mine) => {
                match other {
                    Self::All => false,
                    Self::Some(theirs) => mine.is_disjoint(theirs),
                }
            }
        }
    }

    /// Returns a bitset of allowed archetype indexes if this access is `Some`.
    pub fn bitset(&self) -> Option<&BitSet> {
        match self {
            Self::All => None,
            Self::Some(bitset) => Some(bitset),
        }
    }
}

/// Error returned when a SubWorld split fails due to insufficient access permissions.
#[derive(Clone, Debug)]
pub struct SubWorldAccessError {
    message: String,
}

impl fmt::Display for SubWorldAccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SubWorld access error: {}", self.message)
    }
}

impl std::error::Error for SubWorldAccessError {}

/// Describes a conflict between two systems detected during schedule validation.
#[derive(Clone, Debug)]
pub struct ScheduleConflict {
    /// Name of the first system involved in the conflict.
    pub system_a: String,
    /// Name of the second system involved in the conflict.
    pub system_b: String,
    /// Description of what resource or component caused the conflict.
    pub resource: String,
}

impl fmt::Display for ScheduleConflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Schedule conflict: system '{}' and system '{}' both require conflicting access to '{}'",
            self.system_a, self.system_b, self.resource
        )
    }
}

/// Collection of schedule conflicts detected during validation.
#[derive(Clone, Debug, Default)]
pub struct ScheduleConflicts {
    /// List of detected conflicts.
    pub conflicts: Vec<ScheduleConflict>,
}

impl ScheduleConflicts {
    /// Returns true if there are no conflicts.
    pub fn is_empty(&self) -> bool {
        self.conflicts.is_empty()
    }
}

impl fmt::Display for ScheduleConflicts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.conflicts.is_empty() {
            write!(f, "No schedule conflicts detected")
        } else {
            writeln!(f, "Schedule conflicts detected:")?;
            for conflict in &self.conflicts {
                writeln!(f, "  - {}", conflict)?;
            }
            Ok(())
        }
    }
}

/// Describes which components are available for access.
#[derive(Clone, Debug)]
pub enum ComponentAccess<'a> {
    /// All component types are allowed.
    All,
    /// Some component types are allowed.
    Allow(Cow<'a, Permissions<ComponentTypeId>>),
    /// Some component types are disallowed.
    Disallow(Cow<'a, Permissions<ComponentTypeId>>),
}

impl<'a> ComponentAccess<'a> {
    /// Returns `true` if the given component is accessible for reads.
    pub fn allows_read(&self, component: ComponentTypeId) -> bool {
        match self {
            Self::All => true,
            Self::Allow(components) => components.reads().contains(&component),
            Self::Disallow(components) => !components.reads().contains(&component),
        }
    }

    /// Returns `true` if the given component is accessible for writes.
    pub fn allows_write(&self, component: ComponentTypeId) -> bool {
        match self {
            Self::All => true,
            Self::Allow(components) => components.writes().contains(&component),
            Self::Disallow(components) => !components.writes().contains(&component),
        }
    }

    /// Splits this permission set into two; the left access only allows the permissions given, while the right
    /// allows only what is left from this set after subtracting said permissions.
    ///
    /// # Panics
    /// Panics if the requested access is not available in this permission set.
    pub(crate) fn split(&mut self, access: Permissions<ComponentTypeId>) -> (Self, Self) {
        match self.try_split(access) {
            Ok(result) => result,
            Err(e) => panic!("{}", e),
        }
    }

    /// Attempts to split this permission set into two. Returns an error if the requested access
    /// is not available in this permission set.
    ///
    /// The left access only allows the permissions given, while the right
    /// allows only what is left from this set after subtracting said permissions.
    pub fn try_split(
        &mut self,
        access: Permissions<ComponentTypeId>,
    ) -> Result<(Self, Self), SubWorldAccessError> {
        fn append_incompatible(
            denied: &mut Permissions<ComponentTypeId>,
            to_deny: &Permissions<ComponentTypeId>,
        ) {
            // reads are now denied writes
            for read in to_deny.reads() {
                denied.push_write(*read);
            }

            // writes are now entirely denied
            for write in to_deny.writes() {
                denied.push(*write);
            }
        }

        fn incompatible(
            permissions: &Permissions<ComponentTypeId>,
        ) -> Permissions<ComponentTypeId> {
            let mut denied = Permissions::new();
            // if the current permission allows reads, then everything else must deny writes
            for read in permissions.reads_only() {
                denied.push_write(*read);
            }

            // if the current permission allows writes, then everything else must deny all
            for write in permissions.writes() {
                denied.push(*write);
            }

            denied
        }

        let result = match self {
            Self::All => {
                let denied = incompatible(&access);
                (
                    Self::Allow(Cow::Owned(access)),
                    Self::Disallow(Cow::Owned(denied)),
                )
            }
            Self::Allow(allowed) => {
                if !allowed.is_superset(&access) {
                    return Err(SubWorldAccessError {
                        message: format!(
                            "view accesses components unavailable in this world: world allows only {}, view requires {}",
                            allowed, access
                        ),
                    });
                }

                let mut allowed = allowed.clone();
                allowed.to_mut().subtract(&access);

                (Self::Allow(Cow::Owned(access)), Self::Allow(allowed))
            }
            Self::Disallow(denied) => {
                if !denied.is_disjoint(&access) {
                    return Err(SubWorldAccessError {
                        message: format!(
                            "view accesses components unavailable in this world: world disallows {}, view requires {}",
                            denied, access
                        ),
                    });
                }

                let mut denied = denied.clone();
                append_incompatible(denied.to_mut(), &access);

                (Self::Allow(Cow::Owned(access)), Self::Disallow(denied))
            }
        };

        Ok(result)
    }

    /// Validates that this component access is internally consistent and can support
    /// the expected read/write patterns. Returns the set of conflicts if any.
    pub fn validate(&self) -> Result<(), SubWorldAccessError> {
        match self {
            Self::All => Ok(()),
            Self::Allow(permissions) => {
                // Check that the permissions are well-formed
                if permissions.reads().is_empty() && permissions.writes().is_empty() {
                    return Err(SubWorldAccessError {
                        message: "ComponentAccess::Allow with empty permissions".to_string(),
                    });
                }
                Ok(())
            }
            Self::Disallow(permissions) => {
                if permissions.reads().is_empty() && permissions.writes().is_empty() {
                    return Err(SubWorldAccessError {
                        message: "ComponentAccess::Disallow with empty permissions".to_string(),
                    });
                }
                Ok(())
            }
        }
    }
}

/// Provides access to a subset of the entities of a `World`.
///
/// To access a component mutably in a world, such as inside a [query](crate::query) or via an
/// [`EntryMut`], you need to borrow the entire world mutably. This prevents you from accessing
/// any other data in the world at the same time.
///
/// In some cases, we can work around this by splitting the world. We can split a world around the
/// component types requested by a [`View`]. This will create two subworlds, the left one allowing
/// access only to the components (and mutability) declared by the view, while the right subworld
/// will allow access to everything _but_ those components.
///
/// Subworlds can be recursively further split.
pub struct SubWorld<'a> {
    world: &'a World,
    components: ComponentAccess<'a>,
    archetypes: Option<&'a BitSet>,
}

impl<'a> SubWorld<'a> {
    /// Constructs a new SubWorld.
    ///
    /// # Safety
    /// Queries assume that this type has been constructed correctly. Ensure that sub-worlds represent
    /// disjoint portions of a world and that the world is not used while any of its sub-worlds are alive.
    pub unsafe fn new_unchecked(
        world: &'a World,
        components: ComponentAccess<'a>,
        archetypes: Option<&'a BitSet>,
    ) -> Self {
        Self {
            world,
            components,
            archetypes,
        }
    }

    /// Splits the world into two. The left world allows access only to the data declared by the view;
    /// the right world allows access to all else.
    ///
    /// # Examples
    ///
    /// ```
    /// # use legion::*;
    /// # struct Position;
    /// # let mut world = World::default();
    /// let (left, right) = world.split::<&mut Position>();
    /// ```
    ///
    /// With the above, 'left' contains a sub-world with access _only_ to `&Position` and `&mut Position`,
    /// and `right` contains a sub-world with access to everything _but_ `&Position` and `&mut Position`.
    ///
    /// ```
    /// # use legion::*;
    /// # struct Position;
    /// # let mut world = World::default();
    /// let (left, right) = world.split::<&Position>();
    /// ```
    ///
    /// In this second example, `left` is provided access _only_ to `&Position`. `right` is granted permission
    /// to everything _but_ `&mut Position`.
    ///
    /// # Panics
    /// Panics if the requested access is not available in this SubWorld's permission set.
    pub fn split<'b, T: IntoView>(&'b mut self) -> (SubWorld<'b>, SubWorld<'b>)
    where
        'a: 'b,
    {
        let permissions = T::View::requires_permissions();
        let (left, right) = self.components.split(permissions);

        (
            SubWorld {
                world: self.world,
                components: left,
                archetypes: self.archetypes,
            },
            SubWorld {
                world: self.world,
                components: right,
                archetypes: self.archetypes,
            },
        )
    }

    /// Attempts to split the world into two. Returns an error if the requested access
    /// is not available in this SubWorld's permission set.
    ///
    /// The left world allows access only to the data declared by the view;
    /// the right world allows access to all else.
    pub fn try_split<'b, T: IntoView>(
        &'b mut self,
    ) -> Result<(SubWorld<'b>, SubWorld<'b>), SubWorldAccessError>
    where
        'a: 'b,
    {
        let permissions = T::View::requires_permissions();
        let (left, right) = self.components.try_split(permissions)?;

        Ok((
            SubWorld {
                world: self.world,
                components: left,
                archetypes: self.archetypes,
            },
            SubWorld {
                world: self.world,
                components: right,
                archetypes: self.archetypes,
            },
        ))
    }

    /// Splits the world into two. The left world allows access only to the data declared by the query's view;
    /// the right world allows access to all else.
    pub fn split_for_query<'q, V: IntoView, F: EntityFilter>(
        &mut self,
        _: &'q Query<V, F>,
    ) -> (SubWorld, SubWorld) {
        self.split::<V>()
    }

    /// Attempts to split the world for the given query. Returns an error if the requested access
    /// is not available in this SubWorld's permission set.
    pub fn try_split_for_query<'q, V: IntoView, F: EntityFilter>(
        &mut self,
        _: &'q Query<V, F>,
    ) -> Result<(SubWorld, SubWorld), SubWorldAccessError> {
        self.try_split::<V>()
    }

    /// Returns a reference to the component access permissions for this SubWorld.
    pub fn component_access(&self) -> &ComponentAccess<'a> {
        &self.components
    }

    fn validate_archetype_access(&self, ArchetypeIndex(arch_index): ArchetypeIndex) -> bool {
        if let Some(archetypes) = self.archetypes {
            archetypes.contains(arch_index as usize)
        } else {
            true
        }
    }
}

impl<'a> EntityStore for SubWorld<'a> {
    fn get_component_storage<V: for<'b> View<'b>>(
        &self,
    ) -> Result<StorageAccessor, EntityAccessError> {
        if V::validate_access(&self.components) {
            Ok(self
                .world
                .get_component_storage::<V>()
                .unwrap()
                .with_allowed_archetypes(self.archetypes))
        } else {
            Err(EntityAccessError::AccessDenied)
        }
    }

    fn entry_ref(&self, entity: Entity) -> Result<EntryRef, EntityAccessError> {
        let entry = self.world.entry_ref(entity)?;

        if !self.validate_archetype_access(entry.location().archetype()) {
            return Err(EntityAccessError::AccessDenied);
        }

        Ok(EntryRef {
            allowed_components: self.components.clone(),
            ..entry
        })
    }

    fn entry_mut(&mut self, entity: Entity) -> Result<EntryMut, EntityAccessError> {
        // safety: protected by &mut self and subworld access validation
        let entry = unsafe { self.world.entry_unchecked(entity)? };

        if !self.validate_archetype_access(entry.location().archetype()) {
            return Err(EntityAccessError::AccessDenied);
        }

        Ok(EntryMut {
            allowed_components: self.components.clone(),
            ..entry
        })
    }

    fn id(&self) -> WorldId {
        self.world.id()
    }
}

impl<'a> From<&'a mut World> for SubWorld<'a> {
    fn from(world: &'a mut World) -> Self {
        Self {
            world,
            components: ComponentAccess::All,
            archetypes: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        internals::{
            query::view::{read::Read, write::Write},
            world::{EntityStore, World},
        },
        world::ComponentError,
    };

    #[test]
    fn split_write_permissions() {
        let mut world = World::default();
        let entity = world.push((1usize, false));

        let (mut left, mut right) = world.split::<Write<usize>>();

        let mut left_entry = left.entry_mut(entity).unwrap();
        let mut right_entry = right.entry_mut(entity).unwrap();

        assert!(left_entry.get_component::<usize>().is_ok());
        assert!(left_entry.get_component_mut::<usize>().is_ok());

        assert!(matches!(
            left_entry.get_component::<bool>(),
            Err(ComponentError::Denied { .. })
        ));
        assert!(matches!(
            left_entry.get_component_mut::<bool>(),
            Err(ComponentError::Denied { .. })
        ));

        assert!(right_entry.get_component::<bool>().is_ok());
        assert!(right_entry.get_component_mut::<bool>().is_ok());

        assert!(matches!(
            right_entry.get_component::<usize>(),
            Err(ComponentError::Denied { .. })
        ));
        assert!(matches!(
            right_entry.get_component_mut::<usize>(),
            Err(ComponentError::Denied { .. })
        ));
    }

    #[test]
    fn split_read_permissions() {
        let mut world = World::default();
        let entity = world.push((1usize, false));

        let (mut left, mut right) = world.split::<Read<usize>>();

        let mut left_entry = left.entry_mut(entity).unwrap();
        let mut right_entry = right.entry_mut(entity).unwrap();

        assert!(left_entry.get_component::<usize>().is_ok());
        assert!(matches!(
            left_entry.get_component_mut::<usize>(),
            Err(ComponentError::Denied { .. })
        ));

        assert!(matches!(
            left_entry.get_component::<bool>(),
            Err(ComponentError::Denied { .. })
        ));
        assert!(matches!(
            left_entry.get_component_mut::<bool>(),
            Err(ComponentError::Denied { .. })
        ));

        assert!(right_entry.get_component::<bool>().is_ok());
        assert!(right_entry.get_component_mut::<bool>().is_ok());

        assert!(right_entry.get_component::<usize>().is_ok());
        assert!(matches!(
            right_entry.get_component_mut::<usize>(),
            Err(ComponentError::Denied { .. })
        ));
    }

    #[test]
    fn complex_split() {
        struct A;
        struct B;
        struct C;

        let mut world = World::default();
        let entity = world.push((A, B, C));

        let (mut left, mut right) = world.split::<(Read<A>, Write<B>)>();

        let mut left_entry = left.entry_mut(entity).unwrap();
        let mut right_entry = right.entry_mut(entity).unwrap();

        // left should read but not write A
        assert!(left_entry.get_component::<A>().is_ok());
        assert!(matches!(
            left_entry.get_component_mut::<A>(),
            Err(ComponentError::Denied { .. })
        ));

        // left should read and write B
        assert!(left_entry.get_component::<B>().is_ok());
        assert!(left_entry.get_component_mut::<B>().is_ok());

        // left should not be able to read or write C
        assert!(matches!(
            left_entry.get_component::<C>(),
            Err(ComponentError::Denied { .. })
        ));
        assert!(matches!(
            left_entry.get_component_mut::<C>(),
            Err(ComponentError::Denied { .. })
        ));

        // right should be able to read but not write A
        assert!(right_entry.get_component::<A>().is_ok());
        assert!(matches!(
            right_entry.get_component_mut::<A>(),
            Err(ComponentError::Denied { .. })
        ));

        // right should not be able to read or write B
        assert!(matches!(
            right_entry.get_component::<B>(),
            Err(ComponentError::Denied { .. })
        ));
        assert!(matches!(
            right_entry.get_component_mut::<B>(),
            Err(ComponentError::Denied { .. })
        ));

        // right should read and write C
        assert!(right_entry.get_component::<C>().is_ok());
        assert!(right_entry.get_component_mut::<C>().is_ok());
    }

    #[test]
    fn try_split_insufficient_permissions() {
        let mut world = World::default();
        let entity = world.push((1usize, false));

        // Only declare read access for usize
        let (mut left, mut right) = world.split::<Read<usize>>();

        // Try to split left for write access - should fail
        let result = left.try_split::<Write<usize>>();
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err.to_string().contains("unavailable"));
    }

    #[test]
    fn try_split_valid_permissions() {
        let mut world = World::default();
        let entity = world.push((1usize, false));

        // Declare write access for usize
        let (mut left, mut right) = world.split::<Write<usize>>();

        // Try to split left for write access - should succeed
        let result = left.try_split::<Write<usize>>();
        assert!(result.is_ok());
    }

    #[test]
    fn schedule_conflict_formatting() {
        let conflict = super::ScheduleConflict {
            system_a: "physics".to_string(),
            system_b: "render".to_string(),
            resource: "DeltaTime".to_string(),
        };
        let msg = conflict.to_string();
        assert!(msg.contains("physics"));
        assert!(msg.contains("render"));
        assert!(msg.contains("DeltaTime"));
    }

    #[test]
    fn schedule_conflicts_empty() {
        let conflicts = super::ScheduleConflicts::default();
        assert!(conflicts.is_empty());
    }

    #[test]
    fn schedule_conflicts_display() {
        let mut conflicts = super::ScheduleConflicts::default();
        conflicts.conflicts.push(super::ScheduleConflict {
            system_a: "a".to_string(),
            system_b: "b".to_string(),
            resource: "R".to_string(),
        });
        let msg = conflicts.to_string();
        assert!(msg.contains("a"));
        assert!(msg.contains("b"));
        assert!(msg.contains("R"));
    }
}
