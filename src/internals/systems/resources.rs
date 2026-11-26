//! Contains types related to defining shared resources which can be accessed inside systems.
//!
//! Use resources to share persistent data between systems or to provide a system with state
//! external to entities.

use std::{
    any::TypeId,
    collections::{HashMap, HashSet},
    fmt::{Display, Formatter},
    hash::{BuildHasherDefault, Hasher},
    marker::PhantomData,
};

use atomic_refcell::{AtomicRef, AtomicRefCell, AtomicRefMut};
use downcast_rs::{impl_downcast, Downcast};

use crate::internals::{
    hash::ComponentTypeIdHasher,
    query::view::{read::Read, write::Write, ReadOnly},
};

/// Unique ID for a resource.
#[derive(Copy, Clone, Debug, Eq, PartialOrd, Ord)]
pub struct ResourceTypeId {
    type_id: TypeId,
    #[cfg(debug_assertions)]
    name: &'static str,
}

impl ResourceTypeId {
    /// Returns the resource type ID of the given resource type.
    pub fn of<T: Resource>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            #[cfg(debug_assertions)]
            name: std::any::type_name::<T>(),
        }
    }
}

impl std::hash::Hash for ResourceTypeId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_id.hash(state);
    }
}

impl PartialEq for ResourceTypeId {
    fn eq(&self, other: &Self) -> bool {
        self.type_id.eq(&other.type_id)
    }
}

/// Identifies a specific generation of a resource.
///
/// Every time a resource of a given type is inserted into [`Resources`], the version of that
/// resource type is incremented. Comparing versions allows cached handles to detect that the
/// resource they were fetched from has since been removed and replaced with a new instance.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ResourceVersion(u64);

impl ResourceVersion {
    /// The version of the first insertion of a resource type.
    pub const INITIAL: Self = Self(1);

    fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

impl Display for ResourceVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

impl Display for ResourceTypeId {
    #[cfg(debug_assertions)]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }

    #[cfg(not(debug_assertions))]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.type_id)
    }
}

/// Blanket trait for resource types.
pub trait Resource: 'static + Downcast {}
impl<T> Resource for T where T: 'static {}
impl_downcast!(Resource);

/// Trait which is implemented for tuples of resources and singular resources. This abstracts
/// fetching resources to allow for ergonomic fetching.
///
/// # Example:
/// ```
/// struct TypeA(usize);
/// struct TypeB(usize);
///
/// # use legion::*;
/// # use legion::systems::ResourceSet;
/// let mut resources = Resources::default();
/// resources.insert(TypeA(55));
/// resources.insert(TypeB(12));
///
/// {
///     let (a, mut b) = <(Read<TypeA>, Write<TypeB>)>::fetch_mut(&mut resources);
///     assert_ne!(a.0, b.0);
///     b.0 = a.0;
/// }
///
/// {
///     let (a, b) = <(Read<TypeA>, Read<TypeB>)>::fetch(&resources);
///     assert_eq!(a.0, b.0);
/// }
/// ```
pub trait ResourceSet<'a> {
    /// The resource reference returned during a fetch.
    type Result: 'a;

    /// Fetches all defined resources, without checking mutability.
    ///
    /// # Safety
    /// It is up to the end user to validate proper mutability rules across the resources being accessed.
    unsafe fn fetch_unchecked(resources: &'a UnsafeResources) -> Self::Result;

    /// Fetches all defined resources.
    fn fetch_mut(resources: &'a mut Resources) -> Self::Result {
        // safe because mutable borrow ensures exclusivity
        unsafe { Self::fetch_unchecked(&resources.internal) }
    }

    /// Fetches all defined resources.
    fn fetch(resources: &'a Resources) -> Self::Result
    where
        Self: ReadOnly,
    {
        unsafe { Self::fetch_unchecked(&resources.internal) }
    }
}

impl<'a> ResourceSet<'a> for () {
    type Result = ();

    unsafe fn fetch_unchecked(_: &UnsafeResources) -> Self::Result {}
}

impl<'a, T: Resource> ResourceSet<'a> for Read<T> {
    type Result = AtomicRef<'a, T>;

    unsafe fn fetch_unchecked(resources: &'a UnsafeResources) -> Self::Result {
        let type_id = &ResourceTypeId::of::<T>();
        resources
            .get(&type_id)
            .map(|x| x.get::<T>())
            .unwrap_or_else(|| panic_nonexistent_resource(type_id))
    }
}

impl<'a, T: Resource> ResourceSet<'a> for Write<T> {
    type Result = AtomicRefMut<'a, T>;

    unsafe fn fetch_unchecked(resources: &'a UnsafeResources) -> Self::Result {
        let type_id = &ResourceTypeId::of::<T>();
        resources
            .get(&type_id)
            .map(|x| x.get_mut::<T>())
            .unwrap_or_else(|| panic_nonexistent_resource(type_id))
    }
}

fn panic_nonexistent_resource(type_id: &ResourceTypeId) -> ! {
    #[cfg(debug_assertions)]
    panic!("resource {} does not exist", type_id.name);
    #[cfg(not(debug_assertions))]
    panic!("some resource does not exist");
}

macro_rules! resource_tuple {
    ($head_ty:ident) => {
        impl_resource_tuple!($head_ty);
    };
    ($head_ty:ident, $( $tail_ty:ident ),*) => (
        impl_resource_tuple!($head_ty, $( $tail_ty ),*);
        resource_tuple!($( $tail_ty ),*);
    );
}

macro_rules! impl_resource_tuple {
    ( $( $ty: ident ),* ) => {
        #[allow(unused_parens, non_snake_case)]
        impl<'a, $( $ty: ResourceSet<'a> ),*> ResourceSet<'a> for ($( $ty, )*)
        {
            type Result = ($( $ty::Result, )*);

            unsafe fn fetch_unchecked(resources: &'a UnsafeResources) -> Self::Result {
                ($( $ty::fetch_unchecked(resources), )*)
            }
        }
    };
}

#[cfg(feature = "extended-tuple-impls")]
resource_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z);

#[cfg(not(feature = "extended-tuple-impls"))]
resource_tuple!(A, B, C, D, E, F, G, H);

pub struct ResourceCell {
    data: AtomicRefCell<Box<dyn Resource>>,
}

impl ResourceCell {
    fn new(resource: Box<dyn Resource>) -> Self {
        Self {
            data: AtomicRefCell::new(resource),
        }
    }

    fn into_inner(self) -> Box<dyn Resource> {
        self.data.into_inner()
    }

    /// # Safety
    /// Types which are !Sync should only be retrieved on the thread which owns the resource
    /// collection.
    pub fn get<T: Resource>(&self) -> AtomicRef<T> {
        let borrow = self.data.borrow();
        AtomicRef::map(borrow, |inner| inner.downcast_ref::<T>().unwrap())
    }

    /// # Safety
    /// Types which are !Send should only be retrieved on the thread which owns the resource
    /// collection.
    pub fn get_mut<T: Resource>(&self) -> AtomicRefMut<T> {
        let borrow = self.data.borrow_mut(); // panics if this is borrowed already

        AtomicRefMut::map(borrow, |inner| inner.downcast_mut::<T>().unwrap())
    }
}

/// A container for resources which performs runtime borrow checking
/// but _does not_ ensure that `!Sync` resources aren't accessed across threads.
#[derive(Default)]
pub struct UnsafeResources {
    map: HashMap<ResourceTypeId, ResourceCell, BuildHasherDefault<ComponentTypeIdHasher>>,
    /// The current generation of each resource type. Survives removal, so that a resource
    /// which is re-inserted after being removed is assigned a fresh version.
    versions: HashMap<ResourceTypeId, ResourceVersion, BuildHasherDefault<ComponentTypeIdHasher>>,
    /// Resources whose removal has been requested but which will only be physically dropped
    /// at the next scheduling boundary. Logically these resources are already gone: systems
    /// which declare a dependency on them are skipped, and fetches fail.
    pending_removals: HashSet<ResourceTypeId, BuildHasherDefault<ComponentTypeIdHasher>>,
}

unsafe impl Send for UnsafeResources {}
unsafe impl Sync for UnsafeResources {}

impl UnsafeResources {
    fn contains(&self, type_id: &ResourceTypeId) -> bool {
        self.map.contains_key(type_id)
    }

    /// Returns `true` if a resource of the given type is present and not pending removal.
    ///
    /// Systems which declare a resource dependency which is not available are skipped by
    /// the scheduler.
    pub(crate) fn is_available(&self, type_id: &ResourceTypeId) -> bool {
        self.contains(type_id) && !self.pending_removals.contains(type_id)
    }

    /// Returns the current version of the given resource type, if it has ever been inserted.
    fn version(&self, type_id: &ResourceTypeId) -> Option<ResourceVersion> {
        self.versions.get(type_id).copied()
    }

    /// # Safety
    /// Resources which are `!Send` must be retrieved or inserted only on the main thread.
    unsafe fn insert<T: Resource>(&mut self, resource: T) {
        let type_id = ResourceTypeId::of::<T>();
        // inserting assigns a fresh version, and cancels any pending removal of the old instance
        let version = self
            .versions
            .get(&type_id)
            .copied()
            .map_or(ResourceVersion::INITIAL, ResourceVersion::next);
        self.versions.insert(type_id, version);
        self.pending_removals.remove(&type_id);
        self.map.insert(type_id, ResourceCell::new(Box::new(resource)));
    }

    /// # Safety
    /// Resources which are `!Send` must be retrieved or inserted only on the main thread.
    unsafe fn remove(&mut self, type_id: &ResourceTypeId) -> Option<Box<dyn Resource>> {
        self.pending_removals.remove(type_id);
        self.map.remove(type_id).map(|cell| cell.into_inner())
    }

    /// Marks the given resource type for removal at the next scheduling boundary.
    ///
    /// The resource remains physically stored (so that outstanding borrows stay valid) but is
    /// no longer available: dependent systems are skipped and new fetches fail. Returns `true`
    /// if the resource was present and not already pending removal.
    ///
    /// # Safety
    /// Resources which are `!Send` must be retrieved or inserted only on the main thread.
    unsafe fn remove_deferred(&mut self, type_id: &ResourceTypeId) -> bool {
        if !self.contains(type_id) {
            return false;
        }
        self.pending_removals.insert(*type_id)
    }

    /// Physically drops all resources whose removal was deferred. Called by the scheduler at
    /// flush boundaries, when no systems are executing and so no borrows can be outstanding.
    fn apply_pending_removals(&mut self) {
        let pending = std::mem::take(&mut self.pending_removals);
        for type_id in pending {
            self.map.remove(&type_id);
        }
    }

    fn get(&self, type_id: &ResourceTypeId) -> Option<&ResourceCell> {
        // resources pending removal are logically already gone
        if self.pending_removals.contains(type_id) {
            return None;
        }
        self.map.get(type_id)
    }

    /// # Safety
    /// Resources which are `!Sync` must be retrieved or inserted only on the main thread.
    unsafe fn merge(&mut self, mut other: Self) {
        // Merge resources, retaining our local ones but moving in any non-existant ones
        for resource in other.map.drain() {
            if !self.map.contains_key(&resource.0) {
                let version = self
                    .versions
                    .get(&resource.0)
                    .copied()
                    .map_or(ResourceVersion::INITIAL, ResourceVersion::next);
                self.versions.insert(resource.0, version);
                self.pending_removals.remove(&resource.0);
                self.map.insert(resource.0, resource.1);
            }
        }
    }
}

/// Resources container. Shared resources stored here can be retrieved in systems.
#[derive(Default)]
pub struct Resources {
    internal: UnsafeResources,
    // marker to make `Resources` !Send and !Sync
    _not_send_sync: PhantomData<*const u8>,
}

impl Resources {
    pub(crate) fn internal(&self) -> &UnsafeResources {
        &self.internal
    }

    /// Creates an accessor to resources which are Send and Sync, which itself can be sent
    /// between threads.
    pub fn sync(&mut self) -> SyncResources {
        SyncResources {
            internal: &self.internal,
        }
    }

    /// Returns `true` if type `T` exists in the store. Otherwise, returns `false`.
    pub fn contains<T: Resource>(&self) -> bool {
        self.internal.contains(&ResourceTypeId::of::<T>())
    }

    /// Inserts the instance of `T` into the store. If the type already exists, it will be silently
    /// overwritten. If you would like to retain the instance of the resource that already exists,
    /// call `remove` first to retrieve it.
    pub fn insert<T: Resource>(&mut self, value: T) {
        // safety:
        // this type is !Send and !Sync, and so can only be accessed from the thread which
        // owns the resources collection
        unsafe {
            self.internal.insert(value);
        }
    }

    /// Removes the type `T` from this store if it exists.
    ///
    /// # Returns
    /// If the type `T` was stored, the inner instance of `T is returned. Otherwise, `None`.
    pub fn remove<T: Resource>(&mut self) -> Option<T> {
        // safety:
        // this type is !Send and !Sync, and so can only be accessed from the thread which
        // owns the resources collection
        unsafe {
            let resource = self
                .internal
                .remove(&ResourceTypeId::of::<T>())?
                .downcast::<T>()
                .ok()?;
            Some(*resource)
        }
    }

    /// Marks the type `T` for removal at the next scheduling boundary.
    ///
    /// Unlike [`Resources::remove`], the resource is not dropped immediately. It remains
    /// physically stored - keeping any outstanding borrows valid - but is logically already
    /// gone: fetches of `T` will fail, and systems which declare a dependency on `T` are
    /// skipped by the scheduler. The stored value is physically dropped when the schedule
    /// next flushes its command buffers (or when [`Resources::apply_pending_removals`] is
    /// called manually if no schedule is in use).
    ///
    /// Re-inserting `T` before the boundary cancels the pending removal and assigns the new
    /// instance a fresh [`ResourceVersion`].
    ///
    /// # Returns
    /// `true` if the resource was present and not already pending removal.
    pub fn remove_deferred<T: Resource>(&mut self) -> bool {
        // safety:
        // this type is !Send and !Sync, and so can only be accessed from the thread which
        // owns the resources collection
        unsafe { self.internal.remove_deferred(&ResourceTypeId::of::<T>()) }
    }

    /// Physically drops all resources whose removal was deferred with
    /// [`Resources::remove_deferred`].
    ///
    /// This is called automatically by [`Schedule`](crate::systems::Schedule) at each command
    /// buffer flush boundary. Only call this manually when driving systems without a schedule,
    /// and only when no system borrows are outstanding.
    pub fn apply_pending_removals(&mut self) {
        self.internal.apply_pending_removals();
    }

    /// Returns the current version of the resource type `T`, if it has ever been inserted.
    ///
    /// The version is incremented every time `T` is inserted, allowing stale handles to
    /// detect that the resource they were fetched from has been replaced.
    pub fn version<T: Resource>(&self) -> Option<ResourceVersion> {
        self.internal.version(&ResourceTypeId::of::<T>())
    }

    /// Returns `true` if a resource of type `T` is present and not pending removal.
    ///
    /// Systems which declare a dependency on an unavailable resource are skipped by the
    /// scheduler.
    pub fn is_available<T: Resource>(&self) -> bool {
        self.internal.is_available(&ResourceTypeId::of::<T>())
    }

    pub(crate) fn is_available_id(&self, type_id: &ResourceTypeId) -> bool {
        self.internal.is_available(type_id)
    }

    /// Retrieve an immutable reference to  `T` from the store if it exists. Otherwise, return `None`.
    ///
    /// # Panics
    /// Panics if the resource is already borrowed mutably.
    pub fn get<T: Resource>(&self) -> Option<AtomicRef<T>> {
        // safety:
        // this type is !Send and !Sync, and so can only be accessed from the thread which
        // owns the resources collection
        let type_id = &ResourceTypeId::of::<T>();
        self.internal.get(&type_id).map(|x| x.get::<T>())
    }

    /// Retrieve a mutable reference to  `T` from the store if it exists. Otherwise, return `None`.
    pub fn get_mut<T: Resource>(&self) -> Option<AtomicRefMut<T>> {
        // safety:
        // this type is !Send and !Sync, and so can only be accessed from the thread which
        // owns the resources collection
        let type_id = &ResourceTypeId::of::<T>();
        self.internal.get(&type_id).map(|x| x.get_mut::<T>())
    }

    /// Attempts to retrieve an immutable reference to `T` from the store. If it does not exist,
    /// the closure `f` is called to construct the object and it is then inserted into the store.
    pub fn get_or_insert_with<T: Resource, F: FnOnce() -> T>(&mut self, f: F) -> AtomicRef<T> {
        let type_id = &ResourceTypeId::of::<T>();
        if !self.internal.is_available(type_id) {
            // either absent, or pending removal - in which case the old instance is logically
            // gone and is replaced with a fresh version
            self.insert((f)());
        }
        // safety:
        // this type is !Send and !Sync, and so can only be accessed from the thread which
        // owns the resources collection
        self.internal
            .get(type_id)
            .map(|x| x.get::<T>())
            .unwrap_or_else(|| panic_nonexistent_resource(type_id))
    }

    /// Attempts to retrieve a mutable reference to `T` from the store. If it does not exist,
    /// the closure `f` is called to construct the object and it is then inserted into the store.
    pub fn get_mut_or_insert_with<T: Resource, F: FnOnce() -> T>(
        &mut self,
        f: F,
    ) -> AtomicRefMut<T> {
        let type_id = &ResourceTypeId::of::<T>();
        if !self.internal.is_available(type_id) {
            self.insert((f)());
        }
        // safety:
        // this type is !Send and !Sync, and so can only be accessed from the thread which
        // owns the resources collection
        self.internal
            .get(type_id)
            .map(|x| x.get_mut::<T>())
            .unwrap_or_else(|| panic_nonexistent_resource(type_id))
    }

    /// Attempts to retrieve an immutable reference to `T` from the store. If it does not exist,
    /// the provided value is inserted and then a reference to it is returned.
    pub fn get_or_insert<T: Resource>(&mut self, value: T) -> AtomicRef<T> {
        self.get_or_insert_with(|| value)
    }

    /// Attempts to retrieve a mutable reference to `T` from the store. If it does not exist,
    /// the provided value is inserted and then a reference to it is returned.
    pub fn get_mut_or_insert<T: Resource>(&mut self, value: T) -> AtomicRefMut<T> {
        self.get_mut_or_insert_with(|| value)
    }

    /// Attempts to retrieve an immutable reference to `T` from the store. If it does not exist,
    /// the default constructor for `T` is called.
    ///
    /// `T` must implement `Default` for this method.
    pub fn get_or_default<T: Resource + Default>(&mut self) -> AtomicRef<T> {
        self.get_or_insert_with(T::default)
    }

    /// Attempts to retrieve a mutable reference to `T` from the store. If it does not exist,
    /// the default constructor for `T` is called.
    ///
    /// `T` must implement `Default` for this method.
    pub fn get_mut_or_default<T: Resource + Default>(&mut self) -> AtomicRefMut<T> {
        self.get_mut_or_insert_with(T::default)
    }

    /// Performs merging of two resource storages, which occurs during a world merge.
    /// This merge will retain any already-existant resources in the local world, while moving any
    /// new resources from the source world into this one, consuming the resources.
    pub fn merge(&mut self, other: Resources) {
        // safety:
        // this type is !Send and !Sync, and so can only be accessed from the thread which
        // owns the resources collection
        unsafe {
            self.internal.merge(other.internal);
        }
    }
}

/// A resource collection which is `Send` and `Sync`, but which only allows access to resources
/// which are `Sync`.
pub struct SyncResources<'a> {
    internal: &'a UnsafeResources,
}

impl<'a> SyncResources<'a> {
    /// Retrieve an immutable reference to  `T` from the store if it exists. Otherwise, return `None`.
    ///
    /// # Panics
    /// Panics if the resource is already borrowed mutably.
    pub fn get<T: Resource + Sync>(&self) -> Option<AtomicRef<T>> {
        let type_id = &ResourceTypeId::of::<T>();
        self.internal.get(&type_id).map(|x| x.get::<T>())
    }

    /// Retrieve a mutable reference to  `T` from the store if it exists. Otherwise, return `None`.
    pub fn get_mut<T: Resource + Send>(&self) -> Option<AtomicRefMut<T>> {
        let type_id = &ResourceTypeId::of::<T>();
        self.internal.get(&type_id).map(|x| x.get_mut::<T>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_read_write_test() {
        struct TestOne {
            value: String,
        }

        struct TestTwo {
            value: String,
        }

        struct NotSync {
            ptr: *const u8,
        }

        let mut resources = Resources::default();
        resources.insert(TestOne {
            value: "one".to_string(),
        });

        resources.insert(TestTwo {
            value: "two".to_string(),
        });

        resources.insert(NotSync {
            ptr: std::ptr::null(),
        });

        assert_eq!(resources.get::<TestOne>().unwrap().value, "one");
        assert_eq!(resources.get::<TestTwo>().unwrap().value, "two");
        assert_eq!(resources.get::<NotSync>().unwrap().ptr, std::ptr::null());

        // test re-ownership
        let owned = resources.remove::<TestTwo>();
        assert_eq!(owned.unwrap().value, "two");
    }

    #[test]
    fn deferred_removal_takes_effect_at_boundary() {
        struct TestRes(i32);

        let mut resources = Resources::default();
        resources.insert(TestRes(1));

        // request deferred removal: logically gone, physically still stored
        assert!(resources.remove_deferred::<TestRes>());
        assert!(!resources.is_available::<TestRes>());
        assert!(resources.get::<TestRes>().is_none());
        assert!(resources.contains::<TestRes>());

        // a second request is a no-op
        assert!(!resources.remove_deferred::<TestRes>());

        // the scheduling boundary physically drops the resource
        resources.apply_pending_removals();
        assert!(!resources.contains::<TestRes>());
        assert!(!resources.is_available::<TestRes>());

        // removing a non-existant resource does nothing
        assert!(!resources.remove_deferred::<TestRes>());
    }

    #[test]
    fn reinsertion_after_removal_gets_new_version() {
        struct TestRes(i32);

        let mut resources = Resources::default();
        assert_eq!(resources.version::<TestRes>(), None);

        resources.insert(TestRes(1));
        let v1 = resources.version::<TestRes>().unwrap();
        assert_eq!(v1, ResourceVersion::INITIAL);

        resources.remove_deferred::<TestRes>();
        resources.apply_pending_removals();

        // the version survives removal, so re-insertion is visibly a new generation
        resources.insert(TestRes(2));
        let v2 = resources.version::<TestRes>().unwrap();
        assert!(v2 > v1);
        assert_eq!(resources.get::<TestRes>().unwrap().0, 2);
    }

    #[test]
    fn reinsertion_cancels_pending_removal() {
        struct TestRes(i32);

        let mut resources = Resources::default();
        resources.insert(TestRes(1));
        let v1 = resources.version::<TestRes>().unwrap();

        resources.remove_deferred::<TestRes>();
        // re-insert before the boundary: cancels the pending removal, bumps the version
        resources.insert(TestRes(2));

        assert!(resources.is_available::<TestRes>());
        assert!(resources.version::<TestRes>().unwrap() > v1);

        // the boundary must not drop the re-inserted resource
        resources.apply_pending_removals();
        assert!(resources.contains::<TestRes>());
        assert_eq!(resources.get::<TestRes>().unwrap().0, 2);
    }

    #[test]
    fn get_or_insert_replaces_pending_removal() {
        struct TestRes(i32);

        let mut resources = Resources::default();
        resources.insert(TestRes(1));
        let v1 = resources.version::<TestRes>().unwrap();

        resources.remove_deferred::<TestRes>();
        // the logically-removed instance is replaced with a fresh version
        assert_eq!(resources.get_or_insert::<TestRes>(TestRes(2)).0, 2);
        assert!(resources.version::<TestRes>().unwrap() > v1);
    }
}
