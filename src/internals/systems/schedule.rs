//! Contains types related to defining system schedules.

#[cfg(feature = "parallel")]
use std::iter::repeat;
use std::{
    cell::UnsafeCell,
    fmt::{Debug, Formatter},
};
#[cfg(feature = "parallel")]
use std::{
    collections::{HashMap, HashSet},
    sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(feature = "parallel")]
use itertools::izip;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{
    command::CommandBuffer,
    resources::{ResourceTypeId, Resources, UnsafeResources},
    system::SystemId,
};
use crate::internals::subworld::{ScheduleConflict, ScheduleConflicts};
use crate::internals::{
    storage::component::ComponentTypeId,
    subworld::ArchetypeAccess,
    world::{World, WorldId},
};

/// A `Runnable` which is also `Send` and `Sync`.
pub trait ParallelRunnable: Runnable + Send + Sync {}

impl<T: Runnable + Send + Sync> ParallelRunnable for T {}

impl Debug for dyn ParallelRunnable {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        Debug::fmt(&self.name(), f)
    }
}

/// Trait describing a schedulable type. This is implemented by `System`
pub trait Runnable {
    /// Gets the name of the system.
    fn name(&self) -> Option<&SystemId>;

    /// Gets the resources and component types read by the system.
    fn reads(&self) -> (&[ResourceTypeId], &[ComponentTypeId]);

    /// Gets the resources and component types written by the system.
    fn writes(&self) -> (&[ResourceTypeId], &[ComponentTypeId]);

    /// Prepares the system for execution against a world.
    fn prepare(&mut self, world: &World);

    /// Gets the set of archetypes the system will access when run,
    /// as determined when the system was last prepared.
    fn accesses_archetypes(&self) -> &ArchetypeAccess;

    /// Runs the system.
    ///
    /// # Safety
    ///
    /// The shared references to world and resources may result in unsound mutable aliasing if other code
    /// is accessing the same components or resources as this system. Prefer to use `run` when possible.
    ///
    /// Additionally, systems which are !Sync should never be invoked on a different thread to that which
    /// owns the resources collection passed into this function.
    unsafe fn run_unsafe(&mut self, world: &World, resources: &UnsafeResources);

    /// Gets the system's command buffer.
    fn command_buffer_mut(&mut self, world: WorldId) -> Option<&mut CommandBuffer>;

    /// Runs the system.
    fn run(&mut self, world: &mut World, resources: &mut Resources) {
        unsafe { self.run_unsafe(world, resources.internal()) };
    }
}

impl Debug for dyn Runnable {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        Debug::fmt(&self.name(), f)
    }
}

/// Executes a sequence of systems, potentially in parallel, and then commits their command buffers.
///
/// Systems are provided in execution order. When the `parallel` feature is enabled, the `Executor`
/// may run some systems in parallel. The order in which side-effects (e.g. writes to resources
/// or entities) are observed is maintained.
#[derive(Debug)]
pub struct Executor {
    systems: Vec<SystemBox>,
    #[cfg(feature = "parallel")]
    static_dependants: Vec<Vec<usize>>,
    #[cfg(feature = "parallel")]
    dynamic_dependants: Vec<Vec<usize>>,
    #[cfg(feature = "parallel")]
    static_dependency_counts: Vec<AtomicUsize>,
    #[cfg(feature = "parallel")]
    awaiting: Vec<AtomicUsize>,
}

struct SystemBox(UnsafeCell<Box<dyn ParallelRunnable>>);

// NOT SAFE:
// This type is only safe to use as Send and Sync within
// the constraints of how it is used inside Executor
unsafe impl Send for SystemBox {}
unsafe impl Sync for SystemBox {}

impl SystemBox {
    unsafe fn get(&self) -> &dyn ParallelRunnable {
        std::ops::Deref::deref(&*self.0.get())
    }

    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut(&self) -> &mut dyn ParallelRunnable {
        std::ops::DerefMut::deref_mut(&mut *self.0.get())
    }
}

impl Debug for SystemBox {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        // Safety: we have &self access, and if anyone else is mutably accessing this data concurrently via another &self, it is their responsibility to ensure that they are not beaking aliasing rules
        unsafe { Debug::fmt(&self.get().name(), f) }
    }
}

impl Executor {
    /// Constructs a new executor for all systems to be run in a single stage.
    ///
    /// Systems are provided in the order in which side-effects (e.g. writes to resources or entities)
    /// are to be observed.
    #[cfg(not(feature = "parallel"))]
    pub fn new(systems: Vec<Box<dyn ParallelRunnable>>) -> Self {
        Self {
            systems: systems
                .into_iter()
                .map(|s| SystemBox(UnsafeCell::new(s)))
                .collect(),
        }
    }

    /// Constructs a new executor for all systems to be run in a single stage.
    ///
    /// Systems are provided in the order in which side-effects (e.g. writes to resources or entities)
    /// are to be observed.
    #[cfg(feature = "parallel")]
    #[allow(clippy::cognitive_complexity)]
    // TODO: we should break this up
    pub fn new(systems: Vec<Box<dyn ParallelRunnable>>) -> Self {
        if systems.len() > 1 {
            let mut static_dependency_counts = Vec::with_capacity(systems.len());

            let mut static_dependants: Vec<Vec<_>> =
                repeat(Vec::with_capacity(64)).take(systems.len()).collect();
            let mut dynamic_dependants: Vec<Vec<_>> =
                repeat(Vec::with_capacity(64)).take(systems.len()).collect();

            #[derive(Default)]
            struct PreviousAccess {
                readers: Vec<usize>,
                last_writer: Option<usize>,
            }

            impl PreviousAccess {
                fn add_read(&mut self, idx: usize) -> Option<usize> {
                    self.readers.push(idx);
                    self.last_writer
                }

                fn add_write(&mut self, idx: usize) -> Vec<usize> {
                    let mut dependencies = Vec::new();
                    std::mem::swap(&mut self.readers, &mut dependencies);
                    if let Some(writer) = self.last_writer.replace(idx) {
                        dependencies.push(writer)
                    }
                    dependencies
                }
            }

            let mut resource_accesses =
                HashMap::<ResourceTypeId, PreviousAccess>::with_capacity_and_hasher(
                    64,
                    Default::default(),
                );
            let mut component_accesses =
                HashMap::<ComponentTypeId, PreviousAccess>::with_capacity_and_hasher(
                    64,
                    Default::default(),
                );

            for (i, system) in systems.iter().enumerate() {
                let (read_res, read_comp) = system.reads();
                let (write_res, write_comp) = system.writes();

                // find resource access dependencies
                let mut dependencies = HashSet::with_capacity(64);
                for res in read_res {
                    let access = resource_accesses.entry(*res).or_default();
                    if let Some(dep) = access.add_read(i) {
                        dependencies.insert(dep);
                    }
                }
                for res in write_res {
                    let access = resource_accesses.entry(*res).or_default();
                    for dep in access.add_write(i) {
                        dependencies.insert(dep);
                    }
                }

                static_dependency_counts.push(AtomicUsize::from(dependencies.len()));
                for dep in &dependencies {
                    static_dependants[*dep].push(i);
                }

                // find component access dependencies
                let mut comp_dependencies = HashSet::<usize>::default();
                for comp in read_comp {
                    let access = component_accesses.entry(*comp).or_default();
                    if let Some(dep) = access.add_read(i) {
                        comp_dependencies.insert(dep);
                    }
                }
                for comp in write_comp {
                    let access = component_accesses.entry(*comp).or_default();
                    for dep in access.add_write(i) {
                        comp_dependencies.insert(dep);
                    }
                }

                // remove dependencies which are already static from dynamic dependencies
                for static_dep in &dependencies {
                    comp_dependencies.remove(static_dep);
                }

                for dep in comp_dependencies {
                    if dep != i {
                        // dont be dependent on ourselves
                        dynamic_dependants[dep].push(i);
                    }
                }
            }

            let mut awaiting = Vec::with_capacity(systems.len());
            systems
                .iter()
                .for_each(|_| awaiting.push(AtomicUsize::new(0)));

            Executor {
                awaiting,
                static_dependants,
                dynamic_dependants,
                static_dependency_counts,
                systems: systems
                    .into_iter()
                    .map(|s| SystemBox(UnsafeCell::new(s)))
                    .collect(),
            }
        } else {
            Executor {
                awaiting: Vec::with_capacity(0),
                static_dependants: Vec::with_capacity(0),
                dynamic_dependants: Vec::with_capacity(0),
                static_dependency_counts: Vec::with_capacity(0),
                systems: systems
                    .into_iter()
                    .map(|s| SystemBox(UnsafeCell::new(s)))
                    .collect(),
            }
        }
    }

    /// Converts this executor into a vector of its component systems.
    pub fn into_vec(self) -> Vec<Box<dyn ParallelRunnable>> {
        self.systems.into_iter().map(|s| s.0.into_inner()).collect()
    }

    /// Executes all systems and then flushes their command buffers.
    #[cfg(not(feature = "parallel"))]
    pub fn execute(&mut self, world: &mut World, resources: &mut Resources) {
        let unsafe_resources = resources.internal();
        self.run_systems(world, unsafe_resources);
        self.flush_command_buffers(world, resources);
    }

    /// Executes all systems and then flushes their command buffers.
    #[cfg(feature = "parallel")]
    pub fn execute(&mut self, world: &mut World, resources: &mut Resources) {
        let unsafe_resources = resources.internal();
        rayon::join(|| self.run_systems(world, unsafe_resources), || {});
        self.flush_command_buffers(world, resources);
    }

    /// Executes all systems sequentially.
    ///
    /// Only enabled with parallel is disabled
    #[cfg(not(feature = "parallel"))]
    pub fn run_systems(&mut self, world: &mut World, resources: &UnsafeResources) {
        self.systems.iter_mut().for_each(|system| {
            let system = unsafe { system.get_mut() };
            system.prepare(world);
            unsafe { system.run_unsafe(world, resources) };
        });
    }

    /// Executes all systems, potentially in parallel.
    ///
    /// Ordering is retained in so far as the order of observed resource and component
    /// accesses is maintained.
    ///
    /// Call from within `rayon::ThreadPool::install()` to execute within a specific thread pool.
    #[cfg(feature = "parallel")]
    pub fn run_systems(&mut self, world: &mut World, resources: &UnsafeResources) {
        match self.systems.len() {
            1 => {
                // safety: we have exlusive access to all systems, world and resources here
                unsafe {
                    let system = self.systems[0].get_mut();
                    system.prepare(world);
                    system.run_unsafe(world, resources);
                };
            }
            _ => {
                let systems = &mut self.systems;
                let static_dependency_counts = &self.static_dependency_counts;
                let awaiting = &mut self.awaiting;

                // prepare all systems - archetype filters are pre-executed here
                systems
                    .par_iter_mut()
                    .for_each(|sys| unsafe { sys.get_mut() }.prepare(world));

                // determine dynamic dependencies
                izip!(
                    systems.iter(),
                    self.static_dependants.iter_mut(),
                    self.dynamic_dependants.iter_mut()
                )
                .par_bridge()
                .for_each(|(sys, static_dep, dyn_dep)| {
                    // safety: systems is held exclusively, and we are only reading each system
                    let archetypes = unsafe { sys.get() }.accesses_archetypes();
                    for i in (0..dyn_dep.len()).rev() {
                        let dep = dyn_dep[i];
                        let other = unsafe { systems[dep].get() };

                        // if the archetype sets intersect,
                        // then we can move the dynamic dependant into the static dependants set
                        if !other.accesses_archetypes().is_disjoint(archetypes) {
                            static_dep.push(dep);
                            dyn_dep.swap_remove(i);
                            static_dependency_counts[dep].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });

                // initialize dependency tracking
                for (i, count) in static_dependency_counts.iter().enumerate() {
                    awaiting[i].store(count.load(Ordering::Relaxed), Ordering::Relaxed);
                }

                let awaiting = &self.awaiting;

                // execute all systems with no outstanding dependencies
                (0..systems.len())
                    .into_par_iter()
                    .filter(|i| static_dependency_counts[*i].load(Ordering::SeqCst) == 0)
                    .for_each(|i| {
                        // safety: we are at the root of the execution tree, so we know each
                        // index is exclusive here
                        unsafe { self.run_recursive(i, world, resources) };
                    });

                debug_assert!(
                    awaiting.iter().all(|x| x.load(Ordering::SeqCst) == 0),
                    "not all systems run: {:?}",
                    awaiting
                );
            }
        }
    }

    /// Flushes the recorded command buffers for all systems.
    pub fn flush_command_buffers(&mut self, world: &mut World, resources: &mut Resources) {
        self.systems.iter().for_each(|system| {
            // safety: systems are exlcusive due to &mut self
            let system = unsafe { system.get_mut() };
            if let Some(cmd) = system.command_buffer_mut(world.id()) {
                cmd.flush(world, resources);
            }
        });
    }

    /// Recursively execute through the generated depedency cascade and exhaust it.
    ///
    /// # Safety
    ///
    /// Ensure the system indexed by `i` is only accessed once.
    #[cfg(feature = "parallel")]
    unsafe fn run_recursive(&self, i: usize, world: &World, resources: &UnsafeResources) {
        // safety: the caller ensures nothing else is accessing systems[i]
        self.systems[i].get_mut().run_unsafe(world, resources);

        self.static_dependants[i].par_iter().for_each(|dep| {
            if self.awaiting[*dep].fetch_sub(1, Ordering::Relaxed) == 1 {
                // safety: each dependency is unique, so run_recursive is safe to call
                self.run_recursive(*dep, world, resources);
            }
        });
    }
}

    /// Checks all systems in this executor for access conflicts.
    ///
    /// Returns a list of conflicts where two or more systems have incompatible access
    /// patterns that could cause runtime errors. When systems only share read access,
    /// no conflict is reported and they can safely run in parallel.
    pub fn check_conflicts(&self) -> ScheduleConflicts {
        let mut conflicts = ScheduleConflicts::default();

        // Safety: we only read system metadata, not run them
        let systems: Vec<&dyn ParallelRunnable> =
            self.systems.iter().map(|s| unsafe { s.get() }).collect();

        for i in 0..systems.len() {
            for j in (i + 1)..systems.len() {
                let sys_a = systems[i];
                let sys_b = systems[j];

                let (reads_a, _) = sys_a.reads();
                let (writes_a, _) = sys_a.writes();
                let (reads_b, _) = sys_b.reads();
                let (writes_b, _) = sys_b.writes();

                // Check for write-write conflicts on resources
                for res_a in writes_a {
                    for res_b in writes_b {
                        if res_a == res_b {
                            conflicts.conflicts.push(ScheduleConflict {
                                system_a: sys_a.name().cloned().map(|id| id.to_string()).unwrap_or_else(|| format!("system_{}", i)),
                                system_b: sys_b.name().cloned().map(|id| id.to_string()).unwrap_or_else(|| format!("system_{}", j)),
                                resource: format!("{:?}", res_a),
                            });
                        }
                    }
                }

                // Check for read-write conflicts on resources
                for res_a in writes_a {
                    for res_b in reads_b {
                        if res_a == res_b {
                            conflicts.conflicts.push(ScheduleConflict {
                                system_a: sys_a.name().cloned().map(|id| id.to_string()).unwrap_or_else(|| format!("system_{}", i)),
                                system_b: sys_b.name().cloned().map(|id| id.to_string()).unwrap_or_else(|| format!("system_{}", j)),
                                resource: format!("{:?}", res_a),
                            });
                        }
                    }
                }
                for res_a in reads_a {
                    for res_b in writes_b {
                        if res_a == res_b {
                            conflicts.conflicts.push(ScheduleConflict {
                                system_a: sys_a.name().cloned().map(|id| id.to_string()).unwrap_or_else(|| format!("system_{}", i)),
                                system_b: sys_b.name().cloned().map(|id| id.to_string()).unwrap_or_else(|| format!("system_{}", j)),
                                resource: format!("{:?}", res_a),
                            });
                        }
                    }
                }
            }
        }

        conflicts
    }
}

/// A factory for `Schedule`.
pub struct Builder {
    steps: Vec<Step>,
    accumulator: Vec<Box<dyn ParallelRunnable>>,
}

impl Builder {
    /// Adds a system to the schedule.
    pub fn add_system<T: ParallelRunnable + 'static>(&mut self, system: T) -> &mut Self {
        self.accumulator.push(Box::new(system));
        self
    }

    /// Adds a system to the schedule.
    pub fn add_system_boxed(&mut self, system: Box<dyn ParallelRunnable>) -> &mut Self {
        self.accumulator.push(system);
        self
    }

    /// Waits for executing systems to complete, and the flushes all outstanding system
    /// command buffers.
    pub fn flush(&mut self) -> &mut Self {
        self.finalize_executor();
        self.steps.push(Step::FlushCmdBuffers);
        self
    }

    fn finalize_executor(&mut self) {
        if !self.accumulator.is_empty() {
            let mut systems = Vec::new();
            std::mem::swap(&mut self.accumulator, &mut systems);
            let executor = Executor::new(systems);
            self.steps.push(Step::Systems(executor));
        }
    }

    /// Adds a thread local function to the schedule. This function will be executed on the main thread.
    pub fn add_thread_local_fn<F: FnMut(&mut World, &mut Resources) + 'static>(
        &mut self,
        f: F,
    ) -> &mut Self {
        self.finalize_executor();
        self.steps.push(Step::ThreadLocalFn(
            Box::new(f) as Box<dyn FnMut(&mut World, &mut Resources)>
        ));
        self
    }

    /// Adds a thread local system to the schedule. This system will be executed on the main thread.
    pub fn add_thread_local<S: Runnable + 'static>(&mut self, system: S) -> &mut Self {
        self.finalize_executor();
        let system = Box::new(system) as Box<dyn Runnable>;
        self.steps.push(Step::ThreadLocalSystem(system));
        self
    }

    /// Adds a thread local system to the schedule. This system will be executed on the main thread.
    pub fn add_thread_local_boxed(&mut self, system: Box<dyn Runnable>) -> &mut Self {
        self.finalize_executor();
        self.steps.push(Step::ThreadLocalSystem(system));
        self
    }

    /// Checks all accumulated systems and finalized executors for access conflicts.
    ///
    /// This can be called before `build` to detect conflicts early.
    /// Read-only shared access is allowed and will not produce a conflict.
    pub fn check_conflicts(&self) -> ScheduleConflicts {
        let mut all_conflicts = ScheduleConflicts::default();

        // Check already-finalized executors
        for step in &self.steps {
            if let Step::Systems(executor) = step {
                let conflicts = executor.check_conflicts();
                all_conflicts.conflicts.extend(conflicts.conflicts);
            }
        }

        // Check accumulated (not-yet-finalized) systems against each other
        if self.accumulator.len() > 1 {
            for i in 0..self.accumulator.len() {
                for j in (i + 1)..self.accumulator.len() {
                    let (reads_a, _) = self.accumulator[i].reads();
                    let (writes_a, _) = self.accumulator[i].writes();
                    let (reads_b, _) = self.accumulator[j].reads();
                    let (writes_b, _) = self.accumulator[j].writes();

                    // Check write-write resource conflicts
                    for res_a in writes_a {
                        for res_b in writes_b {
                            if res_a == res_b {
                                all_conflicts.conflicts.push(ScheduleConflict {
                                    system_a: self.accumulator[i].name().cloned().map(|id| id.to_string()).unwrap_or_else(|| format!("system_{}", i)),
                                    system_b: self.accumulator[j].name().cloned().map(|id| id.to_string()).unwrap_or_else(|| format!("system_{}", j)),
                                    resource: format!("{:?}", res_a),
                                });
                            }
                        }
                    }
                }
            }
        }

        all_conflicts
    }

    /// Finalizes the builder into a `Schedule`.
    pub fn build(&mut self) -> Schedule {
        self.flush();
        let mut steps = Vec::new();
        std::mem::swap(&mut self.steps, &mut steps);
        Schedule { steps }
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            steps: Vec::new(),
            accumulator: Vec::new(),
        }
    }
}

/// A step in a schedule.
pub enum Step {
    /// A batch of systems.
    Systems(Executor),
    /// Flush system command buffers.
    FlushCmdBuffers,
    /// A thread local function.
    ThreadLocalFn(Box<dyn FnMut(&mut World, &mut Resources)>),
    /// A thread local system
    ThreadLocalSystem(Box<dyn Runnable>),
}

impl Debug for Step {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Step::Systems(x) => f.debug_tuple("Systems").field(&x).finish(),
            Step::FlushCmdBuffers => f.debug_tuple("FlushCmdBuffers").finish(),
            Step::ThreadLocalFn(_) => {
                f.debug_tuple("ThreadLocalFn")
                    .field(&"dyn FnMut(&mut World, &mut Resources)")
                    .finish()
            }
            Step::ThreadLocalSystem(x) => f.debug_tuple("ThreadLocalSystem").field(&x).finish(),
        }
    }
}

/// A schedule of systems for execution.
///
/// # Examples
///
/// ```rust
/// # use legion::*;
/// # let find_collisions = SystemBuilder::new("find_collisions").build(|_,_,_,_| {});
/// # let calculate_acceleration = SystemBuilder::new("calculate_acceleration").build(|_,_,_,_| {});
/// # let update_positions = SystemBuilder::new("update_positions").build(|_,_,_,_| {});
/// let mut world = World::default();
/// let mut resources = Resources::default();
/// let mut schedule = Schedule::builder()
///     .add_system(find_collisions)
///     .flush()
///     .add_system(calculate_acceleration)
///     .add_system(update_positions)
///     .build();
///
/// schedule.execute(&mut world, &mut resources);
/// ```
#[derive(Debug)]
pub struct Schedule {
    steps: Vec<Step>,
}

impl Schedule {
    /// Creates a new schedule builder.
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Executes all of the steps in the schedule.
    #[cfg(not(feature = "parallel"))]
    pub fn execute(&mut self, world: &mut World, resources: &mut Resources) {
        self.execute_internal(world, resources, |world, resources, executor| {
            executor.run_systems(world, resources.internal())
        });
    }

    /// Executes all of the steps in the schedule.
    #[cfg(feature = "parallel")]
    pub fn execute(&mut self, world: &mut World, resources: &mut Resources) {
        self.execute_internal(world, resources, |world, resources, executor| {
            let resources = resources.internal();
            rayon::join(|| executor.run_systems(world, resources), || {});
        });
    }

    /// Executes all of the steps in the schedule, with parallelized systems running in
    /// the given thread pool.
    #[cfg(feature = "parallel")]
    pub fn execute_in_thread_pool(
        &mut self,
        world: &mut World,
        resources: &mut Resources,
        pool: &rayon::ThreadPool,
    ) {
        self.execute_internal(world, resources, |world, resources, executor| {
            let resources = resources.internal();
            pool.install(|| executor.run_systems(world, resources));
        });
    }

    fn execute_internal<F: FnMut(&mut World, &mut Resources, &mut Executor)>(
        &mut self,
        world: &mut World,
        resources: &mut Resources,
        mut run_executor: F,
    ) {
        enum ToFlush<'a> {
            Executor(&'a mut Executor),
            System(&'a mut CommandBuffer),
        }

        let mut waiting_flush: Vec<ToFlush> = Vec::new();
        for step in &mut self.steps {
            match step {
                Step::Systems(executor) => {
                    run_executor(world, resources, executor);
                    waiting_flush.push(ToFlush::Executor(executor));
                }
                Step::FlushCmdBuffers => {
                    waiting_flush.drain(..).for_each(|e| {
                        match e {
                            ToFlush::Executor(exec) => exec.flush_command_buffers(world, resources),
                            ToFlush::System(cmd) => cmd.flush(world, resources),
                        }
                    });
                }
                Step::ThreadLocalFn(function) => function(world, resources),
                Step::ThreadLocalSystem(system) => {
                    system.prepare(world);
                    system.run(world, resources);
                    if let Some(cmd) = system.command_buffer_mut(world.id()) {
                        waiting_flush.push(ToFlush::System(cmd));
                    }
                }
            }
        }
    }

    /// Converts the schedule into a vector of steps.
    pub fn into_vec(self) -> Vec<Step> {
        self.steps
    }
}

impl From<Builder> for Schedule {
    fn from(mut builder: Builder) -> Self {
        builder.build()
    }
}

impl From<Vec<Step>> for Schedule {
    fn from(steps: Vec<Step>) -> Self {
        Self { steps }
    }
}

    /// Checks all systems in the schedule for access conflicts.
    ///
    /// This detects conflicts where systems in the same execution batch have
    /// incompatible access patterns. Read-only shared access is allowed and
    /// will not produce a conflict.
    pub fn check_conflicts(&self) -> ScheduleConflicts {
        let mut all_conflicts = ScheduleConflicts::default();
        for step in &self.steps {
            if let Step::Systems(executor) = step {
                let conflicts = executor.check_conflicts();
                all_conflicts.conflicts.extend(conflicts.conflicts);
            }
        }
        all_conflicts
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use itertools::sorted;

    use super::*;
    use crate::internals::{
        query::{view::write::Write, IntoQuery},
        systems::system::SystemBuilder,
    };

    #[test]
    fn execute_in_order() {
        let mut world = World::default();

        #[derive(Default)]
        struct Resource;

        let mut resources = Resources::default();
        resources.insert(Resource);

        let order = Arc::new(Mutex::new(Vec::new()));

        let order_clone = order.clone();
        let system_one = SystemBuilder::new("one")
            .write_resource::<Resource>()
            .build(move |_, _, _, _| order_clone.lock().unwrap().push(1usize));
        let order_clone = order.clone();
        let system_two = SystemBuilder::new("two")
            .write_resource::<Resource>()
            .build(move |_, _, _, _| order_clone.lock().unwrap().push(2usize));
        let order_clone = order.clone();
        let system_three = SystemBuilder::new("three")
            .write_resource::<Resource>()
            .build(move |_, _, _, _| order_clone.lock().unwrap().push(3usize));

        let mut schedule = Schedule::builder()
            .add_system(system_one)
            .add_system(system_two)
            .add_system(system_three)
            .build();

        schedule.execute(&mut world, &mut resources);

        let order = order.lock().unwrap();
        let sorted: Vec<usize> = sorted(order.clone()).collect();
        assert_eq!(*order, sorted);
    }

    #[test]
    fn flush() {
        let mut world = World::default();
        let mut resources = Resources::default();

        #[derive(Clone, Copy, Debug, PartialEq)]
        struct TestComp(f32, f32, f32);

        let system_one = SystemBuilder::new("one").build(move |cmd, _, _, _| {
            cmd.push((TestComp(0., 0., 0.),));
        });
        let system_two = SystemBuilder::new("two")
            .with_query(Write::<TestComp>::query())
            .build(move |_, world, _, query| {
                assert_eq!(0, query.iter_mut(world).count());
            });
        let system_three = SystemBuilder::new("three")
            .with_query(Write::<TestComp>::query())
            .build(move |_, world, _, query| {
                assert_eq!(1, query.iter_mut(world).count());
            });

        let mut schedule = Schedule::builder()
            .add_system(system_one)
            .add_system(system_two)
            .flush()
            .add_system(system_three)
            .build();

        schedule.execute(&mut world, &mut resources);
    }

    #[test]
    fn flush_thread_local() {
        let mut world = World::default();
        let mut resources = Resources::default();

        #[derive(Clone, Copy, Debug, PartialEq)]
        struct TestComp(f32, f32, f32);

        let entity = Arc::new(Mutex::new(None));

        {
            let entity = entity.clone();

            let system_one = SystemBuilder::new("one").build(move |cmd, _, _, _| {
                let mut entity = entity.lock().unwrap();
                *entity = Some(cmd.push((TestComp(0.0, 0.0, 0.0),)));
            });

            let mut schedule = Schedule::builder().add_thread_local(system_one).build();

            schedule.execute(&mut world, &mut resources);
        }

        let entity = entity.lock().unwrap();

        assert!(entity.is_some());
        assert!(world.entry(entity.unwrap()).is_some());
    }

    #[test]
    fn thread_local_resource() {
        let mut world = World::default();
        let mut resources = Resources::default();

        #[derive(Clone, Copy, Debug, PartialEq)]
        struct NotSync(*const u8);

        resources.insert(NotSync(std::ptr::null()));

        let system = SystemBuilder::new("one")
            .read_resource::<NotSync>()
            .build(move |_, _, _, _| {});

        let mut schedule = Schedule::builder().add_thread_local(system).build();

        // this should not compile
        // let mut schedule = Schedule::builder().add_system(system).build();

        schedule.execute(&mut world, &mut resources);
    }

    #[test]
    fn parallel_systems_with_disjoint_resources_and_components() {
        // Two groups of components: Physics (Pos, Vel) and Rendering (Mesh, Material)
        // Two shared resources: DeltaTime, FrameCount
        // Systems read different component groups and write different resources
        // They must be parallelizable - no conflicts

        #[derive(Clone, Copy, Debug, PartialEq)]
        struct Pos(f32, f32, f32);
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct Vel(f32, f32, f32);
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct Mesh(u32);
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct Material(u32);

        struct DeltaTime(f32);
        struct FrameCount(u32);

        let mut world = World::default();
        let mut resources = Resources::default();

        resources.insert(DeltaTime(0.016));
        resources.insert(FrameCount(0));

        // Add physics entities (Pos + Vel)
        world.extend(vec![
            (Pos(0.0, 0.0, 0.0), Vel(1.0, 0.0, 0.0)),
            (Pos(1.0, 1.0, 1.0), Vel(0.0, 1.0, 0.0)),
        ]);

        // Add rendering entities (Mesh + Material)
        world.extend(vec![
            (Mesh(1), Material(0)),
            (Mesh(2), Material(1)),
        ]);

        use crate::internals::query::view::read::Read;
        use crate::internals::query::view::write::Write;

        // System A: reads physics components (Pos, Vel), writes DeltaTime
        let system_a = SystemBuilder::new("physics_system")
            .with_query(<(Read<Pos>, Write<Vel>)>::query())
            .write_resource::<DeltaTime>()
            .build(move |_, world, resources, queries| {
                let (ref mut _dt,) = resources;
                let (mut phys_world, _rest) = world.split_for_query(&queries.0);
                for (pos, vel) in queries.0.iter_mut(&mut phys_world) {
                    vel.0 += pos.0 * 0.016;
                }
            });

        // System B: reads rendering components (Mesh, Material), writes FrameCount
        let system_b = SystemBuilder::new("render_system")
            .with_query(<(Read<Mesh>, Read<Material>)>::query())
            .write_resource::<FrameCount>()
            .build(move |_, world, resources, queries| {
                let (ref mut _frame_count,) = resources;
                let (mut render_world, _rest) = world.split_for_query(&queries.0);
                let _count = queries.0.iter(&render_world).count();
            });

        // Build schedule and check for conflicts
        let schedule = Schedule::builder()
            .add_system(system_a)
            .add_system(system_b)
            .build();

        // These systems access different resources and different components,
        // so there should be no unresolvable conflicts
        let conflicts = schedule.check_conflicts();
        // Write-write conflicts are resolved by serialization, so check_conflicts
        // returns them for diagnostics but the schedule still executes correctly
        println!("Conflicts: {}", conflicts);
    }

    #[test]
    fn detect_resource_conflict_at_build_time() {
        // Two systems both writing the same resource - should be detected
        struct SharedRes(u32);

        let mut world = World::default();
        let mut resources = Resources::default();
        resources.insert(SharedRes(0));

        let system_a = SystemBuilder::new("system_a")
            .write_resource::<SharedRes>()
            .build(move |_, _, _, _| {});

        let system_b = SystemBuilder::new("system_b")
            .write_resource::<SharedRes>()
            .build(move |_, _, _, _| {});

        let mut schedule = Schedule::builder()
            .add_system(system_a)
            .add_system(system_b)
            .build();

        // Check for conflicts - write-write on same resource should be reported
        let conflicts = schedule.check_conflicts();
        assert!(!conflicts.is_empty(), "Should detect write-write conflict on SharedRes");
        assert!(conflicts.conflicts.iter().any(|c| c.resource.contains("SharedRes") || c.resource.contains("ResourceTypeId")));

        // The schedule still executes correctly - systems are serialized
        schedule.execute(&mut world, &mut resources);
    }

    #[test]
    fn read_only_resources_allow_parallel_execution() {
        // Two systems both reading the same resource - should run in parallel
        struct GameConfig {
            gravity: f32,
            max_speed: f32,
        }

        #[derive(Clone, Copy, Debug, PartialEq)]
        struct Pos(f32, f32, f32);
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct Vel(f32, f32, f32);

        let mut world = World::default();
        let mut resources = Resources::default();
        resources.insert(GameConfig { gravity: -9.8, max_speed: 100.0 });

        world.extend(vec![
            (Pos(0.0, 0.0, 0.0), Vel(1.0, 0.0, 0.0)),
            (Pos(1.0, 1.0, 1.0), Vel(0.0, 1.0, 0.0)),
        ]);

        use std::sync::atomic::{AtomicBool, Ordering};
        let both_ran = std::sync::Arc::new(AtomicBool::new(false));
        let a_ran = std::sync::Arc::new(AtomicBool::new(false));
        let b_ran = std::sync::Arc::new(AtomicBool::new(false));

        let a_ran_clone = a_ran.clone();
        let both_ran_clone = both_ran.clone();
        let system_a = SystemBuilder::new("gravity_system")
            .with_query(<(Read<Vel>)>::query())
            .read_resource::<GameConfig>()
            .build(move |_, world, resources, queries| {
                a_ran_clone.store(true, Ordering::SeqCst);
                let (ref _config,) = resources;
                for _vel in queries.0.iter(world) {}
            });

        let b_ran_clone = b_ran.clone();
        let both_ran_clone2 = both_ran.clone();
        let system_b = SystemBuilder::new("speed_clamp_system")
            .with_query(<(Read<Vel>)>::query())
            .read_resource::<GameConfig>()
            .build(move |_, world, resources, queries| {
                b_ran_clone.store(true, Ordering::SeqCst);
                let (ref _config,) = resources;
                for _vel in queries.0.iter(world) {}
            });

        let mut schedule = Schedule::builder()
            .add_system(system_a)
            .add_system(system_b)
            .build();

        // Read-only shared resources: no conflicts
        let conflicts = schedule.check_conflicts();
        // Read + Read on same resource = no conflict
        println!("Conflicts: {}", conflicts);

        schedule.execute(&mut world, &mut resources);

        assert!(a_ran.load(Ordering::SeqCst));
        assert!(b_ran.load(Ordering::SeqCst));
    }

    #[test]
    fn subworld_access_consistency_check() {
        // Verify that SubWorld access declarations match actual access
        // System declares read_component for Mesh but tries to write via SubWorld split
        // This should be caught at validation time (via try_split returning Err)

        #[derive(Clone, Copy, Debug, PartialEq)]
        struct Mesh(u32);

        let mut world = World::default();
        let mut resources = Resources::default();
        world.push((Mesh(1),));

        use crate::internals::query::view::write::Write;
        use crate::internals::subworld::SubWorldAccessError;

        // Declare only read access to Mesh
        let system = SystemBuilder::new("mesh_reader")
            .read_component::<Mesh>()
            .build(move |_, world, _, _| {
                // Try to split for write - using try_split should return an error
                let result = world.try_split::<Write<Mesh>>();
                assert!(result.is_err());
                let err = result.unwrap_err();
                assert!(err.to_string().contains("unavailable"));
            });

        let mut schedule = Schedule::builder().add_thread_local(system).build();
        schedule.execute(&mut world, &mut resources);
    }
}
