//! Sample: removing a resource at runtime and re-adding it.
//!
//! A resource removal requested mid-schedule takes effect at the next scheduling
//! boundary (command buffer flush). Systems which declare a dependency on the
//! removed resource are explicitly skipped instead of reading stale state, and
//! re-inserting the resource hands out a new `ResourceVersion`.

use std::sync::{Arc, Mutex};

use legion::{systems::ResourceVersion, Resources, Schedule, SystemBuilder, World};

#[derive(Debug, PartialEq)]
struct Config(i32);

#[test]
fn deferred_resource_removal_and_reinsertion() {
    let mut world = World::default();
    let mut resources = Resources::default();

    // version one of the resource
    resources.insert(Config(1));
    let v1 = resources.version::<Config>().unwrap();
    assert_eq!(v1, ResourceVersion::INITIAL);

    let observed = Arc::new(Mutex::new(Vec::new()));
    let removal_requested = Arc::new(Mutex::new(false));

    // a system which depends on `Config`; records every value it observes
    let reader = {
        let observed = observed.clone();
        SystemBuilder::new("reader")
            .read_resource::<Config>()
            .build(move |_, _, config, _| observed.lock().unwrap().push(config.0))
    };

    // a second dependent system, scheduled after the removal is requested;
    // it must be skipped while the removal is pending
    let late_reader = {
        let observed = observed.clone();
        SystemBuilder::new("late_reader")
            .read_resource::<Config>()
            .build(move |_, _, config, _| observed.lock().unwrap().push(config.0 * 10))
    };

    let mut schedule = Schedule::builder()
        .add_system(reader)
        .add_thread_local_fn(move |_, resources| {
            // request removal mid-schedule (once): takes effect at the next flush boundary
            let mut requested = removal_requested.lock().unwrap();
            if !*requested {
                assert!(resources.remove_deferred::<Config>());
                *requested = true;
            }
        })
        .add_system(late_reader)
        .build();

    schedule.execute(&mut world, &mut resources);

    // the first reader observed version one; the late reader was skipped
    assert_eq!(*observed.lock().unwrap(), vec![1]);

    // the schedule's final flush was the scheduling boundary: the resource is gone
    assert!(!resources.contains::<Config>());
    assert!(!resources.is_available::<Config>());

    // re-insert: version two of the resource
    resources.insert(Config(2));
    let v2 = resources.version::<Config>().unwrap();
    assert!(v2 > v1);

    // dependent systems run again and observe the new version
    schedule.execute(&mut world, &mut resources);
    assert_eq!(*observed.lock().unwrap(), vec![1, 2, 20]);
}

#[test]
fn dependent_systems_skip_while_removal_pending() {
    let mut world = World::default();
    let mut resources = Resources::default();

    resources.insert(Config(1));

    let ran = Arc::new(Mutex::new(Vec::new()));

    // removes the resource via the command buffer; the removal is deferred to
    // the flush at the end of this executor's step
    let remover = SystemBuilder::new("remover").build(|cmd, _, _, _| {
        cmd.exec_mut(|_, resources| {
            resources.remove_deferred::<Config>();
        });
    });

    // depends on `Config`; runs in a later step, after the removal was requested
    let dependent = {
        let ran = ran.clone();
        SystemBuilder::new("dependent")
            .read_resource::<Config>()
            .build(move |_, _, _, _| ran.lock().unwrap().push("dependent"))
    };

    // does not depend on `Config`; must still run
    let independent = {
        let ran = ran.clone();
        SystemBuilder::new("independent").build(move |_, _, _, _| ran.lock().unwrap().push("independent"))
    };

    let mut schedule = Schedule::builder()
        .add_system(remover)
        .flush()
        .add_system(dependent)
        .add_system(independent)
        .build();

    schedule.execute(&mut world, &mut resources);

    // the dependent system was explicitly skipped; the independent one ran
    assert_eq!(*ran.lock().unwrap(), vec!["independent"]);
    assert!(!resources.contains::<Config>());
}
