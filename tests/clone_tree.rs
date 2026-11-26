//! Tests for transactional entity tree cloning with internal reference
//! rewriting, external reference preservation and rollback on clone failure.

use std::error::Error;
use std::fmt;

use legion::world::{CloneError, CloneRegistry};
use legion::*;

/// A plain data component carried by every node.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Label(u32);

/// Points at the node's parent inside the tree.
///
/// This deliberately implements `Clone` but not `Copy`: a component containing
/// an [`Entity`] reference must go through [`Entity`]'s [`Clone`] implementation
/// for internal references to be rewritten, and deriving `Copy` would bypass it.
#[derive(Clone, Debug, PartialEq)]
struct Parent(Entity);

/// Points at the first child of a node.
#[derive(Clone, Debug, PartialEq)]
struct FirstChild(Entity);

/// A reference to an entity outside the cloned tree.
#[derive(Clone, Debug, PartialEq)]
struct ExternalRef(Entity);

/// A second distinct external reference component.
#[derive(Clone, Debug, PartialEq)]
struct ExternalRef2(Entity);

/// A component whose clone can be made to fail on demand.
#[derive(Clone, Debug, PartialEq)]
struct Fragile {
    fail: bool,
    payload: String,
}

#[derive(Debug)]
struct FragileError;

impl fmt::Display for FragileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fragile component refused to clone")
    }
}

impl Error for FragileError {}

/// Builds a three-node tree:
///
/// ```text
/// root (Label 1) --FirstChild--> middle
///  └─ middle (Label 2) --Parent--> root, --FirstChild--> leaf
///      └─ leaf (Label 3) --Parent--> middle
/// ```
///
/// Every node is reachable from `root` through the `FirstChild` edges. Two
/// entities outside the tree are created and referenced from *inside* the tree
/// (one from the root, one from the leaf) through `ExternalRef` components, so
/// that external reference preservation can be asserted.
fn build_sample() -> (World, [Entity; 3], Entity, Entity) {
    let mut world = World::default();

    let root = world.push((Label(1),));
    let middle = world.push((Label(2), Parent(root)));
    let leaf = world.push((Label(3), Parent(middle)));
    world.entry(root).unwrap().add_component(FirstChild(middle));
    world.entry(middle).unwrap().add_component(FirstChild(leaf));

    // two entities outside the tree, referenced from inside the tree
    let external_a = world.push((Label(100),));
    let external_b = world.push((Label(200),));
    world
        .entry(root)
        .unwrap()
        .add_component(ExternalRef(external_a));
    world
        .entry(leaf)
        .unwrap()
        .add_component(ExternalRef(external_b));

    (world, [root, middle, leaf], external_a, external_b)
}

/// Tree edges: follow every `Parent` and `FirstChild` reference stored on an
/// entity. `ExternalRef` deliberately points outside the tree and is therefore
/// not treated as an edge.
fn tree_edges(world: &World, entity: Entity) -> Vec<Entity> {
    let entry = match world.entry_ref(entity) {
        Ok(entry) => entry,
        Err(_) => return Vec::new(),
    };
    let mut edges = Vec::new();
    if let Ok(parent) = entry.get_component::<Parent>() {
        edges.push(parent.0);
    }
    if let Ok(child) = entry.get_component::<FirstChild>() {
        edges.push(child.0);
    }
    edges
}

fn registry() -> CloneRegistry {
    let mut registry = CloneRegistry::new();
    registry.register_copy::<Label>();
    registry.register_clone::<Parent>();
    registry.register_clone::<FirstChild>();
    registry.register_clone::<ExternalRef>();
    registry
}

#[test]
fn clone_tree_rewrites_internal_references() {
    let (mut world, [root, middle, leaf], external_a, external_b) = build_sample();
    let original_len = world.len();

    let mut registry = registry();
    let mapping = world
        .clone_tree(
            [root],
            |world, entity| tree_edges(world, entity),
            &mut registry,
        )
        .unwrap();

    // the sample holds five entities (three tree nodes + two external); only
    // the three tree nodes are cloned, so three new entities are committed
    assert_eq!(world.len(), original_len + 3);
    let cloned_root = mapping[&root];
    let cloned_middle = mapping[&middle];
    let cloned_leaf = mapping[&leaf];

    // all three nodes were cloned with their plain components
    assert_eq!(
        *world
            .entry(cloned_root)
            .unwrap()
            .get_component::<Label>()
            .unwrap(),
        Label(1)
    );
    assert_eq!(
        *world
            .entry(cloned_middle)
            .unwrap()
            .get_component::<Label>()
            .unwrap(),
        Label(2)
    );
    assert_eq!(
        *world
            .entry(cloned_leaf)
            .unwrap()
            .get_component::<Label>()
            .unwrap(),
        Label(3)
    );

    // internal references are rewritten to the corresponding clones
    assert_eq!(
        world
            .entry(cloned_middle)
            .unwrap()
            .get_component::<Parent>()
            .unwrap()
            .0,
        cloned_root
    );
    assert_eq!(
        world
            .entry(cloned_leaf)
            .unwrap()
            .get_component::<Parent>()
            .unwrap()
            .0,
        cloned_middle
    );
    assert_eq!(
        world
            .entry(cloned_root)
            .unwrap()
            .get_component::<FirstChild>()
            .unwrap()
            .0,
        cloned_middle
    );

    // external references keep pointing at their original targets
    assert_eq!(
        world
            .entry(cloned_root)
            .unwrap()
            .get_component::<ExternalRef>()
            .unwrap()
            .0,
        external_a
    );
    assert_eq!(
        world
            .entry(cloned_leaf)
            .unwrap()
            .get_component::<ExternalRef>()
            .unwrap()
            .0,
        external_b
    );
}

#[test]
fn clone_tree_preserves_external_references_explicitly() {
    let mut world = World::default();

    // external targets
    let external_x = world.push((Label(900),));
    let external_y = world.push((Label(901),));

    // single tree node referencing both external entities. ExternalRef is not
    // treated as a tree edge, so neither target becomes part of the tree.
    let root = world.push((Label(1), ExternalRef(external_x), ExternalRef2(external_y)));

    let mut registry = CloneRegistry::new();
    registry.register_copy::<Label>();
    registry.register_clone::<ExternalRef>();
    registry.register_clone::<ExternalRef2>();
    let mapping = world
        .clone_tree([root], |_, _| Vec::<Entity>::new(), &mut registry)
        .unwrap();

    assert_eq!(mapping.len(), 1);
    let cloned = mapping[&root];
    let entry = world.entry(cloned).unwrap();

    // neither external entity was cloned, so both references stay unchanged
    assert_eq!(entry.get_component::<ExternalRef>().unwrap().0, external_x);
    assert_eq!(entry.get_component::<ExternalRef2>().unwrap().0, external_y);
    assert!(world.contains(external_x));
    assert!(world.contains(external_y));
}

#[test]
fn clone_tree_keeps_original_tree_untouched() {
    let (mut world, [root, middle, leaf], external_a, external_b) = build_sample();

    let mut registry = registry();
    let _mapping = world
        .clone_tree(
            [root],
            |world, entity| tree_edges(world, entity),
            &mut registry,
        )
        .unwrap();

    // original entities and their references are unchanged
    assert_eq!(
        world
            .entry(middle)
            .unwrap()
            .get_component::<Parent>()
            .unwrap()
            .0,
        root
    );
    assert_eq!(
        world
            .entry(leaf)
            .unwrap()
            .get_component::<Parent>()
            .unwrap()
            .0,
        middle
    );
    assert_eq!(
        world
            .entry(root)
            .unwrap()
            .get_component::<FirstChild>()
            .unwrap()
            .0,
        middle
    );
    assert_eq!(
        world
            .entry(root)
            .unwrap()
            .get_component::<ExternalRef>()
            .unwrap()
            .0,
        external_a
    );
    assert_eq!(
        world
            .entry(leaf)
            .unwrap()
            .get_component::<ExternalRef>()
            .unwrap()
            .0,
        external_b
    );
}

#[test]
fn clone_tree_rolls_back_when_a_component_clone_fails() {
    let mut world = World::default();

    let root = world.push((Label(1),));
    let child = world.push((
        Label(2),
        Parent(root),
        Fragile {
            fail: true,
            payload: "will fail".to_string(),
        },
    ));
    // make the child reachable from the root so its Fragile component is cloned
    world.entry(root).unwrap().add_component(FirstChild(child));

    let before = world.len();
    let root_components_before = world
        .entry(root)
        .unwrap()
        .archetype()
        .layout()
        .component_types()
        .len();

    let mut registry = CloneRegistry::new();
    registry.register_copy::<Label>();
    registry.register_clone::<Parent>();
    registry.register_clone::<FirstChild>();
    registry.register_try_clone::<Fragile, _, _>(|fragile| {
        if fragile.fail {
            Err(FragileError)
        } else {
            Ok(fragile.clone())
        }
    });

    let result = world.clone_tree(
        [root],
        |world, entity| tree_edges(world, entity),
        &mut registry,
    );

    match result {
        Err(CloneError::ComponentCloneFailed { .. }) => {}
        other => panic!("expected ComponentCloneFailed, got {:?}", other),
    }

    // nothing was committed: entity count and existing entities are unchanged
    assert_eq!(world.len(), before);
    assert!(world.contains(root));
    assert!(world.contains(child));

    // the original data is still intact
    assert_eq!(
        *world.entry(root).unwrap().get_component::<Label>().unwrap(),
        Label(1)
    );
    assert_eq!(
        world
            .entry(child)
            .unwrap()
            .get_component::<Parent>()
            .unwrap()
            .0,
        root
    );
    assert_eq!(
        world
            .entry(child)
            .unwrap()
            .get_component::<Fragile>()
            .unwrap()
            .payload,
        "will fail"
    );
    let root_components_after = world
        .entry(root)
        .unwrap()
        .archetype()
        .layout()
        .component_types()
        .len();
    assert_eq!(root_components_before, root_components_after);
}

#[test]
fn clone_tree_leaves_world_and_mapping_intact_on_panic() {
    /// A component whose clone always panics rather than returning an error.
    #[derive(Clone, Debug, PartialEq)]
    struct Explodes;

    let mut world = World::default();
    let root = world.push((Label(1),));
    let child = world.push((Label(2), Parent(root), Explodes));
    world.entry(root).unwrap().add_component(FirstChild(child));
    let before = world.len();

    let mut registry = CloneRegistry::new();
    registry.register_copy::<Label>();
    registry.register_clone::<Parent>();
    registry.register_clone::<FirstChild>();
    registry.register_try_clone::<Explodes, FragileError, _>(|_| panic!("clone exploded"));

    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world
            .clone_tree([root], |w, e| tree_edges(w, e), &mut registry)
            .unwrap();
    }));
    assert!(panic.is_err());

    // the world was not modified by the aborted clone
    assert_eq!(world.len(), before);
    assert!(world.contains(root));
    assert!(world.contains(child));

    // the thread-local mapping was restored: a bare Entity::clone keeps its
    // identity, and a subsequent non-panicking clone still works
    let id = world.push((Label(9),));
    assert_eq!(id, id.clone());

    let mut healthy = CloneRegistry::new();
    healthy.register_copy::<Label>();
    healthy.register_clone::<Parent>();
    healthy.register_clone::<FirstChild>();
    // Explodes is intentionally left unregistered, so it is skipped
    let mapping = world
        .clone_tree([root], |w, e| tree_edges(w, e), &mut healthy)
        .unwrap();
    assert_eq!(mapping.len(), 2);
    assert_eq!(
        world
            .entry(mapping[&child])
            .unwrap()
            .get_component::<Parent>()
            .unwrap()
            .0,
        mapping[&root]
    );
}

#[test]
fn clone_tree_succeeds_when_fallible_clone_passes() {
    let mut world = World::default();
    let root = world.push((
        Label(1),
        Fragile {
            fail: false,
            payload: "ok".to_string(),
        },
    ));

    let mut registry = CloneRegistry::new();
    registry.register_copy::<Label>();
    registry.register_try_clone::<Fragile, _, _>(|fragile| {
        if fragile.fail {
            Err(FragileError)
        } else {
            Ok(fragile.clone())
        }
    });

    let mapping = world
        .clone_tree([root], |_, _| Vec::<Entity>::new(), &mut registry)
        .unwrap();
    let cloned = mapping[&root];
    assert_eq!(
        world
            .entry(cloned)
            .unwrap()
            .get_component::<Fragile>()
            .unwrap(),
        &Fragile {
            fail: false,
            payload: "ok".to_string()
        }
    );
}

#[test]
fn clone_tree_reports_missing_entity_without_changes() {
    let mut world = World::default();
    let root = world.push((Label(1),));
    let before = world.len();

    let mut registry = registry();
    let missing = {
        // allocate an ID then remove the entity so it cannot resolve
        let id = world.push((Label(999),));
        world.remove(id);
        id
    };

    let result = world.clone_tree([root, missing], |_, _| Vec::<Entity>::new(), &mut registry);

    match result {
        Err(CloneError::EntityNotFound(entity)) => assert_eq!(entity, missing),
        other => panic!("expected EntityNotFound, got {:?}", other),
    }
    assert_eq!(world.len(), before);
    assert!(world.contains(root));
}

#[test]
fn clone_tree_is_cycle_safe_and_deduplicates() {
    let mut world = World::default();

    // two mutually referencing entities form a cycle
    let a = world.push((Label(1),));
    let b = world.push((Label(2), Parent(a)));
    world.entry(a).unwrap().add_component(FirstChild(b));

    let mut registry = registry();
    let mapping = world
        .clone_tree(
            [a],
            |world, entity| tree_edges(world, entity),
            &mut registry,
        )
        .unwrap();

    // each entity is cloned exactly once despite the cycle
    assert_eq!(mapping.len(), 2);
    let cloned_a = mapping[&a];
    let cloned_b = mapping[&b];
    assert_ne!(cloned_a, cloned_b);

    // the cloned cycle points entirely inside the new tree
    assert_eq!(
        world
            .entry(cloned_b)
            .unwrap()
            .get_component::<Parent>()
            .unwrap()
            .0,
        cloned_a
    );
    assert_eq!(
        world
            .entry(cloned_a)
            .unwrap()
            .get_component::<FirstChild>()
            .unwrap()
            .0,
        cloned_b
    );
}

#[test]
fn clone_tree_skips_unregistered_components() {
    let mut world = World::default();
    let root = world.push((Label(1),));
    world.entry(root).unwrap().add_component(Fragile {
        fail: true, // must never be invoked because the type is not registered
        payload: "ignored".to_string(),
    });

    let mut registry = CloneRegistry::new();
    registry.register_copy::<Label>();

    let mapping = world
        .clone_tree([root], |_, _| Vec::<Entity>::new(), &mut registry)
        .unwrap();
    let cloned = mapping[&root];

    // the clone carries Label but not the unregistered Fragile component
    let entry = world.entry(cloned).unwrap();
    assert!(entry.get_component::<Label>().is_ok());
    assert!(entry.get_component::<Fragile>().is_err());
}

#[test]
fn clone_tree_restores_thread_local_mapping_after_use() {
    let mut world = World::default();
    let root = world.push((Label(1),));

    let mut registry = registry();
    world
        .clone_tree([root], |_, _| Vec::<Entity>::new(), &mut registry)
        .unwrap();

    // after cloning, a bare Entity::clone outside a clone context must yield
    // the same ID (external behavior of Entity is restored)
    let id = world.push((Label(2),));
    assert_eq!(id, id.clone());
}

/// Mirrors the `World::clone_tree` doc example, whose doctest cannot be run in
/// this environment because of a pre-existing unrelated compile error in the
/// crate's unit-test harness.
#[test]
fn clone_tree_doc_example_semantics() {
    #[derive(Clone, Copy)]
    struct DocLabel(&'static str);
    #[derive(Clone)]
    struct DocParent(Entity);
    #[derive(Clone)]
    struct DocChild(Entity);

    let mut world = World::default();
    let root = world.push((DocLabel("root"),));
    let child = world.push((DocLabel("child"), DocParent(root)));
    world.entry(root).unwrap().add_component(DocChild(child));

    let mut registry = CloneRegistry::new();
    registry.register_clone::<DocLabel>();
    registry.register_clone::<DocParent>();
    registry.register_clone::<DocChild>();

    let mapping = world
        .clone_tree(
            [root],
            |world, entity| {
                let entry = match world.entry_ref(entity) {
                    Ok(entry) => entry,
                    Err(_) => return Vec::new(),
                };
                let mut edges = Vec::new();
                if let Ok(parent) = entry.get_component::<DocParent>() {
                    edges.push(parent.0);
                }
                if let Ok(c) = entry.get_component::<DocChild>() {
                    edges.push(c.0);
                }
                edges
            },
            &mut registry,
        )
        .unwrap();

    let cloned_child = mapping[&child];
    let cloned_parent = world
        .entry(cloned_child)
        .unwrap()
        .get_component::<DocParent>()
        .unwrap()
        .0;
    assert_eq!(cloned_parent, mapping[&root]);
}
