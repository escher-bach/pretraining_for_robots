use std::collections::{BTreeMap, BTreeSet};

use pretraining_g0_contract::optimal_first_actions;
use pretraining_term_compiler::{
    ActionRole, G0CorpusSpec, G0FeatureSchedule, G0Topology, PlanLengthDistribution, PublicEvent,
    StateSpaceTerm,
};

#[test]
fn plan_length_quotas_are_exact_and_tie_break_shorter_first() {
    let distribution = PlanLengthDistribution {
        weights: vec![1, 2, 1],
    };
    assert_eq!(distribution.quotas(10).unwrap(), vec![3, 5, 2]);
    assert_eq!(
        PlanLengthDistribution {
            weights: vec![1, 1, 1],
        }
        .quotas(2)
        .unwrap(),
        vec![1, 1, 0]
    );
    assert_eq!(
        PlanLengthDistribution {
            weights: vec![0, 2, 0],
        }
        .quotas(7)
        .unwrap(),
        vec![0, 7, 0]
    );
}

#[test]
fn generated_chain_has_the_declared_non_degenerate_length_histogram() {
    let worlds = G0CorpusSpec {
        seed: 100,
        count: 6,
        topology: G0Topology::Chain,
        feature_schedule: pretraining_term_compiler::G0FeatureSchedule::Plain,
        plan_length_distribution: PlanLengthDistribution {
            weights: vec![1, 1, 1],
        },
        min_states: 4,
        max_states: 4,
        min_horizon: 3,
        max_horizon: 3,
    }
    .generate()
    .expect("chain corpus is valid");
    let lengths: BTreeMap<usize, usize> = worlds
        .iter()
        .filter_map(|world| {
            world
                .compile()
                .expect("world lowers")
                .shortest_non_fallback_success_length()
        })
        .fold(BTreeMap::new(), |mut counts, length| {
            *counts.entry(length).or_default() += 1;
            counts
        });
    assert_eq!(lengths, BTreeMap::from([(1, 2), (2, 2), (3, 2)]));
}

#[test]
fn every_declared_topology_generates_a_replayable_exact_route() {
    let topologies = [
        G0Topology::Chain,
        G0Topology::Star,
        G0Topology::Tree,
        G0Topology::Dag,
        G0Topology::Grid,
        G0Topology::Lattice,
        G0Topology::Graph,
        G0Topology::Ring,
    ];
    for (seed, topology) in topologies.into_iter().enumerate() {
        let spec = G0CorpusSpec {
            seed: seed as u64,
            count: 2,
            topology,
            feature_schedule: pretraining_term_compiler::G0FeatureSchedule::Plain,
            plan_length_distribution: PlanLengthDistribution {
                weights: vec![1, 1],
            },
            min_states: 3,
            max_states: 5,
            min_horizon: 2,
            max_horizon: 2,
        };
        let worlds = spec.generate().expect("topology generates");
        assert_eq!(worlds, spec.generate().expect("topology replays"));
        assert!(worlds.iter().all(|world| {
            world
                .compile()
                .expect("topology lowers")
                .shortest_non_fallback_success_length()
                .is_some()
        }));
    }
}

#[test]
fn topology_labels_match_their_structural_edge_families() {
    let make = |topology| {
        G0CorpusSpec {
            seed: 19,
            count: 1,
            topology,
            feature_schedule: pretraining_term_compiler::G0FeatureSchedule::Plain,
            plan_length_distribution: PlanLengthDistribution { weights: vec![1] },
            min_states: 6,
            max_states: 6,
            min_horizon: 1,
            max_horizon: 1,
        }
        .generate()
        .expect("structural sample")
        .remove(0)
    };
    let graph_edges = |world: &pretraining_term_compiler::GeneratedWorld| {
        let StateSpaceTerm::Graph { transitions, .. } = &world.term.environment.state_space else {
            return BTreeSet::new();
        };
        transitions
            .iter()
            .map(|edge| {
                (
                    edge.source.min(edge.destination),
                    edge.source.max(edge.destination),
                )
            })
            .collect::<BTreeSet<_>>()
    };

    let chain = graph_edges(&make(G0Topology::Chain));
    assert!(chain.contains(&(0, 1)) && chain.contains(&(4, 5)));
    let star = graph_edges(&make(G0Topology::Star));
    assert_eq!(star.len(), 5);
    assert!(star
        .iter()
        .all(|(source, destination)| *source == 0 || *destination == 0));
    let tree = graph_edges(&make(G0Topology::Tree));
    assert_eq!(tree.len(), 5, "a tree has states - 1 undirected edges");
    let dag = &make(G0Topology::Dag).term.environment.state_space;
    if let StateSpaceTerm::Graph { transitions, .. } = dag {
        assert!(transitions
            .iter()
            .all(|edge| edge.source < edge.destination));
    } else {
        panic!("DAG must lower to Graph");
    }
    let grid = graph_edges(&make(G0Topology::Grid));
    let lattice = graph_edges(&make(G0Topology::Lattice));
    assert!(
        lattice.len() > grid.len(),
        "toroidal lattice adds wrap edges"
    );
    assert!(matches!(
        make(G0Topology::Ring).term.environment.state_space,
        StateSpaceTerm::Ring { .. }
    ));
}

#[test]
fn corpus_is_seed_reproducible_and_each_world_lowers() {
    let spec = G0CorpusSpec {
        seed: 1234,
        count: 32,
        ..G0CorpusSpec::default()
    };
    let first = spec.generate().expect("generated corpus is valid");
    let second = spec.generate().expect("same seed replays");
    assert_eq!(first, second);
    assert_eq!(first.len(), 32);
    for (index, world) in first.iter().enumerate() {
        assert_eq!(world.generator_metadata.seed, 1234);
        assert!(world.generator_metadata.index >= index);
        let report = G0CorpusSpec::assess_world(world).expect("admission report");
        assert!(report.accepted, "{report:?}");
        assert!(world.term.compile().is_ok());
    }

    let hashes: BTreeSet<_> = first.iter().map(|world| world.term.family_hash()).collect();
    assert_eq!(hashes.len(), first.len(), "semantic worlds must be unique");
    assert!(hashes.len() > 1, "corpus should not collapse to one world");
    let has_ring = first.iter().any(|world| {
        matches!(
            world.term.environment.state_space,
            pretraining_term_compiler::StateSpaceTerm::Ring { .. }
        )
    });
    let has_graph = first.iter().any(|world| {
        matches!(
            world.term.environment.state_space,
            pretraining_term_compiler::StateSpaceTerm::Graph { .. }
        )
    });
    assert!(
        has_ring && has_graph,
        "mixed corpus should contain both topologies"
    );
}

#[test]
fn mixed_corpus_does_not_bind_forward_motion_to_one_action_id() {
    let worlds = G0CorpusSpec {
        seed: 17,
        count: 64,
        ..G0CorpusSpec::default()
    }
    .generate()
    .expect("mixed corpus is valid");
    let mut optimal_ids = BTreeSet::new();
    for world in worlds {
        let compiled = world.compile().expect("world lowers");
        optimal_ids.extend(optimal_first_actions(
            &compiled.fragment,
            &compiled.contract,
        ));
    }
    assert!(optimal_ids.contains(&0));
    assert!(optimal_ids.contains(&1));
}

#[test]
fn corpus_can_hold_one_topology_and_rejects_invalid_ranges() {
    let graph = G0CorpusSpec {
        seed: 7,
        count: 4,
        topology: G0Topology::Graph,
        feature_schedule: pretraining_term_compiler::G0FeatureSchedule::Plain,
        plan_length_distribution: PlanLengthDistribution {
            weights: vec![1, 1],
        },
        min_states: 5,
        max_states: 5,
        min_horizon: 2,
        max_horizon: 2,
    }
    .generate()
    .expect("graph corpus is valid");
    assert!(graph.iter().all(|world| matches!(
        world.term.environment.state_space,
        pretraining_term_compiler::StateSpaceTerm::Graph { .. }
    )));

    let invalid = G0CorpusSpec {
        min_states: 1,
        ..G0CorpusSpec::default()
    };
    assert!(invalid.generate().is_err());
}

#[test]
fn fixed_seed_corpus_varies_action_count_and_ring_meanings() {
    let worlds = G0CorpusSpec {
        seed: 0xC0FFEE,
        count: 24,
        topology: G0Topology::Ring,
        feature_schedule: pretraining_term_compiler::G0FeatureSchedule::Plain,
        plan_length_distribution: PlanLengthDistribution {
            weights: vec![1, 1],
        },
        min_states: 6,
        max_states: 6,
        min_horizon: 2,
        max_horizon: 2,
    }
    .generate()
    .expect("fixed-seed ring corpus is valid");
    let action_counts: BTreeSet<_> = worlds
        .iter()
        .map(|world| world.term.body.actuation.actuators.len())
        .collect();
    assert!(
        action_counts.len() >= 2,
        "action count should vary: {action_counts:?}"
    );
    assert!(action_counts.iter().all(|count| (4..=6).contains(count)));
    let displacements: BTreeSet<_> = worlds
        .iter()
        .flat_map(|world| {
            world
                .term
                .body
                .actuation
                .actuators
                .iter()
                .filter(|actuator| actuator.role == ActionRole::Movement)
                .map(|actuator| actuator.displacement)
        })
        .collect();
    assert_eq!(displacements, BTreeSet::from([-1, 1]));
}

#[test]
fn fixed_seed_graphs_have_sampled_branching_rows() {
    let worlds = G0CorpusSpec {
        seed: 0xBAD5EED,
        count: 12,
        topology: G0Topology::Graph,
        feature_schedule: pretraining_term_compiler::G0FeatureSchedule::Plain,
        plan_length_distribution: PlanLengthDistribution {
            weights: vec![1, 1, 1],
        },
        min_states: 6,
        max_states: 6,
        min_horizon: 3,
        max_horizon: 3,
    }
    .generate()
    .expect("fixed-seed graph corpus is valid");
    assert!(worlds.iter().all(|world| {
        let StateSpaceTerm::Graph {
            states,
            transitions,
            ..
        } = &world.term.environment.state_space
        else {
            return false;
        };
        let mut outgoing = BTreeMap::<usize, usize>::new();
        for transition in transitions {
            *outgoing.entry(transition.source).or_default() += 1;
        }
        transitions.len() > *states && outgoing.values().any(|degree| *degree >= 2)
    }));
}

#[test]
fn fixed_seed_replay_and_action_ids_are_diverse() {
    let spec = G0CorpusSpec {
        seed: 91,
        count: 24,
        topology: G0Topology::Mixed,
        feature_schedule: pretraining_term_compiler::G0FeatureSchedule::Plain,
        plan_length_distribution: PlanLengthDistribution {
            weights: vec![1, 1, 1],
        },
        min_states: 5,
        max_states: 7,
        min_horizon: 2,
        max_horizon: 3,
    };
    let first = spec.generate().expect("fixed-seed corpus is valid");
    assert_eq!(first, spec.generate().expect("replay is valid"));
    let movement_ids: BTreeSet<_> = first
        .iter()
        .flat_map(|world| {
            world
                .term
                .body
                .actuation
                .actuators
                .iter()
                .filter(|actuator| actuator.role == ActionRole::Movement)
                .map(|actuator| actuator.id)
        })
        .collect();
    assert!(movement_ids.len() > 4, "movement IDs should be permuted");
}

#[test]
fn coverage_schedule_emits_each_isolated_construct_without_changing_topology_or_length() {
    let spec = G0CorpusSpec {
        seed: 0xC0DE,
        count: 11,
        topology: G0Topology::Mixed,
        feature_schedule: G0FeatureSchedule::Coverage,
        plan_length_distribution: PlanLengthDistribution { weights: vec![1] },
        min_states: 4,
        max_states: 8,
        min_horizon: 1,
        max_horizon: 1,
    };
    let worlds = spec.generate().expect("coverage schedule is feasible");
    let mut arms = BTreeSet::new();
    for world in &worlds {
        let mut plain = world.term.clone();
        plain.calibration = None;
        plain.restorations.clear();
        plain.disturbance = None;
        plain.scaffold = None;
        plain.couplings.clear();
        plain.interrupts.clear();
        plain.restrictions.clear();
        plain.reveals.clear();
        assert_eq!(
            world.term.environment.topology_diagnostics(),
            plain.environment.topology_diagnostics()
        );
        assert_eq!(
            world
                .compile()
                .expect("coverage world lowers")
                .shortest_optimal_non_fallback_plan_length(),
            Some(1)
        );
        let arm = if !world.term.interrupts.is_empty() {
            "interrupt"
        } else if world.term.calibration.is_some() {
            "calibration"
        } else if !world.term.restorations.is_empty() {
            "restoration"
        } else if world.term.disturbance.is_some() {
            "disturbance"
        } else if world.term.scaffold.is_some() {
            "scaffold"
        } else if !world.term.couplings.is_empty() {
            "coupling"
        } else if !world.term.reveals.is_empty() {
            "reveal"
        } else if world
            .term
            .restrictions
            .iter()
            .any(|restriction| restriction.kind() == "action")
        {
            "restriction-action"
        } else if world
            .term
            .restrictions
            .iter()
            .any(|restriction| restriction.kind() == "viability")
        {
            "restriction-viability"
        } else if world
            .term
            .restrictions
            .iter()
            .any(|restriction| restriction.kind() == "resource")
        {
            "restriction-resource"
        } else {
            "plain"
        };
        arms.insert(arm);
        let action = world
            .term
            .body
            .actuation
            .actuators
            .iter()
            .find(|actuator| actuator.role == ActionRole::Movement)
            .map(|actuator| actuator.id)
            .into_iter()
            .collect::<Vec<_>>();
        let events = world.compile().unwrap().public_event_view(&action);
        match arm {
            "calibration" => assert!(events.groups.iter().any(|group| {
                group.events.iter().any(|event| {
                    matches!(
                        event,
                        PublicEvent::Boundary(
                            pretraining_term_compiler::PublicBoundary::CalibrationReset
                        )
                    )
                })
            })),
            "restoration" => assert!(events.groups.iter().any(|group| {
                group
                    .events
                    .iter()
                    .any(|event| matches!(event, PublicEvent::RestorationAnnouncement { .. }))
            })),
            "disturbance" | "scaffold" | "reveal" => assert!(events.groups.iter().any(|group| {
                group
                    .events
                    .iter()
                    .any(|event| matches!(event, PublicEvent::Signal { .. }))
            })),
            _ => {}
        }
    }
    assert_eq!(
        arms,
        BTreeSet::from([
            "calibration",
            "coupling",
            "disturbance",
            "interrupt",
            "plain",
            "restoration",
            "restriction-action",
            "restriction-resource",
            "restriction-viability",
            "reveal",
            "scaffold",
        ])
    );
}

#[test]
fn ring_transitions_are_local_and_protected_route_edges_survive_random_blocks() {
    let worlds = G0CorpusSpec {
        seed: 0x51DE,
        count: 4,
        topology: G0Topology::Ring,
        feature_schedule: G0FeatureSchedule::Plain,
        plan_length_distribution: PlanLengthDistribution {
            weights: vec![1, 1],
        },
        min_states: 6,
        max_states: 6,
        min_horizon: 2,
        max_horizon: 2,
    }
    .generate()
    .expect("ring worlds are feasible");
    for world in worlds {
        let compiled = world.compile().expect("ring lowers");
        let table = compiled.public_transition_table();
        let states = table.states;
        for row in &table.rows {
            if row.destination != row.address.source {
                let distance = row.destination.abs_diff(row.address.source);
                let cyclic = distance.min(states - distance);
                assert_eq!(cyclic, 1, "non-self ring transition must be local: {row:?}");
            }
        }
        let StateSpaceTerm::Ring { .. } = &world.term.environment.state_space else {
            unreachable!();
        };
        let start = world.term.environment.start;
        let goal = match world.term.norm.expression {
            pretraining_g0_contract::Norm::Settle { cell } => cell,
            _ => unreachable!(),
        };
        let route = world
            .term
            .body
            .actuation
            .actuators
            .iter()
            .find(|actuator| actuator.role == ActionRole::Movement)
            .expect("ring has a route actuator")
            .id;
        let mut cell = start;
        for _ in 0..states {
            if cell == goal {
                break;
            }
            let row = table
                .rows
                .iter()
                .find(|row| {
                    row.address.executed == 0
                        && row.address.source == cell
                        && row.address.actuator == route
                })
                .expect("route transition row exists");
            assert_ne!(row.destination, cell, "protected route edge was blocked");
            cell = row.destination;
        }
        assert_eq!(cell, goal, "protected route must reach the declared goal");
    }
}

#[test]
fn interrupt_event_has_visible_suppressed_and_resumed_phases_against_control() {
    let spec = G0CorpusSpec {
        seed: 0x1A7E,
        count: 7,
        topology: G0Topology::Mixed,
        feature_schedule: G0FeatureSchedule::Coverage,
        plan_length_distribution: PlanLengthDistribution { weights: vec![1] },
        min_states: 4,
        max_states: 8,
        min_horizon: 2,
        max_horizon: 2,
    };
    let world = spec
        .generate()
        .expect("coverage prefix is feasible")
        .into_iter()
        .find(|world| !world.term.interrupts.is_empty())
        .expect("coverage emits an interrupt arm");
    let route = world
        .term
        .body
        .actuation
        .actuators
        .iter()
        .find(|actuator| actuator.role == ActionRole::Movement)
        .expect("route actuator")
        .id;
    let alternate = world
        .term
        .body
        .actuation
        .actuators
        .iter()
        .find(|actuator| actuator.role == ActionRole::Movement && actuator.id != route)
        .expect("alternate actuator")
        .id;
    let compiled = world.compile().expect("interrupt world lowers");
    let mut control = world.term.clone();
    control.interrupts.clear();
    let control_environment = control.environment.clone();
    let control = control.compile().expect("matched control lowers");
    let signal_values = |view: &pretraining_term_compiler::PublicEventView| {
        view.groups
            .iter()
            .flat_map(|group| group.events.iter())
            .filter_map(|event| match event {
                PublicEvent::Signal { value, .. } => Some(*value),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        signal_values(&compiled.public_event_view(&[route, alternate])),
        vec![9_001, 9_001]
    );
    assert_eq!(
        signal_values(&control.public_event_view(&[route, alternate])),
        vec![9_001, 9_001, 9_001]
    );
    assert_eq!(
        compiled.shortest_optimal_non_fallback_plan_length(),
        control.shortest_optimal_non_fallback_plan_length()
    );
    assert_eq!(
        world.term.environment.topology_diagnostics(),
        control_environment.topology_diagnostics()
    );
}
