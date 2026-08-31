use pretraining_g0_contract::{
    identification_diameter, trajectory, value_bounds, AmbiguitySet, Fragment, IndexSet, Norm,
};
use pretraining_term_compiler::{
    ActionRole, Actuation, Actuator, BodyTerm, CalibrationTerm, EnvironmentTerm, GenerationSpec,
    GenerationTemplate, GraphTransition, MissingEdgeBehavior, Morphology, NormTerm, Port,
    PortDirection, PortValue, ScoringTerm, Sensorium, StateSpaceTerm, TwinScope, Visibility,
    Wiring, WorldTerm,
};

const ACTIONS: [u16; 4] = [0, 1, 2, 3];

fn fixture(support: IndexSet, twin: bool) -> WorldTerm {
    let mut transitions = Vec::new();
    let advance = [1, 2, 3, 0, 4];
    let retreat = [3, 4, 1, 2, 1];
    for state in 0..5 {
        transitions.push(GraphTransition {
            source: state,
            actuator: 0,
            destination: advance[state],
        });
        transitions.push(GraphTransition {
            source: state,
            actuator: 1,
            destination: if twin && state <= 2 {
                state
            } else {
                retreat[state]
            },
        });
        transitions.push(GraphTransition {
            source: state,
            actuator: 2,
            destination: state,
        });
    }
    WorldTerm {
        name: if twin { "graph-twin" } else { "graph-witness" }.into(),
        horizon: 2,
        ports: vec![
            Port::new("learner_action", PortDirection::Output, PortValue::Command).public(),
            Port::new("body_command", PortDirection::Input, PortValue::Command).public(),
            Port::new("cell", PortDirection::Output, PortValue::Cell).public(),
            Port::new("reveal", PortDirection::Output, PortValue::Signal).public(),
        ],
        wiring: vec![Wiring {
            from: "learner_action".into(),
            to: "body_command".into(),
        }],
        body: BodyTerm {
            morphology: Morphology { segments: 2 },
            actuation: Actuation {
                command_port: "body_command".into(),
                actuators: vec![
                    Actuator {
                        id: 0,
                        name: "advance".into(),
                        displacement: 0,
                        role: ActionRole::Movement,
                    },
                    Actuator {
                        id: 1,
                        name: "retreat".into(),
                        displacement: 0,
                        role: ActionRole::Movement,
                    },
                    Actuator {
                        id: 2,
                        name: "hold".into(),
                        displacement: 0,
                        role: ActionRole::Hold,
                    },
                    Actuator {
                        id: 3,
                        name: "fallback".into(),
                        displacement: 0,
                        role: ActionRole::Fallback,
                    },
                ],
                supported: support,
            },
            sensorium: Sensorium {
                cell_port: "cell".into(),
                publishes_cell: true,
            },
        },
        environment: EnvironmentTerm::graph(0, 5, transitions, MissingEdgeBehavior::SelfLoop),
        norm: NormTerm {
            expression: Norm::Settle { cell: 4 },
            visibility: Visibility::Public,
        },
        scoring: ScoringTerm {
            goal_reward: 100,
            action_cost: 1,
            fallback_reward: 50,
            violation_penalty: -100,
        },
        calibration: Some(CalibrationTerm {
            pulses: vec![0, 0, 0, 1],
            publishes_cells: true,
        }),
        restorations: Vec::new(),
        disturbance: None,
        scaffold: None,
        couplings: Vec::new(),
        interrupts: Vec::new(),
        restrictions: Vec::new(),
        reveals: Vec::new(),
    }
}

fn sequences() -> Vec<Vec<u16>> {
    pretraining_g0_contract::sequences_of_length(&ACTIONS, 2)
}

#[test]
fn five_state_graph_is_non_cycle_and_missing_edges_self_loop() {
    let term = fixture(IndexSet::from_indices([0, 2, 3]), false);
    let diagnostics = term.environment.topology_diagnostics();
    assert_eq!(diagnostics.state_count, 5);
    assert_eq!(diagnostics.transition_rows, 15);
    assert_eq!(diagnostics.degree_sequence, vec![2, 3, 2, 2, 1]);
    assert!(!diagnostics.is_simple_cycle);

    let compiled = term.compile().expect("graph fixture validates");
    assert_eq!(compiled.audit_metadata().topology, diagnostics);
    for state in 0..5 {
        for action in ACTIONS {
            assert!(compiled.fragment.step(&compiled.contract, state, 0, action) < 5);
        }
    }
    assert_eq!(compiled.fragment.step(&compiled.contract, 4, 0, 3), 4);
    assert_eq!(compiled.fragment.step(&compiled.contract, 4, 0, 2), 4);

    let mut duplicate = fixture(IndexSet::from_indices([0, 2, 3]), false);
    if let StateSpaceTerm::Graph { transitions, .. } = &mut duplicate.environment.state_space {
        transitions.push(GraphTransition {
            source: 0,
            actuator: 0,
            destination: 1,
        });
    }
    assert!(matches!(
        duplicate.compile(),
        Err(pretraining_term_compiler::CompileError::Invalid(_))
    ));
}

#[test]
fn five_state_twin_exhausts_all_sequences_and_identifies_after_calibration() {
    let body = fixture(IndexSet::from_indices([0, 2, 3]), false)
        .compile()
        .expect("body validates");
    let twin = fixture(IndexSet::from_indices([0, 1, 2, 3]), true)
        .compile()
        .expect("twin validates");
    let unrestricted = fixture(IndexSet::from_indices([0, 1, 2, 3]), false)
        .compile()
        .expect("unrestricted validates");
    assert_ne!(body.family_hash, twin.family_hash);
    let all = sequences();
    assert_eq!(all.len(), 16);
    for sequence in &all {
        let body_path = trajectory(&body.fragment, &body.contract, sequence);
        let twin_path = trajectory(&twin.fragment, &twin.contract, sequence);
        assert_eq!(body_path, twin_path, "{sequence:?}");
        assert_eq!(
            body.fragment.value(&body.contract, &body_path, sequence),
            twin.fragment.value(&twin.contract, &twin_path, sequence),
            "{sequence:?}"
        );
    }
    let (body_ceiling, body_optimal) = value_bounds(&body.fragment, &body.contract);
    let (twin_ceiling, twin_optimal) = value_bounds(&twin.fragment, &twin.contract);
    let (unrestricted_ceiling, _) = value_bounds(&unrestricted.fragment, &unrestricted.contract);
    assert!(body_ceiling < unrestricted_ceiling);
    assert_eq!(body_ceiling, twin_ceiling);
    assert_eq!(body_optimal, twin_optimal);
    assert_ne!(body.public_view(&[]).trace, twin.public_view(&[]).trace);

    let mut blind_body = fixture(IndexSet::from_indices([0, 2, 3]), false);
    blind_body.calibration = None;
    let mut blind_unrestricted = fixture(IndexSet::from_indices([0, 1, 2, 3]), false);
    blind_unrestricted.calibration = None;
    let blind_body = blind_body.compile().unwrap();
    let blind_unrestricted = blind_unrestricted.compile().unwrap();
    let blind = AmbiguitySet::uniform(vec![
        blind_body.contract.clone(),
        blind_unrestricted.contract.clone(),
    ]);
    assert_eq!(
        identification_diameter(&blind_body.fragment, &blind, &[]),
        2
    );
    let calibrated =
        AmbiguitySet::uniform(vec![body.contract.clone(), unrestricted.contract.clone()]);
    assert_eq!(identification_diameter(&body.fragment, &calibrated, &[]), 1);
}

#[test]
fn generated_graph_receipt_has_exact_counts_scope_and_replay_hash() {
    let spec = GenerationSpec {
        seed: 11,
        count: 2,
        template: GenerationTemplate::BranchingGraph,
        min_cells: 5,
        max_cells: 5,
        min_horizon: 2,
        max_horizon: 2,
    };
    let first = spec.generate().expect("graph candidates pass receipts");
    let second = spec.generate().expect("replay is deterministic");
    assert_eq!(first, second);
    for family in first {
        assert!(family.receipt.valid);
        assert_eq!(family.receipt.sequences_checked, 16);
        assert!(family.receipt.topology_total);
        assert_eq!(family.receipt.topology_state_count, 5);
        assert_eq!(family.receipt.topology_transition_rows, 9);
        assert_eq!(family.receipt.topology_degree_sequence, vec![1, 3, 2, 2, 2]);
        assert!(!family.receipt.topology_is_simple_cycle);
        assert_eq!(
            family.receipt.twin_scope,
            TwinScope::ConservativeTrajectoryCells
        );
        assert!(family.receipt.twin_publicly_distinct);
        assert_eq!(family.receipt.generator_seed, 11);
        assert!(family.receipt.generator_index < 2);
        assert_eq!(
            family.receipt.family_hash,
            family.body_limited.term.family_hash()
        );
        assert_eq!(
            family.body_limited.term.family_hash(),
            family.body_limited.compile().unwrap().family_hash
        );
    }
}

#[test]
fn graph_coupling_and_ring_displacement_are_typed_rejections() {
    let mut term = fixture(IndexSet::from_indices([0, 1, 2, 3]), false);
    term.couplings
        .push(pretraining_term_compiler::CouplingTerm {
            coupling: pretraining_g0_contract::Coupling::new(
                0,
                pretraining_g0_contract::CouplingRule::Sum,
            ),
            writers: Vec::new(),
            inactive_value: 0,
        });
    assert!(matches!(
        term.compile(),
        Err(pretraining_term_compiler::CompileError::Unsupported(_))
    ));

    let mut ringish_graph = fixture(IndexSet::from_indices([0, 1, 2, 3]), false);
    ringish_graph.body.actuation.actuators[0].displacement = 1;
    assert!(matches!(
        ringish_graph.compile(),
        Err(pretraining_term_compiler::CompileError::Unsupported(_))
    ));

    let ring =
        pretraining_term_compiler::base_term("ring", 5, 2, 4, IndexSet::from_indices([0, 1]));
    let compiled = ring
        .compile()
        .expect("ring compatibility fixture validates");
    assert_eq!(
        compiled.audit_metadata().family_hash_schema_version,
        pretraining_term_compiler::FAMILY_HASH_SCHEMA_VERSION
    );
    assert_eq!(
        ring.legacy_ring_family_hash().as_deref(),
        Some("c202709f5983bda2d930afe7073ad96aec5b4707af1acc69874ce08a0c840987"),
        "version-1 ring encoding remains reproducible for migration"
    );
    assert_ne!(
        ring.family_hash(),
        ring.legacy_ring_family_hash().unwrap(),
        "version-2 hash migration must not be mistaken for unchanged identity"
    );
}
