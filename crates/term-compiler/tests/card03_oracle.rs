//! The executable portion of the Card 03 oracle.
//!
//! These tests deliberately stop at the semantics the current generic compiler
//! can represent.  They do not weaken the full oracle in
//! `experiments/term-compiler/CARD03-ORACLE.md`: calibration prelude state,
//! support restoration, Card 03 scoring, and learner rendering remain explicit
//! follow-up gates rather than being approximated here.

use pretraining_card03_affordance::{
    all_sequences, body_environment_swap, card_cases, full_body, run, Action, Affordance, Contract,
    Outcome,
};
use pretraining_g0_contract::{trajectory, Fragment, Norm};
use pretraining_term_compiler::{
    Actuation, Actuator, BodyTerm, EnvironmentTerm, Morphology, NormTerm, Port, PortDirection,
    PortValue, Sensorium, Visibility, Wiring, WorldTerm,
};

const CELLS: usize = 9;
const HORIZON: usize = 2;
const ACTIONS: [Action; 5] = [
    Action::Hold,
    Action::Step,
    Action::Leap,
    Action::Back,
    Action::Fallback,
];

fn actuator(action: Action) -> Actuator {
    Actuator {
        id: action.index() as u16,
        name: action.name().to_string(),
        displacement: action.displacement() as i32,
    }
}

fn term_for(name: &str, contract: &Contract) -> WorldTerm {
    // Hold and Fallback are always executable in the handwritten card; only
    // movement support belongs to the body restriction.
    let mut supported = contract.support;
    supported.insert(Action::Hold.index());
    supported.insert(Action::Fallback.index());

    WorldTerm {
        name: name.to_string(),
        horizon: HORIZON,
        ports: vec![
            Port::new("learner_action", PortDirection::Output, PortValue::Command).public(),
            Port::new("body_command", PortDirection::Input, PortValue::Command).public(),
            Port::new("cell", PortDirection::Output, PortValue::Cell).public(),
            // The signal is present so the term has a typed home for a future
            // reveal implementation; static cases intentionally emit none.
            Port::new("reveal", PortDirection::Output, PortValue::Signal).public(),
        ],
        wiring: vec![Wiring {
            from: "learner_action".into(),
            to: "body_command".into(),
        }],
        body: BodyTerm {
            morphology: Morphology { cells: CELLS },
            actuation: Actuation {
                command_port: "body_command".into(),
                actuators: ACTIONS.into_iter().map(actuator).collect(),
                supported,
            },
            sensorium: Sensorium {
                cell_port: "cell".into(),
                publishes_cell: true,
            },
        },
        environment: EnvironmentTerm {
            start: contract.start,
            blocked_edges: contract
                .blocked_edges
                .iter()
                .map(|(cell, action)| (*cell, *action as u16))
                .collect(),
        },
        norm: NormTerm {
            expression: Norm::Settle {
                cell: contract.goal,
            },
            visibility: Visibility::Public,
        },
        disturbance: None,
        scaffold: None,
        couplings: Vec::new(),
        interrupts: Vec::new(),
        restrictions: Vec::new(),
        reveals: Vec::new(),
    }
}

fn ids(actions: &[Action]) -> Vec<u16> {
    actions.iter().map(|action| action.index() as u16).collect()
}

/// Roll the compiled transition while retaining Card 03's fallback semantics.
/// This is intentionally a test oracle, not a second production evaluator: it
/// lets the harness compare the compiler's transition path before the compiler
/// gains the Card 03-specific scoring primitive.
fn compiled_outcome_with_goal(
    term: &pretraining_term_compiler::CompiledWorld,
    goal: usize,
    actions: &[Action],
) -> Outcome {
    let mut cell = term.fragment.start(&term.contract);
    let mut path = vec![cell];
    let mut fallback_step = None;
    for (executed, action) in actions.iter().copied().enumerate() {
        if fallback_step.is_some() {
            path.push(cell);
            continue;
        }
        if action == Action::Fallback {
            fallback_step = Some(executed);
            path.push(cell);
            continue;
        }
        cell = term
            .fragment
            .step(&term.contract, cell, executed, action.index() as u16);
        path.push(cell);
    }
    if let Some(step) = fallback_step {
        return Outcome {
            value: 50 - step as i32,
            reached_goal: false,
            fell_back: true,
            fallback_step: Some(step),
            final_cell: cell,
        };
    }
    let settle = (0..path.len()).find(|index| path[*index..].iter().all(|entry| *entry == goal));
    Outcome {
        value: settle.map_or(0, |steps| 100 - steps as i32),
        reached_goal: settle.is_some(),
        fell_back: false,
        fallback_step: None,
        final_cell: cell,
    }
}

#[test]
fn static_card03_cases_match_all_transition_and_outcome_sequences() {
    for case in card_cases()
        .into_iter()
        .filter(|case| case.contract.restore.is_none())
    {
        let term = term_for(case.kind.label(), &case.contract);
        let compiled = term.compile().expect("static Card 03 term validates");
        assert_eq!(compiled.actions(), ids(&ACTIONS));

        for sequence in all_sequences() {
            let handwritten_path = trajectory(&Affordance, &case.contract, &sequence);
            let compiled_path = trajectory(&compiled.fragment, &compiled.contract, &ids(&sequence));
            assert_eq!(
                compiled_path,
                handwritten_path,
                "{} / {sequence:?}",
                case.kind.label()
            );

            let expected = run(&case.contract, &sequence);
            let actual = compiled_outcome_with_goal(&compiled, case.contract.goal, &sequence);
            assert_eq!(actual, expected, "{} / {sequence:?}", case.kind.label());
        }
    }
}

#[test]
fn body_environment_swap_preserves_static_semantics_and_hash_provenance() {
    let mut checked = 0;
    for case in card_cases()
        .into_iter()
        .filter(|case| case.contract.support != full_body() && case.contract.restore.is_none())
    {
        let swapped = body_environment_swap(&case.contract);
        let body = term_for("body-limited", &case.contract)
            .compile()
            .expect("body-limited term validates");
        let environment = term_for("environment-limited", &swapped)
            .compile()
            .expect("environment-limited term validates");

        assert_ne!(
            body.family_hash, environment.family_hash,
            "the two causes must remain distinct in the family hash"
        );
        for sequence in all_sequences() {
            let body_path = trajectory(&body.fragment, &body.contract, &ids(&sequence));
            let environment_path = trajectory(
                &environment.fragment,
                &environment.contract,
                &ids(&sequence),
            );
            assert_eq!(body_path, environment_path, "{sequence:?}");
            assert_eq!(
                compiled_outcome_with_goal(&body, case.contract.goal, &sequence),
                compiled_outcome_with_goal(&environment, case.contract.goal, &sequence),
                "{sequence:?}"
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 6);
}

#[test]
fn public_view_has_no_support_or_edge_fields_and_privileged_view_keeps_them() {
    let case = card_cases()
        .into_iter()
        .find(|case| case.kind.is_witness())
        .expect("Card 03 has a witness");
    let body = term_for("body", &case.contract).compile().unwrap();
    let swapped_contract = body_environment_swap(&case.contract);
    let environment = term_for("environment", &swapped_contract)
        .compile()
        .unwrap();

    // Public views are a trace-only type and do not expose the private support
    // or edge representation. The typed public ports are shared too.
    assert_eq!(body.public_view(&[]), environment.public_view(&[]));
    assert!(body
        .audit_metadata()
        .public_ports
        .iter()
        .all(|port| port != "support" && port != "blocked_edges"));
    assert_ne!(body.privileged_view(&[]), environment.privileged_view(&[]));
}

#[test]
fn the_current_compiler_gates_are_explicitly_separate_from_pending_card03_gates() {
    // This assertion is deliberately about status, so future implementation
    // work cannot silently call the partial harness a full Card 03 audit.
    let static_cases = card_cases()
        .into_iter()
        .filter(|case| case.contract.restore.is_none())
        .count();
    assert_eq!(static_cases, 10);
    assert_eq!(12 - static_cases, 2);
    // Pending: calibration prelude/public trace, support restoration, compiler
    // value(), exact ambiguity/orbit/query parity, and profiled rendering.
}
