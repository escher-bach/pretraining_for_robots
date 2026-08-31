//! The executable portion of the Card 03 oracle.
//!
//! These tests cover the executable finite semantics the current generic
//! compiler can represent. They do not weaken the full oracle in
//! `experiments/term-compiler/CARD03-ORACLE.md`; learner-facing rendering and
//! full audit/query parity remain explicit follow-up gates.

use pretraining_card03_affordance::{
    all_sequences, body_environment_swap, card_cases, full_body, run, Action, Affordance, Contract,
    Outcome,
};
use pretraining_g0_contract::{trajectory, Fragment, Norm};
use pretraining_term_compiler::{
    ActionRole, Actuation, Actuator, BodyTerm, CalibrationTerm, EnvironmentTerm, Morphology,
    NormTerm, Port, PortDirection, PortValue, ScoringTerm, Sensorium, SupportRestoration,
    Visibility, Wiring, WorldTerm,
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
        role: match action {
            Action::Hold => ActionRole::Hold,
            Action::Fallback => ActionRole::Fallback,
            Action::Step | Action::Leap | Action::Back => ActionRole::Movement,
        },
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
            // Restoration announcements and ordinary reveals share this
            // explicit public signal boundary.
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
        scoring: ScoringTerm {
            goal_reward: 100,
            action_cost: 1,
            fallback_reward: 50,
            violation_penalty: -100,
        },
        calibration: Some(CalibrationTerm {
            pulses: contract
                .calibration_pulses()
                .into_iter()
                .map(|action| action.index() as u16)
                .collect(),
            publishes_cells: true,
        }),
        restorations: contract
            .restore
            .map(|restore| SupportRestoration {
                actuator: restore.actuator as u16,
                after_step: restore.after_step,
                announcement_port: "reveal".into(),
                announcement_value: (restore.actuator * 100 + restore.after_step + 1) as i64,
            })
            .into_iter()
            .collect(),
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

/// Map generic compiler facts to the handwritten oracle's named outcome.  The
/// compiler, rather than this harness, owns transition, absorption, and value.
fn compiled_outcome_with_goal(
    term: &pretraining_term_compiler::CompiledWorld,
    goal: usize,
    actions: &[Action],
) -> Outcome {
    let outcome = term.outcome(&ids(actions));
    Outcome {
        value: outcome.value,
        reached_goal: outcome.norm_met && outcome.final_cell == goal,
        fell_back: outcome.fell_back,
        fallback_step: outcome.fallback_step,
        final_cell: outcome.final_cell,
    }
}

#[test]
fn card03_cases_match_all_transition_and_outcome_sequences_including_restoration() {
    for case in card_cases() {
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
    assert!(body
        .audit_metadata()
        .public_ports
        .iter()
        .all(|port| port != "support" && port != "blocked_edges"));
    assert_ne!(body.privileged_view(&[]), environment.privileged_view(&[]));
}

#[test]
fn calibration_is_unscored_public_and_restoration_is_announced_before_scoring() {
    let mut visible_swaps = 0usize;
    for case in card_cases()
        .into_iter()
        .filter(|case| case.contract.restore.is_none())
    {
        let compiled = term_for(case.kind.label(), &case.contract)
            .compile()
            .unwrap();
        let trace = compiled.public_view(&[]).trace;
        let calibration: Vec<i64> = case
            .contract
            .calibration_trace()
            .into_iter()
            .map(|cell| cell as i64)
            .collect();
        assert_eq!(&trace[..calibration.len()], calibration);
        let swapped = term_for("environment", &body_environment_swap(&case.contract))
            .compile()
            .unwrap();
        visible_swaps += usize::from(compiled.public_view(&[]) != swapped.public_view(&[]));
    }
    assert_eq!(visible_swaps, 3);

    for case in card_cases()
        .into_iter()
        .filter(|case| case.contract.restore.is_some())
    {
        let compiled = term_for(case.kind.label(), &case.contract)
            .compile()
            .unwrap();
        let trace = compiled.public_view(&[]).trace;
        let calibration_len = case.contract.calibration_trace().len();
        let restoration = case.contract.restore.unwrap();
        assert_eq!(
            trace[calibration_len + 1],
            (restoration.actuator * 100 + restoration.after_step + 1) as i64
        );
        // Restoration follows the absolute scored clock: unsupported at action
        // zero, available at action one for the audited fixture.
        assert_eq!(
            compiled.fragment.step(
                &compiled.contract,
                case.contract.start,
                0,
                restoration.actuator as u16
            ),
            case.contract.start
        );
        assert_ne!(
            compiled.fragment.step(
                &compiled.contract,
                case.contract.start,
                1,
                restoration.actuator as u16
            ),
            case.contract.start
        );
    }
}
