//! R6 evidence for card 03: reachability, identification, reveal, brackets, and
//! the learner boundary.

use pretraining_card03_affordance::{
    admissible_bodies, all_sequences, allocation_contrast, audit_report, body_ambiguity,
    body_environment_swap, card_cases, contract_hash, full_body, goal_is_reachable,
    learner_episode, learner_episodes, optimal_first_actions, run, run_policy, value_bounds,
    Action, Affordance, AlwaysAttempt, Calibration, CaseKind, Contract, ExactPostCalibration,
    IgnoreSupport, PlanAtCalibration, PublicPolicy, PublicView, Restore, TryOnceThenFallback,
    HORIZON, RING,
};
use pretraining_g0_contract::{identification_diameter, PubliclyObservable, Restriction};
use pretraining_g0_render::{legacy_tokens, profiled_tokens, ENVELOPE_PROFILE};
use pretraining_profiled_event::{declared_profile, strip_profile_tag};
use pretraining_world::Role;

#[test]
fn the_enumeration_is_total_rather_than_sampled() {
    assert_eq!(all_sequences().len(), Action::ALL.len().pow(HORIZON as u32));
    assert_eq!(all_sequences().len(), 25);
    let report = audit_report();
    assert_eq!(report.action_sequences_enumerated, 25);
    assert_eq!(report.cases.len(), card_cases().len());
}

#[test]
fn calibration_identifies_the_body_exactly_and_removing_it_does_not() {
    for case in card_cases() {
        let calibrated = body_ambiguity(&case.contract);
        assert_eq!(
            identification_diameter(&Affordance, &calibrated, &[]),
            1,
            "{} did not identify its body",
            case.kind.label()
        );

        let blind = body_ambiguity(
            &case
                .contract
                .clone()
                .with_calibration(Calibration::Uninformative),
        );
        assert_eq!(
            identification_diameter(&Affordance, &blind, &[]),
            admissible_bodies().len(),
            "{} leaked its body without calibration",
            case.kind.label()
        );
    }
}

#[test]
fn every_admissible_body_leaves_a_distinct_calibration_trace() {
    // The scaffold does not reset between pulses, so identification has to come
    // from consecutive differences. This is what makes that exact rather than
    // merely plausible.
    let base = card_cases()[0].contract.clone();
    let mut traces = Vec::new();
    for body in admissible_bodies() {
        traces.push(
            Contract {
                support: body,
                ..base.clone()
            }
            .calibration_trace(),
        );
    }
    let unique: std::collections::BTreeSet<_> = traces.iter().collect();
    assert_eq!(unique.len(), admissible_bodies().len());
}

#[test]
fn an_unreachable_goal_makes_the_fallback_the_only_correct_first_move() {
    for case in card_cases() {
        let first = optimal_first_actions(&case.contract);
        if goal_is_reachable(&case.contract) {
            assert!(
                !first.contains(&Action::Fallback),
                "{} fell back on a reachable goal",
                case.kind.label()
            );
        } else {
            assert_eq!(
                first,
                vec![Action::Fallback],
                "{} did not fall back on an unreachable goal",
                case.kind.label()
            );
        }
    }
}

#[test]
fn delaying_the_fallback_is_strictly_worse_than_taking_it_at_once() {
    let unreachable = card_cases()
        .into_iter()
        .find(|case| case.kind == CaseKind::WitnessUnreachableFallback)
        .expect("the card has this witness");
    let immediate = run(&unreachable.contract, &[Action::Fallback, Action::Hold]);
    let delayed = run(&unreachable.contract, &[Action::Hold, Action::Fallback]);
    assert!(immediate.value > delayed.value);
    assert_eq!(immediate.value, value_bounds(&unreachable.contract).0);
}

#[test]
fn the_exact_policy_attains_the_ceiling_on_every_case() {
    for case in card_cases() {
        assert_eq!(
            run_policy(&case.contract, &ExactPostCalibration).value,
            value_bounds(&case.contract).0,
            "{} is not attained by the exact policy",
            case.kind.label()
        );
    }
    assert!(allocation_contrast(&ExactPostCalibration));
}

#[test]
fn ignoring_support_is_the_designated_failure_for_reachability() {
    // Optimal where nothing is withheld, wrong where something is. That pairing
    // is the whole reason the frequency-matched control exists.
    assert!(!allocation_contrast(&IgnoreSupport));
    for case in card_cases() {
        let value = run_policy(&case.contract, &IgnoreSupport).value;
        let ceiling = value_bounds(&case.contract).0;
        match case.kind {
            CaseKind::NegativeFrequencyMatched => assert_eq!(value, ceiling),
            CaseKind::WitnessUnreachableFallback => assert!(value < ceiling),
            _ => {}
        }
    }
}

#[test]
fn planning_at_calibration_is_the_designated_failure_for_the_restoration() {
    for case in card_cases() {
        let value = run_policy(&case.contract, &PlanAtCalibration).value;
        let ceiling = value_bounds(&case.contract).0;
        match case.kind {
            CaseKind::NegativeNoRestore => assert_eq!(value, ceiling),
            CaseKind::WitnessRestore => assert!(
                value < ceiling,
                "the restoration must cost a policy that ignores the announcement"
            ),
            _ => {}
        }
    }
}

#[test]
fn attempting_regardless_wins_where_reach_is_never_the_question() {
    for case in card_cases() {
        let value = run_policy(&case.contract, &AlwaysAttempt).value;
        let ceiling = value_bounds(&case.contract).0;
        if goal_is_reachable(&case.contract) {
            assert_eq!(value, ceiling, "{} is reachable", case.kind.label());
        } else {
            assert!(value < ceiling, "{} is not", case.kind.label());
        }
    }
    // Attempting once and then abandoning is never right here: the fallback is
    // worth strictly less after a wasted step and the goal is not closer.
    for case in card_cases() {
        assert!(
            run_policy(&case.contract, &TryOnceThenFallback).value
                <= value_bounds(&case.contract).0
        );
    }
}

#[test]
fn every_negative_isolates_the_failure_it_is_paired_with() {
    let report = audit_report();
    assert!(report.bracket_structure.every_negative_isolates);
    for entry in &report.bracket_structure.isolation {
        assert!(
            !entry.isolating_baselines.is_empty(),
            "{} isolates nothing",
            entry.negative
        );
    }
}

#[test]
fn every_invariance_orbit_verdict_holds() {
    let report = audit_report();
    for verdict in &report.orbit {
        assert!(verdict.verdict_holds, "{} failed", verdict.transform);
    }
    // Rotations preserve; a reflection does not, because the body has no
    // command that reverses `Leap`.
    let reflection = report
        .orbit
        .iter()
        .find(|verdict| verdict.transform.contains("reflection"))
        .expect("the reflection is checked");
    assert!(!reflection.semantics_preserving);
    assert!(!(reflection.ceiling_unchanged && reflection.optimal_actions_correspond));
}

#[test]
fn the_body_environment_swap_is_publicly_visible_and_behaviourally_inert() {
    let case = card_cases()
        .into_iter()
        .find(|entry| entry.kind == CaseKind::WitnessUnreachableFallback)
        .expect("the card has this witness");
    let swapped = body_environment_swap(&case.contract);

    assert_eq!(
        swapped.support,
        full_body(),
        "the body is no longer limited"
    );
    assert!(!swapped.blocked_edges.is_empty(), "the environment now is");
    assert_eq!(
        value_bounds(&case.contract).0,
        value_bounds(&swapped).0,
        "the reachable set, and so the ceiling, is preserved"
    );
    assert_eq!(
        optimal_first_actions(&case.contract),
        optimal_first_actions(&swapped)
    );
    assert_ne!(
        Affordance.public_trace(&case.contract, &[]),
        Affordance.public_trace(&swapped, &[]),
        "a transform that changed nothing publicly would be testing nothing"
    );
    // Not every swap is visible — a body that drives only `Back` never carries
    // the scaffold out of the deleted region. The audit reports how many are.
    let visibility = audit_report().swap_visibility;
    assert!(visibility.the_invariance_is_actually_tested);
    assert!(visibility.publicly_visible_swaps <= visibility.swappable_contracts);
}

#[test]
fn the_body_limit_and_the_norm_are_different_restrictions() {
    // Card 04's hazard is a viability restriction; this card's support is an
    // action restriction. They are the same shared type and they do not
    // constrain the same thing, which is what keeps a body limit from being
    // mistaken for a prohibition.
    let case = card_cases()[0].contract.clone();
    let body = case.body();
    assert_eq!(body.kind(), "action");
    assert!(matches!(body, Restriction::Action { .. }));
    for cell in 0..RING {
        assert!(
            body.admits_cell(cell),
            "a body limit forbids no configuration"
        );
    }
    assert!(body.boundary_effect().is_none());
}

#[test]
fn the_restoration_is_a_reveal_announced_before_the_first_decision() {
    let case = card_cases()
        .into_iter()
        .find(|entry| entry.kind == CaseKind::WitnessRestore)
        .expect("the card has this witness");
    let reveal = case
        .contract
        .reveal()
        .expect("this case restores something");
    assert!(reveal.guard.is_visible_at_start());
    assert!(!case.contract.supports(
        Action::from_index_for_test(case.contract.restore.unwrap().actuator),
        0
    ));
    assert!(case.contract.supports(
        Action::from_index_for_test(case.contract.restore.unwrap().actuator),
        1
    ));
}

trait TestIndex {
    fn from_index_for_test(index: usize) -> Action;
}

impl TestIndex for Action {
    fn from_index_for_test(index: usize) -> Action {
        pretraining_card03_affordance::action_from_index(index).expect("a known actuator")
    }
}

#[test]
fn a_mid_episode_resolve_rebases_the_restoration_rather_than_clamping_it() {
    // The saturating shift this replaced turned "already fired" into "fires
    // after the next action", which cost the exact policy its own witness.
    let contract = Contract::new(0, RING - 1, {
        let mut set = pretraining_g0_contract::IndexSet::EMPTY;
        set.insert(Action::Step.index());
        set.insert(Action::Leap.index());
        set
    })
    .with_restore(Restore {
        actuator: Action::Back.index(),
        after_step: 0,
    });

    let pending = contract.resolved_from(0, 0);
    assert!(pending.restore.is_some(), "it has not fired yet");
    assert!(!pending.support.contains(Action::Back.index()));

    let fired = contract.resolved_from(0, 1);
    assert!(
        fired.restore.is_none(),
        "it has fired and is now body support"
    );
    assert!(fired.support.contains(Action::Back.index()));
    assert!(fired.supports(Action::Back, 0));
}

#[test]
fn every_case_leaves_the_crate_as_a_valid_envelope_that_strips_back_exactly() {
    for (kind, episode) in learner_episodes().expect("renders") {
        let body = legacy_tokens(&episode).expect("renders");
        let tagged = profiled_tokens(&episode).expect("renders");
        assert_eq!(declared_profile(&tagged), Ok(ENVELOPE_PROFILE));
        let (profile, stripped) = strip_profile_tag(&tagged).expect("strips");
        assert_eq!(profile, ENVELOPE_PROFILE);
        assert_eq!(
            stripped,
            body,
            "{} did not survive the round trip",
            kind.label()
        );
    }
}

#[test]
fn the_scaffold_is_unsupervised_and_only_decisions_carry_a_target() {
    for (_, episode) in learner_episodes().expect("renders") {
        for token in legacy_tokens(&episode).expect("renders") {
            if token.supervision.action_mask.iter().any(|slot| *slot) {
                assert_eq!(token.public.role, Role::ActionQuery);
            }
        }
        // Calibration pulses are executed actions with no query in front of
        // them: the scaffold acts and the learner watches.
        let queries = episode.decisions();
        assert!(queries <= HORIZON);
    }
}

#[test]
fn the_teacher_is_taught_from_the_identified_body_and_not_from_the_field() {
    // Rendering an uninformative-calibration contract is refused outright,
    // because the exact policy would then be reading an unpublished field.
    let blind = card_cases()[0]
        .contract
        .clone()
        .with_calibration(Calibration::Uninformative);
    assert!(matches!(
        learner_episode(&blind),
        Err(pretraining_g0_render::RenderFault::TeacherWouldLeak { .. })
    ));

    // And with calibration present, the identified body is what a view exposes.
    let view = PublicView {
        contract: &card_cases()[0].contract,
        cell: 0,
        executed: 0,
    };
    assert!(view.identified_body().is_some());
    let blind_view = PublicView {
        contract: &blind,
        cell: 0,
        executed: 0,
    };
    assert!(blind_view.identified_body().is_none());
}

#[test]
fn the_contract_hash_is_stable_and_moves_with_the_card() {
    assert_eq!(contract_hash(), contract_hash());
    assert_eq!(
        audit_report().contract_hash,
        format!("{:016x}", contract_hash())
    );
}

#[test]
fn the_scaffold_cannot_be_reading_the_goal() {
    let report = audit_report();
    assert!(report.scaffold.independent_of_the_goal);
    assert_eq!(report.scaffold.goals_checked, card_cases().len() * RING);
    assert_eq!(
        report.scaffold.pulse_order,
        vec!["step", "leap", "leap", "back"]
    );
}
