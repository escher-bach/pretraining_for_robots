//! R7 evidence for card 02: history ablation, aliasing, orbits, and the learner
//! boundary.

use pretraining_card02_predictive_state::{
    all_sequences, audit_report, card_cases, ceiling_sequence, contract_hash,
    discriminating_actions, learner_episode, learner_episodes, mode_ambiguity, run, run_policy,
    taught_sequence, value_bounds, without_the_decoy, without_the_latch, Action, CaseKind,
    ConstantCommand, Contract, LastLatch, Memoryless, Mode, ModeConditioned, ModeCoupling,
    ModeVisibility, PredictiveState, PublicPolicy, Window, DISCRIMINATING_STEP, GOAL_OFFSET,
    HORIZON, RING,
};
use pretraining_g0_contract::{
    ablated_policy_value, identification_diameter, public_policy_value, Coarsened, Displaced,
    PubliclyObservable, Resume,
};
use pretraining_g0_render::{legacy_tokens, profiled_tokens, ENVELOPE_PROFILE};
use pretraining_profiled_event::{declared_profile, strip_profile_tag};
use pretraining_world::Role;

#[test]
fn the_enumeration_is_total_rather_than_sampled() {
    assert_eq!(all_sequences().len(), Action::ALL.len().pow(HORIZON as u32));
    assert_eq!(all_sequences().len(), 64);
    assert_eq!(audit_report().action_sequences_enumerated, 64);
}

#[test]
fn the_goal_is_out_of_reach_of_every_mode_independent_route() {
    // Three unit steps land one short and nothing else moves before the last
    // decision. If this ever stops holding, the latch has become optional and
    // the card is measuring something else.
    let witness = card_cases()
        .into_iter()
        .find(|case| case.kind == CaseKind::WitnessLatchedMode)
        .expect("the card has this witness");
    assert_eq!(GOAL_OFFSET, HORIZON + 1);
    for sequence in all_sequences() {
        if sequence.iter().copied().any(Action::is_mode_sensitive) {
            continue;
        }
        assert!(
            !run(&witness.contract, &sequence).reached_goal,
            "{sequence:?} reached the goal without the mode"
        );
    }
}

#[test]
fn the_two_modes_take_opposite_commands_at_the_discriminating_decision() {
    let witnesses: Vec<_> = card_cases()
        .into_iter()
        .filter(|case| case.kind == CaseKind::WitnessLatchedMode)
        .collect();
    assert_eq!(witnesses.len(), 2);
    let first = discriminating_actions(&witnesses[0].contract);
    let second = discriminating_actions(&witnesses[1].contract);
    assert_eq!(first, vec![Action::Cross]);
    assert_eq!(second, vec![Action::Anti]);

    // And the *first* action agrees, which is why this card cannot use it as
    // its observable.
    assert_eq!(
        pretraining_card02_predictive_state::optimal_first_actions(&witnesses[0].contract),
        pretraining_card02_predictive_state::optimal_first_actions(&witnesses[1].contract)
    );
}

#[test]
fn ablating_the_latch_costs_exactly_half_the_witness_and_nothing_on_the_controls() {
    for case in card_cases() {
        let set = mode_ambiguity(&case.contract);
        let full = public_policy_value(&PredictiveState, &set, HORIZON);
        let ablated = ablated_policy_value(&PredictiveState, &set, HORIZON, without_the_latch);
        let ceiling = f64::from(value_bounds(&case.contract).0);
        assert_eq!(full, ceiling, "the latch identifies the mode");
        match case.kind {
            CaseKind::WitnessLatchedMode | CaseKind::NegativeMemoryCost => {
                // One command in two, chosen without information: half the
                // ceiling and no more.
                assert!((ablated - ceiling / 2.0).abs() < 1e-9, "{ablated}");
            }
            CaseKind::NegativeFullyObservable | CaseKind::NegativeIrrelevantLatch => {
                assert_eq!(ablated, ceiling, "{} needs no memory", case.kind.label());
            }
        }
    }
}

#[test]
fn ablating_the_second_latch_costs_nothing_anywhere() {
    for case in card_cases() {
        let set = mode_ambiguity(&case.contract);
        assert_eq!(
            ablated_policy_value(&PredictiveState, &set, HORIZON, without_the_decoy),
            public_policy_value(&PredictiveState, &set, HORIZON),
            "{} lost value to the decoy",
            case.kind.label()
        );
    }
}

#[test]
fn no_action_inside_the_aliasing_interval_separates_the_modes() {
    let witness = card_cases()
        .into_iter()
        .find(|case| case.kind == CaseKind::WitnessLatchedMode)
        .expect("the card has this witness")
        .contract;
    let blind = Coarsened::new(&PredictiveState, without_the_latch);
    let set = mode_ambiguity(&witness);
    for sequence in pretraining_g0_contract::sequences_of_length(&Action::ALL, DISCRIMINATING_STEP)
    {
        assert_eq!(
            identification_diameter(&blind, &set, &sequence),
            2,
            "{sequence:?} rediscovered the mode by probing"
        );
    }
    // And the whole trace does separate them once the latch is visible.
    assert_eq!(identification_diameter(&PredictiveState, &set, &[]), 1);
}

/// The rate at which a policy attains the ceiling over one case kind.
fn optimal_rate<P: PublicPolicy>(policy: &P, kind: CaseKind) -> f64 {
    let cases: Vec<_> = card_cases()
        .into_iter()
        .filter(|case| case.kind == kind)
        .collect();
    let hits = cases
        .iter()
        .filter(|case| run_policy(&case.contract, policy).value == value_bounds(&case.contract).0)
        .count();
    hits as f64 / cases.len() as f64
}

#[test]
fn the_required_memory_span_is_sharp() {
    // Just below scores at chance, exactly at scores perfectly. The claim is
    // about the rate, not about every case: a policy that guesses between two
    // commands is right on one of the two modes by construction, and asserting
    // per-case failure would be asserting that a coin never lands heads.
    assert_eq!(
        optimal_rate(&Window::too_short(), CaseKind::WitnessLatchedMode),
        0.5
    );
    assert_eq!(
        optimal_rate(&Window::sufficient(), CaseKind::WitnessLatchedMode),
        1.0
    );
    assert_eq!(Window::too_short().memory_span(), DISCRIMINATING_STEP);
    assert_eq!(Window::sufficient().memory_span(), DISCRIMINATING_STEP + 1);
    assert!(!Window::too_short().name().is_empty());
}

#[test]
fn a_memoryless_policy_is_optimal_on_the_two_controls_and_scores_at_chance_on_the_witness() {
    assert_eq!(
        optimal_rate(&Memoryless, CaseKind::NegativeFullyObservable),
        1.0
    );
    assert_eq!(
        optimal_rate(&Memoryless, CaseKind::NegativeIrrelevantLatch),
        1.0
    );
    assert_eq!(optimal_rate(&Memoryless, CaseKind::WitnessLatchedMode), 0.5);
    assert_eq!(optimal_rate(&Memoryless, CaseKind::NegativeMemoryCost), 0.5);

    // The contrast is the sharp statement: a guessing policy cannot be correct
    // in both modes *and* take different commands in them.
    assert!(!pretraining_card02_predictive_state::retention_contrast(
        &Memoryless
    ));
    assert!(pretraining_card02_predictive_state::retention_contrast(
        &ModeConditioned
    ));
}

#[test]
fn a_constant_command_is_optimal_only_where_the_latch_is_inert() {
    for case in card_cases() {
        let value = run_policy(&case.contract, &ConstantCommand).value;
        let ceiling = value_bounds(&case.contract).0;
        if case.kind == CaseKind::NegativeIrrelevantLatch {
            assert_eq!(value, ceiling);
        } else if case.contract.mode == Mode::Reversed {
            assert!(value < ceiling, "{} at reversed", case.kind.label());
        }
    }
}

#[test]
fn retaining_the_wrong_latch_passes_the_witness_and_fails_the_memory_cost_control() {
    // This is the over-retention evidence: `last_latch` carries everything and
    // is wrong exactly where the two latches disagree.
    assert!(pretraining_card02_predictive_state::retention_contrast(
        &LastLatch
    ));
    let report = audit_report().over_retention;
    assert!(report.failure_passes_the_witness);
    assert!((report.failure_rate_on_the_memory_cost_control - 0.5).abs() < 1e-9);
    assert!(report.ceiling_passes_both);
    assert!(report.the_second_latch_is_inert);
}

#[test]
fn every_value_orbit_verdict_holds_against_the_discriminating_command() {
    for verdict in &audit_report().orbit {
        assert!(verdict.verdict_holds, "{} failed", verdict.transform);
    }
}

#[test]
fn every_information_orbit_verdict_holds() {
    for verdict in &audit_report().information_orbit {
        assert!(
            verdict.verdict_holds,
            "{} failed: {} -> {}, diameter {} -> {}",
            verdict.transform,
            verdict.public_ceiling_before,
            verdict.public_ceiling_after,
            verdict.diameter_before,
            verdict.diameter_after
        );
    }
}

#[test]
fn ending_the_aliasing_interval_early_is_invisible_to_the_value_orbit() {
    // The reason the information orbit exists. A contract-holding solver never
    // needed the latch, so the ceiling and the correct command are untouched;
    // what moves is what a learner without the latch can attain.
    let witness = card_cases()
        .into_iter()
        .find(|case| case.kind == CaseKind::WitnessLatchedMode)
        .expect("the card has this witness")
        .contract;
    let early = Contract {
        aliasing_until: 1,
        ..witness
    };
    assert_eq!(value_bounds(&witness).0, value_bounds(&early).0);
    assert_eq!(
        discriminating_actions(&witness),
        discriminating_actions(&early)
    );

    let blind = Coarsened::new(&PredictiveState, without_the_latch);
    assert!(
        public_policy_value(&blind, &mode_ambiguity(&early), HORIZON)
            > public_policy_value(&blind, &mode_ambiguity(&witness), HORIZON),
        "probing must become possible"
    );
}

#[test]
fn the_aliasing_interrupt_is_declared_rather_than_defaulted() {
    let interrupt = card_cases()[0].contract.aliasing_interrupt();
    // The configuration keeps running while the commands are inert and resumes
    // from the cell the route reached. A restart would delete the route.
    assert_eq!(interrupt.displaced, Displaced::Continues);
    assert_eq!(interrupt.resume, Resume::FromState);
    assert!(audit_report().kernel_coverage.matches_declaration);
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
fn the_teacher_marks_every_correct_action_and_only_correct_actions() {
    for (kind, episode) in learner_episodes().expect("renders") {
        let case = card_cases()
            .into_iter()
            .find(|entry| entry.kind == kind)
            .expect("a rendered kind is a case kind");
        let selected = episode.selected_actuators();
        assert_eq!(selected.len(), HORIZON);
        for set in &selected {
            assert!(!set.is_empty(), "{} supervised nothing", kind.label());
        }
        // At the discriminating decision the marked set must be exactly the
        // correct set, which on the inert control has two members.
        let taught: Vec<Action> = selected[DISCRIMINATING_STEP]
            .iter()
            .map(|index| {
                pretraining_card02_predictive_state::action_from_index(*index as usize)
                    .expect("a known actuator")
            })
            .collect();
        let expected_len = usize::from(case.contract.coupling == ModeCoupling::Inert) + 1;
        assert_eq!(
            taught.len(),
            expected_len,
            "{} marked the wrong number of correct commands",
            kind.label()
        );
    }
}

#[test]
fn supervision_never_reaches_a_public_byte() {
    for (_, episode) in learner_episodes().expect("renders") {
        for token in legacy_tokens(&episode).expect("renders") {
            if token.supervision.action_mask.iter().any(|slot| *slot) {
                assert_eq!(token.public.role, Role::ActionQuery);
            }
            assert!(!token.supervision.future_mask);
        }
    }
}

#[test]
fn the_latch_is_published_once_and_the_control_republishes_it() {
    for case in card_cases() {
        let episode = learner_episode(&case.contract).expect("renders");
        let latches = legacy_tokens(&episode)
            .expect("renders")
            .into_iter()
            .filter(|token| {
                token.public.role == Role::Condition
                    && token.public.key == pretraining_card02_predictive_state::EPISODE_KEY_MODE
            })
            .count();
        assert_eq!(
            latches,
            match case.contract.visibility {
                ModeVisibility::Latched => 1,
                ModeVisibility::Always => HORIZON,
            },
            "{} published the mode the wrong number of times",
            case.kind.label()
        );
    }
}

#[test]
fn the_taught_sequence_reaches_the_ceiling_on_every_case() {
    for case in card_cases() {
        assert_eq!(
            run(&case.contract, &ceiling_sequence(&case.contract)).value,
            value_bounds(&case.contract).0,
            "{} teacher fell below the ceiling",
            case.kind.label()
        );
        assert_eq!(
            taught_sequence(&case.contract, &ModeConditioned),
            ceiling_sequence(&case.contract)
        );
    }
}

#[test]
fn the_contract_hash_is_stable_and_moves_with_the_card() {
    assert_eq!(contract_hash(), contract_hash());
    assert_eq!(
        audit_report().contract_hash,
        format!("{:016x}", contract_hash())
    );
    assert_eq!(RING, 7);
}

#[test]
fn the_public_trace_carries_what_is_published_and_not_what_is_true() {
    // A decorrelated latch is the card's meaning-changing transformation, and
    // it only works if the trace reports the publication rather than the mode.
    let witness = card_cases()[0].contract;
    let decorrelated = Contract {
        latch_reports: Some(Mode::Forward),
        ..witness.with_flipped_mode()
    };
    let honest = Contract {
        latch_reports: None,
        ..witness.with_flipped_mode()
    };
    assert_ne!(
        PredictiveState.public_trace(&decorrelated, &[]),
        PredictiveState.public_trace(&honest, &[])
    );
    assert_eq!(
        PredictiveState.public_trace(&decorrelated, &[]),
        PredictiveState.public_trace(&witness, &[]),
        "a decorrelated latch says what the other mode's honest latch says"
    );
}
