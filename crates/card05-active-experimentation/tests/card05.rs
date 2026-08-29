//! R8 evidence for card 05: value of information, matched controls, the
//! informative-but-worthless separation, and the learner boundary.

use pretraining_card05_active_experimentation::{
    action_from_index, all_sequences, attains_public_ceiling, audit_report, card_cases,
    contract_hash, instance_ambiguity, learner_episode, learner_episodes, mean_public_ceiling,
    mean_value, optimal_first_actions, privileged_ceiling, probe_contrast, probe_rate,
    public_ceiling, run, run_policy, Action, ActiveExperimentation, AlwaysProbe, Bit, CaseKind,
    Contract, ExactPublic, GateVisibility, NeverProbe, PeekInstead, PrivilegedGateKnown,
    ShamInstead, BUDGET, GOAL_CELL, HORIZON,
};
use pretraining_g0_contract::{
    agent_equivalence, epistemic_value, identification_diameter, public_optimal_actions_at,
    ResourceScope, Restriction,
};
use pretraining_g0_render::{legacy_tokens, profiled_tokens, ENVELOPE_PROFILE};
use pretraining_profiled_event::{declared_profile, strip_profile_tag};
use pretraining_world::Role;

#[test]
fn the_enumeration_is_total_rather_than_sampled() {
    assert_eq!(all_sequences().len(), Action::ALL.len().pow(HORIZON as u32));
    assert_eq!(all_sequences().len(), 25);
    assert_eq!(audit_report().action_sequences_enumerated, 25);
}

#[test]
fn the_witness_has_a_positive_ambiguity_gap_and_the_first_one_in_the_portfolio() {
    // Cards 04 and 02 report vacuous gaps by construction. This is the family
    // where the shared privileged-versus-public comparison finally says
    // something, and the number is the price of the probe.
    for case in card_cases() {
        if case.kind != CaseKind::WitnessProbeThenCommit {
            continue;
        }
        assert_eq!(privileged_ceiling(&case.contract), 99.0);
        assert_eq!(public_ceiling(&case.contract), 98.0);
        assert_eq!(
            privileged_ceiling(&case.contract) - public_ceiling(&case.contract),
            1.0
        );
    }
}

#[test]
fn probing_is_the_correct_public_opening_exactly_on_the_witness() {
    for case in card_cases() {
        let opening = public_optimal_actions_at(
            &ActiveExperimentation,
            &instance_ambiguity(&case.contract),
            &[],
            HORIZON,
        );
        assert_eq!(
            opening.contains(&Action::Probe),
            case.kind == CaseKind::WitnessProbeThenCommit,
            "{} opened with {:?}",
            case.kind.label(),
            opening
        );
    }
}

#[test]
fn the_privileged_opening_is_a_commit_and_would_have_taught_the_gate() {
    // The reason the teacher is the public policy. `value_bounds` reads the
    // gate, so its answer on the witness is "commit at once to the right side",
    // which is exactly the fact the learner is supposed to buy.
    for case in card_cases() {
        if case.kind != CaseKind::WitnessProbeThenCommit {
            continue;
        }
        let privileged = optimal_first_actions(&case.contract);
        assert!(privileged.iter().all(|action| action.is_commit()));
        assert!(!privileged.contains(&Action::Probe));
    }
}

#[test]
fn an_informative_action_can_be_worthless() {
    // The `M5 -> M11b` contrast. `Peek` shrinks the surviving set exactly as
    // much as `Probe` does and moves the ceiling not at all.
    let witness = card_cases()
        .into_iter()
        .find(|case| case.kind == CaseKind::WitnessProbeThenCommit)
        .expect("the card has this witness")
        .contract;
    let set = instance_ambiguity(&witness);
    let table = epistemic_value(&ActiveExperimentation, &set, HORIZON, |action: Action| {
        action.name().to_string()
    });
    let entry = |wanted: Action| {
        table
            .iter()
            .find(|row| row.action == wanted.name())
            .expect("every action is reported")
            .clone()
    };
    let probe = entry(Action::Probe);
    let peek = entry(Action::Peek);
    let sham = entry(Action::Sham);

    assert_eq!(probe.ambiguity_reduction, peek.ambiguity_reduction);
    assert_eq!(sham.ambiguity_reduction, 0);
    assert!(probe.public_value > peek.public_value);
    assert_eq!(peek.public_value, sham.public_value);
}

#[test]
fn the_inconsequential_bit_is_observable_and_outcome_irrelevant() {
    // `agent_equivalence` reports the two halves separately, and this is where
    // they come apart: an intervention distinguishes the two values, and no
    // intervention gives them different outcomes.
    let witness = card_cases()[0].contract;
    let noise = agent_equivalence(
        &ActiveExperimentation,
        &witness,
        &witness.with_flipped_noise(),
        HORIZON,
        |action: Action| action.name().to_string(),
    );
    assert!(!noise.observationally_indistinguishable);
    assert!(noise.outcome_indistinguishable);
    assert!(!noise.equivalent);
    assert_eq!(noise.separating_sequence, Some(vec!["peek".to_string()]));

    let gate = agent_equivalence(
        &ActiveExperimentation,
        &witness,
        &witness.with_flipped_gate(),
        HORIZON,
        |action: Action| action.name().to_string(),
    );
    assert!(!gate.outcome_indistinguishable, "the gate changes outcomes");
}

#[test]
fn the_matched_control_pairs_cost_and_movement_and_differs_only_in_information() {
    let report = audit_report();
    assert!(report.the_matched_control_holds_where_the_probe_is_matched_and_informative);
    // Each control breaks exactly one clause, and the audit names which.
    assert_eq!(
        report.clause_each_control_breaks[CaseKind::WitnessProbeThenCommit.label()],
        "none"
    );
    assert_eq!(
        report.clause_each_control_breaks[CaseKind::NegativeGatePublic.label()],
        "only the informative action reduces ambiguity"
    );
    assert_eq!(
        report.clause_each_control_breaks[CaseKind::NegativeProbeTooExpensive.label()],
        "equal cost"
    );
}

#[test]
fn the_gate_is_invisible_until_it_is_bought() {
    let report = audit_report().leakage;
    assert!(report.the_gate_is_invisible_until_probed.holds);
    assert_eq!(report.initial_diameter, 4, "both bits are open");
    assert_eq!(report.diameter_after_probe, 2, "the gate closes");
    assert_eq!(report.diameter_after_peek, 2, "the other bit closes");
    assert_eq!(report.diameter_after_sham, 4, "nothing closes");

    let witness = card_cases()[0].contract;
    let set = instance_ambiguity(&witness);
    assert_eq!(
        identification_diameter(&ActiveExperimentation, &set, &[Action::Sham, Action::Sham]),
        4
    );
}

#[test]
fn admission_is_judged_in_expectation_because_no_episode_attains_the_public_ceiling() {
    // With the gate hidden, the public ceiling is an average. A blind commit is
    // right on half the instances; scoring it per case would reject the optimal
    // policy on the other half.
    let expensive = CaseKind::NegativeProbeTooExpensive;
    assert_eq!(mean_public_ceiling(expensive), 49.5);
    for case in card_cases() {
        if case.kind != expensive {
            continue;
        }
        let value = run_policy(&case.contract, &NeverProbe).value;
        assert!(value == 99 || value == 0, "no episode lands on 49.5");
    }
    assert!(attains_public_ceiling(&NeverProbe, expensive));
    assert!(!attains_public_ceiling(
        &NeverProbe,
        CaseKind::WitnessProbeThenCommit
    ));
}

#[test]
fn never_probing_is_the_designated_failure_for_every_control() {
    for kind in CaseKind::NEGATIVES {
        assert!(
            attains_public_ceiling(&NeverProbe, kind),
            "{} should not need a probe",
            kind.label()
        );
    }
    assert!(mean_value(&NeverProbe, CaseKind::WitnessProbeThenCommit) < 98.0);
    assert!(audit_report().bracket_structure.every_negative_isolates);
}

#[test]
fn probing_regardless_pays_for_information_it_cannot_use() {
    assert!(!probe_contrast(&AlwaysProbe));
    assert_eq!(probe_rate(&AlwaysProbe, CaseKind::NegativeGatePublic), 1.0);
    // And loses exactly the step it spent.
    assert!(
        mean_value(&AlwaysProbe, CaseKind::NegativeGatePublic)
            < mean_public_ceiling(CaseKind::NegativeGatePublic)
    );
    assert!(probe_contrast(&ExactPublic));
}

#[test]
fn buying_the_worthless_bit_costs_the_same_as_buying_nothing() {
    for kind in CaseKind::ALL {
        assert_eq!(
            mean_value(&PeekInstead, kind),
            mean_value(&ShamInstead, kind),
            "{} separated an inconsequential probe from doing nothing",
            kind.label()
        );
    }
}

#[test]
fn the_privileged_policy_is_a_reference_and_never_a_teacher() {
    // It beats the public ceiling on the witness, which is precisely why it
    // cannot supervise: its advantage is the gate.
    assert!(
        mean_value(&PrivilegedGateKnown, CaseKind::WitnessProbeThenCommit)
            > mean_public_ceiling(CaseKind::WitnessProbeThenCommit)
    );
    for (_, episode) in learner_episodes().expect("renders") {
        for set in episode.selected_actuators() {
            let taught: Vec<Action> = set
                .iter()
                .map(|index| action_from_index(*index as usize).expect("a known actuator"))
                .collect();
            assert!(!taught.contains(&Action::Peek));
        }
    }
}

#[test]
fn the_budget_is_a_shared_resource_restriction() {
    let budget = card_cases()[0].contract.budget();
    assert_eq!(budget.kind(), "resource");
    assert!(matches!(
        budget,
        Restriction::Resource {
            scope: ResourceScope::Shared,
            ..
        }
    ));
    assert_eq!(budget.remaining_budget(1), Some(BUDGET - 1));
    // Shared rather than local is what makes an expensive probe crowd out the
    // commit instead of merely costing points.
    let expensive = Contract::new(Bit::Left, Bit::Left).with_probe_cost(BUDGET);
    let outcome = run(&expensive, &[Action::Probe, Action::CommitLeft]);
    assert!(!outcome.committed, "the commit could not be afforded");
    assert_eq!(outcome.value, 0);
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
fn no_rendered_episode_publishes_the_gate_before_it_was_bought() {
    let audit = audit_report().rendering;
    assert!(audit.no_episode_leaks_the_gate);
    assert!(audit.the_gate_is_invisible_in_the_rendered_prefix);
    assert!(audit.the_inconsequential_bit_is_invisible_unless_bought);
    assert!(audit.the_teacher_probes_exactly_on_the_witness);

    // And the gate-public control does publish it, at the start, on purpose.
    let public = card_cases()
        .into_iter()
        .find(|case| case.kind == CaseKind::NegativeGatePublic)
        .expect("the card has this control");
    assert_eq!(public.contract.visibility, GateVisibility::Public);
    let episode = learner_episode(&public.contract).expect("renders");
    let first_execution = legacy_tokens(&episode)
        .expect("renders")
        .iter()
        .position(|token| token.public.role == Role::ActionExecuted)
        .expect("every episode executes something");
    let gate_rows: Vec<usize> = legacy_tokens(&episode)
        .expect("renders")
        .iter()
        .enumerate()
        .filter(|(_, token)| {
            token.public.role == Role::Condition
                && token.public.key == pretraining_card05_active_experimentation::EPISODE_KEY_GATE
        })
        .map(|(index, _)| index)
        .collect();
    assert!(gate_rows.iter().any(|index| *index < first_execution));
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
fn the_goal_cell_is_the_one_the_episode_names() {
    for (_, episode) in learner_episodes().expect("renders") {
        let goals: Vec<u16> = episode
            .groups
            .iter()
            .flat_map(|group| group.facts.iter())
            .filter_map(|fact| match fact {
                pretraining_g0_render::G0Fact::Goal { key, .. } => Some(*key),
                _ => None,
            })
            .collect();
        assert_eq!(goals, vec![GOAL_CELL as u16]);
    }
}

#[test]
fn every_invariance_orbit_verdict_holds() {
    for verdict in &audit_report().orbit {
        assert!(verdict.verdict_holds, "{} failed", verdict.transform);
    }
}

#[test]
fn the_contract_hash_is_stable_and_moves_with_the_card() {
    assert_eq!(contract_hash(), contract_hash());
    assert_eq!(
        audit_report().contract_hash,
        format!("{:016x}", contract_hash())
    );
}
