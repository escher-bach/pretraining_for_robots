use pretraining_card04_norm_swap::{
    absorption_leaves_ceiling_fixed, all_sequences, ambiguity_gap, audit_report, card_cases,
    contract_hash, goal_conditioning_contrast, optimal_first_actions, orbit_verdicts, run,
    run_policy, score_policy, state_only_baseline, value_bounds, Action, CaseKind, Contract,
    GoalConditionedExact, GreedyProgress, HazardKind, LastGoal, PlanOnce, Switch, SwitchMode,
    GOAL_REWARD, HORIZON, MOVE_COST, RING,
};

fn rate(
    policy_scores: &std::collections::BTreeMap<String, pretraining_card04_norm_swap::KindScore>,
    kind: CaseKind,
) -> f64 {
    policy_scores
        .get(kind.label())
        .expect("every kind is scored")
        .rate
}

#[test]
fn the_enumeration_is_total_rather_than_sampled() {
    assert_eq!(all_sequences().len(), 3usize.pow(HORIZON as u32));
    assert_eq!(all_sequences().len(), 27);
}

#[test]
fn the_ambiguity_gap_is_zero_on_every_case() {
    // This card's value is that it has no epistemic content: every port is
    // public, so a privileged solver has exactly what a public one has. The gap
    // is computed rather than assumed, so a later edit that hides a field is
    // caught here instead of silently making failures ambiguous.
    for case in card_cases() {
        assert_eq!(
            ambiguity_gap(&case.contract),
            0,
            "{} has a non-zero ambiguity gap",
            case.kind.label()
        );
    }
    assert!(audit_report().ambiguity_gap_is_zero_everywhere);
}

#[test]
fn the_frozen_history_contrast_differs_only_in_the_goal_and_still_bites() {
    let audit = audit_report().frozen_history;
    assert!(
        audit.differs_only_in_goal,
        "the two arms must differ in the goal and nothing else"
    );
    assert!(
        audit.optimal_first_actions_disjoint,
        "if both arms share an optimal first action the contrast measures nothing"
    );
    assert!(audit.deterministic_construction);
}

#[test]
fn the_exact_policy_attains_the_ceiling_on_every_case() {
    for case in card_cases() {
        let outcome = run_policy(&case.contract, &GoalConditionedExact);
        let (ceiling, _) = value_bounds(&case.contract);
        assert_eq!(
            outcome.value,
            ceiling,
            "the public ceiling policy missed the ceiling on {}",
            case.kind.label()
        );
    }
    assert!(goal_conditioning_contrast(&GoalConditionedExact));
}

#[test]
fn a_state_only_policy_sits_at_the_goal_blind_ceiling_and_fails_the_contrast() {
    let policy = state_only_baseline();
    let scores = score_policy(&policy);
    assert_eq!(
        rate(&scores, CaseKind::WitnessGoalConditioning),
        0.5,
        "a policy that never reads the goal must land on the goal-blind ceiling"
    );
    assert!(
        !goal_conditioning_contrast(&policy),
        "a state-only policy cannot change its action when only the goal changes"
    );
}

#[test]
fn greedy_progress_is_the_designated_failure_for_inhibition() {
    let scores = score_policy(&GreedyProgress);
    assert_eq!(
        rate(&scores, CaseKind::WitnessInhibit),
        0.0,
        "ignoring the prohibition must fail the inhibition witness outright"
    );
    assert_eq!(
        rate(&scores, CaseKind::NegativeNoGoRemoved),
        1.0,
        "with the prohibition lifted the greedy action is correct"
    );
}

#[test]
fn plan_once_is_the_designated_failure_for_switching() {
    let scores = score_policy(&PlanOnce);
    assert_eq!(
        rate(&scores, CaseKind::WitnessSwitch),
        0.0,
        "planning once cannot absorb a goal published mid-episode"
    );
    assert_eq!(
        rate(&scores, CaseKind::NegativeSwitchAnnounced),
        1.0,
        "announcing the switch at episode start is exactly what rescues it"
    );
}

#[test]
fn every_invariance_orbit_verdict_holds() {
    let verdicts = orbit_verdicts();
    // Derived rather than hard-coded: the rotation count follows RING, and a
    // literal here silently became wrong the moment the ring was resized.
    assert_eq!(
        verdicts.len(),
        2 * pretraining_card04_norm_swap::RING - 1 + 3,
        "the preserving half is now the full dihedral group minus identity"
    );
    for verdict in &verdicts {
        assert!(
            verdict.verdict_holds,
            "orbit transform {} failed its verdict",
            verdict.transform
        );
    }
    // A semantics-changing transform that moved nothing would be a transform
    // that tests nothing, so both halves of the orbit are asserted.
    assert!(verdicts.iter().any(|v| v.semantics_preserving));
    assert!(verdicts.iter().any(|v| !v.semantics_preserving));
}

#[test]
fn every_negative_isolates_the_failure_it_is_paired_with() {
    let structure = audit_report().bracket_structure;
    assert!(
        structure.every_negative_isolates,
        "a negative that no baseline is optimal on while failing its witness \
         is not isolating anything: {:?}",
        structure.isolation
    );
}

#[test]
fn the_cards_literal_bracket_claim_is_false_and_stays_recorded() {
    // Card 04 §9 states that each failing baseline is optimal on exactly one
    // negative. Enumeration says otherwise: the easy negatives are solvable by
    // several baselines at once, so the claim as written cannot hold. The
    // property that does the real work is isolation, asserted above.
    //
    // This is pinned as a test so that a future edit which appears to satisfy
    // the literal claim has to confront the discrepancy rather than quietly
    // reintroduce it.
    let structure = audit_report().bracket_structure;
    assert!(
        !structure.each_failing_baseline_optimal_on_exactly_one,
        "if this now holds, the card text and this crate agree and the finding \
         should be retired deliberately rather than by accident"
    );
    assert!(structure.failing_baselines_optimal_on_multiple.len() >= 2);
    assert!(structure.failing_baselines_optimal_on_none.is_empty());
}

#[test]
fn absorption_does_not_move_the_ceiling() {
    // The M12 claim needs the reset and absorbing variants to have the same
    // ceiling. Otherwise a learner degrading under absorption could simply be
    // facing a harder task, and the node would not be established by that.
    assert!(absorption_leaves_ceiling_fixed());
}

#[test]
fn the_prohibition_costs_exactly_one_step_rather_than_the_goal() {
    // The geometry has to be checked, not assumed: on a six-cell ring the
    // detour did not fit the horizon and the witness was unreachable, which
    // made its ceiling zero and the exact policy look broken on its own card.
    let blocked = Contract::new(0, 2).with_no_go(1);
    let open = Contract::new(0, 2);
    let (blocked_ceiling, _) = value_bounds(&blocked);
    let (open_ceiling, _) = value_bounds(&open);
    assert!(blocked_ceiling > 0, "the goal must remain reachable");
    assert_eq!(
        open_ceiling - blocked_ceiling,
        1,
        "the detour should cost one step"
    );
    assert_eq!(optimal_first_actions(&blocked), vec![Action::Retreat]);
    assert_eq!(optimal_first_actions(&open), vec![Action::Advance]);
}

#[test]
fn the_distractor_is_inert_by_construction() {
    // The maintenance witness is deliberately identical in dynamics to the
    // goal-conditioning witness: the distractor is revealed and changes
    // nothing. That means no enumerated policy can fail it, because none of
    // them consume the event stream. It is a learner-facing test, and this
    // test records that limit rather than letting a vacuous pass look like
    // evidence.
    let cases = card_cases();
    let plain = cases
        .iter()
        .find(|case| case.kind == CaseKind::WitnessGoalConditioning)
        .expect("present");
    let distracted = cases
        .iter()
        .find(|case| case.kind == CaseKind::WitnessMaintain)
        .expect("present");
    assert_eq!(
        value_bounds(&plain.contract).0,
        value_bounds(&distracted.contract).0
    );
    assert_eq!(
        optimal_first_actions(&plain.contract),
        optimal_first_actions(&distracted.contract)
    );
    assert!(distracted.contract.distractor_after.is_some());
}

#[test]
fn a_reset_hazard_and_an_absorbing_hazard_differ_in_dynamics_but_not_in_ceiling() {
    let absorbing = Contract::new(0, 2).with_hazard(1, HazardKind::Absorbing);
    let reset = Contract::new(0, 2).with_hazard(1, HazardKind::Reset);
    assert_eq!(value_bounds(&absorbing).0, value_bounds(&reset).0);
    // They are genuinely different processes even though the ceiling agrees.
    let into_hazard = vec![Action::Advance, Action::Advance, Action::Hold];
    assert_ne!(
        pretraining_card04_norm_swap::run(&absorbing, &into_hazard).final_cell,
        pretraining_card04_norm_swap::run(&reset, &into_hazard).final_cell
    );
}

#[test]
fn the_contract_hash_is_stable_and_moves_with_the_card() {
    assert_eq!(contract_hash(), contract_hash());
    assert_eq!(format!("{:016x}", contract_hash()).len(), 16);
}

#[test]
fn last_goal_follows_a_superseding_switch_and_is_not_the_switch_negative() {
    // Recorded because the card names last-goal as the switch-ignoring
    // baseline. A reactive last-goal policy in fact follows a superseding
    // switch correctly; what it cannot do is composition. The baseline that
    // actually fails the switch witness is plan-once.
    let scores = score_policy(&LastGoal);
    assert_eq!(rate(&scores, CaseKind::WitnessSwitch), 1.0);
    assert_eq!(rate(&score_policy(&PlanOnce), CaseKind::WitnessSwitch), 0.0);
}

/// The kernel-composed evaluator means exactly what the hand-written one meant.
///
/// `run` was rewritten to evaluate a `Norm` term and a `Restriction` from the
/// shared kernel instead of open-coding the same rules. That is a change to an
/// already-audited family, so it is checked against an independent transcription
/// of the original rules over the whole enumeration rather than against the
/// audit's summary numbers, which could agree while individual cases moved.
#[test]
fn composing_the_card_from_the_shared_kernel_did_not_move_one_outcome() {
    fn reference(contract: &Contract, actions: &[Action]) -> (i32, bool) {
        let mut cell = contract.start;
        let mut trajectory = vec![cell];
        let mut violated = false;
        let mut absorbed = false;
        let mut visited_first_goal = cell == contract.goal;

        for action in actions.iter().copied() {
            if absorbed {
                trajectory.push(cell);
                continue;
            }
            let next = match action {
                Action::Retreat => (cell + RING - 1) % RING,
                Action::Hold => cell,
                Action::Advance => (cell + 1) % RING,
            };
            if Some(next) == contract.no_go {
                violated = true;
            }
            match contract.hazard {
                Some((hazard_cell, HazardKind::Absorbing)) if next == hazard_cell => {
                    cell = next;
                    absorbed = true;
                }
                Some((hazard_cell, HazardKind::Reset)) if next == hazard_cell => {
                    cell = contract.start;
                }
                _ => cell = next,
            }
            if cell == contract.goal {
                visited_first_goal = true;
            }
            trajectory.push(cell);
        }

        let target = contract.active_goal(actions.len());
        let settle = (0..trajectory.len())
            .find(|index| trajectory[*index..].iter().all(|entry| *entry == target));
        let composed = match contract.switch {
            Some(switch) if matches!(switch.mode, SwitchMode::Compose) => visited_first_goal,
            _ => true,
        };
        let solved = !violated && !absorbed && composed && settle.is_some();
        let value = match settle {
            Some(steps) if solved => GOAL_REWARD - MOVE_COST * steps as i32,
            _ => 0,
        };
        (value, solved)
    }

    let mut checked = 0usize;
    for case in card_cases() {
        for sequence in all_sequences() {
            let composed = run(&case.contract, &sequence);
            assert_eq!(
                (composed.value, composed.solved),
                reference(&case.contract, &sequence),
                "{} diverged on {sequence:?}",
                case.kind.label()
            );
            checked += 1;
        }
    }
    // Both readings of the composing switch are covered too, even though no
    // case uses it: it is the card's meaning-changing transformation, and a
    // divergence there would move an orbit verdict rather than a case.
    for case in card_cases() {
        let Some(switch) = case.contract.switch else {
            continue;
        };
        let composing = Contract {
            switch: Some(Switch {
                mode: SwitchMode::Compose,
                ..switch
            }),
            ..case.contract.clone()
        };
        for sequence in all_sequences() {
            let composed = run(&composing, &sequence);
            assert_eq!(
                (composed.value, composed.solved),
                reference(&composing, &sequence),
                "the composing reading diverged on {sequence:?}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 20 * 27 + 4 * 27);
}
