//! The exact audit: brackets, orbit verdicts, leakage, and the structural
//! property the card states in prose.
//!
//! Enumeration, orbit checking, and bracket analysis now come from
//! `pretraining-g0-contract`. What stays here is only what is specific to this card:
//! which transforms belong in its orbit, which negative is paired with which
//! witness, and which fields the frozen-history contrast must hold fixed.

use pretraining_g0_contract::{
    analyse_bracket, check_orbit, BaselineEvidence, BracketStructure, KindScore, OrbitVerdict,
    Symmetry,
};
use serde::{Deserialize, Serialize};

use crate::{
    ambiguity_gap, card_cases, contract_hash, goal_conditioning_contrast, optimal_first_actions,
    run_policy, score_policy, state_only_baseline, value_bounds, Action, Case, CaseKind, Contract,
    GoalConditionedExact, GreedyProgress, HazardKind, LastGoal, NormSwap, PlanOnce, PublicPolicy,
    Switch, SwitchMode, CONFIGURATION, HORIZON, RING,
};

/// Move a contract through one element of the ring symmetry group.
pub fn transform(contract: &Contract, symmetry: Symmetry) -> Contract {
    let map = |cell: usize| symmetry.apply(cell);
    Contract {
        start: map(contract.start),
        goal: map(contract.goal),
        no_go: contract.no_go.map(map),
        hazard: contract.hazard.map(|(cell, kind)| (map(cell), kind)),
        distractor_after: contract.distractor_after,
        switch: contract.switch.map(|switch| Switch {
            goal: map(switch.goal),
            ..switch
        }),
    }
}

fn mirror(action: Action) -> Action {
    match action {
        Action::Advance => Action::Retreat,
        Action::Retreat => Action::Advance,
        Action::Hold => Action::Hold,
    }
}

fn identity(action: Action) -> Action {
    action
}

/// Run the whole invariance orbit.
///
/// The semantics-preserving half is now the ring's complete symmetry group
/// rather than a hand-picked subset, because the shared environment supplies
/// its own group. A reflection also exchanges the two directions of travel, so
/// optimal actions must correspond under mirroring rather than be equal.
pub fn orbit_verdicts() -> Vec<OrbitVerdict> {
    let contracts: Vec<Contract> = card_cases().into_iter().map(|case| case.contract).collect();
    let mut verdicts = Vec::new();

    for symmetry in CONFIGURATION.symmetries() {
        if symmetry == Symmetry::identity(RING) {
            continue;
        }
        let map = if symmetry.swaps_directions() {
            mirror
        } else {
            identity
        };
        verdicts.push(check_orbit(
            &NormSwap,
            &contracts,
            &symmetry.name(),
            true,
            |contract| transform(contract, symmetry),
            map,
        ));
    }

    // Semantics-changing, surface-similar. Each must move something; a
    // transform that changes nothing would be testing nothing.
    let pick = |kind: CaseKind| -> Contract {
        card_cases()
            .into_iter()
            .find(|case| case.kind == kind)
            .expect("the card has this witness")
            .contract
    };

    verdicts.push(check_orbit(
        &NormSwap,
        &[pick(CaseKind::WitnessGoalConditioning)],
        "goal_encoding_denotes_a_different_predicate",
        false,
        |contract| Contract {
            goal: (contract.goal + 1) % RING,
            ..contract.clone()
        },
        identity,
    ));

    verdicts.push(check_orbit(
        &NormSwap,
        &[pick(CaseKind::WitnessInhibit)],
        "no_go_keeps_its_form_and_covers_a_different_state",
        false,
        |contract| Contract {
            no_go: Some((contract.no_go.unwrap_or(0) + 3) % RING),
            ..contract.clone()
        },
        identity,
    ));

    verdicts.push(check_orbit(
        &NormSwap,
        &[pick(CaseKind::WitnessSwitch)],
        "second_goal_composes_instead_of_superseding",
        false,
        |contract| Contract {
            switch: contract.switch.map(|switch| Switch {
                mode: SwitchMode::Compose,
                ..switch
            }),
            ..contract.clone()
        },
        identity,
    ));

    verdicts
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenHistoryAudit {
    pub differs_only_in_goal: bool,
    pub optimal_first_actions_disjoint: bool,
    pub deterministic_construction: bool,
}

/// The card's one named leakage risk: if the goal draw shares randomness with
/// anything else, the two arms differ in more than the goal and the measurement
/// is void. This construction consumes no randomness, and the audit checks the
/// stronger structural property directly rather than trusting that.
pub fn frozen_history_audit() -> FrozenHistoryAudit {
    let pair: Vec<Case> = card_cases()
        .into_iter()
        .filter(|case| case.kind == CaseKind::WitnessGoalConditioning)
        .collect();
    let (left, right) = (&pair[0].contract, &pair[1].contract);

    let differs_only_in_goal = left.start == right.start
        && left.no_go == right.no_go
        && left.hazard == right.hazard
        && left.distractor_after == right.distractor_after
        && left.switch == right.switch
        && left.goal != right.goal;

    let left_first = optimal_first_actions(left);
    let right_first = optimal_first_actions(right);
    let disjoint = !left_first.iter().any(|action| right_first.contains(action));

    FrozenHistoryAudit {
        differs_only_in_goal,
        optimal_first_actions_disjoint: disjoint,
        deterministic_construction: card_cases()
            .iter()
            .map(|case| (case.kind, case.contract.clone()))
            .eq(card_cases()
                .iter()
                .map(|case| (case.kind, case.contract.clone()))),
    }
}

/// Which witness each negative is paired against.
pub fn paired_witness(negative: CaseKind) -> CaseKind {
    match negative {
        CaseKind::NegativeSingleGoal | CaseKind::NegativeGoalPredictable => {
            CaseKind::WitnessGoalConditioning
        }
        CaseKind::NegativeNoGoRemoved => CaseKind::WitnessInhibit,
        CaseKind::NegativeSwitchAnnounced => CaseKind::WitnessSwitch,
        other => other,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineReport {
    pub name: String,
    pub scores: std::collections::BTreeMap<String, KindScore>,
    pub passes_goal_conditioning_contrast: bool,
    pub optimal_on_negatives: Vec<String>,
}

fn baseline_report<P: PublicPolicy>(policy: &P, label: &str) -> BaselineReport {
    let scores = score_policy(policy);
    let cases = card_cases();
    let mut optimal_on = Vec::new();
    for kind in CaseKind::NEGATIVES {
        let selected: Vec<&Case> = cases.iter().filter(|case| case.kind == kind).collect();
        let all_optimal = selected
            .iter()
            .all(|case| run_policy(&case.contract, policy).value == value_bounds(&case.contract).0);
        if all_optimal {
            optimal_on.push(kind.label().to_string());
        }
    }
    BaselineReport {
        name: label.to_string(),
        scores,
        passes_goal_conditioning_contrast: goal_conditioning_contrast(policy),
        optimal_on_negatives: optimal_on,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseBracket {
    pub kind: String,
    pub start: usize,
    pub goal: usize,
    pub ceiling: i32,
    pub optimal_first_actions: Vec<String>,
    pub ambiguity_gap: i32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditReport {
    pub card: String,
    pub trunk: String,
    pub ring: usize,
    pub horizon: usize,
    pub action_sequences_enumerated: usize,
    pub contract_hash: String,
    pub cases: Vec<CaseBracket>,
    pub ambiguity_gap_is_zero_everywhere: bool,
    pub frozen_history: FrozenHistoryAudit,
    pub orbit: Vec<OrbitVerdict>,
    pub baselines: Vec<BaselineReport>,
    pub bracket_structure: BracketStructure,
    pub not_claimed: Vec<String>,
}

/// Build the complete audit. No learner is constructed anywhere in this path.
pub fn audit_report() -> AuditReport {
    let cases = card_cases();
    let brackets: Vec<CaseBracket> = cases
        .iter()
        .map(|case| CaseBracket {
            kind: case.kind.label().to_string(),
            start: case.contract.start,
            goal: case.contract.goal,
            ceiling: value_bounds(&case.contract).0,
            optimal_first_actions: optimal_first_actions(&case.contract)
                .into_iter()
                .map(|action| action.name().to_string())
                .collect(),
            ambiguity_gap: ambiguity_gap(&case.contract),
        })
        .collect();

    let baselines = vec![
        baseline_report(&GoalConditionedExact, "goal_conditioned_exact"),
        baseline_report(&state_only_baseline(), "state_only_goal_predictable"),
        baseline_report(
            &crate::state_only_for(CaseKind::NegativeSingleGoal),
            "state_only_single_goal",
        ),
        baseline_report(&LastGoal, "last_goal"),
        baseline_report(&GreedyProgress, "greedy_progress"),
        baseline_report(&PlanOnce, "plan_once"),
    ];

    let evidence: Vec<BaselineEvidence> = baselines
        .iter()
        .map(|report| BaselineEvidence {
            name: report.name.clone(),
            scores: report.scores.clone(),
            optimal_on_negatives: report.optimal_on_negatives.clone(),
            is_ceiling: report.name == "goal_conditioned_exact",
        })
        .collect();
    let pairing: Vec<(String, String)> = CaseKind::NEGATIVES
        .iter()
        .map(|kind| {
            (
                kind.label().to_string(),
                paired_witness(*kind).label().to_string(),
            )
        })
        .collect();

    AuditReport {
        card: "04-norm-swap".to_string(),
        trunk: "T3".to_string(),
        ring: RING,
        horizon: HORIZON,
        action_sequences_enumerated: 3usize.pow(HORIZON as u32),
        contract_hash: format!("{:016x}", contract_hash()),
        ambiguity_gap_is_zero_everywhere: brackets.iter().all(|entry| entry.ambiguity_gap == 0),
        cases: brackets,
        frozen_history: frozen_history_audit(),
        orbit: orbit_verdicts(),
        bracket_structure: analyse_bracket(&evidence, &pairing),
        baselines,
        not_claimed: vec![
            "No learner was constructed, loaded, trained, or evaluated.".to_string(),
            "This is one card built as a world. It is not a capability result.".to_string(),
            "No GPU, remote, or multi-world run is authorized or implied.".to_string(),
            "The hazard variant shows the ceiling is unchanged by absorption; it does not show that any learner degrades.".to_string(),
            "The M12 node is not established. Establishing it requires a learner contrast this crate cannot perform.".to_string(),
            "This card does not render onto the profiled event path, so it is not yet a portfolio family under the seed gate.".to_string(),
        ],
    }
}

/// Whether the absorbing/reset distinction moves the exact ceiling.
///
/// The `M12` claim needs it not to: a learner degrading under absorption cannot
/// then be blamed on a harder task.
pub fn absorption_leaves_ceiling_fixed() -> bool {
    card_cases()
        .iter()
        .filter(|case| case.kind == CaseKind::WitnessViability)
        .all(|case| {
            let reset = Contract {
                hazard: case
                    .contract
                    .hazard
                    .map(|(cell, _)| (cell, HazardKind::Reset)),
                ..case.contract.clone()
            };
            value_bounds(&case.contract).0 == value_bounds(&reset).0
        })
}
