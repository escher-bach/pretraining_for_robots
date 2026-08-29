//! The exact audit: brackets, orbit verdicts, leakage, and the structural
//! property the card states in prose.
//!
//! Enumeration, orbit checking, and bracket analysis now come from
//! `pretraining-g0-contract`. What stays here is only what is specific to this card:
//! which transforms belong in its orbit, which negative is paired with which
//! witness, and which fields the frozen-history contrast must hold fixed.

use pretraining_g0_contract::{
    analyse_bracket, check_orbit, BaselineEvidence, BracketStructure, KernelUse, KindScore,
    OrbitVerdict, Symmetry,
};
use serde::{Deserialize, Serialize};

use crate::{
    ambiguity_gap, card_cases, contract_hash, goal_conditioning_contrast, kernel_use, norm_of,
    optimal_first_actions, published_first_actions, render_audit, run_policy, score_policy,
    state_only_baseline, unannounced_reveal_cost, value_bounds, Action, Case, CaseKind, Contract,
    GoalConditionedExact, GreedyProgress, HazardKind, LastGoal, NormSwap, PlanOnce,
    PublicGoalConditioned, PublicPolicy, RenderAudit, Switch, SwitchMode, CONFIGURATION, HORIZON,
    RING,
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

/// What the card's two information views actually differ by.
///
/// The card's prose says public and privileged information coincide and the gap
/// is zero everywhere. That is true of four of the five witnesses and false of
/// the fifth: an *unannounced* second goal is by construction not published
/// until it fires, so a solver reading the contract outperforms every policy
/// restricted to what has been said.
///
/// The original audit could not see this. [`ambiguity_gap`] compares
/// `privileged_value` with `value`, and this card does not override the former,
/// so it was comparing a quantity with itself. That number is kept below and
/// reported as vacuous rather than deleted, because the same shared primitive is
/// informative for a card that does override it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InformationBoundary {
    pub kind: String,
    pub start: usize,
    pub goal: usize,
    /// The enumerated ceiling, which reads every contract field.
    pub privileged_ceiling: i32,
    /// What the exact policy for the norm *as published* attains.
    pub published_norm_value: i32,
    /// Their difference: what the unannounced reveal costs.
    pub unannounced_reveal_cost: i32,
    /// The first action a privileged solver takes.
    pub privileged_first_actions: Vec<String>,
    /// The first action published information justifies. These differ exactly
    /// where the reveal costs something.
    pub published_first_actions: Vec<String>,
    /// `ambiguity_gap` on this case, which is zero by construction and is
    /// reported so the vacuity is visible rather than reassuring.
    pub vacuous_value_function_gap: i32,
}

fn information_boundary(case: &Case) -> InformationBoundary {
    let privileged = value_bounds(&case.contract).0;
    let cost = unannounced_reveal_cost(&case.contract);
    InformationBoundary {
        kind: case.kind.label().to_string(),
        start: case.contract.start,
        goal: case.contract.goal,
        privileged_ceiling: privileged,
        published_norm_value: privileged - cost,
        unannounced_reveal_cost: cost,
        privileged_first_actions: optimal_first_actions(&case.contract)
            .into_iter()
            .map(|action| action.name().to_string())
            .collect(),
        published_first_actions: published_first_actions(&case.contract)
            .into_iter()
            .map(|action| action.name().to_string())
            .collect(),
        vacuous_value_function_gap: ambiguity_gap(&case.contract),
    }
}

/// Whether the composition matches the coverage table in `EMBODIED-PROCESS.md`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelCoverage {
    pub declared: KernelUse,
    pub composed: KernelUse,
    pub matches_declaration: bool,
    /// The norm connectives the card's *cases* contain, gathered from the norm
    /// terms rather than from this file's prose.
    pub norm_connectives: Vec<String>,
    /// The connectives that appear only under a meaning-changing transformation.
    ///
    /// Conjunction lands here rather than above, and that is the card as
    /// specified: `CARDS.md` lists a *superseding* second goal as a variant and
    /// the composing reading as a transformation that changes the norm. Reported
    /// separately so "card 04 exercises the norm algebra" cannot be read as
    /// "card 04 trains on all three connectives".
    pub norm_connectives_only_under_transformation: Vec<String>,
}

fn kernel_coverage() -> KernelCoverage {
    let declared = KernelUse::declared("04").expect("card 04 is in the coverage table");
    let composed = kernel_use();
    let mut connectives: Vec<String> = card_cases()
        .iter()
        .flat_map(|case| {
            norm_of(&case.contract)
                .connectives()
                .into_iter()
                .map(str::to_string)
        })
        .collect();
    connectives.sort();
    connectives.dedup();
    let mut transformed: Vec<String> = card_cases()
        .iter()
        .filter(|case| case.contract.switch.is_some())
        .flat_map(|case| {
            let composing = Contract {
                switch: case.contract.switch.map(|switch| Switch {
                    mode: SwitchMode::Compose,
                    ..switch
                }),
                ..case.contract.clone()
            };
            norm_of(&composing)
                .connectives()
                .into_iter()
                .map(str::to_string)
        })
        .filter(|name| !connectives.contains(name))
        .collect();
    transformed.sort();
    transformed.dedup();
    KernelCoverage {
        matches_declaration: declared == composed,
        declared,
        composed,
        norm_connectives: connectives,
        norm_connectives_only_under_transformation: transformed,
    }
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
    /// The corrected information-boundary report. See [`InformationBoundary`].
    pub information_boundary: Vec<InformationBoundary>,
    /// The card's prose claim, evaluated instead of repeated.
    pub published_information_suffices_everywhere: bool,
    pub kernel_coverage: KernelCoverage,
    pub rendering: RenderAudit,
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
        baseline_report(&PublicGoalConditioned, "public_goal_conditioned"),
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
            // Both ceilings are excluded from the failing-baseline analysis.
            // A ceiling is not a failure mode: including the public one would
            // let it "isolate" the announced-switch negative, which would be a
            // statement about the information boundary dressed up as a
            // statement about a bracket.
            is_ceiling: matches!(
                report.name.as_str(),
                "goal_conditioned_exact" | "public_goal_conditioned"
            ),
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

    let boundary: Vec<InformationBoundary> = cases.iter().map(information_boundary).collect();
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
        published_information_suffices_everywhere: boundary
            .iter()
            .all(|entry| entry.unannounced_reveal_cost == 0),
        information_boundary: boundary,
        kernel_coverage: kernel_coverage(),
        rendering: render_audit().expect("the card renders through the shared boundary"),
        not_claimed: vec![
            "No learner was constructed, loaded, trained, or evaluated.".to_string(),
            "This is one card built as a world. It is not a capability result.".to_string(),
            "No GPU, remote, or multi-world run is authorized or implied.".to_string(),
            "The hazard variant shows the ceiling is unchanged by absorption; it does not show that any learner degrades.".to_string(),
            "The M12 node is not established. Establishing it requires a learner contrast this crate cannot perform.".to_string(),
            "`ambiguity_gap_is_zero_everywhere` is vacuous for this card: it compares the value function with itself, because the fragment does not override `privileged_value`. The non-vacuous quantity is `information_boundary`.".to_string(),
            "The card's prose claim that public and privileged information coincide is false on the two unannounced-switch witnesses, where the reveal costs one move. It holds on the other eighteen cases.".to_string(),
            "`published_norm_value` is the value of the exact policy for the norm as published. It is not a ceiling over every prior a scheduler might hold about pending switches; this family declares no such prior.".to_string(),
            "Rendering through the shared boundary makes this family learner-facing. It is not yet a frontier family: that needs a bounded pilot with a usable progress signal.".to_string(),
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
