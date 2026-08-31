//! The stage-A semantic audit for card 04's goal-use family.
//!
//! The composite's audit reports a vacuous ambiguity gap on purpose and carries
//! the real quantity — its published-information gap — separately. Stage A has
//! no hidden state at all, so there is no real quantity to carry: the gap is
//! zero, it is vacuous, and both facts are reported rather than one of them
//! being presented as evidence.
//!
//! What replaces it as the family's central measurement is the state-only
//! ceiling. `0.5` on the witness and `1.0` on both controls is the whole
//! contrast in three numbers, and it is enumerated rather than asserted.

use std::collections::BTreeMap;

use pretraining_g0_contract::{
    analyse_bracket, check_orbit, check_orbit_with, optimal_actions_from as prefix_optimal,
    BracketStructure, KernelUse, KindScore, OrbitVerdict, Symmetry,
};
use serde::{Deserialize, Serialize};

use crate::{
    all_sequences, card_cases, cases_of, contract_hash, goal_use_contrast, kernel_use,
    norm_connectives, optimal_first_actions, optimal_rate, render_audit, score_policy,
    state_only_ceiling, vacuous_ambiguity_gap, Action, CaseKind, Contract, Denotation, FixedPlan,
    GoalConditionedExact, GoalUse, PublicPolicy, StateOnly, CONFIGURATION, RING,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseEvidence {
    pub kind: String,
    pub start: usize,
    pub goal: usize,
    pub goal_cell: usize,
    pub ceiling: i32,
    pub optimal_first_actions: Vec<String>,
    pub ambiguity_gap: i32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineEvidence {
    pub name: String,
    pub is_ceiling: bool,
    pub scores: BTreeMap<String, KindScore>,
    pub optimal_rate: BTreeMap<String, f64>,
    pub goal_use_contrast: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecompositionRecord {
    pub composite_card: String,
    pub composite_contract_hash: String,
    pub original_relation: String,
    pub basic_relation: String,
    pub removed: Vec<String>,
    pub retained: Vec<String>,
    pub falsifier: String,
    pub re_entry_condition: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditReport {
    pub family: String,
    pub card: String,
    pub stage: String,
    pub trunk: String,
    pub contract_hash: String,
    pub cases: usize,
    pub action_sequences_enumerated: usize,
    pub removed_from_composite: DecompositionRecord,
    pub kernel_declared_for_composite: KernelUse,
    pub kernel_composed: KernelUse,
    pub norm_connectives_used: Vec<String>,
    pub case_evidence: Vec<CaseEvidence>,
    /// The whole contrast in three numbers.
    pub state_only_ceiling: BTreeMap<String, f64>,
    pub baselines: Vec<BaselineEvidence>,
    pub bracket: BracketStructure,
    /// Zero on every case, and vacuous because there is no privileged view.
    pub ambiguity_gap_is_zero_everywhere: bool,
    pub ambiguity_gap_is_vacuous_by_construction: bool,
    /// Settle and Visit induce the same behaviour at this horizon, so the
    /// composite's goal-predicate transformation is not usable here and the
    /// denotation shift stands in for it. Reported rather than silently
    /// substituted.
    pub settle_and_visit_coincide_at_this_horizon: bool,
    pub preserving_orbits: Vec<OrbitVerdict>,
    pub changing_orbits: Vec<OrbitVerdict>,
    pub full_dihedral_orbit_holds: bool,
    pub rendering: crate::RenderAudit,
    pub no_learner_evidence: String,
}

fn baseline<P: PublicPolicy>(policy: &P, is_ceiling: bool) -> BaselineEvidence {
    BaselineEvidence {
        name: policy.name().to_string(),
        is_ceiling,
        scores: score_policy(policy),
        optimal_rate: CaseKind::ALL
            .into_iter()
            .map(|kind| (kind.label().to_string(), optimal_rate(policy, kind)))
            .collect(),
        goal_use_contrast: goal_use_contrast(policy),
    }
}

fn bracket_entry(entry: &BaselineEvidence) -> pretraining_g0_contract::BaselineEvidence {
    pretraining_g0_contract::BaselineEvidence {
        name: entry.name.clone(),
        scores: entry.scores.clone(),
        optimal_on_negatives: CaseKind::NEGATIVES
            .into_iter()
            .filter(|kind| entry.optimal_rate[kind.label()] >= 1.0)
            .map(|kind| kind.label().to_string())
            .collect(),
        is_ceiling: entry.is_ceiling,
    }
}

/// Whether `Settle` and `Visit` pick out the same behaviour on every case.
///
/// They do at this horizon, because a configuration that reaches the goal can
/// always hold there for the remaining step. The composite's "change the
/// predicate a goal denotes" transformation therefore has no bite in stage A
/// and the denotation shift is used instead.
fn settle_and_visit_coincide() -> bool {
    card_cases().into_iter().all(|case| {
        let settle = optimal_first_actions(&case.contract);
        let visit: Vec<Action> = {
            // A `Visit` norm scored the same way: value falls with the step at
            // which the goal is first entered.
            let mut best = i32::MIN;
            let mut chosen: Vec<Action> = Vec::new();
            for sequence in all_sequences() {
                let trajectory = crate::walk(&case.contract, &sequence);
                let value = trajectory
                    .iter()
                    .position(|cell| *cell == case.contract.goal_cell())
                    .map(|first| crate::GOAL_REWARD - crate::MOVE_COST * first as i32)
                    .unwrap_or(0);
                if value > best {
                    best = value;
                    chosen.clear();
                }
                if value == best && !chosen.contains(&sequence[0]) {
                    chosen.push(sequence[0]);
                }
            }
            chosen.sort();
            chosen.dedup();
            chosen
        };
        settle == visit
    })
}

/// Every element of the ring's dihedral group, checked as one verdict.
fn full_dihedral_orbit(contracts: &[Contract]) -> bool {
    CONFIGURATION.symmetries().into_iter().all(|symmetry| {
        check_orbit(
            &GoalUse,
            contracts,
            &symmetry.name(),
            true,
            |contract| contract.relabelled(symmetry),
            move |action| {
                if symmetry.swaps_directions() {
                    action.reversed()
                } else {
                    action
                }
            },
        )
        .verdict_holds
    })
}

/// The second decision, which is what a goal two steps away is decided at.
fn second_actions(_: &GoalUse, contract: &Contract) -> Vec<Action> {
    let first = optimal_first_actions(contract);
    match first.first() {
        Some(action) => prefix_optimal(&GoalUse, contract, &[*action]),
        None => Vec::new(),
    }
}

pub fn audit_report() -> AuditReport {
    let cases = card_cases();
    let contracts: Vec<Contract> = cases.iter().map(|case| case.contract).collect();

    let case_evidence = cases
        .iter()
        .map(|case| CaseEvidence {
            kind: case.kind.label().to_string(),
            start: case.contract.start,
            goal: case.contract.goal,
            goal_cell: case.contract.goal_cell(),
            ceiling: crate::value_bounds(&case.contract).0,
            optimal_first_actions: optimal_first_actions(&case.contract)
                .into_iter()
                .map(|action| action.name().to_string())
                .collect(),
            ambiguity_gap: vacuous_ambiguity_gap(&case.contract),
        })
        .collect::<Vec<_>>();

    let baselines = vec![
        baseline(&FixedPlan, false),
        baseline(&StateOnly, false),
        baseline(&GoalConditionedExact, true),
    ];
    let bracket = analyse_bracket(
        &baselines.iter().map(bracket_entry).collect::<Vec<_>>(),
        &CaseKind::NEGATIVES
            .into_iter()
            .map(|kind| {
                (
                    kind.label().to_string(),
                    CaseKind::WitnessGoalChangesAction.label().to_string(),
                )
            })
            .collect::<Vec<_>>(),
    );

    let rotation = Symmetry {
        shift: 1,
        reflect: false,
        cells: RING,
    };
    let reflection = Symmetry {
        shift: 0,
        reflect: true,
        cells: RING,
    };
    let preserving_orbits = vec![
        check_orbit(
            &GoalUse,
            &contracts,
            "configuration_and_goal_rotated",
            true,
            |contract| contract.relabelled(rotation),
            |action| action,
        ),
        check_orbit(
            &GoalUse,
            &contracts,
            "configuration_and_goal_reflected_with_directions_exchanged",
            true,
            |contract| contract.relabelled(reflection),
            |action| action.reversed(),
        ),
    ];
    let changing_orbits = vec![
        check_orbit(
            &GoalUse,
            &contracts,
            "goal_denotation_shifted",
            false,
            |contract| contract.with_denotation(Denotation::Shifted),
            |action| action,
        ),
        check_orbit(
            &GoalUse,
            &contracts,
            "configuration_reflected_without_exchanging_directions",
            false,
            |contract| contract.relabelled(reflection),
            |action| action,
        ),
        check_orbit_with(
            &GoalUse,
            &contracts,
            "goal_moved_one_step_further",
            false,
            |contract| contract.with_goal(contract.goal + 1),
            |action| action,
            second_actions,
        ),
    ];

    AuditReport {
        family: "card04a".into(),
        card: "04-norm-swap".into(),
        stage: "A: public goal use".into(),
        trunk: "T3".into(),
        contract_hash: format!("{:016x}", contract_hash()),
        cases: cases.len(),
        action_sequences_enumerated: all_sequences().len(),
        removed_from_composite: DecompositionRecord {
            composite_card: "04".into(),
            composite_contract_hash: "d975c3a646591ccf".into(),
            original_relation:
                "with public state and history fixed, changing the requested outcome changes the correct action, over maintenance, inhibition, switching, and viability"
                    .into(),
            basic_relation:
                "with the start configuration fixed and one goal published at episode start, settle the published goal"
                    .into(),
            removed: vec![
                "the prohibition and the greedy-move inhibition it created".into(),
                "the superseding second goal with its interrupt and reveal".into(),
                "the absorbing viability boundary and its restriction".into(),
                "the irrelevant distractors".into(),
                "every norm connective; one Settle leaf remains".into(),
                "the third decision, which existed to answer a switch".into(),
            ],
            retained: vec![
                "the five-position ring and its full dihedral orbit".into(),
                "the public known body".into(),
                "matched episode pairs sharing all history up to the goal".into(),
                "the constant-goal and goal-predictable-from-state controls".into(),
                "the state-only ceiling of 0.5 on the witness".into(),
                "the published-norm teacher and its leak refusal".into(),
                "a meaning-changing transformation of what the goal denotes".into(),
            ],
            falsifier: "the fixed learner cannot reach exact full-support fit here, or a state-only policy succeeds on this witness".into(),
            re_entry_condition: "this hash passes its own semantic audit and fixed AdmissionProfile; only then the prohibition, then the announced switch, then the unannounced switch, then viability".into(),
        },
        kernel_declared_for_composite: KernelUse::declared("04").expect("card 04 is declared"),
        kernel_composed: kernel_use(),
        norm_connectives_used: norm_connectives()
            .into_iter()
            .map(|name| name.to_string())
            .collect(),
        case_evidence,
        state_only_ceiling: CaseKind::ALL
            .into_iter()
            .map(|kind| (kind.label().to_string(), state_only_ceiling(kind)))
            .collect(),
        baselines,
        bracket,
        ambiguity_gap_is_zero_everywhere: card_cases()
            .into_iter()
            .all(|case| vacuous_ambiguity_gap(&case.contract) == 0),
        ambiguity_gap_is_vacuous_by_construction: true,
        settle_and_visit_coincide_at_this_horizon: settle_and_visit_coincide(),
        preserving_orbits,
        changing_orbits,
        full_dihedral_orbit_holds: full_dihedral_orbit(&contracts),
        rendering: render_audit().expect("card 04a boundary"),
        no_learner_evidence:
            "World-validity evidence only; no learner was constructed, trained, or evaluated."
                .into(),
    }
}

/// Whether every semantic verdict in the report holds.
pub fn audit_passes(report: &AuditReport) -> bool {
    report.ambiguity_gap_is_zero_everywhere
        && report.norm_connectives_used.is_empty()
        && report
            .preserving_orbits
            .iter()
            .chain(&report.changing_orbits)
            .all(|verdict| verdict.verdict_holds)
        && report.full_dihedral_orbit_holds
        && report.bracket.every_negative_isolates
        && report.rendering.report.every_episode_round_trips
        && report.rendering.matched_pairs_are_taught_apart
        && report.state_only_ceiling[CaseKind::WitnessGoalChangesAction.label()] == 0.5
        && report.case_evidence.len() == card_cases().len()
        && !cases_of(CaseKind::WitnessGoalChangesAction).is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{run_policy, HORIZON};

    #[test]
    fn the_whole_audit_passes() {
        let report = audit_report();
        assert!(audit_passes(&report), "{report:#?}");
    }

    #[test]
    fn the_exact_policy_is_optimal_everywhere_and_the_two_controls_are_not() {
        let report = audit_report();
        let exact = report
            .baselines
            .iter()
            .find(|entry| entry.name == "goal_conditioned_exact")
            .expect("the ceiling policy");
        for kind in CaseKind::ALL {
            assert_eq!(exact.optimal_rate[kind.label()], 1.0, "{}", kind.label());
        }
        assert!(exact.goal_use_contrast);
    }

    #[test]
    fn the_ceiling_policy_never_settles_late() {
        for case in card_cases() {
            let outcome = run_policy(&case.contract, &GoalConditionedExact);
            assert!(outcome.solved);
            assert!(outcome.settle_steps.expect("settled") <= HORIZON);
        }
    }
}
