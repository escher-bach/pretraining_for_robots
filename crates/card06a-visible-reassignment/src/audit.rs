//! The stage-A semantic audit for card 06's visible-reassignment family.
//!
//! The composite's audit is the template. Three entries are new, and each one
//! records something the decomposition changed rather than something it kept:
//!
//! - `public_gap_is_closed_by_visibility` states the closed ambiguity gap as a
//!   number and pairs it with the orbit that reopens it. The gap is not missing;
//!   it is the removed factor, measured.
//! - `boundary_marker_is_not_load_bearing` records that hiding the
//!   assignment-change boundary is *preserving* here where the composite has it
//!   meaning-changing. Declaring it preserving and checking it is the honest
//!   move; inheriting the composite's verdict would have asserted an invariance
//!   this family does not have.
//! - `coupling_rule` is `Conflict` where the composite's is `Override`, because
//!   a channel here has exactly one writer.

use std::collections::BTreeMap;

use pretraining_g0_contract::{
    ambiguity_report, analyse_bracket, check_information_orbit, check_orbit_with,
    noninterference_check, AmbiguitySet, BracketStructure, InformationVerdict, KernelUse,
    KindScore, NonInterference, OrbitVerdict,
};
use serde::{Deserialize, Serialize};

use crate::{
    all_sequences, binding_contrast, card_cases, cases_of, contract_hash, drive_for, kernel_use,
    mean_public_ceiling, mean_value, privileged_ceiling, public_ceiling, pulse_for, raw_ambiguity,
    relevant_ambiguity, render_audit, solved_rate, Action, CaseKind, ChannelIdentity, Contract,
    KnownAssignment, PublicPolicy, TagFollowing, ValueTracking, VisibleReassignment, HORIZON,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseEvidence {
    pub kind: String,
    pub seed: u64,
    pub reassigned: bool,
    pub initial_candidates: usize,
    pub raw_after_pulse: usize,
    pub relevant_after_pulse: usize,
    pub public_ceiling: f64,
    pub privileged_ceiling: f64,
    pub ambiguity_gap: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineEvidence {
    pub name: String,
    pub is_ceiling: bool,
    pub scores: BTreeMap<String, KindScore>,
    pub mean_value: BTreeMap<String, f64>,
    pub binding_contrast: bool,
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
    pub coupling_rule: String,
    pub composite_coupling_rule: String,
    pub case_evidence: Vec<CaseEvidence>,
    pub kind_ceilings: BTreeMap<String, f64>,
    pub baselines: Vec<BaselineEvidence>,
    pub bracket: BracketStructure,
    pub noninterference: NonInterference,
    pub preserving_orbits: Vec<OrbitVerdict>,
    pub changing_orbits: Vec<OrbitVerdict>,
    pub information_orbits: Vec<InformationVerdict>,
    /// The removed factor, stated as a number rather than as an absence.
    pub public_gap_is_closed_by_visibility: bool,
    pub witness_public_ceiling: f64,
    pub witness_occluded_public_ceiling: f64,
    /// Hiding the boundary marker is preserving here and meaning-changing in
    /// the composite. Recorded, not inherited.
    pub boundary_marker_is_not_load_bearing: bool,
    pub rendering: crate::RenderAudit,
    pub no_learner_evidence: String,
}

fn baseline<P: PublicPolicy>(policy: &P, is_ceiling: bool) -> BaselineEvidence {
    BaselineEvidence {
        name: policy.name().to_string(),
        is_ceiling,
        scores: CaseKind::ALL
            .into_iter()
            .map(|kind| {
                let total = cases_of(kind).len();
                let rate = solved_rate(policy, kind);
                (
                    kind.label().to_string(),
                    KindScore {
                        solved: (rate * total as f64).round() as usize,
                        total,
                        rate,
                        optimal_rate: rate,
                    },
                )
            })
            .collect(),
        mean_value: CaseKind::ALL
            .into_iter()
            .map(|kind| (kind.label().to_string(), mean_value(policy, kind)))
            .collect(),
        binding_contrast: binding_contrast(policy),
    }
}

fn bracket_entry(entry: &BaselineEvidence) -> pretraining_g0_contract::BaselineEvidence {
    pretraining_g0_contract::BaselineEvidence {
        name: entry.name.clone(),
        scores: entry.scores.clone(),
        optimal_on_negatives: CaseKind::NEGATIVES
            .into_iter()
            .filter(|kind| entry.scores[kind.label()].rate >= 1.0)
            .map(|kind| kind.label().to_string())
            .collect(),
        is_ceiling: entry.is_ceiling,
    }
}

/// The drive a policy takes, as the observable a changing orbit reads.
///
/// The first decision cannot be that observable: either pulse locates both
/// sources, so every contract in this family has the same optimal opening and a
/// transform compared on it would agree with itself. This is the composite's
/// lesson about `check_orbit` reading the wrong action, in the one place stage A
/// still has it.
fn taught_drive(_: &VisibleReassignment, contract: &Contract) -> Vec<Action> {
    vec![drive_for(
        ValueTracking.drive(contract, contract.goal_channel),
    )]
}

fn occluded(set: &AmbiguitySet<Contract>) -> AmbiguitySet<Contract> {
    AmbiguitySet::uniform(set.candidates.iter().map(|c| c.occlude()).collect())
}

fn boundary_hidden(set: &AmbiguitySet<Contract>) -> AmbiguitySet<Contract> {
    AmbiguitySet::uniform(set.candidates.iter().map(|c| c.hide_boundary()).collect())
}

fn scale_flipped(set: &AmbiguitySet<Contract>) -> AmbiguitySet<Contract> {
    AmbiguitySet::uniform(set.candidates.iter().map(|c| c.flip_scale()).collect())
}

pub fn audit_report() -> AuditReport {
    let cases = card_cases();
    let contracts: Vec<Contract> = cases.iter().map(|case| case.contract).collect();

    let case_evidence = cases
        .iter()
        .map(|case| {
            let set = crate::assignment_ambiguity(&case.contract);
            let report = ambiguity_report(&VisibleReassignment, &set, HORIZON);
            let prefix = [pulse_for(case.contract.goal_channel)];
            CaseEvidence {
                kind: case.kind.label().to_string(),
                seed: case.contract.seed,
                reassigned: case.contract.reassigned(),
                initial_candidates: report.candidates,
                raw_after_pulse: raw_ambiguity(&case.contract, &prefix),
                relevant_after_pulse: relevant_ambiguity(&case.contract, &prefix),
                public_ceiling: public_ceiling(&case.contract),
                privileged_ceiling: privileged_ceiling(&case.contract),
                ambiguity_gap: report.ambiguity_gap,
            }
        })
        .collect::<Vec<_>>();

    let baselines = vec![
        baseline(&ChannelIdentity, false),
        baseline(&TagFollowing, false),
        baseline(&ValueTracking, true),
        baseline(&KnownAssignment, true),
    ];
    let bracket = analyse_bracket(
        &baselines.iter().map(bracket_entry).collect::<Vec<_>>(),
        &CaseKind::NEGATIVES
            .into_iter()
            .map(|kind| {
                (
                    kind.label().to_string(),
                    CaseKind::Witness.label().to_string(),
                )
            })
            .collect::<Vec<_>>(),
    );

    let preserving_orbits = vec![
        check_orbit_with(
            &VisibleReassignment,
            &contracts,
            "channel_labels_and_actions_permuted",
            true,
            |contract| contract.relabel_channels(),
            |action| action.flip_channel(),
            taught_drive,
        ),
        check_orbit_with(
            &VisibleReassignment,
            &contracts,
            "source_labels_permuted",
            true,
            |contract| contract.relabel_sources(),
            |action| action,
            taught_drive,
        ),
        check_orbit_with(
            &VisibleReassignment,
            &contracts,
            "common_value_scale_flipped",
            true,
            |contract| contract.flip_scale(),
            |action| action,
            taught_drive,
        ),
    ];
    let changing_orbits = vec![
        check_orbit_with(
            &VisibleReassignment,
            &contracts,
            "goal_names_the_other_source",
            false,
            |contract| contract.change_named_source(),
            |action| action,
            taught_drive,
        ),
        check_orbit_with(
            &VisibleReassignment,
            &contracts,
            "assignment_after_the_boundary_flipped",
            false,
            |contract| contract.flip_after(),
            |action| action,
            taught_drive,
        ),
    ];

    let witness = cases
        .iter()
        .find(|case| case.kind == CaseKind::Witness && case.contract.reassigned())
        .expect("a reassigned witness")
        .contract;
    let witness_set = crate::assignment_ambiguity(&witness);
    let information_orbits = vec![
        check_information_orbit(
            &VisibleReassignment,
            &witness_set,
            "values_made_invisible_across_the_boundary",
            false,
            occluded,
            HORIZON,
        ),
        check_information_orbit(
            &VisibleReassignment,
            &witness_set,
            "assignment_change_boundary_hidden",
            true,
            boundary_hidden,
            HORIZON,
        ),
        check_information_orbit(
            &VisibleReassignment,
            &witness_set,
            "common_value_scale_flipped",
            true,
            scale_flipped,
            HORIZON,
        ),
    ];

    let counterpart = Contract {
        before: witness.before.flipped(),
        after: witness.after.flipped(),
        ..witness
    };
    let noninterference = noninterference_check(
        &VisibleReassignment,
        "hidden assignments do not separate public traces before a pulse",
        &witness,
        &counterpart,
        HORIZON,
        |actions| actions.iter().all(|a| a.pulse_channel().is_none()),
        |action| action.name().into(),
    );

    let witness_public = public_ceiling(&witness);
    let occluded_public = public_ceiling(&witness.occlude());

    AuditReport {
        family: "card06a".into(),
        card: "06-perceptual-organization".into(),
        stage: "A: visible reassignment".into(),
        trunk: "T4".into(),
        contract_hash: format!("{:016x}", contract_hash()),
        cases: cases.len(),
        action_sequences_enumerated: all_sequences().len(),
        removed_from_composite: DecompositionRecord {
            composite_card: "06".into(),
            composite_contract_hash: "76a08f38947c8cae".into(),
            original_relation:
                "persistent-cause binding across hidden channel reassignment, absence, continued latent motion, matched-marginal noise, and a goal named by interaction history"
                    .into(),
            basic_relation:
                "binding the history-named source across one public channel-change boundary while both source values remain continuously visible"
                    .into(),
            removed: vec![
                "absence: the occlusion interrupt and its displaced-process semantics".into(),
                "latent evolution while absent".into(),
                "matched-marginal occlusion noise and the second channel writer it created".into(),
                "the Override coupling rule, replaced by a single-writer Conflict rule".into(),
                "the shuffled-covariance and frozen-during-absence controls, which control for factors that are gone".into(),
            ],
            retained: vec![
                "source exchangeability and the hidden assignment".into(),
                "the goal named by interaction history".into(),
                "action effects: pulse to interrogate, drive to act".into(),
                "source and channel permutation orbits".into(),
                "non-interference before the pulse".into(),
                "the channel-locked and identity-tag contrasts".into(),
                "agent-equivalence quotienting of residual label ambiguity".into(),
            ],
            falsifier: "the fixed learner cannot reach exact full-support fit here, or a channel-identity policy succeeds on this witness".into(),
            re_entry_condition: "this hash passes its own semantic audit and fixed AdmissionProfile; only then absence without hidden evolution, then evolution, then matched noise, one factor at a time".into(),
        },
        kernel_declared_for_composite: KernelUse::declared("06").expect("card 06 is declared"),
        kernel_composed: kernel_use(),
        coupling_rule: format!("{:?}", witness.coupling().rule),
        composite_coupling_rule: "Override".into(),
        case_evidence,
        kind_ceilings: CaseKind::ALL
            .into_iter()
            .map(|kind| (kind.label().to_string(), mean_public_ceiling(kind)))
            .collect(),
        baselines,
        bracket,
        noninterference,
        preserving_orbits,
        changing_orbits,
        information_orbits,
        public_gap_is_closed_by_visibility: (witness_public
            - privileged_ceiling(&witness))
        .abs()
            < 1e-9
            && occluded_public < witness_public,
        witness_public_ceiling: witness_public,
        witness_occluded_public_ceiling: occluded_public,
        boundary_marker_is_not_load_bearing: true,
        rendering: render_audit().expect("card 06a boundary"),
        no_learner_evidence:
            "World-validity evidence only; no learner was constructed, trained, or evaluated."
                .into(),
    }
}

pub fn audit_passes(report: &AuditReport) -> bool {
    report.noninterference.holds
        && report
            .preserving_orbits
            .iter()
            .chain(&report.changing_orbits)
            .all(|verdict| verdict.verdict_holds)
        && report
            .information_orbits
            .iter()
            .all(|verdict| verdict.verdict_holds)
        && report.bracket.every_negative_isolates
        && report.public_gap_is_closed_by_visibility
        && report.rendering.report.every_episode_round_trips
        && report.rendering.the_taught_drive_follows_the_perturbation
        && report.rendering.no_assignment_in_the_opening
        && report.case_evidence.len() == card_cases().len()
        && report.kernel_composed.shared_coupling
        && !report.kernel_composed.interrupt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{run_policy, Assignment, Variant};

    #[test]
    fn the_whole_audit_passes() {
        let report = audit_report();
        assert!(audit_passes(&report), "{report:#?}");
    }

    #[test]
    fn the_exact_policy_is_the_only_baseline_that_binds() {
        let report = audit_report();
        for entry in &report.baselines {
            let binds = entry.name == "value_tracking" || entry.name == "known_assignment";
            assert_eq!(entry.binding_contrast, binds, "{}", entry.name);
        }
    }

    #[test]
    fn every_witness_case_is_solved_by_following_the_perturbation() {
        for case in cases_of(CaseKind::Witness) {
            assert!(run_policy(&case.contract, &ValueTracking) > 0);
        }
        let locked = Contract::new(
            Assignment::Straight,
            Assignment::Straight,
            Variant::ChannelLocked,
            0,
            9,
        );
        assert!(run_policy(&locked, &ChannelIdentity) > 0);
    }
}
