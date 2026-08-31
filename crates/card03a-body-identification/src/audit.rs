//! The stage-A semantic audit for card 03's body-identification family.
//!
//! The composite's two hardest-won checks are the ones kept verbatim in spirit.
//!
//! The **information orbit** is where the calibration scaffold is shown to be
//! load-bearing. Making calibration uninformative leaves the value function and
//! the enumerated optimum untouched — a contract-holding solver never needed
//! the scaffold — so a value orbit would report a pass and mean nothing. What
//! it moves is the public ceiling and the identification diameter, and those are
//! what it is checked against.
//!
//! The **body/environment swap** is checked for having any bite at all. The
//! composite reports how many of its swaps are publicly visible rather than
//! claiming the transform bites everywhere; this stage reports the same count
//! and, because its withholding is uniform over every occupied cell, expects it
//! to be zero. A transform that changes nothing publicly is still a real
//! invariance over two structurally different contracts, and saying which of
//! those two things is being claimed is the point of reporting the number.

use std::collections::BTreeMap;

use pretraining_g0_contract::{
    ambiguity_report, analyse_bracket, check_information_orbit, check_orbit,
    identification_diameter, noninterference_check, AmbiguitySet, BracketStructure,
    InformationVerdict, KernelUse, KindScore, NonInterference, OrbitVerdict, PubliclyObservable,
    Symmetry,
};
use serde::{Deserialize, Serialize};

use crate::{
    all_sequences, card_cases, cases_of, contract_hash, identification_contrast, kernel_use,
    optimal_first_actions, optimal_rate, privileged_ceiling, public_ceiling,
    pulse_order_is_goal_independent, render_audit, score_policy, support_ambiguity, Action,
    BodyIdentification, Calibration, CaseKind, Contract, FollowCalibration, IdentifiedSupportExact,
    IgnoreSupport, PrivilegedBodyKnown, PublicPolicy, PublicView, HORIZON, RING,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseEvidence {
    pub kind: String,
    pub start: usize,
    pub goal: usize,
    pub identified_support: Vec<String>,
    pub announced_restoration: Option<String>,
    pub candidates: usize,
    pub diameter_after_calibration: usize,
    pub public_ceiling: f64,
    pub privileged_ceiling: f64,
    pub ambiguity_gap: f64,
    pub optimal_command: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineEvidence {
    pub name: String,
    pub is_ceiling: bool,
    pub scores: BTreeMap<String, KindScore>,
    pub optimal_rate: BTreeMap<String, f64>,
    pub identification_contrast: bool,
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
    pub case_evidence: Vec<CaseEvidence>,
    pub baselines: Vec<BaselineEvidence>,
    pub bracket: BracketStructure,
    pub noninterference: NonInterference,
    pub preserving_orbits: Vec<OrbitVerdict>,
    pub changing_orbits: Vec<OrbitVerdict>,
    pub information_orbits: Vec<InformationVerdict>,
    /// The scaffold's pulse order is a constant and carries no goal content.
    pub pulse_order_is_goal_independent: bool,
    /// Calibration identifies the body exactly on every case.
    pub calibration_identifies_the_body_exactly: bool,
    /// How many body/environment swaps are visible in a public trace.
    pub publicly_visible_swaps: usize,
    pub swappable_contracts: usize,
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
        identification_contrast: identification_contrast(policy),
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

fn uninformative(set: &AmbiguitySet<Contract>) -> AmbiguitySet<Contract> {
    AmbiguitySet::uniform(
        set.candidates
            .iter()
            .map(|contract| contract.with_calibration(Calibration::Uninformative))
            .collect(),
    )
}

fn rotated(set: &AmbiguitySet<Contract>) -> AmbiguitySet<Contract> {
    let symmetry = Symmetry {
        shift: 3,
        reflect: false,
        cells: RING,
    };
    AmbiguitySet::uniform(
        set.candidates
            .iter()
            .map(|contract| contract.relabelled(symmetry))
            .collect(),
    )
}

/// How many body/environment swaps a public trace can tell apart.
///
/// Reported rather than assumed, because the composite's version of this number
/// is not zero and inheriting either answer would be a claim rather than a
/// measurement.
fn visible_swaps() -> (usize, usize) {
    let cases = card_cases();
    let mut visible = 0usize;
    let mut swappable = 0usize;
    for case in &cases {
        let swapped = case.contract.swapped_to_environment();
        if swapped == case.contract {
            continue;
        }
        swappable += 1;
        let differs = all_sequences().into_iter().any(|sequence| {
            BodyIdentification.public_trace(&case.contract, &sequence)
                != BodyIdentification.public_trace(&swapped, &sequence)
        });
        if differs {
            visible += 1;
        }
    }
    (visible, swappable)
}

pub fn audit_report() -> AuditReport {
    let cases = card_cases();
    let contracts: Vec<Contract> = cases.iter().map(|case| case.contract.clone()).collect();

    let case_evidence = cases
        .iter()
        .map(|case| {
            let set = support_ambiguity(&case.contract);
            let report = ambiguity_report(&BodyIdentification, &set, HORIZON);
            CaseEvidence {
                kind: case.kind.label().to_string(),
                start: case.contract.start,
                goal: case.contract.goal(),
                identified_support: PublicView::identify(&case.contract)
                    .into_iter()
                    .map(|action| action.name().to_string())
                    .collect(),
                announced_restoration: case
                    .contract
                    .announced_restoration()
                    .and_then(Action::from_index)
                    .map(|action| action.name().to_string()),
                candidates: report.candidates,
                diameter_after_calibration: identification_diameter(&BodyIdentification, &set, &[]),
                public_ceiling: public_ceiling(&case.contract),
                privileged_ceiling: privileged_ceiling(&case.contract),
                ambiguity_gap: report.ambiguity_gap,
                optimal_command: optimal_first_actions(&case.contract)
                    .into_iter()
                    .map(|action| action.name().to_string())
                    .collect(),
            }
        })
        .collect::<Vec<_>>();

    let baselines = vec![
        baseline(&IgnoreSupport, false),
        baseline(&FollowCalibration, false),
        baseline(&IdentifiedSupportExact, true),
        baseline(&PrivilegedBodyKnown, true),
    ];
    let bracket = analyse_bracket(
        &baselines.iter().map(bracket_entry).collect::<Vec<_>>(),
        &CaseKind::NEGATIVES
            .into_iter()
            .map(|kind| {
                (
                    kind.label().to_string(),
                    CaseKind::WitnessIdentifiedSupport.label().to_string(),
                )
            })
            .collect::<Vec<_>>(),
    );

    let rotation = Symmetry {
        shift: 3,
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
            &BodyIdentification,
            &contracts,
            "configuration_rotated",
            true,
            |contract| contract.relabelled(rotation),
            |action| action,
        ),
        check_orbit(
            &BodyIdentification,
            &contracts,
            "body_limitation_expressed_as_environment_deletion",
            true,
            |contract| contract.swapped_to_environment(),
            |action| action,
        ),
    ];
    let changing_orbits = vec![
        check_orbit(
            &BodyIdentification,
            &contracts,
            "aliased_support_exchanged",
            false,
            |contract| contract.swap_aliased_support(),
            |action| action,
        ),
        // Declared meaning-changing rather than preserving: a reflection sends
        // the `+2` edge to a `-2` edge, and this body has no actuator for it.
        // The composite's invariance group is rotations only for exactly this
        // reason, and the verdict is what establishes it.
        check_orbit(
            &BodyIdentification,
            &contracts,
            "configuration_reflected",
            false,
            |contract| contract.relabelled(reflection),
            |action| action,
        ),
    ];

    let witness = cases
        .iter()
        .find(|case| case.kind == CaseKind::WitnessIdentifiedSupport)
        .expect("a witness")
        .contract
        .clone();
    let witness_set = support_ambiguity(&witness);
    let information_orbits = vec![
        check_information_orbit(
            &BodyIdentification,
            &witness_set,
            "calibration_made_uninformative",
            false,
            uninformative,
            HORIZON,
        ),
        check_information_orbit(
            &BodyIdentification,
            &witness_set,
            "configuration_rotated",
            true,
            rotated,
            HORIZON,
        ),
    ];

    let blind = witness.with_calibration(Calibration::Uninformative);
    let noninterference = noninterference_check(
        &BodyIdentification,
        "an unidentified body does not separate public traces before the decision",
        &blind,
        &blind.swap_aliased_support(),
        HORIZON,
        |actions| actions.is_empty(),
        |action| action.name().into(),
    );

    let (visible, swappable) = visible_swaps();

    AuditReport {
        family: "card03a".into(),
        card: "03-affordance".into(),
        stage: "A: public body identification".into(),
        trunk: "T1".into(),
        contract_hash: format!("{:016x}", contract_hash()),
        cases: cases.len(),
        action_sequences_enumerated: all_sequences().len(),
        removed_from_composite: DecompositionRecord {
            composite_card: "03".into(),
            composite_contract_hash: "2442b372a18e1d66".into(),
            original_relation:
                "before a failed attempt, change action allocation according to what this body can bring about in this environment, including an immediate minimum-cost fallback when the goal is unreachable"
                    .into(),
            basic_relation:
                "after the free exact calibration scaffold, drive the published goal cell with the command this body actually supports"
                    .into(),
            removed: vec![
                "the planning and fallback decision, and the Fallback action".into(),
                "the unreachable goal".into(),
                "the two-decision budget that made planning necessary".into(),
                "the frequency-matched budget control, which controls for a budget that is gone".into(),
                "the no-restore negative, which was the unreachable-fallback witness".into(),
            ],
            retained: vec![
                "the nine-cell ring and its rotation-only invariance group".into(),
                "the free mandatory exact calibration scaffold and its constant pulse order".into(),
                "hidden actuator support with exact post-calibration identification".into(),
                "the announced restoration control".into(),
                "the body/environment swap invariance".into(),
                "the uninformative-calibration information orbit".into(),
                "the ignore-support baseline".into(),
            ],
            falsifier: "the fixed learner cannot reach exact full-support fit here, or an ignore-support policy succeeds on this witness".into(),
            re_entry_condition: "this hash passes its own semantic audit and fixed AdmissionProfile; only then the fallback decision, then the frequency-matched budget control, then unreachable goals".into(),
        },
        kernel_declared_for_composite: KernelUse::declared("03").expect("card 03 is declared"),
        kernel_composed: kernel_use(),
        case_evidence,
        baselines,
        bracket,
        noninterference,
        preserving_orbits,
        changing_orbits,
        information_orbits,
        pulse_order_is_goal_independent: pulse_order_is_goal_independent(),
        calibration_identifies_the_body_exactly: card_cases().into_iter().all(|case| {
            let identified = PublicView::identify(&case.contract);
            Action::MOVEMENTS.into_iter().all(|action| {
                identified.contains(&action) == case.contract.support.contains(action.index())
            })
        }),
        publicly_visible_swaps: visible,
        swappable_contracts: swappable,
        rendering: render_audit().expect("card 03a boundary"),
        no_learner_evidence:
            "World-validity evidence only; no learner was constructed, trained, or evaluated."
                .into(),
    }
}

pub fn audit_passes(report: &AuditReport) -> bool {
    report.pulse_order_is_goal_independent
        && report.calibration_identifies_the_body_exactly
        && report.noninterference.holds
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
        && report.rendering.report.every_episode_round_trips
        && report.rendering.the_taught_command_is_the_supported_one
        && report.rendering.the_scaffold_shape_is_body_independent
        && report.kernel_composed == report.kernel_declared_for_composite
        && report.case_evidence.len() == card_cases().len()
        && !cases_of(CaseKind::WitnessIdentifiedSupport).is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_whole_audit_passes() {
        let report = audit_report();
        assert!(audit_passes(&report), "{report:#?}");
    }

    #[test]
    fn every_case_identifies_exactly_and_has_no_scored_gap() {
        let report = audit_report();
        for case in &report.case_evidence {
            assert_eq!(case.diameter_after_calibration, 1, "{case:?}");
            assert_eq!(case.ambiguity_gap, 0.0, "{case:?}");
        }
    }

    #[test]
    fn the_scaffold_is_load_bearing_only_in_the_information_orbit() {
        let report = audit_report();
        let uninformative = report
            .information_orbits
            .iter()
            .find(|verdict| verdict.transform == "calibration_made_uninformative")
            .expect("the orbit");
        assert!(uninformative.public_ceiling_after < uninformative.public_ceiling_before);
        assert!(uninformative.diameter_after > uninformative.diameter_before);
        assert!(uninformative.verdict_holds);
    }
}
