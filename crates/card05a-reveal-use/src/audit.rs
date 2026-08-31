//! The stage-A semantic audit for card 05's reveal-use family.
//!
//! Three things here are not in the composite's audit and exist because this is
//! a *decomposition* rather than a new card:
//!
//! 1. `removed_from_composite` states which decision left, in the vocabulary of
//!    the certificate, so a reader can check the implementation against it;
//! 2. `no_action_buys_information` is the executable form of "the purchase
//!    decision is gone" — it is [`epistemic_value`] over every action rather
//!    than a claim; and
//! 3. `matched_control_verdict` is still run, on `Sham` against itself, and is
//!    reported as *not holding* on its ambiguity clause. That is the correct
//!    answer for this family and the reason to run it: a stage that quietly
//!    dropped the check would leave the restoration in stage B unanchored.

use std::collections::BTreeMap;

use pretraining_g0_contract::{
    ambiguity_report, analyse_bracket, check_information_orbit, check_orbit, epistemic_value,
    matched_control_verdict, noninterference_check, AmbiguitySet, BracketStructure,
    InformationVerdict, KernelUse, KindScore, MatchedControlVerdict, NonInterference, OrbitVerdict,
};
use serde::{Deserialize, Serialize};

use crate::{
    all_sequences, attains_public_ceiling, card_cases, cases_of, contract_hash, instance_ambiguity,
    kernel_use, mean_public_ceiling, mean_value, privileged_ceiling, public_ceiling, relabel,
    render_audit, reveal_use_contrast, score_policy, Action, Bit, BlindCommit, CaseKind, Contract,
    ExactPublic, GateCoupling, PrivilegedGateKnown, PublicPolicy, RevealMode, RevealUse,
    ShamThenBlindCommit, HORIZON,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseEvidence {
    pub kind: String,
    pub gate: String,
    pub decoy: String,
    pub candidates: usize,
    pub initial_diameter: usize,
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
    pub attains_public_ceiling: BTreeMap<String, bool>,
    pub follows_the_gate: BTreeMap<String, f64>,
    pub reveal_use_contrast: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KindCeiling {
    pub kind: String,
    pub mean_public_ceiling: f64,
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
    pub kind_ceilings: Vec<KindCeiling>,
    pub baselines: Vec<BaselineEvidence>,
    pub bracket: BracketStructure,
    /// Whether any action buys information the episode can still use.
    ///
    /// A commit reduces the surviving set, because its outcome is published;
    /// that reduction arrives after the irreversible decision and is not a
    /// purchase. The flag therefore quantifies over non-committing actions and
    /// additionally requires that no opening beat an immediate commit.
    pub no_action_buys_information: bool,
    pub epistemic_value_of_every_action: Vec<(String, f64, usize)>,
    pub matched_control: MatchedControlVerdict,
    pub noninterference: NonInterference,
    pub preserving_orbits: Vec<OrbitVerdict>,
    pub changing_orbits: Vec<OrbitVerdict>,
    pub information_orbits: Vec<InformationVerdict>,
    pub rendering: crate::RenderAudit,
    pub no_learner_evidence: String,
}

fn baseline<P: PublicPolicy>(policy: &P, is_ceiling: bool) -> BaselineEvidence {
    BaselineEvidence {
        name: policy.name().to_string(),
        is_ceiling,
        scores: score_policy(policy),
        mean_value: CaseKind::ALL
            .into_iter()
            .map(|kind| (kind.label().to_string(), mean_value(policy, kind)))
            .collect(),
        attains_public_ceiling: CaseKind::ALL
            .into_iter()
            .map(|kind| {
                (
                    kind.label().to_string(),
                    attains_public_ceiling(policy, kind),
                )
            })
            .collect(),
        follows_the_gate: CaseKind::ALL
            .into_iter()
            .map(|kind| {
                (
                    kind.label().to_string(),
                    crate::gate_following_rate(policy, kind),
                )
            })
            .collect(),
        reveal_use_contrast: reveal_use_contrast(policy),
    }
}

fn bracket_entry(entry: &BaselineEvidence) -> pretraining_g0_contract::BaselineEvidence {
    pretraining_g0_contract::BaselineEvidence {
        name: entry.name.clone(),
        scores: entry.scores.clone(),
        optimal_on_negatives: CaseKind::NEGATIVES
            .into_iter()
            .filter(|kind| entry.attains_public_ceiling[kind.label()])
            .map(|kind| kind.label().to_string())
            .collect(),
        is_ceiling: entry.is_ceiling,
    }
}

/// The witness ambiguity set with the reveal publishing a decoy instead of the
/// gate: the uninformative-reveal transformation.
fn uninformative(set: &AmbiguitySet<Contract>) -> AmbiguitySet<Contract> {
    AmbiguitySet::uniform(
        set.candidates
            .iter()
            .map(|contract| contract.with_reveal(RevealMode::PublishesDecoy))
            .collect(),
    )
}

fn withheld(set: &AmbiguitySet<Contract>) -> AmbiguitySet<Contract> {
    AmbiguitySet::uniform(
        set.candidates
            .iter()
            .map(|contract| contract.with_reveal(RevealMode::Withholds))
            .collect(),
    )
}

fn decoy_flipped(set: &AmbiguitySet<Contract>) -> AmbiguitySet<Contract> {
    AmbiguitySet::uniform(
        set.candidates
            .iter()
            .map(|contract| contract.with_flipped_decoy())
            .collect(),
    )
}

pub fn audit_report() -> AuditReport {
    let cases = card_cases();
    let contracts: Vec<Contract> = cases.iter().map(|case| case.contract).collect();

    let case_evidence = cases
        .iter()
        .map(|case| {
            let set = instance_ambiguity(&case.contract);
            let report = ambiguity_report(&RevealUse, &set, HORIZON);
            CaseEvidence {
                kind: case.kind.label().to_string(),
                gate: case.contract.gate.name().to_string(),
                decoy: case.contract.decoy.name().to_string(),
                candidates: report.candidates,
                initial_diameter: report.initial_diameter,
                public_ceiling: public_ceiling(&case.contract),
                privileged_ceiling: privileged_ceiling(&case.contract),
                ambiguity_gap: report.ambiguity_gap,
            }
        })
        .collect();

    let baselines = vec![
        baseline(&BlindCommit, false),
        baseline(&ShamThenBlindCommit, false),
        baseline(&ExactPublic, true),
        baseline(&PrivilegedGateKnown, true),
    ];
    let bracket = analyse_bracket(
        &baselines.iter().map(bracket_entry).collect::<Vec<_>>(),
        &CaseKind::NEGATIVES
            .into_iter()
            .map(|kind| {
                (
                    kind.label().to_string(),
                    CaseKind::WitnessRevealThenCommit.label().to_string(),
                )
            })
            .collect::<Vec<_>>(),
    );

    let hidden = CaseKind::NegativeGateHidden.contract_of(Bit::Left, Bit::Left);
    let hidden_set = instance_ambiguity(&hidden);
    let epistemic: Vec<(String, f64, usize)> =
        epistemic_value(&RevealUse, &hidden_set, HORIZON, |a| a.name().into())
            .into_iter()
            .map(|entry| (entry.action, entry.public_value, entry.ambiguity_reduction))
            .collect();
    let no_purchase = epistemic
        .iter()
        .filter(|(name, _, _)| {
            !Action::ALL
                .into_iter()
                .any(|action| action.is_commit() && action.name() == name)
        })
        .all(|(_, _, reduction)| *reduction == 0)
        && epistemic
            .iter()
            .map(|(_, value, _)| *value)
            .fold(f64::NEG_INFINITY, f64::max)
            <= public_ceiling(&hidden) + 1e-9;

    // Run the composite's matched-control check on the seat the probe used to
    // occupy. It reports that `Sham` reduces nothing, which is the audited form
    // of "there is nothing here to buy".
    let matched_control = matched_control_verdict(
        &RevealUse,
        &hidden_set,
        Action::Sham,
        Action::Sham,
        |_| 1,
        |_| 0,
        |action| action.name().into(),
    );

    let noninterference = noninterference_check(
        &RevealUse,
        "a withheld gate does not separate public traces before a commit",
        &hidden,
        &hidden.with_flipped_gate(),
        HORIZON,
        |actions| actions.iter().all(|action| !action.is_commit()),
        |action| action.name().into(),
    );

    let reflection = pretraining_g0_contract::Symmetry {
        shift: 0,
        reflect: true,
        cells: crate::RING,
    };
    let preserving_orbits = vec![
        check_orbit(
            &RevealUse,
            &contracts,
            "gate_and_commit_labels_exchanged",
            true,
            |contract| relabel(contract, reflection),
            |action| action.mirrored(),
        ),
        check_orbit(
            &RevealUse,
            &contracts,
            "decoy_flipped",
            true,
            |contract| contract.with_flipped_decoy(),
            |action| action,
        ),
    ];
    let changing_orbits = vec![
        check_orbit(
            &RevealUse,
            &contracts,
            "commits_made_equally_valuable",
            false,
            |contract| contract.with_coupling(GateCoupling::Irrelevant),
            |action| action,
        ),
        check_orbit(
            &RevealUse,
            &contracts,
            "gate_flipped_without_relabelling_the_commits",
            false,
            |contract| contract.with_flipped_gate(),
            |action| action,
        ),
    ];

    let witness = CaseKind::WitnessRevealThenCommit.contract_of(Bit::Left, Bit::Left);
    let witness_set = instance_ambiguity(&witness);
    let information_orbits = vec![
        check_information_orbit(
            &RevealUse,
            &witness_set,
            "reveal_made_uninformative",
            false,
            uninformative,
            HORIZON,
        ),
        check_information_orbit(
            &RevealUse,
            &witness_set,
            "reveal_withheld",
            false,
            withheld,
            HORIZON,
        ),
        check_information_orbit(
            &RevealUse,
            &witness_set,
            "decoy_flipped",
            true,
            decoy_flipped,
            HORIZON,
        ),
    ];

    AuditReport {
        family: "card05a".into(),
        card: "05-active-experimentation".into(),
        stage: "A: reveal use".into(),
        trunk: "T2".into(),
        contract_hash: format!("{:016x}", contract_hash()),
        cases: cases.len(),
        action_sequences_enumerated: all_sequences().len(),
        removed_from_composite: DecompositionRecord {
            composite_card: "05".into(),
            composite_contract_hash: "cbe39880124b9d2d".into(),
            original_relation: "pay for information exactly when it changes a later decision"
                .into(),
            basic_relation:
                "commit to the action favoured by a gate a free reveal has already published".into(),
            removed: vec![
                "the Probe action and the gate reveal it bought".into(),
                "the probe cost and the raised-cost control".into(),
                "the Peek action and the inconsequential second bit".into(),
                "the value-of-information comparison against blind commitment".into(),
            ],
            retained: vec![
                "the hidden binary gate and its uniform prior".into(),
                "the irreversible commit and its two gated outcomes".into(),
                "the matched-cost non-informative Sham action".into(),
                "the gate-hidden and equally-valuable controls".into(),
                "the uninformative-reveal information orbit".into(),
                "non-interference before the reveal fires".into(),
                "expectation-over-case-kind scoring".into(),
                "the shared budget restriction".into(),
            ],
            falsifier: "the fixed learner cannot reach exact full-support fit here, or a fixed blind-commit policy succeeds on this witness".into(),
            re_entry_condition: "this hash passes its own semantic audit and fixed AdmissionProfile; only then a free probe, then a matched-cost probe, then Peek and raised cost".into(),
        },
        kernel_declared_for_composite: KernelUse::declared("05").expect("card 05 is declared"),
        kernel_composed: kernel_use(),
        case_evidence,
        kind_ceilings: CaseKind::ALL
            .into_iter()
            .map(|kind| KindCeiling {
                kind: kind.label().to_string(),
                mean_public_ceiling: mean_public_ceiling(kind),
            })
            .collect(),
        baselines,
        bracket,
        no_action_buys_information: no_purchase,
        epistemic_value_of_every_action: epistemic,
        matched_control,
        noninterference,
        preserving_orbits,
        changing_orbits,
        information_orbits,
        rendering: render_audit().expect("card 05a boundary"),
        no_learner_evidence:
            "World-validity evidence only; no learner was constructed, trained, or evaluated."
                .into(),
    }
}

/// Whether every semantic verdict in the report holds.
///
/// The matched control is excluded deliberately: this family has no informative
/// action, so its ambiguity clause is *supposed* to fail, and folding it into a
/// conjunction would either hide that or make the family look broken.
pub fn audit_passes(report: &AuditReport) -> bool {
    report.no_action_buys_information
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
        && report.rendering.the_witness_teaches_one_commit
        && report
            .rendering
            .a_withheld_gate_is_invisible_before_the_commit
        && report.kernel_composed == report.kernel_declared_for_composite
        && report.case_evidence.len() == card_cases().len()
        && !cases_of(CaseKind::WitnessRevealThenCommit).is_empty()
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
    fn the_matched_control_reports_the_missing_purchase_rather_than_hiding_it() {
        let report = audit_report();
        assert!(report.matched_control.equal_cost);
        assert!(report.matched_control.equal_immediate_value_movement);
        assert!(
            !report
                .matched_control
                .only_the_informative_action_reduces_ambiguity,
            "stage A has no informative action; the verdict must say so"
        );
    }

    #[test]
    fn the_witness_has_no_ambiguity_gap_and_the_hidden_control_has_all_of_it() {
        let report = audit_report();
        for case in &report.case_evidence {
            if case.kind == CaseKind::WitnessRevealThenCommit.label() {
                assert_eq!(case.ambiguity_gap, 0.0);
            }
            if case.kind == CaseKind::NegativeGateHidden.label() {
                assert!(case.ambiguity_gap > 49.0);
            }
        }
    }
}
