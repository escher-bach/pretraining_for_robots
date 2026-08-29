use crate::{
    assignment_ambiguity, card_cases, contract_hash, exact_assignment_posterior_value, kernel_use,
    mean_value, relevant_ambiguity, render_audit, Action, AssumeNothingMoved, CaseKind,
    ChannelIdentity, KnownAssignment, PerChannel, PerceptualOrganization, TagFollowing, HORIZON,
};
use pretraining_g0_contract::{
    ambiguity_report, check_information_orbit, check_orbit, check_orbit_with,
    identification_diameter, noninterference_check, optimal_actions_from, InformationVerdict,
    KernelUse, NonInterference, OrbitVerdict,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseEvidence {
    pub kind: String,
    pub seed: u64,
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
    pub values: BTreeMap<String, f64>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditReport {
    pub card: String,
    pub trunk: String,
    pub contract_hash: String,
    pub action_sequences_enumerated: usize,
    pub cases: Vec<CaseEvidence>,
    pub kernel_declared: KernelUse,
    pub kernel_composed: KernelUse,
    pub coupling_rule: String,
    pub witness_interrupt: String,
    pub frozen_control_interrupt: String,
    pub occluded_writer_order: Vec<String>,
    pub noninterference: NonInterference,
    pub baselines: Vec<BaselineEvidence>,
    pub preserving_orbits: Vec<OrbitVerdict>,
    pub changing_orbits: Vec<OrbitVerdict>,
    pub information_orbits: Vec<InformationVerdict>,
    pub rendering: crate::RenderAudit,
    pub no_learner_evidence: String,
}
fn baseline(name: &str, values: BTreeMap<String, f64>) -> BaselineEvidence {
    BaselineEvidence {
        name: name.into(),
        values,
    }
}
fn second_actions(_: &PerceptualOrganization, c: &crate::Contract) -> Vec<Action> {
    optimal_actions_from(
        &PerceptualOrganization,
        c,
        &[if c.goal_channel == 0 {
            Action::PulseZero
        } else {
            Action::PulseOne
        }],
    )
}

pub fn audit_report() -> AuditReport {
    let cases = card_cases();
    let evidence = cases
        .iter()
        .map(|case| {
            let set = assignment_ambiguity(&case.contract);
            let report = ambiguity_report(&PerceptualOrganization, &set, HORIZON);
            let pulse = if case.contract.goal_channel == 0 {
                Action::PulseZero
            } else {
                Action::PulseOne
            };
            CaseEvidence {
                kind: case.kind.label().into(),
                seed: case.contract.seed,
                initial_candidates: report.candidates,
                raw_after_pulse: identification_diameter(&PerceptualOrganization, &set, &[pulse]),
                relevant_after_pulse: relevant_ambiguity(&case.contract, &[pulse]),
                public_ceiling: report.public_ceiling,
                privileged_ceiling: report.privileged_ceiling,
                ambiguity_gap: report.ambiguity_gap,
            }
        })
        .collect();
    let baselines = vec![
        baseline(
            "per_channel",
            CaseKind::ALL
                .into_iter()
                .map(|k| (k.label().into(), mean_value(&PerChannel, k)))
                .collect(),
        ),
        baseline(
            "channel_identity",
            CaseKind::ALL
                .into_iter()
                .map(|k| (k.label().into(), mean_value(&ChannelIdentity, k)))
                .collect(),
        ),
        baseline(
            "tag_following",
            CaseKind::ALL
                .into_iter()
                .map(|k| (k.label().into(), mean_value(&TagFollowing, k)))
                .collect(),
        ),
        baseline(
            "assume_nothing_moved",
            CaseKind::ALL
                .into_iter()
                .map(|k| (k.label().into(), mean_value(&AssumeNothingMoved, k)))
                .collect(),
        ),
        baseline(
            "exact_assignment_posterior",
            CaseKind::ALL
                .into_iter()
                .map(|k| (k.label().into(), exact_assignment_posterior_value(k)))
                .collect(),
        ),
        baseline(
            "known_assignment",
            CaseKind::ALL
                .into_iter()
                .map(|k| (k.label().into(), mean_value(&KnownAssignment, k)))
                .collect(),
        ),
    ];
    let contracts: Vec<_> = cases.iter().map(|case| case.contract).collect();
    let preserving_orbits = vec![
        check_orbit(
            &PerceptualOrganization,
            &contracts,
            "channel_labels_and_actions_permuted",
            true,
            |c| c.relabel_channels(),
            |a| a.flip_channel(),
        ),
        check_orbit(
            &PerceptualOrganization,
            &contracts,
            "source_labels_permuted",
            true,
            |c| c.relabel_sources(),
            |a| a,
        ),
        check_orbit(
            &PerceptualOrganization,
            &contracts,
            "common_value_scale_flipped",
            true,
            |c| c.flip_scale(),
            |a| a,
        ),
        check_orbit(
            &PerceptualOrganization,
            &contracts,
            "occlusion_boundary_timing_flipped",
            true,
            |c| c.flip_timing(),
            |a| a,
        ),
    ];
    let changing_orbits = vec![
        check_orbit_with(
            &PerceptualOrganization,
            &contracts,
            "goal_history_channel_changed",
            false,
            |c| c.change_goal(),
            |a| a,
            second_actions,
        ),
        check_orbit(
            &PerceptualOrganization,
            &contracts,
            "occlusion_noise_visibly_mismatched",
            false,
            |c| c.mismatched_noise(),
            |a| a,
        ),
    ];
    let witness = cases
        .iter()
        .find(|case| case.kind == CaseKind::Witness)
        .expect("witness")
        .contract;
    let information_orbits = vec![check_information_orbit(
        &PerceptualOrganization,
        &assignment_ambiguity(&witness),
        "assignment_boundary_is_hidden",
        false,
        |set| {
            let candidates = set.candidates.iter().map(|c| c.hide_boundary()).collect();
            pretraining_g0_contract::AmbiguitySet::uniform(candidates)
        },
        HORIZON,
    )];
    let counterpart = crate::Contract {
        before: witness.before.flipped(),
        after: witness.after.flipped(),
        ..witness
    };
    let noninterference = noninterference_check(
        &PerceptualOrganization,
        "hidden assignments do not separate before a pulse",
        &witness,
        &counterpart,
        HORIZON,
        |actions| actions.iter().all(|a| a.pulse_channel().is_none()),
        |a| a.name().into(),
    );
    let frozen = cases
        .iter()
        .find(|case| case.kind == CaseKind::FrozenDuringAbsence)
        .expect("frozen")
        .contract;
    AuditReport {
        card: "06-perceptual-organization".into(),
        trunk: "T4".into(),
        contract_hash: format!("{:016x}", contract_hash()),
        action_sequences_enumerated: crate::all_sequences().len(),
        cases: evidence,
        kernel_declared: KernelUse::declared("06").unwrap(),
        kernel_composed: kernel_use(),
        coupling_rule: format!("{:?}", witness.coupling().rule),
        witness_interrupt: format!("{:?}", witness.occlusion().displaced),
        frozen_control_interrupt: format!("{:?}", frozen.occlusion().displaced),
        occluded_writer_order: vec!["source".into(), "matched_marginal_noise".into()],
        noninterference,
        baselines,
        preserving_orbits,
        changing_orbits,
        information_orbits,
        rendering: render_audit().expect("card 06 boundary"),
        no_learner_evidence:
            "World-validity evidence only; no learner was constructed, trained, or evaluated."
                .into(),
    }
}
