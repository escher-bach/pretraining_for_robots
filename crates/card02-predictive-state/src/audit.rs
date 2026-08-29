//! The exact audit for card 02: memory value, non-interference, orbits, and
//! brackets.
//!
//! Three instrument choices are specific to this card and each one is a
//! correction of the obvious default.
//!
//! **The observable is the discriminating command, not the first action.** Both
//! modes start with the same move, so `optimal_first_actions` reports agreement
//! exactly where the card claims a difference. The orbit checks read
//! [`crate::discriminating_actions`] through `check_orbit_with`.
//!
//! **The gap that matters is against an ablated history, not against privileged
//! state.** The mode is published, so the ordinary privileged-versus-public gap
//! is zero and saying so says nothing — the same vacuity card 04's audit had.
//! The load-bearing number is the full public ceiling minus the ceiling of a
//! policy whose trace has had the latch removed.
//!
//! **Two of the declared transformations are invisible to the value orbit.**
//! Ending the aliasing interval early leaves the world's ceiling and its optimal
//! actions untouched — a contract-holding solver never needed the latch. What it
//! changes is what a learner without the latch can attain, so it is checked
//! against the *coarsened* fragment through `check_information_orbit`.

use std::collections::BTreeMap;

use pretraining_g0_contract::{
    ablated_policy_value, analyse_bracket, check_information_orbit, check_orbit_with,
    identification_diameter, noninterference_check, privileged_value_bound, public_policy_value,
    AmbiguitySet, BaselineEvidence, BracketStructure, Coarsened, InformationVerdict, KernelUse,
    KindScore, NonInterference, OrbitVerdict, Symmetry,
};
use serde::{Deserialize, Serialize};

use crate::{
    card_cases, contract_hash, discriminating_actions, kernel_use, mode_ambiguity, render_audit,
    retention_contrast, run_policy, score_policy, taught_sequence, transform, value_bounds,
    without_the_decoy, without_the_latch, Action, Case, CaseKind, ConstantCommand, Contract,
    LastLatch, Memoryless, Mode, ModeConditioned, ModeCoupling, ModeVisibility, PredictiveState,
    PublicPolicy, RenderAudit, Window, CONFIGURATION, DISCRIMINATING_STEP, HORIZON, RING,
};

/// One case's exact bracket.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseBracket {
    pub kind: String,
    pub mode: String,
    pub decoy: Option<String>,
    pub ceiling: i32,
    /// The correct command at the discriminating decision, having taken the
    /// mode-independent route to it.
    pub discriminating_actions: Vec<String>,
    /// The first action, which is the same in both modes and is reported so the
    /// reason this card cannot use it as its observable is visible.
    pub optimal_first_actions: Vec<String>,
}

fn case_bracket(case: &Case) -> CaseBracket {
    CaseBracket {
        kind: case.kind.label().to_string(),
        mode: case.contract.mode.name().to_string(),
        decoy: case.contract.decoy.map(|mode| mode.name().to_string()),
        ceiling: value_bounds(&case.contract).0,
        discriminating_actions: discriminating_actions(&case.contract)
            .into_iter()
            .map(|action| action.name().to_string())
            .collect(),
        optimal_first_actions: crate::optimal_first_actions(&case.contract)
            .into_iter()
            .map(|action| action.name().to_string())
            .collect(),
    }
}

/// What retaining the latch is worth, exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryValue {
    pub kind: String,
    pub mode: String,
    /// The ceiling of the best policy measurable in the whole public trace.
    pub public_ceiling: f64,
    /// The same policy class with the latch removed from the trace.
    pub ceiling_without_the_latch: f64,
    /// Their difference: what the memory buys.
    pub memory_value: f64,
    /// The same ablation applied to the decoy instead. Zero everywhere is the
    /// card's second claim — that the extra latch is genuinely worthless.
    pub decoy_value: f64,
    /// Privileged minus public, which is zero because the mode is published.
    /// Reported so its vacuity is visible rather than reassuring.
    pub vacuous_privileged_gap: f64,
    /// Decisions from the latch to the decision that needs it.
    pub required_memory_span: usize,
}

fn memory_value(case: &Case) -> MemoryValue {
    let set = mode_ambiguity(&case.contract);
    let public = public_policy_value(&PredictiveState, &set, HORIZON);
    let ablated = ablated_policy_value(&PredictiveState, &set, HORIZON, without_the_latch);
    let decoy_ablated = ablated_policy_value(&PredictiveState, &set, HORIZON, without_the_decoy);
    MemoryValue {
        kind: case.kind.label().to_string(),
        mode: case.contract.mode.name().to_string(),
        public_ceiling: public,
        ceiling_without_the_latch: ablated,
        memory_value: public - ablated,
        decoy_value: public - decoy_ablated,
        vacuous_privileged_gap: privileged_value_bound(&PredictiveState, &set, HORIZON) - public,
        required_memory_span: DISCRIMINATING_STEP + 1,
    }
}

/// Whether the aliasing interval leaks the mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AliasingReport {
    /// No sequence confined to the aliasing interval separates the two modes.
    pub interval_is_opaque: NonInterference,
    /// With the latch ablated, the diameter stays at two throughout it.
    pub diameter_during_aliasing_without_the_latch: Vec<usize>,
    /// Probing is possible only once the interval ends, by which point there is
    /// no decision left to use the answer.
    pub the_mode_cannot_be_probed_in_time: bool,
}

fn aliasing_report() -> AliasingReport {
    let witness = card_cases()
        .into_iter()
        .find(|case| case.kind == CaseKind::WitnessLatchedMode)
        .expect("the card has this witness")
        .contract;
    let flipped = witness.with_flipped_mode();

    // Restricted to prefixes shorter than the discriminating decision: those are
    // exactly the histories inside the aliasing interval.
    let opaque = noninterference_check(
        &Coarsened::new(&PredictiveState, without_the_latch),
        "the aliasing interval separates no two modes",
        &witness,
        &flipped,
        DISCRIMINATING_STEP,
        |_: &[Action]| true,
        |action: Action| action.name().to_string(),
    );

    let blind = Coarsened::new(&PredictiveState, without_the_latch);
    let set = mode_ambiguity(&witness);
    let mut diameters = Vec::new();
    for length in 0..DISCRIMINATING_STEP {
        let probe = vec![Action::Cross; length];
        diameters.push(identification_diameter(&blind, &set, &probe));
    }

    AliasingReport {
        the_mode_cannot_be_probed_in_time: diameters.iter().all(|diameter| *diameter == 2),
        interval_is_opaque: opaque,
        diameter_during_aliasing_without_the_latch: diameters,
    }
}

/// The value orbit, read against the discriminating command.
pub fn orbit_verdicts() -> Vec<OrbitVerdict> {
    let contracts: Vec<Contract> = card_cases().into_iter().map(|case| case.contract).collect();
    let identity = |action: Action| action;
    let discriminating =
        |_: &PredictiveState, contract: &Contract| discriminating_actions(contract);
    let mut verdicts = Vec::new();

    // The whole dihedral group preserves this card: `Step` is the only unit
    // move and the two mode-sensitive commands are already each other's mirror.
    for symmetry in CONFIGURATION.symmetries() {
        if symmetry == Symmetry::identity(RING) {
            continue;
        }
        verdicts.push(check_orbit_with(
            &PredictiveState,
            &contracts,
            &symmetry.name(),
            true,
            |contract| transform(contract, symmetry),
            identity,
            discriminating,
        ));
    }

    // Relabelling the modes and exchanging the two commands is the card's
    // stated permutation. It has to be checked *with* the action map: the
    // correct command changes name, and requiring it to stay the same would
    // reject an invariance the card claims.
    verdicts.push(check_orbit_with(
        &PredictiveState,
        &contracts,
        "mode_labels_and_the_two_commands_are_exchanged_together",
        true,
        |contract| contract.with_flipped_mode(),
        |action| match action {
            Action::Cross => Action::Anti,
            Action::Anti => Action::Cross,
            other => other,
        },
        discriminating,
    ));

    // Flipping the decoy is the over-retention test: nothing about the correct
    // behaviour may move.
    verdicts.push(check_orbit_with(
        &PredictiveState,
        &contracts
            .iter()
            .filter(|contract| contract.decoy.is_some())
            .cloned()
            .collect::<Vec<_>>(),
        "the_second_latch_is_flipped",
        true,
        |contract| Contract {
            decoy: contract.decoy.map(Mode::flipped),
            ..*contract
        },
        identity,
        discriminating,
    ));

    // Exchanging the commands *without* relabelling the modes changes which one
    // is correct. Under the first-action observable this would look like a pass;
    // it is caught only because the observable is the discriminating command.
    verdicts.push(check_orbit_with(
        &PredictiveState,
        &contracts
            .iter()
            .filter(|contract| contract.coupling == ModeCoupling::Discriminating)
            .cloned()
            .collect::<Vec<_>>(),
        "which_command_advances_is_changed_without_relabelling_the_mode",
        false,
        |contract| contract.with_flipped_mode(),
        identity,
        discriminating,
    ));

    verdicts
}

/// The information orbit, read against the coarsened fragment.
pub fn information_verdicts() -> Vec<InformationVerdict> {
    let witness = card_cases()
        .into_iter()
        .find(|case| case.kind == CaseKind::WitnessLatchedMode)
        .expect("the card has this witness")
        .contract;
    let blind = Coarsened::new(&PredictiveState, without_the_latch);
    let set = mode_ambiguity(&witness);

    let mut verdicts = vec![
        // Moving the latch later inside the aliasing interval shortens the span
        // a learner must carry, and changes nothing a learner can attain.
        check_information_orbit(
            &blind,
            &set,
            "the_latch_moves_within_the_aliasing_interval",
            true,
            |set: &AmbiguitySet<Contract>| {
                AmbiguitySet::uniform(
                    set.candidates
                        .iter()
                        .map(|contract| Contract {
                            latch_at: DISCRIMINATING_STEP - 1,
                            ..*contract
                        })
                        .collect(),
                )
            },
            HORIZON,
        ),
        // Publishing a constant instead of the mode decorrelates the latch from
        // what it is supposed to name. Nothing about the world changes and the
        // learner is left unable to act.
        //
        // Checked against the *full* fragment, not the coarsened one. Ablating
        // the latch and then decorrelating it changes nothing, because there is
        // nothing left to decorrelate; running it that way reported a
        // meaning-changing transform as inert.
        check_information_orbit(
            &PredictiveState,
            &set,
            "the_latch_is_decorrelated_from_the_mode",
            false,
            |set: &AmbiguitySet<Contract>| {
                AmbiguitySet::uniform(
                    set.candidates
                        .iter()
                        .map(|contract| Contract {
                            latch_reports: Some(Mode::Forward),
                            ..*contract
                        })
                        .collect(),
                )
            },
            HORIZON,
        ),
        // Ending the aliasing interval early makes the mode probeable, so a
        // learner that kept nothing can still recover it in time. The value
        // orbit cannot see this at all.
        check_information_orbit(
            &blind,
            &set,
            "the_aliasing_interval_ends_early",
            false,
            |set: &AmbiguitySet<Contract>| {
                AmbiguitySet::uniform(
                    set.candidates
                        .iter()
                        .map(|contract| Contract {
                            aliasing_until: 1,
                            ..*contract
                        })
                        .collect(),
                )
            },
            HORIZON,
        ),
    ];

    // Republishing the mode is the fully-observable control, stated here as the
    // transformation it is: it closes the same gap the latch ablation opens.
    verdicts.push(check_information_orbit(
        &blind,
        &set,
        "the_mode_is_republished_every_step",
        false,
        |set: &AmbiguitySet<Contract>| {
            AmbiguitySet::uniform(
                set.candidates
                    .iter()
                    .map(|contract| Contract {
                        visibility: ModeVisibility::Always,
                        ..*contract
                    })
                    .collect(),
            )
        },
        HORIZON,
    ));

    verdicts
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineReport {
    pub name: String,
    pub memory_span: usize,
    pub scores: BTreeMap<String, KindScore>,
    pub passes_retention_contrast: bool,
    pub optimal_on_negatives: Vec<String>,
}

fn baseline_report<P: PublicPolicy>(policy: &P) -> BaselineReport {
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
        name: policy.name().to_string(),
        memory_span: policy.memory_span(),
        scores: score_policy(policy),
        passes_retention_contrast: retention_contrast(policy),
        optimal_on_negatives: optimal_on,
    }
}

/// Which witness each isolation negative is paired against.
pub fn paired_witness(negative: CaseKind) -> CaseKind {
    match negative {
        CaseKind::NegativeFullyObservable | CaseKind::NegativeIrrelevantLatch => {
            CaseKind::WitnessLatchedMode
        }
        other => other,
    }
}

/// The memory-cost control's evidence, which the isolation bracket cannot carry.
///
/// An isolation negative is a case where something *simpler* is optimal and the
/// witness is not. The memory-cost control is the opposite shape: nothing
/// simpler is optimal on it, and what it catches is a policy that retains *too
/// much*. Forcing it into the bracket would report `isolates: false` and read as
/// a defect rather than as a different kind of evidence, so it is reported here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverRetentionReport {
    pub designated_failure: String,
    /// The over-retentive policy is correct on the witness.
    pub failure_passes_the_witness: bool,
    /// And wrong on exactly the memory-cost cases where the two latches
    /// disagree.
    pub failure_rate_on_the_memory_cost_control: f64,
    /// The ceiling policy is correct on both.
    pub ceiling_passes_both: bool,
    /// Flipping the second latch moves no correct action anywhere.
    pub the_second_latch_is_inert: bool,
}

fn over_retention_report() -> OverRetentionReport {
    let cases = card_cases();
    let memory_cost: Vec<&Case> = cases
        .iter()
        .filter(|case| case.kind == CaseKind::NegativeMemoryCost)
        .collect();
    let wrong = memory_cost
        .iter()
        .filter(|case| {
            run_policy(&case.contract, &LastLatch).value != value_bounds(&case.contract).0
        })
        .count();
    let inert = cases.iter().all(|case| {
        let flipped = Contract {
            decoy: case.contract.decoy.map(Mode::flipped),
            ..case.contract
        };
        discriminating_actions(&case.contract) == discriminating_actions(&flipped)
            && value_bounds(&case.contract).0 == value_bounds(&flipped).0
    });
    OverRetentionReport {
        designated_failure: LastLatch.name().to_string(),
        failure_passes_the_witness: retention_contrast(&LastLatch),
        failure_rate_on_the_memory_cost_control: wrong as f64 / memory_cost.len() as f64,
        ceiling_passes_both: retention_contrast(&ModeConditioned)
            && memory_cost.iter().all(|case| {
                run_policy(&case.contract, &ModeConditioned).value == value_bounds(&case.contract).0
            }),
        the_second_latch_is_inert: inert,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelCoverage {
    pub declared: KernelUse,
    pub composed: KernelUse,
    pub matches_declaration: bool,
    pub interrupt_resume: String,
    pub interrupt_displaced: String,
}

fn kernel_coverage() -> KernelCoverage {
    let declared = KernelUse::declared("02").expect("card 02 is in the coverage table");
    let composed = kernel_use();
    let interrupt = card_cases()[0].contract.aliasing_interrupt();
    KernelCoverage {
        matches_declaration: declared == composed,
        declared,
        composed,
        interrupt_resume: format!("{:?}", interrupt.resume),
        interrupt_displaced: format!("{:?}", interrupt.displaced),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditReport {
    pub card: String,
    pub trunk: String,
    pub ring: usize,
    pub horizon: usize,
    pub discriminating_step: usize,
    pub action_sequences_enumerated: usize,
    pub contract_hash: String,
    pub cases: Vec<CaseBracket>,
    pub memory: Vec<MemoryValue>,
    /// The latch is worth something exactly where it is load-bearing: on the
    /// witness and on the memory-cost control, and on neither of the two
    /// controls built to make it unnecessary.
    pub memory_is_worth_something_exactly_where_the_latch_is_load_bearing: bool,
    /// The second latch is worth nothing anywhere.
    pub the_decoy_is_worth_nothing_anywhere: bool,
    pub aliasing: AliasingReport,
    pub orbit: Vec<OrbitVerdict>,
    pub information_orbit: Vec<InformationVerdict>,
    pub baselines: Vec<BaselineReport>,
    pub bracket_structure: BracketStructure,
    pub over_retention: OverRetentionReport,
    pub kernel_coverage: KernelCoverage,
    pub rendering: RenderAudit,
    pub not_claimed: Vec<String>,
}

/// Build the complete audit. No learner is constructed anywhere in this path.
pub fn audit_report() -> AuditReport {
    let cases = card_cases();
    let memory: Vec<MemoryValue> = cases.iter().map(memory_value).collect();

    let baselines = vec![
        baseline_report(&ModeConditioned),
        baseline_report(&Memoryless),
        baseline_report(&Window::too_short()),
        baseline_report(&Window::sufficient()),
        baseline_report(&LastLatch),
        baseline_report(&ConstantCommand),
    ];
    let evidence: Vec<BaselineEvidence> = baselines
        .iter()
        .map(|report| BaselineEvidence {
            name: report.name.clone(),
            scores: report.scores.clone(),
            optimal_on_negatives: report.optimal_on_negatives.clone(),
            is_ceiling: report.name == "mode_conditioned",
        })
        .collect();
    let pairing: Vec<(String, String)> = [
        CaseKind::NegativeFullyObservable,
        CaseKind::NegativeIrrelevantLatch,
    ]
    .iter()
    .map(|kind| {
        (
            kind.label().to_string(),
            paired_witness(*kind).label().to_string(),
        )
    })
    .collect();

    AuditReport {
        card: "02-predictive-state".to_string(),
        trunk: "T1".to_string(),
        ring: RING,
        horizon: HORIZON,
        discriminating_step: DISCRIMINATING_STEP,
        action_sequences_enumerated: Action::ALL.len().pow(HORIZON as u32),
        contract_hash: format!("{:016x}", contract_hash()),
        cases: cases.iter().map(case_bracket).collect(),
        memory_is_worth_something_exactly_where_the_latch_is_load_bearing: memory.iter().all(
            |entry| {
                let is_witness = entry.kind == CaseKind::WitnessLatchedMode.label()
                    || entry.kind == CaseKind::NegativeMemoryCost.label();
                (entry.memory_value > 1e-9) == is_witness
            },
        ),
        the_decoy_is_worth_nothing_anywhere: memory
            .iter()
            .all(|entry| entry.decoy_value.abs() < 1e-9),
        memory,
        aliasing: aliasing_report(),
        orbit: orbit_verdicts(),
        information_orbit: information_verdicts(),
        bracket_structure: analyse_bracket(&evidence, &pairing),
        baselines,
        over_retention: over_retention_report(),
        kernel_coverage: kernel_coverage(),
        rendering: render_audit().expect("the card renders through the shared boundary"),
        not_claimed: vec![
            "No learner was constructed, loaded, trained, or evaluated.".to_string(),
            "This is one card built as a world. It is not a capability result.".to_string(),
            "No GPU, remote, or multi-world run is authorized or implied.".to_string(),
            "The variance variant named in CARDS.md is not implemented. It needs an objective under which outcome spread changes the optimal action, and the deterministic G0 fragment has none. The `P6 -> M2` dispute therefore stays open, undecided by this card.".to_string(),
            "The mode is public. The privileged-minus-public gap is zero and vacuous; the load-bearing number is the ceiling lost by ablating the latch.".to_string(),
            "The memory-cost control is not an isolation negative. Nothing simpler is optimal on it; it catches a policy that retains too much, and its evidence is `over_retention` rather than the bracket.".to_string(),
            "One pair of episodes renders to the same public stream: the Forward witness and the Forward irrelevant-latch control. They differ only in what a *different* command would have done, so no on-policy episode can separate them. That control's evidence is the off-policy baseline scoring below, not the rendered corpus.".to_string(),
            "`M2` is not established. The card supplies the instrument; establishing the node needs a learner contrast this crate cannot perform.".to_string(),
        ],
    }
}

/// The sequence the ceiling policy produces, exposed for the rendering.
pub fn ceiling_sequence(contract: &Contract) -> Vec<Action> {
    taught_sequence(contract, &ModeConditioned)
}
