//! The exact audit for card 03: identification, reachability, brackets, and
//! orbits.
//!
//! Two things here are card-specific rather than shared, and both are instrument
//! choices worth stating.
//!
//! First, the invariance group. `check_orbit` is handed the ring's *rotations*,
//! not its dihedral group. Card 04 could use the whole group because its actions
//! are `{-1, 0, +1}`; this card's are `{0, +1, +2, -1}` and a reflection has no
//! image for `+2`. A reflection is therefore run as a meaning-*changing*
//! transform, and it has to move something.
//!
//! Second, the calibration transformation. `check_orbit` compares ceilings and
//! optimal actions, and making calibration uninformative changes neither: the
//! world is the same world, and a solver holding the contract still solves it.
//! What changes is *identifiability*, so that transform is checked with
//! `identify` instead. Running it through the orbit machinery would have
//! produced a verdict that looked like a pass and meant nothing.

use std::collections::BTreeMap;

use pretraining_g0_contract::{
    analyse_bracket, check_orbit, identification_diameter, noninterference_check,
    privileged_value_bound, public_policy_value, BaselineEvidence, BracketStructure, KernelUse,
    KindScore, OrbitVerdict, Symmetry,
};
use serde::{Deserialize, Serialize};

use crate::{
    admissible_bodies, allocation_contrast, body_ambiguity, body_environment_swap, card_cases,
    contract_hash, full_body, goal_is_reachable, kernel_use, optimal_first_actions, render_audit,
    rotate, run_policy, score_policy, value_bounds, Action, Affordance, AlwaysAttempt, Calibration,
    Case, CaseKind, Contract, ExactPostCalibration, IgnoreSupport, PlanAtCalibration, PublicPolicy,
    RenderAudit, TryOnceThenFallback, CONFIGURATION, HORIZON, RING,
};

/// One case's exact bracket.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseBracket {
    pub kind: String,
    pub start: usize,
    pub goal: usize,
    pub supported_actuators: Vec<String>,
    pub goal_is_reachable: bool,
    pub ceiling: i32,
    pub optimal_first_actions: Vec<String>,
    /// Whether the fallback is the whole of the correct first move.
    pub fallback_is_optimal: bool,
}

fn case_bracket(case: &Case) -> CaseBracket {
    let first = optimal_first_actions(&case.contract);
    CaseBracket {
        kind: case.kind.label().to_string(),
        start: case.contract.start,
        goal: case.contract.goal,
        supported_actuators: Action::MOVEMENTS
            .into_iter()
            .filter(|action| case.contract.support.contains(action.index()))
            .map(|action| action.name().to_string())
            .collect(),
        goal_is_reachable: goal_is_reachable(&case.contract),
        ceiling: value_bounds(&case.contract).0,
        fallback_is_optimal: first == vec![Action::Fallback],
        optimal_first_actions: first
            .into_iter()
            .map(|action| action.name().to_string())
            .collect(),
    }
}

/// What calibration does to the identification observable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentificationReport {
    pub kind: String,
    pub goal: usize,
    /// Every body the family admits. Before the scaffold runs, nothing has been
    /// observed, so this is the diameter a learner starts from.
    pub bodies_admitted: usize,
    /// The surviving set after the calibration this contract actually delivers.
    pub diameter_after_calibration: usize,
    /// The same contract with calibration made uninformative.
    pub diameter_after_uninformative_calibration: usize,
    /// The exact optimal policy value under the post-calibration belief.
    pub public_ceiling: f64,
    /// Each surviving body solved by a solver that knows which it is.
    pub privileged_ceiling: f64,
    /// Their difference. Zero is the card's claim, and it holds only because
    /// calibration is mandatory, free, and exact.
    pub ambiguity_gap: f64,
    /// The same gap when calibration shows nothing, which is what makes the
    /// scaffold load-bearing rather than decorative.
    pub ambiguity_gap_without_calibration: f64,
}

fn identification_report(case: &Case) -> IdentificationReport {
    let calibrated = body_ambiguity(&case.contract);
    let blind_contract = case
        .contract
        .clone()
        .with_calibration(Calibration::Uninformative);
    let blind = body_ambiguity(&blind_contract);
    let public = public_policy_value(&Affordance, &calibrated, HORIZON);
    let privileged = privileged_value_bound(&Affordance, &calibrated, HORIZON);
    let blind_public = public_policy_value(&Affordance, &blind, HORIZON);
    let blind_privileged = privileged_value_bound(&Affordance, &blind, HORIZON);
    IdentificationReport {
        kind: case.kind.label().to_string(),
        goal: case.contract.goal,
        bodies_admitted: admissible_bodies().len(),
        diameter_after_calibration: identification_diameter(&Affordance, &calibrated, &[]),
        diameter_after_uninformative_calibration: identification_diameter(&Affordance, &blind, &[]),
        public_ceiling: public,
        privileged_ceiling: privileged,
        ambiguity_gap: privileged - public,
        ambiguity_gap_without_calibration: blind_privileged - blind_public,
    }
}

/// Whether the scaffold could be reading the goal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScaffoldIndependence {
    pub pulse_order: Vec<String>,
    /// The pulse order and the resulting trace are unchanged when the goal is
    /// moved to every other cell, with everything else held fixed.
    pub independent_of_the_goal: bool,
    pub goals_checked: usize,
}

fn scaffold_independence() -> ScaffoldIndependence {
    let mut independent = true;
    let mut checked = 0usize;
    for case in card_cases() {
        let reference_pulses = case.contract.calibration_pulses();
        let reference_trace = case.contract.calibration_trace();
        for goal in 0..RING {
            let moved = Contract {
                goal,
                ..case.contract.clone()
            };
            independent &= moved.calibration_pulses() == reference_pulses;
            independent &= moved.calibration_trace() == reference_trace;
            checked += 1;
        }
    }
    ScaffoldIndependence {
        pulse_order: Action::PULSE_ORDER
            .into_iter()
            .map(|action| action.name().to_string())
            .collect(),
        independent_of_the_goal: independent,
        goals_checked: checked,
    }
}

/// The invariance orbit.
pub fn orbit_verdicts() -> Vec<OrbitVerdict> {
    let contracts: Vec<Contract> = card_cases().into_iter().map(|case| case.contract).collect();
    let identity = |action: Action| action;
    let mut verdicts = Vec::new();

    // Rotations only. See the module note.
    for symmetry in CONFIGURATION.symmetries() {
        if symmetry.reflect || symmetry == Symmetry::identity(RING) {
            continue;
        }
        verdicts.push(check_orbit(
            &Affordance,
            &contracts,
            &symmetry.name(),
            true,
            |contract| rotate(contract, symmetry),
            identity,
        ));
    }

    // The body/environment swap: a different reason for the same reachable set.
    //
    // Restricted to contracts that are body-limited and carry no restoration.
    // A full body has nothing to swap, so including one would pad the orbit
    // with a transform that is the identity — a vacuous pass. A restoration is
    // excluded for a substantive reason: its environment-side twin would have
    // to be a *time-dependent edge list*, and "a reveal means the same thing on
    // either side of the body/environment boundary" is a second claim this card
    // does not make. The enumeration found this by failing rather than by
    // argument.
    let swappable: Vec<Contract> = contracts
        .iter()
        .filter(|contract| contract.support != full_body() && contract.restore.is_none())
        .cloned()
        .collect();
    verdicts.push(check_orbit(
        &Affordance,
        &swappable,
        "body_limit_becomes_an_equivalent_environment_deletion",
        true,
        body_environment_swap,
        identity,
    ));

    // A reflection is *not* in this card's invariance group. It exchanges the
    // two unit directions and leaves `Leap` without an image, so it must move
    // something; a card that reused card 04's dihedral orbit here would be
    // claiming an invariance its body does not have.
    let reflection = Symmetry {
        shift: 0,
        reflect: true,
        cells: RING,
    };
    verdicts.push(check_orbit(
        &Affordance,
        &contracts,
        "reflection_which_has_no_image_for_the_two_edge_command",
        false,
        |contract| rotate(contract, reflection),
        identity,
    ));

    // Changing which actuator is withheld changes the reachable set.
    verdicts.push(check_orbit(
        &Affordance,
        &contracts
            .iter()
            .filter(|contract| contract.support != full_body())
            .cloned()
            .collect::<Vec<_>>(),
        "a_different_actuator_is_withheld",
        false,
        |contract| Contract {
            support: full_body().difference(contract.support),
            ..contract.clone()
        },
        identity,
    ));

    verdicts
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineReport {
    pub name: String,
    pub scores: BTreeMap<String, KindScore>,
    pub passes_allocation_contrast: bool,
    pub optimal_on_negatives: Vec<String>,
}

fn baseline_report<P: PublicPolicy>(policy: &P, label: &str) -> BaselineReport {
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
        scores: score_policy(policy),
        passes_allocation_contrast: allocation_contrast(policy),
        optimal_on_negatives: optimal_on,
    }
}

/// Which witness each negative is paired against.
pub fn paired_witness(negative: CaseKind) -> CaseKind {
    match negative {
        CaseKind::NegativeFrequencyMatched => CaseKind::WitnessUnreachableFallback,
        CaseKind::NegativeNoRestore => CaseKind::WitnessRestore,
        other => other,
    }
}

/// Whether a body/environment swap is publicly visible, per contract.
///
/// The invariance claim is that behaviour depends on the reachable set and not
/// on why it is limited. That claim is only tested where the two arms are
/// publicly *different*; where a swap happens to be invisible, the orbit verdict
/// is comparing an episode with itself. The counts are reported so a reader can
/// see how much of the orbit is doing work.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwapVisibility {
    pub swappable_contracts: usize,
    pub publicly_visible_swaps: usize,
    /// At least one swap must be visible, or the whole transform is vacuous.
    pub the_invariance_is_actually_tested: bool,
}

fn swap_visibility() -> SwapVisibility {
    use pretraining_g0_contract::PubliclyObservable;
    let swappable: Vec<Contract> = card_cases()
        .into_iter()
        .map(|case| case.contract)
        .filter(|contract| contract.support != full_body() && contract.restore.is_none())
        .collect();
    let visible = swappable
        .iter()
        .filter(|contract| {
            Affordance.public_trace(contract, &[])
                != Affordance.public_trace(&body_environment_swap(contract), &[])
        })
        .count();
    SwapVisibility {
        swappable_contracts: swappable.len(),
        publicly_visible_swaps: visible,
        the_invariance_is_actually_tested: visible > 0,
    }
}

/// Whether the reachable set can be told from the ordinary observation stream
/// before an attempt is made.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttemptFreeIdentification {
    /// Two bodies differing in one actuator, with calibration deleted, are
    /// publicly identical until one of them is attempted.
    pub without_calibration_support_is_only_learned_by_attempting: bool,
    pub separating_sequence_without_calibration: Option<Vec<String>>,
    /// With calibration, the same two bodies are already separated at the first
    /// scored decision.
    pub calibration_separates_them_before_any_attempt: bool,
}

fn attempt_free_identification() -> AttemptFreeIdentification {
    let base = card_cases()
        .into_iter()
        .find(|case| case.kind == CaseKind::WitnessUnreachableFallback)
        .expect("the card has this witness")
        .contract;
    let blind = base.clone().with_calibration(Calibration::Uninformative);
    let blind_capable = Contract {
        support: full_body(),
        ..blind.clone()
    };

    // Restricted to sequences that never command the withheld actuator: those
    // are exactly the histories in which the body has not been probed.
    let withheld: Vec<Action> = Action::MOVEMENTS
        .into_iter()
        .filter(|action| !base.support.contains(action.index()))
        .collect();
    let verdict = noninterference_check(
        &Affordance,
        "an unprobed body is invisible without calibration",
        &blind,
        &blind_capable,
        HORIZON,
        |sequence: &[Action]| !sequence.iter().any(|action| withheld.contains(action)),
        |action: Action| action.name().to_string(),
    );

    let calibrated_capable = Contract {
        support: full_body(),
        ..base.clone()
    };
    AttemptFreeIdentification {
        without_calibration_support_is_only_learned_by_attempting: verdict.holds,
        separating_sequence_without_calibration: verdict.separating_sequence,
        calibration_separates_them_before_any_attempt: Affordance.public_trace_of(&base)
            != Affordance.public_trace_of(&calibrated_capable),
    }
}

impl Affordance {
    fn public_trace_of(&self, contract: &Contract) -> Vec<i64> {
        use pretraining_g0_contract::PubliclyObservable;
        self.public_trace(contract, &[])
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelCoverage {
    pub declared: KernelUse,
    pub composed: KernelUse,
    pub matches_declaration: bool,
    pub restriction_kinds_used: Vec<String>,
}

fn kernel_coverage() -> KernelCoverage {
    let declared = KernelUse::declared("03").expect("card 03 is in the coverage table");
    let composed = kernel_use();
    let mut kinds: Vec<String> = card_cases()
        .iter()
        .map(|case| case.contract.body().kind().to_string())
        .collect();
    kinds.sort();
    kinds.dedup();
    KernelCoverage {
        matches_declaration: declared == composed,
        declared,
        composed,
        restriction_kinds_used: kinds,
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
    pub identification: Vec<IdentificationReport>,
    pub calibration_identifies_the_body_everywhere: bool,
    pub scored_phase_ambiguity_gap_is_zero_everywhere: bool,
    pub uninformative_calibration_reopens_the_gap: bool,
    pub scaffold: ScaffoldIndependence,
    pub swap_visibility: SwapVisibility,
    pub attempt_free_identification: AttemptFreeIdentification,
    pub orbit: Vec<OrbitVerdict>,
    pub baselines: Vec<BaselineReport>,
    pub bracket_structure: BracketStructure,
    pub kernel_coverage: KernelCoverage,
    pub rendering: RenderAudit,
    pub not_claimed: Vec<String>,
}

/// Build the complete audit. No learner is constructed anywhere in this path.
pub fn audit_report() -> AuditReport {
    let cases = card_cases();
    let identification: Vec<IdentificationReport> =
        cases.iter().map(identification_report).collect();

    let baselines = vec![
        baseline_report(&ExactPostCalibration, "exact_post_calibration"),
        baseline_report(&IgnoreSupport, "ignore_support"),
        baseline_report(&PlanAtCalibration, "plan_at_calibration"),
        baseline_report(&AlwaysAttempt, "always_attempt"),
        baseline_report(&TryOnceThenFallback, "try_once_then_fallback"),
    ];
    let evidence: Vec<BaselineEvidence> = baselines
        .iter()
        .map(|report| BaselineEvidence {
            name: report.name.clone(),
            scores: report.scores.clone(),
            optimal_on_negatives: report.optimal_on_negatives.clone(),
            is_ceiling: report.name == "exact_post_calibration",
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
        card: "03-affordance".to_string(),
        trunk: "T1".to_string(),
        ring: RING,
        horizon: HORIZON,
        action_sequences_enumerated: Action::ALL.len().pow(HORIZON as u32),
        contract_hash: format!("{:016x}", contract_hash()),
        cases: cases.iter().map(case_bracket).collect(),
        calibration_identifies_the_body_everywhere: identification
            .iter()
            .all(|entry| entry.diameter_after_calibration == 1),
        scored_phase_ambiguity_gap_is_zero_everywhere: identification
            .iter()
            .all(|entry| entry.ambiguity_gap.abs() < 1e-9),
        uninformative_calibration_reopens_the_gap: identification
            .iter()
            .any(|entry| entry.ambiguity_gap_without_calibration > 1e-9),
        identification,
        scaffold: scaffold_independence(),
        swap_visibility: swap_visibility(),
        attempt_free_identification: attempt_free_identification(),
        orbit: orbit_verdicts(),
        bracket_structure: analyse_bracket(&evidence, &pairing),
        baselines,
        kernel_coverage: kernel_coverage(),
        rendering: render_audit().expect("the card renders through the shared boundary"),
        not_claimed: vec![
            "No learner was constructed, loaded, trained, or evaluated.".to_string(),
            "This is one card built as a world. It is not a capability result.".to_string(),
            "No GPU, remote, or multi-world run is authorized or implied.".to_string(),
            "The nine-cell ring and two-step budget are forced by the frequency-matched control, not chosen: a smaller ring leaves no goal that a fully capable body cannot reach on the budget.".to_string(),
            "The absorbing-wasted-budget variant named in CARDS.md is not implemented. It is a variant of the witness rather than part of it, and no admission decision in R6 or R10 turns on it.".to_string(),
            "The invariance group checked here is the ring's rotations, not its dihedral group. A reflection is not a symmetry of a body whose commands are 0, +1, +2, and -1.".to_string(),
            "Calibration is scaffold: mandatory, free, and outside the scored budget. The card's zero scored-phase ambiguity gap is a consequence of that choice and would not survive making calibration optional.".to_string(),
            "Two case labels name identical contracts: the no-restore negative *is* the unreachable-fallback witness, differing only in which family it is scored inside. Ten of twelve episodes are distinct, and a training mixture must count episodes rather than labels.".to_string(),
            "`M3` is not established. The card supplies the instrument; establishing the node needs a learner contrast this crate cannot perform.".to_string(),
        ],
    }
}
