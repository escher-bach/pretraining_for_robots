//! The exact audit for card 05: value of information, matched controls, and the
//! separation between seeking information and seeking useful information.
//!
//! This is the first family in the portfolio whose privileged and public
//! ceilings genuinely differ, so it is the first place the shared
//! `value_bounds`/`ambiguity_gap` pair says something. Cards 04 and 02 report
//! their gaps as vacuous on purpose; here the number is the price of the probe.
//!
//! Three quantities carry the card, all derived from the shared query algebra
//! rather than added as card APIs:
//!
//! - **`epistemic_value`** per first action. `Probe` moves the public ceiling and
//!   shrinks the surviving set; `Peek` shrinks the surviving set and moves
//!   nothing; `Sham` does neither. That table *is* the `M5 -> M11b` contrast.
//! - **`matched_control_verdict`** on the `Probe`/`Sham` pair: equal cost, equal
//!   immediate movement, and only one of them informative.
//! - **`agent_equivalence`** on the two values of the inconsequential bit. They
//!   are observationally distinguishable — `Peek` separates them — and outcome
//!   identical. Reporting the two halves separately is what makes "informative
//!   but worthless" a measurement instead of a label.

use std::collections::BTreeMap;

use pretraining_g0_contract::{
    agent_equivalence, analyse_bracket, check_orbit, epistemic_value, identification_diameter,
    matched_control_verdict, noninterference_check, ActionValue, BaselineEvidence,
    BracketStructure, EquivalenceCertificate, KernelUse, KindScore, MatchedControlVerdict,
    NonInterference, OrbitVerdict, Restriction, Symmetry,
};
use serde::{Deserialize, Serialize};

use crate::{
    attains_public_ceiling, card_cases, contract_hash, instance_ambiguity, kernel_use,
    mean_public_ceiling, mean_value, optimal_first_actions, privileged_ceiling, probe_contrast,
    probe_rate, public_ceiling, relabel, render_audit, score_policy, Action, ActiveExperimentation,
    AlwaysProbe, Case, CaseKind, Contract, ExactPublic, GateCoupling, GateVisibility, NeverProbe,
    PeekInstead, PrivilegedGateKnown, PublicPolicy, RenderAudit, ShamInstead, BUDGET, HORIZON,
    RING,
};

/// One case's exact bracket, with both ceilings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseBracket {
    pub kind: String,
    pub gate: String,
    pub noise: String,
    pub probe_cost: usize,
    /// What a solver reading the gate attains.
    pub privileged_ceiling: f64,
    /// What the best policy measurable in public history attains.
    pub public_ceiling: f64,
    /// The same policy class with the probe removed from the action set.
    pub public_ceiling_without_a_probe: f64,
    pub ambiguity_gap: f64,
    /// What buying the information is worth: the public ceiling minus what the
    /// same learner could do if the probe did not exist.
    pub value_of_the_probe: f64,
    /// The privileged enumeration's first actions, which read the gate and are
    /// reported so the reason they are not the teacher is visible.
    pub privileged_first_actions: Vec<String>,
    /// Whether probing is the correct public opening.
    pub probe_is_correct: bool,
}

/// The public ceiling of a learner whose action set has no probe in it.
///
/// Deleting the action rather than coarsening the trace, because "the probe did
/// not exist" and "the probe existed and told me nothing" are different worlds
/// and only the first is the counterfactual the card's value-of-information
/// quantity is about. The second is `Sham`, and it is a policy rather than a
/// world.
fn ceiling_without_a_probe(contract: &Contract) -> f64 {
    struct WithoutProbe;
    impl pretraining_g0_contract::Fragment for WithoutProbe {
        type Action = Action;
        type Contract = Contract;
        fn actions(&self) -> Vec<Action> {
            Action::ALL
                .into_iter()
                .filter(|action| *action != Action::Probe)
                .collect()
        }
        fn horizon(&self) -> usize {
            HORIZON
        }
        fn start(&self, contract: &Contract) -> usize {
            ActiveExperimentation.start(contract)
        }
        fn step(&self, contract: &Contract, cell: usize, executed: usize, action: Action) -> usize {
            ActiveExperimentation.step(contract, cell, executed, action)
        }
        fn value(&self, contract: &Contract, path: &[usize], actions: &[Action]) -> i32 {
            ActiveExperimentation.value(contract, path, actions)
        }
    }
    impl pretraining_g0_contract::PubliclyObservable for WithoutProbe {
        fn public_trace(&self, contract: &Contract, actions: &[Action]) -> Vec<i64> {
            ActiveExperimentation.public_trace(contract, actions)
        }
    }
    pretraining_g0_contract::public_policy_value(
        &WithoutProbe,
        &instance_ambiguity(contract),
        HORIZON,
    )
}

fn case_bracket(case: &Case) -> CaseBracket {
    let public = public_ceiling(&case.contract);
    let privileged = privileged_ceiling(&case.contract);
    let without = ceiling_without_a_probe(&case.contract);
    let opening = pretraining_g0_contract::public_optimal_actions_at(
        &ActiveExperimentation,
        &instance_ambiguity(&case.contract),
        &[],
        HORIZON,
    );
    CaseBracket {
        kind: case.kind.label().to_string(),
        gate: case.contract.gate.name().to_string(),
        noise: case.contract.noise.name().to_string(),
        probe_cost: case.contract.probe_cost,
        privileged_ceiling: privileged,
        public_ceiling: public,
        public_ceiling_without_a_probe: without,
        ambiguity_gap: privileged - public,
        value_of_the_probe: public - without,
        privileged_first_actions: optimal_first_actions(&case.contract)
            .into_iter()
            .map(|action| action.name().to_string())
            .collect(),
        probe_is_correct: opening.contains(&Action::Probe),
    }
}

/// What each opening action is worth, and what it removes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InformationReport {
    pub kind: String,
    pub actions: Vec<ActionValue>,
    pub matched_control: MatchedControlVerdict,
    /// The two values of the inconsequential bit, compared as hidden states.
    pub inconsequential_bit: EquivalenceCertificate,
    /// The two values of the gate, for contrast.
    pub gate_bit: EquivalenceCertificate,
    /// `Peek` shrinks the surviving set and moves no ceiling. This is the
    /// `M5 -> M11b` contrast as a single boolean.
    pub an_informative_action_can_be_worthless: bool,
    /// Whether the gate is still hidden here, which is when the matched control
    /// is meaningful at all.
    pub the_gate_is_hidden: bool,
}

fn information_report(case: &Case) -> InformationReport {
    let set = instance_ambiguity(&case.contract);
    let actions = epistemic_value(&ActiveExperimentation, &set, HORIZON, |action: Action| {
        action.name().to_string()
    });
    let value_of = |wanted: Action| {
        actions
            .iter()
            .find(|entry| entry.action == wanted.name())
            .expect("every action is reported")
    };
    let peek = value_of(Action::Peek);
    let sham = value_of(Action::Sham);

    InformationReport {
        kind: case.kind.label().to_string(),
        matched_control: matched_control_verdict(
            &ActiveExperimentation,
            &set,
            Action::Probe,
            Action::Sham,
            |action| case.contract.cost_of(action),
            // Neither moves the configuration at all, which is what "equal
            // immediate value movement" means for a card whose only movement is
            // an irreversible commit.
            |_action| 0,
            |action: Action| action.name().to_string(),
        ),
        inconsequential_bit: agent_equivalence(
            &ActiveExperimentation,
            &case.contract,
            &case.contract.with_flipped_noise(),
            HORIZON,
            |action: Action| action.name().to_string(),
        ),
        gate_bit: agent_equivalence(
            &ActiveExperimentation,
            &case.contract,
            &case.contract.with_flipped_gate(),
            HORIZON,
            |action: Action| action.name().to_string(),
        ),
        an_informative_action_can_be_worthless: peek.ambiguity_reduction > 0
            && (peek.public_value - sham.public_value).abs() < 1e-9,
        the_gate_is_hidden: case.contract.visibility == GateVisibility::Hidden,
        actions,
    }
}

/// Whether the gate leaks anywhere it should not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LeakReport {
    /// No sequence that has not probed separates the two gate values.
    pub the_gate_is_invisible_until_probed: NonInterference,
    /// The diameter over both bits before anything is done.
    pub initial_diameter: usize,
    /// And after each of the three non-committing actions.
    pub diameter_after_probe: usize,
    pub diameter_after_peek: usize,
    pub diameter_after_sham: usize,
}

fn leak_report() -> LeakReport {
    let witness = card_cases()
        .into_iter()
        .find(|case| case.kind == CaseKind::WitnessProbeThenCommit)
        .expect("the card has this witness")
        .contract;
    let set = instance_ambiguity(&witness);
    LeakReport {
        the_gate_is_invisible_until_probed: noninterference_check(
            &ActiveExperimentation,
            "an unprobed gate separates no two instances",
            &witness,
            &witness.with_flipped_gate(),
            HORIZON,
            // Every history in which nothing has been committed and nothing has
            // been probed. A commit reveals the gate through its own outcome,
            // which is the card working rather than leaking.
            |sequence: &[Action]| {
                !sequence
                    .iter()
                    .any(|action| *action == Action::Probe || action.is_commit())
            },
            |action: Action| action.name().to_string(),
        ),
        initial_diameter: identification_diameter(&ActiveExperimentation, &set, &[]),
        diameter_after_probe: identification_diameter(
            &ActiveExperimentation,
            &set,
            &[Action::Probe],
        ),
        diameter_after_peek: identification_diameter(&ActiveExperimentation, &set, &[Action::Peek]),
        diameter_after_sham: identification_diameter(&ActiveExperimentation, &set, &[Action::Sham]),
    }
}

/// The invariance orbit.
pub fn orbit_verdicts() -> Vec<OrbitVerdict> {
    let contracts: Vec<Contract> = card_cases().into_iter().map(|case| case.contract).collect();
    let identity = |action: Action| action;
    let mut verdicts = Vec::new();

    // Exchanging the two gate values *and* the two commits is a relabelling.
    verdicts.push(check_orbit(
        &ActiveExperimentation,
        &contracts,
        "gate_values_and_commits_are_exchanged_together",
        true,
        |contract| {
            relabel(
                contract,
                Symmetry {
                    shift: 0,
                    reflect: true,
                    cells: RING,
                },
            )
        },
        |action| match action {
            Action::CommitLeft => Action::CommitRight,
            Action::CommitRight => Action::CommitLeft,
            other => other,
        },
        // The privileged first action is the right observable for a *relabelling*
        // of the gate: the whole point is that the correct commit follows the
        // label. A public observable would report agreement, because the public
        // opening is `Probe` on both sides.
    ));

    // Flipping the inconsequential bit alone must move nothing.
    verdicts.push(check_orbit(
        &ActiveExperimentation,
        &contracts,
        "the_inconsequential_bit_is_flipped",
        true,
        |contract| contract.with_flipped_noise(),
        identity,
    ));

    // Exchanging which gate value a commit favours, *without* relabelling the
    // gate, changes which commit is correct.
    verdicts.push(check_orbit(
        &ActiveExperimentation,
        &contracts
            .iter()
            .filter(|contract| contract.coupling == GateCoupling::Discriminating)
            .cloned()
            .collect::<Vec<_>>(),
        "which_gate_value_a_commit_favours_is_changed",
        false,
        |contract| contract.with_flipped_gate(),
        identity,
    ));

    // Making both commits succeed removes the decision value entirely.
    verdicts.push(check_orbit(
        &ActiveExperimentation,
        &contracts
            .iter()
            .filter(|contract| contract.coupling == GateCoupling::Discriminating)
            .cloned()
            .collect::<Vec<_>>(),
        "both_commits_are_made_to_succeed",
        false,
        |contract| Contract {
            coupling: GateCoupling::Irrelevant,
            ..*contract
        },
        identity,
    ));

    verdicts
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineReport {
    pub name: String,
    pub scores: BTreeMap<String, KindScore>,
    pub probe_rates: BTreeMap<String, f64>,
    /// Mean value over each kind, which is the scale the public ceiling lives on.
    pub mean_values: BTreeMap<String, f64>,
    pub passes_probe_contrast: bool,
    pub optimal_on_negatives: Vec<String>,
}

fn baseline_report<P: PublicPolicy>(policy: &P) -> BaselineReport {
    BaselineReport {
        name: policy.name().to_string(),
        scores: score_policy(policy),
        probe_rates: CaseKind::ALL
            .into_iter()
            .map(|kind| (kind.label().to_string(), probe_rate(policy, kind)))
            .collect(),
        mean_values: CaseKind::ALL
            .into_iter()
            .map(|kind| (kind.label().to_string(), mean_value(policy, kind)))
            .collect(),
        passes_probe_contrast: probe_contrast(policy),
        // In expectation, not per case. See `score_policy`: with the gate still
        // hidden the public ceiling is an average and no episode attains it.
        optimal_on_negatives: CaseKind::NEGATIVES
            .into_iter()
            .filter(|kind| attains_public_ceiling(policy, *kind))
            .map(|kind| kind.label().to_string())
            .collect(),
    }
}

/// Which witness each negative is paired against.
pub fn paired_witness(_negative: CaseKind) -> CaseKind {
    CaseKind::WitnessProbeThenCommit
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelCoverage {
    pub declared: KernelUse,
    pub composed: KernelUse,
    pub matches_declaration: bool,
    pub restriction_kind: String,
    pub resource_scope: String,
    pub gate_reveal_guard: String,
}

fn kernel_coverage() -> KernelCoverage {
    let declared = KernelUse::declared("05").expect("card 05 is in the coverage table");
    let composed = kernel_use();
    let witness = card_cases()[0].contract;
    let budget = witness.budget();
    KernelCoverage {
        matches_declaration: declared == composed,
        declared,
        composed,
        restriction_kind: budget.kind().to_string(),
        resource_scope: match budget {
            Restriction::Resource { scope, .. } => format!("{scope:?}"),
            _ => "not a resource".to_string(),
        },
        gate_reveal_guard: format!("{:?}", witness.gate_reveal().guard),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditReport {
    pub card: String,
    pub trunk: String,
    pub ring: usize,
    pub horizon: usize,
    pub budget: usize,
    pub action_sequences_enumerated: usize,
    pub contract_hash: String,
    pub cases: Vec<CaseBracket>,
    /// Probing is the correct public opening on the witness and on no control.
    pub probing_is_correct_exactly_on_the_witness: bool,
    /// The ambiguity gap is positive on the witness. This is the first family in
    /// the portfolio for which that is true.
    pub the_ambiguity_gap_is_positive_on_the_witness: bool,
    pub information: Vec<InformationReport>,
    /// The matched non-informative control holds where the card claims it: on
    /// the witness and on the equally-valuable control.
    ///
    /// Not "everywhere", and the exceptions are the point. The verdict has three
    /// clauses — equal cost, equal immediate movement, and only the informative
    /// action reducing ambiguity — and each control is built to break exactly
    /// one of them. Publishing the gate breaks informativeness; making the probe
    /// consume the whole budget breaks cost matching. An audit that demanded the
    /// property everywhere would be demanding that the controls fail to control.
    pub the_matched_control_holds_where_the_probe_is_matched_and_informative: bool,
    /// Which clause each control breaks, so the exceptions are legible rather
    /// than merely tolerated.
    pub clause_each_control_breaks: BTreeMap<String, String>,
    /// The mean public ceiling per kind, which is the scale admission is judged
    /// on.
    pub mean_public_ceilings: BTreeMap<String, f64>,
    /// An action can shrink the surviving set and still be worthless.
    pub an_informative_action_can_be_worthless: bool,
    pub leakage: LeakReport,
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
    let brackets: Vec<CaseBracket> = cases.iter().map(case_bracket).collect();
    let information: Vec<InformationReport> = cases.iter().map(information_report).collect();

    let baselines = vec![
        baseline_report(&ExactPublic),
        baseline_report(&NeverProbe),
        baseline_report(&AlwaysProbe),
        baseline_report(&PeekInstead),
        baseline_report(&ShamInstead),
        baseline_report(&PrivilegedGateKnown),
    ];
    let evidence: Vec<BaselineEvidence> = baselines
        .iter()
        .map(|report| BaselineEvidence {
            name: report.name.clone(),
            scores: report.scores.clone(),
            optimal_on_negatives: report.optimal_on_negatives.clone(),
            // Both ceilings are excluded from the failure-mode analysis. The
            // privileged one is not a policy a learner could run.
            is_ceiling: matches!(
                report.name.as_str(),
                "exact_public" | "privileged_gate_known"
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

    AuditReport {
        card: "05-active-experimentation".to_string(),
        trunk: "T2".to_string(),
        ring: RING,
        horizon: HORIZON,
        budget: BUDGET,
        action_sequences_enumerated: Action::ALL.len().pow(HORIZON as u32),
        contract_hash: format!("{:016x}", contract_hash()),
        probing_is_correct_exactly_on_the_witness: brackets.iter().all(|entry| {
            entry.probe_is_correct == (entry.kind == CaseKind::WitnessProbeThenCommit.label())
        }),
        the_ambiguity_gap_is_positive_on_the_witness: brackets
            .iter()
            .filter(|entry| entry.kind == CaseKind::WitnessProbeThenCommit.label())
            .all(|entry| entry.ambiguity_gap > 1e-9),
        cases: brackets,
        the_matched_control_holds_where_the_probe_is_matched_and_informative: information
            .iter()
            .filter(|entry| {
                entry.kind == CaseKind::WitnessProbeThenCommit.label()
                    || entry.kind == CaseKind::NegativeGateIrrelevant.label()
            })
            .all(|entry| entry.matched_control.holds),
        clause_each_control_breaks: information
            .iter()
            .map(|entry| {
                let verdict = &entry.matched_control;
                let broken = if !verdict.equal_cost {
                    "equal cost"
                } else if !verdict.equal_immediate_value_movement {
                    "equal immediate movement"
                } else if !verdict.only_the_informative_action_reduces_ambiguity {
                    "only the informative action reduces ambiguity"
                } else {
                    "none"
                };
                (entry.kind.clone(), broken.to_string())
            })
            .collect(),
        mean_public_ceilings: CaseKind::ALL
            .into_iter()
            .map(|kind| (kind.label().to_string(), mean_public_ceiling(kind)))
            .collect(),
        an_informative_action_can_be_worthless: information
            .iter()
            .any(|entry| entry.an_informative_action_can_be_worthless),
        information,
        leakage: leak_report(),
        orbit: orbit_verdicts(),
        bracket_structure: analyse_bracket(&evidence, &pairing),
        baselines,
        kernel_coverage: kernel_coverage(),
        rendering: render_audit().expect("the card renders through the shared boundary"),
        not_claimed: vec![
            "No learner was constructed, loaded, trained, or evaluated.".to_string(),
            "This is one card built as a world. It is not a capability result.".to_string(),
            "No GPU, remote, or multi-world run is authorized or implied.".to_string(),
            "The ring's adjacency plays no part here: commits jump to an outcome cell. The rotation orbit is therefore a relabelling orbit, not a geometric one.".to_string(),
            "The high-prediction-error variant named in CARDS.md is not implemented. It needs a prediction objective to make error measurable, and the finite-G0 profile emits no future query, so novelty-driven probing has nothing to be driven by. The `M2 -> M5` dispute is not decided by this card.".to_string(),
            "`Peek` is not in the card text. It was added because `Sham` alone cannot separate seeking information from seeking information that changes a decision, which is the `M5 -> M11b` dispute; the card as written would have left that dispute unmeasured.".to_string(),
            "The family parameters — gate visibility, coupling, and probe cost — are published. Without them the witness and the equally-valuable control would be indistinguishable at the first decision and no policy could behave differently in them.".to_string(),
            "Admission is judged in expectation over a case kind, not per case. With the gate hidden the public ceiling is an average that no single episode attains, and a per-case test would reject the optimal blind commit on every instance where the coin fell the other way.".to_string(),
            "Sixteen episodes render to seven distinct public streams. The inconsequential bit never appears, because the teacher never buys it; its evidence is the off-policy epistemic-value table, not the corpus. The equally-valuable control collapses all four instances into one episode for the same reason.".to_string(),
            "`M5` is not established. The card supplies the instrument; establishing the node needs a learner contrast this crate cannot perform.".to_string(),
        ],
    }
}
