//! Card 05 — Active Experimentation, built as an audited world.
//!
//! The claim is narrow on purpose: the learner pays for information **exactly
//! when it changes a later decision**. Probe frequency alone is not evidence, so
//! the family is built so that three separately-motivated actions cost the same
//! and move the outcome the same amount, and differ only in what they tell the
//! learner and whether that telling matters.
//!
//! | Action | Costs a step | Moves the outcome | Reduces ambiguity | Changes the value bound |
//! |---|---|---|---|---|
//! | `Probe` | yes | no | yes | yes |
//! | `Peek` | yes | no | yes | **no** |
//! | `Sham` | yes | no | **no** | no |
//!
//! `Sham` is the card's declared matched non-informative control. `Peek` is the
//! sharper one and it is not in the card text: it reveals a second hidden bit
//! that no outcome depends on. Together they separate "seeks information" from
//! "seeks information that changes a decision", which is the `M5 -> M11b`
//! dispute `EMBODIED-PROCESS.md` leaves open. This card measures it rather than
//! naming it.
//!
//! # Relation to card 02
//!
//! The two cards share an effect structure and differ in exactly the dimension
//! their trunks differ in. In both, a hidden bit decides which of two mirrored
//! commands reaches the goal. Card 02 **publishes** that bit once and asks the
//! learner to carry it, which is memory. Card 05 **withholds** it and offers an
//! action that buys it, which is epistemic action. Nothing else about the
//! contrast changes, so a later cross-card transfer claim is not confounded with
//! world difficulty.
//!
//! # Where the evidence is
//!
//! This is the first family in the portfolio with a genuinely non-zero ambiguity
//! gap. A solver holding the gate commits at once for `99`; the best policy
//! restricted to public history probes first and gets `98`; a policy with no
//! probe available gets `49.5` in expectation. The gap of `1` is what the probe
//! costs, and `49.5` is what not having it costs.
//!
//! The configuration structure is [`pretraining_g0_contract::Ring`] again, used
//! here as a labelled outcome set: `0` undecided, `1` the goal, `2` the miss.
//! Its adjacency plays no part — commits jump — so the rotation orbit is a
//! relabelling orbit, and that is stated rather than dressed up.

mod audit;
mod render;
pub use audit::*;
pub use render::*;

use std::collections::BTreeMap;

use pretraining_g0_contract::{
    AmbiguitySet, ContractHasher, Fragment, Guard, KernelUse, PubliclyObservable, ResourceScope,
    Restriction, Reveal, Ring, Symmetry,
};
use serde::{Deserialize, Serialize};

pub use pretraining_g0_contract::{BracketStructure, Isolation, KindScore, OrbitVerdict};

/// Three cells: undecided, the goal, and the miss.
pub const RING: usize = 3;
pub const CONFIGURATION: Ring = Ring::new(RING);

pub const START_CELL: usize = 0;
pub const GOAL_CELL: usize = 1;
pub const MISS_CELL: usize = 2;

/// Two decisions, and two units of budget.
///
/// They are separate numbers because the expensive-probe control makes one
/// action consume both units while still being one decision. Collapsing them
/// would delete that control.
pub const HORIZON: usize = 2;
pub const BUDGET: usize = 2;

pub const GOAL_REWARD: i32 = 100;
pub const STEP_COST: i32 = 1;

/// Probing must be affordable in the witness and must still leave a commit.
const _: () = assert!(BUDGET >= 2);

/// A hidden binary value.
///
/// The same type names the gate and the inconsequential bit, because they are
/// the same kind of thing and differ only in whether any outcome depends on
/// them. Giving them different types would make that difference look structural.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Bit {
    Left,
    Right,
}

impl Bit {
    pub const ALL: [Self; 2] = [Self::Left, Self::Right];

    pub const fn index(self) -> usize {
        match self {
            Self::Left => 0,
            Self::Right => 1,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
        }
    }

    pub const fn flipped(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Action {
    /// Reveals the gate. Costs a step and moves nothing.
    Probe,
    /// The declared matched control: same cost, same movement, reveals nothing.
    Sham,
    /// Reveals the inconsequential bit. Same cost, same movement, informative
    /// about something no outcome depends on.
    Peek,
    /// Irreversible. Reaches the goal when the gate is `Left`.
    CommitLeft,
    /// Irreversible. Reaches the goal when the gate is `Right`.
    CommitRight,
}

impl Action {
    pub const ALL: [Self; 5] = [
        Self::Probe,
        Self::Sham,
        Self::Peek,
        Self::CommitLeft,
        Self::CommitRight,
    ];

    /// The three actions that cost a step and move nothing.
    pub const NON_COMMITTING: [Self; 3] = [Self::Probe, Self::Sham, Self::Peek];

    pub const fn index(self) -> usize {
        match self {
            Self::Probe => 0,
            Self::Sham => 1,
            Self::Peek => 2,
            Self::CommitLeft => 3,
            Self::CommitRight => 4,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Probe => "probe",
            Self::Sham => "sham",
            Self::Peek => "peek",
            Self::CommitLeft => "commit_left",
            Self::CommitRight => "commit_right",
        }
    }

    pub const fn is_commit(self) -> bool {
        matches!(self, Self::CommitLeft | Self::CommitRight)
    }

    /// The gate value this commit succeeds under.
    pub const fn favours(self) -> Option<Bit> {
        match self {
            Self::CommitLeft => Some(Bit::Left),
            Self::CommitRight => Some(Bit::Right),
            _ => None,
        }
    }

    fn from_index(index: usize) -> Option<Self> {
        Self::ALL.into_iter().find(|action| action.index() == index)
    }
}

/// Whether the gate is published at episode start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GateVisibility {
    Hidden,
    /// The gate-is-public control: uncertainty is removed, so probing only
    /// wastes a step.
    Public,
}

/// Whether the gate decides which commit reaches the goal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GateCoupling {
    Discriminating,
    /// The equally-valuable control: uncertainty is retained and both commits
    /// reach the goal, so the gate is worth nothing.
    Irrelevant,
}

/// One fully specified episode contract.
///
/// `visibility`, `coupling`, and `probe_cost` are **family** parameters and are
/// published; `gate` and `noise` are the instance and are not. That split is the
/// one `EMBODIED-PROCESS.md` declares, and it is load-bearing here: without the
/// family parameters in public view the witness and the equally-valuable control
/// would be indistinguishable at the first decision and no policy could behave
/// differently in them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contract {
    pub gate: Bit,
    /// The bit `Peek` reveals. No outcome depends on it.
    pub noise: Bit,
    pub visibility: GateVisibility,
    pub coupling: GateCoupling,
    /// Units of budget one probe consumes.
    pub probe_cost: usize,
}

impl Contract {
    pub fn new(gate: Bit, noise: Bit) -> Self {
        Self {
            gate,
            noise,
            visibility: GateVisibility::Hidden,
            coupling: GateCoupling::Discriminating,
            probe_cost: 1,
        }
    }

    pub fn with_visibility(mut self, visibility: GateVisibility) -> Self {
        self.visibility = visibility;
        self
    }

    pub fn with_coupling(mut self, coupling: GateCoupling) -> Self {
        self.coupling = coupling;
        self
    }

    pub fn with_probe_cost(mut self, cost: usize) -> Self {
        self.probe_cost = cost;
        self
    }

    /// The budget as a shared resource restriction.
    ///
    /// Shared rather than local: one pool covers probing and committing, which
    /// is what makes an expensive probe crowd out the commit instead of merely
    /// costing points.
    pub fn budget(&self) -> Restriction {
        Restriction::Resource {
            budget: BUDGET,
            scope: ResourceScope::Shared,
        }
    }

    /// The reveal that publishes the gate, and the guard it fires on.
    pub fn gate_reveal(&self) -> Reveal<usize> {
        Reveal::new(
            match self.visibility {
                GateVisibility::Public => Guard::AtStart,
                GateVisibility::Hidden => Guard::OnAction(Action::Probe.index() as u16),
            },
            self.gate.index(),
        )
    }

    /// The reveal that publishes the inconsequential bit.
    pub fn noise_reveal(&self) -> Reveal<usize> {
        Reveal::new(
            Guard::OnAction(Action::Peek.index() as u16),
            self.noise.index(),
        )
    }

    pub fn cost_of(&self, action: Action) -> usize {
        match action {
            Action::Probe => self.probe_cost,
            _ => 1,
        }
    }

    /// Whether this commit reaches the goal.
    pub fn commit_succeeds(&self, action: Action) -> bool {
        match self.coupling {
            // Both commits reach the goal, so nothing the gate says changes an
            // outcome. The gate still exists and `Probe` still reveals it.
            GateCoupling::Irrelevant => true,
            GateCoupling::Discriminating => action.favours() == Some(self.gate),
        }
    }

    /// The same contract with its gate flipped.
    pub fn with_flipped_gate(&self) -> Self {
        Self {
            gate: self.gate.flipped(),
            ..*self
        }
    }

    /// The same contract with its inconsequential bit flipped.
    pub fn with_flipped_noise(&self) -> Self {
        Self {
            noise: self.noise.flipped(),
            ..*self
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub value: i32,
    pub reached_goal: bool,
    pub probed: bool,
    pub committed: bool,
    pub spent: usize,
    pub final_cell: usize,
}

/// Execute a complete action sequence under the shared budget.
///
/// A commit is absorbing: everything after the first one is unscored. An action
/// that does not fit the remaining budget is not taken at all, which is how an
/// expensive probe crowds out the commit rather than merely costing points.
pub fn run(contract: &Contract, actions: &[Action]) -> Outcome {
    let budget = contract.budget();
    let mut spent = 0usize;
    let mut cell = START_CELL;
    let mut probed = false;
    let mut committed = false;
    let mut reached = false;

    for action in actions.iter().copied() {
        if committed {
            break;
        }
        let cost = contract.cost_of(action);
        if budget.remaining_budget(spent).unwrap_or(0) < cost {
            break;
        }
        spent += cost;
        match action {
            Action::Probe => probed = true,
            Action::Sham | Action::Peek => {}
            Action::CommitLeft | Action::CommitRight => {
                committed = true;
                reached = contract.commit_succeeds(action);
                cell = if reached { GOAL_CELL } else { MISS_CELL };
            }
        }
    }

    Outcome {
        value: if reached {
            GOAL_REWARD - STEP_COST * spent as i32
        } else {
            0
        },
        reached_goal: reached,
        probed,
        committed,
        spent,
        final_cell: cell,
    }
}

pub struct ActiveExperimentation;

impl Fragment for ActiveExperimentation {
    type Action = Action;
    type Contract = Contract;

    fn actions(&self) -> Vec<Action> {
        Action::ALL.to_vec()
    }

    fn horizon(&self) -> usize {
        HORIZON
    }

    fn start(&self, _contract: &Contract) -> usize {
        START_CELL
    }

    fn step(&self, contract: &Contract, cell: usize, executed: usize, action: Action) -> usize {
        // The cell is a function of the whole prefix, not of one transition: a
        // commit is only taken when the budget allows it, and the budget depends
        // on what came before. `run` is the authority; this reproduces its
        // final-cell rule for the shared enumerator.
        if cell != START_CELL {
            return cell;
        }
        if !action.is_commit() {
            return START_CELL;
        }
        let _ = executed;
        if contract.commit_succeeds(action) {
            GOAL_CELL
        } else {
            MISS_CELL
        }
    }

    fn value(&self, contract: &Contract, _trajectory: &[usize], actions: &[Action]) -> i32 {
        run(contract, actions).value
    }
}

/// Tags separating the two publications in the trace from cell indices.
pub const GATE_TAG: i64 = 100;
pub const NOISE_TAG: i64 = 200;

impl PubliclyObservable for ActiveExperimentation {
    /// The family parameters, then one entry per action: what it revealed and
    /// where the configuration ended up.
    ///
    /// The gate appears only once an admissible reveal has fired. That is the
    /// whole information structure of the card, so it is worth being explicit:
    /// nothing in this function reads `contract.gate` except behind
    /// [`Contract::gate_reveal`]'s guard.
    fn public_trace(&self, contract: &Contract, actions: &[Action]) -> Vec<i64> {
        let mut trace = vec![
            contract.visibility as i64,
            contract.coupling as i64,
            contract.probe_cost as i64,
        ];
        if contract.visibility == GateVisibility::Public {
            trace.push(GATE_TAG + contract.gate.index() as i64);
        }

        let budget = contract.budget();
        let mut spent = 0usize;
        let mut cell = START_CELL;
        let mut committed = false;
        for action in actions.iter().copied() {
            if committed {
                break;
            }
            let cost = contract.cost_of(action);
            if budget.remaining_budget(spent).unwrap_or(0) < cost {
                // Refusing an unaffordable action is itself public: the learner
                // sees that nothing happened.
                trace.push(-1);
                break;
            }
            spent += cost;
            match action {
                Action::Probe => trace.push(GATE_TAG + contract.gate.index() as i64),
                Action::Peek => trace.push(NOISE_TAG + contract.noise.index() as i64),
                Action::Sham => trace.push(0),
                Action::CommitLeft | Action::CommitRight => {
                    committed = true;
                    cell = if contract.commit_succeeds(action) {
                        GOAL_CELL
                    } else {
                        MISS_CELL
                    };
                    trace.push(cell as i64);
                }
            }
        }
        trace
    }
}

pub fn all_sequences() -> Vec<Vec<Action>> {
    pretraining_g0_contract::sequences_of_length(&Action::ALL, HORIZON)
}

pub fn value_bounds(contract: &Contract) -> (i32, Vec<Vec<Action>>) {
    pretraining_g0_contract::value_bounds(&ActiveExperimentation, contract)
}

pub fn optimal_first_actions(contract: &Contract) -> Vec<Action> {
    pretraining_g0_contract::optimal_first_actions(&ActiveExperimentation, contract)
}

/// The hidden realizations a learner cannot separate at episode start.
///
/// Both bits, not just the gate. The inconsequential bit has to be in the set or
/// `Peek` would reduce no ambiguity and the `M5 -> M11b` contrast would be
/// unmeasurable. The realized contract is first so `identify` reports the class
/// the learner is actually in.
pub fn instance_ambiguity(contract: &Contract) -> AmbiguitySet<Contract> {
    let mut candidates = vec![*contract];
    for gate in Bit::ALL {
        for noise in Bit::ALL {
            let candidate = Contract {
                gate,
                noise,
                ..*contract
            };
            if candidate != *contract {
                candidates.push(candidate);
            }
        }
    }
    AmbiguitySet::uniform(candidates)
}

/// The gate pair alone, used where the inconsequential bit would only dilute a
/// count.
pub fn gate_ambiguity(contract: &Contract) -> AmbiguitySet<Contract> {
    AmbiguitySet::uniform(vec![*contract, contract.with_flipped_gate()])
}

/// Which sub-claim or control a case belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CaseKind {
    /// The gate is hidden, decides the commit, and the probe is affordable.
    WitnessProbeThenCommit,
    /// The gate is published at episode start.
    NegativeGatePublic,
    /// Both commits reach the goal.
    NegativeGateIrrelevant,
    /// The probe consumes the whole budget.
    NegativeProbeTooExpensive,
}

impl CaseKind {
    pub const ALL: [Self; 4] = [
        Self::WitnessProbeThenCommit,
        Self::NegativeGatePublic,
        Self::NegativeGateIrrelevant,
        Self::NegativeProbeTooExpensive,
    ];

    pub const NEGATIVES: [Self; 3] = [
        Self::NegativeGatePublic,
        Self::NegativeGateIrrelevant,
        Self::NegativeProbeTooExpensive,
    ];

    pub const fn is_witness(self) -> bool {
        matches!(self, Self::WitnessProbeThenCommit)
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::WitnessProbeThenCommit => "witness_probe_then_commit",
            Self::NegativeGatePublic => "negative_gate_public",
            Self::NegativeGateIrrelevant => "negative_gate_irrelevant",
            Self::NegativeProbeTooExpensive => "negative_probe_too_expensive",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Case {
    pub kind: CaseKind,
    pub contract: Contract,
}

/// The whole card as a finite set of cases.
///
/// Every kind carries all four combinations of the two hidden bits. Varying the
/// inconsequential bit inside every kind is what makes "the learner did not act
/// on it" a measurable claim rather than an untested assumption.
pub fn card_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for gate in Bit::ALL {
        for noise in Bit::ALL {
            let base = Contract::new(gate, noise);
            cases.push(Case {
                kind: CaseKind::WitnessProbeThenCommit,
                contract: base,
            });
            cases.push(Case {
                kind: CaseKind::NegativeGatePublic,
                contract: base.with_visibility(GateVisibility::Public),
            });
            cases.push(Case {
                kind: CaseKind::NegativeGateIrrelevant,
                contract: base.with_coupling(GateCoupling::Irrelevant),
            });
            cases.push(Case {
                kind: CaseKind::NegativeProbeTooExpensive,
                contract: base.with_probe_cost(BUDGET),
            });
        }
    }
    cases
}

/// Public information available to a policy at one decision.
pub struct PublicView<'a> {
    pub contract: &'a Contract,
    pub executed: usize,
    /// The gate, if a reveal has published it.
    pub revealed_gate: Option<Bit>,
    /// The inconsequential bit, if a peek has published it.
    pub revealed_noise: Option<Bit>,
}

pub trait PublicPolicy {
    fn name(&self) -> &'static str;
    fn act(&self, view: &PublicView<'_>) -> Action;
}

/// Roll a policy forward, publishing only what the reveals allow.
pub fn run_policy<P: PublicPolicy>(contract: &Contract, policy: &P) -> Outcome {
    let mut actions = Vec::with_capacity(HORIZON);
    let mut revealed_gate = match contract.visibility {
        GateVisibility::Public => Some(contract.gate),
        GateVisibility::Hidden => None,
    };
    let mut revealed_noise = None;
    let budget = contract.budget();
    let mut spent = 0usize;
    for executed in 0..HORIZON {
        let action = policy.act(&PublicView {
            contract,
            executed,
            revealed_gate,
            revealed_noise,
        });
        actions.push(action);
        let cost = contract.cost_of(action);
        if budget.remaining_budget(spent).unwrap_or(0) < cost {
            break;
        }
        spent += cost;
        match action {
            Action::Probe => revealed_gate = Some(contract.gate),
            Action::Peek => revealed_noise = Some(contract.noise),
            _ => {}
        }
        if action.is_commit() {
            break;
        }
    }
    run(contract, &actions)
}

/// The commit a revealed gate makes correct.
fn commit_for(gate: Bit) -> Action {
    match gate {
        Bit::Left => Action::CommitLeft,
        Bit::Right => Action::CommitRight,
    }
}

/// The public ceiling as a policy: probes exactly where probing pays.
///
/// It never reads `contract.gate` directly; it reads what a reveal published.
pub struct ExactPublic;

impl PublicPolicy for ExactPublic {
    fn name(&self) -> &'static str {
        "exact_public"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        if let Some(gate) = view.revealed_gate {
            return commit_for(gate);
        }
        let affordable = view.contract.cost_of(Action::Probe) + 1 <= BUDGET;
        let useful = view.contract.coupling == GateCoupling::Discriminating;
        if affordable && useful && view.executed == 0 {
            Action::Probe
        } else {
            // Nothing to learn or nothing left to learn with: commit. Which side
            // is arbitrary and the audit reports that it is.
            Action::CommitLeft
        }
    }
}

/// The privileged ceiling: commits at once to the side the gate favours.
///
/// It reads the gate whether or not anything published it, which is why it is a
/// reference and never a teacher.
pub struct PrivilegedGateKnown;

impl PublicPolicy for PrivilegedGateKnown {
    fn name(&self) -> &'static str {
        "privileged_gate_known"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        commit_for(view.contract.gate)
    }
}

/// Never buys information.
pub struct NeverProbe;

impl PublicPolicy for NeverProbe {
    fn name(&self) -> &'static str {
        "never_probe"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        match view.revealed_gate {
            Some(gate) => commit_for(gate),
            None => Action::CommitLeft,
        }
    }
}

/// Always buys information, whatever it is worth.
pub struct AlwaysProbe;

impl PublicPolicy for AlwaysProbe {
    fn name(&self) -> &'static str {
        "always_probe"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        if view.executed == 0 {
            Action::Probe
        } else {
            match view.revealed_gate {
                Some(gate) => commit_for(gate),
                None => Action::CommitLeft,
            }
        }
    }
}

/// Buys information that no outcome depends on.
///
/// It is the difference between seeking information and seeking information that
/// changes a decision, run as a policy.
pub struct PeekInstead;

impl PublicPolicy for PeekInstead {
    fn name(&self) -> &'static str {
        "peek_instead"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        if view.executed == 0 {
            Action::Peek
        } else {
            match view.revealed_gate {
                Some(gate) => commit_for(gate),
                None => Action::CommitLeft,
            }
        }
    }
}

/// Spends a step on the matched non-informative action.
///
/// It pays exactly what probing costs and learns nothing, so any advantage the
/// prober has over it is information rather than delay.
pub struct ShamInstead;

impl PublicPolicy for ShamInstead {
    fn name(&self) -> &'static str {
        "sham_instead"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        if view.executed == 0 {
            Action::Sham
        } else {
            match view.revealed_gate {
                Some(gate) => commit_for(gate),
                None => Action::CommitLeft,
            }
        }
    }
}

/// Score one policy across every case kind, keeping the kinds separate.
///
/// `optimal_rate` counts cases at the **privileged** ceiling — the best any
/// solver could do on that instance — and is descriptive only. It is not the
/// admission test, for a reason this card is the first to run into: with a gate
/// still hidden, the public ceiling is an *expectation* over instances and no
/// single episode attains it. A blind commit is right on half the instances and
/// wrong on the other half, and calling it sub-optimal on each is a category
/// error. [`attains_public_ceiling`] is the test that respects that.
pub fn score_policy<P: PublicPolicy>(policy: &P) -> BTreeMap<String, KindScore> {
    let cases = card_cases();
    let mut scores = BTreeMap::new();
    for kind in CaseKind::ALL {
        let selected: Vec<&Case> = cases.iter().filter(|case| case.kind == kind).collect();
        let mut solved = 0usize;
        let mut optimal = 0usize;
        for case in &selected {
            let outcome = run_policy(&case.contract, policy);
            if outcome.reached_goal {
                solved += 1;
            }
            if f64::from(outcome.value) >= privileged_ceiling(&case.contract) - 1e-9 {
                optimal += 1;
            }
        }
        let total = selected.len();
        scores.insert(
            kind.label().to_string(),
            KindScore {
                solved,
                total,
                rate: solved as f64 / total as f64,
                optimal_rate: optimal as f64 / total as f64,
            },
        );
    }
    scores
}

/// A policy's mean value over one case kind.
pub fn mean_value<P: PublicPolicy>(policy: &P, kind: CaseKind) -> f64 {
    let cases: Vec<Case> = card_cases()
        .into_iter()
        .filter(|case| case.kind == kind)
        .collect();
    let total: i32 = cases
        .iter()
        .map(|case| run_policy(&case.contract, policy).value)
        .sum();
    f64::from(total) / cases.len() as f64
}

/// The public ceiling averaged over one case kind.
pub fn mean_public_ceiling(kind: CaseKind) -> f64 {
    let cases: Vec<Case> = card_cases()
        .into_iter()
        .filter(|case| case.kind == kind)
        .collect();
    cases
        .iter()
        .map(|case| public_ceiling(&case.contract))
        .sum::<f64>()
        / cases.len() as f64
}

/// Whether a policy attains the public ceiling *in expectation* over a kind.
///
/// The right admission test for a family with residual uncertainty. The
/// per-case test would reject the optimal blind commit on the half of instances
/// where the coin fell the other way.
pub fn attains_public_ceiling<P: PublicPolicy>(policy: &P, kind: CaseKind) -> bool {
    mean_value(policy, kind) >= mean_public_ceiling(kind) - 1e-9
}

/// The exact value of the best policy measurable in public history.
pub fn public_ceiling(contract: &Contract) -> f64 {
    pretraining_g0_contract::public_policy_value(
        &ActiveExperimentation,
        &instance_ambiguity(contract),
        HORIZON,
    )
}

/// The exact value of a solver that reads the gate.
pub fn privileged_ceiling(contract: &Contract) -> f64 {
    pretraining_g0_contract::privileged_value_bound(
        &ActiveExperimentation,
        &instance_ambiguity(contract),
        HORIZON,
    )
}

/// The rate at which a policy buys the consequential information.
pub fn probe_rate<P: PublicPolicy>(policy: &P, kind: CaseKind) -> f64 {
    let cases: Vec<Case> = card_cases()
        .into_iter()
        .filter(|case| case.kind == kind)
        .collect();
    let probes = cases
        .iter()
        .filter(|case| run_policy(&case.contract, policy).probed)
        .count();
    probes as f64 / cases.len() as f64
}

/// The card's central contrast: probe on the witness, not on any control.
pub fn probe_contrast<P: PublicPolicy>(policy: &P) -> bool {
    probe_rate(policy, CaseKind::WitnessProbeThenCommit) > 0.99
        && CaseKind::NEGATIVES
            .into_iter()
            .all(|kind| probe_rate(policy, kind) < 0.01)
}

/// Which kernel constructs this card composes.
pub fn kernel_use() -> KernelUse {
    KernelUse {
        directed_wiring: true,
        shared_coupling: false,
        interrupt: false,
        restrict: true,
        reveal: true,
        norm_algebra: false,
    }
}

/// Move a contract through one ring symmetry.
///
/// The ring's adjacency plays no part in this card, so this is a relabelling of
/// the outcome cells and the commits that reach them. It is still a real
/// transform: it exchanges which commit succeeds under which gate.
pub fn relabel(contract: &Contract, symmetry: Symmetry) -> Contract {
    if symmetry.reflect {
        Contract {
            gate: contract.gate.flipped(),
            ..*contract
        }
    } else {
        *contract
    }
}

/// A stable hash of the contract set.
pub fn contract_hash() -> u64 {
    let mut hasher = ContractHasher::new();
    hasher
        .absorb(RING as u64)
        .absorb(HORIZON as u64)
        .absorb(BUDGET as u64)
        .absorb(GOAL_REWARD as u64)
        .absorb(STEP_COST as u64);
    for case in card_cases() {
        hasher
            .absorb(case.kind as u64)
            .absorb(case.contract.gate as u64)
            .absorb(case.contract.noise as u64)
            .absorb(case.contract.visibility as u64)
            .absorb(case.contract.coupling as u64)
            .absorb(case.contract.probe_cost as u64);
    }
    hasher.finish()
}

pub fn action_from_index(index: usize) -> Option<Action> {
    Action::from_index(index)
}
