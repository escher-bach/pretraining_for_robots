//! Card 05 stage A — reveal use, the basic relation under card 05's
//! decomposition certificate.
//!
//! The composite asks whether the learner *buys* information exactly when it
//! changes a later decision. This family asks the strictly weaker half: when a
//! free reveal has already published the gate, does the learner commit to the
//! side that gate favours? The purchase decision is gone — there is no probe,
//! no probe cost, and no value-of-information comparison — and the audit
//! reports that absence as a measurement rather than asserting it: for every
//! action in this family, [`pretraining_g0_contract::epistemic_value`] reports
//! an ambiguity reduction of zero.
//!
//! | Action | Costs a step | Moves the outcome | Reduces ambiguity |
//! |---|---|---|---|
//! | `Sham` | yes | no | no |
//! | `CommitLeft` | yes | yes | no |
//! | `CommitRight` | yes | yes | no |
//!
//! `Sham` is retained from the composite even though nothing is now matched
//! against it. Deleting it would delete the structural fact that a step may be
//! spent without progress, which is the slot a probe occupies in the composite;
//! keeping it means stage B restores an informative action into an existing
//! seat rather than changing the action space.
//!
//! # What the two controls break
//!
//! - **Gate hidden.** The reveal never fires. Nothing published separates the
//!   two gate values, so the public ceiling is the blind average and no
//!   published content can be used.
//! - **Commits equally valuable.** The reveal fires and publishes the gate, and
//!   both commits reach the goal. The content is present, informative about the
//!   gate, and irrelevant to the decision.
//!
//! Between them the two case-kind controls say that the witness needs a reveal
//! that fires in a family where the gate decides the commit. The third clause —
//! that the content published is the *gate* rather than some other bit — is not
//! a case kind, because a decoy that costs nothing to follow is worth exactly
//! as much as a blind commit and a control that changes no value is testing
//! nothing. It is checked where it does bite, on the public ceiling, as the
//! uninformative-reveal information orbit. This is card 03's lesson reused: the
//! value orbit is blind to information transformations.

mod audit;
mod render;
pub use audit::*;
pub use render::*;

use std::collections::BTreeMap;

use pretraining_g0_contract::{
    AmbiguitySet, ContractHasher, Fragment, Guard, GuardContext, KernelUse, PubliclyObservable,
    ResourceScope, Restriction, Reveal, Ring, Symmetry,
};
use serde::{Deserialize, Serialize};

pub use pretraining_g0_contract::{BracketStructure, Isolation, KindScore, OrbitVerdict};

/// Three cells: undecided, the goal, and the miss. Inherited unchanged from the
/// composite so a later cross-version claim is about one configuration space.
pub const RING: usize = 3;
pub const CONFIGURATION: Ring = Ring::new(RING);

pub const START_CELL: usize = 0;
pub const GOAL_CELL: usize = 1;
pub const MISS_CELL: usize = 2;

/// Two decisions and two units of budget, as in the composite.
///
/// The horizon stays at two even though one commit ends the episode: the seat
/// the probe used to occupy is still there, occupied by `Sham`. A horizon of
/// one would make stage B a change of shape rather than a restored decision.
pub const HORIZON: usize = 2;
pub const BUDGET: usize = 2;

pub const GOAL_REWARD: i32 = 100;
pub const STEP_COST: i32 = 1;

/// A hidden binary value: the gate, or the decoy the uninformative reveal
/// publishes. One type, because they differ only in whether any outcome depends
/// on them.
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
    /// Costs a step, moves nothing, publishes nothing.
    Sham,
    /// Irreversible. Reaches the goal when the gate is `Left`.
    CommitLeft,
    /// Irreversible. Reaches the goal when the gate is `Right`.
    CommitRight,
}

impl Action {
    pub const ALL: [Self; 3] = [Self::Sham, Self::CommitLeft, Self::CommitRight];

    pub const fn index(self) -> usize {
        match self {
            Self::Sham => 0,
            Self::CommitLeft => 1,
            Self::CommitRight => 2,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Sham => "sham",
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
            Self::Sham => None,
        }
    }

    /// The commit that exchanges the two gate values, for the relabelling orbit.
    pub const fn mirrored(self) -> Self {
        match self {
            Self::CommitLeft => Self::CommitRight,
            Self::CommitRight => Self::CommitLeft,
            Self::Sham => Self::Sham,
        }
    }

    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.into_iter().find(|action| action.index() == index)
    }
}

/// What the episode-start reveal does. A published family parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RevealMode {
    /// The witness: the reveal fires and its content is the gate.
    PublishesGate,
    /// The gate-hidden control: the reveal never fires.
    Withholds,
    /// The uninformative-reveal control: the reveal fires and its content is a
    /// bit no outcome depends on.
    PublishesDecoy,
}

impl RevealMode {
    pub const ALL: [Self; 3] = [Self::PublishesGate, Self::Withholds, Self::PublishesDecoy];

    pub const fn name(self) -> &'static str {
        match self {
            Self::PublishesGate => "publishes_gate",
            Self::Withholds => "withholds",
            Self::PublishesDecoy => "publishes_decoy",
        }
    }
}

/// Whether the gate decides which commit reaches the goal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GateCoupling {
    Discriminating,
    /// The equally-valuable control: both commits reach the goal, so the gate is
    /// published, informative, and worth nothing.
    Irrelevant,
}

/// One fully specified episode contract.
///
/// `reveal_mode` and `coupling` are family parameters and are published; `gate`
/// and `decoy` are the instance and are not. Without the family parameters in
/// public view the witness and the equally-valuable control would be the same
/// episode up to the first decision, which is the reason the composite
/// publishes its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contract {
    pub gate: Bit,
    /// The bit the uninformative reveal publishes. No outcome depends on it.
    pub decoy: Bit,
    pub reveal_mode: RevealMode,
    pub coupling: GateCoupling,
}

/// Trace tags, so a publication is never mistaken for a cell index.
pub const GATE_TAG: i64 = 100;
pub const DECOY_TAG: i64 = 200;

impl Contract {
    pub fn new(gate: Bit, decoy: Bit) -> Self {
        Self {
            gate,
            decoy,
            reveal_mode: RevealMode::PublishesGate,
            coupling: GateCoupling::Discriminating,
        }
    }

    pub fn with_reveal(mut self, reveal_mode: RevealMode) -> Self {
        self.reveal_mode = reveal_mode;
        self
    }

    pub fn with_coupling(mut self, coupling: GateCoupling) -> Self {
        self.coupling = coupling;
        self
    }

    /// The budget as a shared resource restriction, unchanged from the composite.
    pub fn budget(&self) -> Restriction {
        Restriction::Resource {
            budget: BUDGET,
            scope: ResourceScope::Shared,
        }
    }

    /// The reveal that publishes the gate.
    ///
    /// `Guard::Never` is what the gate-hidden control is built from: the
    /// construct stays in the composition and stops firing, so the two arms are
    /// structurally identical and differ in one guard.
    pub fn gate_reveal(&self) -> Reveal<usize> {
        Reveal::new(
            match self.reveal_mode {
                RevealMode::PublishesGate => Guard::AtStart,
                RevealMode::Withholds | RevealMode::PublishesDecoy => Guard::Never,
            },
            self.gate.index(),
        )
    }

    /// The reveal that publishes the decoy.
    pub fn decoy_reveal(&self) -> Reveal<usize> {
        Reveal::new(
            match self.reveal_mode {
                RevealMode::PublishesDecoy => Guard::AtStart,
                RevealMode::PublishesGate | RevealMode::Withholds => Guard::Never,
            },
            self.decoy.index(),
        )
    }

    /// What the episode-start reveals put into public view, if anything.
    ///
    /// The only path from `gate` into the public trace. Everything downstream
    /// reads this rather than the field, which is what the non-interference
    /// check verifies rather than assumes.
    pub fn published_at_start(&self) -> Option<i64> {
        let context = GuardContext {
            executed: 0,
            last_action: None,
            cell: START_CELL,
        };
        if let Some(value) = self.gate_reveal().published(context) {
            return Some(GATE_TAG + value as i64);
        }
        self.decoy_reveal()
            .published(context)
            .map(|value| DECOY_TAG + value as i64)
    }

    /// Every action costs one unit. There is no cost asymmetry left to exploit.
    pub fn cost_of(&self, _action: Action) -> usize {
        1
    }

    /// Whether this commit reaches the goal.
    pub fn commit_succeeds(&self, action: Action) -> bool {
        match self.coupling {
            GateCoupling::Irrelevant => action.is_commit(),
            GateCoupling::Discriminating => action.favours() == Some(self.gate),
        }
    }

    pub fn with_flipped_gate(&self) -> Self {
        Self {
            gate: self.gate.flipped(),
            ..*self
        }
    }

    pub fn with_flipped_decoy(&self) -> Self {
        Self {
            decoy: self.decoy.flipped(),
            ..*self
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub value: i32,
    pub reached_goal: bool,
    pub committed: bool,
    pub spent: usize,
    pub final_cell: usize,
}

/// Execute a complete action sequence under the shared budget.
///
/// A commit is absorbing: everything after the first one is unscored.
pub fn run(contract: &Contract, actions: &[Action]) -> Outcome {
    let budget = contract.budget();
    let mut spent = 0usize;
    let mut cell = START_CELL;
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
        if action.is_commit() {
            committed = true;
            reached = contract.commit_succeeds(action);
            cell = if reached { GOAL_CELL } else { MISS_CELL };
        }
    }

    Outcome {
        value: if reached {
            GOAL_REWARD - STEP_COST * spent as i32
        } else {
            0
        },
        reached_goal: reached,
        committed,
        spent,
        final_cell: cell,
    }
}

pub struct RevealUse;

impl Fragment for RevealUse {
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

    fn step(&self, contract: &Contract, cell: usize, _executed: usize, action: Action) -> usize {
        if cell != START_CELL || !action.is_commit() {
            return cell;
        }
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

impl PubliclyObservable for RevealUse {
    /// The family parameters, whatever the start reveals published, then one
    /// entry per executed action.
    ///
    /// Nothing here reads `contract.gate` except through
    /// [`Contract::published_at_start`] and through the outcome of a commit the
    /// learner has already made.
    fn public_trace(&self, contract: &Contract, actions: &[Action]) -> Vec<i64> {
        let mut trace = vec![contract.reveal_mode as i64, contract.coupling as i64];
        if let Some(published) = contract.published_at_start() {
            trace.push(published);
        }

        let budget = contract.budget();
        let mut spent = 0usize;
        let mut committed = false;
        for action in actions.iter().copied() {
            if committed {
                break;
            }
            let cost = contract.cost_of(action);
            if budget.remaining_budget(spent).unwrap_or(0) < cost {
                trace.push(-1);
                break;
            }
            spent += cost;
            if action.is_commit() {
                committed = true;
                trace.push(if contract.commit_succeeds(action) {
                    GOAL_CELL as i64
                } else {
                    MISS_CELL as i64
                });
            } else {
                trace.push(0);
            }
        }
        trace
    }
}

pub fn all_sequences() -> Vec<Vec<Action>> {
    pretraining_g0_contract::sequences_of_length(&Action::ALL, HORIZON)
}

pub fn value_bounds(contract: &Contract) -> (i32, Vec<Vec<Action>>) {
    pretraining_g0_contract::value_bounds(&RevealUse, contract)
}

pub fn optimal_first_actions(contract: &Contract) -> Vec<Action> {
    pretraining_g0_contract::optimal_first_actions(&RevealUse, contract)
}

/// The hidden realizations a learner cannot separate before the reveal fires.
///
/// Both bits, so that "the decoy was published and nothing used it" is a
/// measurable claim rather than an assumption.
pub fn instance_ambiguity(contract: &Contract) -> AmbiguitySet<Contract> {
    let mut candidates = vec![*contract];
    for gate in Bit::ALL {
        for decoy in Bit::ALL {
            let candidate = Contract {
                gate,
                decoy,
                ..*contract
            };
            if candidate != *contract {
                candidates.push(candidate);
            }
        }
    }
    AmbiguitySet::uniform(candidates)
}

/// Which sub-claim or control a case belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CaseKind {
    /// The reveal fires, publishes the gate, and the gate decides the commit.
    WitnessRevealThenCommit,
    /// The reveal never fires.
    NegativeGateHidden,
    /// The reveal fires and publishes the gate; both commits reach the goal.
    NegativeCommitsEquallyValuable,
}

impl CaseKind {
    pub const ALL: [Self; 3] = [
        Self::WitnessRevealThenCommit,
        Self::NegativeGateHidden,
        Self::NegativeCommitsEquallyValuable,
    ];

    pub const NEGATIVES: [Self; 2] = [
        Self::NegativeGateHidden,
        Self::NegativeCommitsEquallyValuable,
    ];

    pub const fn is_witness(self) -> bool {
        matches!(self, Self::WitnessRevealThenCommit)
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::WitnessRevealThenCommit => "witness_reveal_then_commit",
            Self::NegativeGateHidden => "negative_gate_hidden",
            Self::NegativeCommitsEquallyValuable => "negative_commits_equally_valuable",
        }
    }

    pub const fn contract_of(self, gate: Bit, decoy: Bit) -> Contract {
        let base = Contract {
            gate,
            decoy,
            reveal_mode: RevealMode::PublishesGate,
            coupling: GateCoupling::Discriminating,
        };
        match self {
            Self::WitnessRevealThenCommit => base,
            Self::NegativeGateHidden => Contract {
                reveal_mode: RevealMode::Withholds,
                ..base
            },
            Self::NegativeCommitsEquallyValuable => Contract {
                coupling: GateCoupling::Irrelevant,
                ..base
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct Case {
    pub kind: CaseKind,
    pub contract: Contract,
}

/// The whole family as a finite set of cases: three kinds over both bits.
///
/// The decoy varies inside every kind and is published in none of them, so the
/// twelve case labels render as six distinct public episodes. That is the same
/// accounting fact the composite reports when its inconsequential bit is never
/// bought, and it is why a training mixture counts episodes rather than labels.
pub fn card_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for gate in Bit::ALL {
        for decoy in Bit::ALL {
            for kind in CaseKind::ALL {
                cases.push(Case {
                    kind,
                    contract: kind.contract_of(gate, decoy),
                });
            }
        }
    }
    cases
}

/// Public information available to a policy at one decision.
pub struct PublicView<'a> {
    pub contract: &'a Contract,
    pub executed: usize,
    /// The gate, if the reveal published it. Never the field itself.
    pub revealed_gate: Option<Bit>,
    /// The decoy, if the reveal published it.
    pub revealed_decoy: Option<Bit>,
}

pub trait PublicPolicy {
    fn name(&self) -> &'static str;
    fn act(&self, view: &PublicView<'_>) -> Action;
}

/// Roll a policy forward, publishing only what the reveals allow.
pub fn run_policy<P: PublicPolicy>(contract: &Contract, policy: &P) -> Outcome {
    let revealed_gate = match contract.reveal_mode {
        RevealMode::PublishesGate => Some(contract.gate),
        RevealMode::Withholds | RevealMode::PublishesDecoy => None,
    };
    let revealed_decoy = match contract.reveal_mode {
        RevealMode::PublishesDecoy => Some(contract.decoy),
        RevealMode::PublishesGate | RevealMode::Withholds => None,
    };
    let mut actions = Vec::with_capacity(HORIZON);
    for executed in 0..HORIZON {
        let action = policy.act(&PublicView {
            contract,
            executed,
            revealed_gate,
            revealed_decoy,
        });
        actions.push(action);
        if action.is_commit() {
            break;
        }
    }
    run(contract, &actions)
}

/// The commit a gate value makes correct.
pub fn commit_for(gate: Bit) -> Action {
    match gate {
        Bit::Left => Action::CommitLeft,
        Bit::Right => Action::CommitRight,
    }
}

/// The public ceiling as a policy: commit on the published gate, blind
/// otherwise.
pub struct ExactPublic;

impl PublicPolicy for ExactPublic {
    fn name(&self) -> &'static str {
        "exact_public"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        match view.revealed_gate {
            Some(gate) => commit_for(gate),
            // Nothing published bears on the decision. Which side is arbitrary,
            // and the audit reports that it is.
            None => Action::CommitLeft,
        }
    }
}

/// The privileged reference: commits on the gate whether or not anything
/// published it. Never a teacher.
pub struct PrivilegedGateKnown;

impl PublicPolicy for PrivilegedGateKnown {
    fn name(&self) -> &'static str {
        "privileged_gate_known"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        commit_for(view.contract.gate)
    }
}

/// Ignores every publication and commits one way.
pub struct BlindCommit;

impl PublicPolicy for BlindCommit {
    fn name(&self) -> &'static str {
        "blind_commit"
    }

    fn act(&self, _view: &PublicView<'_>) -> Action {
        Action::CommitLeft
    }
}

/// Spends the free step and then commits blind.
///
/// It pays exactly what a stage-B probe would cost and learns nothing, so any
/// advantage a reveal-reader has over it is content rather than delay.
pub struct ShamThenBlindCommit;

impl PublicPolicy for ShamThenBlindCommit {
    fn name(&self) -> &'static str {
        "sham_then_blind_commit"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        if view.executed == 0 {
            Action::Sham
        } else {
            Action::CommitLeft
        }
    }
}

/// Score one policy across every case kind, keeping the kinds separate.
///
/// `optimal_rate` counts cases at the privileged ceiling and is descriptive
/// only: with the gate withheld the public ceiling is an expectation no single
/// episode attains, so the admission test is [`attains_public_ceiling`].
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

pub fn cases_of(kind: CaseKind) -> Vec<Case> {
    card_cases()
        .into_iter()
        .filter(|case| case.kind == kind)
        .collect()
}

/// A policy's mean value over one case kind.
pub fn mean_value<P: PublicPolicy>(policy: &P, kind: CaseKind) -> f64 {
    let cases = cases_of(kind);
    let total: i32 = cases
        .iter()
        .map(|case| run_policy(&case.contract, policy).value)
        .sum();
    f64::from(total) / cases.len() as f64
}

/// The public ceiling averaged over one case kind.
pub fn mean_public_ceiling(kind: CaseKind) -> f64 {
    let cases = cases_of(kind);
    cases
        .iter()
        .map(|case| public_ceiling(&case.contract))
        .sum::<f64>()
        / cases.len() as f64
}

/// Whether a policy attains the public ceiling in expectation over a kind.
pub fn attains_public_ceiling<P: PublicPolicy>(policy: &P, kind: CaseKind) -> bool {
    mean_value(policy, kind) >= mean_public_ceiling(kind) - 1e-9
}

/// The exact value of the best policy measurable in public history.
pub fn public_ceiling(contract: &Contract) -> f64 {
    pretraining_g0_contract::public_policy_value(&RevealUse, &instance_ambiguity(contract), HORIZON)
}

/// The exact value of a solver that reads the gate.
pub fn privileged_ceiling(contract: &Contract) -> f64 {
    pretraining_g0_contract::privileged_value_bound(
        &RevealUse,
        &instance_ambiguity(contract),
        HORIZON,
    )
}

/// The rate at which a policy commits to the side the gate actually favours.
///
/// The stage-A counterpart of the composite's probe rate: what is measured is
/// use of published content, not purchase of it.
pub fn gate_following_rate<P: PublicPolicy>(policy: &P, kind: CaseKind) -> f64 {
    let cases = cases_of(kind);
    let following = cases
        .iter()
        .filter(|case| {
            let mut actions = Vec::new();
            let revealed_gate = match case.contract.reveal_mode {
                RevealMode::PublishesGate => Some(case.contract.gate),
                _ => None,
            };
            let revealed_decoy = match case.contract.reveal_mode {
                RevealMode::PublishesDecoy => Some(case.contract.decoy),
                _ => None,
            };
            for executed in 0..HORIZON {
                let action = policy.act(&PublicView {
                    contract: &case.contract,
                    executed,
                    revealed_gate,
                    revealed_decoy,
                });
                actions.push(action);
                if action.is_commit() {
                    break;
                }
            }
            actions
                .iter()
                .find(|action| action.is_commit())
                .is_some_and(|action| action.favours() == Some(case.contract.gate))
        })
        .count();
    following as f64 / cases.len() as f64
}

/// The family's central contrast: the published gate is followed on the
/// witness, and no control can be followed better than chance.
pub fn reveal_use_contrast<P: PublicPolicy>(policy: &P) -> bool {
    gate_following_rate(policy, CaseKind::WitnessRevealThenCommit) > 0.99
        && gate_following_rate(policy, CaseKind::NegativeGateHidden) <= 0.51
}

/// Which kernel constructs this family composes.
///
/// Identical to the composite's declared row. The decomposition removes a
/// *decision*, not a construct: the reveal and the shared budget both stay, and
/// what leaves is the probe that made the budget a thing worth spending.
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
/// The ring's adjacency plays no part, so this is a relabelling of the outcome
/// cells and of the commits that reach them. It still exchanges which commit
/// succeeds under which gate, so it is not vacuous.
pub fn relabel(contract: &Contract, symmetry: Symmetry) -> Contract {
    if symmetry.reflect {
        contract.with_flipped_gate()
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
        .absorb(STEP_COST as u64)
        .absorb(Action::ALL.len() as u64);
    for case in card_cases() {
        hasher
            .absorb(case.kind as u64)
            .absorb(case.contract.gate as u64)
            .absorb(case.contract.decoy as u64)
            .absorb(case.contract.reveal_mode as u64)
            .absorb(case.contract.coupling as u64);
    }
    hasher.finish()
}

pub fn action_from_index(index: usize) -> Option<Action> {
    Action::from_index(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretraining_g0_contract::{epistemic_value, noninterference_check};

    #[test]
    fn the_purchase_decision_is_gone_and_the_audit_can_see_it() {
        // The composite's whole content is that one action reduces the
        // surviving set *before* the decision that uses it. Stage A must have
        // no such action, and this is the shared query that says so rather than
        // a comment.
        //
        // A commit does reduce the set, because its outcome is published. That
        // reduction is not a purchase: it arrives after the irreversible
        // decision it would have informed, and the check is written to say
        // exactly that rather than to be quietly weakened into "some action
        // reduces nothing".
        let hidden = CaseKind::NegativeGateHidden.contract_of(Bit::Left, Bit::Left);
        let set = instance_ambiguity(&hidden);
        let entries = epistemic_value(&RevealUse, &set, HORIZON, |a| a.name().into());
        for entry in &entries {
            let commits = Action::ALL
                .into_iter()
                .any(|action| action.is_commit() && action.name() == entry.action);
            if !commits {
                assert_eq!(
                    entry.ambiguity_reduction, 0,
                    "{} bought information in a family with nothing to buy",
                    entry.action
                );
            }
        }
        let best = entries
            .iter()
            .map(|entry| entry.public_value)
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(
            best,
            public_ceiling(&hidden),
            "no opening is worth more than committing at once, so nothing is worth waiting for"
        );
    }

    #[test]
    fn the_reveal_closes_the_gap_and_withholding_it_reopens_the_gap() {
        let witness = CaseKind::WitnessRevealThenCommit.contract_of(Bit::Right, Bit::Left);
        let hidden = CaseKind::NegativeGateHidden.contract_of(Bit::Right, Bit::Left);
        assert_eq!(public_ceiling(&witness), privileged_ceiling(&witness));
        assert!(public_ceiling(&hidden) < privileged_ceiling(&hidden));
        assert_eq!(public_ceiling(&hidden), 49.5);
    }

    #[test]
    fn a_withheld_gate_does_not_separate_traces_before_a_commit() {
        let left = CaseKind::NegativeGateHidden.contract_of(Bit::Left, Bit::Left);
        let right = left.with_flipped_gate();
        let verdict = noninterference_check(
            &RevealUse,
            "a withheld gate is invisible until a commit resolves",
            &left,
            &right,
            HORIZON,
            |actions| actions.iter().all(|action| !action.is_commit()),
            |action| action.name().into(),
        );
        assert!(verdict.holds, "{verdict:?}");
    }

    #[test]
    fn only_a_policy_that_reads_the_publication_follows_the_gate() {
        assert!(reveal_use_contrast(&ExactPublic));
        assert!(!reveal_use_contrast(&BlindCommit));
        assert!(!reveal_use_contrast(&ShamThenBlindCommit));
    }
}
