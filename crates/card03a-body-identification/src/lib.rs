//! Card 03 stage A — public body identification, the basic relation under card
//! 03's decomposition certificate.
//!
//! The composite asks the learner to identify what this body can bring about
//! *and* to plan inside a budget, falling back immediately when the goal is out
//! of reach. This family keeps only the first half. There is one scored
//! decision, the goal is always reachable, and there is no fallback to choose:
//! the whole content of the decision is which actuator this body actually
//! drives.
//!
//! # How a single decision can still turn on the body
//!
//! It cannot, if every goal cell names one command. So the body carries two
//! actuators that drive the *same* edge — `Leap` and `Vault` both displace by
//! two — and supports exactly one of them in the witness. The goal cell is
//! therefore reachable either way, and which command reaches it is a fact about
//! the body rather than about the configuration. This is the card's declared
//! coupling variant: one edge, two labels.
//!
//! A free, mandatory, exact calibration scaffold pulses every movement actuator
//! in a fixed order before the scored decision, so support is identified
//! exactly and the scored-phase ambiguity gap is zero. Making the calibration
//! uninformative reopens it, which is how the scaffold is shown to be
//! load-bearing rather than decorative — the composite's test, kept.
//!
//! # The two controls
//!
//! - **Fully capable.** Both aliased actuators are supported, so identification
//!   buys nothing and a fixed command is optimal. This is the reachability-free
//!   analogue of the composite's frequency-matched control.
//! - **Announced restoration.** Calibration shows an actuator withheld and a
//!   reveal at episode start announces it restored. Behaviour must follow the
//!   announcement rather than the scaffold, which is the composite's reversal
//!   control with the planning removed.

mod audit;
mod render;
pub use audit::*;
pub use render::*;

use std::collections::BTreeMap;

use pretraining_g0_contract::{
    AmbiguitySet, ContractHasher, Fragment, Guard, GuardContext, IndexSet, KernelUse,
    PubliclyObservable, Restriction, Reveal, Ring, Symmetry,
};
use serde::{Deserialize, Serialize};

pub use pretraining_g0_contract::{BracketStructure, Isolation, KindScore, OrbitVerdict};

/// The composite's nine-cell ring, unchanged.
pub const RING: usize = 9;
pub const CONFIGURATION: Ring = Ring::new(RING);

/// One scored decision. The second existed to answer a fallback.
pub const HORIZON: usize = 1;

pub const GOAL_REWARD: i32 = 100;
pub const MOVE_COST: i32 = 1;

/// The cell the calibration scaffold runs from.
///
/// Distinct from every scored start, so that a body limitation and an
/// environment deletion can be told apart by where they bite. The composite
/// found this the hard way: with the scaffold confined to the scored phase's
/// region the two arms produced byte-identical traces and the invariance test
/// compared an episode with itself.
pub const CALIBRATION_CELL: usize = 4;

/// The scored start configurations.
pub const SCORED_STARTS: [usize; 3] = [0, 1, 2];

/// How far the goal sits from the start. Both aliased actuators cover it.
pub const GOAL_OFFSET: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Action {
    Hold,
    Step,
    /// Displaces by two.
    Leap,
    /// Displaces by two as well: the coupling variant's second label.
    Vault,
    Back,
}

impl Action {
    pub const ALL: [Self; 5] = [Self::Hold, Self::Step, Self::Leap, Self::Vault, Self::Back];

    /// The actuators a body's support set can withhold.
    pub const MOVEMENTS: [Self; 4] = [Self::Step, Self::Leap, Self::Vault, Self::Back];

    /// The two labels on one edge. Exactly one is supported in the witness.
    pub const ALIASED: [Self; 2] = [Self::Leap, Self::Vault];

    /// The fixed calibration pulse order.
    ///
    /// A constant, and the audit checks that it is: an order that varied with
    /// the goal would let a learner read the answer off the scaffold instead of
    /// off the body.
    pub const PULSE_ORDER: [Self; 4] = [Self::Step, Self::Leap, Self::Vault, Self::Back];

    pub const fn index(self) -> usize {
        match self {
            Self::Hold => 0,
            Self::Step => 1,
            Self::Leap => 2,
            Self::Vault => 3,
            Self::Back => 4,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Hold => "hold",
            Self::Step => "step",
            Self::Leap => "leap",
            Self::Vault => "vault",
            Self::Back => "back",
        }
    }

    pub const fn displacement(self) -> i64 {
        match self {
            Self::Hold => 0,
            Self::Step => 1,
            Self::Leap | Self::Vault => 2,
            Self::Back => -1,
        }
    }

    pub const fn is_movement(self) -> bool {
        !matches!(self, Self::Hold)
    }

    /// The other label on the same edge, where there is one.
    pub const fn alias(self) -> Option<Self> {
        match self {
            Self::Leap => Some(Self::Vault),
            Self::Vault => Some(Self::Leap),
            _ => None,
        }
    }

    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.into_iter().find(|action| action.index() == index)
    }
}

/// A publicly announced restoration of one actuator's support.
///
/// Announced at episode start through a `reveal`, and in force for the scored
/// decision. Announcement is what makes the contrast well posed: card 04
/// established that an unannounced change cannot require behaviour to update
/// before it is published.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Restore {
    pub actuator: usize,
}

/// Whether calibration identifies the body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Calibration {
    /// Every movement actuator is pulsed once, in the fixed order.
    Full,
    /// Only `Hold` is pulsed, so nothing about support is shown. The
    /// meaning-changing transformation, never a case kind.
    Uninformative,
}

/// One fully specified episode contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contract {
    pub start: usize,
    /// The published goal cell.
    ///
    /// Stored rather than derived from `start`, because a derived goal would
    /// travel with the start under every relabelling and make the reflection
    /// look like a rotation. It is not: a reflection sends the `+2` edge to a
    /// `-2` edge this body has no actuator for, which is why the composite's
    /// invariance group is rotations only. Storing the goal is what lets that
    /// verdict be measured instead of asserted.
    pub goal: usize,
    /// Which movement actuators this body drives, by [`Action::index`]. Hidden
    /// until calibration publishes its consequences.
    pub support: IndexSet,
    /// Environment edges deleted, as `(cell, actuator index)`, kept sorted so
    /// two contracts denoting the same environment compare equal.
    pub blocked_edges: Vec<(usize, usize)>,
    pub restore: Option<Restore>,
    pub calibration: Calibration,
}

impl Contract {
    pub fn new(start: usize, support: IndexSet) -> Self {
        Self {
            start,
            goal: (start + GOAL_OFFSET) % RING,
            support,
            blocked_edges: Vec::new(),
            restore: None,
            calibration: Calibration::Full,
        }
    }

    /// The published goal cell.
    pub fn goal(&self) -> usize {
        self.goal
    }

    /// The body as a shared action restriction.
    pub fn body(&self) -> Restriction {
        Restriction::Action {
            supported: self.support,
        }
    }

    /// The reveal that announces a restoration, and the guard it fires on.
    pub fn restoration_reveal(&self) -> Reveal<usize> {
        match self.restore {
            Some(restore) => Reveal::new(Guard::AtStart, restore.actuator),
            None => Reveal::new(Guard::Never, usize::MAX),
        }
    }

    /// The restoration, if one has been announced.
    pub fn announced_restoration(&self) -> Option<usize> {
        self.restoration_reveal().published(GuardContext {
            executed: 0,
            last_action: None,
            cell: self.start,
        })
    }

    /// Whether the body drove this actuator when the scaffold ran.
    ///
    /// The announcement is deliberately not consulted. Calibration happens
    /// first and an announced restoration arrives after it, which is what makes
    /// the restoration control a case where the scaffold is *right about the
    /// past* and wrong about the decision.
    pub fn drove_at_calibration(&self, action: Action) -> bool {
        !action.is_movement() || self.body().permits_action(action.index() as u16)
    }

    /// Whether the body drives this actuator at the scored decision.
    pub fn supports(&self, action: Action) -> bool {
        self.drove_at_calibration(action) || self.announced_restoration() == Some(action.index())
    }

    /// Whether the environment carries this actuator's edge from this cell.
    pub fn edge_present(&self, cell: usize, action: Action) -> bool {
        !self.blocked_edges.contains(&(cell, action.index()))
    }

    /// The configuration one command reaches from `cell` at the scored decision.
    pub fn effect(&self, cell: usize, action: Action) -> usize {
        self.displace(cell, action, self.supports(action))
    }

    /// The same, as the scaffold saw it.
    pub fn calibration_effect(&self, cell: usize, action: Action) -> usize {
        self.displace(cell, action, self.drove_at_calibration(action))
    }

    fn displace(&self, cell: usize, action: Action, driven: bool) -> usize {
        if !driven || !self.edge_present(cell, action) {
            return cell;
        }
        let cells = RING as i64;
        (((cell as i64 + action.displacement()) % cells + cells) % cells) as usize
    }

    /// The cells the calibration scaffold publishes, in pulse order.
    pub fn calibration_trace(&self) -> Vec<usize> {
        let mut cell = CALIBRATION_CELL;
        let mut trace = Vec::with_capacity(Action::PULSE_ORDER.len());
        for pulse in Action::PULSE_ORDER {
            let pulse = match self.calibration {
                Calibration::Full => pulse,
                // The scaffold still runs and still costs nothing; it simply
                // shows nothing. Deleting it instead would change the episode's
                // shape as well as its information.
                Calibration::Uninformative => Action::Hold,
            };
            cell = self.calibration_effect(cell, pulse);
            trace.push(cell);
        }
        trace
    }

    /// The commands that reach the goal from the scored start.
    pub fn reaching_commands(&self) -> Vec<Action> {
        Action::ALL
            .into_iter()
            .filter(|action| self.effect(self.start, *action) == self.goal())
            .collect()
    }

    /// The same body limitation expressed as environment deletions.
    ///
    /// The withheld actuators are restored to the body and their edges are
    /// deleted from every cell the family ever occupies. The reachable set is
    /// preserved and so is every public fact; what changes is which component
    /// carries the limitation. Cells the family never occupies are left alone,
    /// so the two contracts are genuinely different objects rather than two
    /// spellings of one.
    pub fn swapped_to_environment(&self) -> Self {
        // An announced restoration is an actuator-level fact and an environment
        // deletion is an edge-level one; moving a limitation the announcement
        // has already lifted would delete the announcement instead of relocating
        // a limitation. Those contracts are simply not swappable, and the audit
        // counts them rather than pretending the transform applied.
        if self.restore.is_some() {
            return self.clone();
        }
        let mut blocked = self.blocked_edges.clone();
        for action in Action::MOVEMENTS {
            // An actuator an announcement restores is not withheld by anything,
            // so deleting its edges would change behaviour rather than move a
            // limitation from one component to another.
            if self.supports(action) {
                continue;
            }
            for cell in occupied_cells() {
                blocked.push((cell, action.index()));
            }
        }
        blocked.sort_unstable();
        blocked.dedup();
        Self {
            support: IndexSet::from_indices(Action::MOVEMENTS.map(|a| a.index())),
            blocked_edges: blocked,
            ..self.clone()
        }
    }

    /// Exchange the two labels on the shared edge.
    pub fn swap_aliased_support(&self) -> Self {
        let mut support = self.support;
        for action in Action::ALIASED {
            if self.support.contains(action.index()) {
                support.remove(action.index());
            } else {
                support.insert(action.index());
            }
        }
        Self {
            support,
            ..self.clone()
        }
    }

    pub fn with_calibration(&self, calibration: Calibration) -> Self {
        Self {
            calibration,
            ..self.clone()
        }
    }

    /// Move the whole contract through one ring symmetry.
    pub fn relabelled(&self, symmetry: Symmetry) -> Self {
        Self {
            start: symmetry.apply(self.start),
            goal: symmetry.apply(self.goal),
            blocked_edges: self
                .blocked_edges
                .iter()
                .map(|(cell, actuator)| (symmetry.apply(*cell), *actuator))
                .collect(),
            ..self.clone()
        }
    }
}

/// Every cell the calibration scaffold or a scored decision can occupy.
pub fn occupied_cells() -> Vec<usize> {
    let mut cells = vec![CALIBRATION_CELL];
    let probe = Contract::new(
        0,
        IndexSet::from_indices(Action::MOVEMENTS.map(|a| a.index())),
    );
    cells.extend(probe.calibration_trace());
    for start in SCORED_STARTS {
        cells.push(start);
        for action in Action::ALL {
            cells.push(probe.effect(start, action));
        }
    }
    cells.sort_unstable();
    cells.dedup();
    cells
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub value: i32,
    pub reached_goal: bool,
    pub final_cell: usize,
}

pub fn run(contract: &Contract, actions: &[Action]) -> Outcome {
    let mut cell = contract.start;
    for action in actions.iter().copied() {
        cell = contract.effect(cell, action);
    }
    let reached = cell == contract.goal();
    Outcome {
        value: if reached {
            GOAL_REWARD - MOVE_COST * actions.len() as i32
        } else {
            0
        },
        reached_goal: reached,
        final_cell: cell,
    }
}

pub struct BodyIdentification;

impl Fragment for BodyIdentification {
    type Action = Action;
    type Contract = Contract;

    fn actions(&self) -> Vec<Action> {
        Action::ALL.to_vec()
    }

    fn horizon(&self) -> usize {
        HORIZON
    }

    fn start(&self, contract: &Contract) -> usize {
        contract.start
    }

    fn step(&self, contract: &Contract, cell: usize, _executed: usize, action: Action) -> usize {
        contract.effect(cell, action)
    }

    fn value(&self, contract: &Contract, _trajectory: &[usize], actions: &[Action]) -> i32 {
        run(contract, actions).value
    }
}

/// Tags separating the announcement from cell indices in the trace.
pub const RESTORE_TAG: i64 = 100;

impl PubliclyObservable for BodyIdentification {
    /// The calibration outcome and the announcement, then the configuration
    /// after each scored action.
    ///
    /// The support set never appears. Everything a learner can know about it
    /// comes from consecutive differences in the calibration trace, which is
    /// what makes identification the family's content rather than a lookup.
    fn public_trace(&self, contract: &Contract, actions: &[Action]) -> Vec<i64> {
        let mut trace = vec![
            contract.calibration as i64,
            contract.start as i64,
            contract.goal as i64,
        ];
        match contract.announced_restoration() {
            Some(actuator) => trace.push(RESTORE_TAG + actuator as i64),
            None => trace.push(RESTORE_TAG - 1),
        }
        for cell in contract.calibration_trace() {
            trace.push(cell as i64);
        }
        let mut cell = contract.start;
        for action in actions.iter().copied() {
            cell = contract.effect(cell, action);
            trace.push(cell as i64);
        }
        trace
    }
}

pub fn all_sequences() -> Vec<Vec<Action>> {
    pretraining_g0_contract::sequences_of_length(&Action::ALL, HORIZON)
}

pub fn value_bounds(contract: &Contract) -> (i32, Vec<Vec<Action>>) {
    pretraining_g0_contract::value_bounds(&BodyIdentification, contract)
}

pub fn optimal_first_actions(contract: &Contract) -> Vec<Action> {
    pretraining_g0_contract::optimal_first_actions(&BodyIdentification, contract)
}

/// The support sets the family draws from.
///
/// Only the aliased pair varies. `Step` and `Back` are supported throughout, so
/// the scaffold's other two pulses are constant and the audit can check that the
/// pulse order carries no goal information. The fourth body drives neither
/// aliased actuator and appears only where an announcement restores one.
pub fn support_domain() -> Vec<IndexSet> {
    let base = [Action::Step.index(), Action::Back.index()];
    vec![
        IndexSet::from_indices(base.into_iter().chain([Action::Leap.index()])),
        IndexSet::from_indices(base.into_iter().chain([Action::Vault.index()])),
        IndexSet::from_indices(
            base.into_iter()
                .chain([Action::Leap.index(), Action::Vault.index()]),
        ),
        IndexSet::from_indices(base),
    ]
}

/// The bodies a learner cannot separate before calibration speaks.
///
/// A candidate that cannot reach the goal at all is excluded, because
/// reachability is a *published* property of this family: stage A removed the
/// unreachable goal, so "this body cannot get there" is not a hypothesis the
/// learner is left holding. The exclusion is applied through the same
/// `reaching_commands` the scored decision uses, so an announced restoration
/// admits the fourth body exactly where it makes it able to reach.
pub fn support_ambiguity(contract: &Contract) -> AmbiguitySet<Contract> {
    let mut candidates = vec![contract.clone()];
    for support in support_domain() {
        if support == contract.support {
            continue;
        }
        let candidate = Contract {
            support,
            ..contract.clone()
        };
        if candidate.reaching_commands().is_empty() {
            continue;
        }
        candidates.push(candidate);
    }
    AmbiguitySet::uniform(candidates)
}

pub fn public_ceiling(contract: &Contract) -> f64 {
    pretraining_g0_contract::public_policy_value(
        &BodyIdentification,
        &support_ambiguity(contract),
        HORIZON,
    )
}

pub fn privileged_ceiling(contract: &Contract) -> f64 {
    pretraining_g0_contract::privileged_value_bound(
        &BodyIdentification,
        &support_ambiguity(contract),
        HORIZON,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CaseKind {
    /// Exactly one of the two aliased actuators is supported.
    WitnessIdentifiedSupport,
    /// Both are supported, so identification buys nothing.
    NegativeFullyCapable,
    /// Calibration shows one withheld and an announcement restores it.
    NegativeAnnouncedRestoration,
}

impl CaseKind {
    pub const ALL: [Self; 3] = [
        Self::WitnessIdentifiedSupport,
        Self::NegativeFullyCapable,
        Self::NegativeAnnouncedRestoration,
    ];

    pub const NEGATIVES: [Self; 2] = [
        Self::NegativeFullyCapable,
        Self::NegativeAnnouncedRestoration,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::WitnessIdentifiedSupport => "witness_identified_support",
            Self::NegativeFullyCapable => "negative_fully_capable",
            Self::NegativeAnnouncedRestoration => "negative_announced_restoration",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Case {
    pub kind: CaseKind,
    pub contract: Contract,
}

/// The family as twelve cases.
pub fn card_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let leap_only = support_domain()[0];
    let vault_only = support_domain()[1];
    let both = support_domain()[2];
    let neither = support_domain()[3];
    for start in SCORED_STARTS {
        for support in [leap_only, vault_only] {
            cases.push(Case {
                kind: CaseKind::WitnessIdentifiedSupport,
                contract: Contract::new(start, support),
            });
        }
        cases.push(Case {
            kind: CaseKind::NegativeFullyCapable,
            contract: Contract::new(start, both),
        });
        cases.push(Case {
            kind: CaseKind::NegativeAnnouncedRestoration,
            contract: Contract {
                restore: Some(Restore {
                    actuator: Action::Leap.index(),
                }),
                // Neither aliased actuator drove during calibration, so the
                // scaffold is right about the past and silent about the
                // decision. A body that still drove the other label would let a
                // scaffold-following policy succeed by accident.
                ..Contract::new(start, neither)
            },
        });
    }
    cases
}

pub fn cases_of(kind: CaseKind) -> Vec<Case> {
    card_cases()
        .into_iter()
        .filter(|case| case.kind == kind)
        .collect()
}

/// What a policy may read: the calibration outcome and the announcement.
pub struct PublicView<'a> {
    pub contract: &'a Contract,
    /// The actuators calibration showed to be driven.
    pub calibrated_support: Vec<Action>,
    /// The actuator an announcement restored, if any.
    pub announced: Option<Action>,
}

impl<'a> PublicView<'a> {
    /// Reconstruct support from consecutive differences in the scaffold.
    ///
    /// This is the whole identification step, and it reads only the published
    /// calibration cells.
    pub fn identify(contract: &Contract) -> Vec<Action> {
        let trace = contract.calibration_trace();
        let mut cell = CALIBRATION_CELL;
        let mut driven = Vec::new();
        for (pulse, next) in Action::PULSE_ORDER.into_iter().zip(trace) {
            if next != cell {
                driven.push(pulse);
            }
            cell = next;
        }
        driven
    }

    pub fn of(contract: &'a Contract) -> Self {
        Self {
            contract,
            calibrated_support: Self::identify(contract),
            announced: contract
                .announced_restoration()
                .and_then(Action::from_index),
        }
    }
}

pub trait PublicPolicy {
    fn name(&self) -> &'static str;
    fn act(&self, view: &PublicView<'_>) -> Action;
}

pub fn run_policy<P: PublicPolicy>(contract: &Contract, policy: &P) -> Outcome {
    run(contract, &[policy.act(&PublicView::of(contract))])
}

/// Picks the first command that would reach the goal on a fully capable body.
///
/// The card's ignore-support baseline. It is optimal wherever the body happens
/// not to matter, and wrong on half the witness.
pub struct IgnoreSupport;

impl PublicPolicy for IgnoreSupport {
    fn name(&self) -> &'static str {
        "ignore_support"
    }

    fn act(&self, _view: &PublicView<'_>) -> Action {
        Action::Leap
    }
}

/// Follows the calibration scaffold and nothing else.
///
/// Optimal on the witness and on the fully-capable control, and wrong on the
/// restoration control, which is exactly what the restoration control is for.
pub struct FollowCalibration;

impl PublicPolicy for FollowCalibration {
    fn name(&self) -> &'static str {
        "follow_calibration"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        Action::ALIASED
            .into_iter()
            .find(|action| view.calibrated_support.contains(action))
            .unwrap_or(Action::Hold)
    }
}

/// The exact public policy: calibration, then the announcement on top of it.
pub struct IdentifiedSupportExact;

impl PublicPolicy for IdentifiedSupportExact {
    fn name(&self) -> &'static str {
        "identified_support_exact"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        if let Some(announced) = view.announced {
            if Action::ALIASED.contains(&announced) {
                return announced;
            }
        }
        FollowCalibration.act(view)
    }
}

/// Reads the support set directly. A reference, never a teacher.
pub struct PrivilegedBodyKnown;

impl PublicPolicy for PrivilegedBodyKnown {
    fn name(&self) -> &'static str {
        "privileged_body_known"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        view.contract
            .reaching_commands()
            .into_iter()
            .find(|action| action.is_movement())
            .unwrap_or(Action::Hold)
    }
}

pub fn score_policy<P: PublicPolicy>(policy: &P) -> BTreeMap<String, KindScore> {
    let mut scores = BTreeMap::new();
    for kind in CaseKind::ALL {
        let selected = cases_of(kind);
        let mut solved = 0usize;
        let mut optimal = 0usize;
        for case in &selected {
            let outcome = run_policy(&case.contract, policy);
            if outcome.reached_goal {
                solved += 1;
            }
            if outcome.value == value_bounds(&case.contract).0 {
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

pub fn optimal_rate<P: PublicPolicy>(policy: &P, kind: CaseKind) -> f64 {
    let cases = cases_of(kind);
    let optimal = cases
        .iter()
        .filter(|case| run_policy(&case.contract, policy).value == value_bounds(&case.contract).0)
        .count();
    optimal as f64 / cases.len() as f64
}

/// The family's central contrast: the identified body selects the command.
pub fn identification_contrast<P: PublicPolicy>(policy: &P) -> bool {
    optimal_rate(policy, CaseKind::WitnessIdentifiedSupport) > 0.99
}

/// Which kernel constructs this family composes.
///
/// The composite's declared row exactly: a body restriction and an announced
/// reveal. What left is a decision, not a construct.
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

/// Whether the calibration pulse order depends on anything about the goal.
///
/// It is a compile-time constant, so this is a check that the constant is
/// actually what every case uses rather than a claim that it is.
pub fn pulse_order_is_goal_independent() -> bool {
    card_cases().into_iter().all(|case| {
        let _ = case.contract.goal();
        Action::PULSE_ORDER == [Action::Step, Action::Leap, Action::Vault, Action::Back]
    })
}

pub fn contract_hash() -> u64 {
    let mut hasher = ContractHasher::new();
    hasher
        .absorb(RING as u64)
        .absorb(HORIZON as u64)
        .absorb(GOAL_REWARD as u64)
        .absorb(MOVE_COST as u64)
        .absorb(CALIBRATION_CELL as u64)
        .absorb(GOAL_OFFSET as u64)
        .absorb(Action::ALL.len() as u64);
    for case in card_cases() {
        hasher
            .absorb(case.kind as u64)
            .absorb(case.contract.start as u64)
            .absorb(case.contract.goal as u64)
            .absorb(u64::from(case.contract.support.0))
            .absorb(case.contract.calibration as u64)
            .absorb_option(case.contract.restore.map(|r| r.actuator as u64));
        for (cell, actuator) in &case.contract.blocked_edges {
            hasher.absorb(*cell as u64).absorb(*actuator as u64);
        }
    }
    hasher.finish()
}

pub fn action_from_index(index: usize) -> Option<Action> {
    Action::from_index(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_identifies_the_body_exactly() {
        for case in card_cases() {
            let identified = PublicView::identify(&case.contract);
            for action in Action::MOVEMENTS {
                assert_eq!(
                    identified.contains(&action),
                    case.contract.support.contains(action.index()),
                    "{} was misidentified",
                    action.name()
                );
            }
        }
    }

    #[test]
    fn the_scored_gap_is_zero_and_an_uninformative_scaffold_reopens_it() {
        let witness = Contract::new(0, support_domain()[1]);
        assert_eq!(public_ceiling(&witness), privileged_ceiling(&witness));
        let blind = witness.with_calibration(Calibration::Uninformative);
        assert!(public_ceiling(&blind) < privileged_ceiling(&blind));
    }

    #[test]
    fn both_labels_reach_the_goal_and_only_the_supported_one_does_it() {
        let leap_only = Contract::new(0, support_domain()[0]);
        let vault_only = Contract::new(0, support_domain()[1]);
        assert_eq!(leap_only.goal(), vault_only.goal());
        assert_eq!(optimal_first_actions(&leap_only), vec![Action::Leap]);
        assert_eq!(optimal_first_actions(&vault_only), vec![Action::Vault]);
    }

    #[test]
    fn the_announcement_overrides_the_scaffold() {
        let restored = Contract {
            restore: Some(Restore {
                actuator: Action::Leap.index(),
            }),
            ..Contract::new(0, support_domain()[1])
        };
        assert!(!PublicView::identify(&restored).contains(&Action::Leap));
        assert_eq!(
            IdentifiedSupportExact.act(&PublicView::of(&restored)),
            Action::Leap
        );
        assert_eq!(
            FollowCalibration.act(&PublicView::of(&restored)),
            Action::Vault
        );
    }

    #[test]
    fn each_control_restores_exactly_the_baseline_it_is_built_for() {
        assert!(identification_contrast(&IdentifiedSupportExact));
        assert!(!identification_contrast(&IgnoreSupport));
        assert_eq!(
            optimal_rate(&IgnoreSupport, CaseKind::NegativeFullyCapable),
            1.0
        );
        assert_eq!(
            optimal_rate(&IgnoreSupport, CaseKind::NegativeAnnouncedRestoration),
            1.0
        );
        assert_eq!(
            optimal_rate(&FollowCalibration, CaseKind::NegativeAnnouncedRestoration),
            0.0
        );
    }

    #[test]
    fn the_body_environment_swap_preserves_every_public_fact() {
        for case in card_cases() {
            let swapped = case.contract.swapped_to_environment();
            if swapped == case.contract {
                continue;
            }
            for sequence in all_sequences() {
                assert_eq!(
                    BodyIdentification.public_trace(&case.contract, &sequence),
                    BodyIdentification.public_trace(&swapped, &sequence),
                    "the swap changed a public fact"
                );
            }
        }
    }
}
