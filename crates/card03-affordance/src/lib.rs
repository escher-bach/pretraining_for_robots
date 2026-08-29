//! Card 03 — Affordance and Reachability, built as an audited world.
//!
//! The claim is about *allocation before failure*: the learner should change
//! what it attempts according to what this body can bring about in this
//! environment, and it should do so on its first scored decision rather than
//! after an attempt has already failed. Everything below exists to make that
//! sentence falsifiable.
//!
//! # What is shared and what is not
//!
//! The configuration structure is [`pretraining_g0_contract::Ring`] — the same
//! type card 04 uses, with a different cell count, which is a family parameter
//! and not a second environment. The body limitation is
//! [`Restriction::Action`], the same type card 04 uses for its viability
//! boundary in a different variant, so "restriction" names one object across
//! the portfolio. The support restoration is [`Reveal`], the same construct
//! card 04's announced goal switch uses.
//!
//! What is *not* shared is the invariance group. Card 04's actions are
//! `{-1, 0, +1}` and the full dihedral group of the ring preserves its meaning.
//! This card's actions are `{0, +1, +2, -1}`, which no reflection preserves:
//! there is no `-2`. So only the rotation subgroup is semantics-preserving here,
//! and [`orbit_verdicts`] checks a reflection as a *meaning-changing*
//! transformation rather than assuming the environment's group is the card's.
//!
//! # Geometry
//!
//! Nine cells and two scored decisions. Both numbers are forced rather than
//! chosen, by the one control that is hardest to build:
//!
//! - the frequency-matched control needs goals a **fully capable** body cannot
//!   reach inside the budget. A fully capable body's two-step displacement set
//!   is `{-2, -1, 0, +1, +2, +3, +4}`, so the ring must have more than seven
//!   cells for any goal to be budget-unreachable at all. Nine leaves exactly
//!   two such cells, `+5` and `+6`.
//! - a support-limited body without `Back` reaches `{0, +1, +2, +3, +4}`, so
//!   `+8` is unreachable to it and one step away for a capable one. That pair is
//!   the witness.
//!
//! Card 04 recorded the same kind of finding — a five-cell ring rather than six,
//! because the detour had to fit the horizon. Neither number is in the card text.
//!
//! # Where the epistemic content is, and is not
//!
//! Calibration is scaffold: it is mandatory, it consumes no scored budget, and
//! its pulse order is a constant that cannot depend on the goal. After it, the
//! body is exactly identified, so the public and privileged views of the scored
//! phase coincide and any learner shortfall is skill rather than information.
//! The ambiguity lives entirely *before* calibration and is reported as an
//! identification diameter, not as a value gap. The uninformative-calibration
//! transformation is what turns that diameter back on, and its verdict is
//! computed from the shared query algebra rather than asserted.

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

/// Nine cells. See the module note: seven or fewer make every goal reachable
/// inside the budget for a fully capable body, which deletes the
/// frequency-matched control rather than weakening it.
pub const RING: usize = 9;

/// The shared configuration structure, at this card's parameter.
pub const CONFIGURATION: Ring = Ring::new(RING);

/// Two scored decisions after calibration.
pub const HORIZON: usize = 2;

pub const GOAL_REWARD: i32 = 100;
pub const MOVE_COST: i32 = 1;

/// What an immediate fallback secures.
///
/// It has to sit strictly between the worst successful attempt and outright
/// failure: below `GOAL_REWARD - MOVE_COST * HORIZON` so a reachable goal is
/// always worth attempting, and above zero so an unreachable one is worth
/// abandoning. Checked at compile time rather than trusted.
pub const FALLBACK_REWARD: i32 = 50;

const _: () = assert!(FALLBACK_REWARD < GOAL_REWARD - MOVE_COST * HORIZON as i32);
const _: () = assert!(FALLBACK_REWARD > 0);

/// The body's action repertoire.
///
/// `Leap` is the coupling variant the card names: one command traverses two
/// edges together. `Hold` and `Fallback` are part of every body; only the three
/// movement actuators are subject to support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Action {
    Hold,
    Step,
    Leap,
    Back,
    Fallback,
}

impl Action {
    pub const ALL: [Self; 5] = [
        Self::Hold,
        Self::Step,
        Self::Leap,
        Self::Back,
        Self::Fallback,
    ];

    /// The actuators a body may or may not drive.
    pub const MOVEMENTS: [Self; 3] = [Self::Step, Self::Leap, Self::Back];

    /// The fixed calibration pulse order.
    ///
    /// A constant, and the audit checks that it is: an order that varied with
    /// the goal would let a learner read the goal's reachability off the
    /// scaffold instead of off the body.
    ///
    /// `Leap` appears twice, and that is not padding. The environment-deletion
    /// twin of a body limitation has to delete the withheld edge at every cell
    /// the *scored* phase can command from, or the reachable set would not be
    /// preserved. With a three-pulse scaffold the calibration never left that
    /// region, so the two arms produced byte-identical traces and the invariance
    /// test compared a thing with itself. The extra `Leap` carries the scaffold
    /// to a cell outside the task's reach, where the body limit and the
    /// environment deletion visibly differ. Identification is unaffected: the
    /// three support bits are still read off consecutive differences.
    pub const PULSE_ORDER: [Self; 4] = [Self::Step, Self::Leap, Self::Leap, Self::Back];

    pub const fn index(self) -> usize {
        match self {
            Self::Hold => 0,
            Self::Step => 1,
            Self::Leap => 2,
            Self::Back => 3,
            Self::Fallback => 4,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Hold => "hold",
            Self::Step => "step",
            Self::Leap => "leap",
            Self::Back => "back",
            Self::Fallback => "fallback",
        }
    }

    /// The displacement this actuator commands when it is supported.
    pub const fn displacement(self) -> i64 {
        match self {
            Self::Hold | Self::Fallback => 0,
            Self::Step => 1,
            Self::Leap => 2,
            Self::Back => -1,
        }
    }

    /// Whether the body's support set can withhold this actuator.
    pub const fn is_movement(self) -> bool {
        matches!(self, Self::Step | Self::Leap | Self::Back)
    }

    fn from_index(index: usize) -> Option<Self> {
        Self::ALL.into_iter().find(|action| action.index() == index)
    }
}

/// A publicly announced restoration of one actuator's support.
///
/// It is announced at episode start and takes effect after a stated step. The
/// announcement is what makes the contrast well posed: card 04 established that
/// an *unannounced* change cannot require behaviour to update before it is
/// published, because falling back was correct on the information available.
/// Here fallback is absorbing, so an unannounced restoration would arrive after
/// the episode had already ended correctly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Restore {
    pub actuator: usize,
    pub after_step: usize,
}

/// Whether calibration identifies the body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Calibration {
    /// Every movement actuator is pulsed once, in the fixed order.
    Full,
    /// Only `Hold` is pulsed, so nothing about support is shown. This is the
    /// card's meaning-changing transformation, not a variant of the witness.
    Uninformative,
}

/// One fully specified episode contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contract {
    pub start: usize,
    pub goal: usize,
    /// Which movement actuators this body drives, by [`Action::index`].
    pub support: IndexSet,
    /// Environment edges deleted at a specific cell, as `(cell, actuator index)`.
    ///
    /// Kept sorted so two contracts denoting the same environment compare and
    /// hash equal. This is what makes the body/environment swap a real
    /// transformation rather than a relabelling: a body limit withdraws an
    /// actuator everywhere, an environment deletion withdraws one edge.
    pub blocked_edges: Vec<(usize, usize)>,
    pub restore: Option<Restore>,
    pub calibration: Calibration,
}

impl Contract {
    pub fn new(start: usize, goal: usize, support: IndexSet) -> Self {
        Self {
            start,
            goal,
            support,
            blocked_edges: Vec::new(),
            restore: None,
            calibration: Calibration::Full,
        }
    }

    pub fn with_blocked_edges(mut self, edges: &[(usize, usize)]) -> Self {
        let mut sorted = edges.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        self.blocked_edges = sorted;
        self
    }

    pub fn with_restore(mut self, restore: Restore) -> Self {
        self.restore = Some(restore);
        self
    }

    pub fn with_calibration(mut self, calibration: Calibration) -> Self {
        self.calibration = calibration;
        self
    }

    /// The body limitation as a shared restriction.
    pub fn body(&self) -> Restriction {
        Restriction::Action {
            supported: self.support,
        }
    }

    /// The restoration as a shared reveal, announced at episode start.
    pub fn reveal(&self) -> Option<Reveal<usize>> {
        self.restore
            .map(|restore| Reveal::new(Guard::AtStart, restore.actuator))
    }

    /// Whether the actuator is driven by the body after `executed` actions.
    ///
    /// Support is a body fact and the restoration is a public event, so this is
    /// a function of the step index. It says nothing about where the body is.
    pub fn supports(&self, action: Action, executed: usize) -> bool {
        if !action.is_movement() {
            return true;
        }
        if self.body().permits_action(action.index() as u16) {
            return true;
        }
        match self.restore {
            Some(restore) if restore.actuator == action.index() => {
                Guard::AfterStep(restore.after_step).fired(GuardContext {
                    executed,
                    last_action: None,
                    cell: self.start,
                })
            }
            _ => false,
        }
    }

    /// Whether this environment carries the edge leaving `cell` under `action`.
    pub fn edge_present(&self, cell: usize, action: Action) -> bool {
        !self.blocked_edges.contains(&(cell, action.index()))
    }

    /// The configuration after one command, with both limitations applied.
    ///
    /// The two reasons a command can fail to move the body are deliberately
    /// applied by different objects — [`Restriction::Action`] for the body, the
    /// edge list for the environment — because the card's invariance claim is
    /// that behaviour depends on the resulting reachable set and not on which
    /// of the two produced it.
    pub fn advance(&self, cell: usize, executed: usize, action: Action) -> usize {
        if !self.supports(action, executed) || !self.edge_present(cell, action) {
            return cell;
        }
        let displacement = action.displacement();
        let shifted = cell as i64 + displacement;
        shifted.rem_euclid(RING as i64) as usize
    }

    /// The public calibration trace: the cell after each scaffold pulse.
    ///
    /// Pulses are not reset between them, so the trace is cumulative and each
    /// effect is read from consecutive differences. That is a harder
    /// identification problem than resetting after every pulse, and it is still
    /// exact: `+1`, `+2`, and `-1` are pairwise distinct and each distinct from
    /// `0` on a nine-cell ring.
    pub fn calibration_trace(&self) -> Vec<usize> {
        let mut cell = self.start;
        let mut trace = vec![cell];
        for action in self.calibration_pulses() {
            // Calibration runs before the scored phase, so the restoration has
            // not fired: step zero is the right index and passing `executed`
            // here would be passing the scored clock into the scaffold.
            cell = self.advance(cell, 0, action);
            trace.push(cell);
        }
        trace
    }

    /// This contract as seen from a mid-episode decision.
    ///
    /// A policy that re-solves from step `executed` over the remaining budget
    /// hands the solver a *fresh* episode, and a fresh episode's step clock
    /// starts at zero. Anything time-dependent has to be rebased, and clamping
    /// the restoration's step index with a saturating subtraction is not
    /// rebasing: it turns "already fired" into "fires after the next action",
    /// which is one step too late and cost the exact policy its own witness
    /// before this method existed.
    ///
    /// A fired restoration is therefore folded into the support set, where it is
    /// no longer time-dependent at all, and only a pending one keeps a shifted
    /// index.
    pub fn resolved_from(&self, cell: usize, executed: usize) -> Self {
        let mut probe = self.clone();
        probe.start = cell;
        probe.restore = match self.restore {
            Some(restore) if executed > restore.after_step => {
                probe.support.insert(restore.actuator);
                None
            }
            Some(restore) => Some(Restore {
                after_step: restore.after_step - executed,
                ..restore
            }),
            None => None,
        };
        probe
    }

    /// The pulses this contract's calibration actually delivers.
    pub fn calibration_pulses(&self) -> Vec<Action> {
        match self.calibration {
            Calibration::Full => Action::PULSE_ORDER.to_vec(),
            // Not "no calibration": the scaffold still runs and still costs the
            // same number of public records, so the two arms differ in what is
            // shown rather than in how long the episode is.
            Calibration::Uninformative => vec![Action::Hold; Action::PULSE_ORDER.len()],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub value: i32,
    pub reached_goal: bool,
    pub fell_back: bool,
    /// Decisions taken before the fallback, when there was one.
    pub fallback_step: Option<usize>,
    pub final_cell: usize,
}

/// Execute a complete scored action sequence.
///
/// Fallback is absorbing, and it is absorbing in the *value* rather than in the
/// transition: anything after the first fallback is unscored. That keeps the
/// configuration space the ring rather than the ring plus a terminal cell, which
/// would put a non-ring element inside the environment's symmetry group.
pub fn run(contract: &Contract, actions: &[Action]) -> Outcome {
    let mut cell = contract.start;
    let mut trajectory = vec![cell];
    let mut fallback_step = None;
    for (executed, action) in actions.iter().copied().enumerate() {
        if fallback_step.is_some() {
            trajectory.push(cell);
            continue;
        }
        if action == Action::Fallback {
            fallback_step = Some(executed);
            trajectory.push(cell);
            continue;
        }
        cell = contract.advance(cell, executed, action);
        trajectory.push(cell);
    }

    if let Some(step) = fallback_step {
        return Outcome {
            value: FALLBACK_REWARD - MOVE_COST * step as i32,
            reached_goal: false,
            fell_back: true,
            fallback_step: Some(step),
            final_cell: cell,
        };
    }

    let settle = (0..trajectory.len()).find(|index| {
        trajectory[*index..]
            .iter()
            .all(|entry| *entry == contract.goal)
    });
    Outcome {
        value: settle.map_or(0, |steps| GOAL_REWARD - MOVE_COST * steps as i32),
        reached_goal: settle.is_some(),
        fell_back: false,
        fallback_step: None,
        final_cell: cell,
    }
}

/// The card as an exhaustively auditable fragment.
pub struct Affordance;

impl Fragment for Affordance {
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

    fn step(&self, contract: &Contract, cell: usize, executed: usize, action: Action) -> usize {
        contract.advance(cell, executed, action)
    }

    fn value(&self, contract: &Contract, _trajectory: &[usize], actions: &[Action]) -> i32 {
        run(contract, actions).value
    }
}

impl PubliclyObservable for Affordance {
    /// Everything the learner has seen: the calibration trace, the announced
    /// restoration, and the scored configurations up to a fallback.
    ///
    /// Support is not here. It reaches the trace only through the cells the
    /// calibration pulses produced, which is the whole design: a body is
    /// identified by what it did, not by a published field.
    fn public_trace(&self, contract: &Contract, actions: &[Action]) -> Vec<i64> {
        let mut trace: Vec<i64> = contract
            .calibration_trace()
            .into_iter()
            .map(|cell| cell as i64)
            .collect();
        trace.push(contract.goal as i64);
        // The announcement is public by construction, so it belongs in the
        // trace; leaving it out would make two publicly different episodes
        // count as indistinguishable.
        trace.push(match contract.restore {
            Some(restore) => (restore.actuator * 100 + restore.after_step + 1) as i64,
            None => 0,
        });
        let mut cell = contract.start;
        for (executed, action) in actions.iter().copied().enumerate() {
            if action == Action::Fallback {
                trace.push(-1);
                break;
            }
            cell = contract.advance(cell, executed, action);
            trace.push(cell as i64);
        }
        trace
    }
}

pub fn all_sequences() -> Vec<Vec<Action>> {
    pretraining_g0_contract::sequences_of_length(&Action::ALL, HORIZON)
}

pub fn value_bounds(contract: &Contract) -> (i32, Vec<Vec<Action>>) {
    pretraining_g0_contract::value_bounds(&Affordance, contract)
}

pub fn value_bounds_over(contract: &Contract, horizon: usize) -> (i32, Vec<Vec<Action>>) {
    pretraining_g0_contract::value_bounds_over(&Affordance, contract, horizon)
}

pub fn optimal_first_actions(contract: &Contract) -> Vec<Action> {
    pretraining_g0_contract::optimal_first_actions(&Affordance, contract)
}

/// Whether the goal can be settled on inside the budget.
///
/// Derived from the enumeration rather than from a closed-form distance, because
/// a closed form would have to re-derive support, blocked edges, and the
/// restoration — three chances to disagree with the world.
pub fn goal_is_reachable(contract: &Contract) -> bool {
    all_sequences()
        .into_iter()
        .any(|sequence| run(contract, &sequence).reached_goal)
}

/// Every body the family admits, which is the pre-calibration ambiguity set.
///
/// All eight subsets of the three movement actuators. A body is a hidden
/// parameter drawn from this domain; calibration is what collapses it.
pub fn admissible_bodies() -> Vec<IndexSet> {
    let movements: Vec<usize> = Action::MOVEMENTS.into_iter().map(Action::index).collect();
    let mut bodies = Vec::with_capacity(1 << movements.len());
    for mask in 0..(1u32 << movements.len()) {
        let mut set = IndexSet::EMPTY;
        for (bit, index) in movements.iter().enumerate() {
            if mask >> bit & 1 == 1 {
                set.insert(*index);
            }
        }
        bodies.push(set);
    }
    bodies
}

/// The ambiguity set a learner faces before calibration has told it anything.
///
/// Every admissible body under this contract's goal, environment, and
/// announcement. The realized body is first so `identify` reports the class the
/// learner is actually in.
pub fn body_ambiguity(contract: &Contract) -> AmbiguitySet<Contract> {
    let mut candidates = vec![contract.clone()];
    for body in admissible_bodies() {
        if body == contract.support {
            continue;
        }
        candidates.push(Contract {
            support: body,
            ..contract.clone()
        });
    }
    AmbiguitySet::uniform(candidates)
}

/// Which sub-claim or control a case belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CaseKind {
    /// The goal is out of this body's reach; the first scored decision must be
    /// the fallback.
    WitnessUnreachableFallback,
    /// The same body, a reachable goal; falling back is now wrong.
    WitnessReachableAttempt,
    /// Support is publicly restored mid-episode, so an unreachable goal becomes
    /// reachable and the fallback becomes wrong.
    WitnessRestore,
    /// A fully capable body with goals that fail only on the budget.
    NegativeFrequencyMatched,
    /// The restoration case with the restoration removed.
    NegativeNoRestore,
}

impl CaseKind {
    pub const ALL: [Self; 5] = [
        Self::WitnessUnreachableFallback,
        Self::WitnessReachableAttempt,
        Self::WitnessRestore,
        Self::NegativeFrequencyMatched,
        Self::NegativeNoRestore,
    ];

    pub const NEGATIVES: [Self; 2] = [Self::NegativeFrequencyMatched, Self::NegativeNoRestore];

    pub const fn is_witness(self) -> bool {
        matches!(
            self,
            Self::WitnessUnreachableFallback | Self::WitnessReachableAttempt | Self::WitnessRestore
        )
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::WitnessUnreachableFallback => "witness_unreachable_fallback",
            Self::WitnessReachableAttempt => "witness_reachable_attempt",
            Self::WitnessRestore => "witness_restore",
            Self::NegativeFrequencyMatched => "negative_frequency_matched",
            Self::NegativeNoRestore => "negative_no_restore",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Case {
    pub kind: CaseKind,
    pub contract: Contract,
}

fn body(actions: &[Action]) -> IndexSet {
    IndexSet::from_indices(actions.iter().map(|action| action.index()))
}

/// The full body, which is what the frequency-matched control uses.
pub fn full_body() -> IndexSet {
    body(&Action::MOVEMENTS)
}

/// The whole card as a finite set of cases.
///
/// The two witness kinds share their bodies with each other case by case, so a
/// witness pair differs in the goal alone. That is the same frozen-history
/// construction card 04 uses, applied to the body instead of the history: if the
/// body moved too, a change of behaviour would not be attributable to
/// reachability.
pub fn card_cases() -> Vec<Case> {
    let forward = body(&[Action::Step, Action::Leap]);
    let backward = body(&[Action::Back]);
    let mut cases = Vec::new();

    // A body without `Back` cannot reach `-1` in two steps; a capable one does
    // it in one. A body with only `Back` cannot reach `+2`.
    cases.push(Case {
        kind: CaseKind::WitnessUnreachableFallback,
        contract: Contract::new(0, RING - 1, forward),
    });
    cases.push(Case {
        kind: CaseKind::WitnessUnreachableFallback,
        contract: Contract::new(0, 2, backward),
    });

    // The same two bodies, goals they can reach. `+3` is `Step` then `Leap`;
    // `-2` is `Back` twice.
    cases.push(Case {
        kind: CaseKind::WitnessReachableAttempt,
        contract: Contract::new(0, 3, forward),
    });
    cases.push(Case {
        kind: CaseKind::WitnessReachableAttempt,
        contract: Contract::new(0, RING - 2, backward),
    });

    // Announced restoration: unreachable at the first decision, reachable after.
    // The restored actuator has to be one that actually makes the goal
    // reachable in the remaining budget. `Step` does not: from the start, a
    // single `+1` at the second decision cannot reach `+2`. The enumeration
    // caught that — the case was silently a second unreachable-fallback witness
    // with a decorative announcement.
    for (goal, support, restored) in [
        (RING - 1, forward, Action::Back),
        (2, backward, Action::Leap),
    ] {
        cases.push(Case {
            kind: CaseKind::WitnessRestore,
            contract: Contract::new(0, goal, support).with_restore(Restore {
                actuator: restored.index(),
                after_step: 0,
            }),
        });
    }

    // The same contracts without the announcement.
    for (goal, support) in [(RING - 1, forward), (2, backward)] {
        cases.push(Case {
            kind: CaseKind::NegativeNoRestore,
            contract: Contract::new(0, goal, support),
        });
    }

    // A fully capable body: two goals it cannot reach on the budget alone and
    // two it can, matching the witness family's fallback fraction exactly.
    for goal in [5usize, 6] {
        cases.push(Case {
            kind: CaseKind::NegativeFrequencyMatched,
            contract: Contract::new(0, goal, full_body()),
        });
    }
    for goal in [1usize, 3] {
        cases.push(Case {
            kind: CaseKind::NegativeFrequencyMatched,
            contract: Contract::new(0, goal, full_body()),
        });
    }

    cases
}

/// The environment-limited twin of a body-limited contract.
///
/// The body is made fully capable and the withheld actuator's edges are deleted
/// at every cell the scored phase can start a command from, which leaves the
/// reachable set identical. The two are publicly *distinguishable* — calibration
/// walks past the deleted region and sees the difference — and behaviourally
/// *equivalent*, which is exactly the pair the invariance claim needs. A
/// transformation that changed nothing publicly would be testing nothing.
pub fn body_environment_swap(contract: &Contract) -> Contract {
    let withheld: Vec<Action> = Action::MOVEMENTS
        .into_iter()
        .filter(|action| !contract.support.contains(action.index()))
        .collect();
    let reach = scored_reachable_cells(contract);
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for action in &withheld {
        for cell in &reach {
            edges.push((*cell, action.index()));
        }
    }
    Contract {
        support: full_body(),
        ..contract.clone()
    }
    .with_blocked_edges(&edges)
}

/// Every cell a scored command can be issued from within the budget.
fn scored_reachable_cells(contract: &Contract) -> Vec<usize> {
    let mut cells = vec![contract.start];
    for sequence in all_sequences() {
        let mut cell = contract.start;
        for (executed, action) in sequence.iter().copied().enumerate() {
            if action == Action::Fallback {
                break;
            }
            cell = contract.advance(cell, executed, action);
            if !cells.contains(&cell) {
                cells.push(cell);
            }
        }
    }
    cells.sort_unstable();
    cells
}

/// Public information available to a policy at one scored decision.
pub struct PublicView<'a> {
    pub contract: &'a Contract,
    pub cell: usize,
    pub executed: usize,
}

impl PublicView<'_> {
    /// What calibration established about the body, as a restriction.
    ///
    /// A policy reads this rather than the contract's support field. When
    /// calibration is uninformative it returns nothing, and a policy that
    /// needed it has to say what it does without it.
    pub fn identified_body(&self) -> Option<Restriction> {
        match self.contract.calibration {
            Calibration::Full => Some(self.contract.body()),
            Calibration::Uninformative => None,
        }
    }
}

pub trait PublicPolicy {
    fn name(&self) -> &'static str;
    fn act(&self, view: &PublicView<'_>) -> Action;
}

/// Roll a policy forward over the scored phase and score it.
pub fn run_policy<P: PublicPolicy>(contract: &Contract, policy: &P) -> Outcome {
    let mut cell = contract.start;
    let mut actions = Vec::with_capacity(HORIZON);
    for executed in 0..HORIZON {
        let action = policy.act(&PublicView {
            contract,
            cell,
            executed,
        });
        actions.push(action);
        if action == Action::Fallback {
            break;
        }
        cell = contract.advance(cell, executed, action);
    }
    while actions.len() < HORIZON {
        actions.push(Action::Hold);
    }
    run(contract, &actions)
}

/// The public ceiling as a policy: re-solves the contract exactly at each step.
///
/// It reads the identified body and the announced restoration, both of which are
/// public after calibration, and nothing else.
pub struct ExactPostCalibration;

impl PublicPolicy for ExactPostCalibration {
    fn name(&self) -> &'static str {
        "exact_post_calibration"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        let probe = view.contract.resolved_from(view.cell, view.executed);
        let remaining = HORIZON.saturating_sub(view.executed);
        // A fallback taken later is worth less, so a policy re-solving from a
        // later step must be scored on the sub-episode it actually faces.
        let (_, optimal) = value_bounds_over(&probe, remaining);
        optimal
            .first()
            .and_then(|sequence| sequence.first().copied())
            .unwrap_or(Action::Hold)
    }
}

/// Never falls back; drives toward the goal on the shortest nominal route.
pub struct AlwaysAttempt;

impl PublicPolicy for AlwaysAttempt {
    fn name(&self) -> &'static str {
        "always_attempt"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        nominal_step_toward(view.cell, view.contract.goal)
    }
}

/// Plans as though every actuator were supported and every edge present.
///
/// It therefore falls back exactly when the goal exceeds the budget for a fully
/// capable body, which is optimal on the frequency-matched control and wrong on
/// the witness. That is the pairing the bracket reads.
pub struct IgnoreSupport;

impl PublicPolicy for IgnoreSupport {
    fn name(&self) -> &'static str {
        "ignore_support"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        let capable = Contract {
            start: view.cell,
            support: full_body(),
            blocked_edges: Vec::new(),
            restore: None,
            ..view.contract.clone()
        };
        let remaining = HORIZON.saturating_sub(view.executed);
        let (_, optimal) = value_bounds_over(&capable, remaining);
        optimal
            .first()
            .and_then(|sequence| sequence.first().copied())
            .unwrap_or(Action::Hold)
    }
}

/// Attempts once, then falls back whatever happened.
pub struct TryOnceThenFallback;

impl PublicPolicy for TryOnceThenFallback {
    fn name(&self) -> &'static str {
        "try_once_then_fallback"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        if view.executed == 0 {
            nominal_step_toward(view.cell, view.contract.goal)
        } else {
            Action::Fallback
        }
    }
}

/// Fixes its plan on what calibration showed and ignores any announcement.
///
/// Optimal when nothing is restored, wrong the moment something is, which is
/// what makes the restoration witness attributable.
pub struct PlanAtCalibration;

impl PublicPolicy for PlanAtCalibration {
    fn name(&self) -> &'static str {
        "plan_at_calibration"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        let mut probe = view.contract.resolved_from(view.cell, view.executed);
        // The whole point of this baseline: it never reads the announcement, so
        // a restoration that has already fired is discarded rather than folded
        // in. `resolved_from` would have folded it in, which is why the field is
        // cleared after and not before.
        probe.support = view.contract.support;
        probe.restore = None;
        let remaining = HORIZON.saturating_sub(view.executed);
        let (_, optimal) = value_bounds_over(&probe, remaining);
        optimal
            .first()
            .and_then(|sequence| sequence.first().copied())
            .unwrap_or(Action::Hold)
    }
}

/// The nominal shortest command toward a goal, ignoring support entirely.
fn nominal_step_toward(from: usize, to: usize) -> Action {
    let forward = CONFIGURATION.forward_distance(from, to);
    match forward {
        0 => Action::Hold,
        1 => Action::Step,
        2 => Action::Leap,
        _ if RING - forward == 1 => Action::Back,
        _ if forward <= 2 + 2 => Action::Leap,
        _ => Action::Back,
    }
}

/// Score one policy across every case kind, keeping the kinds separate.
pub fn score_policy<P: PublicPolicy>(policy: &P) -> BTreeMap<String, KindScore> {
    let cases = card_cases();
    let mut scores = BTreeMap::new();
    for kind in CaseKind::ALL {
        let selected: Vec<&Case> = cases.iter().filter(|case| case.kind == kind).collect();
        let mut solved = 0usize;
        let mut optimal = 0usize;
        for case in &selected {
            let outcome = run_policy(&case.contract, policy);
            let (ceiling, _) = value_bounds(&case.contract);
            // "Solved" is doing the right thing, which for an unreachable goal
            // is falling back. A success rate defined as reaching the goal would
            // score the correct policy zero on its own witness.
            if outcome.value == ceiling {
                optimal += 1;
                solved += 1;
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

/// The card's central contrast: with the body fixed, an unreachable goal must
/// produce the fallback on the first scored decision and a reachable one must
/// not.
pub fn allocation_contrast<P: PublicPolicy>(policy: &P) -> bool {
    let cases = card_cases();
    let pairs: Vec<(&Case, &Case)> = cases
        .iter()
        .filter(|case| case.kind == CaseKind::WitnessUnreachableFallback)
        .filter_map(|unreachable| {
            cases
                .iter()
                .find(|other| {
                    other.kind == CaseKind::WitnessReachableAttempt
                        && other.contract.support == unreachable.contract.support
                })
                .map(|reachable| (unreachable, reachable))
        })
        .collect();
    if pairs.is_empty() {
        return false;
    }
    pairs.into_iter().all(|(unreachable, reachable)| {
        let first = |case: &Case| {
            policy.act(&PublicView {
                contract: &case.contract,
                cell: case.contract.start,
                executed: 0,
            })
        };
        first(unreachable) == Action::Fallback && first(reachable) != Action::Fallback
    })
}

/// Which kernel constructs this card composes.
pub fn kernel_use() -> KernelUse {
    KernelUse {
        directed_wiring: true,
        shared_coupling: false,
        interrupt: false,
        restrict: true,
        reveal: card_cases()
            .iter()
            .any(|case| case.contract.restore.is_some()),
        norm_algebra: false,
    }
}

/// Move a contract through one ring rotation.
///
/// Only rotations. See the module note: a reflection exchanges `+1` with `-1`
/// but has no image for `+2`, so it is not a symmetry of this body and is
/// checked as a meaning-changing transformation instead.
pub fn rotate(contract: &Contract, symmetry: Symmetry) -> Contract {
    let map = |cell: usize| symmetry.apply(cell);
    Contract {
        start: map(contract.start),
        goal: map(contract.goal),
        support: contract.support,
        blocked_edges: {
            let mut edges: Vec<(usize, usize)> = contract
                .blocked_edges
                .iter()
                .map(|(cell, action)| (map(*cell), *action))
                .collect();
            edges.sort_unstable();
            edges
        },
        restore: contract.restore,
        calibration: contract.calibration,
    }
}

/// A stable hash of the contract set.
pub fn contract_hash() -> u64 {
    let mut hasher = ContractHasher::new();
    hasher
        .absorb(RING as u64)
        .absorb(HORIZON as u64)
        .absorb(GOAL_REWARD as u64)
        .absorb(MOVE_COST as u64)
        .absorb(FALLBACK_REWARD as u64);
    for case in card_cases() {
        hasher
            .absorb(case.kind as u64)
            .absorb(case.contract.start as u64)
            .absorb(case.contract.goal as u64)
            .absorb(u64::from(case.contract.support.0))
            .absorb(case.contract.calibration as u64);
        for (cell, action) in &case.contract.blocked_edges {
            hasher.absorb((*cell as u64) << 8 | *action as u64);
        }
        hasher.absorb_option(
            case.contract
                .restore
                .map(|restore| (restore.actuator as u64) << 16 | restore.after_step as u64),
        );
    }
    hasher.finish()
}

/// Recover an action from its actuator index, for decoding a rendered episode.
pub fn action_from_index(index: usize) -> Option<Action> {
    Action::from_index(index)
}
