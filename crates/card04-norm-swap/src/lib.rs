//! Card 04 — Norm Swap, built as an audited world.
//!
//! Behaviour is a function of the *relation* between the current situation and
//! the requested outcome, not of the situation alone. The card splits that into
//! three sub-claims that are deliberately never collapsed into one executive
//! score: `M4` goal conditioning, `M10` maintain/inhibit/switch, and `M12`
//! viability.
//!
//! Everything here is exact by exhaustive enumeration over `3^3 = 27` action
//! sequences per case. No learner exists in this crate and none is run. The
//! card has no epistemic content by design: every port is public, so the public
//! and privileged ceilings coincide and the ambiguity gap is zero everywhere.
//! That is what makes any gap between a learner and the ceiling unambiguously
//! skill rather than information.

mod audit;
mod render;
pub use audit::*;
pub use render::*;

use std::collections::BTreeMap;

use pretraining_g0_contract::{
    BoundaryEffect, ContractHasher, Fragment, Guard, GuardContext, IndexSet, KernelUse, Norm,
    Restriction, Reveal, Ring,
};
use serde::{Deserialize, Serialize};

pub use pretraining_g0_contract::{BracketStructure, Isolation, KindScore, OrbitVerdict, Symmetry};

/// Cells are arranged in a ring so that no cell is an edge. A ring is chosen
/// over a line because it gives every goal two routes of different length,
/// which is what lets a forbidden greedy action have a correct alternative.
///
/// Five, not six, and the difference is not cosmetic. With six cells a goal two
/// steps along has a four-step detour, which does not fit the three-step
/// horizon: blocking the short route made the inhibition and viability
/// witnesses *unreachable*, so their ceiling was zero and the exact policy
/// scored zero on its own card. Five cells make the detour exactly three steps,
/// so the prohibition costs a step without making the goal impossible. The
/// enumeration found this; nothing in the card text says it.
pub const RING: usize = 5;

/// The shared configuration structure. Card 03 reuses *this object*, not a copy
/// of these numbers, which is what keeps a later cross-card claim from being
/// confounded with world difficulty.
pub const CONFIGURATION: Ring = Ring::new(RING);

/// Three steps. The action support is `3^3 = 27` sequences per case, so every
/// baseline, ceiling, and optimal set below is total rather than sampled.
pub const HORIZON: usize = 3;

/// A blocked short route must leave an alternative that fits the horizon.
///
/// Checked at compile time rather than in a test, because it is a property of
/// the constants themselves and a six-cell ring violated it silently: the
/// detour did not fit, the inhibition and viability witnesses became
/// unreachable, and their ceiling was zero. Resizing the ring now fails the
/// build.
const _: () = assert!(RING - 2 <= HORIZON);

pub const GOAL_REWARD: i32 = 100;
pub const MOVE_COST: i32 = 1;

/// The body is published and stable. This card holds identification at zero on
/// purpose, so a goal effect can never be confounded with an embodiment effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Action {
    Retreat,
    Hold,
    Advance,
}

impl Action {
    pub const ALL: [Self; 3] = [Self::Retreat, Self::Hold, Self::Advance];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Retreat => "retreat",
            Self::Hold => "hold",
            Self::Advance => "advance",
        }
    }

    fn apply(self, cell: usize) -> usize {
        match self {
            Self::Retreat => CONFIGURATION.retreat(cell),
            Self::Hold => cell,
            Self::Advance => CONFIGURATION.advance(cell),
        }
    }
}

/// How a second published goal relates to the first.
///
/// The two readings produce identical public histories up to the switch and
/// completely different correct behaviour afterwards, which is why the card
/// calls this the cheapest available test of whether a learner read the norm's
/// structure rather than its most recent value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwitchMode {
    /// Only the second goal matters at the end.
    Supersede,
    /// The first goal must have been visited, and the second must hold at the end.
    Compose,
}

/// Whether entering the hazard resets the episode or ends it.
///
/// The optimal policy avoids the hazard under both, so the privileged ceiling is
/// identical. That identity is the enumerable content of the `M12` variant: any
/// learner degradation under `Absorbing` cannot be attributed to the ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HazardKind {
    Reset,
    Absorbing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Switch {
    /// Number of executed actions after which the second goal is published.
    pub after_step: usize,
    pub goal: usize,
    pub mode: SwitchMode,
    /// When true the second goal is published at episode start instead, so a
    /// policy that plans once and executes blindly can still succeed.
    pub announced: bool,
}

/// One fully published episode contract.
///
/// Nothing in it is hidden. `identify` is trivial and unused by this card.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contract {
    pub start: usize,
    pub goal: usize,
    pub no_go: Option<usize>,
    pub hazard: Option<(usize, HazardKind)>,
    /// Step index after which a revealed and irrelevant event occurs.
    pub distractor_after: Option<usize>,
    pub switch: Option<Switch>,
}

impl Contract {
    pub fn new(start: usize, goal: usize) -> Self {
        Self {
            start,
            goal,
            no_go: None,
            hazard: None,
            distractor_after: None,
            switch: None,
        }
    }

    pub fn with_no_go(mut self, cell: usize) -> Self {
        self.no_go = Some(cell);
        self
    }

    pub fn with_hazard(mut self, cell: usize, kind: HazardKind) -> Self {
        self.hazard = Some((cell, kind));
        self
    }

    pub fn with_distractor(mut self, after: usize) -> Self {
        self.distractor_after = Some(after);
        self
    }

    pub fn with_switch(mut self, switch: Switch) -> Self {
        self.switch = Some(switch);
        self
    }

    /// The goal in force after `executed` actions.
    pub fn active_goal(&self, executed: usize) -> usize {
        match self.switch {
            Some(switch) if executed > switch.after_step => switch.goal,
            _ => self.goal,
        }
    }

    /// What a policy is entitled to know before choosing its `executed`-th
    /// action. An announced switch is visible from the start; an unannounced
    /// one only after it fires.
    pub fn published_switch(&self, executed: usize) -> Option<Switch> {
        match self.switch {
            Some(switch) if switch.announced || executed > switch.after_step => Some(switch),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub value: i32,
    pub solved: bool,
    pub violated_no_go: bool,
    pub absorbed: bool,
    pub final_cell: usize,
}

/// The card's norm, composed from the shared algebra.
///
/// Every connective this card claims in `EMBODIED-PROCESS.md`'s coverage table
/// appears here as a real operator rather than as a branch inside an evaluator:
/// supersession carries the mid-episode goal event, conjunction carries the
/// composing reading of that event, and priority carries the prohibition and
/// the viability boundary. `run` evaluates *this*, so a later edit that changes
/// the card's meaning has to change a norm term.
pub fn norm_of(contract: &Contract) -> Norm {
    let outcome = match contract.switch {
        Some(switch) => {
            let after = match switch.mode {
                SwitchMode::Supersede => Norm::Settle { cell: switch.goal },
                // Composition is conjunction, not replacement: the first goal
                // must still have been visited.
                SwitchMode::Compose => Norm::both(
                    Norm::Visit {
                        cell: contract.goal,
                    },
                    Norm::Settle { cell: switch.goal },
                ),
            };
            Norm::supersede(
                Norm::Settle {
                    cell: contract.goal,
                },
                after,
                Guard::AfterStep(switch.after_step),
            )
        }
        None => Norm::Settle {
            cell: contract.goal,
        },
    };
    // Priority, not conjunction. A prohibition breached cannot be offset by
    // reaching the goal, and the two connectives differ exactly there.
    let with_prohibition = match contract.no_go {
        Some(cell) => Norm::priority(Norm::Avoid { cell }, outcome),
        None => outcome,
    };
    match contract.hazard {
        // A reset boundary is a dynamics fact and carries no norm term: the
        // configuration is returned to the start and the episode continues.
        Some((cell, HazardKind::Absorbing)) => {
            Norm::priority(Norm::Avoid { cell }, with_prohibition)
        }
        _ => with_prohibition,
    }
}

/// The viability restriction a hazard imposes on the configuration space.
pub fn viability_of(contract: &Contract) -> Option<Restriction> {
    contract.hazard.map(|(cell, kind)| Restriction::Viability {
        inadmissible: IndexSet::from_indices([cell]),
        effect: match kind {
            HazardKind::Absorbing => BoundaryEffect::Absorbing,
            HazardKind::Reset => BoundaryEffect::Reset,
        },
    })
}

/// The reveal that publishes a second goal, and the guard it fires on.
pub fn reveal_of(contract: &Contract) -> Option<Reveal<usize>> {
    contract.switch.map(|switch| {
        Reveal::new(
            if switch.announced {
                Guard::AtStart
            } else {
                Guard::AfterStep(switch.after_step)
            },
            switch.goal,
        )
    })
}

/// Which kernel constructs this card actually composes.
///
/// Reported from the case set rather than declared, so the coverage table in
/// `EMBODIED-PROCESS.md` can be checked against the code instead of trusted.
pub fn kernel_use() -> KernelUse {
    KernelUse {
        directed_wiring: true,
        // The unannounced goal event displaces the running pursuit; the body
        // resumes from the cell it had reached rather than restarting, which is
        // why `interrupt` and not only the norm algebra is claimed.
        interrupt: card_cases()
            .iter()
            .any(|case| case.contract.switch.is_some()),
        restrict: card_cases()
            .iter()
            .any(|case| case.contract.hazard.is_some()),
        reveal: card_cases()
            .iter()
            .any(|case| case.contract.switch.is_some()),
        norm_algebra: true,
        shared_coupling: false,
    }
}

/// Walk the configuration forward under the viability restriction.
fn walk(contract: &Contract, actions: &[Action]) -> (Vec<usize>, bool, bool) {
    let viability = viability_of(contract);
    let mut cell = contract.start;
    let mut trajectory = vec![cell];
    let mut violated = false;
    let mut absorbed = false;
    for action in actions.iter().copied() {
        if absorbed {
            trajectory.push(cell);
            continue;
        }
        let next = action.apply(cell);
        if Some(next) == contract.no_go {
            violated = true;
        }
        match viability.as_ref() {
            Some(restriction) if !restriction.admits_cell(next) => {
                match restriction.boundary_effect() {
                    Some(BoundaryEffect::Absorbing) => {
                        cell = next;
                        absorbed = true;
                    }
                    Some(BoundaryEffect::Reset) => cell = contract.start,
                    None => cell = next,
                }
            }
            _ => cell = next,
        }
        trajectory.push(cell);
    }
    (trajectory, violated, absorbed)
}

/// Execute a complete action sequence against a contract.
///
/// Value is `GOAL_REWARD` minus one `MOVE_COST` per step before the
/// configuration settles, and zero whenever the norm was broken. Success and
/// value are returned separately: a binary success rate and a cost-aware
/// contract value answer different questions and are never merged into one
/// number.
///
/// Cost counts steps before settling rather than moves made. Charging per move
/// would let a policy that waits and then goes tie with one that goes
/// immediately, which blunts the paired first-action contrast the card rests on.
pub fn run(contract: &Contract, actions: &[Action]) -> Outcome {
    let (trajectory, violated, absorbed) = walk(contract, actions);
    let final_cell = *trajectory.last().expect("a trajectory contains its start");
    let verdict = norm_of(contract).evaluate(
        &trajectory,
        GuardContext {
            executed: actions.len(),
            last_action: None,
            cell: final_cell,
        },
    );
    let value = match verdict.settle_steps {
        Some(steps) if verdict.met => GOAL_REWARD - MOVE_COST * steps as i32,
        _ => 0,
    };

    Outcome {
        value,
        solved: verdict.met,
        violated_no_go: violated,
        absorbed,
        final_cell,
    }
}

/// The card as a fragment: this is what makes it exhaustively auditable by the
/// shared machinery rather than by its own copy of an enumerator.
pub struct NormSwap;

impl Fragment for NormSwap {
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

    fn step(&self, contract: &Contract, cell: usize, action: Action) -> usize {
        let next = action.apply(cell);
        match contract.hazard {
            Some((hazard, HazardKind::Absorbing)) if cell == hazard => hazard,
            Some((hazard, HazardKind::Absorbing)) if next == hazard => hazard,
            Some((hazard, HazardKind::Reset)) if next == hazard => contract.start,
            _ => next,
        }
    }

    fn value(&self, contract: &Contract, _trajectory: &[usize], actions: &[Action]) -> i32 {
        run(contract, actions).value
    }
}

/// Every action sequence of length `HORIZON`.
pub fn all_sequences() -> Vec<Vec<Action>> {
    pretraining_g0_contract::sequences_of_length(&Action::ALL, HORIZON)
}

/// The exact ceiling and every sequence achieving it.
///
/// Because every port of this card is public, this is simultaneously the public
/// and the privileged ceiling. The ambiguity gap is therefore zero by
/// construction, and [`ambiguity_gap`] checks that rather than assuming it.
pub fn value_bounds(contract: &Contract) -> (i32, Vec<Vec<Action>>) {
    pretraining_g0_contract::value_bounds(&NormSwap, contract)
}

/// The ceiling over a stated remaining horizon.
pub fn value_bounds_over(contract: &Contract, horizon: usize) -> (i32, Vec<Vec<Action>>) {
    pretraining_g0_contract::value_bounds_over(&NormSwap, contract, horizon)
}

/// The optimal first actions, which is what the paired contrast reads.
pub fn optimal_first_actions(contract: &Contract) -> Vec<Action> {
    pretraining_g0_contract::optimal_first_actions(&NormSwap, contract)
}

/// The ambiguity gap, which this card asserts is zero everywhere.
pub fn ambiguity_gap(contract: &Contract) -> i32 {
    pretraining_g0_contract::ambiguity_gap(&NormSwap, contract)
}

/// Public information available to a policy at one decision point.
pub struct PublicView<'a> {
    pub contract: &'a Contract,
    pub cell: usize,
    pub executed: usize,
}

impl PublicView<'_> {
    /// The goal a policy can currently see published.
    pub fn visible_goal(&self) -> usize {
        match self.contract.published_switch(self.executed) {
            Some(switch) if switch.announced && self.executed <= switch.after_step => {
                self.contract.goal
            }
            Some(switch) => switch.goal,
            None => self.contract.goal,
        }
    }
}

pub trait PublicPolicy {
    fn name(&self) -> &'static str;
    fn act(&self, view: &PublicView<'_>) -> Action;
}

/// Roll a policy forward and score it. The policy never sees a hidden field,
/// because this card has none.
pub fn run_policy<P: PublicPolicy>(contract: &Contract, policy: &P) -> Outcome {
    let mut cell = contract.start;
    let mut actions = Vec::with_capacity(HORIZON);
    for executed in 0..HORIZON {
        let view = PublicView {
            contract,
            cell,
            executed,
        };
        let action = policy.act(&view);
        actions.push(action);
        cell = action.apply(cell);
        if let Some((hazard_cell, HazardKind::Absorbing)) = contract.hazard {
            if cell == hazard_cell {
                break;
            }
        }
        if let Some((hazard_cell, HazardKind::Reset)) = contract.hazard {
            if cell == hazard_cell {
                cell = contract.start;
            }
        }
    }
    while actions.len() < HORIZON {
        actions.push(Action::Hold);
    }
    run(contract, &actions)
}

fn ring_step_toward(from: usize, to: usize) -> Action {
    if from == to {
        return Action::Hold;
    }
    let forward = (to + RING - from) % RING;
    let backward = (from + RING - to) % RING;
    if forward <= backward {
        Action::Advance
    } else {
        Action::Retreat
    }
}

/// The public ceiling as a policy: re-solves the contract exactly at every step.
pub struct GoalConditionedExact;

impl PublicPolicy for GoalConditionedExact {
    fn name(&self) -> &'static str {
        "goal_conditioned_exact"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        let mut probe = view.contract.clone();
        probe.start = view.cell;
        // Re-solving from the current cell requires shifting any pending switch
        // into the remaining horizon rather than the original one.
        if let Some(mut switch) = probe.switch {
            switch.after_step = switch.after_step.saturating_sub(view.executed);
            probe.switch = Some(switch);
        }
        let remaining = HORIZON.saturating_sub(view.executed);
        let (_, optimal) = value_bounds_over(&probe, remaining);
        optimal
            .first()
            .and_then(|s| s.first().copied())
            .unwrap_or(Action::Hold)
    }
}

/// Maps configuration to action and never reads the goal channel.
///
/// It is parameterized by the goal it assumes from the starting cell, which is
/// what makes it optimal on the single-goal and goal-predictable negatives and
/// unable to pass the witness.
pub struct StateOnly {
    pub assumed_goal: [usize; RING],
}

impl PublicPolicy for StateOnly {
    fn name(&self) -> &'static str {
        "state_only"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        let target = self.assumed_goal[view.contract.start];
        let candidate = ring_step_toward(view.cell, target);
        avoid_forbidden(view, candidate, target)
    }
}

/// Acts toward the most recently published goal and reads no norm structure.
pub struct LastGoal;

impl PublicPolicy for LastGoal {
    fn name(&self) -> &'static str {
        "last_goal"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        let target = view.visible_goal();
        let candidate = ring_step_toward(view.cell, target);
        avoid_forbidden(view, candidate, target)
    }
}

/// Reduces distance to the active goal and ignores the no-go and the hazard.
pub struct GreedyProgress;

impl PublicPolicy for GreedyProgress {
    fn name(&self) -> &'static str {
        "greedy_progress"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        ring_step_toward(view.cell, view.visible_goal())
    }
}

/// Plans once at episode start from what is published then, and executes the
/// plan blindly. It succeeds when a switch is announced in advance and fails
/// when the same switch arrives mid-episode.
pub struct PlanOnce;

impl PublicPolicy for PlanOnce {
    fn name(&self) -> &'static str {
        "plan_once"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        let mut probe = view.contract.clone();
        // What is knowable at episode start: an unannounced switch is not.
        if let Some(switch) = probe.switch {
            if !switch.announced {
                probe.switch = None;
            }
        }
        let (_, optimal) = value_bounds(&probe);
        optimal
            .first()
            .and_then(|s| s.get(view.executed).copied())
            .unwrap_or(Action::Hold)
    }
}

/// Steer around a published prohibition when the greedy step would enter it.
fn avoid_forbidden(view: &PublicView<'_>, candidate: Action, target: usize) -> Action {
    let enters_forbidden = |action: Action| {
        let next = action.apply(view.cell);
        Some(next) == view.contract.no_go
            || view.contract.hazard.map(|(cell, _)| cell) == Some(next)
    };
    if !enters_forbidden(candidate) {
        return candidate;
    }
    Action::ALL
        .into_iter()
        .filter(|action| !enters_forbidden(*action))
        .min_by_key(|action| {
            let next = action.apply(view.cell);
            let forward = (target + RING - next) % RING;
            let backward = (next + RING - target) % RING;
            forward.min(backward)
        })
        .unwrap_or(Action::Hold)
}

/// Which sub-claim or negative a case belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CaseKind {
    /// `M4`: frozen public history, different goal, action must change.
    WitnessGoalConditioning,
    /// `M10` maintenance: a revealed and irrelevant distractor event.
    WitnessMaintain,
    /// `M10` inhibition: the greedy action is forbidden.
    WitnessInhibit,
    /// `M10` switching: a second goal arrives mid-episode.
    WitnessSwitch,
    /// `M12`: an absorbing boundary next to the direct route.
    WitnessViability,
    /// 5.1 — the goal is constant across the family.
    NegativeSingleGoal,
    /// 5.2 — the goal is a deterministic function of the starting cell.
    NegativeGoalPredictable,
    /// 5.3 — the prohibition is lifted, so the greedy action becomes correct.
    NegativeNoGoRemoved,
    /// 5.4 — the switch is published at episode start.
    NegativeSwitchAnnounced,
}

impl CaseKind {
    pub const ALL: [Self; 9] = [
        Self::WitnessGoalConditioning,
        Self::WitnessMaintain,
        Self::WitnessInhibit,
        Self::WitnessSwitch,
        Self::WitnessViability,
        Self::NegativeSingleGoal,
        Self::NegativeGoalPredictable,
        Self::NegativeNoGoRemoved,
        Self::NegativeSwitchAnnounced,
    ];

    pub const NEGATIVES: [Self; 4] = [
        Self::NegativeSingleGoal,
        Self::NegativeGoalPredictable,
        Self::NegativeNoGoRemoved,
        Self::NegativeSwitchAnnounced,
    ];

    pub const fn is_witness(self) -> bool {
        matches!(
            self,
            Self::WitnessGoalConditioning
                | Self::WitnessMaintain
                | Self::WitnessInhibit
                | Self::WitnessSwitch
                | Self::WitnessViability
        )
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::WitnessGoalConditioning => "witness_goal_conditioning",
            Self::WitnessMaintain => "witness_maintain",
            Self::WitnessInhibit => "witness_inhibit",
            Self::WitnessSwitch => "witness_switch",
            Self::WitnessViability => "witness_viability",
            Self::NegativeSingleGoal => "negative_single_goal",
            Self::NegativeGoalPredictable => "negative_goal_predictable",
            Self::NegativeNoGoRemoved => "negative_no_go_removed",
            Self::NegativeSwitchAnnounced => "negative_switch_announced",
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
/// The frozen-history construction is structural rather than sampled: the two
/// goal-conditioning cases are built from one base contract with only the goal
/// field replaced, so nothing but the goal can differ. The card warns that a
/// shared random draw would void the contrast, and this construction consumes
/// no randomness at all.
pub fn card_cases() -> Vec<Case> {
    let mut cases = Vec::new();

    // M4 — frozen history, opposite goals, start at cell 0.
    let base = Contract::new(0, 2);
    cases.push(Case {
        kind: CaseKind::WitnessGoalConditioning,
        contract: Contract {
            goal: 2,
            ..base.clone()
        },
    });
    cases.push(Case {
        kind: CaseKind::WitnessGoalConditioning,
        contract: Contract {
            goal: 3,
            ..base.clone()
        },
    });

    // M10 maintenance — the distractor is revealed and changes nothing.
    for goal in [2usize, 3] {
        cases.push(Case {
            kind: CaseKind::WitnessMaintain,
            contract: Contract::new(0, goal).with_distractor(1),
        });
    }

    // M10 inhibition — goal two steps advance, prohibition on the first advance
    // cell, so the greedy action is correct-looking and wrong.
    cases.push(Case {
        kind: CaseKind::WitnessInhibit,
        contract: Contract::new(0, 2).with_no_go(1),
    });
    cases.push(Case {
        kind: CaseKind::WitnessInhibit,
        contract: Contract::new(0, 3).with_no_go(4),
    });

    // M10 switching — unannounced second goal after one executed action.
    for (first, second, mode) in [
        (2usize, 3usize, SwitchMode::Supersede),
        (3, 2, SwitchMode::Supersede),
    ] {
        cases.push(Case {
            kind: CaseKind::WitnessSwitch,
            contract: Contract::new(0, first).with_switch(Switch {
                after_step: 0,
                goal: second,
                mode,
                announced: false,
            }),
        });
    }

    // M12 viability — absorbing hazard adjacent to the direct route.
    cases.push(Case {
        kind: CaseKind::WitnessViability,
        contract: Contract::new(0, 2).with_hazard(1, HazardKind::Absorbing),
    });
    cases.push(Case {
        kind: CaseKind::WitnessViability,
        contract: Contract::new(0, 3).with_hazard(4, HazardKind::Absorbing),
    });

    // 5.1 — one constant goal for the whole family.
    for start in [0usize, 1, 4] {
        cases.push(Case {
            kind: CaseKind::NegativeSingleGoal,
            contract: Contract::new(start, 2),
        });
    }

    // 5.2 — the goal is a deterministic function of the starting cell.
    for start in [0usize, 1, 4] {
        cases.push(Case {
            kind: CaseKind::NegativeGoalPredictable,
            contract: Contract::new(start, predictable_goal(start)),
        });
    }

    // 5.3 — the inhibition cases with the prohibition lifted.
    cases.push(Case {
        kind: CaseKind::NegativeNoGoRemoved,
        contract: Contract::new(0, 2),
    });
    cases.push(Case {
        kind: CaseKind::NegativeNoGoRemoved,
        contract: Contract::new(0, 3),
    });

    // 5.4 — the switch cases with the second goal announced at episode start.
    for (first, second) in [(2usize, 3usize), (3, 2)] {
        cases.push(Case {
            kind: CaseKind::NegativeSwitchAnnounced,
            contract: Contract::new(0, first).with_switch(Switch {
                after_step: 0,
                goal: second,
                mode: SwitchMode::Supersede,
                announced: true,
            }),
        });
    }

    cases
}

/// The deterministic goal map used by negative 5.2.
pub fn predictable_goal(start: usize) -> usize {
    (start + 2) % RING
}

/// The best configuration-to-goal map for one case family.
///
/// A state-only policy is necessarily *family-relative*. Negatives 5.1 and 5.2
/// share starting cells and demand different goals at them, so no single
/// configuration-to-goal map is optimal on both. The card reads as though one
/// state-only policy covers both; enumeration says otherwise, and the honest
/// construction is one map per family.
pub fn state_only_for(kind: CaseKind) -> StateOnly {
    let mut assumed = [0usize; RING];
    for (start, slot) in assumed.iter_mut().enumerate() {
        let observed = card_cases()
            .into_iter()
            .find(|case| case.kind == kind && case.contract.start == start)
            .map(|case| case.contract.goal);
        *slot = observed.unwrap_or_else(|| predictable_goal(start));
    }
    StateOnly {
        assumed_goal: assumed,
    }
}

/// The witness-facing state-only baseline. It must commit to one goal at the
/// witness start cell and therefore cannot satisfy both goal-conditioning cases.
pub fn state_only_baseline() -> StateOnly {
    state_only_for(CaseKind::NegativeGoalPredictable)
}

/// Score one policy across every case kind, keeping the kinds separate.
pub fn score_policy<P: PublicPolicy>(policy: &P) -> BTreeMap<String, KindScore> {
    let cases = card_cases();
    let mut scores: BTreeMap<String, KindScore> = BTreeMap::new();
    for kind in CaseKind::ALL {
        let selected: Vec<&Case> = cases.iter().filter(|case| case.kind == kind).collect();
        let mut solved = 0usize;
        let mut optimal = 0usize;
        for case in &selected {
            let outcome = run_policy(&case.contract, policy);
            if outcome.solved {
                solved += 1;
            }
            let (ceiling, _) = value_bounds(&case.contract);
            if outcome.value == ceiling {
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

/// The paired goal-conditioning contrast: the first action must be correct in
/// both frozen-history cases and must differ between them.
pub fn goal_conditioning_contrast<P: PublicPolicy>(policy: &P) -> bool {
    let cases: Vec<Case> = card_cases()
        .into_iter()
        .filter(|case| case.kind == CaseKind::WitnessGoalConditioning)
        .collect();
    if cases.len() != 2 {
        return false;
    }
    let mut firsts = Vec::new();
    for case in &cases {
        let view = PublicView {
            contract: &case.contract,
            cell: case.contract.start,
            executed: 0,
        };
        let action = policy.act(&view);
        if !optimal_first_actions(&case.contract).contains(&action) {
            return false;
        }
        firsts.push(action);
    }
    firsts[0] != firsts[1]
}

/// A stable hash of the contract set, so a silent change to the card's meaning
/// is detectable rather than something a reader has to notice.
pub fn contract_hash() -> u64 {
    let mut hasher = ContractHasher::new();
    hasher
        .absorb(RING as u64)
        .absorb(HORIZON as u64)
        .absorb(GOAL_REWARD as u64)
        .absorb(MOVE_COST as u64);
    for case in card_cases() {
        hasher
            .absorb(case.kind as u64)
            .absorb(case.contract.start as u64)
            .absorb(case.contract.goal as u64);
        hasher.absorb_option(case.contract.no_go.map(|c| c as u64));
        hasher.absorb_option(
            case.contract
                .hazard
                .map(|(cell, kind)| (cell as u64) << 8 | kind as u64),
        );
        hasher.absorb_option(case.contract.distractor_after.map(|a| a as u64));
        hasher.absorb_option(case.contract.switch.map(|switch| {
            (switch.after_step as u64) << 24
                | (switch.goal as u64) << 16
                | (switch.mode as u64) << 8
                | switch.announced as u64
        }));
    }
    hasher.finish()
}
