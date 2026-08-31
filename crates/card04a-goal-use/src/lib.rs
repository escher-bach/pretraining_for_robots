//! Card 04 stage A — public goal use, the basic relation under card 04's
//! decomposition certificate.
//!
//! The composite holds public state and history fixed and changes the requested
//! outcome, over a composite of goal maintenance, a prohibition, an unannounced
//! mid-episode supersession, and an absorbing viability boundary. This family
//! keeps only the first: one goal, published at episode start, on a ring with a
//! known body and nothing epistemically withheld. Two episodes that share a
//! start configuration and differ only in the published goal must take
//! different first actions.
//!
//! Everything the composite added on top is gone, and with it every norm
//! connective: the norm here is a single `Settle` leaf. That is deliberate.
//! `EMBODIED-PROCESS.md` makes supersession and priority card 04's central
//! constructs, so a stage that kept them would not be a decomposition of the
//! composite — it would be the composite with fewer cases.
//!
//! # The two controls
//!
//! - **Constant goal.** The published goal never varies and the start does, so
//!   a policy that reads the configuration and ignores the goal channel is
//!   optimal.
//! - **Goal predictable from state.** The goal varies and is a fixed function
//!   of the start, so a policy that ignores both the goal channel *and* the
//!   configuration is optimal.
//!
//! Each removes one reason to read the goal, and each restores a baseline that
//! fails the witness. The witness's state-only ceiling is exactly `0.5`,
//! because every start configuration carries two goals with different correct
//! first actions.
//!
//! # Where the composite's epistemic content went
//!
//! Nowhere: it never had any. Public and privileged information coincide on
//! eighteen of the composite's twenty cases, and the two exceptions are the
//! unannounced switch, which this stage removes. So the ambiguity gap here is
//! zero on every case and *vacuous* — [`Fragment::privileged_value`] is not
//! overridden, so the reported quantity compares the value function with
//! itself. The audit says so rather than presenting a zero gap as evidence.

mod audit;
mod render;
pub use audit::*;
pub use render::*;

use std::collections::BTreeMap;

use pretraining_g0_contract::{
    ambiguity_gap, ContractHasher, Fragment, GuardContext, KernelUse, Norm, PubliclyObservable,
    Ring, Symmetry,
};
use serde::{Deserialize, Serialize};

pub use pretraining_g0_contract::{BracketStructure, Isolation, KindScore, OrbitVerdict};

/// The composite's five-position ring, unchanged.
pub const RING: usize = 5;
pub const CONFIGURATION: Ring = Ring::new(RING);

/// Two decisions rather than the composite's three.
///
/// Three exist in the composite to leave room for a switch to fire and be
/// answered. With the switch removed the third decision would only be a longer
/// hold, which adds budget without adding contrast.
pub const HORIZON: usize = 2;

pub const GOAL_REWARD: i32 = 100;
pub const MOVE_COST: i32 = 1;

/// How far a shifted denotation moves the cell a goal symbol names.
pub const DENOTATION_SHIFT: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Action {
    Retreat,
    Hold,
    Advance,
}

impl Action {
    pub const ALL: [Self; 3] = [Self::Retreat, Self::Hold, Self::Advance];

    pub const fn index(self) -> usize {
        match self {
            Self::Retreat => 0,
            Self::Hold => 1,
            Self::Advance => 2,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Retreat => "retreat",
            Self::Hold => "hold",
            Self::Advance => "advance",
        }
    }

    pub fn apply(self, cell: usize) -> usize {
        match self {
            Self::Retreat => CONFIGURATION.retreat(cell),
            Self::Hold => cell,
            Self::Advance => CONFIGURATION.advance(cell),
        }
    }

    /// The image of this action under a reflection, which exchanges the two
    /// directions of travel.
    pub const fn reversed(self) -> Self {
        match self {
            Self::Retreat => Self::Advance,
            Self::Advance => Self::Retreat,
            Self::Hold => Self::Hold,
        }
    }

    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.into_iter().find(|action| action.index() == index)
    }
}

/// What the published goal symbol denotes.
///
/// A published family parameter, so that changing it is a change to the world
/// *and* to the public trace. Leaving it unpublished would convert a
/// meaning-changing transformation into hidden state, which is precisely what
/// this stage is built not to have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Denotation {
    /// The symbol names the cell to settle on.
    Direct,
    /// The symbol names a cell `DENOTATION_SHIFT` positions away.
    Shifted,
}

impl Denotation {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Shifted => "shifted",
        }
    }
}

/// Which arm of the family a case belongs to. A published family parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Regime {
    /// Goals vary within a start configuration.
    GoalVaries,
    /// One goal for every episode.
    ConstantGoal,
    /// Goals vary and are a fixed function of the start configuration.
    GoalPredictableFromState,
}

impl Regime {
    pub const ALL: [Self; 3] = [
        Self::GoalVaries,
        Self::ConstantGoal,
        Self::GoalPredictableFromState,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::GoalVaries => "goal_varies",
            Self::ConstantGoal => "constant_goal",
            Self::GoalPredictableFromState => "goal_predictable_from_state",
        }
    }
}

/// One fully specified episode contract. Every field is public.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contract {
    pub start: usize,
    /// The published goal symbol. What it denotes depends on `denotation`.
    pub goal: usize,
    pub denotation: Denotation,
    pub regime: Regime,
}

impl Contract {
    pub fn new(start: usize, goal: usize, regime: Regime) -> Self {
        Self {
            start,
            goal,
            denotation: Denotation::Direct,
            regime,
        }
    }

    /// The cell the published goal symbol names.
    pub fn goal_cell(&self) -> usize {
        match self.denotation {
            Denotation::Direct => self.goal,
            Denotation::Shifted => (self.goal + DENOTATION_SHIFT) % RING,
        }
    }

    /// The norm in force: one leaf, no connectives.
    pub fn norm(&self) -> Norm {
        Norm::Settle {
            cell: self.goal_cell(),
        }
    }

    pub fn with_goal(&self, goal: usize) -> Self {
        Self {
            goal: goal % RING,
            ..*self
        }
    }

    pub fn with_denotation(&self, denotation: Denotation) -> Self {
        Self {
            denotation,
            ..*self
        }
    }

    /// Move the whole contract through one ring symmetry.
    ///
    /// Both the start and the goal move, which is what makes this a relabelling
    /// rather than a different task. A reflection additionally exchanges the two
    /// directions of travel, so the action map is not the identity there.
    pub fn relabelled(&self, symmetry: Symmetry) -> Self {
        Self {
            start: symmetry.apply(self.start),
            goal: symmetry.apply(self.goal),
            ..*self
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub value: i32,
    pub solved: bool,
    pub settle_steps: Option<usize>,
    pub final_cell: usize,
}

/// Roll a sequence forward. There is no hazard, no reset, and no absorption.
pub fn walk(contract: &Contract, actions: &[Action]) -> Vec<usize> {
    let mut cell = contract.start;
    let mut trajectory = vec![cell];
    for action in actions.iter().copied() {
        cell = action.apply(cell);
        trajectory.push(cell);
    }
    trajectory
}

/// Execute a complete action sequence against a contract.
///
/// Value is `GOAL_REWARD` minus one `MOVE_COST` per step before the
/// configuration settles, and zero when the norm is unmet. Cost counts steps
/// before settling rather than moves made, exactly as in the composite, so a
/// policy that waits and then goes cannot tie with one that goes at once.
pub fn run(contract: &Contract, actions: &[Action]) -> Outcome {
    let trajectory = walk(contract, actions);
    let final_cell = *trajectory.last().expect("a trajectory contains its start");
    let verdict = contract.norm().evaluate(
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
        settle_steps: verdict.settle_steps,
        final_cell,
    }
}

pub struct GoalUse;

impl Fragment for GoalUse {
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

    fn step(&self, _contract: &Contract, cell: usize, _executed: usize, action: Action) -> usize {
        action.apply(cell)
    }

    fn value(&self, contract: &Contract, _trajectory: &[usize], actions: &[Action]) -> i32 {
        run(contract, actions).value
    }
}

impl PubliclyObservable for GoalUse {
    /// The family parameters, the goal, the start, then the configuration after
    /// each action.
    ///
    /// Every field of the contract is here, which is the point: this family has
    /// no privileged view, and the audit's zero ambiguity gap is a consequence
    /// of that rather than a claim about it.
    fn public_trace(&self, contract: &Contract, actions: &[Action]) -> Vec<i64> {
        let mut trace = vec![
            contract.regime as i64,
            contract.denotation as i64,
            contract.goal as i64,
            contract.start as i64,
        ];
        for cell in walk(contract, actions).into_iter().skip(1) {
            trace.push(cell as i64);
        }
        trace
    }
}

pub fn all_sequences() -> Vec<Vec<Action>> {
    pretraining_g0_contract::sequences_of_length(&Action::ALL, HORIZON)
}

pub fn value_bounds(contract: &Contract) -> (i32, Vec<Vec<Action>>) {
    pretraining_g0_contract::value_bounds(&GoalUse, contract)
}

pub fn optimal_first_actions(contract: &Contract) -> Vec<Action> {
    pretraining_g0_contract::optimal_first_actions(&GoalUse, contract)
}

/// The actions that attain the ceiling from a mid-episode prefix.
pub fn optimal_actions_from(contract: &Contract, prefix: &[Action]) -> Vec<Action> {
    pretraining_g0_contract::optimal_actions_from(&GoalUse, contract, prefix)
}

/// The gap between a privileged and a public solver.
///
/// Zero on every case, and vacuous: `privileged_value` is not overridden, so
/// this compares the value function with itself. It is computed rather than
/// asserted so that an edit which hides a field is caught.
pub fn vacuous_ambiguity_gap(contract: &Contract) -> i32 {
    ambiguity_gap(&GoalUse, contract)
}

/// Which sub-claim or control a case belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CaseKind {
    /// Matched pairs: one start configuration, two goals, two correct actions.
    WitnessGoalChangesAction,
    /// One goal for every episode.
    NegativeConstantGoal,
    /// The goal is a fixed function of the start configuration.
    NegativeGoalPredictableFromState,
}

impl CaseKind {
    pub const ALL: [Self; 3] = [
        Self::WitnessGoalChangesAction,
        Self::NegativeConstantGoal,
        Self::NegativeGoalPredictableFromState,
    ];

    pub const NEGATIVES: [Self; 2] = [
        Self::NegativeConstantGoal,
        Self::NegativeGoalPredictableFromState,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::WitnessGoalChangesAction => "witness_goal_changes_action",
            Self::NegativeConstantGoal => "negative_constant_goal",
            Self::NegativeGoalPredictableFromState => "negative_goal_predictable_from_state",
        }
    }

    pub const fn regime(self) -> Regime {
        match self {
            Self::WitnessGoalChangesAction => Regime::GoalVaries,
            Self::NegativeConstantGoal => Regime::ConstantGoal,
            Self::NegativeGoalPredictableFromState => Regime::GoalPredictableFromState,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Case {
    pub kind: CaseKind,
    pub contract: Contract,
}

/// The cell every constant-goal episode settles on.
pub const CONSTANT_GOAL_CELL: usize = 0;

/// The whole family as sixteen cases.
///
/// Sixteen is the composite's own distinct-episode count, and that is on
/// purpose: the stage is meant to differ from the composite in the relation it
/// asks for, not in how much data it offers.
///
/// The witness is four start configurations, each with the goal one step
/// forward and one step back. Those two goals have different correct first
/// actions and share every public fact before the goal, which is the matched
/// pair the composite's claim is stated over.
pub fn card_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for start in 0..4 {
        for offset in [1usize, RING - 1] {
            cases.push(Case {
                kind: CaseKind::WitnessGoalChangesAction,
                contract: Contract::new(
                    start,
                    (start + offset) % RING,
                    CaseKind::WitnessGoalChangesAction.regime(),
                ),
            });
        }
    }
    for start in 1..5 {
        cases.push(Case {
            kind: CaseKind::NegativeConstantGoal,
            contract: Contract::new(
                start,
                CONSTANT_GOAL_CELL,
                CaseKind::NegativeConstantGoal.regime(),
            ),
        });
    }
    for start in 0..4 {
        cases.push(Case {
            kind: CaseKind::NegativeGoalPredictableFromState,
            contract: Contract::new(
                start,
                (start + 1) % RING,
                CaseKind::NegativeGoalPredictableFromState.regime(),
            ),
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

/// A policy that sees the whole public contract, because all of it is public.
pub trait PublicPolicy {
    fn name(&self) -> &'static str;
    /// The whole plan, which is sufficient here: nothing is revealed mid-episode.
    fn plan(&self, contract: &Contract) -> Vec<Action>;
}

pub fn run_policy<P: PublicPolicy>(contract: &Contract, policy: &P) -> Outcome {
    run(contract, &policy.plan(contract))
}

/// The ceiling: reads the goal and settles on it as early as possible.
pub struct GoalConditionedExact;

impl PublicPolicy for GoalConditionedExact {
    fn name(&self) -> &'static str {
        "goal_conditioned_exact"
    }

    fn plan(&self, contract: &Contract) -> Vec<Action> {
        let mut prefix = Vec::with_capacity(HORIZON);
        for _ in 0..HORIZON {
            let choices = optimal_actions_from(contract, &prefix);
            match choices.first() {
                Some(action) => prefix.push(*action),
                None => break,
            }
        }
        prefix
    }
}

/// Ignores the goal channel and the configuration both: one fixed plan.
pub struct FixedPlan;

impl PublicPolicy for FixedPlan {
    fn name(&self) -> &'static str {
        "fixed_plan"
    }

    fn plan(&self, _contract: &Contract) -> Vec<Action> {
        vec![Action::Advance, Action::Hold]
    }
}

/// Reads the configuration and the regime and ignores the goal channel.
///
/// The plan is the exact state-only optimum: for each published regime and
/// start configuration, the fixed sequence with the best mean value over the
/// goals that regime actually presents there. Computing it rather than guessing
/// it is what makes the `0.5` witness ceiling a measurement.
pub struct StateOnly;

impl StateOnly {
    fn best_plan(regime: Regime, start: usize) -> Vec<Action> {
        let peers: Vec<Contract> = card_cases()
            .into_iter()
            .map(|case| case.contract)
            .filter(|contract| contract.regime == regime && contract.start == start)
            .collect();
        let mut best = f64::NEG_INFINITY;
        let mut chosen = vec![Action::Hold; HORIZON];
        for sequence in all_sequences() {
            let mean = if peers.is_empty() {
                0.0
            } else {
                peers
                    .iter()
                    .map(|contract| f64::from(run(contract, &sequence).value))
                    .sum::<f64>()
                    / peers.len() as f64
            };
            if mean > best {
                best = mean;
                chosen = sequence;
            }
        }
        chosen
    }
}

impl PublicPolicy for StateOnly {
    fn name(&self) -> &'static str {
        "state_only"
    }

    fn plan(&self, contract: &Contract) -> Vec<Action> {
        Self::best_plan(contract.regime, contract.start)
    }
}

/// Score one policy across every case kind, keeping the kinds separate.
pub fn score_policy<P: PublicPolicy>(policy: &P) -> BTreeMap<String, KindScore> {
    let mut scores = BTreeMap::new();
    for kind in CaseKind::ALL {
        let selected = cases_of(kind);
        let mut solved = 0usize;
        let mut optimal = 0usize;
        for case in &selected {
            let outcome = run_policy(&case.contract, policy);
            if outcome.solved {
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

/// The fraction of a kind's cases on which a policy attains the ceiling.
pub fn optimal_rate<P: PublicPolicy>(policy: &P, kind: CaseKind) -> f64 {
    let cases = cases_of(kind);
    let optimal = cases
        .iter()
        .filter(|case| run_policy(&case.contract, policy).value == value_bounds(&case.contract).0)
        .count();
    optimal as f64 / cases.len() as f64
}

/// The state-only ceiling on one case kind.
///
/// On the witness this is `0.5`: every start configuration carries two goals
/// whose correct first actions differ, so no goal-blind plan can be right on
/// more than one of them.
pub fn state_only_ceiling(kind: CaseKind) -> f64 {
    optimal_rate(&StateOnly, kind)
}

/// The family's central contrast: changing only the goal changes the action.
pub fn goal_use_contrast<P: PublicPolicy>(policy: &P) -> bool {
    optimal_rate(policy, CaseKind::WitnessGoalChangesAction) > 0.99
}

/// Which kernel constructs this family composes.
///
/// Three of the composite's five are gone. The interrupt and the reveal carried
/// the mid-episode switch, the restriction carried the viability boundary, and
/// the norm algebra's connectives had nothing left to connect once the second
/// goal and the prohibition were removed.
pub fn kernel_use() -> KernelUse {
    KernelUse {
        directed_wiring: true,
        shared_coupling: false,
        interrupt: false,
        restrict: false,
        reveal: false,
        norm_algebra: false,
    }
}

/// The norm connectives this family's cases actually use.
pub fn norm_connectives() -> Vec<&'static str> {
    let mut found: Vec<&'static str> = card_cases()
        .into_iter()
        .flat_map(|case| case.contract.norm().connectives())
        .collect();
    found.sort_unstable();
    found.dedup();
    found
}

/// A stable hash of the contract set.
pub fn contract_hash() -> u64 {
    let mut hasher = ContractHasher::new();
    hasher
        .absorb(RING as u64)
        .absorb(HORIZON as u64)
        .absorb(GOAL_REWARD as u64)
        .absorb(MOVE_COST as u64)
        .absorb(Action::ALL.len() as u64);
    for case in card_cases() {
        hasher
            .absorb(case.kind as u64)
            .absorb(case.contract.start as u64)
            .absorb(case.contract.goal as u64)
            .absorb(case.contract.denotation as u64)
            .absorb(case.contract.regime as u64);
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
    fn a_matched_pair_shares_its_configuration_and_takes_opposite_actions() {
        let forward = Contract::new(0, 1, Regime::GoalVaries);
        let backward = Contract::new(0, RING - 1, Regime::GoalVaries);
        assert_eq!(forward.start, backward.start);
        assert_eq!(optimal_first_actions(&forward), vec![Action::Advance]);
        assert_eq!(optimal_first_actions(&backward), vec![Action::Retreat]);
    }

    #[test]
    fn the_state_only_ceiling_on_the_witness_is_exactly_one_half() {
        assert_eq!(
            state_only_ceiling(CaseKind::WitnessGoalChangesAction),
            0.5,
            "each start carries two goals with different correct actions"
        );
        assert_eq!(state_only_ceiling(CaseKind::NegativeConstantGoal), 1.0);
        assert_eq!(
            state_only_ceiling(CaseKind::NegativeGoalPredictableFromState),
            1.0
        );
    }

    #[test]
    fn each_control_restores_exactly_the_baseline_it_is_built_for() {
        assert!(goal_use_contrast(&GoalConditionedExact));
        assert!(!goal_use_contrast(&StateOnly));
        assert!(!goal_use_contrast(&FixedPlan));
        assert_eq!(
            optimal_rate(&FixedPlan, CaseKind::NegativeConstantGoal),
            0.25
        );
        assert_eq!(
            optimal_rate(&FixedPlan, CaseKind::NegativeGoalPredictableFromState),
            1.0
        );
    }

    #[test]
    fn every_case_has_a_vacuous_and_zero_ambiguity_gap() {
        for case in card_cases() {
            assert_eq!(vacuous_ambiguity_gap(&case.contract), 0);
        }
    }

    #[test]
    fn the_norm_has_no_connectives_left() {
        assert!(norm_connectives().is_empty());
        let declared = KernelUse::declared("04").expect("card 04 is declared");
        assert!(declared.norm_algebra && declared.interrupt && declared.restrict);
        assert!(!kernel_use().norm_algebra);
    }
}
