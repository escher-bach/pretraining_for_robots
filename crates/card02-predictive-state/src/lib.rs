//! Card 02 — Predictive State, built as an audited world.
//!
//! The claim is about *selective* retention: the learner should keep exactly the
//! earlier public information that changes action-conditioned futures, and drop
//! equally salient information that does not. Both halves are needed. A family
//! that only rewarded remembering would be passed by a learner that remembers
//! everything, which is not the capability.
//!
//! # The witness
//!
//! Seven cells, three decisions. A public latch fires before the first decision
//! and sets a mode. During the first two decisions the two modes are
//! *observationally identical*: the two mode-sensitive commands are inert, so no
//! action can rediscover what the latch said. At the third decision they
//! separate — one command advances by two and the other retreats by two, and
//! which is which is the mode.
//!
//! The goal sits four cells ahead, so the unique route is two unit steps and
//! then one mode-correct command. Nothing mode-independent reaches it: three
//! unit steps land one short. That is what makes the latch necessary rather than
//! merely available.
//!
//! # Where the evidence is
//!
//! The mode is **public**. It is published once and never repeated, so the
//! ordinary ambiguity gap — privileged value against public value — is zero, and
//! saying so would be saying nothing. Card 04 was audited that way and the
//! number turned out vacuous.
//!
//! The load-bearing quantity here is the ceiling of a policy that *cannot see the
//! latch*, computed by coarsening the public trace and re-solving exactly. The
//! difference between that and the full public ceiling is what the memory is
//! worth. Everything the card claims is a statement about that difference or
//! about the identification diameter, not about the value-function gap.

mod audit;
mod render;
pub use audit::*;
pub use render::*;

use std::collections::BTreeMap;

use pretraining_g0_contract::{
    AmbiguitySet, ContractHasher, Displaced, Fragment, Guard, Interrupt, KernelUse,
    PubliclyObservable, Resume, Ring, Symmetry,
};
use serde::{Deserialize, Serialize};

pub use pretraining_g0_contract::{BracketStructure, Isolation, KindScore, OrbitVerdict};

/// Seven cells: enough that the goal four ahead is distinct from the cell a
/// wrong mode-command lands on, and no larger.
pub const RING: usize = 7;

pub const CONFIGURATION: Ring = Ring::new(RING);

/// Three decisions: two of aliasing and one that discriminates.
pub const HORIZON: usize = 3;

/// The decision at which the mode-sensitive commands become effective.
pub const DISCRIMINATING_STEP: usize = 2;

/// How far ahead the goal sits.
///
/// Four, so the route is `Step, Step, +2`. Three would be reachable by unit
/// steps alone and the latch would carry no consequence; five would be
/// unreachable in the budget.
pub const GOAL_OFFSET: usize = 4;

pub const GOAL_REWARD: i32 = 100;
pub const MOVE_COST: i32 = 1;

/// Neither the aliased route nor the discriminating command may be skippable.
const _: () = assert!(GOAL_OFFSET > HORIZON);
const _: () = assert!(GOAL_OFFSET <= HORIZON - 1 + 2);

/// The hidden mode a latch sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Mode {
    Forward,
    Reversed,
}

impl Mode {
    pub const ALL: [Self; 2] = [Self::Forward, Self::Reversed];

    pub const fn index(self) -> usize {
        match self {
            Self::Forward => 0,
            Self::Reversed => 1,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Reversed => "reversed",
        }
    }

    pub const fn flipped(self) -> Self {
        match self {
            Self::Forward => Self::Reversed,
            Self::Reversed => Self::Forward,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Action {
    Hold,
    Step,
    /// Advances by two in `Forward` mode and retreats by two in `Reversed`.
    Cross,
    /// The mirror of `Cross`.
    Anti,
}

impl Action {
    pub const ALL: [Self; 4] = [Self::Hold, Self::Step, Self::Cross, Self::Anti];

    /// The two commands whose meaning the mode decides.
    pub const MODE_SENSITIVE: [Self; 2] = [Self::Cross, Self::Anti];

    pub const fn index(self) -> usize {
        match self {
            Self::Hold => 0,
            Self::Step => 1,
            Self::Cross => 2,
            Self::Anti => 3,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Hold => "hold",
            Self::Step => "step",
            Self::Cross => "cross",
            Self::Anti => "anti",
        }
    }

    pub const fn is_mode_sensitive(self) -> bool {
        matches!(self, Self::Cross | Self::Anti)
    }

    fn from_index(index: usize) -> Option<Self> {
        Self::ALL.into_iter().find(|action| action.index() == index)
    }
}

/// When the mode is published.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ModeVisibility {
    /// Once, before the first decision. The learner has to carry it.
    Latched,
    /// Before every decision, so a memoryless policy suffices. This is the
    /// fully-observable control.
    Always,
}

/// Whether the mode changes what the mode-sensitive commands do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ModeCoupling {
    /// The witness: the mode decides which command advances.
    Discriminating,
    /// The irrelevant-latch control: the latch fires and nothing depends on it.
    Inert,
}

/// One fully specified episode contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contract {
    pub start: usize,
    pub mode: Mode,
    pub visibility: ModeVisibility,
    pub coupling: ModeCoupling,
    /// A second latch that is published and never consulted.
    ///
    /// It exists so that carrying *everything* is distinguishable from carrying
    /// the right thing: a policy keyed on the most recent latch reads this one
    /// and is wrong exactly when it disagrees with the mode.
    pub decoy: Option<Mode>,
    /// The decision after which the aliasing interval ends.
    ///
    /// The witness sets it so the mode-sensitive commands are inert until the
    /// last decision, which is what makes the mode unprobeable. Lowering it is
    /// the card's "end the aliasing interval early" transformation.
    pub aliasing_until: usize,
    /// The decision before which the latch fires.
    ///
    /// Anywhere inside the aliasing interval is admissible; moving it changes
    /// how long the learner must carry the mode and nothing else, which is one
    /// of the card's preserving transformations.
    pub latch_at: usize,
    /// What the latch publishes, when that is not the true mode.
    ///
    /// `None` is the honest latch. `Some` decorrelates it: the world still has a
    /// mode and the learner is still told something, but what it is told no
    /// longer identifies what it needs. This is the card's declared
    /// meaning-changing transformation and it is why the published value is a
    /// separate field from the mode rather than the same one read twice.
    pub latch_reports: Option<Mode>,
}

impl Contract {
    pub fn new(start: usize, mode: Mode) -> Self {
        Self {
            start,
            mode,
            visibility: ModeVisibility::Latched,
            coupling: ModeCoupling::Discriminating,
            decoy: None,
            aliasing_until: DISCRIMINATING_STEP,
            latch_at: 0,
            latch_reports: None,
        }
    }

    pub fn with_visibility(mut self, visibility: ModeVisibility) -> Self {
        self.visibility = visibility;
        self
    }

    pub fn with_coupling(mut self, coupling: ModeCoupling) -> Self {
        self.coupling = coupling;
        self
    }

    pub fn with_decoy(mut self, decoy: Mode) -> Self {
        self.decoy = Some(decoy);
        self
    }

    pub fn with_aliasing_until(mut self, step: usize) -> Self {
        self.aliasing_until = step;
        self
    }

    pub fn goal(&self) -> usize {
        (self.start + GOAL_OFFSET) % RING
    }

    /// The interrupt that ends the aliasing interval.
    ///
    /// The displaced process continues and resumes from the state it reached:
    /// the configuration is not reset when the commands come alive, which is why
    /// the two unit steps taken during aliasing still count toward the route.
    pub fn aliasing_interrupt(&self) -> Interrupt {
        Interrupt::new(
            Guard::AfterStep(self.aliasing_until.saturating_sub(1)),
            Displaced::Continues,
            Resume::FromState,
        )
    }

    /// Whether the mode-sensitive commands have effect yet.
    pub fn commands_are_live(&self, executed: usize) -> bool {
        executed >= self.aliasing_until
    }

    /// Whether the mode is published before the decision at `executed`.
    pub fn mode_is_published(&self, executed: usize) -> bool {
        match self.visibility {
            ModeVisibility::Always => true,
            ModeVisibility::Latched => executed == self.latch_at,
        }
    }

    /// What the latch says, which is the true mode unless it is decorrelated.
    pub fn reported_mode(&self) -> Mode {
        self.latch_reports.unwrap_or(self.mode)
    }

    /// The displacement a command produces at this step.
    pub fn displacement(&self, executed: usize, action: Action) -> i64 {
        match action {
            Action::Hold => 0,
            Action::Step => 1,
            Action::Cross | Action::Anti if !self.commands_are_live(executed) => 0,
            Action::Cross | Action::Anti => {
                let forward = match self.coupling {
                    // The control makes both commands advance, so the latch
                    // still fires and still says something true about a mode
                    // that no longer changes any effect.
                    ModeCoupling::Inert => true,
                    ModeCoupling::Discriminating => {
                        (self.mode == Mode::Forward) == (action == Action::Cross)
                    }
                };
                if forward {
                    2
                } else {
                    -2
                }
            }
        }
    }

    pub fn advance(&self, cell: usize, executed: usize, action: Action) -> usize {
        let shifted = cell as i64 + self.displacement(executed, action);
        shifted.rem_euclid(RING as i64) as usize
    }

    /// The same contract with its mode flipped, which is the pair the ablation
    /// and non-interference checks are run over.
    pub fn with_flipped_mode(&self) -> Self {
        Self {
            mode: self.mode.flipped(),
            ..*self
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub value: i32,
    pub reached_goal: bool,
    pub final_cell: usize,
}

/// Execute a complete action sequence.
pub fn run(contract: &Contract, actions: &[Action]) -> Outcome {
    let mut cell = contract.start;
    let mut trajectory = vec![cell];
    for (executed, action) in actions.iter().copied().enumerate() {
        cell = contract.advance(cell, executed, action);
        trajectory.push(cell);
    }
    let goal = contract.goal();
    let settle =
        (0..trajectory.len()).find(|index| trajectory[*index..].iter().all(|entry| *entry == goal));
    Outcome {
        value: settle.map_or(0, |steps| GOAL_REWARD - MOVE_COST * steps as i32),
        reached_goal: settle.is_some(),
        final_cell: cell,
    }
}

pub struct PredictiveState;

impl Fragment for PredictiveState {
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

/// Mode publications are tagged in the trace rather than living at a fixed
/// index.
///
/// A fixed index only works while the latch fires before the first decision. The
/// card's own preserving transformation moves it later inside the aliasing
/// interval, and with a positional coarsening that transform silently stopped
/// ablating anything: the ablated ceiling jumped from `48.5` to `97` and a
/// preserving transform read as meaning-changing. Tagging lets
/// [`without_the_latch`] find the first mode publication wherever it is, and
/// leave any later republication alone — which is what keeps the
/// fully-observable control observable.
pub const MODE_TAG: i64 = 100;
/// The decoy's tag, distinct so ablating one cannot remove the other.
pub const DECOY_TAG: i64 = 200;

impl PubliclyObservable for PredictiveState {
    /// The start cell, the latch, the decoy, the goal, then one entry per
    /// decision: the cell reached, and the mode again on the steps where it is
    /// republished.
    fn public_trace(&self, contract: &Contract, actions: &[Action]) -> Vec<i64> {
        let mut trace = vec![contract.start as i64];
        trace.push(match contract.decoy {
            Some(mode) => DECOY_TAG + mode.index() as i64,
            None => 0,
        });
        trace.push(contract.goal() as i64);
        // What is *published*, not what is true. A decorrelated latch tells the
        // learner something that no longer identifies the mode, and a trace that
        // wrote the true mode would hide that.
        if contract.mode_is_published(0) {
            trace.push(MODE_TAG + contract.reported_mode().index() as i64);
        }
        let mut cell = contract.start;
        for (executed, action) in actions.iter().copied().enumerate() {
            cell = contract.advance(cell, executed, action);
            trace.push(cell as i64);
            // Republication is itself public, so an `Always` contract's trace
            // says the mode again at every decision. That is what makes the
            // fully-observable control solvable without memory.
            if contract.mode_is_published(executed + 1) {
                trace.push(MODE_TAG + contract.reported_mode().index() as i64);
            }
        }
        trace
    }
}

/// The trace with the first mode publication removed and nothing else.
///
/// This is the ablation the card asks for: a learner that saw the latch fire but
/// did not keep it. Later republications are left in place, because a control
/// that republishes has genuinely told the learner again — removing those too
/// would make the fully-observable control unsolvable and turn a control into a
/// second witness.
pub fn without_the_latch(trace: &[i64]) -> Vec<i64> {
    let mut coarsened = trace.to_vec();
    if let Some(position) = coarsened
        .iter()
        .position(|entry| (MODE_TAG..DECOY_TAG).contains(entry))
    {
        coarsened[position] = 0;
    }
    coarsened
}

/// The trace with the decoy removed, used to show the decoy is inert.
pub fn without_the_decoy(trace: &[i64]) -> Vec<i64> {
    let mut coarsened = trace.to_vec();
    if let Some(position) = coarsened.iter().position(|entry| *entry >= DECOY_TAG) {
        coarsened[position] = 0;
    }
    coarsened
}

pub fn all_sequences() -> Vec<Vec<Action>> {
    pretraining_g0_contract::sequences_of_length(&Action::ALL, HORIZON)
}

pub fn value_bounds(contract: &Contract) -> (i32, Vec<Vec<Action>>) {
    pretraining_g0_contract::value_bounds(&PredictiveState, contract)
}

pub fn value_bounds_over(contract: &Contract, horizon: usize) -> (i32, Vec<Vec<Action>>) {
    pretraining_g0_contract::value_bounds_over(&PredictiveState, contract, horizon)
}

pub fn optimal_first_actions(contract: &Contract) -> Vec<Action> {
    pretraining_g0_contract::optimal_first_actions(&PredictiveState, contract)
}

/// The mode pair a learner cannot separate during aliasing.
pub fn mode_ambiguity(contract: &Contract) -> AmbiguitySet<Contract> {
    AmbiguitySet::uniform(vec![*contract, contract.with_flipped_mode()])
}

/// The correct command at the discriminating decision, having taken the route.
///
/// `optimal_first_actions` is the wrong observable for this card: the first move
/// is `Step` in both modes and the contrast lives at the last decision. Reading
/// the first action would report agreement where the card claims a difference.
pub fn discriminating_actions(contract: &Contract) -> Vec<Action> {
    let prefix = route_prefix();
    let mut cell = contract.start;
    for (executed, action) in prefix.iter().copied().enumerate() {
        cell = contract.advance(cell, executed, action);
    }
    let mut best = i32::MIN;
    let mut chosen = Vec::new();
    for action in Action::ALL {
        let mut sequence = prefix.clone();
        sequence.push(action);
        let value = run(contract, &sequence).value;
        if value > best {
            best = value;
            chosen.clear();
        }
        if value == best {
            chosen.push(action);
        }
    }
    let _ = cell;
    chosen
}

/// The mode-independent prefix every optimal route shares.
pub fn route_prefix() -> Vec<Action> {
    vec![Action::Step; DISCRIMINATING_STEP]
}

/// Which sub-claim or control a case belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CaseKind {
    /// The latch is the only thing that decides the last command.
    WitnessLatchedMode,
    /// The mode is republished every step, so no memory is required.
    NegativeFullyObservable,
    /// The latch fires and no effect depends on it.
    NegativeIrrelevantLatch,
    /// Two latches, only one of which matters.
    NegativeMemoryCost,
}

impl CaseKind {
    pub const ALL: [Self; 4] = [
        Self::WitnessLatchedMode,
        Self::NegativeFullyObservable,
        Self::NegativeIrrelevantLatch,
        Self::NegativeMemoryCost,
    ];

    pub const NEGATIVES: [Self; 3] = [
        Self::NegativeFullyObservable,
        Self::NegativeIrrelevantLatch,
        Self::NegativeMemoryCost,
    ];

    pub const fn is_witness(self) -> bool {
        matches!(self, Self::WitnessLatchedMode)
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::WitnessLatchedMode => "witness_latched_mode",
            Self::NegativeFullyObservable => "negative_fully_observable",
            Self::NegativeIrrelevantLatch => "negative_irrelevant_latch",
            Self::NegativeMemoryCost => "negative_memory_cost",
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
/// Every kind carries both modes. A kind with one mode would let a constant
/// command score perfectly on it, and the negative would then be measuring the
/// constant rather than the control.
pub fn card_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for mode in Mode::ALL {
        cases.push(Case {
            kind: CaseKind::WitnessLatchedMode,
            contract: Contract::new(0, mode),
        });
        cases.push(Case {
            kind: CaseKind::NegativeFullyObservable,
            contract: Contract::new(0, mode).with_visibility(ModeVisibility::Always),
        });
        cases.push(Case {
            kind: CaseKind::NegativeIrrelevantLatch,
            contract: Contract::new(0, mode).with_coupling(ModeCoupling::Inert),
        });
        // Matched pairs: the decoy agrees with the mode in one and disagrees in
        // the other, and the correct command is the same in both. A policy keyed
        // on the most recent latch is wrong on exactly half.
        for decoy in Mode::ALL {
            cases.push(Case {
                kind: CaseKind::NegativeMemoryCost,
                contract: Contract::new(0, mode).with_decoy(decoy),
            });
        }
    }
    cases
}

/// Public information available to a policy at one decision.
pub struct PublicView<'a> {
    pub contract: &'a Contract,
    pub cell: usize,
    pub executed: usize,
    /// What the policy has retained, which is the thing under test.
    pub recalled_mode: Option<Mode>,
    pub recalled_decoy: Option<Mode>,
}

pub trait PublicPolicy {
    fn name(&self) -> &'static str;
    /// How many decisions back this policy can see a publication.
    ///
    /// The rollout uses it to decide what to put in the view, so a policy cannot
    /// quietly read something its declared span excludes.
    fn memory_span(&self) -> usize;
    fn act(&self, view: &PublicView<'_>) -> Action;
}

/// Roll a policy forward, giving it only what its declared span retains.
pub fn run_policy<P: PublicPolicy>(contract: &Contract, policy: &P) -> Outcome {
    let mut cell = contract.start;
    let mut actions = Vec::with_capacity(HORIZON);
    for executed in 0..HORIZON {
        // A publication at decision `k` is inside a span of `w` at decision
        // `executed` when `executed - k < w`. The latch fires at `0`, so seeing
        // it at the discriminating decision needs a span of at least three.
        let visible =
            |published_at: usize| executed.saturating_sub(published_at) < policy.memory_span();
        let mode_published_at: Option<usize> = (0..=executed)
            .rev()
            .find(|step| contract.mode_is_published(*step));
        let recalled_mode = mode_published_at
            .filter(|step| visible(*step))
            .map(|_| contract.reported_mode());
        let recalled_decoy = contract.decoy.filter(|_| visible(contract.latch_at));
        let action = policy.act(&PublicView {
            contract,
            cell,
            executed,
            recalled_mode,
            recalled_decoy,
        });
        actions.push(action);
        cell = contract.advance(cell, executed, action);
    }
    run(contract, &actions)
}

/// The ceiling policy: takes the route, then the command the recalled mode makes
/// correct.
pub struct ModeConditioned;

impl PublicPolicy for ModeConditioned {
    fn name(&self) -> &'static str {
        "mode_conditioned"
    }

    fn memory_span(&self) -> usize {
        HORIZON
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        if view.executed < DISCRIMINATING_STEP {
            return Action::Step;
        }
        match view.recalled_mode {
            Some(Mode::Forward) | None => Action::Cross,
            Some(Mode::Reversed) => Action::Anti,
        }
    }
}

/// Retains nothing across decisions.
///
/// It still sees a republished mode, which is what makes it optimal on the
/// fully-observable control and unable to pass the witness.
pub struct Memoryless;

impl PublicPolicy for Memoryless {
    fn name(&self) -> &'static str {
        "memoryless"
    }

    fn memory_span(&self) -> usize {
        1
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        ModeConditioned.act(view)
    }
}

/// A fixed window, used just below and just above the required span.
pub struct Window {
    pub span: usize,
    label: &'static str,
}

impl Window {
    /// One decision short of what the witness needs.
    pub fn too_short() -> Self {
        Self {
            span: DISCRIMINATING_STEP,
            label: "window_below_the_required_span",
        }
    }

    /// Exactly what the witness needs.
    pub fn sufficient() -> Self {
        Self {
            span: DISCRIMINATING_STEP + 1,
            label: "window_at_the_required_span",
        }
    }
}

impl PublicPolicy for Window {
    fn name(&self) -> &'static str {
        self.label
    }

    fn memory_span(&self) -> usize {
        self.span
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        ModeConditioned.act(view)
    }
}

/// Acts on the most recently published latch of any kind.
///
/// It is right whenever the decoy agrees with the mode and wrong whenever it
/// does not, which is the whole content of the memory-cost control.
pub struct LastLatch;

impl PublicPolicy for LastLatch {
    fn name(&self) -> &'static str {
        "last_latch"
    }

    fn memory_span(&self) -> usize {
        HORIZON
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        if view.executed < DISCRIMINATING_STEP {
            return Action::Step;
        }
        let effective = view.recalled_decoy.or(view.recalled_mode);
        match effective {
            Some(Mode::Reversed) => Action::Anti,
            _ => Action::Cross,
        }
    }
}

/// Takes the route and then always issues the same command.
pub struct ConstantCommand;

impl PublicPolicy for ConstantCommand {
    fn name(&self) -> &'static str {
        "constant_command"
    }

    fn memory_span(&self) -> usize {
        1
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        if view.executed < DISCRIMINATING_STEP {
            Action::Step
        } else {
            Action::Cross
        }
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

/// The card's central contrast: the discriminating command must be correct in
/// both modes and must differ between them.
pub fn retention_contrast<P: PublicPolicy>(policy: &P) -> bool {
    let witnesses: Vec<Case> = card_cases()
        .into_iter()
        .filter(|case| case.kind == CaseKind::WitnessLatchedMode)
        .collect();
    if witnesses.len() != 2 {
        return false;
    }
    let commands: Vec<Action> = witnesses
        .iter()
        .map(|case| taught_sequence(&case.contract, policy)[DISCRIMINATING_STEP])
        .collect();
    let correct = witnesses
        .iter()
        .zip(&commands)
        .all(|(case, command)| discriminating_actions(&case.contract).contains(command));
    correct && commands[0] != commands[1]
}

/// The action sequence a policy actually produces.
pub fn taught_sequence<P: PublicPolicy>(contract: &Contract, policy: &P) -> Vec<Action> {
    let mut cell = contract.start;
    let mut actions = Vec::with_capacity(HORIZON);
    for executed in 0..HORIZON {
        let visible =
            |published_at: usize| executed.saturating_sub(published_at) < policy.memory_span();
        let mode_published_at: Option<usize> = (0..=executed)
            .rev()
            .find(|step| contract.mode_is_published(*step));
        let action = policy.act(&PublicView {
            contract,
            cell,
            executed,
            recalled_mode: mode_published_at
                .filter(|step| visible(*step))
                .map(|_| contract.reported_mode()),
            recalled_decoy: contract.decoy.filter(|_| visible(contract.latch_at)),
        });
        actions.push(action);
        cell = contract.advance(cell, executed, action);
    }
    actions
}

/// Which kernel constructs this card composes.
pub fn kernel_use() -> KernelUse {
    KernelUse {
        directed_wiring: true,
        shared_coupling: false,
        // The aliasing interval ends by interrupt: the mode-sensitive commands
        // displace the inert ones, and the configuration resumes from the state
        // the route reached rather than restarting.
        interrupt: true,
        restrict: false,
        reveal: false,
        norm_algebra: false,
    }
}

/// Move a contract through one ring symmetry.
///
/// Reflections are included. Unlike card 03, every command here has an image:
/// `Step` is the only unit move and the two mode-sensitive commands are already
/// each other's mirror, so a reflection maps the card onto itself with the mode
/// flipped.
pub fn transform(contract: &Contract, symmetry: Symmetry) -> Contract {
    Contract {
        start: symmetry.apply(contract.start),
        ..*contract
    }
}

/// A stable hash of the contract set.
pub fn contract_hash() -> u64 {
    let mut hasher = ContractHasher::new();
    hasher
        .absorb(RING as u64)
        .absorb(HORIZON as u64)
        .absorb(GOAL_OFFSET as u64)
        .absorb(DISCRIMINATING_STEP as u64)
        .absorb(GOAL_REWARD as u64)
        .absorb(MOVE_COST as u64);
    for case in card_cases() {
        hasher
            .absorb(case.kind as u64)
            .absorb(case.contract.start as u64)
            .absorb(case.contract.mode as u64)
            .absorb(case.contract.visibility as u64)
            .absorb(case.contract.coupling as u64)
            .absorb(case.contract.aliasing_until as u64)
            .absorb(case.contract.latch_at as u64);
        hasher.absorb_option(case.contract.decoy.map(|mode| mode as u64));
        hasher.absorb_option(case.contract.latch_reports.map(|mode| mode as u64));
    }
    hasher.finish()
}

pub fn action_from_index(index: usize) -> Option<Action> {
    Action::from_index(index)
}
