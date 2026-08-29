//! A second finite process carrying the same goal-sensitive relation through
//! entirely different surface facts.
//!
//! The relation under test is the one the first process established: hold the
//! situation fixed, change only the requested outcome, and the correct action
//! must change. Everything on the surface is different. There is no line, no
//! coordinate, no direction, and no signed displacement. There are three
//! containers, one tracked item, one blocker, and exactly one free container at
//! all times. An action evicts whatever occupies a named container into the free
//! one.
//!
//! The point of a second process is discrimination. If a learner solved the
//! first process by internalising "the goal is left, so command a negative
//! number", that rule is not merely wrong here — it is not even expressible. If
//! it instead learned "read the requested outcome, compare it with the current
//! situation, and act on the difference", that transfers. Nothing in this crate
//! tests which of those happened; it builds and audits the instrument that could.
//!
//! Every quantity below is exact by exhaustive enumeration over four semantic
//! commands and a two-step horizon: sixteen sequences. No learner is involved,
//! no training is performed, and no transfer is claimed.

use pretraining_world::{LearningToken, PublicToken, Role, Supervision, PAYLOAD_DIM};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const CONTAINER_COUNT: usize = 3;
pub const HORIZON: u8 = 2;
pub const GOAL_BLIND_CEILING: f64 = 0.5;
pub const PROCESS_VERSION: &str = "container-eviction-selection-0.1.0";

/// A command channel counts as actuated at or above this normalized value.
pub const COMMAND_THRESHOLD: f32 = 0.5;

/// Schema tag distinguishing the item band from the blocker band.
///
/// This is the one place the process gives a categorical fact a numeric
/// encoding, and it is declared rather than hidden. Two records inside one
/// simultaneity group must be distinguishable by something other than their
/// order, and the first process uses the same mechanism to distinguish its five
/// position keys.
pub const ITEM_BAND_TAG: f32 = 1.0;
pub const BLOCKER_BAND_TAG: f32 = -1.0;

const BOUNDARY_TASK_RESET: f32 = 0.0;
const BOUNDARY_EPISODE_END: f32 = 1.0;

// ---------------------------------------------------------------------------
// Semantics
// ---------------------------------------------------------------------------

/// Which of the two occupants sits in a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Occupant {
    Item,
    Blocker,
}

/// The complete situation: where the tracked item is and where the blocker is.
///
/// With three containers and two occupants there is always exactly one free
/// container, which is what makes eviction total and the process finite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Occupancy {
    pub item: u8,
    pub blocker: u8,
}

impl Occupancy {
    pub fn new(item: u8, blocker: u8) -> Result<Self, String> {
        if item as usize >= CONTAINER_COUNT || blocker as usize >= CONTAINER_COUNT {
            return Err("an occupant must be in a declared container".into());
        }
        if item == blocker {
            return Err("a container holds at most one occupant".into());
        }
        Ok(Self { item, blocker })
    }

    /// The unique container holding nothing.
    pub fn free(self) -> u8 {
        (0..CONTAINER_COUNT as u8)
            .find(|container| *container != self.item && *container != self.blocker)
            .expect("three containers and two occupants leave exactly one free")
    }

    pub fn occupant(self, container: u8) -> Option<Occupant> {
        if container == self.item {
            Some(Occupant::Item)
        } else if container == self.blocker {
            Some(Occupant::Blocker)
        } else {
            None
        }
    }
}

/// One semantic command.
///
/// The alphabet is categorical. Unlike the first process's single signed
/// control, no command is between any two others, and no command is the negation
/// of another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Command {
    Hold,
    Evict(u8),
    /// More than one channel was actuated at once. The world treats this as a
    /// refused step, and it is counted separately so a decoder cannot quietly
    /// report it as a hold.
    Overreach,
}

impl Command {
    /// The four semantic commands a policy may intend. `Overreach` is a decoding
    /// outcome, not an intention, so it is not in the alphabet.
    pub fn alphabet() -> Vec<Self> {
        let mut alphabet = vec![Self::Hold];
        alphabet.extend((0..CONTAINER_COUNT as u8).map(Self::Evict));
        alphabet
    }

    pub fn as_str(self) -> String {
        match self {
            Self::Hold => "hold".into(),
            Self::Evict(container) => format!("evict_{container}"),
            Self::Overreach => "overreach".into(),
        }
    }
}

/// Evict the occupant of the named container into the free container.
///
/// Evicting a free container, holding, and overreaching all leave the situation
/// unchanged. Because the evicted occupant lands in the container that was free,
/// the named container becomes the new free one and the invariant is preserved.
pub fn transition(state: Occupancy, command: Command) -> Occupancy {
    match command {
        Command::Hold | Command::Overreach => state,
        Command::Evict(container) => match state.occupant(container) {
            None => state,
            Some(Occupant::Item) => Occupancy {
                item: state.free(),
                blocker: state.blocker,
            },
            Some(Occupant::Blocker) => Occupancy {
                item: state.item,
                blocker: state.free(),
            },
        },
    }
}

/// Whether a complete command sequence ever puts the item in the goal container.
pub fn sequence_succeeds(start: Occupancy, goal: u8, commands: &[Command]) -> bool {
    let mut state = start;
    if state.item == goal {
        return true;
    }
    for command in commands {
        state = transition(state, *command);
        if state.item == goal {
            return true;
        }
    }
    false
}

/// Every command sequence of the full horizon.
pub fn enumerated_sequences() -> Vec<Vec<Command>> {
    let alphabet = Command::alphabet();
    let mut sequences = vec![Vec::new()];
    for _ in 0..HORIZON {
        let mut extended = Vec::new();
        for prefix in &sequences {
            for command in &alphabet {
                let mut next = prefix.clone();
                next.push(*command);
                extended.push(next);
            }
        }
        sequences = extended;
    }
    sequences
}

/// The commands that can begin a successful sequence for this case.
pub fn optimal_first_commands(start: Occupancy, goal: u8) -> Vec<Command> {
    let mut first_commands: Vec<Command> = enumerated_sequences()
        .into_iter()
        .filter(|sequence| sequence_succeeds(start, goal, sequence))
        .map(|sequence| sequence[0])
        .collect();
    first_commands.sort_unstable();
    first_commands.dedup();
    first_commands
}

/// The exact success ceiling of any policy that cannot read the goal, on the
/// balanced witness pair. Obtained by enumerating all sixteen sequences.
pub fn hidden_goal_public_ceiling() -> f64 {
    let start = witness_start();
    let goals = witness_goals();
    let mut best: f64 = 0.0;
    for sequence in enumerated_sequences() {
        let successes = goals
            .iter()
            .filter(|goal| sequence_succeeds(start, **goal, &sequence))
            .count();
        best = best.max(successes as f64 / goals.len() as f64);
    }
    best
}

/// The situation both witnesses share: the item in container 0, the blocker in
/// container 1, container 2 free.
pub fn witness_start() -> Occupancy {
    Occupancy::new(0, 1).expect("the witness start is a legal situation")
}

/// The two requested outcomes. One names an occupied container, the other names
/// the free one, so the correct first command is not a copy of the goal.
pub fn witness_goals() -> [u8; 2] {
    [1, 2]
}

// ---------------------------------------------------------------------------
// Presentation: names
// ---------------------------------------------------------------------------

/// A public naming of the process.
///
/// Three key bands: which container holds the item, which holds the blocker, and
/// which command channel evicts it. The item and blocker bands live in the
/// observation namespace and must not collide with each other; the command band
/// is an actuator namespace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Presentation {
    pub name: String,
    item_keys: [u16; CONTAINER_COUNT],
    blocker_keys: [u16; CONTAINER_COUNT],
    evict_keys: [u16; CONTAINER_COUNT],
}

impl Presentation {
    pub fn canonical() -> Self {
        Self {
            name: "canonical".into(),
            item_keys: [100, 101, 102],
            blocker_keys: [110, 111, 112],
            evict_keys: [120, 121, 122],
        }
    }

    /// A consistent renaming that preserves every public meaning.
    ///
    /// It breaks numeric rank, contiguity, and the arithmetic offset between the
    /// bands, which is exactly what the canonical naming accidentally offers.
    pub fn renamed() -> Self {
        Self {
            name: "renamed".into(),
            item_keys: [307, 303, 305],
            blocker_keys: [341, 344, 342],
            evict_keys: [363, 361, 362],
        }
    }

    pub fn item_key(&self, container: u8) -> u16 {
        self.item_keys[container as usize]
    }

    pub fn blocker_key(&self, container: u8) -> u16 {
        self.blocker_keys[container as usize]
    }

    pub fn evict_key(&self, container: u8) -> u16 {
        self.evict_keys[container as usize]
    }

    pub fn container_of_item_key(&self, key: u16) -> Option<u8> {
        self.item_keys
            .iter()
            .position(|candidate| *candidate == key)
            .map(|container| container as u8)
    }

    pub fn container_of_evict_key(&self, key: u16) -> Option<u8> {
        self.evict_keys
            .iter()
            .position(|candidate| *candidate == key)
            .map(|container| container as u8)
    }

    /// The public schema: one entry per container, binding its three keys.
    ///
    /// The binding is published by co-membership in one simultaneity group, not
    /// by any numeric relationship between the key values.
    pub fn schema(&self) -> Vec<ContainerSchema> {
        (0..CONTAINER_COUNT as u8)
            .map(|container| ContainerSchema {
                item_key: self.item_key(container),
                blocker_key: self.blocker_key(container),
                evict_key: self.evict_key(container),
            })
            .collect()
    }

    /// Every public key this presentation uses, for disjointness checks.
    pub fn all_observation_keys(&self) -> Vec<u16> {
        self.item_keys
            .iter()
            .chain(self.blocker_keys.iter())
            .copied()
            .collect()
    }

    pub fn all_actuator_keys(&self) -> Vec<u16> {
        self.evict_keys.to_vec()
    }
}

/// One container's public identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContainerSchema {
    pub item_key: u16,
    pub blocker_key: u16,
    pub evict_key: u16,
}

// ---------------------------------------------------------------------------
// Presentation: order
// ---------------------------------------------------------------------------

/// Two orderings of the same public facts.
///
/// Only order changes: the same schema entries, the same observations, the same
/// commands, and the same dynamics. Unlike the first process, the invariance
/// this axis tests is **measured** here rather than assumed, because a policy
/// that reads the first declared container is order sensitive and must be caught.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializationOrder {
    Canonical,
    Permuted,
}

impl SerializationOrder {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Permuted => "permuted",
        }
    }

    /// The order in which the per-container schema groups are published.
    fn group_order(self) -> [usize; CONTAINER_COUNT] {
        match self {
            Self::Canonical => [0, 1, 2],
            Self::Permuted => [2, 0, 1],
        }
    }

    /// The order in which the command channels are queried.
    fn channel_order(self) -> [usize; CONTAINER_COUNT] {
        match self {
            Self::Canonical => [0, 1, 2],
            Self::Permuted => [1, 2, 0],
        }
    }

    /// Whether the three records inside a schema group are reversed.
    fn reverse_within_group(self) -> bool {
        matches!(self, Self::Permuted)
    }
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaseKind {
    Witness,
    FixedGoalControl,
    StatePredictsGoalControl,
    HiddenGoalLeakageCheck,
    RenamedWitness,
}

impl CaseKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Witness => "witness",
            Self::FixedGoalControl => "fixed_goal_control",
            Self::StatePredictsGoalControl => "state_predicts_goal_control",
            Self::HiddenGoalLeakageCheck => "hidden_goal_leakage_check",
            Self::RenamedWitness => "renamed_witness",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvictionCase {
    pub id: String,
    pub kind: CaseKind,
    pub start: Occupancy,
    goal: u8,
    pub goal_visible: bool,
    pub presentation: Presentation,
}

impl EvictionCase {
    pub fn goal_for_verification(&self) -> u8 {
        self.goal
    }

    pub fn public_observation(
        &self,
        state: Occupancy,
        steps_remaining: u8,
        order: SerializationOrder,
    ) -> PublicObservation {
        let schema = self.presentation.schema();
        let containers = order
            .group_order()
            .into_iter()
            .map(|container| schema[container])
            .collect();
        PublicObservation {
            containers,
            item_here_key: self.presentation.item_key(state.item),
            blocker_here_key: self.presentation.blocker_key(state.blocker),
            goal_key: self
                .goal_visible
                .then(|| self.presentation.item_key(self.goal)),
            steps_remaining,
        }
    }

    pub fn with_presentation(&self, presentation: Presentation) -> Self {
        let mut case = self.clone();
        case.presentation = presentation;
        case
    }
}

/// Everything a policy may see. It contains no occupancy the observations do not
/// state, no hidden goal, and no generator identifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicObservation {
    /// The container schema, in the order it was published.
    pub containers: Vec<ContainerSchema>,
    pub item_here_key: u16,
    pub blocker_here_key: u16,
    pub goal_key: Option<u16>,
    pub steps_remaining: u8,
}

impl PublicObservation {
    pub fn container_of_item_key(&self, key: u16) -> Option<&ContainerSchema> {
        self.containers
            .iter()
            .find(|container| container.item_key == key)
    }

    pub fn container_of_blocker_key(&self, key: u16) -> Option<&ContainerSchema> {
        self.containers
            .iter()
            .find(|container| container.blocker_key == key)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvictionSuite {
    pub cases: Vec<EvictionCase>,
}

impl EvictionSuite {
    pub fn standard() -> Self {
        let canonical = Presentation::canonical();
        let renamed = Presentation::renamed();
        let start = witness_start();
        let [blocked_goal, free_goal] = witness_goals();
        Self {
            cases: vec![
                case(
                    "witness-goal-occupied",
                    CaseKind::Witness,
                    start,
                    blocked_goal,
                    true,
                    &canonical,
                ),
                case(
                    "witness-goal-free",
                    CaseKind::Witness,
                    start,
                    free_goal,
                    true,
                    &canonical,
                ),
                case(
                    "fixed-goal-occupied",
                    CaseKind::FixedGoalControl,
                    start,
                    blocked_goal,
                    true,
                    &canonical,
                ),
                case(
                    "state-predicts-blocker-in-1",
                    CaseKind::StatePredictsGoalControl,
                    Occupancy::new(0, 1).expect("legal"),
                    1,
                    true,
                    &canonical,
                ),
                case(
                    "state-predicts-blocker-in-2",
                    CaseKind::StatePredictsGoalControl,
                    Occupancy::new(0, 2).expect("legal"),
                    2,
                    true,
                    &canonical,
                ),
                case(
                    "hidden-goal-occupied",
                    CaseKind::HiddenGoalLeakageCheck,
                    start,
                    blocked_goal,
                    false,
                    &canonical,
                ),
                case(
                    "hidden-goal-free",
                    CaseKind::HiddenGoalLeakageCheck,
                    start,
                    free_goal,
                    false,
                    &canonical,
                ),
                case(
                    "renamed-witness-goal-occupied",
                    CaseKind::RenamedWitness,
                    start,
                    blocked_goal,
                    true,
                    &renamed,
                ),
                case(
                    "renamed-witness-goal-free",
                    CaseKind::RenamedWitness,
                    start,
                    free_goal,
                    true,
                    &renamed,
                ),
            ],
        }
    }

    pub fn cases_of_kind(&self, kind: CaseKind) -> impl Iterator<Item = &EvictionCase> {
        self.cases.iter().filter(move |case| case.kind == kind)
    }
}

fn case(
    id: &str,
    kind: CaseKind,
    start: Occupancy,
    goal: u8,
    goal_visible: bool,
    presentation: &Presentation,
) -> EvictionCase {
    EvictionCase {
        id: id.into(),
        kind,
        start,
        goal,
        goal_visible,
        presentation: presentation.clone(),
    }
}

// ---------------------------------------------------------------------------
// Reference policies
// ---------------------------------------------------------------------------

/// A policy chooses a command channel, or `None` to hold.
pub trait PublicPolicy {
    fn choose_actuator_key(&self, observation: &PublicObservation) -> Option<u16>;
}

/// Reads the requested outcome and the current situation, and compares them.
///
/// If the goal container is occupied by the blocker, clear it. Otherwise move
/// the item. This is the relation the process is built to test: the answer is
/// neither a copy of the goal nor a function of the situation alone.
#[derive(Debug, Clone, Copy)]
pub struct GoalAwarePolicy;

impl PublicPolicy for GoalAwarePolicy {
    fn choose_actuator_key(&self, observation: &PublicObservation) -> Option<u16> {
        let item = observation.container_of_item_key(observation.item_here_key)?;
        let blocker = observation.container_of_blocker_key(observation.blocker_here_key)?;
        let Some(goal_key) = observation.goal_key else {
            // With no requested outcome there is nothing to compare against.
            // Moving the item is the best a public policy can do, and the
            // enumerated ceiling says how far that gets.
            return Some(item.evict_key);
        };
        let goal = observation.container_of_item_key(goal_key)?;
        if goal.item_key == item.item_key {
            return None;
        }
        if goal.blocker_key == blocker.blocker_key {
            Some(goal.evict_key)
        } else {
            Some(item.evict_key)
        }
    }
}

/// Ignores the requested outcome. Clears the blocker, then moves the item.
///
/// This solves the control in which the situation predicts the goal, and cannot
/// exceed the enumerated ceiling on the witness pair.
#[derive(Debug, Clone, Copy)]
pub struct StateOnlyPolicy;

impl PublicPolicy for StateOnlyPolicy {
    fn choose_actuator_key(&self, observation: &PublicObservation) -> Option<u16> {
        let item = observation.container_of_item_key(observation.item_here_key)?;
        let blocker = observation.container_of_blocker_key(observation.blocker_here_key)?;
        if observation.steps_remaining == HORIZON {
            Some(blocker.evict_key)
        } else {
            Some(item.evict_key)
        }
    }
}

/// Computes the correct relation, but recovers the container binding from
/// arithmetic between key numbers instead of from the published schema.
///
/// Under the canonical naming the command key is the item key plus twenty and
/// the blocker key is the item key plus ten. That is an accident of the naming,
/// not a public fact, and a consistent renaming destroys it. This is the
/// analogue of the first process's stable-key shortcut, and it exists so the
/// suite can demonstrate a policy that looks correct until the names change.
#[derive(Debug, Clone, Copy)]
pub struct KeyArithmeticShortcutPolicy;

impl PublicPolicy for KeyArithmeticShortcutPolicy {
    fn choose_actuator_key(&self, observation: &PublicObservation) -> Option<u16> {
        let item_container_key = observation.item_here_key;
        let blocker_container_key = observation.blocker_here_key.checked_sub(10)?;
        let Some(goal_key) = observation.goal_key else {
            return Some(item_container_key + 20);
        };
        if goal_key == item_container_key {
            return None;
        }
        if goal_key == blocker_container_key {
            Some(goal_key + 20)
        } else {
            Some(item_container_key + 20)
        }
    }
}

/// Always evicts whichever container was published first.
///
/// It reads no goal and no observation, so it is both a constant baseline and
/// the policy that exposes a presentation-order shortcut.
#[derive(Debug, Clone, Copy)]
pub struct FirstDeclaredContainerPolicy;

impl PublicPolicy for FirstDeclaredContainerPolicy {
    fn choose_actuator_key(&self, observation: &PublicObservation) -> Option<u16> {
        observation
            .containers
            .first()
            .map(|container| container.evict_key)
    }
}

/// Evicts the goal container, whatever is in it.
///
/// This is the natural wrong answer: it treats the requested outcome as the
/// answer rather than as one side of a comparison. It is included because a
/// process whose goal can simply be copied would not test the relation at all.
#[derive(Debug, Clone, Copy)]
pub struct CopyTheGoalPolicy;

impl PublicPolicy for CopyTheGoalPolicy {
    fn choose_actuator_key(&self, observation: &PublicObservation) -> Option<u16> {
        let goal_key = observation.goal_key?;
        observation
            .container_of_item_key(goal_key)
            .map(|container| container.evict_key)
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Score {
    pub successes: usize,
    pub cases: usize,
    pub rate: f64,
}

impl Score {
    fn from_counts(successes: usize, cases: usize) -> Self {
        Self {
            successes,
            cases,
            rate: if cases == 0 {
                0.0
            } else {
                successes as f64 / cases as f64
            },
        }
    }
}

/// The same evidence shape the first process reports, so the two processes can
/// be read side by side without translating field names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyEvidence {
    pub witness: Score,
    pub fixed_goal_control: Score,
    pub state_predicts_goal_control: Score,
    pub hidden_goal_check: Score,
    pub renamed_witness: Score,
    pub counterfactual_goal_sensitivity: f64,
    /// Fraction of cases whose complete command sequence is unchanged when the
    /// simultaneous schema facts and command channels are published in a
    /// different order. Measured, not assumed.
    pub presentation_order_invariance: f64,
}

pub fn evaluate_policy<P: PublicPolicy>(suite: &EvictionSuite, policy: &P) -> PolicyEvidence {
    PolicyEvidence {
        witness: score_kind(suite, policy, CaseKind::Witness),
        fixed_goal_control: score_kind(suite, policy, CaseKind::FixedGoalControl),
        state_predicts_goal_control: score_kind(suite, policy, CaseKind::StatePredictsGoalControl),
        hidden_goal_check: score_kind(suite, policy, CaseKind::HiddenGoalLeakageCheck),
        renamed_witness: score_kind(suite, policy, CaseKind::RenamedWitness),
        counterfactual_goal_sensitivity: counterfactual_goal_sensitivity(suite, policy),
        presentation_order_invariance: presentation_order_invariance(suite, policy),
    }
}

fn score_kind<P: PublicPolicy>(suite: &EvictionSuite, policy: &P, kind: CaseKind) -> Score {
    let cases: Vec<_> = suite.cases_of_kind(kind).collect();
    let successes = cases
        .iter()
        .filter(|case| run_public_policy(case, policy, SerializationOrder::Canonical))
        .count();
    Score::from_counts(successes, cases.len())
}

/// Run a policy against one case and report whether the item reached the goal.
pub fn run_public_policy<P: PublicPolicy>(
    case: &EvictionCase,
    policy: &P,
    order: SerializationOrder,
) -> bool {
    policy_commands(case, policy, order)
        .map(|commands| sequence_succeeds(case.start, case.goal, &commands))
        .unwrap_or(false)
}

/// The complete command sequence a policy produces, or `None` if it ever named a
/// channel this presentation does not declare.
pub fn policy_commands<P: PublicPolicy>(
    case: &EvictionCase,
    policy: &P,
    order: SerializationOrder,
) -> Option<Vec<Command>> {
    let mut state = case.start;
    let mut commands = Vec::new();
    for step in 0..HORIZON {
        if state.item == case.goal {
            break;
        }
        let observation = case.public_observation(state, HORIZON - step, order);
        let command = match policy.choose_actuator_key(&observation) {
            None => Command::Hold,
            Some(key) => Command::Evict(case.presentation.container_of_evict_key(key)?),
        };
        commands.push(command);
        state = transition(state, command);
    }
    Some(commands)
}

fn counterfactual_goal_sensitivity<P: PublicPolicy>(suite: &EvictionSuite, policy: &P) -> f64 {
    let cases: Vec<_> = suite.cases_of_kind(CaseKind::Witness).collect();
    assert_eq!(cases.len(), 2, "the standard witness is a goal pair");
    let choices: Vec<Option<Command>> = cases
        .iter()
        .map(|case| {
            let observation =
                case.public_observation(case.start, HORIZON, SerializationOrder::Canonical);
            policy.choose_actuator_key(&observation).and_then(|key| {
                case.presentation
                    .container_of_evict_key(key)
                    .map(Command::Evict)
            })
        })
        .collect();

    let each_correct = cases.iter().zip(&choices).all(|(case, choice)| {
        choice
            .is_some_and(|command| optimal_first_commands(case.start, case.goal).contains(&command))
    });
    let changed = choices[0] != choices[1];
    if each_correct && changed {
        1.0
    } else {
        0.0
    }
}

fn presentation_order_invariance<P: PublicPolicy>(suite: &EvictionSuite, policy: &P) -> f64 {
    let mut invariant = 0usize;
    for case in &suite.cases {
        let canonical = policy_commands(case, policy, SerializationOrder::Canonical);
        let permuted = policy_commands(case, policy, SerializationOrder::Permuted);
        if canonical == permuted {
            invariant += 1;
        }
    }
    invariant as f64 / suite.cases.len() as f64
}

// ---------------------------------------------------------------------------
// Serialization into the existing public-event layout
// ---------------------------------------------------------------------------

fn event_payload(value: f32, lower: f32, upper: f32, aux0: f32, aux1: f32) -> [f32; PAYLOAD_DIM] {
    [value, lower, upper, aux0, aux1, 1.0, 0.0, 0.0]
}

/// A live episode encoded in the same structured public-event layout the learner
/// already consumes.
///
/// The first process rendered its three moves as one bounded control whose sign
/// meant direction. This process renders its commands as one channel per
/// container, actuated at or above a threshold. Nothing about the action space
/// is ordered, and no channel is the negation of another.
#[derive(Debug, Clone)]
pub struct EvictionRollout {
    case: EvictionCase,
    order: SerializationOrder,
    tokens: Vec<LearningToken>,
    state: Occupancy,
    steps: u8,
    done: bool,
    success: bool,
    commands: Vec<Command>,
    next_event: u16,
}

impl EvictionRollout {
    pub fn new(case: EvictionCase, order: SerializationOrder) -> Self {
        let state = case.start;
        let mut rollout = Self {
            case,
            order,
            tokens: Vec::new(),
            state,
            steps: 0,
            done: false,
            success: false,
            commands: Vec::new(),
            next_event: 0,
        };
        rollout.append_schema();
        rollout.append_event(vec![(
            Role::Boundary,
            0,
            event_payload(BOUNDARY_TASK_RESET, -1.0, 1.0, 0.0, 0.0),
        )]);
        if rollout.case.goal_visible {
            let goal_key = rollout
                .case
                .presentation
                .item_key(rollout.case.goal_for_verification());
            rollout.append_event(vec![(
                Role::Goal,
                goal_key,
                event_payload(1.0, 0.0, 1.0, 0.0, 0.0),
            )]);
        }
        rollout.success = rollout.state.item == rollout.case.goal_for_verification();
        rollout.append_observation();
        if rollout.success {
            rollout.done = true;
            rollout.append_end();
        } else {
            rollout.append_action_query();
        }
        rollout
    }

    pub fn tokens(&self) -> &[LearningToken] {
        &self.tokens
    }

    pub fn case(&self) -> &EvictionCase {
        &self.case
    }

    pub fn order(&self) -> SerializationOrder {
        self.order
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn success(&self) -> bool {
        self.success
    }

    pub fn commands(&self) -> &[Command] {
        &self.commands
    }

    /// The current situation. This is public: every occupancy it reports has
    /// already been serialized as an observation record.
    pub fn state(&self) -> Occupancy {
        self.state
    }

    /// Control steps taken so far.
    pub fn steps(&self) -> u8 {
        self.steps
    }

    /// The token positions of the open action query, in publication order.
    ///
    /// A caller supplies one normalized value per position, in this order.
    pub fn current_query_positions(&self) -> Vec<usize> {
        if self.done {
            return Vec::new();
        }
        let last_event = self
            .tokens
            .iter()
            .rev()
            .find(|token| token.public.role == Role::ActionQuery)
            .map(|token| token.public.event);
        let Some(event) = last_event else {
            return Vec::new();
        };
        self.tokens
            .iter()
            .enumerate()
            .filter(|(_, token)| {
                token.public.role == Role::ActionQuery && token.public.event == event
            })
            .map(|(position, _)| position)
            .collect()
    }

    /// Advance one step from one normalized value per queried command channel.
    pub fn step_normalized(&mut self, normalized: &[f32]) -> Result<Command, String> {
        if self.done {
            return Err("cannot step a completed eviction episode".into());
        }
        let positions = self.current_query_positions();
        if normalized.len() != positions.len() {
            return Err(format!(
                "expected {} command values, received {}",
                positions.len(),
                normalized.len()
            ));
        }
        if normalized.iter().any(|value| !value.is_finite()) {
            return Err("every command value must be finite".into());
        }

        let actuated: Vec<u8> = positions
            .iter()
            .zip(normalized)
            .filter(|(_, value)| **value >= COMMAND_THRESHOLD)
            .map(|(position, _)| {
                self.case
                    .presentation
                    .container_of_evict_key(self.tokens[*position].public.key)
                    .expect("a query names a declared command channel")
            })
            .collect();
        let command = match actuated.len() {
            0 => Command::Hold,
            1 => Command::Evict(actuated[0]),
            _ => Command::Overreach,
        };
        self.commands.push(command);

        self.append_executed(command);
        self.state = transition(self.state, command);
        self.steps += 1;
        if self.state.item == self.case.goal_for_verification() {
            self.success = true;
        }
        self.append_observation();
        self.done = self.success || self.steps >= HORIZON;
        if self.done {
            self.append_end();
        } else {
            self.append_action_query();
        }
        Ok(command)
    }

    fn append_schema(&mut self) {
        let schema = self.case.presentation.schema();
        for container in self.order.group_order() {
            let entry = schema[container];
            let mut records = vec![
                (
                    Role::SchemaObservation,
                    entry.item_key,
                    event_payload(ITEM_BAND_TAG, -1.0, 1.0, 0.0, 0.0),
                ),
                (
                    Role::SchemaObservation,
                    entry.blocker_key,
                    event_payload(BLOCKER_BAND_TAG, -1.0, 1.0, 0.0, 0.0),
                ),
                (
                    Role::SchemaActuator,
                    entry.evict_key,
                    event_payload(0.0, -1.0, 1.0, 1.0, 0.0),
                ),
            ];
            if self.order.reverse_within_group() {
                records.reverse();
            }
            self.append_event(records);
        }
    }

    fn append_observation(&mut self) {
        let mut records = vec![
            (
                Role::Observation,
                self.case.presentation.item_key(self.state.item),
                event_payload(1.0, 0.0, 1.0, 0.0, 0.0),
            ),
            (
                Role::Observation,
                self.case.presentation.blocker_key(self.state.blocker),
                event_payload(1.0, 0.0, 1.0, 0.0, 0.0),
            ),
        ];
        if self.order.reverse_within_group() {
            records.reverse();
        }
        self.append_event(records);
    }

    fn append_action_query(&mut self) {
        let remaining = f32::from(HORIZON - self.steps) / f32::from(HORIZON);
        let records = self
            .order
            .channel_order()
            .into_iter()
            .map(|container| {
                (
                    Role::ActionQuery,
                    self.case.presentation.evict_key(container as u8),
                    event_payload(0.0, -1.0, 1.0, remaining, 0.0),
                )
            })
            .collect();
        self.append_event(records);
    }

    fn append_executed(&mut self, command: Command) {
        let records = self
            .order
            .channel_order()
            .into_iter()
            .map(|container| {
                let actuated =
                    matches!(command, Command::Evict(chosen) if chosen as usize == container);
                (
                    Role::ActionExecuted,
                    self.case.presentation.evict_key(container as u8),
                    event_payload(if actuated { 1.0 } else { 0.0 }, -1.0, 1.0, 1.0, 0.0),
                )
            })
            .collect();
        self.append_event(records);
    }

    fn append_end(&mut self) {
        self.append_event(vec![(
            Role::Boundary,
            0,
            event_payload(BOUNDARY_EPISODE_END, -1.0, 1.0, 0.0, 0.0),
        )]);
    }

    fn append_event(&mut self, records: Vec<(Role, u16, [f32; PAYLOAD_DIM])>) {
        let event = self.next_event;
        self.next_event = self
            .next_event
            .checked_add(1)
            .expect("eviction event counter overflow");
        self.tokens.extend(
            records
                .into_iter()
                .map(|(role, key, payload)| LearningToken {
                    public: PublicToken {
                        role,
                        key,
                        event,
                        payload,
                    },
                    supervision: Supervision::default(),
                }),
        );
    }
}

pub fn standard_eviction_rollouts(order: SerializationOrder) -> Vec<EvictionRollout> {
    EvictionSuite::standard()
        .cases
        .into_iter()
        .map(|case| EvictionRollout::new(case, order))
        .collect()
}

// ---------------------------------------------------------------------------
// The reference audit
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceSummary {
    pub observation_key_bands: Vec<String>,
    pub actuator_channels: usize,
    pub action_geometry: String,
    pub goal_form: String,
    pub schema_form: String,
    pub declared_numeric_encoding: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceAudit {
    pub process: String,
    pub process_version: String,
    pub containers: usize,
    pub horizon: u8,
    pub semantic_command_alphabet: Vec<String>,
    pub enumerated_sequences: usize,
    pub exhaustive_public_hidden_goal_ceiling: f64,
    pub optimal_first_commands: BTreeMap<String, Vec<String>>,
    pub policy_evidence: BTreeMap<String, PolicyEvidence>,
    pub surface: SurfaceSummary,
    pub not_claimed: Vec<String>,
}

pub fn reference_audit() -> ReferenceAudit {
    let suite = EvictionSuite::standard();
    let mut optimal = BTreeMap::new();
    for case in &suite.cases {
        optimal.insert(
            case.id.clone(),
            optimal_first_commands(case.start, case.goal)
                .into_iter()
                .map(Command::as_str)
                .collect(),
        );
    }

    ReferenceAudit {
        process: "three containers, one tracked item, one blocker, one free container; \
                  the same situation with two different requested outcomes"
            .into(),
        process_version: PROCESS_VERSION.into(),
        containers: CONTAINER_COUNT,
        horizon: HORIZON,
        semantic_command_alphabet: Command::alphabet()
            .into_iter()
            .map(Command::as_str)
            .collect(),
        enumerated_sequences: enumerated_sequences().len(),
        exhaustive_public_hidden_goal_ceiling: hidden_goal_public_ceiling(),
        optimal_first_commands: optimal,
        policy_evidence: BTreeMap::from([
            (
                "copy_the_goal".into(),
                evaluate_policy(&suite, &CopyTheGoalPolicy),
            ),
            (
                "first_declared_container".into(),
                evaluate_policy(&suite, &FirstDeclaredContainerPolicy),
            ),
            (
                "goal_aware".into(),
                evaluate_policy(&suite, &GoalAwarePolicy),
            ),
            (
                "key_arithmetic_shortcut".into(),
                evaluate_policy(&suite, &KeyArithmeticShortcutPolicy),
            ),
            (
                "state_only".into(),
                evaluate_policy(&suite, &StateOnlyPolicy),
            ),
        ]),
        surface: SurfaceSummary {
            observation_key_bands: vec!["item-in-container".into(), "blocker-in-container".into()],
            actuator_channels: CONTAINER_COUNT,
            action_geometry: "categorical: one command channel per container, actuated at or \
                              above a threshold; no channel is between or opposite to another"
                .into(),
            goal_form: "one selected item-band key naming the container the item must end in"
                .into(),
            schema_form: "one simultaneity group per container binding its item key, blocker \
                          key, and command channel; the binding is co-membership, not arithmetic"
                .into(),
            declared_numeric_encoding: "the schema value slot tags the item band as +1.0 and the \
                                        blocker band as -1.0; this is the one categorical fact \
                                        given a numeric encoding, and it is declared rather than \
                                        hidden"
                .into(),
        },
        not_claimed: vec![
            "no learner was run, and no training of any kind was performed".into(),
            "no transfer claim: a second process that shares a relation is an instrument for \
             testing transfer, not evidence of it"
                .into(),
            "no claim that this process is matched in difficulty to the first; only that its \
             surface vocabulary is disjoint and its goal-blind ceiling is identical"
                .into(),
            "no claim about which representation a learner should receive".into(),
        ],
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn witness_pair() -> (EvictionCase, EvictionCase) {
        let suite = EvictionSuite::standard();
        let cases: Vec<_> = suite.cases_of_kind(CaseKind::Witness).cloned().collect();
        (cases[0].clone(), cases[1].clone())
    }

    #[test]
    fn the_witnesses_share_a_situation_and_differ_only_in_the_requested_outcome() {
        let (left, right) = witness_pair();
        assert_eq!(left.start, right.start);
        assert_eq!(left.presentation, right.presentation);
        assert!(left.goal_visible && right.goal_visible);
        assert_ne!(left.goal_for_verification(), right.goal_for_verification());
    }

    #[test]
    fn the_correct_first_command_changes_when_only_the_goal_changes() {
        let (left, right) = witness_pair();
        let left_first = optimal_first_commands(left.start, left.goal_for_verification());
        let right_first = optimal_first_commands(right.start, right.goal_for_verification());

        // When the requested container is occupied, exactly one first command
        // can begin a solution. When it is free, one step suffices, so the
        // second step is slack and three first commands survive. That asymmetry
        // is intrinsic to eviction dynamics and is recorded rather than tuned
        // away: with a free destination the item can always be moved directly.
        assert_eq!(left_first, vec![Command::Evict(1)]);
        assert_eq!(
            right_first,
            vec![Command::Hold, Command::Evict(0), Command::Evict(2)]
        );

        // What the paired contrast needs is that the two sets are disjoint, so
        // no single first command can begin a solution to both.
        assert!(left_first
            .iter()
            .all(|command| !right_first.contains(command)));
        assert!(!left_first.is_empty() && !right_first.is_empty());
    }

    #[test]
    fn the_first_command_contrast_is_necessary_but_not_sufficient() {
        // Evicting the requested container is an acceptable *first* command for
        // both witnesses, and the two choices differ, so a policy that simply
        // copies the goal satisfies the paired first-action counterfactual.
        // It nevertheless solves neither witness, because the copy is wrong at
        // the second step in both. The first-action contrast is therefore a
        // necessary condition, not a sufficient one, and a gate that reports it
        // alone can be passed by a policy that never solves anything.
        let (left, right) = witness_pair();
        for case in [&left, &right] {
            assert!(
                optimal_first_commands(case.start, case.goal_for_verification())
                    .contains(&Command::Evict(case.goal_for_verification())),
                "{} does not admit the copied goal as a first command",
                case.id
            );
        }

        let evidence = evaluate_policy(&EvictionSuite::standard(), &CopyTheGoalPolicy);
        assert_eq!(evidence.counterfactual_goal_sensitivity, 1.0);
        assert_eq!(evidence.witness.rate, 0.0);

        // The conjunction is what rejects it: the first process's gate requires
        // the witness score and the counterfactual together, and the witness
        // score is what fails here.
        assert!(evidence.witness.rate < GOAL_BLIND_CEILING);
    }

    #[test]
    fn the_hidden_goal_ceiling_is_exactly_one_half_by_enumeration() {
        assert_eq!(enumerated_sequences().len(), 16);
        assert_eq!(hidden_goal_public_ceiling(), GOAL_BLIND_CEILING);
        // No sequence solves both requested outcomes.
        let start = witness_start();
        for sequence in enumerated_sequences() {
            let solved = witness_goals()
                .iter()
                .filter(|goal| sequence_succeeds(start, **goal, &sequence))
                .count();
            assert!(solved <= 1, "{sequence:?} solved both goals");
        }
    }

    #[test]
    fn every_case_is_solvable_by_some_enumerated_sequence() {
        for case in &EvictionSuite::standard().cases {
            assert!(
                enumerated_sequences().into_iter().any(|sequence| {
                    sequence_succeeds(case.start, case.goal_for_verification(), &sequence)
                }),
                "{} has no solution",
                case.id
            );
        }
    }

    #[test]
    fn the_goal_aware_policy_solves_every_visible_case_and_stops_at_the_ceiling() {
        let evidence = evaluate_policy(&EvictionSuite::standard(), &GoalAwarePolicy);
        assert_eq!(evidence.witness.rate, 1.0);
        assert_eq!(evidence.fixed_goal_control.rate, 1.0);
        assert_eq!(evidence.state_predicts_goal_control.rate, 1.0);
        assert_eq!(evidence.renamed_witness.rate, 1.0);
        assert_eq!(evidence.hidden_goal_check.rate, GOAL_BLIND_CEILING);
        assert_eq!(evidence.counterfactual_goal_sensitivity, 1.0);
        assert_eq!(evidence.presentation_order_invariance, 1.0);
    }

    #[test]
    fn the_state_only_policy_is_exposed_by_the_paired_witness() {
        let evidence = evaluate_policy(&EvictionSuite::standard(), &StateOnlyPolicy);
        assert_eq!(evidence.state_predicts_goal_control.rate, 1.0);
        assert_eq!(evidence.witness.rate, GOAL_BLIND_CEILING);
        assert_eq!(evidence.counterfactual_goal_sensitivity, 0.0);
        assert!(evidence.hidden_goal_check.rate <= GOAL_BLIND_CEILING);
    }

    #[test]
    fn the_key_arithmetic_shortcut_solves_the_canonical_cases_and_fails_after_renaming() {
        let evidence = evaluate_policy(&EvictionSuite::standard(), &KeyArithmeticShortcutPolicy);
        assert_eq!(evidence.witness.rate, 1.0);
        assert_eq!(evidence.counterfactual_goal_sensitivity, 1.0);
        assert_eq!(
            evidence.renamed_witness.rate, 0.0,
            "a consistent renaming must destroy the key arithmetic"
        );
    }

    #[test]
    fn copying_the_goal_cannot_solve_the_paired_witness() {
        let evidence = evaluate_policy(&EvictionSuite::standard(), &CopyTheGoalPolicy);
        assert!(
            evidence.witness.rate <= GOAL_BLIND_CEILING,
            "the goal must not be copyable into the answer"
        );
    }

    #[test]
    fn a_policy_that_reads_the_first_declared_container_is_exposed_by_reordering() {
        let evidence = evaluate_policy(&EvictionSuite::standard(), &FirstDeclaredContainerPolicy);
        assert!(evidence.witness.rate <= GOAL_BLIND_CEILING);
        assert!(
            evidence.presentation_order_invariance < 1.0,
            "the order check must actually catch an order-sensitive policy"
        );
    }

    #[test]
    fn presentation_order_is_measured_rather_than_assumed() {
        // The correct policy is order invariant and a positional one is not, so
        // the measurement separates them instead of reporting a constant.
        let suite = EvictionSuite::standard();
        assert_eq!(presentation_order_invariance(&suite, &GoalAwarePolicy), 1.0);
        assert!(presentation_order_invariance(&suite, &FirstDeclaredContainerPolicy) < 1.0);
    }

    #[test]
    fn no_hidden_goal_case_serializes_a_goal_record() {
        for rollout in standard_eviction_rollouts(SerializationOrder::Canonical) {
            let has_goal = rollout
                .tokens()
                .iter()
                .any(|token| token.public.role == Role::Goal);
            assert_eq!(
                has_goal,
                rollout.case().goal_visible,
                "{} serialized the wrong goal visibility",
                rollout.case().id
            );
        }
    }

    #[test]
    fn opposite_goals_differ_only_in_the_goal_record() {
        let suite = EvictionSuite::standard();
        let cases: Vec<_> = suite.cases_of_kind(CaseKind::Witness).collect();
        let left = EvictionRollout::new(cases[0].clone(), SerializationOrder::Canonical);
        let right = EvictionRollout::new(cases[1].clone(), SerializationOrder::Canonical);
        assert_eq!(left.tokens().len(), right.tokens().len());
        let differing: Vec<_> = left
            .tokens()
            .iter()
            .zip(right.tokens())
            .filter(|(a, b)| a.public != b.public)
            .collect();
        assert_eq!(differing.len(), 1);
        assert_eq!(differing[0].0.public.role, Role::Goal);
    }

    #[test]
    fn hidden_goal_pairs_serialize_identical_public_records() {
        let suite = EvictionSuite::standard();
        let cases: Vec<_> = suite
            .cases_of_kind(CaseKind::HiddenGoalLeakageCheck)
            .collect();
        let left = EvictionRollout::new(cases[0].clone(), SerializationOrder::Canonical);
        let right = EvictionRollout::new(cases[1].clone(), SerializationOrder::Canonical);
        let public = |rollout: &EvictionRollout| -> Vec<PublicToken> {
            rollout
                .tokens()
                .iter()
                .map(|token| token.public.clone())
                .collect()
        };
        assert_eq!(
            public(&left),
            public(&right),
            "a hidden goal must leave no public trace"
        );
    }

    #[test]
    fn reordering_changes_presentation_and_not_the_fact_set() {
        let case = EvictionSuite::standard().cases[0].clone();
        let canonical = EvictionRollout::new(case.clone(), SerializationOrder::Canonical);
        let permuted = EvictionRollout::new(case, SerializationOrder::Permuted);

        // A fact is a role, a key, and a payload. Which simultaneity group it
        // sits in is structure; the group's numeric index and the order of the
        // groups are presentation, so neither belongs in the comparison.
        let fact = |token: &LearningToken| {
            format!(
                "{:?}|{}|{:?}",
                token.public.role, token.public.key, token.public.payload
            )
        };
        let all_facts = |rollout: &EvictionRollout| {
            let mut facts: Vec<String> = rollout.tokens().iter().map(fact).collect();
            facts.sort();
            facts
        };
        let grouped_facts = |rollout: &EvictionRollout| {
            let mut groups: BTreeMap<u16, Vec<String>> = BTreeMap::new();
            for token in rollout.tokens() {
                groups
                    .entry(token.public.event)
                    .or_default()
                    .push(fact(token));
            }
            let mut partition: Vec<Vec<String>> = groups
                .into_values()
                .map(|mut facts| {
                    facts.sort();
                    facts
                })
                .collect();
            partition.sort();
            partition
        };
        assert_ne!(
            canonical
                .tokens()
                .iter()
                .map(|token| token.public.key)
                .collect::<Vec<_>>(),
            permuted
                .tokens()
                .iter()
                .map(|token| token.public.key)
                .collect::<Vec<_>>(),
            "the permuted order must actually differ"
        );
        assert_eq!(
            all_facts(&canonical),
            all_facts(&permuted),
            "reordering must not change which facts are published"
        );
        assert_eq!(
            grouped_facts(&canonical),
            grouped_facts(&permuted),
            "reordering must not change which facts are simultaneous"
        );
    }

    #[test]
    fn a_rollout_ends_on_success_or_on_the_horizon() {
        for order in [SerializationOrder::Canonical, SerializationOrder::Permuted] {
            for mut rollout in standard_eviction_rollouts(order) {
                let mut steps = 0;
                while !rollout.is_done() {
                    let positions = rollout.current_query_positions();
                    assert_eq!(positions.len(), CONTAINER_COUNT);
                    let commands = vec![0.0f32; CONTAINER_COUNT];
                    rollout.step_normalized(&commands).expect("a live step");
                    steps += 1;
                    assert!(steps <= HORIZON);
                }
                assert!(rollout.commands().len() <= HORIZON as usize);
                assert!(rollout
                    .tokens()
                    .last()
                    .is_some_and(|token| token.public.role == Role::Boundary));
            }
        }
    }

    #[test]
    fn the_goal_aware_command_sequence_drives_a_rollout_to_success() {
        for case in EvictionSuite::standard().cases {
            if !case.goal_visible {
                continue;
            }
            let mut rollout = EvictionRollout::new(case.clone(), SerializationOrder::Canonical);
            while !rollout.is_done() {
                let positions = rollout.current_query_positions();
                let observation = rollout.case.public_observation(
                    rollout.state,
                    HORIZON - rollout.steps,
                    rollout.order,
                );
                let chosen = GoalAwarePolicy.choose_actuator_key(&observation);
                let commands: Vec<f32> = positions
                    .iter()
                    .map(|position| {
                        let key = rollout.tokens[*position].public.key;
                        if Some(key) == chosen {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect();
                rollout.step_normalized(&commands).expect("a live step");
            }
            assert!(rollout.success(), "{} was not solved", case.id);
        }
    }

    #[test]
    fn actuating_two_channels_is_refused_and_recorded_as_such() {
        let case = EvictionSuite::standard().cases[0].clone();
        let mut rollout = EvictionRollout::new(case, SerializationOrder::Canonical);
        let before = rollout.state;
        let command = rollout
            .step_normalized(&[1.0, 1.0, 0.0])
            .expect("a live step");
        assert_eq!(command, Command::Overreach);
        assert_eq!(rollout.state, before, "a refused step changes nothing");
        assert_ne!(
            command,
            Command::Hold,
            "an overreach must not be reported as a hold"
        );
    }

    #[test]
    fn evicting_an_empty_container_changes_nothing() {
        let state = witness_start();
        assert_eq!(transition(state, Command::Evict(state.free())), state);
        assert_eq!(transition(state, Command::Hold), state);
        assert_eq!(transition(state, Command::Overreach), state);
    }

    #[test]
    fn eviction_preserves_the_single_free_container() {
        for item in 0..CONTAINER_COUNT as u8 {
            for blocker in 0..CONTAINER_COUNT as u8 {
                let Ok(state) = Occupancy::new(item, blocker) else {
                    continue;
                };
                for command in Command::alphabet() {
                    let next = transition(state, command);
                    assert_ne!(next.item, next.blocker);
                    assert!(Occupancy::new(next.item, next.blocker).is_ok());
                }
            }
        }
    }

    #[test]
    fn no_supervision_is_written_into_a_public_payload() {
        for rollout in standard_eviction_rollouts(SerializationOrder::Canonical) {
            for token in rollout.tokens() {
                assert_eq!(token.supervision, Supervision::default());
                assert_eq!(token.public.payload[5], 1.0);
                assert_eq!(token.public.payload[6], 0.0);
                assert_eq!(token.public.payload[7], 0.0);
            }
            // An action query never carries the command it is asking for.
            for token in rollout
                .tokens()
                .iter()
                .filter(|token| token.public.role == Role::ActionQuery)
            {
                assert_eq!(token.public.payload[0], 0.0);
            }
        }
    }

    #[test]
    fn the_audit_reports_the_enumerated_facts() {
        let audit = reference_audit();
        assert_eq!(audit.enumerated_sequences, 16);
        assert_eq!(audit.exhaustive_public_hidden_goal_ceiling, 0.5);
        assert_eq!(audit.semantic_command_alphabet.len(), 4);
        assert_eq!(audit.policy_evidence.len(), 5);
        assert!(audit
            .not_claimed
            .iter()
            .any(|line| line.contains("transfer")));
        assert!(serde_json::to_string(&audit).is_ok());
    }
}
