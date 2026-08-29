//! A finite diagnostic for separating goal-conditioned behavior from shortcuts.
//!
//! This crate deliberately implements one small process rather than a general
//! world language or solver. Every reference quantity is obtained by exhaustive
//! enumeration over a five-position line and a two-action horizon.

use pretraining_world::{LearningToken, PublicToken, Role, Supervision, PAYLOAD_DIM};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const LINE_LENGTH: u8 = 5;
pub const HORIZON: u8 = 2;
pub const GOAL_BLIND_CEILING: f64 = 0.5;
pub const DIAGNOSTIC_SERIALIZATION_VERSION: &str = "goal-conditioned-continuous-control-0.1.0";

const CANONICAL_CONTROL_KEY: u16 = 30;
const RENAMED_CONTROL_KEY: u16 = 31;
const MOVE_THRESHOLD: f32 = 1.0 / 3.0;
const BOUNDARY_TASK_RESET: f32 = 0.0;
const BOUNDARY_EPISODE_END: f32 = 1.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Move {
    Left,
    Stay,
    Right,
}

impl Move {
    pub const ALL: [Self; 3] = [Self::Left, Self::Stay, Self::Right];

    fn displacement(self) -> i8 {
        match self {
            Self::Left => -1,
            Self::Stay => 0,
            Self::Right => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PositionSchema {
    pub key: u16,
    pub coordinate: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionSchema {
    pub key: u16,
    pub displacement: i8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Presentation {
    pub name: String,
    position_keys: [u16; LINE_LENGTH as usize],
    action_keys: [u16; 3],
}

impl Presentation {
    pub fn canonical() -> Self {
        Self {
            name: "canonical".into(),
            position_keys: [0, 1, 2, 3, 4],
            action_keys: [0, 1, 2],
        }
    }

    pub fn renamed() -> Self {
        Self {
            name: "renamed".into(),
            position_keys: [12, 10, 14, 11, 13],
            action_keys: [22, 20, 21],
        }
    }

    pub fn position_key(&self, coordinate: u8) -> u16 {
        self.position_keys[coordinate as usize]
    }

    pub fn action_key(&self, action: Move) -> u16 {
        self.action_keys[action_index(action)]
    }

    pub fn decode_action(&self, key: u16) -> Option<Move> {
        self.action_keys
            .iter()
            .position(|candidate| *candidate == key)
            .map(|index| Move::ALL[index])
    }

    pub fn position_schema(&self) -> [PositionSchema; LINE_LENGTH as usize] {
        std::array::from_fn(|coordinate| PositionSchema {
            key: self.position_keys[coordinate],
            coordinate: coordinate as u8,
        })
    }

    pub fn action_schema(&self) -> [ActionSchema; 3] {
        std::array::from_fn(|index| ActionSchema {
            key: self.action_keys[index],
            displacement: Move::ALL[index].displacement(),
        })
    }

    fn continuous_control_key(&self) -> u16 {
        match self.name.as_str() {
            "canonical" => CANONICAL_CONTROL_KEY,
            "renamed" => RENAMED_CONTROL_KEY,
            _ => RENAMED_CONTROL_KEY,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicObservation {
    pub position_key: u16,
    pub goal_key: Option<u16>,
    pub steps_remaining: u8,
    pub positions: [PositionSchema; LINE_LENGTH as usize],
    pub actions: [ActionSchema; 3],
}

impl PublicObservation {
    pub fn position(&self) -> Option<u8> {
        self.positions
            .iter()
            .find(|entry| entry.key == self.position_key)
            .map(|entry| entry.coordinate)
    }

    pub fn goal(&self) -> Option<u8> {
        let key = self.goal_key?;
        self.positions
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| entry.coordinate)
    }

    pub fn action_key_for_displacement(&self, displacement: i8) -> Option<u16> {
        self.actions
            .iter()
            .find(|entry| entry.displacement == displacement)
            .map(|entry| entry.key)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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
pub struct DiagnosticCase {
    pub id: String,
    pub kind: CaseKind,
    pub start: u8,
    goal: u8,
    pub goal_visible: bool,
    pub presentation: Presentation,
}

impl DiagnosticCase {
    pub fn public_observation(&self, position: u8, steps_remaining: u8) -> PublicObservation {
        PublicObservation {
            position_key: self.presentation.position_key(position),
            goal_key: self
                .goal_visible
                .then(|| self.presentation.position_key(self.goal)),
            steps_remaining,
            positions: self.presentation.position_schema(),
            actions: self.presentation.action_schema(),
        }
    }

    pub fn goal_for_verification(&self) -> u8 {
        self.goal
    }

    pub fn with_presentation(&self, presentation: Presentation) -> Self {
        let mut case = self.clone();
        case.presentation = presentation;
        case
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiagnosticSuite {
    pub cases: Vec<DiagnosticCase>,
}

impl DiagnosticSuite {
    pub fn standard() -> Self {
        let canonical = Presentation::canonical();
        let renamed = Presentation::renamed();
        Self {
            cases: vec![
                case("witness-left", CaseKind::Witness, 2, 0, true, &canonical),
                case("witness-right", CaseKind::Witness, 2, 4, true, &canonical),
                case(
                    "fixed-goal-left",
                    CaseKind::FixedGoalControl,
                    2,
                    0,
                    true,
                    &canonical,
                ),
                case(
                    "state-predicts-left",
                    CaseKind::StatePredictsGoalControl,
                    1,
                    0,
                    true,
                    &canonical,
                ),
                case(
                    "state-predicts-right",
                    CaseKind::StatePredictsGoalControl,
                    3,
                    4,
                    true,
                    &canonical,
                ),
                case(
                    "hidden-goal-left",
                    CaseKind::HiddenGoalLeakageCheck,
                    2,
                    0,
                    false,
                    &canonical,
                ),
                case(
                    "hidden-goal-right",
                    CaseKind::HiddenGoalLeakageCheck,
                    2,
                    4,
                    false,
                    &canonical,
                ),
                case(
                    "renamed-witness-left",
                    CaseKind::RenamedWitness,
                    2,
                    0,
                    true,
                    &renamed,
                ),
                case(
                    "renamed-witness-right",
                    CaseKind::RenamedWitness,
                    2,
                    4,
                    true,
                    &renamed,
                ),
            ],
        }
    }

    pub fn cases_of_kind(&self, kind: CaseKind) -> impl Iterator<Item = &DiagnosticCase> {
        self.cases.iter().filter(move |case| case.kind == kind)
    }
}

fn case(
    id: &str,
    kind: CaseKind,
    start: u8,
    goal: u8,
    goal_visible: bool,
    presentation: &Presentation,
) -> DiagnosticCase {
    DiagnosticCase {
        id: id.into(),
        kind,
        start,
        goal,
        goal_visible,
        presentation: presentation.clone(),
    }
}

pub trait PublicPolicy {
    fn choose_action_key(&self, observation: &PublicObservation) -> u16;
}

#[derive(Debug, Clone, Copy)]
pub struct GoalAwarePolicy;

impl PublicPolicy for GoalAwarePolicy {
    fn choose_action_key(&self, observation: &PublicObservation) -> u16 {
        let position = observation
            .position()
            .expect("public position key is declared");
        let goal = observation.goal().unwrap_or(0);
        let displacement = (goal as i8 - position as i8).signum();
        observation
            .action_key_for_displacement(displacement)
            .expect("all three displacements are declared")
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StateOnlyPolicy;

impl PublicPolicy for StateOnlyPolicy {
    fn choose_action_key(&self, observation: &PublicObservation) -> u16 {
        let position = observation
            .position()
            .expect("public position key is declared");
        let displacement = if position <= 2 { -1 } else { 1 };
        observation
            .action_key_for_displacement(displacement)
            .expect("left and right are declared")
    }
}

/// Deliberately ignores the public schema and assumes stable numeric keys.
#[derive(Debug, Clone, Copy)]
pub struct StableKeyShortcutPolicy;

impl PublicPolicy for StableKeyShortcutPolicy {
    fn choose_action_key(&self, observation: &PublicObservation) -> u16 {
        match observation.goal_key {
            Some(goal_key) if goal_key > observation.position_key => 2,
            _ => 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AlwaysLeftPolicy;

impl PublicPolicy for AlwaysLeftPolicy {
    fn choose_action_key(&self, observation: &PublicObservation) -> u16 {
        observation
            .action_key_for_displacement(-1)
            .expect("left is declared")
    }
}

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
            rate: successes as f64 / cases as f64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyEvidence {
    pub witness: Score,
    pub fixed_goal_control: Score,
    pub state_predicts_goal_control: Score,
    pub hidden_goal_check: Score,
    pub renamed_witness: Score,
    pub counterfactual_goal_sensitivity: f64,
    /// Fraction of semantic cases whose complete action sequence is unchanged
    /// when simultaneous schema tokens are reordered.
    #[serde(default = "unit_rate")]
    pub presentation_order_invariance: f64,
}

pub fn evaluate_policy<P: PublicPolicy>(suite: &DiagnosticSuite, policy: &P) -> PolicyEvidence {
    PolicyEvidence {
        witness: score_kind(suite, policy, CaseKind::Witness),
        fixed_goal_control: score_kind(suite, policy, CaseKind::FixedGoalControl),
        state_predicts_goal_control: score_kind(suite, policy, CaseKind::StatePredictsGoalControl),
        hidden_goal_check: score_kind(suite, policy, CaseKind::HiddenGoalLeakageCheck),
        renamed_witness: score_kind(suite, policy, CaseKind::RenamedWitness),
        counterfactual_goal_sensitivity: counterfactual_goal_sensitivity(suite, policy),
        presentation_order_invariance: 1.0,
    }
}

fn unit_rate() -> f64 {
    1.0
}

fn score_kind<P: PublicPolicy>(suite: &DiagnosticSuite, policy: &P, kind: CaseKind) -> Score {
    let cases: Vec<_> = suite.cases_of_kind(kind).collect();
    let successes = cases
        .iter()
        .filter(|case| run_public_policy(case, policy))
        .count();
    Score::from_counts(successes, cases.len())
}

pub fn run_public_policy<P: PublicPolicy>(case: &DiagnosticCase, policy: &P) -> bool {
    let mut position = case.start;
    if position == case.goal {
        return true;
    }
    for step in 0..HORIZON {
        let observation = case.public_observation(position, HORIZON - step);
        let Some(action) = case
            .presentation
            .decode_action(policy.choose_action_key(&observation))
        else {
            return false;
        };
        position = transition(position, action);
        if position == case.goal {
            return true;
        }
    }
    false
}

pub fn transition(position: u8, action: Move) -> u8 {
    (position as i8 + action.displacement()).clamp(0, LINE_LENGTH as i8 - 1) as u8
}

/// Two presentations of the same event set. Only the order of entries inside
/// a simultaneous schema event changes; event meaning and episode dynamics do
/// not. Comparing them exposes policies that use token position as a shortcut.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializationOrder {
    Canonical,
    Permuted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingPresentationArm {
    Fixed,
    Orbit,
}

impl TrainingPresentationArm {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Orbit => "orbit",
        }
    }
}

impl SerializationOrder {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Permuted => "permuted",
        }
    }
}

/// A live diagnostic episode encoded in the same structured public-event ABI
/// consumed by the pretraining learner.
///
/// The diagnostic has three discrete moves, but the established learner emits
/// one continuous value per control. We therefore expose one bounded movement
/// control and decode negative/near-zero/positive values as left/stay/right.
/// That ordered action geometry is an explicit representation choice.
#[derive(Debug, Clone)]
pub struct DiagnosticRollout {
    case: DiagnosticCase,
    order: SerializationOrder,
    tokens: Vec<LearningToken>,
    position: u8,
    steps: u8,
    done: bool,
    success: bool,
    actions: Vec<Move>,
    next_event: u16,
}

impl DiagnosticRollout {
    pub fn new(case: DiagnosticCase, order: SerializationOrder) -> Self {
        let position = case.start;
        let mut rollout = Self {
            case,
            order,
            tokens: Vec::new(),
            position,
            steps: 0,
            done: false,
            success: false,
            actions: Vec::new(),
            next_event: 0,
        };
        rollout.append_schema();
        rollout.append_event(vec![(
            Role::Boundary,
            0,
            event_payload(BOUNDARY_TASK_RESET, -1.0, 1.0, 0.0, 0.0),
        )]);
        if rollout.case.goal_visible {
            rollout.append_event(vec![(
                Role::Goal,
                rollout
                    .case
                    .presentation
                    .position_key(rollout.case.goal_for_verification()),
                event_payload(1.0, 0.0, 1.0, 0.0, 0.0),
            )]);
        }
        rollout.append_observation();
        rollout.append_action_query();
        rollout
    }

    pub fn tokens(&self) -> &[LearningToken] {
        &self.tokens
    }

    pub fn case(&self) -> &DiagnosticCase {
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

    pub fn action_displacements(&self) -> Vec<i8> {
        self.actions
            .iter()
            .map(|action| action.displacement())
            .collect()
    }

    pub fn current_query_position(&self) -> Option<usize> {
        if self.done {
            return None;
        }
        self.tokens
            .iter()
            .enumerate()
            .rev()
            .find_map(|(position, token)| {
                (token.public.role == Role::ActionQuery).then_some(position)
            })
    }

    pub fn step_normalized(&mut self, normalized: f32) -> Result<(), String> {
        if self.done {
            return Err("cannot step a completed diagnostic episode".into());
        }
        if !normalized.is_finite() {
            return Err("diagnostic action must be finite".into());
        }
        let action = decode_normalized_move(normalized);
        self.actions.push(action);
        self.append_event(vec![(
            Role::ActionExecuted,
            self.case.presentation.continuous_control_key(),
            event_payload(action.displacement() as f32, -1.0, 1.0, 1.0, 0.0),
        )]);
        self.position = transition(self.position, action);
        self.steps += 1;
        self.success = self.position == self.case.goal_for_verification();
        self.append_observation();
        self.done = self.success || self.steps >= HORIZON;
        if self.done {
            self.append_event(vec![(
                Role::Boundary,
                0,
                event_payload(BOUNDARY_EPISODE_END, -1.0, 1.0, 0.0, 0.0),
            )]);
        } else {
            self.append_action_query();
        }
        Ok(())
    }

    fn supervise_current_query(&mut self, action: Move) {
        let position = self
            .current_query_position()
            .expect("a live diagnostic episode has one action query");
        let supervision = &mut self.tokens[position].supervision;
        supervision.action_target[0] = action.displacement() as f32;
        supervision.action_mask[0] = true;
    }

    fn append_schema(&mut self) {
        let mut entries: Vec<_> = self
            .case
            .presentation
            .position_schema()
            .into_iter()
            .map(|entry| {
                (
                    Role::SchemaObservation,
                    entry.key,
                    event_payload(normalize_position(entry.coordinate), -1.0, 1.0, 0.0, 0.0),
                )
            })
            .collect();
        entries.push((
            Role::SchemaActuator,
            self.case.presentation.continuous_control_key(),
            event_payload(0.0, -1.0, 1.0, 1.0, 0.0),
        ));
        if self.order == SerializationOrder::Permuted {
            // Fixed, replay-stable permutation of six simultaneous entries.
            // This changes presentation only; all public facts are identical.
            let permutation = [2usize, 5, 4, 0, 3, 1];
            entries = permutation
                .into_iter()
                .map(|index| entries[index])
                .collect();
        }
        self.append_event(entries);
    }

    fn append_observation(&mut self) {
        self.append_event(vec![(
            Role::Observation,
            self.case.presentation.position_key(self.position),
            event_payload(1.0, 0.0, 1.0, 0.0, 0.0),
        )]);
    }

    fn append_action_query(&mut self) {
        self.append_event(vec![(
            Role::ActionQuery,
            self.case.presentation.continuous_control_key(),
            event_payload(
                0.0,
                -1.0,
                1.0,
                (HORIZON - self.steps) as f32 / HORIZON as f32,
                0.0,
            ),
        )]);
    }

    fn append_event(&mut self, entries: Vec<(Role, u16, [f32; PAYLOAD_DIM])>) {
        let event = self.next_event;
        self.next_event = self
            .next_event
            .checked_add(1)
            .expect("diagnostic event counter overflow");
        self.tokens.extend(
            entries
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

pub fn standard_diagnostic_rollouts(order: SerializationOrder) -> Vec<DiagnosticRollout> {
    DiagnosticSuite::standard()
        .cases
        .into_iter()
        .map(|case| DiagnosticRollout::new(case, order))
        .collect()
}

/// Matched teacher records for a small representation probe.
///
/// Both arms contain the same four balanced visible semantic cases twice. The fixed arm
/// repeats one presentation. The orbit arm uses that presentation once and a
/// consistently renamed plus reordered presentation once. Targets and record
/// count are therefore matched while presentation diversity changes.
pub fn teacher_training_records(arm: TrainingPresentationArm) -> Vec<Vec<LearningToken>> {
    let base_cases: Vec<_> = DiagnosticSuite::standard()
        .cases
        .into_iter()
        .filter(|case| {
            matches!(
                case.kind,
                CaseKind::Witness | CaseKind::StatePredictsGoalControl
            )
        })
        .collect();
    let mut records = Vec::with_capacity(base_cases.len() * 2);
    for case in base_cases {
        records.push(teacher_trajectory(
            case.clone(),
            SerializationOrder::Canonical,
        ));
        let (second_case, second_order) = match arm {
            TrainingPresentationArm::Fixed => (case, SerializationOrder::Canonical),
            TrainingPresentationArm::Orbit => (
                case.with_presentation(Presentation::renamed()),
                SerializationOrder::Permuted,
            ),
        };
        records.push(teacher_trajectory(second_case, second_order));
    }
    records
}

fn teacher_trajectory(case: DiagnosticCase, order: SerializationOrder) -> Vec<LearningToken> {
    let mut rollout = DiagnosticRollout::new(case, order);
    while !rollout.is_done() {
        let displacement =
            (rollout.case.goal_for_verification() as i8 - rollout.position as i8).signum();
        let action = match displacement {
            -1 => Move::Left,
            0 => Move::Stay,
            1 => Move::Right,
            _ => unreachable!(),
        };
        rollout.supervise_current_query(action);
        rollout
            .step_normalized(action.displacement() as f32)
            .expect("teacher action is finite and episode is live");
    }
    rollout.tokens
}

pub fn decode_normalized_move(normalized: f32) -> Move {
    if normalized < -MOVE_THRESHOLD {
        Move::Left
    } else if normalized > MOVE_THRESHOLD {
        Move::Right
    } else {
        Move::Stay
    }
}

fn normalize_position(position: u8) -> f32 {
    position as f32 / ((LINE_LENGTH - 1) as f32 / 2.0) - 1.0
}

fn event_payload(value: f32, lower: f32, upper: f32, aux0: f32, aux1: f32) -> [f32; PAYLOAD_DIM] {
    [value, lower, upper, aux0, aux1, 1.0, 0.0, 0.0]
}

fn counterfactual_goal_sensitivity<P: PublicPolicy>(suite: &DiagnosticSuite, policy: &P) -> f64 {
    let cases: Vec<_> = suite.cases_of_kind(CaseKind::Witness).collect();
    assert_eq!(cases.len(), 2, "the standard witness is a goal pair");
    let choices: Vec<_> = cases
        .iter()
        .map(|case| {
            let observation = case.public_observation(case.start, HORIZON);
            case.presentation
                .decode_action(policy.choose_action_key(&observation))
        })
        .collect();
    let individually_correct = cases.iter().zip(&choices).all(|(case, choice)| {
        choice.is_some_and(|action| optimal_first_moves(case).contains(&action))
    });
    if individually_correct && choices[0] != choices[1] {
        1.0
    } else {
        0.0
    }
}

pub fn optimal_first_moves(case: &DiagnosticCase) -> Vec<Move> {
    Move::ALL
        .into_iter()
        .filter(|first| {
            Move::ALL
                .into_iter()
                .any(|second| sequence_succeeds(case.start, case.goal, &[*first, second]))
        })
        .collect()
}

fn sequence_succeeds(start: u8, goal: u8, actions: &[Move]) -> bool {
    let mut position = start;
    if position == goal {
        return true;
    }
    for action in actions {
        position = transition(position, *action);
        if position == goal {
            return true;
        }
    }
    false
}

pub fn hidden_goal_public_ceiling() -> f64 {
    let goals = [0, 4];
    let mut best: f64 = 0.0;
    for first in Move::ALL {
        for second in Move::ALL {
            let successes = goals
                .iter()
                .filter(|goal| sequence_succeeds(2, **goal, &[first, second]))
                .count();
            best = best.max(successes as f64 / goals.len() as f64);
        }
    }
    best
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransferEvidence {
    pub gain_over_scratch: f64,
    pub gain_over_matched_control: f64,
    pub prior_skill_retention: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointEvidence {
    /// A cheap inner-loop measure. It may improve for invalid reasons and never
    /// decides promotion by itself.
    pub headline_metric: f64,
    pub diagnostics: PolicyEvidence,
    pub transfer: Option<TransferEvidence>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgressThresholds {
    pub minimum_headline_gain: f64,
    pub minimum_witness_gain: f64,
    pub minimum_witness_rate: f64,
    pub minimum_control_rate: f64,
    pub minimum_renamed_rate: f64,
    #[serde(default = "unit_rate")]
    pub minimum_presentation_order_invariance: f64,
    pub maximum_hidden_goal_rate: f64,
    pub minimum_transfer_advantage: f64,
    pub minimum_retention: f64,
}

impl Default for ProgressThresholds {
    fn default() -> Self {
        Self {
            minimum_headline_gain: 0.05,
            minimum_witness_gain: 0.25,
            minimum_witness_rate: 0.95,
            minimum_control_rate: 0.95,
            minimum_renamed_rate: 0.95,
            minimum_presentation_order_invariance: 0.95,
            maximum_hidden_goal_rate: GOAL_BLIND_CEILING,
            minimum_transfer_advantage: 0.10,
            minimum_retention: 0.95,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgressClass {
    InvalidEvidence,
    NoMeasuredProgress,
    FalseProgress,
    LocalMeaningfulProgress,
    LocalButNonTransferring,
    TransferBackedProgress,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromotionDecision {
    pub class: ProgressClass,
    pub accept_local_step: bool,
    pub accept_as_project_progress: bool,
    pub requires_transfer_test: bool,
    pub passed_checks: Vec<String>,
    pub failed_checks: Vec<String>,
}

pub fn classify_progress(
    previous: &CheckpointEvidence,
    candidate: &CheckpointEvidence,
    thresholds: &ProgressThresholds,
) -> PromotionDecision {
    if !valid_evidence(previous) || !valid_evidence(candidate) {
        return PromotionDecision {
            class: ProgressClass::InvalidEvidence,
            accept_local_step: false,
            accept_as_project_progress: false,
            requires_transfer_test: false,
            passed_checks: vec![],
            failed_checks: vec![
                "one or more evidence values are non-finite or outside its declared range".into(),
            ],
        };
    }

    let headline_gain = candidate.headline_metric - previous.headline_metric;
    let witness_gain = candidate.diagnostics.witness.rate - previous.diagnostics.witness.rate;
    if headline_gain < thresholds.minimum_headline_gain
        && witness_gain < thresholds.minimum_witness_gain
    {
        return PromotionDecision {
            class: ProgressClass::NoMeasuredProgress,
            accept_local_step: false,
            accept_as_project_progress: false,
            requires_transfer_test: false,
            passed_checks: vec![],
            failed_checks: vec![format!(
                "headline gain {headline_gain:.3} and witness gain {witness_gain:.3} are both below threshold"
            )],
        };
    }

    let mut passed = Vec::new();
    let mut failed = Vec::new();
    check(
        candidate.diagnostics.witness.rate >= thresholds.minimum_witness_rate,
        "solves the paired-goal witness",
        &mut passed,
        &mut failed,
    );
    check(
        witness_gain >= thresholds.minimum_witness_gain,
        "improves the paired-goal witness",
        &mut passed,
        &mut failed,
    );
    check(
        candidate.diagnostics.counterfactual_goal_sensitivity >= 1.0,
        "changes action correctly when only the goal changes",
        &mut passed,
        &mut failed,
    );
    check(
        candidate.diagnostics.fixed_goal_control.rate >= thresholds.minimum_control_rate,
        "passes the fixed-goal apparatus control",
        &mut passed,
        &mut failed,
    );
    check(
        candidate.diagnostics.state_predicts_goal_control.rate >= thresholds.minimum_control_rate,
        "passes the state-predictable apparatus control",
        &mut passed,
        &mut failed,
    );
    check(
        candidate.diagnostics.renamed_witness.rate >= thresholds.minimum_renamed_rate,
        "survives renaming of every public key",
        &mut passed,
        &mut failed,
    );
    check(
        candidate.diagnostics.presentation_order_invariance
            >= thresholds.minimum_presentation_order_invariance,
        "is unchanged when simultaneous public facts are reordered",
        &mut passed,
        &mut failed,
    );
    check(
        candidate.diagnostics.hidden_goal_check.rate
            <= thresholds.maximum_hidden_goal_rate + f64::EPSILON,
        "does not exceed the public ceiling when the goal is hidden",
        &mut passed,
        &mut failed,
    );

    if !failed.is_empty() {
        return PromotionDecision {
            class: ProgressClass::FalseProgress,
            accept_local_step: false,
            accept_as_project_progress: false,
            requires_transfer_test: false,
            passed_checks: passed,
            failed_checks: failed,
        };
    }

    let Some(transfer) = &candidate.transfer else {
        return PromotionDecision {
            class: ProgressClass::LocalMeaningfulProgress,
            accept_local_step: true,
            accept_as_project_progress: false,
            requires_transfer_test: true,
            passed_checks: passed,
            failed_checks: vec![],
        };
    };

    let transfer_passes = transfer.gain_over_scratch >= thresholds.minimum_transfer_advantage
        && transfer.gain_over_matched_control >= thresholds.minimum_transfer_advantage
        && transfer.prior_skill_retention >= thresholds.minimum_retention;
    if transfer_passes {
        passed.push("improves held-out acquisition over scratch and a matched control".into());
        passed.push("retains the previously measured behavior".into());
        PromotionDecision {
            class: ProgressClass::TransferBackedProgress,
            accept_local_step: true,
            accept_as_project_progress: true,
            requires_transfer_test: false,
            passed_checks: passed,
            failed_checks: vec![],
        }
    } else {
        failed.push(
            "local improvement did not produce the required held-out advantage with retention"
                .into(),
        );
        PromotionDecision {
            class: ProgressClass::LocalButNonTransferring,
            accept_local_step: false,
            accept_as_project_progress: false,
            requires_transfer_test: false,
            passed_checks: passed,
            failed_checks: failed,
        }
    }
}

fn check(condition: bool, description: &str, passed: &mut Vec<String>, failed: &mut Vec<String>) {
    if condition {
        passed.push(description.into());
    } else {
        failed.push(description.into());
    }
}

fn valid_evidence(evidence: &CheckpointEvidence) -> bool {
    let rates = [
        evidence.diagnostics.witness.rate,
        evidence.diagnostics.fixed_goal_control.rate,
        evidence.diagnostics.state_predicts_goal_control.rate,
        evidence.diagnostics.hidden_goal_check.rate,
        evidence.diagnostics.renamed_witness.rate,
        evidence.diagnostics.counterfactual_goal_sensitivity,
        evidence.diagnostics.presentation_order_invariance,
    ];
    let rates_are_valid = rates
        .into_iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(&value));
    let transfer_is_valid = evidence.transfer.as_ref().is_none_or(|transfer| {
        transfer.gain_over_scratch.is_finite()
            && (-1.0..=1.0).contains(&transfer.gain_over_scratch)
            && transfer.gain_over_matched_control.is_finite()
            && (-1.0..=1.0).contains(&transfer.gain_over_matched_control)
            && transfer.prior_skill_retention.is_finite()
            && (0.0..=1.0).contains(&transfer.prior_skill_retention)
    });
    evidence.headline_metric.is_finite() && rates_are_valid && transfer_is_valid
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceAudit {
    pub process: String,
    pub exhaustive_public_hidden_goal_ceiling: f64,
    pub policy_evidence: BTreeMap<String, PolicyEvidence>,
    pub example_previous_evidence: CheckpointEvidence,
    pub example_candidate_evidence: BTreeMap<String, CheckpointEvidence>,
    pub example_decisions: BTreeMap<String, PromotionDecision>,
}

pub fn reference_audit() -> ReferenceAudit {
    let suite = DiagnosticSuite::standard();
    let goal_aware = evaluate_policy(&suite, &GoalAwarePolicy);
    let state_only = evaluate_policy(&suite, &StateOnlyPolicy);
    let stable_key = evaluate_policy(&suite, &StableKeyShortcutPolicy);
    let always_left = evaluate_policy(&suite, &AlwaysLeftPolicy);
    let thresholds = ProgressThresholds::default();

    let previous = CheckpointEvidence {
        headline_metric: 0.40,
        diagnostics: state_only.clone(),
        transfer: None,
    };
    let false_candidate = CheckpointEvidence {
        headline_metric: 0.80,
        diagnostics: stable_key.clone(),
        transfer: None,
    };
    let local_candidate = CheckpointEvidence {
        headline_metric: 0.80,
        diagnostics: goal_aware.clone(),
        transfer: None,
    };
    let transfer_candidate = CheckpointEvidence {
        headline_metric: 0.80,
        diagnostics: goal_aware.clone(),
        transfer: Some(TransferEvidence {
            gain_over_scratch: 0.20,
            gain_over_matched_control: 0.15,
            prior_skill_retention: 0.98,
        }),
    };

    ReferenceAudit {
        process: "five-position line; same start, opposite visible goals, two-step horizon".into(),
        exhaustive_public_hidden_goal_ceiling: hidden_goal_public_ceiling(),
        policy_evidence: BTreeMap::from([
            ("always_left".into(), always_left),
            ("goal_aware".into(), goal_aware),
            ("stable_key_shortcut".into(), stable_key),
            ("state_only".into(), state_only),
        ]),
        example_previous_evidence: previous.clone(),
        example_candidate_evidence: BTreeMap::from([
            ("false_progress".into(), false_candidate.clone()),
            ("local_meaningful_progress".into(), local_candidate.clone()),
            (
                "transfer_backed_progress".into(),
                transfer_candidate.clone(),
            ),
        ]),
        example_decisions: BTreeMap::from([
            (
                "false_progress".into(),
                classify_progress(&previous, &false_candidate, &thresholds),
            ),
            (
                "local_meaningful_progress".into(),
                classify_progress(&previous, &local_candidate, &thresholds),
            ),
            (
                "transfer_backed_progress".into(),
                classify_progress(&previous, &transfer_candidate, &thresholds),
            ),
        ]),
    }
}

fn action_index(action: Move) -> usize {
    match action {
        Move::Left => 0,
        Move::Stay => 1,
        Move::Right => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn public_fingerprint(tokens: &[LearningToken]) -> Vec<(u8, u16, u16, [u32; PAYLOAD_DIM])> {
        let mut fingerprint: Vec<_> = tokens
            .iter()
            .map(|token| {
                (
                    token.public.role as u8,
                    token.public.key,
                    token.public.event,
                    token.public.payload.map(f32::to_bits),
                )
            })
            .collect();
        fingerprint.sort_unstable();
        fingerprint
    }

    fn checkpoint(headline: f64, diagnostics: PolicyEvidence) -> CheckpointEvidence {
        CheckpointEvidence {
            headline_metric: headline,
            diagnostics,
            transfer: None,
        }
    }

    #[test]
    fn every_standard_case_is_solvable_by_exhaustive_search() {
        for case in DiagnosticSuite::standard().cases {
            assert!(!optimal_first_moves(&case).is_empty(), "{}", case.id);
        }
    }

    #[test]
    fn hidden_goal_ceiling_is_exactly_one_half() {
        assert_eq!(hidden_goal_public_ceiling(), GOAL_BLIND_CEILING);
    }

    #[test]
    fn goal_aware_policy_passes_the_witness_and_the_renaming() {
        let evidence = evaluate_policy(&DiagnosticSuite::standard(), &GoalAwarePolicy);
        assert_eq!(evidence.witness.rate, 1.0);
        assert_eq!(evidence.renamed_witness.rate, 1.0);
        assert_eq!(evidence.counterfactual_goal_sensitivity, 1.0);
        assert_eq!(evidence.hidden_goal_check.rate, GOAL_BLIND_CEILING);
    }

    #[test]
    fn state_only_policy_is_exposed_by_the_paired_goal_witness() {
        let evidence = evaluate_policy(&DiagnosticSuite::standard(), &StateOnlyPolicy);
        assert_eq!(evidence.state_predicts_goal_control.rate, 1.0);
        assert_eq!(evidence.fixed_goal_control.rate, 1.0);
        assert_eq!(evidence.witness.rate, GOAL_BLIND_CEILING);
        assert_eq!(evidence.counterfactual_goal_sensitivity, 0.0);
    }

    #[test]
    fn stable_key_shortcut_is_exposed_by_renaming() {
        let evidence = evaluate_policy(&DiagnosticSuite::standard(), &StableKeyShortcutPolicy);
        assert_eq!(evidence.witness.rate, 1.0);
        assert_eq!(evidence.renamed_witness.rate, 0.0);
    }

    #[test]
    fn hidden_goal_observations_do_not_contain_the_goal() {
        for case in DiagnosticSuite::standard().cases_of_kind(CaseKind::HiddenGoalLeakageCheck) {
            let observation = case.public_observation(case.start, HORIZON);
            assert_eq!(observation.goal_key, None);
        }
    }

    #[test]
    fn learner_witness_prefixes_differ_only_in_the_visible_goal_key() {
        let suite = DiagnosticSuite::standard();
        let cases: Vec<_> = suite.cases_of_kind(CaseKind::Witness).cloned().collect();
        let left = DiagnosticRollout::new(cases[0].clone(), SerializationOrder::Canonical);
        let right = DiagnosticRollout::new(cases[1].clone(), SerializationOrder::Canonical);
        assert_eq!(left.tokens().len(), right.tokens().len());
        let differences: Vec<_> = left
            .tokens()
            .iter()
            .zip(right.tokens())
            .filter(|(left, right)| left.public != right.public)
            .collect();
        assert_eq!(differences.len(), 1);
        assert_eq!(differences[0].0.public.role, Role::Goal);
        assert_eq!(differences[0].1.public.role, Role::Goal);
    }

    #[test]
    fn hidden_goal_learner_prefixes_are_byte_for_byte_identical() {
        let suite = DiagnosticSuite::standard();
        let cases: Vec<_> = suite
            .cases_of_kind(CaseKind::HiddenGoalLeakageCheck)
            .cloned()
            .collect();
        let left = DiagnosticRollout::new(cases[0].clone(), SerializationOrder::Canonical);
        let right = DiagnosticRollout::new(cases[1].clone(), SerializationOrder::Canonical);
        assert_eq!(left.tokens(), right.tokens());
    }

    #[test]
    fn schema_reordering_changes_only_presentation() {
        for case in DiagnosticSuite::standard().cases {
            let canonical = DiagnosticRollout::new(case.clone(), SerializationOrder::Canonical);
            let permuted = DiagnosticRollout::new(case, SerializationOrder::Permuted);
            assert_ne!(canonical.tokens(), permuted.tokens());
            assert_eq!(
                public_fingerprint(canonical.tokens()),
                public_fingerprint(permuted.tokens())
            );
        }
    }

    #[test]
    fn continuous_actions_decode_and_drive_the_real_diagnostic_transition() {
        let case = DiagnosticSuite::standard()
            .cases_of_kind(CaseKind::Witness)
            .next()
            .unwrap()
            .clone();
        let mut solved = DiagnosticRollout::new(case.clone(), SerializationOrder::Canonical);
        solved.step_normalized(-1.0).unwrap();
        solved.step_normalized(-1.0).unwrap();
        assert!(solved.is_done());
        assert!(solved.success());
        assert_eq!(solved.action_displacements(), vec![-1, -1]);

        let mut stalled = DiagnosticRollout::new(case, SerializationOrder::Canonical);
        stalled.step_normalized(0.0).unwrap();
        stalled.step_normalized(0.0).unwrap();
        assert!(stalled.is_done());
        assert!(!stalled.success());
    }

    #[test]
    fn learner_queries_never_serialize_a_target() {
        for rollout in standard_diagnostic_rollouts(SerializationOrder::Canonical) {
            for token in rollout
                .tokens()
                .iter()
                .filter(|token| token.public.role == Role::ActionQuery)
            {
                assert_eq!(token.public.payload[0], 0.0);
                assert!(!token.supervision.action_mask.iter().any(|value| *value));
                assert!(!token.supervision.future_mask);
            }
        }
    }

    #[test]
    fn representation_probe_arms_match_semantics_records_and_targets() {
        let fixed = teacher_training_records(TrainingPresentationArm::Fixed);
        let orbit = teacher_training_records(TrainingPresentationArm::Orbit);
        assert_eq!(fixed.len(), 8);
        assert_eq!(orbit.len(), fixed.len());
        let target_count = |records: &[Vec<LearningToken>]| {
            records
                .iter()
                .flatten()
                .flat_map(|token| token.supervision.action_mask)
                .filter(|value| *value)
                .count()
        };
        assert_eq!(target_count(&fixed), target_count(&orbit));
        assert!(fixed.chunks_exact(2).all(|pair| pair[0] == pair[1]));
        assert!(orbit.chunks_exact(2).all(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn every_teacher_probe_record_solves_its_visible_case() {
        for arm in [
            TrainingPresentationArm::Fixed,
            TrainingPresentationArm::Orbit,
        ] {
            for record in teacher_training_records(arm) {
                let targets: Vec<_> = record
                    .iter()
                    .filter(|token| token.supervision.action_mask[0])
                    .map(|token| decode_normalized_move(token.supervision.action_target[0]))
                    .collect();
                assert!(!targets.is_empty());
                assert!(targets.len() <= HORIZON as usize);
            }
        }
    }

    #[test]
    fn paired_witnesses_differ_only_in_the_public_goal() {
        let suite = DiagnosticSuite::standard();
        let cases: Vec<_> = suite.cases_of_kind(CaseKind::Witness).collect();
        let left = cases[0].public_observation(cases[0].start, HORIZON);
        let right = cases[1].public_observation(cases[1].start, HORIZON);
        assert_eq!(left.position_key, right.position_key);
        assert_eq!(left.steps_remaining, right.steps_remaining);
        assert_eq!(left.positions, right.positions);
        assert_eq!(left.actions, right.actions);
        assert_ne!(left.goal_key, right.goal_key);
    }

    #[test]
    fn gate_rejects_a_better_headline_score_from_a_label_shortcut() {
        let suite = DiagnosticSuite::standard();
        let previous = checkpoint(0.40, evaluate_policy(&suite, &StateOnlyPolicy));
        let candidate = checkpoint(0.80, evaluate_policy(&suite, &StableKeyShortcutPolicy));
        let decision = classify_progress(&previous, &candidate, &ProgressThresholds::default());
        assert_eq!(decision.class, ProgressClass::FalseProgress);
        assert!(!decision.accept_local_step);
        assert!(decision
            .failed_checks
            .iter()
            .any(|reason| reason.contains("renaming")));
    }

    #[test]
    fn gate_accepts_local_behavior_but_does_not_call_it_project_progress() {
        let suite = DiagnosticSuite::standard();
        let previous = checkpoint(0.40, evaluate_policy(&suite, &StateOnlyPolicy));
        let candidate = checkpoint(0.80, evaluate_policy(&suite, &GoalAwarePolicy));
        let decision = classify_progress(&previous, &candidate, &ProgressThresholds::default());
        assert_eq!(decision.class, ProgressClass::LocalMeaningfulProgress);
        assert!(decision.accept_local_step);
        assert!(!decision.accept_as_project_progress);
        assert!(decision.requires_transfer_test);
    }

    #[test]
    fn gate_rejects_leakage_even_when_every_visible_case_is_solved() {
        let suite = DiagnosticSuite::standard();
        let previous = checkpoint(0.40, evaluate_policy(&suite, &StateOnlyPolicy));
        let mut leaky = evaluate_policy(&suite, &GoalAwarePolicy);
        leaky.hidden_goal_check = Score::from_counts(2, 2);
        let candidate = checkpoint(0.90, leaky);
        let decision = classify_progress(&previous, &candidate, &ProgressThresholds::default());
        assert_eq!(decision.class, ProgressClass::FalseProgress);
        assert!(decision
            .failed_checks
            .iter()
            .any(|reason| reason.contains("hidden")));
    }

    #[test]
    fn gate_rejects_progress_that_depends_on_schema_token_order() {
        let suite = DiagnosticSuite::standard();
        let previous = checkpoint(0.40, evaluate_policy(&suite, &StateOnlyPolicy));
        let mut order_sensitive = evaluate_policy(&suite, &GoalAwarePolicy);
        order_sensitive.presentation_order_invariance = 0.5;
        let candidate = checkpoint(0.90, order_sensitive);
        let decision = classify_progress(&previous, &candidate, &ProgressThresholds::default());
        assert_eq!(decision.class, ProgressClass::FalseProgress);
        assert!(decision
            .failed_checks
            .iter()
            .any(|reason| reason.contains("reordered")));
    }

    #[test]
    fn gate_requires_matched_transfer_and_retention_for_project_progress() {
        let suite = DiagnosticSuite::standard();
        let previous = checkpoint(0.40, evaluate_policy(&suite, &StateOnlyPolicy));
        let diagnostics = evaluate_policy(&suite, &GoalAwarePolicy);
        let failed_transfer = CheckpointEvidence {
            headline_metric: 0.80,
            diagnostics: diagnostics.clone(),
            transfer: Some(TransferEvidence {
                gain_over_scratch: 0.20,
                gain_over_matched_control: 0.02,
                prior_skill_retention: 0.98,
            }),
        };
        let rejected =
            classify_progress(&previous, &failed_transfer, &ProgressThresholds::default());
        assert_eq!(rejected.class, ProgressClass::LocalButNonTransferring);
        assert!(!rejected.accept_as_project_progress);

        let passed_transfer = CheckpointEvidence {
            headline_metric: 0.80,
            diagnostics,
            transfer: Some(TransferEvidence {
                gain_over_scratch: 0.20,
                gain_over_matched_control: 0.15,
                prior_skill_retention: 0.98,
            }),
        };
        let accepted =
            classify_progress(&previous, &passed_transfer, &ProgressThresholds::default());
        assert_eq!(accepted.class, ProgressClass::TransferBackedProgress);
        assert!(accepted.accept_as_project_progress);
    }

    #[test]
    fn negative_transfer_gain_is_valid_evidence_and_is_rejected_as_nontransferring() {
        let suite = DiagnosticSuite::standard();
        let previous = checkpoint(0.40, evaluate_policy(&suite, &StateOnlyPolicy));
        let candidate = CheckpointEvidence {
            headline_metric: 0.80,
            diagnostics: evaluate_policy(&suite, &GoalAwarePolicy),
            transfer: Some(TransferEvidence {
                gain_over_scratch: -0.10,
                gain_over_matched_control: -0.20,
                prior_skill_retention: 0.98,
            }),
        };
        let decision = classify_progress(&previous, &candidate, &ProgressThresholds::default());
        assert_eq!(decision.class, ProgressClass::LocalButNonTransferring);
    }

    #[test]
    fn audit_is_standard_json_and_contains_all_decision_classes() {
        let audit = reference_audit();
        let json = serde_json::to_string(&audit).unwrap();
        let restored: ReferenceAudit = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, audit);
        assert_eq!(
            audit.example_decisions["false_progress"].class,
            ProgressClass::FalseProgress
        );
        assert_eq!(
            audit.example_decisions["local_meaningful_progress"].class,
            ProgressClass::LocalMeaningfulProgress
        );
        assert_eq!(
            audit.example_decisions["transfer_backed_progress"].class,
            ProgressClass::TransferBackedProgress
        );
    }
}
