//! An isolated, finite compiler for typed embodied-world terms.
//!
//! This crate is a spike, not a replacement for a card evaluator.  It gives
//! the shared kernel a deliberately small executable interpretation over a
//! finite transition system: body support and sensorium are first-class, ports
//! carry both a value type and a visibility, and every public event has an
//! explicit source. A ring remains a compact state-space constructor; it is
//! not the executor's only topology.
//! Its output implements the existing finite audit interfaces, so the existing
//! query algebra can audit generated terms without being changed.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use pretraining_g0_contract::{
    identification_diameter, value_bounds, AcceptanceReport, AmbiguitySet, BoundaryEffect,
    Coupling, CouplingRule, Displaced, Fragment, Guard, GuardContext, IndexSet, Interrupt,
    KernelUse, Norm, PubliclyObservable, Restriction,
};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Version of the canonical term encoding used by [`WorldTerm::family_hash`].
/// Version 1 was the ring-only shape; version 2 separates body morphology from
/// environment state space and therefore intentionally changes ring digests.
pub const FAMILY_HASH_SCHEMA_VERSION: u16 = 2;

/// Where a declared port may be observed.  Ports are private by default in a
/// handwritten term because [`Port::new`] uses `Privileged`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Visibility {
    Public,
    Privileged,
    Generator,
}

/// The finite values the first compiler understands.  New semantic values must
/// be added deliberately; an untyped integer channel is not a back door.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PortValue {
    Command,
    Cell,
    Signal,
    /// A body-support descriptor is audit-only.  It may not be exposed as a
    /// public query because that would collapse body identification by schema.
    Support,
}

/// A monitor is input-only.  This is how the no-monitor-to-actuator rule is
/// made structural rather than a convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PortDirection {
    Input,
    Output,
    Monitor,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Port {
    pub name: String,
    pub direction: PortDirection,
    pub value: PortValue,
    pub visibility: Visibility,
}

impl Port {
    pub fn new(name: impl Into<String>, direction: PortDirection, value: PortValue) -> Self {
        Self {
            name: name.into(),
            direction,
            value,
            visibility: Visibility::Privileged,
        }
    }

    pub fn public(mut self) -> Self {
        self.visibility = Visibility::Public;
        self
    }

    pub fn with_visibility(mut self, visibility: Visibility) -> Self {
        self.visibility = visibility;
        self
    }
}

/// `P ▷ Q`, named rather than positional so a generated term can be audited.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Wiring {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Morphology {
    /// A body-local structural count. It is deliberately independent from the
    /// number of environment states.
    pub segments: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Actuator {
    pub id: u16,
    pub name: String,
    pub displacement: i32,
    pub role: ActionRole,
}

/// Action roles are semantic: only a movement can be body-supported, while a
/// fallback terminates scoring without adding a fake terminal configuration to
/// the ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionRole {
    Movement,
    Hold,
    Fallback,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Actuation {
    pub command_port: String,
    pub actuators: Vec<Actuator>,
    /// Body support is distinct from an environment edge deletion.  This is
    /// the structural distinction Card 03 needs to retain.
    pub supported: IndexSet,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sensorium {
    pub cell_port: String,
    pub publishes_cell: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BodyTerm {
    pub morphology: Morphology,
    pub actuation: Actuation,
    pub sensorium: Sensorium,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentTerm {
    pub start: usize,
    pub state_space: StateSpaceTerm,
}

/// Diagnostics computed from the executable environment topology. These are
/// privileged audit facts and never learner-visible data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyDiagnostics {
    pub state_count: usize,
    pub transition_rows: usize,
    pub degree_sequence: Vec<usize>,
    pub is_simple_cycle: bool,
}

/// The environment owns the finite state space and its topology.  A graph
/// table names body-owned actions but never owns their availability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateSpaceTerm {
    /// Compact cyclic constructor retained for existing ring contracts.
    Ring {
        states: usize,
        /// Environmental edge deletions. Missing ring edges self-loop.
        blocked_edges: Vec<(usize, u16)>,
    },
    /// Explicit deterministic transition graph. Entries absent from the table
    /// follow the declared missing-edge behavior.
    Graph {
        states: usize,
        transitions: Vec<GraphTransition>,
        missing_edge: MissingEdgeBehavior,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphTransition {
    pub source: usize,
    pub actuator: u16,
    pub destination: usize,
}

/// Missing graph edges are explicit semantics, never an accidental map miss.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissingEdgeBehavior {
    SelfLoop,
}

impl EnvironmentTerm {
    pub fn ring(start: usize, states: usize, blocked_edges: Vec<(usize, u16)>) -> Self {
        Self {
            start,
            state_space: StateSpaceTerm::Ring {
                states,
                blocked_edges,
            },
        }
    }

    pub fn graph(
        start: usize,
        states: usize,
        transitions: Vec<GraphTransition>,
        missing_edge: MissingEdgeBehavior,
    ) -> Self {
        Self {
            start,
            state_space: StateSpaceTerm::Graph {
                states,
                transitions,
                missing_edge,
            },
        }
    }

    pub fn state_count(&self) -> usize {
        match &self.state_space {
            StateSpaceTerm::Ring { states, .. } | StateSpaceTerm::Graph { states, .. } => *states,
        }
    }

    fn is_ring(&self) -> bool {
        matches!(self.state_space, StateSpaceTerm::Ring { .. })
    }

    pub fn topology_diagnostics(&self) -> TopologyDiagnostics {
        let states = self.state_count();
        let (transition_rows, edges, is_simple_cycle) = match &self.state_space {
            StateSpaceTerm::Ring { blocked_edges, .. } => (
                blocked_edges.len(),
                (0..states)
                    .map(|state| {
                        let next = (state + 1) % states;
                        if state < next {
                            (state, next)
                        } else {
                            (next, state)
                        }
                    })
                    .collect::<BTreeSet<_>>(),
                states >= 3,
            ),
            StateSpaceTerm::Graph { transitions, .. } => {
                let mut edges = BTreeSet::new();
                for transition in transitions {
                    if transition.source < states
                        && transition.destination < states
                        && transition.source != transition.destination
                    {
                        edges.insert((
                            transition.source.min(transition.destination),
                            transition.source.max(transition.destination),
                        ));
                    }
                }
                let mut degree = vec![0usize; states];
                for (from, to) in &edges {
                    degree[*from] += 1;
                    degree[*to] += 1;
                }
                let connected = if states == 0 {
                    false
                } else {
                    let mut seen = BTreeSet::new();
                    let mut frontier = vec![0usize];
                    while let Some(state) = frontier.pop() {
                        if !seen.insert(state) {
                            continue;
                        }
                        for (from, to) in &edges {
                            if *from == state && !seen.contains(to) {
                                frontier.push(*to);
                            } else if *to == state && !seen.contains(from) {
                                frontier.push(*from);
                            }
                        }
                    }
                    seen.len() == states
                };
                let simple = connected
                    && states >= 3
                    && edges.len() == states
                    && degree.iter().all(|degree| *degree == 2);
                (transitions.len(), edges, simple)
            }
        };
        let mut degree = vec![0usize; states];
        for (from, to) in &edges {
            degree[*from] += 1;
            degree[*to] += 1;
        }
        TopologyDiagnostics {
            state_count: states,
            transition_rows,
            degree_sequence: degree,
            is_simple_cycle,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormTerm {
    pub expression: Norm,
    /// A private norm is allowed for exact upper-bound experiments, but only a
    /// public norm is encoded into the learner-visible trace.
    pub visibility: Visibility,
}

/// Reward/cost semantics are term data rather than a card-local evaluator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScoringTerm {
    pub goal_reward: i32,
    pub action_cost: i32,
    pub fallback_reward: i32,
    pub violation_penalty: i32,
}

impl Default for ScoringTerm {
    fn default() -> Self {
        Self {
            goal_reward: 100,
            action_cost: 1,
            fallback_reward: 0,
            violation_penalty: -100,
        }
    }
}

/// An action prelude.  It runs before the first scored decision with the
/// scored clock held at zero, and its cumulative cells may be public.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationTerm {
    pub pulses: Vec<u16>,
    pub publishes_cells: bool,
}

/// A public, announced restoration of one body actuator.  The support effect
/// begins only after `after_step` scored actions, never during calibration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportRestoration {
    pub actuator: u16,
    pub after_step: usize,
    pub announcement_port: String,
    pub announcement_value: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignalTerm {
    pub output_port: String,
    pub guard: Guard,
    pub value: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessTerm {
    pub name: String,
    pub signals: Vec<SignalTerm>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoupledWriter {
    pub guard: Guard,
    pub value: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CouplingTerm {
    pub coupling: Coupling,
    /// The declared list order is the writer order for `Override`.
    pub writers: Vec<CoupledWriter>,
    /// Defined output when no writer guard is active.  This closes the error
    /// case that `Coupling::resolve` intentionally exposes at the kernel level.
    pub inactive_value: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterruptTerm {
    pub specification: Interrupt,
    /// The named disturbance or scaffold process displaced by the interrupt.
    pub interrupted_process: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevealTerm {
    pub source_visibility: Visibility,
    pub output_port: String,
    pub guard: Guard,
    pub value: i64,
}

/// One complete, serializable world term.  Generator metadata is deliberately
/// absent: it belongs to [`GenerationSpec`] and never enters a public trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldTerm {
    pub name: String,
    pub horizon: usize,
    pub ports: Vec<Port>,
    pub wiring: Vec<Wiring>,
    pub body: BodyTerm,
    pub environment: EnvironmentTerm,
    pub norm: NormTerm,
    pub scoring: ScoringTerm,
    pub calibration: Option<CalibrationTerm>,
    pub restorations: Vec<SupportRestoration>,
    pub disturbance: Option<ProcessTerm>,
    pub scaffold: Option<ProcessTerm>,
    pub couplings: Vec<CouplingTerm>,
    pub interrupts: Vec<InterruptTerm>,
    pub restrictions: Vec<Restriction>,
    pub reveals: Vec<RevealTerm>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    Invalid(String),
    Unsupported(String),
    UnknownPort(String),
    IllTypedWire { from: String, to: String },
    VisibilityLeak { from: String, to: String },
    MonitorAsSource(String),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid world term: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported world operation: {message}"),
            Self::UnknownPort(port) => write!(f, "unknown port `{port}`"),
            Self::IllTypedWire { from, to } => write!(f, "ill-typed wire `{from}` -> `{to}`"),
            Self::VisibilityLeak { from, to } => {
                write!(
                    f,
                    "wire `{from}` -> `{to}` leaks non-public data into public view"
                )
            }
            Self::MonitorAsSource(port) => write!(f, "monitor port `{port}` cannot be a source"),
        }
    }
}

impl std::error::Error for CompileError {}

fn permits_flow(from: Visibility, to: Visibility) -> bool {
    match (from, to) {
        (Visibility::Public, _) => true,
        (Visibility::Privileged, Visibility::Privileged) => true,
        // Generator values may instantiate a private parameter but can never
        // become learner-visible by ordinary wiring.
        (Visibility::Generator, Visibility::Privileged | Visibility::Generator) => true,
        _ => false,
    }
}

fn guard_context(executed: usize, action: Option<u16>, cell: usize) -> GuardContext {
    GuardContext {
        executed,
        last_action: action,
        cell,
    }
}

/// Frozen version-2 public norm token.  It remains the only norm rendering in
/// `PublicView` until a separately versioned learner-event profile exists.
fn norm_code(norm: &Norm) -> i64 {
    fn walk(norm: &Norm, state: &mut u64) {
        let mix = |state: &mut u64, value: u64| {
            *state ^= value;
            *state = state.wrapping_mul(0x100000001b3);
        };
        match norm {
            Norm::Settle { cell } => mix(state, 1 ^ *cell as u64),
            Norm::Visit { cell } => mix(state, 2 ^ *cell as u64),
            Norm::Avoid { cell } => mix(state, 3 ^ *cell as u64),
            Norm::Both(left, right) => {
                mix(state, 4);
                walk(left, state);
                walk(right, state);
            }
            Norm::Supersede {
                before,
                after,
                guard,
            } => {
                mix(state, 5 ^ guard_code(*guard));
                walk(before, state);
                walk(after, state);
            }
            Norm::Priority { high, low } => {
                mix(state, 6);
                walk(high, state);
                walk(low, state);
            }
        }
    }
    let mut state = 0xcbf29ce484222325;
    walk(norm, &mut state);
    state as i64
}

fn guard_code(guard: Guard) -> u64 {
    match guard {
        Guard::AtStart => 11,
        Guard::AfterStep(step) => 12 ^ step as u64,
        Guard::OnAction(action) => 13 ^ u64::from(action),
        Guard::OnCellEntry(cell) => 14 ^ cell as u64,
        Guard::Never => 15,
    }
}

/// What one symbolic-goal atom requires of the configuration.
///
/// These are the norm's leaves rather than new vocabulary.  The compiler
/// cannot publish a requirement the world does not evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GoalPredicate {
    Settle,
    Visit,
    Avoid,
}

impl GoalPredicate {
    const fn code(self) -> i64 {
        match self {
            Self::Settle => 1,
            Self::Visit => 2,
            Self::Avoid => 3,
        }
    }

    fn from_code(code: i64) -> Option<Self> {
        match code {
            1 => Some(Self::Settle),
            2 => Some(Self::Visit),
            3 => Some(Self::Avoid),
            _ => None,
        }
    }
}

/// One symbolic requirement on configuration-channel content.
///
/// `cell` is a possible value of the sensorium's cell observation channel, not
/// the channel/key itself.  A future learner adapter must add the observation
/// port explicitly rather than conflating it with this content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoalAtom {
    pub predicate: GoalPredicate,
    pub cell: usize,
}

/// A structured symbolic diagnostic for a goal, distinct from its denotation.
///
/// [`Norm`] is the denotation: the world evaluates it, and a privileged norm is
/// evaluated without being published at all.  This diagnostic is derived from
/// the norm by one total function ([`SymbolicGoalDiagnostic::from_norm`]) so a future
/// symbolic adapter can be checked against evaluator semantics.
///
/// It does not replace the version-2 `norm_code` public token.  That token is
/// opaque and non-injective, but it is frozen for trace replay.  A structured
/// learner carrier needs a separately versioned, framed event profile and a
/// lowering onto the canonical learner boundary.
///
/// # Declared limit
///
/// [`NormTerm`] carries one visibility for its whole expression. The
/// visibility-gated API below therefore exposes this diagnostic only for a
/// public norm; it does not establish that all its subterms are safe to publish
/// in a future carrier. Subterm visibility remains a separate change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolicGoalDiagnostic {
    Atom(GoalAtom),
    /// Both requirements apply.
    All {
        left: Box<SymbolicGoalDiagnostic>,
        right: Box<SymbolicGoalDiagnostic>,
    },
    /// `after` supersedes `before` once the announced guard fires.
    Then {
        before: Box<SymbolicGoalDiagnostic>,
        after: Box<SymbolicGoalDiagnostic>,
        guard: Guard,
    },
    /// `high` wins wherever the two conflict.
    Preferred {
        high: Box<SymbolicGoalDiagnostic>,
        low: Box<SymbolicGoalDiagnostic>,
    },
}

const CARRIER_ATOM: i64 = 1;
const CARRIER_ALL: i64 = 2;
const CARRIER_THEN: i64 = 3;
const CARRIER_PREFERRED: i64 = 4;

impl SymbolicGoalDiagnostic {
    /// Derive a symbolic diagnostic from a norm.
    ///
    /// Total by construction for diagnostics. It is not a public-rendering
    /// function: `CompiledWorld` gates diagnostic access on norm visibility,
    /// and the v2 renderer continues to use `norm_code`.
    pub fn from_norm(norm: &Norm) -> Self {
        match norm {
            Norm::Settle { cell } => Self::atom(GoalPredicate::Settle, *cell),
            Norm::Visit { cell } => Self::atom(GoalPredicate::Visit, *cell),
            Norm::Avoid { cell } => Self::atom(GoalPredicate::Avoid, *cell),
            Norm::Both(left, right) => Self::All {
                left: Box::new(Self::from_norm(left)),
                right: Box::new(Self::from_norm(right)),
            },
            Norm::Supersede {
                before,
                after,
                guard,
            } => Self::Then {
                before: Box::new(Self::from_norm(before)),
                after: Box::new(Self::from_norm(after)),
                guard: *guard,
            },
            Norm::Priority { high, low } => Self::Preferred {
                high: Box::new(Self::from_norm(high)),
                low: Box::new(Self::from_norm(low)),
            },
        }
    }

    fn atom(predicate: GoalPredicate, cell: usize) -> Self {
        Self::Atom(GoalAtom { predicate, cell })
    }

    /// Encode the diagnostic in self-delimiting prefix order.
    ///
    /// Prefix order, so the encoding is self-delimiting and a decoder needs no
    /// separate length field.  The flat vector is a *rendering*; this type is
    /// what carries meaning, the same way the canonical public record stands
    /// behind the learner ABI's float rows rather than the other way round.
    pub fn encode_diagnostic(&self) -> Vec<i64> {
        let mut rendered = Vec::new();
        self.render_into(&mut rendered);
        rendered
    }

    fn render_into(&self, out: &mut Vec<i64>) {
        match self {
            Self::Atom(atom) => {
                out.push(CARRIER_ATOM);
                out.push(atom.predicate.code());
                out.push(atom.cell as i64);
            }
            Self::All { left, right } => {
                out.push(CARRIER_ALL);
                left.render_into(out);
                right.render_into(out);
            }
            Self::Then {
                before,
                after,
                guard,
            } => {
                out.push(CARRIER_THEN);
                out.extend(render_guard(*guard));
                before.render_into(out);
                after.render_into(out);
            }
            Self::Preferred { high, low } => {
                out.push(CARRIER_PREFERRED);
                high.render_into(out);
                low.render_into(out);
            }
        }
    }

    /// Decode one diagnostic from the head of an encoded slice, returning it and
    /// the number of slots it consumed.
    ///
    /// Invertibility is checked rather than asserted. This is not a v2 public
    /// trace decoder.
    pub fn decode_diagnostic(rendered: &[i64]) -> Result<(Self, usize), CompileError> {
        let malformed =
            || CompileError::Invalid("goal diagnostic encoding is malformed".to_string());
        let tag = rendered.first().copied().ok_or_else(malformed)?;
        match tag {
            CARRIER_ATOM => {
                let predicate =
                    GoalPredicate::from_code(rendered.get(1).copied().ok_or_else(malformed)?)
                        .ok_or_else(malformed)?;
                let cell = usize::try_from(rendered.get(2).copied().ok_or_else(malformed)?)
                    .map_err(|_| malformed())?;
                Ok((Self::atom(predicate, cell), 3))
            }
            CARRIER_ALL | CARRIER_PREFERRED => {
                let (first, first_used) = Self::decode_diagnostic(&rendered[1..])?;
                let (second, second_used) = Self::decode_diagnostic(&rendered[1 + first_used..])?;
                let carrier = if tag == CARRIER_ALL {
                    Self::All {
                        left: Box::new(first),
                        right: Box::new(second),
                    }
                } else {
                    Self::Preferred {
                        high: Box::new(first),
                        low: Box::new(second),
                    }
                };
                Ok((carrier, 1 + first_used + second_used))
            }
            CARRIER_THEN => {
                let guard = decode_guard(
                    rendered.get(1).copied().ok_or_else(malformed)?,
                    rendered.get(2).copied().ok_or_else(malformed)?,
                )
                .ok_or_else(malformed)?;
                let tail = rendered.get(3..).ok_or_else(malformed)?;
                let (before, before_used) = Self::decode_diagnostic(tail)?;
                let (after, after_used) = Self::decode_diagnostic(&tail[before_used..])?;
                Ok((
                    Self::Then {
                        before: Box::new(before),
                        after: Box::new(after),
                        guard,
                    },
                    3 + before_used + after_used,
                ))
            }
            _ => Err(malformed()),
        }
    }
}

/// An announced guard as two diagnostic slots: a tag and its argument.
///
/// Two slots rather than one mixed integer.  The replaced encoding folded the
/// argument into the tag with `XOR`, which is not injective over the variants
/// it had to separate.
fn render_guard(guard: Guard) -> [i64; 2] {
    match guard {
        Guard::AtStart => [1, 0],
        Guard::AfterStep(step) => [2, step as i64],
        Guard::OnAction(action) => [3, i64::from(action)],
        Guard::OnCellEntry(cell) => [4, cell as i64],
        Guard::Never => [5, 0],
    }
}

fn decode_guard(tag: i64, argument: i64) -> Option<Guard> {
    match tag {
        1 if argument == 0 => Some(Guard::AtStart),
        2 => usize::try_from(argument).ok().map(Guard::AfterStep),
        3 => u16::try_from(argument).ok().map(Guard::OnAction),
        4 => usize::try_from(argument).ok().map(Guard::OnCellEntry),
        5 if argument == 0 => Some(Guard::Never),
        _ => None,
    }
}

impl WorldTerm {
    /// Validate every structural invariant before any executable object is
    /// made.  Validation is intentionally conservative around conflict
    /// coupling: two declared writers are rejected rather than depending on a
    /// future guard analysis to prove that they never coincide.
    pub fn validate(&self) -> Result<(), CompileError> {
        if self.name.trim().is_empty() {
            return Err(CompileError::Invalid("world has no name".into()));
        }
        if !(1..=IndexSet::CAPACITY).contains(&self.body.morphology.segments) {
            return Err(CompileError::Invalid(
                "morphology segments must be in 1..=32".into(),
            ));
        }
        if self.horizon == 0 || self.horizon > 8 {
            return Err(CompileError::Invalid(
                "horizon must be in 1..=8 for exact enumeration".into(),
            ));
        }
        let states = self.environment.state_count();
        if !(2..=IndexSet::CAPACITY).contains(&states) {
            return Err(CompileError::Invalid(
                "environment state count must be in 2..=32".into(),
            ));
        }
        if self.environment.start >= states {
            return Err(CompileError::Invalid(
                "environment start is outside its state space".into(),
            ));
        }
        if self
            .norm
            .expression
            .referenced_cells()
            .into_iter()
            .any(|cell| cell >= states)
        {
            return Err(CompileError::Invalid(
                "a norm names a cell outside the environment state space".into(),
            ));
        }
        let mut ports = BTreeMap::new();
        for port in &self.ports {
            if port.value == PortValue::Support && port.visibility == Visibility::Public {
                return Err(CompileError::Invalid(
                    "body support cannot be a public port".into(),
                ));
            }
            if port.name.trim().is_empty() || ports.insert(port.name.as_str(), port).is_some() {
                return Err(CompileError::Invalid(
                    "ports must have unique non-empty names".into(),
                ));
            }
        }
        let command = ports
            .get(self.body.actuation.command_port.as_str())
            .ok_or_else(|| CompileError::UnknownPort(self.body.actuation.command_port.clone()))?;
        if !(command.direction == PortDirection::Input
            && command.value == PortValue::Command
            && command.visibility == Visibility::Public)
        {
            return Err(CompileError::Invalid(
                "body command port must be a public command input".into(),
            ));
        }
        let sensor = ports
            .get(self.body.sensorium.cell_port.as_str())
            .ok_or_else(|| CompileError::UnknownPort(self.body.sensorium.cell_port.clone()))?;
        if !(sensor.direction == PortDirection::Output && sensor.value == PortValue::Cell) {
            return Err(CompileError::Invalid(
                "body cell port must be a cell output".into(),
            ));
        }
        if self.body.sensorium.publishes_cell && sensor.visibility != Visibility::Public {
            return Err(CompileError::Invalid(
                "a published cell sensor must use a public port".into(),
            ));
        }
        let mut actuator_ids = BTreeSet::new();
        let mut fallback_count = 0usize;
        for actuator in &self.body.actuation.actuators {
            if usize::from(actuator.id) >= IndexSet::CAPACITY || !actuator_ids.insert(actuator.id) {
                return Err(CompileError::Invalid(
                    "actuator ids must be unique values in 0..32".into(),
                ));
            }
            if actuator.role == ActionRole::Fallback {
                fallback_count += 1;
            }
            if actuator.role != ActionRole::Movement
                && !self
                    .body
                    .actuation
                    .supported
                    .contains(usize::from(actuator.id))
            {
                return Err(CompileError::Invalid(
                    "hold and fallback actions must remain body-supported".into(),
                ));
            }
        }
        if actuator_ids.is_empty() {
            return Err(CompileError::Invalid(
                "body needs at least one actuator".into(),
            ));
        }
        if fallback_count > 1 {
            return Err(CompileError::Invalid(
                "a body may declare at most one fallback action".into(),
            ));
        }
        if self
            .body
            .actuation
            .supported
            .iter()
            .any(|id| !actuator_ids.contains(&(id as u16)))
        {
            return Err(CompileError::Invalid(
                "body support refers to an undeclared actuator".into(),
            ));
        }
        match &self.environment.state_space {
            StateSpaceTerm::Ring { blocked_edges, .. } => {
                if blocked_edges
                    .iter()
                    .any(|(state, action)| *state >= states || !actuator_ids.contains(action))
                {
                    return Err(CompileError::Invalid(
                        "blocked ring edge refers to an unknown state or actuator".into(),
                    ));
                }
            }
            StateSpaceTerm::Graph { transitions, .. } => {
                if self
                    .body
                    .actuation
                    .actuators
                    .iter()
                    .any(|actuator| actuator.displacement != 0)
                {
                    return Err(CompileError::Unsupported(
                        "graph actuators must use explicit transition rows; nonzero displacement is ring-only"
                            .into(),
                    ));
                }
                let mut entries = BTreeSet::new();
                for transition in transitions {
                    if transition.source >= states
                        || transition.destination >= states
                        || !actuator_ids.contains(&transition.actuator)
                    {
                        return Err(CompileError::Invalid(
                            "graph transition refers to an unknown state or actuator".into(),
                        ));
                    }
                    if !entries.insert((transition.source, transition.actuator)) {
                        return Err(CompileError::Invalid(
                            "graph transition table must be deterministic".into(),
                        ));
                    }
                }
            }
        }
        if self.scoring.action_cost < 0 {
            return Err(CompileError::Invalid(
                "action cost cannot be negative".into(),
            ));
        }
        if let Some(calibration) = &self.calibration {
            if calibration.pulses.is_empty() {
                return Err(CompileError::Invalid(
                    "a calibration prelude needs at least one pulse".into(),
                ));
            }
            if calibration
                .pulses
                .iter()
                .any(|action| !actuator_ids.contains(action))
            {
                return Err(CompileError::Invalid(
                    "calibration pulse refers to an undeclared actuator".into(),
                ));
            }
        }
        let mut restored = BTreeSet::new();
        for restoration in &self.restorations {
            let actuator = self
                .body
                .actuation
                .actuators
                .iter()
                .find(|actuator| actuator.id == restoration.actuator)
                .ok_or_else(|| {
                    CompileError::Invalid("restoration refers to an undeclared actuator".into())
                })?;
            if actuator.role != ActionRole::Movement {
                return Err(CompileError::Invalid(
                    "only a movement actuator may be restored".into(),
                ));
            }
            if self
                .body
                .actuation
                .supported
                .contains(usize::from(restoration.actuator))
                || !restored.insert(restoration.actuator)
            {
                return Err(CompileError::Invalid(
                    "a restoration must name one initially unsupported actuator once".into(),
                ));
            }
            let port = ports
                .get(restoration.announcement_port.as_str())
                .ok_or_else(|| CompileError::UnknownPort(restoration.announcement_port.clone()))?;
            if !(port.direction == PortDirection::Output
                && port.value == PortValue::Signal
                && port.visibility == Visibility::Public)
            {
                return Err(CompileError::Invalid(
                    "a support restoration must have a public signal announcement".into(),
                ));
            }
        }
        for wire in &self.wiring {
            let from = ports
                .get(wire.from.as_str())
                .ok_or_else(|| CompileError::UnknownPort(wire.from.clone()))?;
            let to = ports
                .get(wire.to.as_str())
                .ok_or_else(|| CompileError::UnknownPort(wire.to.clone()))?;
            if from.direction == PortDirection::Monitor {
                return Err(CompileError::MonitorAsSource(wire.from.clone()));
            }
            if from.direction != PortDirection::Output
                || !matches!(to.direction, PortDirection::Input | PortDirection::Monitor)
                || from.value != to.value
            {
                return Err(CompileError::IllTypedWire {
                    from: wire.from.clone(),
                    to: wire.to.clone(),
                });
            }
            if !permits_flow(from.visibility, to.visibility) {
                return Err(CompileError::VisibilityLeak {
                    from: wire.from.clone(),
                    to: wire.to.clone(),
                });
            }
        }
        let process_names: BTreeSet<&str> = [self.disturbance.as_ref(), self.scaffold.as_ref()]
            .into_iter()
            .flatten()
            .map(|process| process.name.as_str())
            .collect();
        if process_names.len() != self.disturbance.iter().count() + self.scaffold.iter().count() {
            return Err(CompileError::Invalid("process names must be unique".into()));
        }
        for process in [self.disturbance.as_ref(), self.scaffold.as_ref()]
            .into_iter()
            .flatten()
        {
            if process.name.trim().is_empty() {
                return Err(CompileError::Invalid("process has no name".into()));
            }
            for signal in &process.signals {
                let output = ports
                    .get(signal.output_port.as_str())
                    .ok_or_else(|| CompileError::UnknownPort(signal.output_port.clone()))?;
                if output.direction != PortDirection::Output || output.value != PortValue::Signal {
                    return Err(CompileError::Invalid(
                        "process signal must name a signal output port".into(),
                    ));
                }
                if output.visibility == Visibility::Generator {
                    return Err(CompileError::Invalid(
                        "generator metadata cannot be emitted as a process signal".into(),
                    ));
                }
            }
        }
        for reveal in &self.reveals {
            let output = ports
                .get(reveal.output_port.as_str())
                .ok_or_else(|| CompileError::UnknownPort(reveal.output_port.clone()))?;
            if !(output.direction == PortDirection::Output
                && output.value == PortValue::Signal
                && output.visibility == Visibility::Public)
            {
                return Err(CompileError::Invalid(
                    "a reveal must publish through a public signal output".into(),
                ));
            }
        }
        for interrupt in &self.interrupts {
            if !process_names.contains(interrupt.interrupted_process.as_str()) {
                return Err(CompileError::Invalid(
                    "interrupt names an unknown process".into(),
                ));
            }
        }
        for restriction in &self.restrictions {
            match restriction {
                Restriction::Action { supported }
                    if supported
                        .iter()
                        .any(|id| !actuator_ids.contains(&(id as u16))) =>
                {
                    return Err(CompileError::Invalid(
                        "action restriction refers to an undeclared actuator".into(),
                    ));
                }
                Restriction::Viability { inadmissible, .. }
                    if inadmissible.iter().any(|cell| cell >= states) =>
                {
                    return Err(CompileError::Invalid(
                        "viability restriction refers to an unknown cell".into(),
                    ));
                }
                Restriction::Resource { budget: 0, .. } => {
                    return Err(CompileError::Invalid(
                        "resource restriction needs a nonzero budget".into(),
                    ));
                }
                _ => {}
            }
        }
        for coupling in &self.couplings {
            if coupling.coupling.rule != CouplingRule::Sum && coupling.writers.is_empty() {
                return Err(CompileError::Invalid(
                    "override/conflict coupling needs a writer".into(),
                ));
            }
            if coupling.coupling.rule == CouplingRule::Conflict && coupling.writers.len() > 1 {
                return Err(CompileError::Invalid(
                    "conflict coupling may declare at most one writer".into(),
                ));
            }
        }
        if !self.environment.is_ring() && !self.couplings.is_empty() {
            return Err(CompileError::Unsupported(
                "displacement couplings are ring-only; graph transitions must not approximate them"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn derived_kernel_use(&self) -> KernelUse {
        KernelUse {
            directed_wiring: !self.wiring.is_empty(),
            shared_coupling: !self.couplings.is_empty(),
            interrupt: !self.interrupts.is_empty(),
            restrict: !self.restrictions.is_empty()
                || self.body.actuation.supported.len() != self.body.actuation.actuators.len(),
            reveal: !self.reveals.is_empty() || !self.restorations.is_empty(),
            norm_algebra: !self.norm.expression.connectives().is_empty(),
        }
    }

    pub fn compile(&self) -> Result<CompiledWorld, CompileError> {
        self.compile_with_metadata(None)
    }

    pub fn compile_with_metadata(
        &self,
        generator_metadata: Option<GeneratorMetadata>,
    ) -> Result<CompiledWorld, CompileError> {
        self.validate()?;
        let resource_budget = self
            .restrictions
            .iter()
            .filter_map(|restriction| match restriction {
                Restriction::Resource { budget, .. } => Some(*budget),
                _ => None,
            })
            .min();
        let horizon = resource_budget.map_or(self.horizon, |budget| self.horizon.min(budget));
        let transition_table = self.lower_transition_table(horizon);
        Ok(CompiledWorld {
            fragment: CompiledFragment {
                actions: self
                    .body
                    .actuation
                    .actuators
                    .iter()
                    .map(|actuator| actuator.id)
                    .collect(),
                horizon,
            },
            contract: CompiledContract {
                program: CompiledProgram {
                    states: self.environment.state_count(),
                    start: self.environment.start,
                    actuators: self.body.actuation.actuators.clone(),
                    supported: self.body.actuation.supported,
                    publishes_cell: self.body.sensorium.publishes_cell,
                    cell_port: self.body.sensorium.cell_port.clone(),
                    transition_table,
                    state_space: self.environment.state_space.clone(),
                    horizon,
                    norm: self.norm.expression.clone(),
                    norm_public: self.norm.visibility == Visibility::Public,
                    scoring: self.scoring.clone(),
                    calibration: self.calibration.clone(),
                    restorations: self.restorations.clone(),
                    restrictions: self.restrictions.clone(),
                    disturbance: self.disturbance.clone(),
                    scaffold: self.scaffold.clone(),
                    interrupts: self.interrupts.clone(),
                    reveals: self.reveals.clone(),
                    ports: self
                        .ports
                        .iter()
                        .map(|port| (port.name.clone(), port.clone()))
                        .collect(),
                    public_port_ids: public_port_ids(&self.ports),
                },
            },
            kernel_use: self.derived_kernel_use(),
            generator_metadata,
            family_hash: self.family_hash(),
        })
    }

    /// Lower all finite transition semantics before execution. Ring movement,
    /// delayed support, restrictions, and ring-only coupling are compiled into
    /// the same `(state, executed, actuator) -> state` table used for an
    /// explicit graph. Runtime stepping is therefore a table lookup, never a
    /// topology-specific evaluator.
    fn lower_transition_table(&self, horizon: usize) -> BTreeMap<(usize, usize, u16), usize> {
        let states = self.environment.state_count();
        let graph_edges: BTreeMap<(usize, u16), usize> = match &self.environment.state_space {
            StateSpaceTerm::Graph { transitions, .. } => transitions
                .iter()
                .map(|edge| ((edge.source, edge.actuator), edge.destination))
                .collect(),
            StateSpaceTerm::Ring { .. } => BTreeMap::new(),
        };
        let ring_blocks: BTreeSet<(usize, u16)> = match &self.environment.state_space {
            StateSpaceTerm::Ring { blocked_edges, .. } => blocked_edges.iter().copied().collect(),
            StateSpaceTerm::Graph { .. } => BTreeSet::new(),
        };
        let lowered_couplings: Vec<LoweredCoupling> = self
            .couplings
            .iter()
            .map(LoweredCoupling::from_term)
            .collect();
        let mut table = BTreeMap::new();
        for executed in 0..=horizon {
            for state in 0..states {
                for actuator in &self.body.actuation.actuators {
                    let next = if self
                        .restrictions
                        .iter()
                        .any(|restriction| !restriction.admits_cell(state))
                        || actuator.role == ActionRole::Fallback
                        || !(self
                            .body
                            .actuation
                            .supported
                            .contains(usize::from(actuator.id))
                            || self.restorations.iter().any(|restoration| {
                                restoration.actuator == actuator.id
                                    && executed > restoration.after_step
                            }))
                        || !self
                            .restrictions
                            .iter()
                            .all(|restriction| restriction.permits_action(actuator.id))
                    {
                        state
                    } else {
                        let moved = match &self.environment.state_space {
                            StateSpaceTerm::Ring { .. }
                                if ring_blocks.contains(&(state, actuator.id)) =>
                            {
                                state
                            }
                            StateSpaceTerm::Ring { .. } => {
                                let context = guard_context(executed + 1, Some(actuator.id), state);
                                let coupling_delta: i32 = lowered_couplings
                                    .iter()
                                    .map(|coupling| coupling.resolve(context))
                                    .sum();
                                (state as i32 + actuator.displacement + coupling_delta)
                                    .rem_euclid(states as i32)
                                    as usize
                            }
                            StateSpaceTerm::Graph { missing_edge, .. } => graph_edges
                                .get(&(state, actuator.id))
                                .copied()
                                .unwrap_or(match missing_edge {
                                    MissingEdgeBehavior::SelfLoop => state,
                                }),
                        };
                        self.restrictions
                            .iter()
                            .fold(moved, |candidate, restriction| {
                                if restriction.admits_cell(candidate) {
                                    candidate
                                } else {
                                    match restriction.boundary_effect() {
                                        Some(BoundaryEffect::Reset) => self.environment.start,
                                        Some(BoundaryEffect::Absorbing) => candidate,
                                        None => candidate,
                                    }
                                }
                            })
                    };
                    table.insert((state, executed, actuator.id), next);
                }
            }
        }
        table
    }

    /// Stable, provenance-aware family identifier.  It excludes the world
    /// label and generator replay metadata, canonicalizes sets/maps, and keeps
    /// body support distinct from environment edge deletions even when their
    /// transitions happen to agree.
    pub fn family_hash(&self) -> String {
        let mut canonical = self.clone();
        canonical.name.clear();
        canonical
            .ports
            .sort_by(|left, right| left.name.cmp(&right.name));
        canonical
            .wiring
            .sort_by(|left, right| (&left.from, &left.to).cmp(&(&right.from, &right.to)));
        canonical
            .body
            .actuation
            .actuators
            .sort_by_key(|actuator| actuator.id);
        match &mut canonical.environment.state_space {
            StateSpaceTerm::Ring { blocked_edges, .. } => {
                blocked_edges.sort_unstable();
                blocked_edges.dedup();
            }
            StateSpaceTerm::Graph { transitions, .. } => {
                transitions.sort_by_key(|edge| (edge.source, edge.actuator, edge.destination))
            }
        }
        // Preserve orders that affect execution: process signals, norm trees,
        // restriction precedence, restoration announcements, and Override
        // writer order.  `serde_json` provides a maintained canonical byte
        // encoding for this already-canonical term, and BLAKE3 provides the
        // stable digest rather than a local hash implementation.
        let bytes = serde_json::to_vec(&canonical).expect("world terms serialize");
        blake3::hash(&bytes).to_hex().to_string()
    }

    /// Compute the version-1 digest for a ring term using the former
    /// `morphology.cells`/`environment.blocked_edges` serialization shape.
    /// This is a migration aid for stored ring fixtures; new receipts use the
    /// version-2 [`family_hash`](Self::family_hash) and its explicit schema
    /// version.
    pub fn legacy_ring_family_hash(&self) -> Option<String> {
        if !self.environment.is_ring() {
            return None;
        }

        #[derive(Serialize)]
        struct LegacyMorphology {
            cells: usize,
        }
        #[derive(Serialize)]
        struct LegacyBody<'a> {
            morphology: LegacyMorphology,
            actuation: &'a Actuation,
            sensorium: &'a Sensorium,
        }
        #[derive(Serialize)]
        struct LegacyEnvironment {
            start: usize,
            blocked_edges: Vec<(usize, u16)>,
        }
        #[derive(Serialize)]
        struct LegacyWorld<'a> {
            name: &'a str,
            horizon: usize,
            ports: &'a [Port],
            wiring: &'a [Wiring],
            body: LegacyBody<'a>,
            environment: LegacyEnvironment,
            norm: &'a NormTerm,
            scoring: &'a ScoringTerm,
            calibration: &'a Option<CalibrationTerm>,
            restorations: &'a [SupportRestoration],
            disturbance: &'a Option<ProcessTerm>,
            scaffold: &'a Option<ProcessTerm>,
            couplings: &'a [CouplingTerm],
            interrupts: &'a [InterruptTerm],
            restrictions: &'a [Restriction],
            reveals: &'a [RevealTerm],
        }

        let mut canonical = self.clone();
        canonical.name.clear();
        canonical
            .ports
            .sort_by(|left, right| left.name.cmp(&right.name));
        canonical
            .wiring
            .sort_by(|left, right| (&left.from, &left.to).cmp(&(&right.from, &right.to)));
        canonical
            .body
            .actuation
            .actuators
            .sort_by_key(|actuator| actuator.id);
        let (start, mut blocked_edges) = match &canonical.environment.state_space {
            StateSpaceTerm::Ring { blocked_edges, .. } => {
                (canonical.environment.start, blocked_edges.clone())
            }
            StateSpaceTerm::Graph { .. } => return None,
        };
        blocked_edges.sort_unstable();
        blocked_edges.dedup();
        let legacy = LegacyWorld {
            name: &canonical.name,
            horizon: canonical.horizon,
            ports: &canonical.ports,
            wiring: &canonical.wiring,
            body: LegacyBody {
                morphology: LegacyMorphology {
                    cells: canonical.environment.state_count(),
                },
                actuation: &canonical.body.actuation,
                sensorium: &canonical.body.sensorium,
            },
            environment: LegacyEnvironment {
                start,
                blocked_edges,
            },
            norm: &canonical.norm,
            scoring: &canonical.scoring,
            calibration: &canonical.calibration,
            restorations: &canonical.restorations,
            disturbance: &canonical.disturbance,
            scaffold: &canonical.scaffold,
            couplings: &canonical.couplings,
            interrupts: &canonical.interrupts,
            restrictions: &canonical.restrictions,
            reveals: &canonical.reveals,
        };
        let bytes = serde_json::to_vec(&legacy).expect("legacy world terms serialize");
        Some(blake3::hash(&bytes).to_hex().to_string())
    }
}

fn public_port_ids(ports: &[Port]) -> BTreeMap<String, PublicPortId> {
    let mut names: Vec<&str> = ports
        .iter()
        .filter(|port| port.visibility == Visibility::Public)
        .map(|port| port.name.as_str())
        .collect();
    names.sort_unstable();
    names
        .into_iter()
        .enumerate()
        .map(|(index, name)| {
            (
                name.to_string(),
                PublicPortId(u16::try_from(index).expect("validated G0 port count")),
            )
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CompiledProgram {
    states: usize,
    start: usize,
    actuators: Vec<Actuator>,
    supported: IndexSet,
    publishes_cell: bool,
    cell_port: String,
    transition_table: BTreeMap<(usize, usize, u16), usize>,
    state_space: StateSpaceTerm,
    horizon: usize,
    norm: Norm,
    norm_public: bool,
    scoring: ScoringTerm,
    calibration: Option<CalibrationTerm>,
    restorations: Vec<SupportRestoration>,
    restrictions: Vec<Restriction>,
    disturbance: Option<ProcessTerm>,
    scaffold: Option<ProcessTerm>,
    interrupts: Vec<InterruptTerm>,
    reveals: Vec<RevealTerm>,
    ports: BTreeMap<String, Port>,
    public_port_ids: BTreeMap<String, PublicPortId>,
}

/// Runtime coupling is total because the compiler has provided a declared
/// inactive value and rejected ambiguous conflict writer sets.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LoweredCoupling {
    rule: CouplingRule,
    writers: Vec<CoupledWriter>,
    inactive_value: i32,
}

impl LoweredCoupling {
    fn from_term(term: &CouplingTerm) -> Self {
        Self {
            rule: term.coupling.rule,
            writers: term.writers.clone(),
            inactive_value: term.inactive_value,
        }
    }

    fn resolve(&self, context: GuardContext) -> i32 {
        let writes: Vec<i32> = self
            .writers
            .iter()
            .filter(|writer| writer.guard.fired(context))
            .map(|writer| writer.value)
            .collect();
        match self.rule {
            CouplingRule::Sum => {
                if writes.is_empty() {
                    self.inactive_value
                } else {
                    writes.into_iter().sum()
                }
            }
            CouplingRule::Override | CouplingRule::Conflict => {
                writes.last().copied().unwrap_or(self.inactive_value)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledContract {
    program: CompiledProgram,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledWorld {
    pub fragment: CompiledFragment,
    pub contract: CompiledContract,
    pub kernel_use: KernelUse,
    pub generator_metadata: Option<GeneratorMetadata>,
    pub family_hash: String,
}

/// Stateless interpreter for a [`CompiledContract`].  Keeping program state in
/// the contract permits an ambiguity set to contain several compiled worlds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledFragment {
    actions: Vec<u16>,
    horizon: usize,
}

impl CompiledProgram {
    fn actuator(&self, action: u16) -> Option<&Actuator> {
        self.actuators
            .iter()
            .find(|candidate| candidate.id == action)
    }

    fn interrupted(&self, process: &str, context: GuardContext) -> bool {
        self.interrupts.iter().any(|interrupt| {
            interrupt.interrupted_process == process
                && interrupt.specification.active(context)
                && interrupt.specification.displaced == Displaced::Frozen
        })
    }

    fn public_events(&self, executed: usize, action: Option<u16>, cell: usize) -> Vec<i64> {
        let context = guard_context(executed, action, cell);
        let mut events = Vec::new();
        for process in [self.disturbance.as_ref(), self.scaffold.as_ref()]
            .into_iter()
            .flatten()
        {
            if self.interrupted(process.name.as_str(), context) {
                continue;
            }
            for signal in &process.signals {
                let port = &self.ports[signal.output_port.as_str()];
                if port.visibility == Visibility::Public && signal.guard.fired(context) {
                    events.push(signal.value);
                }
            }
        }
        for reveal in &self.reveals {
            if reveal.guard.fired(context) {
                events.push(reveal.value);
            }
        }
        events
    }

    fn public_event_facts(
        &self,
        executed: usize,
        action: Option<u16>,
        cell: usize,
    ) -> Vec<PublicEvent> {
        let context = guard_context(executed, action, cell);
        let mut events = Vec::new();
        for process in [self.disturbance.as_ref(), self.scaffold.as_ref()]
            .into_iter()
            .flatten()
        {
            if self.interrupted(process.name.as_str(), context) {
                continue;
            }
            for signal in &process.signals {
                let Some(port) = self.ports.get(signal.output_port.as_str()) else {
                    continue;
                };
                if port.visibility == Visibility::Public && signal.guard.fired(context) {
                    if let Some(id) = self.public_port_ids.get(signal.output_port.as_str()) {
                        events.push(PublicEvent::Signal {
                            port: *id,
                            value: signal.value,
                        });
                    }
                }
            }
        }
        for reveal in &self.reveals {
            if reveal.guard.fired(context) {
                if let Some(id) = self.public_port_ids.get(reveal.output_port.as_str()) {
                    events.push(PublicEvent::Signal {
                        port: *id,
                        value: reveal.value,
                    });
                }
            }
        }
        events
    }

    fn transition(&self, cell: usize, executed: usize, action: u16) -> usize {
        self.transition_table
            .get(&(cell, executed.min(self.horizon), action))
            .copied()
            .unwrap_or(cell)
    }

    fn calibration_trace(&self) -> Option<Vec<usize>> {
        let calibration = self.calibration.as_ref()?;
        let mut cell = self.start;
        let mut trace = vec![cell];
        for action in &calibration.pulses {
            // A prelude action is not a scored decision.  Its support is read
            // at scored index zero and can never consume or advance that clock.
            cell = self.transition(cell, 0, *action);
            trace.push(cell);
        }
        Some(trace)
    }

    fn fallback_step(&self, actions: &[u16]) -> Option<usize> {
        actions.iter().position(|action| {
            self.actuator(*action)
                .is_some_and(|actuator| actuator.role == ActionRole::Fallback)
        })
    }
}

impl Fragment for CompiledFragment {
    type Action = u16;
    type Contract = CompiledContract;

    fn actions(&self) -> Vec<Self::Action> {
        self.actions.clone()
    }

    fn horizon(&self) -> usize {
        self.horizon
    }

    fn start(&self, contract: &Self::Contract) -> usize {
        contract.program.start
    }

    fn step(&self, contract: &Self::Contract, cell: usize, executed: usize, action: u16) -> usize {
        contract.program.transition(cell, executed, action)
    }

    fn value(
        &self,
        contract: &Self::Contract,
        trajectory: &[usize],
        actions: &[Self::Action],
    ) -> i32 {
        if let Some(step) = contract.program.fallback_step(actions) {
            return contract.program.scoring.fallback_reward
                - contract.program.scoring.action_cost * step as i32;
        }
        let last = *trajectory.last().unwrap_or(&contract.program.start);
        let last_action = actions.last().copied();
        let verdict = contract
            .program
            .norm
            .evaluate(trajectory, guard_context(actions.len(), last_action, last));
        if verdict.met {
            let cost_steps = verdict.settle_steps.unwrap_or(actions.len());
            contract.program.scoring.goal_reward
                - contract.program.scoring.action_cost * cost_steps as i32
        } else if verdict.violated_prohibition {
            contract.program.scoring.violation_penalty
        } else {
            0
        }
    }
}

impl PubliclyObservable for CompiledFragment {
    fn public_trace(&self, contract: &Self::Contract, actions: &[Self::Action]) -> Vec<i64> {
        let program = &contract.program;
        let mut cell = program.start;
        let mut trace = Vec::new();
        if let Some(calibration) = program.calibration_trace() {
            if program
                .calibration
                .as_ref()
                .is_some_and(|calibration| calibration.publishes_cells)
            {
                trace.extend(calibration.into_iter().map(|cell| cell as i64));
            }
        } else if program.publishes_cell {
            trace.push(cell as i64);
        }
        if program.norm_public {
            trace.push(norm_code(&program.norm));
        }
        trace.extend(
            program
                .restorations
                .iter()
                .map(|restoration| restoration.announcement_value),
        );
        trace.extend(program.public_events(0, None, cell));
        for (executed, action) in actions.iter().enumerate() {
            if program
                .actuator(*action)
                .is_some_and(|actuator| actuator.role == ActionRole::Fallback)
            {
                trace.push(-1);
                break;
            }
            cell = self.step(contract, cell, executed, *action);
            if program.publishes_cell {
                trace.push(cell as i64);
            }
            trace.extend(program.public_events(executed + 1, Some(*action), cell));
        }
        trace
    }
}

impl CompiledWorld {
    pub fn horizon(&self) -> usize {
        self.contract.program.horizon
    }

    pub fn actions(&self) -> Vec<u16> {
        self.fragment.actions()
    }

    /// Public schema metadata keyed by the compiler-assigned port identity.
    /// Port names are intentionally absent so callers cannot accidentally put
    /// compiler labels into learner payloads.
    pub fn public_ports(&self) -> Vec<PublicPortDescriptor> {
        let program = &self.contract.program;
        program
            .public_port_ids
            .iter()
            .filter_map(|(name, id)| {
                program.ports.get(name).map(|port| PublicPortDescriptor {
                    id: *id,
                    direction: port.direction,
                    value: port.value,
                })
            })
            .collect()
    }

    /// Exact shortest optimal non-fallback prefix. First enumerate the full
    /// horizon value ceiling, retain its non-fallback sequences, and then find
    /// the earliest prefix on one of those sequences whose own outcome reports
    /// norm satisfaction. This is an optimal-continuation definition and stays
    /// correct for Visit and composed norms whose prefix value differs from the
    /// fixed-horizon value. The empty prefix is included for completeness.
    pub fn shortest_optimal_non_fallback_plan_length(&self) -> Option<usize> {
        let (_, optimal_sequences) = value_bounds(&self.fragment, &self.contract);
        let actions: Vec<u16> = self
            .actions()
            .into_iter()
            .filter(|action| {
                self.contract
                    .program
                    .actuator(*action)
                    .is_some_and(|actuator| actuator.role != ActionRole::Fallback)
            })
            .collect();
        for length in 0..=self.horizon() {
            for sequence in &optimal_sequences {
                if sequence.iter().any(|action| !actions.contains(action)) {
                    continue;
                }
                let full = self.outcome(sequence);
                if full.fell_back {
                    continue;
                }
                let prefix = self.outcome(&sequence[..length]);
                if prefix.norm_met && !prefix.fell_back {
                    return Some(length);
                }
            }
        }
        None
    }

    /// Compatibility alias for callers using the pre-precise API name.
    pub fn shortest_non_fallback_success_length(&self) -> Option<usize> {
        self.shortest_optimal_non_fallback_plan_length()
    }

    /// Execute a scored sequence once, exposing only generic scoring facts.
    /// The v2 public trace exposes only its frozen norm token; callers that
    /// need to inspect the semantic objective use an audit view or the
    /// visibility-gated symbolic diagnostic below.
    pub fn outcome(&self, actions: &[u16]) -> ExecutionOutcome {
        let fallback_step = self.contract.program.fallback_step(actions);
        // A fallback is absorbing in scoring/public execution, not in the
        // ring's `Fragment::step` state.  Continue to use the generic
        // trajectory for enumeration, but report the terminal configuration
        // from the scored prefix so later supplied actions are unscored.
        let scored_actions = fallback_step.map_or(actions, |step| &actions[..step]);
        let trajectory =
            pretraining_g0_contract::trajectory(&self.fragment, &self.contract, scored_actions);
        let final_cell = *trajectory.last().expect("a trajectory includes start");
        let last_action = actions.last().copied();
        let norm_met = fallback_step.is_none()
            && self
                .contract
                .program
                .norm
                .evaluate(
                    &trajectory,
                    guard_context(actions.len(), last_action, final_cell),
                )
                .met;
        ExecutionOutcome {
            value: self.fragment.value(
                &self.contract,
                &pretraining_g0_contract::trajectory(&self.fragment, &self.contract, actions),
                actions,
            ),
            norm_met,
            fell_back: fallback_step.is_some(),
            fallback_step,
            final_cell,
        }
    }

    /// Learner-visible data only.  There is no conversion from this view to
    /// the private program fields used by a semantic audit.
    pub fn public_view(&self, actions: &[u16]) -> PublicView {
        PublicView {
            trace: self.fragment.public_trace(&self.contract, actions),
        }
    }

    /// Typed learner-visible events. This is additive to the frozen v2 trace:
    /// old audits and hashes continue to use [`Self::public_view`], while new
    /// corpus adapters can consume source-identified events without parsing
    /// integers by range.
    pub fn public_event_view(&self, actions: &[u16]) -> PublicEventView {
        let program = &self.contract.program;
        let mut groups = Vec::new();
        let port_id = |name: &str| program.public_port_ids[name];
        let cell_event = |cell: usize| PublicEvent::Observation {
            port: port_id(program.cell_port.as_str()),
            value: PublicEventValue::Cell(cell),
        };

        let mut cell = program.start;
        if let Some(calibration) = &program.calibration {
            let mut events = vec![PublicEvent::Boundary(PublicBoundary::CalibrationReset)];
            if calibration.publishes_cells {
                events.push(cell_event(cell));
            }
            groups.push(PublicEventGroup {
                time: PublicEventTime::Calibration { pulse: 0 },
                events,
            });
            for (pulse, actuator) in calibration.pulses.iter().copied().enumerate() {
                cell = program.transition(cell, 0, actuator);
                let mut events = vec![PublicEvent::ActionExecuted {
                    actuator,
                    fallback: program
                        .actuator(actuator)
                        .is_some_and(|candidate| candidate.role == ActionRole::Fallback),
                }];
                if calibration.publishes_cells {
                    events.push(cell_event(cell));
                }
                groups.push(PublicEventGroup {
                    time: PublicEventTime::Calibration { pulse: pulse + 1 },
                    events,
                });
            }
        }

        // Calibration is an isolated prelude. The scored task always begins
        // from the declared environment start, just as `outcome` and the
        // frozen trace do; calibration may publish evidence but cannot move
        // the scored state implicitly.
        cell = program.start;
        let mut opening = vec![PublicEvent::Boundary(PublicBoundary::TaskReset)];
        if program.publishes_cell {
            opening.push(cell_event(program.start));
        }
        if program.norm_public {
            opening.push(PublicEvent::Goal {
                diagnostic: SymbolicGoalDiagnostic::from_norm(&program.norm),
            });
        }
        for restoration in &program.restorations {
            opening.push(PublicEvent::RestorationAnnouncement {
                port: port_id(restoration.announcement_port.as_str()),
                actuator: restoration.actuator,
                activates_after: restoration.after_step,
                value: restoration.announcement_value,
            });
        }
        opening.extend(program.public_event_facts(0, None, cell));
        groups.push(PublicEventGroup {
            time: PublicEventTime::BeforeAction { executed: 0 },
            events: opening,
        });

        for (executed, action) in actions.iter().copied().enumerate() {
            let fallback = program
                .actuator(action)
                .is_some_and(|candidate| candidate.role == ActionRole::Fallback);
            let mut effects = vec![PublicEvent::ActionExecuted {
                actuator: action,
                fallback,
            }];
            if !fallback {
                cell = program.transition(cell, executed, action);
            }
            if program.publishes_cell && !fallback {
                effects.push(cell_event(cell));
            }
            if !fallback {
                effects.extend(program.public_event_facts(executed + 1, Some(action), cell));
            }
            groups.push(PublicEventGroup {
                time: PublicEventTime::AfterAction {
                    executed: executed + 1,
                },
                events: effects,
            });
            if fallback {
                break;
            }
        }
        groups.push(PublicEventGroup {
            time: PublicEventTime::EpisodeEnd,
            events: vec![PublicEvent::Boundary(PublicBoundary::EpisodeEnd)],
        });
        PublicEventView { groups }
    }

    /// The fully lowered, public transition function. Every action is present
    /// at every scored clock and source, including explicit fallback rows.
    pub fn public_transition_table(&self) -> PublicTransitionTable {
        let program = &self.contract.program;
        let mut rows =
            Vec::with_capacity((program.horizon + 1) * program.states * program.actuators.len());
        for executed in 0..=program.horizon {
            for source in 0..program.states {
                for actuator in &program.actuators {
                    rows.push(PublicTransition {
                        address: PublicTransitionAddress {
                            executed,
                            source,
                            actuator: actuator.id,
                        },
                        destination: program.transition(source, executed, actuator.id),
                        fallback: actuator.role == ActionRole::Fallback,
                    });
                }
            }
        }
        PublicTransitionTable {
            states: program.states,
            horizon: program.horizon,
            rows,
        }
    }

    /// A structured diagnostic corresponding to a public norm, if one exists.
    ///
    /// This value is deliberately not rendered by [`Self::public_view`]. The
    /// version-2 trace remains a one-slot opaque norm token; a future adapter
    /// must lower this diagnostic through its own framed event profile.
    pub fn public_goal_diagnostic(&self) -> Option<SymbolicGoalDiagnostic> {
        self.contract
            .program
            .norm_public
            .then(|| SymbolicGoalDiagnostic::from_norm(&self.contract.program.norm))
    }

    /// Explicit audit view.  The caller receives semantic state, never a
    /// renderer-ready record, so privileged data cannot enter public episodes
    /// by an accidental type conversion.
    pub fn privileged_view(&self, actions: &[u16]) -> PrivilegedView {
        PrivilegedView {
            trajectory: pretraining_g0_contract::trajectory(
                &self.fragment,
                &self.contract,
                actions,
            ),
            state_space: self.contract.program.state_space.clone(),
            supported: self.contract.program.supported,
        }
    }

    pub fn audit_metadata(&self) -> AuditMetadata {
        AuditMetadata {
            family_hash: self.family_hash.clone(),
            family_hash_schema_version: FAMILY_HASH_SCHEMA_VERSION,
            kernel_use: self.kernel_use.clone(),
            public_ports: self
                .contract
                .program
                .ports
                .values()
                .filter(|port| port.visibility == Visibility::Public)
                .map(|port| port.name.clone())
                .collect(),
            topology: EnvironmentTerm {
                start: self.contract.program.start,
                state_space: self.contract.program.state_space.clone(),
            }
            .topology_diagnostics(),
        }
    }
}

/// A deterministic identity for a learner-visible compiler port. IDs are
/// assigned from the sorted public port names at compile time; names remain
/// compiler metadata and never enter a learner event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PublicPortId(pub u16);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicPortDescriptor {
    pub id: PublicPortId,
    pub direction: PortDirection,
    pub value: PortValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PublicBoundary {
    CalibrationReset,
    TaskReset,
    EpisodeEnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PublicEventTime {
    Calibration { pulse: usize },
    BeforeAction { executed: usize },
    AfterAction { executed: usize },
    EpisodeEnd,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PublicEventValue {
    Cell(usize),
    Scalar(i64),
    Selection,
}

/// A typed event that is safe to hand to a corpus adapter. It contains no
/// generator metadata, private support, or latent transition state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PublicEvent {
    Boundary(PublicBoundary),
    Observation {
        port: PublicPortId,
        value: PublicEventValue,
    },
    Goal {
        diagnostic: SymbolicGoalDiagnostic,
    },
    Signal {
        port: PublicPortId,
        value: i64,
    },
    RestorationAnnouncement {
        port: PublicPortId,
        actuator: u16,
        activates_after: usize,
        value: i64,
    },
    ActionExecuted {
        actuator: u16,
        fallback: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicEventGroup {
    pub time: PublicEventTime,
    pub events: Vec<PublicEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicEventView {
    pub groups: Vec<PublicEventGroup>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PublicTransitionAddress {
    pub executed: usize,
    pub source: usize,
    pub actuator: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicTransition {
    pub address: PublicTransitionAddress,
    pub destination: usize,
    pub fallback: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicTransitionTable {
    pub states: usize,
    pub horizon: usize,
    pub rows: Vec<PublicTransition>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicView {
    pub trace: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionOutcome {
    pub value: i32,
    pub norm_met: bool,
    pub fell_back: bool,
    pub fallback_step: Option<usize>,
    pub final_cell: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrivilegedView {
    pub trajectory: Vec<usize>,
    pub state_space: StateSpaceTerm,
    pub supported: IndexSet,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditMetadata {
    pub family_hash: String,
    pub family_hash_schema_version: u16,
    pub kernel_use: KernelUse,
    pub public_ports: Vec<String>,
    pub topology: TopologyDiagnostics,
}

/// Bounded construction metadata.  It is never copied into [`WorldTerm`] or a
/// public trace; replay is instead the pair `(specification, index)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationSpec {
    pub seed: u64,
    pub count: usize,
    /// Ring remains the default for compatibility; graph generation is an
    /// explicit non-ring template, never an accidental change of the old
    /// constructor.
    pub template: GenerationTemplate,
    pub min_cells: usize,
    pub max_cells: usize,
    pub min_horizon: usize,
    pub max_horizon: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenerationTemplate {
    Ring,
    BranchingGraph,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratorMetadata {
    pub seed: u64,
    pub index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedWorld {
    pub term: WorldTerm,
    pub generator_metadata: GeneratorMetadata,
}

impl GeneratedWorld {
    pub fn compile(&self) -> Result<CompiledWorld, CompileError> {
        self.term
            .compile_with_metadata(Some(self.generator_metadata.clone()))
    }
}

/// A small standalone G0 corpus specification.
///
/// [`GenerationSpec`] produces paired embodiment contrasts with an exact
/// receipt. This specification is the intentionally lighter corpus surface:
/// it emits independently varied finite worlds for pretraining and compiles
/// every term before returning it. The generator metadata remains separate
/// from the term, so replay coordinates do not affect semantic identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct G0CorpusSpec {
    pub seed: u64,
    pub count: usize,
    pub topology: G0Topology,
    /// Deterministic feature coverage for the standalone corpus.  The
    /// schedule is generator metadata and never enters the learner episode.
    pub feature_schedule: G0FeatureSchedule,
    /// Target weights for the exact shortest non-fallback success length.
    /// The first weight is length one, the second length two, and so on.
    pub plan_length_distribution: PlanLengthDistribution,
    pub min_states: usize,
    pub max_states: usize,
    pub min_horizon: usize,
    pub max_horizon: usize,
}

/// Feature mixtures are deliberately a small explicit contract rather than a
/// collection of independent booleans.  `Coverage` rotates isolated feature
/// arms while preserving the exact plan-length quotas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum G0FeatureSchedule {
    Plain,
    Coverage,
}

impl Default for G0FeatureSchedule {
    fn default() -> Self {
        Self::Coverage
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum G0FeatureArm {
    Plain,
    Calibration,
    Restoration,
    DisturbanceEvent,
    ScaffoldEvent,
    Coupling,
    InterruptEvent,
    RevealEvent,
    RestrictionAction,
    RestrictionViability,
    RestrictionResource,
}

const G0_COVERAGE_ARMS: [G0FeatureArm; 11] = [
    G0FeatureArm::Plain,
    G0FeatureArm::Calibration,
    G0FeatureArm::Restoration,
    G0FeatureArm::DisturbanceEvent,
    G0FeatureArm::ScaffoldEvent,
    G0FeatureArm::Coupling,
    G0FeatureArm::InterruptEvent,
    G0FeatureArm::RevealEvent,
    G0FeatureArm::RestrictionAction,
    G0FeatureArm::RestrictionViability,
    G0FeatureArm::RestrictionResource,
];

/// Topology distribution used by [`G0CorpusSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum G0Topology {
    Chain,
    Star,
    Tree,
    Dag,
    Grid,
    Lattice,
    Ring,
    Graph,
    /// Select a topology independently for each generated index.
    Mixed,
}

/// A finite, serializable target distribution over exact plan lengths.
///
/// Weights are nonnegative integers with a positive total, which makes quota
/// realization independent of floating point arithmetic. A vector of length
/// `N` declares buckets for lengths `1..=N`; zero weights omit a bucket.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanLengthDistribution {
    pub weights: Vec<u64>,
}

impl PlanLengthDistribution {
    pub fn new(weights: Vec<u64>) -> Result<Self, CompileError> {
        let distribution = Self { weights };
        distribution.validate()?;
        Ok(distribution)
    }

    pub fn validate(&self) -> Result<(), CompileError> {
        if self.weights.is_empty() {
            return Err(CompileError::Invalid(
                "G0 plan-length distribution needs weights for lengths 1..=N".into(),
            ));
        }
        if self.weights.len() > 8 {
            return Err(CompileError::Invalid(
                "G0 plan-length distribution supports at most eight lengths".into(),
            ));
        }
        let Some(total) = self
            .weights
            .iter()
            .try_fold(0u64, |total, weight| total.checked_add(*weight))
        else {
            return Err(CompileError::Invalid(
                "G0 plan-length distribution weights overflow".into(),
            ));
        };
        if total == 0 {
            return Err(CompileError::Invalid(
                "G0 plan-length distribution needs a positive total weight".into(),
            ));
        }
        Ok(())
    }

    pub fn max_length(&self) -> usize {
        self.weights.len()
    }

    pub fn max_requested_length(&self) -> usize {
        self.weights
            .iter()
            .rposition(|weight| *weight > 0)
            .map_or(0, |index| index + 1)
    }

    /// Largest-remainder quotas. Ties are assigned to the shorter plan first,
    /// giving a total deterministic function of `(weights, count)`.
    pub fn quotas(&self, count: usize) -> Result<Vec<usize>, CompileError> {
        self.validate()?;
        let total_weight = self
            .weights
            .iter()
            .try_fold(0u64, |total, weight| total.checked_add(*weight))
            .expect("validated distribution has a finite weight sum");
        let mut quotas = Vec::with_capacity(self.weights.len());
        let mut remainders = Vec::with_capacity(self.weights.len());
        let mut assigned = 0usize;
        for weight in &self.weights {
            let numerator = (count as u128) * (*weight as u128);
            let quotient = (numerator / total_weight as u128) as usize;
            let remainder = numerator % total_weight as u128;
            quotas.push(quotient);
            remainders.push(remainder);
            assigned += quotient;
        }
        let mut leftover = count - assigned;
        let mut order: Vec<usize> = (0..self.weights.len()).collect();
        order.sort_by(|left, right| {
            remainders[*right]
                .cmp(&remainders[*left])
                .then_with(|| left.cmp(right))
        });
        for index in order {
            if leftover == 0 {
                break;
            }
            quotas[index] += 1;
            leftover -= 1;
        }
        Ok(quotas)
    }
}

impl Default for G0CorpusSpec {
    fn default() -> Self {
        Self {
            seed: 0,
            count: 16,
            topology: G0Topology::Mixed,
            feature_schedule: G0FeatureSchedule::Coverage,
            plan_length_distribution: PlanLengthDistribution {
                weights: vec![1, 1, 1, 1, 1, 1],
            },
            min_states: 4,
            max_states: 8,
            min_horizon: 2,
            max_horizon: 6,
        }
    }
}

fn rectangular_shapes(min_states: usize, max_states: usize) -> Vec<(usize, usize)> {
    let mut shapes = Vec::new();
    for rows in 2..=max_states {
        for columns in 2..=max_states {
            let states = rows.saturating_mul(columns);
            if states >= min_states && states <= max_states {
                shapes.push((rows, columns));
            }
        }
    }
    shapes
}

fn topology_feasible_in_range(
    topology: G0Topology,
    min_states: usize,
    max_states: usize,
    target_length: usize,
) -> bool {
    match topology {
        G0Topology::Chain | G0Topology::Dag | G0Topology::Graph => max_states >= target_length + 1,
        G0Topology::Star => {
            target_length <= 2
                && max_states >= target_length + 1
                && max_states <= usize::from(G0_HOLD_ID) + 1
        }
        // A branched tree reserves two nodes for the branch and retains a
        // backbone route of length `states - 2`.
        G0Topology::Tree => max_states >= target_length + 2,
        G0Topology::Ring => max_states >= target_length.saturating_mul(2) + 1,
        G0Topology::Grid | G0Topology::Lattice => rectangular_shapes(min_states, max_states)
            .into_iter()
            .any(|(rows, columns)| {
                let diameter = if topology == G0Topology::Grid {
                    rows + columns - 2
                } else {
                    (rows / 2) + (columns / 2)
                };
                diameter >= target_length
            }),
        G0Topology::Mixed => false,
    }
}

fn choose_topology_states(
    topology: G0Topology,
    min_states: usize,
    max_states: usize,
    target_length: usize,
    rng: &mut ChaCha8Rng,
) -> Option<usize> {
    if matches!(topology, G0Topology::Grid | G0Topology::Lattice) {
        let mut choices: Vec<usize> = rectangular_shapes(min_states, max_states)
            .into_iter()
            .filter(|(rows, columns)| {
                let diameter = if topology == G0Topology::Grid {
                    rows + columns - 2
                } else {
                    (rows / 2) + (columns / 2)
                };
                diameter >= target_length
            })
            .map(|(rows, columns)| rows * columns)
            .collect();
        choices.sort_unstable();
        choices.dedup();
        return choices.choose(rng).copied();
    }
    let required = match topology {
        G0Topology::Star => target_length + 1,
        G0Topology::Tree => target_length + 2,
        G0Topology::Ring => target_length.saturating_mul(2) + 1,
        _ => target_length + 1,
    };
    let lower = min_states.max(required);
    (lower <= max_states).then(|| rng.gen_range(lower..=max_states))
}

fn g0_viability_cell(term: &WorldTerm, target_length: usize) -> Option<usize> {
    let states = term.environment.state_count();
    let mut adjacency = vec![Vec::new(); states];
    match &term.environment.state_space {
        StateSpaceTerm::Ring { blocked_edges, .. } => {
            for cell in 0..states {
                for actuator in term
                    .body
                    .actuation
                    .actuators
                    .iter()
                    .filter(|actuator| actuator.role == ActionRole::Movement)
                {
                    if !blocked_edges.contains(&(cell, actuator.id)) {
                        adjacency[cell].push(
                            (cell as i32 + actuator.displacement).rem_euclid(states as i32)
                                as usize,
                        );
                    }
                }
            }
        }
        StateSpaceTerm::Graph { transitions, .. } => {
            for transition in transitions {
                adjacency[transition.source].push(transition.destination);
            }
        }
    }
    let distances = |origin: usize, graph: &[Vec<usize>]| {
        let mut distance = vec![usize::MAX; states];
        let mut queue = vec![origin];
        distance[origin] = 0;
        let mut cursor = 0;
        while let Some(&cell) = queue.get(cursor) {
            cursor += 1;
            let next_distance = distance[cell].saturating_add(1);
            for &next in &graph[cell] {
                if distance[next] == usize::MAX {
                    distance[next] = next_distance;
                    queue.push(next);
                }
            }
        }
        distance
    };
    let goal = match term.norm.expression {
        Norm::Settle { cell } => cell,
        _ => return None,
    };
    let from_start = distances(term.environment.start, &adjacency);
    let mut reverse = vec![Vec::new(); states];
    for (source, destinations) in adjacency.iter().enumerate() {
        for &destination in destinations {
            reverse[destination].push(source);
        }
    }
    let to_goal = distances(goal, &reverse);
    (0..states).find(|cell| {
        *cell != goal && from_start[*cell].saturating_add(to_goal[*cell]) > target_length
    })
}

/// Apply one isolated corpus feature after the environment topology has been
/// constructed.  `false` means that the arm is not feasible for this term and
/// the caller should reject the candidate without weakening its plan quota.
fn apply_g0_feature(term: &mut WorldTerm, arm: G0FeatureArm, target_length: usize) -> bool {
    let movement: Vec<u16> = term
        .body
        .actuation
        .actuators
        .iter()
        .filter(|actuator| actuator.role == ActionRole::Movement)
        .map(|actuator| actuator.id)
        .collect();
    let Some(&route) = movement.first() else {
        return arm == G0FeatureArm::Plain;
    };
    let Some(&alternate) = movement.last().filter(|id| **id != route) else {
        return arm == G0FeatureArm::Plain;
    };
    let signal_guard = Guard::AfterStep(target_length.saturating_sub(1));
    match arm {
        G0FeatureArm::Plain => {}
        G0FeatureArm::Calibration => {
            term.calibration = Some(CalibrationTerm {
                pulses: vec![route, G0_HOLD_ID],
                publishes_cells: true,
            });
        }
        G0FeatureArm::Restoration => {
            term.body.actuation.supported.remove(usize::from(alternate));
            term.restorations.push(SupportRestoration {
                actuator: alternate,
                // Activate before the final route step whenever possible so
                // this arm changes lowered transitions, not just metadata.
                after_step: target_length.saturating_sub(2),
                announcement_port: "event".into(),
                announcement_value: 6_000 + target_length as i64,
            });
        }
        G0FeatureArm::DisturbanceEvent => {
            term.disturbance = Some(ProcessTerm {
                name: "g0-disturbance".into(),
                signals: vec![SignalTerm {
                    output_port: "event".into(),
                    guard: signal_guard,
                    value: 7_000 + target_length as i64,
                }],
            });
        }
        G0FeatureArm::ScaffoldEvent => {
            term.scaffold = Some(ProcessTerm {
                name: "g0-scaffold".into(),
                signals: vec![SignalTerm {
                    output_port: "event".into(),
                    guard: Guard::AtStart,
                    value: 8_000 + target_length as i64,
                }],
            });
        }
        G0FeatureArm::Coupling => {
            if !term.environment.is_ring() {
                return false;
            }
            term.couplings.push(CouplingTerm {
                coupling: Coupling::new(0, CouplingRule::Sum),
                writers: vec![CoupledWriter {
                    guard: Guard::OnAction(alternate),
                    value: 1,
                }],
                inactive_value: 0,
            });
        }
        G0FeatureArm::InterruptEvent => {
            term.disturbance = Some(ProcessTerm {
                name: "g0-interrupt-process".into(),
                signals: vec![SignalTerm {
                    output_port: "event".into(),
                    // The process is visible before the interruption, then
                    // the point interrupt suppresses it only for the route
                    // action, and it resumes on a later action.
                    guard: Guard::AtStart,
                    value: 9_000 + target_length as i64,
                }],
            });
            term.interrupts.push(InterruptTerm {
                specification: Interrupt::new(
                    Guard::OnAction(route),
                    Displaced::Frozen,
                    pretraining_g0_contract::Resume::FromState,
                ),
                interrupted_process: "g0-interrupt-process".into(),
            });
        }
        G0FeatureArm::RevealEvent => {
            term.reveals.push(RevealTerm {
                source_visibility: Visibility::Privileged,
                output_port: "event".into(),
                guard: signal_guard,
                value: 10_000 + target_length as i64,
            });
        }
        G0FeatureArm::RestrictionAction => {
            let mut supported = term.body.actuation.supported;
            supported.remove(usize::from(alternate));
            term.restrictions.push(Restriction::Action { supported });
        }
        G0FeatureArm::RestrictionViability => {
            let Some(cell) = g0_viability_cell(term, target_length) else {
                return false;
            };
            term.restrictions.push(Restriction::Viability {
                inadmissible: IndexSet::from_indices([cell]),
                effect: BoundaryEffect::Reset,
            });
        }
        G0FeatureArm::RestrictionResource => {
            term.restrictions.push(Restriction::Resource {
                budget: target_length.max(1),
                scope: pretraining_g0_contract::ResourceScope::Shared,
            });
        }
    }
    true
}

/// The shared admission predicate has a singleton specialization here. It is
/// still exact (the same finite value ceiling and first-action set), but avoids
/// the public-belief recursion, which is vacuous when there is no hidden
/// alternative. This keeps corpus materialization bounded at the default
/// count/horizon while retaining the shared report shape.
fn assess_g0_singleton(world: &CompiledWorld) -> AcceptanceReport {
    let actions = world.actions();
    let (public_ceiling_i32, optimal_sequences) = value_bounds(&world.fragment, &world.contract);
    let mut optimal_first = BTreeSet::new();
    for sequence in &optimal_sequences {
        if let Some(action) = sequence.first() {
            optimal_first.insert(*action);
        }
    }
    let start_goal_met = world.outcome(&[]).norm_met;
    AcceptanceReport {
        goal_holds_at_start: start_goal_met,
        action_count: actions.len(),
        optimal_first_action_count: optimal_first.len(),
        public_ceiling: f64::from(public_ceiling_i32),
        privileged_ceiling: f64::from(public_ceiling_i32),
        ambiguity_gap: 0.0,
        informative_action_attains_public_ceiling: false,
        hidden_state_noninterference: true,
        hidden_pairs_checked: 0,
        accepted: !start_goal_met && optimal_first.len() < actions.len(),
    }
}

impl G0CorpusSpec {
    /// Run the shared exact semantic admission filter for one public world.
    /// G0 corpus worlds have no hidden alternatives, so the ambiguity set is a
    /// singleton and there are no hidden-state pairs to audit.
    pub fn assess_world(world: &GeneratedWorld) -> Result<AcceptanceReport, CompileError> {
        let compiled = world.compile()?;
        Ok(assess_g0_singleton(&compiled))
    }

    fn scheduled_feature(&self, ordinal: usize, topology: G0Topology) -> G0FeatureArm {
        if self.feature_schedule == G0FeatureSchedule::Plain {
            return G0FeatureArm::Plain;
        }
        for offset in 0..G0_COVERAGE_ARMS.len() {
            let arm = G0_COVERAGE_ARMS[(ordinal + offset) % G0_COVERAGE_ARMS.len()];
            // Coupling is a ring transition overlay in the current finite
            // semantics. Do not silently place it on a graph term.
            if arm == G0FeatureArm::Coupling && topology != G0Topology::Ring {
                continue;
            }
            return arm;
        }
        G0FeatureArm::Plain
    }

    pub fn validate(&self) -> Result<(), CompileError> {
        if self.count == 0 || self.count > 1024 {
            return Err(CompileError::Invalid(
                "G0 corpus count must be in 1..=1024".into(),
            ));
        }
        if !(2..=IndexSet::CAPACITY).contains(&self.min_states)
            || self.min_states > self.max_states
            || self.max_states > IndexSet::CAPACITY
        {
            return Err(CompileError::Invalid(
                "G0 state range must be in 2..=32".into(),
            ));
        }
        if self.min_horizon == 0 || self.min_horizon > self.max_horizon || self.max_horizon > 8 {
            return Err(CompileError::Invalid(
                "G0 horizon range must be in 1..=8".into(),
            ));
        }
        self.plan_length_distribution.validate()?;
        if self.plan_length_distribution.max_requested_length() > self.max_horizon {
            return Err(CompileError::Invalid(format!(
                "G0 plan-length support 1..={} exceeds max horizon {}",
                self.plan_length_distribution.max_requested_length(),
                self.max_horizon
            )));
        }
        if self.max_states < self.plan_length_distribution.max_requested_length() + 1 {
            return Err(CompileError::Invalid(format!(
                "G0 state range cannot realize plan length {}: need at least {} states",
                self.plan_length_distribution.max_requested_length(),
                self.plan_length_distribution.max_requested_length() + 1
            )));
        }
        if self.topology != G0Topology::Mixed
            && self
                .plan_length_distribution
                .weights
                .iter()
                .enumerate()
                .any(|(index, weight)| {
                    *weight > 0
                        && !topology_feasible_in_range(
                            self.topology,
                            self.min_states,
                            self.max_states,
                            index + 1,
                        )
                })
        {
            return Err(CompileError::Invalid(
                "G0 topology cannot realize every positive plan-length bucket in the declared state range"
                    .into(),
            ));
        }
        Ok(())
    }

    /// Generate a reproducible corpus of valid finite deterministic worlds.
    ///
    /// Graph worlds use a guaranteed in-budget route plus sampled transition
    /// rows, which gives the corpus structural variation beyond changing
    /// labels or state counts. Ring worlds vary their actuator meanings and
    /// environment edge deletions. All
    /// emitted terms use the existing validator/lowering path as an admission
    /// check; no unvalidated candidate is returned.
    pub fn generate(&self) -> Result<Vec<GeneratedWorld>, CompileError> {
        self.validate()?;
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut worlds = Vec::with_capacity(self.count);
        let mut semantic_hashes = BTreeSet::new();
        let mut seen_features = BTreeSet::new();
        let quotas = self.plan_length_distribution.quotas(self.count)?;
        let mut requested_lengths = Vec::with_capacity(self.count);
        for (index, quota) in quotas.iter().enumerate() {
            requested_lengths.extend(std::iter::repeat(index + 1).take(*quota));
        }
        // Rejection is deterministic because both the RNG stream and the
        // candidate index advance exactly once per attempt. The cap prevents
        // an accidentally impossible admission predicate from looping.
        let max_attempts = self.count.saturating_mul(32).saturating_add(64);
        let mut attempts = 0usize;
        let mut last_feature = G0FeatureArm::Plain;
        let mut last_target_length = 0usize;
        while worlds.len() < self.count && attempts < max_attempts {
            let index = attempts;
            attempts += 1;
            let target_length = requested_lengths[worlds.len()];
            let horizon = rng.gen_range(self.min_horizon.max(target_length)..=self.max_horizon);
            let mut topology = match self.topology {
                topology @ (G0Topology::Chain
                | G0Topology::Star
                | G0Topology::Tree
                | G0Topology::Dag
                | G0Topology::Grid
                | G0Topology::Lattice
                | G0Topology::Ring
                | G0Topology::Graph) => topology,
                G0Topology::Mixed => match rng.gen_range(0..8) {
                    0 => G0Topology::Chain,
                    1 => G0Topology::Star,
                    2 => G0Topology::Tree,
                    3 => G0Topology::Dag,
                    4 => G0Topology::Grid,
                    5 => G0Topology::Lattice,
                    6 => G0Topology::Ring,
                    _ => G0Topology::Graph,
                },
            };
            // Coupling lowers to displacement arithmetic and is therefore
            // deliberately ring-only. Force that scheduled arm onto a ring
            // in Mixed mode so coverage does not depend on a lucky draw.
            if self.feature_schedule == G0FeatureSchedule::Coverage
                && G0_COVERAGE_ARMS[worlds.len() % G0_COVERAGE_ARMS.len()] == G0FeatureArm::Coupling
                && self.topology == G0Topology::Mixed
                && topology_feasible_in_range(
                    G0Topology::Ring,
                    self.min_states,
                    self.max_states,
                    target_length,
                )
            {
                topology = G0Topology::Ring;
            }
            if !topology_feasible_in_range(
                topology,
                self.min_states,
                self.max_states,
                target_length,
            ) {
                continue;
            }
            let Some(states) = choose_topology_states(
                topology,
                self.min_states,
                self.max_states,
                target_length,
                &mut rng,
            ) else {
                continue;
            };
            let segments = rng.gen_range(1..=4);
            // Keep exact semantic admission bounded at longer horizons: the
            // action alphabet still varies across worlds, while avoiding an
            // exponential query tree for every six-step candidate.
            let movement_count = match topology {
                G0Topology::Star => states.saturating_sub(1),
                G0Topology::Grid | G0Topology::Lattice | G0Topology::Tree => 4,
                _ if horizon >= 5 => 2,
                _ => rng.gen_range(2..=4),
            };
            let mut term = match topology {
                G0Topology::Ring => g0_ring_term(
                    format!("g0-world-{index}"),
                    states,
                    horizon,
                    target_length,
                    segments,
                    movement_count,
                    &mut rng,
                ),
                topology => g0_graph_term(
                    format!("g0-world-{index}"),
                    states,
                    horizon,
                    target_length,
                    segments,
                    movement_count,
                    topology,
                    &mut rng,
                ),
            };
            let feature = self.scheduled_feature(worlds.len(), topology);
            last_feature = feature;
            last_target_length = target_length;
            if !apply_g0_feature(&mut term, feature, target_length) {
                continue;
            }
            let world = GeneratedWorld {
                term,
                generator_metadata: GeneratorMetadata {
                    seed: self.seed,
                    index,
                },
            };
            // Names and generator coordinates are excluded from semantic
            // identity, so do not emit the same world twice in one stream.
            if !semantic_hashes.insert(world.term.family_hash()) {
                continue;
            }
            let compiled = world.compile()?;
            let report = assess_g0_singleton(&compiled);
            let measured = compiled.shortest_optimal_non_fallback_plan_length();
            if report.accepted && measured == Some(target_length) {
                seen_features.insert(feature);
                worlds.push(world);
            }
        }
        if worlds.len() != self.count {
            return Err(CompileError::Invalid(format!(
                "G0 corpus admitted {} of {} worlds after {} deterministic attempts (last arm {last_feature:?}, length {last_target_length})",
                worlds.len(),
                self.count,
                attempts
            )));
        }
        if self.feature_schedule == G0FeatureSchedule::Coverage
            && self.topology == G0Topology::Mixed
            && self.count >= G0_COVERAGE_ARMS.len()
        {
            let missing: Vec<G0FeatureArm> = G0_COVERAGE_ARMS
                .iter()
                .copied()
                .filter(|arm| !seen_features.contains(arm))
                .collect();
            if !missing.is_empty() {
                return Err(CompileError::Invalid(format!(
                    "G0 Coverage schedule could not realize feature arms: {missing:?}"
                )));
            }
        }
        Ok(worlds)
    }
}

/// A generated embodiment contrast, not three unrelated sampled worlds.  The
/// body-limited witness is paired with its unrestricted control and with an
/// environment-edge twin that preserves the witness's executable behavior.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedEmbodimentFamily {
    pub body_limited: GeneratedWorld,
    pub unrestricted_control: GeneratedWorld,
    pub environment_twin: GeneratedWorld,
    pub receipt: EmbodimentValidityReceipt,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbodimentValidityReceipt {
    pub sequences_checked: usize,
    pub topology_total: bool,
    pub topology_state_count: usize,
    pub topology_transition_rows: usize,
    pub topology_degree_sequence: Vec<usize>,
    pub topology_is_simple_cycle: bool,
    pub twin_scope: TwinScope,
    pub family_hash: String,
    pub generator_seed: u64,
    pub generator_index: usize,
    pub goal_differs_from_start: bool,
    pub body_limitation_changes_ceiling: bool,
    pub twin_trajectories_equal: bool,
    pub twin_values_equal: bool,
    pub twin_optimal_sequences_equal: bool,
    /// The preserving twin must still be observable through the calibration
    /// prelude; otherwise the orbit would compare a public episode with itself.
    pub twin_publicly_distinct: bool,
    /// Generic query-algebra evidence: the public prelude collapses the two
    /// body candidates from an initial ambiguity class to one survivor.
    pub calibration_identifies_body: bool,
    pub valid: bool,
}

/// Scope used when constructing environment-side withheld transitions.
/// `ConservativeTrajectoryCells` is intentionally explicit in receipts: it is
/// not exact command-site provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TwinScope {
    ExactCommandSites,
    ConservativeTrajectoryCells,
}

impl GeneratedEmbodimentFamily {
    /// Re-run the exact finite filter that admits one generated family.  The
    /// receipt is evidence about executable terms, not learner evidence.
    pub fn verify(&self) -> Result<EmbodimentValidityReceipt, CompileError> {
        let body = self.body_limited.compile()?;
        let unrestricted = self.unrestricted_control.compile()?;
        let twin = self.environment_twin.compile()?;
        let topology = self.body_limited.term.environment.topology_diagnostics();
        let topology_total = body.contract.program.transition_table.len()
            == topology.state_count * (body.horizon() + 1) * body.actions().len();
        if body.actions() != unrestricted.actions()
            || body.actions() != twin.actions()
            || body.horizon() != unrestricted.horizon()
            || body.horizon() != twin.horizon()
        {
            return Err(CompileError::Invalid(
                "an embodiment family must share action alphabet and scored horizon".into(),
            ));
        }
        // Composite norms need an exact, declared nondegeneracy witness rather
        // than a leaf heuristic. The existing generator has only bare
        // `Settle` goals, so retain its original admission rule until that
        // witness and its receipt vocabulary are specified.
        let goal_differs_from_start = match &self.body_limited.term.norm.expression {
            Norm::Settle { cell } | Norm::Visit { cell } => *cell != body.contract.program.start,
            _ => false,
        };
        let (body_ceiling, body_optimal) =
            pretraining_g0_contract::value_bounds(&body.fragment, &body.contract);
        let (unrestricted_ceiling, _) =
            pretraining_g0_contract::value_bounds(&unrestricted.fragment, &unrestricted.contract);
        let (twin_ceiling, twin_optimal) =
            pretraining_g0_contract::value_bounds(&twin.fragment, &twin.contract);
        let mut trajectories_equal = true;
        let mut values_equal = true;
        let actions = body.actions();
        let sequences = pretraining_g0_contract::sequences_of_length(&actions, body.horizon());
        for sequence in &sequences {
            let body_path =
                pretraining_g0_contract::trajectory(&body.fragment, &body.contract, sequence);
            let twin_path =
                pretraining_g0_contract::trajectory(&twin.fragment, &twin.contract, sequence);
            trajectories_equal &= body_path == twin_path;
            values_equal &= body.fragment.value(&body.contract, &body_path, sequence)
                == twin.fragment.value(&twin.contract, &twin_path, sequence);
        }
        let twin_publicly_distinct = body.public_view(&[]).trace != twin.public_view(&[]).trace;
        let calibration_identifies_body = body_identification_after_calibration(
            &self.body_limited.term,
            &self.unrestricted_control.term,
        )?;
        let receipt = EmbodimentValidityReceipt {
            sequences_checked: sequences.len(),
            topology_total,
            topology_state_count: topology.state_count,
            topology_transition_rows: topology.transition_rows,
            topology_degree_sequence: topology.degree_sequence.clone(),
            topology_is_simple_cycle: topology.is_simple_cycle,
            twin_scope: TwinScope::ConservativeTrajectoryCells,
            family_hash: body.family_hash.clone(),
            generator_seed: self.body_limited.generator_metadata.seed,
            generator_index: self.body_limited.generator_metadata.index,
            goal_differs_from_start,
            body_limitation_changes_ceiling: body_ceiling != unrestricted_ceiling,
            twin_trajectories_equal: trajectories_equal,
            twin_values_equal: values_equal,
            twin_optimal_sequences_equal: body_ceiling == twin_ceiling
                && body_optimal == twin_optimal,
            twin_publicly_distinct,
            calibration_identifies_body,
            valid: goal_differs_from_start
                && topology_total
                && body_ceiling != unrestricted_ceiling
                && (self.body_limited.term.environment.is_ring() || !topology.is_simple_cycle)
                && trajectories_equal
                && values_equal
                && body_ceiling == twin_ceiling
                && body_optimal == twin_optimal
                && twin_publicly_distinct
                && calibration_identifies_body,
        };
        Ok(receipt)
    }
}

/// Use the shared query algebra to check the generated body's identification
/// claim.  The same two terms are compared once without their prelude and once
/// with it; no card-specific query or evaluator is introduced here.
fn body_identification_after_calibration(
    body_term: &WorldTerm,
    unrestricted_term: &WorldTerm,
) -> Result<bool, CompileError> {
    let mut blind_body_term = body_term.clone();
    blind_body_term.calibration = None;
    let mut blind_unrestricted_term = unrestricted_term.clone();
    blind_unrestricted_term.calibration = None;
    let blind_body = blind_body_term.compile()?;
    let blind_unrestricted = blind_unrestricted_term.compile()?;
    let blind_set = AmbiguitySet::uniform(vec![
        blind_body.contract.clone(),
        blind_unrestricted.contract.clone(),
    ]);
    let initial_diameter = identification_diameter(&blind_body.fragment, &blind_set, &[]);

    let body = body_term.compile()?;
    let unrestricted = unrestricted_term.compile()?;
    let calibrated_set =
        AmbiguitySet::uniform(vec![body.contract.clone(), unrestricted.contract.clone()]);
    let calibrated_diameter = identification_diameter(&body.fragment, &calibrated_set, &[]);
    Ok(initial_diameter == 2 && calibrated_diameter == 1)
}

/// Enumerate the configurations a term can expose to a scored command.  This
/// derives the twin's deleted edges from executable body semantics rather than
/// assuming that the sampled horizon happens to be the reachable region.
fn scored_reachable_cells(term: &WorldTerm) -> Result<Vec<usize>, CompileError> {
    let compiled = term.compile()?;
    let actions = compiled.actions();
    let sequences = pretraining_g0_contract::sequences_of_length(&actions, compiled.horizon());
    let mut cells = BTreeSet::new();
    for sequence in sequences {
        cells.extend(pretraining_g0_contract::trajectory(
            &compiled.fragment,
            &compiled.contract,
            &sequence,
        ));
    }
    Ok(cells.into_iter().collect())
}

impl Default for GenerationSpec {
    fn default() -> Self {
        Self {
            seed: 0,
            count: 4,
            template: GenerationTemplate::Ring,
            min_cells: 5,
            max_cells: 6,
            min_horizon: 2,
            max_horizon: 3,
        }
    }
}

impl GenerationSpec {
    pub fn validate(&self) -> Result<(), CompileError> {
        if self.count == 0 || self.count > 64 {
            return Err(CompileError::Invalid(
                "generator count must be in 1..=64".into(),
            ));
        }
        if !(2..=8).contains(&self.min_cells)
            || self.min_cells > self.max_cells
            || self.max_cells > 8
        {
            return Err(CompileError::Invalid(
                "generated cells must be in 2..=8".into(),
            ));
        }
        if self.min_horizon == 0 || self.min_horizon > self.max_horizon || self.max_horizon > 6 {
            return Err(CompileError::Invalid(
                "generated horizons must be in 1..=6".into(),
            ));
        }
        Ok(())
    }

    /// The only general generator entry point emits receipt-filtered paired
    /// families.  It deliberately does not emit independent random worlds.
    pub fn generate(&self) -> Result<Vec<GeneratedEmbodimentFamily>, CompileError> {
        match self.template {
            GenerationTemplate::Ring => self.generate_embodiment_families(),
            GenerationTemplate::BranchingGraph => self.generate_branching_graph_families(),
        }
    }

    /// Deterministically construct receipt-filtered embodiment contrasts.  A
    /// rejected candidate is not emitted, which prevents a sampled world from
    /// being mistaken for a valid family merely because it compiled.
    pub fn generate_embodiment_families(
        &self,
    ) -> Result<Vec<GeneratedEmbodimentFamily>, CompileError> {
        self.validate()?;
        if self.max_cells < 4 || self.max_horizon < 2 {
            return Err(CompileError::Invalid(
                "embodiment families need at least four cells and two scored steps".into(),
            ));
        }
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut families = Vec::with_capacity(self.count);
        for index in 0..self.count {
            let lower_cells = self.min_cells.max(4);
            if lower_cells > self.max_cells {
                return Err(CompileError::Invalid(
                    "embodiment family cell range is empty".into(),
                ));
            }
            let cells = rng.gen_range(lower_cells..=self.max_cells);
            let max_horizon = self.max_horizon.min(cells - 2);
            let min_horizon = self.min_horizon.max(2);
            if min_horizon > max_horizon {
                return Err(CompileError::Invalid(
                    "no horizon leaves the body-limited goal outside forward reach".into(),
                ));
            }
            let horizon = rng.gen_range(min_horizon..=max_horizon);
            let goal = cells - 1;
            let full_support = IndexSet::from_indices([0, 1, 2, 3]);
            let limited_support = IndexSet::from_indices([0, 2, 3]);
            let body_term = embodiment_term(
                format!("embodiment-{index}-body"),
                cells,
                horizon,
                goal,
                limited_support,
                Vec::new(),
            );
            let unrestricted_term = embodiment_term(
                format!("embodiment-{index}-unrestricted"),
                cells,
                horizon,
                goal,
                full_support,
                Vec::new(),
            );
            // Delete the withheld actuator only where the scored phase can
            // command from.  The calibration prelude intentionally exits that
            // region, making the provenance-changing twin publicly visible.
            let scored_cells = scored_reachable_cells(&body_term)?;
            let withheld = full_support.difference(limited_support);
            let twin_edges = scored_cells
                .into_iter()
                .flat_map(|cell| withheld.iter().map(move |actuator| (cell, actuator as u16)))
                .collect();
            let twin_term = embodiment_term(
                format!("embodiment-{index}-environment"),
                cells,
                horizon,
                goal,
                full_support,
                twin_edges,
            );
            let metadata = GeneratorMetadata {
                seed: self.seed,
                index,
            };
            let mut family = GeneratedEmbodimentFamily {
                body_limited: GeneratedWorld {
                    term: body_term,
                    generator_metadata: metadata.clone(),
                },
                unrestricted_control: GeneratedWorld {
                    term: unrestricted_term,
                    generator_metadata: metadata.clone(),
                },
                environment_twin: GeneratedWorld {
                    term: twin_term,
                    generator_metadata: metadata,
                },
                receipt: EmbodimentValidityReceipt {
                    sequences_checked: 0,
                    topology_total: false,
                    topology_state_count: 0,
                    topology_transition_rows: 0,
                    topology_degree_sequence: Vec::new(),
                    topology_is_simple_cycle: false,
                    twin_scope: TwinScope::ConservativeTrajectoryCells,
                    family_hash: String::new(),
                    generator_seed: 0,
                    generator_index: 0,
                    goal_differs_from_start: false,
                    body_limitation_changes_ceiling: false,
                    twin_trajectories_equal: false,
                    twin_values_equal: false,
                    twin_optimal_sequences_equal: false,
                    twin_publicly_distinct: false,
                    calibration_identifies_body: false,
                    valid: false,
                },
            };
            family.receipt = family.verify()?;
            if !family.receipt.valid {
                return Err(CompileError::Invalid(
                    "generated embodiment candidate failed its exact validity filter".into(),
                ));
            }
            families.push(family);
        }
        Ok(families)
    }

    /// Deterministically construct non-cyclic branching graph contrasts. The
    /// witness withholds `retreat`; its environment twin represents the same
    /// scored self-loops as missing graph entries, while retaining a calibration
    ///-only retreat edge outside the scored-reachable state set.
    pub fn generate_branching_graph_families(
        &self,
    ) -> Result<Vec<GeneratedEmbodimentFamily>, CompileError> {
        self.validate()?;
        if self.max_cells < 5 || self.max_horizon < 2 {
            return Err(CompileError::Invalid(
                "branching graph families need at least five states and two scored steps".into(),
            ));
        }
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut families = Vec::with_capacity(self.count);
        for index in 0..self.count {
            let lower_states = self.min_cells.max(5);
            if lower_states > self.max_cells {
                return Err(CompileError::Invalid(
                    "branching graph family state range is empty".into(),
                ));
            }
            let states = rng.gen_range(lower_states..=self.max_cells);
            let max_horizon = self.max_horizon.min(states - 3);
            let min_horizon = self.min_horizon.max(2);
            if min_horizon > max_horizon {
                return Err(CompileError::Invalid(
                    "no graph horizon leaves a calibration-only branch state".into(),
                ));
            }
            let horizon = rng.gen_range(min_horizon..=max_horizon);
            let goal = states - 1;
            let full_support = IndexSet::from_indices([0, 1, 2, 3]);
            let limited_support = IndexSet::from_indices([0, 2, 3]);
            let body_term = branching_graph_term(
                format!("graph-{index}-body"),
                states,
                horizon,
                goal,
                limited_support,
                BTreeSet::new(),
            );
            if body_term.environment.topology_diagnostics().is_simple_cycle {
                return Err(CompileError::Invalid(
                    "branching graph template produced a simple cycle".into(),
                ));
            }
            let unrestricted_term = branching_graph_term(
                format!("graph-{index}-unrestricted"),
                states,
                horizon,
                goal,
                full_support,
                BTreeSet::new(),
            );
            let scored_states: BTreeSet<usize> =
                scored_reachable_cells(&body_term)?.into_iter().collect();
            let twin_term = branching_graph_term(
                format!("graph-{index}-environment"),
                states,
                horizon,
                goal,
                full_support,
                scored_states,
            );
            let metadata = GeneratorMetadata {
                seed: self.seed,
                index,
            };
            let mut family = GeneratedEmbodimentFamily {
                body_limited: GeneratedWorld {
                    term: body_term,
                    generator_metadata: metadata.clone(),
                },
                unrestricted_control: GeneratedWorld {
                    term: unrestricted_term,
                    generator_metadata: metadata.clone(),
                },
                environment_twin: GeneratedWorld {
                    term: twin_term,
                    generator_metadata: metadata,
                },
                receipt: EmbodimentValidityReceipt {
                    sequences_checked: 0,
                    topology_total: false,
                    topology_state_count: 0,
                    topology_transition_rows: 0,
                    topology_degree_sequence: Vec::new(),
                    topology_is_simple_cycle: false,
                    twin_scope: TwinScope::ConservativeTrajectoryCells,
                    family_hash: String::new(),
                    generator_seed: 0,
                    generator_index: 0,
                    goal_differs_from_start: false,
                    body_limitation_changes_ceiling: false,
                    twin_trajectories_equal: false,
                    twin_values_equal: false,
                    twin_optimal_sequences_equal: false,
                    twin_publicly_distinct: false,
                    calibration_identifies_body: false,
                    valid: false,
                },
            };
            family.receipt = family.verify()?;
            if !family.receipt.valid {
                return Err(CompileError::Invalid(
                    "generated branching graph candidate failed its exact validity filter".into(),
                ));
            }
            families.push(family);
        }
        Ok(families)
    }
}

fn embodiment_term(
    name: impl Into<String>,
    cells: usize,
    horizon: usize,
    goal: usize,
    supported: IndexSet,
    blocked_edges: Vec<(usize, u16)>,
) -> WorldTerm {
    WorldTerm {
        name: name.into(),
        horizon,
        ports: vec![
            Port::new("learner_action", PortDirection::Output, PortValue::Command).public(),
            Port::new("body_command", PortDirection::Input, PortValue::Command).public(),
            Port::new("cell", PortDirection::Output, PortValue::Cell).public(),
            Port::new("reveal", PortDirection::Output, PortValue::Signal).public(),
        ],
        wiring: vec![Wiring {
            from: "learner_action".into(),
            to: "body_command".into(),
        }],
        body: BodyTerm {
            morphology: Morphology { segments: 2 },
            actuation: Actuation {
                command_port: "body_command".into(),
                actuators: vec![
                    Actuator {
                        id: 0,
                        name: "advance".into(),
                        displacement: 1,
                        role: ActionRole::Movement,
                    },
                    Actuator {
                        id: 1,
                        name: "retreat".into(),
                        displacement: -1,
                        role: ActionRole::Movement,
                    },
                    Actuator {
                        id: 2,
                        name: "hold".into(),
                        displacement: 0,
                        role: ActionRole::Hold,
                    },
                    Actuator {
                        id: 3,
                        name: "fallback".into(),
                        displacement: 0,
                        role: ActionRole::Fallback,
                    },
                ],
                supported,
            },
            sensorium: Sensorium {
                cell_port: "cell".into(),
                publishes_cell: true,
            },
        },
        environment: EnvironmentTerm::ring(0, cells, blocked_edges),
        norm: NormTerm {
            expression: Norm::Settle { cell: goal },
            visibility: Visibility::Public,
        },
        scoring: ScoringTerm {
            goal_reward: 100,
            action_cost: 1,
            fallback_reward: 50,
            violation_penalty: -100,
        },
        calibration: Some(CalibrationTerm {
            // Advance beyond every scored-reachable cell, then pulse the
            // withheld retreat actuator.  The body twin remains still while
            // the environment twin retreats, so the preserving transform is
            // publicly non-vacuous through the prelude.
            pulses: std::iter::repeat(0)
                .take(horizon + 1)
                .chain(std::iter::once(1))
                .collect(),
            publishes_cells: true,
        }),
        restorations: Vec::new(),
        disturbance: None,
        scaffold: None,
        couplings: Vec::new(),
        interrupts: Vec::new(),
        restrictions: Vec::new(),
        reveals: Vec::new(),
    }
}

/// A non-ring environment with an advance chain and a second outgoing
/// intervention (`retreat`) to the goal. `missing_retreat_from` provides the
/// environment-side counterpart of body support: absent graph entries have the
/// explicitly declared self-loop meaning.
fn branching_graph_term(
    name: impl Into<String>,
    states: usize,
    horizon: usize,
    goal: usize,
    supported: IndexSet,
    missing_retreat_from: BTreeSet<usize>,
) -> WorldTerm {
    let mut transitions = Vec::new();
    // The chain reaches state `horizon + 2` only during the unscored
    // calibration prelude. It is therefore not in the conservative twin's
    // scored trajectory-cell deletion set, making provenance visible without
    // changing scored behavior.
    for state in 0..=horizon + 1 {
        transitions.push(GraphTransition {
            source: state,
            actuator: 0,
            destination: state + 1,
        });
    }
    for state in 0..states {
        if !missing_retreat_from.contains(&state) {
            transitions.push(GraphTransition {
                source: state,
                actuator: 1,
                // The first retreat is the short route to the goal; later
                // rows remain observable as genuine graph interventions.
                destination: if state == 1 {
                    goal
                } else {
                    state.saturating_sub(1)
                },
            });
        }
    }
    WorldTerm {
        name: name.into(),
        horizon,
        ports: vec![
            Port::new("learner_action", PortDirection::Output, PortValue::Command).public(),
            Port::new("body_command", PortDirection::Input, PortValue::Command).public(),
            Port::new("cell", PortDirection::Output, PortValue::Cell).public(),
            Port::new("reveal", PortDirection::Output, PortValue::Signal).public(),
        ],
        wiring: vec![Wiring {
            from: "learner_action".into(),
            to: "body_command".into(),
        }],
        body: BodyTerm {
            morphology: Morphology { segments: 2 },
            actuation: Actuation {
                command_port: "body_command".into(),
                actuators: vec![
                    Actuator {
                        id: 0,
                        name: "advance".into(),
                        // Graph action meaning is supplied only by explicit
                        // transition rows; ring displacement is inapplicable.
                        displacement: 0,
                        role: ActionRole::Movement,
                    },
                    Actuator {
                        id: 1,
                        name: "retreat".into(),
                        displacement: 0,
                        role: ActionRole::Movement,
                    },
                    Actuator {
                        id: 2,
                        name: "hold".into(),
                        displacement: 0,
                        role: ActionRole::Hold,
                    },
                    Actuator {
                        id: 3,
                        name: "fallback".into(),
                        displacement: 0,
                        role: ActionRole::Fallback,
                    },
                ],
                supported,
            },
            sensorium: Sensorium {
                cell_port: "cell".into(),
                publishes_cell: true,
            },
        },
        environment: EnvironmentTerm::graph(0, states, transitions, MissingEdgeBehavior::SelfLoop),
        norm: NormTerm {
            expression: Norm::Settle { cell: goal },
            visibility: Visibility::Public,
        },
        scoring: ScoringTerm {
            goal_reward: 100,
            action_cost: 1,
            fallback_reward: 50,
            violation_penalty: -100,
        },
        calibration: Some(CalibrationTerm {
            pulses: std::iter::repeat(0)
                .take(horizon + 2)
                .chain(std::iter::once(1))
                .collect(),
            publishes_cells: true,
        }),
        restorations: Vec::new(),
        disturbance: None,
        scaffold: None,
        couplings: Vec::new(),
        interrupts: Vec::new(),
        restrictions: Vec::new(),
        reveals: Vec::new(),
    }
}

const G0_HOLD_ID: u16 = 30;
const G0_FALLBACK_ID: u16 = 31;

/// Pick movement identifiers from the body-action namespace while reserving
/// the final two identifiers for the stable hold and fallback actions.  The
/// random permutation is intentional: a learner must not be able to bind
/// "forward" to one fixed command number across generated worlds.
fn g0_movement_ids(count: usize, rng: &mut ChaCha8Rng) -> Vec<u16> {
    let mut ids: Vec<u16> = (0..G0_HOLD_ID).collect();
    ids.shuffle(rng);
    ids.truncate(count);
    // Keep the legacy low-number action channels represented in the corpus so
    // downstream adapters can exercise them, while the remaining channels and
    // their meanings still vary per world.  This is a corpus-level coverage
    // guarantee, not a semantic preference: ring displacements and graph rows
    // are assigned after this permutation.
    if !ids.contains(&0) {
        ids[0] = 0;
    }
    if !ids.contains(&1) {
        ids[1] = 1;
    }
    ids
}

fn g0_world_term(
    name: impl Into<String>,
    horizon: usize,
    goal: usize,
    segments: usize,
    movement: Vec<Actuator>,
    environment: EnvironmentTerm,
) -> WorldTerm {
    let mut actuators = movement;
    actuators.push(Actuator {
        id: G0_HOLD_ID,
        name: "hold".into(),
        displacement: 0,
        role: ActionRole::Hold,
    });
    actuators.push(Actuator {
        id: G0_FALLBACK_ID,
        name: "fallback".into(),
        displacement: 0,
        role: ActionRole::Fallback,
    });
    WorldTerm {
        name: name.into(),
        horizon,
        ports: vec![
            Port::new("learner_action", PortDirection::Output, PortValue::Command).public(),
            Port::new("body_command", PortDirection::Input, PortValue::Command).public(),
            Port::new("cell", PortDirection::Output, PortValue::Cell).public(),
            Port::new("event", PortDirection::Output, PortValue::Signal).public(),
        ],
        wiring: vec![Wiring {
            from: "learner_action".into(),
            to: "body_command".into(),
        }],
        body: BodyTerm {
            morphology: Morphology { segments },
            actuation: Actuation {
                command_port: "body_command".into(),
                supported: IndexSet::from_indices(
                    actuators.iter().map(|actuator| usize::from(actuator.id)),
                ),
                actuators,
            },
            sensorium: Sensorium {
                cell_port: "cell".into(),
                publishes_cell: true,
            },
        },
        environment,
        norm: NormTerm {
            expression: Norm::Settle { cell: goal },
            visibility: Visibility::Public,
        },
        scoring: ScoringTerm {
            goal_reward: 100,
            action_cost: 1,
            fallback_reward: 50,
            violation_penalty: -100,
        },
        calibration: None,
        restorations: Vec::new(),
        disturbance: None,
        scaffold: None,
        couplings: Vec::new(),
        interrupts: Vec::new(),
        restrictions: Vec::new(),
        reveals: Vec::new(),
    }
}

fn g0_ring_term(
    name: impl Into<String>,
    states: usize,
    horizon: usize,
    target_length: usize,
    segments: usize,
    movement_count: usize,
    rng: &mut ChaCha8Rng,
) -> WorldTerm {
    let movement_ids = g0_movement_ids(movement_count, rng);
    // A ring route is deliberately length-controlled. Every movement advances
    // by one, while all non-route movement edges at route cells are blocked;
    // therefore no actuator can jump directly to the goal.
    let direction = if rng.gen_bool(0.5) { 1i32 } else { -1i32 };
    let start = rng.gen_range(0..states);
    let goal = (start as i32 + direction * target_length as i32).rem_euclid(states as i32) as usize;
    let movement: Vec<Actuator> = movement_ids
        .iter()
        .enumerate()
        .map(|(index, id)| Actuator {
            id: *id,
            name: format!("move-{index}"),
            displacement: if index == 0 {
                direction
            } else {
                // A ring movement is always local.  IDs still vary, and
                // duplicate directions are intentional actuator aliases.
                if rng.gen_bool(0.5) {
                    1
                } else {
                    -1
                }
            },
            role: ActionRole::Movement,
        })
        .collect();
    let route_id = movement[0].id;
    let mut blocked = BTreeSet::new();
    let mut protected_route = BTreeSet::new();
    for step in 0..target_length {
        let cell = (start as i32 + direction * step as i32).rem_euclid(states as i32) as usize;
        protected_route.insert((cell, route_id));
        for actuator in &movement {
            if actuator.id != route_id {
                blocked.insert((cell, actuator.id));
            }
        }
    }
    // Retain additional environmental variation away from the protected route.
    for cell in 0..states {
        for actuator in &movement {
            if !blocked.contains(&(cell, actuator.id))
                && !protected_route.contains(&(cell, actuator.id))
                && rng.gen_bool(0.16)
            {
                blocked.insert((cell, actuator.id));
            }
        }
    }
    g0_world_term(
        name,
        horizon,
        goal,
        segments,
        movement,
        EnvironmentTerm::ring(start, states, blocked.into_iter().collect()),
    )
}

fn push_edge(edges: &mut Vec<(usize, usize)>, source: usize, destination: usize) {
    if source != destination && !edges.contains(&(source, destination)) {
        edges.push((source, destination));
    }
}

fn topology_edges(
    states: usize,
    target_length: usize,
    topology: G0Topology,
    rng: &mut ChaCha8Rng,
) -> (usize, usize, Vec<(usize, usize)>) {
    let mut edges = Vec::new();
    match topology {
        G0Topology::Chain => {
            for state in 0..states.saturating_sub(1) {
                push_edge(&mut edges, state, state + 1);
            }
            (0, target_length, edges)
        }
        G0Topology::Star => {
            for leaf in 1..states {
                push_edge(&mut edges, 0, leaf);
                push_edge(&mut edges, leaf, 0);
            }
            if target_length == 1 {
                (0, 1, edges)
            } else {
                (1, 2, edges)
            }
        }
        G0Topology::Tree => {
            // A branched tree with a long backbone. The final state is a leaf
            // attached to node one, leaving the backbone distance intact.
            for state in 0..states.saturating_sub(2) {
                push_edge(&mut edges, state, state + 1);
                push_edge(&mut edges, state + 1, state);
            }
            let branch = states - 1;
            push_edge(&mut edges, 1, branch);
            push_edge(&mut edges, branch, 1);
            (0, target_length, edges)
        }
        G0Topology::Dag => {
            for state in 0..states.saturating_sub(1) {
                push_edge(&mut edges, state, state + 1);
            }
            // Extra forward edges live after the requested goal, so they do
            // not introduce a shorter path to it.
            for source in target_length..states {
                for destination in source + 2..states {
                    push_edge(&mut edges, source, destination);
                }
            }
            (0, target_length, edges)
        }
        G0Topology::Grid | G0Topology::Lattice => {
            let (rows, columns) = rectangular_shapes(2, states)
                .into_iter()
                .find(|(rows, columns)| rows * columns == states)
                .unwrap_or((1, states));
            let index = |row: usize, column: usize| row * columns + column;
            let toroidal = topology == G0Topology::Lattice;
            for row in 0..rows {
                for column in 0..columns {
                    let here = index(row, column);
                    if toroidal || column + 1 < columns {
                        push_edge(&mut edges, here, index(row, (column + 1) % columns));
                        push_edge(&mut edges, index(row, (column + 1) % columns), here);
                    }
                    if toroidal || row + 1 < rows {
                        push_edge(&mut edges, here, index((row + 1) % rows, column));
                        push_edge(&mut edges, index((row + 1) % rows, column), here);
                    }
                }
            }
            let mut goal = 0;
            'search: for row in 0..rows {
                for column in 0..columns {
                    let distance = if toroidal {
                        row.min(rows - row) + column.min(columns - column)
                    } else {
                        row + column
                    };
                    if distance == target_length {
                        goal = index(row, column);
                        break 'search;
                    }
                }
            }
            (0, goal, edges)
        }
        G0Topology::Graph => {
            for state in 0..target_length {
                push_edge(&mut edges, state, state + 1);
            }
            let route: BTreeSet<usize> = (0..=target_length).collect();
            // Random extra rows are restricted so no edge can enter the route
            // or goal from outside it. This keeps the generic graph's exact
            // shortest path while allowing cycles and branching elsewhere.
            for _ in 0..states.saturating_mul(8) {
                let source = rng.gen_range(0..states);
                let destination = rng.gen_range(0..states);
                if source != destination && !route.contains(&destination) {
                    push_edge(&mut edges, source, destination);
                }
            }
            for source in target_length + 1..states {
                for destination in source + 1..states {
                    if (source + destination) % 3 == 0 {
                        push_edge(&mut edges, source, destination);
                    }
                }
            }
            (0, target_length, edges)
        }
        G0Topology::Ring | G0Topology::Mixed => unreachable!(),
    }
}

fn g0_graph_term(
    name: impl Into<String>,
    states: usize,
    horizon: usize,
    target_length: usize,
    segments: usize,
    movement_count: usize,
    topology: G0Topology,
    rng: &mut ChaCha8Rng,
) -> WorldTerm {
    let movement_ids = g0_movement_ids(movement_count, rng);
    let movement: Vec<Actuator> = movement_ids
        .iter()
        .enumerate()
        .map(|(index, id)| Actuator {
            id: *id,
            name: format!("move-{index}"),
            displacement: 0,
            role: ActionRole::Movement,
        })
        .collect();

    let (start, goal, edges) = topology_edges(states, target_length, topology, rng);
    let mut by_source = vec![Vec::new(); states];
    for (source, destination) in edges {
        if !by_source[source].contains(&destination) {
            by_source[source].push(destination);
        }
    }
    let mut transitions = Vec::new();
    for (source, destinations) in by_source.into_iter().enumerate() {
        for (offset, destination) in destinations
            .into_iter()
            .take(movement_ids.len())
            .enumerate()
        {
            transitions.push(GraphTransition {
                source,
                actuator: movement_ids[offset],
                destination,
            });
        }
    }
    g0_world_term(
        name,
        horizon,
        goal,
        segments,
        movement,
        EnvironmentTerm::graph(start, states, transitions, MissingEdgeBehavior::SelfLoop),
    )
}

/// A small valid term used by examples, tests, and deterministic generation.
pub fn base_term(
    name: impl Into<String>,
    cells: usize,
    horizon: usize,
    goal: usize,
    supported: IndexSet,
) -> WorldTerm {
    WorldTerm {
        name: name.into(),
        horizon,
        ports: vec![
            Port::new("learner_action", PortDirection::Output, PortValue::Command).public(),
            Port::new("body_command", PortDirection::Input, PortValue::Command).public(),
            Port::new("cell", PortDirection::Output, PortValue::Cell).public(),
            Port::new("reveal", PortDirection::Output, PortValue::Signal).public(),
        ],
        wiring: vec![Wiring {
            from: "learner_action".into(),
            to: "body_command".into(),
        }],
        body: BodyTerm {
            morphology: Morphology { segments: 2 },
            actuation: Actuation {
                command_port: "body_command".into(),
                actuators: vec![
                    Actuator {
                        id: 0,
                        name: "advance".into(),
                        displacement: 1,
                        role: ActionRole::Movement,
                    },
                    Actuator {
                        id: 1,
                        name: "retreat".into(),
                        displacement: -1,
                        role: ActionRole::Movement,
                    },
                ],
                supported,
            },
            sensorium: Sensorium {
                cell_port: "cell".into(),
                publishes_cell: true,
            },
        },
        environment: EnvironmentTerm::ring(0, cells, Vec::new()),
        norm: NormTerm {
            expression: Norm::Settle { cell: goal },
            visibility: Visibility::Public,
        },
        scoring: ScoringTerm::default(),
        calibration: None,
        restorations: Vec::new(),
        disturbance: None,
        scaffold: None,
        couplings: Vec::new(),
        interrupts: Vec::new(),
        restrictions: Vec::new(),
        reveals: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use pretraining_g0_contract::{
        noninterference_check, trajectory, value_bounds, PubliclyObservable,
    };

    use super::*;

    #[test]
    fn a_body_support_limit_is_not_an_environment_edge_limit() {
        let body_limited = base_term("body", 5, 2, 2, IndexSet::from_indices([0]));
        let environment_limited = WorldTerm {
            body: BodyTerm {
                actuation: Actuation {
                    supported: IndexSet::from_indices([0, 1]),
                    ..body_limited.body.actuation.clone()
                },
                ..body_limited.body.clone()
            },
            environment: EnvironmentTerm::ring(0, 5, vec![(0, 1)]),
            ..body_limited.clone()
        };
        let body = body_limited.compile().expect("valid body term");
        let environment = environment_limited
            .compile()
            .expect("valid environment term");
        assert_eq!(body.fragment.step(&body.contract, 1, 0, 1), 1);
        assert_eq!(environment.fragment.step(&environment.contract, 1, 0, 1), 0);
    }

    #[test]
    fn explicit_kernel_constructs_are_derived_not_hand_written() {
        let mut term = base_term("all-constructs", 5, 3, 2, IndexSet::from_indices([0, 1]));
        term.norm.expression = Norm::both(Norm::Visit { cell: 1 }, Norm::Settle { cell: 2 });
        term.restrictions.push(Restriction::Resource {
            budget: 2,
            scope: pretraining_g0_contract::ResourceScope::Shared,
        });
        term.couplings.push(CouplingTerm {
            coupling: Coupling::new(0, CouplingRule::Sum),
            writers: vec![CoupledWriter {
                guard: Guard::Never,
                value: 1,
            }],
            inactive_value: 0,
        });
        term.scaffold = Some(ProcessTerm {
            name: "scaffold".into(),
            signals: Vec::new(),
        });
        term.interrupts.push(InterruptTerm {
            specification: Interrupt::new(
                Guard::AfterStep(0),
                Displaced::Frozen,
                pretraining_g0_contract::Resume::FromState,
            ),
            interrupted_process: "scaffold".into(),
        });
        term.reveals.push(RevealTerm {
            source_visibility: Visibility::Privileged,
            output_port: "reveal".into(),
            guard: Guard::AfterStep(0),
            value: 7,
        });
        let compiled = term.compile().expect("valid composed term");
        assert_eq!(compiled.horizon(), 2);
        let (ceiling, _) = value_bounds(&compiled.fragment, &compiled.contract);
        assert_eq!(ceiling, 98, "the compiled Fragment uses its term horizon");
        assert_eq!(
            compiled.kernel_use,
            KernelUse {
                directed_wiring: true,
                shared_coupling: true,
                interrupt: true,
                restrict: true,
                reveal: true,
                norm_algebra: true
            }
        );
    }

    #[test]
    fn lowered_coupling_has_a_total_inactive_case_and_conflicts_fail_at_compile_time() {
        let mut term = base_term("coupling", 5, 2, 3, IndexSet::from_indices([0, 1]));
        term.couplings.push(CouplingTerm {
            coupling: Coupling::new(0, CouplingRule::Override),
            writers: vec![CoupledWriter {
                guard: Guard::Never,
                value: 99,
            }],
            inactive_value: 1,
        });
        let compiled = term.compile().unwrap();
        assert_eq!(compiled.fragment.step(&compiled.contract, 0, 0, 0), 2);

        term.couplings[0] = CouplingTerm {
            coupling: Coupling::new(0, CouplingRule::Conflict),
            writers: vec![
                CoupledWriter {
                    guard: Guard::AtStart,
                    value: 1,
                },
                CoupledWriter {
                    guard: Guard::AtStart,
                    value: 2,
                },
            ],
            inactive_value: 0,
        };
        assert!(matches!(term.compile(), Err(CompileError::Invalid(_))));
    }

    #[test]
    fn restoration_and_calibration_do_not_share_a_clock() {
        let mut term = base_term("restoration", 5, 2, 4, IndexSet::from_indices([0]));
        term.calibration = Some(CalibrationTerm {
            pulses: vec![0, 1],
            publishes_cells: true,
        });
        term.restorations.push(SupportRestoration {
            actuator: 1,
            after_step: 0,
            announcement_port: "reveal".into(),
            announcement_value: 401,
        });
        let compiled = term.compile().unwrap();
        assert_eq!(compiled.public_view(&[]).trace[..3], [0, 1, 1]);
        assert!(compiled.kernel_use.reveal);
        assert_eq!(compiled.fragment.step(&compiled.contract, 0, 0, 1), 0);
        assert_eq!(compiled.fragment.step(&compiled.contract, 0, 1, 1), 4);
        assert!(compiled.public_view(&[]).trace.contains(&401));
    }

    #[test]
    fn fallback_is_absorbing_in_outcomes_and_scoring() {
        let term = embodiment_term(
            "fallback",
            5,
            2,
            4,
            IndexSet::from_indices([0, 2, 3]),
            Vec::new(),
        );
        let compiled = term.compile().unwrap();
        let outcome = compiled.outcome(&[3, 0]);
        assert_eq!(outcome.value, 50);
        assert!(outcome.fell_back);
        assert_eq!(outcome.fallback_step, Some(0));
        assert_eq!(outcome.final_cell, 0);
        assert_eq!(compiled.public_view(&[3, 0]).trace.last(), Some(&-1));
    }

    #[test]
    fn direct_privileged_to_public_wiring_is_rejected() {
        let mut term = base_term("leak", 4, 2, 1, IndexSet::from_indices([0, 1]));
        term.ports.push(Port::new(
            "secret",
            PortDirection::Output,
            PortValue::Signal,
        ));
        term.ports
            .push(Port::new("public_sink", PortDirection::Input, PortValue::Signal).public());
        term.wiring.push(Wiring {
            from: "secret".into(),
            to: "public_sink".into(),
        });
        assert!(matches!(
            term.compile(),
            Err(CompileError::VisibilityLeak { .. })
        ));
    }

    #[test]
    fn ill_typed_wiring_is_rejected_before_execution() {
        let mut term = base_term("type-error", 4, 2, 1, IndexSet::from_indices([0, 1]));
        term.ports
            .push(Port::new("cell_sink", PortDirection::Input, PortValue::Cell).public());
        term.wiring.push(Wiring {
            from: "learner_action".into(),
            to: "cell_sink".into(),
        });
        assert!(matches!(
            term.compile(),
            Err(CompileError::IllTypedWire { .. })
        ));
    }

    #[test]
    fn a_monitor_cannot_actuate_and_support_cannot_be_queried_publicly() {
        let mut monitor_feedback =
            base_term("monitor-feedback", 4, 2, 1, IndexSet::from_indices([0, 1]));
        monitor_feedback.ports.push(
            Port::new("audit", PortDirection::Monitor, PortValue::Command)
                .with_visibility(Visibility::Privileged),
        );
        monitor_feedback.wiring.push(Wiring {
            from: "audit".into(),
            to: "body_command".into(),
        });
        assert!(matches!(
            monitor_feedback.compile(),
            Err(CompileError::MonitorAsSource(_))
        ));

        let mut public_support =
            base_term("support-query", 4, 2, 1, IndexSet::from_indices([0, 1]));
        public_support
            .ports
            .push(Port::new("support", PortDirection::Output, PortValue::Support).public());
        assert!(matches!(
            public_support.compile(),
            Err(CompileError::Invalid(_))
        ));
    }

    #[test]
    fn private_signal_changes_do_not_change_the_public_trace() {
        let mut left = base_term("left", 4, 2, 1, IndexSet::from_indices([0, 1]));
        left.ports.push(Port::new(
            "latent",
            PortDirection::Output,
            PortValue::Signal,
        ));
        left.disturbance = Some(ProcessTerm {
            name: "hidden".into(),
            signals: vec![SignalTerm {
                output_port: "latent".into(),
                guard: Guard::AtStart,
                value: 1,
            }],
        });
        let mut right = left.clone();
        right.name = "right".into();
        right.disturbance.as_mut().unwrap().signals[0].value = 99;
        let left = left.compile().unwrap();
        let right = right.compile().unwrap();
        assert_eq!(
            left.fragment.public_trace(&left.contract, &[0]),
            right.fragment.public_trace(&right.contract, &[0])
        );
        let report = noninterference_check(
            &left.fragment,
            "latent value",
            &left.contract,
            &right.contract,
            2,
            |_| true,
            |action| action.to_string(),
        );
        assert!(report.holds, "{report:?}");
    }

    #[test]
    fn generated_families_are_bounded_deterministic_and_receipt_valid() {
        let specification = GenerationSpec {
            seed: 42,
            count: 6,
            ..GenerationSpec::default()
        };
        let first = specification.generate().unwrap();
        let second = specification.generate().unwrap();
        assert_eq!(first, second);
        for family in first {
            assert!(family.receipt.valid, "{:#?}", family.receipt);
            assert_eq!(family.verify().unwrap(), family.receipt);
            let compiled = family.body_limited.compile().unwrap();
            let actions = compiled.actions();
            let path = trajectory(
                &compiled.fragment,
                &compiled.contract,
                &vec![actions[0]; compiled.horizon()],
            );
            assert_eq!(path.len(), compiled.horizon() + 1);
        }
    }

    #[test]
    fn generated_twins_are_publicly_distinct_and_calibration_is_query_audited() {
        let families = GenerationSpec {
            seed: 7,
            count: 8,
            ..GenerationSpec::default()
        }
        .generate()
        .unwrap();
        assert_eq!(families.len(), 8);
        for family in families {
            assert!(family.receipt.valid);
            assert!(family.receipt.twin_publicly_distinct);
            assert!(family.receipt.calibration_identifies_body);
            let body = family.body_limited.compile().unwrap();
            let twin = family.environment_twin.compile().unwrap();
            assert_ne!(
                body.public_view(&[]).trace,
                twin.public_view(&[]).trace,
                "the generated preserving twin must be visible in its prelude"
            );
        }
    }

    #[test]
    fn family_hash_is_semantic_and_replay_metadata_is_separate() {
        let left = base_term("label-a", 5, 2, 2, IndexSet::from_indices([0, 1]));
        let mut right = left.clone();
        right.name = "label-b".into();
        assert_eq!(
            left.family_hash(),
            right.family_hash(),
            "labels are generator metadata"
        );
        match &mut right.environment.state_space {
            StateSpaceTerm::Ring { blocked_edges, .. } => blocked_edges.push((0, 0)),
            StateSpaceTerm::Graph { .. } => panic!("base term uses the ring constructor"),
        }
        assert_ne!(
            left.family_hash(),
            right.family_hash(),
            "edge provenance is semantic"
        );
        match &mut right.environment.state_space {
            StateSpaceTerm::Ring { blocked_edges, .. } => blocked_edges.clear(),
            StateSpaceTerm::Graph { .. } => panic!("base term uses the ring constructor"),
        }
        right.calibration = Some(CalibrationTerm {
            pulses: vec![0],
            publishes_cells: true,
        });
        assert_ne!(
            left.family_hash(),
            right.family_hash(),
            "prelude behavior is semantic"
        );

        let first = GeneratedWorld {
            term: left.clone(),
            generator_metadata: GeneratorMetadata { seed: 1, index: 0 },
        }
        .compile()
        .unwrap();
        let second = GeneratedWorld {
            term: left,
            generator_metadata: GeneratorMetadata { seed: 2, index: 9 },
        }
        .compile()
        .unwrap();
        assert_eq!(first.family_hash, second.family_hash);
        assert_ne!(first.generator_metadata, second.generator_metadata);
    }
    fn superseding_term(guard: Guard) -> WorldTerm {
        let mut term = base_term("supersede", 5, 3, 2, IndexSet::from_indices([0, 1]));
        term.norm.expression = Norm::Supersede {
            before: Box::new(Norm::Settle { cell: 2 }),
            after: Box::new(Norm::Settle { cell: 3 }),
            guard,
        };
        term
    }

    #[test]
    fn a_public_goal_has_a_structured_diagnostic_without_changing_its_v2_trace() {
        let norms = [
            Norm::Settle { cell: 3 },
            Norm::Avoid { cell: 1 },
            Norm::both(Norm::Visit { cell: 1 }, Norm::Settle { cell: 2 }),
            Norm::Supersede {
                before: Box::new(Norm::Settle { cell: 1 }),
                after: Box::new(Norm::Settle { cell: 3 }),
                guard: Guard::AfterStep(1),
            },
            Norm::Priority {
                high: Box::new(Norm::Avoid { cell: 4 }),
                low: Box::new(Norm::Settle { cell: 3 }),
            },
        ];
        let mut published = BTreeSet::new();
        for norm in &norms {
            let carrier = SymbolicGoalDiagnostic::from_norm(norm);
            let encoded = carrier.encode_diagnostic();
            let (decoded, used) =
                SymbolicGoalDiagnostic::decode_diagnostic(&encoded).expect("a diagnostic decodes");
            assert_eq!(decoded, carrier);
            assert_eq!(used, encoded.len());
            assert!(published.insert(encoded));

            let mut term = base_term("carrier", 5, 3, 3, IndexSet::from_indices([0, 1]));
            term.norm.expression = norm.clone();
            let compiled = term.compile().expect("valid term");
            assert_eq!(compiled.public_goal_diagnostic(), Some(carrier));
            let states = term.environment.state_count();
            assert!(norm.referenced_cells().iter().all(|cell| *cell < states));
            assert_eq!(compiled.public_view(&[]).trace, vec![0, norm_code(norm)]);
        }
    }

    #[test]
    fn v2_public_norm_token_has_frozen_golden_values() {
        let settled = base_term("v2-settle", 5, 3, 3, IndexSet::from_indices([0, 1]))
            .compile()
            .expect("valid term");
        assert_eq!(
            settled.public_view(&[]).trace,
            vec![0, -5_808_588_758_991_127_739]
        );

        let superseded = superseding_term(Guard::Never)
            .compile()
            .expect("valid term");
        assert_eq!(
            superseded.public_view(&[]).trace,
            vec![0, 3_454_054_311_722_615_336]
        );
    }

    #[test]
    fn v2_keeps_the_known_supersession_collision_while_diagnostics_separate_it() {
        // The v2 encoding folded the guard into the supersession tag with XOR,
        // and `12 ^ 3` is the code `Never` used. Replay compatibility requires
        // that collision to remain in the one-slot public trace.
        let announced = superseding_term(Guard::AfterStep(3))
            .compile()
            .expect("valid term");
        let never = superseding_term(Guard::Never)
            .compile()
            .expect("valid term");
        assert_ne!(
            announced.public_goal_diagnostic(),
            never.public_goal_diagnostic()
        );
        assert_eq!(
            announced.public_view(&[]).trace,
            never.public_view(&[]).trace
        );
        assert_eq!(announced.public_view(&[]).trace.len(), 2);
    }

    #[test]
    fn a_privileged_goal_has_no_public_diagnostic() {
        let mut hidden = base_term("hidden-goal", 5, 2, 2, IndexSet::from_indices([0, 1]));
        hidden.norm.visibility = Visibility::Privileged;
        let mut moved = hidden.clone();
        moved.norm.expression = Norm::Settle { cell: 3 };

        let hidden = hidden.compile().expect("valid term");
        let moved = moved.compile().expect("valid term");
        assert_eq!(hidden.public_goal_diagnostic(), None);
        assert_eq!(hidden.public_view(&[]).trace, moved.public_view(&[]).trace);
        // The denotation still differs; only its diagnostic is withheld.
        assert!(hidden.outcome(&[0, 0]).norm_met);
        assert!(!moved.outcome(&[0, 0]).norm_met);
    }

    #[test]
    fn a_goal_outside_the_state_space_is_rejected_before_execution() {
        let mut absent = base_term("absent-goal", 5, 2, 2, IndexSet::from_indices([0, 1]));
        absent.norm.expression = Norm::Settle { cell: 9 };
        assert!(absent.compile().is_err());

        let mut private_absent = absent.clone();
        private_absent.norm.visibility = Visibility::Privileged;
        assert!(private_absent.compile().is_err());

        // A guard naming a missing cell is the same defect one level in: an
        // announcement whose timing can never be read.
        let mut guarded = base_term("absent-guard", 5, 2, 2, IndexSet::from_indices([0, 1]));
        guarded.norm.expression = Norm::Supersede {
            before: Box::new(Norm::Settle { cell: 1 }),
            after: Box::new(Norm::Settle { cell: 3 }),
            guard: Guard::OnCellEntry(7),
        };
        assert!(guarded.compile().is_err());

        guarded.norm.visibility = Visibility::Privileged;
        assert!(guarded.compile().is_err());
    }

    #[test]
    fn typed_events_keep_same_valued_public_signals_on_distinct_ports() {
        let mut term = base_term("typed-signals", 5, 2, 2, IndexSet::from_indices([0, 1]));
        term.ports
            .push(Port::new("signal-a", PortDirection::Output, PortValue::Signal).public());
        term.ports
            .push(Port::new("signal-b", PortDirection::Output, PortValue::Signal).public());
        term.disturbance = Some(ProcessTerm {
            name: "disturbance".into(),
            signals: vec![
                SignalTerm {
                    output_port: "signal-a".into(),
                    guard: Guard::AtStart,
                    value: 1,
                },
                SignalTerm {
                    output_port: "signal-b".into(),
                    guard: Guard::AtStart,
                    value: 1,
                },
            ],
        });
        let compiled = term.compile().expect("valid term");
        let signals: Vec<_> = compiled
            .public_event_view(&[])
            .groups
            .iter()
            .flat_map(|group| group.events.iter())
            .filter_map(|event| match event {
                PublicEvent::Signal { port, value } => Some((*port, *value)),
                _ => None,
            })
            .collect();
        assert_eq!(signals.len(), 2);
        assert_eq!(signals[0].1, signals[1].1);
        assert_ne!(signals[0].0, signals[1].0);
    }

    #[test]
    fn typed_events_exclude_private_signals() {
        let mut left = base_term(
            "typed-private-left",
            5,
            2,
            2,
            IndexSet::from_indices([0, 1]),
        );
        left.ports.push(Port::new(
            "private-signal",
            PortDirection::Output,
            PortValue::Signal,
        ));
        left.disturbance = Some(ProcessTerm {
            name: "private".into(),
            signals: vec![SignalTerm {
                output_port: "private-signal".into(),
                guard: Guard::AtStart,
                value: 7,
            }],
        });
        let mut right = left.clone();
        right.name = "typed-private-right".into();
        right.disturbance.as_mut().unwrap().signals[0].value = 99;
        let left = left.compile().unwrap();
        let right = right.compile().unwrap();
        assert_eq!(left.public_event_view(&[]), right.public_event_view(&[]));
    }

    #[test]
    fn calibration_events_include_pulses_and_observations_without_scored_time() {
        let mut term = base_term("typed-calibration", 5, 2, 2, IndexSet::from_indices([0, 1]));
        term.calibration = Some(CalibrationTerm {
            pulses: vec![0, 1],
            publishes_cells: true,
        });
        let compiled = term.compile().unwrap();
        let view = compiled.public_event_view(&[]);
        let calibration: Vec<_> = view
            .groups
            .iter()
            .filter(|group| matches!(group.time, PublicEventTime::Calibration { .. }))
            .collect();
        assert_eq!(calibration.len(), 3);
        assert!(matches!(
            calibration[1].events[0],
            PublicEvent::ActionExecuted { actuator: 0, .. }
        ));
        assert!(calibration[1]
            .events
            .iter()
            .any(|event| matches!(event, PublicEvent::Observation { .. })));
        assert!(view
            .groups
            .iter()
            .any(|group| matches!(group.time, PublicEventTime::BeforeAction { executed: 0 })));
    }

    #[test]
    fn calibration_reset_restores_the_declared_scored_start() {
        let mut term = base_term(
            "typed-calibration-reset",
            5,
            2,
            2,
            IndexSet::from_indices([0, 1]),
        );
        term.calibration = Some(CalibrationTerm {
            pulses: vec![0],
            publishes_cells: true,
        });
        let compiled = term.compile().expect("valid term");
        let view = compiled.public_event_view(&[0]);
        let opening = view
            .groups
            .iter()
            .find(|group| matches!(group.time, PublicEventTime::BeforeAction { executed: 0 }))
            .expect("task opening");
        assert!(opening.events.iter().any(|event| matches!(
            event,
            PublicEvent::Observation {
                value: PublicEventValue::Cell(0),
                ..
            }
        )));
        let first_after = view
            .groups
            .iter()
            .find(|group| matches!(group.time, PublicEventTime::AfterAction { executed: 1 }))
            .expect("first scored action");
        assert!(first_after.events.iter().any(|event| matches!(
            event,
            PublicEvent::Observation {
                value: PublicEventValue::Cell(1),
                ..
            }
        )));
    }

    #[test]
    fn shortest_optimal_plan_includes_zero_when_the_compiled_goal_starts_met() {
        let compiled = embodiment_term(
            "zero-plan",
            5,
            2,
            0,
            IndexSet::from_indices([0, 1, 2, 3]),
            Vec::new(),
        )
        .compile()
        .expect("valid term");
        assert_eq!(compiled.shortest_non_fallback_success_length(), Some(0));
    }

    #[test]
    fn restoration_event_declares_activation_timing_separately() {
        let mut term = base_term("typed-restoration", 5, 2, 4, IndexSet::from_indices([0]));
        term.restorations.push(SupportRestoration {
            actuator: 1,
            after_step: 0,
            announcement_port: "reveal".into(),
            announcement_value: 401,
        });
        let compiled = term.compile().unwrap();
        let view = compiled.public_event_view(&[1]);
        let announcement = view
            .groups
            .iter()
            .flat_map(|group| group.events.iter())
            .find_map(|event| match event {
                PublicEvent::RestorationAnnouncement {
                    actuator,
                    activates_after,
                    ..
                } => Some((*actuator, *activates_after)),
                _ => None,
            })
            .expect("restoration announcement");
        assert_eq!(announcement, (1, 0));
        assert_eq!(
            compiled.fragment.step(&compiled.contract, 0, 0, 1),
            0,
            "unsupported before the declared activation boundary"
        );
        assert_ne!(
            compiled.fragment.step(&compiled.contract, 0, 1, 1),
            0,
            "supported after the declared activation boundary"
        );
    }

    #[test]
    fn an_in_range_signal_is_not_an_observation_in_typed_events() {
        let mut term = base_term(
            "typed-in-range-signal",
            5,
            2,
            2,
            IndexSet::from_indices([0, 1]),
        );
        term.disturbance = Some(ProcessTerm {
            name: "signal".into(),
            signals: vec![SignalTerm {
                output_port: "reveal".into(),
                guard: Guard::AtStart,
                value: 1,
            }],
        });
        let compiled = term.compile().unwrap();
        let view = compiled.public_event_view(&[]);
        assert!(view
            .groups
            .iter()
            .flat_map(|group| group.events.iter())
            .any(|event| matches!(event, PublicEvent::Signal { value: 1, .. })));
        assert!(!view
            .groups
            .iter()
            .flat_map(|group| group.events.iter())
            .any(|event| matches!(
                event,
                PublicEvent::Observation {
                    value: PublicEventValue::Cell(1),
                    ..
                }
            )));
    }

    #[test]
    fn public_transition_table_is_exact_across_time_source_and_action() {
        let mut term = base_term("typed-transitions", 5, 2, 4, IndexSet::from_indices([0]));
        term.restorations.push(SupportRestoration {
            actuator: 1,
            after_step: 0,
            announcement_port: "reveal".into(),
            announcement_value: 401,
        });
        let compiled = term.compile().unwrap();
        let table = compiled.public_transition_table();
        assert_eq!(
            table.rows.len(),
            (table.horizon + 1) * table.states * compiled.actions().len()
        );
        for row in &table.rows {
            assert_eq!(
                row.destination,
                compiled.fragment.step(
                    &compiled.contract,
                    row.address.source,
                    row.address.executed,
                    row.address.actuator,
                )
            );
        }
        let restored = table
            .rows
            .iter()
            .find(|row| {
                row.address.source == 0 && row.address.actuator == 1 && row.address.executed == 1
            })
            .expect("restored row");
        assert_ne!(restored.destination, 0);
        assert!(table.rows.iter().all(|row| {
            row.fallback
                == compiled
                    .contract
                    .program
                    .actuator(row.address.actuator)
                    .is_some_and(|actuator| actuator.role == ActionRole::Fallback)
        }));
    }
}
