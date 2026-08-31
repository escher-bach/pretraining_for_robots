//! An isolated, finite compiler for typed embodied-world terms.
//!
//! This crate is a spike, not a replacement for a card evaluator.  It gives
//! the shared kernel a deliberately small executable interpretation over a
//! ring world: body support and sensorium are first-class, ports carry both a
//! value type and a visibility, and every public event has an explicit source.
//! Its output implements the existing finite audit interfaces, so the existing
//! query algebra can audit generated terms without being changed.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use pretraining_g0_contract::{
    identification_diameter, AmbiguitySet, BoundaryEffect, Coupling, CouplingRule, Displaced,
    Fragment, Guard, GuardContext, IndexSet, Interrupt, KernelUse, Norm, PubliclyObservable,
    Restriction,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

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
    /// The finite configuration cells occupied by the body/environment pair.
    pub cells: usize,
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
    /// A local environmental deletion, deliberately not an actuator-support
    /// change.  Entries are `(cell, actuator_id)`.
    pub blocked_edges: Vec<(usize, u16)>,
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
    UnknownPort(String),
    IllTypedWire { from: String, to: String },
    VisibilityLeak { from: String, to: String },
    MonitorAsSource(String),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid world term: {message}"),
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

impl WorldTerm {
    /// Validate every structural invariant before any executable object is
    /// made.  Validation is intentionally conservative around conflict
    /// coupling: two declared writers are rejected rather than depending on a
    /// future guard analysis to prove that they never coincide.
    pub fn validate(&self) -> Result<(), CompileError> {
        if self.name.trim().is_empty() {
            return Err(CompileError::Invalid("world has no name".into()));
        }
        if !(2..=IndexSet::CAPACITY).contains(&self.body.morphology.cells) {
            return Err(CompileError::Invalid(
                "morphology cells must be in 2..=32".into(),
            ));
        }
        if self.horizon == 0 || self.horizon > 8 {
            return Err(CompileError::Invalid(
                "horizon must be in 1..=8 for exact enumeration".into(),
            ));
        }
        if self.environment.start >= self.body.morphology.cells {
            return Err(CompileError::Invalid(
                "environment start is outside morphology".into(),
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
        for (cell, action) in &self.environment.blocked_edges {
            if *cell >= self.body.morphology.cells || !actuator_ids.contains(action) {
                return Err(CompileError::Invalid(
                    "blocked edge refers to an unknown cell or actuator".into(),
                ));
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
                    if inadmissible
                        .iter()
                        .any(|cell| cell >= self.body.morphology.cells) =>
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
                    cells: self.body.morphology.cells,
                    start: self.environment.start,
                    actuators: self.body.actuation.actuators.clone(),
                    supported: self.body.actuation.supported,
                    publishes_cell: self.body.sensorium.publishes_cell,
                    blocked_edges: self.environment.blocked_edges.iter().copied().collect(),
                    horizon,
                    norm: self.norm.expression.clone(),
                    norm_public: self.norm.visibility == Visibility::Public,
                    scoring: self.scoring.clone(),
                    calibration: self.calibration.clone(),
                    restorations: self.restorations.clone(),
                    restrictions: self.restrictions.clone(),
                    couplings: self
                        .couplings
                        .iter()
                        .map(LoweredCoupling::from_term)
                        .collect(),
                    disturbance: self.disturbance.clone(),
                    scaffold: self.scaffold.clone(),
                    interrupts: self.interrupts.clone(),
                    reveals: self.reveals.clone(),
                    ports: self
                        .ports
                        .iter()
                        .map(|port| (port.name.clone(), port.clone()))
                        .collect(),
                },
            },
            kernel_use: self.derived_kernel_use(),
            generator_metadata,
            family_hash: self.family_hash(),
        })
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
        canonical.environment.blocked_edges.sort_unstable();
        canonical.environment.blocked_edges.dedup();
        // Preserve orders that affect execution: process signals, norm trees,
        // restriction precedence, restoration announcements, and Override
        // writer order.  `serde_json` provides a maintained canonical byte
        // encoding for this already-canonical term, and BLAKE3 provides the
        // stable digest rather than a local hash implementation.
        let bytes = serde_json::to_vec(&canonical).expect("world terms serialize");
        blake3::hash(&bytes).to_hex().to_string()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CompiledProgram {
    cells: usize,
    start: usize,
    actuators: Vec<Actuator>,
    supported: IndexSet,
    publishes_cell: bool,
    blocked_edges: BTreeSet<(usize, u16)>,
    horizon: usize,
    norm: Norm,
    norm_public: bool,
    scoring: ScoringTerm,
    calibration: Option<CalibrationTerm>,
    restorations: Vec<SupportRestoration>,
    restrictions: Vec<Restriction>,
    couplings: Vec<LoweredCoupling>,
    disturbance: Option<ProcessTerm>,
    scaffold: Option<ProcessTerm>,
    interrupts: Vec<InterruptTerm>,
    reveals: Vec<RevealTerm>,
    ports: BTreeMap<String, Port>,
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

    fn action_supported(&self, action: u16, executed: usize) -> bool {
        self.supported.contains(usize::from(action))
            || self.restorations.iter().any(|restoration| {
                restoration.actuator == action && executed > restoration.after_step
            })
    }

    fn action_permitted(&self, action: u16, executed: usize) -> bool {
        self.action_supported(action, executed)
            && self
                .restrictions
                .iter()
                .all(|restriction| restriction.permits_action(action))
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

    fn coupling_displacement(&self, context: GuardContext) -> i32 {
        self.couplings
            .iter()
            .map(|term| term.resolve(context))
            .sum()
    }

    fn transition(&self, cell: usize, executed: usize, action: u16) -> usize {
        if self
            .restrictions
            .iter()
            .any(|restriction| !restriction.admits_cell(cell))
        {
            return cell;
        }
        let Some(actuator) = self.actuator(action) else {
            return cell;
        };
        if actuator.role == ActionRole::Fallback
            || !self.action_permitted(action, executed)
            || self.blocked_edges.contains(&(cell, action))
        {
            return cell;
        }
        let context = guard_context(executed + 1, Some(action), cell);
        let displacement = actuator.displacement + self.coupling_displacement(context);
        let moved = (cell as i32 + displacement).rem_euclid(self.cells as i32) as usize;
        for restriction in &self.restrictions {
            if !restriction.admits_cell(moved) {
                return match restriction.boundary_effect() {
                    Some(BoundaryEffect::Reset) => self.start,
                    Some(BoundaryEffect::Absorbing) => moved,
                    None => moved,
                };
            }
        }
        moved
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

    /// Execute a scored sequence once, exposing only generic scoring facts.
    /// A caller that needs a named goal can inspect the term's norm separately;
    /// no card-specific outcome type is embedded in the compiler.
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
            blocked_edges: self
                .contract
                .program
                .blocked_edges
                .iter()
                .copied()
                .collect(),
            supported: self.contract.program.supported,
        }
    }

    pub fn audit_metadata(&self) -> AuditMetadata {
        AuditMetadata {
            family_hash: self.family_hash.clone(),
            kernel_use: self.kernel_use.clone(),
            public_ports: self
                .contract
                .program
                .ports
                .values()
                .filter(|port| port.visibility == Visibility::Public)
                .map(|port| port.name.clone())
                .collect(),
        }
    }
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
    pub blocked_edges: Vec<(usize, u16)>,
    pub supported: IndexSet,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditMetadata {
    pub family_hash: String,
    pub kernel_use: KernelUse,
    pub public_ports: Vec<String>,
}

/// Bounded construction metadata.  It is never copied into [`WorldTerm`] or a
/// public trace; replay is instead the pair `(specification, index)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationSpec {
    pub seed: u64,
    pub count: usize,
    pub min_cells: usize,
    pub max_cells: usize,
    pub min_horizon: usize,
    pub max_horizon: usize,
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

impl GeneratedEmbodimentFamily {
    /// Re-run the exact finite filter that admits one generated family.  The
    /// receipt is evidence about executable terms, not learner evidence.
    pub fn verify(&self) -> Result<EmbodimentValidityReceipt, CompileError> {
        let body = self.body_limited.compile()?;
        let unrestricted = self.unrestricted_control.compile()?;
        let twin = self.environment_twin.compile()?;
        if body.actions() != unrestricted.actions()
            || body.actions() != twin.actions()
            || body.horizon() != unrestricted.horizon()
            || body.horizon() != twin.horizon()
        {
            return Err(CompileError::Invalid(
                "an embodiment family must share action alphabet and scored horizon".into(),
            ));
        }
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
            goal_differs_from_start,
            body_limitation_changes_ceiling: body_ceiling != unrestricted_ceiling,
            twin_trajectories_equal: trajectories_equal,
            twin_values_equal: values_equal,
            twin_optimal_sequences_equal: body_ceiling == twin_ceiling
                && body_optimal == twin_optimal,
            twin_publicly_distinct,
            calibration_identifies_body,
            valid: goal_differs_from_start
                && body_ceiling != unrestricted_ceiling
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
        self.generate_embodiment_families()
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
            morphology: Morphology { cells },
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
        environment: EnvironmentTerm {
            start: 0,
            blocked_edges,
        },
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
            morphology: Morphology { cells },
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
        environment: EnvironmentTerm {
            start: 0,
            blocked_edges: Vec::new(),
        },
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
            environment: EnvironmentTerm {
                start: 0,
                blocked_edges: vec![(0, 1)],
            },
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
        right.environment.blocked_edges.push((0, 0));
        assert_ne!(
            left.family_hash(),
            right.family_hash(),
            "edge provenance is semantic"
        );
        right.environment.blocked_edges.clear();
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
}
