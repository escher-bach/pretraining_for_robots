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
    BoundaryEffect, Coupling, CouplingRule, Displaced, Fragment, Guard, GuardContext, IndexSet,
    Interrupt, KernelUse, Norm, PubliclyObservable, Restriction,
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
        for actuator in &self.body.actuation.actuators {
            if usize::from(actuator.id) >= IndexSet::CAPACITY || !actuator_ids.insert(actuator.id) {
                return Err(CompileError::Invalid(
                    "actuator ids must be unique values in 0..32".into(),
                ));
            }
        }
        if actuator_ids.is_empty() {
            return Err(CompileError::Invalid(
                "body needs at least one actuator".into(),
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
            reveal: !self.reveals.is_empty(),
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
                    restrictions: self.restrictions.clone(),
                    couplings: self.couplings.clone(),
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
        let mut pieces = vec![
            format!(
                "cells={};horizon={};start={};support={}",
                self.body.morphology.cells,
                self.horizon,
                self.environment.start,
                self.body.actuation.supported.0
            ),
            format!(
                "norm={:?};norm_view={:?}",
                self.norm.expression, self.norm.visibility
            ),
            format!(
                "sensor={}:{}",
                self.body.sensorium.publishes_cell, self.body.sensorium.cell_port
            ),
        ];
        let mut ports: Vec<_> = self
            .ports
            .iter()
            .map(|port| format!("port={:?}", port))
            .collect();
        ports.sort();
        pieces.extend(ports);
        let mut wiring: Vec<_> = self
            .wiring
            .iter()
            .map(|wire| format!("wire={}:{}", wire.from, wire.to))
            .collect();
        wiring.sort();
        pieces.extend(wiring);
        let mut actuators: Vec<_> = self
            .body
            .actuation
            .actuators
            .iter()
            .map(|actuator| {
                format!(
                    "act={}:{}:{}",
                    actuator.id, actuator.name, actuator.displacement
                )
            })
            .collect();
        actuators.sort();
        pieces.extend(actuators);
        let mut edges: Vec<_> = self
            .environment
            .blocked_edges
            .iter()
            .map(|edge| format!("edge={}:{}", edge.0, edge.1))
            .collect();
        edges.sort();
        pieces.extend(edges);
        let mut restrictions: Vec<_> = self
            .restrictions
            .iter()
            .map(|restriction| format!("restrict={restriction:?}"))
            .collect();
        restrictions.sort();
        pieces.extend(restrictions);
        let mut reveals: Vec<_> = self
            .reveals
            .iter()
            .map(|reveal| format!("reveal={reveal:?}"))
            .collect();
        reveals.sort();
        pieces.extend(reveals);
        for coupling in &self.couplings {
            // Writer order is semantic under Override, so it is intentionally
            // retained rather than sorted.
            pieces.push(format!("coupling={coupling:?}"));
        }
        for process in [self.disturbance.as_ref(), self.scaffold.as_ref()]
            .into_iter()
            .flatten()
        {
            let mut signals: Vec<_> = process
                .signals
                .iter()
                .map(|signal| format!("{signal:?}"))
                .collect();
            signals.sort();
            pieces.push(format!("process={}:{}", process.name, signals.join(",")));
        }
        let mut interrupts: Vec<_> = self
            .interrupts
            .iter()
            .map(|interrupt| format!("interrupt={interrupt:?}"))
            .collect();
        interrupts.sort();
        pieces.extend(interrupts);
        let mut state = 0xcbf29ce484222325u64;
        for byte in pieces.join("|").bytes() {
            state ^= u64::from(byte);
            state = state.wrapping_mul(0x100000001b3);
        }
        format!("{state:016x}")
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
    restrictions: Vec<Restriction>,
    couplings: Vec<CouplingTerm>,
    disturbance: Option<ProcessTerm>,
    scaffold: Option<ProcessTerm>,
    interrupts: Vec<InterruptTerm>,
    reveals: Vec<RevealTerm>,
    ports: BTreeMap<String, Port>,
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

    fn action_supported(&self, action: u16) -> bool {
        self.supported.contains(usize::from(action))
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
            .map(|term| {
                let writes: Vec<f64> = term
                    .writers
                    .iter()
                    .filter(|writer| writer.guard.fired(context))
                    .map(|writer| writer.value as f64)
                    .collect();
                term.coupling.resolve(&writes).unwrap_or(0.0) as i32
            })
            .sum()
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
        let program = &contract.program;
        if program
            .restrictions
            .iter()
            .any(|restriction| !restriction.admits_cell(cell))
        {
            return cell;
        }
        let Some(actuator) = program.actuator(action) else {
            return cell;
        };
        if !program.action_supported(action) || program.blocked_edges.contains(&(cell, action)) {
            return cell;
        }
        let context = guard_context(executed + 1, Some(action), cell);
        let displacement = actuator.displacement + program.coupling_displacement(context);
        let moved = (cell as i32 + displacement).rem_euclid(program.cells as i32) as usize;
        for restriction in &program.restrictions {
            if !restriction.admits_cell(moved) {
                return match restriction.boundary_effect() {
                    Some(BoundaryEffect::Reset) => program.start,
                    Some(BoundaryEffect::Absorbing) => moved,
                    None => moved,
                };
            }
        }
        moved
    }

    fn value(
        &self,
        contract: &Self::Contract,
        trajectory: &[usize],
        actions: &[Self::Action],
    ) -> i32 {
        let last = *trajectory.last().unwrap_or(&contract.program.start);
        let last_action = actions.last().copied();
        let verdict = contract
            .program
            .norm
            .evaluate(trajectory, guard_context(actions.len(), last_action, last));
        if verdict.met {
            100 - actions.len() as i32
        } else if verdict.violated_prohibition {
            -100
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
        if program.publishes_cell {
            trace.push(cell as i64);
        }
        if program.norm_public {
            trace.push(norm_code(&program.norm));
        }
        trace.extend(program.public_events(0, None, cell));
        for (executed, action) in actions.iter().enumerate() {
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

impl Default for GenerationSpec {
    fn default() -> Self {
        Self {
            seed: 0,
            count: 4,
            min_cells: 3,
            max_cells: 6,
            min_horizon: 2,
            max_horizon: 4,
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

    pub fn generate(&self) -> Result<Vec<GeneratedWorld>, CompileError> {
        self.validate()?;
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let mut worlds = Vec::with_capacity(self.count);
        for index in 0..self.count {
            let cells = rng.gen_range(self.min_cells..=self.max_cells);
            let horizon = rng.gen_range(self.min_horizon..=self.max_horizon);
            let goal = rng.gen_range(0..cells);
            let support = if index % 2 == 0 {
                IndexSet::from_indices([0, 1])
            } else {
                IndexSet::from_indices([0])
            };
            let mut term = base_term(format!("generated-{index}"), cells, horizon, goal, support);
            if index % 3 == 0 {
                term.reveals.push(RevealTerm {
                    source_visibility: Visibility::Privileged,
                    output_port: "reveal".into(),
                    guard: Guard::AfterStep(0),
                    value: index as i64,
                });
            }
            let generated = GeneratedWorld {
                term,
                generator_metadata: GeneratorMetadata {
                    seed: self.seed,
                    index,
                },
            };
            generated.compile()?;
            worlds.push(generated);
        }
        Ok(worlds)
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
                    },
                    Actuator {
                        id: 1,
                        name: "retreat".into(),
                        displacement: -1,
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
    fn generated_terms_are_bounded_deterministic_and_executable() {
        let specification = GenerationSpec {
            seed: 42,
            count: 6,
            ..GenerationSpec::default()
        };
        let first = specification.generate().unwrap();
        let second = specification.generate().unwrap();
        assert_eq!(first, second);
        for term in first {
            let compiled = term.compile().unwrap();
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
