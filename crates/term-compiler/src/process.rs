//! A compositional continuous-process IR for the trajectory world.
//!
//! The original term compiler executes finite symbolic transition systems.
//! This module keeps its useful typed boundary (named ports, visibility,
//! validation, deterministic generation) and adds the process graph needed by
//! the trajectory trainer.  It deliberately does not render learner events:
//! the Python world owns numerical integration and publishes only its public
//! event schema.  `runtime_private` is an execution input, never a learner
//! input.

use std::collections::{BTreeMap, BTreeSet};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

pub const PROCESS_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessPortKind {
    Action,
    Goal,
    State,
    Disturbance,
    Observation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessVisibility {
    Public,
    Private,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessPort {
    pub name: String,
    pub kind: ProcessPortKind,
    pub visibility: ProcessVisibility,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProcessNodeKind {
    /// An externally supplied action or goal.  Inputs are public only when
    /// they are emitted in a public query; the process itself never emits
    /// teacher labels.
    Input {
        channel: ProcessPortKind,
    },
    /// A first-order body/environment integrator.
    Plant {
        state_dim: usize,
        gain: f64,
    },
    /// A memory component.  `alpha` is the retained previous command weight.
    ActuatorLag {
        alpha: f64,
    },
    GoalSwitch {
        at_step: usize,
        magnitude: f64,
    },
    Disturbance {
        start_step: usize,
        amplitude: f64,
    },
    /// The only node allowed to publish learner-visible process outputs.
    Output {
        channel: ProcessPortKind,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessNode {
    pub id: String,
    pub kind: ProcessNodeKind,
    pub inputs: Vec<ProcessPort>,
    pub outputs: Vec<ProcessPort>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessWiring {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProcessFamily {
    Reaching,
    ActuatorLag,
    LagGoalSwitchDisturbance,
    GoalSwitch,
    Disturbance,
    GoalSwitchDisturbance,
    ActuatorLagGoalSwitch,
    ActuatorLagDisturbance,
}

impl ProcessFamily {
    fn uses_lag(self) -> bool {
        matches!(
            self,
            Self::ActuatorLag
                | Self::LagGoalSwitchDisturbance
                | Self::ActuatorLagGoalSwitch
                | Self::ActuatorLagDisturbance
        )
    }

    fn uses_goal_switch(self) -> bool {
        matches!(
            self,
            Self::LagGoalSwitchDisturbance
                | Self::GoalSwitch
                | Self::GoalSwitchDisturbance
                | Self::ActuatorLagGoalSwitch
        )
    }

    fn uses_disturbance(self) -> bool {
        matches!(
            self,
            Self::LagGoalSwitchDisturbance
                | Self::Disturbance
                | Self::GoalSwitchDisturbance
                | Self::ActuatorLagDisturbance
        )
    }

    fn label(self) -> &'static str {
        match self {
            Self::Reaching => "reaching",
            Self::ActuatorLag => "actuator-lag",
            Self::GoalSwitch => "goal-switch",
            Self::Disturbance => "disturbance",
            Self::GoalSwitchDisturbance => "goal-switch-disturbance",
            Self::ActuatorLagGoalSwitch => "actuator-lag-goal-switch",
            Self::ActuatorLagDisturbance => "actuator-lag-disturbance",
            Self::LagGoalSwitchDisturbance => "actuator-lag-goal-switch-disturbance",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoalSwitchConfig {
    pub step: usize,
    pub magnitude: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisturbanceConfig {
    pub start_step: usize,
    pub amplitude: f64,
}

/// Numerical parameters are kept in a separately named runtime section.  A
/// public trajectory exporter may use this section to simulate a world, but
/// the learner loader must consume only `public_schema` and recorded events.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimePrivate {
    pub dynamics_gain: f64,
    pub lag_alpha: Option<f64>,
    pub goal_switch: Option<GoalSwitchConfig>,
    pub disturbance: Option<DisturbanceConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicSchema {
    pub schema: String,
    pub sensor_dim: usize,
    pub action_dim: usize,
    pub channels: Vec<String>,
    pub event_kinds: Vec<String>,
}

/// Fully compiled, serializable process program.  This is the stable bridge
/// consumed by the Python trajectory world; no generator seed is present.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompiledProcess {
    pub schema_version: u16,
    pub family: ProcessFamily,
    pub name: String,
    pub horizon: usize,
    pub sensor_dim: usize,
    pub action_dim: usize,
    pub nodes: Vec<ProcessNode>,
    pub wiring: Vec<ProcessWiring>,
    pub public_schema: PublicSchema,
    #[serde(rename = "runtime_private")]
    pub runtime_private: RuntimePrivate,
}

/// Replay coordinates are intentionally outside `CompiledProcess` and are not
/// included in public traces or public schema metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneratedProcess {
    pub process: CompiledProcess,
    pub generator_seed: u64,
    pub generator_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessCompileError {
    Invalid(String),
    DuplicateNode(String),
    DuplicatePort(String),
    UnknownEndpoint(String),
    KindMismatch { from: String, to: String },
    VisibilityLeak(String),
    Cycle,
    MissingNodeKind(&'static str),
}

impl std::fmt::Display for ProcessCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid process: {s}"),
            Self::DuplicateNode(s) => write!(f, "duplicate process node `{s}`"),
            Self::DuplicatePort(s) => write!(f, "duplicate process port `{s}`"),
            Self::UnknownEndpoint(s) => write!(f, "unknown process endpoint `{s}`"),
            Self::KindMismatch { from, to } => write!(f, "incompatible wire `{from}` -> `{to}`"),
            Self::VisibilityLeak(s) => write!(f, "private process output leaks through `{s}`"),
            Self::Cycle => write!(f, "process wiring contains a cycle"),
            Self::MissingNodeKind(s) => write!(f, "process is missing {s} node"),
        }
    }
}

impl std::error::Error for ProcessCompileError {}

fn port(id: &str, kind: ProcessPortKind, visibility: ProcessVisibility) -> ProcessPort {
    ProcessPort {
        name: id.into(),
        kind,
        visibility,
    }
}

fn node(
    id: &str,
    kind: ProcessNodeKind,
    inputs: Vec<ProcessPort>,
    outputs: Vec<ProcessPort>,
) -> ProcessNode {
    ProcessNode {
        id: id.into(),
        kind,
        inputs,
        outputs,
    }
}

fn base_nodes(
    sensor_dim: usize,
    action_dim: usize,
    family: ProcessFamily,
    horizon: usize,
) -> (Vec<ProcessNode>, Vec<ProcessWiring>) {
    let action = node(
        "action",
        ProcessNodeKind::Input {
            channel: ProcessPortKind::Action,
        },
        vec![],
        vec![port(
            "out",
            ProcessPortKind::Action,
            ProcessVisibility::Public,
        )],
    );
    let goal = node(
        "goal",
        ProcessNodeKind::Input {
            channel: ProcessPortKind::Goal,
        },
        vec![],
        vec![port(
            "out",
            ProcessPortKind::Goal,
            ProcessVisibility::Public,
        )],
    );
    let plant = node(
        "plant",
        ProcessNodeKind::Plant {
            state_dim: 2,
            gain: 1.0,
        },
        vec![
            port(
                "action",
                ProcessPortKind::Action,
                ProcessVisibility::Private,
            ),
            port(
                "disturbance",
                ProcessPortKind::Disturbance,
                ProcessVisibility::Private,
            ),
        ],
        vec![port(
            "state",
            ProcessPortKind::State,
            ProcessVisibility::Private,
        )],
    );
    let observation = node(
        "observation",
        ProcessNodeKind::Output {
            channel: ProcessPortKind::Observation,
        },
        vec![port(
            "state",
            ProcessPortKind::State,
            ProcessVisibility::Private,
        )],
        vec![port(
            "out",
            ProcessPortKind::Observation,
            ProcessVisibility::Public,
        )],
    );
    let mut nodes = vec![action, goal, plant, observation];
    let mut wiring = vec![
        ProcessWiring {
            from: "action.out".into(),
            to: "plant.action".into(),
        },
        ProcessWiring {
            from: "plant.state".into(),
            to: "observation.state".into(),
        },
    ];
    if family.uses_lag() {
        let lag = node(
            "actuator_lag",
            ProcessNodeKind::ActuatorLag { alpha: 0.75 },
            vec![port(
                "command",
                ProcessPortKind::Action,
                ProcessVisibility::Private,
            )],
            vec![port(
                "out",
                ProcessPortKind::Action,
                ProcessVisibility::Private,
            )],
        );
        wiring[0] = ProcessWiring {
            from: "action.out".into(),
            to: "actuator_lag.command".into(),
        };
        wiring.insert(
            1,
            ProcessWiring {
                from: "actuator_lag.out".into(),
                to: "plant.action".into(),
            },
        );
        nodes.push(lag);
    }
    if family.uses_goal_switch() {
        let switch = node(
            "goal_switch",
            ProcessNodeKind::GoalSwitch {
                at_step: horizon / 2,
                magnitude: 0.22,
            },
            vec![port(
                "goal",
                ProcessPortKind::Goal,
                ProcessVisibility::Private,
            )],
            vec![port(
                "out",
                ProcessPortKind::Goal,
                ProcessVisibility::Private,
            )],
        );
        let goal_observation = node(
            "goal_observation",
            ProcessNodeKind::Output {
                channel: ProcessPortKind::Goal,
            },
            vec![port(
                "goal",
                ProcessPortKind::Goal,
                ProcessVisibility::Private,
            )],
            vec![port(
                "out",
                ProcessPortKind::Goal,
                ProcessVisibility::Public,
            )],
        );
        wiring.push(ProcessWiring {
            from: "goal.out".into(),
            to: "goal_switch.goal".into(),
        });
        wiring.push(ProcessWiring {
            from: "goal_switch.out".into(),
            to: "goal_observation.goal".into(),
        });
        nodes.push(switch);
        nodes.push(goal_observation);
    }
    // Disturbance is an independent optional operator and therefore must be
    // emitted for disturbance-only and lag-plus-disturbance compositions too.
    if family.uses_disturbance() {
        let disturbance = node(
            "disturbance",
            ProcessNodeKind::Disturbance {
                start_step: horizon / 3,
                amplitude: 0.035,
            },
            vec![],
            vec![port(
                "out",
                ProcessPortKind::Disturbance,
                ProcessVisibility::Private,
            )],
        );
        wiring.push(ProcessWiring {
            from: "disturbance.out".into(),
            to: "plant.disturbance".into(),
        });
        nodes.push(disturbance);
    }
    // A zero disturbance source makes the plant input total for the two
    // simpler families; it is private and cannot appear in a public event.
    if !family.uses_disturbance() {
        let disturbance = node(
            "zero_disturbance",
            ProcessNodeKind::Disturbance {
                start_step: horizon + 1,
                amplitude: 0.0,
            },
            vec![],
            vec![port(
                "out",
                ProcessPortKind::Disturbance,
                ProcessVisibility::Private,
            )],
        );
        wiring.push(ProcessWiring {
            from: "zero_disturbance.out".into(),
            to: "plant.disturbance".into(),
        });
        nodes.push(disturbance);
    }
    let _ = (sensor_dim, action_dim);
    (nodes, wiring)
}

fn public_schema(sensor_dim: usize, action_dim: usize) -> PublicSchema {
    PublicSchema {
        schema: "trajectory-public-events-v2".into(),
        sensor_dim,
        action_dim,
        channels: vec!["sensor".into(), "goal".into(), "target_action".into()],
        event_kinds: vec![
            "reset".into(),
            "observation".into(),
            "goal".into(),
            "action_query".into(),
            "action_executed".into(),
            "episode_end".into(),
        ],
    }
}

fn lower_runtime(
    nodes: &[ProcessNode],
    horizon: usize,
) -> Result<RuntimePrivate, ProcessCompileError> {
    let mut dynamics_gain = None;
    let mut lag_alpha = None;
    let mut goal_switch = None;
    let mut disturbance = None;
    for node in nodes {
        match node.kind {
            ProcessNodeKind::Plant { gain, .. } => {
                if dynamics_gain.replace(gain).is_some() {
                    return Err(ProcessCompileError::Invalid(
                        "process has multiple plant nodes".into(),
                    ));
                }
            }
            ProcessNodeKind::ActuatorLag { alpha } => {
                if lag_alpha.replace(alpha).is_some() {
                    return Err(ProcessCompileError::Invalid(
                        "process has multiple actuator lag nodes".into(),
                    ));
                }
            }
            ProcessNodeKind::GoalSwitch { at_step, magnitude } => {
                if at_step >= horizon
                    || goal_switch
                        .replace(GoalSwitchConfig {
                            step: at_step,
                            magnitude,
                        })
                        .is_some()
                {
                    return Err(ProcessCompileError::Invalid(
                        "process has an invalid or duplicate goal switch".into(),
                    ));
                }
            }
            ProcessNodeKind::Disturbance {
                start_step,
                amplitude,
            } => {
                if amplitude != 0.0 {
                    if disturbance
                        .replace(DisturbanceConfig {
                            start_step,
                            amplitude,
                        })
                        .is_some()
                    {
                        return Err(ProcessCompileError::Invalid(
                            "process has multiple disturbance nodes".into(),
                        ));
                    }
                }
            }
            ProcessNodeKind::Input { .. } | ProcessNodeKind::Output { .. } => {}
        }
    }
    Ok(RuntimePrivate {
        dynamics_gain: dynamics_gain.ok_or(ProcessCompileError::MissingNodeKind("plant"))?,
        lag_alpha,
        goal_switch,
        disturbance: disturbance.filter(|value| value.amplitude != 0.0),
    })
}

impl CompiledProcess {
    /// Lower numerical execution parameters from the validated graph nodes.
    /// Callers that need the full contract should call [`Self::validate`],
    /// which also checks that the serialized runtime section matches this
    /// lowering.
    pub fn lowered_runtime_private(&self) -> Result<RuntimePrivate, ProcessCompileError> {
        lower_runtime(&self.nodes, self.horizon)
    }

    pub fn validate(&self) -> Result<(), ProcessCompileError> {
        if self.schema_version != PROCESS_SCHEMA_VERSION {
            return Err(ProcessCompileError::Invalid(
                "unsupported process schema version".into(),
            ));
        }
        if self.horizon == 0
            || self.horizon > 4096
            || self.action_dim == 0
            || self.action_dim > 64
            || self.sensor_dim == 0
            || self.sensor_dim > 256
        {
            return Err(ProcessCompileError::Invalid(
                "dimensions or horizon outside bounds".into(),
            ));
        }
        if self.public_schema.sensor_dim != self.sensor_dim
            || self.public_schema.action_dim != self.action_dim
        {
            return Err(ProcessCompileError::Invalid(
                "public schema dimensions disagree with process".into(),
            ));
        }
        let mut nodes = BTreeMap::new();
        let mut endpoints = BTreeMap::new();
        let mut has_plant = false;
        let mut has_observation = false;
        for n in &self.nodes {
            if nodes.insert(n.id.clone(), n).is_some() {
                return Err(ProcessCompileError::DuplicateNode(n.id.clone()));
            }
            for p in n.inputs.iter().chain(n.outputs.iter()) {
                let endpoint = format!("{}.{}", n.id, p.name);
                if endpoints
                    .insert(endpoint.clone(), (p, n, p.visibility))
                    .is_some()
                {
                    return Err(ProcessCompileError::DuplicatePort(endpoint));
                }
                if p.visibility == ProcessVisibility::Public
                    && !matches!(
                        n.kind,
                        ProcessNodeKind::Input { .. } | ProcessNodeKind::Output { .. }
                    )
                {
                    return Err(ProcessCompileError::VisibilityLeak(format!(
                        "{n_id}.{name}",
                        n_id = n.id,
                        name = p.name
                    )));
                }
            }
            has_plant |= matches!(n.kind, ProcessNodeKind::Plant { .. });
            has_observation |= matches!(
                n.kind,
                ProcessNodeKind::Output {
                    channel: ProcessPortKind::Observation
                }
            );
            if let ProcessNodeKind::ActuatorLag { alpha } = n.kind {
                if !(0.0..1.0).contains(&alpha) {
                    return Err(ProcessCompileError::Invalid(
                        "lag alpha must be in (0,1)".into(),
                    ));
                }
            }
        }
        if !has_plant {
            return Err(ProcessCompileError::MissingNodeKind("plant"));
        }
        if !has_observation {
            return Err(ProcessCompileError::MissingNodeKind("observation output"));
        }
        let mut edges = BTreeSet::new();
        let mut incoming = BTreeMap::<String, usize>::new();
        let mut used_outputs = BTreeSet::<String>::new();
        for wire in &self.wiring {
            let (from, from_node, from_visibility) = endpoints
                .get(&wire.from)
                .ok_or_else(|| ProcessCompileError::UnknownEndpoint(wire.from.clone()))?;
            let (to, to_node, _) = endpoints
                .get(&wire.to)
                .ok_or_else(|| ProcessCompileError::UnknownEndpoint(wire.to.clone()))?;
            if !from_node.outputs.iter().any(|p| p.name == from.name)
                || !to_node.inputs.iter().any(|p| p.name == to.name)
            {
                return Err(ProcessCompileError::KindMismatch {
                    from: wire.from.clone(),
                    to: wire.to.clone(),
                });
            }
            if from.kind != to.kind {
                return Err(ProcessCompileError::KindMismatch {
                    from: wire.from.clone(),
                    to: wire.to.clone(),
                });
            }
            if *from_visibility == ProcessVisibility::Private
                && to.visibility == ProcessVisibility::Public
                && !matches!(to_node.kind, ProcessNodeKind::Output { .. })
            {
                return Err(ProcessCompileError::VisibilityLeak(format!(
                    "{} -> {}",
                    wire.from, wire.to
                )));
            }
            if !edges.insert((from_node.id.clone(), to_node.id.clone())) {
                return Err(ProcessCompileError::Invalid(format!(
                    "duplicate wire `{}`",
                    wire.from
                )));
            }
            *incoming.entry(wire.to.clone()).or_default() += 1;
            used_outputs.insert(wire.from.clone());
        }
        for node in &self.nodes {
            for input in &node.inputs {
                let endpoint = format!("{}.{}", node.id, input.name);
                if incoming.get(&endpoint).copied().unwrap_or(0) != 1 {
                    return Err(ProcessCompileError::Invalid(format!(
                        "input `{endpoint}` must have exactly one wire"
                    )));
                }
            }
            if !matches!(
                node.kind,
                ProcessNodeKind::Input { .. } | ProcessNodeKind::Output { .. }
            ) {
                for output in &node.outputs {
                    let endpoint = format!("{}.{}", node.id, output.name);
                    if !used_outputs.contains(&endpoint) {
                        return Err(ProcessCompileError::Invalid(format!(
                            "internal output `{endpoint}` is disconnected"
                        )));
                    }
                }
            }
        }
        // Cycles are reserved for temporal state recurrence; the only legal
        // state recurrence is internal to Plant/Lag, so the wiring graph must
        // remain a DAG and cannot hide a second execution semantics.
        let mut indegree = BTreeMap::<String, usize>::new();
        let mut adjacency = BTreeMap::<String, Vec<String>>::new();
        for id in nodes.keys() {
            indegree.insert(id.clone(), 0);
        }
        for (from, to) in edges {
            *indegree.get_mut(&to).unwrap() += 1;
            adjacency.entry(from).or_default().push(to);
        }
        let mut queue: Vec<String> = indegree
            .iter()
            .filter(|(_, d)| **d == 0)
            .map(|(id, _)| id.clone())
            .collect();
        let mut visited = 0;
        while let Some(id) = queue.pop() {
            visited += 1;
            for next in adjacency.get(&id).into_iter().flatten() {
                let d = indegree.get_mut(next).unwrap();
                *d -= 1;
                if *d == 0 {
                    queue.push(next.clone());
                }
            }
        }
        if visited != nodes.len() {
            return Err(ProcessCompileError::Cycle);
        }
        let lowered = self.lowered_runtime_private()?;
        if lowered != self.runtime_private {
            return Err(ProcessCompileError::Invalid(
                "runtime_private does not match process graph lowering".into(),
            ));
        }
        Ok(())
    }
}

pub fn compile_process(
    family: ProcessFamily,
    name: impl Into<String>,
    sensor_dim: usize,
    action_dim: usize,
    horizon: usize,
) -> Result<CompiledProcess, ProcessCompileError> {
    let (nodes, wiring) = base_nodes(sensor_dim, action_dim, family, horizon);
    let runtime_private = lower_runtime(&nodes, horizon)?;
    let process = CompiledProcess {
        schema_version: PROCESS_SCHEMA_VERSION,
        family,
        name: name.into(),
        horizon,
        sensor_dim,
        action_dim,
        nodes,
        wiring,
        public_schema: public_schema(sensor_dim, action_dim),
        runtime_private,
    };
    process.validate()?;
    Ok(process)
}

/// Deterministically emits a mixed corpus.  Every family is a structural graph
/// change: lag inserts a stateful actuator node; the composed family also adds
/// timed goal-switch and disturbance nodes and their wires.
pub fn generate_processes(
    seed: u64,
    count: usize,
) -> Result<Vec<GeneratedProcess>, ProcessCompileError> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut generated = Vec::with_capacity(count);
    for index in 0..count {
        // Enumerate the complete legal product of the three optional
        // operators.  The process graph, rather than this label, determines
        // runtime behavior; cycling masks gives every composition support in
        // a bounded export without inventing arbitrary DAG semantics.
        let family = match index % 8 {
            0 => ProcessFamily::Reaching,
            1 => ProcessFamily::ActuatorLag,
            2 => ProcessFamily::GoalSwitch,
            3 => ProcessFamily::Disturbance,
            4 => ProcessFamily::ActuatorLagGoalSwitch,
            5 => ProcessFamily::ActuatorLagDisturbance,
            6 => ProcessFamily::GoalSwitchDisturbance,
            _ => ProcessFamily::LagGoalSwitchDisturbance,
        };
        // Export a balanced support: every legal process composition is
        // emitted for each crossed sensor/body realization.  The seed still
        // controls horizons and names, while dimensions are never missing by
        // chance from a finite launch manifest.
        let realization = (index / 8) % 4;
        let sensor_dim = if realization / 2 == 0 { 8 } else { 10 };
        let action_dim = if realization % 2 == 0 { 2 } else { 4 };
        let horizon = rng.gen_range(12..=24);
        let name = format!("{}-{:04}", family.label(), index);
        let process = compile_process(family, name, sensor_dim, action_dim, horizon)?;
        generated.push(GeneratedProcess {
            process,
            generator_seed: seed,
            generator_index: index,
        });
    }
    Ok(generated)
}
