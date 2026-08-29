//! The calibrated-monomial developmental world family.
//!
//! The learner-visible transcript, public-prefix oracle, latent transition
//! executor, and verifier are kept in one Rust crate so the scientific
//! information boundary is exact and the hot data path remains batched.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeMap;

pub const WORLD_VERSION: &str = "calibrated-monomial-0.2.0";
pub const ORACLE_VERSION: &str = "public-prefix-oracle-0.2.0";
pub const TOKEN_ABI_VERSION: &str = "physical-event-abi-0.2.0";
pub const PAYLOAD_DIM: usize = 8;
pub const ACTION_HORIZON: usize = 16;

const INSTANCE_DOMAIN: u64 = 0x494E_5354_414E_4345;
const TASK_DOMAIN: u64 = 0x5441_534B_5F5F_5F5F;
const CALIBRATION_DOMAIN: u64 = 0x4341_4C49_4252_4154;
const PRESENTATION_DOMAIN: u64 = 0x5052_4553_454E_545F;
const BOUNDARY_CALIBRATION_RESET: f32 = -1.0;
const BOUNDARY_TASK_RESET: f32 = 0.0;
const BOUNDARY_EPISODE_END: f32 = 1.0;
const REQUESTED_ACTION_STEPS: usize = 1;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Pad = 0,
    SchemaObservation = 1,
    SchemaActuator = 2,
    Boundary = 3,
    Condition = 4,
    Goal = 5,
    Observation = 6,
    ActionQuery = 7,
    ActionExecuted = 8,
    FutureQuery = 9,
    Feedback = 10,
}

impl Role {
    pub const COUNT: usize = 11;
}

#[derive(Debug, Clone, PartialEq)]
pub struct PublicToken {
    pub role: Role,
    pub key: u16,
    pub event: u16,
    pub payload: [f32; PAYLOAD_DIM],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Supervision {
    pub action_target: [f32; ACTION_HORIZON],
    pub action_mask: [bool; ACTION_HORIZON],
    pub future_target: f32,
    pub future_mask: bool,
}

impl Default for Supervision {
    fn default() -> Self {
        Self {
            action_target: [0.0; ACTION_HORIZON],
            action_mask: [false; ACTION_HORIZON],
            future_target: 0.0,
            future_mask: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LearningToken {
    pub public: PublicToken,
    pub supervision: Supervision,
}

#[derive(Debug, Clone)]
pub struct FamilyConfig {
    pub d_min: usize,
    pub d_max: usize,
    pub gain_min: f32,
    pub gain_max: f32,
    pub action_limit: f32,
    pub calibration_pulse: f32,
    pub calibration_margin: f32,
    pub task_state_limit: f32,
    pub one_step_total_action: f32,
    pub multi_step_total_action: f32,
    pub max_control_steps: usize,
    pub success_tolerance: f32,
}

impl Default for FamilyConfig {
    fn default() -> Self {
        Self {
            d_min: 1,
            d_max: 4,
            gain_min: 0.75,
            gain_max: 1.25,
            action_limit: 0.20,
            calibration_pulse: 0.10,
            calibration_margin: 0.10,
            task_state_limit: 0.80,
            one_step_total_action: 0.10,
            multi_step_total_action: 0.35,
            max_control_steps: 4,
            success_tolerance: 0.05,
        }
    }
}

impl FamilyConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.d_min == 0 || self.d_min > self.d_max || self.d_max > u16::MAX as usize {
            return Err("dimension bounds must satisfy 1 <= d_min <= d_max <= u16::MAX".into());
        }
        if !(0.0 < self.gain_min && self.gain_min <= self.gain_max) {
            return Err("gain bounds must satisfy 0 < gain_min <= gain_max".into());
        }
        if !(0.0 < self.calibration_pulse && self.calibration_pulse <= self.action_limit) {
            return Err("calibration pulse must be positive and within the action limit".into());
        }
        if self.gain_max * self.calibration_pulse > 1.0 - self.calibration_margin {
            return Err("calibration can clip under the declared gain bound".into());
        }
        if !(self.one_step_total_action <= self.action_limit
            && self.multi_step_total_action > self.action_limit)
        {
            return Err(
                "one-step and multi-step action totals do not straddle action_limit".into(),
            );
        }
        if self.gain_max * self.multi_step_total_action > self.task_state_limit {
            return Err("multi-step goal can exceed the task-state safety support".into());
        }
        if self.max_control_steps
            < (self.multi_step_total_action / self.action_limit).ceil() as usize
        {
            return Err("max_control_steps cannot realize the multi-step support".into());
        }
        if !(0.0 < self.success_tolerance && self.success_tolerance < 0.1) {
            return Err("success_tolerance must be positive and small".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Instance {
    pub d: usize,
    /// `effect_of_actuator[j]` is the observation channel affected by actuator j.
    pub effect_of_actuator: Vec<usize>,
    pub gain_of_actuator: Vec<f32>,
    pub seed: u64,
    pub index: u64,
}

impl Instance {
    pub fn validate(&self, cfg: &FamilyConfig) -> Result<(), String> {
        if self.d < cfg.d_min || self.d > cfg.d_max {
            return Err("instance dimension is outside family support".into());
        }
        if self.effect_of_actuator.len() != self.d || self.gain_of_actuator.len() != self.d {
            return Err("instance vectors do not match dimension".into());
        }
        let mut seen = vec![false; self.d];
        for (&effect, &gain) in self.effect_of_actuator.iter().zip(&self.gain_of_actuator) {
            if effect >= self.d || seen[effect] {
                return Err("effect map is not a permutation".into());
            }
            seen[effect] = true;
            if gain.abs() < cfg.gain_min - 1.0e-6 || gain.abs() > cfg.gain_max + 1.0e-6 {
                return Err("gain is outside declared support".into());
            }
        }
        Ok(())
    }

    pub fn transition(&self, x: &[f32], u: &[f32]) -> Result<Vec<f32>, String> {
        if x.len() != self.d || u.len() != self.d {
            return Err("transition vectors do not match instance dimension".into());
        }
        let mut next = x.to_vec();
        for (j, &command) in u.iter().enumerate() {
            let i = self.effect_of_actuator[j];
            next[i] = (next[i] + self.gain_of_actuator[j] * command).clamp(-1.0, 1.0);
        }
        Ok(next)
    }

    pub fn latent_teacher(&self, x: &[f32], goal: &[f32], action_limit: f32) -> Vec<f32> {
        let mut action = vec![0.0; self.d];
        for (j, output) in action.iter_mut().enumerate() {
            let i = self.effect_of_actuator[j];
            *output =
                ((goal[i] - x[i]) / self.gain_of_actuator[j]).clamp(-action_limit, action_limit);
        }
        action
    }
}

#[derive(Debug, Clone)]
pub struct PublicOracle {
    pub d: usize,
    pub effect_of_actuator: Vec<usize>,
    pub gain_of_actuator: Vec<f32>,
    pub action_limit: f32,
}

impl PublicOracle {
    /// Reconstruct the monomial action-effect law using only serialized public
    /// tokens before the task reset. Supervision fields are not accepted.
    pub fn from_public_prefix(tokens: &[PublicToken]) -> Result<Self, String> {
        let task_event = tokens
            .iter()
            .find(|t| {
                t.role == Role::Boundary && (t.payload[0] - BOUNDARY_TASK_RESET).abs() < 1.0e-6
            })
            .map(|t| t.event)
            .ok_or("public prefix has no task-reset boundary")?;

        let mut observation_keys = Vec::new();
        let mut actuator_keys = Vec::new();
        let mut action_limit = None;
        for token in tokens.iter().filter(|t| t.event < task_event) {
            match token.role {
                Role::SchemaObservation => observation_keys.push(token.key as usize),
                Role::SchemaActuator => {
                    actuator_keys.push(token.key as usize);
                    action_limit = Some(token.payload[2].abs());
                }
                _ => {}
            }
        }
        observation_keys.sort_unstable();
        observation_keys.dedup();
        actuator_keys.sort_unstable();
        actuator_keys.dedup();
        let d = observation_keys.len();
        if d == 0 || actuator_keys.len() != d {
            return Err("public schema does not contain equal nonzero sensor/actuator sets".into());
        }
        if observation_keys != (0..d).collect::<Vec<_>>()
            || actuator_keys != (0..d).collect::<Vec<_>>()
        {
            return Err("encounter-local keys must be contiguous for this world version".into());
        }
        let action_limit = action_limit.ok_or("public schema omits action bounds")?;

        let mut observations: BTreeMap<u16, Vec<Option<f32>>> = BTreeMap::new();
        let mut actions: BTreeMap<u16, Vec<Option<f32>>> = BTreeMap::new();
        for token in tokens.iter().filter(|t| t.event < task_event) {
            match token.role {
                Role::Observation => {
                    let row = observations
                        .entry(token.event)
                        .or_insert_with(|| vec![None; d]);
                    row[token.key as usize] = Some(token.payload[0]);
                }
                Role::ActionExecuted => {
                    let row = actions.entry(token.event).or_insert_with(|| vec![None; d]);
                    row[token.key as usize] = Some(token.payload[0] * action_limit);
                }
                _ => {}
            }
        }

        let mut effect = vec![usize::MAX; d];
        let mut gain = vec![0.0f32; d];
        for (&event, row) in &actions {
            let before = observations
                .range(..event)
                .next_back()
                .ok_or("calibration action has no preceding public observation")?
                .1;
            let after = observations
                .range((event + 1)..task_event)
                .next()
                .ok_or("calibration action has no following public observation")?
                .1;
            let nonzero: Vec<(usize, f32)> = row
                .iter()
                .enumerate()
                .filter_map(|(j, value)| value.and_then(|u| (u.abs() > 1.0e-7).then_some((j, u))))
                .collect();
            if nonzero.len() != 1 {
                return Err("each calibration action must pulse exactly one actuator".into());
            }
            let (j, u) = nonzero[0];
            let mut changed = Vec::new();
            for i in 0..d {
                let delta = after[i].ok_or("incomplete post-calibration observation")?
                    - before[i].ok_or("incomplete pre-calibration observation")?;
                if delta.abs() > 1.0e-7 {
                    changed.push((i, delta));
                }
            }
            if changed.len() != 1 {
                return Err("calibration pulse must change exactly one observation channel".into());
            }
            if effect[j] != usize::MAX {
                return Err("actuator was calibrated more than once".into());
            }
            effect[j] = changed[0].0;
            gain[j] = changed[0].1 / u;
        }
        if effect.contains(&usize::MAX) {
            return Err("public calibration does not cover every actuator".into());
        }
        let mut seen = vec![false; d];
        for &i in &effect {
            if seen[i] {
                return Err("reconstructed effect law is not monomial".into());
            }
            seen[i] = true;
        }
        Ok(Self {
            d,
            effect_of_actuator: effect,
            gain_of_actuator: gain,
            action_limit,
        })
    }

    pub fn action(&self, x: &[f32], goal: &[f32]) -> Result<Vec<f32>, String> {
        if x.len() != self.d || goal.len() != self.d {
            return Err("oracle input does not match public schema".into());
        }
        let mut action = vec![0.0; self.d];
        for (j, output) in action.iter_mut().enumerate() {
            let i = self.effect_of_actuator[j];
            *output = ((goal[i] - x[i]) / self.gain_of_actuator[j])
                .clamp(-self.action_limit, self.action_limit);
        }
        Ok(action)
    }
}

#[derive(Debug, Clone)]
pub struct Trajectory {
    pub tokens: Vec<LearningToken>,
    pub d: usize,
    pub seed: u64,
    pub index: u64,
    pub control_steps: usize,
    pub oracle_reconstruction_error: f32,
}

#[derive(Debug, Clone)]
struct Serializer {
    tokens: Vec<LearningToken>,
    next_event: u16,
}

impl Serializer {
    fn new() -> Self {
        Self {
            tokens: Vec::new(),
            next_event: 0,
        }
    }

    fn segment(&mut self, entries: Vec<(Role, u16, [f32; PAYLOAD_DIM], Supervision)>) {
        let event = self.next_event;
        self.next_event = self
            .next_event
            .checked_add(1)
            .expect("event counter overflow");
        for (role, key, payload, supervision) in entries {
            self.tokens.push(LearningToken {
                public: PublicToken {
                    role,
                    key,
                    event,
                    payload,
                },
                supervision,
            });
        }
    }

    fn public(&self) -> Vec<PublicToken> {
        self.tokens.iter().map(|t| t.public.clone()).collect()
    }
}

fn payload(value: f32, lower: f32, upper: f32, aux0: f32, aux1: f32) -> [f32; PAYLOAD_DIM] {
    [value, lower, upper, aux0, aux1, 1.0, 0.0, 0.0]
}

fn boundary_payload(kind: f32) -> [f32; PAYLOAD_DIM] {
    payload(kind, -1.0, 1.0, 0.0, 0.0)
}

fn action_payload(value: f32, cfg: &FamilyConfig) -> [f32; PAYLOAD_DIM] {
    payload(
        value,
        -cfg.action_limit,
        cfg.action_limit,
        1.0,
        REQUESTED_ACTION_STEPS as f32 / ACTION_HORIZON as f32,
    )
}

fn shuffled_keys(d: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let mut keys: Vec<usize> = (0..d).collect();
    keys.shuffle(rng);
    keys
}

fn mix_seed(seed: u64, index: u64) -> u64 {
    let mut z = seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn domain_rng(seed: u64, index: u64, domain: u64, salt: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(mix_seed(seed ^ domain ^ salt.rotate_left(17), index))
}

pub fn sample_instance(cfg: &FamilyConfig, seed: u64, index: u64) -> Result<Instance, String> {
    cfg.validate()?;
    let span = cfg.d_max - cfg.d_min + 1;
    let d = cfg.d_min + index as usize % span;
    let mut rng = domain_rng(seed, index, INSTANCE_DOMAIN, 0);
    let mut effect_of_actuator: Vec<usize> = (0..d).collect();
    effect_of_actuator.shuffle(&mut rng);
    let gain_of_actuator = (0..d)
        .map(|_| {
            let magnitude = rng.gen_range(cfg.gain_min..=cfg.gain_max);
            let sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
            sign * magnitude
        })
        .collect();
    let instance = Instance {
        d,
        effect_of_actuator,
        gain_of_actuator,
        seed,
        index,
    };
    instance.validate(cfg)?;
    Ok(instance)
}

fn append_schema(
    serializer: &mut Serializer,
    instance: &Instance,
    cfg: &FamilyConfig,
    rng: &mut ChaCha8Rng,
) {
    let mut entries = Vec::with_capacity(2 * instance.d);
    let mut tagged: Vec<(Role, usize)> = (0..instance.d)
        .map(|i| (Role::SchemaObservation, i))
        .chain((0..instance.d).map(|j| (Role::SchemaActuator, j)))
        .collect();
    tagged.shuffle(rng);
    for (role, key) in tagged {
        let p = match role {
            Role::SchemaObservation => payload(0.0, -1.0, 1.0, 0.0, 0.0),
            Role::SchemaActuator => action_payload(0.0, cfg),
            _ => unreachable!(),
        };
        entries.push((role, key as u16, p, Supervision::default()));
    }
    serializer.segment(entries);
}

fn append_calibration(
    serializer: &mut Serializer,
    instance: &Instance,
    cfg: &FamilyConfig,
    presentation_rng: &mut ChaCha8Rng,
    calibration_rng: &mut ChaCha8Rng,
) -> Result<(), String> {
    let calibration_order = shuffled_keys(instance.d, presentation_rng);
    for &pulse_actuator in &calibration_order {
        serializer.segment(vec![(
            Role::Boundary,
            0,
            boundary_payload(BOUNDARY_CALIBRATION_RESET),
            Supervision::default(),
        )]);
        let before = vec![0.0; instance.d];
        append_observation(serializer, &before, presentation_rng);
        let mut action = vec![0.0; instance.d];
        let pulse_sign = if calibration_rng.gen_bool(0.5) {
            1.0
        } else {
            -1.0
        };
        action[pulse_actuator] = pulse_sign * cfg.calibration_pulse;
        let action_order = shuffled_keys(instance.d, presentation_rng);
        append_executed_action(serializer, &action, cfg, &action_order);
        let after = instance.transition(&before, &action)?;
        let observation_order = shuffled_keys(instance.d, presentation_rng);
        append_future_queries(serializer, &after, &observation_order, 1, true);
        append_observation(serializer, &after, presentation_rng);
    }
    Ok(())
}

fn sample_task(
    instance: &Instance,
    cfg: &FamilyConfig,
    rng: &mut ChaCha8Rng,
) -> (Vec<f32>, Vec<f32>) {
    let mut start = vec![0.0; instance.d];
    for value in &mut start {
        *value = rng.gen_range(-0.10..=0.10);
    }
    let mut total_action = vec![0.0; instance.d];
    for value in &mut total_action {
        let magnitude = if rng.gen_bool(1.0 / 3.0) {
            cfg.one_step_total_action
        } else {
            cfg.multi_step_total_action
        };
        let sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        *value = sign * magnitude;
    }
    let goal = instance
        .transition(&start, &total_action)
        .expect("task vectors match instance dimension");
    debug_assert!(goal.iter().all(|v| v.abs() <= cfg.task_state_limit + 0.11));
    (start, goal)
}

type PublicPrefixBuild = (Serializer, PublicOracle, Vec<f32>, Vec<f32>, ChaCha8Rng);

fn build_public_prefix(
    instance: &Instance,
    cfg: &FamilyConfig,
    presentation_salt: u64,
) -> Result<PublicPrefixBuild, String> {
    // Semantic and presentation streams are domain-separated. In particular,
    // changing a legal serialization order cannot change the task itself.
    let mut task_rng = domain_rng(instance.seed, instance.index, TASK_DOMAIN, 0);
    let (start, goal) = sample_task(instance, cfg, &mut task_rng);
    let mut calibration_rng = domain_rng(instance.seed, instance.index, CALIBRATION_DOMAIN, 0);
    let mut presentation_rng = domain_rng(
        instance.seed,
        instance.index,
        PRESENTATION_DOMAIN,
        presentation_salt,
    );
    let mut serializer = Serializer::new();
    append_schema(&mut serializer, instance, cfg, &mut presentation_rng);
    append_calibration(
        &mut serializer,
        instance,
        cfg,
        &mut presentation_rng,
        &mut calibration_rng,
    )?;
    serializer.segment(vec![(
        Role::Boundary,
        0,
        boundary_payload(BOUNDARY_TASK_RESET),
        Supervision::default(),
    )]);
    let oracle = PublicOracle::from_public_prefix(&serializer.public())?;
    let condition_entries = shuffled_keys(instance.d, &mut presentation_rng)
        .into_iter()
        .map(|i| {
            (
                Role::Goal,
                i as u16,
                payload(goal[i], -1.0, 1.0, 0.0, 0.0),
                Supervision::default(),
            )
        })
        .collect();
    serializer.segment(condition_entries);
    append_observation(&mut serializer, &start, &mut presentation_rng);
    Ok((serializer, oracle, start, goal, presentation_rng))
}

pub fn generate_trajectory(
    cfg: &FamilyConfig,
    seed: u64,
    index: u64,
) -> Result<Trajectory, String> {
    let instance = sample_instance(cfg, seed, index)?;
    let (mut serializer, oracle, mut x, goal, mut rng) = build_public_prefix(&instance, cfg, 0)?;
    let mut max_oracle_error = 0.0f32;
    for j in 0..instance.d {
        max_oracle_error =
            max_oracle_error.max((oracle.gain_of_actuator[j] - instance.gain_of_actuator[j]).abs());
        if oracle.effect_of_actuator[j] != instance.effect_of_actuator[j] {
            return Err("public oracle reconstructed the wrong actuator permutation".into());
        }
    }

    let mut control_steps = 0usize;
    loop {
        let oracle_action = oracle.action(&x, &goal)?;
        let latent_action = instance.latent_teacher(&x, &goal, cfg.action_limit);
        for (a, b) in oracle_action.iter().zip(&latent_action) {
            max_oracle_error = max_oracle_error.max((a - b).abs());
        }
        let action_order = shuffled_keys(instance.d, &mut rng);
        append_action_queries(&mut serializer, Some(&oracle_action), cfg, &action_order);
        append_executed_action(&mut serializer, &oracle_action, cfg, &action_order);
        let next = instance.transition(&x, &oracle_action)?;
        let observation_order = shuffled_keys(instance.d, &mut rng);
        append_future_queries(&mut serializer, &next, &observation_order, 1, true);
        let success = append_feedback(&mut serializer, &next, &goal, cfg.success_tolerance);
        append_observation(&mut serializer, &next, &mut rng);
        x = next;
        control_steps += 1;
        if success || control_steps >= cfg.max_control_steps {
            serializer.segment(vec![(
                Role::Boundary,
                0,
                boundary_payload(BOUNDARY_EPISODE_END),
                Supervision::default(),
            )]);
            if !success {
                return Err(
                    "public teacher failed to solve an admitted task within max_control_steps"
                        .into(),
                );
            }
            break;
        }
    }

    Ok(Trajectory {
        tokens: serializer.tokens,
        d: instance.d,
        seed,
        index,
        control_steps,
        oracle_reconstruction_error: max_oracle_error,
    })
}

#[derive(Debug, Clone)]
pub struct RolloutEpisode {
    pub instance: Instance,
    pub oracle: PublicOracle,
    pub tokens: Vec<LearningToken>,
    pub x: Vec<f32>,
    pub goal: Vec<f32>,
    pub query_order: Vec<usize>,
    pub steps: usize,
    pub done: bool,
    pub success: bool,
    rng: ChaCha8Rng,
}

impl RolloutEpisode {
    pub fn new(cfg: &FamilyConfig, seed: u64, index: u64) -> Result<Self, String> {
        let instance = sample_instance(cfg, seed, index)?;
        let (mut serializer, oracle, x, goal, mut rng) = build_public_prefix(&instance, cfg, 0)?;
        let query_order = shuffled_keys(instance.d, &mut rng);
        append_action_queries(&mut serializer, None, cfg, &query_order);
        Ok(Self {
            instance,
            oracle,
            tokens: serializer.tokens,
            x,
            goal,
            query_order,
            steps: 0,
            done: false,
            success: false,
            rng,
        })
    }

    pub fn current_query_positions(&self) -> Vec<usize> {
        let event = self
            .tokens
            .iter()
            .rev()
            .find(|t| t.public.role == Role::ActionQuery)
            .map(|t| t.public.event)
            .expect("rollout always ends in action queries while live");
        self.tokens
            .iter()
            .enumerate()
            .filter_map(|(i, t)| {
                (t.public.role == Role::ActionQuery && t.public.event == event).then_some(i)
            })
            .collect()
    }

    pub fn step_normalized(
        &mut self,
        cfg: &FamilyConfig,
        actions_in_query_order: &[f32],
    ) -> Result<(), String> {
        if self.done {
            return Err("cannot step a completed rollout".into());
        }
        if actions_in_query_order.len() != self.instance.d {
            return Err("predicted action count does not match current query set".into());
        }
        let mut action = vec![0.0; self.instance.d];
        for (&key, &normalized) in self.query_order.iter().zip(actions_in_query_order) {
            action[key] = normalized.clamp(-1.0, 1.0) * cfg.action_limit;
        }
        append_executed_action(
            &mut SerializerProxy::new(&mut self.tokens),
            &action,
            cfg,
            &self.query_order,
        );
        let next = self.instance.transition(&self.x, &action)?;
        let future_order = shuffled_keys(self.instance.d, &mut self.rng);
        append_future_queries(
            &mut SerializerProxy::new(&mut self.tokens),
            &next,
            &future_order,
            1,
            false,
        );
        self.success = append_feedback(
            &mut SerializerProxy::new(&mut self.tokens),
            &next,
            &self.goal,
            cfg.success_tolerance,
        );
        append_observation(
            &mut SerializerProxy::new(&mut self.tokens),
            &next,
            &mut self.rng,
        );
        self.x = next;
        self.steps += 1;
        self.done = self.success || self.steps >= cfg.max_control_steps;
        if self.done {
            SerializerProxy::new(&mut self.tokens).segment(vec![(
                Role::Boundary,
                0,
                boundary_payload(BOUNDARY_EPISODE_END),
                Supervision::default(),
            )]);
        } else {
            self.query_order = shuffled_keys(self.instance.d, &mut self.rng);
            append_action_queries(
                &mut SerializerProxy::new(&mut self.tokens),
                None,
                cfg,
                &self.query_order,
            );
        }
        Ok(())
    }

    pub fn terminal_error(&self) -> f32 {
        self.x
            .iter()
            .zip(&self.goal)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }
}

/// Adapter that appends segments to an existing token vector while preserving
/// monotonically increasing event IDs. It lets online rollouts reuse the exact
/// serializer functions used by static teacher trajectories.
struct SerializerProxy<'a> {
    tokens: &'a mut Vec<LearningToken>,
    next_event: u16,
}

impl<'a> SerializerProxy<'a> {
    fn new(tokens: &'a mut Vec<LearningToken>) -> Self {
        let next_event = tokens.last().map(|t| t.public.event + 1).unwrap_or(0);
        Self { tokens, next_event }
    }

    fn segment(&mut self, entries: Vec<(Role, u16, [f32; PAYLOAD_DIM], Supervision)>) {
        let event = self.next_event;
        for (role, key, payload, supervision) in entries {
            self.tokens.push(LearningToken {
                public: PublicToken {
                    role,
                    key,
                    event,
                    payload,
                },
                supervision,
            });
        }
        self.next_event += 1;
    }
}

trait SegmentSink {
    fn segment(&mut self, entries: Vec<(Role, u16, [f32; PAYLOAD_DIM], Supervision)>);
}

impl SegmentSink for Serializer {
    fn segment(&mut self, entries: Vec<(Role, u16, [f32; PAYLOAD_DIM], Supervision)>) {
        Serializer::segment(self, entries)
    }
}

impl SegmentSink for SerializerProxy<'_> {
    fn segment(&mut self, entries: Vec<(Role, u16, [f32; PAYLOAD_DIM], Supervision)>) {
        SerializerProxy::segment(self, entries)
    }
}

// Generic wrappers allow the static and online serializers to share exact
// event construction without exposing hidden dynamics to Python.
fn append_executed_action<S: SegmentSink>(
    serializer: &mut S,
    u: &[f32],
    cfg: &FamilyConfig,
    order: &[usize],
) {
    let entries = order
        .iter()
        .map(|&j| {
            (
                Role::ActionExecuted,
                j as u16,
                action_payload(u[j] / cfg.action_limit, cfg),
                Supervision::default(),
            )
        })
        .collect();
    serializer.segment(entries);
}

fn append_future_queries<S: SegmentSink>(
    serializer: &mut S,
    next: &[f32],
    order: &[usize],
    horizon: usize,
    supervised: bool,
) {
    assert!((1..=ACTION_HORIZON).contains(&horizon));
    let entries = order
        .iter()
        .map(|&i| {
            let mut supervision = Supervision::default();
            if supervised {
                supervision.future_target = next[i];
                supervision.future_mask = true;
            }
            (
                Role::FutureQuery,
                i as u16,
                payload(0.0, -1.0, 1.0, horizon as f32 / ACTION_HORIZON as f32, 0.0),
                supervision,
            )
        })
        .collect();
    serializer.segment(entries);
}

fn append_action_queries<S: SegmentSink>(
    serializer: &mut S,
    oracle_action: Option<&[f32]>,
    cfg: &FamilyConfig,
    order: &[usize],
) {
    let entries = order
        .iter()
        .map(|&j| {
            let mut supervision = Supervision::default();
            if let Some(action) = oracle_action {
                supervision.action_target[0] = action[j] / cfg.action_limit;
                supervision.action_mask[0] = true;
            }
            (
                Role::ActionQuery,
                j as u16,
                action_payload(0.0, cfg),
                supervision,
            )
        })
        .collect();
    serializer.segment(entries);
}

fn append_feedback<S: SegmentSink>(
    serializer: &mut S,
    x: &[f32],
    goal: &[f32],
    tolerance: f32,
) -> bool {
    let error = x
        .iter()
        .zip(goal)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let success = error <= tolerance;
    serializer.segment(vec![(
        Role::Feedback,
        0,
        payload(error / 2.0, 0.0, 1.0, if success { 1.0 } else { 0.0 }, 0.0),
        Supervision::default(),
    )]);
    success
}

fn append_observation<S: SegmentSink>(serializer: &mut S, x: &[f32], rng: &mut ChaCha8Rng) {
    let entries = shuffled_keys(x.len(), rng)
        .into_iter()
        .map(|i| {
            (
                Role::Observation,
                i as u16,
                payload(x[i], -1.0, 1.0, 0.0, 0.0),
                Supervision::default(),
            )
        })
        .collect();
    serializer.segment(entries);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_unsafe_calibration() {
        let cfg = FamilyConfig {
            calibration_pulse: 1.0,
            ..FamilyConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn dimensions_are_stratified() {
        let cfg = FamilyConfig::default();
        let dimensions: Vec<usize> = (0..8)
            .map(|i| sample_instance(&cfg, 7, i).unwrap().d)
            .collect();
        assert_eq!(dimensions, vec![1, 2, 3, 4, 1, 2, 3, 4]);
    }

    #[test]
    fn generation_is_replay_deterministic() {
        let cfg = FamilyConfig::default();
        let a = generate_trajectory(&cfg, 19, 11).unwrap();
        let b = generate_trajectory(&cfg, 19, 11).unwrap();
        assert_eq!(a.tokens, b.tokens);
        assert_eq!(a.control_steps, b.control_steps);
    }

    #[test]
    fn public_oracle_matches_hidden_dynamics() {
        let cfg = FamilyConfig::default();
        for index in 0..128 {
            let trajectory = generate_trajectory(&cfg, 23, index).unwrap();
            assert!(trajectory.oracle_reconstruction_error < 1.0e-5);
            assert!(trajectory.control_steps >= 1);
            assert!(trajectory.control_steps <= cfg.max_control_steps);
        }
    }

    #[test]
    fn calibration_never_clips() {
        let cfg = FamilyConfig::default();
        for index in 0..128 {
            let instance = sample_instance(&cfg, 29, index).unwrap();
            for j in 0..instance.d {
                let mut u = vec![0.0; instance.d];
                u[j] = cfg.calibration_pulse;
                let next = instance.transition(&vec![0.0; instance.d], &u).unwrap();
                assert!(next.iter().all(|x| x.abs() <= 1.0 - cfg.calibration_margin));
            }
        }
    }

    #[test]
    fn no_target_is_serialized_in_action_query_payload() {
        let cfg = FamilyConfig::default();
        let trajectory = generate_trajectory(&cfg, 31, 3).unwrap();
        for token in trajectory
            .tokens
            .iter()
            .filter(|t| t.public.role == Role::ActionQuery)
        {
            assert_eq!(token.public.payload[0], 0.0);
            assert!(token.supervision.action_mask[0]);
        }
    }

    #[test]
    fn goal_and_future_query_have_distinct_explicit_roles() {
        let cfg = FamilyConfig::default();
        let trajectory = generate_trajectory(&cfg, 33, 5).unwrap();
        assert!(trajectory
            .tokens
            .iter()
            .any(|token| token.public.role == Role::Goal));
        assert!(!trajectory
            .tokens
            .iter()
            .any(|token| token.public.role == Role::Condition));
        for token in trajectory
            .tokens
            .iter()
            .filter(|token| token.public.role == Role::FutureQuery)
        {
            assert!((token.public.payload[3] - 1.0 / ACTION_HORIZON as f32).abs() < 1.0e-7);
        }
    }

    #[test]
    fn presentation_changes_do_not_change_task_or_reconstructed_dynamics() {
        let cfg = FamilyConfig::default();
        for index in 0..64 {
            let instance = sample_instance(&cfg, 35, index).unwrap();
            let (_, oracle_a, start_a, goal_a, _) =
                build_public_prefix(&instance, &cfg, 0).unwrap();
            let (_, oracle_b, start_b, goal_b, _) =
                build_public_prefix(&instance, &cfg, 1).unwrap();
            assert_eq!(start_a, start_b);
            assert_eq!(goal_a, goal_b);
            assert_eq!(oracle_a.effect_of_actuator, oracle_b.effect_of_actuator);
            assert_eq!(oracle_a.gain_of_actuator, oracle_b.gain_of_actuator);
        }
    }

    #[test]
    fn static_oracle_trajectory_matches_online_public_rollout_exactly() {
        let cfg = FamilyConfig::default();
        for index in 0..32 {
            let static_trajectory = generate_trajectory(&cfg, 39, index).unwrap();
            let mut online = RolloutEpisode::new(&cfg, 39, index).unwrap();
            while !online.done {
                let action = online.oracle.action(&online.x, &online.goal).unwrap();
                let normalized_by_query: Vec<f32> = online
                    .query_order
                    .iter()
                    .map(|&j| action[j] / cfg.action_limit)
                    .collect();
                online.step_normalized(&cfg, &normalized_by_query).unwrap();
            }
            let static_public: Vec<&PublicToken> = static_trajectory
                .tokens
                .iter()
                .map(|token| &token.public)
                .collect();
            let online_public: Vec<&PublicToken> =
                online.tokens.iter().map(|token| &token.public).collect();
            assert_eq!(static_public, online_public);
        }
    }

    #[test]
    fn rollout_uses_model_actions_and_stays_bounded() {
        let cfg = FamilyConfig::default();
        let mut rollout = RolloutEpisode::new(&cfg, 37, 3).unwrap();
        while !rollout.done {
            let zeros = vec![0.0; rollout.instance.d];
            rollout.step_normalized(&cfg, &zeros).unwrap();
        }
        assert!(!rollout.success);
        assert_eq!(rollout.steps, cfg.max_control_steps);
        assert!(rollout.x.iter().all(|v| (-1.0..=1.0).contains(v)));
    }

    #[test]
    fn public_oracle_solves_online_rollout() {
        let cfg = FamilyConfig::default();
        let mut rollout = RolloutEpisode::new(&cfg, 41, 7).unwrap();
        while !rollout.done {
            let action = rollout.oracle.action(&rollout.x, &rollout.goal).unwrap();
            let normalized_by_query: Vec<f32> = rollout
                .query_order
                .iter()
                .map(|&j| action[j] / cfg.action_limit)
                .collect();
            rollout.step_normalized(&cfg, &normalized_by_query).unwrap();
        }
        assert!(rollout.success);
        assert!(rollout.terminal_error() <= cfg.success_tolerance);
    }
}
