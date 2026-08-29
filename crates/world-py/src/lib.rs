//! Thin batched Python boundary for the Rust worlds.
//!
//! Learner tensors contain only public event fields. Methods prefixed
//! `privileged_` are validator/evaluator surfaces and must never be passed to
//! the model as inputs.

use pretraining_eviction_world::{
    optimal_first_commands, standard_eviction_rollouts, transition as eviction_transition, Command,
    EvictionRollout, SerializationOrder as EvictionSerializationOrder, PROCESS_VERSION,
};
use pretraining_goal_conditioned_world::{
    classify_progress, standard_diagnostic_rollouts, teacher_training_records, CheckpointEvidence,
    DiagnosticRollout, ProgressThresholds, SerializationOrder, TrainingPresentationArm,
    DIAGNOSTIC_SERIALIZATION_VERSION,
};
use pretraining_profiled_event::{
    tag_legacy_episode, InterpretationProfile, PROFILED_TOKEN_ABI_VERSION,
};
use pretraining_world::{
    generate_trajectory, FamilyConfig, LearningToken, Role, RolloutEpisode, ACTION_HORIZON,
    ORACLE_VERSION, PAYLOAD_DIM, TOKEN_ABI_VERSION, WORLD_VERSION,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

fn parse_serialization_order(value: &str) -> PyResult<SerializationOrder> {
    match value {
        "canonical" => Ok(SerializationOrder::Canonical),
        "permuted" => Ok(SerializationOrder::Permuted),
        _ => Err(PyValueError::new_err(
            "serialization_order must be 'canonical' or 'permuted'",
        )),
    }
}

fn parse_eviction_serialization_order(value: &str) -> PyResult<EvictionSerializationOrder> {
    match value {
        "canonical" => Ok(EvictionSerializationOrder::Canonical),
        "permuted" => Ok(EvictionSerializationOrder::Permuted),
        _ => Err(PyValueError::new_err(
            "serialization_order must be 'canonical' or 'permuted'",
        )),
    }
}

fn parse_training_arm(value: &str) -> PyResult<TrainingPresentationArm> {
    match value {
        "fixed" => Ok(TrainingPresentationArm::Fixed),
        "orbit" => Ok(TrainingPresentationArm::Orbit),
        _ => Err(PyValueError::new_err("arm must be 'fixed' or 'orbit'")),
    }
}

fn py_config(
    d_min: usize,
    d_max: usize,
    gain_min: f32,
    gain_max: f32,
    action_limit: f32,
    calibration_pulse: f32,
    max_control_steps: usize,
) -> PyResult<FamilyConfig> {
    let cfg = FamilyConfig {
        d_min,
        d_max,
        gain_min,
        gain_max,
        action_limit,
        calibration_pulse,
        max_control_steps,
        ..FamilyConfig::default()
    };
    cfg.validate().map_err(PyValueError::new_err)?;
    Ok(cfg)
}

#[derive(Default)]
struct PaddedBatch {
    role_ids: Vec<Vec<u8>>,
    key_ids: Vec<Vec<i64>>,
    position_ids: Vec<Vec<i64>>,
    payloads: Vec<Vec<Vec<f32>>>,
    attention_mask: Vec<Vec<i64>>,
    action_targets: Vec<Vec<Vec<f32>>>,
    action_target_mask: Vec<Vec<Vec<f32>>>,
    future_targets: Vec<Vec<f32>>,
    future_target_mask: Vec<Vec<f32>>,
    lengths: Vec<usize>,
}

fn pad_records(records: &[&[LearningToken]], max_tokens: usize) -> Result<PaddedBatch, String> {
    let mut batch = PaddedBatch::default();
    for record in records {
        if record.len() > max_tokens {
            return Err(format!(
                "trajectory length {} exceeds max_tokens {}; truncation is forbidden",
                record.len(),
                max_tokens
            ));
        }
        let mut roles = vec![Role::Pad as u8; max_tokens];
        let mut keys = vec![0i64; max_tokens];
        let mut positions = vec![0i64; max_tokens];
        let mut payloads = vec![vec![0.0f32; PAYLOAD_DIM]; max_tokens];
        let mut attention = vec![0i64; max_tokens];
        let mut action_targets = vec![vec![0.0f32; ACTION_HORIZON]; max_tokens];
        let mut action_mask = vec![vec![0.0f32; ACTION_HORIZON]; max_tokens];
        let mut future_targets = vec![0.0f32; max_tokens];
        let mut future_mask = vec![0.0f32; max_tokens];
        for (position, token) in record.iter().enumerate() {
            roles[position] = token.public.role as u8;
            keys[position] = token.public.key as i64;
            positions[position] = token.public.event as i64;
            payloads[position].copy_from_slice(&token.public.payload);
            attention[position] = 1;
            action_targets[position].copy_from_slice(&token.supervision.action_target);
            for h in 0..ACTION_HORIZON {
                action_mask[position][h] = if token.supervision.action_mask[h] {
                    1.0
                } else {
                    0.0
                };
            }
            future_targets[position] = token.supervision.future_target;
            future_mask[position] = if token.supervision.future_mask {
                1.0
            } else {
                0.0
            };
        }
        batch.role_ids.push(roles);
        batch.key_ids.push(keys);
        batch.position_ids.push(positions);
        batch.payloads.push(payloads);
        batch.attention_mask.push(attention);
        batch.action_targets.push(action_targets);
        batch.action_target_mask.push(action_mask);
        batch.future_targets.push(future_targets);
        batch.future_target_mask.push(future_mask);
        batch.lengths.push(record.len());
    }
    Ok(batch)
}

fn pad_records_with_optional_profile(
    records: &[&[LearningToken]],
    max_tokens: usize,
    profile: Option<InterpretationProfile>,
) -> Result<PaddedBatch, String> {
    let Some(profile) = profile else {
        return pad_records(records, max_tokens);
    };
    let tagged = records
        .iter()
        .map(|record| tag_legacy_episode(profile, record).map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let borrowed: Vec<&[LearningToken]> = tagged.iter().map(Vec::as_slice).collect();
    pad_records(&borrowed, max_tokens)
}

fn interpretation_profile(
    profiled: bool,
    profile: InterpretationProfile,
) -> Option<InterpretationProfile> {
    profiled.then_some(profile)
}

fn token_abi_for(profiled: bool) -> &'static str {
    if profiled {
        PROFILED_TOKEN_ABI_VERSION
    } else {
        TOKEN_ABI_VERSION
    }
}

fn profile_offset(profiled: bool) -> usize {
    usize::from(profiled)
}

fn padded_to_dict<'py>(py: Python<'py>, batch: PaddedBatch) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    output.set_item("role_ids", batch.role_ids)?;
    output.set_item("key_ids", batch.key_ids)?;
    output.set_item("position_ids", batch.position_ids)?;
    output.set_item("payloads", batch.payloads)?;
    output.set_item("attention_mask", batch.attention_mask)?;
    output.set_item("action_targets", batch.action_targets)?;
    output.set_item("action_target_mask", batch.action_target_mask)?;
    output.set_item("future_targets", batch.future_targets)?;
    output.set_item("future_target_mask", batch.future_target_mask)?;
    output.set_item("lengths", batch.lengths)?;
    Ok(output)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    *, seed, start_index, batch_size, max_tokens,
    d_min=1, d_max=4, gain_min=0.75, gain_max=1.25,
    action_limit=0.20, calibration_pulse=0.10, max_control_steps=4,
    profiled=false
))]
fn generate_training_batch(
    py: Python<'_>,
    seed: u64,
    start_index: u64,
    batch_size: usize,
    max_tokens: usize,
    d_min: usize,
    d_max: usize,
    gain_min: f32,
    gain_max: f32,
    action_limit: f32,
    calibration_pulse: f32,
    max_control_steps: usize,
    profiled: bool,
) -> PyResult<Py<PyAny>> {
    if batch_size == 0 || max_tokens == 0 {
        return Err(PyValueError::new_err(
            "batch_size and max_tokens must be nonzero",
        ));
    }
    let cfg = py_config(
        d_min,
        d_max,
        gain_min,
        gain_max,
        action_limit,
        calibration_pulse,
        max_control_steps,
    )?;
    let trajectories = (0..batch_size)
        .map(|offset| generate_trajectory(&cfg, seed, start_index + offset as u64))
        .collect::<Result<Vec<_>, _>>()
        .map_err(PyValueError::new_err)?;
    let records: Vec<&[LearningToken]> = trajectories.iter().map(|t| t.tokens.as_slice()).collect();
    let padded = pad_records_with_optional_profile(
        &records,
        max_tokens,
        interpretation_profile(
            profiled,
            InterpretationProfile::ChannelValuesWithRequestedSpan,
        ),
    )
    .map_err(PyValueError::new_err)?;
    let output = padded_to_dict(py, padded)?;
    output.set_item(
        "dimensions",
        trajectories.iter().map(|t| t.d).collect::<Vec<_>>(),
    )?;
    output.set_item(
        "indices",
        trajectories.iter().map(|t| t.index).collect::<Vec<_>>(),
    )?;
    output.set_item(
        "control_steps",
        trajectories
            .iter()
            .map(|t| t.control_steps)
            .collect::<Vec<_>>(),
    )?;
    output.set_item("world_version", WORLD_VERSION)?;
    output.set_item("oracle_version", ORACLE_VERSION)?;
    output.set_item("token_abi_version", token_abi_for(profiled))?;
    output.set_item(
        "interpretation_profile",
        profiled.then(|| InterpretationProfile::ChannelValuesWithRequestedSpan.as_str()),
    )?;
    Ok(output.into_any().unbind())
}

#[pyfunction]
#[pyo3(signature = (*, arm, max_tokens, profiled=false))]
fn generate_goal_conditioning_training_batch(
    py: Python<'_>,
    arm: &str,
    max_tokens: usize,
    profiled: bool,
) -> PyResult<Py<PyAny>> {
    if max_tokens == 0 {
        return Err(PyValueError::new_err("max_tokens must be nonzero"));
    }
    let arm = parse_training_arm(arm)?;
    let records = teacher_training_records(arm);
    let borrowed: Vec<&[LearningToken]> = records.iter().map(Vec::as_slice).collect();
    let padded = pad_records_with_optional_profile(
        &borrowed,
        max_tokens,
        interpretation_profile(
            profiled,
            InterpretationProfile::KeySelectionsWithRemainingHorizon,
        ),
    )
    .map_err(PyValueError::new_err)?;
    let output = padded_to_dict(py, padded)?;
    output.set_item("arm", arm.as_str())?;
    output.set_item("semantic_cases", records.len() / 2)?;
    output.set_item("records", records.len())?;
    output.set_item(
        "diagnostic_serialization_version",
        DIAGNOSTIC_SERIALIZATION_VERSION,
    )?;
    output.set_item("token_abi_version", token_abi_for(profiled))?;
    output.set_item(
        "interpretation_profile",
        profiled.then(|| InterpretationProfile::KeySelectionsWithRemainingHorizon.as_str()),
    )?;
    Ok(output.into_any().unbind())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    *, seed, start_index, count,
    d_min=1, d_max=4, gain_min=0.75, gain_max=1.25,
    action_limit=0.20, calibration_pulse=0.10, max_control_steps=4
))]
fn validate_generated_worlds(
    py: Python<'_>,
    seed: u64,
    start_index: u64,
    count: usize,
    d_min: usize,
    d_max: usize,
    gain_min: f32,
    gain_max: f32,
    action_limit: f32,
    calibration_pulse: f32,
    max_control_steps: usize,
) -> PyResult<Py<PyAny>> {
    let cfg = py_config(
        d_min,
        d_max,
        gain_min,
        gain_max,
        action_limit,
        calibration_pulse,
        max_control_steps,
    )?;
    if count == 0 {
        return Err(PyValueError::new_err("count must be nonzero"));
    }
    let mut dimension_counts = vec![0usize; d_max + 1];
    let mut max_length = 0usize;
    let mut min_length = usize::MAX;
    let mut max_oracle_error = 0.0f32;
    let mut action_targets = 0usize;
    let mut future_targets = 0usize;
    for offset in 0..count {
        let trajectory = generate_trajectory(&cfg, seed, start_index + offset as u64)
            .map_err(PyValueError::new_err)?;
        dimension_counts[trajectory.d] += 1;
        max_length = max_length.max(trajectory.tokens.len());
        min_length = min_length.min(trajectory.tokens.len());
        max_oracle_error = max_oracle_error.max(trajectory.oracle_reconstruction_error);
        for token in &trajectory.tokens {
            action_targets += token.supervision.action_mask.iter().filter(|&&v| v).count();
            future_targets += usize::from(token.supervision.future_mask);
        }
    }
    let output = PyDict::new(py);
    output.set_item("count", count)?;
    output.set_item("dimension_counts", dimension_counts)?;
    output.set_item("min_length", min_length)?;
    output.set_item("max_length", max_length)?;
    output.set_item("max_oracle_error", max_oracle_error)?;
    output.set_item("action_targets", action_targets)?;
    output.set_item("future_targets", future_targets)?;
    output.set_item("world_version", WORLD_VERSION)?;
    output.set_item("oracle_version", ORACLE_VERSION)?;
    output.set_item("token_abi_version", TOKEN_ABI_VERSION)?;
    Ok(output.into_any().unbind())
}

#[pyclass(name = "RolloutBatch")]
struct PyRolloutBatch {
    cfg: FamilyConfig,
    episodes: Vec<RolloutEpisode>,
    max_tokens: usize,
    profiled: bool,
}

#[pymethods]
impl PyRolloutBatch {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        *, seed, start_index, batch_size, max_tokens,
        d_min=1, d_max=4, gain_min=0.75, gain_max=1.25,
        action_limit=0.20, calibration_pulse=0.10, max_control_steps=4,
        profiled=false
    ))]
    fn new(
        seed: u64,
        start_index: u64,
        batch_size: usize,
        max_tokens: usize,
        d_min: usize,
        d_max: usize,
        gain_min: f32,
        gain_max: f32,
        action_limit: f32,
        calibration_pulse: f32,
        max_control_steps: usize,
        profiled: bool,
    ) -> PyResult<Self> {
        let cfg = py_config(
            d_min,
            d_max,
            gain_min,
            gain_max,
            action_limit,
            calibration_pulse,
            max_control_steps,
        )?;
        let episodes = (0..batch_size)
            .map(|offset| RolloutEpisode::new(&cfg, seed, start_index + offset as u64))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        if episodes
            .iter()
            .any(|episode| episode.tokens.len() + profile_offset(profiled) > max_tokens)
        {
            return Err(PyValueError::new_err(
                "initial rollout prefix exceeds max_tokens",
            ));
        }
        Ok(Self {
            cfg,
            episodes,
            max_tokens,
            profiled,
        })
    }

    fn learner_batch(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let records: Vec<&[LearningToken]> =
            self.episodes.iter().map(|e| e.tokens.as_slice()).collect();
        let padded = pad_records_with_optional_profile(
            &records,
            self.max_tokens,
            interpretation_profile(
                self.profiled,
                InterpretationProfile::ChannelValuesWithRequestedSpan,
            ),
        )
        .map_err(PyValueError::new_err)?;
        let output = padded_to_dict(py, padded)?;
        let offset = profile_offset(self.profiled) as i64;
        let mut positions = vec![vec![-1i64; self.cfg.d_max]; self.episodes.len()];
        let mut keys = vec![vec![-1i64; self.cfg.d_max]; self.episodes.len()];
        for (row, episode) in self.episodes.iter().enumerate() {
            if episode.done {
                continue;
            }
            let current = episode.current_query_positions();
            for (slot, (&position, &key)) in current.iter().zip(&episode.query_order).enumerate() {
                positions[row][slot] = position as i64 + offset;
                keys[row][slot] = key as i64;
            }
        }
        output.set_item("query_positions", positions)?;
        output.set_item("query_keys", keys)?;
        output.set_item(
            "dimensions",
            self.episodes
                .iter()
                .map(|e| e.instance.d)
                .collect::<Vec<_>>(),
        )?;
        output.set_item(
            "done",
            self.episodes.iter().map(|e| e.done).collect::<Vec<_>>(),
        )?;
        output.set_item("token_abi_version", token_abi_for(self.profiled))?;
        Ok(output.into_any().unbind())
    }

    /// Apply one normalized action list in the current public query order for
    /// every episode. Completed episodes must receive an empty list.
    fn step(&mut self, actions: Vec<Vec<f32>>) -> PyResult<()> {
        if actions.len() != self.episodes.len() {
            return Err(PyValueError::new_err(
                "expected one action list per rollout episode",
            ));
        }
        for (episode, action) in self.episodes.iter_mut().zip(actions) {
            if episode.done {
                if !action.is_empty() {
                    return Err(PyValueError::new_err(
                        "completed episode received a nonempty action",
                    ));
                }
                continue;
            }
            episode
                .step_normalized(&self.cfg, &action)
                .map_err(PyValueError::new_err)?;
            if episode.tokens.len() + profile_offset(self.profiled) > self.max_tokens {
                return Err(PyValueError::new_err(format!(
                    "rollout grew to {} tokens beyond max_tokens {}; truncation is forbidden",
                    episode.tokens.len(),
                    self.max_tokens
                )));
            }
        }
        Ok(())
    }

    fn all_done(&self) -> bool {
        self.episodes.iter().all(|episode| episode.done)
    }

    fn summary(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let output = PyDict::new(py);
        let success: Vec<bool> = self.episodes.iter().map(|e| e.success).collect();
        let errors: Vec<f32> = self.episodes.iter().map(|e| e.terminal_error()).collect();
        let steps: Vec<usize> = self.episodes.iter().map(|e| e.steps).collect();
        output.set_item("success", success)?;
        output.set_item("terminal_error", errors)?;
        output.set_item("steps", steps)?;
        output.set_item(
            "dimensions",
            self.episodes
                .iter()
                .map(|e| e.instance.d)
                .collect::<Vec<_>>(),
        )?;
        Ok(output.into_any().unbind())
    }

    /// Validation-only oracle actions, normalized and ordered exactly like the
    /// public action queries. Never use this method to construct model inputs.
    fn privileged_oracle_actions(&self) -> PyResult<Vec<Vec<f32>>> {
        self.episodes
            .iter()
            .map(|episode| {
                if episode.done {
                    return Ok(Vec::new());
                }
                let action = episode.oracle.action(&episode.x, &episode.goal)?;
                Ok(episode
                    .query_order
                    .iter()
                    .map(|&j| action[j] / self.cfg.action_limit)
                    .collect())
            })
            .collect::<Result<Vec<_>, String>>()
            .map_err(PyValueError::new_err)
    }
}

/// CPU-friendly diagnostic episodes using the exact learner-facing tensor ABI.
/// Privileged goals remain inside Rust and never appear in `learner_batch` when
/// the corresponding case declares them hidden.
#[pyclass(name = "GoalConditioningRolloutBatch")]
struct PyGoalConditioningRolloutBatch {
    episodes: Vec<DiagnosticRollout>,
    max_tokens: usize,
    serialization_order: SerializationOrder,
    profiled: bool,
}

#[pymethods]
impl PyGoalConditioningRolloutBatch {
    #[new]
    #[pyo3(signature = (*, max_tokens, serialization_order="canonical", profiled=false))]
    fn new(max_tokens: usize, serialization_order: &str, profiled: bool) -> PyResult<Self> {
        if max_tokens == 0 {
            return Err(PyValueError::new_err("max_tokens must be nonzero"));
        }
        let serialization_order = parse_serialization_order(serialization_order)?;
        let episodes = standard_diagnostic_rollouts(serialization_order);
        if episodes
            .iter()
            .any(|episode| episode.tokens().len() + profile_offset(profiled) > max_tokens)
        {
            return Err(PyValueError::new_err(
                "initial diagnostic prefix exceeds max_tokens",
            ));
        }
        Ok(Self {
            episodes,
            max_tokens,
            serialization_order,
            profiled,
        })
    }

    fn learner_batch(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let records: Vec<&[LearningToken]> = self
            .episodes
            .iter()
            .map(|episode| episode.tokens())
            .collect();
        let padded = pad_records_with_optional_profile(
            &records,
            self.max_tokens,
            interpretation_profile(
                self.profiled,
                InterpretationProfile::KeySelectionsWithRemainingHorizon,
            ),
        )
        .map_err(PyValueError::new_err)?;
        let output = padded_to_dict(py, padded)?;
        let offset = profile_offset(self.profiled);
        output.set_item(
            "query_positions",
            self.episodes
                .iter()
                .map(|episode| {
                    episode
                        .current_query_position()
                        .map(|position| (position + offset) as i64)
                        .unwrap_or(-1)
                })
                .collect::<Vec<_>>(),
        )?;
        output.set_item(
            "case_ids",
            self.episodes
                .iter()
                .map(|episode| episode.case().id.clone())
                .collect::<Vec<_>>(),
        )?;
        output.set_item(
            "case_kinds",
            self.episodes
                .iter()
                .map(|episode| episode.case().kind.as_str())
                .collect::<Vec<_>>(),
        )?;
        output.set_item(
            "done",
            self.episodes
                .iter()
                .map(DiagnosticRollout::is_done)
                .collect::<Vec<_>>(),
        )?;
        output.set_item("serialization_order", self.serialization_order.as_str())?;
        output.set_item(
            "diagnostic_serialization_version",
            DIAGNOSTIC_SERIALIZATION_VERSION,
        )?;
        output.set_item("token_abi_version", token_abi_for(self.profiled))?;
        Ok(output.into_any().unbind())
    }

    /// Apply one normalized continuous movement value to every live episode.
    /// Completed episodes must receive an empty list.
    fn step(&mut self, actions: Vec<Vec<f32>>) -> PyResult<()> {
        if actions.len() != self.episodes.len() {
            return Err(PyValueError::new_err(
                "expected one action list per diagnostic episode",
            ));
        }
        for (episode, action) in self.episodes.iter_mut().zip(actions) {
            if episode.is_done() {
                if !action.is_empty() {
                    return Err(PyValueError::new_err(
                        "completed diagnostic episode received a nonempty action",
                    ));
                }
                continue;
            }
            if action.len() != 1 {
                return Err(PyValueError::new_err(
                    "live diagnostic episode requires exactly one movement value",
                ));
            }
            episode
                .step_normalized(action[0])
                .map_err(PyValueError::new_err)?;
            if episode.tokens().len() + profile_offset(self.profiled) > self.max_tokens {
                return Err(PyValueError::new_err(format!(
                    "diagnostic rollout grew to {} tokens beyond max_tokens {}; truncation is forbidden",
                    episode.tokens().len(),
                    self.max_tokens
                )));
            }
        }
        Ok(())
    }

    fn all_done(&self) -> bool {
        self.episodes.iter().all(DiagnosticRollout::is_done)
    }

    fn summary(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let output = PyDict::new(py);
        output.set_item(
            "case_ids",
            self.episodes
                .iter()
                .map(|episode| episode.case().id.clone())
                .collect::<Vec<_>>(),
        )?;
        output.set_item(
            "case_kinds",
            self.episodes
                .iter()
                .map(|episode| episode.case().kind.as_str())
                .collect::<Vec<_>>(),
        )?;
        output.set_item(
            "success",
            self.episodes
                .iter()
                .map(DiagnosticRollout::success)
                .collect::<Vec<_>>(),
        )?;
        output.set_item(
            "action_displacements",
            self.episodes
                .iter()
                .map(DiagnosticRollout::action_displacements)
                .collect::<Vec<_>>(),
        )?;
        output.set_item("serialization_order", self.serialization_order.as_str())?;
        output.set_item("token_abi_version", token_abi_for(self.profiled))?;
        Ok(output.into_any().unbind())
    }
}

/// Container-process episodes exposed through the same public tensor boundary.
/// The new path is profiled by default because it has no legacy checkpoint
/// lineage. Privileged goals and optimality checks appear only in evaluator
/// methods, never in `learner_batch`.
#[pyclass(name = "EvictionRolloutBatch")]
struct PyEvictionRolloutBatch {
    episodes: Vec<EvictionRollout>,
    max_tokens: usize,
    serialization_order: EvictionSerializationOrder,
    profiled: bool,
}

#[pymethods]
impl PyEvictionRolloutBatch {
    #[new]
    #[pyo3(signature = (*, max_tokens, serialization_order="canonical", profiled=true))]
    fn new(max_tokens: usize, serialization_order: &str, profiled: bool) -> PyResult<Self> {
        if max_tokens == 0 {
            return Err(PyValueError::new_err("max_tokens must be nonzero"));
        }
        let serialization_order = parse_eviction_serialization_order(serialization_order)?;
        let episodes = standard_eviction_rollouts(serialization_order);
        if episodes
            .iter()
            .any(|episode| episode.tokens().len() + profile_offset(profiled) > max_tokens)
        {
            return Err(PyValueError::new_err(
                "initial eviction prefix exceeds max_tokens",
            ));
        }
        Ok(Self {
            episodes,
            max_tokens,
            serialization_order,
            profiled,
        })
    }

    fn learner_batch(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let records: Vec<&[LearningToken]> =
            self.episodes.iter().map(EvictionRollout::tokens).collect();
        let padded = pad_records_with_optional_profile(
            &records,
            self.max_tokens,
            interpretation_profile(
                self.profiled,
                InterpretationProfile::KeySelectionsWithRemainingHorizon,
            ),
        )
        .map_err(PyValueError::new_err)?;
        let output = padded_to_dict(py, padded)?;
        let offset = profile_offset(self.profiled);
        output.set_item(
            "query_positions",
            self.episodes
                .iter()
                .map(|episode| {
                    episode
                        .current_query_positions()
                        .into_iter()
                        .map(|position| (position + offset) as i64)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )?;
        output.set_item(
            "query_keys",
            self.episodes
                .iter()
                .map(|episode| {
                    episode
                        .current_query_positions()
                        .into_iter()
                        .map(|position| episode.tokens()[position].public.key as i64)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )?;
        output.set_item(
            "case_ids",
            self.episodes
                .iter()
                .map(|episode| episode.case().id.clone())
                .collect::<Vec<_>>(),
        )?;
        output.set_item(
            "case_kinds",
            self.episodes
                .iter()
                .map(|episode| episode.case().kind.as_str())
                .collect::<Vec<_>>(),
        )?;
        output.set_item(
            "done",
            self.episodes
                .iter()
                .map(EvictionRollout::is_done)
                .collect::<Vec<_>>(),
        )?;
        output.set_item("serialization_order", self.serialization_order.as_str())?;
        output.set_item("process_version", PROCESS_VERSION)?;
        output.set_item("token_abi_version", token_abi_for(self.profiled))?;
        Ok(output.into_any().unbind())
    }

    /// Apply one normalized value per currently queried container channel.
    /// Completed episodes must receive an empty list.
    fn step(&mut self, actions: Vec<Vec<f32>>) -> PyResult<()> {
        if actions.len() != self.episodes.len() {
            return Err(PyValueError::new_err(
                "expected one action list per eviction episode",
            ));
        }
        for (episode, action) in self.episodes.iter_mut().zip(actions) {
            if episode.is_done() {
                if !action.is_empty() {
                    return Err(PyValueError::new_err(
                        "completed eviction episode received a nonempty action",
                    ));
                }
                continue;
            }
            episode
                .step_normalized(&action)
                .map_err(PyValueError::new_err)?;
            if episode.tokens().len() + profile_offset(self.profiled) > self.max_tokens {
                return Err(PyValueError::new_err(format!(
                    "eviction rollout grew to {} records beyond max_tokens {}; truncation is forbidden",
                    episode.tokens().len() + profile_offset(self.profiled),
                    self.max_tokens
                )));
            }
        }
        Ok(())
    }

    fn all_done(&self) -> bool {
        self.episodes.iter().all(EvictionRollout::is_done)
    }

    fn summary(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let output = PyDict::new(py);
        output.set_item(
            "case_ids",
            self.episodes
                .iter()
                .map(|episode| episode.case().id.clone())
                .collect::<Vec<_>>(),
        )?;
        output.set_item(
            "case_kinds",
            self.episodes
                .iter()
                .map(|episode| episode.case().kind.as_str())
                .collect::<Vec<_>>(),
        )?;
        output.set_item(
            "success",
            self.episodes
                .iter()
                .map(EvictionRollout::success)
                .collect::<Vec<_>>(),
        )?;
        output.set_item(
            "commands",
            self.episodes
                .iter()
                .map(|episode| {
                    episode
                        .commands()
                        .iter()
                        .map(|command| command.as_str())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )?;
        output.set_item(
            "first_action_optimal",
            self.episodes
                .iter()
                .map(|episode| {
                    episode.commands().first().is_some_and(|command| {
                        optimal_first_commands(
                            episode.case().start,
                            episode.case().goal_for_verification(),
                        )
                        .contains(command)
                    })
                })
                .collect::<Vec<_>>(),
        )?;
        output.set_item(
            "steps",
            self.episodes
                .iter()
                .map(EvictionRollout::steps)
                .collect::<Vec<_>>(),
        )?;
        output.set_item("serialization_order", self.serialization_order.as_str())?;
        output.set_item("process_version", PROCESS_VERSION)?;
        output.set_item("token_abi_version", token_abi_for(self.profiled))?;
        Ok(output.into_any().unbind())
    }

    /// Validation-only optimal actions in the current public query order.
    /// These values must never be included in `learner_batch`.
    fn privileged_oracle_actions(&self) -> Vec<Vec<f32>> {
        self.episodes
            .iter()
            .map(|episode| {
                if episode.is_done() {
                    return Vec::new();
                }
                let goal = episode.case().goal_for_verification();
                let state = episode.state();
                let command = Command::alphabet()
                    .into_iter()
                    .find(|candidate| eviction_transition(state, *candidate).item == goal)
                    .or_else(|| optimal_first_commands(state, goal).into_iter().next())
                    .unwrap_or(Command::Hold);
                episode
                    .current_query_positions()
                    .into_iter()
                    .map(|position| {
                        let key = episode.tokens()[position].public.key;
                        let container = episode
                            .case()
                            .presentation
                            .container_of_evict_key(key)
                            .expect("every query key names a declared container");
                        f32::from(matches!(command, Command::Evict(chosen) if chosen == container))
                    })
                    .collect()
            })
            .collect()
    }
}

#[pyfunction]
#[pyo3(signature = (*, previous_json, candidate_json, thresholds_json=None))]
fn classify_goal_progress(
    previous_json: &str,
    candidate_json: &str,
    thresholds_json: Option<&str>,
) -> PyResult<String> {
    let previous: CheckpointEvidence = serde_json::from_str(previous_json)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let candidate: CheckpointEvidence = serde_json::from_str(candidate_json)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let thresholds = thresholds_json
        .map(serde_json::from_str::<ProgressThresholds>)
        .transpose()
        .map_err(|error| PyValueError::new_err(error.to_string()))?
        .unwrap_or_default();
    serde_json::to_string(&classify_progress(&previous, &candidate, &thresholds))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
fn versions(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let output = PyDict::new(py);
    output.set_item("world", WORLD_VERSION)?;
    output.set_item("oracle", ORACLE_VERSION)?;
    output.set_item("token_abi", TOKEN_ABI_VERSION)?;
    output.set_item("profiled_token_abi", PROFILED_TOKEN_ABI_VERSION)?;
    output.set_item("role_count", Role::COUNT)?;
    output.set_item("payload_dim", PAYLOAD_DIM)?;
    output.set_item("action_horizon", ACTION_HORIZON)?;
    output.set_item("diagnostic_serialization", DIAGNOSTIC_SERIALIZATION_VERSION)?;
    output.set_item("eviction_process", PROCESS_VERSION)?;
    Ok(output.into_any().unbind())
}

#[pymodule]
fn pretraining_world_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(generate_training_batch, module)?)?;
    module.add_function(wrap_pyfunction!(
        generate_goal_conditioning_training_batch,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(validate_generated_worlds, module)?)?;
    module.add_function(wrap_pyfunction!(versions, module)?)?;
    module.add_function(wrap_pyfunction!(classify_goal_progress, module)?)?;
    module.add_class::<PyRolloutBatch>()?;
    module.add_class::<PyGoalConditioningRolloutBatch>()?;
    module.add_class::<PyEvictionRolloutBatch>()?;
    Ok(())
}
