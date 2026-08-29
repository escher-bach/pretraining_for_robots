//! The one learner event boundary every finite G0 family renders through.
//!
//! Seed-gate condition 2 in `DEVELOPMENT-PATH.md` is that the portfolio
//! families "express genuinely different process relations through **one
//! self-describing learner event boundary**". That sentence rules out the
//! obvious implementation, which is for each card crate to emit
//! `LearningToken`s directly the way the two legacy worlds do. Five hand-written
//! emitters would be five boundaries wearing one ABI version, and the canonical
//! audit already found what that costs: two producers disagreeing about slot
//! meaning with no way for a decoder to tell.
//!
//! So a card does not emit rows. It emits a **public transcript** — an ordered
//! list of simultaneity groups of typed public facts — and this crate is the
//! only thing that turns a transcript into bytes. The path is:
//!
//! ```text
//! card transcript -> canonical Episode -> PublicRow + SupervisionRow
//!   -> LearningToken -> profiled envelope -> padded batch
//! ```
//!
//! Two properties follow from routing through the canonical record rather than
//! around it, and both are checked rather than asserted:
//!
//! - **invertibility.** [`boundary_check`] renders a transcript and decodes the
//!   result back, requiring the decoded canonical episode to equal the one that
//!   was rendered. A family whose events cannot survive the round trip is not
//!   admitted, which is what makes "the learner sees exactly these facts" a
//!   statement about the wire and not about intent.
//! - **separation.** [`render_public`] takes only the public half of an
//!   `Episode`, so no rendering path can consult supervision. The teacher target
//!   travels in a table addressed by record position, built here and never read
//!   by the public renderer.
//!
//! # What a transcript may not contain
//!
//! There is no fact kind for privileged state, and no place to put one: a
//! [`G0Fact`] is a public record and the supervision channel carries a single
//! scalar per action query. A card that wants to leak has to change this type.

use pretraining_canonical_event::{
    decode_episode, render_public, render_supervision, ActionTarget, ChannelContent, ChannelRole,
    ConditionCode, Episode, EventGroup, LocalKey, Profile, PublicEpisode, PublicFact, PublicRecord,
    PublicRow, Quantity, QueryHorizon, RecordAddress, StepSpan, SupervisionRecord,
    SupervisionTable, ACTION_HORIZON,
};
use pretraining_profiled_event::{tag_legacy_episode, InterpretationProfile};
use pretraining_world::{LearningToken, PublicToken, Role, Supervision};
use serde::{Deserialize, Serialize};

/// Re-exported so a card crate depends on this boundary and not on the
/// record layer underneath it. A card that reached past this seam could
/// construct a public fact the transcript type does not admit.
pub use pretraining_canonical_event::{BoundarySubtype, KeyNamespace};

/// The width of the learner's action head.
///
/// Cards express step counts as a fraction of this rather than of their own
/// horizon. Sixteen is a power of two, so every such fraction is exact in the
/// `f32` payload; a horizon of three is not, and the renderer refuses `1/3`
/// rather than rounding it. That refusal is why this constant is public.
pub const ACTION_HEAD: usize = pretraining_canonical_event::ACTION_HORIZON;

/// Express a step count as an exactly representable fraction of the action head.
pub fn step_fraction(steps: usize) -> f64 {
    assert!(
        steps <= ACTION_HEAD,
        "a step count of {steps} exceeds the action head"
    );
    steps as f64 / ACTION_HEAD as f64
}
pub use pretraining_profiled_event::PROFILED_TOKEN_ABI_VERSION;

/// The canonical profile every finite G0 family declares.
pub const CANONICAL_PROFILE: Profile = Profile::FiniteG0;
/// The envelope code that selects it in a learner-visible header.
pub const ENVELOPE_PROFILE: InterpretationProfile = InterpretationProfile::FiniteG0Discrete;

/// The teacher's target for the action query it selects.
///
/// The action head is a `tanh`, so `1.0` is its supremum rather than a value it
/// attains. It is used anyway, because the two legacy worlds already supervise
/// this head with saturating targets and a second convention would make a later
/// mixture comparison depend on which family a record came from. The cost is
/// that the selected query's residual never reaches zero; the pilot reads
/// argmax accuracy, which is unaffected by a common offset.
pub const SELECTED_TARGET: f64 = 1.0;
/// The target for every action query the teacher did not select.
pub const REJECTED_TARGET: f64 = -1.0;

/// The slot of the action head the teacher writes.
///
/// One slot, not a horizon of them: a G0 decision is a choice among the
/// actuators offered at this step, and the query rows already carry the
/// alternatives. Spreading one categorical choice across the head would make
/// the head's width part of the family semantics.
pub const SCORED_ACTION_SLOT: usize = 0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderFault {
    /// A transcript with no decision cannot supervise anything.
    NoActionQuery,
    /// Two action queries in one group name the same actuator.
    DuplicateActuator { group: usize, actuator: u16 },
    /// A decision offered no alternatives, so the choice is not a choice.
    SingleAlternative { group: usize },
    /// A decision asserted no correct action, so it supervises nothing.
    NoSelectedAction { group: usize },
    /// The declared control horizon does not fit the action head.
    HorizonExceedsActionHead { horizon: usize },
    /// Remaining steps exceeded the declared horizon.
    RemainingExceedsHorizon { remaining: usize, horizon: usize },
    /// The transcript names an actuator the schema does not declare.
    UndeclaredActuator { actuator: u16 },
    /// The transcript names an observation channel the schema does not declare.
    UndeclaredChannel { channel: u16 },
    /// The canonical layer refused the episode.
    Canonical { detail: String },
    /// Rendering and decoding disagreed.
    NotInvertible { detail: String },
    /// The transcript would have to be taught by a policy reading unpublished
    /// state.
    ///
    /// A shared fault rather than a card-local one, because every family faces
    /// the same rule: `AGENTS.md` forbids privileged state from becoming
    /// learner input *or supervision*, and a teacher is supervision. Card 04
    /// found this the hard way — its audited optimal first action on the
    /// unannounced-switch witness was privileged — so the boundary carries a
    /// way to refuse rather than leaving each card to remember.
    TeacherWouldLeak { detail: String },
}

impl std::fmt::Display for RenderFault {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoActionQuery => write!(formatter, "the transcript contains no action query"),
            Self::DuplicateActuator { group, actuator } => write!(
                formatter,
                "group {group} queries actuator {actuator} more than once"
            ),
            Self::SingleAlternative { group } => write!(
                formatter,
                "group {group} offers one action, so the decision is not a choice"
            ),
            Self::NoSelectedAction { group } => write!(
                formatter,
                "group {group} marks no action correct, so it supervises nothing"
            ),
            Self::HorizonExceedsActionHead { horizon } => write!(
                formatter,
                "a control horizon of {horizon} exceeds the {ACTION_HORIZON}-step action head"
            ),
            Self::RemainingExceedsHorizon { remaining, horizon } => write!(
                formatter,
                "{remaining} remaining steps exceed the declared horizon {horizon}"
            ),
            Self::UndeclaredActuator { actuator } => {
                write!(formatter, "actuator {actuator} is not in the schema")
            }
            Self::UndeclaredChannel { channel } => {
                write!(
                    formatter,
                    "observation channel {channel} is not in the schema"
                )
            }
            Self::Canonical { detail } => write!(formatter, "{detail}"),
            Self::NotInvertible { detail } => write!(formatter, "{detail}"),
            Self::TeacherWouldLeak { detail } => write!(
                formatter,
                "the teacher for this transcript would read unpublished state: {detail}"
            ),
        }
    }
}

impl std::error::Error for RenderFault {}

/// One declared public port.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Port {
    pub key: u16,
    /// The schema reference value for the port, in its declared bounds.
    pub reference: f64,
    pub lower: f64,
    pub upper: f64,
}

impl Port {
    pub fn unit(key: u16) -> Self {
        Self {
            key,
            reference: 0.0,
            lower: 0.0,
            upper: 1.0,
        }
    }

    pub fn signed(key: u16) -> Self {
        Self {
            key,
            reference: 0.0,
            lower: -1.0,
            upper: 1.0,
        }
    }

    fn quantity(&self) -> Quantity {
        Quantity::normalized(self.reference, self.lower, self.upper)
    }
}

/// The body and interface a family publishes at episode start.
///
/// This is the entire basis on which a learner can tell one family from
/// another. There is no family identifier: the envelope declares a *reading* of
/// the slots and nothing about which world produced them, so which family this
/// is has to be inferred from the ports and the events — which is the
/// identification content trunk T1 is about.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PortSchema {
    pub observations: Vec<Port>,
    pub actuators: Vec<Port>,
}

impl PortSchema {
    pub fn declares_actuator(&self, key: u16) -> bool {
        self.actuators.iter().any(|port| port.key == key)
    }

    pub fn declares_channel(&self, key: u16) -> bool {
        self.observations.iter().any(|port| port.key == key)
    }
}

/// A public content reading: a channel value or the selection of a named key.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Content {
    /// The numeric value of the named channel.
    Value { value: f64, lower: f64, upper: f64 },
    /// The named key is the selected one, with no magnitude.
    Selection,
}

impl Content {
    fn canonical(self) -> ChannelContent {
        match self {
            Self::Value {
                value,
                lower,
                upper,
            } => ChannelContent::Value(Quantity::normalized(value, lower, upper)),
            Self::Selection => ChannelContent::Selection {
                indicator: Quantity::normalized(1.0, 0.0, 1.0),
            },
        }
    }
}

/// One public fact a family may publish.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum G0Fact {
    Boundary(BoundarySubtype),
    Goal {
        key: u16,
        namespace: KeyNamespace,
        content: Content,
    },
    Observation {
        key: u16,
        content: Content,
    },
    /// The `reveal` construct: a public condition on a named key.
    Condition {
        key: u16,
        namespace: KeyNamespace,
        code: u16,
        value: f64,
        lower: f64,
        upper: f64,
    },
    /// One alternative offered at this decision.
    ///
    /// `selected` says the teacher asserts this action is **correct**, not that
    /// it is the one that will be executed. More than one may be selected in a
    /// decision, and that is the point: where several actions attain the
    /// ceiling the contract is indifferent between them, and marking one would
    /// teach a preference the world does not have. It also made two
    /// structurally different card 02 cases render byte-identically, because
    /// the tie-break happened to agree.
    ///
    /// It becomes a supervision entry and never a payload slot.
    ActionQuery {
        actuator: u16,
        remaining: usize,
        selected: bool,
    },
    /// The action that was actually taken, published as a public fact.
    ActionExecuted {
        actuator: u16,
    },
}

/// A simultaneity group: facts that share one event index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct G0Group {
    pub facts: Vec<G0Fact>,
}

impl G0Group {
    pub fn new(facts: Vec<G0Fact>) -> Self {
        Self { facts }
    }

    pub fn one(fact: G0Fact) -> Self {
        Self { facts: vec![fact] }
    }
}

/// A family's complete public transcript for one episode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct G0Episode {
    pub schema: PortSchema,
    /// The episode's control horizon in decisions.
    pub horizon: usize,
    pub groups: Vec<G0Group>,
}

impl G0Episode {
    pub fn new(schema: PortSchema, horizon: usize, groups: Vec<G0Group>) -> Self {
        Self {
            schema,
            horizon,
            groups,
        }
    }

    /// The number of decisions the transcript contains.
    pub fn decisions(&self) -> usize {
        self.groups
            .iter()
            .filter(|group| {
                group
                    .facts
                    .iter()
                    .any(|fact| matches!(fact, G0Fact::ActionQuery { .. }))
            })
            .count()
    }

    /// The actuators the teacher marked correct at each decision, in order.
    ///
    /// A set per decision rather than one actuator, because a contract may be
    /// indifferent between several. A pilot scores membership in this set, not
    /// equality with one member.
    pub fn selected_actuators(&self) -> Vec<Vec<u16>> {
        self.groups
            .iter()
            .filter_map(|group| {
                let selected: Vec<u16> = group
                    .facts
                    .iter()
                    .filter_map(|fact| match fact {
                        G0Fact::ActionQuery {
                            actuator, selected, ..
                        } if *selected => Some(*actuator),
                        _ => None,
                    })
                    .collect();
                let queried = group
                    .facts
                    .iter()
                    .any(|fact| matches!(fact, G0Fact::ActionQuery { .. }));
                queried.then_some(selected)
            })
            .collect()
    }
}

fn validate(episode: &G0Episode) -> Result<(), RenderFault> {
    if episode.horizon > ACTION_HORIZON {
        return Err(RenderFault::HorizonExceedsActionHead {
            horizon: episode.horizon,
        });
    }
    let mut any_query = false;
    for (index, group) in episode.groups.iter().enumerate() {
        let mut actuators: Vec<u16> = Vec::new();
        let mut selected_count = 0usize;
        for fact in &group.facts {
            match fact {
                G0Fact::ActionQuery {
                    actuator,
                    remaining,
                    selected,
                } => {
                    any_query = true;
                    selected_count += usize::from(*selected);
                    if !episode.schema.declares_actuator(*actuator) {
                        return Err(RenderFault::UndeclaredActuator {
                            actuator: *actuator,
                        });
                    }
                    if *remaining > episode.horizon {
                        return Err(RenderFault::RemainingExceedsHorizon {
                            remaining: *remaining,
                            horizon: episode.horizon,
                        });
                    }
                    if actuators.contains(actuator) {
                        return Err(RenderFault::DuplicateActuator {
                            group: index,
                            actuator: *actuator,
                        });
                    }
                    actuators.push(*actuator);
                }
                G0Fact::ActionExecuted { actuator } => {
                    if !episode.schema.declares_actuator(*actuator) {
                        return Err(RenderFault::UndeclaredActuator {
                            actuator: *actuator,
                        });
                    }
                }
                G0Fact::Observation { key, .. } => {
                    if !episode.schema.declares_channel(*key) {
                        return Err(RenderFault::UndeclaredChannel { channel: *key });
                    }
                }
                _ => {}
            }
        }
        if actuators.len() == 1 {
            return Err(RenderFault::SingleAlternative { group: index });
        }
        if !actuators.is_empty() && selected_count == 0 {
            return Err(RenderFault::NoSelectedAction { group: index });
        }
    }
    if !any_query {
        return Err(RenderFault::NoActionQuery);
    }
    Ok(())
}

fn schema_group(schema: &PortSchema) -> Result<EventGroup, RenderFault> {
    let mut records = Vec::with_capacity(schema.observations.len() + schema.actuators.len());
    for port in &schema.observations {
        records.push(
            PublicRecord::new(
                LocalKey::new(KeyNamespace::Observation, port.key),
                PublicFact::ChannelSchema {
                    channel: ChannelRole::Observation,
                    reference: port.quantity(),
                    command_span: None,
                },
            )
            .map_err(|detail| RenderFault::Canonical { detail })?,
        );
    }
    for port in &schema.actuators {
        records.push(
            PublicRecord::new(
                LocalKey::new(KeyNamespace::Actuator, port.key),
                PublicFact::ChannelSchema {
                    channel: ChannelRole::Actuator,
                    reference: port.quantity(),
                    command_span: None,
                },
            )
            .map_err(|detail| RenderFault::Canonical { detail })?,
        );
    }
    Ok(EventGroup { group: 0, records })
}

fn fact_record(
    fact: &G0Fact,
    horizon: usize,
) -> Result<(PublicRecord, Option<SupervisionRecord>), RenderFault> {
    let canonical = |detail: String| RenderFault::Canonical { detail };
    Ok(match fact {
        G0Fact::Boundary(subtype) => (
            PublicRecord::new(
                LocalKey::new(KeyNamespace::Episode, 0),
                PublicFact::Boundary { subtype: *subtype },
            )
            .map_err(canonical)?,
            None,
        ),
        G0Fact::Goal {
            key,
            namespace,
            content,
        } => (
            PublicRecord::new(
                LocalKey::new(*namespace, *key),
                PublicFact::Goal {
                    content: content.canonical(),
                },
            )
            .map_err(canonical)?,
            None,
        ),
        G0Fact::Observation { key, content } => (
            PublicRecord::new(
                LocalKey::new(KeyNamespace::Observation, *key),
                PublicFact::Observation {
                    content: content.canonical(),
                },
            )
            .map_err(canonical)?,
            None,
        ),
        G0Fact::Condition {
            key,
            namespace,
            code,
            value,
            lower,
            upper,
        } => (
            PublicRecord::new(
                LocalKey::new(*namespace, *key),
                PublicFact::Condition {
                    namespace: *namespace,
                    code: ConditionCode(*code),
                    quantity: Quantity::normalized(*value, *lower, *upper),
                },
            )
            .map_err(canonical)?,
            None,
        ),
        G0Fact::ActionQuery {
            actuator,
            remaining,
            selected,
        } => {
            let span = StepSpan::new(*remaining as u16, horizon as u16).map_err(canonical)?;
            (
                PublicRecord::new(
                    LocalKey::new(KeyNamespace::Actuator, *actuator),
                    PublicFact::ActionQuery {
                        command: Quantity::normalized(0.0, -1.0, 1.0),
                        horizon: QueryHorizon::RemainingFraction { remaining: span },
                    },
                )
                .map_err(canonical)?,
                Some(SupervisionRecord {
                    action_targets: vec![ActionTarget {
                        step: SCORED_ACTION_SLOT,
                        value: if *selected {
                            SELECTED_TARGET
                        } else {
                            REJECTED_TARGET
                        },
                    }],
                    future_target: None,
                }),
            )
        }
        G0Fact::ActionExecuted { actuator } => (
            PublicRecord::new(
                LocalKey::new(KeyNamespace::Actuator, *actuator),
                PublicFact::ActionExecuted {
                    command: Quantity::normalized(1.0, 0.0, 1.0),
                    actuator_marker: true,
                },
            )
            .map_err(canonical)?,
            None,
        ),
    })
}

/// Build the canonical episode and its separately addressed supervision.
pub fn canonical_episode(episode: &G0Episode) -> Result<Episode, RenderFault> {
    validate(episode)?;
    let mut groups = vec![schema_group(&episode.schema)?];
    let mut supervision = SupervisionTable::default();
    for (offset, group) in episode.groups.iter().enumerate() {
        let group_index = offset + 1;
        let mut records = Vec::with_capacity(group.facts.len());
        for (record_index, fact) in group.facts.iter().enumerate() {
            let (record, entry) = fact_record(fact, episode.horizon)?;
            if let Some(entry) = entry {
                supervision.set(
                    RecordAddress {
                        group_index,
                        record_index,
                    },
                    entry,
                );
            }
            records.push(record);
        }
        groups.push(EventGroup {
            group: group_index as u16,
            records,
        });
    }
    Episode::new(PublicEpisode::new(CANONICAL_PROFILE, groups), supervision)
        .map_err(|detail| RenderFault::Canonical { detail })
}

fn to_learning_tokens(episode: &Episode) -> Result<Vec<LearningToken>, RenderFault> {
    let public = render_public(episode.public()).map_err(|error| RenderFault::Canonical {
        detail: error.to_string(),
    })?;
    let supervision = render_supervision(episode).map_err(|error| RenderFault::Canonical {
        detail: error.to_string(),
    })?;
    Ok(public
        .into_iter()
        .zip(supervision)
        .map(|(row, targets)| LearningToken {
            public: PublicToken {
                role: role_from_code(row.role),
                key: row.key,
                event: row.group,
                payload: row.payload,
            },
            supervision: Supervision {
                action_target: targets.action_target,
                action_mask: targets.action_mask,
                future_target: targets.future_target,
                future_mask: targets.future_mask,
            },
        })
        .collect())
}

/// The role codes are shared with `pretraining-world`; this is the only place
/// the two spellings meet, and it is total rather than defaulting.
fn role_from_code(code: u8) -> Role {
    match code {
        1 => Role::SchemaObservation,
        2 => Role::SchemaActuator,
        3 => Role::Boundary,
        4 => Role::Condition,
        5 => Role::Goal,
        6 => Role::Observation,
        7 => Role::ActionQuery,
        8 => Role::ActionExecuted,
        9 => Role::FutureQuery,
        10 => Role::Feedback,
        other => unreachable!("the canonical renderer emitted role {other}"),
    }
}

/// Render a transcript onto the legacy `0.2.0` body, without the envelope.
pub fn legacy_tokens(episode: &G0Episode) -> Result<Vec<LearningToken>, RenderFault> {
    to_learning_tokens(&canonical_episode(episode)?)
}

/// Render a transcript onto the learner-visible `0.3.1` envelope.
pub fn profiled_tokens(episode: &G0Episode) -> Result<Vec<LearningToken>, RenderFault> {
    let body = legacy_tokens(episode)?;
    tag_legacy_episode(ENVELOPE_PROFILE, &body).map_err(|error| RenderFault::Canonical {
        detail: error.to_string(),
    })
}

fn rows_of(tokens: &[LearningToken]) -> Vec<PublicRow> {
    tokens
        .iter()
        .map(|token| PublicRow {
            role: token.public.role as u8,
            key: token.public.key,
            group: token.public.event,
            payload: token.public.payload,
        })
        .collect()
}

/// The record count and public-fact fingerprint of one rendered transcript.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryEvidence {
    pub records: usize,
    pub profiled_records: usize,
    pub decisions: usize,
    pub supervised_records: usize,
    /// Invariant to presentation order inside a group, and to nothing else.
    pub fingerprint: u64,
    pub round_trips: bool,
}

/// Render, decode, and require the two canonical episodes to be equal.
///
/// This is the check that makes "the learner sees exactly these facts" a
/// property of the wire. It is deliberately an equality on the *canonical*
/// episode rather than on the rows: equal rows would be satisfied by a renderer
/// and a decoder that shared a mistake, while equality after a decode that
/// refuses every departure from the profile's field schema would not.
pub fn boundary_check(episode: &G0Episode) -> Result<BoundaryEvidence, RenderFault> {
    let canonical = canonical_episode(episode)?;
    let tokens = to_learning_tokens(&canonical)?;
    let decoded = decode_episode(CANONICAL_PROFILE, &rows_of(&tokens)).map_err(|error| {
        RenderFault::NotInvertible {
            detail: format!("the rendered rows did not decode: {error}"),
        }
    })?;
    if &decoded != canonical.public() {
        return Err(RenderFault::NotInvertible {
            detail: "the decoded episode differs from the one that was rendered".into(),
        });
    }
    let profiled = profiled_tokens(episode)?;
    let supervised = tokens
        .iter()
        .filter(|token| token.supervision.action_mask.iter().any(|slot| *slot))
        .count();
    Ok(BoundaryEvidence {
        records: tokens.len(),
        profiled_records: profiled.len(),
        decisions: episode.decisions(),
        supervised_records: supervised,
        fingerprint: canonical.public().presentation_free_fingerprint(),
        round_trips: true,
    })
}

/// The evidence a family reports about its rendering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderingReport {
    pub canonical_profile: String,
    pub envelope_profile: String,
    pub envelope_abi: String,
    pub episodes: usize,
    pub max_records: usize,
    pub max_profiled_records: usize,
    pub total_decisions: usize,
    pub every_episode_round_trips: bool,
    /// Distinct fingerprints, so a family that renders every case identically
    /// is visible rather than merely suspected.
    ///
    /// A count below `episodes` is not automatically a defect: two case labels
    /// can name the same world and differ only in which family they are scored
    /// inside, or in a counterfactual no on-policy episode reveals. It *is*
    /// something a training mixture has to account for, because identical
    /// episodes are duplicated data rather than additional coverage.
    pub distinct_fingerprints: usize,
    /// The episode indices that render to the same public stream, grouped.
    ///
    /// Reported rather than left to be inferred from the count, because *which*
    /// episodes collide is the interesting part: two labels naming one contract
    /// is bookkeeping, while a control that collides with its own witness means
    /// the control's evidence is off-policy and cannot come from the corpus.
    pub colliding_episodes: Vec<Vec<usize>>,
}

/// Summarize the rendering of a whole family.
pub fn rendering_report(episodes: &[G0Episode]) -> Result<RenderingReport, RenderFault> {
    let mut fingerprints = Vec::with_capacity(episodes.len());
    let mut max_records = 0usize;
    let mut max_profiled = 0usize;
    let mut decisions = 0usize;
    for episode in episodes {
        let evidence = boundary_check(episode)?;
        max_records = max_records.max(evidence.records);
        max_profiled = max_profiled.max(evidence.profiled_records);
        decisions += evidence.decisions;
        fingerprints.push(evidence.fingerprint);
    }
    let mut groups: std::collections::BTreeMap<u64, Vec<usize>> = std::collections::BTreeMap::new();
    for (index, fingerprint) in fingerprints.iter().enumerate() {
        groups.entry(*fingerprint).or_default().push(index);
    }
    let colliding: Vec<Vec<usize>> = groups
        .into_values()
        .filter(|indices| indices.len() > 1)
        .collect();
    fingerprints.sort_unstable();
    fingerprints.dedup();
    Ok(RenderingReport {
        canonical_profile: CANONICAL_PROFILE.as_str().to_string(),
        envelope_profile: ENVELOPE_PROFILE.as_str().to_string(),
        envelope_abi: pretraining_profiled_event::PROFILED_TOKEN_ABI_VERSION.to_string(),
        episodes: episodes.len(),
        max_records,
        max_profiled_records: max_profiled,
        total_decisions: decisions,
        every_episode_round_trips: true,
        distinct_fingerprints: fingerprints.len(),
        colliding_episodes: colliding,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> PortSchema {
        PortSchema {
            observations: vec![Port::unit(0), Port::unit(1)],
            actuators: vec![Port::signed(0), Port::signed(1)],
        }
    }

    fn minimal() -> G0Episode {
        G0Episode::new(
            schema(),
            2,
            vec![
                G0Group::one(G0Fact::Boundary(BoundarySubtype::TaskReset)),
                G0Group::one(G0Fact::Observation {
                    key: 0,
                    content: Content::Selection,
                }),
                G0Group::new(vec![
                    G0Fact::ActionQuery {
                        actuator: 0,
                        remaining: 2,
                        selected: true,
                    },
                    G0Fact::ActionQuery {
                        actuator: 1,
                        remaining: 2,
                        selected: false,
                    },
                ]),
                G0Group::one(G0Fact::ActionExecuted { actuator: 0 }),
                G0Group::one(G0Fact::Boundary(BoundarySubtype::EpisodeEnd)),
            ],
        )
    }

    #[test]
    fn a_minimal_transcript_round_trips_through_the_boundary() {
        let evidence = boundary_check(&minimal()).expect("renders");
        assert!(evidence.round_trips);
        assert_eq!(evidence.decisions, 1);
        assert_eq!(evidence.supervised_records, 2);
        assert_eq!(evidence.profiled_records, evidence.records + 1);
    }

    #[test]
    fn supervision_is_addressed_and_never_reaches_a_payload_slot() {
        let episode = minimal();
        let mut flipped = episode.clone();
        for group in &mut flipped.groups {
            for fact in &mut group.facts {
                if let G0Fact::ActionQuery {
                    actuator, selected, ..
                } = fact
                {
                    *selected = *actuator == 1;
                }
            }
        }
        let base = legacy_tokens(&episode).expect("renders");
        let moved = legacy_tokens(&flipped).expect("renders");
        assert_eq!(
            base.iter().map(|t| t.public.clone()).collect::<Vec<_>>(),
            moved.iter().map(|t| t.public.clone()).collect::<Vec<_>>(),
            "moving the teacher's choice must not move one public byte"
        );
        assert_ne!(
            base.iter()
                .map(|t| t.supervision.clone())
                .collect::<Vec<_>>(),
            moved
                .iter()
                .map(|t| t.supervision.clone())
                .collect::<Vec<_>>(),
        );
        assert_eq!(episode.selected_actuators(), vec![vec![0]]);
        assert_eq!(flipped.selected_actuators(), vec![vec![1]]);
    }

    #[test]
    fn all_three_content_readings_survive_one_profile() {
        let mut episode = minimal();
        episode.groups.insert(
            2,
            G0Group::new(vec![
                G0Fact::Goal {
                    key: 1,
                    namespace: KeyNamespace::Observation,
                    content: Content::Selection,
                },
                G0Fact::Observation {
                    key: 1,
                    content: Content::Value {
                        value: 0.25,
                        lower: 0.0,
                        upper: 1.0,
                    },
                },
                G0Fact::Condition {
                    key: 0,
                    namespace: KeyNamespace::Actuator,
                    code: 7,
                    value: 1.0,
                    lower: 0.0,
                    upper: 1.0,
                },
            ]),
        );
        let evidence = boundary_check(&episode).expect("a mixed group round trips");
        assert!(evidence.round_trips);
    }

    #[test]
    fn a_condition_may_name_any_namespace() {
        for namespace in KeyNamespace::ALL {
            let mut episode = minimal();
            episode.groups.insert(
                1,
                G0Group::one(G0Fact::Condition {
                    key: 0,
                    namespace,
                    code: 3,
                    value: 0.0,
                    lower: 0.0,
                    upper: 1.0,
                }),
            );
            let evidence = boundary_check(&episode)
                .unwrap_or_else(|error| panic!("{}: {error}", namespace.as_str()));
            assert!(evidence.round_trips);
        }
    }

    #[test]
    fn undeclared_ports_and_degenerate_decisions_are_refused() {
        let mut stray = minimal();
        stray.groups[3] = G0Group::one(G0Fact::ActionExecuted { actuator: 9 });
        assert_eq!(
            legacy_tokens(&stray),
            Err(RenderFault::UndeclaredActuator { actuator: 9 })
        );

        let mut alone = minimal();
        alone.groups[2] = G0Group::one(G0Fact::ActionQuery {
            actuator: 0,
            remaining: 2,
            selected: true,
        });
        assert_eq!(
            legacy_tokens(&alone),
            Err(RenderFault::SingleAlternative { group: 2 })
        );

        let mut unselected = minimal();
        unselected.groups[2] = G0Group::new(vec![
            G0Fact::ActionQuery {
                actuator: 0,
                remaining: 2,
                selected: false,
            },
            G0Fact::ActionQuery {
                actuator: 1,
                remaining: 2,
                selected: false,
            },
        ]);
        assert_eq!(
            legacy_tokens(&unselected),
            Err(RenderFault::NoSelectedAction { group: 2 })
        );

        let mut duplicated = minimal();
        duplicated.groups[2] = G0Group::new(vec![
            G0Fact::ActionQuery {
                actuator: 0,
                remaining: 2,
                selected: true,
            },
            G0Fact::ActionQuery {
                actuator: 0,
                remaining: 2,
                selected: false,
            },
        ]);
        assert_eq!(
            legacy_tokens(&duplicated),
            Err(RenderFault::DuplicateActuator {
                group: 2,
                actuator: 0
            })
        );

        let mut long = minimal();
        long.horizon = ACTION_HORIZON + 1;
        assert_eq!(
            legacy_tokens(&long),
            Err(RenderFault::HorizonExceedsActionHead {
                horizon: ACTION_HORIZON + 1
            })
        );
    }

    #[test]
    fn the_envelope_declares_the_shared_profile_and_not_the_family() {
        let one = minimal();
        let mut other = minimal();
        other.schema.observations.push(Port::unit(2));
        other.groups.insert(
            1,
            G0Group::one(G0Fact::Observation {
                key: 2,
                content: Content::Value {
                    value: -0.5,
                    lower: -1.0,
                    upper: 1.0,
                },
            }),
        );
        let first = profiled_tokens(&one).expect("renders");
        let second = profiled_tokens(&other).expect("renders");
        assert_eq!(
            first[0].public, second[0].public,
            "the header must not distinguish two families"
        );
        assert_ne!(first[1..], second[1..]);
    }

    #[test]
    fn a_family_report_counts_distinct_renderings() {
        let episodes = vec![minimal(), minimal()];
        let report = rendering_report(&episodes).expect("renders");
        assert_eq!(report.episodes, 2);
        assert_eq!(report.distinct_fingerprints, 1);
        assert_eq!(report.colliding_episodes, vec![vec![0, 1]]);
        assert_eq!(report.envelope_abi, "physical-event-abi-0.3.1");
        assert_eq!(report.canonical_profile, "finite-g0-discrete-0.1.0");
        assert!(report.every_episode_round_trips);
    }
}

#[cfg(test)]
mod indifference_tests {
    use super::*;

    /// A decision the contract is indifferent about marks every correct action.
    #[test]
    fn several_actions_may_be_correct_at_one_decision() {
        let episode = G0Episode::new(
            PortSchema {
                observations: vec![Port::unit(0)],
                actuators: vec![Port::signed(0), Port::signed(1), Port::signed(2)],
            },
            1,
            vec![
                G0Group::one(G0Fact::Boundary(BoundarySubtype::TaskReset)),
                G0Group::one(G0Fact::Observation {
                    key: 0,
                    content: Content::Selection,
                }),
                G0Group::new(vec![
                    G0Fact::ActionQuery {
                        actuator: 0,
                        remaining: 1,
                        selected: true,
                    },
                    G0Fact::ActionQuery {
                        actuator: 1,
                        remaining: 1,
                        selected: true,
                    },
                    G0Fact::ActionQuery {
                        actuator: 2,
                        remaining: 1,
                        selected: false,
                    },
                ]),
                G0Group::one(G0Fact::ActionExecuted { actuator: 0 }),
                G0Group::one(G0Fact::Boundary(BoundarySubtype::EpisodeEnd)),
            ],
        );
        let evidence = boundary_check(&episode).expect("renders");
        assert!(evidence.round_trips);
        assert_eq!(episode.selected_actuators(), vec![vec![0, 1]]);

        let tokens = legacy_tokens(&episode).expect("renders");
        let targets: Vec<f32> = tokens
            .iter()
            .filter(|token| token.public.role == Role::ActionQuery)
            .map(|token| token.supervision.action_target[SCORED_ACTION_SLOT])
            .collect();
        assert_eq!(
            targets,
            vec![
                SELECTED_TARGET as f32,
                SELECTED_TARGET as f32,
                REJECTED_TARGET as f32
            ]
        );
    }
}
