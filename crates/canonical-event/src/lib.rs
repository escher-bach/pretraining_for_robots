//! A canonical, explicitly typed public-event record for robot pretraining.
//!
//! The representation audit recommended one canonical public event record whose
//! fields are named and typed, with supervision structurally separate, and with
//! the existing eight-float layout demoted to a thin renderer of that record.
//! This crate is that specification, written as executable code so the claims
//! are checked rather than asserted.
//!
//! What this crate deliberately does **not** do:
//!
//! - it does not modify `pretraining-world`, `pretraining-world-py`, or the diagnostic
//!   crate, and it is not linked into them;
//! - it does not migrate the world ABI or any checkpoint ABI; and
//! - it does not introduce a byte renderer, a tokenizer, or a model.
//!
//! It renders onto the existing `physical-event-abi-0.2.0` float layout and is
//! tested against that layout's real production output, so adopting it later is
//! a decision about which layer owns meaning, not a rewrite of the wire format.
//!
//! # The central finding
//!
//! `physical-event-abi-0.2.0` is not one self-describing format. The two
//! producers in this repository disagree about what the auxiliary payload slots
//! mean for the same role, and in one case the disagreeing values coincide
//! numerically. A decoder therefore needs a [`Profile`] tag; the ABI version
//! string alone is not enough to recover meaning. See [`Profile::field_schema`]
//! and the profile-collision tests.

pub mod corpus;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Version of the canonical record specified here.
pub const CANONICAL_RECORD_VERSION: &str = "canonical-public-event-0.1.0";
/// The float layout this crate renders onto. Owned by `pretraining-world`.
pub const RENDER_TARGET_ABI: &str = "physical-event-abi-0.2.0";
/// Payload width of the rendered layout.
pub const PAYLOAD_DIM: usize = 8;
/// Width of the learner's action head in the rendered layout.
pub const ACTION_HORIZON: usize = 16;
/// The finite diagnostic's control horizon, needed to read its action-query
/// remaining-step fraction. It is a profile constant, not an ABI constant,
/// which is itself part of the finding.
pub const DIAGNOSTIC_CONTROL_HORIZON: u16 = 2;

const SLOT_VALUE: usize = 0;
const SLOT_LOWER: usize = 1;
const SLOT_UPPER: usize = 2;
const SLOT_AUX0: usize = 3;
const SLOT_AUX1: usize = 4;
const SLOT_PRESENCE: usize = 5;
const SLOT_RESERVED_A: usize = 6;
const SLOT_RESERVED_B: usize = 7;

const ROLE_PAD: u8 = 0;

// ---------------------------------------------------------------------------
// Typed key namespaces
// ---------------------------------------------------------------------------

/// The namespace a local key lives in.
///
/// The rendered layout has a single flat key space: observation channel `0`,
/// actuator channel `0`, and the episode-scalar key `0` all render to the
/// number `0`. The namespace is recoverable from the role, so rendering stays
/// invertible, but a learner that embeds the key slot alone sees one vector for
/// three typed keys. Naming the namespace is what makes a joint renaming well
/// defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum KeyNamespace {
    /// Public observation channels, including future queries about them.
    Observation,
    /// Public actuator channels.
    Actuator,
    /// Episode-scalar keys that name no channel, such as boundaries.
    Episode,
}

impl KeyNamespace {
    pub const ALL: [Self; 3] = [Self::Observation, Self::Actuator, Self::Episode];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Observation => "observation",
            Self::Actuator => "actuator",
            Self::Episode => "episode",
        }
    }

    /// The numeric encoding used by condition records, which are the only
    /// records whose namespace is not recoverable from the role.
    fn code(self) -> f64 {
        match self {
            Self::Observation => 0.0,
            Self::Actuator => 1.0,
            Self::Episode => 2.0,
        }
    }

    fn from_code(code: f64) -> Option<Self> {
        Self::ALL.into_iter().find(|entry| entry.code() == code)
    }
}

/// An encounter-local public key, typed by namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LocalKey {
    pub namespace: KeyNamespace,
    pub name: u16,
}

impl LocalKey {
    pub fn new(namespace: KeyNamespace, name: u16) -> Self {
        Self { namespace, name }
    }
}

// ---------------------------------------------------------------------------
// Quantities, units, and spans
// ---------------------------------------------------------------------------

/// A declared affine unit: `physical = normalized * scale + offset`.
///
/// The rendered layout carries no unit slot. Units are canonical-record
/// metadata; two quantities that differ only in unit render identically. That is
/// a fact about the current ABI, recorded here rather than hidden.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Unit {
    pub label: String,
    pub scale: f64,
    pub offset: f64,
}

impl Unit {
    /// The identity unit: the normalized value is the physical value.
    pub fn normalized() -> Self {
        Self {
            label: "normalized".into(),
            scale: 1.0,
            offset: 0.0,
        }
    }

    pub fn affine(label: &str, scale: f64, offset: f64) -> Result<Self, String> {
        if !scale.is_finite() || !offset.is_finite() || scale == 0.0 {
            return Err("an affine unit needs a finite non-zero scale and a finite offset".into());
        }
        Ok(Self {
            label: label.into(),
            scale,
            offset,
        })
    }

    pub fn to_physical(&self, normalized: f64) -> f64 {
        normalized * self.scale + self.offset
    }

    pub fn from_physical(&self, physical: f64) -> f64 {
        (physical - self.offset) / self.scale
    }

    pub fn is_identity(&self) -> bool {
        self.scale == 1.0 && self.offset == 0.0
    }
}

/// A public numeric quantity: a normalized value, its declared public bounds,
/// and the unit those numbers stand for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Quantity {
    pub value: f64,
    pub lower: f64,
    pub upper: f64,
    pub unit: Unit,
}

impl Quantity {
    pub fn normalized(value: f64, lower: f64, upper: f64) -> Self {
        Self {
            value,
            lower,
            upper,
            unit: Unit::normalized(),
        }
    }

    pub fn with_unit(mut self, unit: Unit) -> Self {
        self.unit = unit;
        self
    }

    /// The value in the declared physical unit.
    pub fn physical(&self) -> f64 {
        self.unit.to_physical(self.value)
    }

    /// True when the numbers are finite and the bounds contain the value.
    pub fn is_well_formed(&self) -> bool {
        self.value.is_finite()
            && self.lower.is_finite()
            && self.upper.is_finite()
            && self.lower <= self.upper
            && self.lower <= self.value
            && self.value <= self.upper
    }
}

/// A number of control steps expressed as a fraction of a stated horizon.
///
/// The layout stores only the fraction. Recovering the step count is exact
/// because the horizon is fixed by the profile; a fraction that is not an exact
/// multiple is a decode error rather than a rounded guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StepSpan {
    pub steps: u16,
    pub of_horizon: u16,
}

impl StepSpan {
    pub fn new(steps: u16, of_horizon: u16) -> Result<Self, String> {
        if of_horizon == 0 {
            return Err("a step span needs a non-zero horizon".into());
        }
        if steps > of_horizon {
            return Err("a step span cannot exceed its horizon".into());
        }
        Ok(Self { steps, of_horizon })
    }

    pub fn fraction(self) -> f64 {
        f64::from(self.steps) / f64::from(self.of_horizon)
    }
}

/// How a boundary is subtyped.
///
/// The layout stores this as `-1.0`, `0.0`, or `1.0` in a numeric slot, which
/// puts three categorical kinds on an ordered line. The canonical record keeps
/// them categorical; the numeric encoding exists only inside the renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BoundarySubtype {
    CalibrationReset,
    TaskReset,
    EpisodeEnd,
}

impl BoundarySubtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CalibrationReset => "calibration_reset",
            Self::TaskReset => "task_reset",
            Self::EpisodeEnd => "episode_end",
        }
    }

    fn code(self) -> f64 {
        match self {
            Self::CalibrationReset => -1.0,
            Self::TaskReset => 0.0,
            Self::EpisodeEnd => 1.0,
        }
    }

    fn from_code(code: f64) -> Option<Self> {
        if code == -1.0 {
            Some(Self::CalibrationReset)
        } else if code == 0.0 {
            Some(Self::TaskReset)
        } else if code == 1.0 {
            Some(Self::EpisodeEnd)
        } else {
            None
        }
    }
}

/// A contract-local categorical name for what a condition record asserts.
///
/// It is deliberately opaque, like a key name. Naming the classes here — a
/// prohibition, a hazard, a restored actuator, a revealed gate — would move
/// card ontology into the record layer, and the finite families disagree about
/// which classes exist. What the layer does fix is that the code is
/// categorical: it is a `u16` in the record and only becomes a number inside
/// the renderer, which is the same treatment [`BoundarySubtype`] receives.
///
/// The rendered layout still puts it in a numeric slot, so a learner embedding
/// that slot sees an ordered line where the contract declares an unordered set.
/// That is a recorded property of `physical-event-abi-0.2.0`, not a claim that
/// the ordering means anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ConditionCode(pub u16);

impl ConditionCode {
    /// The largest code that survives the render/decode round trip exactly.
    ///
    /// `f32` represents every integer below `2^24`; the bound is stated rather
    /// than assumed so a family that grows its condition vocabulary fails here
    /// instead of silently rounding two classes together.
    pub const MAX: u16 = 4096;

    pub fn is_representable(self) -> bool {
        self.0 <= Self::MAX
    }
}

/// Which of the two channel kinds a schema record declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ChannelRole {
    Observation,
    Actuator,
}

/// What a goal or observation record says about its key.
///
/// The two production profiles disagree here, and the disagreement is invisible
/// in the floats: `calibrated-monomial-0.2.0` writes the channel's numeric
/// value, while `goal-conditioned-continuous-control-0.1.0` writes a presence
/// indicator and carries the content in the key identity. Making the two
/// readings distinct canonical variants is the point of this type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChannelContent {
    /// The public numeric value of the named channel.
    Value(Quantity),
    /// The named key is the selected one; the indicator carries no magnitude.
    Selection { indicator: Quantity },
}

impl ChannelContent {
    fn quantity(&self) -> &Quantity {
        match self {
            Self::Value(quantity) => quantity,
            Self::Selection { indicator } => indicator,
        }
    }
}

/// What an action query says about the command it is asking for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QueryHorizon {
    /// The fraction of this episode's control horizon that remains.
    RemainingFraction { remaining: StepSpan },
    /// An actuator marker plus how much of the action head one command covers.
    ActuatorSpan { marker: bool, requested: StepSpan },
}

// ---------------------------------------------------------------------------
// The canonical public fact
// ---------------------------------------------------------------------------

/// The kind of a public fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventKind {
    SchemaObservation,
    SchemaActuator,
    Boundary,
    Condition,
    Goal,
    Observation,
    ActionQuery,
    ActionExecuted,
    FutureQuery,
    Feedback,
}

impl EventKind {
    /// Every kind this crate can represent, in role-code order.
    pub const ALL: [Self; 10] = [
        Self::SchemaObservation,
        Self::SchemaActuator,
        Self::Boundary,
        Self::Condition,
        Self::Goal,
        Self::Observation,
        Self::ActionQuery,
        Self::ActionExecuted,
        Self::FutureQuery,
        Self::Feedback,
    ];

    /// The role code used by `physical-event-abi-0.2.0`.
    ///
    /// Role `0` is padding, which is a batching concern and is deliberately not
    /// representable as a canonical fact.
    pub fn role_code(self) -> u8 {
        match self {
            Self::SchemaObservation => 1,
            Self::SchemaActuator => 2,
            Self::Boundary => 3,
            Self::Condition => 4,
            Self::Goal => 5,
            Self::Observation => 6,
            Self::ActionQuery => 7,
            Self::ActionExecuted => 8,
            Self::FutureQuery => 9,
            Self::Feedback => 10,
        }
    }

    pub fn from_role_code(code: u8) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.role_code() == code)
    }

    /// The namespace of the key this kind names.
    ///
    /// [`EventKind::Condition`] is the one kind whose namespace is not fixed by
    /// its role; the value returned here is only its default. Read
    /// [`PublicFact::namespace`] instead when a fact is available.
    pub fn namespace(self) -> KeyNamespace {
        match self {
            Self::SchemaObservation
            | Self::Condition
            | Self::Goal
            | Self::Observation
            | Self::FutureQuery => KeyNamespace::Observation,
            Self::SchemaActuator | Self::ActionQuery | Self::ActionExecuted => {
                KeyNamespace::Actuator
            }
            Self::Boundary | Self::Feedback => KeyNamespace::Episode,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::SchemaObservation => "schema_observation",
            Self::SchemaActuator => "schema_actuator",
            Self::Boundary => "boundary",
            Self::Condition => "condition",
            Self::Goal => "goal",
            Self::Observation => "observation",
            Self::ActionQuery => "action_query",
            Self::ActionExecuted => "action_executed",
            Self::FutureQuery => "future_query",
            Self::Feedback => "feedback",
        }
    }
}

/// One public fact, with every field named and typed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PublicFact {
    /// Declares one public channel and its schema reference value.
    ChannelSchema {
        channel: ChannelRole,
        reference: Quantity,
        /// Present only for actuator channels, and only in profiles that write
        /// a command span into the schema record.
        command_span: Option<StepSpan>,
    },
    Boundary {
        subtype: BoundarySubtype,
    },
    /// A publicly revealed condition on a named key.
    ///
    /// The namespace travels with the fact rather than being implied by the
    /// role, because the finite families reveal conditions about all three: a
    /// prohibited configuration cell, a restored actuator, and an episode-scalar
    /// gate. A single implied namespace would force two of those three to lie
    /// about what they name.
    Condition {
        namespace: KeyNamespace,
        code: ConditionCode,
        quantity: Quantity,
    },
    Goal {
        content: ChannelContent,
    },
    Observation {
        content: ChannelContent,
    },
    ActionQuery {
        command: Quantity,
        horizon: QueryHorizon,
    },
    ActionExecuted {
        command: Quantity,
        actuator_marker: bool,
    },
    FutureQuery {
        command: Quantity,
        horizon: StepSpan,
    },
    Feedback {
        error: Quantity,
        success: bool,
    },
}

impl PublicFact {
    pub fn kind(&self) -> EventKind {
        match self {
            Self::ChannelSchema { channel, .. } => match channel {
                ChannelRole::Observation => EventKind::SchemaObservation,
                ChannelRole::Actuator => EventKind::SchemaActuator,
            },
            Self::Boundary { .. } => EventKind::Boundary,
            Self::Condition { .. } => EventKind::Condition,
            Self::Goal { .. } => EventKind::Goal,
            Self::Observation { .. } => EventKind::Observation,
            Self::ActionQuery { .. } => EventKind::ActionQuery,
            Self::ActionExecuted { .. } => EventKind::ActionExecuted,
            Self::FutureQuery { .. } => EventKind::FutureQuery,
            Self::Feedback { .. } => EventKind::Feedback,
        }
    }

    /// The namespace this fact names.
    ///
    /// Every kind except [`PublicFact::Condition`] takes it from the role. A
    /// condition carries its own, which is why this is a method on the fact and
    /// not only on the kind.
    pub fn namespace(&self) -> KeyNamespace {
        match self {
            Self::Condition { namespace, .. } => *namespace,
            _ => self.kind().namespace(),
        }
    }
}

/// One public record: a typed key and a typed fact. It has no supervision field
/// and no generator field, and cannot be given one without changing this type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PublicRecord {
    pub key: LocalKey,
    pub fact: PublicFact,
}

impl PublicRecord {
    pub fn new(key: LocalKey, fact: PublicFact) -> Result<Self, String> {
        if key.namespace != fact.namespace() {
            return Err(format!(
                "a {} fact names a {} key, not a {} key",
                fact.kind().as_str(),
                fact.namespace().as_str(),
                key.namespace.as_str()
            ));
        }
        Ok(Self { key, fact })
    }

    /// A deterministic total order used to compare simultaneous record sets.
    ///
    /// It is an ordering, not a magnitude comparison: floats are ordered by bit
    /// pattern so the comparison is total and replay-stable.
    fn order_key(&self) -> (u8, u16, u8, Vec<u64>) {
        let (tag, numbers) = fact_order_payload(&self.fact);
        (
            self.key.namespace as u8,
            self.key.name,
            tag,
            numbers.into_iter().map(f64::to_bits).collect(),
        )
    }
}

fn quantity_numbers(quantity: &Quantity) -> Vec<f64> {
    vec![
        quantity.value,
        quantity.lower,
        quantity.upper,
        quantity.unit.scale,
        quantity.unit.offset,
    ]
}

fn content_numbers(content: &ChannelContent) -> Vec<f64> {
    let mut numbers = quantity_numbers(content.quantity());
    numbers.push(match content {
        ChannelContent::Value(_) => 0.0,
        ChannelContent::Selection { .. } => 1.0,
    });
    numbers
}

fn fact_order_payload(fact: &PublicFact) -> (u8, Vec<f64>) {
    match fact {
        PublicFact::ChannelSchema {
            channel,
            reference,
            command_span,
        } => {
            let mut numbers = quantity_numbers(reference);
            numbers.push(*channel as u8 as f64);
            numbers.push(command_span.map_or(-1.0, StepSpan::fraction));
            (0, numbers)
        }
        PublicFact::Boundary { subtype } => (1, vec![subtype.code()]),
        PublicFact::Condition {
            namespace,
            code,
            quantity,
        } => {
            let mut numbers = quantity_numbers(quantity);
            numbers.push(*namespace as u8 as f64);
            numbers.push(f64::from(code.0));
            (2, numbers)
        }
        PublicFact::Goal { content } => (3, content_numbers(content)),
        PublicFact::Observation { content } => (4, content_numbers(content)),
        PublicFact::ActionQuery { command, horizon } => {
            let mut numbers = quantity_numbers(command);
            match horizon {
                QueryHorizon::RemainingFraction { remaining } => {
                    numbers.push(0.0);
                    numbers.push(remaining.fraction());
                }
                QueryHorizon::ActuatorSpan { marker, requested } => {
                    numbers.push(if *marker { 1.0 } else { 0.0 });
                    numbers.push(requested.fraction());
                }
            }
            (5, numbers)
        }
        PublicFact::ActionExecuted {
            command,
            actuator_marker,
        } => {
            let mut numbers = quantity_numbers(command);
            numbers.push(if *actuator_marker { 1.0 } else { 0.0 });
            (6, numbers)
        }
        PublicFact::FutureQuery { command, horizon } => {
            let mut numbers = quantity_numbers(command);
            numbers.push(horizon.fraction());
            (7, numbers)
        }
        PublicFact::Feedback { error, success } => {
            let mut numbers = quantity_numbers(error);
            numbers.push(if *success { 1.0 } else { 0.0 });
            (8, numbers)
        }
    }
}

// ---------------------------------------------------------------------------
// Groups and episodes
// ---------------------------------------------------------------------------

/// A simultaneity group: the records that share one event index.
///
/// The order of `records` is presentation, not meaning. Two groups with the
/// same record multiset denote the same public fact set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventGroup {
    pub group: u16,
    pub records: Vec<PublicRecord>,
}

/// A complete public episode. It carries no supervision, no seed, no instance
/// index, and no hidden state, because it has nowhere to put them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PublicEpisode {
    pub version: String,
    pub profile: Profile,
    pub groups: Vec<EventGroup>,
}

impl PublicEpisode {
    pub fn new(profile: Profile, groups: Vec<EventGroup>) -> Self {
        Self {
            version: CANONICAL_RECORD_VERSION.into(),
            profile,
            groups,
        }
    }

    pub fn record_count(&self) -> usize {
        self.groups.iter().map(|group| group.records.len()).sum()
    }

    /// The episode with the records inside every group sorted deterministically.
    ///
    /// This removes within-group presentation order and nothing else. It does
    /// **not** remove key names: see
    /// [`PublicEpisode::presentation_free_fingerprint`].
    pub fn canonicalize_within_groups(&self) -> Self {
        let mut canonical = self.clone();
        for group in &mut canonical.groups {
            group.records.sort_by_key(PublicRecord::order_key);
        }
        canonical
    }

    /// A hash of the episode's meaning modulo within-group presentation order.
    ///
    /// This is invariant to record order inside a simultaneity group. It is
    /// deliberately **not** invariant to joint key renaming: a rename-invariant
    /// canonical form is a graph-canonization problem, and claiming one here
    /// would be claiming an invariance this crate does not compute. Renaming is
    /// instead checked as exact renderer equivariance against a stated renaming,
    /// which is decidable.
    pub fn presentation_free_fingerprint(&self) -> u64 {
        let canonical = self.canonicalize_within_groups();
        let mut hash = FNV_OFFSET;
        for group in &canonical.groups {
            fnv_number(&mut hash, f64::from(group.group));
            for record in &canonical_records(group) {
                let (namespace, name, tag, numbers) = record.order_key();
                fnv_number(&mut hash, f64::from(namespace));
                fnv_number(&mut hash, f64::from(name));
                fnv_number(&mut hash, f64::from(tag));
                for number in numbers {
                    fnv_bytes(&mut hash, &number.to_le_bytes());
                }
            }
        }
        hash
    }
}

fn canonical_records(group: &EventGroup) -> Vec<PublicRecord> {
    group.records.clone()
}

// ---------------------------------------------------------------------------
// Supervision, kept structurally separate
// ---------------------------------------------------------------------------

/// The address of one record inside an episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RecordAddress {
    pub group_index: usize,
    pub record_index: usize,
}

/// One supervised action step.
///
/// A step that is not supervised has no entry at all. Absence replaces the
/// parallel boolean mask: an unmasked target cannot be read, because it does not
/// exist.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ActionTarget {
    pub step: usize,
    pub value: f64,
}

/// Privileged teacher supervision attached to one record.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SupervisionRecord {
    pub action_targets: Vec<ActionTarget>,
    pub future_target: Option<f64>,
}

impl SupervisionRecord {
    pub fn is_empty(&self) -> bool {
        self.action_targets.is_empty() && self.future_target.is_none()
    }
}

/// Supervision for a whole episode, addressed by record position.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SupervisionTable {
    pub entries: BTreeMap<RecordAddress, SupervisionRecord>,
}

impl SupervisionTable {
    pub fn set(&mut self, address: RecordAddress, record: SupervisionRecord) {
        if record.is_empty() {
            self.entries.remove(&address);
        } else {
            self.entries.insert(address, record);
        }
    }

    pub fn get(&self, address: RecordAddress) -> Option<&SupervisionRecord> {
        self.entries.get(&address)
    }
}

/// A public episode together with its privileged supervision.
///
/// The two halves are separate values. [`render_public`] takes only a
/// [`PublicEpisode`], so no rendering path can consult supervision even by
/// mistake; that is a property of the signature, not of reviewer discipline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Episode {
    public: PublicEpisode,
    supervision: SupervisionTable,
}

impl Episode {
    pub fn new(public: PublicEpisode, supervision: SupervisionTable) -> Result<Self, String> {
        for address in supervision.entries.keys() {
            let group = public.groups.get(address.group_index).ok_or_else(|| {
                format!("supervision names missing group {}", address.group_index)
            })?;
            if address.record_index >= group.records.len() {
                return Err(format!(
                    "supervision names missing record {} in group {}",
                    address.record_index, address.group_index
                ));
            }
        }
        Ok(Self {
            public,
            supervision,
        })
    }

    pub fn public(&self) -> &PublicEpisode {
        &self.public
    }

    pub fn supervision(&self) -> &SupervisionTable {
        &self.supervision
    }

    /// Permute the records inside every simultaneity group, carrying each
    /// record's supervision with it.
    ///
    /// This is a presentation change. The semantic content is untouched, and the
    /// presentation seed is a separate argument from anything semantic.
    pub fn reorder_within_groups(&self, presentation_seed: u64) -> Self {
        let mut state = presentation_seed ^ 0x5052_4553_454E_545F;
        let mut groups = Vec::with_capacity(self.public.groups.len());
        let mut supervision = SupervisionTable::default();
        for (group_index, group) in self.public.groups.iter().enumerate() {
            let mut order: Vec<usize> = (0..group.records.len()).collect();
            for position in (1..order.len()).rev() {
                let draw = (splitmix64(&mut state) % (position as u64 + 1)) as usize;
                order.swap(position, draw);
            }
            let mut records = Vec::with_capacity(order.len());
            for (new_index, &old_index) in order.iter().enumerate() {
                records.push(group.records[old_index].clone());
                if let Some(entry) = self.supervision.get(RecordAddress {
                    group_index,
                    record_index: old_index,
                }) {
                    supervision.set(
                        RecordAddress {
                            group_index,
                            record_index: new_index,
                        },
                        entry.clone(),
                    );
                }
            }
            groups.push(EventGroup {
                group: group.group,
                records,
            });
        }
        Self {
            public: PublicEpisode {
                version: self.public.version.clone(),
                profile: self.public.profile,
                groups,
            },
            supervision,
        }
    }

    /// Replace every public key name through a bijective renaming.
    ///
    /// Supervision addresses are positions, so they are unchanged: renaming keys
    /// must not move records.
    pub fn rename_keys(&self, renaming: &KeyRenaming) -> Result<Self, String> {
        renaming.validate()?;
        let mut groups = Vec::with_capacity(self.public.groups.len());
        for group in &self.public.groups {
            let mut records = Vec::with_capacity(group.records.len());
            for record in &group.records {
                let mut renamed = record.clone();
                renamed.key.name = renaming.apply(record.key)?;
                records.push(renamed);
            }
            groups.push(EventGroup {
                group: group.group,
                records,
            });
        }
        Ok(Self {
            public: PublicEpisode {
                version: self.public.version.clone(),
                profile: self.public.profile,
                groups,
            },
            supervision: self.supervision.clone(),
        })
    }
}

/// A joint renaming of public key names: one bijection per namespace.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct KeyRenaming {
    pub maps: BTreeMap<KeyNamespace, BTreeMap<u16, u16>>,
}

impl KeyRenaming {
    pub fn with(mut self, namespace: KeyNamespace, pairs: &[(u16, u16)]) -> Self {
        let entry = self.maps.entry(namespace).or_default();
        for (from, to) in pairs {
            entry.insert(*from, *to);
        }
        self
    }

    /// A renaming is valid only when it is injective inside every namespace.
    ///
    /// A non-injective renaming would merge two public keys and silently change
    /// the task, so it is rejected rather than applied.
    pub fn validate(&self) -> Result<(), String> {
        for (namespace, map) in &self.maps {
            let mut seen: Vec<u16> = map.values().copied().collect();
            seen.sort_unstable();
            let before = seen.len();
            seen.dedup();
            if seen.len() != before {
                return Err(format!(
                    "renaming for the {} namespace is not injective",
                    namespace.as_str()
                ));
            }
        }
        Ok(())
    }

    fn apply(&self, key: LocalKey) -> Result<u16, String> {
        match self.maps.get(&key.namespace) {
            None => Ok(key.name),
            Some(map) => map.get(&key.name).copied().ok_or_else(|| {
                format!(
                    "renaming for the {} namespace does not cover key {}",
                    key.namespace.as_str(),
                    key.name
                )
            }),
        }
    }

    /// The inverse renaming, used to check exact equivariance.
    pub fn inverse(&self) -> Result<Self, String> {
        self.validate()?;
        let mut maps = BTreeMap::new();
        for (namespace, map) in &self.maps {
            let mut inverted = BTreeMap::new();
            for (from, to) in map {
                inverted.insert(*to, *from);
            }
            maps.insert(*namespace, inverted);
        }
        Ok(Self { maps })
    }
}

// ---------------------------------------------------------------------------
// Profiles: what the auxiliary slots mean
// ---------------------------------------------------------------------------

/// A producer's reading of `physical-event-abi-0.2.0`.
///
/// The ABI version string does not identify a profile, and the two profiles
/// below disagree about slot meanings for the same role. A decoder that is not
/// told the profile cannot recover meaning; that is the finding this type exists
/// to make unavoidable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Profile {
    /// `pretraining-world`'s calibrated monomial family.
    CalibratedMonomial,
    /// `pretraining-goal-conditioned-world`'s finite goal diagnostic.
    GoalConditionedDiagnostic,
    /// The shared reading used by every finite G0 capability-card family.
    ///
    /// One profile, not one per card. The seed gate requires that the portfolio
    /// families express different process relations *through one
    /// self-describing learner event boundary*; a per-family profile code would
    /// hand the learner family identity for free and defeat the identification
    /// content of trunk T1. So this profile fixes what the slots mean and says
    /// nothing about which family produced a row — the family has to be
    /// inferred from the schema and the events, which is the point.
    ///
    /// It differs from the two legacy profiles in three ways, each forced by a
    /// portfolio requirement rather than chosen:
    ///
    /// 1. goal and observation records carry a **content-kind flag**, because
    ///    card 06 needs channel values while cards 02–05 need key selections and
    ///    both must decode under one profile;
    /// 2. the action query carries **its own horizon**, because the families
    ///    have different budgets and a profile-level horizon constant is exactly
    ///    the non-self-describing property the canonical audit found; and
    /// 3. condition records are emitted, carrying a namespace and a
    ///    contract-local code, because `reveal` is a live construct for cards
    ///    03, 04, and 05.
    FiniteG0,
}

/// What one payload slot means for one kind under one profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotSpec {
    pub slot: usize,
    pub meaning: &'static str,
}

/// The field schema of one kind under one profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KindSchema {
    pub kind: &'static str,
    pub role_code: u8,
    pub namespace: &'static str,
    pub slots: Vec<SlotSpec>,
}

impl Profile {
    pub const ALL: [Self; 3] = [
        Self::CalibratedMonomial,
        Self::GoalConditionedDiagnostic,
        Self::FiniteG0,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::CalibratedMonomial => "calibrated-monomial-0.2.0",
            Self::GoalConditionedDiagnostic => "goal-conditioned-continuous-control-0.1.0",
            Self::FiniteG0 => "finite-g0-discrete-0.1.0",
        }
    }

    /// Whether goal and observation records under this profile declare their
    /// content kind in the row rather than inheriting it from the profile.
    pub fn content_kind_is_declared(self) -> bool {
        matches!(self, Self::FiniteG0)
    }

    /// The complete, explicit field schema.
    ///
    /// Slots `0..=2` are always the value and its declared bounds; slot `5` is
    /// always presence; slots `6` and `7` are reserved and must be zero. Only
    /// the auxiliary slots vary, and that is exactly where the two profiles
    /// disagree.
    pub fn field_schema(self) -> Vec<KindSchema> {
        EventKind::ALL
            .into_iter()
            .filter(|kind| self.emits(*kind))
            .map(|kind| KindSchema {
                kind: kind.as_str(),
                role_code: kind.role_code(),
                namespace: kind.namespace().as_str(),
                slots: self.slot_specs(kind),
            })
            .collect()
    }

    /// Whether this profile's producer ever emits this kind.
    pub fn emits(self, kind: EventKind) -> bool {
        match self {
            // `Condition` is declared by the role space and emitted by nobody.
            Self::CalibratedMonomial => kind != EventKind::Condition,
            Self::GoalConditionedDiagnostic => !matches!(
                kind,
                EventKind::Condition | EventKind::FutureQuery | EventKind::Feedback
            ),
            // The finite families emit no future query and no feedback. Both
            // omissions are deliberate. A future target for a family with
            // hidden state would have to be read off privileged state, which
            // the information boundary forbids as supervision; and terminal
            // outcome feedback would publish, after the fact, exactly the
            // hidden mode or gate that cards 02 and 05 are built to withhold.
            // Emitting either for some families and not others would also make
            // supervision density a family correlate, which the mixture
            // accounting must not have. The action head carries the whole
            // learner signal for this profile.
            Self::FiniteG0 => !matches!(kind, EventKind::FutureQuery | EventKind::Feedback),
        }
    }

    fn slot_specs(self, kind: EventKind) -> Vec<SlotSpec> {
        vec![
            SlotSpec {
                slot: SLOT_VALUE,
                meaning: self.value_meaning(kind),
            },
            SlotSpec {
                slot: SLOT_LOWER,
                meaning: "declared public lower bound",
            },
            SlotSpec {
                slot: SLOT_UPPER,
                meaning: "declared public upper bound",
            },
            SlotSpec {
                slot: SLOT_AUX0,
                meaning: self.aux0_meaning(kind),
            },
            SlotSpec {
                slot: SLOT_AUX1,
                meaning: self.aux1_meaning(kind),
            },
            SlotSpec {
                slot: SLOT_PRESENCE,
                meaning: "presence flag, always 1.0 for a real record",
            },
            SlotSpec {
                slot: SLOT_RESERVED_A,
                meaning: "reserved, must be 0.0",
            },
            SlotSpec {
                slot: SLOT_RESERVED_B,
                meaning: "reserved, must be 0.0",
            },
        ]
    }

    fn value_meaning(self, kind: EventKind) -> &'static str {
        match (self, kind) {
            (_, EventKind::Boundary) => "categorical boundary subtype encoded as -1.0/0.0/1.0",
            (Self::CalibratedMonomial, EventKind::Goal | EventKind::Observation) => {
                "normalized value of the named observation channel"
            }
            (Self::GoalConditionedDiagnostic, EventKind::Goal | EventKind::Observation) => {
                "presence indicator; the content is the key identity"
            }
            (Self::FiniteG0, EventKind::Goal | EventKind::Observation) => {
                "channel value, or a selection indicator; slot 3 says which"
            }
            (Self::FiniteG0, EventKind::Condition) => "normalized value of the revealed condition",
            (Self::CalibratedMonomial, EventKind::Feedback) => "half the maximum public error",
            (_, EventKind::SchemaObservation) => "schema reference value for the channel",
            (_, EventKind::SchemaActuator) => "schema reference command, always 0.0",
            (_, EventKind::ActionQuery | EventKind::FutureQuery) => "placeholder, always 0.0",
            (_, EventKind::ActionExecuted) => "normalized executed command",
            (_, EventKind::Condition) => "normalized value of the named condition",
            (_, EventKind::Feedback) => "normalized public error",
        }
    }

    fn aux0_meaning(self, kind: EventKind) -> &'static str {
        match (self, kind) {
            (
                Self::CalibratedMonomial,
                EventKind::SchemaActuator | EventKind::ActionQuery | EventKind::ActionExecuted,
            ) => "actuator marker, always 1.0",
            (Self::CalibratedMonomial, EventKind::FutureQuery) => {
                "requested horizon as a fraction of the action head"
            }
            (Self::CalibratedMonomial, EventKind::Feedback) => "success bit",
            (
                Self::GoalConditionedDiagnostic,
                EventKind::SchemaActuator | EventKind::ActionExecuted,
            ) => "actuator marker, always 1.0",
            (Self::GoalConditionedDiagnostic, EventKind::ActionQuery) => {
                "control steps remaining as a fraction of the episode horizon"
            }
            (Self::FiniteG0, EventKind::SchemaActuator | EventKind::ActionExecuted) => {
                "actuator marker, always 1.0"
            }
            (Self::FiniteG0, EventKind::Goal | EventKind::Observation) => {
                "content kind: 0.0 is a channel value, 1.0 is a key selection"
            }
            (Self::FiniteG0, EventKind::Condition) => "contract-local condition code",
            (Self::FiniteG0, EventKind::ActionQuery) => {
                "control steps remaining as a fraction of the action head"
            }
            _ => "unused, always 0.0",
        }
    }

    fn aux1_meaning(self, kind: EventKind) -> &'static str {
        match (self, kind) {
            (
                Self::CalibratedMonomial,
                EventKind::SchemaActuator | EventKind::ActionQuery | EventKind::ActionExecuted,
            ) => "requested command steps as a fraction of the action head",
            (Self::FiniteG0, EventKind::ActionQuery) => {
                "this episode's control horizon as a fraction of the action head"
            }
            (Self::FiniteG0, EventKind::Condition) => {
                "namespace of the named key: 0.0 observation, 1.0 actuator, 2.0 episode"
            }
            _ => "unused, always 0.0",
        }
    }

    /// A hash over the complete field schema, not merely over a role count and a
    /// payload width. Changing what any slot means changes this value.
    pub fn schema_hash(self) -> u64 {
        let mut hash = FNV_OFFSET;
        fnv_bytes(&mut hash, CANONICAL_RECORD_VERSION.as_bytes());
        fnv_bytes(&mut hash, RENDER_TARGET_ABI.as_bytes());
        fnv_bytes(&mut hash, self.as_str().as_bytes());
        fnv_number(&mut hash, PAYLOAD_DIM as f64);
        fnv_number(&mut hash, ACTION_HORIZON as f64);
        for kind_schema in self.field_schema() {
            fnv_bytes(&mut hash, kind_schema.kind.as_bytes());
            fnv_number(&mut hash, f64::from(kind_schema.role_code));
            fnv_bytes(&mut hash, kind_schema.namespace.as_bytes());
            for slot in kind_schema.slots {
                fnv_number(&mut hash, slot.slot as f64);
                fnv_bytes(&mut hash, slot.meaning.as_bytes());
            }
        }
        hash
    }
}

// ---------------------------------------------------------------------------
// The rendered float layout
// ---------------------------------------------------------------------------

/// One row of `physical-event-abi-0.2.0`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PublicRow {
    pub role: u8,
    pub key: u16,
    pub group: u16,
    pub payload: [f32; PAYLOAD_DIM],
}

/// Rendered supervision for one row, in the production parallel-array form.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SupervisionRow {
    pub action_target: [f32; ACTION_HORIZON],
    pub action_mask: [bool; ACTION_HORIZON],
    pub future_target: f32,
    pub future_mask: bool,
}

impl Default for SupervisionRow {
    fn default() -> Self {
        Self {
            action_target: [0.0; ACTION_HORIZON],
            action_mask: [false; ACTION_HORIZON],
            future_target: 0.0,
            future_mask: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderError {
    /// A value cannot be written into the layout without losing precision.
    NotExactlyRepresentable { field: String, value: String },
    /// The episode does not fit the declared context and would be truncated.
    CapacityExceeded { produced: usize, capacity: usize },
    /// The record is malformed for its profile.
    Unsupported { detail: String },
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotExactlyRepresentable { field, value } => write!(
                formatter,
                "{field} value {value} is not exactly representable in the rendered layout"
            ),
            Self::CapacityExceeded { produced, capacity } => write!(
                formatter,
                "rendering produced {produced} rows but the declared capacity is {capacity}"
            ),
            Self::Unsupported { detail } => write!(formatter, "{detail}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecodeError {
    UnknownRole {
        role: u8,
    },
    /// Padding is a batching concern and is not part of an episode.
    PaddingRow {
        index: usize,
    },
    ProfileDoesNotEmit {
        profile: &'static str,
        kind: &'static str,
    },
    ReservedSlotNotZero {
        index: usize,
        slot: usize,
        value: String,
    },
    RecordNotPresent {
        index: usize,
        value: String,
    },
    NonContiguousGroups {
        index: usize,
        expected: u16,
        found: u16,
    },
    UnexpectedSlotValue {
        index: usize,
        slot: usize,
        value: String,
        expected: &'static str,
    },
    Empty,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownRole { role } => write!(formatter, "role {role} is not a public fact"),
            Self::PaddingRow { index } => write!(
                formatter,
                "row {index} is padding; padding is a batching concern, not an episode record"
            ),
            Self::ProfileDoesNotEmit { profile, kind } => {
                write!(formatter, "profile {profile} never emits a {kind} record")
            }
            Self::ReservedSlotNotZero { index, slot, value } => write!(
                formatter,
                "row {index} reserved slot {slot} holds {value}; reserved slots must be 0.0"
            ),
            Self::RecordNotPresent { index, value } => write!(
                formatter,
                "row {index} has presence flag {value}; a real record must set it to 1.0"
            ),
            Self::NonContiguousGroups {
                index,
                expected,
                found,
            } => write!(
                formatter,
                "row {index} opens group {found} but the next group must be {expected}"
            ),
            Self::UnexpectedSlotValue {
                index,
                slot,
                value,
                expected,
            } => write!(
                formatter,
                "row {index} slot {slot} holds {value}; expected {expected}"
            ),
            Self::Empty => write!(formatter, "an episode needs at least one record"),
        }
    }
}

fn exact_f32(field: &str, value: f64) -> Result<f32, RenderError> {
    let narrowed = value as f32;
    if f64::from(narrowed) == value {
        Ok(narrowed)
    } else {
        Err(RenderError::NotExactlyRepresentable {
            field: field.into(),
            value: format!("{value}"),
        })
    }
}

fn render_payload(
    value: f64,
    lower: f64,
    upper: f64,
    aux0: f64,
    aux1: f64,
    field: &str,
) -> Result<[f32; PAYLOAD_DIM], RenderError> {
    let mut payload = [0.0f32; PAYLOAD_DIM];
    payload[SLOT_VALUE] = exact_f32(&format!("{field}.value"), value)?;
    payload[SLOT_LOWER] = exact_f32(&format!("{field}.lower"), lower)?;
    payload[SLOT_UPPER] = exact_f32(&format!("{field}.upper"), upper)?;
    payload[SLOT_AUX0] = exact_f32(&format!("{field}.aux0"), aux0)?;
    payload[SLOT_AUX1] = exact_f32(&format!("{field}.aux1"), aux1)?;
    payload[SLOT_PRESENCE] = 1.0;
    Ok(payload)
}

/// Render a public episode onto the float layout.
///
/// The signature takes only the public half of an episode. Supervision is not
/// reachable from here.
pub fn render_public(episode: &PublicEpisode) -> Result<Vec<PublicRow>, RenderError> {
    let profile = episode.profile;
    let mut rows = Vec::with_capacity(episode.record_count());
    for group in &episode.groups {
        for record in &group.records {
            let kind = record.fact.kind();
            if !profile.emits(kind) {
                return Err(RenderError::Unsupported {
                    detail: format!(
                        "profile {} never emits a {} record",
                        profile.as_str(),
                        kind.as_str()
                    ),
                });
            }
            if record.key.namespace != record.fact.namespace() {
                return Err(RenderError::Unsupported {
                    detail: format!(
                        "a {} fact names a {} key",
                        kind.as_str(),
                        record.key.namespace.as_str()
                    ),
                });
            }
            let payload = render_fact(profile, &record.fact)?;
            rows.push(PublicRow {
                role: kind.role_code(),
                key: record.key.name,
                group: group.group,
                payload,
            });
        }
    }
    Ok(rows)
}

/// Render with a declared context capacity.
///
/// Exceeding the capacity is an error. Nothing is truncated, because a silently
/// shortened episode is a changed task.
pub fn render_public_with_capacity(
    episode: &PublicEpisode,
    capacity: usize,
) -> Result<Vec<PublicRow>, RenderError> {
    let rows = render_public(episode)?;
    if rows.len() > capacity {
        return Err(RenderError::CapacityExceeded {
            produced: rows.len(),
            capacity,
        });
    }
    Ok(rows)
}

fn render_fact(profile: Profile, fact: &PublicFact) -> Result<[f32; PAYLOAD_DIM], RenderError> {
    match fact {
        PublicFact::ChannelSchema {
            channel,
            reference,
            command_span,
        } => {
            let actuator = matches!(channel, ChannelRole::Actuator);
            if !actuator && command_span.is_some() {
                return Err(RenderError::Unsupported {
                    detail: "an observation channel schema cannot declare a command span".into(),
                });
            }
            render_payload(
                reference.value,
                reference.lower,
                reference.upper,
                if actuator { 1.0 } else { 0.0 },
                command_span.map_or(0.0, StepSpan::fraction),
                "channel_schema",
            )
        }
        PublicFact::Boundary { subtype } => {
            render_payload(subtype.code(), -1.0, 1.0, 0.0, 0.0, "boundary")
        }
        PublicFact::Condition {
            namespace,
            code,
            quantity,
        } => {
            // A condition shares its role with the `0.3.x` profile declaration,
            // and that declaration's payload states a lower bound above its
            // upper bound. Requiring a well-formed quantity here is therefore
            // not housekeeping: it is what makes the header signature
            // unreachable for any renderable condition record, so the envelope
            // can admit conditions without admitting an ambiguity.
            if !quantity.is_well_formed() {
                return Err(RenderError::Unsupported {
                    detail: format!(
                        "a condition quantity must be finite with lower <= value <= upper,                          got value {} in [{}, {}]",
                        quantity.value, quantity.lower, quantity.upper
                    ),
                });
            }
            if !code.is_representable() {
                return Err(RenderError::Unsupported {
                    detail: format!(
                        "condition code {} exceeds the exactly representable bound {}",
                        code.0,
                        ConditionCode::MAX
                    ),
                });
            }
            render_payload(
                quantity.value,
                quantity.lower,
                quantity.upper,
                f64::from(code.0),
                namespace.code(),
                "condition",
            )
        }
        PublicFact::Goal { content } | PublicFact::Observation { content } => {
            let is_selection = matches!(content, ChannelContent::Selection { .. });
            let aux0 = if profile.content_kind_is_declared() {
                // The row says which reading applies, so one profile can carry
                // both. This is the only place the two readings coexist.
                if is_selection {
                    1.0
                } else {
                    0.0
                }
            } else {
                let expects_selection = matches!(profile, Profile::GoalConditionedDiagnostic);
                if is_selection != expects_selection {
                    return Err(RenderError::Unsupported {
                        detail: format!(
                            "profile {} reads goal and observation records as {}",
                            profile.as_str(),
                            if expects_selection {
                                "key selections"
                            } else {
                                "channel values"
                            }
                        ),
                    });
                }
                0.0
            };
            let quantity = content.quantity();
            render_payload(
                quantity.value,
                quantity.lower,
                quantity.upper,
                aux0,
                0.0,
                "channel_content",
            )
        }
        PublicFact::ActionQuery { command, horizon } => {
            let (aux0, aux1) = match (profile, horizon) {
                (
                    Profile::GoalConditionedDiagnostic,
                    QueryHorizon::RemainingFraction { remaining },
                ) => (remaining.fraction(), 0.0),
                (Profile::CalibratedMonomial, QueryHorizon::ActuatorSpan { marker, requested }) => {
                    (if *marker { 1.0 } else { 0.0 }, requested.fraction())
                }
                (Profile::FiniteG0, QueryHorizon::RemainingFraction { remaining }) => {
                    // Both numbers are expressed against the action head rather
                    // than against the episode horizon, so the row is decodable
                    // without a per-family constant. That constant is the
                    // non-self-describing property the canonical audit found in
                    // the legacy diagnostic profile.
                    if usize::from(remaining.of_horizon) > ACTION_HORIZON {
                        return Err(RenderError::Unsupported {
                            detail: format!(
                                "a finite-G0 episode horizon of {} exceeds the action head",
                                remaining.of_horizon
                            ),
                        });
                    }
                    (
                        f64::from(remaining.steps) / ACTION_HORIZON as f64,
                        f64::from(remaining.of_horizon) / ACTION_HORIZON as f64,
                    )
                }
                _ => {
                    return Err(RenderError::Unsupported {
                        detail: format!(
                            "profile {} does not use this action-query horizon form",
                            profile.as_str()
                        ),
                    })
                }
            };
            render_payload(
                command.value,
                command.lower,
                command.upper,
                aux0,
                aux1,
                "action_query",
            )
        }
        PublicFact::ActionExecuted {
            command,
            actuator_marker,
        } => {
            let aux1 = match profile {
                Profile::CalibratedMonomial => 1.0 / ACTION_HORIZON as f64,
                Profile::GoalConditionedDiagnostic | Profile::FiniteG0 => 0.0,
            };
            render_payload(
                command.value,
                command.lower,
                command.upper,
                if *actuator_marker { 1.0 } else { 0.0 },
                aux1,
                "action_executed",
            )
        }
        PublicFact::FutureQuery { command, horizon } => render_payload(
            command.value,
            command.lower,
            command.upper,
            horizon.fraction(),
            0.0,
            "future_query",
        ),
        PublicFact::Feedback { error, success } => render_payload(
            error.value,
            error.lower,
            error.upper,
            if *success { 1.0 } else { 0.0 },
            0.0,
            "feedback",
        ),
    }
}

/// Render supervision into the production parallel-array form.
pub fn render_supervision(episode: &Episode) -> Result<Vec<SupervisionRow>, RenderError> {
    let mut rows = Vec::with_capacity(episode.public.record_count());
    for (group_index, group) in episode.public.groups.iter().enumerate() {
        for record_index in 0..group.records.len() {
            let mut row = SupervisionRow::default();
            if let Some(entry) = episode.supervision.get(RecordAddress {
                group_index,
                record_index,
            }) {
                for target in &entry.action_targets {
                    if target.step >= ACTION_HORIZON {
                        return Err(RenderError::Unsupported {
                            detail: format!(
                                "action target step {} exceeds the action head",
                                target.step
                            ),
                        });
                    }
                    row.action_target[target.step] = exact_f32("action_target", target.value)?;
                    row.action_mask[target.step] = true;
                }
                if let Some(future) = entry.future_target {
                    row.future_target = exact_f32("future_target", future)?;
                    row.future_mask = true;
                }
            }
            rows.push(row);
        }
    }
    Ok(rows)
}

/// Recover supervision from the production parallel-array form.
pub fn decode_supervision(
    episode: &PublicEpisode,
    rows: &[SupervisionRow],
) -> Result<SupervisionTable, String> {
    if rows.len() != episode.record_count() {
        return Err(format!(
            "expected {} supervision rows, found {}",
            episode.record_count(),
            rows.len()
        ));
    }
    let mut table = SupervisionTable::default();
    let mut cursor = 0usize;
    for (group_index, group) in episode.groups.iter().enumerate() {
        for record_index in 0..group.records.len() {
            let row = &rows[cursor];
            cursor += 1;
            let mut entry = SupervisionRecord::default();
            for step in 0..ACTION_HORIZON {
                if row.action_mask[step] {
                    entry.action_targets.push(ActionTarget {
                        step,
                        value: f64::from(row.action_target[step]),
                    });
                }
            }
            if row.future_mask {
                entry.future_target = Some(f64::from(row.future_target));
            }
            table.set(
                RecordAddress {
                    group_index,
                    record_index,
                },
                entry,
            );
        }
    }
    Ok(table)
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// Recover a canonical episode from rendered rows under a stated profile.
///
/// Every departure from the profile's field schema is an error. The decoder
/// never guesses a meaning and never fills in a default for a slot it did not
/// understand.
pub fn decode_episode(profile: Profile, rows: &[PublicRow]) -> Result<PublicEpisode, DecodeError> {
    if rows.is_empty() {
        return Err(DecodeError::Empty);
    }
    let mut groups: Vec<EventGroup> = Vec::new();
    let mut expected_group = 0u16;
    for (index, row) in rows.iter().enumerate() {
        if row.role == ROLE_PAD {
            return Err(DecodeError::PaddingRow { index });
        }
        let kind = EventKind::from_role_code(row.role)
            .ok_or(DecodeError::UnknownRole { role: row.role })?;
        if !profile.emits(kind) {
            return Err(DecodeError::ProfileDoesNotEmit {
                profile: profile.as_str(),
                kind: kind.as_str(),
            });
        }
        for slot in [SLOT_RESERVED_A, SLOT_RESERVED_B] {
            if row.payload[slot] != 0.0 {
                return Err(DecodeError::ReservedSlotNotZero {
                    index,
                    slot,
                    value: format!("{}", row.payload[slot]),
                });
            }
        }
        if row.payload[SLOT_PRESENCE] != 1.0 {
            return Err(DecodeError::RecordNotPresent {
                index,
                value: format!("{}", row.payload[SLOT_PRESENCE]),
            });
        }
        match groups.last() {
            Some(open) if open.group == row.group => {}
            _ => {
                if row.group != expected_group {
                    return Err(DecodeError::NonContiguousGroups {
                        index,
                        expected: expected_group,
                        found: row.group,
                    });
                }
                groups.push(EventGroup {
                    group: row.group,
                    records: Vec::new(),
                });
                expected_group = expected_group.saturating_add(1);
            }
        }
        let fact = decode_fact(profile, kind, index, row)?;
        // The namespace comes from the fact rather than the kind, because a
        // condition record carries its own and every other kind's fact reports
        // the role's namespace unchanged.
        let namespace = fact.namespace();
        groups
            .last_mut()
            .expect("a group was just opened")
            .records
            .push(PublicRecord {
                key: LocalKey::new(namespace, row.key),
                fact,
            });
    }
    Ok(PublicEpisode::new(profile, groups))
}

fn quantity_from(row: &PublicRow) -> Quantity {
    Quantity::normalized(
        f64::from(row.payload[SLOT_VALUE]),
        f64::from(row.payload[SLOT_LOWER]),
        f64::from(row.payload[SLOT_UPPER]),
    )
}

fn span_from_fraction(
    index: usize,
    slot: usize,
    fraction: f32,
    horizon: u16,
) -> Result<StepSpan, DecodeError> {
    let steps = f64::from(fraction) * f64::from(horizon);
    let rounded = steps.round();
    if (steps - rounded).abs() > 1e-6 || rounded < 0.0 || rounded > f64::from(horizon) {
        return Err(DecodeError::UnexpectedSlotValue {
            index,
            slot,
            value: format!("{fraction}"),
            expected: "an exact multiple of one step of the declared horizon",
        });
    }
    StepSpan::new(rounded as u16, horizon).map_err(|_| DecodeError::UnexpectedSlotValue {
        index,
        slot,
        value: format!("{fraction}"),
        expected: "a step count within the declared horizon",
    })
}

fn expect_flag(index: usize, slot: usize, value: f32) -> Result<bool, DecodeError> {
    if value == 0.0 {
        Ok(false)
    } else if value == 1.0 {
        Ok(true)
    } else {
        Err(DecodeError::UnexpectedSlotValue {
            index,
            slot,
            value: format!("{value}"),
            expected: "0.0 or 1.0",
        })
    }
}

fn expect_zero(index: usize, slot: usize, value: f32) -> Result<(), DecodeError> {
    if value == 0.0 {
        Ok(())
    } else {
        Err(DecodeError::UnexpectedSlotValue {
            index,
            slot,
            value: format!("{value}"),
            expected: "0.0 for a slot this profile does not use",
        })
    }
}

fn decode_fact(
    profile: Profile,
    kind: EventKind,
    index: usize,
    row: &PublicRow,
) -> Result<PublicFact, DecodeError> {
    let aux0 = row.payload[SLOT_AUX0];
    let aux1 = row.payload[SLOT_AUX1];
    match kind {
        EventKind::SchemaObservation => {
            expect_zero(index, SLOT_AUX0, aux0)?;
            expect_zero(index, SLOT_AUX1, aux1)?;
            Ok(PublicFact::ChannelSchema {
                channel: ChannelRole::Observation,
                reference: quantity_from(row),
                command_span: None,
            })
        }
        EventKind::SchemaActuator => {
            if !expect_flag(index, SLOT_AUX0, aux0)? {
                return Err(DecodeError::UnexpectedSlotValue {
                    index,
                    slot: SLOT_AUX0,
                    value: format!("{aux0}"),
                    expected: "1.0, the actuator marker",
                });
            }
            let command_span = match profile {
                Profile::CalibratedMonomial => Some(span_from_fraction(
                    index,
                    SLOT_AUX1,
                    aux1,
                    ACTION_HORIZON as u16,
                )?),
                Profile::GoalConditionedDiagnostic | Profile::FiniteG0 => {
                    expect_zero(index, SLOT_AUX1, aux1)?;
                    None
                }
            };
            Ok(PublicFact::ChannelSchema {
                channel: ChannelRole::Actuator,
                reference: quantity_from(row),
                command_span,
            })
        }
        EventKind::Boundary => {
            expect_zero(index, SLOT_AUX0, aux0)?;
            expect_zero(index, SLOT_AUX1, aux1)?;
            let subtype = BoundarySubtype::from_code(f64::from(row.payload[SLOT_VALUE])).ok_or(
                DecodeError::UnexpectedSlotValue {
                    index,
                    slot: SLOT_VALUE,
                    value: format!("{}", row.payload[SLOT_VALUE]),
                    expected: "-1.0, 0.0, or 1.0",
                },
            )?;
            Ok(PublicFact::Boundary { subtype })
        }
        EventKind::Condition => {
            let code = f64::from(aux0);
            if code < 0.0 || code.fract() != 0.0 || code > f64::from(ConditionCode::MAX) {
                return Err(DecodeError::UnexpectedSlotValue {
                    index,
                    slot: SLOT_AUX0,
                    value: format!("{aux0}"),
                    expected: "a whole condition code within the declared bound",
                });
            }
            let namespace = KeyNamespace::from_code(f64::from(aux1)).ok_or(
                DecodeError::UnexpectedSlotValue {
                    index,
                    slot: SLOT_AUX1,
                    value: format!("{aux1}"),
                    expected: "0.0, 1.0, or 2.0, a key namespace",
                },
            )?;
            let quantity = quantity_from(row);
            // Symmetric with the renderer, and load-bearing rather than tidy.
            // The `0.3.x` profile declaration shares this role and states a
            // lower bound above its upper bound, so refusing a malformed
            // quantity here is what keeps a consumer that skipped the envelope
            // from silently reading a header as a public condition.
            if !quantity.is_well_formed() {
                return Err(DecodeError::UnexpectedSlotValue {
                    index,
                    slot: SLOT_LOWER,
                    value: format!("{}", row.payload[SLOT_LOWER]),
                    expected: "a finite lower bound at or below the value and its upper bound",
                });
            }
            Ok(PublicFact::Condition {
                namespace,
                code: ConditionCode(code as u16),
                quantity,
            })
        }
        EventKind::Goal | EventKind::Observation => {
            expect_zero(index, SLOT_AUX1, aux1)?;
            let quantity = quantity_from(row);
            let content = if profile.content_kind_is_declared() {
                if expect_flag(index, SLOT_AUX0, aux0)? {
                    ChannelContent::Selection {
                        indicator: quantity,
                    }
                } else {
                    ChannelContent::Value(quantity)
                }
            } else {
                expect_zero(index, SLOT_AUX0, aux0)?;
                match profile {
                    Profile::CalibratedMonomial => ChannelContent::Value(quantity),
                    _ => ChannelContent::Selection {
                        indicator: quantity,
                    },
                }
            };
            Ok(if kind == EventKind::Goal {
                PublicFact::Goal { content }
            } else {
                PublicFact::Observation { content }
            })
        }
        EventKind::ActionQuery => {
            let horizon = match profile {
                Profile::CalibratedMonomial => QueryHorizon::ActuatorSpan {
                    marker: expect_flag(index, SLOT_AUX0, aux0)?,
                    requested: span_from_fraction(index, SLOT_AUX1, aux1, ACTION_HORIZON as u16)?,
                },
                Profile::GoalConditionedDiagnostic => {
                    expect_zero(index, SLOT_AUX1, aux1)?;
                    QueryHorizon::RemainingFraction {
                        remaining: span_from_fraction(
                            index,
                            SLOT_AUX0,
                            aux0,
                            DIAGNOSTIC_CONTROL_HORIZON,
                        )?,
                    }
                }
                Profile::FiniteG0 => {
                    let head = ACTION_HORIZON as u16;
                    let horizon = span_from_fraction(index, SLOT_AUX1, aux1, head)?;
                    if horizon.steps == 0 {
                        return Err(DecodeError::UnexpectedSlotValue {
                            index,
                            slot: SLOT_AUX1,
                            value: format!("{aux1}"),
                            expected: "a non-zero episode control horizon",
                        });
                    }
                    let remaining = span_from_fraction(index, SLOT_AUX0, aux0, head)?;
                    QueryHorizon::RemainingFraction {
                        remaining: StepSpan::new(remaining.steps, horizon.steps).map_err(|_| {
                            DecodeError::UnexpectedSlotValue {
                                index,
                                slot: SLOT_AUX0,
                                value: format!("{aux0}"),
                                expected: "remaining steps within this episode's horizon",
                            }
                        })?,
                    }
                }
            };
            Ok(PublicFact::ActionQuery {
                command: quantity_from(row),
                horizon,
            })
        }
        EventKind::ActionExecuted => {
            let actuator_marker = expect_flag(index, SLOT_AUX0, aux0)?;
            match profile {
                Profile::CalibratedMonomial => {
                    span_from_fraction(index, SLOT_AUX1, aux1, ACTION_HORIZON as u16)?;
                }
                Profile::GoalConditionedDiagnostic | Profile::FiniteG0 => {
                    expect_zero(index, SLOT_AUX1, aux1)?
                }
            }
            Ok(PublicFact::ActionExecuted {
                command: quantity_from(row),
                actuator_marker,
            })
        }
        EventKind::FutureQuery => {
            expect_zero(index, SLOT_AUX1, aux1)?;
            Ok(PublicFact::FutureQuery {
                command: quantity_from(row),
                horizon: span_from_fraction(index, SLOT_AUX0, aux0, ACTION_HORIZON as u16)?,
            })
        }
        EventKind::Feedback => {
            expect_zero(index, SLOT_AUX1, aux1)?;
            Ok(PublicFact::Feedback {
                error: quantity_from(row),
                success: expect_flag(index, SLOT_AUX0, aux0)?,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Hashing and presentation helpers
// ---------------------------------------------------------------------------

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
    *hash ^= 0xff;
    *hash = hash.wrapping_mul(FNV_PRIME);
}

fn fnv_number(hash: &mut u64, value: f64) {
    fnv_bytes(hash, &value.to_bits().to_le_bytes());
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
