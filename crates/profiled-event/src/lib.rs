//! A reversible, learner-visible interpretation envelope for physical events.
//!
//! `physical-event-abi-0.2.0` fixes tensor shape but does not identify what all
//! payload slots mean. This crate leaves that legacy body byte-for-byte intact
//! and prefixes one public record selecting the decoder that gives it meaning.
//! The adapter contains no world state, task answer, seed, or supervision.

use std::fmt;

use pretraining_canonical_event::Profile as CanonicalProfile;
use pretraining_world::{LearningToken, PublicToken, Role, Supervision, PAYLOAD_DIM};

/// The unchanged layout accepted and produced by the existing generators.
pub const LEGACY_TOKEN_ABI_VERSION: &str = pretraining_world::TOKEN_ABI_VERSION;

/// The opt-in envelope defined by this crate.
///
/// `0.3.1` widens what a body may contain and adds a profile code; it does not
/// change the wire layout. The patch bump is not cosmetic — a `0.3.0` consumer
/// would reject both additions, so the admitted set is consumer-visible and has
/// to be named.
pub const PROFILED_TOKEN_ABI_VERSION: &str = "physical-event-abi-0.3.1";

/// The role the profile declaration uses.
///
/// It was chosen at `0.3.0` because no `0.2.0` producer emitted it, and the
/// body guard was correspondingly blunt: any `Condition` record was refused.
/// The finite G0 families broke that, because `reveal` is a live construct for
/// cards 03, 04, and 05 and a revealed condition is exactly what it emits.
///
/// The guard is therefore narrowed to the property it was actually protecting:
/// a body record must not be *indistinguishable from a header*. That is a
/// three-part signature — this role, event `0`, and the exact tag payload — and
/// [`validate_no_header_collision`] refuses only that. The narrowing is safe by
/// construction rather than by inspection: the tag payload declares a lower
/// bound of `3.0` above an upper bound of `1.0`, and a condition record's
/// quantity is required to be well formed, so no renderable condition can carry
/// it. `header_signature_is_unreachable_for_a_well_formed_condition` checks that
/// claim rather than restating it.
pub const PROFILE_TAG_ROLE: Role = Role::Condition;
pub const PROFILE_TAG_EVENT: u16 = 0;

/// Slots 0..=2 are the semantic version tuple. Slot 5 is the usual presence
/// bit. The other slots are fixed at zero so corruption is detectable.
pub const PROFILE_TAG_PAYLOAD: [f32; PAYLOAD_DIM] = [0.0, 3.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

/// A meaning system for the legacy rows, rather than a world or generator ID.
///
/// The categorical code travels in the public key field of the profile record.
/// It is not written into a continuous payload coordinate.
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InterpretationProfile {
    /// Goals and observations are channel values; action-query auxiliaries are
    /// an actuator marker and requested command span.
    ChannelValuesWithRequestedSpan = 1,
    /// Goals and observations select named keys; the action-query auxiliary is
    /// the fraction of the episode horizon remaining.
    KeySelectionsWithRemainingHorizon = 2,
    /// The shared reading used by every finite G0 capability-card family: goal
    /// and observation records declare their own content kind, the action query
    /// carries its own horizon, and condition records are emitted.
    ///
    /// One code for the whole portfolio, not one per card. A per-card code
    /// would publish family identity, and the seed gate requires the families to
    /// be distinguished by their process relations through a single boundary.
    FiniteG0Discrete = 3,
}

impl InterpretationProfile {
    pub const ALL: [Self; 3] = [
        Self::ChannelValuesWithRequestedSpan,
        Self::KeySelectionsWithRemainingHorizon,
        Self::FiniteG0Discrete,
    ];

    pub const fn code(self) -> u16 {
        self as u16
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ChannelValuesWithRequestedSpan => "channel-values-with-requested-span",
            Self::KeySelectionsWithRemainingHorizon => "key-selections-with-remaining-horizon",
            Self::FiniteG0Discrete => "finite-g0-discrete",
        }
    }

    /// The canonical decoder selected by this public declaration.
    pub const fn canonical_profile(self) -> CanonicalProfile {
        match self {
            Self::ChannelValuesWithRequestedSpan => CanonicalProfile::CalibratedMonomial,
            Self::KeySelectionsWithRemainingHorizon => CanonicalProfile::GoalConditionedDiagnostic,
            Self::FiniteG0Discrete => CanonicalProfile::FiniteG0,
        }
    }

    fn from_code(code: u16) -> Result<Self, EnvelopeError> {
        match code {
            1 => Ok(Self::ChannelValuesWithRequestedSpan),
            2 => Ok(Self::KeySelectionsWithRemainingHorizon),
            3 => Ok(Self::FiniteG0Discrete),
            _ => Err(EnvelopeError::UnknownProfile { code }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeError {
    EmptyLegacyEpisode,
    EmptyTaggedEpisode,
    PaddingToken {
        index: usize,
    },
    /// A body record carries the header's exact three-part signature.
    HeaderSignatureInBody {
        index: usize,
    },
    FirstGroupNotZero {
        found: u16,
    },
    NonContiguousGroups {
        index: usize,
        previous: u16,
        found: u16,
    },
    EventOverflow,
    MissingProfileHeader,
    MalformedProfileHeader,
    SupervisedProfileHeader,
    UnknownProfile {
        code: u16,
    },
    ProfileHeaderNotIsolated,
}

impl fmt::Display for EnvelopeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLegacyEpisode => write!(formatter, "cannot tag an empty legacy episode"),
            Self::EmptyTaggedEpisode => write!(formatter, "the tagged episode is empty"),
            Self::PaddingToken { index } => {
                write!(
                    formatter,
                    "legacy episode contains padding at record {index}"
                )
            }
            Self::HeaderSignatureInBody { index } => write!(
                formatter,
                "record {index} carries the exact signature of a profile declaration"
            ),
            Self::FirstGroupNotZero { found } => {
                write!(formatter, "the first event group is {found}, not 0")
            }
            Self::NonContiguousGroups {
                index,
                previous,
                found,
            } => write!(
                formatter,
                "record {index} moves from event group {previous} to non-contiguous group {found}"
            ),
            Self::EventOverflow => write!(formatter, "the legacy event index cannot be shifted"),
            Self::MissingProfileHeader => {
                write!(formatter, "the first record is not a profile declaration")
            }
            Self::MalformedProfileHeader => {
                write!(
                    formatter,
                    "the profile declaration has the wrong 0.3.0 signature"
                )
            }
            Self::SupervisedProfileHeader => {
                write!(
                    formatter,
                    "the profile declaration must not contain supervision"
                )
            }
            Self::UnknownProfile { code } => {
                write!(formatter, "profile code {code} is not declared by this ABI")
            }
            Self::ProfileHeaderNotIsolated => {
                write!(
                    formatter,
                    "the profile declaration must be its own event group"
                )
            }
        }
    }
}

impl std::error::Error for EnvelopeError {}

/// Construct the sole record added by the envelope.
pub fn profile_header(profile: InterpretationProfile) -> LearningToken {
    LearningToken {
        public: PublicToken {
            role: PROFILE_TAG_ROLE,
            key: profile.code(),
            event: PROFILE_TAG_EVENT,
            payload: PROFILE_TAG_PAYLOAD,
        },
        supervision: Supervision::default(),
    }
}

/// Prefix a profile declaration and shift every legacy event group by one.
///
/// The input is borrowed and never modified. Removing the declaration with
/// [`strip_profile_tag`] reconstructs an exactly equal `0.2.0` sequence.
pub fn tag_legacy_episode(
    profile: InterpretationProfile,
    legacy: &[LearningToken],
) -> Result<Vec<LearningToken>, EnvelopeError> {
    validate_legacy_body(legacy)?;
    // Shifting every group by one is the only thing that can overflow, so the
    // guard belongs here rather than in the body validator: a legacy episode
    // that fills the group index is well formed, it just cannot be wrapped.
    if legacy
        .last()
        .is_some_and(|token| token.public.event == u16::MAX)
    {
        return Err(EnvelopeError::EventOverflow);
    }

    let mut tagged = Vec::with_capacity(legacy.len() + 1);
    tagged.push(profile_header(profile));
    tagged.extend(legacy.iter().cloned().map(|mut token| {
        token.public.event += 1;
        token
    }));
    Ok(tagged)
}

/// Validate a complete tagged sequence and return its declared decoder.
pub fn declared_profile(tagged: &[LearningToken]) -> Result<InterpretationProfile, EnvelopeError> {
    validate_tagged(tagged)
}

/// Remove and validate the declaration, reconstructing the exact legacy body.
pub fn strip_profile_tag(
    tagged: &[LearningToken],
) -> Result<(InterpretationProfile, Vec<LearningToken>), EnvelopeError> {
    let profile = validate_tagged(tagged)?;
    let legacy: Vec<_> = tagged[1..]
        .iter()
        .cloned()
        .map(|mut token| {
            token.public.event -= 1;
            token
        })
        .collect();
    validate_legacy_body(&legacy)?;
    Ok((profile, legacy))
}

fn validate_tagged(tagged: &[LearningToken]) -> Result<InterpretationProfile, EnvelopeError> {
    let header = tagged.first().ok_or(EnvelopeError::EmptyTaggedEpisode)?;
    if header.public.role != PROFILE_TAG_ROLE || header.public.event != PROFILE_TAG_EVENT {
        return Err(EnvelopeError::MissingProfileHeader);
    }
    if header.public.payload != PROFILE_TAG_PAYLOAD {
        return Err(EnvelopeError::MalformedProfileHeader);
    }
    if header.supervision != Supervision::default() {
        return Err(EnvelopeError::SupervisedProfileHeader);
    }
    let profile = InterpretationProfile::from_code(header.public.key)?;
    if tagged.len() == 1 {
        return Err(EnvelopeError::EmptyLegacyEpisode);
    }
    if tagged[1].public.event != 1 {
        return Err(EnvelopeError::ProfileHeaderNotIsolated);
    }

    let mut previous = 1u16;
    for (index, token) in tagged.iter().enumerate().skip(1) {
        if token.public.role == Role::Pad {
            return Err(EnvelopeError::PaddingToken { index });
        }
        validate_no_header_collision(index, token)?;
        let found = token.public.event;
        if found != previous && found != previous.saturating_add(1) {
            return Err(EnvelopeError::NonContiguousGroups {
                index,
                previous,
                found,
            });
        }
        previous = found;
    }
    Ok(profile)
}

/// Refuse exactly the records a validator could mistake for a header.
///
/// The event index is part of the signature because a header is always the
/// isolated first group; a record carrying the tag payload later in the episode
/// could never be read as one. Refusing it anyway would be refusing more than
/// the ambiguity, which is what `0.3.0` did to every condition record.
fn validate_no_header_collision(index: usize, token: &LearningToken) -> Result<(), EnvelopeError> {
    if token.public.role == PROFILE_TAG_ROLE
        && token.public.event == PROFILE_TAG_EVENT
        && token.public.payload == PROFILE_TAG_PAYLOAD
    {
        return Err(EnvelopeError::HeaderSignatureInBody { index });
    }
    Ok(())
}

fn validate_legacy_body(legacy: &[LearningToken]) -> Result<(), EnvelopeError> {
    let first = legacy.first().ok_or(EnvelopeError::EmptyLegacyEpisode)?;
    if first.public.event != 0 {
        return Err(EnvelopeError::FirstGroupNotZero {
            found: first.public.event,
        });
    }

    let mut previous = 0u16;
    for (index, token) in legacy.iter().enumerate() {
        if token.public.role == Role::Pad {
            return Err(EnvelopeError::PaddingToken { index });
        }
        validate_no_header_collision(index, token)?;
        let found = token.public.event;
        if found != previous && found != previous + 1 {
            return Err(EnvelopeError::NonContiguousGroups {
                index,
                previous,
                found,
            });
        }
        previous = found;
    }
    Ok(())
}
