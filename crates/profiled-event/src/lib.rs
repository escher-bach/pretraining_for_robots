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
pub const PROFILED_TOKEN_ABI_VERSION: &str = "physical-event-abi-0.3.0";

/// The profile declaration uses a role that no `0.2.0` producer emits.
pub const PROFILE_TAG_ROLE: Role = Role::Condition;
pub const PROFILE_TAG_EVENT: u16 = 0;

/// Slots 0..=2 are the semantic version tuple. Slot 5 is the usual presence
/// bit. The other slots are fixed at zero so corruption is detectable.
pub const PROFILE_TAG_PAYLOAD: [f32; PAYLOAD_DIM] = [0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

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
}

impl InterpretationProfile {
    pub const ALL: [Self; 2] = [
        Self::ChannelValuesWithRequestedSpan,
        Self::KeySelectionsWithRemainingHorizon,
    ];

    pub const fn code(self) -> u16 {
        self as u16
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ChannelValuesWithRequestedSpan => "channel-values-with-requested-span",
            Self::KeySelectionsWithRemainingHorizon => "key-selections-with-remaining-horizon",
        }
    }

    /// The canonical decoder selected by this public declaration.
    pub const fn canonical_profile(self) -> CanonicalProfile {
        match self {
            Self::ChannelValuesWithRequestedSpan => CanonicalProfile::CalibratedMonomial,
            Self::KeySelectionsWithRemainingHorizon => CanonicalProfile::GoalConditionedDiagnostic,
        }
    }

    fn from_code(code: u16) -> Result<Self, EnvelopeError> {
        match code {
            1 => Ok(Self::ChannelValuesWithRequestedSpan),
            2 => Ok(Self::KeySelectionsWithRemainingHorizon),
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
    ReservedConditionRole {
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
            Self::ReservedConditionRole { index } => write!(
                formatter,
                "record {index} uses the role reserved for the profile declaration"
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
        if token.public.role == PROFILE_TAG_ROLE {
            return Err(EnvelopeError::ReservedConditionRole { index });
        }
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
        if token.public.role == PROFILE_TAG_ROLE {
            return Err(EnvelopeError::ReservedConditionRole { index });
        }
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
