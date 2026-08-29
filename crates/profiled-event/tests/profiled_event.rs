use pretraining_canonical_event::{
    decode_episode, render_public, ChannelContent, PublicFact, PublicRow, QueryHorizon,
};
use pretraining_eviction_world::{standard_eviction_rollouts, SerializationOrder as EvictionOrder};
use pretraining_goal_conditioned_world::{
    standard_diagnostic_rollouts, SerializationOrder as DiagnosticOrder,
};
use pretraining_profiled_event::{
    declared_profile, profile_header, strip_profile_tag, tag_legacy_episode, EnvelopeError,
    InterpretationProfile, LEGACY_TOKEN_ABI_VERSION, PROFILED_TOKEN_ABI_VERSION, PROFILE_TAG_EVENT,
    PROFILE_TAG_PAYLOAD, PROFILE_TAG_ROLE,
};
use pretraining_world::{
    generate_trajectory, FamilyConfig, LearningToken, PublicToken, Role, Supervision,
};

fn rows(tokens: &[LearningToken]) -> Vec<PublicRow> {
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

fn ambiguous_action_query() -> LearningToken {
    LearningToken {
        public: PublicToken {
            role: Role::ActionQuery,
            key: 0,
            event: 0,
            payload: [0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        },
        supervision: Supervision::default(),
    }
}

#[test]
fn the_new_format_is_opt_in_and_the_legacy_version_is_unchanged() {
    assert_eq!(LEGACY_TOKEN_ABI_VERSION, "physical-event-abi-0.2.0");
    assert_eq!(PROFILED_TOKEN_ABI_VERSION, "physical-event-abi-0.3.0");
    assert_eq!(Role::COUNT, 11, "the learner role table did not change");
    assert_eq!(
        pretraining_world::PAYLOAD_DIM,
        8,
        "the payload width did not change"
    );
}

#[test]
fn the_header_is_public_categorical_and_unsupervised() {
    for profile in InterpretationProfile::ALL {
        let header = profile_header(profile);
        assert_eq!(header.public.role, PROFILE_TAG_ROLE);
        assert_eq!(header.public.key, profile.code());
        assert_eq!(header.public.event, PROFILE_TAG_EVENT);
        assert_eq!(header.public.payload, PROFILE_TAG_PAYLOAD);
        assert_eq!(header.supervision, Supervision::default());
    }
    assert_ne!(
        InterpretationProfile::ALL[0].code(),
        InterpretationProfile::ALL[1].code()
    );
}

#[test]
fn every_sampled_numeric_episode_round_trips_exactly() {
    let config = FamilyConfig::default();
    for seed in 0..4 {
        for index in 0..16 {
            let original = generate_trajectory(&config, seed, index)
                .expect("the admitted world generates")
                .tokens;
            let snapshot = original.clone();
            let tagged = tag_legacy_episode(
                InterpretationProfile::ChannelValuesWithRequestedSpan,
                &original,
            )
            .expect("valid legacy events can be tagged");
            assert_eq!(original, snapshot, "tagging mutated the legacy sequence");
            assert_eq!(tagged.len(), original.len() + 1);

            let (profile, recovered) =
                strip_profile_tag(&tagged).expect("the tag has an exact inverse");
            assert_eq!(
                profile,
                InterpretationProfile::ChannelValuesWithRequestedSpan
            );
            assert_eq!(recovered, original);
        }
    }
}

#[test]
fn both_selection_processes_round_trip_under_one_semantic_profile() {
    for order in [DiagnosticOrder::Canonical, DiagnosticOrder::Permuted] {
        for mut rollout in standard_diagnostic_rollouts(order) {
            while !rollout.is_done() {
                rollout.step_normalized(-1.0).expect("a live step succeeds");
            }
            let original = rollout.tokens().to_vec();
            let tagged = tag_legacy_episode(
                InterpretationProfile::KeySelectionsWithRemainingHorizon,
                &original,
            )
            .expect("diagnostic rows can be tagged");
            let (profile, recovered) = strip_profile_tag(&tagged).expect("tag strips");
            assert_eq!(
                profile,
                InterpretationProfile::KeySelectionsWithRemainingHorizon
            );
            assert_eq!(recovered, original);
        }
    }

    for order in [EvictionOrder::Canonical, EvictionOrder::Permuted] {
        for mut rollout in standard_eviction_rollouts(order) {
            while !rollout.is_done() {
                let commands = vec![0.0; rollout.current_query_positions().len()];
                rollout
                    .step_normalized(&commands)
                    .expect("a live step succeeds");
            }
            let original = rollout.tokens().to_vec();
            let tagged = tag_legacy_episode(
                InterpretationProfile::KeySelectionsWithRemainingHorizon,
                &original,
            )
            .expect("eviction rows can be tagged");
            let (profile, recovered) = strip_profile_tag(&tagged).expect("tag strips");
            assert_eq!(
                profile,
                InterpretationProfile::KeySelectionsWithRemainingHorizon
            );
            assert_eq!(recovered, original);
        }
    }
}

#[test]
fn a_profile_tag_resolves_the_previously_ambiguous_row() {
    let legacy = vec![ambiguous_action_query()];
    let numeric_tagged = tag_legacy_episode(
        InterpretationProfile::ChannelValuesWithRequestedSpan,
        &legacy,
    )
    .expect("tags");
    let selection_tagged = tag_legacy_episode(
        InterpretationProfile::KeySelectionsWithRemainingHorizon,
        &legacy,
    )
    .expect("tags");

    let (numeric_profile, numeric_body) =
        strip_profile_tag(&numeric_tagged).expect("numeric tag decodes");
    let (selection_profile, selection_body) =
        strip_profile_tag(&selection_tagged).expect("selection tag decodes");
    assert_eq!(numeric_body, selection_body, "the legacy row is identical");

    let numeric = decode_episode(numeric_profile.canonical_profile(), &rows(&numeric_body))
        .expect("the declared numeric decoder accepts the row");
    let selection = decode_episode(
        selection_profile.canonical_profile(),
        &rows(&selection_body),
    )
    .expect("the declared selection decoder accepts the row");
    assert_ne!(
        numeric.groups[0].records[0].fact,
        selection.groups[0].records[0].fact
    );
    assert!(matches!(
        numeric.groups[0].records[0].fact,
        PublicFact::ActionQuery {
            horizon: QueryHorizon::ActuatorSpan { .. },
            ..
        }
    ));
    assert!(matches!(
        selection.groups[0].records[0].fact,
        PublicFact::ActionQuery {
            horizon: QueryHorizon::RemainingFraction { .. },
            ..
        }
    ));
}

#[test]
fn the_declared_profile_drives_a_lossless_canonical_round_trip() {
    let original = generate_trajectory(&FamilyConfig::default(), 19, 7)
        .expect("generates")
        .tokens;
    let tagged = tag_legacy_episode(
        InterpretationProfile::ChannelValuesWithRequestedSpan,
        &original,
    )
    .expect("tags");
    let (profile, body) = strip_profile_tag(&tagged).expect("strips");
    let canonical =
        decode_episode(profile.canonical_profile(), &rows(&body)).expect("profile selects decoder");
    assert_eq!(render_public(&canonical).expect("renders"), rows(&body));
}

#[test]
fn the_shared_profile_header_does_not_reveal_which_selection_world_emitted_it() {
    let diagnostic = standard_diagnostic_rollouts(DiagnosticOrder::Canonical)
        .into_iter()
        .next()
        .expect("suite is non-empty");
    let eviction = standard_eviction_rollouts(EvictionOrder::Canonical)
        .into_iter()
        .next()
        .expect("suite is non-empty");
    let diagnostic_tagged = tag_legacy_episode(
        InterpretationProfile::KeySelectionsWithRemainingHorizon,
        diagnostic.tokens(),
    )
    .expect("tags");
    let eviction_tagged = tag_legacy_episode(
        InterpretationProfile::KeySelectionsWithRemainingHorizon,
        eviction.tokens(),
    )
    .expect("tags");

    assert_eq!(diagnostic_tagged[0], eviction_tagged[0]);
}

#[test]
fn malformed_or_implicit_sequences_are_rejected() {
    let legacy = vec![ambiguous_action_query()];
    assert_eq!(
        declared_profile(&legacy),
        Err(EnvelopeError::MissingProfileHeader)
    );

    let mut unknown = tag_legacy_episode(
        InterpretationProfile::ChannelValuesWithRequestedSpan,
        &legacy,
    )
    .expect("tags");
    unknown[0].public.key = 99;
    assert_eq!(
        declared_profile(&unknown),
        Err(EnvelopeError::UnknownProfile { code: 99 })
    );

    let mut malformed = unknown.clone();
    malformed[0].public.key = InterpretationProfile::ChannelValuesWithRequestedSpan.code();
    malformed[0].public.payload[2] = 1.0;
    assert_eq!(
        declared_profile(&malformed),
        Err(EnvelopeError::MalformedProfileHeader)
    );

    let mut supervised = tag_legacy_episode(
        InterpretationProfile::ChannelValuesWithRequestedSpan,
        &legacy,
    )
    .expect("tags");
    supervised[0].supervision.future_mask = true;
    assert_eq!(
        declared_profile(&supervised),
        Err(EnvelopeError::SupervisedProfileHeader)
    );

    let mut shared_group = tag_legacy_episode(
        InterpretationProfile::ChannelValuesWithRequestedSpan,
        &legacy,
    )
    .expect("tags");
    shared_group[1].public.event = 0;
    assert_eq!(
        declared_profile(&shared_group),
        Err(EnvelopeError::ProfileHeaderNotIsolated)
    );
}

#[test]
fn padding_condition_records_gaps_and_overflow_cannot_enter_the_envelope() {
    let mut padding = ambiguous_action_query();
    padding.public.role = Role::Pad;
    assert_eq!(
        tag_legacy_episode(
            InterpretationProfile::ChannelValuesWithRequestedSpan,
            &[padding]
        ),
        Err(EnvelopeError::PaddingToken { index: 0 })
    );

    let mut condition = ambiguous_action_query();
    condition.public.role = Role::Condition;
    assert_eq!(
        tag_legacy_episode(
            InterpretationProfile::ChannelValuesWithRequestedSpan,
            &[condition]
        ),
        Err(EnvelopeError::ReservedConditionRole { index: 0 })
    );

    let mut gap = ambiguous_action_query();
    let mut later = gap.clone();
    later.public.event = 2;
    assert_eq!(
        tag_legacy_episode(
            InterpretationProfile::ChannelValuesWithRequestedSpan,
            &[gap.clone(), later]
        ),
        Err(EnvelopeError::NonContiguousGroups {
            index: 1,
            previous: 0,
            found: 2,
        })
    );

    gap.public.event = u16::MAX;
    assert_eq!(
        tag_legacy_episode(
            InterpretationProfile::ChannelValuesWithRequestedSpan,
            &[gap]
        ),
        Err(EnvelopeError::FirstGroupNotZero { found: u16::MAX })
    );
}

#[test]
fn profile_names_describe_meaning_not_generators() {
    assert_eq!(
        InterpretationProfile::ChannelValuesWithRequestedSpan.as_str(),
        "channel-values-with-requested-span"
    );
    assert_eq!(
        InterpretationProfile::KeySelectionsWithRemainingHorizon.as_str(),
        "key-selections-with-remaining-horizon"
    );
    assert_ne!(
        InterpretationProfile::ChannelValuesWithRequestedSpan
            .canonical_profile()
            .schema_hash(),
        InterpretationProfile::KeySelectionsWithRemainingHorizon
            .canonical_profile()
            .schema_hash()
    );

    // The distinction is about content: a value and a key selection decode to
    // different canonical facts even if a downstream task gives them the same
    // English label.
    let value = ChannelContent::Value(pretraining_canonical_event::Quantity::normalized(
        1.0, 0.0, 1.0,
    ));
    let selection = ChannelContent::Selection {
        indicator: pretraining_canonical_event::Quantity::normalized(1.0, 0.0, 1.0),
    };
    assert_ne!(value, selection);
}

/// Collect every record produced by all three legacy generators, driving the
/// live processes with several distinct controls so different serializer
/// branches run.
fn every_legacy_record() -> Vec<(&'static str, LearningToken)> {
    let mut records = Vec::new();

    let config = FamilyConfig::default();
    for seed in 0..4 {
        for index in 0..16 {
            let trajectory = generate_trajectory(&config, seed, index)
                .expect("the admitted world generates")
                .tokens;
            records.extend(
                trajectory
                    .into_iter()
                    .map(|token| ("numeric-control", token)),
            );
        }
    }

    for order in [DiagnosticOrder::Canonical, DiagnosticOrder::Permuted] {
        for control in [-1.0f32, 0.0, 1.0] {
            for mut rollout in standard_diagnostic_rollouts(order) {
                while !rollout.is_done() {
                    rollout
                        .step_normalized(control)
                        .expect("a live step succeeds");
                }
                records.extend(
                    rollout
                        .tokens()
                        .iter()
                        .cloned()
                        .map(|token| ("line-diagnostic", token)),
                );
            }
        }
    }

    for order in [EvictionOrder::Canonical, EvictionOrder::Permuted] {
        for control in [-1.0f32, 0.0, 1.0] {
            for mut rollout in standard_eviction_rollouts(order) {
                while !rollout.is_done() {
                    let commands = vec![control; rollout.current_query_positions().len()];
                    rollout
                        .step_normalized(&commands)
                        .expect("a live step succeeds");
                }
                records.extend(
                    rollout
                        .tokens()
                        .iter()
                        .cloned()
                        .map(|token| ("container-eviction", token)),
                );
            }
        }
    }

    records
}

#[test]
fn no_legacy_producer_emits_the_role_this_envelope_reserves() {
    let records = every_legacy_record();
    assert!(
        records.len() > 1_000,
        "the survey must actually cover the generators, saw {} records",
        records.len()
    );

    for (producer, token) in &records {
        assert_ne!(
            token.public.role, PROFILE_TAG_ROLE,
            "{producer} emits the role reserved for the profile declaration"
        );
        assert_ne!(
            token.public.role,
            Role::Pad,
            "{producer} emits padding inside an episode"
        );
    }
}

#[test]
fn a_tagged_episode_is_refused_by_the_canonical_decoder_rather_than_silently_read() {
    let mut rollout = standard_diagnostic_rollouts(DiagnosticOrder::Canonical)
        .into_iter()
        .next()
        .expect("suite is non-empty");
    while !rollout.is_done() {
        rollout.step_normalized(-1.0).expect("a live step succeeds");
    }
    let tagged = tag_legacy_episode(
        InterpretationProfile::KeySelectionsWithRemainingHorizon,
        rollout.tokens(),
    )
    .expect("tags");

    // A consumer that skips the envelope gets an error, not a reading. The
    // declaration is not decodable as a public fact under any profile.
    for canonical in pretraining_canonical_event::Profile::ALL {
        assert!(
            matches!(
                decode_episode(canonical, &rows(&tagged)),
                Err(pretraining_canonical_event::DecodeError::ProfileDoesNotEmit { .. })
            ),
            "a tagged episode must not decode under {}",
            canonical.schema_hash()
        );
    }

    let (profile, body) = strip_profile_tag(&tagged).expect("strips");
    decode_episode(profile.canonical_profile(), &rows(&body))
        .expect("the declared decoder reads the body once the tag is consumed");
}

#[test]
fn stripping_and_retagging_reproduces_the_tagged_episode() {
    let config = FamilyConfig::default();
    for seed in 0..3 {
        let original = generate_trajectory(&config, seed, 5)
            .expect("generates")
            .tokens;
        for profile in InterpretationProfile::ALL {
            let tagged = tag_legacy_episode(profile, &original).expect("tags");
            let (declared, body) = strip_profile_tag(&tagged).expect("strips");
            assert_eq!(declared, profile);
            assert_eq!(
                tag_legacy_episode(declared, &body).expect("retags"),
                tagged,
                "the envelope is not a bijection on tagged episodes"
            );
        }
    }
}

#[test]
fn a_legacy_episode_that_fills_the_group_index_is_valid_but_cannot_be_wrapped() {
    // Contiguity means the only way to reach the last group is to occupy every
    // one. Such a body is well formed; the envelope simply has nowhere to put
    // its declaration, and says so instead of wrapping around to group 0.
    let full: Vec<LearningToken> = (0..=u16::MAX)
        .map(|group| {
            let mut token = ambiguous_action_query();
            token.public.event = group;
            token
        })
        .collect();

    assert_eq!(
        tag_legacy_episode(InterpretationProfile::ChannelValuesWithRequestedSpan, &full),
        Err(EnvelopeError::EventOverflow)
    );

    // One group short, the same body wraps and unwraps exactly.
    let room = &full[..full.len() - 1];
    let tagged = tag_legacy_episode(InterpretationProfile::ChannelValuesWithRequestedSpan, room)
        .expect("a body with one group to spare can be tagged");
    assert_eq!(tagged.last().expect("non-empty").public.event, u16::MAX);
    let (_, recovered) = strip_profile_tag(&tagged).expect("strips");
    assert_eq!(recovered, room);
}
