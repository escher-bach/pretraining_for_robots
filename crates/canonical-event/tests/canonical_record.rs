//! The exact test matrix the representation audit asked for.
//!
//! The matrix has two halves and both are needed. Invariances say what must stay
//! the same under a presentation change; sensitivities say what must still
//! change when the task changes. A serializer that satisfies only the first half
//! can be "robust" because it deleted the relation the task teaches.
//!
//! Every claim here is about the **apparatus**: what the record and the renderer
//! preserve. None of it is a claim about learner invariance, which needs a
//! learner and is out of scope for this crate.

use std::collections::BTreeMap;

use pretraining_canonical_event::corpus::{
    diagnostic_episode, monomial_episode, reference_episodes, DIAGNOSTIC_CONTROL_KEY, LINE_LENGTH,
};
use pretraining_canonical_event::{
    decode_episode, decode_supervision, render_public, render_public_with_capacity,
    render_supervision, ActionTarget, ChannelContent, DecodeError, Episode, EventKind,
    KeyNamespace, KeyRenaming, LocalKey, Profile, PublicEpisode, PublicFact, PublicRecord,
    PublicRow, Quantity, RecordAddress, RenderError, StepSpan, SupervisionRecord, SupervisionRow,
    Unit, ACTION_HORIZON, CANONICAL_RECORD_VERSION, PAYLOAD_DIM,
};
use pretraining_goal_conditioned_world::{standard_diagnostic_rollouts, SerializationOrder};
use pretraining_world::{LearningToken, Role};

// ---------------------------------------------------------------------------
// Agreement with the layout this crate renders onto
// ---------------------------------------------------------------------------

#[test]
fn role_codes_and_widths_match_the_production_abi() {
    assert_eq!(PAYLOAD_DIM, pretraining_world::PAYLOAD_DIM);
    assert_eq!(ACTION_HORIZON, pretraining_world::ACTION_HORIZON);
    assert_eq!(
        EventKind::SchemaObservation.role_code(),
        Role::SchemaObservation as u8
    );
    assert_eq!(
        EventKind::SchemaActuator.role_code(),
        Role::SchemaActuator as u8
    );
    assert_eq!(EventKind::Boundary.role_code(), Role::Boundary as u8);
    assert_eq!(EventKind::Condition.role_code(), Role::Condition as u8);
    assert_eq!(EventKind::Goal.role_code(), Role::Goal as u8);
    assert_eq!(EventKind::Observation.role_code(), Role::Observation as u8);
    assert_eq!(EventKind::ActionQuery.role_code(), Role::ActionQuery as u8);
    assert_eq!(
        EventKind::ActionExecuted.role_code(),
        Role::ActionExecuted as u8
    );
    assert_eq!(EventKind::FutureQuery.role_code(), Role::FutureQuery as u8);
    assert_eq!(EventKind::Feedback.role_code(), Role::Feedback as u8);
    // Role 0 is padding and is deliberately not a canonical fact, so the
    // canonical kinds are one fewer than the production role count.
    assert_eq!(EventKind::ALL.len() + 1, Role::COUNT);
}

// ---------------------------------------------------------------------------
// Round trip: information preservation
// ---------------------------------------------------------------------------

#[test]
fn every_reference_episode_round_trips_exactly() {
    for (name, episode) in reference_episodes() {
        let rows = render_public(episode.public()).expect("the corpus renders");
        let decoded = decode_episode(episode.public().profile, &rows).expect("the corpus decodes");
        assert_eq!(
            &decoded,
            episode.public(),
            "{name} did not survive the canonical round trip"
        );
        let rerendered = render_public(&decoded).expect("a decoded episode re-renders");
        assert_eq!(rows, rerendered, "{name} re-rendered to different rows");
        assert_eq!(decoded.version, CANONICAL_RECORD_VERSION);
    }
}

#[test]
fn supervision_round_trips_and_unsupervised_steps_stay_absent() {
    for (name, episode) in reference_episodes() {
        let rows = render_supervision(&episode).expect("the corpus supervision renders");
        assert_eq!(rows.len(), episode.public().record_count());
        let decoded =
            decode_supervision(episode.public(), &rows).expect("the corpus supervision decodes");
        assert_eq!(
            &decoded,
            episode.supervision(),
            "{name} lost supervision in the round trip"
        );
        // Absence is the only representation of "not supervised": there is no
        // masked-off value sitting behind a false flag.
        for entry in decoded.entries.values() {
            for target in &entry.action_targets {
                assert!(target.step < ACTION_HORIZON);
            }
            assert!(!entry.is_empty());
        }
    }
}

#[test]
fn perturbing_supervision_cannot_change_the_public_rendering() {
    let episode = monomial_episode();
    let before = render_public(episode.public()).expect("renders");

    let mut perturbed_supervision = episode.supervision().clone();
    for entry in perturbed_supervision.entries.values_mut() {
        for target in &mut entry.action_targets {
            target.value = -target.value + 0.5;
        }
        if let Some(future) = entry.future_target {
            entry.future_target = Some(future + 0.25);
        }
    }
    let perturbed = Episode::new(episode.public().clone(), perturbed_supervision)
        .expect("addresses still exist");

    let after = render_public(perturbed.public()).expect("renders");
    assert_eq!(
        before, after,
        "public rendering must not depend on privileged targets"
    );
    assert_ne!(
        render_supervision(&episode).expect("renders"),
        render_supervision(&perturbed).expect("renders"),
        "the perturbation must actually have changed the targets"
    );
}

// ---------------------------------------------------------------------------
// Loud failures rather than silent repair
// ---------------------------------------------------------------------------

fn one_observation_row(value: f32) -> PublicRow {
    PublicRow {
        role: EventKind::Observation.role_code(),
        key: 0,
        group: 0,
        payload: [value, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    }
}

#[test]
fn padding_rows_are_rejected() {
    let mut row = one_observation_row(0.5);
    row.role = 0;
    assert_eq!(
        decode_episode(Profile::CalibratedMonomial, &[row]),
        Err(DecodeError::PaddingRow { index: 0 })
    );
}

#[test]
fn reserved_slots_must_be_zero() {
    let mut row = one_observation_row(0.5);
    row.payload[6] = 1.0;
    assert!(matches!(
        decode_episode(Profile::CalibratedMonomial, &[row]),
        Err(DecodeError::ReservedSlotNotZero { slot: 6, .. })
    ));
}

#[test]
fn the_presence_flag_must_be_set() {
    let mut row = one_observation_row(0.5);
    row.payload[5] = 0.0;
    assert!(matches!(
        decode_episode(Profile::CalibratedMonomial, &[row]),
        Err(DecodeError::RecordNotPresent { .. })
    ));
}

#[test]
fn non_contiguous_groups_are_rejected() {
    let mut row = one_observation_row(0.5);
    row.group = 3;
    assert_eq!(
        decode_episode(Profile::CalibratedMonomial, &[row]),
        Err(DecodeError::NonContiguousGroups {
            index: 0,
            expected: 0,
            found: 3
        })
    );
}

#[test]
fn a_profile_rejects_a_kind_its_producer_never_emits() {
    let row = PublicRow {
        role: EventKind::Feedback.role_code(),
        key: 0,
        group: 0,
        payload: [0.25, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    };
    assert!(matches!(
        decode_episode(Profile::GoalConditionedDiagnostic, &[row]),
        Err(DecodeError::ProfileDoesNotEmit { .. })
    ));
    assert!(decode_episode(Profile::CalibratedMonomial, &[row]).is_ok());
}

#[test]
fn exceeding_the_declared_capacity_is_an_error_not_a_truncation() {
    let episode = monomial_episode();
    let full = render_public(episode.public()).expect("renders");
    let error = render_public_with_capacity(episode.public(), full.len() - 1)
        .expect_err("a shortened episode is a changed task");
    assert_eq!(
        error,
        RenderError::CapacityExceeded {
            produced: full.len(),
            capacity: full.len() - 1
        }
    );
    assert!(render_public_with_capacity(episode.public(), full.len()).is_ok());
}

#[test]
fn a_value_that_would_lose_precision_is_rejected() {
    let record = PublicRecord::new(
        LocalKey::new(KeyNamespace::Observation, 0),
        PublicFact::Observation {
            content: ChannelContent::Value(Quantity::normalized(0.1, -1.0, 1.0)),
        },
    )
    .expect("well typed");
    let episode = PublicEpisode::new(
        Profile::CalibratedMonomial,
        vec![pretraining_canonical_event::EventGroup {
            group: 0,
            records: vec![record],
        }],
    );
    assert!(matches!(
        render_public(&episode),
        Err(RenderError::NotExactlyRepresentable { .. })
    ));
}

// ---------------------------------------------------------------------------
// Invariances: presentation must not be meaning
// ---------------------------------------------------------------------------

#[test]
fn within_group_reordering_preserves_the_record_multiset_and_its_supervision() {
    let episode = monomial_episode();
    let mut saw_a_real_permutation = false;

    for seed in 0..32u64 {
        let reordered = episode.reorder_within_groups(seed);

        // The group structure is untouched.
        assert_eq!(
            reordered.public().groups.len(),
            episode.public().groups.len()
        );
        for (before, after) in episode
            .public()
            .groups
            .iter()
            .zip(&reordered.public().groups)
        {
            assert_eq!(before.group, after.group);
            let mut before_sorted = before.records.clone();
            let mut after_sorted = after.records.clone();
            before_sorted.sort_by_key(|record| format!("{record:?}"));
            after_sorted.sort_by_key(|record| format!("{record:?}"));
            assert_eq!(
                before_sorted, after_sorted,
                "reordering changed the record multiset"
            );
        }
        if render_public(reordered.public()).expect("renders")
            != render_public(episode.public()).expect("renders")
        {
            saw_a_real_permutation = true;
        }

        // Supervision travelled with its record rather than with its position.
        let paired_before = paired_targets(&episode);
        let paired_after = paired_targets(&reordered);
        assert_eq!(
            paired_before, paired_after,
            "reordering separated a target from its record"
        );

        // The canonical form and the fingerprint are order free.
        assert_eq!(
            reordered.public().canonicalize_within_groups(),
            episode.public().canonicalize_within_groups()
        );
        assert_eq!(
            reordered.public().presentation_free_fingerprint(),
            episode.public().presentation_free_fingerprint()
        );
    }

    assert!(
        saw_a_real_permutation,
        "the invariance test never actually permuted anything"
    );
}

/// Every supervised target, keyed by the record it supervises rather than by
/// where that record happens to sit.
fn paired_targets(episode: &Episode) -> BTreeMap<String, Vec<ActionTarget>> {
    let mut paired = BTreeMap::new();
    for (group_index, group) in episode.public().groups.iter().enumerate() {
        for (record_index, record) in group.records.iter().enumerate() {
            if let Some(entry) = episode.supervision().get(RecordAddress {
                group_index,
                record_index,
            }) {
                paired.insert(
                    format!("{}:{:?}", group.group, record),
                    entry.action_targets.clone(),
                );
            }
        }
    }
    paired
}

#[test]
fn joint_key_renaming_is_exactly_equivariant_on_the_rendered_rows() {
    let episode = monomial_episode();
    let renaming = KeyRenaming::default()
        .with(KeyNamespace::Observation, &[(0, 41), (1, 17)])
        .with(KeyNamespace::Actuator, &[(0, 9), (1, 200)]);
    let renamed = episode.rename_keys(&renaming).expect("a valid renaming");

    let before = render_public(episode.public()).expect("renders");
    let after = render_public(renamed.public()).expect("renders");
    assert_eq!(before.len(), after.len());

    for (original, renamed_row) in before.iter().zip(&after) {
        let kind = EventKind::from_role_code(original.role).expect("a known role");
        let expected_key = match kind.namespace() {
            KeyNamespace::Observation => match original.key {
                0 => 41,
                1 => 17,
                other => other,
            },
            KeyNamespace::Actuator => match original.key {
                0 => 9,
                1 => 200,
                other => other,
            },
            KeyNamespace::Episode => original.key,
        };
        assert_eq!(renamed_row.role, original.role);
        assert_eq!(renamed_row.group, original.group);
        assert_eq!(renamed_row.payload, original.payload);
        assert_eq!(renamed_row.key, expected_key);
    }

    // Renaming back reproduces the original rows exactly.
    let inverse = renaming.inverse().expect("a bijection inverts");
    let restored = renamed.rename_keys(&inverse).expect("a valid renaming");
    assert_eq!(render_public(restored.public()).expect("renders"), before);
    assert_eq!(restored.supervision(), episode.supervision());
}

#[test]
fn a_non_injective_renaming_is_rejected() {
    let renaming = KeyRenaming::default().with(KeyNamespace::Observation, &[(0, 5), (1, 5)]);
    assert!(renaming.validate().is_err());
    assert!(monomial_episode().rename_keys(&renaming).is_err());
}

#[test]
fn a_renaming_must_cover_every_key_it_claims_to_rename() {
    let partial = KeyRenaming::default().with(KeyNamespace::Observation, &[(0, 5)]);
    let error = monomial_episode()
        .rename_keys(&partial)
        .expect_err("channel 1 is unnamed");
    assert!(error.contains("does not cover key 1"), "{error}");
}

#[test]
fn typed_namespaces_separate_keys_that_the_flat_layout_merges() {
    // Observation channel 0 and actuator channel 0 render to the same key slot.
    let episode = monomial_episode();
    let rows = render_public(episode.public()).expect("renders");
    let observation_zero = rows
        .iter()
        .find(|row| row.role == EventKind::Observation.role_code() && row.key == 0)
        .expect("an observation on channel 0");
    let actuator_zero = rows
        .iter()
        .find(|row| row.role == EventKind::ActionExecuted.role_code() && row.key == 0)
        .expect("a command on actuator 0");
    assert_eq!(observation_zero.key, actuator_zero.key);

    // The canonical record keeps them apart, so a renaming can touch one
    // without touching the other.
    let renaming = KeyRenaming::default().with(KeyNamespace::Observation, &[(0, 6), (1, 7)]);
    let renamed = episode.rename_keys(&renaming).expect("valid");
    let renamed_rows = render_public(renamed.public()).expect("renders");
    assert!(renamed_rows
        .iter()
        .any(|row| row.role == EventKind::Observation.role_code() && row.key == 6));
    assert!(renamed_rows
        .iter()
        .any(|row| row.role == EventKind::ActionExecuted.role_code() && row.key == 0));

    // A record whose key namespace contradicts its fact cannot be built.
    assert!(PublicRecord::new(
        LocalKey::new(KeyNamespace::Actuator, 0),
        PublicFact::Observation {
            content: ChannelContent::Value(Quantity::normalized(0.0, -1.0, 1.0)),
        },
    )
    .is_err());
}

#[test]
fn an_affine_unit_change_leaves_the_rendering_identical_and_round_trips_physically() {
    let normalized = Quantity::normalized(0.5, -1.0, 1.0);
    let unit = Unit::affine("metres", 0.25, 0.125).expect("a valid affine unit");
    let in_metres = normalized.clone().with_unit(unit.clone());

    let build = |quantity: Quantity| {
        PublicEpisode::new(
            Profile::CalibratedMonomial,
            vec![pretraining_canonical_event::EventGroup {
                group: 0,
                records: vec![PublicRecord::new(
                    LocalKey::new(KeyNamespace::Observation, 0),
                    PublicFact::Observation {
                        content: ChannelContent::Value(quantity),
                    },
                )
                .expect("well typed")],
            }],
        )
    };

    assert_eq!(
        render_public(&build(normalized.clone())).expect("renders"),
        render_public(&build(in_metres.clone())).expect("renders"),
        "the 0.2.0 layout carries no unit, so a unit change must not change bytes"
    );

    assert_eq!(in_metres.physical(), 0.25);
    assert_eq!(unit.from_physical(in_metres.physical()), normalized.value);

    // Because the layout drops the unit, decoding returns the identity unit.
    // That loss is real and is recorded here rather than papered over.
    let decoded = decode_episode(
        Profile::CalibratedMonomial,
        &render_public(&build(in_metres)).expect("renders"),
    )
    .expect("decodes");
    match &decoded.groups[0].records[0].fact {
        PublicFact::Observation {
            content: ChannelContent::Value(quantity),
        } => assert!(quantity.unit.is_identity()),
        other => panic!("unexpected fact {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Sensitivities: the record must still change when the task changes
// ---------------------------------------------------------------------------

#[test]
fn changing_only_the_goal_changes_the_canonical_episode() {
    let left = diagnostic_episode(0, -1.0);
    let right = diagnostic_episode(LINE_LENGTH - 1, 1.0);

    assert_ne!(left.public(), right.public());
    assert_ne!(
        left.public().presentation_free_fingerprint(),
        right.public().presentation_free_fingerprint()
    );

    // The schema and the reset are shared; the goal record is what differs.
    assert_eq!(left.public().groups[0], right.public().groups[0]);
    assert_eq!(left.public().groups[1], right.public().groups[1]);
    assert_ne!(left.public().groups[2], right.public().groups[2]);

    // The supervised first action differs too, which is the point of the pair.
    let first_target = |episode: &Episode| {
        episode
            .supervision()
            .get(RecordAddress {
                group_index: 4,
                record_index: 0,
            })
            .expect("the first action query is supervised")
            .action_targets
            .clone()
    };
    assert_ne!(first_target(&left), first_target(&right));
}

#[test]
fn changing_a_later_group_leaves_the_earlier_prefix_identical() {
    let episode = monomial_episode();
    let rows = render_public(episode.public()).expect("renders");

    let mut altered = episode.public().clone();
    let last = altered.groups.len() - 1;
    altered.groups[last].records[0] = PublicRecord::new(
        LocalKey::new(KeyNamespace::Episode, 0),
        PublicFact::Boundary {
            subtype: pretraining_canonical_event::BoundarySubtype::TaskReset,
        },
    )
    .expect("well typed");
    let altered_rows = render_public(&altered).expect("renders");

    assert_ne!(rows, altered_rows);
    assert_eq!(
        rows[..rows.len() - 1],
        altered_rows[..altered_rows.len() - 1],
        "a later event must not disturb an earlier prefix"
    );
}

#[test]
fn actuator_bounds_do_not_bound_the_actuator_value_slot() {
    // In `calibrated-monomial-0.2.0` an executed command writes the normalized
    // command into the value slot but the physical limit into the bound slots.
    // The two are in different units, so the declared bounds do not contain the
    // declared value. This is recorded as a finding, not silently repaired: the
    // production ABI is not this crate's to change.
    let episode = monomial_episode();
    let executed: Vec<_> = episode
        .public()
        .groups
        .iter()
        .flat_map(|group| &group.records)
        .filter_map(|record| match &record.fact {
            PublicFact::ActionExecuted { command, .. } => Some(command.clone()),
            _ => None,
        })
        .collect();
    assert!(!executed.is_empty());
    assert!(
        executed.iter().any(|command| !command.is_well_formed()),
        "the corpus should exhibit the unit mismatch it documents"
    );
}

// ---------------------------------------------------------------------------
// The profile finding
// ---------------------------------------------------------------------------

#[test]
fn the_same_bytes_decode_to_different_facts_under_the_two_profiles() {
    let row = one_observation_row(0.5);
    let monomial = decode_episode(Profile::CalibratedMonomial, &[row]).expect("decodes");
    let diagnostic = decode_episode(Profile::GoalConditionedDiagnostic, &[row]).expect("decodes");

    assert_ne!(
        monomial.groups[0].records[0].fact,
        diagnostic.groups[0].records[0].fact
    );
    assert!(matches!(
        monomial.groups[0].records[0].fact,
        PublicFact::Observation {
            content: ChannelContent::Value(_)
        }
    ));
    assert!(matches!(
        diagnostic.groups[0].records[0].fact,
        PublicFact::Observation {
            content: ChannelContent::Selection { .. }
        }
    ));
}

#[test]
fn the_profiles_disagree_about_the_action_query_auxiliary_slot() {
    // The identical row: aux0 is 1.0 and aux1 is 0.0.
    let row = PublicRow {
        role: EventKind::ActionQuery.role_code(),
        key: 0,
        group: 0,
        payload: [0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    };
    let monomial = decode_episode(Profile::CalibratedMonomial, &[row]).expect("decodes");
    let diagnostic = decode_episode(Profile::GoalConditionedDiagnostic, &[row]).expect("decodes");

    match &monomial.groups[0].records[0].fact {
        PublicFact::ActionQuery {
            horizon: pretraining_canonical_event::QueryHorizon::ActuatorSpan { marker, requested },
            ..
        } => {
            assert!(
                *marker,
                "the monomial profile reads aux0 as an actuator marker"
            );
            assert_eq!(*requested, StepSpan::new(0, ACTION_HORIZON as u16).unwrap());
        }
        other => panic!("unexpected fact {other:?}"),
    }
    match &diagnostic.groups[0].records[0].fact {
        PublicFact::ActionQuery {
            horizon: pretraining_canonical_event::QueryHorizon::RemainingFraction { remaining },
            ..
        } => assert_eq!(
            remaining.steps, 2,
            "the diagnostic profile reads the same 1.0 as a full remaining horizon"
        ),
        other => panic!("unexpected fact {other:?}"),
    }
}

#[test]
fn the_schema_hash_separates_the_profiles_and_is_stable() {
    let monomial = Profile::CalibratedMonomial.schema_hash();
    let diagnostic = Profile::GoalConditionedDiagnostic.schema_hash();
    assert_ne!(monomial, diagnostic);
    assert_eq!(monomial, Profile::CalibratedMonomial.schema_hash());
    assert_eq!(diagnostic, Profile::GoalConditionedDiagnostic.schema_hash());

    // The schema is not merely a role count and a payload width: every emitted
    // kind carries a meaning for every slot.
    for profile in Profile::ALL {
        for kind_schema in profile.field_schema() {
            assert_eq!(kind_schema.slots.len(), PAYLOAD_DIM);
            for slot in &kind_schema.slots {
                assert!(!slot.meaning.is_empty());
            }
        }
    }
}

#[test]
fn the_fingerprint_ignores_within_group_order_but_not_key_names() {
    let episode = monomial_episode();
    let reordered = episode.reorder_within_groups(7);
    assert_eq!(
        episode.public().presentation_free_fingerprint(),
        reordered.public().presentation_free_fingerprint()
    );

    let renamed = episode
        .rename_keys(&KeyRenaming::default().with(KeyNamespace::Observation, &[(0, 40), (1, 41)]))
        .expect("valid");
    assert_ne!(
        episode.public().presentation_free_fingerprint(),
        renamed.public().presentation_free_fingerprint(),
        "the fingerprint deliberately does not claim rename invariance"
    );
}

// ---------------------------------------------------------------------------
// Conformance with the real production serializer
// ---------------------------------------------------------------------------

fn to_public_row(token: &LearningToken) -> PublicRow {
    PublicRow {
        role: token.public.role as u8,
        key: token.public.key,
        group: token.public.event,
        payload: token.public.payload,
    }
}

fn to_supervision_row(token: &LearningToken) -> SupervisionRow {
    SupervisionRow {
        action_target: token.supervision.action_target,
        action_mask: token.supervision.action_mask,
        future_target: token.supervision.future_target,
        future_mask: token.supervision.future_mask,
    }
}

#[test]
fn production_diagnostic_rollouts_round_trip_through_the_canonical_record() {
    for order in [SerializationOrder::Canonical, SerializationOrder::Permuted] {
        let mut rollouts = standard_diagnostic_rollouts(order);
        assert!(!rollouts.is_empty());
        for rollout in &mut rollouts {
            while !rollout.is_done() {
                rollout.step_normalized(-1.0).expect("a live episode steps");
            }
            let rows: Vec<PublicRow> = rollout.tokens().iter().map(to_public_row).collect();
            let supervision_rows: Vec<SupervisionRow> =
                rollout.tokens().iter().map(to_supervision_row).collect();

            let episode = decode_episode(Profile::GoalConditionedDiagnostic, &rows)
                .expect("the real serializer decodes into the canonical record");
            assert_eq!(
                render_public(&episode).expect("re-renders"),
                rows,
                "case {} lost information in the canonical round trip",
                rollout.case().id
            );

            let table =
                decode_supervision(&episode, &supervision_rows).expect("supervision decodes");
            let bundle = Episode::new(episode, table).expect("addresses exist");
            assert_eq!(
                render_supervision(&bundle).expect("re-renders"),
                supervision_rows,
                "case {} lost supervision in the canonical round trip",
                rollout.case().id
            );

            // Every fact kind recovered is one the diagnostic profile declares.
            for group in &bundle.public().groups {
                for record in &group.records {
                    assert!(Profile::GoalConditionedDiagnostic.emits(record.fact.kind()));
                    assert_eq!(record.key.namespace, record.fact.namespace());
                }
            }
        }
    }
}

#[test]
fn the_production_control_key_is_an_actuator_key_not_an_observation_key() {
    let mut rollouts = standard_diagnostic_rollouts(SerializationOrder::Canonical);
    let rollout = rollouts.first_mut().expect("a standard suite is non-empty");
    rollout.step_normalized(-1.0).expect("steps");
    let rows: Vec<PublicRow> = rollout.tokens().iter().map(to_public_row).collect();
    let episode = decode_episode(Profile::GoalConditionedDiagnostic, &rows).expect("decodes");

    let actuator_names: Vec<u16> = episode
        .groups
        .iter()
        .flat_map(|group| &group.records)
        .filter(|record| record.key.namespace == KeyNamespace::Actuator)
        .map(|record| record.key.name)
        .collect();
    assert!(!actuator_names.is_empty());
    assert!(actuator_names
        .iter()
        .all(|name| *name == DIAGNOSTIC_CONTROL_KEY));
}

#[test]
fn an_unsupervised_record_produces_no_supervision_entry() {
    let mut rollouts = standard_diagnostic_rollouts(SerializationOrder::Canonical);
    let rollout = rollouts.first_mut().expect("non-empty");
    while !rollout.is_done() {
        rollout.step_normalized(-1.0).expect("steps");
    }
    let rows: Vec<PublicRow> = rollout.tokens().iter().map(to_public_row).collect();
    let supervision_rows: Vec<SupervisionRow> =
        rollout.tokens().iter().map(to_supervision_row).collect();
    let episode = decode_episode(Profile::GoalConditionedDiagnostic, &rows).expect("decodes");
    let table = decode_supervision(&episode, &supervision_rows).expect("decodes");

    // The live diagnostic rollout carries no teacher targets at all, so the
    // table is empty rather than full of zeroed placeholders.
    assert!(
        table.entries.is_empty(),
        "an unsupervised rollout produced {} entries",
        table.entries.len()
    );
    assert!(SupervisionRecord::default().is_empty());
}
