//! Checks that relate the second process to things outside it.
//!
//! Two questions are asked here. First, is the second process's surface really
//! different from the first process's, or does it merely look different in
//! prose? Second, does it serialize into the canonical public-event record, so
//! that a later comparison between the two processes is a comparison of content
//! rather than of two incomparable encodings?
//!
//! Neither question is about a learner, and neither answer is evidence of
//! transfer.

use std::collections::BTreeSet;

use pretraining_canonical_event::{decode_episode, render_public, Profile, PublicRow};
use pretraining_eviction_world::{
    standard_eviction_rollouts, EvictionRollout, GoalAwarePolicy, PublicPolicy, SerializationOrder,
    BLOCKER_BAND_TAG, GOAL_BLIND_CEILING, HORIZON, ITEM_BAND_TAG,
};
use pretraining_goal_conditioned_world::{
    standard_diagnostic_rollouts, DiagnosticRollout, SerializationOrder as FirstOrder,
};
use pretraining_world::{LearningToken, Role};

/// Drive every first-process episode to completion so its full vocabulary is
/// serialized, not only its opening prefix.
fn first_process_tokens() -> Vec<LearningToken> {
    let mut tokens = Vec::new();
    for order in [FirstOrder::Canonical, FirstOrder::Permuted] {
        let mut rollouts: Vec<DiagnosticRollout> = standard_diagnostic_rollouts(order);
        for rollout in &mut rollouts {
            while !rollout.is_done() {
                rollout.step_normalized(-1.0).expect("a live step");
            }
            tokens.extend(rollout.tokens().iter().cloned());
        }
        let mut rollouts: Vec<DiagnosticRollout> = standard_diagnostic_rollouts(order);
        for rollout in &mut rollouts {
            while !rollout.is_done() {
                rollout.step_normalized(1.0).expect("a live step");
            }
            tokens.extend(rollout.tokens().iter().cloned());
        }
    }
    tokens
}

/// Drive every second-process episode to completion under the goal-aware policy
/// and under a plain hold, so both the acting and the idle vocabulary appear.
fn second_process_tokens() -> Vec<LearningToken> {
    let mut tokens = Vec::new();
    for order in [SerializationOrder::Canonical, SerializationOrder::Permuted] {
        for mut rollout in standard_eviction_rollouts(order) {
            while !rollout.is_done() {
                let commands = goal_aware_commands(&rollout);
                rollout.step_normalized(&commands).expect("a live step");
            }
            tokens.extend(rollout.tokens().iter().cloned());
        }
        for mut rollout in standard_eviction_rollouts(order) {
            while !rollout.is_done() {
                let width = rollout.current_query_positions().len();
                rollout
                    .step_normalized(&vec![0.0; width])
                    .expect("a live step");
            }
            tokens.extend(rollout.tokens().iter().cloned());
        }
    }
    tokens
}

fn goal_aware_commands(rollout: &EvictionRollout) -> Vec<f32> {
    let positions = rollout.current_query_positions();
    let observation = rollout.case().public_observation(
        rollout.state(),
        HORIZON - rollout.steps(),
        rollout.order(),
    );
    let chosen = GoalAwarePolicy.choose_actuator_key(&observation);
    positions
        .iter()
        .map(|position| {
            let key = rollout.tokens()[*position].public.key;
            if Some(key) == chosen {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn keys_in_namespace(tokens: &[LearningToken], roles: &[Role]) -> BTreeSet<u16> {
    tokens
        .iter()
        .filter(|token| roles.contains(&token.public.role))
        .map(|token| token.public.key)
        .collect()
}

#[test]
fn the_two_processes_name_disjoint_channels() {
    let first = first_process_tokens();
    let second = second_process_tokens();

    let observation_roles = [
        Role::SchemaObservation,
        Role::Observation,
        Role::Goal,
        Role::FutureQuery,
    ];
    let actuator_roles = [
        Role::SchemaActuator,
        Role::ActionQuery,
        Role::ActionExecuted,
    ];

    let first_observations = keys_in_namespace(&first, &observation_roles);
    let second_observations = keys_in_namespace(&second, &observation_roles);
    let first_actuators = keys_in_namespace(&first, &actuator_roles);
    let second_actuators = keys_in_namespace(&second, &actuator_roles);

    assert!(!first_observations.is_empty() && !second_observations.is_empty());
    assert!(
        first_observations.is_disjoint(&second_observations),
        "observation channels overlap: {:?}",
        first_observations
            .intersection(&second_observations)
            .collect::<Vec<_>>()
    );
    assert!(
        first_actuators.is_disjoint(&second_actuators),
        "command channels overlap: {:?}",
        first_actuators
            .intersection(&second_actuators)
            .collect::<Vec<_>>()
    );

    // The episode-scalar boundary key is shared on purpose. It names no channel;
    // it marks the structure of an episode, and both processes have episodes.
    let first_episode = keys_in_namespace(&first, &[Role::Boundary]);
    let second_episode = keys_in_namespace(&second, &[Role::Boundary]);
    assert_eq!(first_episode, second_episode);
    assert_eq!(first_episode, BTreeSet::from([0]));
}

#[test]
fn the_action_geometry_differs() {
    let first = first_process_tokens();
    let second = second_process_tokens();

    let first_channels = keys_in_namespace(&first, &[Role::ActionQuery]);
    let second_channels = keys_in_namespace(&second, &[Role::ActionQuery]);

    // The first process asks for one signed control per step. The second asks
    // for one channel per container and expects at most one to be actuated.
    assert_eq!(
        first_channels.len(),
        2,
        "one control key per presentation, two presentations"
    );
    assert_eq!(
        second_channels.len(),
        6,
        "three container channels per presentation, two presentations"
    );

    let executed_values = |tokens: &[LearningToken]| -> Vec<f32> {
        tokens
            .iter()
            .filter(|token| token.public.role == Role::ActionExecuted)
            .map(|token| token.public.payload[0])
            .collect()
    };
    assert!(
        executed_values(&first).iter().any(|value| *value < 0.0),
        "the first process commands a signed displacement"
    );
    assert!(
        executed_values(&second).iter().all(|value| *value >= 0.0),
        "the second process actuates a channel; no command is the negation of another"
    );
}

#[test]
fn the_second_process_publishes_no_ordered_coordinate_system() {
    let distinct_schema_values = |tokens: &[LearningToken]| -> BTreeSet<String> {
        tokens
            .iter()
            .filter(|token| token.public.role == Role::SchemaObservation)
            .map(|token| format!("{}", token.public.payload[0]))
            .collect()
    };

    // The first process's observation schema publishes five distinct coordinates
    // spanning the line, which is what makes "left" and "right" meaningful.
    let first = distinct_schema_values(&first_process_tokens());
    assert_eq!(first.len(), 5);

    // The second process's observation schema publishes exactly two values, and
    // they are band tags rather than positions. There is no coordinate to
    // compare, so no rule of the form "move towards a smaller coordinate" is
    // expressible against it.
    let second = distinct_schema_values(&second_process_tokens());
    assert_eq!(
        second,
        BTreeSet::from([format!("{ITEM_BAND_TAG}"), format!("{BLOCKER_BAND_TAG}")])
    );
}

#[test]
fn the_two_processes_share_the_goal_blind_ceiling() {
    // Same relation, same exact ceiling, obtained independently by enumeration
    // in each process. This is what makes their witness scores comparable.
    assert_eq!(
        pretraining_eviction_world::hidden_goal_public_ceiling(),
        GOAL_BLIND_CEILING
    );
    assert_eq!(
        pretraining_goal_conditioned_world::hidden_goal_public_ceiling(),
        GOAL_BLIND_CEILING
    );
}

#[test]
fn the_second_process_serializes_into_the_canonical_public_event_record() {
    // The second process adopts the first process's profile conventions on
    // purpose: selections rather than channel values, a remaining-step fraction
    // on the action query, and no auxiliary command span. That choice is what
    // makes the two processes comparable at the level of content, and the
    // canonical record is what checks the choice was actually honoured.
    for order in [SerializationOrder::Canonical, SerializationOrder::Permuted] {
        for mut rollout in standard_eviction_rollouts(order) {
            while !rollout.is_done() {
                let commands = goal_aware_commands(&rollout);
                rollout.step_normalized(&commands).expect("a live step");
            }
            let rows: Vec<PublicRow> = rollout
                .tokens()
                .iter()
                .map(|token| PublicRow {
                    role: token.public.role as u8,
                    key: token.public.key,
                    group: token.public.event,
                    payload: token.public.payload,
                })
                .collect();

            let episode = decode_episode(Profile::GoalConditionedDiagnostic, &rows)
                .expect("the second process decodes into the canonical record");
            assert_eq!(
                render_public(&episode).expect("re-renders"),
                rows,
                "case {} lost information in the canonical round trip",
                rollout.case().id
            );
        }
    }
}
