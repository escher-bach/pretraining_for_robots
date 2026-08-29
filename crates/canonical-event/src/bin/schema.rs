//! Emit the canonical record's field schema and its round-trip verdicts as one
//! JSON artifact, so the specification can be reviewed without reading Rust.

use pretraining_canonical_event::corpus::reference_episodes;
use pretraining_canonical_event::{
    decode_episode, decode_supervision, render_public, render_supervision, EventKind, KindSchema,
    Profile, PublicRow, ACTION_HORIZON, CANONICAL_RECORD_VERSION, PAYLOAD_DIM, RENDER_TARGET_ABI,
};
use serde::Serialize;

#[derive(Serialize)]
struct ProfileReport {
    name: &'static str,
    schema_hash: String,
    emitted_kinds: Vec<&'static str>,
    never_emitted_kinds: Vec<&'static str>,
    field_schema: Vec<KindSchema>,
}

#[derive(Serialize)]
struct RoundTripReport {
    episode: String,
    profile: &'static str,
    groups: usize,
    records: usize,
    public_round_trip_exact: bool,
    supervision_round_trip_exact: bool,
    supervised_records: usize,
}

#[derive(Serialize)]
struct CollisionReport {
    row: Vec<f32>,
    role: &'static str,
    reading_under_calibrated_monomial: String,
    reading_under_goal_conditioned_diagnostic: String,
}

#[derive(Serialize)]
struct Artifact {
    canonical_record_version: &'static str,
    render_target_abi: &'static str,
    payload_dim: usize,
    action_horizon: usize,
    claim: &'static str,
    profiles: Vec<ProfileReport>,
    round_trips: Vec<RoundTripReport>,
    profile_collisions: Vec<CollisionReport>,
    not_claimed: Vec<&'static str>,
}

fn profile_report(profile: Profile) -> ProfileReport {
    ProfileReport {
        name: profile.as_str(),
        schema_hash: format!("{:#018x}", profile.schema_hash()),
        emitted_kinds: EventKind::ALL
            .into_iter()
            .filter(|kind| profile.emits(*kind))
            .map(EventKind::as_str)
            .collect(),
        never_emitted_kinds: EventKind::ALL
            .into_iter()
            .filter(|kind| !profile.emits(*kind))
            .map(EventKind::as_str)
            .collect(),
        field_schema: profile.field_schema(),
    }
}

fn collision(role: EventKind, payload: [f32; PAYLOAD_DIM]) -> CollisionReport {
    let row = PublicRow {
        role: role.role_code(),
        key: 0,
        group: 0,
        payload,
    };
    let read = |profile: Profile| match decode_episode(profile, &[row]) {
        Ok(episode) => format!("{:?}", episode.groups[0].records[0].fact),
        Err(error) => format!("rejected: {error}"),
    };
    CollisionReport {
        row: payload.to_vec(),
        role: role.as_str(),
        reading_under_calibrated_monomial: read(Profile::CalibratedMonomial),
        reading_under_goal_conditioned_diagnostic: read(Profile::GoalConditionedDiagnostic),
    }
}

fn main() {
    let mut round_trips = Vec::new();
    for (name, episode) in reference_episodes() {
        let rows = render_public(episode.public()).expect("the corpus renders");
        let decoded = decode_episode(episode.public().profile, &rows).expect("the corpus decodes");
        let supervision_rows = render_supervision(&episode).expect("supervision renders");
        let decoded_supervision =
            decode_supervision(episode.public(), &supervision_rows).expect("supervision decodes");
        round_trips.push(RoundTripReport {
            episode: name.to_string(),
            profile: episode.public().profile.as_str(),
            groups: episode.public().groups.len(),
            records: episode.public().record_count(),
            public_round_trip_exact: &decoded == episode.public()
                && render_public(&decoded).expect("re-renders") == rows,
            supervision_round_trip_exact: &decoded_supervision == episode.supervision(),
            supervised_records: episode.supervision().entries.len(),
        });
    }

    let artifact = Artifact {
        canonical_record_version: CANONICAL_RECORD_VERSION,
        render_target_abi: RENDER_TARGET_ABI,
        payload_dim: PAYLOAD_DIM,
        action_horizon: ACTION_HORIZON,
        claim: "the public information of an episode can be written as an explicitly \
                typed record and recovered exactly from the existing float layout, \
                once the producer profile is stated",
        profiles: Profile::ALL.into_iter().map(profile_report).collect(),
        round_trips,
        profile_collisions: vec![
            collision(
                EventKind::Observation,
                [0.5, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            ),
            collision(EventKind::Goal, [0.5, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            collision(
                EventKind::ActionQuery,
                [0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            ),
        ],
        not_claimed: vec![
            "no claim that a learner is invariant to renaming or reordering; this is apparatus only",
            "no rename-invariant canonical form, which would be a graph-canonization claim",
            "no migration of the production world or checkpoint ABI",
            "no byte renderer and no second tokenizer",
            "no learner was run to produce this artifact",
        ],
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&artifact).expect("the artifact is serializable")
    );
}
