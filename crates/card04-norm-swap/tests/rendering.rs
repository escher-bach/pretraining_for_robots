//! R5 boundary and integration evidence for card 04.
//!
//! The unit tests inside `render.rs` check that the transcript says the right
//! things. These check the other half: that what leaves the crate is a valid
//! `0.3.1` envelope, that it strips back to exactly the body that was rendered,
//! and that no supervision reaches a public byte.

use pretraining_card04_norm_swap::{
    card_cases, learner_episode, learner_episodes, port_schema, render_audit, CaseKind,
    CONDITION_GOAL_SUPERSEDES_AT, CONDITION_HAZARD, CONDITION_PROHIBITED,
};
use pretraining_g0_render::{
    legacy_tokens, profiled_tokens, ENVELOPE_PROFILE, PROFILED_TOKEN_ABI_VERSION,
};
use pretraining_profiled_event::{declared_profile, strip_profile_tag};
use pretraining_world::Role;

#[test]
fn every_case_leaves_the_crate_as_a_valid_envelope_that_strips_back_exactly() {
    for (kind, episode) in learner_episodes() {
        let body = legacy_tokens(&episode).expect("renders");
        let tagged = profiled_tokens(&episode).expect("renders");
        assert_eq!(
            declared_profile(&tagged),
            Ok(ENVELOPE_PROFILE),
            "{} declared the wrong profile",
            kind.label()
        );
        let (profile, stripped) = strip_profile_tag(&tagged).expect("strips");
        assert_eq!(profile, ENVELOPE_PROFILE);
        assert_eq!(
            stripped,
            body,
            "{} did not survive the round trip",
            kind.label()
        );
    }
}

#[test]
fn supervision_lives_only_in_the_supervision_channel() {
    for (_, episode) in learner_episodes() {
        for token in legacy_tokens(&episode).expect("renders") {
            let supervised = token.supervision.action_mask.iter().any(|slot| *slot);
            if supervised {
                assert_eq!(
                    token.public.role,
                    Role::ActionQuery,
                    "only an action query carries a target"
                );
                assert!(
                    !token.supervision.future_mask,
                    "this profile emits no future query, so it may not carry a future target"
                );
            }
            // The public payload of a query row is the same whatever the teacher
            // chose: the command placeholder, its bounds, remaining steps, the
            // horizon, and the presence flag. No slot varies with supervision.
            if token.public.role == Role::ActionQuery {
                assert_eq!(token.public.payload[0], 0.0);
                assert_eq!(token.public.payload[5], 1.0);
            }
        }
    }
}

#[test]
fn the_episode_fits_the_declared_learner_context() {
    let audit = render_audit().expect("renders");
    // The checked-in profile declares a 2048-token context. The margin is
    // reported rather than assumed so a later family that grows cannot silently
    // rely on it.
    assert!(
        audit.report.max_profiled_records <= 2048,
        "an episode of {} records does not fit the declared context",
        audit.report.max_profiled_records
    );
    assert_eq!(audit.report.envelope_abi, PROFILED_TOKEN_ABI_VERSION);
    assert_eq!(audit.report.episodes, card_cases().len());
}

#[test]
fn the_schema_publishes_the_whole_body_and_nothing_conditional() {
    let schema = port_schema();
    assert_eq!(schema.observations.len(), 5, "five configuration cells");
    assert_eq!(
        schema.actuators.len(),
        3,
        "three always-supported actuators"
    );
    // Card 04 holds identification at zero on purpose, so every case publishes
    // the same ports. A later card varies this; if this assertion ever needs
    // relaxing for card 04, the card's claim has changed.
    for (_, episode) in learner_episodes() {
        assert_eq!(episode.schema, schema);
    }
}

#[test]
fn the_conditions_a_case_publishes_match_the_contract_fields_it_has() {
    for case in card_cases() {
        let facts: Vec<_> = learner_episode(&case.contract)
            .groups
            .iter()
            .flat_map(|group| group.facts.clone())
            .collect();
        let has = |wanted: u16| {
            facts.iter().any(|fact| {
                matches!(fact, pretraining_g0_render::G0Fact::Condition { code, .. } if *code == wanted)
            })
        };
        assert_eq!(
            has(CONDITION_PROHIBITED),
            case.contract.no_go.is_some(),
            "{} published the wrong prohibition state",
            case.kind.label()
        );
        assert_eq!(
            has(CONDITION_HAZARD),
            case.contract.hazard.is_some(),
            "{} published the wrong hazard state",
            case.kind.label()
        );
        assert_eq!(
            has(CONDITION_GOAL_SUPERSEDES_AT),
            case.contract.switch.is_some(),
            "{} published the wrong switch state",
            case.kind.label()
        );
    }
}

#[test]
fn an_announced_switch_is_visible_before_the_first_decision_and_an_unannounced_one_is_not() {
    let first_decision_group = |kind: CaseKind| {
        let case = card_cases()
            .into_iter()
            .find(|case| case.kind == kind)
            .expect("the card has this case");
        let episode = learner_episode(&case.contract);
        let index = episode
            .groups
            .iter()
            .position(|group| {
                group
                    .facts
                    .iter()
                    .any(|fact| matches!(fact, pretraining_g0_render::G0Fact::ActionQuery { .. }))
            })
            .expect("every case has a decision");
        episode.groups[..index]
            .iter()
            .flat_map(|group| group.facts.clone())
            .collect::<Vec<_>>()
    };

    let announced = first_decision_group(CaseKind::NegativeSwitchAnnounced);
    let unannounced = first_decision_group(CaseKind::WitnessSwitch);
    let goals = |facts: &[pretraining_g0_render::G0Fact]| {
        facts
            .iter()
            .filter(|fact| matches!(fact, pretraining_g0_render::G0Fact::Goal { .. }))
            .count()
    };
    assert_eq!(goals(&announced), 2, "both outcomes are on the table");
    assert_eq!(
        goals(&unannounced),
        1,
        "only the first outcome is published"
    );
}
