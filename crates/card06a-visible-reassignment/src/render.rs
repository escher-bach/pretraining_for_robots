//! Card 06 stage A rendered onto the shared learner event boundary.
//!
//! Two details are carried over from the composite deliberately.
//!
//! The channels publish **values**, not selections: the `FiniteG0` profile's
//! content-kind flag exists for exactly this, and the perturbation is a
//! magnitude rather than a choice of key.
//!
//! The opening decision is taught as a **set**. Either pulse locates both
//! sources, so the world is indifferent between them, and marking one would
//! teach a tie-break the contract does not have. The episode then executes the
//! goal channel's pulse, which is the realization the case declares — the
//! teacher asserts what is correct, not what was done.

use pretraining_g0_contract::public_optimal_actions_at;
use pretraining_g0_render::{
    boundary_check, rendering_report, BoundaryEvidence, BoundarySubtype, Content, G0Episode,
    G0Fact, G0Group, KeyNamespace, Port, PortSchema, RenderFault, RenderingReport,
};
use serde::{Deserialize, Serialize};

use crate::{
    assignment_ambiguity, card_cases, pulse_for, Action, Case, CaseKind, Contract, Source,
    VisibleReassignment, CHANNELS, HORIZON,
};

/// The arm of the family. A published family parameter.
pub const CONDITION_VARIANT: u16 = 1;
/// Which source the goal names. A published family parameter.
pub const CONDITION_NAMING: u16 = 2;
/// Whether the assignment-change boundary is announced.
pub const CONDITION_BOUNDARY: u16 = 3;
/// The identity tag, where the family publishes one.
pub const CONDITION_TAG: u16 = 4;

pub const EPISODE_KEY_FAMILY: u16 = 0;
pub const EPISODE_KEY_TAG: u16 = 1;

pub fn port_schema() -> PortSchema {
    PortSchema {
        observations: (0..CHANNELS as u16)
            .map(|key| Port {
                key,
                reference: 0.0,
                lower: -2.0,
                upper: 2.0,
            })
            .collect(),
        actuators: Action::ALL
            .into_iter()
            .map(|action| Port::signed(action.index() as u16))
            .collect(),
    }
}

fn condition(key: u16, code: u16, value: f64) -> G0Fact {
    G0Fact::Condition {
        key,
        namespace: KeyNamespace::Episode,
        code,
        value,
        lower: 0.0,
        upper: 4.0,
    }
}

fn observation(channel: usize, value: i32) -> G0Fact {
    G0Fact::Observation {
        key: channel as u16,
        content: Content::Value {
            value: f64::from(value),
            lower: -2.0,
            upper: 2.0,
        },
    }
}

pub fn learner_episode(contract: &Contract) -> Result<G0Episode, RenderFault> {
    let set = assignment_ambiguity(contract);
    let mut opening = vec![
        G0Fact::Goal {
            key: contract.goal_channel as u16,
            namespace: KeyNamespace::Observation,
            content: Content::Selection,
        },
        condition(
            EPISODE_KEY_FAMILY,
            CONDITION_VARIANT,
            contract.variant as u8 as f64,
        ),
        condition(
            EPISODE_KEY_FAMILY,
            CONDITION_NAMING,
            contract.naming as u8 as f64,
        ),
        condition(
            EPISODE_KEY_FAMILY,
            CONDITION_BOUNDARY,
            f64::from(u8::from(contract.boundary_visible)),
        ),
        observation(0, 0),
        observation(1, 0),
    ];
    if contract.tags_visible() {
        opening.push(condition(
            EPISODE_KEY_TAG,
            CONDITION_TAG,
            contract.before.channel_of(Source::Left) as f64,
        ));
    }

    let mut groups = vec![
        G0Group::one(G0Fact::Boundary(BoundarySubtype::TaskReset)),
        G0Group::new(opening),
    ];

    let first = public_optimal_actions_at(&VisibleReassignment, &set, &[], HORIZON);
    if first.is_empty() {
        return Err(RenderFault::NoSelectedAction {
            group: groups.len(),
        });
    }
    groups.push(G0Group::new(
        Action::ALL
            .into_iter()
            .map(|candidate| G0Fact::ActionQuery {
                actuator: candidate.index() as u16,
                remaining: HORIZON,
                selected: first.contains(&candidate),
            })
            .collect(),
    ));

    // The case declares which pulse is realized. It has to be one the teacher
    // called correct, or the episode would supervise one action and execute
    // another.
    let pulse = pulse_for(contract.goal_channel);
    if !first.contains(&pulse) {
        return Err(RenderFault::TeacherWouldLeak {
            detail: "the realized pulse is not among the publicly optimal openings".into(),
        });
    }
    groups.push(G0Group::one(G0Fact::ActionExecuted {
        actuator: pulse.index() as u16,
    }));

    let pulsed = contract.goal_channel;
    let mut after_boundary = Vec::new();
    if contract.boundary_visible {
        after_boundary.push(G0Fact::Boundary(BoundarySubtype::CalibrationReset));
    }
    for channel in 0..CHANNELS {
        // A channel with no writer publishes nothing rather than publishing a
        // zero that would read as a real value.
        if let Some(value) = contract.channel_value(channel, pulsed) {
            after_boundary.push(observation(channel, value));
        }
    }
    if contract.tags_visible() {
        after_boundary.push(condition(
            EPISODE_KEY_TAG,
            CONDITION_TAG,
            contract.after.channel_of(Source::Left) as f64,
        ));
    }
    groups.push(G0Group::new(after_boundary));

    let second = public_optimal_actions_at(&VisibleReassignment, &set, &[pulse], HORIZON);
    if second.is_empty() {
        return Err(RenderFault::NoSelectedAction {
            group: groups.len(),
        });
    }
    groups.push(G0Group::new(
        Action::ALL
            .into_iter()
            .map(|candidate| G0Fact::ActionQuery {
                actuator: candidate.index() as u16,
                remaining: 1,
                selected: second.contains(&candidate),
            })
            .collect(),
    ));
    groups.push(G0Group::one(G0Fact::Boundary(BoundarySubtype::EpisodeEnd)));

    Ok(G0Episode::new(port_schema(), HORIZON, groups))
}

pub fn learner_episodes() -> Result<Vec<(CaseKind, G0Episode)>, RenderFault> {
    card_cases()
        .into_iter()
        .map(|case| learner_episode(&case.contract).map(|episode| (case.kind, episode)))
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseRendering {
    pub kind: String,
    pub seed: u64,
    pub evidence: BoundaryEvidence,
    pub taught_opening: Vec<String>,
    pub taught_drive: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderAudit {
    pub report: RenderingReport,
    pub cases: Vec<CaseRendering>,
    /// Channels carry values, not selections.
    pub values_not_selections: bool,
    /// No episode publishes an assignment before something is pulsed.
    pub no_assignment_in_the_opening: bool,
    /// Either pulse is taught as correct, because the world is indifferent.
    pub both_openings_are_taught: bool,
    /// The taught drive follows the perturbation on every witness episode.
    pub the_taught_drive_follows_the_perturbation: bool,
}

pub fn render_audit() -> Result<RenderAudit, RenderFault> {
    let cases: Vec<Case> = card_cases();
    let episodes: Vec<G0Episode> = cases
        .iter()
        .map(|case| learner_episode(&case.contract))
        .collect::<Result<_, _>>()?;
    let report = rendering_report(&episodes)?;

    let mut rendered = Vec::with_capacity(cases.len());
    let mut both_openings = true;
    let mut follows = true;
    let mut no_assignment = true;

    for (case, episode) in cases.iter().zip(&episodes) {
        let taught = episode.selected_actuators();
        let names = |set: &Vec<u16>| -> Vec<String> {
            set.iter()
                .map(|index| {
                    crate::action_from_index(*index as usize)
                        .expect("a known actuator")
                        .name()
                        .to_string()
                })
                .collect()
        };
        let opening = taught.first().map(names).unwrap_or_default();
        let drive = taught.get(1).map(names).unwrap_or_default();
        both_openings &= opening.len() == 2;

        if case.kind == CaseKind::Witness {
            let expected = crate::drive_for(crate::PublicPolicy::drive(
                &crate::ValueTracking,
                &case.contract,
                case.contract.goal_channel,
            ));
            follows &= drive == vec![expected.name().to_string()];
        }

        // Nothing before the first executed action may depend on the hidden
        // assignment. The tag variant publishes a tag, which is a *published*
        // fact of that family, so it is compared against its own twin.
        let twin = Contract {
            before: case.contract.before.flipped(),
            after: case.contract.after.flipped(),
            ..case.contract
        };
        if !case.contract.tags_visible() {
            let twin_episode = learner_episode(&twin)?;
            let prefix = |episode: &G0Episode| episode.groups[..2].to_vec();
            no_assignment &= prefix(episode) == prefix(&twin_episode);
        }

        rendered.push(CaseRendering {
            kind: case.kind.label().to_string(),
            seed: case.contract.seed,
            evidence: boundary_check(episode)?,
            taught_opening: opening,
            taught_drive: drive,
        });
    }

    Ok(RenderAudit {
        report,
        cases: rendered,
        values_not_selections: true,
        no_assignment_in_the_opening: no_assignment,
        both_openings_are_taught: both_openings,
        the_taught_drive_follows_the_perturbation: follows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rendering_carries_the_whole_claim() {
        let audit = render_audit().expect("renders");
        assert!(audit.report.every_episode_round_trips);
        assert!(audit.both_openings_are_taught);
        assert!(audit.the_taught_drive_follows_the_perturbation);
        assert!(audit.no_assignment_in_the_opening);
    }

    #[test]
    fn a_reassigned_witness_is_taught_the_other_channel() {
        for case in card_cases() {
            if case.kind != CaseKind::Witness || !case.contract.reassigned() {
                continue;
            }
            let episode = learner_episode(&case.contract).expect("renders");
            let taught = episode.selected_actuators();
            let pulsed = case.contract.goal_channel;
            assert_eq!(
                taught[1],
                vec![crate::drive_for(1 - pulsed).index() as u16],
                "a reassigned source is not in the channel it was pulsed in"
            );
        }
    }
}
