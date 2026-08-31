//! Card 03 stage A rendered onto the shared learner event boundary.
//!
//! The scaffold is part of the episode, not a preamble to it: the four
//! calibration pulses are published as executed actions with their
//! consequences, and only then does the scored decision open. That ordering is
//! the composite's "observe, then act" lesson made concrete — a family whose
//! scaffold speaks before the first decision must be rendered so the learner
//! hears it before deciding, or the public ceiling it is taught against is not
//! the ceiling it actually has.
//!
//! The teacher is [`pretraining_g0_contract::public_optimal_actions_at`], which
//! reads the calibration outcome and the announcement and never the support
//! set.

use pretraining_g0_contract::public_optimal_actions_at;
use pretraining_g0_render::{
    boundary_check, rendering_report, BoundaryEvidence, BoundarySubtype, Content, G0Episode,
    G0Fact, G0Group, KeyNamespace, Port, PortSchema, RenderFault, RenderingReport,
};
use serde::{Deserialize, Serialize};

use crate::{
    card_cases, support_ambiguity, Action, BodyIdentification, Case, CaseKind, Contract,
    CALIBRATION_CELL, HORIZON, RING,
};

/// The actuator an announcement restores.
pub const CONDITION_RESTORED: u16 = 1;
/// Whether the scaffold shows anything. A family fact.
pub const CONDITION_CALIBRATION: u16 = 2;

pub const EPISODE_KEY_BODY: u16 = 0;

pub fn port_schema() -> PortSchema {
    PortSchema {
        observations: (0..RING as u16).map(Port::unit).collect(),
        actuators: Action::ALL
            .into_iter()
            .map(|action| Port::signed(action.index() as u16))
            .collect(),
    }
}

fn observation(cell: usize) -> G0Fact {
    G0Fact::Observation {
        key: cell as u16,
        content: Content::Selection,
    }
}

fn condition(code: u16, value: f64) -> G0Fact {
    G0Fact::Condition {
        key: EPISODE_KEY_BODY,
        namespace: KeyNamespace::Episode,
        code,
        value,
        lower: 0.0,
        upper: 8.0,
    }
}

/// Render one contract as a learner-visible episode taught by the public policy.
pub fn learner_episode(contract: &Contract) -> Result<G0Episode, RenderFault> {
    let set = support_ambiguity(contract);
    let mut groups = vec![
        G0Group::one(G0Fact::Boundary(BoundarySubtype::TaskReset)),
        G0Group::new(vec![
            G0Fact::Goal {
                key: contract.goal() as u16,
                namespace: KeyNamespace::Observation,
                content: Content::Selection,
            },
            condition(CONDITION_CALIBRATION, contract.calibration as u8 as f64),
        ]),
    ];

    // The scaffold. It is free — no scored decision is offered during it — and
    // mandatory, so every episode carries the same four pulses in the same
    // order and a learner cannot read the goal off the scaffold's shape.
    groups.push(G0Group::one(G0Fact::Boundary(
        BoundarySubtype::CalibrationReset,
    )));
    groups.push(G0Group::one(observation(CALIBRATION_CELL)));
    for (pulse, cell) in Action::PULSE_ORDER
        .into_iter()
        .zip(contract.calibration_trace())
    {
        let published = match contract.calibration {
            crate::Calibration::Full => pulse,
            crate::Calibration::Uninformative => Action::Hold,
        };
        groups.push(G0Group::one(G0Fact::ActionExecuted {
            actuator: published.index() as u16,
        }));
        groups.push(G0Group::one(observation(cell)));
    }

    // The announcement arrives after the scaffold, which is what the
    // restoration control turns on.
    if let Some(actuator) = contract.announced_restoration() {
        groups.push(G0Group::one(condition(CONDITION_RESTORED, actuator as f64)));
    }

    groups.push(G0Group::one(observation(contract.start)));

    let correct = public_optimal_actions_at(&BodyIdentification, &set, &[], HORIZON);
    if correct.is_empty() {
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
                selected: correct.contains(&candidate),
            })
            .collect(),
    ));
    let chosen = correct[0];
    groups.push(G0Group::one(G0Fact::ActionExecuted {
        actuator: chosen.index() as u16,
    }));
    groups.push(G0Group::one(observation(
        contract.effect(contract.start, chosen),
    )));
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
    pub start: usize,
    pub evidence: BoundaryEvidence,
    pub taught_command: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderAudit {
    pub report: RenderingReport,
    pub cases: Vec<CaseRendering>,
    /// The scaffold is published before the only decision, in every episode.
    pub the_scaffold_speaks_before_the_decision: bool,
    /// The taught command is the one this body actually drives.
    pub the_taught_command_is_the_supported_one: bool,
    /// Two bodies differing only in support render the same scaffold shape:
    /// same pulses, same order, different consequences.
    pub the_scaffold_shape_is_body_independent: bool,
}

pub fn render_audit() -> Result<RenderAudit, RenderFault> {
    let cases: Vec<Case> = card_cases();
    let episodes: Vec<G0Episode> = cases
        .iter()
        .map(|case| learner_episode(&case.contract))
        .collect::<Result<_, _>>()?;
    let report = rendering_report(&episodes)?;

    let mut rendered = Vec::with_capacity(cases.len());
    let mut scaffold_first = true;
    let mut taught_supported = true;
    let mut shape_stable = true;

    for (case, episode) in cases.iter().zip(&episodes) {
        let decision = episode
            .groups
            .iter()
            .position(|group| {
                group
                    .facts
                    .iter()
                    .any(|fact| matches!(fact, G0Fact::ActionQuery { .. }))
            })
            .expect("every episode has a decision");
        let pulses = episode.groups[..decision]
            .iter()
            .flat_map(|group| group.facts.iter())
            .filter(|fact| matches!(fact, G0Fact::ActionExecuted { .. }))
            .count();
        scaffold_first &= pulses == Action::PULSE_ORDER.len();

        let taught: Vec<String> = episode
            .selected_actuators()
            .first()
            .map(|set| {
                set.iter()
                    .map(|index| {
                        crate::action_from_index(*index as usize)
                            .expect("a known actuator")
                            .name()
                            .to_string()
                    })
                    .collect()
            })
            .unwrap_or_default();
        taught_supported &= taught.iter().all(|name| {
            Action::ALL
                .into_iter()
                .find(|action| action.name() == name)
                .is_some_and(|action| {
                    case.contract.effect(case.contract.start, action) == case.contract.goal()
                })
        });

        // The pulse *sequence* must not depend on the body: only what the
        // pulses do may. A scaffold whose shape varied would let a learner read
        // support off the episode's structure instead of off its content.
        let twin = Contract {
            support: case.contract.swap_aliased_support().support,
            ..case.contract.clone()
        };
        let shape = |episode: &G0Episode| -> Vec<u16> {
            episode
                .groups
                .iter()
                .flat_map(|group| group.facts.iter())
                .filter_map(|fact| match fact {
                    G0Fact::ActionExecuted { actuator } => Some(*actuator),
                    _ => None,
                })
                .take(Action::PULSE_ORDER.len())
                .collect()
        };
        shape_stable &= shape(episode) == shape(&learner_episode(&twin)?);

        rendered.push(CaseRendering {
            kind: case.kind.label().to_string(),
            start: case.contract.start,
            evidence: boundary_check(episode)?,
            taught_command: taught,
        });
    }

    Ok(RenderAudit {
        report,
        cases: rendered,
        the_scaffold_speaks_before_the_decision: scaffold_first,
        the_taught_command_is_the_supported_one: taught_supported,
        the_scaffold_shape_is_body_independent: shape_stable,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rendering_carries_the_whole_claim() {
        let audit = render_audit().expect("renders");
        assert!(audit.report.every_episode_round_trips);
        assert!(audit.the_scaffold_speaks_before_the_decision);
        assert!(audit.the_taught_command_is_the_supported_one);
        assert!(audit.the_scaffold_shape_is_body_independent);
    }

    #[test]
    fn the_two_witness_bodies_are_taught_different_commands() {
        let cases = crate::cases_of(CaseKind::WitnessIdentifiedSupport);
        for pair in cases.chunks(2) {
            let taught: Vec<Vec<u16>> = pair
                .iter()
                .map(|case| {
                    learner_episode(&case.contract)
                        .expect("renders")
                        .selected_actuators()[0]
                        .clone()
                })
                .collect();
            assert_eq!(taught.len(), 2);
            assert_ne!(taught[0], taught[1]);
        }
    }

    #[test]
    fn the_restoration_is_published_after_the_scaffold_and_before_the_decision() {
        let case = crate::cases_of(CaseKind::NegativeAnnouncedRestoration)
            .into_iter()
            .next()
            .expect("a restoration case");
        let episode = learner_episode(&case.contract).expect("renders");
        let position = |predicate: fn(&G0Fact) -> bool| {
            episode
                .groups
                .iter()
                .position(|group| group.facts.iter().any(predicate))
                .expect("present")
        };
        let announced = position(
            |fact| matches!(fact, G0Fact::Condition { code, .. } if *code == CONDITION_RESTORED),
        );
        let decision = position(|fact| matches!(fact, G0Fact::ActionQuery { .. }));
        assert!(announced < decision);
        assert_eq!(
            episode.selected_actuators()[0],
            vec![Action::Leap.index() as u16],
            "the announcement, not the scaffold, decides"
        );
    }
}
