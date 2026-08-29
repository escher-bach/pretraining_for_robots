//! Card 03 rendered onto the shared learner event boundary.
//!
//! The scaffold is the interesting part. Calibration is not a separate protocol:
//! it is the same public vocabulary the scored phase uses — an executed action
//! and the configuration it produced — bracketed by the two boundary subtypes
//! the ABI already has. A learner cannot tell the scaffold from the task by the
//! record kinds; it tells them apart by the boundary, which is what makes
//! "identify the body from what it did" a real problem rather than a lookup.
//!
//! What is deliberately *not* published:
//!
//! - the support set. It reaches the learner only through the cells the pulses
//!   produced. [`crate::AuditReport::calibration_identifies_the_body_everywhere`]
//!   is what establishes that this is enough.
//! - the reachability of the goal. That is the quantity the card asks the
//!   learner to compute, so publishing it would delete the card.
//!
//! What *is* published, because the contrast requires it, is the support
//! restoration — announced at episode start, with the decision index it takes
//! effect at. Card 04 established why: an unannounced change cannot require
//! behaviour to update before it is published. Here the point is sharper still,
//! because the fallback is absorbing: an unannounced restoration would arrive
//! after a correctly-ended episode and could not change anything at all.

use pretraining_g0_render::{
    boundary_check, rendering_report, step_fraction, BoundaryEvidence, BoundarySubtype, Content,
    G0Episode, G0Fact, G0Group, KeyNamespace, Port, PortSchema, RenderFault, RenderingReport,
};
use serde::{Deserialize, Serialize};

use crate::{
    card_cases, goal_is_reachable, optimal_first_actions, Action, Calibration, Case, CaseKind,
    Contract, ExactPostCalibration, PublicPolicy, PublicView, HORIZON, RING,
};

/// The named actuator is supported from the carried decision index onward.
pub const CONDITION_SUPPORT_RESTORED_AT: u16 = 1;

/// The body and interface card 03 publishes.
///
/// Every case publishes the same five actuators. That is the card, not an
/// oversight: the *interface* is constant and the *support* is not, so a learner
/// that read affordance off the schema would be reading the wrong thing. The
/// schema is where card 03 and card 04 look identical and where their claims
/// diverge.
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

/// Render one contract as a learner-visible episode taught by the exact
/// post-calibration policy.
///
/// Refuses a contract whose calibration is uninformative. The exact teacher
/// reads the body, which is public *because calibration identified it*; with an
/// uninformative scaffold the same policy would be reading an unpublished field,
/// and the shared boundary has a fault for exactly that.
pub fn learner_episode(contract: &Contract) -> Result<G0Episode, RenderFault> {
    if contract.calibration != Calibration::Full {
        return Err(RenderFault::TeacherWouldLeak {
            detail: "the exact policy reads a body that an uninformative calibration never showed"
                .into(),
        });
    }

    let mut groups = vec![
        G0Group::one(G0Fact::Boundary(BoundarySubtype::CalibrationReset)),
        G0Group::one(observation(contract.start)),
    ];

    // The scaffold acts and the learner watches. Pulses carry no action query,
    // so nothing here is scored and nothing here is supervised.
    let mut cell = contract.start;
    for pulse in contract.calibration_pulses() {
        groups.push(G0Group::one(G0Fact::ActionExecuted {
            actuator: pulse.index() as u16,
        }));
        cell = contract.advance(cell, 0, pulse);
        groups.push(G0Group::one(observation(cell)));
    }

    groups.push(G0Group::one(G0Fact::Boundary(BoundarySubtype::TaskReset)));
    let mut opening = vec![
        observation(contract.start),
        G0Fact::Goal {
            key: contract.goal as u16,
            namespace: KeyNamespace::Observation,
            content: Content::Selection,
        },
    ];
    if let Some(restore) = contract.restore {
        opening.push(G0Fact::Condition {
            key: restore.actuator as u16,
            namespace: KeyNamespace::Actuator,
            code: CONDITION_SUPPORT_RESTORED_AT,
            // The decision index from which the actuator is driven, as a
            // fraction of the action head so the payload is exact.
            value: step_fraction(restore.after_step + 1),
            lower: 0.0,
            upper: 1.0,
        });
    }
    groups.push(G0Group::new(opening));

    cell = contract.start;
    for executed in 0..HORIZON {
        let chosen = ExactPostCalibration.act(&PublicView {
            contract,
            cell,
            executed,
        });
        groups.push(G0Group::new(
            Action::ALL
                .into_iter()
                .map(|action| G0Fact::ActionQuery {
                    actuator: action.index() as u16,
                    remaining: HORIZON - executed,
                    selected: action == chosen,
                })
                .collect(),
        ));
        groups.push(G0Group::one(G0Fact::ActionExecuted {
            actuator: chosen.index() as u16,
        }));
        if chosen == Action::Fallback {
            // The fallback is absorbing, so there is no further decision to
            // offer. Continuing to query would publish choices that cannot
            // change the outcome and would inflate the supervised record count
            // of exactly the arm the card cares most about.
            break;
        }
        cell = contract.advance(cell, executed, chosen);
        groups.push(G0Group::one(observation(cell)));
    }

    groups.push(G0Group::one(G0Fact::Boundary(BoundarySubtype::EpisodeEnd)));
    Ok(G0Episode::new(port_schema(), HORIZON, groups))
}

/// Every case rendered, keeping its kind so a pilot can score arms apart.
pub fn learner_episodes() -> Result<Vec<(CaseKind, G0Episode)>, RenderFault> {
    card_cases()
        .into_iter()
        .map(|case| learner_episode(&case.contract).map(|episode| (case.kind, episode)))
        .collect()
}

/// One case's boundary evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseRendering {
    pub kind: String,
    pub goal: usize,
    pub evidence: BoundaryEvidence,
    pub taught_first_action: String,
    pub optimal_first_actions: Vec<String>,
    pub goal_is_reachable: bool,
    /// Decisions actually offered. A fallback ends the episode, so an
    /// unreachable-goal arm is shorter than a reachable one.
    pub decisions_offered: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderAudit {
    pub report: RenderingReport,
    pub cases: Vec<CaseRendering>,
    /// The teacher's first move is always in the exact optimal set.
    pub teacher_is_optimal_everywhere: bool,
    /// The teacher falls back exactly on the unreachable goals.
    pub fallback_is_taught_exactly_when_the_goal_is_out_of_reach: bool,
    /// An uninformative-calibration contract cannot be rendered at all.
    pub uninformative_calibration_is_refused: bool,
}

pub fn render_audit() -> Result<RenderAudit, RenderFault> {
    let cases: Vec<Case> = card_cases();
    let episodes: Vec<G0Episode> = cases
        .iter()
        .map(|case| learner_episode(&case.contract))
        .collect::<Result<_, _>>()?;
    let report = rendering_report(&episodes)?;

    let mut rendered = Vec::with_capacity(cases.len());
    let mut teacher_optimal = true;
    let mut fallback_matches_reach = true;
    for (case, episode) in cases.iter().zip(&episodes) {
        let evidence = boundary_check(episode)?;
        let taught = ExactPostCalibration.act(&PublicView {
            contract: &case.contract,
            cell: case.contract.start,
            executed: 0,
        });
        let optimal = optimal_first_actions(&case.contract);
        teacher_optimal &= optimal.contains(&taught);
        let reachable = goal_is_reachable(&case.contract);
        fallback_matches_reach &= (taught == Action::Fallback) == !reachable;
        rendered.push(CaseRendering {
            kind: case.kind.label().to_string(),
            goal: case.contract.goal,
            decisions_offered: evidence.decisions,
            evidence,
            taught_first_action: taught.name().to_string(),
            optimal_first_actions: optimal
                .into_iter()
                .map(|action| action.name().to_string())
                .collect(),
            goal_is_reachable: reachable,
        });
    }

    let blind = cases[0]
        .contract
        .clone()
        .with_calibration(Calibration::Uninformative);
    Ok(RenderAudit {
        report,
        cases: rendered,
        teacher_is_optimal_everywhere: teacher_optimal,
        fallback_is_taught_exactly_when_the_goal_is_out_of_reach: fallback_matches_reach,
        uninformative_calibration_is_refused: matches!(
            learner_episode(&blind),
            Err(RenderFault::TeacherWouldLeak { .. })
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_teacher_falls_back_exactly_when_the_goal_is_out_of_reach() {
        let audit = render_audit().expect("renders");
        assert!(audit.teacher_is_optimal_everywhere);
        assert!(audit.fallback_is_taught_exactly_when_the_goal_is_out_of_reach);
        assert!(audit.uninformative_calibration_is_refused);
        assert!(audit.report.every_episode_round_trips);
    }

    #[test]
    fn the_scored_phase_never_publishes_the_support_set() {
        for (_, episode) in learner_episodes().expect("renders") {
            for group in &episode.groups {
                for fact in &group.facts {
                    if let G0Fact::Condition { code, .. } = fact {
                        assert_eq!(
                            *code, CONDITION_SUPPORT_RESTORED_AT,
                            "the only condition this card publishes is the restoration"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn two_bodies_with_the_same_goal_differ_only_in_the_calibration_trace() {
        let unreachable = card_cases()
            .into_iter()
            .find(|case| case.kind == CaseKind::WitnessUnreachableFallback)
            .expect("the card has this witness");
        let capable = Contract {
            support: crate::full_body(),
            ..unreachable.contract.clone()
        };
        let limited = learner_episode(&unreachable.contract).expect("renders");
        let full = learner_episode(&capable).expect("renders");

        let scaffold = |episode: &G0Episode| {
            let end = episode
                .groups
                .iter()
                .position(|group| {
                    group
                        .facts
                        .iter()
                        .any(|fact| matches!(fact, G0Fact::Boundary(BoundarySubtype::TaskReset)))
                })
                .expect("every episode has a task reset");
            episode.groups[..end].to_vec()
        };
        assert_ne!(
            scaffold(&limited),
            scaffold(&full),
            "calibration must separate the two bodies"
        );
        assert_eq!(
            limited.schema, full.schema,
            "and the interface must not: the schema is the same for both"
        );
    }

    #[test]
    fn an_announced_restoration_is_visible_before_the_first_decision() {
        let restore = card_cases()
            .into_iter()
            .find(|case| case.kind == CaseKind::WitnessRestore)
            .expect("the card has this witness");
        let plain = card_cases()
            .into_iter()
            .find(|case| case.kind == CaseKind::NegativeNoRestore)
            .expect("the card has this negative");

        let before_first_decision = |contract: &Contract| {
            let episode = learner_episode(contract).expect("renders");
            let index = episode
                .groups
                .iter()
                .position(|group| {
                    group
                        .facts
                        .iter()
                        .any(|fact| matches!(fact, G0Fact::ActionQuery { .. }))
                })
                .expect("every case has a decision");
            episode.groups[..index]
                .iter()
                .flat_map(|group| group.facts.clone())
                .collect::<Vec<_>>()
        };
        let has_condition = |facts: &[G0Fact]| {
            facts
                .iter()
                .any(|fact| matches!(fact, G0Fact::Condition { .. }))
        };
        assert!(has_condition(&before_first_decision(&restore.contract)));
        assert!(!has_condition(&before_first_decision(&plain.contract)));
    }

    #[test]
    fn the_fallback_ends_the_episode_rather_than_padding_it() {
        for (kind, episode) in learner_episodes().expect("renders") {
            let fell_back = episode.groups.iter().any(|group| {
                group.facts.iter().any(|fact| {
                    matches!(fact, G0Fact::ActionExecuted { actuator }
                        if *actuator == Action::Fallback.index() as u16)
                })
            });
            if fell_back {
                assert_eq!(
                    episode.decisions(),
                    1,
                    "{} offered a decision after an absorbing fallback",
                    kind.label()
                );
            }
        }
    }

    #[test]
    fn the_taught_sequence_reaches_the_ceiling_on_every_case() {
        for case in card_cases() {
            let outcome = crate::run_policy(&case.contract, &ExactPostCalibration);
            assert_eq!(
                outcome.value,
                crate::value_bounds(&case.contract).0,
                "{} teacher fell below the ceiling",
                case.kind.label()
            );
        }
    }
}
