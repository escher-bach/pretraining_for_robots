//! Card 04 stage A rendered onto the shared learner event boundary.
//!
//! The composite found the rule this module obeys: its audited optimal first
//! action on the two unannounced-switch witnesses was privileged, and teaching
//! it would have taught the learner an unpublished goal. Stage A removes the
//! switch, so the enumerated optimum is public everywhere — but the renderer
//! *checks* that rather than inheriting the conclusion. An episode whose
//! ambiguity gap is not zero is refused with
//! [`RenderFault::TeacherWouldLeak`], which is the same refusal the composite
//! needed and the reason it is a shared fault rather than a card-local one.

use pretraining_g0_render::{
    boundary_check, rendering_report, BoundaryEvidence, BoundarySubtype, Content, G0Episode,
    G0Fact, G0Group, KeyNamespace, Port, PortSchema, RenderFault, RenderingReport,
};
use serde::{Deserialize, Serialize};

use crate::{
    card_cases, optimal_actions_from, vacuous_ambiguity_gap, Action, Case, CaseKind, Contract,
    HORIZON, RING,
};

/// Which arm of the family this episode belongs to. A family fact.
pub const CONDITION_REGIME: u16 = 1;
/// What the published goal symbol denotes. A family fact.
pub const CONDITION_DENOTATION: u16 = 2;

pub const EPISODE_KEY_FAMILY: u16 = 0;

/// Five configuration cells and three actuators.
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
        key: EPISODE_KEY_FAMILY,
        namespace: KeyNamespace::Episode,
        code,
        value,
        lower: 0.0,
        upper: 4.0,
    }
}

/// Render one contract as a learner-visible episode taught by the public policy.
pub fn learner_episode(contract: &Contract) -> Result<G0Episode, RenderFault> {
    // The check that makes the teacher below legitimate. It is cheap, exact,
    // and it fires before anything is emitted.
    if vacuous_ambiguity_gap(contract) != 0 {
        return Err(RenderFault::TeacherWouldLeak {
            detail: format!(
                "goal {} from cell {} has a privileged advantage; the enumerated optimum is not public",
                contract.goal, contract.start
            ),
        });
    }

    let mut groups = vec![
        G0Group::one(G0Fact::Boundary(BoundarySubtype::TaskReset)),
        G0Group::new(vec![
            G0Fact::Goal {
                key: contract.goal as u16,
                namespace: KeyNamespace::Observation,
                content: Content::Selection,
            },
            condition(CONDITION_REGIME, contract.regime as u8 as f64),
            condition(CONDITION_DENOTATION, contract.denotation as u8 as f64),
        ]),
        G0Group::one(observation(contract.start)),
    ];

    let mut prefix: Vec<Action> = Vec::with_capacity(HORIZON);
    let mut cell = contract.start;
    for executed in 0..HORIZON {
        let correct = optimal_actions_from(contract, &prefix);
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
                    remaining: HORIZON - executed,
                    selected: correct.contains(&candidate),
                })
                .collect(),
        ));
        let chosen = correct[0];
        groups.push(G0Group::one(G0Fact::ActionExecuted {
            actuator: chosen.index() as u16,
        }));
        prefix.push(chosen);
        cell = chosen.apply(cell);
        groups.push(G0Group::one(observation(cell)));
    }

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
    pub goal: usize,
    pub evidence: BoundaryEvidence,
    pub taught_opening: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderAudit {
    pub report: RenderingReport,
    pub cases: Vec<CaseRendering>,
    /// No episode's teacher reads anything the learner cannot see.
    pub every_teacher_is_public: bool,
    /// The two goals of a matched pair are taught different first actions.
    pub matched_pairs_are_taught_apart: bool,
}

pub fn render_audit() -> Result<RenderAudit, RenderFault> {
    let cases: Vec<Case> = card_cases();
    let episodes: Vec<G0Episode> = cases
        .iter()
        .map(|case| learner_episode(&case.contract))
        .collect::<Result<_, _>>()?;
    let report = rendering_report(&episodes)?;

    let mut rendered = Vec::with_capacity(cases.len());
    for (case, episode) in cases.iter().zip(&episodes) {
        rendered.push(CaseRendering {
            kind: case.kind.label().to_string(),
            start: case.contract.start,
            goal: case.contract.goal,
            evidence: boundary_check(episode)?,
            taught_opening: episode
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
                .unwrap_or_default(),
        });
    }

    // Every witness start carries exactly two goals. Their taught openings must
    // differ, or the family's whole claim is untaught.
    let mut pairs_differ = true;
    for start in 0..RING {
        let openings: Vec<Vec<String>> = rendered
            .iter()
            .zip(&cases)
            .filter(|(_, case)| {
                case.kind == CaseKind::WitnessGoalChangesAction && case.contract.start == start
            })
            .map(|(entry, _)| entry.taught_opening.clone())
            .collect();
        if openings.len() == 2 {
            pairs_differ &= openings[0] != openings[1];
        }
    }

    Ok(RenderAudit {
        report,
        cases: rendered,
        every_teacher_is_public: true,
        matched_pairs_are_taught_apart: pairs_differ,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Denotation, Regime};

    #[test]
    fn the_rendering_carries_the_whole_claim() {
        let audit = render_audit().expect("renders");
        assert!(audit.report.every_episode_round_trips);
        assert!(audit.matched_pairs_are_taught_apart);
        assert_eq!(audit.report.episodes, card_cases().len());
    }

    #[test]
    fn every_episode_publishes_its_goal_before_the_first_decision() {
        for (kind, episode) in learner_episodes().expect("renders") {
            let first_decision = episode
                .groups
                .iter()
                .position(|group| {
                    group
                        .facts
                        .iter()
                        .any(|fact| matches!(fact, G0Fact::ActionQuery { .. }))
                })
                .expect("every episode has a decision");
            let has_goal = episode.groups[..first_decision]
                .iter()
                .flat_map(|group| group.facts.iter())
                .any(|fact| matches!(fact, G0Fact::Goal { .. }));
            assert!(has_goal, "{} withheld its goal", kind.label());
        }
    }

    #[test]
    fn a_shifted_denotation_changes_what_is_taught_without_hiding_anything() {
        let direct = Contract::new(0, 1, Regime::GoalVaries);
        let shifted = direct.with_denotation(Denotation::Shifted);
        let taught = |contract: &Contract| {
            learner_episode(contract)
                .expect("renders")
                .selected_actuators()[0]
                .clone()
        };
        assert_ne!(taught(&direct), taught(&shifted));
        // And the change is visible: the denotation is a published family fact,
        // so this is a different world the learner is told about, not hidden
        // state it must guess.
        assert_eq!(vacuous_ambiguity_gap(&shifted), 0);
    }
}
