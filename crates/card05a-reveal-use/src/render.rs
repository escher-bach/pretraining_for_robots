//! Card 05 stage A rendered onto the shared learner event boundary.
//!
//! The teacher is [`pretraining_g0_contract::public_optimal_actions_at`], as in
//! the composite and for the same reason: `value_bounds` reads the gate, so its
//! answer on the gate-hidden control is "commit to the side the gate favours",
//! which is the privileged answer and teaching it would be teaching the gate.
//!
//! What that produces here is worth stating, because it is the whole stage-A
//! contrast in one line. On the witness the published gate partitions the
//! belief before the first decision, so exactly one commit is taught. On the
//! gate-hidden control nothing partitions it and both commits are taught — the
//! boundary admits a *set* of correct actions, and a rendering that picked one
//! would be teaching a tie-break the world does not have.

use pretraining_g0_contract::public_optimal_actions_at;
use pretraining_g0_render::{
    boundary_check, legacy_tokens, rendering_report, step_fraction, BoundaryEvidence,
    BoundarySubtype, Content, G0Episode, G0Fact, G0Group, KeyNamespace, Port, PortSchema,
    RenderFault, RenderingReport,
};
use serde::{Deserialize, Serialize};

use crate::{
    card_cases, instance_ambiguity, Action, Case, CaseKind, Contract, GateCoupling, RevealMode,
    RevealUse, BUDGET, GOAL_CELL, HORIZON, MISS_CELL, RING, START_CELL,
};

/// The gate's value, published by the reveal.
pub const CONDITION_GATE: u16 = 1;
/// A published bit that is not the gate.
pub const CONDITION_DECOY: u16 = 2;
/// Whether the two commits have different outcomes. A family fact.
pub const CONDITION_COMMITS_DIFFER: u16 = 3;
/// Whether the episode-start reveal fires at all. A family fact.
pub const CONDITION_REVEAL_FIRES: u16 = 4;
/// The remaining budget, republished after every action.
pub const CONDITION_BUDGET_REMAINING: u16 = 5;

pub const EPISODE_KEY_GATE: u16 = 0;
pub const EPISODE_KEY_DECOY: u16 = 1;
pub const EPISODE_KEY_OUTCOMES: u16 = 2;
pub const EPISODE_KEY_BUDGET: u16 = 3;

/// Three outcome cells and three actuators.
///
/// `Sham` is declared exactly like the two commits. Nothing in the schema says
/// which actuator ends the episode.
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

fn condition(key: u16, code: u16, value: f64) -> G0Fact {
    G0Fact::Condition {
        key,
        namespace: KeyNamespace::Episode,
        code,
        value,
        lower: 0.0,
        upper: 1.0,
    }
}

/// Render one contract as a learner-visible episode taught by the public policy.
pub fn learner_episode(contract: &Contract) -> Result<G0Episode, RenderFault> {
    let set = instance_ambiguity(contract);
    let mut groups = vec![G0Group::one(G0Fact::Boundary(BoundarySubtype::TaskReset))];

    let mut opening = vec![
        G0Fact::Goal {
            key: GOAL_CELL as u16,
            namespace: KeyNamespace::Observation,
            content: Content::Selection,
        },
        condition(
            EPISODE_KEY_OUTCOMES,
            CONDITION_COMMITS_DIFFER,
            match contract.coupling {
                GateCoupling::Discriminating => 1.0,
                GateCoupling::Irrelevant => 0.0,
            },
        ),
        condition(
            EPISODE_KEY_GATE,
            CONDITION_REVEAL_FIRES,
            match contract.reveal_mode {
                RevealMode::Withholds => 0.0,
                RevealMode::PublishesGate | RevealMode::PublishesDecoy => 1.0,
            },
        ),
        condition(
            EPISODE_KEY_BUDGET,
            CONDITION_BUDGET_REMAINING,
            step_fraction(BUDGET),
        ),
    ];
    // The single path from the instance into public view. It is written as a
    // match on what the reveal published rather than on the field, so a later
    // edit that publishes an unguarded gate has to delete this structure first.
    match contract.reveal_mode {
        RevealMode::PublishesGate => opening.push(condition(
            EPISODE_KEY_GATE,
            CONDITION_GATE,
            contract.gate.index() as f64,
        )),
        RevealMode::PublishesDecoy => opening.push(condition(
            EPISODE_KEY_DECOY,
            CONDITION_DECOY,
            contract.decoy.index() as f64,
        )),
        RevealMode::Withholds => {}
    }
    groups.push(G0Group::new(opening));
    groups.push(G0Group::one(observation(START_CELL)));

    let mut prefix: Vec<Action> = Vec::with_capacity(HORIZON);
    let mut spent = 0usize;
    let mut cell = START_CELL;
    for executed in 0..HORIZON {
        let correct = public_optimal_actions_at(&RevealUse, &set, &prefix, HORIZON);
        if correct.is_empty() {
            break;
        }
        let chosen = correct[0];
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
        groups.push(G0Group::one(G0Fact::ActionExecuted {
            actuator: chosen.index() as u16,
        }));
        prefix.push(chosen);

        let cost = contract.cost_of(chosen);
        if spent + cost > BUDGET {
            break;
        }
        spent += cost;
        if chosen.is_commit() {
            cell = if contract.commit_succeeds(chosen) {
                GOAL_CELL
            } else {
                MISS_CELL
            };
        }
        groups.push(G0Group::new(vec![condition(
            EPISODE_KEY_BUDGET,
            CONDITION_BUDGET_REMAINING,
            step_fraction(BUDGET - spent),
        )]));
        groups.push(G0Group::one(observation(cell)));
        if chosen.is_commit() {
            break;
        }
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseRendering {
    pub kind: String,
    pub gate: String,
    pub decoy: String,
    pub evidence: BoundaryEvidence,
    pub taught_opening: Vec<String>,
    /// Whether the taught opening is a single action or an admitted set.
    pub taught_opening_is_a_set: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderAudit {
    pub report: RenderingReport,
    pub cases: Vec<CaseRendering>,
    /// The witness teaches exactly one commit, because the gate was published.
    pub the_witness_teaches_one_commit: bool,
    /// The gate-hidden control teaches both commits rather than a tie-break.
    pub the_hidden_control_teaches_both_commits: bool,
    /// Two episodes differing only in a withheld gate agree until a commit.
    pub a_withheld_gate_is_invisible_before_the_commit: bool,
    /// Two episodes differing only in the decoy are identical throughout.
    pub the_decoy_never_appears: bool,
}

fn public_rows(episode: &G0Episode) -> Result<Vec<pretraining_world::PublicToken>, RenderFault> {
    Ok(legacy_tokens(episode)?
        .into_iter()
        .map(|token| token.public)
        .collect())
}

pub fn render_audit() -> Result<RenderAudit, RenderFault> {
    let cases: Vec<Case> = card_cases();
    let episodes: Vec<G0Episode> = cases
        .iter()
        .map(|case| learner_episode(&case.contract))
        .collect::<Result<_, _>>()?;
    let report = rendering_report(&episodes)?;

    let mut rendered = Vec::with_capacity(cases.len());
    let mut witness_single = true;
    let mut hidden_both = true;
    let mut gate_invisible = true;
    let mut decoy_invisible = true;

    for (case, episode) in cases.iter().zip(&episodes) {
        let evidence = boundary_check(episode)?;
        let taught: Vec<Vec<u16>> = episode.selected_actuators();
        let opening: Vec<String> = taught
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
        let is_set = opening.len() > 1;

        match case.kind {
            CaseKind::WitnessRevealThenCommit => witness_single &= opening.len() == 1,
            CaseKind::NegativeGateHidden => hidden_both &= opening.len() == 2,
            CaseKind::NegativeCommitsEquallyValuable => {}
        }

        if case.contract.reveal_mode == RevealMode::Withholds {
            let twin = learner_episode(&case.contract.with_flipped_gate())?;
            let own = public_rows(episode)?;
            let other = public_rows(&twin)?;
            let shared = own.iter().zip(&other).take_while(|(a, b)| a == b).count();
            let first_execution = own
                .iter()
                .position(|row| row.role == pretraining_world::Role::ActionExecuted)
                .expect("every episode executes something");
            gate_invisible &= shared > first_execution;
        }

        let decoy_twin = learner_episode(&case.contract.with_flipped_decoy())?;
        decoy_invisible &= public_rows(episode)? == public_rows(&decoy_twin)?;

        rendered.push(CaseRendering {
            kind: case.kind.label().to_string(),
            gate: case.contract.gate.name().to_string(),
            decoy: case.contract.decoy.name().to_string(),
            evidence,
            taught_opening: opening,
            taught_opening_is_a_set: is_set,
        });
    }

    Ok(RenderAudit {
        report,
        cases: rendered,
        the_witness_teaches_one_commit: witness_single,
        the_hidden_control_teaches_both_commits: hidden_both,
        a_withheld_gate_is_invisible_before_the_commit: gate_invisible,
        the_decoy_never_appears: decoy_invisible,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{commit_for, Bit};

    #[test]
    fn the_rendering_carries_the_whole_claim() {
        let audit = render_audit().expect("renders");
        assert!(audit.report.every_episode_round_trips);
        assert!(audit.the_witness_teaches_one_commit);
        assert!(audit.the_hidden_control_teaches_both_commits);
        assert!(audit.a_withheld_gate_is_invisible_before_the_commit);
        assert!(audit.the_decoy_never_appears);
    }

    #[test]
    fn the_witness_commits_to_the_published_side_at_the_first_decision() {
        for case in card_cases() {
            if case.kind != CaseKind::WitnessRevealThenCommit {
                continue;
            }
            let episode = learner_episode(&case.contract).expect("renders");
            let taught = episode.selected_actuators();
            assert_eq!(
                taught[0],
                vec![commit_for(case.contract.gate).index() as u16],
                "the first decision follows the published gate"
            );
        }
    }

    #[test]
    fn the_decoy_is_carried_but_never_published_by_any_case() {
        // The decoy exists in every contract and appears in no episode, which is
        // what makes the uninformative-reveal orbit a transformation of this
        // family rather than a different one.
        let with_decoy =
            Contract::new(Bit::Left, Bit::Right).with_reveal(RevealMode::PublishesDecoy);
        let episode = learner_episode(&with_decoy).expect("renders");
        let published: Vec<u16> = episode
            .groups
            .iter()
            .flat_map(|group| group.facts.iter())
            .filter_map(|fact| match fact {
                G0Fact::Condition { code, .. } => Some(*code),
                _ => None,
            })
            .collect();
        assert!(published.contains(&CONDITION_DECOY));
        for (_, case_episode) in learner_episodes().expect("renders") {
            let codes: Vec<u16> = case_episode
                .groups
                .iter()
                .flat_map(|group| group.facts.iter())
                .filter_map(|fact| match fact {
                    G0Fact::Condition { code, .. } => Some(*code),
                    _ => None,
                })
                .collect();
            assert!(!codes.contains(&CONDITION_DECOY));
        }
    }

    #[test]
    fn the_family_parameters_are_published_before_the_first_decision() {
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
            let codes: Vec<u16> = episode.groups[..first_decision]
                .iter()
                .flat_map(|group| group.facts.iter())
                .filter_map(|fact| match fact {
                    G0Fact::Condition { code, .. } => Some(*code),
                    _ => None,
                })
                .collect();
            for required in [CONDITION_COMMITS_DIFFER, CONDITION_REVEAL_FIRES] {
                assert!(
                    codes.contains(&required),
                    "{} withheld a family parameter",
                    kind.label()
                );
            }
        }
    }
}
