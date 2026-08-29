//! Card 05 rendered onto the shared learner event boundary.
//!
//! The teacher here cannot be the enumerated optimum. `value_bounds` reads the
//! gate, so its answer is "commit at once to the side the gate favours" — the
//! privileged answer, and teaching it would be teaching the gate. Card 04
//! established the rule the hard way; this card is the one where following it
//! changes the taught action on *every* witness episode rather than on two.
//!
//! So the teacher is [`pretraining_g0_contract::public_optimal_actions_at`]: the
//! actions optimal under the belief the learner actually holds, re-conditioned
//! after each action on what that action published. On the witness it opens with
//! `Probe` and then commits to the revealed side; on the gate-public control it
//! commits immediately; on the expensive-probe control it commits blind.
//!
//! # What is published, and why each piece has to be
//!
//! - **The family parameters.** Gate visibility, whether the two commits differ,
//!   and what a probe costs. Without them the witness and the equally-valuable
//!   control are the same episode up to the first decision, and no policy could
//!   probe in one and not the other.
//! - **The reveals.** The gate after a probe, the inconsequential bit after a
//!   peek. Each is a condition record on its own episode key.
//! - **Nothing else.** The gate does not appear before its reveal fires, and the
//!   audit's non-interference check is what establishes that rather than this
//!   comment.

use pretraining_g0_contract::public_optimal_actions_at;
use pretraining_g0_render::{
    boundary_check, legacy_tokens, rendering_report, step_fraction, BoundaryEvidence,
    BoundarySubtype, Content, G0Episode, G0Fact, G0Group, KeyNamespace, Port, PortSchema,
    RenderFault, RenderingReport,
};
use serde::{Deserialize, Serialize};

use crate::{
    card_cases, instance_ambiguity, Action, ActiveExperimentation, Bit, Case, CaseKind, Contract,
    GateCoupling, GateVisibility, BUDGET, GOAL_CELL, HORIZON, MISS_CELL, RING, START_CELL,
};

/// The gate's value, published by a reveal.
pub const CONDITION_GATE: u16 = 1;
/// The inconsequential bit's value, published by a peek.
pub const CONDITION_NOISE: u16 = 2;
/// Whether the two commits have different outcomes. A family fact.
pub const CONDITION_COMMITS_DIFFER: u16 = 3;
/// What one probe costs, as a fraction of the action head. A family fact.
pub const CONDITION_PROBE_COST: u16 = 4;
/// The remaining budget, republished after every action.
pub const CONDITION_BUDGET_REMAINING: u16 = 5;

pub const EPISODE_KEY_GATE: u16 = 0;
pub const EPISODE_KEY_NOISE: u16 = 1;
pub const EPISODE_KEY_OUTCOMES: u16 = 2;
pub const EPISODE_KEY_BUDGET: u16 = 3;

/// The body and interface card 05 publishes: three outcome cells and five
/// actuators.
///
/// The three non-committing actuators are declared identically. Nothing in the
/// schema says which of them buys information, which is the point: that is what
/// the learner has to work out.
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
            EPISODE_KEY_BUDGET,
            CONDITION_PROBE_COST,
            step_fraction(contract.probe_cost),
        ),
        condition(
            EPISODE_KEY_BUDGET,
            CONDITION_BUDGET_REMAINING,
            step_fraction(BUDGET),
        ),
    ];
    if contract.visibility == GateVisibility::Public {
        opening.push(condition(
            EPISODE_KEY_GATE,
            CONDITION_GATE,
            contract.gate.index() as f64,
        ));
    }
    groups.push(G0Group::new(opening));
    groups.push(G0Group::one(observation(START_CELL)));

    let mut prefix: Vec<Action> = Vec::with_capacity(HORIZON);
    let mut spent = 0usize;
    let mut cell = START_CELL;
    for executed in 0..HORIZON {
        // The whole correct set under the belief the learner holds now. Nothing
        // here reads the gate; `public_optimal_actions_at` conditions on what
        // the reveals have published and on nothing else.
        let correct = public_optimal_actions_at(&ActiveExperimentation, &set, &prefix, HORIZON);
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

        let mut consequences = vec![condition(
            EPISODE_KEY_BUDGET,
            CONDITION_BUDGET_REMAINING,
            step_fraction(BUDGET - spent),
        )];
        match chosen {
            Action::Probe => consequences.push(condition(
                EPISODE_KEY_GATE,
                CONDITION_GATE,
                contract.gate.index() as f64,
            )),
            Action::Peek => consequences.push(condition(
                EPISODE_KEY_NOISE,
                CONDITION_NOISE,
                contract.noise.index() as f64,
            )),
            Action::CommitLeft | Action::CommitRight => {
                cell = if contract.commit_succeeds(chosen) {
                    GOAL_CELL
                } else {
                    MISS_CELL
                };
            }
            Action::Sham => {}
        }
        groups.push(G0Group::new(consequences));
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
    pub noise: String,
    pub evidence: BoundaryEvidence,
    pub taught_opening: Vec<String>,
    pub probes: bool,
    /// Whether the gate value appears anywhere before a probe was executed.
    pub gate_published_before_it_was_bought: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderAudit {
    pub report: RenderingReport,
    pub cases: Vec<CaseRendering>,
    /// The teacher probes on the witness and on nothing else.
    pub the_teacher_probes_exactly_on_the_witness: bool,
    /// No episode publishes the gate before it was bought, except the control
    /// built to publish it.
    pub no_episode_leaks_the_gate: bool,
    /// Two episodes differing only in the gate are identical up to the reveal.
    pub the_gate_is_invisible_in_the_rendered_prefix: bool,
    /// Two episodes differing only in the inconsequential bit are identical
    /// unless something peeked.
    pub the_inconsequential_bit_is_invisible_unless_bought: bool,
}

/// The public rows of one episode.
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
    let mut probes_exactly = true;
    let mut no_leak = true;
    let mut gate_hidden = true;
    let mut noise_hidden = true;

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
        let probes = opening.iter().any(|name| name == Action::Probe.name());
        probes_exactly &= probes == (case.kind == CaseKind::WitnessProbeThenCommit);

        // Walk the rendered facts and check that no gate condition precedes the
        // executed probe, unless the family publishes the gate outright.
        let mut executed_probe = false;
        let mut leaked = false;
        for group in &episode.groups {
            for fact in &group.facts {
                match fact {
                    G0Fact::ActionExecuted { actuator }
                        if *actuator == Action::Probe.index() as u16 =>
                    {
                        executed_probe = true
                    }
                    G0Fact::Condition { code, .. } if *code == CONDITION_GATE => {
                        leaked |=
                            !executed_probe && case.contract.visibility == GateVisibility::Hidden;
                    }
                    _ => {}
                }
            }
        }
        no_leak &= !leaked;

        if case.contract.visibility == GateVisibility::Hidden {
            let twin = learner_episode(&case.contract.with_flipped_gate())?;
            let own = public_rows(episode)?;
            let other = public_rows(&twin)?;
            let shared = own.iter().zip(&other).take_while(|(a, b)| a == b).count();
            // Everything before the first executed action must agree: the gate
            // may only enter the stream through a reveal it paid for.
            let first_execution = own
                .iter()
                .position(|row| row.role == pretraining_world::Role::ActionExecuted)
                .expect("every episode executes something");
            gate_hidden &= shared > first_execution;
        }

        let noise_twin = learner_episode(&case.contract.with_flipped_noise())?;
        let peeked = episode.groups.iter().any(|group| {
            group.facts.iter().any(|fact| {
                matches!(fact, G0Fact::ActionExecuted { actuator }
                    if *actuator == Action::Peek.index() as u16)
            })
        });
        if !peeked {
            noise_hidden &= public_rows(episode)? == public_rows(&noise_twin)?;
        }

        rendered.push(CaseRendering {
            kind: case.kind.label().to_string(),
            gate: case.contract.gate.name().to_string(),
            noise: case.contract.noise.name().to_string(),
            evidence,
            taught_opening: opening,
            probes,
            gate_published_before_it_was_bought: leaked,
        });
    }

    Ok(RenderAudit {
        report,
        cases: rendered,
        the_teacher_probes_exactly_on_the_witness: probes_exactly,
        no_episode_leaks_the_gate: no_leak,
        the_gate_is_invisible_in_the_rendered_prefix: gate_hidden,
        the_inconsequential_bit_is_invisible_unless_bought: noise_hidden,
    })
}

/// The commit the revealed gate makes correct, for a reader of the rendering.
pub fn commit_for(gate: Bit) -> Action {
    match gate {
        Bit::Left => Action::CommitLeft,
        Bit::Right => Action::CommitRight,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rendering_carries_the_whole_claim() {
        let audit = render_audit().expect("renders");
        assert!(audit.report.every_episode_round_trips);
        assert!(audit.the_teacher_probes_exactly_on_the_witness);
        assert!(audit.no_episode_leaks_the_gate);
        assert!(audit.the_gate_is_invisible_in_the_rendered_prefix);
        assert!(audit.the_inconsequential_bit_is_invisible_unless_bought);
    }

    #[test]
    fn the_teacher_never_peeks() {
        // `Peek` is informative and worthless. A teacher that took it would be
        // teaching information seeking rather than useful information seeking,
        // which is the distinction the card exists to draw.
        for (kind, episode) in learner_episodes().expect("renders") {
            for set in episode.selected_actuators() {
                assert!(
                    !set.contains(&(Action::Peek.index() as u16)),
                    "{} taught a worthless probe",
                    kind.label()
                );
            }
        }
    }

    #[test]
    fn the_witness_opens_with_the_probe_and_then_commits_to_the_revealed_side() {
        for case in card_cases() {
            if case.kind != CaseKind::WitnessProbeThenCommit {
                continue;
            }
            let episode = learner_episode(&case.contract).expect("renders");
            let taught = episode.selected_actuators();
            assert_eq!(taught.len(), 2);
            assert_eq!(taught[0], vec![Action::Probe.index() as u16]);
            assert_eq!(
                taught[1],
                vec![commit_for(case.contract.gate).index() as u16],
                "the second decision follows the reveal"
            );
        }
    }

    #[test]
    fn the_gate_public_control_commits_at_once() {
        for case in card_cases() {
            if case.kind != CaseKind::NegativeGatePublic {
                continue;
            }
            let episode = learner_episode(&case.contract).expect("renders");
            let taught = episode.selected_actuators();
            assert_eq!(
                taught[0],
                vec![commit_for(case.contract.gate).index() as u16]
            );
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
            for required in [CONDITION_COMMITS_DIFFER, CONDITION_PROBE_COST] {
                assert!(
                    codes.contains(&required),
                    "{} withheld a family parameter",
                    kind.label()
                );
            }
        }
    }
}
