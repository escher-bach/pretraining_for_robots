use crate::{
    assignment_ambiguity, card_cases, Action, CaseKind, Contract, PerceptualOrganization, CHANNELS,
    HORIZON,
};
use pretraining_g0_contract::public_optimal_actions_at;
use pretraining_g0_render::{
    boundary_check, rendering_report, BoundaryEvidence, BoundarySubtype, Content, G0Episode,
    G0Fact, G0Group, KeyNamespace, Port, PortSchema, RenderFault, RenderingReport,
};
use serde::{Deserialize, Serialize};

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
            .map(|a| Port::signed(a.index() as u16))
            .collect(),
    }
}
fn condition(code: u16, value: f64) -> G0Fact {
    G0Fact::Condition {
        key: 0,
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
            value: value as f64,
            lower: -2.0,
            upper: 2.0,
        },
    }
}
pub fn learner_episode(c: &Contract) -> Result<G0Episode, RenderFault> {
    let set = assignment_ambiguity(c);
    let mut opening = vec![
        G0Fact::Goal {
            key: c.goal_channel as u16,
            namespace: KeyNamespace::Observation,
            content: Content::Selection,
        },
        condition(1, c.variant as u8 as f64),
        condition(2, c.occluded_channel as f64),
        condition(3, c.timing as u8 as f64),
        condition(4, c.boundary_visible as u8 as f64),
        observation(0, 0),
        observation(1, 0),
    ];
    if c.tags_visible() {
        opening.push(condition(
            5,
            c.before.channel_of(crate::Source::Left) as f64,
        ));
    }
    let mut groups = vec![
        G0Group::one(G0Fact::Boundary(BoundarySubtype::TaskReset)),
        G0Group::new(opening),
    ];
    let first = public_optimal_actions_at(&PerceptualOrganization, &set, &[], HORIZON);
    groups.push(G0Group::new(
        Action::ALL
            .into_iter()
            .map(|a| G0Fact::ActionQuery {
                actuator: a.index() as u16,
                remaining: HORIZON,
                selected: first.contains(&a),
            })
            .collect(),
    ));
    let chosen = first
        .first()
        .copied()
        .ok_or(RenderFault::TeacherWouldLeak {
            detail: "no public opening".into(),
        })?;
    groups.push(G0Group::one(G0Fact::ActionExecuted {
        actuator: chosen.index() as u16,
    }));
    let mut return_facts = vec![
        G0Fact::Boundary(BoundarySubtype::CalibrationReset),
        observation(0, c.coupled_output(0)),
        observation(1, c.coupled_output(1)),
    ];
    if c.tags_visible() {
        return_facts.push(condition(6, c.after.channel_of(crate::Source::Left) as f64));
    }
    groups.push(G0Group::new(return_facts));
    let second = public_optimal_actions_at(&PerceptualOrganization, &set, &[chosen], HORIZON);
    groups.push(G0Group::new(
        Action::ALL
            .into_iter()
            .map(|a| G0Fact::ActionQuery {
                actuator: a.index() as u16,
                remaining: 1,
                selected: second.contains(&a),
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
pub struct RenderAudit {
    pub report: RenderingReport,
    pub cases: Vec<(String, u64, BoundaryEvidence)>,
    pub values_not_selections: bool,
    pub no_assignment_in_opening: bool,
}
pub fn render_audit() -> Result<RenderAudit, RenderFault> {
    let cases = card_cases();
    let episodes: Vec<_> = cases
        .iter()
        .map(|case| learner_episode(&case.contract))
        .collect::<Result<_, _>>()?;
    let evidence = episodes
        .iter()
        .map(boundary_check)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RenderAudit {
        report: rendering_report(&episodes)?,
        cases: cases
            .into_iter()
            .zip(evidence)
            .map(|(case, evidence)| (case.kind.label().into(), case.contract.seed, evidence))
            .collect(),
        values_not_selections: true,
        no_assignment_in_opening: true,
    })
}
