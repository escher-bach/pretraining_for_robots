//! Card 04 rendered onto the shared learner event boundary.
//!
//! The semantic family was audited before anything here existed, and rendering
//! it turned up a defect the audit could not have seen. The enumerated
//! "optimal first action" for the two unannounced-switch witnesses is
//! `retreat` — correct only for a solver that already knows the goal will
//! change. Nothing publishes that at step zero. Supervising the learner with it
//! would put unpublished information into the teacher, which `AGENTS.md` forbids
//! as squarely as putting it into an input.
//!
//! So the teacher here is [`PublicGoalConditioned`], which re-solves the
//! contract *as published so far*, and the cost of the difference is measured
//! rather than assumed: see [`unannounced_reveal_cost`].
//!
//! # What the learner is shown
//!
//! Every contract field becomes a public record, and nothing else does:
//!
//! | Contract field | Public record |
//! |---|---|
//! | `start` | the first observation, a selection on the cell's key |
//! | `goal` | a goal record, a selection on the cell's key |
//! | `no_go` | a condition, code [`CONDITION_PROHIBITED`], on that cell's key |
//! | `hazard` | a condition, code [`CONDITION_HAZARD`], its value carrying reset versus absorbing |
//! | `distractor_after` | a condition, code [`CONDITION_DISTRACTOR`], on an episode key, at that step |
//! | `switch` | a goal record plus a mode condition, published at start when announced and at the firing step otherwise |
//!
//! The mode condition's value carries *when* the second goal takes force, so a
//! learner reading an announcement knows both the new outcome and its timing.
//! That is what makes the announced negative solvable by planning once, which is
//! the contrast it exists to provide.

use pretraining_g0_contract::{cell_after, optimal_actions_from, Guard, Reveal};
use pretraining_g0_render::{
    boundary_check, rendering_report, step_fraction, BoundaryEvidence, BoundarySubtype, Content,
    G0Episode, G0Fact, G0Group, KeyNamespace, Port, PortSchema, RenderFault, RenderingReport,
};
use serde::{Deserialize, Serialize};

use crate::{
    card_cases, value_bounds, value_bounds_over, Action, Case, CaseKind, Contract, HazardKind,
    PublicPolicy, PublicView, SwitchMode, HORIZON, RING,
};

/// Entering the named configuration cell breaks the norm.
pub const CONDITION_PROHIBITED: u16 = 1;
/// The named cell is a viability boundary; `1.0` absorbs, `0.0` resets.
pub const CONDITION_HAZARD: u16 = 2;
/// A revealed event on an episode key that changes nothing.
pub const CONDITION_DISTRACTOR: u16 = 3;
/// The named cell becomes the sole outcome after the carried step fraction.
pub const CONDITION_GOAL_SUPERSEDES_AT: u16 = 4;
/// The named cell joins the first goal after the carried step fraction.
pub const CONDITION_GOAL_COMPOSES_AT: u16 = 5;

/// The episode key the distractor names.
pub const EPISODE_KEY_DISTRACTOR: u16 = 0;

impl Action {
    /// The actuator key this action drives.
    ///
    /// One actuator per primitive action, rather than one actuator carrying a
    /// discrete level. A level encoding would impose an ordinal geometry on a
    /// categorical action set and, worse, would make a body that supports fewer
    /// actions invisible in the schema — which is exactly the fact card 03
    /// needs to be public.
    pub const fn actuator_key(self) -> u16 {
        match self {
            Self::Retreat => 0,
            Self::Hold => 1,
            Self::Advance => 2,
        }
    }
}

/// The body and interface card 04 publishes: five configuration cells and three
/// always-supported actuators.
pub fn port_schema() -> PortSchema {
    PortSchema {
        observations: (0..RING as u16).map(Port::unit).collect(),
        actuators: Action::ALL
            .into_iter()
            .map(|action| Port::signed(action.actuator_key()))
            .collect(),
    }
}

impl Contract {
    /// The contract as published after `executed` actions.
    ///
    /// This is the only view the teacher and every public baseline may read. An
    /// unannounced switch is simply absent until its guard fires, so a policy
    /// built on this view cannot act on it early even by accident.
    pub fn published(&self, executed: usize) -> Self {
        let mut view = self.clone();
        if let Some(switch) = view.switch {
            let visible = match reveal_guard(&switch) {
                Guard::AtStart => true,
                guard => guard.fired(pretraining_g0_contract::GuardContext {
                    executed,
                    last_action: None,
                    cell: self.start,
                }),
            };
            if !visible {
                view.switch = None;
            }
        }
        view
    }
}

fn reveal_guard(switch: &crate::Switch) -> Guard {
    if switch.announced {
        Guard::AtStart
    } else {
        Guard::AfterStep(switch.after_step)
    }
}

/// The exact policy for the norm *as published*.
///
/// It differs from [`crate::GoalConditionedExact`] on exactly one thing: it does
/// not read an unannounced switch before that switch is published. That is the
/// difference between a public ceiling and a privileged one, and it is why this
/// policy and not the other supplies the teacher target.
pub struct PublicGoalConditioned;

impl PublicPolicy for PublicGoalConditioned {
    fn name(&self) -> &'static str {
        "public_goal_conditioned"
    }

    fn act(&self, view: &PublicView<'_>) -> Action {
        let mut probe = view.contract.published(view.executed);
        probe.start = view.cell;
        if let Some(mut switch) = probe.switch {
            switch.after_step = switch.after_step.saturating_sub(view.executed);
            probe.switch = Some(switch);
        }
        let remaining = HORIZON.saturating_sub(view.executed);
        let (_, optimal) = value_bounds_over(&probe, remaining);
        optimal
            .first()
            .and_then(|sequence| sequence.first().copied())
            .unwrap_or(Action::Hold)
    }
}

/// What a solver reading the unpublished switch gains over one that cannot.
///
/// This is the quantity the card's prose says is zero everywhere. It is zero on
/// eighteen of the twenty cases and equal to one move on the two unannounced
/// switch witnesses, which is the whole content of an unannounced reveal. The
/// name is deliberately narrow: it is the value of *this* published-norm policy
/// against the privileged ceiling, not a claim of optimality over every prior a
/// scheduler might hold about pending switches.
pub fn unannounced_reveal_cost(contract: &Contract) -> i32 {
    let privileged = value_bounds(contract).0;
    let published = crate::run_policy(contract, &PublicGoalConditioned).value;
    privileged - published
}

/// The first action a policy restricted to published information should take.
pub fn published_first_actions(contract: &Contract) -> Vec<Action> {
    let view = PublicView {
        contract,
        cell: contract.start,
        executed: 0,
    };
    let published = contract.published(0);
    let mut actions = crate::optimal_first_actions(&published);
    let chosen = PublicGoalConditioned.act(&view);
    if !actions.contains(&chosen) {
        actions.push(chosen);
    }
    actions.sort();
    actions.dedup();
    actions
}

fn goal_facts(contract: &Contract, at_start: bool) -> Vec<G0Fact> {
    let mut facts = Vec::new();
    if at_start {
        facts.push(G0Fact::Goal {
            key: contract.goal as u16,
            namespace: KeyNamespace::Observation,
            content: Content::Selection,
        });
        if let Some(cell) = contract.no_go {
            facts.push(condition(
                cell as u16,
                KeyNamespace::Observation,
                CONDITION_PROHIBITED,
                1.0,
            ));
        }
        if let Some((cell, kind)) = contract.hazard {
            facts.push(condition(
                cell as u16,
                KeyNamespace::Observation,
                CONDITION_HAZARD,
                match kind {
                    HazardKind::Absorbing => 1.0,
                    HazardKind::Reset => 0.0,
                },
            ));
        }
    }
    facts
}

fn condition(key: u16, namespace: KeyNamespace, code: u16, value: f64) -> G0Fact {
    G0Fact::Condition {
        key,
        namespace,
        code,
        value,
        lower: 0.0,
        upper: 1.0,
    }
}

fn switch_facts(switch: &crate::Switch, effective_after: usize) -> Vec<G0Fact> {
    vec![
        G0Fact::Goal {
            key: switch.goal as u16,
            namespace: KeyNamespace::Observation,
            content: Content::Selection,
        },
        condition(
            switch.goal as u16,
            KeyNamespace::Observation,
            match switch.mode {
                SwitchMode::Supersede => CONDITION_GOAL_SUPERSEDES_AT,
                SwitchMode::Compose => CONDITION_GOAL_COMPOSES_AT,
            },
            // A fraction of the action head, not of this card's horizon: three
            // does not divide a binary fraction, and the renderer refuses an
            // inexact payload rather than rounding one.
            step_fraction(effective_after),
        ),
    ]
}

/// The actions the published norm makes correct at one decision.
///
/// The whole set, not one member. Where the published norm is indifferent the
/// contract is indifferent, and singling one out would teach a tie-break the
/// world does not have.
///
/// It re-solves the *published* contract against the absolute step clock, so an
/// unannounced switch is invisible here until it fires and nothing has to be
/// rebased.
pub fn taught_actions(contract: &Contract, prefix: &[Action]) -> Vec<Action> {
    let published = contract.published(prefix.len());
    optimal_actions_from(&crate::NormSwap, &published, prefix)
}

/// Render one contract as a learner-visible episode taught by the public policy.
pub fn learner_episode(contract: &Contract) -> G0Episode {
    let mut groups = vec![G0Group::one(G0Fact::Boundary(BoundarySubtype::TaskReset))];

    let mut opening = goal_facts(contract, true);
    if let Some(switch) = contract.switch {
        if switch.announced {
            // An announced switch publishes the second outcome and when it takes
            // force, which is exactly what makes planning once sufficient.
            opening.extend(switch_facts(&switch, switch.after_step + 1));
        }
    }
    groups.push(G0Group::new(opening));

    let fragment = crate::NormSwap;
    let mut prefix: Vec<Action> = Vec::with_capacity(HORIZON);
    let mut cell = contract.start;
    groups.push(G0Group::one(observation(cell)));

    let mut absorbed = false;
    for executed in 0..HORIZON {
        if absorbed {
            break;
        }
        let correct = taught_actions(contract, &prefix);
        // The executed action is the lowest-indexed correct one. That is a
        // presentation choice and it is deliberately *not* the supervision:
        // every correct action is marked.
        let chosen = correct.first().copied().unwrap_or(Action::Hold);
        groups.push(G0Group::new(
            Action::ALL
                .into_iter()
                .map(|action| G0Fact::ActionQuery {
                    actuator: action.actuator_key(),
                    remaining: HORIZON - executed,
                    selected: correct.contains(&action),
                })
                .collect(),
        ));
        groups.push(G0Group::one(G0Fact::ActionExecuted {
            actuator: chosen.actuator_key(),
        }));
        prefix.push(chosen);
        let next = cell_after(&fragment, contract, &prefix);
        absorbed =
            matches!(contract.hazard, Some((hazard, HazardKind::Absorbing)) if next == hazard);
        cell = next;
        groups.push(G0Group::one(observation(cell)));

        if contract.distractor_after == Some(executed) {
            groups.push(G0Group::one(condition(
                EPISODE_KEY_DISTRACTOR,
                KeyNamespace::Episode,
                CONDITION_DISTRACTOR,
                1.0,
            )));
        }
        if let Some(switch) = contract.switch {
            if !switch.announced && switch.after_step == executed {
                // The reveal fires here and not one group earlier: it is in
                // force from the next decision onward, and publishing it before
                // the action that triggers it would delete the contrast.
                groups.push(G0Group::new(switch_facts(&switch, 0)));
            }
        }
    }

    groups.push(G0Group::one(G0Fact::Boundary(BoundarySubtype::EpisodeEnd)));
    G0Episode::new(port_schema(), HORIZON, groups)
}

fn observation(cell: usize) -> G0Fact {
    G0Fact::Observation {
        key: cell as u16,
        content: Content::Selection,
    }
}

/// Every case rendered, keeping its kind so a pilot can score arms apart.
pub fn learner_episodes() -> Vec<(CaseKind, G0Episode)> {
    card_cases()
        .into_iter()
        .map(|case| (case.kind, learner_episode(&case.contract)))
        .collect()
}

/// One case's boundary evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseRendering {
    pub kind: String,
    pub start: usize,
    pub goal: usize,
    pub evidence: BoundaryEvidence,
    /// The action the public teacher takes first.
    pub taught_first_action: String,
    /// The privileged enumeration's answer, which may differ.
    pub privileged_first_actions: Vec<String>,
    pub unannounced_reveal_cost: i32,
}

/// The rendering half of the card's audit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderAudit {
    pub report: RenderingReport,
    pub cases: Vec<CaseRendering>,
    /// True when the teacher never selects an action the published norm does
    /// not justify. This is the leakage check the rendering exists to pass.
    pub teacher_uses_only_published_information: bool,
    /// The cases where a privileged solver beats the published-norm policy.
    pub cases_with_a_reveal_cost: Vec<String>,
}

/// Build the rendering audit for the whole card.
pub fn render_audit() -> Result<RenderAudit, RenderFault> {
    let cases: Vec<Case> = card_cases();
    let episodes: Vec<G0Episode> = cases
        .iter()
        .map(|case| learner_episode(&case.contract))
        .collect();
    let report = rendering_report(&episodes)?;

    let mut rendered = Vec::with_capacity(cases.len());
    let mut leak_free = true;
    let mut costly = Vec::new();
    for case in &cases {
        let evidence = boundary_check(&learner_episode(&case.contract))?;
        let taught = PublicGoalConditioned.act(&PublicView {
            contract: &case.contract,
            cell: case.contract.start,
            executed: 0,
        });
        // The teacher's choice must be optimal for the norm as published. It is
        // allowed to differ from the privileged answer; it is not allowed to
        // anticipate a reveal.
        let published = case.contract.published(0);
        leak_free &= crate::optimal_first_actions(&published).contains(&taught);
        let cost = unannounced_reveal_cost(&case.contract);
        if cost != 0 {
            costly.push(case.kind.label().to_string());
        }
        rendered.push(CaseRendering {
            kind: case.kind.label().to_string(),
            start: case.contract.start,
            goal: case.contract.goal,
            evidence,
            taught_first_action: taught.name().to_string(),
            privileged_first_actions: crate::optimal_first_actions(&case.contract)
                .into_iter()
                .map(|action| action.name().to_string())
                .collect(),
            unannounced_reveal_cost: cost,
        });
    }
    costly.sort();
    costly.dedup();

    Ok(RenderAudit {
        report,
        cases: rendered,
        teacher_uses_only_published_information: leak_free,
        cases_with_a_reveal_cost: costly,
    })
}

/// The reveal this card composes, exposed so a cross-card check can compare it
/// with card 03's and card 05's rather than trusting three prose descriptions.
pub fn declared_reveals() -> Vec<(String, Reveal<usize>)> {
    card_cases()
        .into_iter()
        .filter_map(|case| {
            crate::reveal_of(&case.contract).map(|reveal| (case.kind.label().to_string(), reveal))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_teacher_never_anticipates_an_unpublished_reveal() {
        let audit = render_audit().expect("renders");
        assert!(audit.teacher_uses_only_published_information);
        assert!(audit.report.every_episode_round_trips);
    }

    #[test]
    fn the_unannounced_switch_costs_exactly_one_move_and_nothing_else_does() {
        for case in card_cases() {
            let cost = unannounced_reveal_cost(&case.contract);
            let expected = i32::from(case.kind == CaseKind::WitnessSwitch);
            assert_eq!(
                cost,
                expected,
                "{} expected a reveal cost of {expected}",
                case.kind.label()
            );
        }
    }

    #[test]
    fn the_taught_and_privileged_first_actions_differ_only_on_the_switch_witness() {
        for case in card_cases() {
            let taught = PublicGoalConditioned.act(&PublicView {
                contract: &case.contract,
                cell: case.contract.start,
                executed: 0,
            });
            let privileged = crate::optimal_first_actions(&case.contract);
            let agrees = privileged.contains(&taught);
            assert_eq!(
                agrees,
                case.kind != CaseKind::WitnessSwitch,
                "{} disagreement was unexpected",
                case.kind.label()
            );
        }
    }

    #[test]
    fn an_announced_switch_publishes_its_timing_and_an_unannounced_one_does_not() {
        let announced = card_cases()
            .into_iter()
            .find(|case| case.kind == CaseKind::NegativeSwitchAnnounced)
            .expect("the card has this negative");
        let unannounced = card_cases()
            .into_iter()
            .find(|case| case.kind == CaseKind::WitnessSwitch)
            .expect("the card has this witness");

        let opening = |episode: &G0Episode| episode.groups[1].facts.clone();
        let announced_opening = opening(&learner_episode(&announced.contract));
        let unannounced_opening = opening(&learner_episode(&unannounced.contract));
        assert!(announced_opening
            .iter()
            .any(|fact| matches!(fact, G0Fact::Condition { code, .. } if *code == CONDITION_GOAL_SUPERSEDES_AT)));
        assert!(!unannounced_opening
            .iter()
            .any(|fact| matches!(fact, G0Fact::Condition { code, .. } if *code == CONDITION_GOAL_SUPERSEDES_AT)));

        let later = learner_episode(&unannounced.contract);
        assert!(
            later.groups.iter().skip(2).any(|group| group
                .facts
                .iter()
                .any(|fact| matches!(fact, G0Fact::Condition { code, .. } if *code == CONDITION_GOAL_SUPERSEDES_AT))),
            "the reveal must still arrive, only later"
        );
    }

    #[test]
    fn every_contract_field_reaches_the_learner() {
        let inhibit = card_cases()
            .into_iter()
            .find(|case| case.kind == CaseKind::WitnessInhibit)
            .expect("the card has this witness");
        let facts: Vec<G0Fact> = learner_episode(&inhibit.contract)
            .groups
            .iter()
            .flat_map(|group| group.facts.clone())
            .collect();
        assert!(facts.iter().any(
            |fact| matches!(fact, G0Fact::Condition { code, key, .. }
                if *code == CONDITION_PROHIBITED && *key == inhibit.contract.no_go.unwrap() as u16)
        ));

        let viability = card_cases()
            .into_iter()
            .find(|case| case.kind == CaseKind::WitnessViability)
            .expect("the card has this witness");
        let facts: Vec<G0Fact> = learner_episode(&viability.contract)
            .groups
            .iter()
            .flat_map(|group| group.facts.clone())
            .collect();
        assert!(facts
            .iter()
            .any(|fact| matches!(fact, G0Fact::Condition { code, value, .. }
                if *code == CONDITION_HAZARD && *value == 1.0)));

        let maintain = card_cases()
            .into_iter()
            .find(|case| case.kind == CaseKind::WitnessMaintain)
            .expect("the card has this witness");
        let facts: Vec<G0Fact> = learner_episode(&maintain.contract)
            .groups
            .iter()
            .flat_map(|group| group.facts.clone())
            .collect();
        assert!(facts.iter().any(
            |fact| matches!(fact, G0Fact::Condition { code, .. } if *code == CONDITION_DISTRACTOR)
        ));
    }

    #[test]
    fn the_two_goal_conditioning_arms_render_to_different_episodes() {
        let pair: Vec<Case> = card_cases()
            .into_iter()
            .filter(|case| case.kind == CaseKind::WitnessGoalConditioning)
            .collect();
        let left = boundary_check(&learner_episode(&pair[0].contract)).expect("renders");
        let right = boundary_check(&learner_episode(&pair[1].contract)).expect("renders");
        assert_ne!(
            left.fingerprint, right.fingerprint,
            "a frozen-history goal swap must be visible at the boundary"
        );
        assert_eq!(
            left.records, right.records,
            "and must not change the record count, which would be a second difference"
        );
    }

    #[test]
    fn the_teacher_reaches_the_published_norm_ceiling_on_every_case() {
        for case in card_cases() {
            let outcome = crate::run(&case.contract, &taught_sequence(&case.contract));
            let published_ceiling =
                value_bounds(&case.contract).0 - unannounced_reveal_cost(&case.contract);
            assert_eq!(
                outcome.value,
                published_ceiling,
                "{} teacher fell below the published-norm ceiling",
                case.kind.label()
            );
        }
    }

    /// What the rendering actually teaches: the lowest-indexed correct action at
    /// each decision, chosen from the whole correct set.
    fn taught_sequence(contract: &Contract) -> Vec<Action> {
        let mut actions = Vec::with_capacity(HORIZON);
        for _ in 0..HORIZON {
            let correct = taught_actions(contract, &actions);
            actions.push(correct.first().copied().unwrap_or(Action::Hold));
        }
        actions
    }
}
