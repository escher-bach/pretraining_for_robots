//! Card 02 rendered onto the shared learner event boundary.
//!
//! The rendering is where "the mode is public but must be carried" becomes a
//! property of the byte stream rather than a description. The latch is one
//! condition record, published once, in a group of its own, and never repeated;
//! the observations that follow are identical between the two modes right up to
//! the discriminating decision. Two episodes differing only in mode therefore
//! agree record-for-record until the last executed action.
//!
//! That is checked, not asserted: [`the_two_modes_render_identically_until_the_
//! discriminating_decision`] compares the rendered prefixes.
//!
//! The teacher is the mode-conditioned ceiling policy. Nothing privileged
//! reaches it — the mode it conditions on is exactly what the latch published,
//! which is why a decorrelated-latch contract is refused rather than rendered:
//! there the ceiling policy would be reading the true mode, and the latch said
//! something else.

use pretraining_g0_render::{
    boundary_check, legacy_tokens, rendering_report, step_fraction, BoundaryEvidence,
    BoundarySubtype, Content, G0Episode, G0Fact, G0Group, KeyNamespace, Port, PortSchema,
    RenderFault, RenderingReport,
};
use pretraining_world::{PublicToken, Role};
use serde::{Deserialize, Serialize};

use pretraining_g0_contract::{cell_after, optimal_actions_from};

use crate::{
    card_cases, ceiling_sequence, discriminating_actions, Action, Case, CaseKind, Contract, Mode,
    ModeVisibility, PredictiveState, DISCRIMINATING_STEP, HORIZON, RING,
};

/// The mode this episode is in, as most recently published.
pub const CONDITION_MODE_LATCH: u16 = 1;
/// A second latch that no effect depends on.
pub const CONDITION_DECOY_LATCH: u16 = 2;

/// The episode key the mode latch names.
pub const EPISODE_KEY_MODE: u16 = 0;
/// The episode key the decoy names. A different key, because a learner that had
/// to tell them apart by publication order alone would be solving a different
/// problem from the one the card states.
pub const EPISODE_KEY_DECOY: u16 = 1;

/// The body and interface card 02 publishes: seven cells and four actuators.
///
/// The two mode-sensitive commands are declared like any other, with no hint
/// that their effect is conditional. Publishing that would hand the learner the
/// structure it is meant to infer.
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

fn latch(key: u16, code: u16, mode: Mode) -> G0Fact {
    G0Fact::Condition {
        key,
        namespace: KeyNamespace::Episode,
        code,
        // Two modes, so the value is a bit. It is a categorical name occupying a
        // numeric slot, which is the recorded limitation of this layout rather
        // than a claim that the two modes are ordered.
        value: mode.index() as f64,
        lower: 0.0,
        upper: 1.0,
    }
}

/// Render one contract as a learner-visible episode taught by the ceiling
/// policy.
///
/// Refuses a decorrelated latch. There the published value no longer identifies
/// the mode, so the ceiling policy would be conditioning on something the
/// learner was never told — the same leak card 04's unannounced switch produced,
/// and the shared boundary has a fault for it.
pub fn learner_episode(contract: &Contract) -> Result<G0Episode, RenderFault> {
    if contract.latch_reports.is_some() {
        return Err(RenderFault::TeacherWouldLeak {
            detail: "the ceiling policy reads a mode that a decorrelated latch never published"
                .into(),
        });
    }

    let mut groups = vec![G0Group::one(G0Fact::Boundary(BoundarySubtype::TaskReset))];
    let mut opening = vec![G0Fact::Goal {
        key: contract.goal() as u16,
        namespace: KeyNamespace::Observation,
        content: Content::Selection,
    }];
    if let Some(decoy) = contract.decoy {
        opening.push(latch(EPISODE_KEY_DECOY, CONDITION_DECOY_LATCH, decoy));
    }
    groups.push(G0Group::new(opening));
    groups.push(G0Group::one(observation(contract.start)));

    let mut prefix: Vec<Action> = Vec::with_capacity(HORIZON);
    let mut cell;
    for executed in 0..HORIZON {
        if contract.mode_is_published(executed) {
            // Its own group, so the latch is a distinct public event rather than
            // something a learner could confuse with the observation beside it.
            groups.push(G0Group::one(latch(
                EPISODE_KEY_MODE,
                CONDITION_MODE_LATCH,
                contract.reported_mode(),
            )));
        }
        // The whole correct set. On the inert control both commands advance, so
        // both are marked — which is what stops that control from rendering
        // byte-identically to the witness whose tie-break happened to agree.
        let correct = optimal_actions_from(&PredictiveState, contract, &prefix);
        let action = correct.first().copied().unwrap_or(Action::Hold);
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
            actuator: action.index() as u16,
        }));
        prefix.push(action);
        cell = cell_after(&PredictiveState, contract, &prefix);
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

/// The learner-visible rows of one episode, with the mode latch removed.
///
/// Two things about this are deliberate.
///
/// It is the **public rows**, not the transcript. A transcript group carries the
/// teacher's `selected` flag, and comparing transcripts reports a divergence at
/// the discriminating decision that no learner can see — the flag becomes a
/// supervision entry and never a payload slot. The aliasing claim is about what
/// the learner is shown, so it has to be asked of what the learner is shown.
///
/// It **removes the latch**. The latch record is exactly where the two modes
/// differ, and that is the point of the card; requiring the raw streams to agree
/// would be requiring the latch not to work.
fn public_rows_without_the_latch(episode: &G0Episode) -> Result<Vec<PublicToken>, RenderFault> {
    Ok(legacy_tokens(episode)?
        .into_iter()
        .map(|token| token.public)
        .filter(|public| !(public.role == Role::Condition && public.key == EPISODE_KEY_MODE))
        .collect())
}

/// How many rows two latch-free renderings share before they first differ.
fn shared_prefix(left: &[PublicToken], right: &[PublicToken]) -> usize {
    left.iter().zip(right).take_while(|(a, b)| a == b).count()
}

/// The index of the last query row of the discriminating decision.
///
/// Decisions are found as maximal runs of query rows rather than by decoding the
/// remaining-step payload, so the check does not depend on the slot encoding it
/// is meant to be independent of.
fn discriminating_decision_end(rows: &[PublicToken]) -> usize {
    let mut decision = 0usize;
    let mut index = 0usize;
    while index < rows.len() {
        if rows[index].role != Role::ActionQuery {
            index += 1;
            continue;
        }
        let start = index;
        while index < rows.len() && rows[index].role == Role::ActionQuery {
            index += 1;
        }
        if decision == DISCRIMINATING_STEP {
            return index - 1;
        }
        decision += 1;
        let _ = start;
    }
    panic!("every episode offers every decision");
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseRendering {
    pub kind: String,
    pub mode: String,
    pub evidence: BoundaryEvidence,
    pub taught_sequence: Vec<String>,
    pub taught_discriminating_action: String,
    pub correct_discriminating_actions: Vec<String>,
    /// The latch-free public row at which this episode first differs from its
    /// mode twin.
    pub diverges_from_its_mode_twin_at_row: usize,
    /// The last public row of the discriminating decision's query group.
    pub discriminating_decision_ends_at_row: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderAudit {
    pub report: RenderingReport,
    pub cases: Vec<CaseRendering>,
    /// The teacher's discriminating command is correct on every case.
    pub teacher_is_correct_everywhere: bool,
    /// With the latch records set aside, two episodes differing only in mode are
    /// identical until after the discriminating command is executed. This is the
    /// aliasing claim, stated about the rendered stream rather than about the
    /// semantics. The latch itself of course differs; that is what it is for.
    pub the_modes_are_indistinguishable_until_the_discriminating_action: bool,
    /// A decorrelated latch cannot be rendered at all.
    pub a_decorrelated_latch_is_refused: bool,
    /// The latch appears exactly once in a latched episode and at every decision
    /// in a republishing one.
    pub latch_counts_match_the_visibility: bool,
}

pub fn render_audit() -> Result<RenderAudit, RenderFault> {
    let cases: Vec<Case> = card_cases();
    let episodes: Vec<G0Episode> = cases
        .iter()
        .map(|case| learner_episode(&case.contract))
        .collect::<Result<_, _>>()?;
    let report = rendering_report(&episodes)?;

    let mut rendered = Vec::with_capacity(cases.len());
    let mut teacher_correct = true;
    let mut aliased = true;
    let mut counts_match = true;
    for (case, episode) in cases.iter().zip(&episodes) {
        let evidence = boundary_check(episode)?;
        let taught: Vec<Action> = episode
            .selected_actuators()
            .iter()
            .map(|set| crate::action_from_index(set[0] as usize).expect("a known actuator"))
            .collect();
        let correct = discriminating_actions(&case.contract);
        teacher_correct &= correct.contains(&taught[DISCRIMINATING_STEP]);

        let twin = learner_episode(&case.contract.with_flipped_mode())?;
        let own = public_rows_without_the_latch(episode)?;
        let other = public_rows_without_the_latch(&twin)?;
        let prefix = shared_prefix(&own, &other);
        let decision = discriminating_decision_end(&own);
        // The two modes must still agree at the group that *offers* the
        // discriminating decision. They may differ at the next group, which is
        // the executed action, and that is the earliest a difference is allowed.
        if case.contract.coupling == crate::ModeCoupling::Discriminating
            && case.contract.visibility == ModeVisibility::Latched
        {
            aliased &= prefix > decision;
        }

        let latch_records = episode
            .groups
            .iter()
            .flat_map(|group| group.facts.iter())
            .filter(|fact| {
                matches!(fact, G0Fact::Condition { code, .. } if *code == CONDITION_MODE_LATCH)
            })
            .count();
        counts_match &= latch_records
            == match case.contract.visibility {
                ModeVisibility::Latched => 1,
                ModeVisibility::Always => HORIZON,
            };

        rendered.push(CaseRendering {
            kind: case.kind.label().to_string(),
            mode: case.contract.mode.name().to_string(),
            evidence,
            taught_discriminating_action: taught[DISCRIMINATING_STEP].name().to_string(),
            taught_sequence: taught
                .iter()
                .map(|action| action.name().to_string())
                .collect(),
            correct_discriminating_actions: correct
                .into_iter()
                .map(|action| action.name().to_string())
                .collect(),
            diverges_from_its_mode_twin_at_row: prefix,
            discriminating_decision_ends_at_row: decision,
        });
    }

    let decorrelated = Contract {
        latch_reports: Some(Mode::Forward),
        ..card_cases()[0].contract
    };
    Ok(RenderAudit {
        report,
        cases: rendered,
        teacher_is_correct_everywhere: teacher_correct,
        the_modes_are_indistinguishable_until_the_discriminating_action: aliased,
        a_decorrelated_latch_is_refused: matches!(
            learner_episode(&decorrelated),
            Err(RenderFault::TeacherWouldLeak { .. })
        ),
        latch_counts_match_the_visibility: counts_match,
    })
}

/// The fraction of the action head one decision of this card occupies, exposed
/// so a mixture accountant can compare families on one scale.
pub fn decision_fraction() -> f64 {
    step_fraction(HORIZON)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rendering_carries_the_whole_claim() {
        let audit = render_audit().expect("renders");
        assert!(audit.report.every_episode_round_trips);
        assert!(audit.teacher_is_correct_everywhere);
        assert!(audit.the_modes_are_indistinguishable_until_the_discriminating_action);
        assert!(audit.a_decorrelated_latch_is_refused);
        assert!(audit.latch_counts_match_the_visibility);
    }

    #[test]
    fn the_two_modes_render_identically_until_the_discriminating_decision() {
        let witness = card_cases()
            .into_iter()
            .find(|case| case.kind == CaseKind::WitnessLatchedMode)
            .expect("the card has this witness");
        let forward = learner_episode(&witness.contract).expect("renders");
        let reversed = learner_episode(&witness.contract.with_flipped_mode()).expect("renders");

        // The latch row itself differs, so the comparison is over everything
        // *except* it: that is the point of the card — one record separates two
        // otherwise identical streams, and it arrives long before it is needed.
        let left = public_rows_without_the_latch(&forward).expect("renders");
        let right = public_rows_without_the_latch(&reversed).expect("renders");
        let decision = discriminating_decision_end(&left);
        assert_eq!(
            left[..=decision],
            right[..=decision],
            "the modes must be indistinguishable once the latch is removed"
        );
        assert_ne!(
            left[decision + 1..],
            right[decision + 1..],
            "and must separate immediately after the discriminating command"
        );
        assert_ne!(forward, reversed, "and the latch itself must differ");
    }

    #[test]
    fn the_republishing_control_says_the_mode_at_every_decision() {
        let control = card_cases()
            .into_iter()
            .find(|case| case.kind == CaseKind::NegativeFullyObservable)
            .expect("the card has this control");
        let episode = learner_episode(&control.contract).expect("renders");
        let latches = episode
            .groups
            .iter()
            .flat_map(|group| group.facts.iter())
            .filter(|fact| {
                matches!(fact, G0Fact::Condition { code, .. } if *code == CONDITION_MODE_LATCH)
            })
            .count();
        assert_eq!(latches, HORIZON);
    }

    #[test]
    fn the_decoy_travels_on_its_own_key() {
        let case = card_cases()
            .into_iter()
            .find(|case| case.kind == CaseKind::NegativeMemoryCost)
            .expect("the card has this control");
        let episode = learner_episode(&case.contract).expect("renders");
        let keys: Vec<u16> = episode
            .groups
            .iter()
            .flat_map(|group| group.facts.iter())
            .filter_map(|fact| match fact {
                G0Fact::Condition { key, .. } => Some(*key),
                _ => None,
            })
            .collect();
        assert!(keys.contains(&EPISODE_KEY_MODE));
        assert!(keys.contains(&EPISODE_KEY_DECOY));
        assert_ne!(EPISODE_KEY_MODE, EPISODE_KEY_DECOY);
    }

    #[test]
    fn flipping_the_decoy_moves_no_taught_action() {
        for case in card_cases() {
            let Some(decoy) = case.contract.decoy else {
                continue;
            };
            let flipped = Contract {
                decoy: Some(decoy.flipped()),
                ..case.contract
            };
            assert_eq!(
                ceiling_sequence(&case.contract),
                ceiling_sequence(&flipped),
                "the second latch must not move the teacher"
            );
            let base = learner_episode(&case.contract).expect("renders");
            let moved = learner_episode(&flipped).expect("renders");
            assert_ne!(base, moved, "but it must still be published");
        }
    }
}
