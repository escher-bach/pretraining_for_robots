//! Card 06 stage A — visible reassignment, the basic relation card 06's
//! decomposition certificate already names.
//!
//! The certificate is explicit about the target, so this crate implements it
//! rather than choosing it: *binding the history-named source across one public
//! channel-change boundary while both source values remain continuously
//! visible*. Absence, latent evolution while absent, and matched-marginal
//! occlusion noise are gone. Source exchangeability, the hidden assignment, the
//! history-named goal, action effects, the source and channel permutations,
//! non-interference, and the channel-locked and identity-tag contrasts stay.
//!
//! # What the mechanism is
//!
//! Both channels publish their source's value at every step, and both sources
//! read `0` until something distinguishes them. A pulse perturbs the source
//! currently in the pulsed channel; the assignment boundary then fires, and the
//! perturbation appears wherever that source now is. The goal names a source by
//! that interaction — "the source that was in channel `g`" — never by index, so
//! a policy that drives channel `g` again is following channel identity and a
//! policy that follows the perturbation is following the source.
//!
//! # The one thing this stage gives away, and why that is the point
//!
//! The composite's public/known-assignment gap is large: with the source
//! occluded, no public history locates it. Here the gap is zero, because a
//! continuously visible perturbation locates the source exactly. That is not a
//! weakened world — it is the removal the certificate asks for, stated as a
//! number. Memory through absence is precisely the thing whose removal closes
//! the gap, and the audit's `values_made_invisible_across_the_boundary`
//! information orbit is what shows that restoring the occlusion reopens it.
//!
//! # An honest divergence from the composite
//!
//! The composite treats hiding the assignment-change boundary as
//! meaning-changing. Here it is *preserving*: with the values continuously
//! visible, the marker carries nothing the values do not already carry. The
//! audit declares and checks it as preserving and records the divergence, since
//! inheriting the composite's verdict would have asserted an invariance this
//! family does not have.

mod audit;
mod render;
pub use audit::*;
pub use render::*;

use std::collections::BTreeMap;

use pretraining_g0_contract::{
    agent_equivalence, identify, AmbiguitySet, ContractHasher, Coupling, CouplingRule, Fragment,
    KernelUse, PubliclyObservable,
};
use serde::{Deserialize, Serialize};

pub const HORIZON: usize = 2;
pub const CHANNELS: usize = 2;
pub const GOAL_REWARD: i32 = 100;
pub const STEP_COST: i32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Source {
    Left,
    Right,
}

impl Source {
    pub const ALL: [Self; 2] = [Self::Left, Self::Right];

    pub const fn other(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
        }
    }
}

/// Which source sits in which channel. Hidden.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Assignment {
    Straight,
    Swapped,
}

impl Assignment {
    pub const ALL: [Self; 2] = [Self::Straight, Self::Swapped];

    pub const fn source_at(self, channel: usize) -> Source {
        match (self, channel) {
            (Self::Straight, 0) | (Self::Swapped, 1) => Source::Left,
            _ => Source::Right,
        }
    }

    pub const fn channel_of(self, source: Source) -> usize {
        match (self, source) {
            (Self::Straight, Source::Left) | (Self::Swapped, Source::Right) => 0,
            _ => 1,
        }
    }

    pub const fn flipped(self) -> Self {
        match self {
            Self::Straight => Self::Swapped,
            Self::Swapped => Self::Straight,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Straight => "straight",
            Self::Swapped => "swapped",
        }
    }
}

/// The sign of the common value scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Sign {
    Negative,
    Positive,
}

impl Sign {
    pub const ALL: [Self; 2] = [Self::Negative, Self::Positive];

    pub const fn value(self) -> i32 {
        match self {
            Self::Negative => -1,
            Self::Positive => 1,
        }
    }

    pub const fn flipped(self) -> Self {
        match self {
            Self::Negative => Self::Positive,
            Self::Positive => Self::Negative,
        }
    }
}

/// Which source an otherwise identical goal names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GoalNaming {
    /// The source that was in the goal channel before the boundary.
    HistoryNamed,
    /// The other one. The declared meaning-changing transformation.
    Complement,
}

/// Whether the channels keep publishing values after the boundary.
///
/// `Occluded` is not a case kind. It is the transformation the information
/// orbit applies to walk one step back toward the composite, and it exists so
/// that "continuous visibility is what stage A rests on" is a measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Visibility {
    Continuous,
    Occluded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Variant {
    /// The assignment may change across the public boundary.
    Witness,
    /// The assignment cannot change, so channel identity is enough.
    ChannelLocked,
    /// Every source carries a permanent public tag, so no history is needed.
    IdentityTag,
}

impl Variant {
    pub const ALL: [Self; 3] = [Self::Witness, Self::ChannelLocked, Self::IdentityTag];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Witness => "witness",
            Self::ChannelLocked => "channel_locked",
            Self::IdentityTag => "identity_tag",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Action {
    PulseZero,
    PulseOne,
    DriveZero,
    DriveOne,
}

impl Action {
    pub const ALL: [Self; 4] = [
        Self::PulseZero,
        Self::PulseOne,
        Self::DriveZero,
        Self::DriveOne,
    ];

    pub const fn index(self) -> usize {
        match self {
            Self::PulseZero => 0,
            Self::PulseOne => 1,
            Self::DriveZero => 2,
            Self::DriveOne => 3,
        }
    }

    pub const fn pulse_channel(self) -> Option<usize> {
        match self {
            Self::PulseZero => Some(0),
            Self::PulseOne => Some(1),
            _ => None,
        }
    }

    pub const fn drive_channel(self) -> Option<usize> {
        match self {
            Self::DriveZero => Some(0),
            Self::DriveOne => Some(1),
            _ => None,
        }
    }

    pub const fn flip_channel(self) -> Self {
        match self {
            Self::PulseZero => Self::PulseOne,
            Self::PulseOne => Self::PulseZero,
            Self::DriveZero => Self::DriveOne,
            Self::DriveOne => Self::DriveZero,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::PulseZero => "pulse_zero",
            Self::PulseOne => "pulse_one",
            Self::DriveZero => "drive_zero",
            Self::DriveOne => "drive_one",
        }
    }

    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.into_iter().find(|action| action.index() == index)
    }
}

pub fn pulse_for(channel: usize) -> Action {
    if channel == 0 {
        Action::PulseZero
    } else {
        Action::PulseOne
    }
}

pub fn drive_for(channel: usize) -> Action {
    if channel == 0 {
        Action::DriveZero
    } else {
        Action::DriveOne
    }
}

/// Public family parameters plus the private assignment realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contract {
    /// Hidden: the assignment before the public boundary.
    pub before: Assignment,
    /// Hidden: the assignment after it.
    pub after: Assignment,
    pub variant: Variant,
    /// Published: the channel whose occupant the goal names.
    pub goal_channel: usize,
    pub naming: GoalNaming,
    pub scale: Sign,
    pub visibility: Visibility,
    pub boundary_visible: bool,
    pub seed: u64,
}

impl Contract {
    pub fn new(
        before: Assignment,
        after: Assignment,
        variant: Variant,
        goal_channel: usize,
        seed: u64,
    ) -> Self {
        Self {
            before,
            after,
            variant,
            goal_channel,
            naming: GoalNaming::HistoryNamed,
            scale: Sign::Positive,
            visibility: Visibility::Continuous,
            boundary_visible: true,
            seed,
        }
    }

    /// Each channel is one shared variable with exactly one writer.
    ///
    /// `Conflict` rather than the composite's `Override` is the executable form
    /// of "the occlusion noise is gone": a second writer here is a contract
    /// error rather than a resolution rule, and stage B restores `Override` when
    /// it restores the noise.
    pub fn coupling(&self) -> Coupling {
        Coupling::new(0, CouplingRule::Conflict)
    }

    pub fn tags_visible(&self) -> bool {
        self.variant == Variant::IdentityTag
    }

    /// The source the goal names, by the interaction rather than by index.
    pub fn named_source(&self) -> Source {
        let occupant = self.before.source_at(self.goal_channel);
        match self.naming {
            GoalNaming::HistoryNamed => occupant,
            GoalNaming::Complement => occupant.other(),
        }
    }

    /// The source a pulse of `channel` perturbs.
    pub fn perturbed_by(&self, channel: usize) -> Source {
        self.before.source_at(channel)
    }

    /// One channel's published value after the boundary, resolved through the
    /// declared coupling.
    ///
    /// A channel has exactly one writer: the source assigned to it. Under
    /// `Occluded` there is no writer at all and the resolution fails, which is
    /// what makes the occlusion transformation a change to the composition
    /// rather than a change to a rendering.
    pub fn channel_value(&self, channel: usize, pulsed: usize) -> Option<i32> {
        if self.visibility == Visibility::Occluded {
            return None;
        }
        let writer = if self.after.source_at(channel) == self.perturbed_by(pulsed) {
            self.scale.value()
        } else {
            0
        };
        self.coupling()
            .resolve(&[f64::from(writer)])
            .ok()
            .map(|value| value as i32)
    }

    /// Whether driving `channel` reaches the named source.
    pub fn drive_succeeds(&self, channel: usize) -> bool {
        self.after.source_at(channel) == self.named_source()
    }

    pub fn relabel_channels(&self) -> Self {
        Self {
            before: self.before.flipped(),
            after: self.after.flipped(),
            goal_channel: 1 - self.goal_channel,
            ..*self
        }
    }

    pub fn relabel_sources(&self) -> Self {
        Self {
            before: self.before.flipped(),
            after: self.after.flipped(),
            ..*self
        }
    }

    pub fn flip_scale(&self) -> Self {
        Self {
            scale: self.scale.flipped(),
            ..*self
        }
    }

    pub fn flip_after(&self) -> Self {
        Self {
            after: self.after.flipped(),
            ..*self
        }
    }

    pub fn change_named_source(&self) -> Self {
        Self {
            naming: match self.naming {
                GoalNaming::HistoryNamed => GoalNaming::Complement,
                GoalNaming::Complement => GoalNaming::HistoryNamed,
            },
            ..*self
        }
    }

    pub fn hide_boundary(&self) -> Self {
        Self {
            boundary_visible: false,
            ..*self
        }
    }

    pub fn occlude(&self) -> Self {
        Self {
            visibility: Visibility::Occluded,
            ..*self
        }
    }

    /// Whether the assignment actually moved. Never published.
    pub fn reassigned(&self) -> bool {
        self.before != self.after
    }
}

/// Score a complete action sequence.
///
/// An episode is only informative once something has been pulsed, so a
/// sequence that opens with a drive scores nothing. Which channel is pulsed is
/// deliberately unconstrained: with two exchangeable sources either pulse
/// locates both, and the audit reports that indifference rather than hiding it
/// behind a required opening.
pub fn run(contract: &Contract, actions: &[Action]) -> i32 {
    if actions.len() < HORIZON || actions[0].pulse_channel().is_none() {
        return 0;
    }
    match actions[1].drive_channel() {
        Some(channel) if contract.drive_succeeds(channel) => {
            GOAL_REWARD - HORIZON as i32 * STEP_COST
        }
        _ => 0,
    }
}

pub struct VisibleReassignment;

impl Fragment for VisibleReassignment {
    type Action = Action;
    type Contract = Contract;

    fn actions(&self) -> Vec<Action> {
        Action::ALL.to_vec()
    }

    fn horizon(&self) -> usize {
        HORIZON
    }

    fn start(&self, _: &Contract) -> usize {
        0
    }

    fn step(&self, contract: &Contract, cell: usize, executed: usize, action: Action) -> usize {
        if executed == 1
            && action
                .drive_channel()
                .is_some_and(|channel| contract.drive_succeeds(channel))
        {
            1
        } else {
            cell
        }
    }

    fn value(&self, contract: &Contract, _: &[usize], actions: &[Action]) -> i32 {
        run(contract, actions)
    }
}

impl PubliclyObservable for VisibleReassignment {
    fn public_trace(&self, contract: &Contract, actions: &[Action]) -> Vec<i64> {
        let mut trace = vec![
            contract.variant as i64,
            contract.goal_channel as i64,
            contract.naming as i64,
            contract.scale as i64,
            contract.visibility as i64,
            contract.boundary_visible as i64,
        ];
        if contract.tags_visible() {
            trace.push(20 + contract.before.channel_of(Source::Left) as i64);
        }
        if actions.is_empty() {
            return trace;
        }
        trace.push(actions[0].index() as i64);
        let Some(pulsed) = actions[0].pulse_channel() else {
            // Nothing was interrogated, so nothing distinguishes the sources and
            // the boundary publishes no information about them.
            return trace;
        };
        if contract.boundary_visible {
            trace.push(30);
        }
        for channel in 0..CHANNELS {
            match contract.channel_value(channel, pulsed) {
                Some(value) => trace.push(value as i64),
                None => trace.push(-9),
            }
        }
        if contract.tags_visible() {
            trace.push(40 + contract.after.channel_of(Source::Left) as i64);
        }
        if actions.len() > 1 {
            trace.push(actions[1].index() as i64);
            trace.push(i64::from(run(contract, actions) > 0));
        }
        trace
    }
}

pub fn all_sequences() -> Vec<Vec<Action>> {
    pretraining_g0_contract::sequences_of_length(&Action::ALL, HORIZON)
}

/// The hidden assignment pairs a learner cannot separate at episode start.
///
/// The seed is generator metadata and is held fixed across candidates: coupling
/// a replay seed to an assignment would itself be a leakage path, and giving the
/// realized pair extra prior mass because its seed differs would be an error the
/// composite documents.
pub fn assignment_ambiguity(contract: &Contract) -> AmbiguitySet<Contract> {
    let mut candidates = vec![*contract];
    for before in Assignment::ALL {
        for after in Assignment::ALL {
            if contract.variant == Variant::ChannelLocked && before != after {
                continue;
            }
            if before == contract.before && after == contract.after {
                continue;
            }
            candidates.push(Contract {
                before,
                after,
                ..*contract
            });
        }
    }
    AmbiguitySet::uniform(candidates)
}

/// Surviving ambiguity modulo agent equivalence.
///
/// Raw residual ambiguity after a pulse is two: an unmoved straight assignment
/// and an unmoved swapped one publish the same values. They are the same world
/// as far as any admissible intervention is concerned, and the quotient is what
/// says so.
pub fn relevant_ambiguity(contract: &Contract, prefix: &[Action]) -> usize {
    let set = assignment_ambiguity(contract);
    let survivors = identify(&VisibleReassignment, &set, prefix);
    let mut representatives: Vec<usize> = Vec::new();
    'candidate: for index in survivors {
        for representative in &representatives {
            if agent_equivalence(
                &VisibleReassignment,
                &set.candidates[index],
                &set.candidates[*representative],
                HORIZON,
                |action| action.name().into(),
            )
            .equivalent
            {
                continue 'candidate;
            }
        }
        representatives.push(index);
    }
    representatives.len()
}

pub fn raw_ambiguity(contract: &Contract, prefix: &[Action]) -> usize {
    pretraining_g0_contract::identification_diameter(
        &VisibleReassignment,
        &assignment_ambiguity(contract),
        prefix,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CaseKind {
    Witness,
    ChannelLocked,
    IdentityTag,
}

impl CaseKind {
    pub const ALL: [Self; 3] = [Self::Witness, Self::ChannelLocked, Self::IdentityTag];

    pub const NEGATIVES: [Self; 2] = [Self::ChannelLocked, Self::IdentityTag];

    pub const fn variant(self) -> Variant {
        match self {
            Self::Witness => Variant::Witness,
            Self::ChannelLocked => Variant::ChannelLocked,
            Self::IdentityTag => Variant::IdentityTag,
        }
    }

    pub const fn label(self) -> &'static str {
        self.variant().label()
    }
}

#[derive(Debug, Clone)]
pub struct Case {
    pub kind: CaseKind,
    pub contract: Contract,
}

/// The family as twenty cases.
///
/// The channel-locked arm carries only the pairs its own contract permits, so
/// it is a smaller arm by construction rather than by sampling.
pub fn card_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let mut seed = 0x60A_u64;
    for kind in CaseKind::ALL {
        for before in Assignment::ALL {
            for after in Assignment::ALL {
                if kind == CaseKind::ChannelLocked && before != after {
                    continue;
                }
                for goal_channel in 0..CHANNELS {
                    let mut contract =
                        Contract::new(before, after, kind.variant(), goal_channel, seed);
                    if seed & 1 == 1 {
                        contract.scale = Sign::Negative;
                    }
                    cases.push(Case { kind, contract });
                    seed += 1;
                }
            }
        }
    }
    cases
}

pub fn cases_of(kind: CaseKind) -> Vec<Case> {
    card_cases()
        .into_iter()
        .filter(|case| case.kind == kind)
        .collect()
}

pub trait PublicPolicy {
    fn name(&self) -> &'static str;
    /// The channel to drive, given the channel that was pulsed.
    fn drive(&self, contract: &Contract, pulsed: usize) -> usize;
}

/// Run a policy: pulse the goal channel, then drive where the policy says.
pub fn run_policy<P: PublicPolicy>(contract: &Contract, policy: &P) -> i32 {
    let pulsed = contract.goal_channel;
    run(
        contract,
        &[pulse_for(pulsed), drive_for(policy.drive(contract, pulsed))],
    )
}

/// Drives the channel it pulsed. Correct exactly when nothing was reassigned.
pub struct ChannelIdentity;

impl PublicPolicy for ChannelIdentity {
    fn name(&self) -> &'static str {
        "channel_identity"
    }

    fn drive(&self, _contract: &Contract, pulsed: usize) -> usize {
        pulsed
    }
}

/// Follows a published identity tag where one exists, and channel identity
/// where none does.
pub struct TagFollowing;

impl PublicPolicy for TagFollowing {
    fn name(&self) -> &'static str {
        "tag_following"
    }

    fn drive(&self, contract: &Contract, pulsed: usize) -> usize {
        if contract.tags_visible() {
            contract.after.channel_of(contract.named_source())
        } else {
            pulsed
        }
    }
}

/// Follows the perturbation across the boundary. The exact public policy.
///
/// It reads only the published channel values: the channel showing the
/// perturbation holds the source that was pulsed, and under a history-named
/// goal that is the source to drive.
pub struct ValueTracking;

impl PublicPolicy for ValueTracking {
    fn name(&self) -> &'static str {
        "value_tracking"
    }

    fn drive(&self, contract: &Contract, pulsed: usize) -> usize {
        let perturbed = (0..CHANNELS)
            .find(|channel| {
                contract.channel_value(*channel, pulsed) == Some(contract.scale.value())
            })
            .unwrap_or(pulsed);
        // Public throughout: the naming mode is a published family parameter
        // and which channel was pulsed is the learner's own history. Nothing
        // here consults `before`, which is what keeps this a public policy and
        // not a second privileged reference.
        let names_the_pulsed_source =
            (pulsed == contract.goal_channel) == (contract.naming == GoalNaming::HistoryNamed);
        if names_the_pulsed_source {
            perturbed
        } else {
            1 - perturbed
        }
    }
}

/// Reads the assignment directly. A reference, never a teacher.
pub struct KnownAssignment;

impl PublicPolicy for KnownAssignment {
    fn name(&self) -> &'static str {
        "known_assignment"
    }

    fn drive(&self, contract: &Contract, _pulsed: usize) -> usize {
        contract.after.channel_of(contract.named_source())
    }
}

pub fn mean_value<P: PublicPolicy>(policy: &P, kind: CaseKind) -> f64 {
    let cases = cases_of(kind);
    cases
        .iter()
        .map(|case| f64::from(run_policy(&case.contract, policy)))
        .sum::<f64>()
        / cases.len() as f64
}

pub fn solved_rate<P: PublicPolicy>(policy: &P, kind: CaseKind) -> f64 {
    let cases = cases_of(kind);
    let solved = cases
        .iter()
        .filter(|case| run_policy(&case.contract, policy) > 0)
        .count();
    solved as f64 / cases.len() as f64
}

pub fn score_policy<P: PublicPolicy>(policy: &P) -> BTreeMap<String, f64> {
    CaseKind::ALL
        .into_iter()
        .map(|kind| (kind.label().to_string(), mean_value(policy, kind)))
        .collect()
}

/// The family's central contrast: the witness needs the perturbation, and the
/// two controls do not.
pub fn binding_contrast<P: PublicPolicy>(policy: &P) -> bool {
    solved_rate(policy, CaseKind::Witness) > 0.99
}

pub fn public_ceiling(contract: &Contract) -> f64 {
    pretraining_g0_contract::public_policy_value(
        &VisibleReassignment,
        &assignment_ambiguity(contract),
        HORIZON,
    )
}

pub fn privileged_ceiling(contract: &Contract) -> f64 {
    pretraining_g0_contract::privileged_value_bound(
        &VisibleReassignment,
        &assignment_ambiguity(contract),
        HORIZON,
    )
}

pub fn mean_public_ceiling(kind: CaseKind) -> f64 {
    let cases = cases_of(kind);
    cases
        .iter()
        .map(|case| public_ceiling(&case.contract))
        .sum::<f64>()
        / cases.len() as f64
}

/// Which kernel constructs this family composes.
///
/// The interrupt is gone with the occlusion it carried. The coupling stays,
/// with a single writer and a `Conflict` rule where the composite has two
/// writers and an `Override` rule.
pub fn kernel_use() -> KernelUse {
    KernelUse {
        directed_wiring: true,
        shared_coupling: true,
        interrupt: false,
        restrict: false,
        reveal: false,
        norm_algebra: false,
    }
}

pub fn contract_hash() -> u64 {
    let mut hasher = ContractHasher::new();
    hasher
        .absorb(HORIZON as u64)
        .absorb(CHANNELS as u64)
        .absorb(GOAL_REWARD as u64)
        .absorb(STEP_COST as u64)
        .absorb(Action::ALL.len() as u64);
    for case in card_cases() {
        let contract = case.contract;
        hasher
            .absorb(case.kind as u64)
            .absorb(contract.before as u64)
            .absorb(contract.after as u64)
            .absorb(contract.variant as u64)
            .absorb(contract.goal_channel as u64)
            .absorb(contract.naming as u64)
            .absorb(contract.scale as u64)
            .absorb(contract.visibility as u64)
            .absorb(contract.boundary_visible as u64)
            .absorb(contract.seed);
    }
    hasher.finish()
}

pub fn action_from_index(index: usize) -> Option<Action> {
    Action::from_index(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretraining_g0_contract::noninterference_check;

    #[test]
    fn the_perturbation_moves_with_its_source_across_the_boundary() {
        let moved = Contract::new(
            Assignment::Straight,
            Assignment::Swapped,
            Variant::Witness,
            0,
            1,
        );
        assert_eq!(moved.perturbed_by(0), Source::Left);
        assert_eq!(moved.channel_value(1, 0), Some(1));
        assert_eq!(moved.channel_value(0, 0), Some(0));
        assert!(moved.drive_succeeds(1) && !moved.drive_succeeds(0));

        let held = Contract {
            after: Assignment::Straight,
            ..moved
        };
        assert_eq!(held.channel_value(0, 0), Some(1));
        assert!(held.drive_succeeds(0));
    }

    #[test]
    fn channel_identity_holds_only_where_nothing_was_reassigned() {
        assert!(binding_contrast(&ValueTracking));
        assert!(binding_contrast(&KnownAssignment));
        assert!(!binding_contrast(&ChannelIdentity));
        assert!(!binding_contrast(&TagFollowing));
        assert_eq!(solved_rate(&ChannelIdentity, CaseKind::ChannelLocked), 1.0);
        assert_eq!(solved_rate(&TagFollowing, CaseKind::IdentityTag), 1.0);
        assert_eq!(solved_rate(&TagFollowing, CaseKind::ChannelLocked), 1.0);
    }

    #[test]
    fn the_hidden_assignment_does_not_leak_before_a_pulse() {
        let left = Contract::new(
            Assignment::Straight,
            Assignment::Straight,
            Variant::Witness,
            0,
            1,
        );
        let right = Contract {
            before: Assignment::Swapped,
            after: Assignment::Swapped,
            ..left
        };
        let verdict = noninterference_check(
            &VisibleReassignment,
            "the assignment is invisible until something is pulsed",
            &left,
            &right,
            HORIZON,
            |actions| actions.iter().all(|a| a.pulse_channel().is_none()),
            |action| action.name().into(),
        );
        assert!(verdict.holds, "{verdict:?}");
    }

    #[test]
    fn residual_ambiguity_is_real_and_agent_irrelevant() {
        let witness = Contract::new(
            Assignment::Straight,
            Assignment::Straight,
            Variant::Witness,
            0,
            1,
        );
        let prefix = [pulse_for(witness.goal_channel)];
        assert_eq!(raw_ambiguity(&witness, &prefix), 2);
        assert_eq!(relevant_ambiguity(&witness, &prefix), 1);
    }

    #[test]
    fn continuous_visibility_is_what_closes_the_gap() {
        let witness = Contract::new(
            Assignment::Straight,
            Assignment::Swapped,
            Variant::Witness,
            0,
            1,
        );
        assert_eq!(public_ceiling(&witness), privileged_ceiling(&witness));
        assert!(public_ceiling(&witness.occlude()) < privileged_ceiling(&witness.occlude()));
    }
}
