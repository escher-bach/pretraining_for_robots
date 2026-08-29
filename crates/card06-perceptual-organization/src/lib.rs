//! Card 06 — a finite source-binding world with executable coupling and occlusion.
mod audit;
mod render;
pub use audit::*;
pub use render::*;

use pretraining_g0_contract::{
    agent_equivalence, identify, AmbiguitySet, ContractHasher, Coupling, CouplingRule, Displaced,
    Fragment, Guard, Interrupt, KernelUse, PubliclyObservable, Resume,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

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
}
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BoundaryTiming {
    Early,
    Late,
}
impl BoundaryTiming {
    pub const fn flipped(self) -> Self {
        match self {
            Self::Early => Self::Late,
            Self::Late => Self::Early,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NoiseMode {
    MatchedMarginal,
    VisibleMismatch,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Variant {
    Witness,
    ChannelLocked,
    ShuffledCovariance,
    FrozenDuringAbsence,
    IdentityTag,
}
impl Variant {
    pub const ALL: [Self; 5] = [
        Self::Witness,
        Self::ChannelLocked,
        Self::ShuffledCovariance,
        Self::FrozenDuringAbsence,
        Self::IdentityTag,
    ];
    pub const fn label(self) -> &'static str {
        match self {
            Self::Witness => "witness",
            Self::ChannelLocked => "channel_locked",
            Self::ShuffledCovariance => "shuffled_covariance",
            Self::FrozenDuringAbsence => "frozen_during_absence",
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
}

/// Public contract parameters plus private assignment/noise realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contract {
    pub before: Assignment,
    pub after: Assignment,
    pub noise_sign: Sign,
    pub variant: Variant,
    pub goal_channel: usize,
    pub occluded_channel: usize,
    pub timing: BoundaryTiming,
    pub scale: Sign,
    pub boundary_visible: bool,
    pub noise_mode: NoiseMode,
    pub seed: u64,
}
impl Contract {
    pub fn new(
        before: Assignment,
        after: Assignment,
        noise_sign: Sign,
        variant: Variant,
        goal_channel: usize,
        occluded_channel: usize,
        seed: u64,
    ) -> Self {
        Self {
            before,
            after,
            noise_sign,
            variant,
            goal_channel,
            occluded_channel,
            timing: BoundaryTiming::Early,
            scale: Sign::Positive,
            boundary_visible: true,
            noise_mode: NoiseMode::MatchedMarginal,
            seed,
        }
    }
    pub fn coupling(&self) -> Coupling {
        Coupling::new(0, CouplingRule::Override)
    }
    pub fn occlusion(&self) -> Interrupt {
        Interrupt::new(
            Guard::AfterStep(0),
            if self.variant == Variant::FrozenDuringAbsence {
                Displaced::Frozen
            } else {
                Displaced::Continues
            },
            Resume::FromState,
        )
    }
    pub fn tags_visible(&self) -> bool {
        self.variant == Variant::IdentityTag
    }
    pub const fn named_source(&self) -> Source {
        self.before.source_at(self.goal_channel)
    }
    /// The interrupt controls the latent source transition; frozen does not alter `after`.
    pub fn source_value(&self, source: Source) -> i32 {
        if self.occlusion().displaced == Displaced::Frozen {
            return 0;
        }
        let timing = match self.timing {
            BoundaryTiming::Early => 1,
            BoundaryTiming::Late => -1,
        };
        let relation = if source == self.named_source() { 1 } else { -1 };
        timing * relation * self.scale.value()
    }
    pub fn noise_value(&self, channel: usize) -> i32 {
        if self.occlusion().displaced == Displaced::Frozen {
            return 0;
        }
        let sign = if self.variant == Variant::ShuffledCovariance {
            // Independent generator bits per channel retain the {-1,+1}
            // marginal but sever the source-to-channel covariance.
            if (self.seed >> channel) & 1 == 0 {
                Sign::Negative
            } else {
                Sign::Positive
            }
        } else {
            self.noise_sign
        };
        match self.noise_mode {
            NoiseMode::MatchedMarginal => sign.value() * self.scale.value(),
            NoiseMode::VisibleMismatch => 2 * self.noise_sign.value(),
        }
    }
    /// The source and noise are competing contributions; Override selects the later noise writer.
    pub fn coupled_output(&self, channel: usize) -> i32 {
        let mut writers = vec![self.source_value(self.after.source_at(channel)) as f64];
        if channel == self.occluded_channel
            || self.variant == Variant::ShuffledCovariance
            || !self.boundary_visible
        {
            writers.push(self.noise_value(channel) as f64);
        }
        self.coupling()
            .resolve(&writers)
            .expect("source writer exists") as i32
    }
    pub fn goal_drive_succeeds(&self, channel: usize) -> bool {
        self.after.source_at(channel) == self.named_source()
            && self.noise_mode == NoiseMode::MatchedMarginal
    }
    pub fn relabel_channels(&self) -> Self {
        Self {
            before: self.before.flipped(),
            after: self.after.flipped(),
            goal_channel: 1 - self.goal_channel,
            occluded_channel: 1 - self.occluded_channel,
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
    pub fn flip_timing(&self) -> Self {
        Self {
            timing: self.timing.flipped(),
            ..*self
        }
    }
    pub fn change_goal(&self) -> Self {
        Self {
            goal_channel: 1 - self.goal_channel,
            ..*self
        }
    }
    pub fn hide_boundary(&self) -> Self {
        Self {
            boundary_visible: false,
            ..*self
        }
    }
    pub fn mismatched_noise(&self) -> Self {
        Self {
            noise_mode: NoiseMode::VisibleMismatch,
            ..*self
        }
    }
}

pub fn run(contract: &Contract, actions: &[Action]) -> i32 {
    if actions.len() < HORIZON || actions[0].pulse_channel() != Some(contract.goal_channel) {
        return 0;
    }
    match actions[1].drive_channel() {
        Some(channel) if contract.goal_drive_succeeds(channel) => {
            GOAL_REWARD - HORIZON as i32 * STEP_COST
        }
        _ => 0,
    }
}
pub struct PerceptualOrganization;
impl Fragment for PerceptualOrganization {
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
                .is_some_and(|channel| contract.goal_drive_succeeds(channel))
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
impl PubliclyObservable for PerceptualOrganization {
    fn public_trace(&self, c: &Contract, actions: &[Action]) -> Vec<i64> {
        let mut trace = vec![
            c.variant as i64,
            c.goal_channel as i64,
            c.occluded_channel as i64,
            c.timing as i64,
            c.scale as i64,
            c.boundary_visible as i64,
            c.noise_mode as i64,
        ];
        if c.tags_visible() {
            trace.push(20 + c.before.channel_of(Source::Left) as i64);
        }
        if actions.is_empty() {
            return trace;
        }
        trace.push(actions[0].index() as i64);
        if actions[0].pulse_channel().is_none() {
            return trace;
        }
        trace.push(30 + c.timing as i64);
        trace.push(c.coupled_output(0) as i64);
        trace.push(c.coupled_output(1) as i64);
        if c.tags_visible() {
            trace.push(40 + c.after.channel_of(Source::Left) as i64);
        }
        if actions.len() > 1 {
            trace.push(actions[1].index() as i64);
            trace.push((run(c, actions) > 0) as i64);
        }
        trace
    }
}

fn candidate_seed(before: Assignment, after: Assignment, noise: Sign, base: &Contract) -> u64 {
    let _ = (before, after, noise);
    // Replay seed stays fixed while the posterior varies latent assignments:
    // coupling a generator seed to an assignment would itself be a leakage path.
    base.seed
}
/// Exact finite posterior candidates; shared query functions, not a bespoke posterior, consume it.
pub fn assignment_ambiguity(c: &Contract) -> AmbiguitySet<Contract> {
    let mut candidates = vec![*c];
    for before in Assignment::ALL {
        for after in Assignment::ALL {
            for noise_sign in Sign::ALL {
                if c.variant == Variant::ChannelLocked && after != before {
                    continue;
                }
                let candidate = Contract {
                    before,
                    after,
                    noise_sign,
                    seed: candidate_seed(before, after, noise_sign, c),
                    ..*c
                };
                // Seed is generator metadata, not another latent world state:
                // do not give the realized assignment/noise realization twice
                // the prior mass merely because its replay seed differs.
                if before != c.before || after != c.after || noise_sign != c.noise_sign {
                    candidates.push(candidate);
                }
            }
        }
    }
    AmbiguitySet::uniform(candidates)
}
/// Surviving ambiguity modulo shared finite agent equivalence.
pub fn relevant_ambiguity(c: &Contract, prefix: &[Action]) -> usize {
    let set = assignment_ambiguity(c);
    let survivors = identify(&PerceptualOrganization, &set, prefix);
    let mut reps = Vec::new();
    'candidate: for index in survivors {
        for representative in &reps {
            if agent_equivalence(
                &PerceptualOrganization,
                &set.candidates[index],
                &set.candidates[*representative],
                HORIZON,
                |a| a.name().into(),
            )
            .equivalent
            {
                continue 'candidate;
            }
        }
        reps.push(index);
    }
    reps.len()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CaseKind {
    Witness,
    ChannelLocked,
    ShuffledCovariance,
    FrozenDuringAbsence,
    IdentityTag,
}
impl CaseKind {
    pub const ALL: [Self; 5] = [
        Self::Witness,
        Self::ChannelLocked,
        Self::ShuffledCovariance,
        Self::FrozenDuringAbsence,
        Self::IdentityTag,
    ];
    pub const fn variant(self) -> Variant {
        match self {
            Self::Witness => Variant::Witness,
            Self::ChannelLocked => Variant::ChannelLocked,
            Self::ShuffledCovariance => Variant::ShuffledCovariance,
            Self::FrozenDuringAbsence => Variant::FrozenDuringAbsence,
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
pub fn card_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    let mut seed = 0xC06_u64;
    for kind in CaseKind::ALL {
        for before in Assignment::ALL {
            for after in Assignment::ALL {
                for noise in Sign::ALL {
                    if kind == CaseKind::ChannelLocked && before != after {
                        continue;
                    }
                    let mut c = Contract::new(before, after, noise, kind.variant(), 0, 1, seed);
                    if seed & 1 == 1 {
                        c.goal_channel = 1;
                        c.occluded_channel = 0;
                    }
                    if seed & 2 == 2 {
                        c.timing = BoundaryTiming::Late;
                    }
                    if seed & 4 == 4 {
                        c.scale = Sign::Negative;
                    }
                    cases.push(Case { kind, contract: c });
                    seed += 1;
                }
            }
        }
    }
    cases
}

pub trait PublicPolicy {
    fn name(&self) -> &'static str;
    fn act(&self, contract: &Contract, step: usize) -> Action;
}
pub fn run_policy<P: PublicPolicy>(c: &Contract, policy: &P) -> i32 {
    run(c, &[policy.act(c, 0), policy.act(c, 1)])
}
fn pulse_for(channel: usize) -> Action {
    if channel == 0 {
        Action::PulseZero
    } else {
        Action::PulseOne
    }
}
fn drive_for(channel: usize) -> Action {
    if channel == 0 {
        Action::DriveZero
    } else {
        Action::DriveOne
    }
}
pub struct PerChannel;
impl PublicPolicy for PerChannel {
    fn name(&self) -> &'static str {
        "per_channel"
    }
    fn act(&self, c: &Contract, step: usize) -> Action {
        if step == 0 {
            pulse_for(c.goal_channel)
        } else {
            drive_for(c.goal_channel)
        }
    }
}
pub struct ChannelIdentity;
impl PublicPolicy for ChannelIdentity {
    fn name(&self) -> &'static str {
        "channel_identity"
    }
    fn act(&self, c: &Contract, step: usize) -> Action {
        PerChannel.act(c, step)
    }
}
pub struct AssumeNothingMoved;
impl PublicPolicy for AssumeNothingMoved {
    fn name(&self) -> &'static str {
        "assume_nothing_moved"
    }
    fn act(&self, c: &Contract, step: usize) -> Action {
        PerChannel.act(c, step)
    }
}
pub struct KnownAssignment;
impl PublicPolicy for KnownAssignment {
    fn name(&self) -> &'static str {
        "known_assignment"
    }
    fn act(&self, c: &Contract, step: usize) -> Action {
        if step == 0 {
            pulse_for(c.goal_channel)
        } else {
            drive_for(c.after.channel_of(c.named_source()))
        }
    }
}
pub struct TagFollowing;
impl PublicPolicy for TagFollowing {
    fn name(&self) -> &'static str {
        "tag_following"
    }
    fn act(&self, c: &Contract, step: usize) -> Action {
        if c.tags_visible() {
            KnownAssignment.act(c, step)
        } else {
            PerChannel.act(c, step)
        }
    }
}
pub fn mean_value<P: PublicPolicy>(p: &P, kind: CaseKind) -> f64 {
    let cases: Vec<_> = card_cases()
        .into_iter()
        .filter(|c| c.kind == kind)
        .collect();
    cases
        .iter()
        .map(|c| run_policy(&c.contract, p) as f64)
        .sum::<f64>()
        / cases.len() as f64
}
pub fn score_policy<P: PublicPolicy>(p: &P) -> BTreeMap<String, f64> {
    CaseKind::ALL
        .into_iter()
        .map(|kind| (kind.label().into(), mean_value(p, kind)))
        .collect()
}
pub fn public_ceiling(c: &Contract) -> f64 {
    pretraining_g0_contract::public_policy_value(
        &PerceptualOrganization,
        &assignment_ambiguity(c),
        HORIZON,
    )
}
pub fn privileged_ceiling(c: &Contract) -> f64 {
    pretraining_g0_contract::privileged_value_bound(
        &PerceptualOrganization,
        &assignment_ambiguity(c),
        HORIZON,
    )
}
pub fn exact_assignment_posterior_value(kind: CaseKind) -> f64 {
    let cases: Vec<_> = card_cases()
        .into_iter()
        .filter(|c| c.kind == kind)
        .collect();
    cases
        .iter()
        .map(|c| public_ceiling(&c.contract))
        .sum::<f64>()
        / cases.len() as f64
}
pub fn all_sequences() -> Vec<Vec<Action>> {
    pretraining_g0_contract::sequences_of_length(&Action::ALL, HORIZON)
}
pub fn kernel_use() -> KernelUse {
    KernelUse {
        directed_wiring: true,
        shared_coupling: true,
        interrupt: true,
        restrict: false,
        reveal: false,
        norm_algebra: false,
    }
}
pub fn contract_hash() -> u64 {
    let mut h = ContractHasher::new();
    h.absorb(HORIZON as u64)
        .absorb(CHANNELS as u64)
        .absorb(GOAL_REWARD as u64)
        .absorb(STEP_COST as u64)
        .absorb(Action::ALL.len() as u64);
    for case in card_cases() {
        let c = case.contract;
        h.absorb(case.kind as u64)
            .absorb(c.before as u64)
            .absorb(c.after as u64)
            .absorb(c.noise_sign as u64)
            .absorb(c.variant as u64)
            .absorb(c.goal_channel as u64)
            .absorb(c.occluded_channel as u64)
            .absorb(c.timing as u64)
            .absorb(c.scale as u64)
            .absorb(c.boundary_visible as u64)
            .absorb(c.noise_mode as u64)
            .absorb(c.seed);
    }
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretraining_g0_contract::{identification_diameter, noninterference_check};
    #[test]
    fn override_and_interrupt_execute_the_public_dynamics() {
        let c = Contract::new(
            Assignment::Straight,
            Assignment::Swapped,
            Sign::Negative,
            Variant::Witness,
            0,
            1,
            7,
        );
        assert_eq!(c.occlusion().displaced, Displaced::Continues);
        assert_eq!(c.source_value(Source::Left), 1);
        assert_eq!(
            c.coupled_output(1),
            -1,
            "later noise writer overrides the competing +1 source writer"
        );
        let frozen = Contract {
            variant: Variant::FrozenDuringAbsence,
            ..c
        };
        assert_eq!(frozen.after, c.after);
        assert_eq!(frozen.source_value(Source::Left), 0);
        assert_eq!(frozen.coupled_output(0), 0);
    }
    #[test]
    fn exact_posterior_tracks_binding_not_channel_identity() {
        let w = card_cases()
            .into_iter()
            .find(|c| c.kind == CaseKind::Witness)
            .unwrap()
            .contract;
        let set = assignment_ambiguity(&w);
        assert!(
            identification_diameter(&PerceptualOrganization, &set, &[pulse_for(w.goal_channel)])
                < set.len()
        );
        assert!(
            exact_assignment_posterior_value(CaseKind::Witness)
                > mean_value(&PerChannel, CaseKind::Witness)
        );
        assert_eq!(
            exact_assignment_posterior_value(CaseKind::ShuffledCovariance),
            mean_value(&PerChannel, CaseKind::ShuffledCovariance)
        );
    }
    #[test]
    fn hidden_assignment_does_not_leak_before_intervention() {
        let left = Contract::new(
            Assignment::Straight,
            Assignment::Straight,
            Sign::Positive,
            Variant::Witness,
            0,
            1,
            1,
        );
        let right = Contract {
            before: Assignment::Swapped,
            after: Assignment::Swapped,
            ..left
        };
        assert!(
            noninterference_check(
                &PerceptualOrganization,
                "hidden assignment before pulse",
                &left,
                &right,
                HORIZON,
                |actions| actions.iter().all(|a| a.pulse_channel().is_none()),
                |a| a.name().into()
            )
            .holds
        );
    }
}
