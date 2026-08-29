//! Shared G0 contract and fragment machinery.
//!
//! This is the layer the finite seed families share, extracted from card 04
//! after building it rather than designed before it. That order matters: a
//! single world needs no shared vocabulary, because there is nothing to share
//! with. The layer earns its place at the moment a second card must reuse the
//! first card's environment *as the same object* instead of a copy, which is
//! what keeps a later cross-card claim from being confounded with world
//! difficulty.
//!
//! Five things live here, and deliberately nothing else:
//!
//! 1. an **environment** — a finite configuration space with an adjacency
//!    structure and its symmetry group, shared across cards;
//! 2. a **process kernel** ([`kernel`]) — the five operators and the norm
//!    algebra `EMBODIED-PROCESS.md` declares, as shared executable data rather
//!    than as labels each card re-derives privately;
//! 3. a **fragment** — the trait a card implements to become exhaustively
//!    auditable;
//! 4. the **audit machinery** — enumeration, ceilings, ambiguity gap, orbit
//!    verdicts, and the bracket/isolation analysis, all generic; and
//! 5. the **query algebra** ([`query`]) — identification, public and privileged
//!    ceilings, epistemic value, matched controls, non-interference, and
//!    history ablation, all derived from one ambiguity-set object.
//!
//! Items 2 and 5 were added when cards 02, 03, 05, and 06 arrived. Card 04 alone
//! needed neither: it has no hidden state, so its query answers are trivial, and
//! a single card can open-code its own norm without anything to be consistent
//! with. Both layers exist for the same reason the environment does — the moment
//! a second card claims the same construct, a copy makes a later cross-card
//! claim a claim about two different objects.
//!
//! What is *not* here is any card's contract shape, nor a general interpreter
//! that executes an arbitrary composition. Card 04 publishes a goal, a
//! prohibition, a hazard, a distractor, and a switch; card 03 publishes
//! reachability and reveals. Fixing a single contract struct here would force
//! every later card through card 04's ontology, which is the opposite of a
//! shared layer; and an interpreter would have to fix exactly that struct to
//! have a state to interpret.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub mod kernel;
pub mod query;

pub use kernel::{
    BoundaryEffect, Coupling, CouplingRule, Displaced, Guard, GuardContext, IndexSet, Interrupt,
    KernelUse, Norm, NormVerdict, ResourceScope, Restriction, Resume, Reveal,
};
pub use query::{
    ablated_policy_value, agent_equivalence, ambiguity_report, check_information_orbit,
    epistemic_value, identification_diameter, identify, matched_control_verdict,
    noninterference_check, privileged_value_bound, public_optimal_actions_at, public_policy_value,
    public_policy_value_and_first_actions, ActionValue, AmbiguityReport, AmbiguitySet, Coarsened,
    EquivalenceCertificate, InformationVerdict, MatchedControlVerdict, NonInterference,
    PubliclyObservable, VALUE_EPSILON,
};

/// A finite ring of cells: the configuration structure cards 04 and 03 share.
///
/// A ring is used rather than a line because it gives every target two routes
/// of different length, which is what lets a forbidden or unreachable short
/// route still leave a correct alternative.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ring {
    pub cells: usize,
}

impl Ring {
    pub const fn new(cells: usize) -> Self {
        Self { cells }
    }

    pub const fn advance(&self, cell: usize) -> usize {
        (cell + 1) % self.cells
    }

    pub const fn retreat(&self, cell: usize) -> usize {
        (cell + self.cells - 1) % self.cells
    }

    /// Steps from `from` to `to` going forward.
    pub const fn forward_distance(&self, from: usize, to: usize) -> usize {
        (to + self.cells - from) % self.cells
    }

    /// The shorter of the two routes.
    pub fn distance(&self, from: usize, to: usize) -> usize {
        let forward = self.forward_distance(from, to);
        forward.min(self.cells - forward)
    }

    /// The longer route, which is what a prohibition on the short one forces.
    pub fn detour(&self, from: usize, to: usize) -> usize {
        self.cells - self.distance(from, to)
    }

    /// The ring's symmetry group: rotations and reflections, and nothing else.
    ///
    /// An arbitrary permutation of cell labels does **not** preserve adjacency,
    /// so a card that describes its orbit as "permute configuration labels"
    /// without this restriction would be claiming a false invariance.
    pub fn symmetries(&self) -> Vec<Symmetry> {
        let mut group = Vec::with_capacity(self.cells * 2);
        for shift in 0..self.cells {
            group.push(Symmetry {
                shift,
                reflect: false,
                cells: self.cells,
            });
            group.push(Symmetry {
                shift,
                reflect: true,
                cells: self.cells,
            });
        }
        group
    }
}

/// One element of the ring's dihedral symmetry group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Symmetry {
    pub shift: usize,
    pub reflect: bool,
    pub cells: usize,
}

impl Symmetry {
    pub const fn identity(cells: usize) -> Self {
        Self {
            shift: 0,
            reflect: false,
            cells,
        }
    }

    pub const fn apply(&self, cell: usize) -> usize {
        let base = if self.reflect {
            (self.cells - cell) % self.cells
        } else {
            cell
        };
        (base + self.shift) % self.cells
    }

    /// Whether this symmetry exchanges the two directions of travel, and
    /// therefore the meaning of the two move actions.
    pub const fn swaps_directions(&self) -> bool {
        self.reflect
    }

    pub fn name(&self) -> String {
        if self.reflect {
            format!("reflect_then_rotate_{}", self.shift)
        } else {
            format!("rotate_{}", self.shift)
        }
    }
}

/// A finite world a card can be audited as.
///
/// The value function receives the whole trajectory rather than only the final
/// state, because several cards score *when* a configuration settled, not only
/// whether it did.
pub trait Fragment {
    type Action: Copy + Ord + std::fmt::Debug;
    type Contract: Clone;

    fn actions(&self) -> Vec<Self::Action>;
    fn horizon(&self) -> usize;
    fn start(&self, contract: &Self::Contract) -> usize;
    /// The configuration after `action` is taken with `executed` actions already
    /// behind it.
    ///
    /// The step index is part of the signature because a family may restore or
    /// withdraw an effect mid-episode — card 03 publicly restores an actuator —
    /// and folding that into the cell would make the configuration space carry
    /// the clock. A family with a time-invariant transition simply ignores it.
    fn step(
        &self,
        contract: &Self::Contract,
        cell: usize,
        executed: usize,
        action: Self::Action,
    ) -> usize;
    /// Score a complete rollout. `trajectory[0]` is the starting cell.
    fn value(
        &self,
        contract: &Self::Contract,
        trajectory: &[usize],
        actions: &[Self::Action],
    ) -> i32;
    /// The value a solver with access to every privileged field could reach.
    ///
    /// A card with no epistemic content returns the public value, which is what
    /// makes its ambiguity gap zero. Overriding this is how a card with hidden
    /// state declares a real gap.
    fn privileged_value(
        &self,
        contract: &Self::Contract,
        trajectory: &[usize],
        actions: &[Self::Action],
    ) -> i32 {
        self.value(contract, trajectory, actions)
    }
}

/// Every action sequence of a given length.
pub fn sequences_of_length<A: Copy>(actions: &[A], length: usize) -> Vec<Vec<A>> {
    let mut sequences = vec![Vec::new()];
    for _ in 0..length {
        let mut grown = Vec::with_capacity(sequences.len() * actions.len());
        for prefix in &sequences {
            for action in actions {
                let mut next = prefix.clone();
                next.push(*action);
                grown.push(next);
            }
        }
        sequences = grown;
    }
    sequences
}

/// Roll a fixed sequence forward, returning the trajectory including the start.
pub fn trajectory<F: Fragment>(
    fragment: &F,
    contract: &F::Contract,
    actions: &[F::Action],
) -> Vec<usize> {
    let mut cell = fragment.start(contract);
    let mut path = vec![cell];
    for (executed, action) in actions.iter().enumerate() {
        cell = fragment.step(contract, cell, executed, *action);
        path.push(cell);
    }
    path
}

/// The exact ceiling and every sequence achieving it, over a stated horizon.
///
/// A policy re-solving mid-episode must pass the horizon it actually has left;
/// planning a fresh full horizon from the current cell overstates the budget
/// and makes an "exact" policy inexact.
pub fn value_bounds_over<F: Fragment>(
    fragment: &F,
    contract: &F::Contract,
    horizon: usize,
) -> (i32, Vec<Vec<F::Action>>) {
    let actions = fragment.actions();
    let mut best = i32::MIN;
    let mut optimal = Vec::new();
    for sequence in sequences_of_length(&actions, horizon) {
        let path = trajectory(fragment, contract, &sequence);
        let value = fragment.value(contract, &path, &sequence);
        if value > best {
            best = value;
            optimal.clear();
        }
        if value == best {
            optimal.push(sequence);
        }
    }
    (best, optimal)
}

pub fn value_bounds<F: Fragment>(
    fragment: &F,
    contract: &F::Contract,
) -> (i32, Vec<Vec<F::Action>>) {
    value_bounds_over(fragment, contract, fragment.horizon())
}

/// The optimal first actions, which is what a paired contrast reads.
pub fn optimal_first_actions<F: Fragment>(fragment: &F, contract: &F::Contract) -> Vec<F::Action> {
    let (_, optimal) = value_bounds(fragment, contract);
    let mut first: Vec<F::Action> = optimal.iter().filter_map(|s| s.first().copied()).collect();
    first.sort();
    first.dedup();
    first
}

/// The actions that attain the ceiling from a mid-episode prefix.
///
/// This is the primitive a step-wise policy should use, and it exists because
/// the obvious alternative is wrong in a way that is easy to miss. Re-solving
/// "the remaining episode" hands the solver a fresh contract whose step clock
/// starts at zero, so anything time-dependent — a restored actuator, an
/// interval that ends at a stated step — is evaluated at the wrong index. Card
/// 03 lost its own restoration witness to exactly that before this existed.
///
/// Enumerating *completions of the prefix* keeps the absolute clock, because the
/// value function receives the whole action sequence. Nothing needs rebasing,
/// so nothing can be rebased incorrectly.
///
/// The whole set is returned, not one member. Where several actions attain the
/// ceiling the world is indifferent between them, and a caller that silently
/// took the first would be asserting a preference the contract does not have.
pub fn optimal_actions_from<F: Fragment>(
    fragment: &F,
    contract: &F::Contract,
    prefix: &[F::Action],
) -> Vec<F::Action> {
    let horizon = fragment.horizon();
    if prefix.len() >= horizon {
        return Vec::new();
    }
    let actions = fragment.actions();
    let mut best = i32::MIN;
    let mut chosen = Vec::new();
    for suffix in sequences_of_length(&actions, horizon - prefix.len()) {
        let mut sequence = prefix.to_vec();
        sequence.extend(suffix);
        let path = trajectory(fragment, contract, &sequence);
        let value = fragment.value(contract, &path, &sequence);
        let next = sequence[prefix.len()];
        if value > best {
            best = value;
            chosen.clear();
        }
        if value == best && !chosen.contains(&next) {
            chosen.push(next);
        }
    }
    chosen.sort();
    chosen.dedup();
    chosen
}

/// The configuration reached by following a prefix, with the absolute clock.
pub fn cell_after<F: Fragment>(
    fragment: &F,
    contract: &F::Contract,
    prefix: &[F::Action],
) -> usize {
    *trajectory(fragment, contract, prefix)
        .last()
        .expect("a trajectory contains its start")
}

/// The gap between what a privileged solver and a public one can reach.
///
/// Computed rather than assumed, so a later edit that hides a field is caught
/// instead of quietly making every failure deniable.
pub fn ambiguity_gap<F: Fragment>(fragment: &F, contract: &F::Contract) -> i32 {
    let actions = fragment.actions();
    let mut public_best = i32::MIN;
    let mut privileged_best = i32::MIN;
    for sequence in sequences_of_length(&actions, fragment.horizon()) {
        let path = trajectory(fragment, contract, &sequence);
        public_best = public_best.max(fragment.value(contract, &path, &sequence));
        privileged_best =
            privileged_best.max(fragment.privileged_value(contract, &path, &sequence));
    }
    privileged_best - public_best
}

/// One transform in a card's invariance orbit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrbitVerdict {
    pub transform: String,
    pub semantics_preserving: bool,
    pub ceiling_unchanged: bool,
    pub optimal_actions_correspond: bool,
    pub verdict_holds: bool,
}

/// Check one orbit transform across a set of contracts.
///
/// A semantics-preserving transform must leave the ceiling and the
/// corresponding optimal set fixed. A semantics-changing one must move at least
/// one of them — a transform that changes nothing is testing nothing, and both
/// halves are checked so a vacuous transform cannot pass as a verdict.
///
/// The action set compared is the optimal *first* actions. That is the right
/// observable for a card whose contrast is at the first decision and the wrong
/// one for a card whose contrast is later: card 02's first move is the same in
/// both modes and its whole claim lives at the third. Use [`check_orbit_with`]
/// there, rather than reading agreement where the card claims a difference.
pub fn check_orbit<F, T, M>(
    fragment: &F,
    contracts: &[F::Contract],
    name: &str,
    semantics_preserving: bool,
    transform: T,
    map_action: M,
) -> OrbitVerdict
where
    F: Fragment,
    T: Fn(&F::Contract) -> F::Contract,
    M: Fn(F::Action) -> F::Action,
{
    check_orbit_with(
        fragment,
        contracts,
        name,
        semantics_preserving,
        transform,
        map_action,
        optimal_first_actions,
    )
}

/// [`check_orbit`] against a stated action observable.
pub fn check_orbit_with<F, T, M, O>(
    fragment: &F,
    contracts: &[F::Contract],
    name: &str,
    semantics_preserving: bool,
    transform: T,
    map_action: M,
    observable: O,
) -> OrbitVerdict
where
    F: Fragment,
    T: Fn(&F::Contract) -> F::Contract,
    M: Fn(F::Action) -> F::Action,
    O: Fn(&F, &F::Contract) -> Vec<F::Action>,
{
    let mut ceiling_unchanged = true;
    let mut actions_correspond = true;
    for contract in contracts {
        let moved = transform(contract);
        let (base_ceiling, _) = value_bounds(fragment, contract);
        let (moved_ceiling, _) = value_bounds(fragment, &moved);
        ceiling_unchanged &= base_ceiling == moved_ceiling;

        let mut expected: Vec<F::Action> = observable(fragment, contract)
            .into_iter()
            .map(&map_action)
            .collect();
        expected.sort();
        expected.dedup();
        let mut found = observable(fragment, &moved);
        found.sort();
        found.dedup();
        actions_correspond &= expected == found;
    }
    let verdict_holds = if semantics_preserving {
        ceiling_unchanged && actions_correspond
    } else {
        !(ceiling_unchanged && actions_correspond)
    };
    OrbitVerdict {
        transform: name.to_string(),
        semantics_preserving,
        ceiling_unchanged,
        optimal_actions_correspond: actions_correspond,
        verdict_holds,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KindScore {
    pub solved: usize,
    pub total: usize,
    pub rate: f64,
    pub optimal_rate: f64,
}

/// Whether a negative isolates the failure mode it is paired against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Isolation {
    pub negative: String,
    pub paired_witness: String,
    pub isolating_baselines: Vec<String>,
    pub isolates: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BracketStructure {
    /// The property that does the real work.
    pub every_negative_isolates: bool,
    pub isolation: Vec<Isolation>,
    /// The stronger property card 04 §9 states in prose. Enumeration showed it
    /// false there: easy negatives are solvable by several baselines at once.
    /// It is reported so later cards can be checked against it rather than
    /// inheriting the assumption.
    pub each_failing_baseline_optimal_on_exactly_one: bool,
    pub failing_baselines_optimal_on_multiple: Vec<String>,
    pub failing_baselines_optimal_on_none: Vec<String>,
}

/// One baseline's evidence, in the shape the bracket analysis consumes.
pub struct BaselineEvidence {
    pub name: String,
    /// Case-kind label to score.
    pub scores: BTreeMap<String, KindScore>,
    /// Negative labels this baseline attains the ceiling on everywhere.
    pub optimal_on_negatives: Vec<String>,
    /// Whether this baseline is the card's ceiling policy, which is excluded
    /// from the failing-baseline analysis.
    pub is_ceiling: bool,
}

/// Analyse the bracket: does each negative isolate the failure it is paired with?
///
/// `pairing` maps a negative's label to the witness label it is meant to
/// isolate. A negative isolates when some baseline is optimal on it *and* fails
/// that witness. A negative nothing brackets is not isolating anything, and a
/// baseline optimal nowhere is not bracketing anything.
pub fn analyse_bracket(
    baselines: &[BaselineEvidence],
    pairing: &[(String, String)],
) -> BracketStructure {
    let failing: Vec<&BaselineEvidence> =
        baselines.iter().filter(|entry| !entry.is_ceiling).collect();

    let mut isolation = Vec::new();
    for (negative, witness) in pairing {
        let isolating: Vec<String> = failing
            .iter()
            .filter(|entry| {
                entry.optimal_on_negatives.contains(negative)
                    && entry
                        .scores
                        .get(witness)
                        .map(|score| score.rate < 1.0)
                        .unwrap_or(false)
            })
            .map(|entry| entry.name.clone())
            .collect();
        isolation.push(Isolation {
            negative: negative.clone(),
            paired_witness: witness.clone(),
            isolates: !isolating.is_empty(),
            isolating_baselines: isolating,
        });
    }

    let multiple: Vec<String> = failing
        .iter()
        .filter(|entry| entry.optimal_on_negatives.len() > 1)
        .map(|entry| entry.name.clone())
        .collect();
    let none: Vec<String> = failing
        .iter()
        .filter(|entry| entry.optimal_on_negatives.is_empty())
        .map(|entry| entry.name.clone())
        .collect();

    BracketStructure {
        every_negative_isolates: isolation.iter().all(|entry| entry.isolates),
        isolation,
        each_failing_baseline_optimal_on_exactly_one: multiple.is_empty() && none.is_empty(),
        failing_baselines_optimal_on_multiple: multiple,
        failing_baselines_optimal_on_none: none,
    }
}

/// An FNV-1a accumulator, so a card's contract set hashes stably and a silent
/// change of meaning is detectable rather than something a reader must notice.
pub struct ContractHasher {
    state: u64,
}

impl Default for ContractHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl ContractHasher {
    pub const fn new() -> Self {
        Self {
            state: 0xcbf29ce484222325,
        }
    }

    pub fn absorb(&mut self, value: u64) -> &mut Self {
        self.state ^= value;
        self.state = self.state.wrapping_mul(0x100000001b3);
        self
    }

    pub fn absorb_option(&mut self, value: Option<u64>) -> &mut Self {
        self.absorb(value.unwrap_or(u64::MAX))
    }

    pub const fn finish(&self) -> u64 {
        self.state
    }

    pub fn hex(&self) -> String {
        format!("{:016x}", self.state)
    }
}
