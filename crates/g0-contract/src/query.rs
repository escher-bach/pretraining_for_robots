//! The shared query algebra for finite families with hidden state.
//!
//! Card 04 needed none of this: every one of its ports is public, so its public
//! and privileged ceilings coincide and `identify` is trivial. Cards 02, 05,
//! and 06 are the opposite — their whole content is what the learner cannot yet
//! distinguish — and `EMBODIED-PROCESS.md` requires that the quantities they
//! report be *derived* from one shared interface rather than added as per-card
//! APIs.
//!
//! The single object everything here is built from is the **ambiguity set**: a
//! list of candidate contracts that a card asserts are publicly
//! indistinguishable, together with a prior over them. From it:
//!
//! | Quantity | Derivation |
//! |---|---|
//! | Identification observable | [`identify`] returns the surviving candidates; its size is the diameter. |
//! | Public ceiling | [`public_policy_value`] — the best policy measurable in the public trace. |
//! | Privileged ceiling | [`privileged_value_bound`] — each candidate solved knowing which it is. |
//! | Ambiguity gap | Their difference. |
//! | Epistemic value | [`epistemic_value`] — what committing to a first action does to the public ceiling and to the surviving set. |
//! | Matched non-informative action | [`matched_control_verdict`] — equal cost and value movement, no reduction in the surviving set. |
//!
//! The public ceiling is a **policy** value, not the value of the best fixed
//! action sequence. That distinction is the whole point: a learner that can
//! probe adapts inside the episode, so a fixed-sequence bound would understate
//! the public ceiling and manufacture an ambiguity gap that active
//! experimentation had already closed. The recursion below therefore branches
//! on the public observation after every action, which is exactly what a
//! learner can condition on and nothing more.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{sequences_of_length, trajectory, Fragment};

/// A family whose learner-visible trace can be extracted exactly.
///
/// `public_trace` must return everything the learner sees after executing
/// `actions`, and nothing else. Two contracts are publicly indistinguishable
/// under a prefix exactly when their traces agree on it — so a card that
/// accidentally includes a hidden field here converts its own ambiguity into
/// zero, and the non-interference check below is what catches that.
pub trait PubliclyObservable: Fragment {
    fn public_trace(&self, contract: &Self::Contract, actions: &[Self::Action]) -> Vec<i64>;
}

/// A set of contracts a card asserts are publicly indistinguishable at episode
/// start, with a prior over them.
#[derive(Debug, Clone)]
pub struct AmbiguitySet<C> {
    pub candidates: Vec<C>,
    pub prior: Vec<f64>,
}

impl<C: Clone> AmbiguitySet<C> {
    /// A uniform prior over the candidates.
    pub fn uniform(candidates: Vec<C>) -> Self {
        assert!(
            !candidates.is_empty(),
            "an ambiguity set needs at least one candidate"
        );
        let weight = 1.0 / candidates.len() as f64;
        let prior = vec![weight; candidates.len()];
        Self { candidates, prior }
    }

    pub fn new(candidates: Vec<C>, prior: Vec<f64>) -> Result<Self, String> {
        if candidates.len() != prior.len() {
            return Err("an ambiguity set needs one prior weight per candidate".into());
        }
        if candidates.is_empty() {
            return Err("an ambiguity set needs at least one candidate".into());
        }
        let total: f64 = prior.iter().sum();
        if !(total - 1.0).abs().lt(&1e-9) {
            return Err(format!("the prior sums to {total}, not 1"));
        }
        if prior.iter().any(|weight| *weight < 0.0) {
            return Err("a prior weight cannot be negative".into());
        }
        Ok(Self { candidates, prior })
    }

    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }
}

/// `identify(contract, public, history)`: the candidates still compatible with
/// what a learner has seen after `actions`.
///
/// Indices into the ambiguity set are returned rather than contracts, because
/// the caller needs to carry the prior along and a cloned contract would lose
/// which candidate it was.
pub fn identify<F: PubliclyObservable>(
    fragment: &F,
    set: &AmbiguitySet<F::Contract>,
    actions: &[F::Action],
) -> Vec<usize>
where
    F::Contract: Clone,
{
    let mut survivors = Vec::new();
    let reference = fragment.public_trace(&set.candidates[0], actions);
    let mut traces: BTreeMap<Vec<i64>, Vec<usize>> = BTreeMap::new();
    traces.insert(reference, vec![0]);
    for index in 1..set.candidates.len() {
        let trace = fragment.public_trace(&set.candidates[index], actions);
        traces.entry(trace).or_default().push(index);
    }
    // A learner holding this history is inside exactly one trace class. The
    // observable is the size of the class it is in, so the reported diameter is
    // the class containing candidate 0 — the card's declared realized instance.
    for group in traces.values() {
        if group.contains(&0) {
            survivors.clone_from(group);
        }
    }
    survivors
}

/// The size of the surviving set: the identification observable.
pub fn identification_diameter<F: PubliclyObservable>(
    fragment: &F,
    set: &AmbiguitySet<F::Contract>,
    actions: &[F::Action],
) -> usize
where
    F::Contract: Clone,
{
    identify(fragment, set, actions).len()
}

/// The exact value of the best policy measurable in the public trace.
///
/// The recursion carries a weighted belief over candidates and alternates
/// **observe, then act**, which is the order a learner actually experiences.
/// Partitioning only *after* each action would force the first move to be
/// common to every candidate, and that silently understates the ceiling for any
/// family whose scaffold speaks before the first decision — card 03's
/// calibration is exactly such a scaffold, and it made a family with a
/// provably-zero information gap report a positive one.
///
/// At the horizon each surviving candidate contributes its own realized value
/// under the actions actually taken.
pub fn public_policy_value<F: PubliclyObservable>(
    fragment: &F,
    set: &AmbiguitySet<F::Contract>,
    horizon: usize,
) -> f64
where
    F::Contract: Clone,
{
    let belief: Vec<(usize, f64)> = set
        .prior
        .iter()
        .enumerate()
        .filter(|(_, weight)| **weight > 0.0)
        .map(|(index, weight)| (index, *weight))
        .collect();
    observe_then_act(fragment, set, &belief, &mut Vec::new(), horizon).0
}

/// The public ceiling together with the first action attaining it.
pub fn public_policy_value_and_first_actions<F: PubliclyObservable>(
    fragment: &F,
    set: &AmbiguitySet<F::Contract>,
    horizon: usize,
) -> (f64, Vec<F::Action>)
where
    F::Contract: Clone,
{
    let belief: Vec<(usize, f64)> = set
        .prior
        .iter()
        .enumerate()
        .filter(|(_, weight)| **weight > 0.0)
        .map(|(index, weight)| (index, *weight))
        .collect();
    observe_then_act(fragment, set, &belief, &mut Vec::new(), horizon)
}

/// Tolerance for comparing two exactly-computed rational-valued ceilings.
///
/// Values are small integers divided by candidate counts, so genuinely distinct
/// values differ by far more than this; it exists only to absorb the last bit
/// of a repeated float sum.
pub const VALUE_EPSILON: f64 = 1e-9;

/// Split a belief by what the learner can currently see.
fn partition<F: PubliclyObservable>(
    fragment: &F,
    set: &AmbiguitySet<F::Contract>,
    belief: &[(usize, f64)],
    actions: &[F::Action],
) -> Vec<Vec<(usize, f64)>> {
    let mut classes: BTreeMap<Vec<i64>, Vec<(usize, f64)>> = BTreeMap::new();
    for (index, weight) in belief {
        let trace = fragment.public_trace(&set.candidates[*index], actions);
        classes.entry(trace).or_default().push((*index, *weight));
    }
    classes.into_values().collect()
}

/// Observe first, then act. The returned actions are those optimal in the
/// belief class the *realized* candidate is in, which is the class containing
/// candidate zero.
fn observe_then_act<F: PubliclyObservable>(
    fragment: &F,
    set: &AmbiguitySet<F::Contract>,
    belief: &[(usize, f64)],
    actions: &mut Vec<F::Action>,
    remaining: usize,
) -> (f64, Vec<F::Action>)
where
    F::Contract: Clone,
{
    let mut total = 0.0;
    let mut realized = Vec::new();
    for class in partition(fragment, set, belief, actions) {
        let holds_realized = class.iter().any(|(index, _)| *index == 0);
        let (value, best) = act_then_observe(fragment, set, &class, actions, remaining);
        total += value;
        if holds_realized {
            realized = best;
        }
    }
    (total, realized)
}

fn act_then_observe<F: PubliclyObservable>(
    fragment: &F,
    set: &AmbiguitySet<F::Contract>,
    belief: &[(usize, f64)],
    actions: &mut Vec<F::Action>,
    remaining: usize,
) -> (f64, Vec<F::Action>)
where
    F::Contract: Clone,
{
    if remaining == 0 {
        let value = belief
            .iter()
            .map(|(index, weight)| {
                let contract = &set.candidates[*index];
                let path = trajectory(fragment, contract, actions);
                weight * f64::from(fragment.value(contract, &path, actions))
            })
            .sum();
        return (value, Vec::new());
    }

    let mut best = f64::NEG_INFINITY;
    let mut best_actions = Vec::new();
    for action in fragment.actions() {
        actions.push(action);
        let value = observe_then_act(fragment, set, belief, actions, remaining - 1).0;
        actions.pop();

        if value > best + VALUE_EPSILON {
            best = value;
            best_actions.clear();
        }
        if (value - best).abs() <= VALUE_EPSILON {
            best_actions.push(action);
        }
    }
    best_actions.sort();
    best_actions.dedup();
    (best, best_actions)
}

/// The privileged ceiling: each candidate solved by a solver that knows which
/// one it is, then averaged under the prior.
pub fn privileged_value_bound<F: Fragment>(
    fragment: &F,
    set: &AmbiguitySet<F::Contract>,
    horizon: usize,
) -> f64
where
    F::Contract: Clone,
{
    set.candidates
        .iter()
        .zip(&set.prior)
        .map(|(contract, weight)| {
            let mut best = i32::MIN;
            for sequence in sequences_of_length(&fragment.actions(), horizon) {
                let path = trajectory(fragment, contract, &sequence);
                best = best.max(fragment.privileged_value(contract, &path, &sequence));
            }
            weight * f64::from(best)
        })
        .sum()
}

/// The public and privileged ceilings and their gap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AmbiguityReport {
    pub candidates: usize,
    pub initial_diameter: usize,
    pub public_ceiling: f64,
    pub privileged_ceiling: f64,
    pub ambiguity_gap: f64,
}

pub fn ambiguity_report<F: PubliclyObservable>(
    fragment: &F,
    set: &AmbiguitySet<F::Contract>,
    horizon: usize,
) -> AmbiguityReport
where
    F::Contract: Clone,
{
    let public = public_policy_value(fragment, set, horizon);
    let privileged = privileged_value_bound(fragment, set, horizon);
    AmbiguityReport {
        candidates: set.len(),
        initial_diameter: identification_diameter(fragment, set, &[]),
        public_ceiling: public,
        privileged_ceiling: privileged,
        ambiguity_gap: privileged - public,
    }
}

/// What one first action is worth, and what it does to the surviving set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionValue {
    pub action: String,
    /// The public ceiling reachable when this action is taken first.
    pub public_value: f64,
    /// The identification diameter immediately after taking it.
    pub diameter_after: usize,
    /// Ambiguity removed relative to taking no action.
    pub ambiguity_reduction: usize,
}

/// `epistemic_value`: the expected reduction in relevant ambiguity weighted by
/// the resulting change in value bounds, computed per first action.
///
/// A card reads two things from this: which actions reduce the surviving set,
/// and whether that reduction is worth its cost. Both are needed — an action
/// that reveals something no later decision depends on has a reduction and no
/// value, which is the `M5 → M11b` dispute in `EMBODIED-PROCESS.md`.
pub fn epistemic_value<F: PubliclyObservable>(
    fragment: &F,
    set: &AmbiguitySet<F::Contract>,
    horizon: usize,
    name: impl Fn(F::Action) -> String,
) -> Vec<ActionValue>
where
    F::Contract: Clone,
{
    let before = identification_diameter(fragment, set, &[]);
    let belief: Vec<(usize, f64)> = set
        .prior
        .iter()
        .enumerate()
        .filter(|(_, weight)| **weight > 0.0)
        .map(|(index, weight)| (index, *weight))
        .collect();

    let mut report = Vec::new();
    for action in fragment.actions() {
        let mut actions = vec![action];
        let mut classes: BTreeMap<Vec<i64>, Vec<(usize, f64)>> = BTreeMap::new();
        for (index, weight) in &belief {
            let trace = fragment.public_trace(&set.candidates[*index], &actions);
            classes.entry(trace).or_default().push((*index, *weight));
        }
        let mut value = 0.0;
        for class in classes.values() {
            value += observe_then_act(
                fragment,
                set,
                class,
                &mut actions,
                horizon.saturating_sub(1),
            )
            .0;
        }
        let after = identification_diameter(fragment, set, &actions);
        report.push(ActionValue {
            action: name(action),
            public_value: value,
            diameter_after: after,
            ambiguity_reduction: before.saturating_sub(after),
        });
    }
    report
}

/// Whether a declared control action really is the matched non-informative one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchedControlVerdict {
    pub informative_action: String,
    pub control_action: String,
    /// Both consume the same amount of the budget.
    pub equal_cost: bool,
    /// Neither moves the outcome by itself.
    pub equal_immediate_value_movement: bool,
    /// The informative action shrinks the surviving set and the control does not.
    pub only_the_informative_action_reduces_ambiguity: bool,
    pub holds: bool,
}

/// Check the matched non-informative control an epistemic card must supply.
///
/// `immediate_movement` is the card's own measure of how far one action moves
/// the outcome before any later decision — the quantity that has to be equal for
/// the pair to isolate information from progress.
pub fn matched_control_verdict<F: PubliclyObservable>(
    fragment: &F,
    set: &AmbiguitySet<F::Contract>,
    informative: F::Action,
    control: F::Action,
    cost: impl Fn(F::Action) -> usize,
    immediate_movement: impl Fn(F::Action) -> i32,
    name: impl Fn(F::Action) -> String,
) -> MatchedControlVerdict
where
    F::Contract: Clone,
{
    let before = identification_diameter(fragment, set, &[]);
    let after_informative = identification_diameter(fragment, set, &[informative]);
    let after_control = identification_diameter(fragment, set, &[control]);
    let equal_cost = cost(informative) == cost(control);
    let equal_movement = immediate_movement(informative) == immediate_movement(control);
    let only_informative = after_informative < before && after_control == before;
    MatchedControlVerdict {
        informative_action: name(informative),
        control_action: name(control),
        equal_cost,
        equal_immediate_value_movement: equal_movement,
        only_the_informative_action_reduces_ambiguity: only_informative,
        holds: equal_cost && equal_movement && only_informative,
    }
}

/// The result of holding public history fixed and perturbing hidden state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonInterference {
    pub property: String,
    pub sequences_checked: usize,
    pub holds: bool,
    /// The first action sequence whose public trace separated the two hidden
    /// realizations, when one exists.
    pub separating_sequence: Option<Vec<String>>,
}

/// `noninterference_check(contract, property)` restricted to admissible
/// sequences.
///
/// Card 02 needs it over the aliasing interval, card 05 over every sequence
/// that has not yet probed, and card 06 over the occluded window. The
/// restriction is the card's, because "which histories should not separate
/// these" is a semantic claim; the checking is shared.
pub fn noninterference_check<F: PubliclyObservable>(
    fragment: &F,
    property: &str,
    left: &F::Contract,
    right: &F::Contract,
    horizon: usize,
    admissible: impl Fn(&[F::Action]) -> bool,
    name: impl Fn(F::Action) -> String,
) -> NonInterference {
    let mut checked = 0usize;
    for length in 0..=horizon {
        for sequence in sequences_of_length(&fragment.actions(), length) {
            if !admissible(&sequence) {
                continue;
            }
            checked += 1;
            if fragment.public_trace(left, &sequence) != fragment.public_trace(right, &sequence) {
                return NonInterference {
                    property: property.to_string(),
                    sequences_checked: checked,
                    holds: false,
                    separating_sequence: Some(
                        sequence.iter().map(|action| name(*action)).collect(),
                    ),
                };
            }
        }
    }
    NonInterference {
        property: property.to_string(),
        sequences_checked: checked,
        holds: true,
        separating_sequence: None,
    }
}

/// The ceiling of the best policy that ignores a declared part of its history.
///
/// Card 02 needs "the public ceiling after ablating the latch". Ablation is
/// expressed as a coarsening of the public trace, so the same recursion serves
/// it: the policy still adapts, it just cannot see what the coarsening removed.
pub fn ablated_policy_value<F: PubliclyObservable>(
    fragment: &F,
    set: &AmbiguitySet<F::Contract>,
    horizon: usize,
    coarsen: impl Fn(&[i64]) -> Vec<i64> + Copy,
) -> f64
where
    F::Contract: Clone,
{
    struct Ablated<'a, F, G> {
        inner: &'a F,
        coarsen: G,
    }
    impl<F: Fragment, G> Fragment for Ablated<'_, F, G> {
        type Action = F::Action;
        type Contract = F::Contract;
        fn actions(&self) -> Vec<Self::Action> {
            self.inner.actions()
        }
        fn horizon(&self) -> usize {
            self.inner.horizon()
        }
        fn start(&self, contract: &Self::Contract) -> usize {
            self.inner.start(contract)
        }
        fn step(
            &self,
            contract: &Self::Contract,
            cell: usize,
            executed: usize,
            action: Self::Action,
        ) -> usize {
            self.inner.step(contract, cell, executed, action)
        }
        fn value(
            &self,
            contract: &Self::Contract,
            path: &[usize],
            actions: &[Self::Action],
        ) -> i32 {
            self.inner.value(contract, path, actions)
        }
        fn privileged_value(
            &self,
            contract: &Self::Contract,
            path: &[usize],
            actions: &[Self::Action],
        ) -> i32 {
            self.inner.privileged_value(contract, path, actions)
        }
    }
    impl<F: PubliclyObservable, G: Fn(&[i64]) -> Vec<i64>> PubliclyObservable for Ablated<'_, F, G> {
        fn public_trace(&self, contract: &Self::Contract, actions: &[Self::Action]) -> Vec<i64> {
            (self.coarsen)(&self.inner.public_trace(contract, actions))
        }
    }

    let ablated = Ablated {
        inner: fragment,
        coarsen,
    };
    public_policy_value(&ablated, set, horizon)
}
