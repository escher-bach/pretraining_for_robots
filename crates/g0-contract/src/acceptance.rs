//! Minimal semantic admission for finite generated worlds.
//!
//! This is deliberately a thin adapter over the query algebra.  A generator
//! may sample a term, compile it, and ask this module whether it is worth
//! emitting.  It does not introduce a second solver, a card-specific receipt,
//! or numeric thresholds beyond the query algebra's exact-value tolerance.
//!
//! The compiler owns one term-specific fact that the common [`Fragment`]
//! interface cannot infer: whether the norm is met at the empty history.  The
//! caller supplies that boolean.  Hidden perturbations are supplied as pairs
//! of contracts whose public histories are meant to remain identical; every
//! other check is derived here by exhaustive enumeration.

use serde::{Deserialize, Serialize};

use crate::{
    ambiguity_report, epistemic_value, noninterference_check, optimal_first_actions, AmbiguitySet,
    PubliclyObservable, VALUE_EPSILON,
};

/// Evidence and decision from the finite semantic acceptance filter.
///
/// A report is intentionally made of observables rather than policy knobs.  A
/// generator can retain it alongside its seed/index without creating another
/// lifecycle or gate vocabulary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcceptanceReport {
    /// Whether the empty history already satisfies the generated norm.
    pub goal_holds_at_start: bool,
    /// Number of available actions and number attaining the exact first-step
    /// ceiling.  Equality means the decision is vacuous.
    pub action_count: usize,
    pub optimal_first_action_count: usize,
    /// Exact public/privileged values over the supplied ambiguity set.
    pub public_ceiling: f64,
    pub privileged_ceiling: f64,
    pub ambiguity_gap: f64,
    /// For a positive gap, an ambiguity-reducing first action must attain the
    /// best public-policy ceiling. The gap need not reach the privileged
    /// ceiling: information can cost an action (as in a 98-vs-99 probe).
    pub informative_action_attains_public_ceiling: bool,
    /// All supplied hidden perturbations remained non-interfering over every
    /// action sequence up to the fragment horizon.
    pub hidden_state_noninterference: bool,
    pub hidden_pairs_checked: usize,
    pub accepted: bool,
}

/// Run the minimal exact admission filter for one realized candidate.
///
/// `candidates` is the public ambiguity set, with the realized contract in
/// position zero as required by the shared query algebra.  `hidden_pairs` are
/// audit-only pairs of hidden realizations that are declared to be held under
/// the same public history.  If there is no hidden state, pass an empty slice.
/// `start_goal_met` is evaluated by the concrete compiler because only it knows
/// how its norm maps to a scored outcome.
pub fn assess<F: PubliclyObservable>(
    fragment: &F,
    candidates: &AmbiguitySet<F::Contract>,
    start_goal_met: bool,
    hidden_pairs: &[(F::Contract, F::Contract)],
    action_name: impl Fn(F::Action) -> String + Copy,
) -> AcceptanceReport
where
    F::Contract: Clone,
{
    let actions = fragment.actions();
    let optimal = optimal_first_actions(fragment, &candidates.candidates[0]);
    let report = ambiguity_report(fragment, candidates, fragment.horizon());

    let informative_action_attains_public_ceiling =
        epistemic_value(fragment, candidates, fragment.horizon(), action_name)
            .into_iter()
            .any(|action| {
                action.ambiguity_reduction > 0
                    && (action.public_value - report.public_ceiling).abs() <= VALUE_EPSILON
            });

    let hidden_state_noninterference = hidden_pairs.iter().all(|(left, right)| {
        noninterference_check(
            fragment,
            "generated hidden-state perturbation",
            left,
            right,
            fragment.horizon(),
            |_| true,
            action_name,
        )
        .holds
    });

    let gap_is_admissible =
        report.ambiguity_gap <= VALUE_EPSILON || informative_action_attains_public_ceiling;
    let accepted = !start_goal_met
        && optimal.len() < actions.len()
        && gap_is_admissible
        && hidden_state_noninterference;

    AcceptanceReport {
        goal_holds_at_start: start_goal_met,
        action_count: actions.len(),
        optimal_first_action_count: optimal.len(),
        public_ceiling: report.public_ceiling,
        privileged_ceiling: report.privileged_ceiling,
        ambiguity_gap: report.ambiguity_gap,
        informative_action_attains_public_ceiling,
        hidden_state_noninterference,
        hidden_pairs_checked: hidden_pairs.len(),
        accepted,
    }
}

/// Convenience predicate for generators that only need to reject candidates.
pub fn accepts<F: PubliclyObservable>(
    fragment: &F,
    candidates: &AmbiguitySet<F::Contract>,
    start_goal_met: bool,
    hidden_pairs: &[(F::Contract, F::Contract)],
    action_name: impl Fn(F::Action) -> String + Copy,
) -> bool
where
    F::Contract: Clone,
{
    assess(
        fragment,
        candidates,
        start_goal_met,
        hidden_pairs,
        action_name,
    )
    .accepted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{trajectory, Fragment};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Fixture;

    impl Fragment for Fixture {
        type Action = u8;
        type Contract = bool;

        fn actions(&self) -> Vec<Self::Action> {
            vec![0, 1]
        }

        fn horizon(&self) -> usize {
            1
        }

        fn start(&self, _: &Self::Contract) -> usize {
            0
        }

        fn step(&self, _: &Self::Contract, cell: usize, _: usize, action: Self::Action) -> usize {
            cell + usize::from(action)
        }

        fn value(
            &self,
            contract: &Self::Contract,
            trajectory: &[usize],
            _: &[Self::Action],
        ) -> i32 {
            let moved = trajectory.last().copied().unwrap_or_default() > 0;
            if moved == *contract {
                1
            } else {
                0
            }
        }
    }

    impl PubliclyObservable for Fixture {
        fn public_trace(&self, contract: &Self::Contract, actions: &[Self::Action]) -> Vec<i64> {
            // The boolean is hidden; actions produce no observations.
            let path = trajectory(self, contract, actions);
            vec![path.last().copied().unwrap_or_default() as i64]
        }
    }

    #[test]
    fn accepts_nontrivial_world_without_hidden_ambiguity() {
        let fixture = Fixture;
        let candidates = AmbiguitySet::uniform(vec![true]);
        let report = assess(&fixture, &candidates, false, &[], |action| {
            action.to_string()
        });
        assert!(report.accepted);
        assert_eq!(report.action_count, 2);
        assert_eq!(report.optimal_first_action_count, 1);
    }

    #[test]
    fn rejects_start_goal_and_vacuous_decisions() {
        let fixture = Fixture;
        let candidates = AmbiguitySet::uniform(vec![true]);
        assert!(
            !assess(&fixture, &candidates, true, &[], |action| action
                .to_string())
            .accepted
        );

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct Flat;
        impl Fragment for Flat {
            type Action = u8;
            type Contract = ();
            fn actions(&self) -> Vec<u8> {
                vec![0, 1]
            }
            fn horizon(&self) -> usize {
                1
            }
            fn start(&self, _: &()) -> usize {
                0
            }
            fn step(&self, _: &(), cell: usize, _: usize, _: u8) -> usize {
                cell
            }
            fn value(&self, _: &(), _: &[usize], _: &[u8]) -> i32 {
                0
            }
        }
        impl PubliclyObservable for Flat {
            fn public_trace(&self, _: &(), _: &[u8]) -> Vec<i64> {
                vec![0]
            }
        }
        let flat = Flat;
        let candidates = AmbiguitySet::uniform(vec![()]);
        assert!(!assess(&flat, &candidates, false, &[], |action| action.to_string()).accepted);
    }

    #[test]
    fn rejects_unclosed_ambiguity_and_publicly_interfering_pair() {
        let fixture = Fixture;
        // The two candidates have opposite optimal actions but no public
        // observation can identify them, so the public ceiling stays below the
        // privileged ceiling.
        let candidates = AmbiguitySet::uniform(vec![true, false]);
        let report = assess(&fixture, &candidates, false, &[], |action| {
            action.to_string()
        });
        assert!(report.ambiguity_gap > VALUE_EPSILON);
        assert!(!report.informative_action_attains_public_ceiling);
        assert!(!report.accepted);

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct Leaky;
        impl Fragment for Leaky {
            type Action = u8;
            type Contract = bool;
            fn actions(&self) -> Vec<u8> {
                vec![0]
            }
            fn horizon(&self) -> usize {
                1
            }
            fn start(&self, _: &bool) -> usize {
                0
            }
            fn step(&self, _: &bool, cell: usize, _: usize, _: u8) -> usize {
                cell
            }
            fn value(&self, _: &bool, _: &[usize], _: &[u8]) -> i32 {
                0
            }
        }
        impl PubliclyObservable for Leaky {
            fn public_trace(&self, contract: &bool, _: &[u8]) -> Vec<i64> {
                vec![i64::from(*contract)]
            }
        }
        let leaky = Leaky;
        let candidates = AmbiguitySet::uniform(vec![true]);
        let report = assess(&leaky, &candidates, false, &[(true, false)], |action| {
            action.to_string()
        });
        assert!(!report.hidden_state_noninterference);
        assert!(!report.accepted);
    }

    #[test]
    fn admits_an_informative_probe_with_an_irreducible_cost_gap() {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct Probe;
        impl Fragment for Probe {
            type Action = u8;
            type Contract = bool;

            fn actions(&self) -> Vec<u8> {
                vec![0, 1]
            }

            fn horizon(&self) -> usize {
                2
            }

            fn start(&self, _: &bool) -> usize {
                0
            }

            fn step(&self, _: &bool, cell: usize, _: usize, _: u8) -> usize {
                cell
            }

            fn value(&self, contract: &bool, _: &[usize], actions: &[u8]) -> i32 {
                match actions {
                    // Probe action 0 identifies the hidden side, but consumes
                    // one point of the attainable reward.
                    [0, second] => {
                        if (*second == 1) == *contract {
                            98
                        } else {
                            0
                        }
                    }
                    // A privileged solver can commit first and achieve 99;
                    // a public solver cannot choose the right second action
                    // without first probing.
                    [1, second] => {
                        if *second == u8::from(*contract) {
                            99
                        } else {
                            0
                        }
                    }
                    _ => 0,
                }
            }
        }

        impl PubliclyObservable for Probe {
            fn public_trace(&self, contract: &bool, actions: &[u8]) -> Vec<i64> {
                match actions.first().copied() {
                    // The probe reveals the hidden side.
                    Some(0) => vec![i64::from(*contract)],
                    // A commit gives no information before the next action.
                    _ => vec![0],
                }
            }
        }

        let probe = Probe;
        let candidates = AmbiguitySet::uniform(vec![true, false]);
        let report = assess(&probe, &candidates, false, &[], |action| action.to_string());
        assert_eq!(report.public_ceiling, 98.0);
        assert_eq!(report.privileged_ceiling, 99.0);
        assert_eq!(report.ambiguity_gap, 1.0);
        assert!(report.informative_action_attains_public_ceiling);
        assert!(report.accepted);
    }
}
