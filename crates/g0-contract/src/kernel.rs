//! The executable process kernel.
//!
//! `EMBODIED-PROCESS.md` fixes five operators and a three-connective norm
//! algebra as the smallest set needed to express all eight cards. Until this
//! module existed those constructs were labels in a coverage table: a card
//! could claim `restrict` while its crate open-coded an `if` on a private
//! field, and nothing checked that two cards claiming the same construct meant
//! the same thing. `DEVELOPMENT-PATH.md` states the rule directly — a row is
//! not complete when its process-algebra coverage exists only as a label.
//!
//! So the constructs live here as data, once, and each card composes them
//! rather than re-deriving them. Three consequences are deliberate:
//!
//! 1. a construct is **shared**: card 03's action restriction and card 04's
//!    viability restriction are the same `Restriction` type with different
//!    variants, so a cross-card claim about "restriction" is a claim about one
//!    object;
//! 2. a construct is **declared**: an interrupt must say what happens to the
//!    displaced process and how it resumes, because `EMBODIED-PROCESS.md` makes
//!    that distinction card 07's central control and a silent default would
//!    decide it; and
//! 3. a construct is **inert**: nothing here reads hidden state or emits an
//!    event. These types describe a composition; the cards interpret them.
//!
//! What is *not* here is a scheduler, a simulator, or an interpreter that
//! executes an arbitrary composition. A general process interpreter would have
//! to fix a state type, which is exactly the card ontology this layer refuses
//! to fix. The kernel supplies the vocabulary and the predicates; each card's
//! `Fragment` remains the executable semantics.

use serde::{Deserialize, Serialize};

/// A small dense set over `0..32`, which is every index space a G0 family has.
///
/// A `Vec<bool>` would carry a length that can silently disagree between two
/// cards; a fixed word cannot. Thirty-two is checked rather than assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub struct IndexSet(pub u32);

impl IndexSet {
    pub const EMPTY: Self = Self(0);
    pub const CAPACITY: usize = 32;

    pub fn from_indices(indices: impl IntoIterator<Item = usize>) -> Self {
        let mut set = Self::EMPTY;
        for index in indices {
            set.insert(index);
        }
        set
    }

    /// Every index below `count`.
    pub fn full(count: usize) -> Self {
        assert!(count <= Self::CAPACITY, "index set capacity is 32");
        if count == Self::CAPACITY {
            Self(u32::MAX)
        } else {
            Self((1u32 << count) - 1)
        }
    }

    pub fn insert(&mut self, index: usize) {
        assert!(index < Self::CAPACITY, "index set capacity is 32");
        self.0 |= 1u32 << index;
    }

    pub fn remove(&mut self, index: usize) {
        assert!(index < Self::CAPACITY, "index set capacity is 32");
        self.0 &= !(1u32 << index);
    }

    pub const fn contains(self, index: usize) -> bool {
        index < Self::CAPACITY && (self.0 >> index) & 1 == 1
    }

    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn iter(self) -> impl Iterator<Item = usize> {
        (0..Self::CAPACITY).filter(move |index| self.contains(*index))
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

/// When a guarded construct fires.
///
/// A guard is evaluated against public quantities only. There is deliberately
/// no `WhenHiddenStateIs` variant: a guard that reads privileged state would be
/// a path from the hidden view into public output, which the information
/// boundary forbids. A card that needs a hidden trigger publishes a reveal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Guard {
    /// Before the first action, so a plan-once policy can see it.
    AtStart,
    /// Immediately after `executed` actions have been taken.
    AfterStep(usize),
    /// When the named action index is executed.
    OnAction(u16),
    /// When the configuration enters the named cell.
    OnCellEntry(usize),
    /// Never fires. Present so a control can delete a construct without
    /// deleting the composition it belongs to, which keeps the paired arms
    /// structurally identical.
    Never,
}

/// The public facts a guard is allowed to consult.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuardContext {
    /// Number of actions already executed.
    pub executed: usize,
    /// The action index just executed, if any.
    pub last_action: Option<u16>,
    /// The cell just entered.
    pub cell: usize,
}

impl Guard {
    /// Whether this guard has fired by the time `context` describes.
    ///
    /// `AtStart` is true everywhere including before the first action, which is
    /// what makes an announced reveal visible to a policy that plans once.
    pub fn fired(self, context: GuardContext) -> bool {
        match self {
            Self::AtStart => true,
            Self::AfterStep(step) => context.executed > step,
            Self::OnAction(action) => context.last_action == Some(action),
            Self::OnCellEntry(cell) => context.cell == cell,
            Self::Never => false,
        }
    }

    /// Whether the guard is one a policy can evaluate before acting at all.
    pub fn is_visible_at_start(self) -> bool {
        matches!(self, Self::AtStart)
    }
}

// ---------------------------------------------------------------------------
// restrict_{kind,K}(P)
// ---------------------------------------------------------------------------

/// Whether a restricted resource is shared with other components or local.
///
/// `EMBODIED-PROCESS.md` requires resource restriction to state this, because a
/// shared budget and per-component budgets give different composite policies —
/// which is card 07's declared meaning-changing transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceScope {
    Shared,
    Local,
}

/// What happens when the configuration reaches the viability boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryEffect {
    /// The episode continues from the start configuration.
    Reset,
    /// The configuration is trapped; no later action changes it.
    Absorbing,
}

/// `restrict_{kind,K}(P)`.
///
/// The three kinds are not interchangeable and are not collapsed into one
/// predicate: an unsupported actuator is a body fact whose command is dropped,
/// an inadmissible cell is a norm fact that ends or resets the episode, and a
/// budget is a resource fact that ends it. Cards 03, 04, and 05 use one kind
/// each, and the audit reports which.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Restriction {
    /// Body support: only these action indices have their declared effect.
    Action { supported: IndexSet },
    /// Viability: these cells are inadmissible, with a declared boundary effect.
    Viability {
        inadmissible: IndexSet,
        effect: BoundaryEffect,
    },
    /// Budget: at most `budget` actions, shared or local.
    Resource { budget: usize, scope: ResourceScope },
}

impl Restriction {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Action { .. } => "action",
            Self::Viability { .. } => "viability",
            Self::Resource { .. } => "resource",
        }
    }

    /// Whether an action index retains its declared effect.
    ///
    /// A viability or resource restriction permits every action: it constrains
    /// where the process may go and how long it may run, not what the body can
    /// command. Conflating the two would make a body limit indistinguishable
    /// from a norm, which is the confound card 03 exists to avoid.
    pub fn permits_action(&self, action: u16) -> bool {
        match self {
            Self::Action { supported } => supported.contains(usize::from(action)),
            _ => true,
        }
    }

    pub fn admits_cell(&self, cell: usize) -> bool {
        match self {
            Self::Viability { inadmissible, .. } => !inadmissible.contains(cell),
            _ => true,
        }
    }

    pub fn boundary_effect(&self) -> Option<BoundaryEffect> {
        match self {
            Self::Viability { effect, .. } => Some(*effect),
            _ => None,
        }
    }

    pub fn remaining_budget(&self, spent: usize) -> Option<usize> {
        match self {
            Self::Resource { budget, .. } => Some(budget.saturating_sub(spent)),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// reveal_{C,g}(P)
// ---------------------------------------------------------------------------

/// `reveal_{C,g}(P)`: publish revealable content `C` when guard `g` fires.
///
/// The content is generic because what is revealed differs by card — restored
/// actuator support, a superseding goal, a gate value. What does not differ is
/// that the content becomes public exactly when the guard fires and is not
/// otherwise readable, which is the property the non-interference audit checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reveal<C> {
    pub guard: Guard,
    pub content: C,
}

impl<C: Copy> Reveal<C> {
    pub fn new(guard: Guard, content: C) -> Self {
        Self { guard, content }
    }

    /// The content if the guard has fired, and nothing otherwise.
    ///
    /// Returning `Option` rather than the content plus a flag is what makes a
    /// caller unable to read an unrevealed value by ignoring a boolean.
    pub fn published(&self, context: GuardContext) -> Option<C> {
        self.guard.fired(context).then_some(self.content)
    }
}

// ---------------------------------------------------------------------------
// interrupt_{g,resume}(P,Q)
// ---------------------------------------------------------------------------

/// What happens to the displaced process while the interrupting one runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Displaced {
    /// It keeps evolving unobserved. Card 06's occlusion witness.
    Continues,
    /// It is held. Card 06's frozen-during-absence control.
    Frozen,
}

/// How the displaced process continues once the interrupt ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Resume {
    /// Continue from the state it had reached. Card 07's central control.
    FromState,
    /// Start again from its initial state.
    Restart,
}

/// `interrupt_{g,resume}(P,Q)`.
///
/// Both fields are required. `EMBODIED-PROCESS.md` states that whether the
/// displaced process resumes from its current state or restarts is card 07's
/// central control, so a default would silently decide a contrast.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Interrupt {
    pub guard: Guard,
    pub displaced: Displaced,
    pub resume: Resume,
}

impl Interrupt {
    pub fn new(guard: Guard, displaced: Displaced, resume: Resume) -> Self {
        Self {
            guard,
            displaced,
            resume,
        }
    }

    pub fn active(&self, context: GuardContext) -> bool {
        self.guard.fired(context)
    }
}

// ---------------------------------------------------------------------------
// P ⊗_v[rule] Q
// ---------------------------------------------------------------------------

/// How two writers to one shared variable are resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CouplingRule {
    /// Contributions add.
    Sum,
    /// The selected writer wins and the other contributes nothing.
    Override,
    /// Simultaneous writes are a contract error rather than a silent choice.
    Conflict,
}

/// `P ⊗_v[rule] Q`: coupling through a declared shared variable.
///
/// The rule is part of the contract, not of the implementation. Card 06's
/// channels are `Override`-coupled to sources through a hidden assignment; a
/// `Sum` coupling would make the same wiring a different world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Coupling {
    pub variable: u16,
    pub rule: CouplingRule,
}

impl Coupling {
    pub fn new(variable: u16, rule: CouplingRule) -> Self {
        Self { variable, rule }
    }

    /// Resolve contributions under the declared rule.
    ///
    /// `Override` takes the last writer, which is why the card that uses it must
    /// declare a writer order; `Conflict` refuses two writers instead of picking.
    pub fn resolve(&self, contributions: &[f64]) -> Result<f64, String> {
        match self.rule {
            CouplingRule::Sum => Ok(contributions.iter().sum()),
            CouplingRule::Override => contributions
                .last()
                .copied()
                .ok_or_else(|| format!("variable {} has no writer", self.variable)),
            CouplingRule::Conflict => match contributions {
                [] => Err(format!("variable {} has no writer", self.variable)),
                [only] => Ok(*only),
                _ => Err(format!(
                    "variable {} has {} simultaneous writers under a conflict rule",
                    self.variable,
                    contributions.len()
                )),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Norm algebra
// ---------------------------------------------------------------------------

/// A norm over a public trajectory.
///
/// Norms compose separately from processes, with the three connectives
/// `EMBODIED-PROCESS.md` declares: conjunction, supersession at an event, and
/// priority where they conflict. The leaves are the two outcome conditions the
/// finite cards need.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Norm {
    /// The configuration must settle on this cell and stay there.
    Settle { cell: usize },
    /// This cell must have been visited at some point.
    Visit { cell: usize },
    /// This cell must never be entered.
    Avoid { cell: usize },
    /// `N1 ∧ N2`: both apply.
    Both(Box<Norm>, Box<Norm>),
    /// `N1 ⨟ N2`: `after` supersedes `before` once the guard fires.
    Supersede {
        before: Box<Norm>,
        after: Box<Norm>,
        guard: Guard,
    },
    /// `N1 ≻ N2`: `high` wins wherever the two conflict.
    Priority { high: Box<Norm>, low: Box<Norm> },
}

/// The verdict of one norm on one trajectory.
///
/// `Violated` is separate from `Unmet` because a prohibition breached and a
/// goal not reached are different failures and card 04 scores them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormVerdict {
    pub met: bool,
    pub violated_prohibition: bool,
    /// Steps before the configuration settled, when a settling norm is in force.
    pub settle_steps: Option<usize>,
}

impl NormVerdict {
    const MET: Self = Self {
        met: true,
        violated_prohibition: false,
        settle_steps: None,
    };

    fn unmet() -> Self {
        Self {
            met: false,
            violated_prohibition: false,
            settle_steps: None,
        }
    }

    fn violated() -> Self {
        Self {
            met: false,
            violated_prohibition: true,
            settle_steps: None,
        }
    }
}

impl Norm {
    pub fn both(left: Norm, right: Norm) -> Self {
        Self::Both(Box::new(left), Box::new(right))
    }

    pub fn supersede(before: Norm, after: Norm, guard: Guard) -> Self {
        Self::Supersede {
            before: Box::new(before),
            after: Box::new(after),
            guard,
        }
    }

    pub fn priority(high: Norm, low: Norm) -> Self {
        Self::Priority {
            high: Box::new(high),
            low: Box::new(low),
        }
    }

    /// Evaluate the norm against a complete trajectory.
    ///
    /// `trajectory[0]` is the starting cell, so a trajectory of `n` actions has
    /// `n + 1` entries. Supersession is evaluated at the end of the episode
    /// because that is when the norm in force is determined; a guard that has
    /// fired by then selects `after`.
    pub fn evaluate(&self, trajectory: &[usize], context: GuardContext) -> NormVerdict {
        match self {
            Self::Settle { cell } => {
                let settle = (0..trajectory.len())
                    .find(|index| trajectory[*index..].iter().all(|entry| entry == cell));
                match settle {
                    Some(steps) => NormVerdict {
                        met: true,
                        violated_prohibition: false,
                        settle_steps: Some(steps),
                    },
                    None => NormVerdict::unmet(),
                }
            }
            Self::Visit { cell } => {
                if trajectory.contains(cell) {
                    NormVerdict::MET
                } else {
                    NormVerdict::unmet()
                }
            }
            Self::Avoid { cell } => {
                // The starting cell is not an entry: an episode that begins on a
                // prohibited cell has not been driven there by any action.
                if trajectory[1..].contains(cell) {
                    NormVerdict::violated()
                } else {
                    NormVerdict::MET
                }
            }
            Self::Both(left, right) => {
                let a = left.evaluate(trajectory, context);
                let b = right.evaluate(trajectory, context);
                NormVerdict {
                    met: a.met && b.met,
                    violated_prohibition: a.violated_prohibition || b.violated_prohibition,
                    settle_steps: a.settle_steps.or(b.settle_steps),
                }
            }
            Self::Supersede {
                before,
                after,
                guard,
            } => {
                if guard.fired(context) {
                    after.evaluate(trajectory, context)
                } else {
                    before.evaluate(trajectory, context)
                }
            }
            Self::Priority { high, low } => {
                let strong = high.evaluate(trajectory, context);
                // Priority is not conjunction: where the high norm is violated
                // the low norm's verdict is discarded rather than combined, so a
                // prohibition breach cannot be offset by reaching the goal.
                if strong.violated_prohibition || !strong.met {
                    return strong;
                }
                let weak = low.evaluate(trajectory, context);
                NormVerdict {
                    met: weak.met,
                    violated_prohibition: weak.violated_prohibition,
                    settle_steps: weak.settle_steps,
                }
            }
        }
    }

    /// The connectives this norm uses, for the coverage report.
    pub fn connectives(&self) -> Vec<&'static str> {
        let mut found = Vec::new();
        self.collect_connectives(&mut found);
        found.sort_unstable();
        found.dedup();
        found
    }

    fn collect_connectives(&self, found: &mut Vec<&'static str>) {
        match self {
            Self::Settle { .. } | Self::Visit { .. } | Self::Avoid { .. } => {}
            Self::Both(left, right) => {
                found.push("conjunction");
                left.collect_connectives(found);
                right.collect_connectives(found);
            }
            Self::Supersede { before, after, .. } => {
                found.push("supersession");
                before.collect_connectives(found);
                after.collect_connectives(found);
            }
            Self::Priority { high, low } => {
                found.push("priority");
                high.collect_connectives(found);
                low.collect_connectives(found);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Coverage reporting
// ---------------------------------------------------------------------------

/// One card's declared use of the kernel.
///
/// The point of recording this is that `EMBODIED-PROCESS.md`'s coverage table
/// can be checked against what the code composes, instead of being a claim a
/// reader has to trust.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelUse {
    pub directed_wiring: bool,
    pub shared_coupling: bool,
    pub interrupt: bool,
    pub restrict: bool,
    pub reveal: bool,
    pub norm_algebra: bool,
}

impl KernelUse {
    pub const NONE: Self = Self {
        directed_wiring: false,
        shared_coupling: false,
        interrupt: false,
        restrict: false,
        reveal: false,
        norm_algebra: false,
    };

    /// The row `EMBODIED-PROCESS.md` states for a card.
    pub fn declared(card: &str) -> Option<Self> {
        let row = |coupling, interrupt, restrict, reveal, norm| Self {
            directed_wiring: true,
            shared_coupling: coupling,
            interrupt,
            restrict,
            reveal,
            norm_algebra: norm,
        };
        Some(match card {
            "01" => row(true, false, true, false, false),
            "02" => row(false, true, false, false, false),
            "03" => row(false, false, true, true, false),
            "04" => row(false, true, true, true, true),
            "05" => row(false, false, true, true, false),
            "06" => row(true, true, false, false, false),
            "07" => row(false, true, true, true, true),
            "08" => row(true, true, false, true, false),
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(executed: usize, cell: usize) -> GuardContext {
        GuardContext {
            executed,
            last_action: None,
            cell,
        }
    }

    #[test]
    fn index_set_round_trips() {
        let set = IndexSet::from_indices([0, 3, 31]);
        assert!(set.contains(0) && set.contains(3) && set.contains(31));
        assert!(!set.contains(1));
        assert_eq!(set.len(), 3);
        assert_eq!(set.iter().collect::<Vec<_>>(), vec![0, 3, 31]);
        assert_eq!(IndexSet::full(4), IndexSet::from_indices([0, 1, 2, 3]));
        assert_eq!(IndexSet::full(32).len(), 32);
    }

    #[test]
    fn a_start_guard_is_visible_before_acting_and_a_step_guard_is_not() {
        assert!(Guard::AtStart.fired(context(0, 0)));
        assert!(!Guard::AfterStep(0).fired(context(0, 0)));
        assert!(Guard::AfterStep(0).fired(context(1, 0)));
        assert!(!Guard::Never.fired(context(9, 0)));
        assert!(Guard::AtStart.is_visible_at_start());
        assert!(!Guard::AfterStep(0).is_visible_at_start());
    }

    #[test]
    fn a_restriction_constrains_only_its_own_kind() {
        let body = Restriction::Action {
            supported: IndexSet::from_indices([0, 2]),
        };
        assert!(body.permits_action(0) && !body.permits_action(1));
        assert!(body.admits_cell(7), "a body limit is not a norm");
        assert!(body.remaining_budget(0).is_none());

        let viability = Restriction::Viability {
            inadmissible: IndexSet::from_indices([3]),
            effect: BoundaryEffect::Absorbing,
        };
        assert!(viability.permits_action(1), "a norm is not a body limit");
        assert!(!viability.admits_cell(3) && viability.admits_cell(2));
        assert_eq!(viability.boundary_effect(), Some(BoundaryEffect::Absorbing));

        let resource = Restriction::Resource {
            budget: 3,
            scope: ResourceScope::Shared,
        };
        assert_eq!(resource.remaining_budget(1), Some(2));
        assert_eq!(resource.remaining_budget(9), Some(0));
    }

    #[test]
    fn an_unfired_reveal_yields_nothing_at_all() {
        let reveal = Reveal::new(Guard::AfterStep(1), 7u16);
        assert_eq!(reveal.published(context(1, 0)), None);
        assert_eq!(reveal.published(context(2, 0)), Some(7));
        assert_eq!(
            Reveal::new(Guard::Never, 7u16).published(context(9, 0)),
            None
        );
    }

    #[test]
    fn coupling_rules_differ_and_conflict_refuses_rather_than_picks() {
        let sum = Coupling::new(0, CouplingRule::Sum);
        let over = Coupling::new(0, CouplingRule::Override);
        let conflict = Coupling::new(0, CouplingRule::Conflict);
        assert_eq!(sum.resolve(&[1.0, 2.0]), Ok(3.0));
        assert_eq!(over.resolve(&[1.0, 2.0]), Ok(2.0));
        assert!(conflict.resolve(&[1.0, 2.0]).is_err());
        assert_eq!(conflict.resolve(&[1.0]), Ok(1.0));
        assert!(sum.resolve(&[]).is_ok());
        assert!(over.resolve(&[]).is_err());
    }

    #[test]
    fn priority_is_not_conjunction() {
        // Reaching the goal cannot offset entering the prohibited cell.
        let norm = Norm::priority(Norm::Avoid { cell: 1 }, Norm::Settle { cell: 1 });
        let verdict = norm.evaluate(&[0, 1, 1], context(2, 1));
        assert!(!verdict.met && verdict.violated_prohibition);

        let conjunction = Norm::both(Norm::Avoid { cell: 1 }, Norm::Settle { cell: 1 });
        let both = conjunction.evaluate(&[0, 1, 1], context(2, 1));
        assert!(!both.met && both.violated_prohibition);
        assert_eq!(both.settle_steps, Some(1));
        assert_eq!(
            verdict.settle_steps, None,
            "priority discards the low norm's verdict rather than merging it"
        );
    }

    #[test]
    fn supersession_selects_by_guard_and_conjunction_does_not() {
        let superseded = Norm::supersede(
            Norm::Settle { cell: 2 },
            Norm::Settle { cell: 3 },
            Guard::AfterStep(0),
        );
        assert!(superseded.evaluate(&[0, 4, 3, 3], context(3, 3)).met);
        assert!(!superseded.evaluate(&[0, 1, 2, 2], context(3, 2)).met);

        let composed = Norm::both(Norm::Visit { cell: 2 }, Norm::Settle { cell: 3 });
        assert!(composed.evaluate(&[0, 1, 2, 3], context(3, 3)).met);
        assert!(!composed.evaluate(&[0, 4, 3, 3], context(3, 3)).met);
    }

    #[test]
    fn a_prohibited_starting_cell_is_not_an_entry() {
        let norm = Norm::Avoid { cell: 0 };
        assert!(norm.evaluate(&[0, 1, 2], context(2, 2)).met);
        assert!(!norm.evaluate(&[1, 0, 2], context(2, 2)).met);
    }

    #[test]
    fn connectives_are_reported_from_the_composition() {
        let norm = Norm::supersede(
            Norm::priority(Norm::Avoid { cell: 1 }, Norm::Settle { cell: 2 }),
            Norm::both(Norm::Visit { cell: 2 }, Norm::Settle { cell: 3 }),
            Guard::AfterStep(0),
        );
        assert_eq!(
            norm.connectives(),
            vec!["conjunction", "priority", "supersession"]
        );
        assert!(Norm::Settle { cell: 0 }.connectives().is_empty());
    }

    #[test]
    fn the_declared_coverage_table_is_available_for_every_card() {
        for card in ["01", "02", "03", "04", "05", "06", "07", "08"] {
            assert!(KernelUse::declared(card).is_some(), "card {card}");
        }
        assert!(KernelUse::declared("09").is_none());
        let four = KernelUse::declared("04").expect("card 04");
        assert!(four.norm_algebra && four.interrupt && four.restrict && four.reveal);
        assert!(!four.shared_coupling);
    }
}
