# ADR: factored finite source language over graph execution IR

Status: design decision for the isolated R3c term-compiler experiment. This
document does not authorize a card migration, learner run, renderer change, or
main-worktree integration.

## Decision

Keep the finite labeled transition graph as the executor-facing IR. Add a
factored source-language seam only when a capability requires intentional
objects, properties, relations, preconditions, or effects. The candidate
source language is a deliberately deterministic finite subset of RDDL
(Relational Dynamic Influence Diagram Language): finite object domains,
finite-valued state/non-fluent predicates and functions, finite ground actions,
deterministic conditional probability functions/effects, finite reward and
termination conditions, plus a compiler-declared finite public projection.
Random distributions, continuous variables, external functions, and concurrent
joint actions are outside the first subset.

The compiler should ground an accepted factored model into the existing graph
IR. The source model and its object/relation provenance remain available to
privileged audits and canonical identity; the learner receives only the
declared public projection. The graph IR remains the only runtime executor for
the initial implementation.

This follows the project boundary: the environment supplies persistent
entities and relations, while bodies supply interventions and observations
([`EMBODIED-PROCESS.md`](../../EMBODIED-PROCESS.md), lines 46–63). It also keeps
the common query algebra above the backend rather than adding a card-specific
relational evaluator (same document, lines 132–173).

## Extensional universality is not relational expressivity

For any fixed finite factored model, enumerate every legal valuation of its
fluents as an opaque state ID. For every grounded action, enumerate its result
as a transition row. This produces a finite graph, so the graph IR is
extensionally universal for deterministic finite worlds.

That encoding does not preserve intentional relational expressivity. Once
grounded, it no longer says that two states differ by swapping object names,
that an action has a precondition, that an effect updates the selected object,
or that a relation is invariant under a permutation. Those facts become
uninterpreted IDs and must be rediscovered from a potentially exponential
table. A graph-shape example therefore adds semantic expressivity only when it
introduces state/action-dependent transitions that a ring cannot express; a
tree or another non-ring shape by itself is coverage of the graph executor.

R3c already demonstrates this distinction. `StateSpaceTerm::Ring` uses
modular displacement, while `StateSpaceTerm::Graph` uses explicit
`(state, actuator)` rows and a self-loop default
([`crates/term-compiler/src/lib.rs`](../../crates/term-compiler/src/lib.rs),
lines 163–176 and 971–992). The five-state square-plus-spur witness is a
meaningful non-ring execution test. More trees, cycles, or degree sequences
are valuable coverage and metamorphic tests, but they are not the next
capability layer.

## Capability coverage

The project capability graph is a transfer hypothesis, not a list of compiler
features ([`EMBODIED-PROCESS.md`](../../EMBODIED-PROCESS.md), lines 231–279).
The matrix records what this isolated compiler can represent semantically, not
what a learner has acquired.

| Trunk / capabilities | Current R3c graph compiler | Deterministic finite RDDL source | First admissible evidence |
|---|---|---|---|
| T1 — M1 body/interface identification; M2 action-conditioned prediction; M3 affordance/reachability | Bounded/exact for deterministic finite terms: typed body support, explicit environment transitions, exhaustive trajectories and values. No stochastic/noisy plant. | Adds reusable object/action schemas while grounding to the same exact graph checks. Stochastic RDDL remains rejected. | Preserve Card 03 ring behavior, then pass the non-ring receipt and all-sequence checks. |
| T1 — M4 goal-conditioned inverse control; M6 robust partial-observed regulation | M4 is scalar norm/scoring control; M6 is unsupported because there is no noisy continuous dynamics or exogenous state evolution. | Finite relational goals and deterministic effects are representable; robust filtering and continuous/noisy regulation are not in this subset. | Separate goal/value checks from learner policy evidence; do not claim G1 regulation. |
| T2 — M5 active experimentation; M11b epistemic action selection | Calibration and ambiguity reduction can be audited through the shared query algebra, but the compiler does not implement a learner policy. | A probe action and its information-bearing effect can be factored and grounded. Policy selection remains apparatus/learner work. | Matched informative/non-informative probe with exact public ambiguity and value receipts. |
| T3 — M10 maintain/inhibit/switch; M12 irreversibility | Scalar norms, action/viability/resource restrictions, fallback, calibration, and restoration are representable. Rich temporal process state is not. | Preconditions/effects and finite termination can express local switching and irreversible facts. Scheduler semantics remain outside the subset. | Change one norm/precondition/effect while holding public history fixed; exact bounds must change only when intended. |
| T4 — P7 entity/relation continuity; P9 source/channel binding; M7 relational abstraction; M11a perceptual organization | Unsupported intentionally: states are opaque integers; there are no object sets, properties, relation fluents, public projections over sources, or identity transformations. | This is the first target layer: finite objects, relation fluents, deterministic effects, and explicit public projection. | Visible source reassignment witness below, with object/channel permutation and channel-identity controls. |
| T5 — M9 temporal composition | Partial: finite action sequences, calibration, restoration, and a frozen signal interrupt exist; no scheduler, continuation, restart, or segment state. | A bounded segment state can be encoded as fluents, but this does not supply the missing process scheduler automatically. | Keep as unsupported until an independent composition contract specifies interruption and resume semantics. |
| T6 — P8 other-agent contingency; M8 goal/means inference; H4 physical prompting | Unsupported: no independently acting agent, hidden demonstrated goal, or cross-body action trace. Disturbance/scaffold signals are not other agents. | Static relational scene facts are possible, but demonstrator policy, multiple bodies, and physical prompting require a later multi-agent contract. | Do not open T6 from graph/RDDL source validity; it remains gated on T1 and T4 transfer evidence. |

The current Card 03 contract already asks for a published finite configuration
graph, variable body support, calibration, and a body/environment swap
([`CARDS.md`](../../CARDS.md), lines 169–205). R3c now covers the deterministic
graph portion. Its graph backend rejects displacement coupling rather than
pretending that ring arithmetic is relational ([`crates/term-compiler/src/lib.rs`](../../crates/term-compiler/src/lib.rs), lines
633–646 and 840); the Card 03 coupling variant therefore remains outside this
graph compiler claim.

## Why trees are not the next semantic layer

A tree changes reachability, degree, and dead-end behavior, so it is a useful
non-ring witness and a good test of total self-loop defaults. It still treats
the complete configuration as one opaque state and every action as one opaque
label. Adding a second tree changes the extensional transition family but does
not create an object identity, a relation, or a reusable action schema.

Consequently:

1. retain trees/branching graphs as graph-IR coverage and non-ring validity
   fixtures;
2. do not call a larger topology distribution “relational generalization”;
3. introduce factored source semantics only for the first capability whose
   contrast cannot be stated without object identity or a relation; and
4. ground that source into the same graph executor so trajectory, value,
   public-boundary, hash, and replay audits remain shared.

## Smallest visible-reassignment witness

The first factored witness should implement the decomposed Card 06 relation,
not full occlusion or multi-agent behavior. The project describes this as
binding the history-named source across one public channel-change boundary
while both source values remain visible ([`CARDS.md`](../../CARDS.md), lines
438–447).

Use the smallest finite instance:

```text
objects:       source_a, source_b
channels:      channel_x, channel_y
values:        {0, 1, 2}
relation:      assigned_to(source, channel), a hidden bijection
state:         value(source_a), value(source_b), assigned_to(...), phase
action:        pulse(channel), which increments the source currently assigned
               to that channel, subject to a declared finite precondition
boundary:      a public assignment-change event; the new assignment is hidden
goal:          drive the source identified by interaction history to value 2
observation:   public channel values and boundary event, never source IDs or
               the hidden assignment
```

The two assignments are exchangeable initially. A learner must use the
intervention effect before the boundary to bind a persistent source, then
follow that source after the channel reassignment. Required controls are:

- channel-locked assignment, removing the relation change;
- permanent identity tags, removing the need for binding; and
- a channel-identity goal, changing the relation being tested.

The RDDL-subset sketch is intentionally small:

```text
objects source_a, source_b, channel_x, channel_y
fluent value(source) : {0,1,2}
fluent assigned_to(source, channel) : Boolean
action pulse(channel)
cpf value'(s) = increment(value(s))
  if pulse(channel) and assigned_to(s, channel), else value(s)
```

The assignment-change boundary and public projection are part of the
contract, not generator labels. Grounding produces an ordinary finite graph;
the factored source is retained so audits can test object permutation,
relation preservation, precondition/effect correctness, and the public/hidden
boundary directly.

## Admission and stop gates

Admission requires all of the following before this source model is considered
coherent:

1. The accepted RDDL subset is finite and deterministic; unsupported random,
   continuous, external, or concurrent constructs are rejected explicitly.
2. Grounding produces a total canonical graph with deterministic replay, and
   graph execution passes the existing generic trajectory/value/query tests.
3. The visible-reassignment witness changes the required policy, while each
   control removes exactly its named relational dependency.
4. Object and channel permutations preserve meaning; changing the assignment
   relation or history-named target changes the audited result.
5. Public traces contain only the declared channel values and boundary event;
   object IDs, assignment rows, source values not exposed by the projection,
   generator seed, and case labels remain privileged/generator data.
6. Exact finite receipts report totality, complete sequence coverage, public
   ambiguity before/after the boundary, values, optimal actions, topology,
   source hash, and replay coordinates separately.
7. The factored-source hash changes for semantic relation/effect mutations,
   while seed/index changes do not; the grounded graph hash is derived from the
   same canonical source semantics and provenance.

Stop and classify the result as graph coverage, rather than relational
progress, if object/relation syntax is erased before auditing, if the witness
can be solved from channel identity or a fixed sequence, if a control does not
remove exactly one dependency, or if a source construct is silently relaxed.
Stop the R3c experiment if grounding leaks privileged facts into public output,
replay is sampled rather than exact, or existing ring/Card 03 receipts change
without an explicit versioned decision.

## R3c boundary

This ADR maps only to the isolated R3c row in
[`DEVELOPMENT-PATH.md`](../../DEVELOPMENT-PATH.md), currently described as a
typed embodied-world compiler spike (`lines 179`). Its passing evidence would
be finite world-validity and capability-coverage evidence. It would not promote
the source language into the main cards, authorize learner training or GPU
work, establish T4 transfer, or reopen the project’s R10/R11 decisions.
