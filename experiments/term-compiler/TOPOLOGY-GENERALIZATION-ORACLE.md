# Topology-independent finite environment oracle

This document defines the acceptance gates for extending the isolated term
compiler beyond ring configuration spaces. It is a design oracle, not an
instruction to change the production workspace. A backend is admitted only if
it passes these gates while preserving the existing ring/Card 03 behavior.

## 1. Semantic boundary

Topology belongs to the environment's configuration space, not to morphology.
Morphology describes what the body can do: typed action ports, actuator
interventions, support, delays, gains, and saturation. The environment decides
how a valid intervention changes its configuration.

The finite backend therefore needs this separation:

```text
learner command
  -> typed body intervention (possibly unsupported)
  -> environment transition(state, intervention)
  -> next configuration and public sensor event
```

The body MUST NOT encode graph adjacency as a morphology field. An environment
MUST NOT silently reinterpret an unsupported body action as an absent edge.
Those two causes may produce the same trajectory, but they remain distinct typed
provenance and distinct canonical hash input.

## 2. Total finite transition system

The topology-independent environment is a finite configuration space plus a
total transition relation:

```text
ConfigurationSpace = (states, transition_table, default_transition)
transition : State × Intervention -> State
```

Every state and intervention pair MUST have a deterministic result. The
preferred sparse encoding is an explicit table of present transitions plus an
explicit default. For this backend the default MUST be a self-loop:

```text
if edge(state, intervention) exists: use its target
otherwise:                             next_state = state
```

An explicit self-loop and an absent edge are behaviorally equal but need not be
provenance-equal; if a world cares about that distinction it MUST retain it in
the term and hash. Unknown state IDs, unknown intervention IDs, duplicate
transition keys, and targets outside the state set MUST be rejected before
lowering. Stochastic or nondeterministic rows are outside this gate.

Norm evaluation, scoring, fallback absorption, public traces, and absolute
step-clock guards operate over state IDs exactly as the ring backend operates
over cell IDs. A fallback remains an action outcome, not an extra topology node.

## 3. Non-ring witness

The first non-ring conformance fixture MUST use five states `0..4` with the
underlying undirected topology:

```text
0 -- 1 -- 2 -- 3 -- 0
     |
     4
```

Its degree sequence is `{1,2,2,2,3}`, so it cannot be isomorphic to a simple
cycle, whose every node has degree two. One valid labeled transition table is:

| State | `advance` | `retreat` | `hold` |
|---:|---:|---:|---:|
| 0 | 1 | 3 | 0 |
| 1 | 2 | 4 | 1 |
| 2 | 3 | 1 | 2 |
| 3 | 0 | 2 | 3 |
| 4 | 4 | 1 | 4 |

The table is total; the `hold` column is explicit here for readability. The
fixture's body actions are `advance`, `retreat`, `hold`, and `fallback`, with
`fallback` absorbing scoring/public execution. Start at `0`, goal at `4`, and
use a two-step scored horizon.

The body-limited witness supports `{advance, hold, fallback}`. The unrestricted
control supports all movement actions, so it can reach `4` as
`advance(0 -> 1), retreat(1 -> 4)`. The witness cannot reach `4` in two steps.
The environment twin has full body support but deletes the `retreat` transition
only at the witness's scored command sites. Its scored behavior must match the
witness while its calibration must be publicly distinguishable.

This fixture is admissible only if the backend reports its graph as non-cycle by
an executable topology check, not merely by trusting a fixture label.

## 4. Body/environment twin construction

For a body-limited term with support `S` and a full body with support `F`, let
`W = F - S` be the withheld movement interventions. Enumerate all complete
scored action sequences of the witness. Define `Q_exact` as the set of states at
which an action is issued before the final scored action (the actual command
sites), across the complete finite sequence set.
The environment twin MUST:

1. use the same states, start, goal, action alphabet, scoring, and public
   calibration;
2. use full body support `F`;
3. replace each withheld intervention's row at every `q ∈ Q_exact` (or at every
   `q ∈ Q_conservative` when that declared scope is selected) with the default
   self-loop (or an equivalent explicit self-loop); and
4. leave transitions outside the selected `Q` scope unchanged so a calibration
   prelude can expose the changed provenance.

A backend MAY use a conservative superset `Q_conservative` (for example, every
cell appearing anywhere in an enumerated scored trajectory), but the receipt
MUST say so and the calibration must exit that superset before probing a
withheld action. It MUST NOT call an all-state deletion an exact command-site
construction. In this witness, a two-step complete enumeration has
`Q_exact = {0,1}` while the trajectory-cell superset is
`Q_conservative = {0,1,2}`; the current ring spike uses the latter strategy.

The twin is preserving only when all scored trajectories, values, and optimal
action sets agree. It is non-vacuous only when at least one public prelude trace
differs. For the witness above, use `Q_conservative = {0,1,2}`. A calibration
of `advance, advance, advance, retreat` reaches state `3` before the final
retreat: the body-limited arm cannot execute `retreat` and stays at `3`, while
the twin retreats to `2`. This establishes public distinction without changing
scored behavior.

## 5. Required exact receipts

The topology backend MUST reuse the generic finite audit/query machinery. It
MUST NOT invent topology-specific learner or card APIs for these predicates.
For the non-ring fixture, an exact receipt MUST include at least:

| Predicate | Required evidence |
|---|---|
| Totality | Every state × declared action has a deterministic target, including default self-loops. |
| Non-cycle topology | Executable graph invariant rejects simple-cycle isomorphism; degree sequence is a diagnostic. |
| Sequence coverage | Complete action enumeration over the declared horizon, not one teacher rollout. |
| Body contrast | Body-limited and unrestricted ceilings differ; the goal is not the start state. |
| Twin trajectories | Every witness/twin scored trajectory agrees. |
| Twin values | Every witness/twin scored value agrees, including fallback and norm penalties. |
| Twin optima | Complete ceiling-achieving sequences and deduplicated first-action sets agree. |
| Public distinction | Empty-action public traces differ through the calibration prelude. |
| Identification | `AmbiguitySet` diameter is 2 without the prelude and 1 after the public prelude. |
| Boundary | Public trace contains no support, edge table, topology metadata, seed, or case label. |
| Replay | Same specification and index reproduce terms, receipt, and hash exactly. |

For this fixture the minimum executable run is concrete: five states, three
explicit movement/hold rows per state (15 keyed rows), four declared actions,
and `4^2 = 16` complete scored sequences. The harness must compare all 16
body/twin trajectories and values, compare their complete optimal sequence and
first-action sets, and run the empty-action public trace once before and once
after the four-step calibration prelude. These counts are receipt facts, not
sampling recommendations.

The receipt MUST report whether the twin used exact command sites or a
conservative scope, the topology state count, transition-row count, sequence
count, family hash, and generator seed/index separately. `valid` is the
conjunction of the declared predicates; a rejected candidate MUST NOT be
emitted. The receipt remains finite semantic evidence, not learner or transfer
evidence.

## 6. Ring compatibility gate

Adding the topology abstraction MUST leave the current ring backend as a
specialized, exact backend. Before admitting the new backend:

- every existing ring/Card 03 transition, outcome, public trace, optimal-action
  set, and restoration-clock test passes unchanged;
- the ring fixture's existing family hash and receipts remain byte-compatible;
- an explicit ring adapter round-trips its transition table to the prior ring
  semantics, including modular displacement and default behavior;
- body/environment swaps retain their existing behavior-preserving and
  publicly-distinguishable properties; and
- no ring-only assumption leaks into the topology-independent receipt schema.

The non-ring fixture MUST be tested through the same generic `Fragment`,
`PubliclyObservable`, `AmbiguitySet`, identification, value-bound, and
noninterference interfaces. A pass obtained by converting the graph to an
ordered ring is not topology generalization.

## 7. Coupling and process-algebra limits

The current ring compiler interprets coupling as an additive integer
displacement. That interpretation is not topology-independent. A graph backend
MUST choose one of these explicit outcomes:

- define a typed intervention algebra in which coupling resolves to a declared
  graph intervention and audit that resolution; or
- reject a term containing coupling for that backend as unsupported.

It MUST NOT silently add integers to state IDs, drop coupling terms, or claim
cross-backend equivalence without a typed mapping. The same rule applies to
interrupts: the current spike only suppresses public signals for a frozen named
process and does not implement process continuation/restart. A topology backend
may preserve that declared limit, but it cannot advertise richer interrupt
semantics until it has stateful process tests.

Action restrictions, viability boundaries, and resource restrictions remain
different constructs. A body action restriction changes intervention support;
an environment transition default changes edge availability; a viability
boundary changes admissible states; a resource restriction changes the scored
horizon. Their receipts and hashes MUST retain these distinctions.

## 8. Canonical identity and replay

The family hash MUST include the canonical topology semantics: sorted state
identifiers, transition keys and targets, the default-transition policy, typed
action/intervention meanings, body support, ports/views, norm, scoring,
calibration, restoration, restrictions, and behavior-sensitive process order.
It MUST preserve body support versus environment deletion even when their
composed transitions coincide. Order-insensitive sets and transition rows are
sorted; writer, reveal, process, and norm orders remain semantic where they
affect execution or public output.

The hash MUST exclude `name`, seed, index, and construction metadata. Replay is
the complete generation specification plus index, including the deterministic
RNG path needed to reach that index. Canonical serialization should use the
maintained serializer and BLAKE3 already used by the spike. A digest is a
collision-resistant version identifier, not proof of semantic equivalence;
conformance tests compare canonical bytes and test semantic mutations for a
changed digest absent a cryptographic collision.

## 9. Minimal API implications

The smallest principled extension is an environment configuration-space seam,
not a second card evaluator:

```text
ConfigurationSpace {
    states: StateSet,
    transitions: TransitionTable,
    default: DefaultTransition::SelfLoop,
}

EnvironmentTerm { configuration_space, start }
BodyTerm { typed_interventions, support, sensorium }
compile(term) -> CompiledWorld
```

`TransitionTable` must expose deterministic lookup and a canonical iterator;
`CompiledFragment::step` must pass the body-produced intervention to that
lookup without assuming arithmetic adjacency. The existing generic query
functions should consume the compiled fragment unchanged. Public and
privileged views must remain separate, and generator metadata must have no path
to public output.

The compiler should expose topology diagnostics (state count, row count,
default policy, connectivity/degree summary) as audit metadata rather than
learner observations. If a backend cannot implement a generic operation (for
example coupling on graph interventions), its compile/audit status must be
`Unsupported` or a typed error, never a silent fallback.

## 10. Stop rules

Stop the generalization experiment and preserve the last passing commit if any
of the following occurs:

- the non-ring fixture is isomorphic to, or executed as, a disguised ring;
- missing transitions are not explicit total self-loops;
- body support and environment deletions collapse into one untyped predicate;
- a body/environment twin passes only because its public traces are identical;
- any receipt predicate samples instead of exhaustively enumerating the finite
  support;
- privileged topology, support, edge rows, or generator metadata enter public
  traces or query targets;
- coupling/interrupt behavior is silently ignored or given ring arithmetic; or
- existing ring/Card 03 tests, hashes, or receipts change without a separately
  versioned compatibility decision.

Passing this oracle earns a candidate topology backend and a finite world
validity result. It does not authorize learner training, multi-world lineage,
GPU execution, or replacement of the handwritten card families.
