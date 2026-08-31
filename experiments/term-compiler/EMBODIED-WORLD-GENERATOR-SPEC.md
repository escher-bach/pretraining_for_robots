# Embodied World Generator Specification

## Status and terminology

This document specifies the finite embodied-world generator implemented in
`pretraining-term-compiler`. It describes current executable behavior, not a
proposal for a general world language.

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY**
are normative. A *term* is a serializable world description. A *compiled
world* is the executable result of validating and lowering a term. A *family*
is a related witness, control, and twin, rather than unrelated samples.

## 1. Purpose and non-goals

The generator creates small, deterministic, auditable embodiment contrasts. It
preserves the distinction between a body that cannot drive an actuator and an
environment in which a cell/action edge is absent. Those causes MAY induce
identical scored behavior but MUST remain distinct typed data.

The generator MUST represent a body, environment, normative objective,
signals, and information boundary; validate before execution; expose exact
finite transition/value and public-observation interfaces; and generate
receipt-validated nondegenerate families. It is not a general robotics
simulator, continuous controller, stochastic process language, learner API,
renderer, or universal process scheduler. A valid receipt is finite semantic
evidence, not evidence of usefulness or learnability.

## 2. Mathematical/executable model

The configuration space is either a compact ring `C = {0, ..., n-1}` or a
finite deterministic graph with `2 <= |C| <= 32`. A graph is an explicit table
of `(state, actuator) -> destination` rows plus `MissingEdgeBehavior::SelfLoop`;
unknown states, duplicate keys, and out-of-set destinations are invalid. An
actuator has an identifier in `[0,32)`, an integer displacement `d(a)`, and one
role:

- `Movement` MAY be withheld by body support;
- `Hold` MUST remain body-supported; and
- `Fallback` MUST remain body-supported, does not move the ring, and absorbs
  scored/public execution.

There MUST be at most one fallback. A directly authored term has positive
scored horizon `H <= 8`.

Displacement is a ring-only effect. Graph actuators MUST set `d(a)=0`; their
meaning comes from the typed intervention key and explicit transition rows.
The graph backend MUST reject a nonzero displacement rather than silently
ignoring ring arithmetic.

At scored index `e`, support and edge presence are separate predicates:

```text
supported(a,e) = a in S
                 or exists r: r.actuator=a and e > r.after_step
edge_present(c,a) = (c,a) not in E
```

For a non-fallback ring action whose source cell is admitted by every viability
restriction, that is supported, passes every action restriction, and has an
environment edge, execution is:

```text
c' = (c + d(a) + coupling_delta(e,a,c)) mod n.
```

For a graph action under the same predicates, execution looks up the explicit
row; an absent row applies the declared self-loop default. An unsupported
command still leaves the state unchanged. Fallback is absorbing in outcomes
and the public trace, rather than by adding a terminal state.

## 3. Syntax and data model

The top-level term is:

```text
WorldTerm {
  name, horizon, ports, wiring,
  body, environment, norm, scoring,
  calibration?, restorations[], disturbance?, scaffold?,
  couplings[], interrupts[], restrictions[], reveals[]
}
```

`name` is diagnostic instance text, not family identity. `BodyTerm` contains
body-local `Morphology { segments }`, `Actuation { command_port, actuators,
supported }`, and `Sensorium { cell_port, publishes_cell }`. `EnvironmentTerm`
contains `start` and either a ring with `blocked_edges: (cell, actuator_id)` or
a graph with explicit transition rows and a self-loop default. A blocked or
missing edge MUST NOT be rewritten as body support.

### Norms and scoring

`NormTerm` contains a visibility and an expression. Leaves are `Settle(cell)`
(reach and remain), `Visit(cell)`, and `Avoid(cell)` (do not enter after the
start). Composition is `Both`, `Supersede(before,after,guard)`, or
`Priority(high,low)`. Guards are `AtStart`, `AfterStep(k)`, `OnAction(a)`,
`OnCellEntry(c)`, and `Never`.

### Guard evaluation and norm timing

A guard is a total predicate over `GuardContext { executed, last_action,
cell }`. Its exact truth conditions are:

```text
AtStart         = true
AfterStep(k)    = executed > k
OnAction(a)     = last_action == Some(a)
OnCellEntry(c)  = cell == c
Never           = false
```

`AtStart` is deliberately not a one-time event: it is true in every context,
including after actions. `OnCellEntry` is likewise a cell-equality predicate,
not a proof that the current step crossed an edge into that cell.

The executor uses three contexts. At the start it evaluates public guarded
events with `(0, None, start)`. A non-fallback scored transition at source
cell `c`, action `a`, and prior scored count `e` evaluates coupling guards
with `(e + 1, Some(a), c)`; this is the **source** context. After that
transition it evaluates public process/reveal guards with
`(e + 1, Some(a), c')`, where `c'` is the destination; this is the
**post-transition** context. Calibration pulses call the transition at scored
index zero, so each pulse's coupling context has `executed = 1` and its then
current source cell.

For a non-fallback outcome, norm evaluation receives the complete supplied
trajectory and the final context `(actions.len(), actions.last(),
trajectory.last())`. `Supersede(before, after, guard)` selects `after` when
that final guard fires and otherwise selects `before`; it then evaluates the
selected subtree over the whole trajectory. It is not a per-step switch.
Consequently `Supersede` with `AtStart` always selects `after`, even for a
nonempty trajectory. A fallback outcome uses fallback scoring before norm
evaluation, as specified below.

### Reference public-norm encoding

The reference executor emits a public norm as a signed 64-bit `norm_code`.
This encoding is not a general interchange format, but byte/trace-compatible
implementations MUST compute it exactly. Start with unsigned state
`0xcbf29ce484222325`; for every `mix(v)`, set
`state = (state XOR v) * 0x100000001b3 (mod 2^64)`. Encode leaves as
`mix(1 XOR cell)`, `mix(2 XOR cell)`, and `mix(3 XOR cell)` for `Settle`,
`Visit`, and `Avoid`. Encode `Both` as `mix(4)` followed by its left then right
child; `Supersede` as `mix(5 XOR guard_code)` followed by before then after;
and `Priority` as `mix(6)` followed by high then low. Guard codes are
`AtStart=11`, `AfterStep(k)=12 XOR k`, `OnAction(a)=13 XOR a`,
`OnCellEntry(c)=14 XOR c`, and `Never=15`. The final unsigned state is
reinterpreted as Rust `u64 as i64` and appended to the public trace.

`ScoringTerm` is `(goal_reward, action_cost, fallback_reward,
violation_penalty)`, all signed integers except that `action_cost` MUST be
nonnegative. First fallback at index `k` has value
`fallback_reward - action_cost*k`. Otherwise a met norm has value
`goal_reward - action_cost*q`, where `q` is its settlement index when defined,
or the sequence length. A violated prohibition has `violation_penalty`; an
unmet non-violation has value zero.

### Calibration, restoration, and processes

`CalibrationTerm { pulses, publishes_cells }` is an optional nonempty prelude
of declared actuators. Every pulse MUST execute at scored index zero: it does
not consume horizon or advance restoration time. Its cumulative cells are
public only when `publishes_cells` is true.

`SupportRestoration` names an initially unsupported movement actuator, one
`after_step`, and a public signal port/value. Each actuator MAY be restored at
most once. Its announcement is public before scoring; the effect begins only
when `executed > after_step`.

`ProcessTerm` is a named disturbance or scaffold with guarded signal outputs.
`RevealTerm` explicitly publishes a guarded value through a public signal port.
Its `source_visibility` field is currently provenance/hash data only: runtime
publication is controlled by the output port and guard, and the validator does
not inspect that source field. A term using a private value in a reveal therefore
needs an explicit audit; this spike does not provide a generic hidden-state to
reveal proof.
`CouplingTerm` has a `Sum`, `Override`, or `Conflict` rule, ordered guarded
integer writers, and an `inactive_value`. `InterruptTerm` names a process and
contains a guard, a displaced mode (`Continues` or `Frozen`), and a resume mode
(`FromState` or `Restart`). Restrictions are action support, viability
(reset/absorbing boundary), or resource budget with declared scope.

Several fields are intentionally narrow in this version. `ports` and `wiring`
are validated and hashed, but wiring is not a general runtime dataflow
executor: lowered transitions read the body/environment fields directly.
`Actuation.command_port`, `Sensorium.cell_port`, actuator `name`, and
`Coupling.variable` are typed identity or validation metadata and do not alter
the ring transition by themselves. `Sensorium.publishes_cell` controls cell
records in the public trace, while `NormTerm.visibility` controls whether the
compact norm code is public; neither changes private norm evaluation. A
restoration's announcement port is validated as a public signal port, but the
trace emits its `announcement_value` directly. Disturbance and scaffold signals
share the same guarded public-event mechanism; their distinction is currently
structural/name-level rather than a separate scheduler.

## 4. Types, ports, and visibility

Every port has unique nonempty name, direction, value type, and visibility:

```text
direction = Input | Output | Monitor
value     = Command | Cell | Signal | Support
view      = Public | Privileged | Generator
```

`Monitor` MUST NOT source wiring. `Support` MUST NOT be public. The body
command port MUST be a public command input; the cell port MUST be a cell
output, and MUST be public when its sensorium publishes cells.

For `from -> to`, `from` MUST be an output, `to` an input/monitor, and the
value types MUST agree. Ordinary flows are exactly:

```text
Public      -> Public | Privileged | Generator
Privileged  -> Privileged
Generator   -> Privileged | Generator
```

Thus direct privileged/generator-to-public flow is invalid. A public reveal is
the explicit publication mechanism, not an ordinary-wire exception. Public to
Generator is accepted by the validator, but generator values cannot be emitted
as process signals from a Generator port and ordinary wiring has no runtime
publication side effect in this spike.

## 5. Validation and error model

Validation MUST precede lowering. It rejects an empty name; invalid ring/graph,
horizon, or start; duplicate/missing/malformed ports; bad public sensor;
duplicate/out-of-range/no actuators; unsupported hold/fallback; multiple
fallbacks; unknown support, edge, graph row, pulse, restriction, or restoration
targets; duplicate graph keys; nonzero graph displacement; negative action cost;
malformed wires; monitor sources; visibility leaks; public support ports;
malformed process/reveal outputs; duplicate process names; unknown interrupted
process; zero resource budget; and malformed couplings.

A restoration MUST target one initially unsupported movement action and use a
public signal output. `Override` and `Conflict` MUST have a writer.
`Conflict` MUST have at most one declared writer; this conservative rule avoids
requiring a proof that guards never coincide.

For ring terms, the validator preserves the legacy permissive role/displacement
metadata (the lowered ring transition uses movement displacement and fallback
never moves). For graph terms, every actuator displacement MUST be zero; a
nonzero value is a typed validation error because graph meaning comes from the
explicit transition table.

Errors are `Invalid(message)`, `Unsupported(message)`, `UnknownPort(name)`,
`IllTypedWire {from,to}`, `VisibilityLeak {from,to}`, or
`MonitorAsSource(name)`. Generation also uses `Invalid` when bounds cannot
produce its required contrast or a candidate fails its receipt.

## 6. Compilation and lowering

```text
WorldTerm -> validate -> finite lowered program -> CompiledWorld
```

The result contains `CompiledFragment`, `CompiledContract`, derived
kernel-use metadata, optional generator metadata, and a family hash. The
compiled transition table is total over state, scored index, and declared
actuator. Ring rows use modular displacement; graph rows use explicit lookup
and the self-loop default.
`CompiledFragment` implements finite actions, horizon, start, transition,
value, and public trace. The compiled horizon is the authored horizon capped by
the smallest resource budget, when present.

Couplings are lowered to total functions on ring backends: `Sum` returns the active sum or
`inactive_value`; `Override` returns the last active writer or inactive value;
and validated `Conflict` returns its sole active writer or inactive value. Each
coupling contributes a displacement and all lowered coupling displacements are
summed. Guards receive the transition context (including the current action and
cell); the declared variable identifier is not a runtime storage lookup. A
runtime coupling error MUST NOT be discarded. A graph term containing a
displacement coupling is rejected as unsupported; it is not approximated by
integer state arithmetic.

Interrupt lowering is intentionally partial. A `Frozen` interrupt suppresses
public signals from the named process while its guard is active. `Continues`
does not suppress those signals, and neither `resume` value currently changes
anything because process state and a scheduler are not implemented. The fields
remain serialized and hashed so changing them is a contract change for future
implementations, but this version MUST report this limit rather than claiming
continuation or restart behavior.

## 7. Operational semantics and views

An absorbing viability cell stays unchanged. A reset boundary returns to start.
Other transitions follow Section 2. The public trace is an integer sequence in
this order:

1. public cumulative calibration cells when a calibration exists *and*
   `publishes_cells` is true; when no calibration exists, the start cell when
   the sensorium publishes cells. If calibration exists with
   `publishes_cells=false`, no calibration cell (including no synthesized start
   cell) is emitted;
2. a deterministic compact code for a public norm, if any;
3. restoration announcement values;
4. public guarded process signals and reveals at start;
5. for every scored action until fallback, a public resulting cell (when
   enabled), then public process signals and reveals.

Fallback appends `-1` and terminates the public trace. Later supplied actions
are unscored. Generic trajectories MAY still enumerate a full action sequence,
but final outcome cell uses the prefix before its first fallback.

`PublicView` contains only this trace. `PrivilegedView` explicitly contains a
trajectory, the environment state space, and body support. Audit metadata
contains the family hash, its schema version, kernel-use flags, public port
names, and topology diagnostics. Seed/index metadata is neither public nor
part of a family hash.

For example, if a term has a calibration prelude but sets
`publishes_cells=false` while its sensorium publishes cells, the public trace
does not begin with either the calibration start cell or the cumulative pulse
cells. It begins with the public norm code (if present), then any restoration
announcements and start events; scored cells are still emitted after each
scored action. This differs from a term with no calibration, where the sensorium
does emit the start cell before the norm code.

## 8. Canonical hashing, replay, and conformance levels

The current canonical term encoding is schema version 2. It clears `name`,
sorts ports by name, wiring by endpoints, actuators by identifier, and ring
blocked edges or graph transition rows lexicographically. It preserves
behavior-sensitive orders: norm tree, restrictions, process signals,
restoration announcements, and override writers.
The reference implementation serializes this canonical Rust value using its
derived `serde` representation through `serde_json`, then hashes those bytes
with BLAKE3. The resulting 256-bit digest is a collision-resistant family
identifier, not a mathematical proof of semantic inequality: a hash collision
is possible in principle. Equal hashes are a versioning identifier, never the
sole evidence that two arbitrary programs are equivalent.

This specification has two conformance levels. **Semantic conformance**
requires the types, validation, lowering, operational behavior, public/other
views, and family/receipt predicates in this document. It does not require an
independent implementation to reproduce the reference's derived Rust JSON
field layout, `serde_json` escaping/number serialization, or bytes. A semantic
implementation MAY use another canonical representation and identifier, but
MUST label it as such and MUST NOT claim reference hash compatibility.

**Reference-byte conformance** is narrower and implementation-specific. It
requires the exact current Rust data model, declaration/field order supplied
by the reference's derived `Serialize`, the `serde_json` bytes it produces,
the sorting/normalization above, and BLAKE3 over those exact bytes. Thus its
canonical bytes and hash are reproducible by rebuilding the reference, rather
than a language-neutral wire standard defined by this document. The exact
`norm_code` algorithm above is separately fixed for public-trace
compatibility.

Seed/index changes MUST NOT alter the hash. Version-1 ring fixtures can be
replayed through the explicit legacy ring encoding for migration; their
version-1 digest is not equal to the version-2 digest. Semantic changes, including action
meaning, support, edge, calibration, restoration, norm, or scoring changes,
MUST be treated as a distinct semantic term; they change the reference
canonical term bytes and SHOULD alter the digest absent a cryptographic
collision. Replay is the complete
`GenerationSpec` plus index; `GeneratorMetadata { seed, index }` stays
separate from the term.

## 9. Deterministic family generation

`GenerationSpec` is:

```text
seed, count, template, min_cells, max_cells, min_horizon, max_horizon
```

`template` is `Ring` by default for compatibility or the explicit
`BranchingGraph` non-ring template. The ring template samples:

It requires `1 <= count <= 64`, `2 <= min_cells <= max_cells <= 8`, and
`1 <= min_horizon <= max_horizon <= 6`. Family generation further requires a
possible `n >= 4` and `H >= 2`. It seeds `ChaCha8Rng` with `seed` and, for
index `i`, samples:

```text
n ~ Uniform[max(min_cells,4), max_cells]
H ~ Uniform[max(min_horizon,2), min(max_horizon,n-2)]
start = 0; goal = n-1
```

The branching-graph template requires a possible `n >= 5` and `H >= 2`, then
samples `H <= n-3`. It emits explicit graph rows, a self-loop default, and
zero-displacement action descriptors; graph action meaning comes only from the
rows. Its advance chain reaches `H+2` only in calibration, while retreat from
state `1` is the short unrestricted route to the goal and other retreat rows
remain genuine graph interventions.

An empty range is an error, not a relaxed contrast. `H <= n-2` prevents forward
motion from wrapping to the backward goal in-budget.

Each generated family uses this alphabet:

| ID | Name | Displacement | Role |
|---:|---|---:|---|
| 0 | advance | +1 ring / 0 graph | Movement |
| 1 | retreat | -1 ring / 0 graph | Movement |
| 2 | hold | 0 | Hold |
| 3 | fallback | 0 | Fallback |

It uses public `Settle(goal)`, scoring `(100,1,50,-100)`, and public
calibration of `H+1` ring advances (or `H+2` graph advances) followed by
retreat. It returns exactly these
three terms:

1. **Body-limited witness:** support `{advance,hold,fallback}`, no blocked
   edges.
2. **Unrestricted control:** all four actions supported, no blocked edges.
3. **Environment twin:** full support, with retreat deleted only at the
   conservative set of cells appearing anywhere in a complete witness
   trajectory (including its initial and final cells). The implementation
   currently uses this set, not the exact set of pre-action command sites; it
   may therefore include a terminal cell, which is safe but stronger than
   necessary.

The prelude travels beyond that scored region before retreat. The twin is thus
publicly distinct in calibration while behaviorally preserving the witness
during scoring. The scored-region set is computed by enumerating all action
sequences of the compiled witness fragment, so a future transition extension
must either preserve this conservative interpretation or version the generator
and receipt semantics.

## 10. Exact validity receipt

Every candidate is compiled and verified before emission. Its
`EmbodimentValidityReceipt` means:

| Field | Exact predicate |
|---|---|
| `sequences_checked` | Complete action sequences enumerated over the common alphabet/horizon. |
| `topology_total` | Lowered table has one deterministic result for every state, scored index, and declared action. |
| `topology_state_count` | Number of environment states in the selected ring or graph configuration space. |
| `topology_transition_rows` | Explicit blocked-edge/graph-row count retained by the environment term. |
| `topology_degree_sequence` | Executable undirected graph degree vector, with self-loops ignored. |
| `topology_is_simple_cycle` | Executable topology diagnostic; graph witnesses must be false. |
| `twin_scope` | `ExactCommandSites` or explicit `ConservativeTrajectoryCells`; the current generator uses the latter. |
| `family_hash` | Semantic hash of the body-limited term; seed/index remain separate receipt metadata. |
| `generator_seed`, `generator_index` | Replay coordinates copied from enclosing `GeneratedWorld` metadata; neither enters the family hash. |
| `goal_differs_from_start` | The `Settle`/`Visit` target differs from start. |
| `body_limitation_changes_ceiling` | Witness and unrestricted exact ceilings differ. |
| `twin_trajectories_equal` | Every witness/twin complete trajectory agrees. |
| `twin_values_equal` | Every witness/twin complete value agrees. |
| `twin_optimal_sequences_equal` | Exact ceilings and complete optimal sequence sets agree. |
| `twin_publicly_distinct` | Empty-action public traces differ via the prelude. |
| `calibration_identifies_body` | A uniform two-contract ambiguity set has public diameter 2 without calibration and 1 with it. |
| `valid` | Conjunction of all required predicates above, excluding the count. |

A candidate with `valid = false` MUST NOT be emitted. Generation replay is a
separate acceptance test that reruns `GenerationSpec` plus index and compares
terms, receipts, and hashes; `verify()` itself only recomputes the finite
semantic receipt. Neither is performance evidence.

## 11. Worked example

Let `n=5` and `H=2`. Then `start=0`, `goal=4`, and calibration is
`advance, advance, advance, retreat`. The body-limited witness has support
`{advance,hold,fallback}` and calibration cells `0,1,2,3,3`; retreat is
unsupported. Its scored goal is unreachable by two advances, so immediate
fallback yields 50.

The unrestricted control retreats `0 -> 4`, holds, and yields 99. The twin
blocks retreat at scored-reachable cells `{0,1,2}`, so it matches witness
scored trajectories, values, and optima. In calibration, retreat from 3 is not
blocked: the twin reaches 2 while the body-limited witness remains at 3. This
makes the preserving provenance transformation observable without changing its
scored behavior.

## 12. Conformance requirements

A conforming implementation MUST:

1. enforce Sections 4–6 before execution;
2. preserve body support and environment deletion as distinct data and in the
   family hash;
3. provide the finite transition, scoring, fallback, prelude, restoration, and
   public-trace semantics above;
4. use total lowered coupling and reject invalid conflict shape;
5. use deterministic seeded generation and emit only receipt-valid families;
6. enumerate complete finite action support for every receipt claim;
7. keep public, privileged, and generator views structurally distinct; and
8. expose executable topology diagnostics and receipt state/row/scope facts;
9. reject graph displacement couplings and nonzero graph actuator displacement;
10. use canonical serialization plus BLAKE3 for semantic family identity,
    reporting schema version 2 and the tested legacy ring migration mapping.

A conforming implementation SHOULD expose a `verify()` operation that
recomputes a receipt from returned terms. It MAY expose diagnostics, but those
diagnostics MUST NOT silently become public observations.

## 13. Extension points

Future compatible extensions MAY add value types, richer morphology,
configuration graphs, stochastic transition kernels, sensors, scoring
functions, symbolic action adapters, or stronger model checking. They SHOULD
retain explicit typed causes, information views, replay metadata, canonical
semantic identity, and validity receipts rechecked from executable semantics.
Any extension that changes meaning MUST change the family hash and define new
conformance tests.

## 14. Known limits

- Configuration is a finite deterministic ring or explicit graph only; there
  is no continuous geometry, stochastic kernel, or general process scheduler.
- Public traces are integer sequences, not a renderer or learner event format.
- The generator emits one fixed embodiment-contrast template, not a broad
  distribution over world designs.
- Resource scope is declared but does not yet have distinct local/shared
  multi-process execution semantics.
- Interrupt execution freezes named public process signals when declared
  frozen; it does not model general process state, continuation, restart, or a
  scheduler-level displaced process.
- Public norm publication is a compact deterministic integer code, not a
  separately specified wire format.
- The receipt does not establish robustness outside finite support, rendering
  correctness, learner acquisition, or transfer.

## Appendix A. Non-normative integration note

The implementation is currently an isolated experiment. It has no learner
renderer, training configuration, GPU path, or authority to replace existing
handwritten worlds. Existing worlds remain independent until separately
verified dual-path integration is authorized.
