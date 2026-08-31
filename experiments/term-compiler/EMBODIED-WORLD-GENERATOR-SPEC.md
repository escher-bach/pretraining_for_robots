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

The configuration space is the ring `C = {0, ..., n-1}`, where `2 <= n <= 32`.
An actuator has an identifier in `0..32`, an integer displacement `d(a)`, and
one role:

- `Movement` MAY be withheld by body support;
- `Hold` MUST remain body-supported; and
- `Fallback` MUST remain body-supported, does not move the ring, and absorbs
  scored/public execution.

There MUST be at most one fallback. A directly authored term has positive
scored horizon `H <= 8`.

At scored index `e`, support and edge presence are separate predicates:

```text
supported(a,e) = a in S
                 or exists r: r.actuator=a and e > r.after_step
edge_present(c,a) = (c,a) not in E
```

For a non-fallback action that is supported, passes every action restriction,
and has an environment edge, execution is:

```text
c' = (c + d(a) + coupling_delta(e,a,c)) mod n.
```

An unsupported command or missing edge leaves the cell unchanged. Fallback is
absorbing in outcomes and the public trace, rather than by adding a terminal
cell outside the ring.

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
`Morphology { cells }`, `Actuation { command_port, actuators, supported }`,
and `Sensorium { cell_port, publishes_cell }`. `EnvironmentTerm` contains
`start` and `blocked_edges: (cell, actuator_id)`; a blocked edge MUST NOT be
rewritten as body support.

### Norms and scoring

`NormTerm` contains a visibility and an expression. Leaves are `Settle(cell)`
(reach and remain), `Visit(cell)`, and `Avoid(cell)` (do not enter after the
start). Composition is `Both`, `Supersede(before,after,guard)`, or
`Priority(high,low)`. Guards are `AtStart`, `AfterStep(k)`, `OnAction(a)`,
`OnCellEntry(c)`, and `Never`.

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
`CouplingTerm` has a `Sum`, `Override`, or `Conflict` rule, ordered guarded
integer writers, and an `inactive_value`. `InterruptTerm` names a process and
contains a guard, a displaced mode (`Continues` or `Frozen`), and a resume mode
(`FromState` or `Restart`). Restrictions are action support, viability
(reset/absorbing boundary), or resource budget with declared scope.

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
the explicit publication mechanism, not an ordinary-wire exception.

## 5. Validation and error model

Validation MUST precede lowering. It rejects an empty name; invalid ring,
horizon, or start; duplicate/missing/malformed ports; bad public sensor;
duplicate/out-of-range/no actuators; unsupported hold/fallback; multiple
fallbacks; unknown support, edge, pulse, restriction, or restoration targets;
negative action cost; malformed wires; monitor sources; visibility leaks;
public support ports; malformed process/reveal outputs; duplicate process
names; unknown interrupted process; zero resource budget; and malformed
couplings.

A restoration MUST target one initially unsupported movement action and use a
public signal output. `Override` and `Conflict` MUST have a writer.
`Conflict` MUST have at most one declared writer; this conservative rule avoids
requiring a proof that guards never coincide.

Errors are `Invalid(message)`, `UnknownPort(name)`, `IllTypedWire {from,to}`,
`VisibilityLeak {from,to}`, or `MonitorAsSource(name)`. Generation also uses
`Invalid` when bounds cannot produce its required contrast or a candidate fails
its receipt.

## 6. Compilation and lowering

```text
WorldTerm -> validate -> finite lowered program -> CompiledWorld
```

The result contains `CompiledFragment`, `CompiledContract`, derived
kernel-use metadata, optional generator metadata, and a family hash.
`CompiledFragment` implements finite actions, horizon, start, transition,
value, and public trace. The compiled horizon is the authored horizon capped by
the smallest resource budget, when present.

Couplings are lowered to total functions: `Sum` returns the active sum or
`inactive_value`; `Override` returns the last active writer or inactive value;
and validated `Conflict` returns its sole active writer or inactive value. A
runtime coupling error MUST NOT be discarded.

## 7. Operational semantics and views

An absorbing viability cell stays unchanged. A reset boundary returns to start.
Other transitions follow Section 2. The public trace is an integer sequence in
this order:

1. public cumulative calibration cells, if enabled; otherwise the start cell
   when the sensorium publishes cells;
2. a deterministic compact code for a public norm, if any;
3. restoration announcement values;
4. public guarded process signals and reveals at start;
5. for every scored action until fallback, a public resulting cell (when
   enabled), then public process signals and reveals.

Fallback appends `-1` and terminates the public trace. Later supplied actions
are unscored. Generic trajectories MAY still enumerate a full action sequence,
but final outcome cell uses the prefix before its first fallback.

`PublicView` contains only this trace. `PrivilegedView` explicitly contains a
trajectory, blocked edges, and body support. Audit metadata contains the family
hash, kernel-use flags, and public port names. Seed/index metadata is neither
public nor part of a family hash.

## 8. Canonical hashing and replay

To hash a family, the implementation clears `name`, sorts ports by name,
wiring by endpoints, actuators by identifier, and blocked edges
lexicographically. It preserves behavior-sensitive orders: norm tree,
restrictions, process signals, restoration announcements, and override writers.
It serializes the canonical term using `serde_json` and hashes the bytes with
BLAKE3.

Seed/index changes MUST NOT alter the hash. Semantic changes, including action
meaning, support, edge, calibration, restoration, norm, or scoring changes,
MUST alter the serialized term and hence the hash. Replay is the complete
`GenerationSpec` plus index; `GeneratorMetadata { seed, index }` stays
separate from the term.

## 9. Deterministic family generation

`GenerationSpec` is:

```text
seed, count, min_cells, max_cells, min_horizon, max_horizon
```

It requires `1 <= count <= 64`, `2 <= min_cells <= max_cells <= 8`, and
`1 <= min_horizon <= max_horizon <= 6`. Family generation further requires a
possible `n >= 4` and `H >= 2`. It seeds `ChaCha8Rng` with `seed` and, for
index `i`, samples:

```text
n ~ Uniform[max(min_cells,4), max_cells]
H ~ Uniform[max(min_horizon,2), min(max_horizon,n-2)]
start = 0; goal = n-1
```

An empty range is an error, not a relaxed contrast. `H <= n-2` prevents forward
motion from wrapping to the backward goal in-budget.

Each generated family uses this alphabet:

| ID | Name | Displacement | Role |
|---:|---|---:|---|
| 0 | advance | +1 | Movement |
| 1 | retreat | -1 | Movement |
| 2 | hold | 0 | Hold |
| 3 | fallback | 0 | Fallback |

It uses public `Settle(goal)`, scoring `(100,1,50,-100)`, and public
calibration of `H+1` advances followed by retreat. It returns exactly these
three terms:

1. **Body-limited witness:** support `{advance,hold,fallback}`, no blocked
   edges.
2. **Unrestricted control:** all four actions supported, no blocked edges.
3. **Environment twin:** full support, with retreat deleted only at cells that
   exact enumeration finds reachable at a scored command site in the witness.

The prelude travels beyond that scored region before retreat. The twin is thus
publicly distinct in calibration while behaviorally preserving the witness
during scoring.

## 10. Exact validity receipt

Every candidate is compiled and verified before emission. Its
`EmbodimentValidityReceipt` means:

| Field | Exact predicate |
|---|---|
| `sequences_checked` | Complete action sequences enumerated over the common alphabet/horizon. |
| `goal_differs_from_start` | The `Settle`/`Visit` target differs from start. |
| `body_limitation_changes_ceiling` | Witness and unrestricted exact ceilings differ. |
| `twin_trajectories_equal` | Every witness/twin complete trajectory agrees. |
| `twin_values_equal` | Every witness/twin complete value agrees. |
| `twin_optimal_sequences_equal` | Exact ceilings and complete optimal sequence sets agree. |
| `twin_publicly_distinct` | Empty-action public traces differ via the prelude. |
| `calibration_identifies_body` | A uniform two-contract ambiguity set has public diameter 2 without calibration and 1 with it. |
| `valid` | Conjunction of all required predicates above, excluding the count. |

A candidate with `valid = false` MUST NOT be emitted. The receipt is exact
finite semantic evidence, not a performance result.

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
8. use canonical serialization plus BLAKE3 for semantic family identity.

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

- Configuration is a finite ring only; there is no continuous geometry or
  general graph executor.
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
