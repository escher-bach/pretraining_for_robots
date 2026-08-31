# Typed embodied-world compiler spike

This is an isolated R3c experiment in the `codex/term-compiler-spike`
worktree. It is not connected to any existing card crate, learner renderer,
profile, training configuration, or GPU path.

## Executable boundary

```text
WorldTerm -> validate -> lower -> CompiledFragment + CompiledContract
```

`CompiledFragment` implements the existing finite `Fragment` and
`PubliclyObservable` interfaces. Existing exact enumeration, bounds,
noninterference, ambiguity, and orbit queries can therefore operate on a
compiled term without changing the shared audit kernel.

The first-class term components are:

- `BodyTerm`: `Morphology`, `Actuation`, `Sensorium`, action roles, and body
  support;
- `EnvironmentTerm`: start configuration plus either compact ring edges or an
  explicit finite graph with deterministic self-loop defaults;
- `NormTerm` and `ScoringTerm`: public/private normative expression plus goal,
  action, fallback, and violation values;
- `CalibrationTerm`: an unscored action prelude whose cumulative cells can be
  public;
- `SupportRestoration`: a public announcement whose actuator support begins at
  its declared absolute scored step;
- optional `Disturbance`/`Scaffold` signals, typed ports/wiring, restrictions,
  reveals, interrupts, and lowered coupling.

Validation rejects ill-typed wires, direct non-public-to-public flow,
monitor-to-actuator feedback, public support queries, malformed restrictions,
and ambiguous conflict coupling. Lowered coupling has an explicit inactive
value, so execution never discards a runtime coupling error.

Family hashes use canonical term serialization and BLAKE3 (schema version 2;
the version-1 ring encoding remains available for migration checks). Generator
metadata (seed and index) is separate from terms, public traces, and family
hashes, and is copied explicitly into validity receipts for replay.

## Generated embodiment families

`GenerationSpec::generate()` emits deterministic
`GeneratedEmbodimentFamily` triples, not unrelated sampled worlds:

```text
body-limited witness
unrestricted control
environment-edge twin
```

The environment twin deletes the withheld actuator at a declared scope. The
current generator uses the conservative set of cells appearing anywhere in a
complete scored trajectory, then calibrates beyond that set so the
body/environment provenance remains publicly observable while all scored
behavior is preserved.

Each family carries an `EmbodimentValidityReceipt`. It is an exact finite
semantic filter, with these predicates:

- `sequences_checked`: number of complete action sequences enumerated;
- `topology_total`, state/row counts, degree sequence, and cycle diagnostic:
  executable finite-topology evidence;
- `twin_scope`: exact command sites or conservative trajectory cells;
- `family_hash`, `generator_seed`, and `generator_index`: semantic identity and
  separate replay coordinates;
- `goal_differs_from_start`: the requested goal is nondegenerate;
- `body_limitation_changes_ceiling`: the limited body and unrestricted control
  have different exact ceilings;
- `twin_trajectories_equal`: every body/twin transition trajectory agrees;
- `twin_values_equal`: every body/twin sequence value agrees;
- `twin_optimal_sequences_equal`: their exact ceiling and complete optimal
  sequence set agree;
- `twin_publicly_distinct`: the preserving twin differs through its public
  calibration prelude rather than comparing an episode with itself;
- `calibration_identifies_body`: the shared query algebra reports ambiguity
  diameter two without the prelude and one with it;
- `valid`: conjunction of the required family predicates above.

A candidate that fails the receipt filter is not emitted.

## Verification

Run locally from this worktree:

```powershell
cargo fmt --all -- --check
cargo test -p pretraining-term-compiler
cargo test --workspace --locked
```

The focused compiler suite covers typed-boundary rejection, fallback scoring,
calibration and restoration clocks, total coupling lowering, deterministic
receipt-filtered ring and branching-graph generation, explicit non-cycle
diagnostics, graph self-loop defaults, public/privileged separation, and
semantic hash migration. The Card 03 differential harness compares all 12
existing contracts over all 25 scored action sequences, including fallback and
timed restoration. A separate topology harness exhausts the five-state
non-ring witness's 16 horizon-two sequences.
These results are evidence for this isolated compiler boundary only.

## Remaining limits

- The executor supports finite deterministic rings and explicit graphs; it is
  not a stochastic geometry engine or general process scheduler.
- No compiler term renders through the learner event boundary, and no card has
  migrated from its handwritten evaluator.
- The Card 03 harness has not established full portfolio ambiguity/orbit or
  renderer parity; it is not an admission or transfer result.
- Interrupt execution currently freezes public process signals for a declared
  frozen displacement. It does not yet model general displaced-process state,
  continuation, or restart semantics.
- No learner run, GPU run, remote submission, or main-worktree integration is
  authorized by this spike.
