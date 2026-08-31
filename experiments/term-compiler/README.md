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
- `EnvironmentTerm`: start configuration and local blocked edges;
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

Family hashes use canonical term serialization and BLAKE3. Generator metadata
(seed and index) is separate from terms, public traces, and family hashes.

## Generated embodiment families

`GenerationSpec::generate()` emits deterministic
`GeneratedEmbodimentFamily` triples, not unrelated sampled worlds:

```text
body-limited witness
unrestricted control
environment-edge twin
```

The environment twin deletes the withheld actuator only at configurations that
are executable scored command sites. Its calibration prelude leaves that region
so the body/environment provenance remains publicly observable while all
scored behavior is preserved.

Each family carries an `EmbodimentValidityReceipt`. It is an exact finite
semantic filter, with these predicates:

- `sequences_checked`: number of complete action sequences enumerated;
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
receipt-filtered generation, public/privileged separation, and semantic hashes.
The Card 03 differential harness compares all 12 existing contracts over all
25 scored action sequences, including fallback and timed restoration. It is
evidence for this isolated compiler boundary only.

## Remaining limits

- The executor is finite ring G0 only; it is not a general process scheduler.
- No compiler term renders through the learner event boundary, and no card has
  migrated from its handwritten evaluator.
- The Card 03 harness has not established full portfolio ambiguity/orbit or
  renderer parity; it is not an admission or transfer result.
- Interrupt execution currently freezes public process signals for a declared
  frozen displacement. It does not yet model general displaced-process state,
  continuation, or restart semantics.
- No learner run, GPU run, remote submission, or main-worktree integration is
  authorized by this spike.
