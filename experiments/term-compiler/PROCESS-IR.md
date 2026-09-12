# Continuous process compiler bridge

`pretraining_term_compiler::generate_processes(seed, count)` emits a mixed
corpus of `GeneratedProcess` records. The executable bridge is:

```powershell
cargo run -q -p pretraining-term-compiler --bin process-export -- --seed 7 --count 12 > process-programs.json
```

Each record has three layers:

* `process.nodes` and `process.wiring` are a typed, validated composition;
* `process.public_schema` describes the existing public trajectory event ABI;
* `process.runtime_private` is for the simulator only and is never copied into
  `PublicEvent`, content tokens, supervision, or the trainer batch.

The generator enumerates the complete bounded product of the three optional
operators, cycling through eight legal compositions:

* `Reaching`: action -> plant -> observation;
* `ActuatorLag`: add actuator lag;
* `GoalSwitch`: add a timed goal switch;
* `Disturbance`: add a disturbance process;
* `ActuatorLagGoalSwitch`, `ActuatorLagDisturbance`, and
  `GoalSwitchDisturbance`: each corresponding pair; and
* `LagGoalSwitchDisturbance`: all three operators together.

The generator selects sensor width, actuator width, and horizon independently
for each record. These are structural compositions within this explicit
primitive library; arbitrary DAGs and a general process scheduler remain
unsupported and are rejected by validation.

The seed and index live only in `GeneratedProcess` replay metadata. They are
not fields of the compiled process and do not appear in `public_schema`.
The Python world should select the record, pass `runtime_private` to its
private simulator, and publish the ordinary public history (`sensor`, `goal`,
`action_query`, `action_executed`, `episode_end`). The learner receives only
that history through the existing common-content adapter.

Compilation lowers `runtime_private` from the validated `Plant`, `ActuatorLag`,
`GoalSwitch`, and `Disturbance` nodes. It does not keep a second family preset
as the source of dynamics. Every non-input internal output and every declared
input must participate in exactly one validated wire; removing the lag edge is
therefore a compile error. The Python `ProcessIR.from_compiled_process`
adapter reads the same node graph and cross-checks the serialized lowering.

The prior `WorldTerm` machinery is reused for the part it actually provides:
typed named ports, visibility checks, deterministic serialization, and
validated composition. Its finite symbolic executor cannot integrate the
continuous 2-D state used by this trajectory family, so numerical integration
remains in the Python runtime behind this compiled boundary. No claim is made
that the old finite transition executor runs these continuous terms.

Validation rejects duplicate endpoints, mismatched port kinds, non-DAG wiring,
public outputs from internal nodes, invalid lag coefficients, and missing
plant/observation nodes. The Rust tests cover deterministic replay, JSON
round-trip, all eight operator compositions, structural family differences,
disconnected wiring rejection, and the private/public boundary.
