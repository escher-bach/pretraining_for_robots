# Apparatus

This document describes what the code does and how to run it.

## Active first-system apparatus

The active implementation route is `CalibratedReachV2`, a small sampled
trajectory world in
`python/pretraining_experiments/trajectory_world.py`. It is intentionally
separate from the persisted finite-G0 corpus. Each episode contains public
reset, goal, calibration, observation, action-query, executed-action, and
episode-end events with explicit physical time. Action labels and the latent
calibration receipt are separate objects. No transition table is generated or
sent to the learner.

The public teacher accepts only `PublicHistory` and public action bounds. It
estimates the action-to-sensor map from public calibration using
`scipy.optimize.lsq_linear`, solving a declared bounded residual-plus-
minimum-effort objective. This is a public-history behavior teacher, not an
optimal privileged planner. Optional disturbance timing is hidden from the
teacher and only affects the sensed trajectory.

The common learner boundary is in
`python/pretraining_experiments/common_content.py`. Numeric sensor, goal, and
past-action adapters produce shared content tokens while role, key, physical
time, serialization position, and availability remain structural. The
existing `model.py` payload path remains historical G0 apparatus; it is not
the first-system path.

The checked-in `configs/first_training_system_cpu.toml` is an apparatus smoke
profile: it records the small common core and the deterministic trajectory
mixture used by `FirstTrainingConfig`, then checks save/resume. The scientific
entrypoint is selected explicitly with `mode = "scientific"`; the bounded
review profile is `configs/first_training_system_scientific_cpu.toml`. The
combined CPU readiness command is
`python tools/audit_training_system.py`; its receipt passes the current CPU
boundary. GPU access is authorized for the later fixed profile. The scientific
smoke path now propagates world fields and writes a public closed-loop receipt,
and the checked-in two-update smoke completes in 57.24 seconds with 32 cells,
updates 0/1/2, and closed-loop updates 0/2. The fixed 20-update profile
completed on CPU and on the verified one-T4 GPU run. Its compact GPU receipt is
`audit/runs/pretraining-first-system-f8bcb79/receipt.json`; source and
configuration identities are recorded in
`artifacts/first-training-system/GPU-RESULT.md`. Primary success is evaluator-only physical
state error; public sensor RMSE (`L2 / sqrt(sensor_width)`) is a diagnostic so
8- and 10-channel cells are comparable.
The current public-event contract is synchronous (`available_at == time`);
delayed availability is rejected until an explicit learner availability mask is
implemented and audited. The apparatus smoke is not the first science run.

## Rust workspace

| Crate | Function |
|---|---|
| `pretraining-world` | Generates the calibrated-monomial family, public tokens, public-prefix teacher targets, verifier results, deterministic trajectories, and batched online rollouts. |
| `pretraining-goal-conditioned-world` | Implements an exact five-position goal-swap diagnostic, presentation controls, policy baselines, and the evidence classifier used to distinguish local behaviour from transfer. |
| `pretraining-eviction-world` | Implements the disjoint container-eviction process, exact policy enumeration, hidden-goal ceiling, serialization controls, and reference audit. |
| `pretraining-canonical-event` | Defines typed public records and separate supervision addresses; renders to and decodes from the eight-float event layout under an explicit interpretation profile. |
| `pretraining-profiled-event` | Prefixes a learner-visible profile record, validates the envelope, and reconstructs the exact underlying event sequence. |
| `pretraining-g0-contract` | Supplies the finite ring environment, symmetry transforms, fragment trait, exhaustive sequences, public/privileged bounds, ambiguity gap, orbit checks, baseline isolation, and contract hashing. |
| `pretraining-card04-norm-swap` | Implements Card 04 on the G0 layer and emits its exact audit report. |
| `pretraining-card06-perceptual-organization` | Implements the exact two-source Card 06 binding family, shared coupling/interruption semantics, controls, orbits, rendering, and audit. |
| `pretraining-world-py` | Exposes batched public tensors and online rollout objects to Python through PyO3. Methods named `privileged_*` are evaluation-only. |

The data path is:

```text
Rust family -> public event records -> profiled float rows -> PyO3 batch
-> continuous event embedding -> Hugging Face LlamaModel
-> action/future heads -> Rust rollout -> verifier/evaluator
```

Rust owns transitions, teacher/verifier semantics, targets, replay, and rollout
state. Python does not reproduce those rules.

This is the implemented symbolic apparatus, not the accepted universal
modality architecture. `LEARNER-INPUT-ARCHITECTURE.md` requires the
eight-float projection to become the G0 adapter into one common `[B,T,H]`
content interface shared by symbolic, visual, linguistic, proprioceptive, and
action realizations. The experimental optional continuous-content sidecar is a
backward-compatible probe only and may not be treated as the future grounding
path.

## Rust commands

```powershell
cargo fmt --all -- --check
cargo test --workspace --locked
cargo run -p pretraining-card04-norm-swap --bin card04-audit
cargo run -p pretraining-card06-perceptual-organization --bin card06-audit
cargo run -p pretraining-canonical-event --bin schema
cargo run -p pretraining-eviction-world --bin eviction-audit
cargo run -p pretraining-goal-conditioned-world --bin audit
```

The audit binaries write JSON to standard output.

## Python training surface

| Module | Function |
|---|---|
| `model.py` | Wraps the maintained `LlamaModel` body with continuous event embeddings and action/future regression heads. |
| `data.py` | Converts Rust batches to tensors and checks world/model ABI compatibility. |
| `train.py` | Uses Transformers `Trainer` for optimization, scheduling, accumulation, mixed precision, distributed execution, checkpointing, and resume. |
| `evaluation.py` | Computes held-out error distributions, thresholds, trial curves, and paired before/after deltas. |
| `goal_conditioning.py` | Runs the line-world diagnostic and representation probe through the real learner interface. |
| `eviction_evaluation.py` | Runs full eviction episodes, first-action diagnostics, hidden-goal checks, renaming/order checks, and throughput. |
| `baselines.py` | Evaluates fixed scaled-oracle policies before scheduling training. |
| `benchmarks.py` | Measures batched Rust generation and binding throughput. |
| `runner.py` | Performs the non-interactive Kaggle phase sequence and packages compact evidence. |
| `seed_gate.py` | Validates the immutable R10 contracts, runs the selected-core timing gate and bounded per-family pilots, evaluates grouped G0 decisions, and writes non-transfer receipts. |
| `common_content.py` | Implements the versioned shared content-token boundary, variable-width adapters, structural time/address inputs, and embodiment action decoder for new trajectories. |
| `trajectory_world.py` | Generates the public-only CalibratedReachV2 sensor/goal/action trajectories, public-history teacher, and private calibration receipt. |
| `first_training.py` | Builds the common-content bundle, runs the bounded CPU Trainer save/resume smoke, and dispatches the explicit TOML scientific path; the bounded two-update receipt passes and the reviewed fixed 20-update diagnostic is authorized. |
| `tools/audit_training_system.py` | Runs the CPU public/private, calibration, permutation/ablation, embodiment, model continuation, and closed-loop apparatus checks and writes the compact receipt. |

The finite-G0 Python boundary exposes one deduplicated corpus and mixture API
for Cards 04, 03, 02, 05, and 06. Family names, aliases, hashes, and accounting
indices remain evaluator-only metadata and never enter model tensors.

The checked-in model profile has 12 layers, width 384, six attention heads,
SwiGLU width 1024, payload width 8, action horizon 16, and context limit 2048.
The transformer body is randomly initialized. No pretrained language, vision,
or robot weights are loaded.

The persisted compiler corpus uses `g0_loader.py` and standard PyTorch
`Dataset`/`DataLoader` collation. Its model dictionary includes the nine raw
tensor fields plus loss-only `action_decision_groups`; these select grouped
categorical supervision and never enter the backbone. See
`artifacts/g0-learner-input/R3D-REVIEW.md` for a fresh episode, its reproducible
CPU inspection, and the remaining modality migration requirements.

## Build the Python extension and run tests

Install the pinned Python dependencies in an environment that already contains
a compatible PyTorch build. Kaggle supplies CUDA PyTorch, so it is intentionally
absent from `requirements-kaggle.txt`.

```powershell
python -m pip install -r requirements-kaggle.txt
python -m maturin build --release --locked `
  --manifest-path crates/world-py/Cargo.toml --out dist
$wheel = Get-ChildItem dist/pretraining_world_py-*.whl | Select-Object -First 1
python -m pip install --force-reinstall --no-deps $wheel.FullName
$env:PYTHONPATH = (Resolve-Path python).Path
python -m unittest discover -s python/tests -v
```

## R10 seed gate

The preserved CPU pilot and later lineage contracts are
`configs/r10/seed_gate_cpu.toml` and `configs/r10/lineage_contract.toml`.
The CPU timing command remains reproducible, but R10 is closed and it is not an
instruction to score another pilot:

```powershell
$env:PYTHONPATH = (Resolve-Path python).Path
python -m pretraining_experiments.seed_gate `
  --config configs/r10/seed_gate_cpu.toml `
  --output-root artifacts/r10/seed-gate `
  --preflight-only
```

The preserved local receipt is unscored: four selected-core updates at the
per-family maximum padded length took 8.43 seconds, over the fixed three-second
limit. Do not run the CPU pilots after that verdict. The fixed one-T4 execution
contract was `configs/r10/seed_gate_t4.toml`:

```powershell
python tools/kaggle_run.py launch --experiment r10-seed-gate
```

That first T4 run is retained as apparatus evidence: row-wise action L1 did not
optimize declared grouped ActionQuery argmax, and iterable prefetch overstated
presentations. The one permitted repair used standard categorical cross-entropy
over public query alternatives; group addresses remained loss-only metadata.
Its versioned contract and registry entry were:

```powershell
python tools/kaggle_run.py launch --experiment r10-seed-gate-grouped
```

It is complete and decisive; do not run this command again for R10. Verified
receipt: `audit/runs/pretraining-r10-seed-gate-grouped-e2dc185/receipt.json`.
At source `e2dc1856ab56e45f55d5fa01e63d0bd0f90035b6`, CUDA preflight was 3.0558
seconds for four updates and 0.7171 seconds for full evaluation. Each family
used 256 consumed presentations. Card 02 was frontier-admitted; Cards 04, 03,
05, and 06 were valid but inconclusive, so the classifier is
`seed_gate_incomplete` and R11 remains blocked. The structural gate checks are
audited; the `0.80` final, `0.25` gain, `0.60` every-case-kind, and cadence
barriers were predeclared heuristic/unpowered decision thresholds. The
configuration's `primary` field name is stale: classification actually checked
all `by_case_kind` values, with no outcome changed.

## R10a Card 06 compatibility scale

`configs/r10/card06_compatibility_scale_t4.toml` is a separate diagnostic
profile, not an R10 retry. It preserves Card 06 contract `76a08f38947c8cae`,
the selected core, grouped raw-logit objective/evaluator, seeds, batch, and
optimizer, while declaring a new 256-update cosine horizon. Decision rungs are
64, 128, and 256 updates; the receipt records the earliest exact full-support
fit and whether it remains exact through all later rungs. Stable exact fit may
only open a separately declared generalization profile. Unstable or incomplete
fit sends Card 06 to optimization diagnosis, decomposition, or deferral. None
of these outcomes changes R10 or authorizes R11.

The first submitted scale kernel at source `a9f118018261241812b991811cb33aebf51b1f7c`
stopped before training: the runner resolved the pinned relative model-config
path before the strict profile comparison. Its failed receipt is preserved.
The bounded apparatus repair permits only that equivalent resolved path and
keeps all scientific settings and other execution fields fixed.

The repaired profile completed at source
`f4fd45edcda699f7a2e1fe4ec54c1a0a5117a2fc`. Its fixed classification is
`support_fit_incomplete`: exact full-support fit was false at all three rungs.
The verified receipt is
`audit/runs/pretraining-r10a-c06-scale-f4fd45e/receipt.json`. Do not launch the
profile again; its prospective action is Card 06 decomposition or deferral.

The completed command was:

```powershell
python tools/kaggle_run.py launch --experiment r10a-card06-compatibility-scale
```

At repository creation, 116 Rust tests and 37 Python tests pass.

## Kaggle control plane

Prerequisites:

1. install and authenticate the official `kaggle` CLI;
2. add a public HTTP(S)-reachable Git remote named `origin`;
3. commit and push the exact source to run; and
4. obtain explicit user authorization for the declared GPU run.

The registry is `kaggle/experiments.toml`. Its current entry is the preserved
single-world apparatus run; it is not the future multi-world run contract.

```powershell
python tools/kaggle_run.py launch --experiment single-world-apparatus
python tools/kaggle_run.py status --kernel <owner/slug>
python tools/kaggle_run.py logs --kernel <owner/slug>
python tools/kaggle_run.py collect --kernel <owner/slug>
```

Use `run` instead of `launch` to submit, wait, and collect in one process:

```powershell
python tools/kaggle_run.py run --experiment single-world-apparatus
```

`launch` verifies that `HEAD` is a full SHA reachable on `origin`, derives the
public clone URL, writes a temporary three-cell notebook, and submits it through
`kaggle kernels push`. The notebook clones into `/tmp/pretraining-runtime`,
checks out the exact SHA, and invokes `pretraining_experiments.runner` once.

The runner performs:

```text
environment capture -> pinned install and Rust wheel build
-> Rust/Python tests -> world validation -> CPU benchmark
-> trivial-policy band -> exact two-T4 Trainer run
-> checkpoint/resume -> evaluation -> compact manifest
```

Only declared results are written under `/kaggle/working/pretraining-results`.
`collect` downloads selected JSON and bounded logs, verifies them against the
remote manifest and launch record, and writes a receipt under `audit/runs/`.
Model checkpoints and recovery payloads remain on Kaggle.
