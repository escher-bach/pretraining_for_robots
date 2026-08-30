# Apparatus

This document describes what the code does and how to run it.

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

The finite-G0 Python boundary exposes one deduplicated corpus and mixture API
for Cards 04, 03, 02, 05, and 06. Family names, aliases, hashes, and accounting
indices remain evaluator-only metadata and never enter model tensors.

The checked-in model profile has 12 layers, width 384, six attention heads,
SwiGLU width 1024, payload width 8, action horizon 16, and context limit 2048.
The transformer body is randomly initialized. No pretrained language, vision,
or robot weights are loaded.

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

The fixed CPU pilot and later lineage contracts are
`configs/r10/seed_gate_cpu.toml` and `configs/r10/lineage_contract.toml`.
Run the timing decision before any pilot is scored:

```powershell
$env:PYTHONPATH = (Resolve-Path python).Path
python -m pretraining_experiments.seed_gate `
  --config configs/r10/seed_gate_cpu.toml `
  --output-root artifacts/r10/seed-gate `
  --preflight-only
```

The current local receipt is unscored: four selected-core updates at the
per-family maximum padded length took 8.43 seconds, over the fixed three-second
limit. Do not run the CPU pilots after that verdict. The fixed one-T4 execution
contract is `configs/r10/seed_gate_t4.toml`; launch it through the same exact-SHA
control plane:

```powershell
python tools/kaggle_run.py launch --experiment r10-seed-gate
```

That first T4 run completed but is retained as an apparatus result, not learner
evidence: row-wise action L1 decreased while the declared grouped ActionQuery
argmax did not improve, and iterable prefetch overstated presentations. The
only authorized follow-up preserves every scientific setting and replaces just
that objective with standard categorical cross-entropy over the public query
alternatives; group addresses remain loss-only metadata. Use the versioned
contract and registry entry (never overwrite the first receipt):

```powershell
python tools/kaggle_run.py launch --experiment r10-seed-gate-grouped
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
