# First viable training system

This is the fixed review contract for a bounded calibrated goal-reaching pilot.
The user authorized both the CPU diagnostic and the GPU run. Astra's review
vetoed the original V1 identification premise: a fixed public reactive policy
succeeds on all 32 held-out cells without estimating the calibration map.
The small V2 correction uses independently signed/permuted sensor and actuator
frames. Final nonlearning checks must pass before training starts; no further
user permission is required. Preserve the V1 receipts as historical evidence.

The active progress row is `User-reviewed fixed CPU run` in
`DEVELOPMENT-PATH.md`. Review findings and CPU baseline evidence are in
`artifacts/first-training-system/ASTRA-REVIEW.md` and
`artifacts/first-training-system/astra-baseline-pre-review.json`.

## Fixed profile and evidence

`configs/first_training_system_scientific_cpu.toml` fixes 20 optimizer updates,
batch size 2, learning rate 1e-3, zero weight decay, and the maintained Trainer's
cosine learning-rate schedule. There is no episode scheduler or online DAgger.
The 64-episode training pool contains two instances per cell of the 32-cell
factorial: sensor widths 8/10, actuator widths 2/4, absolute/reset-relative goals,
goal switch absent/step 3, and disturbance absent/present. The held-out pool has
one instance per cell, using seed `100000 + 7919 * cell_index`.

Twenty updates consume 40 presentations, less than one full training-pool epoch;
this does not guarantee equal exposure to every cell. Widths 8/10 are arbitrary
small fixtures for the common boundary, not universal modality restrictions.
Adapters and padding derive from the declared sensor widths. The world remains
two-dimensional and the supported actuator families remain 2/4.

The model has hidden size 64, intermediate size 128, two temporal layers, four
attention heads, and context length 512. Numeric sensors, goals, past actions,
and control records enter the same content-token boundary. Role, key, physical
time, serialization position, and availability are structural. The current
event contract is synchronous: `available_at` equals physical event time.

World horizon is 12, dt 0.1, action bounds [-0.5,0.5], calibration pulse 0.06,
zero observation noise, and disturbance scale 0.02. Teacher-forced masked action
loss is measured at updates 0/5/10/20; closed-loop physical error, success, sensor
RMSE and effort at 0/20. Policies include learner, public teacher, inaction,
fixed public reactive control, and learner ablations for goal, calibration,
and post-calibration action history. Primary success is evaluator-only physical
state error <=0.05. Public sensor RMSE is diagnostic, not the success target.

The V1 audit at `artifacts/first-training-system/audit.json` and scientific smoke
at `artifacts/first-training-system/scientific-smoke-final-review/` preserve
apparatus evidence, including checkpoint continuation. The latter completed
two updates in 57.24 seconds; it is not an acquisition result. The new
`audit-world-only.json` checks V2 world semantics, shortcut controls, information
witnesses, and a nonlearning public-prefix model rollout without optimization.
It does not repeat checkpoint continuation.

```powershell
$env:PYTHONPATH = (Resolve-Path python).Path
python tools/audit_training_system.py --world-only
python -m pretraining_experiments.first_training `
  --config configs/first_training_system_scientific_cpu.toml `
  --output-root artifacts/first-training-system/scientific-cpu-review
```

## Interpretation and stopping

The pilot tests whether a shared core begins acquiring calibrated,
goal-conditioned reaching within one procedurally sampled family. Complete
noiseless basis calibration precedes scored decisions and identifies the
public composite action-to-sensor map. This does not establish active
experimentation, separate body/environment identification, predictive state,
compositional generalization, new-adapter transfer, or grounding.

Disturbance contributes at most 0.002 per axis per step, less than 0.034 in
accumulated norm over the horizon, already below the 0.05 success tolerance.
Success on that arm cannot establish disturbance rejection. After calibration,
the exact plant admits current-observation feedback, so post-calibration action
history is not necessarily useful. Ablation degradation alone may include an
input-distribution shift; it is not an optimal information-bound proof.

Stop at the fixed budget, or earlier on an apparatus/boundary failure,
non-finite loss/actions, missing evaluation support, or failed checkpoint
verification. Preserve failed receipts. A short inconclusive curve is a
learner/support-fit diagnostic, not a verdict against the world or broad
pretraining hypothesis. Do not automatically scale, add a scheduler, or open
grounding from this pilot.

## GPU execution and public boundary

The GPU profile must match the reviewed semantics and evaluation support.
Manual evaluation tensors must follow model device placement; declarations
of unimplemented transform holdouts must be removed. The Kaggle control plane
requires a committed full SHA reachable from public origin and records exact
configuration identity. Checkpoints remain on Kaggle; collect compact verified
receipts. CPU/GPU authorization was present, and both fixed 20-update runs are
now complete. The verified GPU receipt is
`audit/runs/pretraining-first-system-f8bcb79/receipt.json` (source
`f8bcb7977ba77f9c5fb8f6c9f2d008efed1a7b9a`, configuration SHA256
`c9c1776db9a2b87f6ffa0e99602d3f844dae103ab0a8a903fff3c1cef203a183`). It
used one visible T4, world size 1, and 40 episode presentations. CPU and GPU
both reduced held-out action loss `0.025209 -> 0.016089` and mean physical
error `0.245424 -> 0.143723`, while success stayed `2/32` against inaction
`4/32`, fixed reactive `7/32`, and teacher `32/32`. The supplied calibration
prefix is sequential in event order, but its actual physical timestamps
collapse to zero, so timing-sensitive calibration evidence is not established.
The next smallest question is whether the policy learns informative early
decisions versus small late teacher actions; propose a bounded acquisition
diagnostic before considering any scale change.

Public events contain observations, goals, executed/calibration actions,
queries and boundaries. Queries contain no targets. Action supervision is
loss-only; latent state, maps, seeds, contract identity and generator metadata
stay in evaluator receipts and never enter the learner's feature tensors.
The teacher uses public calibration and maintained NumPy/SciPy linear algebra.
