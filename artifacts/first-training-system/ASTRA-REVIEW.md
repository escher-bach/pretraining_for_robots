# Astra pre-run review

**CPU decision: GO for the fixed 20-update calibrated goal-reaching diagnostic.**
The original V1 identification experiment was vetoed. The bounded V2 repair
passes the world-only audit with zero failures/gaps and no optimization.
GPU training additionally requires the trainer/control-plane execution checks;
this review does not certify a CUDA execution that has not occurred.
The user authorized both runs; no new permission is required.

## Decision-changing evidence

On the exact 32-cell held-out cohort, V1's public fixed reactive controller
`u[:2] = clip(8*(public_target-current_sensor)[:2], -.5, .5)` succeeds 32/32
without identifying a calibration map. The public teacher also succeeds 32/32.
Sensor rows 0/1 and actuator columns 0/1 preserve positive coordinate axes in V1.
The audit's missing-calibration teacher exception was an API precondition,
not evidence that calibration is necessary for control.

V2 independently signs/permutates sensor and actuator frames, with offsets
transformed consistently, preserving singular values and public calibration.
The actual V2 cohort gives teacher 32/32 (mean physical error 0.000688), fixed
reactive 7/32 (0.354527), and inaction 4/32 (0.122437). A B/-B signed-action
coordinate witness preserves observations/goals and physical effects; after
removing calibration action records, first-query histories are identical but
public teacher actions differ by 1.01008 in L2. This is real information evidence,
not a missing-field rejection or an optimal calibration-blind policy bound.

Evidence: `astra-baseline-pre-review.json` preserves the initial 6.88-second
V1 check and in-memory signed-frame prototype. `astra-baseline-v2.json` and
`audit-world-only.json` contain actual V2 results, all 96 baseline rollout rows,
the information witness, eight world boundary cases, semantic permutations,
paired embodiment checks, and a bounded 12-step nonlearning common-model rollout.
The audit performs no optimizer updates and does not repeat checkpoint resume.

## Architecture and generation assessment

Widths 8/10 were arbitrary examples. The shared temporal core already supports
variable-width adapters. The restriction was in world validation, adapter
registration and tensor padding; these now derive from configuration while
8/10 remain a fixed profile fixture. Two/four actuators remain declared body
families, and latent state remains 2D. Broader width experimentation is unnecessary.

This is procedural parameter sampling within one family. It does not yet
implement compositional generation of new process structures. G0's typed
compiler remains useful, but its corpus renderer publishes the full transition
table (`crates/g0-corpus/src/lib.rs`, `episode_for`); it is not interchangeable
with an unknown-interface trajectory renderer. Retain this small continuous
seed and the reusable shared boundary/Trainer. Before portfolio expansion,
the next family must demonstrate reuse of the public trajectory, adapter,
training and audit seams, with only the new process semantics added. Do not
build a general compiler or another independent family stack before this pilot.

## Fixed interpretation and execution limits

The run tests initial acquisition of calibrated goal-conditioned reaching
within one generated family. Complete noiseless basis calibration precedes
scored decisions. It cannot establish active experimentation, separate body
and environment identification, predictive state, compositional generalization,
new-adapter transfer or grounding. Post-calibration action history is not
necessary for the exact memoryless plant. Ablation degradation can also reflect
input distribution shift; it does not alone prove an information bound.

Disturbance increments are at most 0.002 per axis per step; their worst-case
accumulated norm across 12 steps is below 0.034, below the 0.05 success tolerance.
Success therefore does not establish disturbance rejection. Twenty updates at
batch 2 consume 40 presentations from a 64-episode pool, not equal exposure to every
cell. Same-prior held-out seeds test within-family generalization only.

Keep the fixed budget and paired before/after closed-loop controls. Lower
teacher-forced action loss alone can reflect predicting small late-episode
actions. Stop on apparatus/boundary failure, non-finite values, missing
checkpoints/evaluation support, or the fixed update budget. Preserve receipts.
An inconclusive 20-update curve is not a failure of the broad project hypothesis
and does not automatically authorize scaling, a scheduler or grounding.

Before GPU execution, verify model-device batch placement in both evaluation
paths, the matching fixed profile, removal of unevaluated transform-holdout
claims, exact committed source/config identity, and compact receipt collection.

## Final GPU readiness review — 2026-09-12

**GO for the authorized fixed 20-update pilot**, following the completed CPU
diagnostic. This bounded code review found no blocking defect in explicit CUDA
selection, Trainer fp16 configuration, evaluation-batch placement, manual
evaluation autocast, scientific runner dispatch, root-seed resolution, or
scientific receipt collection. The runner pins the training subprocess to one
visible GPU and removes inherited distributed ranks, preserving batch 2 and
40 episode presentations even when Kaggle exposes two T4s. The GPU profile
retains the reviewed world, model, optimizer and evaluation support. CUDA
execution itself has not been verified locally; the launch must use the exact
committed source/configuration and retain its execution receipt.

The episode's calibration actions are actual sequential environment
interventions: each applies the declared dt-scaled dynamics and publishes the
resulting calibration observation. Positive/negative basis pairs return the
state to its initial value. They are not hypothetical private-map lookup rows.
However, every calibration event is stamped at time zero. This collapses the
duration of the supplied precontrol calibration scaffold; serialization order
preserves its action/observation sequence. The narrow pilot therefore tests
control after supplied noiseless calibration, not identification through a
physically timed exploration trajectory. The time field is not a faithful
physical clock during this prefix. Retain that limitation when interpreting
the run; any later timing or active-experimentation claim requires an explicit
calibration-time contract and fresh evidence. Episode review views must include
calibration observations and distinguish public events from loss-only targets
and private audit annotations.
