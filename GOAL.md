# Goal

## One sentence

Abstract pretraining should induce **transferable embodied capabilities** by
decomposing goal-directed behaviour into testable capabilities, expressing the
required relations as composable processes, and training across procedurally
generated world families.

## Active objective

Build the first coherent abstract robot pretraining system using a bounded
calibrated goal-reaching family and one shared content-token interface.
Astra's pre-run review found V1 solvable by a calibration-blind positive-axis
controller; V2 removes that shortcut with independently signed/permuted sensor
and actuator frames. Sensor widths 8/10 are small experimental fixtures, not
architecture constraints. The user authorized CPU/GPU execution after the
bounded correction and final readiness checks.

```text
common content boundary + public trajectory
  -> CPU boundary and shortcut evidence
  -> reviewed fixed configuration
  -> one fixed-mixture abstract pilot
  -> acquisition/scale/scheduler diagnosis if warranted
  -> matched grounding experiment
```

The pilot tests calibrated goal-conditioned reaching within one sampled family,
not active experimentation, robust disturbance rejection, compositional
pretraining, or transfer. The original broader objective below remains binding.
Historical R3/R10/R11 gates are preserved evidence, not current blockers.
`FIRST-TRAINING-SYSTEM.md` fixes the configuration and interpretation;
`LEARNER-INPUT-ARCHITECTURE.md` fixes the common continuous content boundary.

## The claim to test

A core pretrained across admitted abstract worlds should learn a held-out
abstract or grounded world faster than:

- an identically initialized scratch core; 
- the same core given a matched amount of alternative pretraining.

The three arms must use the same downstream observations, action adapter, data,
optimizer, and budget. The evidence is the learning curve and retention of
earlier capabilities, not source-world score alone.

## Target capabilities

The source worlds are intended to develop reusable organization for:

- identifying an unfamiliar body, interface, and process dynamics;
- carrying action-conditioned predictive state;
- regulating toward a requested outcome under disturbance or partial
  observation;
- taking actions for information when information changes later control;
- binding persistent causes and preserving behaviour under irrelevant
  realization changes;
- maintaining, inhibiting, and switching goals without violating constraints;
- composing, monitoring, interrupting, and recovering behaviour; and
- inferring an outcome from another agent's physical demonstration and
  realizing it through a different body.

The first four development fronts are identification/regulation, epistemic
action, norm/selective control, and binding/invariance. Composition and physical
prompting are dependent regions opened after their prerequisites are measurable.

## Method

```text
capability claim
  -> minimal behavioural contrast
  -> process composition and information boundary
  -> procedural world family
  -> semantic audit and learner pilot
  -> multi-world abstract pretraining
  -> matched held-out transfer and retention test
```

A world is an interaction among a body, environment, norm, disturbance,
optional other agents, and optional scaffold. The learner receives only public
events. Privileged state may be used by verification and diagnostics but may not
silently become learner input or supervision.

## Boundaries

- The core is modality-free because every symbolic, visual, linguistic,
  proprioceptive, and action realization is adapted into the same canonical
  content-token interface. The symbolic G0 payload is one modality input, not a
  privileged primary route. Modality adapters may participate during abstract
  pretraining as well as downstream transfer; pixels or words may not be
  deferred to an otherwise untrained side channel.
- A demonstration is a public temporal history whose observations and actions
  use those same modality and embodiment adapters. Actor/body addressing may
  preserve provenance, but it is not a substitute for perceptual content.
- Abstract success does not establish visual or physical grounding. Only a
  matched downstream transfer experiment does.
- A card is an experimental contrast, not a curriculum stage or capability
  claim by itself.
- A generated instance is not a new family. A semantic change creates a new
  family version and requires a new audit.
- No particular internal representation is required. Capabilities are defined
  by public histories, actions, outcomes, acquisition speed, and retention.
