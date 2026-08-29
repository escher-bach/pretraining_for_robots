# Development Path

## Working vocabulary

| Term | Meaning |
|---|---|
| Capability | A behavioural relation defined over public histories, actions, and outcomes. |
| Trunk | An independently measurable development front. |
| Card | A bounded contrast with a witness, surgical negatives, invariances, baselines, and a transfer prediction. |
| Process | A typed state-transition component with public and hidden ports. |
| Family contract | Versioned semantics, parameter domain, information boundary, goals/costs, interventions, invariances, and solver status. |
| World family | A procedural distribution generated from one family contract. |
| World instance | One seeded parameter draw from a family. |
| Audit | Semantic, information-boundary, baseline, ambiguity, and invariance checks for one family version. |
| Frontier family | An audited family whose learner-facing pilot yields a usable progress signal. |
| Checkpoint state | Weights plus evidence, admitted families, retention state, remaining budget, and the progress chart. |

## Embodied process model

```text
World = Body || Environment || Norm || Disturbance || OtherAgents || Scaffold
Body  = Morphology || Actuation || Sensorium
```

The learner connects through typed observation and action ports. Components may
be coupled, sequenced, hidden, revealed, renamed, restricted, delayed, fed back,
repeated, or interrupted. These operators are introduced only when a card needs
their semantics; there is no general-purpose language to build in advance.

Every family declares three information views:

- **public:** schema, prior, goal when public, and learner-visible history;
- **privileged:** instantiated hidden state for verification and upper bounds;
- **generator:** seeds and construction metadata for coverage and replay.

The reusable semantic questions are monitoring, identification or
distinguishability, reachability, value bounds, strategy where supported, and
metamorphic invariance. Small finite families use exact enumeration. Harder
families return bounds or monitor-only status; no universal solver is assumed.

## Capability decomposition

| Trunk | Capability | Required instrument |
|---|---|---|
| T1 | Identification and regulation | Separate identification from outcome control; vary body/dynamics while holding the task relation fixed. |
| T2 | Epistemic action | Compare an informative action with an equally costly non-informative action. |
| T3 | Norm structure and selective control | Freeze state/history and change only the requested outcome or constraint. |
| T4 | Binding, invariance, and relational abstraction | Pair semantics-preserving orbits with surface-similar semantics-changing transformations. |
| T5 | Temporal composition | Recombine closed-loop procedures in unseen orders, with interruption and recovery. Depends on T1. |
| T6 | Other agents and physical prompting | Separate demonstrated outcome from demonstrated motion across different bodies. Depends on T1 and T4. |
| B | Continual transfer and retention | Compare learning curves across checkpoints and retain earlier sentinels. This is a regime over every trunk. |

## Cards

Cards are candidate instruments, not mandatory curriculum stages. The first
seed portfolio uses Cards 04, 03, 02, 05, and 06 because together they cover T1
through T4 with finite, CPU-auditable machinery.

| Card | Intended result | Fragment | Path role |
|---|---|---|---|
| 01 Regulation | Feedback rejects disturbance and recovers; a fixed action plan does not. | G1 continuous/noisy | Open after the finite seed unless T1 evidence requires it sooner. |
| 02 Predictive state | Earlier public information is retained exactly when it changes action-conditioned futures. | G0 finite | T1 seed family. |
| 03 Affordance | Behaviour changes with body-relative reachability before a failed attempt reveals it. | G0 finite | T1 seed family and next shared-contract implementation. |
| 04 Norm swap | With history fixed, changing the requested outcome changes action; maintain, inhibit, switch, and viability remain separate contrasts. | G0 finite | T3 seed family; semantics and audit are implemented. |
| 05 Active experimentation | A diagnostic probe is selected for later information value, not immediate progress. | G0 finite | T2 seed family. |
| 06 Perceptual organization | Persistent causes are bound across absence and relabelling while false grouping controls fail. | G0 finite | T4 seed family and first abstract-to-vision contract. |
| 07 Composition | Closed-loop skills are recombined, interrupted, resumed, monitored, and recovered. | G2 bounded | T5 dependent family. |
| 08 Physical prompting | The learner infers another agent's outcome and realizes it through a different body. | G2 bounded | T6 dependent family. |

Every implemented card version must provide: a minimal witness; only the
negatives needed to isolate its claim; preserving and changing transformations;
public and privileged bounds; ambiguity; shortcut checks; a versioned contract
hash; a learner-facing rendering; and a downstream falsifier.

## Family lifecycle

```text
proposed -> candidate -> audited -> frontier -> replay/sentinel -> archived
```

- `candidate`: executable contract and minimal witness exist.
- `audited`: semantics, information boundary, baselines, ambiguity, negatives,
  replay, and invariance pass.
- `frontier`: a bounded pilot proves the common learner/evaluator can interact
  with the family and exposes a nontrivial learning-progress signal.
- A semantic change creates a new version. A new seed does not.
- A failed family gets one bounded repair when the claim stays unchanged;
  otherwise it is deferred or archived and does not block unrelated trunks.

## Seed gate

Multi-world pretraining begins when:

1. frontier families cover T1, T2, T3, and T4;
2. the families express genuinely different process relations through one
   self-describing learner event boundary;
3. each family has a valid baseline/ceiling bracket and no direct leakage;
4. bounded pilots produce usable progress signals;
5. fixed-mixture, scratch, retention, and alternative-pretraining comparators
   are declared;
6. scheduling sentinels are separate from sealed transfer families; and
7. source versions, mixture accounting, checkpoint cadence, total budget, stop
   rules, and compact audit artifacts are fixed.

Cards 01, 07, and 08 and complete visual grounding are not prerequisites for
this gate.

## Progress chart

Every piece of work names one row.

| Row | State transition | Status | Completion evidence |
|---|---|---|---|
| R0 | Standalone repository and neutral namespace | Complete | Rust and Python suites pass from this root. |
| R1 | Common world/training apparatus | Complete | Rust generation and rollout, PyO3 tensors, maintained model body, Trainer checkpoint/resume, and Kaggle control plane exist. |
| R2 | Typed mixed-family event boundary | Complete for existing families | Canonical record plus explicit interpretation profile round-trip existing producers. |
| R3 | Shared finite G0 audit layer | Complete | Enumeration, bounds, ambiguity, invariance orbit, isolation bracket, and contract hash are reusable. |
| R4 | Card 04 semantic family | Audited | Exact 20-case audit, 27 action sequences per case, zero ambiguity gap, and full orbit tests. |
| R5 | Card 04 learner-facing family | **Next** | Render through the profiled event boundary and pass boundary/integration tests. |
| R6 | Card 03 affordance family | **Next** | Implement on G0, audit reachability/reveal/identification, then add learner rendering. |
| R7 | Card 02 predictive-state family | Pending R6 machinery | Exact history-ablation and ambiguity audit plus learner rendering. |
| R8 | Card 05 active-experimentation family | Pending R7 machinery | Exact value-of-information and matched-cost control plus learner rendering. |
| R9 | Card 06 perceptual-organization family | Pending R8 machinery | Source-binding controls and preserving/changing orbit plus learner rendering. |
| R10 | Four-trunk seed portfolio | Blocked by R5–R9 | Bounded per-family pilots and all seven seed-gate conditions. |
| R11 | Multi-world learner lineage | Blocked by R10 and GPU authorization | Matched fixed mixture and adaptive sessions with immutable checkpoints, retention, full cost, and stop rules. |
| R12 | Held-out transfer claim | Blocked by R11 | Learning-curve advantage over scratch and alternative pretraining with retention. |
| R13 | Dependent composition, prompting, and grounding | Evidence-gated | Cards 01/07/08 and visual or physical descendants opened by R11–R12 evidence. |

## Learner lineage after the seed gate

At each immutable checkpoint select one bounded action:

```text
TRAIN_NEW | TRAIN_REPLAY | TRAIN_FIXED_MIXTURE | EVALUATE
ADMIT_FAMILY | MIGRATE_CORE | STOP
```

The selected action fixes its parent checkpoint, family versions, mixture,
objective, budget, seed policy, monitors, and stop rule before execution. A new
family version can enter only at the next checkpoint boundary. The adaptive
lineage is retained only if its capability gain per total cost exceeds the
matched fixed-mixture comparator.

The immediate route is R5 and R6, followed by R7, R8, R9, then R10. Do not wait
for all eight cards before opening R11.
