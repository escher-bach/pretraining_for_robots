# Development Path

`EMBODIED-PROCESS.md` defines the capability graph and process semantics. This
document applies them to the current implementation route and progress state.

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

The learner connects through typed observation and action ports. The process
kernel supports directed wiring, declared shared-variable coupling, interruption
with explicit resume semantics, typed restriction, and guarded reveal. Norms
support conjunction, supersession, and priority. Hiding is the default port
view, delay is a buffer component, renaming belongs to invariance tests, and the
feedback loop closes through the learner rather than inside the world.

Every family declares three information views:

- **public:** schema, prior, goal when public, and learner-visible history;
- **privileged:** instantiated hidden state for verification and upper bounds;
- **generator:** seeds and construction metadata for coverage and replay.

The six reusable semantic questions are monitoring, identification,
reachability, value bounds, strategy where supported, and agent-centred
equivalence. Auditor-only operations check metamorphic invariance and
non-interference. Small finite families use exact enumeration. Harder families
return bounds or monitor-only status; no universal solver is assumed.

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

`CARDS.md` is the implementation authority for each card's witness, controls,
transformations, information boundary, baselines, admission rule, and transfer
falsifier. The table below is only the route through those specifications.

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
| R0 | Standalone repository and neutral namespace | Complete | Goal, meta-process, embodied process and capability graph, path, all eight card contracts, apparatus, and passing Rust/Python suites are available from this root. |
| R1 | Common world/training apparatus | Complete | Rust generation and rollout, PyO3 tensors, maintained model body, Trainer checkpoint/resume, and Kaggle control plane exist. |
| R2 | Typed mixed-family event boundary | Complete for existing families | Canonical record plus explicit interpretation profile round-trip existing producers. |
| R3 | Shared finite G0 audit layer | Complete | Enumeration, bounds, ambiguity, invariance orbit, isolation bracket, and contract hash are reusable. |
| R3a | Executable process kernel and query algebra | Complete | The five operators and the norm algebra are shared data in `pretraining-g0-contract::kernel`; identification, public/privileged policy ceilings, epistemic value, matched controls, non-interference, and history ablation are shared in `::query`. Card 04's evaluator is composed from them with a byte-identical audit. |
| R3b | One learner event boundary | Complete | `pretraining-g0-render` turns a card's public transcript into a canonical episode and then into profiled tokens. One profile serves the whole portfolio, so the envelope never publishes family identity. |
| R4 | Card 04 semantic family | Audited | Exact 20-case audit, 27 action sequences per case, zero ambiguity gap, and full orbit tests. |
| R5 | Card 04 learner-facing family | Complete | All 20 cases render through the shared finite-G0 profile on `physical-event-abi-0.3.1`, decode back exactly, and are taught by a published-information policy. The rendering found that the audited optimal first action on the two unannounced-switch witnesses was privileged; the corrected boundary is reported per case. |
| R6 | Card 03 affordance family | Complete | Nine-cell ring, two scored decisions, twelve cases. Calibration identifies the body exactly, the scored-phase ambiguity gap is zero, making calibration uninformative reopens it, both negatives isolate, all eleven orbit verdicts hold, and every case renders and round-trips. |
| R7 | Card 02 predictive-state family | Complete | Seven-cell ring, three decisions, ten cases. Ablating the latch costs exactly half the ceiling on the witness and nothing on either control; the required memory span is sharp at three; the aliasing interval separates no two modes under any probe; all sixteen value-orbit and four information-orbit verdicts hold. |
| R8 | Card 05 active-experimentation family | **Next** | Exact value-of-information and matched-cost control plus learner rendering. |
| R9 | Card 06 perceptual-organization family | Pending R8 machinery | Source-binding controls and preserving/changing orbit plus learner rendering. |
| R10 | Four-trunk seed portfolio | Blocked by R5–R9 | Bounded per-family pilots and all seven seed-gate conditions. |
| R11 | Multi-world learner lineage | Blocked by R10 and GPU authorization | Matched fixed-mixture and direct hill-climbing sessions over immutable checkpoints, with concurrent preparation permitted, retention, full cost, and stop rules. |
| R12 | Held-out transfer claim | Blocked by R11 | Learning-curve advantage over scratch and alternative pretraining with retention. |
| R13 | Dependent composition, prompting, and grounding | Evidence-gated | Cards 01/07/08 and visual or physical descendants opened by R11–R12 evidence. |

## Learner lineage after the seed gate

### Next steps through the first R11 lineage

The first R11 artifact is an abstract-pretrained checkpoint lineage. R12, not
R11, determines whether a retained checkpoint transfers to a held-out abstract
or grounded family.

R5--R9 are not invitations to add unrelated handwritten task generators. For
each card, first state the witness and controls as compositions of the process
kernel's shared environment, body, norm, disturbance, scaffold, and operators.
Implement the smallest missing reusable composition seam, compile or adapt that
composition into a versioned family contract, and then audit and render it. A
row is not complete when its process-algebra coverage exists only as a label in
`CARDS.md`; the corresponding seam must be executable and reused wherever two
cards claim the same construct.

Proceed in this order:

1. **R5 — finish Card 04 rendering.** Render the audited norm-swap contract
   through the profiled event boundary and pass the boundary and integration
   checks needed to expose it to the common learner.
2. **R6 — implement Card 03.** Reuse the shared G0 environment, add the
   affordance/reachability contract, audit identification and reveal semantics,
   and add learner rendering.
3. **R7 — implement Card 02.** Add the predictive-state family, exact history
   ablation and ambiguity audit, and learner rendering.
4. **R8 — implement Card 05.** Add the active-experimentation family, exact
   value-of-information comparison, matched-cost non-informative control, and
   learner rendering.
5. **R9 — implement Card 06.** Add the perceptual-organization family, source
   binding controls, exact small-instance posterior, preserving/changing orbit,
   and learner rendering.
6. **R10 — close the seed gate.** Run one bounded learner pilot per family,
   retain only families with usable progress signals, and fix the admitted
   contract hashes, source sentinels, fixed-mixture comparator, mixture
   accounting, checkpoint cadence, total budget, and stop rules.
7. **R11 — start matched lineages.** Initialize the adaptive and fixed-mixture
   arms from the same checkpoint. Use the same admitted family versions,
   learner objective, episode accounting, optimizer budget, and evaluation
   support. Emit immutable checkpoints and retain the evidence needed to replay
   every transition.

R5--R10 are the bootstrap required before the first multi-world lineage exists.
They do not require Cards 01, 07, or 08. Once R11 is active, preparation of
additional candidate families need not wait for the current training block to
finish.

### R11 as direct checkpoint hill climbing

R11 is a greedy, local optimization over checkpoint states, not an attempt to
solve a complete curriculum in advance. Define a checkpoint state as:

```text
C[k] = (
    weights,
    admitted_family_contracts,
    source_sentinel_evidence,
    retention_state,
    remaining_budget,
    progress_chart
)
```

Before R11 begins, fix a source score from the declared capability sentinels and
retention penalties. For any bounded candidate action `A` from checkpoint
`C[k]`, its realized local utility is:

```text
utility(A | C[k]) =
    (source_score(Candidate[k+1]) - source_score(C[k]))
    / total_cost(A)
```

The score, weights, retention bounds, cost accounting, and tie rule are part of
the R10 run contract; they cannot be changed after inspecting a candidate. A
sealed R12 transfer family is never a scheduler sentinel and cannot influence
this hill climb.

At each immutable checkpoint boundary:

1. freeze `C[k]` and evaluate its declared source sentinels;
2. admit any fully audited and piloted family versions that are ready, without
   requiring that they be trained immediately;
3. form bounded neighbouring actions from the admitted pool, such as more
   exposure to a new family, replay of an earlier family, or a changed fixed
   mixture;
4. fix each candidate's parent checkpoint, contract hashes, mixture, seeds,
   update budget, monitors, cost accounting, and stop rule;
5. select the candidate with the highest predicted positive local utility,
   using a bounded matched branch only when sentinels cannot rank two live
   candidates cheaply;
6. execute that one training block, evaluate the resulting candidate
   checkpoint, and measure its realized utility and retention;
7. retain the candidate as `C[k+1]` only when it satisfies the declared
   improvement and retention rules; otherwise keep `C[k]` as the parent for the
   next bounded attempt; and
8. compare the adaptive lineage with the matched fixed-mixture lineage at the
   declared cadence. Retain the adaptive method only if its capability gain per
   total cost is better.

This is direct hill climbing because each accepted training block must improve
the declared local checkpoint objective from its immutable parent. It does not
claim global curriculum optimality.

### Concurrent family and instance generation during R11

Two kinds of concurrency are permitted:

- **Family preparation:** while an R11 block trains on immutable admitted
  contracts, other work may implement, audit, render, and prepare candidate
  families. A learner pilot may also run concurrently when separate compute is
  available; otherwise it runs in a checkpoint gap. A candidate becomes
  scheduler-eligible only after its audit and pilot complete, and it can enter
  only at the next checkpoint boundary. A new family that realizes an existing
  card does not require a new card; a new behavioural contrast does.
- **Instance generation:** after the scheduler fixes a training mixture, seeded
  instances from those admitted contract versions may be generated in parallel
  with GPU optimization and placed in a bounded prefetch queue. This is an
  execution optimization, not a scheduler decision and not a semantic family
  change.

Concurrent work must not mutate a contract used by the active block, inspect a
sealed transfer family, change public/privileged boundaries, or escape total
cost accounting. A semantic change produces a new contract hash and waits for a
later admission boundary.

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
