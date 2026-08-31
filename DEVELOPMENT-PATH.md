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
proposed -> candidate -> audited -> compatibility-characterized -> frontier
         -> replay/sentinel -> archived
                         \-> decomposed/deferred -> re-entry
```

- `candidate`: executable contract and minimal witness exist.
- `audited`: semantics, information boundary, baselines, ambiguity, negatives,
  replay, and invariance pass.
- `compatibility-characterized`: a fixed learner, objective, support, and
  budget have been run with a recorded `AdmissionProfile`. It says which
  learner/world pairing was tested; it does not yet assert that the world is
  easy, hard, or developmentally ordered.
- `frontier`: a compatibility-characterized pilot satisfies its declared
  AdmissionProfile and exposes a nontrivial learning-progress signal.
- `decomposed/deferred`: a valid family that is not frontier under its declared
  profile has a small capability decomposition certificate and a precise
  re-entry condition. It is not silently weakened or retried.
- A semantic change creates a new version. A new seed does not.
- A failed family gets one bounded repair when the claim stays unchanged;
  otherwise it is deferred or archived and does not block unrelated trunks.

An `AdmissionProfile` fixes the learner core and initialization policy,
objective, public training support, evaluator, update and presentation budget,
cadence, thresholds, and stop rule. Its thresholds are decision barriers, not
estimates of a natural learnability boundary: they must be declared before the
run and justified as sufficient to choose a path, but are not treated as
scientifically derived without power or calibration evidence. Structural gate
requirements (valid contracts, no leakage, distinct public episodes, fixed
comparators, sealed transfer, and exact accounting) are grounded by the
contracts and audits; numeric cadence and thresholds are not.

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

## Compatibility diagnosis and re-entry

A failed admission barrier is a classification problem, not permission to tune
until a score crosses it. Use the cheapest observation that can distinguish the
following evidence classes before scheduling another learner run.

| Observation | Classification | Permitted next action | Not established |
|---|---|---|---|
| Broken contract, leaked target, wrong public policy, invalid orbit/baseline, or a degenerate solution that survives the card controls | World defect | Repair or replace the contract, then re-audit under a new hash. | A learner limitation or a capability claim. |
| ABI/bounds failure, objective/evaluator mismatch, incorrect accounting, timeout, nondeterminism, or unsupported training path | Apparatus defect | One bounded apparatus repair with worlds, profile budget, and claim fixed; preserve the failed receipt. | Evidence against the world or learner. |
| Audits and apparatus pass, but the fixed profile fails its barrier | Learner/support fit | Mark compatibility-characterized and inconclusive; issue a decomposition certificate or defer. | That the world is defective, impossible to learn, or unsuitable for every learner. |
| Improvement is present but misses a barrier, or support/cadence is too small to characterize the curve | Scaling question | A card-local scaling profile only when the curve and a predeclared decision make it informative. | Admission, transfer, or a reason to alter the portfolio gate. |
| An admitted source signal does not improve a held-out matched descendant versus scratch and alternative pretraining | Generalization/transfer failure | Remove or narrow the predicted graph edge; retain source evidence separately. | That the source world was invalid. |

Every `decomposed/deferred` record carries a **decomposition certificate**:
the original behavioural relation; the proposed more basic relation; which
world component or decision is removed; the controls and invariances retained;
why the resulting target is more compatible with the observed learner profile;
the falsifier; and the exact re-entry condition. A certificate may defer a
composite world to a later trunk, but may not relabel a new capability as a
repair of the old one.

A **scaling profile** is a new, card-local AdmissionProfile, not a relaxed
version of an old gate. It fixes the unchanged contract hash, learner and
objective, support, scale points, accounting, stopping rule, and the decision
that each scale point can change. It must say why the existing learning curve,
rather than hope, warrants scale. Re-entry requires a completed certificate or
scaling profile, a fresh path row, and explicit compute authorization; neither
can reopen a closed portfolio gate.

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
| R8 | Card 05 active-experimentation family | Complete | Three outcome cells, two decisions, sixteen cases. Privileged 99, public 98, no-probe 49.5: the first non-vacuous ambiguity gap in the portfolio. Probing is the correct public opening on the witness and on no control; the matched control holds where the probe is matched and informative, and the audit names which clause each control breaks. |
| R9 | Card 06 perceptual-organization family | Complete | Contract `76a08f38947c8cae`: 36 exact cases and 32 distinct public episodes. Two evolving latent sources execute shared `Override` coupling and continue/freeze interruption; the audit reports the exact shared posterior, agent-equivalence ambiguity, noninterference, four preserving orbits, three meaning-changing/information orbits, all four controls, and learner round-trips. |
| R10 | Four-trunk seed portfolio | Complete — seed gate incomplete | The one permitted grouped-objective repair is audit-verified at source `e2dc1856ab56e45f55d5fa01e63d0bd0f90035b6`: CUDA preflight passed (four updates 3.0558 s; full-corpus evaluation 0.7171 s), and every finite pilot completed with ABI/bounds checks and exactly 256 consumed presentations. Only Card 02 (`74b2d0da16ad3b31`) is frontier-admitted: macro grouped argmax `0.1667 -> 0.9375`, final all-case-kind minimum `0.8333`. Cards 04 (`0.3395 -> 0.4259`, minimum `0.3333`), 03 (`0.4667 -> 0.6000`, minimum `0.2500`), 05 (`0.1250 -> 0.6875`, minimum `0.2500`), and 06 (`0.2500 -> 0.6875`, minimum `0.5625`) are audited and inconclusive under the fixed `0.80` final / `0.25` gain / `0.60` every-case-kind barriers. The configuration's `required_primary_case_kind_argmax` name is stale: classification used all `by_case_kind` entries, not only primary kinds; that stricter implementation changes no outcome. The overall classifier is `seed_gate_incomplete`; no second R10 repair or R11 launch is authorized. Compact evidence: `audit/runs/pretraining-r10-seed-gate-grouped-e2dc185/receipt.json`; run: `https://www.kaggle.com/code/aniruddhavarma/pretraining-r10-seed-gate-grouped-e2dc185`. This is source-family learner evidence, not transfer evidence. |
| R10a | Post-gate compatibility triage | In progress; does not reopen R10 | Use the fixed R10 result to choose the next smallest card-local diagnostic, not a new portfolio gate or R11 launch. Card 04: decompose goal-use from switch/viability and defer the composite. Card 03: decompose body identification from reachability planning; scale only after a separately justified profile. Card 05: decompose reveal use from probe value. Card 06: permit one Card-06-only scale diagnostic because its grouped curve rose `0.2500 -> 0.6875` at 64 updates, while preserving its contract and recording a new AdmissionProfile. The first submitted Card 06 kernel at source `a9f118018261241812b991811cb33aebf51b1f7c` stopped before training because strict validation compared the pinned repository-relative model-config path with the runner-resolved absolute path. Its audit-verified failed receipt is preserved at `audit/runs/pretraining-r10a-c06-scale-a9f1180/receipt.json`; the bounded repair accepts only those equivalent paths and keeps every other profile field strict. |
| R11 | Multi-world learner lineage | Blocked by incomplete R10 seed gate | Matched fixed-mixture and direct hill-climbing sessions over immutable checkpoints require frontier families across T1, T2, T3, and T4. R10 admitted only Card 02 (T1), so no lineage may launch. |
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
