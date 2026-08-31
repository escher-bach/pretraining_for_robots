# Handoff

Current state of the R5–R10b bootstrap, written so the next agent can continue
without re-deriving what was already decided. `DEVELOPMENT-PATH.md` remains the
authority for the progress chart; this document says where the work stopped, what
was learned building it, and what to do next.

## Where the work stopped

**R5–R10c are closed. The preserved first R10 one-T4 pilot exposed a loss/metric
and accounting defect; its one allowed grouped-objective repair completed and
is the decisive result. R10 is `seed_gate_incomplete`: Card 02 alone is
frontier-admitted, while Cards 04, 03, 05, and 06 are audited and
compatibility-characterized but inconclusive. R10a decomposed and deferred
those four. R10b implemented all four decomposition certificates as new audited
family versions and gated them under a new profile: every one is
`support_fit_incomplete`, so every certificate's own falsifier fired. R10c
replicated that profile unchanged on R10's own learner — one T4 at fp16 — and
reached the same verdict, so the falsification is not a device artifact. R11
remains blocked.**

| Row | State | Crate |
|---|---|---|
| R3a executable process kernel and query algebra | Complete | `crates/g0-contract` |
| R3b one learner event boundary | Complete | `crates/g0-render` |
| R5 card 04 norm swap | Complete | `crates/card04-norm-swap` |
| R6 card 03 affordance | Complete | `crates/card03-affordance` |
| R7 card 02 predictive state | Complete | `crates/card02-predictive-state` |
| R8 card 05 active experimentation | Complete | `crates/card05-active-experimentation` |
| R9 card 06 perceptual organization | Complete | `crates/card06-perceptual-organization` |
| R10 seed gate | Complete — `seed_gate_incomplete` | `crates/world-py`, `python/pretraining_experiments/seed_gate.py`, `configs/r10` |
| R10a post-gate compatibility triage | Complete — Card 06 `support_fit_incomplete` | `CARDS.md`, `DEVELOPMENT-PATH.md`, `configs/r10/card06_compatibility_scale_t4.toml` |
| R10b stage-A decomposition gate | Complete — `decomposition_gate_incomplete` | `crates/card04a-goal-use`, `crates/card03a-body-identification`, `crates/card05a-reveal-use`, `crates/card06a-visible-reassignment`, `configs/r10b` |
| R10c matched replication of the R10b profile | Complete — `decomposition_gate_incomplete` | `configs/r10b/decomposition_gate_t4.toml`, `kaggle/experiments.toml` |

`cargo test --workspace --locked` passes with no failures.
`cargo fmt --all -- --check` is clean. The Python suite passes too — 67 tests —
but **only after rebuilding the PyO3 wheel**, because the extension carries both
the `0.3.1` envelope constant and the family corpora themselves, and R10b added
four families to the latter:

```bash
python -m maturin build --release --locked --manifest-path crates/world-py/Cargo.toml --out dist
```

then force-reinstall the wheel from `dist/` and set `PYTHONPATH` to `python/`.
A stale wheel makes four tests fail with an ABI-version mismatch that looks like
a code defect and is not, and a wheel predating R10b reports the stage-A
families as unknown. Note also that the environment has
`transformers 5.3.0` where `requirements-kaggle.txt` pins `4.57.6`; the suite
passes on both, but a Kaggle run uses the pin.

The user authorized the completed Kaggle GPU work, `origin` is the public HTTPS
repository, and the verified decisive receipt is
`audit/runs/pretraining-r10-seed-gate-grouped-e2dc185/receipt.json` for source
`e2dc1856ab56e45f55d5fa01e63d0bd0f90035b6`. Do not relaunch R10.

Nine family audit binaries emit JSON on stdout:

```bash
cargo run -p pretraining-card02-predictive-state --bin card02-audit
```

The seed families are `card02-audit`, `card03-audit`, `card04-audit`,
`card05-audit`, and `card06-audit`. The four R10b stage-A families are
`card04a-audit`, `card03a-audit`, `card05a-audit`, and `card06a-audit`, in
`crates/card04a-goal-use`, `crates/card03a-body-identification`,
`crates/card05a-reveal-use`, and `crates/card06a-visible-reassignment`. Each
stage-A audit carries a `removed_from_composite` record stating, in the
certificate's own vocabulary, which decision or construct left — so an
implementation can be checked against its certificate instead of trusted.

## What R10b closed

R10b is the re-entry row for the four `decomposed/deferred` cards. It did two
things: it implemented every decomposition certificate as a real world version,
and it gated the four under one newly declared AdmissionProfile,
`configs/r10b/decomposition_gate.toml`.

| Family | Contract | Cases / distinct | Decomposes | Macro argmax `0 -> 64` | Case kinds at the rung |
|---|---|---:|---|---|---|
| `card04a` | `06a36ac33c3f952b` | 16 / 16 | Card 04 | `0.3750 -> 0.3750` | witness `0.2500`, controls `0.3750` and `0.5000` |
| `card03a` | `235c0a6ad2efb10d` | 12 / 12 | Card 03 | `0.5000 -> 0.8333` | witness `0.5000`, both controls `1.0000` |
| `card05a` | `4941b2e1e1c8c390` | 12 / 6 | Card 05 | `0.8333 -> 0.8333` | witness `0.5000`, both controls `1.0000` |
| `card06a` | `b6f762bb89601096` | 20 / 14 | Card 06 | `0.2500 -> 0.7500` | all three kinds `0.7500` |

The profile inherits the decisive R10 repair's learner core, initialization
policy, grouped action-query objective, batch, and 64-update / 256-presentation
budget, and changes only the world. Its barrier is the one each certificate
states for itself — predeclared exact full-support fit — and deliberately not
R10's `0.80` final macro, `0.25` gain, and `0.60` every-case-kind numbers, which
were heuristic and unpowered and remain binding for R10 alone. The receipt
carries the counterfactual anyway: all four families miss R10's barrier too, so
the outcome does not depend on which barrier was declared.

**The result is `decomposition_gate_incomplete`, and the honest reading is that
four predictions were wrong.** Each certificate said the composite's difficulty
lay in the factor it removed. Removing the factor did not move the barrier. The
classification is learner/support fit rather than a world or apparatus defect:
every family passes its own semantic audit, and every apparatus check passes —
ABI and bounds, finite parameters, no timeout, exactly 256 presentations per
family, and falling training loss on three of the four. Card 04's stage A is the
starkest: its macro did not move from `0.3750` at any recorded rung and its loss
fell only `1.0985 -> 1.0720`.

Two things follow, and neither is a licence to tune. The profile's predeclared
incomplete action forbids re-running it with different numbers. And the
certificates' staging is not entered: nothing adds absence back to Card 06, or a
probe back to Card 05, on top of a stage that was not itself fit.

One deviation is recorded rather than smoothed over. This gate ran on local CPU
in `469.1 s` at fp32; the decisive R10 result was one T4 at fp16. The learner
core, initialization policy, objective, batch, and budget are the same, but the
pairing is not byte-identical, and "the same fixed learner" is a phrase the
certificates use. That left one apparatus question open, and **R10c closes it**:
the user authorized GPU work, and the *unchanged* profile is replicated on one
T4 at fp16 under `configs/r10b/decomposition_gate_t4.toml`. The validator admits
exactly two device spellings and no third, and a test asserts that the two
configurations differ in nothing else, so the replication cannot drift into a
second profile.

The R10c interpretation was fixed before launch and is in the chart row: all
four exact means the CPU outcome was a device/precision artifact and the
certificates are not falsified; none means the falsification stands on the
matched learner; a mixture is read per family; anything incomplete is `unscored`
and an apparatus defect. No branch admits a composite, reopens R10, authorizes
R11, or permits tuning. **The R10b CPU receipt is preserved either way** — it is
a completed run whose learner deviated, not a failure to be overwritten.

Compact evidence, including per-family receipts and the preflight, is under
`audit/runs/pretraining-r10b-stage-a-decomposition-gate-local/`.

## What R10c settled

R10c ran the R10b profile unchanged on one T4 at fp16, which is R10's own
decisive learner. It is audit-verified at source
`c2b60907c9ce61a5e73ba5a1c0ab9ae7ba72d488` with a clean tree, the remote Rust
and 67 Python tests passing, a CUDA preflight of four updates in `3.3810 s` and
a full-corpus evaluation in `0.6577 s`, and exactly 256 presentations per
family. Run:
`https://www.kaggle.com/code/aniruddhavarma/pretraining-r10c-stage-a-c2b6090`.

| Family | CPU macro `0 -> 64` | T4 macro `0 -> 64` | T4 case kinds at the rung |
|---|---|---|---|
| `card04a` | `0.3750 -> 0.3750` | `0.3750 -> 0.7500` | witness `0.7500`, constant-goal `0.5000`, goal-predictable `1.0000` |
| `card03a` | `0.5000 -> 0.8333` | `0.5000 -> 0.8333` | witness `0.5000`, both controls `1.0000` |
| `card05a` | `0.8333 -> 0.8333` | `0.8333 -> 0.8333` | witness `0.5000`, both controls `1.0000` |
| `card06a` | `0.2500 -> 0.7500` | `0.2500 -> 0.7500` | all three kinds `0.7500` |

**No family reached exact full-support fit, so the predeclared "none" branch
applies and the falsification stands on the matched learner.** The four
certificates predicted that the composites' difficulty lay in the factors they
removed; on the learner the certificates name, removing those factors does not
move the barrier.

Two details are worth carrying forward. Three of the four families reproduce
the CPU numbers *exactly* at every recorded rung, which is a stronger
replication than the row needed and says the measure is not sitting on a
precision boundary. Card 04's stage A is the one the device moved, from flat at
`0.3750` to `0.7500` — so the CPU flatness reported in R10b was partly a
device/precision effect and that sentence in "What R10b closed" should be read
with this row beside it. It still misses the barrier, and its bracket is not
restored either: the control a state-only policy should find easiest, the
constant-goal arm, is its *worst* kind at `0.5000`, below the witness at
`0.7500`. Whatever this learner is failing to acquire, it is not specific to
reading a goal channel.

Two wrinkles for the next reader. The gate receipt's `row` field reads `R10b`,
because it is emitted from the profile rather than from the chart row; the
profile *is* R10b's, unchanged and deliberately so, and the chart row is R10c.
And the first collection of this run wrote a null `scientific_report`, because
`runner.py` files a decomposition-gate receipt under `summary["decomposition_gate"]`
while `tools/kaggle_run.py` only looked for the seed-gate and Card06 keys. That
is fixed, the run was re-collected from the same immutable kernel version, and
the collector now refuses an unrecognized finite-G0 summary key instead of
silently dropping the report. Nothing about the run itself changed.

Compact evidence is under `audit/runs/pretraining-r10c-stage-a-c2b6090/`, with
the verified per-family receipts under its `pretraining-results/`.

## What building the stage-A families found

Four of these are new, and each one is a thing the composite could not have
shown.

**A decomposition can delete its own control.** Card 05's stage A drops the
probe, so the composite's uninformative-reveal control drops with it: a decoy
that costs nothing to follow is worth exactly as much as a blind commit, and a
control that moves no value is testing nothing. It survives as an *information*
orbit instead, where it moves the public ceiling `99 -> 49.5`. The composite's
own lesson — the value orbit is blind to information transformations — is what
makes the reduced family well posed rather than merely smaller.

**A closed ambiguity gap can be the measurement.** Card 06's stage A has a zero
public/known-assignment gap where the composite's is `98` against `49`. That is
not a weakened world; it is the removed factor stated as a number, and the
`values_made_invisible_across_the_boundary` orbit reopens it to `49` on demand.

**An inherited verdict can be false in the reduced family.** The composite
treats hiding Card 06's assignment-change boundary as meaning-changing. With the
values continuously visible it is *preserving*, because the marker carries
nothing the values do not. The stage-A audit declares and checks it as
preserving and records the divergence; inheriting the composite's verdict would
have asserted an invariance this family does not have.

**A single decision needs two labels on one edge.** Card 03's stage A removes
the planning half, and with one decision left every goal cell would name exactly
one command — the body would stop mattering. `Leap` and `Vault` both drive the
`+2` edge and the witness body supports one of them, which is the card's own
declared coupling variant carrying the whole content of the decision.

## What R9 completed

Card 06 is the last seed family and the first executable user of the `⊗`
shared-coupling seam. Contract `76a08f38947c8cae` has 36 exact seeded cases and
32 distinct public episodes.

The implementation has:

- two exchangeable latent sources with hidden drift;
- public observation channels carrying **values** rather than selections — the
  `FiniteG0` profile's content-kind flag exists for this and is so far unused;
- executable `Coupling { rule: Override }` resolving competing source and
  matched-marginal noise writers through a hidden assignment;
- occlusion as an `Interrupt` with `Displaced::Continues` in the witness and
  `Displaced::Frozen` in the frozen-during-absence control;
- a goal naming a source by interaction history, and the four controls the card
  lists: channel-locked, shuffled-covariance, frozen-during-absence, identity-tag.

The exact assignment posterior is computed by `AmbiguitySet` plus
`public_policy_value`. The audit also reports agent-equivalence-quotiented
ambiguity, shared noninterference, real preserving/changing/information orbits,
baseline brackets, seeds, ambiguity gaps, and learner-boundary round-trips.

## What R10 closed

The decisive grouped-objective run is the verified Kaggle T4 receipt at
`audit/runs/pretraining-r10-seed-gate-grouped-e2dc185/receipt.json`, source
`e2dc1856ab56e45f55d5fa01e63d0bd0f90035b6`, run
`https://www.kaggle.com/code/aniruddhavarma/pretraining-r10-seed-gate-grouped-e2dc185`.
CUDA preflight passed: four updates in `3.0558 s`, full-corpus evaluation in
`0.7171 s`. Every pilot passed ABI/bounds checks and consumed exactly 256
presentations.

| Family | Macro grouped argmax | Final all-case-kind minimum | R10 state |
|---|---:|---:|---|
| Card 04 | `0.3395 -> 0.4259` | `0.3333` | Audited, inconclusive |
| Card 03 | `0.4667 -> 0.6000` | `0.2500` | Audited, inconclusive |
| Card 02 | `0.1667 -> 0.9375` | `0.8333` | Frontier-admitted |
| Card 05 | `0.1250 -> 0.6875` | `0.2500` | Audited, inconclusive |
| Card 06 | `0.2500 -> 0.6875` | `0.5625` | Audited, inconclusive |

The classifier used every `by_case_kind` value despite the stale configuration
field name `required_primary_case_kind_argmax`; this is stricter than the name
implies and does not change any outcome. The fixed barriers were final macro
`>= 0.80`, gain `>= 0.25`, and every-case-kind `>= 0.60`. These were
predeclared, useful gate decisions, but heuristic and unpowered rather than
scientifically derived learnability thresholds. In contrast, the gate's
structural barriers — valid audited contracts, leakage-free public rendering,
distinct public support, working ABI/bounds, exact consumed-step accounting,
and sealed transfer — are grounded in executable audits and receipts.

The overall result is `seed_gate_incomplete`. It is source-family learner
evidence only, not transfer evidence. R10 permits no second repair and does
not authorize R11. The original L1 run remains a preserved apparatus failure;
the grouped objective and exact accounting were its one bounded repair.

R10a is complete and was not a rerun of R10: decompose/defer Card 04;
decompose body identification from planning for Card 03 (scale only under a
separately justified profile); decompose reveal use from probe value for Card
05; and allow one Card-06-only scale diagnostic because its curve rose through
64 updates. The user authorized GPU work and the fixed Card 06 profile is
`configs/r10/card06_compatibility_scale_t4.toml`. The first kernel at
`a9f118018261241812b991811cb33aebf51b1f7c` stopped before training because the
validator rejected the runner-resolved absolute spelling of the pinned model
configuration path. Its audit-verified failure receipt is preserved under
`audit/runs/pretraining-r10a-c06-scale-a9f1180/`. The bounded repair treated
only the equivalent relative and resolved paths alike while retaining strict
validation of every other execution field.

The repaired run at source `f4fd45edcda699f7a2e1fe4ec54c1a0a5117a2fc`
completed 256 updates and 1,024 presentations with verified artifacts. Exact
fit was false at 64, 128, and 256 updates. Macro case-kind argmax rose
`0.2500 -> 0.7250 -> 0.7875 -> 0.8750`, but the decisive witness moved
`0.2500 -> 0.6250 -> 0.4375 -> 0.5000`; three simpler controls reached
`1.0000`, and identity-tag reached `0.8750`. This is
`support_fit_incomplete`, not a world or apparatus defect. The composite Card
06 is decomposed/deferred under the certificate in `CARDS.md`. Do not rerun or
tune this profile, reopen R10, or authorize R11. Compact evidence is under
`audit/runs/pretraining-r10a-c06-scale-f4fd45e/`.

## What building these four families actually found

These are the things a reader would otherwise have to rediscover. Each one is a
defect that was found by construction and is now fixed and tested.

**A privileged teacher is easy to write by accident.** Card 04's audited
"optimal first action" on its two unannounced-switch witnesses was `retreat`,
correct only for a solver that already knows the goal will change. Nothing
publishes that at step zero. Every card now teaches from a public policy, and
`RenderFault::TeacherWouldLeak` lets a rendering refuse a contract whose teacher
would read unpublished state. Card 05 refuses a decorrelated latch; card 03
refuses an uninformative calibration.

**A vacuous ambiguity gap reads exactly like a real one.** `ambiguity_gap`
compares `privileged_value` with `value`, and a fragment that does not override
the former is comparing a quantity with itself. Cards 04 and 02 report theirs as
vacuous on purpose and carry the real quantity separately — the
published-information gap and the latch-ablation gap. Card 05 is the first family
where the shared comparison says something.

**The value orbit is blind to information transformations.** Making card 03's
calibration uninformative, or ending card 02's aliasing interval early, moves
neither the ceiling nor the correct action: a contract-holding solver never
needed the scaffold. `check_information_orbit` checks those against what a
coarsened learner can attain and against the identification diameter.

**The value orbit also reads the wrong action.** `check_orbit` compares optimal
*first* actions, and card 02's two modes open with the same move. `check_orbit_with`
takes the observable; card 02 passes its discriminating command.

**Re-solving "the remaining episode" restarts the step clock.** Clamping a
time-dependent reveal with a saturating subtraction turns "already fired" into
"fires after the next action", and it cost card 03 its own restoration witness.
Use `optimal_actions_from` (or `public_optimal_actions_at` where there is hidden
state); both keep the absolute clock and need no rebasing.

**Observe, then act.** The belief recursion originally partitioned only *after*
each action, forcing the first move to be common to every candidate. That
understates the public ceiling for any family whose scaffold speaks before the
first decision, and made card 03 report a positive gap where it provably has
none.

**Supervising one action where several are correct teaches a tie-break the world
does not have.** It also made card 02's irrelevant-latch control render
byte-identically to its witness. The boundary admits a *set* of correct actions
per decision.

**Per-case optimality is wrong for a family with residual uncertainty.** Card
05's public ceiling is an average no single episode attains; the per-case test
rejected the optimal blind commit on half its instances. Admission there is
judged in expectation over a case kind.

**A transform that changes nothing publicly is testing nothing.** Card 03's
body/environment swap was invisible until the calibration scaffold was extended
to probe past the region the environment deletion covers. The audit reports how
many swaps are actually visible rather than claiming the transform bites
everywhere.

## Rendering collisions, which R10 must account for

Distinct public episodes per family, against case count:

| Family | Episodes | Distinct | Why |
|---|---:|---:|---|
| Card 04 | 20 | 16 | four labels name one contract, two name another |
| Card 03 | 12 | 10 | the no-restore negative *is* the unreachable-fallback witness |
| Card 02 | 10 | 9 | the Forward witness and Forward irrelevant-latch control differ only counterfactually |
| Card 05 | 16 | 7 | the inconsequential bit is never bought, so it never appears |
| Card 06 | 36 | 32 | four frozen-source seed pairs share the same public episode |

None of these is a defect; all of them mean a training mixture must count
episodes rather than labels. `RenderingReport::colliding_episodes` names the
groups.

## Deliberate omissions, each with its reason

- **No `FutureQuery` and no `Feedback` in the `FiniteG0` profile.** A future
  target for a family with hidden state would have to be read off privileged
  state, which the information boundary forbids as supervision; terminal outcome
  feedback would publish, after the fact, the very mode or gate cards 02 and 05
  withhold. Emitting either for some families and not others would also make
  supervision density a family correlate. The action head carries the whole
  learner signal.
- **Card 02's variance variant.** Needs an objective under which outcome spread
  changes the optimal action; the deterministic G0 fragment has none. The
  `P6 -> M2` dispute stays open.
- **Card 05's high-prediction-error variant and novelty-driven baseline.** Need a
  prediction objective, which follows from the point above. `M2 -> M5` stays open.
- **Card 03's absorbing-wasted-budget variant.** A variant of the witness rather
  than part of it; no admission decision turns on it.
- **Cards 01, 07, 08.** Not prerequisites for the seed gate.

## Interfaces you will touch

- `crates/g0-contract/src/kernel.rs` — the five operators and the norm algebra,
  as shared data. `KernelUse::declared(card)` holds `EMBODIED-PROCESS.md`'s
  coverage table so a card can be checked against it.
- `crates/g0-contract/src/query.rs` — all six declared queries plus the two
  auditor operations. `AmbiguitySet` is the object everything is derived from.
- `crates/g0-render/src/lib.rs` — a card emits a `G0Episode` transcript and
  nothing else. `boundary_check` renders and decodes and requires equality.
- `physical-event-abi-0.3.1` — the envelope. `0.3.0` refused every condition
  record, which the finite families need for `reveal`; the guard is narrowed to
  the header signature and the canonical decoder refuses a malformed condition
  quantity so a skipped envelope still cannot be read as a public fact.

## One convention worth knowing

Step counts are expressed as fractions of the 16-slot action head, never of a
card's own horizon. Sixteen is a power of two so every such fraction is exact in
the `f32` payload; a horizon of three is not, and the renderer refuses `1/3`
rather than rounding it. `pretraining_g0_render::step_fraction` is the helper.
