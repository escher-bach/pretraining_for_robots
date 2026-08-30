# Handoff

Current state of the R5–R10 bootstrap, written so the next agent can continue
without re-deriving what was already decided. `DEVELOPMENT-PATH.md` remains the
authority for the progress chart; this document says where the work stopped, what
was learned building it, and what to do next.

## Where the work stopped

**R5–R9 are complete. The preserved first R10 one-T4 pilot completed but did
not admit a family: its row-wise loss was mismatched to grouped ActionQuery
argmax and its iterable counters included prefetch. One versioned, bounded
apparatus repair is prepared; it changes neither worlds nor scientific budget.**

| Row | State | Crate |
|---|---|---|
| R3a executable process kernel and query algebra | Complete | `crates/g0-contract` |
| R3b one learner event boundary | Complete | `crates/g0-render` |
| R5 card 04 norm swap | Complete | `crates/card04-norm-swap` |
| R6 card 03 affordance | Complete | `crates/card03-affordance` |
| R7 card 02 predictive state | Complete | `crates/card02-predictive-state` |
| R8 card 05 active experimentation | Complete | `crates/card05-active-experimentation` |
| R9 card 06 perceptual organization | Complete | `crates/card06-perceptual-organization` |
| R10 seed gate | One bounded apparatus repair queued | `crates/world-py`, `python/pretraining_experiments/seed_gate.py`, `configs/r10` |

`cargo test --workspace --locked` passes with no failures.
`cargo fmt --all -- --check` is clean. The Python suite passes too — 41 tests —
but **only after rebuilding the PyO3 wheel**, because the `0.3.1` envelope bump
changed a constant the installed extension carries:

```bash
python -m maturin build --release --locked --manifest-path crates/world-py/Cargo.toml --out dist
```

then force-reinstall the wheel from `dist/` and set `PYTHONPATH` to `python/`.
A stale wheel makes four tests fail with an ABI-version mismatch that looks like
a code defect and is not. Note also that the environment has
`transformers 5.3.0` where `requirements-kaggle.txt` pins `4.57.6`; the suite
passes on both, but a Kaggle run uses the pin.

The user authorized Kaggle GPU work, `origin` is now the public HTTPS repository,
and the first T4 pilot completed; its preserved receipt and the one bounded
repair are described below. The R10 registry entries use the existing exact-SHA
launcher.

Five seed-family audit binaries emit JSON on stdout:

```bash
cargo run -p pretraining-card02-predictive-state --bin card02-audit
```

The others are `card03-audit`, `card04-audit`, `card05-audit`, and
`card06-audit`.

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

## What remains for R10

Gate conditions 2, 3, 5, 6, and 7 are now fixed. The mixed PyO3 API samples
distinct public episodes rather than labels and returns family/hash/alias data
only as evaluator metadata. `configs/r10/seed_gate_cpu.toml` freezes the five
selected-core pilots; `configs/r10/lineage_contract.toml` freezes the later
fixed-mixture, adaptive, scratch, alternative-pretraining, retention, cadence,
budget, and stop contracts. Source sentinels and the sealed goal-conditioned
transfer diagnostic are structurally disjoint.

The first T4 run is retained at
`audit/runs/pretraining-r10-seed-gate-e36b828/receipt.json`. Its preflight
passed (four updates in 2.80 seconds; full-corpus evaluation in 0.72 seconds)
and all five pilots completed, but none met the fixed admission threshold.
Row-wise L1 decreased while grouped argmax stayed flat or fell for every family;
that common failure is an apparatus/objective mismatch, not evidence against
the worlds. Its `episode_presentations = 520` also counted iterable prefetches
rather than the 256 examples actually consumed by 64 updates of four examples.

The sole authorized repair is `configs/r10/seed_gate_t4_grouped.toml` and
registry entry `r10-seed-gate-grouped`: raw action-head logits use standard
categorical cross-entropy across ActionQuery alternatives, with grouping passed
only to the loss adapter, and cost is regenerated from completed steps. Seeds,
family contracts, selected core, optimizer schedule, batch size, thresholds,
evaluation support, and time caps are unchanged. Its result is decisive for
R10; do not make a second repair.

The original CPU timing receipt at
`artifacts/r10/seed-gate/timing-preflight-receipt.json` is apparatus evidence,
not learner evidence: with standard per-family dynamic padding under the fixed
192-token cap, Card 04 evaluation took 2.40 seconds but four optimizer updates
took 8.43 seconds against the fixed three-second stop. The CPU path therefore
stopped before scoring any family.

Next action: commit and push the exact source, then launch
`python tools/kaggle_run.py launch --experiment r10-seed-gate-grouped`.
Collect the compact receipt after the kernel becomes terminal. Do not loosen or
change the worlds, learner core, seeds, budgets, thresholds, or stops; the
grouped loss and exact completed-step accounting are the only repair.

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
