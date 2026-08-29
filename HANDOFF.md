# Handoff

Current state of the R5–R10 bootstrap, written so the next agent can continue
without re-deriving what was already decided. `DEVELOPMENT-PATH.md` remains the
authority for the progress chart; this document says where the work stopped, what
was learned building it, and what to do next.

## Where the work stopped

**R5, R6, R7, and R8 are complete and committed. R9 is next, then R10.**

| Row | State | Crate |
|---|---|---|
| R3a executable process kernel and query algebra | Complete | `crates/g0-contract` |
| R3b one learner event boundary | Complete | `crates/g0-render` |
| R5 card 04 norm swap | Complete | `crates/card04-norm-swap` |
| R6 card 03 affordance | Complete | `crates/card03-affordance` |
| R7 card 02 predictive state | Complete | `crates/card02-predictive-state` |
| R8 card 05 active experimentation | Complete | `crates/card05-active-experimentation` |
| **R9 card 06 perceptual organization** | **Not started** | — |
| R10 seed gate | Not started | — |

`cargo test --workspace --locked` passes: 39 test targets, 0 failures.
`cargo fmt --all -- --check` is clean. The Python suite passes too — 37 tests —
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

Nothing is left half-applied, and no GPU or remote run has been launched or
authorized.

Four audit binaries emit JSON on stdout:

```bash
cargo run -p pretraining-card02-predictive-state --bin card02-audit
```

The others are `card03-audit`, `card04-audit`, `card05-audit`.

## What R9 has to do

Card 06 is the last seed family and the only one needing the `⊗` shared-coupling
seam, which `crates/g0-contract/src/kernel.rs` already defines and nothing yet
uses. Reusing it is not optional decoration — `DEVELOPMENT-PATH.md` requires the
seam to be executable wherever two cards claim the same construct, and cards 01,
06, and 08 all claim this one.

Design sketch that follows from the four families already built:

- exchangeable latent sources on the shared `Ring`, each with a hidden drift;
- public observation channels carrying **values** rather than selections — the
  `FiniteG0` profile's content-kind flag exists for this and is so far unused;
- `Coupling { rule: Override }` from channels to sources through a hidden
  assignment, changing at public boundaries whose *timing* is public and whose
  *new assignment* is not;
- occlusion as an `Interrupt` with `Displaced::Continues` in the witness and
  `Displaced::Frozen` in the frozen-during-absence control;
- a goal naming a source by interaction history, and the four controls the card
  lists: channel-locked, shuffled-covariance, frozen-during-absence, identity-tag.

The exact assignment posterior is what `AmbiguitySet` + `public_policy_value`
compute; do not write a bespoke posterior.

## What R10 has to do

Seven gate conditions in `DEVELOPMENT-PATH.md`. Two are already satisfiable from
what exists (one boundary; valid baseline/ceiling brackets with no direct
leakage), and the rest need work:

1. a bounded CPU learner pilot per family. Local Python has `torch 2.10.0+cpu`
   and `transformers 5.3.0`, and the existing suite passes on it;
2. the five families need a PyO3 surface. `crates/world-py` currently exposes
   only the two legacy worlds. One batching entry point taking a family name and
   returning padded profiled tensors would serve all five;
3. mixture accounting must count **distinct episodes, not case labels** — see
   the collision counts below;
4. the comparators, cadence, budget, and stop rules have to be written down as a
   run contract before any pilot is scored.

Compute rule reminder from `META-PROCESS.md`: local CPU for anything under ten
minutes, Kaggle above that, and **no GPU launch without explicit authorization**.

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
