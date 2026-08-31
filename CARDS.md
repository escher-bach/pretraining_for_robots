# Capability Cards

This file is the complete specification of the eight capability instruments.
It contains the information needed to implement, audit, render, and interpret a
card without consulting another repository.

A card is a controlled comparison, not a task name. It supports its capability
claim only when the learner succeeds on the witness, changes behaviour on the
matched controls, respects the preserving transformations, responds to the
meaning-changing transformations, and satisfies the declared information
boundary. A high score on the witness alone is not evidence.

Cards are candidates, not a mandatory curriculum. The immediate seed portfolio
is Cards 04, 03, 02, 05, and 06. It covers trunks T1 through T4 with finite,
CPU-auditable worlds. Cards 01, 07, and 08 open when their dependencies or the
transfer evidence justify their additional cost.

## Shared implementation contract

Every implemented card version contains:

1. a versioned family contract and parameter domain;
2. a seeded generator that keeps public, privileged, and generator data apart;
3. a minimal positive witness and only the controls listed by the card;
4. a semantics-preserving orbit and paired meaning-changing transformations;
5. public and privileged bounds, an ambiguity report, and shortcut checks;
6. declared baselines, with their expected success and failure conditions;
7. a learner rendering through the common profiled event boundary;
8. replay, boundary, audit, and integration tests;
9. a compact receipt containing the contract hash, seeds, audit results, and
   learner-facing configuration; and
10. a downstream transfer comparison whose falsifier is fixed before training.

No card supplies custom oracle logic. It asks the shared apparatus these six
questions:

- `monitor`: did a trace satisfy a declared property?
- `identify`: which hidden realizations remain compatible with a history?
- `reachability`: can a property be achieved inside the remaining budget?
- `value_bounds`: what value interval is attainable from this history?
- `strategy`: which action set or distribution is justified by the view?
- `agent_equivalence`: can allowed interventions distinguish two hidden states?

The auditor additionally runs `metamorphic_check` and
`noninterference_check`. Memory ablations, probe value, ambiguity gap, public
ceiling, and privileged ceiling are derived from these operations; they do not
become per-card APIs.

The execution fragments are:

| Fragment | World class | Evidence status |
|---|---|---|
| G0 | Small finite discrete process | Exact enumeration is required. |
| G1 | Published linear/noisy process | Standard control and filtering solvers provide exact or declared numerical references. |
| G2 | Hybrid, switched, or multi-agent process | Bounds and monitors replace claims of an exact solver. |

Across all cards, bodies are known unless the card says otherwise, compared
actions have matched costs, resets and absorbing states are declared, and the
ambiguity gap is always reported. Pixels occur only in a downstream transfer
family, never in an abstract witness.

## Card 01: Regulation

**Trunk:** T1 identification and regulation. **Fragment:** G1. **State:**
specified and deferred until the finite seed needs a stronger regulation test.

**Claim.** Feedback rejects an unpredictable disturbance and recovers after an
observation gap or perturbation. A fixed action plan formed before the
disturbance becomes observable cannot do so.

**Witness.** Compose a known body, a published diagonal linear plant, a hidden
coloured disturbance, noisy observations, and a public tracking norm:

```text
x[t+1] = A x[t] + B u[t] + w[t]
y[t]   = x[t] + v[t]
```

The reference changes at public boundary events. The budget, noise laws, plant,
and quadratic action/tracking costs are public. Base episodes reset and contain
no absorbing state. A variant adds a published absorbing viability boundary.

**Controls.** The primary control replaces `y` with a shadow plant signal that
has matching marginals and autocorrelation but is causally unrelated to the
controlled plant. A frozen-disturbance control makes a constant offset
sufficient. The viability variant separates ordinary recovery from control near
irreversible failure.

**Transformations.** Channel permutation, joint sign reversal, and matched
rescaling of time, budget, disturbance, and costs preserve meaning. Reversing
only the measurement sign, shifting the reference without the cost target, or
adding an undeclared observation delay changes the correct feedback relation.

**Boundary and bracket.** Public and privileged solvers differ only because the
privileged view knows the disturbance realization. Report tracking cost,
recovery time, viability, both ceilings, and their gap. Compare against
inaction, best constant action, a fixed proportional controller, and the best
fixed action profile.

**Admission and transfer.** Admit only if the learner beats the fixed-profile
bracket on the witness, loses that advantage on the non-diagnostic control, and
recovers after gaps. The transfer prediction is faster acquisition of Card 07
closed-loop segments than from matched open-loop pretraining. Equal Card 07
curves falsify that prediction.

## Card 02: Predictive State

**Trunk:** T1 identification and regulation. **Fragment:** G0. **State:**
implemented, audited, rendered, and frontier-admitted by the bounded R10
learner pilot. This is source-family learner evidence, not transfer evidence.

**Claim.** The learner retains exactly the earlier public information that
changes action-conditioned futures. It does not retain equally salient but
irrelevant history.

**Witness.** A public early event deterministically sets a hidden mode. During
the following aliasing interval every mode emits identical observations. Later,
one action advances toward the public goal in one mode and retreats in another.
The mode cannot be safely rediscovered by probing. Episodes reset, have a fixed
budget, and charge a per-step cost. A separate variance variant holds predicted
means fixed while making the optimal action depend on outcome variance.

**Controls.** In the fully observable control the mode is emitted every step.
In the irrelevant-latch control, the event occurs but no action consequence
depends on it. In the memory-cost control, two latches occur, only one matters,
and carrying or acting on both is costly.

**Transformations.** Mode, observation, and non-discriminating action labels may
be permuted, and the latch may move within the admissible early interval.
Decorrelating the latch from the mode, changing the discriminating action, or
ending the aliasing interval early changes the required state.

**Boundary and bracket.** The public ambiguity becomes zero when the latch is
observed, so the privileged-minus-public gap is zero and vacuous. The
load-bearing number is the public ceiling after **ablating the latch from
history**: `97` becomes `48.5` on the witness and on the memory-cost control,
and is unchanged on the fully-observable and irrelevant-latch controls. Ablating
the *second* latch costs nothing anywhere. Compare against the exact memoryless
policy, fixed windows just below and above the required span — the boundary is
sharp at three — a constant-command policy, a last-latch policy, and the
mode-conditioned ceiling. No action sequence confined to the aliasing interval
separates the two modes.

Two implementation facts are not in this card text. The card's observable is the
**discriminating command, not the first action**: both modes open with the same
move, so a first-action orbit reports agreement exactly where the card claims a
difference. And two declared transformations — ending the aliasing interval
early, and republishing the mode — leave the world's ceiling and correct actions
untouched, so they are checked against what a latch-ablated learner can attain
rather than through the value orbit, which would have passed them vacuously.

**Admission and transfer.** The audit-verified R10 grouped-ActionQuery pilot
admitted contract `74b2d0da16ad3b31`: over 64 updates and 256 consumed
presentations, full-corpus macro argmax rose from `0.1667` to `0.9375` and the
final all-case-kind minimum was `0.8333`. (The configuration field retained a
stale `primary` name, but the classifier used every `by_case_kind` entry.) This
establishes a usable learner progress signal for this source family only; it
does not establish the transfer
prediction below. The memory-cost case is not an isolation negative: nothing
simpler is optimal on it, and what it catches is a policy that retains *too
much*, so its evidence is a designated-failure table rather than the bracket.
The variance variant is not implemented — it needs an objective
under which outcome spread changes the optimal action, and the deterministic G0
fragment has none, so the `P6 -> M2` dispute stays open and undecided by this
card. The transfer prediction is faster acquisition
of Card 05 than from fully observable pretraining. Equal Card 05 curves falsify
it.

## Card 03: Affordance and Reachability

**Trunk:** T1 identification and regulation. **Fragment:** G0. **State:**
implemented, audited, rendered, and piloted. The R10 pilot was valid but
inconclusive, so this family remains audited and deferred.

**Claim.** Before a failed attempt, the learner changes its action allocation
according to what this body can bring about in this environment.

**Witness.** Use a published finite configuration graph. A variable body drives
only part of the edge-label set; a coupling variant makes one action drive two
edges together. A short public calibration phase shows action effects without
directly publishing actuator support. The public goal may be unreachable. The
correct response is an immediate minimum-cost fallback, scored on the first
post-calibration decision. Episodes reset; a variant makes wasted budget
absorbing.

**Controls.** A frequency-matched control uses a fully capable body and makes
the same fraction of goals fail only because they exceed the budget. A
body/environment swap deletes environment edges instead of body support while
preserving the same reachable set. A reversal control publicly restores support
mid-episode, requiring behaviour to update immediately.

**Transformations.** Configuration, action, and edge labels and calibration
order may be permuted; the body/environment swap must preserve behaviour.
Changing which edges are unsupported, which channels are coupled, or making
calibration uninformative changes the reachable set or its identifiability.

**Boundary and bracket.** After calibration, support identification and
reachability are exact, so the public and privileged ceilings coincide; the
implemented family reports both and the gap is zero on every case. That holds
*because* calibration is mandatory, free, and exact — making it uninformative
reopens the gap, which is how the scaffold is shown to be load-bearing. Pulse
order is a constant and the audit checks its independence from the goal across
every cell. Compare against always-attempt, ignore-support,
try-once-then-fallback, plan-at-calibration, and the exact post-calibration
policy.

Three implementation facts are not in this card text and were found by
enumeration. The ring has nine cells and the budget is two decisions, because a
smaller ring leaves no goal that a fully capable body cannot reach and so
deletes the frequency-matched control. The invariance group is the ring's
*rotations* only: this body's commands are `0`, `+1`, `+2`, and `-1`, and a
reflection has no image for `+2`, so a reflection is checked as
meaning-changing. The calibration scaffold pulses `Leap` twice so that it leaves
the region the scored phase can reach; without that the body/environment swap
produced byte-identical public traces and its invariance verdict compared an
episode with itself. Three of six swappable contracts are publicly visible, and
the audit reports that count rather than claiming the transform bites
everywhere.

**Admission and transfer.** The audit-verified R10 pilot for contract
`2442b372a18e1d66` ran 64 updates and 256 consumed presentations but did not
meet frontier admission: macro argmax was `0.4667 -> 0.6000`, with final
all-case-kind minimum `0.2500`. This is an inconclusive bounded learner result,
not evidence against the family or a transfer result; R10 is closed and permits
no second repair. Its R10a decomposition certificate is stated in full below.
Scale the composite only under a separately justified Card-03-only scaling
profile. Admit in a later, separately declared decision only if unreachable
goals cause fallback on the first scored decision, frequency matching does not
reproduce that choice, the body/environment swap is invariant, and restored
support changes behaviour.
The restoration is *announced* at episode start: the fallback is absorbing, so an
unannounced restoration would arrive after a correctly ended episode and could
not require any behaviour to update. The absorbing-wasted-budget variant is not
implemented and no admission decision turns on it.
The transfer prediction is faster acquisition of Card 05 than from fixed-body
pretraining. Equal curves falsify it.

**Decomposition certificate.** The original relation is body-relative action
allocation before a failed attempt: identifying what this body can bring about
in this environment *and* planning inside a two-decision budget, including an
immediate minimum-cost fallback when the public goal is unreachable. The next
basic relation is public body identification alone — after the free exact
calibration scaffold, drive the published goal cell with the command this body
actually supports, where *which* supported command reaches that cell depends on
the identified support. It removes the planning and fallback decision, the
unreachable goal, and the budget that made planning necessary, leaving one
scored decision whose only content is the identified body. It retains the ring
configuration and its rotation-only invariance group, the mandatory free exact
calibration scaffold and its goal-independent constant pulse order, the hidden
actuator support with its exact post-calibration identification, the announced
restoration control, the body/environment swap invariance, the
uninformative-calibration information orbit, and the always-attempt and
ignore-support baselines. It is more compatible with the observed learner
profile because the R10 pilot's aggregate rose to `0.6000` while the
all-case-kind minimum stayed at `0.2500`: the learner acquired part of the
support and not the composite, and identification is exactly the factor this
family shares with Card 02, the one family R10 admitted. It is falsified if the
same fixed learner cannot reach predeclared exact full-support fit on a newly
audited identification-only family, or if an ignore-support policy succeeds on
that family's witness; the fully-capable control should instead restore the
fixed-command baseline. Re-entry requires that new version and contract hash to
pass its own semantic audit and its own fixed AdmissionProfile; only then may a
later version restore the fallback decision, then the frequency-matched budget
control, then unreachable goals, one factor at a time. The composite remains
deferred until every prerequisite stage has its own evidence.

**Stage A family.** `card03a`, contract `235c0a6ad2efb10d`, in
`crates/card03a-body-identification`. Nine cells, one scored decision, and two
actuators — `Leap` and `Vault` — that drive the same `+2` edge; the witness body
supports exactly one of them, so the published goal is reachable either way and
which command reaches it is a fact about the body rather than the configuration.
This is the card's declared coupling variant used as the whole content of a
single decision. Twelve cases render as twelve distinct public episodes.
Calibration identifies support exactly — diameter `1`, scored gap `0` — and
making it uninformative moves the public ceiling `99 -> 66` and the diameter
`1 -> 3`, which is the same information orbit the composite uses to show the
scaffold is load-bearing. The fully-capable control restores the ignore-support
baseline; the announced restoration arrives after the scaffold and defeats a
scaffold-following policy. Rotations preserve and the reflection is
meaning-changing, because it sends the `+2` edge to a `-2` edge this body has no
actuator for — the composite's rotations-only group, measured rather than
inherited. The body/environment swap preserves every public fact on all six
swappable contracts, and the audit reports that count rather than claiming the
transform bites everywhere. Run
`cargo run -p pretraining-card03a-body-identification --bin card03a-audit`.
Its R10b pilot is `support_fit_incomplete`: macro case-kind argmax rose
`0.5000 -> 0.8333` while both controls reached `1.0000` and the witness held at
`0.5000`. The learner acquired the two arms where the body does not have to be
read and not the one where it does, so removing the planning half did not move
the barrier. The R10c matched replication on a T4 reproduces every number
exactly.

## Card 04: Norm Swap

**Trunk:** T3 norm structure and selective control. **Fragment:** G0. **State:**
implemented, audited, rendered, and compatibility-characterized by R10. The
valid 64-update pilot is inconclusive, so the composite is deferred rather than
frontier-admitted.

**Claim.** With the public state and history fixed, changing the requested
outcome changes the correct action. Goal maintenance, inhibition, switching,
and viability are measured separately.

**Witness.** The implemented finite family uses a five-position ring, horizon
three, and a public known body. Goals and prohibitions are public; nothing is
epistemically hidden. Matched episode pairs share all randomness and history up
to the goal event, then receive different goals. Variants add irrelevant
distractors, a forbidden greedy move, a mid-episode superseding goal, or an
absorbing viability boundary.

**Controls.** A constant single-goal family makes a state-only policy optimal.
A goal-predictable-from-state family varies goals without requiring the goal
channel. Removing the prohibition makes greedy progress correct. Announcing a
switch at episode start makes one-shot planning sufficient.

**Transformations.** Configuration, goal, and action labels and irrelevant
distractor order may be permuted. Changing the predicate denoted by a goal,
moving the prohibited state, or changing a second goal from superseding to
composing changes the norm and therefore the correct action.

**Boundary and bracket.** Public and privileged information coincide on
eighteen of the twenty cases. They do **not** coincide on the two unannounced
switch witnesses: the second goal is by construction unpublished until it fires,
so a solver reading the contract reaches `98` where the exact policy for the
published norm reaches `97`, and the two take opposite first actions. This
correction came from rendering the family, not from the semantic audit, because
the audit's ambiguity gap compared the value function with itself. The audit
enumerates 20 cases and all 27 three-action sequences per case, checks a full
dihedral orbit, and establishes a state-only ceiling of `0.5`. Compare
state-only, last-goal, greedy-progress, plan-once, the exact goal-conditioned
policy, and the published-norm policy; the last two are both ceilings and are
excluded from the failure-mode bracket. Isolation requires that each control has
at least one baseline that is optimal there and fails its paired witness.

The learner is taught by the published-norm policy, never by the privileged
one. Twenty rendered episodes carry sixteen distinct public fingerprints,
because four case labels name contracts that are literally identical and differ
only in which family they are scored inside; a training mixture must count
episodes, not labels.

**Admission and transfer.** The R10 grouped-ActionQuery pilot for
`d975c3a646591ccf` used 64 updates and 256 consumed presentations: macro
argmax `0.3395 -> 0.4259`, final all-case-kind minimum `0.3333`. It misses the
fixed frontier profile and is learner/support-fit evidence, not a semantic
defect or transfer result. R10a therefore decomposes public goal use from the
switch and viability composite and defers this card; re-entry runs through the
certificate below and its own AdmissionProfile, not through another R10
repair. The transfer prediction is faster acquisition of Card 07 segment
selection than from single-goal pretraining. Equal Card 07 curves falsify it.

**Decomposition certificate.** The original relation is norm-conditioned action
selection with public state and history held fixed, composited over goal
maintenance, inhibition by a prohibition, mid-episode supersession by an
unannounced second goal, and an absorbing viability boundary. The next basic
relation is public goal use alone: with the start configuration fixed and one
goal published at episode start, the correct action is the one that settles the
published goal, and changing only the goal changes that action. It removes the
prohibition, the superseding second goal together with the interrupt and reveal
that carry it, the viability restriction, and the irrelevant distractors — and
with them every norm connective, leaving the `Settle` and `Visit` leaves alone.
It retains the ring and its full dihedral orbit, the public known body, matched
episode pairs that share all history up to the goal event, the
constant-single-goal and goal-predictable-from-state controls, the state-only
ceiling of `0.5` on the witness, the published-norm teacher, and the
goal-predicate meaning-changing transformation. It is more compatible with the
observed learner profile because Card 04's R10 pilot had both the lowest final
macro (`0.4259`) and the flattest gain of the five families while its public
and privileged information coincide on eighteen of its twenty cases: almost
nothing was epistemically withheld, so what the learner missed was the
composite norm structure rather than any hidden state. It is falsified if the
same fixed learner cannot reach predeclared exact full-support fit on a newly
audited goal-use family, or if a state-only policy succeeds on that family's
witness; the two controls should instead restore their state-only and
fixed-action baselines. Re-entry requires that new version and contract hash to
pass its own semantic audit and its own fixed AdmissionProfile; only then may a
later version restore the prohibition, then the announced switch, then the
unannounced switch, then the absorbing viability boundary, one factor at a
time. The composite remains deferred until every prerequisite stage has its own
evidence.

**Stage A family.** `card04a`, contract `06a36ac33c3f952b`, in
`crates/card04a-goal-use`. The same five-position ring, two decisions, one goal
published at episode start, and a single `Settle` leaf with no connectives left
to compose. Sixteen cases render as sixteen distinct public episodes — the
composite's own distinct-episode count, kept so that the stage differs from it
in the relation it asks for rather than in how much data it offers. The witness
is four start configurations each carrying the goal one step forward and one
step back, which is what makes the enumerated state-only ceiling exactly `0.5`
against `1.0` on both controls. The ambiguity gap is zero on every case *and*
vacuous, and the audit reports both rather than presenting the zero as evidence.
The full dihedral orbit holds; shifting what the goal symbol denotes is
meaning-changing, and the denotation is published so that transformation is a
change to the world the learner is told about rather than hidden state it must
guess. `Settle` and `Visit` coincide at this horizon, so the composite's
goal-predicate transformation has no bite here; that is recorded rather than
silently substituted. Run
`cargo run -p pretraining-card04a-goal-use --bin card04a-audit`.
Its R10b pilot is `support_fit_incomplete`. On CPU it was the flattest of the
four — macro case-kind argmax never moved from `0.3750` and the training loss
fell only `1.0985 -> 1.0720` — and it is the one family the R10c matched
replication moved, to `0.3750 -> 0.7500` on a T4 at fp16. It misses the barrier
on both, and on the matched learner its bracket is not restored either: the
goal-predictable control reaches `1.0000` and the witness `0.7500`, but the
constant-goal control, where a state-only policy is exactly optimal, is the
worst kind at `0.5000`. Removing the prohibition, the switch, and the viability
boundary left the composite's shortfall in place.

## Card 05: Active Experimentation

**Trunk:** T2 epistemic action. **Fragment:** G0. **State:** implemented,
audited, rendered, and compatibility-characterized by R10. The valid pilot is
inconclusive, so it remains audited rather than frontier-admitted.

**Claim.** The learner pays for information exactly when it changes a later
decision. Probe frequency alone is not evidence.

| Condition | Correct choice |
|---|---|
| Hidden gate affects the later commit | Probe. |
| Gate is already public | Do not probe. |
| Hidden gate does not affect value | Do not probe. |
| Probe costs more than blind commitment | Do not probe. |

**Witness.** A hidden binary gate determines which committed action reaches the
goal. Commitment is irreversible within the episode. One action reveals the
gate, makes no goal progress, and costs one step. A constructed control action
has identical cost and immediate value movement but reveals nothing. Instances
without this matched action are rejected rather than repaired after generation.

**Controls.** Publish the gate at episode start; retain uncertainty but make
both gated outcomes equally valuable; or raise probe cost above its expected
value. Additional variants provide high prediction error unrelated to the goal
and an observation-only probe whose result cannot change any action effect.

**Transformations.** Probe, gate, and observation labels and the pre-commit
availability window may be permuted. Making the reveal independent of the gate,
making both commits successful, or changing which gate value a commit favours
changes the decision value.

**Boundary and bracket.** Before probing, public ambiguity is the gate prior;
after probing, it is zero. The implemented family reports privileged `99`,
public-with-probe `98`, and public-without-probe `49.5`: the first non-vacuous
ambiguity gap in the portfolio, and a value-of-information of `48.5`. Compare
never-probe, always-probe, peek-instead, sham-instead, the exact public policy,
and a privileged gate-known reference. No action sequence that has not probed
separates the two gate values.

Three implementation facts are not in this card text. A third non-committing
action, `Peek`, was added: it buys a second hidden bit that no outcome depends
on. `Sham` alone cannot separate seeking information from seeking information
that *changes a decision*, so without it the `M5 -> M11b` dispute would have
been left unmeasured; with it, `agent_equivalence` reports the two values of
that bit as observationally distinguishable and outcome-identical. The family
parameters — gate visibility, whether the commits differ, and what a probe costs
— are **published**, because without them the witness and the equally-valuable
control are the same episode up to the first decision and no policy could behave
differently in them. And admission is judged **in expectation over a case kind**,
not per case: with the gate hidden the public ceiling is an average that no
single episode attains, so a per-case test would reject the optimal blind commit
on every instance where the coin fell the other way.

The novelty-driven baseline and the high-prediction-error variant are not
implemented. They need a prediction objective to make error measurable, and the
finite-G0 profile emits no future query, so the `M2 -> M5` dispute is not decided
by this card.

**Admission and transfer.** The R10 grouped-ActionQuery pilot for
`cbe39880124b9d2d` used 64 updates and 256 consumed presentations: macro
argmax `0.1250 -> 0.6875`, final all-case-kind minimum `0.2500`. It therefore
misses the fixed frontier barrier despite a rising aggregate score. R10a
decomposes reveal use from the value-of-probing decision under the certificate
below, retaining matched controls and the information boundary; this is a new
card-local decision, not a retrospective relaxation of R10. Admit only if probe
rate is high in the
witness and low in all three primary controls. Each control breaks exactly one clause of the
matched-control verdict and the audit names which: publishing the gate breaks
informativeness, making the probe consume the whole budget breaks cost matching,
and the equally-valuable control breaks none — there the probe is matched and
informative and simply worthless. The transfer prediction is faster
disambiguation in Card 08 than from information-free pretraining. Equal Card 08
curves falsify it.

**Decomposition certificate.** The original relation is paying for information
exactly when it changes a later decision: a matched-cost informative probe
bought before an irreversible commit, scored in expectation over a case kind
against three controls that each break exactly one clause of the
matched-control verdict. The next basic relation is reveal use alone — when a
free reveal publishes the gate at episode start, commit to the action that gate
value favours. It removes the purchase decision itself: the probe, its cost,
and the value-of-information comparison that ranked it against blind
commitment, so that no action in the reduced family reduces the surviving
candidate set and the matched-control verdict reports that absence rather than
asserting it. It retains the hidden binary gate and its prior, the irreversible
commit with its two gated outcomes, the matched-cost non-informative `Sham`
action, the gate-hidden and equally-valuable controls, the uninformative-reveal
information orbit, non-interference before the reveal fires,
expectation-over-case-kind scoring, and the blind-commit and
exact-public-policy baselines. It is more compatible with the observed learner
profile because Card 05's R10 curve rose furthest of the deferred four
(`0.1250 -> 0.6875`) while its all-case-kind minimum stayed at `0.2500`: the
aggregate moved and some case kind did not, and conditioning a commit on
already-published content is the strictly weaker half of deciding to buy that
content first. It is falsified if the same fixed learner cannot reach
predeclared exact full-support fit on a newly audited reveal-use family, or if
a fixed blind-commit policy succeeds on that family's witness; the gate-hidden
and equally-valuable controls should instead restore the blind average.
Re-entry requires that new version and contract hash to pass its own semantic
audit and its own fixed AdmissionProfile; only then may a later version restore
a free probe, then a matched-cost probe carrying a real value-of-information
decision, then the `Peek` and raised-cost controls, one factor at a time. The
composite remains deferred until every prerequisite stage has its own evidence.

**Stage A family.** `card05a`, contract `4941b2e1e1c8c390`, in
`crates/card05a-reveal-use`. Three outcome cells, two decisions, and three
actions: `Sham` and the two commits. Twelve cases render as six distinct public
episodes, because the decoy varies inside every kind and is published in none of
them — the same accounting fact the composite reports when its inconsequential
bit is never bought. On the witness the published gate closes the gap
completely, `99` public against `99` privileged; withholding it leaves `49.5`
against `99`, which is the composite's non-vacuous gap retained as the
gate-hidden control. The removal is audited rather than asserted: no
non-committing action reduces the surviving candidate set, and no opening is
worth more than committing at once. The matched-control check is still run on
`Sham` and reports that its ambiguity clause does *not* hold, which is the
correct answer for a family with nothing to buy and leaves stage B's restoration
anchored to an existing verdict. Run
`cargo run -p pretraining-card05a-reveal-use --bin card05a-audit`.
Its R10b pilot is `support_fit_incomplete`: macro case-kind argmax held at
`0.8333` from initialization to the rung, with both controls at `1.0000` and the
witness at `0.5000`. Conditioning a commit on already-published content is the
weaker half of buying that content, and this learner did not acquire it either.
The R10c matched replication on a T4 reproduces every number exactly.

## Card 06: Perceptual Organization

**Trunk:** T4 binding and relational abstraction. **Fragment:** G0. **State:**
implemented, audited, rendered, compatibility-characterized, and
decomposed/deferred after R10a. It is not frontier-admitted.

**Claim.** The learner binds observations to persistent causes rather than to
channel identity, keeps a cause individuated through absence, and re-identifies
it when it returns through different channels.

**Witness.** Exchangeable latent sources emit into a changing subset of public
channels. Assignment-change boundaries are public but assignments are not.
During occlusion, a source continues evolving while matched-marginal noise
occupies its channels and other sources move. Learner pulses provide
intervention evidence. The public goal names a source by interaction history,
not by index, and asks the learner to drive that source to a value.

**Controls.** Lock sources to channels; destroy cross-channel covariance while
preserving marginals; freeze all sources through absence; or give every source
a permanent identity tag. Each removes one need for history-based source
binding while matching the remaining structure.

**Transformations.** Source, channel, occlusion target, boundary timing, and
common value scale may change without changing meaning. Changing the source
named by an otherwise identical goal, hiding whether an assignment changed, or
using visibly mismatched occlusion noise changes the required relation or
invalidates the contrast.

**Boundary and bracket.** Exactly enumerate the posterior over source-channel
assignments for small instances. Report its within-episode ambiguity curve and
the public/known-assignment gap. Reject positional correlations between goal
naming and channel placement. Compare per-channel, channel-identity,
tag-following, assume-nothing-moved, exact assignment-posterior, and
known-assignment policies.

The implemented finite family uses two exchangeable drifting sources, two
channels, a public assignment-change boundary, and a two-decision horizon. Its
`Override` coupling resolves competing source and matched-marginal noise
writers; the same interruption seam advances source state in the witness and
freezes it without changing assignment in the frozen control. Across 36 seeded
cases, the shared posterior reaches `98` on the witness, while destroying
covariance leaves a public ceiling of `49` against a known-assignment ceiling
of `98`. Raw residual label ambiguity is quotiented to one agent-relevant class
on the witness. The audit computes noninterference and all preserving,
meaning-changing, and information-orbit verdicts rather than recording them as
labels. The 36 cases render as 32 distinct public episodes; mixture accounting
uses the latter.

**Admission and transfer.** The R10 grouped-ActionQuery pilot for
`76a08f38947c8cae` used 64 updates and 256 consumed presentations: macro
argmax `0.2500 -> 0.6875`, final all-case-kind minimum `0.5625`. It is
inconclusive under the fixed profile, not a world defect, but its monotone
improvement across the recorded cadence justifies one Card-06-only scaling
diagnostic with a newly fixed AdmissionProfile. That diagnostic neither
reopens R10 nor authorizes R11. The audit-verified R10a diagnostic completed
256 updates and 1,024 presentations. Exact full-support fit was false at 64,
128, and 256 updates. Macro case-kind argmax rose
`0.2500 -> 0.7250 -> 0.7875 -> 0.8750`, while the witness moved
`0.2500 -> 0.6250 -> 0.4375 -> 0.5000`; channel-locked,
frozen-through-absence, and shuffled-covariance controls reached `1.0000`, and
identity-tag reached `0.8750`. The fixed classification is
`support_fit_incomplete`. The controls show that the learner acquired simpler
shortcuts while failing the history-based binding witness, so this is a
learner/support-fit result rather than evidence of a degenerate world. More
undirected scaling is not authorized.

**Decomposition certificate.** The original relation is persistent-cause
binding across hidden channel reassignment, absence, continued latent motion,
matched-marginal noise, and a goal named by interaction history. The next basic
relation is binding the history-named source across one public channel-change
boundary while both source values remain continuously visible. It removes
absence, latent evolution while absent, and occlusion noise, but retains source
exchangeability, hidden assignment, the history-named goal, action effects,
source/channel permutations, noninterference, and the channel-locked and
identity-tag contrasts. This isolates relational reassignment from memory
through absence, matching the observed ability to learn the controls but not
the composite witness. It is falsified if the same fixed learner cannot reach
predeclared exact full-support fit on a newly audited visible-reassignment
family, or if a channel-identity policy succeeds on its visible-reassignment
witness; the controls should instead restore their associated simple
baselines. Re-entry requires that new version and contract hash pass its
semantic audit and fixed AdmissionProfile; only then may a later version add
absence without hidden
evolution, followed by evolution and matched noise one factor at a time. The
current composite remains deferred until every prerequisite stage has its own
evidence.

**Stage A family.** `card06a`, contract `b6f762bb89601096`, in
`crates/card06a-visible-reassignment`. Two exchangeable sources, two channels,
one public assignment-change boundary, and a two-decision horizon. A pulse
perturbs the source in the pulsed channel and the perturbation then appears
wherever that source went, so the history-named goal is answered by following a
value across a boundary rather than by remembering through an absence. Twenty
cases render as fourteen distinct public episodes. The channel is a shared
variable with exactly one writer, so the coupling rule is `Conflict` where the
composite's is `Override`; restoring the occlusion noise is what restores the
second writer. The public and known-assignment ceilings coincide at `98` — the
composite's `98`/`49` gap closes, and that is the removed factor stated as a
number rather than as an absence, with the
`values_made_invisible_across_the_boundary` information orbit dropping it back
to `49`. Raw residual ambiguity after a pulse is `2` and quotients to one
agent-relevant class. Channel-identity reaches `49` on the witness against `98`
for value tracking, and both surgical controls restore their simple baselines.
One verdict diverges from the composite deliberately: hiding the
assignment-change boundary is *preserving* here, because continuously visible
values already carry what the marker announces. It is declared and checked as
preserving rather than inherited as meaning-changing. Run
`cargo run -p pretraining-card06a-visible-reassignment --bin card06a-audit`.
Its R10b pilot is `support_fit_incomplete`: macro case-kind argmax rose
`0.2500 -> 0.7500` with all three kinds level at `0.7500`, so unlike the
composite's R10a diagnostic there is no control/witness separation to read. The
R10c matched replication on a T4 reproduces every number exactly. The
certificate's staging is therefore not entered: absence is not added on top of a
stage that has not itself been fit.

The original admission condition remains: the learner must beat channel-based
policies on the witness, each surgical control must restore its associated
simple baseline, and both preserving and changing orbits must separate
correctly. The downstream visual family contains near-identical movable
objects, an effector,
occlusion, motion during absence, and goals that name objects by interaction
history. Every arm uses the same fixed pixel encoder trained only on downstream
visual data; only core initialization differs. Compare scratch, this card, a
channel-locked pretraining control, and a shuffled-covariance control. The
predicted advantage is largest under occlusion with intervening motion and
smallest without occlusion. No advantage, or an advantage uniform across all
conditions, falsifies transfer of the named binding relation.

## Card 07: Temporal Composition

**Trunk:** T5 temporal composition. **Fragment:** G2. **State:** specified but
gated on attributable Card 01 regulation and Card 04 goal conditioning.

**Claim.** The learner reuses closed-loop procedures in unseen orders, detects
their completion, handles interruption, and resumes the interrupted procedure
from its current state rather than restarting it.

**Witness.** Three disturbed regulation segments have different dynamics and
sub-outcomes. Training exposes only some orders; test uses new orders. One
shared budget spans the entire sequence. Segment boundaries are absent at test.
An unannounced interrupt inserts a public priority outcome, after which the
previous segment must resume from its current state. A flattened-affordance
variant removes reachability structure while preserving co-occurrence.

**Controls.** Reuse training orders at test; remove disturbance so segments are
open-loop strings; reset an interrupted segment so restarting is correct; or
publish completion boundaries at test.

**Transformations.** Segment identity, channels, durations, budget scale,
interrupt time, and training exposure order may be permuted. Changing the
required order, allowing abandonment after interruption, or replacing the
shared budget with per-segment budgets changes the required composite policy.

**Boundary and bracket.** G2 permits declared bounds, not an exact ceiling.
Restrict the horizon until bounds are decision-useful. Compare a sequence
memorizer, flat policy, always-restart policy, greedy nearest-subgoal policy, and
a known-good scheduler composed from admitted Card 01 controllers. Report any
remaining bound width; if it cannot separate learner performance, the fragment
is inconclusive rather than the learner deficient.

**Admission and transfer.** Admit only after isolated segments are already
reliable and the learner beats memorization and open-loop controls on unseen
orders, detects boundaries, and resumes from state. The transfer prediction is
faster acquisition of Card 08 multi-segment demonstrations than from fixed-order
pretraining. Equal Card 08 curves falsify it.

## Card 08: Physical Prompting

**Trunk:** T6 other agents and physical prompting. **Fragment:** G2. **State:**
specified but gated on T1 execution, Card 04 goal conditioning, and Card 06
entity binding. Card 05 supplies a cross-card disambiguation prediction.

**Claim.** The learner infers the outcome another agent intended, separates it
from that agent's movement, and realizes the outcome through a different body.

**Witness.** Demonstrator and learner act on the same finite configuration but
have different action support and coupling. The demonstrator's action sequence
is unexecutable by the learner, and the closest copy produces a different
outcome; generation must verify both facts. The goal is never published. The
learner observes environmental effects and optionally action tokens. In the
rational-imitation condition, a visible demonstrator-only constraint explains
an inefficient detour, while the learner should take its own efficient route.
A relational-goal variant requires matching a relation rather than the
demonstrator's final absolute configuration.

**Controls.** Give both agents the same body; make the demonstration compatible
with two goals; remove the constraint so an efficient demonstration is
non-diagnostic; or add salient but outcome-irrelevant demonstrator motion.

**Transformations.** Configuration labels, non-copyable demonstrator bodies,
speed, channel assignment, and demonstration count may vary. Changing the goal
behind an ambiguous demonstration, removing the reason for a detour, or
replacing demonstrator-caused effects with disturbance changes the inference
problem.

**Boundary and bracket.** `identify(public, history)` returns the goals still
compatible with the demonstration. It must collapse to one in the witness and
remain greater than one in the ambiguity control. Reject correlations from
demonstration length, start state, body identity, or learner initial state to
goal identity. Compare movement matching, detour copying, attention following,
final-outcome matching, first-hypothesis commitment, and a privileged
goal-known policy.

**Admission and transfer.** Admit only if the learner rejects copying when
bodies differ, takes its own efficient route, responds safely to ambiguous
goals, and exceeds final-state matching on relational goals. A later visual
demonstration contract is written only if Card 06's visual transfer survives;
it must reuse that downstream family and frozen encoder. Equal visual learning
curves from abstract-pretrained and scratch cores falsify the physical-prompt
transfer prediction.

## Card completion record

When a card changes lifecycle state, record it in the progress chart in
`DEVELOPMENT-PATH.md`. A result record is valid only if it names the card and
contract hash, witness and control arms, preserving and changing orbit results,
public and privileged bounds, ambiguity, baselines, seeds, learner checkpoint,
total cost, and the progress-chart decision it changes.
