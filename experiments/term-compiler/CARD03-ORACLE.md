# Card 03 oracle for the term-compiler spike

This is an isolated experimental specification for R6 / the Card 03 affordance
family. It is not a change to the existing card, its contract, or the main
workspace. The existing handwritten implementation is the oracle; a compiled
term must agree with it before any migration or generated-world claim is made.

## Purpose and scope

Card 03 tests whether behavior changes with body-relative reachability before a
failed attempt. It is especially useful for the term compiler because the same
observable transition can arise from two different semantic sources:

```text
body limitation       = the body does not drive an actuator
environment deletion  = the environment has no edge for that actuator
```

The oracle requires these sources to remain typed and inspectable, while their
composed behavior agrees when the resulting reachable set is the same. An
implementation that stores only one opaque `can_move` predicate is behaviorally
insufficient: it could pass a few rollouts while making the body/environment
invariance claim impossible to audit.

The minimum experiment is a differential harness. It should compile a Card 03
term and run the existing handwritten `Affordance` fragment over the same
contracts, sequences, public histories, and rendered episodes. No learner, GPU,
remote run, or main-workspace dependency is required.

## Reference fixture

The fixture is the current audited family, not a new generated family:

| Quantity | Required oracle value |
|---|---:|
| Ring cells | 9 |
| Scored horizon | 2 |
| Actions | `Hold`, `Step`, `Leap`, `Back`, `Fallback` |
| Movement actuators | `Step = +1`, `Leap = +2`, `Back = -1` |
| Full body support | `{Step, Leap, Back}` |
| Admissible hidden bodies | all 8 subsets of the 3 movement actuators |
| Goal reward / movement cost / fallback reward | `100 / 1 / 50` |
| Calibration pulses | `Step, Leap, Leap, Back` |
| Cases | 12 (5 case kinds) |
| Exhaustive scored sequences | `5^2 = 25` |
| Existing family contract hash | `2442b372a18e1d66` |

The term fixture must express, as separate components, a body action-support
restriction, a ring environment with optional blocked edges, a calibration
scaffold, a public goal, and an optional announced support restoration. The
scored transition is the conjunction of body support and environment-edge
presence; failure to move leaves the cell unchanged. `Fallback` is absorbing in
value and ends the rendered episode's decision stream.

The existing `card_cases()` is the case source of truth. Its semantic pairs are:

| Kind | Start → goal | Body/support | Reachable | Ceiling | Optimal first actions |
|---|---|---|---:|---:|---|
| witness unreachable fallback (×2) | `0 → 8`, `0 → 2` | `{Step, Leap}`, `{Back}` | no | 50 | `{Fallback}` |
| witness reachable attempt (×2) | `0 → 3`, `0 → 7` | `{Step, Leap}`, `{Back}` | yes | 98 | `{Step, Leap}`, `{Back}` |
| witness restore (×2) | `0 → 8`, `0 → 2` | same limited bodies; restore `Back` or `Leap` after step 0 | yes after reveal | 98 | `{Hold, Back}`, `{Hold, Step, Leap}` |
| negative no restore (×2) | `0 → 8`, `0 → 2` | same two limited bodies | no | 50 | `{Fallback}` |
| negative frequency matched (×4) | `0 → 5`, `0 → 6`, `0 → 1`, `0 → 3` | full body | no/no/yes/yes | 50/50/99/98 | `{Fallback}`, `{Fallback}`, `{Step}`, `{Step, Leap}` |

The duplicate no-restore labels and unreachable-fallback labels intentionally
share public episodes. Case labels are generator/scoring metadata, not learner
inputs; a harness must retain them for bracket scoring while reporting the
rendered episode fingerprint separately.

## Differential gates

Each gate compares the compiled and handwritten paths on the same fixture. A
failure is a compiler-spike failure, not permission to relax the oracle.

### 1. Transition and exhaustive value parity

For every case and all 25 action sequences of length two, require equality of:

- the step-by-step cell trajectory, including the absolute `executed` index;
- `Outcome` (`value`, `reached_goal`, `fell_back`, `fallback_step`, `final_cell`);
- `goal_is_reachable`;
- the value ceiling and the complete set of ceiling-achieving sequences; and
- the deduplicated optimal first-action set.

The equality must hold for both ordinary cases and the two restoration cases.
Restoration is time-dependent: a restored actuator is unsupported at executed
index 0 and supported at index 1 for the fixture. A compiler that rebases a
mid-episode solve or clamps the clock incorrectly must fail this gate.

The expected aggregate counts are 12 cases and 25 sequences per case. The
compiler may use a generic finite enumerator, but it must not compare only one
teacher trajectory.

### 2. Body/environment separation and swap invariance

For every body-limited, no-restore contract (`support != full_body()` and
`restore == None`), construct the equivalent environment twin using the oracle's
`body_environment_swap` semantics:

1. make the body fully capable;
2. delete each withheld actuator's edge at every cell reachable during the
   scored phase; and
3. keep start, goal, calibration, reward, and action vocabulary fixed.

Require for each of the six swappable contracts:

- body support differs (`limited` versus full);
- the environment-side twin has non-empty blocked edges;
- all 25 sequence outcomes and trajectories are equal;
- value ceilings are equal;
- optimal first-action sets are equal; and
- reachability is equal.

The public distinction is part of the test, not an incidental diagnostic. The
calibration trace/public trace must differ for the three visible swaps, and the
harness must report exactly `6` swappable contracts and `3` publicly visible
swaps for this fixture. It must not demand visibility from every swap: three
contracts do not leave the scored region during calibration and are therefore
publicly identical. Treating an invisible swap as evidence would compare an
episode with itself.

This is the central embodiment oracle: generated semantics must preserve the
behavioral quotient while retaining whether a limitation came from the body or
from the environment. A term compiler that erases that provenance cannot
support Card 03's preserving transformation or Card 08's later body-transfer
claims.

### 3. Action support and reachability

The compiler's typed body term must expose support as an actuator-level
restriction, not as a configuration viability restriction:

- `Hold` and `Fallback` remain available for every body;
- only `Step`, `Leap`, and `Back` are body-supported/unsupported;
- a body restriction forbids no ring cell and has no viability boundary;
- an environment edge deletion forbids a transition from a particular cell;
- `Step`, `Leap`, and `Back` retain their distinct displacements; and
- all eight support subsets are representable and enumerable.

For each case, compare the exact reachable set under the two-step budget. The
frequency-matched full-body negative is required: the ring must be large enough
that a fully capable body cannot reach goals `5` and `6` in two steps. A smaller
ring or an action set that loses `Leap` deletes this control and is not Card 03
parity.

The paired allocation contrast must hold for the exact public policy:

- same body, unreachable goal → first action is `Fallback`;
- same body, reachable goal → first action is not `Fallback`;
- `IgnoreSupport` passes the frequency-matched negative but fails the limited
  unreachable witness; and
- `AlwaysAttempt` passes reachable cases but fails the unreachable witness.

These baseline failures are useful compiler checks because they ensure the
generator has not made the answer a goal-frequency or shortest-route lookup.

### 4. Sensor and public-information boundary

The compiled term must provide separate public and privileged views. Public
records may contain the calibration observations, public goal, announced
restoration condition, executed actions, and scored configuration observations.
They must not contain:

- the support set or a body identifier field;
- blocked-edge metadata;
- `goal_is_reachable` or a privileged ceiling; or
- generator parameters, seeds, or case labels.

The learner-facing port schema is constant across body realizations: nine unit
observations and five signed actuators. Support cannot be inferred from schema
shape. The public trace must include the fixed calibration outputs and the
announcement when present; the scaffold pulse order and resulting trace must be
independent of the goal across all `12 * 9 = 108` goal substitutions.

Identification checks:

- before calibration, the body ambiguity set contains all 8 admissible bodies;
- full calibration leaves diameter 1 for every case;
- uninformative calibration leaves diameter 8 and reopens a positive ambiguity
  gap on at least one case; and
- without calibration, two bodies that never receive a withheld actuator are
  publicly indistinguishable until an attempt, while full calibration separates
  them before any scored attempt.

Rendering checks:

- an uninformative-calibration term must be rejected when using the exact
  post-calibration teacher (`TeacherWouldLeak` semantics);
- only action-query records carry supervision targets;
- restoration is announced before the first scored decision;
- fallback offers one decision and then ends the episode; and
- all 12 episodes pass the existing boundary round-trip, with 10 distinct
  public fingerprints, maximum 43 canonical records, and maximum 44 profiled
  records.

The exact policy must be derived from the identified public body and public
announcement, never from the privileged support or edge fields. This prevents
the compiler from making a privileged teacher look like a public ceiling.

### 5. Audit and orbit parity

The compiled family must reproduce the audit-level predicates, not just the
rollout table:

- calibrated body identification everywhere;
- zero scored-phase ambiguity gap after full calibration;
- a reopened gap under uninformative calibration;
- goal-independent scaffold;
- all eight non-identity rotations semantics-preserving;
- the body-limit/environment-deletion orbit semantics-preserving;
- the reflection with no image for `Leap` semantics-changing; and
- changing the withheld actuator semantics-changing.

The rotation orbit is the ring's rotation subgroup only. Do not silently apply
the full dihedral group: reflection exchanges directions but this action set has
`+2` and no `-2`. The compiled report should preserve the distinction between a
preserving transformation (same ceiling and corresponding actions) and a
meaning-changing transformation (the expected ceiling/action change).

## Contract and hash expectations

The existing fixture's semantic family hash is the compatibility anchor:
`2442b372a18e1d66`. If the compiler claims exact Card 03 parity, the compiled
fixture must either emit this hash or explicitly expose a compatibility mapping
to it. A different opaque hash with no mapping is not parity evidence.

The compiler should make hashing canonical and provenance-aware:

- include typed ports, action meanings, body support domain, environment edge
  semantics, calibration/reveal behavior, scoring, horizons, controls, and
  preserving/changing transformations;
- canonicalize order-insensitive sets and blocked-edge lists;
- distinguish a body restriction from an environment deletion even when their
  composed behavior is equivalent;
- exclude generator-only seeds and construction metadata from the family hash;
- expose an instance seed/replay record separately; and
- change the family hash for a semantic mutation (for example changing
  calibration to uninformative, deleting a different edge, changing `Leap`, or
  changing reveal timing).

For a newly generated family, a new semantic hash is expected and requires a
new audit. A new seed of an unchanged family should retain the family hash and
change only instance metadata. Never overwrite the existing Card 03 receipt or
silently relabel a generated term as contract `2442b372a18e1d66`.

## Minimal compiler API implications

The spike need not implement a universal language. It does need these seams so
the oracle can be expressed without reaching through private fields:

```text
compile(term) -> CompiledFragment
CompiledFragment::{
    fragment(),
    public_view(history),
    privileged_view(state),
    generator_metadata(),
    family_hash(),
    audit_metadata(),
}

Term::{
    world(body, environment, scaffold, goal, scoring),
    body(action_support, actuation_map),
    environment(configuration_graph, blocked_edges),
    reveal(condition, guard),
}
```

Names are illustrative; the required properties are not. In particular:

1. `BodyTerm` and `EnvironmentTerm` must be distinct typed nodes.
2. A compiled fragment must still implement the existing generic `Fragment`
   and `PubliclyObservable` contracts, or provide a thin adapter with identical
   semantics. Existing `value_bounds`, `optimal_actions_from`, ambiguity,
   orbit, and noninterference machinery should be reusable.
3. Public rendering must consume a public view, while audits may consume a
   privileged view explicitly. There must be no implicit conversion from
   privileged fields to learner records.
4. The compiler must expose canonical transitions with the absolute step index;
   a mid-episode query cannot reset or rebase time-dependent reveals.
5. The term must retain case labels, seeds, and construction metadata only in a
   generator/audit channel. They cannot enter public histories or family hashes
   accidentally.
6. A compile error must be available for ill-typed terms: missing port types,
   body/environment wiring that bypasses the declared boundary, a monitor that
   feeds back into an actuator, a public query of support, or a teacher requiring
   unannounced privileged state.

## Stop/merge rule for this spike

Stop at the spike if any differential gate fails, if the compiler can only pass
by weakening the public boundary, or if it cannot represent body/environment
provenance. Do not modify the handwritten Card 03 evaluator to accommodate it.

Only after all gates pass should the parent task consider a follow-on generated
world experiment. Even then, keep the compiler and handwritten evaluator in
dual-path mode and compare them on every generated instance until a separately
versioned family audit is complete.

