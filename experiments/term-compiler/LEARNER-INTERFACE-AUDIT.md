# Learner Interface Audit

Status: standalone experimental specification. This document records the
interface boundary that a future term-compiler family would have to satisfy;
it does not migrate a card, alter the current learner ABI, or make a main-flow
claim.

The motivating question is whether the compiler can eventually generate worlds
whose goals, observations, actions, and demonstrations are richer than the
current finite-card transcripts. The answer is: the current boundary is a
sound public event carrier for one typed body at a time, but it is not yet a
complete carrier for language-conditioned or cross-embodiment demonstrations.
Those additions belong at an explicit interface adapter, not in opaque key or
condition conventions.

## 1. Scope and terminology

This audit separates four things that are easy to conflate:

| Term | Meaning | Current owner |
| --- | --- | --- |
| Goal denotation | The world-level proposition or target relation whose satisfaction is evaluated | World/term semantics; private unless deliberately revealed |
| Goal carrier | The public representation from which the learner may infer or condition on that denotation | Canonical public event, presently a named channel/key plus numeric value or selection |
| Prompt | A learner-facing carrier for a task instruction, such as text, language tokens, or a structured relation | Not yet defined in the canonical event ABI |
| Target/supervision | Privileged loss information used to train or score a prediction at a public record | Separate supervision table/parallel arrays |

The distinction matters for a hidden-goal task: the denotation may remain
private while a demonstration, observation history, or language prompt is
public. A target may identify the correct action for training, but must not
become an input feature.

RDDL is paused at Gate 0 in this spike: no RDDL source lowering or learner
family admission is implied here. RLDS is used only as interchange inspiration
for episode/step/agent separation; it is not imported as a project dependency
or treated as the project's semantic ABI. ICRT is only an architectural
reference for a trajectory learner with continuous event adapters; the
project-owned event meanings and information boundary remain authoritative.

## 2. Current sequence ABI

The finite-G0 path is:

```text
card transcript
  -> G0Episode(schema, horizon, groups)
  -> canonical public Episode + SupervisionTable
  -> PublicRow + SupervisionRow
  -> LearningToken(public, supervision)
  -> optional profiled header
  -> padded Python batch
  -> role/key/payload event embeddings and action/future heads
```

`G0Episode` contains exactly one public `PortSchema`, one decision horizon,
and ordered simultaneity groups ([`crates/g0-render/src/lib.rs:318-325`](../../crates/g0-render/src/lib.rs:318)).
The renderer inserts the schema as group zero ([`crates/g0-render/src/lib.rs:444-472`](../../crates/g0-render/src/lib.rs:444)); card-authored groups are
renumbered after it ([`crates/g0-render/src/lib.rs:574-600`](../../crates/g0-render/src/lib.rs:574)).

The current group vocabulary has these meanings:

| Group fact | Public carrier | Current semantic use |
| --- | --- | --- |
| `Boundary` | Episode key and categorical subtype | Calibration/task/episode brackets |
| `Condition` | Typed key namespace, contract-local code, bounded quantity | Explicitly revealed condition or clock |
| `Goal` | Observation key plus `Selection` or numeric `Value` | Symbolic goal carrier |
| `Observation` | Observation key plus `Selection` or numeric `Value` | Public state/effect carrier |
| `ActionQuery` | Actuator key plus remaining horizon | One alternative in a decision set |
| `ActionExecuted` | Actuator key and executed marker | Public behavior/history |

The `selected` bit on an `ActionQuery` is not public. The renderer converts
it into a supervision entry, while the public query carries a zero command,
its key, and its horizon ([`crates/g0-render/src/lib.rs:532-558`](../../crates/g0-render/src/lib.rs:532)).
The executed action is public ([`crates/g0-render/src/lib.rs:560-570`](../../crates/g0-render/src/lib.rs:560)).

At the canonical layer, `PublicRecord` has no supervision or generator field
([`crates/canonical-event/src/lib.rs:521-527`](../../crates/canonical-event/src/lib.rs:521));
`PublicEpisode` has no seed, instance index, or hidden state
([`crates/canonical-event/src/lib.rs:650-657`](../../crates/canonical-event/src/lib.rs:650)).
Supervision is addressed separately by record position
([`crates/canonical-event/src/lib.rs:720-755`](../../crates/canonical-event/src/lib.rs:720)).

The profiled path prefixes one isolated declaration selecting the interpretation
profile; the header has no supervision ([`crates/profiled-event/src/lib.rs:194-204`](../../crates/profiled-event/src/lib.rs:194)).
It declares the slot interpretation, not the card family. This is important:
family labels must not become a free learner feature.

The padded batch exposes:

```text
role_ids                 [batch, time]
key_ids                  [batch, time]
position_ids             [batch, time]
payloads                 [batch, time, 8]
attention_mask           [batch, time]
action_targets           [batch, time, 16]
action_target_mask       [batch, time, 16]
future_targets           [batch, time]
future_target_mask       [batch, time]
```

This layout is constructed in [`crates/world-py/src/lib.rs:450-564`](../../crates/world-py/src/lib.rs:450).
Python's `MODEL_FIELDS` is the exact model-input allowlist
([`python/pretraining_experiments/data.py:12-22`](../../python/pretraining_experiments/data.py:12));
everything else returned by the extension is metadata and must remain outside
the model dictionary.

## 3. Carriers and compatibility

### 3.1 Symbolic goals

Supported today. `Goal` and `Observation` carry either a numeric channel value
or a selection of a named key ([`crates/canonical-event/src/lib.rs:315-328`](../../crates/canonical-event/src/lib.rs:315)).
Card 03, for example, publishes the goal as a selected observation key
([`crates/card03-affordance/src/render.rs:97-104`](../../crates/card03-affordance/src/render.rs:97)).

This is a carrier, not a universal goal language. A key such as `7` denotes a
contract-local observation channel only because the episode schema and card
semantics give it that meaning.

### 3.2 Language goals and prompts

Not wire-native today. There is no text field, tokenizer, vocabulary, prompt
span, or language modality in `PublicFact::Goal`; its content is only
`ChannelContent` ([`crates/canonical-event/src/lib.rs:466-470`](../../crates/canonical-event/src/lib.rs:466)).
The canonical-event crate explicitly does not introduce a tokenizer or model
([`crates/canonical-event/src/lib.rs:9-18`](../../crates/canonical-event/src/lib.rs:9)).

There is an adapter escape hatch: the Python model accepts an optional
`canonical_content_embeds` tensor with one hidden vector per event position and
adds it to the role/key/payload embedding ([`python/pretraining_experiments/model.py:179-199`](../../python/pretraining_experiments/model.py:179)).
That supports a language encoder which pools or aligns a prompt into event
positions. It does not, by itself, define prompt semantics, causal placement,
or language-token supervision. The standard training loop currently calls the
model with only the batch fields ([`python/pretraining_experiments/train.py:299-308`](../../python/pretraining_experiments/train.py:299));
adapter plumbing is therefore still required.

### 3.3 Goal-observation pairs

Supported when both are symbolic/numeric and public. A goal can name an
observation channel, and subsequent observations can expose values or
selections for that channel. The carrier does not assert that the goal is
reachable, optimal, or satisfied. Those remain world-side denotations and
evaluation facts.

Conditions are typed but intentionally contract-local: they carry a namespace,
code, and bounded quantity ([`crates/canonical-event/src/lib.rs:454-465`](../../crates/canonical-event/src/lib.rs:454)).
They are not a generic privacy mechanism. Any condition placed in a `G0Episode`
is public by construction. Card renderers must therefore decide whether the
condition is an announced fact or a forbidden shortcut.

### 3.4 Single-body action trajectories

Supported. A schema declares observation and actuator ports
([`crates/g0-render/src/lib.rs:211-232`](../../crates/g0-render/src/lib.rs:211));
queries name declared actuators, and execution records publish the chosen
actuator. The current finite renderer can vary the number and identities of
local ports across episodes, subject to the fixed action head and payload ABI.

### 3.5 Demonstrations across embodiments

Not supported as a first-class representation. The current event types have no
actor/agent field, no demonstrator-versus-learner role, no action namespace, and
no per-agent action mask. `KeyNamespace` distinguishes only observation,
actuator, and episode keys ([`crates/canonical-event/src/lib.rs:70-78`](../../crates/canonical-event/src/lib.rs:70)).
One `G0Episode` also has one schema rather than a schema per actor.

Concatenating two transcripts or encoding actor identity in key numbers or
condition codes would be an undocumented convention and a likely shortcut.
It would not establish the Card 08 requirement that a demonstrator action can
be unexecutable by the learner ([`CARDS.md:516-527`](../../CARDS.md:516)).

The smallest principled extension is an actor-scoped interface descriptor:

```text
ActorId
ActionNamespace / PortSet
  declared actions and effect units
  learner-executable mask
PublicAction(actor, namespace-local-action, timestamp)
```

The public record must carry only the actor/action identity needed by the task;
privileged action optimality remains in the target table. Action names must be
canonicalized independently of array position so a demonstrator's alphabet
cannot silently become the learner's output head.

## 4. Prompt, target, and causal boundaries

The current system is intentionally parallel-array rather than a causal-loss
specification. `action_targets` and masks are available at every padded
position, but only query rows receive action supervision. The model's ordinary
loss is masked L1; a grouped action-query objective uses decision-group IDs as
loss addressing rather than learner input ([`python/pretraining_experiments/model.py:214-256`](../../python/pretraining_experiments/model.py:214)).

For a future prompt or demonstration, the conformance rule should be:

1. Prompt/goal carrier positions are public inputs and have no target by
   default.
2. Demonstrator actions are public history only unless a separate explicit
   imitation target is requested.
3. Learner action targets attach only to learner action queries and only where
   the learner's action mask says the action is executable.
4. Future-state or success targets require an explicit public-observation
   contract; they must not be read from hidden world state merely because a
   renderer can compute them.
5. Padding and metadata never receive loss.

This is causal in the information-boundary sense: a target may label a public
prefix, but it must not alter the prefix. The existing renderer tests already
exercise the strongest local invariant: changing the teacher's selected action
does not change public bytes ([`crates/g0-render/src/lib.rs:835-867`](../../crates/g0-render/src/lib.rs:835)).

For demonstrations, this must be strengthened to prohibit target leakage from
the demonstrator's future, the hidden goal, and the learner's unavailable
actions. A target should be addressed by `(episode, time, actor, action)` after
actor support exists, not by an untyped array slot alone.

## 5. Vision and language content embeddings

The current model composes three fixed event embeddings — role, key, and the
eight-float payload — and may add an external continuous embedding
([`python/pretraining_experiments/model.py:187-199`](../../python/pretraining_experiments/model.py:187)).
This gives a useful adapter boundary:

```text
raw modality (image/audio/text)
  -> frozen or separately trained encoder
  -> [batch, event_positions, hidden_size]
  -> canonical_content_embeds
  -> same trajectory transformer and heads
```

The adapter must define alignment. One image per observation event, one pooled
prompt vector at a goal event, and a variable-length prompt packed into
dedicated event positions are different protocols. None should be inferred
from payload slots.

The current ABI can therefore host:

| Carrier | Current status | Required work |
| --- | --- | --- |
| Symbolic goal key/selection | Compatible | None beyond card renderer |
| Numeric goal value | Compatible | Declare bounds/units |
| Text prompt as raw tokens | Incompatible | New adapter/event-position convention |
| Text prompt as pooled embedding | Adapter-compatible | Encoder, alignment, training-path plumbing |
| Numeric observation | Compatible | Existing payload schema |
| Vision observation as raw bytes | Incompatible | New modality transport, not hidden payload abuse |
| Vision observation as aligned embedding | Adapter-compatible | Encoder, alignment, leakage tests |

The existing adapter test proves only that gradients can flow through an
external embedding tensor; it does not prove visual learning or transfer
([`python/tests/test_model_integration.py:177-192`](../../python/tests/test_model_integration.py:177)).

## 6. Metadata and public/private separation

Learner-visible public events must exclude seed, instance index, contract hash,
family name, case labels, privileged state, and generator diagnostics. The
canonical public episode has no fields for the first four
([`crates/canonical-event/src/lib.rs:650-657`](../../crates/canonical-event/src/lib.rs:650)).

The PyO3 G0 functions nevertheless return evaluator metadata alongside model
arrays. The implementation documents that family identity, labels, fingerprints,
and hashes are separate metadata ([`crates/world-py/src/lib.rs:683-688`](../../crates/world-py/src/lib.rs:683));
the Python wrapper enforces the split by tensorizing only `MODEL_FIELDS`
([`python/pretraining_experiments/data.py:118-141`](../../python/pretraining_experiments/data.py:118)).
The separation is thus an API discipline, not a property of an untrusted raw
dictionary. A future interface should return distinct public-input and
evaluator-metadata objects where possible.

Conditions need the same discipline. The shared renderer can reject a teacher
whose optimal action depends on unpublished state (`TeacherWouldLeak`), but it
cannot infer whether a card's condition code is secretly generator metadata
([`crates/g0-render/src/lib.rs:120-129`](../../crates/g0-render/src/lib.rs:120)).
Each source term must construct a public view explicitly and pass admission
tests for hidden-state invariance.

## 7. Compatibility matrix

| Capability | Current canonical/G0 ABI | Term-compiler implication | Admission status |
| --- | --- | --- | --- |
| Typed finite body ports | Yes, one schema per episode | Lower body ports to `PortSchema` | Eligible for isolated adapter test |
| Finite environment trajectory | Yes, as public observation/action groups | Lower execution trace only | Eligible |
| Public conditions/reveals | Yes, opaque contract-local code | Lower only explicitly public reveals | Eligible with leak audit |
| Symbolic goals | Yes | Lower key/selection denotation | Eligible |
| Language goal carrier | No raw carrier; external embedding hook exists | Add prompt adapter and alignment contract | Blocked on interface witness |
| Numeric/selection observations | Yes | Lower typed observations | Eligible |
| Vision observations | No raw modality; external embedding hook exists | Add modality adapter, not payload overloading | Blocked on adapter witness |
| One action alphabet | Yes | Lower action queries/executions | Eligible |
| Multiple actor/action alphabets | No actor/action namespace | Extend canonical interface first | Blocked |
| Separate public inputs and targets | Yes structurally, then parallel arrays | Preserve address identity through extensions | Eligible |
| Causal action loss | Partial: row masks and optional grouping | Add actor/time/action support masks | Blocked for demonstrations |
| RDDL factored source | Not yet lowered | Keep paused at Gate 0 | Paused |

## 8. Minimal conformance witness

Before additional factored work, build one isolated interface fixture, not a
new card and not a main-flow migration. Its public sequence should contain:

```text
schema
task_reset
goal carrier (symbolic first; optional language embedding sidecar)
initial observation
demonstrator action + resulting public observation
learner action-query alternatives
learner executed action + resulting public observation
episode_end
```

The fixture must have two actors with disjoint local action sets. At least one
demonstrator action must be unexecutable by the learner. The hidden goal should
be a relation, not a public absolute cell. The fixture should provide:

- public records;
- a separate target table with learner-only action targets;
- evaluator metadata containing seed/hash/denotation, kept outside model input;
- a per-actor executable-action mask;
- a language/prompt embedding sidecar at the declared goal position; and
- an optional visual observation embedding at one observation position.

Required tests:

1. Public round-trip preserves event kind, actor, action namespace, key, and
   simultaneity.
2. Renaming local action IDs while preserving the declared namespace preserves
   denotation and changes no target for an unrelated actor.
3. Changing only supervision changes no public event or embedding input.
4. Removing the hidden goal from evaluator metadata leaves model inputs
   unchanged.
5. The learner mask rejects demonstrator-only actions before loss or execution.
6. Prompt and vision embeddings have explicit positions and shape checks; no
   modality data is smuggled through condition codes or numeric keys.
7. A causal-prefix test proves that targets from future steps cannot alter the
   input prefix used to predict the current learner action.

The current ABI should pass the single-actor subset and should fail the
two-actor portion with an explicit missing-actor/action-namespace diagnostic.
That failure is the desired result: it identifies the smallest interface change
before a factored source language can be judged.

## 9. Stop gates

Stop and do not lower more factored terms if any of the following is true:

- actor identity is encoded only by numeric key conventions;
- demonstrator and learner actions share an output mask without an explicit
  executable-support declaration;
- a prompt or visual embedding lacks a declared event alignment;
- a condition code carries hidden generator state;
- evaluator metadata reaches `MODEL_FIELDS`;
- targets alter the public event stream;
- the witness cannot distinguish goal denotation from its public carrier; or
- the two-actor witness cannot replay the same public transcript and target
  addressing under canonical renaming.

Passing this witness would establish interface compatibility only. It would not
establish a new card, a learner result, a transfer result, or RDDL admission.
Those remain separate gates after the boundary is coherent.
