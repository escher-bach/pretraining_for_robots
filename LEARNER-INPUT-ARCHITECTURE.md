# Learner Input Architecture

**Status:** Binding common-content architecture, implemented for the new
trajectory path and recorded 2026-09-08. This document is the authority for
how symbolic, visual, linguistic, proprioceptive, and demonstration content
may enter the learner. The historical continuous-content sidecar remains only
as replay apparatus and is not the future grounding path. The new
`common_content.py` boundary and `CalibratedReachV2` public trajectory are
implemented. First-system batching, learner integration, and CPU boundary
evidence are implemented and audited; matched acquisition remains future work.

## Non-negotiable invariant

The learner core has one canonical continuous event-content interface. Every
sensor realization, including the finite-G0 eight-float representation, reaches
the core through an adapter that produces that same interface:

```text
raw symbolic observation ----- symbolic adapter -----\
raw visual observation -------- vision adapter --------> content tokens [T,H]
raw language observation ----- language adapter -------/
raw robot state or action ----- embodiment adapter ----/
                                                        + role/key/time structure
                                                        -> one temporal core
```

The first-system implementation exercises this boundary with numeric sensor,
goal, and past-action adapters. Sensor widths and token counts may vary, while
adapter output width remains shared. This demonstrates an executable common
interface; it is not yet evidence of semantic alignment or transfer to a new
adapter.

For the currently selected core, `H = 384`; the architectural invariant is the
common hidden-width content sequence, not that particular width. The temporal
core must not have a privileged semantic route used only by abstract worlds and
a separate cold route first activated for grounding.

## What the eight-float payload is

The finite-G0 payload is semantically meaningful. Depending on the event role,
its slots encode values, public bounds, horizons, condition codes, presence,
and reserved zeros. Together with role, key, and event group, it is the public
observation language of the existing symbolic G0 families.

It is **not** the universal representation of every future modality. It is one
raw modality accepted by one input adapter:

```text
finite-G0 records [T,8] -> G0 content adapter -> [T,H]
```

The existing `Linear(8,H)` projection is therefore a symbolic/G0 adapter, not a
permanent primary content path to which other modalities are appended.

The eight coordinates also cannot silently become an opaque vision embedding:
their meanings are fixed by the current event ABI. If a learned modality emits
opaque vectors, it uses a declared adapter and content-token contract rather
than pretending those vectors are values, bounds, or condition codes.

## Other sensor realizations

Raw modality widths and sequence lengths may vary; adapter output width does
not. Examples include:

```text
ten-float sensor stream [T,10] -> sensor adapter Linear/MLP(10,H) -> [T,H]
DINO features [N,Dv]           -> projector/resampler(Dv,H)       -> [N,H]
language features [N,Dl]       -> language projector(Dl,H)        -> [N,H]
robot state [T,Dr]             -> embodiment adapter(Dr,H)        -> [T,H]
```

A visual frame need not be collapsed into one eight-float record. Patch or
object tokens may become multiple observation-content tokens. Use a maintained
vision encoder and maintained projection/resampling components. Context-length
compression is explicit and measured; it is not hidden inside the G0 ABI.

Role, public key/address, event time, and boundary structure may be added as
structural embeddings after content adaptation. They must not carry family
identity, private denotations, generator metadata, or answers unavailable from
the public observation.

## Abstract pretraining across sensor realizations

Abstractness belongs to process and world semantics, not to one privileged
learner encoding. A symbolic simulator may retain exact state for generation,
auditing, solving, and evaluation while publishing different learner-visible
sensor realizations of the same process.

Useful abstract-pretraining observation families include coordinate
permutations, episode-varying linear or nonlinear mixes, redundant widths,
partial observations, quantization, noise, tokenized/set-valued observations,
simple visual renderings, and language-rendered public goals. Preserving and
meaning-changing transformations remain part of each contract.

Two claims must remain separate:

1. **Interface identification:** a shared adapter/core must infer an unknown
   episode-level realization from calibration or interaction rather than from a
   family label.
2. **New-adapter transfer:** a new modality adapter is trained with the same
   downstream data in pretrained-core and scratch-core arms; only the core
   initialization differs.

Do not create a unique named adapter per world. Reuse modality families across
many worlds, vary their realization parameters, and hold out transformations so
the learner cannot replace interface identification with family memorization.

## Demonstrations and actions

A demonstration is an ordinary public temporal history, not a separate
semantic carrier bypassing the learner input path. Demonstrator observations,
actions, and environmental effects use the same modality adapters and canonical
content tokens as target-agent interaction. Public actor and embodiment
addresses distinguish whose observation or action an event represents; they do
not replace perceptual content.

Actions require the symmetric boundary:

```text
demonstrated embodiment action -> action encoder -> common content token
core decision representation   -> target action decoder -> executable action
```

The finite-G0 `ActionQuery` rows and fixed 16-slot head are one discrete action
realization, not the universal action interface. A target learner must never be
supervised to emit a demonstrator action that its own embodiment cannot execute.

## Rejected design: symbolic primary path plus dormant sidecar

The experimental model seam implemented this composition:

```text
role embedding + key embedding + Linear(8,H)(payload)
                              + optional external content [T,H]
```

The external content was deliberately absent and inert for the completed
symbolic runs so the R10/R10c raw batch, parameters, and behavior remained
unchanged. That made it a useful non-regression probe, but not a valid grounding
architecture.

As a future transfer path it fails for four reasons:

- abstract training teaches the core to rely on semantic directions produced
  by `Linear(8,H)`;
- real visual or linguistic data does not naturally publish the same symbolic
  state, transition-table, or condition values;
- the external content channel receives no training or alignment during the
  symbolic phase; and
- activating that cold channel while removing familiar symbolic values creates
  a representation shift with no mechanism tying the two realizations together.

An additive content input is not inherently invalid. It is rejected here as
the **primary transfer story** because it was introduced as an optional
sidecar, not trained as the common input abstraction.

## Failure record and user feedback

During the 2026-09-04/05 architecture review, a generated G0 episode exposed
that nearly all learner-visible semantics flowed through the fixed eight-float
payload while the proposed visual/language path was an unused sidecar. The
earlier explanation repeatedly described that sidecar as a sensible future
multimodal route instead of immediately identifying the cold-channel mismatch.

The user was explicitly and justifiably disappointed because this response:

- misunderstood vision and language as later attachments rather than sensor
  realizations inside the worlds used for training;
- treated local ABI/checkpoint compatibility as if it justified the global
  learner architecture;
- failed to notice that the symbolic pathway would be trained while the future
  perceptual pathway remained untouched; and
- obscured the central question of how the system could learn perception and
  transfer when its familiar semantic input disappeared.

The cause in the implementation was a reversal of priorities: preserving a
closed experiment's exact parameterization was allowed to define a new modality
interface. Closed-run compatibility is a constraint on historical replay, not
an architecture requirement for future work. The old sidecar may remain only
as isolated historical apparatus until migration; it must not silently become
the accepted design.

## Required implementation gate

No learner lineage, grounding claim, visual family, language family, or
physical-prompting claim may rely on the experimental sidecar as sufficient.
Before such work proceeds, a versioned implementation must demonstrate:

1. one canonical `[B,T,H]` content-token input to the temporal core;
2. the existing eight-float route implemented and named as a G0 adapter;
3. at least one non-symbolic or pseudo-modal adapter using the same content
   boundary during training, not first appearing at transfer;
4. explicit role/key/time structure separate from modality content;
5. no family identity or privileged simulator state in adapter outputs;
6. multiple tokens or a declared maintained resampler for high-dimensional
   modalities rather than forced one-vector compression;
7. an embodiment-specific action adapter boundary; and
8. matched pretrained-core versus scratch-core acquisition tests using the
   same downstream adapter, observations, actions, optimizer, order, and budget.

This decision changes architecture and future work selection. It does not
retroactively invalidate the semantic audits of G0 worlds, but it prevents
their current learner encoding from being mistaken for a completed grounding
or VLA interface.
