# Prompted-interface protocol fixture

Path row: **R13 — dependent composition, prompting, and grounding**.

State change sought: make the minimum public/private sequence contract for a
future Card 08 experiment executable before introducing a world generator,
learner run, or production ABI change. This is apparatus-contract evidence
only; R11 remains blocked and R13 remains evidence-gated.

## Contract

Version `prompted-interface/0.2` is deliberately standalone and uses only
Python's standard library. It does not modify or replace the production Rust
event ABI.

The JSON bundle has three separate top-level values:

- `public_episode`: an RLDS-shaped episode/step envelope containing only public
  events and a public adapter contract;
- `supervision`: loss-only target records addressed by target `query_event_id`,
  actor, embodiment, and action refs; and
- `private_receipt`: the world-level `GoalDenotation`, exact evaluator,
  goal-grounding receipt, and generator receipt.

Every public event has an explicit address:

```text
event_id, phase, actor, embodiment_id, modality, namespace, local_id
```

Port refs additionally carry `{ actor, embodiment_id, namespace, local_id }`.
Demonstrator events are prompt history. Target `ActionQuery` candidates come
from the target action adapter and must be executable by the target body; a
demonstrator action can never silently become a learner output. The public
adapter declares only versioned scoped local IDs. Query-scoped action outcomes
used for exact labels live exclusively in the private evaluator receipt.

Goals are kept as three separate concepts:

```text
private GoalDenotation -> explicit GoalPublication -> public GoalCarrier
```

The carrier may be symbolic, a language span, a goal-observation span, or a
demonstration. Span and demonstration refs are typed and content-hashed; the
validator rejects missing, trailing, cross-modality, or content-mismatched
references. A public `adapter_alignment_manifest` declares event alignment,
encoder identity/version, expected shape, and sidecar refs without carrying
raw modality data or private metadata.

Target labels are not caller-supplied. `derive_target_supervision()` runs the
tiny exact evaluator over the private query-scoped action-outcome table and
private goal denotation, then `validate()` recomputes and checks the result.
This keeps the evaluator label path explicit while ensuring labels cannot alter
public events or adapter inputs. Each fixture target body exposes a complete
bay/dock/wait action surface, so a private-denotation change can alter labels
without changing the public episode.

The fixture also emits a private `goal_grounding` receipt tying the publication
ID, carrier event/hash, denotation ID, and grounded outcome together. The
validator recomputes the public-carrier marker and rejects stale or tampered
grounding. This is a fixture-exact receipt contract; a future RDDL adapter must
produce the same evidence from exhaustive simulation rather than use this
marker heuristic.

## Run

From the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path experiments/prompted-interface).Path
python -m unittest discover -s experiments/prompted-interface -p 'test_protocol.py' -v
python experiments/prompted-interface/conformance.py --output $env:TEMP/prompted-interface-v02.json
```

The conformance runner executes the CPU-small tests and writes one JSON
artifact with public events, loss-only supervision, and the private receipt.

## Boundary tests

The fixture covers all four carrier kinds, explicit event addressing,
actor/body separation, missing and trailing carrier refs, span modality and
content identity, source-action exclusion, actor-local action renaming,
no-copy target bodies, same-outcome/different-demo preservation,
changed-outcome target response, exact evaluator label derivation,
public-carrier changes, fixed-carrier/private-denotation input invariance,
private-receipt noninterference, tampered carrier/grounding rejection, and
adapter-manifest privacy.

## Deliberate limits

This fixture validates static protocol structure and evaluator boundaries. It
does not prove that transformer logits ignore a suffix after a query, nor that
a language or vision encoder can populate continuous event embeddings. Those
remain future production-profile work requiring a versioned event/model ABI,
causal-mask tests, and maintained modality-adapter plumbing. RLDS is used only
as an episode/step shape, not as a claim that the fixture imports or replaces
an RLDS runtime.
