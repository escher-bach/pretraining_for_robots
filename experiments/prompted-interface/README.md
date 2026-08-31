# Prompted-interface protocol fixture

Path row: **R13 — dependent composition, prompting, and grounding**.

State change sought: make the minimum public/private sequence contract for a
future Card 08 experiment executable before introducing a world generator,
learner run, or production ABI change. This is apparatus-contract evidence only:
R11 remains blocked and R13 remains evidence-gated.

This directory is deliberately standalone. It is not a new canonical-event
profile, does not modify the learner, and has no dependency beyond Python's
standard library.

## Contract

The JSON bundle has three separate top-level values:

- `public_episode`: an RLDS-shaped episode/step envelope with public contract
  metadata and `protocol_events`.
- `supervision`: loss-only records for `ActionQuery`; they are not part of the
  learner-visible episode.
- `private_receipt`: a monitor-only `OutcomeDenotation` and generator receipt.

Each observation or action is an actor- and embodiment-scoped `PortRef`:

```text
{ actor, embodiment_id, namespace, local_id }
```

The public sequence must contain exactly ordered `PromptStart`, `PromptEnd`,
and `TargetStart` boundaries. Demonstrator `ActionExecuted` events are prompt
content only. An `ActionQuery` and its separate label may name only target-body
actuator refs.

`GoalCarrier` accepts `symbolic`, `external_language_span`,
`goal_observation_span`, and `demonstrations`. The fixture does not expose a
private outcome predicate, source assignment, seed, or other receipt data in
the public episode.

## Run

From the repository root:

```powershell
python experiments/prompted-interface/conformance.py --output $env:TEMP/prompted-interface.json
```

The runner executes the CPU-small tests and writes one JSON artifact with the
public episode, loss-only supervision, and private receipt in separate fields.

The tests cover schema acceptance, missing boundary rejection, source-action
rejection in target queries, disjoint action renames, a no-copy paired target
body, same-outcome/different-demo preservation, changed-outcome/same-source
action-surface change, and public/private noninterference.

## Deliberate limits

This fixture validates static protocol structure; it does **not** show that a
causal model's logits ignore a suffix after a query, nor that a language or
vision encoder can populate the learner's optional continuous-content input.
Those require a versioned production event/model ABI, causal-mask tests, and
maintained modality-adapter plumbing. RLDS is used here as an episode/step
shape, not as a claim that the fixture imports or replaces an RLDS runtime.
