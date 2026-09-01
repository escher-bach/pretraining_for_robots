"""A small, executable prompted-interface protocol fixture.

The fixture is intentionally independent of the production Rust event ABI and
learner.  It specifies the smallest versioned boundary needed before a
goal-conditioned, cross-embodiment adapter can be attempted: typed public
events, an explicit goal publication, loss-only target supervision, and an
adapter-alignment manifest.  All data is JSON-compatible and validation uses
only the Python standard library.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


PROTOCOL_VERSION = "prompted-interface/0.2"
RLDS_SHAPE = "rlds/episode-step/1"
ALIGNMENT_MANIFEST_VERSION = "adapter-alignment/0.1"
GOAL_CARRIER_KINDS = {
    "symbolic",
    "external_language_span",
    "goal_observation_span",
    "demonstrations",
}
BOUNDARIES = ("PromptStart", "PromptEnd", "TargetStart")
PHASES = ("schema", "prompt", "target", "terminal")
ACTORS = {"environment", "demonstrator", "target"}
PORT_ACTORS = {"demonstrator", "target"}
MODALITIES = {"symbolic", "numeric", "observation", "language", "vision", "action", "demonstration"}
NAMESPACES = {"observation", "actuator", "content", "goal", "episode"}
PHASE_ORDER = {phase: index for index, phase in enumerate(PHASES)}


class ProtocolError(ValueError):
    """Raised when a bundle would permit an invalid or leaking presentation."""


@dataclass(frozen=True)
class PortRef:
    """A port identity that cannot silently cross actor or embodiment bodies."""

    actor: str
    embodiment_id: str
    namespace: str
    local_id: str

    def to_json(self) -> dict[str, str]:
        return {
            "actor": self.actor,
            "embodiment_id": self.embodiment_id,
            "namespace": self.namespace,
            "local_id": self.local_id,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "PortRef":
        required = {"actor", "embodiment_id", "namespace", "local_id"}
        _require(set(value) == required, "port reference fields are incomplete or have trailing fields")
        result = cls(**{name: value[name] for name in required})
        _require(
            all(isinstance(part, str) and part for part in result.to_json().values()),
            "port reference fields must be non-empty strings",
        )
        _require(result.actor in PORT_ACTORS, "port reference actor must be demonstrator or target")
        _require(result.namespace in {"observation", "actuator"}, "port reference namespace must be observation or actuator")
        return result


def ref(actor: str, embodiment_id: str, namespace: str, local_id: str) -> dict[str, str]:
    return PortRef(actor, embodiment_id, namespace, local_id).to_json()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _carrier_hash(event: Mapping[str, Any]) -> str:
    """Hash the public carrier event, excluding no public content."""

    return _digest(event)


def _default_goal_text(outcome: str) -> str:
    texts = {
        "place-amber-at-bay": "place the amber item at the marked bay",
        "place-amber-at-dock": "place the amber item at the marked dock",
    }
    if outcome not in texts:
        raise ValueError(f"fixture has no default public carrier for outcome {outcome!r}")
    return texts[outcome]


def _default_demo_observation(outcome: str) -> str:
    observations = {
        "place-amber-at-bay": "amber approaches bay",
        "place-amber-at-dock": "amber approaches dock",
    }
    if outcome not in observations:
        raise ValueError(f"fixture has no default demonstration for outcome {outcome!r}")
    return observations[outcome]


def _event(
    kind: str,
    event_id: str,
    phase: str,
    *,
    actor: str = "environment",
    embodiment_id: str = "episode",
    modality: str = "symbolic",
    namespace: str = "episode",
    local_id: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Create an event with its complete, explicit address."""

    return {
        "event_id": event_id,
        "phase": phase,
        "kind": kind,
        "actor": actor,
        "embodiment_id": embodiment_id,
        "modality": modality,
        "namespace": namespace,
        "local_id": local_id or event_id,
        **fields,
    }


def _step(index: int, events: list[dict[str, Any]], *, action: Any = None, terminal: bool = False) -> dict[str, Any]:
    """Use stable RLDS episode/step keys; protocol_events hold our semantics."""

    return {
        "is_first": index == 0,
        "is_last": terminal,
        "is_terminal": terminal,
        "observation": [],
        "action": action,
        "reward": None,
        "discount": 0.0 if terminal else 1.0,
        "protocol_events": events,
    }


def _goal_carrier(kind: str, public_goal_text: str) -> dict[str, Any]:
    if kind == "symbolic":
        return {
            "carrier_id": "carrier-0",
            "kind": kind,
            "public_content": {"predicate": public_goal_text, "content_id": "symbolic-goal-0"},
        }
    if kind == "external_language_span":
        return {
            "carrier_id": "carrier-0",
            "kind": kind,
            "public_content": {
                "span_id": "language-0",
                "content_hash": _digest({"modality": "language", "content": public_goal_text}),
            },
        }
    if kind == "goal_observation_span":
        return {
            "carrier_id": "carrier-0",
            "kind": kind,
            "public_content": {
                "span_id": "goal-observation-0",
                "content_hash": _digest({"modality": "observation", "content": public_goal_text}),
            },
        }
    if kind == "demonstrations":
        return {
            "carrier_id": "carrier-0",
            "kind": kind,
            "public_content": {
                "demonstration_id": "demo-0",
                "event_ids": ["demo-language-0", "demo-observation-0", "demo-action-0"],
                "content_hash": "",  # filled after public events are built
            },
        }
    raise ValueError(f"unknown goal carrier kind: {kind}")


def _target_action_surface(embodiment_id: str) -> list[str]:
    """The complete target action surface for each fixture embodiment."""

    surfaces = {
        "target-wheels": ["target-roll-west", "target-roll-east", "target-wait"],
        "target-arm": ["target-press-west", "target-press-east", "target-wait"],
    }
    if embodiment_id not in surfaces:
        raise ValueError(f"fixture has no exact action surface for {embodiment_id!r}")
    return surfaces[embodiment_id]


def _fixture_action_outcomes(action_ids: list[str], outcome: str) -> dict[str, str]:
    """Build the private exact table used by this fixture's tiny evaluator."""

    outcomes = {
        "target-roll-west": "place-amber-at-bay",
        "target-roll-east": "place-amber-at-dock",
        "target-press-west": "place-amber-at-bay",
        "target-press-east": "place-amber-at-dock",
        "target-drive-left": "place-amber-at-bay",
        "target-drive-right": "place-amber-at-dock",
        "target-wait": "no-op",
    }
    result = {}
    for action_id in action_ids:
        if action_id not in outcomes:
            raise ValueError(f"fixture has no exact semantics for target action {action_id!r}")
        result[action_id] = outcomes[action_id]
    if outcome not in set(result.values()):
        raise ValueError(f"fixture outcome {outcome!r} is not represented by its target action surface")
    return result


def _content_hash_for_events(events: list[Mapping[str, Any]], event_ids: list[str]) -> str:
    by_id = {event["event_id"]: event for event in events}
    selected = []
    for event_id in event_ids:
        event = by_id[event_id]
        selected.append(
            {
                "event_id": event_id,
                "kind": event["kind"],
                "modality": event["modality"],
                "local_id": event["local_id"],
                "value": event.get("value"),
                "port": event.get("port"),
            }
        )
    return _digest(selected)


def _carrier_grounded_outcome(
    carrier: Mapping[str, Any], events: list[Mapping[str, Any]]
) -> str:
    """Resolve the fixture's public carrier marker without reading private data."""

    kind = carrier["kind"]
    content = carrier["public_content"]
    if kind == "symbolic":
        text = content["predicate"]
    elif kind in {"external_language_span", "goal_observation_span"}:
        span = next(event for event in events if event["event_id"] == content["span_id"])
        text = span["content"]
    else:
        referenced = {event["event_id"] for event in events if event["event_id"] in content["event_ids"]}
        observations = [
            event["value"]
            for event in events
            if event["event_id"] in referenced and event["kind"] == "Observation"
        ]
        if len(observations) != 1:
            raise ProtocolError("fixture demonstration needs exactly one grounding observation")
        text = observations[0]
    _require(isinstance(text, str), "fixture carrier grounding text must be a string")
    lowered = text.lower()
    has_bay = "bay" in lowered
    has_dock = "dock" in lowered
    _require(has_bay != has_dock, "fixture carrier must identify exactly one outcome marker")
    return "place-amber-at-bay" if has_bay else "place-amber-at-dock"


def _adapter_manifest(events: list[Mapping[str, Any]]) -> dict[str, Any]:
    entries = []
    for event in events:
        if event["kind"] != "ContentSpan":
            continue
        entries.append(
            {
                "event_id": event["event_id"],
                "modality": event["modality"],
                "alignment": "one-event-one-sidecar",
                "encoder_id": f"external/{event['modality']}",
                "encoder_version": "declared-by-adapter",
                "shape": [384],
                "sidecar_ref": f"sidecar://{event['event_id']}",
            }
        )
    return {"version": ALIGNMENT_MANIFEST_VERSION, "entries": entries}


def make_bundle(
    *,
    goal_carrier_kind: str = "demonstrations",
    source_action: str = "source-slide-left",
    target_embodiment: str = "target-wheels",
    target_action: str = "target-roll-west",
    outcome: str = "place-amber-at-bay",
    public_goal_text: str | None = None,
    private_source_assignment: str = "source-a",
    demo_note: str = "source moved around the obstacle",
    demo_observation: str | None = None,
) -> dict[str, Any]:
    """Return a deterministic public/private paired episode.

    ``outcome`` is a private goal denotation.  Public action labels are derived
    by :func:`derive_target_supervision`, using the exact finite effect table
    and this private denotation; callers cannot independently inject a label.
    """

    if public_goal_text is None:
        public_goal_text = _default_goal_text(outcome)
    if demo_observation is None:
        demo_observation = _default_demo_observation(outcome)

    source_body = "source-slider"
    source_actuator = ref("demonstrator", source_body, "actuator", source_action)
    source_observation = ref("demonstrator", source_body, "observation", "source-camera")
    target_observation = ref("target", target_embodiment, "observation", "target-camera")
    target_action_ids = _target_action_surface(target_embodiment)
    if target_action not in target_action_ids:
        raise ValueError(f"target action {target_action!r} is not in the {target_embodiment} action surface")
    target_candidates = [
        ref("target", target_embodiment, "actuator", action_id)
        for action_id in target_action_ids
    ]

    carrier = _goal_carrier(goal_carrier_kind, public_goal_text)
    prompt_events = [
        _event(
            "GoalCarrier",
            "goal-carrier-0",
            "prompt",
            embodiment_id=target_embodiment,
            modality={
                "symbolic": "symbolic",
                "external_language_span": "language",
                "goal_observation_span": "observation",
                "demonstrations": "demonstration",
            }[goal_carrier_kind],
            namespace="goal",
            local_id="carrier-0",
            value=carrier,
        )
    ]
    if goal_carrier_kind == "external_language_span":
        prompt_events.append(
            _event(
                "ContentSpan",
                "language-0",
                "prompt",
                embodiment_id=target_embodiment,
                modality="language",
                namespace="content",
                local_id="language-0",
                span_id="language-0",
                content=public_goal_text,
            )
        )
    elif goal_carrier_kind == "goal_observation_span":
        prompt_events.append(
            _event(
                "ContentSpan",
                "goal-observation-0",
                "prompt",
                embodiment_id=target_embodiment,
                modality="observation",
                namespace="content",
                local_id="goal-observation-0",
                span_id="goal-observation-0",
                content=public_goal_text,
            )
        )
    elif goal_carrier_kind == "demonstrations":
        prompt_events.extend(
            [
                _event(
                    "ContentSpan",
                    "demo-language-0",
                    "prompt",
                    embodiment_id=source_body,
                    modality="language",
                    namespace="content",
                    local_id="demo-language-0",
                    span_id="demo-language-0",
                    demonstration_id="demo-0",
                    content=demo_note,
                ),
                _event(
                    "Observation",
                    "demo-observation-0",
                    "prompt",
                    actor="demonstrator",
                    embodiment_id=source_body,
                    modality="observation",
                    namespace="observation",
                    local_id="source-camera",
                    demonstration_id="demo-0",
                    port=source_observation,
                    value=demo_observation,
                ),
                _event(
                    "ActionExecuted",
                    "demo-action-0",
                    "prompt",
                    actor="demonstrator",
                    embodiment_id=source_body,
                    modality="action",
                    namespace="actuator",
                    local_id=source_action,
                    demonstration_id="demo-0",
                    port=source_actuator,
                    provenance="demonstration",
                ),
            ]
        )

    schema_event = _event(
        "Schema",
        "schema-0",
        "schema",
        embodiment_id=target_embodiment,
        modality="symbolic",
        namespace="episode",
        local_id="target-interface",
    )
    boundaries = [
        _event("Boundary", "boundary-prompt-start", "prompt", local_id="PromptStart", boundary="PromptStart"),
        _event("Boundary", "boundary-prompt-end", "prompt", local_id="PromptEnd", boundary="PromptEnd"),
        _event("Boundary", "boundary-target-start", "target", local_id="TargetStart", boundary="TargetStart"),
    ]
    target_events = [
        _event(
            "Observation",
            "target-observation-0",
            "target",
            actor="target",
            embodiment_id=target_embodiment,
            modality="observation",
            namespace="observation",
            local_id="target-camera",
            port=target_observation,
            value="amber is visible",
        ),
        _event(
            "ActionQuery",
            "target-query-0",
            "target",
            actor="target",
            embodiment_id=target_embodiment,
            modality="action",
            namespace="actuator",
            local_id="target-query-0",
            query_id="target-query-0",
            candidates=target_candidates,
        ),
    ]
    episode_end = _event("EpisodeEnd", "episode-end-0", "terminal", local_id="episode-end")

    all_events = [schema_event, *boundaries[:1], *prompt_events, boundaries[1], boundaries[2], *target_events, episode_end]
    if goal_carrier_kind == "demonstrations":
        carrier["public_content"]["content_hash"] = _content_hash_for_events(
            all_events, carrier["public_content"]["event_ids"]
        )
        prompt_events[0]["value"] = carrier

    steps = [
        _step(0, [schema_event, boundaries[0]]),
        _step(1, prompt_events),
        _step(2, [boundaries[1]]),
        _step(3, [boundaries[2], target_events[0]]),
        _step(4, [target_events[1]]),
        _step(5, [episode_end], terminal=True),
    ]

    bundle = {
        "public_episode": {
            "rlds_shape": RLDS_SHAPE,
            "protocol_version": PROTOCOL_VERSION,
            "public_contract": {
                "source_embodiment": source_body,
                "target_embodiment": target_embodiment,
                "goal_publication": {
                    "publication_id": "publication-0",
                    "visibility": "public",
                    "denotation_visibility": "private",
                    "carrier_event_id": "goal-carrier-0",
                },
                "action_adapters": {
                    "target": {
                        "version": "action-adapter/0.2",
                        "actor": "target",
                        "embodiment_id": target_embodiment,
                        "namespace": "actuator",
                        "candidates": list(target_action_ids),
                        "executable": list(target_action_ids),
                    }
                },
                "adapter_alignment_manifest": _adapter_manifest(all_events),
            },
            "steps": steps,
        },
        "supervision": {"action_queries": []},
        "private_receipt": {
            "goal_denotation": {
                "denotation_id": "denotation-0",
                "kind": "target-outcome",
                "outcome": outcome,
            },
            "outcome_denotation": {
                "kind": "private-monitor-predicate",
                "predicate": outcome,
            },
            "evaluator": {
                "version": "fixture-evaluator/0.2",
                "action_outcomes": {
                    "target-query-0": _fixture_action_outcomes(target_action_ids, outcome),
                },
            },
            "goal_grounding": {
                "publication_id": "publication-0",
                "carrier_event_id": "goal-carrier-0",
                "carrier_hash": _carrier_hash(prompt_events[0]),
                "denotation_id": "denotation-0",
                "grounded_outcome": outcome,
                "method": "fixture-exact-carrier-marker",
                "version": "goal-grounding/0.2",
            },
            "generator": {"source_assignment": private_source_assignment, "seed": "fixture-seed"},
        },
    }
    bundle["supervision"] = derive_target_supervision(bundle)
    validate(bundle)
    return bundle


def _events(bundle: Mapping[str, Any]) -> Iterable[tuple[int, Mapping[str, Any]]]:
    for index, step in enumerate(bundle["public_episode"]["steps"]):
        for event in step["protocol_events"]:
            yield index, event


def _all_events(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [event for _, event in _events(bundle)]


def _validate_event_address(event: Mapping[str, Any]) -> None:
    required = {"event_id", "phase", "kind", "actor", "embodiment_id", "modality", "namespace", "local_id"}
    _require(set(event) >= required, f"event {event.get('event_id', '<missing>')} is missing an address field")
    _require(isinstance(event["event_id"], str) and event["event_id"], "event ids must be non-empty strings")
    _require(event["phase"] in PHASES, "unsupported event phase")
    _require(event["actor"] in ACTORS, "unsupported event actor")
    _require(isinstance(event["embodiment_id"], str) and event["embodiment_id"], "event embodiment must be non-empty")
    _require(event["modality"] in MODALITIES, "unsupported event modality")
    _require(event["namespace"] in NAMESPACES, "unsupported event namespace")
    _require(isinstance(event["local_id"], str) and event["local_id"], "event local_id must be non-empty")


def _validate_port_event(event: Mapping[str, Any], *, kind: str) -> PortRef:
    port = PortRef.from_json(event.get("port", {}))
    _require(event["actor"] == port.actor, f"{kind} actor does not match its port")
    _require(event["embodiment_id"] == port.embodiment_id, f"{kind} embodiment does not match its port")
    _require(event["namespace"] == port.namespace, f"{kind} namespace does not match its port")
    _require(event["local_id"] == port.local_id, f"{kind} local_id does not match its port")
    expected_modality = "action" if kind in {"ActionExecuted", "ActionQuery"} else "observation"
    _require(event["modality"] == expected_modality, f"{kind} modality does not match its namespace")
    return port


def _validate_alignment_manifest(episode: Mapping[str, Any], events: list[Mapping[str, Any]]) -> None:
    manifest = episode["public_contract"].get("adapter_alignment_manifest")
    _require(isinstance(manifest, Mapping), "public contract needs an adapter-alignment manifest")
    _require(set(manifest) == {"version", "entries"}, "alignment manifest has trailing or missing fields")
    _require(manifest["version"] == ALIGNMENT_MANIFEST_VERSION, "unsupported alignment manifest version")
    spans = {event["event_id"]: event for event in events if event["kind"] == "ContentSpan"}
    entries = manifest["entries"]
    _require(isinstance(entries, list), "alignment manifest entries must be a list")
    seen: set[str] = set()
    for entry in entries:
        required = {"event_id", "modality", "alignment", "encoder_id", "encoder_version", "shape", "sidecar_ref"}
        _require(set(entry) == required, "alignment entry has trailing or missing fields")
        event_id = entry["event_id"]
        _require(event_id in spans, "alignment references a missing or non-content event")
        _require(event_id not in seen, "alignment event ids must be unique")
        seen.add(event_id)
        _require(entry["modality"] == spans[event_id]["modality"], "alignment modality mismatch")
        _require(entry["alignment"] == "one-event-one-sidecar", "unsupported adapter alignment")
        _require(isinstance(entry["shape"], list) and entry["shape"] == [384], "adapter shape must be [384]")
        serialized = json.dumps(entry, sort_keys=True)
        _require(not any(secret in serialized for secret in ("seed", "private", "denotation", "source_assignment")), "private metadata in alignment manifest")
    _require(seen == set(spans), "alignment manifest has missing or trailing span references")


def _exact_target_actions(bundle: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Tiny exact evaluator: private query outcomes meet the private goal."""

    denotation = bundle["private_receipt"].get("goal_denotation")
    _require(isinstance(denotation, Mapping), "private goal denotation is required")
    evaluator = bundle["private_receipt"].get("evaluator")
    _require(isinstance(evaluator, Mapping), "private exact evaluator is required")
    action_outcomes = evaluator.get("action_outcomes")
    _require(isinstance(action_outcomes, Mapping), "private action outcome table is required")
    expected_outcome = denotation.get("outcome")
    _require(isinstance(expected_outcome, str) and expected_outcome, "goal denotation needs an outcome token")
    result: dict[str, list[dict[str, str]]] = {}
    for _, event in _events(bundle):
        if event["kind"] != "ActionQuery":
            continue
        outcomes = action_outcomes.get(event["event_id"])
        _require(isinstance(outcomes, Mapping), "private action outcomes need every query event")
        labels = []
        for candidate in event["candidates"]:
            port = PortRef.from_json(candidate)
            if outcomes.get(port.local_id) == expected_outcome:
                labels.append(port.to_json())
        result[event["event_id"]] = labels
    return result


def derive_target_supervision(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Derive target labels from public candidates plus the exact private monitor."""

    target_body = bundle["public_episode"]["public_contract"]["target_embodiment"]
    labels = _exact_target_actions(bundle)
    return {
        "action_queries": [
            {
                "query_event_id": event_id,
                "target_actor": "target",
                "target_embodiment": target_body,
                "correct_target_actions": actions,
            }
            for event_id, actions in labels.items()
        ]
    }


def validate(bundle: Mapping[str, Any]) -> None:
    """Validate protocol structure and the public/private information boundary."""

    _require(set(bundle) == {"public_episode", "supervision", "private_receipt"}, "bundle fields are fixed")
    episode = bundle["public_episode"]
    _require(episode.get("rlds_shape") == RLDS_SHAPE, "bundle must use the declared RLDS-shaped envelope")
    _require(episode.get("protocol_version") == PROTOCOL_VERSION, "unsupported protocol version")
    _require(isinstance(episode.get("steps"), list) and episode["steps"], "episode needs steps")
    _require(episode["steps"][0].get("is_first") is True, "first step must be marked is_first")
    _require(episode["steps"][-1].get("is_last") is True, "last step must be marked is_last")

    boundaries: dict[str, int] = {}
    goal_carriers: list[tuple[int, Mapping[str, Any]]] = []
    spans: dict[str, Mapping[str, Any]] = {}
    demonstrations: dict[str, list[Mapping[str, Any]]] = {}
    queries: dict[str, tuple[int, Mapping[str, Any]]] = {}
    events = _all_events(bundle)
    event_ids: set[str] = set()
    last_phase = PHASE_ORDER["schema"]
    for step_index, event in _events(bundle):
        _validate_event_address(event)
        _require(event["event_id"] not in event_ids, "event ids must be globally unique")
        event_ids.add(event["event_id"])
        phase = PHASE_ORDER[event["phase"]]
        _require(phase >= last_phase, "event phases must be monotonic")
        last_phase = phase
        kind = event.get("kind")
        if kind == "Boundary":
            _require(event["namespace"] == "episode" and event["local_id"] in BOUNDARIES, "unsupported boundary")
            _require(event["phase"] == ("target" if event["local_id"] == "TargetStart" else "prompt"), "boundary phase mismatch")
            _require(event["local_id"] not in boundaries, "each prompting boundary occurs once")
            boundaries[event["local_id"]] = step_index
        elif kind == "GoalCarrier":
            _require(event["namespace"] == "goal" and event["actor"] == "environment", "GoalCarrier must be an environment goal event")
            _require(event["phase"] == "prompt", "GoalCarrier must be in prompt phase")
            value = event.get("value")
            _require(isinstance(value, Mapping) and set(value) == {"carrier_id", "kind", "public_content"}, "invalid GoalCarrier shape")
            _require(value.get("kind") in GOAL_CARRIER_KINDS, "unknown GoalCarrier")
            goal_carriers.append((step_index, event))
        elif kind == "ContentSpan":
            _require(event["namespace"] == "content", "ContentSpan must use content namespace")
            _require(event["phase"] == "prompt", "ContentSpan must be in prompt phase")
            span_id = event.get("span_id")
            _require(span_id == event["local_id"] and span_id not in spans, "content spans need unique local ids")
            _require(event["modality"] in {"language", "observation", "vision"}, "unsupported content modality")
            spans[span_id] = event
            if "demonstration_id" in event:
                demonstrations.setdefault(event["demonstration_id"], []).append(event)
        elif kind in {"Observation", "ActionExecuted"}:
            port = _validate_port_event(event, kind=kind)
            _require(event["phase"] == ("prompt" if port.actor == "demonstrator" else "target"), "actor event is in the wrong phase")
            if kind == "ActionExecuted" and port.actor == "demonstrator":
                _require(event.get("provenance") == "demonstration", "demonstrator action must be prompt content")
                _require(event["phase"] == "prompt", "demonstrator action must be in prompt phase")
            if "demonstration_id" in event:
                demonstrations.setdefault(event["demonstration_id"], []).append(event)
        elif kind == "ActionQuery":
            _require(event["actor"] == "target" and event["namespace"] == "actuator" and event["modality"] == "action", "only target ActionQuery is supported")
            query_id = event.get("query_id")
            _require(query_id == event["event_id"] and query_id not in queries, "query ids must equal unique event ids")
            _require(isinstance(event.get("candidates"), list) and len(event["candidates"]) >= 2, "target query needs executable alternatives")
            _require(event["phase"] == "target", "target query must be in target phase")
            queries[query_id] = (step_index, event)
        elif kind in {"Schema", "EpisodeEnd"}:
            _require(event["phase"] == ("schema" if kind == "Schema" else "terminal"), "schema/terminal event phase mismatch")
        else:
            raise ProtocolError(f"unsupported public event kind {kind!r}")

    _require(set(boundaries) == set(BOUNDARIES), "PromptStart, PromptEnd, and TargetStart are required")
    _require(boundaries["PromptStart"] < boundaries["PromptEnd"] < boundaries["TargetStart"], "prompt phases are out of order")
    _require(len(goal_carriers) == 1, "exactly one GoalCarrier is required")
    _require(boundaries["PromptStart"] <= goal_carriers[0][0] < boundaries["PromptEnd"], "GoalCarrier belongs in prompt")
    carrier_event = goal_carriers[0][1]
    carrier = carrier_event["value"]
    _require(carrier["carrier_id"] == carrier_event["local_id"], "GoalCarrier id must match its event address")
    content = carrier["public_content"]
    if carrier["kind"] == "symbolic":
        _require(set(content) == {"predicate", "content_id"}, "symbolic carrier has trailing or missing fields")
    elif carrier["kind"] in {"external_language_span", "goal_observation_span"}:
        _require(set(content) == {"span_id", "content_hash"}, "span carrier has trailing or missing references")
        span = spans.get(content["span_id"])
        _require(span is not None, "GoalCarrier references a missing public span")
        expected_modality = "language" if carrier["kind"] == "external_language_span" else "observation"
        _require(span["modality"] == expected_modality, "GoalCarrier span modality mismatch")
        _require(content["content_hash"] == _digest({"modality": span["modality"], "content": span["content"]}), "GoalCarrier span content identity mismatch")
    else:
        _require(set(content) == {"demonstration_id", "event_ids", "content_hash"}, "demonstration carrier has trailing or missing references")
        event_ids_ref = content["event_ids"]
        _require(isinstance(event_ids_ref, list) and event_ids_ref, "demonstration carrier needs event references")
        _require(len(set(event_ids_ref)) == len(event_ids_ref), "demonstration references must be unique")
        selected = [event for event in events if event["event_id"] in event_ids_ref]
        _require(len(selected) == len(event_ids_ref), "demonstration carrier references a missing event")
        _require(all(event.get("demonstration_id") == content["demonstration_id"] for event in selected), "demonstration reference crosses demonstrations or has a trailing foreign reference")
        _require(any(event["kind"] == "ActionExecuted" for event in selected), "demonstration needs an action")
        _require(any(event["kind"] == "Observation" for event in selected), "demonstration needs an observation")
        _require(content["content_hash"] == _content_hash_for_events(events, event_ids_ref), "demonstration content identity mismatch")
        demonstrations.setdefault(content["demonstration_id"], [])
        _require(set(event_ids_ref) == {event["event_id"] for event in demonstrations[content["demonstration_id"]]}, "demonstration references must be complete")

    contract = episode.get("public_contract")
    _require(isinstance(contract, Mapping), "public contract is required")
    _require(isinstance(contract.get("source_embodiment"), str) and contract["source_embodiment"], "source embodiment is required")
    target_body = contract.get("target_embodiment")
    _require(isinstance(target_body, str) and target_body, "target embodiment is required")
    publication = contract.get("goal_publication")
    _require(isinstance(publication, Mapping), "explicit GoalPublication is required")
    _require(set(publication) == {"publication_id", "visibility", "denotation_visibility", "carrier_event_id"}, "GoalPublication has trailing or missing fields")
    _require(publication["visibility"] == "public" and publication["denotation_visibility"] == "private", "goal publication visibility is invalid")
    _require(publication["carrier_event_id"] == carrier_event["event_id"], "GoalPublication references the wrong carrier")

    adapters = contract.get("action_adapters")
    _require(isinstance(adapters, Mapping) and set(adapters) == {"target"}, "target action adapter is required")
    target_adapter = adapters["target"]
    _require(set(target_adapter) == {"version", "actor", "embodiment_id", "namespace", "candidates", "executable"}, "action adapter has trailing or missing fields")
    _require(target_adapter["version"] == "action-adapter/0.2", "unsupported action adapter version")
    _require(target_adapter["actor"] == "target" and target_adapter["embodiment_id"] == target_body and target_adapter["namespace"] == "actuator", "target action adapter scope mismatch")
    _require(isinstance(target_adapter["candidates"], list) and all(isinstance(item, str) and item for item in target_adapter["candidates"]), "action adapter candidates must be local IDs")
    _require(len(set(target_adapter["candidates"])) == len(target_adapter["candidates"]), "action adapter candidates must be unique")
    _require(isinstance(target_adapter["executable"], list) and all(isinstance(item, str) for item in target_adapter["executable"]), "action adapter executable IDs must be strings")
    _require(set(target_adapter["candidates"]) == set(target_adapter["executable"]), "executable candidates must be declared separately and completely")
    _validate_alignment_manifest(episode, events)

    label_map = {item.get("query_event_id"): item for item in bundle["supervision"].get("action_queries", [])}
    _require(set(label_map) == set(queries), "supervision must label each target query exactly once by event id")
    expected_labels = _exact_target_actions(bundle)
    for query_id, (query_step, query) in queries.items():
        _require(query_step >= boundaries["TargetStart"], "target query cannot occur before target phase")
        candidates = [PortRef.from_json(value) for value in query["candidates"]]
        _require(all(item.actor == "target" and item.embodiment_id == target_body and item.namespace == "actuator" for item in candidates), "target query candidate crosses actor or embodiment; source action is not executable by target")
        _require({item.local_id for item in candidates} == set(target_adapter["executable"]), "target query candidates must equal the executable target surface")
        label = label_map[query_id]
        _require(set(label) == {"query_event_id", "target_actor", "target_embodiment", "correct_target_actions"}, "target supervision has trailing or missing fields")
        _require(label["target_actor"] == "target" and label["target_embodiment"] == target_body, "supervision crosses target actor or embodiment")
        labels = [PortRef.from_json(value) for value in label["correct_target_actions"]]
        _require(labels and all(item in candidates for item in labels), "supervision action must be in query candidates")
        _require(label["correct_target_actions"] == expected_labels[query_id], "caller-supplied supervision disagrees with exact evaluator")

    private = bundle["private_receipt"]
    _require(isinstance(private, Mapping) and "goal_denotation" in private and "outcome_denotation" in private and "evaluator" in private and "goal_grounding" in private, "private goal denotation is required")
    _require(private["goal_denotation"].get("outcome") == private["outcome_denotation"].get("predicate"), "private denotation receipts disagree")
    evaluator = private["evaluator"]
    _require(isinstance(evaluator, Mapping) and set(evaluator) == {"version", "action_outcomes"}, "private evaluator has trailing or missing fields")
    _require(evaluator["version"] == "fixture-evaluator/0.2", "unsupported private evaluator version")
    action_outcomes = evaluator["action_outcomes"]
    _require(isinstance(action_outcomes, Mapping), "private action outcome table must be a mapping")
    _require(set(action_outcomes) == set(queries), "private action outcomes must cover every target query exactly")
    for query_id, (_, query) in queries.items():
        outcomes = action_outcomes[query_id]
        _require(isinstance(outcomes, Mapping), "query action outcomes must be mappings")
        _require(set(outcomes) == {PortRef.from_json(item).local_id for item in query["candidates"]}, "private outcomes must cover the complete query action surface")
        _require(all(isinstance(key, str) and isinstance(value, str) and value for key, value in outcomes.items()), "private action outcome entries must be strings")
    grounding = private["goal_grounding"]
    _require(
        isinstance(grounding, Mapping)
        and set(grounding)
        == {
            "publication_id",
            "carrier_event_id",
            "carrier_hash",
            "denotation_id",
            "grounded_outcome",
            "method",
            "version",
        },
        "goal grounding receipt has trailing or missing fields",
    )
    _require(grounding["version"] == "goal-grounding/0.2", "unsupported goal grounding version")
    _require(grounding["method"] == "fixture-exact-carrier-marker", "unsupported goal grounding method")
    _require(grounding["publication_id"] == publication["publication_id"], "goal grounding publication mismatch")
    _require(grounding["carrier_event_id"] == carrier_event["event_id"], "goal grounding carrier mismatch")
    _require(grounding["carrier_hash"] == _carrier_hash(carrier_event), "goal grounding carrier hash mismatch")
    _require(grounding["denotation_id"] == private["goal_denotation"].get("denotation_id"), "goal grounding denotation mismatch")
    _require(grounding["grounded_outcome"] == private["goal_denotation"].get("outcome"), "goal grounding outcome mismatch")
    _require(
        grounding["grounded_outcome"] == _carrier_grounded_outcome(carrier, events),
        "goal grounding disagrees with the public carrier",
    )
    public_json = json.dumps(episode, sort_keys=True)
    _require(not any(token in public_json for token in ("goal_denotation", "outcome_denotation", "source_assignment", "fixture-seed")), "private denotation or generator metadata leaked into public episode")


def public_projection(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """The only learner-visible portion of a rendered bundle."""

    return copy.deepcopy(bundle["public_episode"])


def target_supervision_projection(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Loss-only labels, intentionally separate from the learner-visible episode."""

    return copy.deepcopy(bundle["supervision"])


def learner_input_projection(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Public model inputs, including alignment declarations but no sidecars."""

    return public_projection(bundle)


def rename_actions(bundle: Mapping[str, Any], mapping: Mapping[str, str], *, actor: str) -> dict[str, Any]:
    """Rename one actor's local actuator IDs while preserving exact effects."""

    result = copy.deepcopy(bundle)
    for _, event in _events(result):
        if event.get("kind") == "ActionQuery":
            ports = event["candidates"]
        elif event.get("kind") == "ActionExecuted":
            ports = [event["port"]]
        else:
            continue
        for value in ports:
            port = PortRef.from_json(value)
            if port.actor == actor and port.namespace == "actuator" and port.local_id in mapping:
                value["local_id"] = mapping[port.local_id]
                if event["kind"] == "ActionExecuted":
                    event["local_id"] = mapping[port.local_id]
    adapter = result["public_episode"]["public_contract"]["action_adapters"].get(actor)
    if adapter is not None:
        adapter["candidates"] = [mapping.get(local_id, local_id) for local_id in adapter["candidates"]]
        adapter["executable"] = [mapping.get(local_id, local_id) for local_id in adapter["executable"]]
    evaluator = result["private_receipt"].get("evaluator")
    if actor == "target" and isinstance(evaluator, Mapping):
        for query_id, outcomes in evaluator["action_outcomes"].items():
            renamed = {}
            for local_id, value in outcomes.items():
                renamed[mapping.get(local_id, local_id)] = value
            evaluator["action_outcomes"][query_id] = renamed
    for _, event in _events(result):
        if event.get("kind") == "GoalCarrier" and event["value"]["kind"] == "demonstrations":
            content = event["value"]["public_content"]
            content["content_hash"] = _content_hash_for_events(_all_events(result), content["event_ids"])
    grounding = result["private_receipt"].get("goal_grounding")
    carrier_event = next((event for _, event in _events(result) if event.get("kind") == "GoalCarrier"), None)
    if isinstance(grounding, dict) and carrier_event is not None:
        grounding["carrier_hash"] = _carrier_hash(carrier_event)
    result["supervision"] = derive_target_supervision(result)
    validate(result)
    return result


def assert_public_private_noninterference(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    """Private receipts cannot change public inputs or labels for one history."""

    validate(left)
    validate(right)
    _require(public_projection(left) == public_projection(right), "noninterference comparison requires fixed public episode")
    _require(learner_input_projection(left) == learner_input_projection(right), "private receipt changed model inputs")
    _require(target_supervision_projection(left) == target_supervision_projection(right), "private receipt changed target supervision for identical public history")


def assert_fixed_carrier_private_denotation_invariance(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    """A fixed public carrier has identical learner inputs despite private denotation changes.

    ``right`` may intentionally have different evaluator labels: this helper
    checks the model-input boundary, while ``assert_public_private_noninterference``
    checks the stronger fixed-history label invariant.
    """

    validate(left)
    validate(right)
    left_carrier = next(event for _, event in _events(left) if event["kind"] == "GoalCarrier")
    right_carrier = next(event for _, event in _events(right) if event["kind"] == "GoalCarrier")
    _require(left_carrier == right_carrier, "fixed-carrier comparison requires identical public carrier")
    _require(learner_input_projection(left) == learner_input_projection(right), "private denotation changed learner inputs")
