"""A small, executable prompted-interface protocol fixture.

This module is intentionally independent of the production Rust event ABI and
learner.  It models the contract needed before a Card 08-style experiment can
be admitted: a public, RLDS-shaped episode; loss-only target supervision; and a
private outcome receipt.  All data is plain JSON-compatible Python values so
the fixture has no runtime dependency outside the standard library.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


PROTOCOL_VERSION = "prompted-interface/0.1"
RLDS_SHAPE = "rlds/episode-step/1"
GOAL_CARRIER_KINDS = {
    "symbolic",
    "external_language_span",
    "goal_observation_span",
    "demonstrations",
}
BOUNDARIES = ("PromptStart", "PromptEnd", "TargetStart")


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
        if set(value) != required:
            raise ProtocolError("port reference must contain exactly actor, embodiment_id, namespace, local_id")
        result = cls(**{name: value[name] for name in required})
        if not all(isinstance(part, str) and part for part in result.to_json().values()):
            raise ProtocolError("port reference fields must be non-empty strings")
        if result.actor not in {"demonstrator", "target"}:
            raise ProtocolError("port reference actor must be demonstrator or target")
        if result.namespace not in {"observation", "actuator"}:
            raise ProtocolError("port reference namespace must be observation or actuator")
        return result


def ref(actor: str, embodiment_id: str, namespace: str, local_id: str) -> dict[str, str]:
    return PortRef(actor, embodiment_id, namespace, local_id).to_json()


def _event(kind: str, **fields: Any) -> dict[str, Any]:
    return {"kind": kind, **fields}


def _step(index: int, events: list[dict[str, Any]], *, action: Any = None) -> dict[str, Any]:
    """Use the stable RLDS episode/step keys; protocol_events hold our semantics."""

    return {
        "is_first": index == 0,
        "is_last": False,
        "is_terminal": False,
        "observation": [],
        "action": action,
        "reward": None,
        "discount": 1.0,
        "protocol_events": events,
    }


def _goal_carrier(kind: str, public_goal_text: str) -> dict[str, Any]:
    if kind == "symbolic":
        return {"kind": kind, "public_content": {"predicate": public_goal_text}}
    if kind == "external_language_span":
        return {
            "kind": kind,
            "public_content": {"span_id": "language-0", "text": public_goal_text},
        }
    if kind == "goal_observation_span":
        return {
            "kind": kind,
            "public_content": {"span_id": "goal-observation-0", "observation": public_goal_text},
        }
    if kind == "demonstrations":
        return {"kind": kind, "public_content": {"demonstration_ids": ["demo-0"]}}
    raise ValueError(f"unknown goal carrier kind: {kind}")


def make_bundle(
    *,
    goal_carrier_kind: str = "demonstrations",
    source_action: str = "source-slide-left",
    target_embodiment: str = "target-wheels",
    target_action: str = "target-roll-west",
    outcome: str = "place-amber-at-bay",
    public_goal_text: str = "place the amber item at the marked bay",
    private_source_assignment: str = "source-a",
    demo_note: str = "source moved around the obstacle",
    demo_observation: str = "amber approaches bay",
) -> dict[str, Any]:
    """Return a deterministic, valid paired demonstration/target episode.

    ``outcome`` lives only in the private receipt.  The public carrier gives the
    learner evidence, while the separate supervision record supplies the
    target-body action set.  Fixture tests can vary those two objects to check
    that a renderer cannot secretly use a private receipt as a teacher.
    """

    source_body = "source-slider"
    source_actuator = ref("demonstrator", source_body, "actuator", source_action)
    source_observation = ref("demonstrator", source_body, "observation", "source-camera")
    target_observation = ref("target", target_embodiment, "observation", "target-camera")
    target_actuator = ref("target", target_embodiment, "actuator", target_action)
    target_alternative = ref("target", target_embodiment, "actuator", "target-wait")

    prompt_events = [_event("GoalCarrier", value=_goal_carrier(goal_carrier_kind, public_goal_text))]
    if goal_carrier_kind == "external_language_span":
        prompt_events.append(
            _event("ContentSpan", span_id="language-0", modality="language", content=public_goal_text)
        )
    elif goal_carrier_kind == "goal_observation_span":
        prompt_events.append(
            _event(
                "ContentSpan",
                span_id="goal-observation-0",
                modality="observation",
                content=public_goal_text,
            )
        )
    elif goal_carrier_kind == "demonstrations":
        prompt_events.extend(
            [
                _event("ContentSpan", span_id="language-0", modality="language", content=demo_note),
                _event("Observation", port=source_observation, value=demo_observation),
                _event("ActionExecuted", port=source_actuator, provenance="demonstration"),
            ]
        )

    steps = [
        _step(0, [_event("Boundary", boundary="PromptStart")]),
        _step(1, prompt_events),
        _step(2, [_event("Boundary", boundary="PromptEnd")]),
        _step(
            3,
            [
                _event("Boundary", boundary="TargetStart"),
                _event("Observation", port=target_observation, value="amber is visible"),
            ],
        ),
        _step(
            4,
            [
                _event(
                    "ActionQuery",
                    query_id="target-q0",
                    actor="target",
                    embodiment_id=target_embodiment,
                    candidates=[target_actuator, target_alternative],
                )
            ],
        ),
    ]
    steps[-1]["is_last"] = True

    return {
        "public_episode": {
            "rlds_shape": RLDS_SHAPE,
            "protocol_version": PROTOCOL_VERSION,
            "public_contract": {
                "source_embodiment": source_body,
                "target_embodiment": target_embodiment,
                "target_action_adapter": {
                    "adapter_id": f"adapter/{target_embodiment}",
                    "version": "1",
                    "accepts": "target actuator ActionRef only",
                },
                "prompt_modalities": {
                    "symbolic": ["symbolic"],
                    "external_language_span": ["language"],
                    "goal_observation_span": ["observation"],
                    "demonstrations": ["demonstration", "language"],
                }[goal_carrier_kind],
            },
            "steps": steps,
        },
        "supervision": {
            "action_queries": [
                {
                    "query_id": "target-q0",
                    "correct_target_actions": [target_actuator],
                }
            ]
        },
        "private_receipt": {
            "outcome_denotation": {
                "kind": "private-monitor-predicate",
                "predicate": outcome,
            },
            "generator": {"source_assignment": private_source_assignment, "seed": "fixture-seed"},
        },
    }


def _events(bundle: Mapping[str, Any]) -> Iterable[tuple[int, Mapping[str, Any]]]:
    for index, step in enumerate(bundle["public_episode"]["steps"]):
        for event in step["protocol_events"]:
            yield index, event


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def validate(bundle: Mapping[str, Any]) -> None:
    """Validate the public contract and loss-only target labels.

    This validator intentionally cannot prove causal transformer logits or the
    correctness of a future language/vision encoder.  It proves only protocol
    structure and information-boundary properties that are meaningful without a
    learner.
    """

    _require(set(bundle) == {"public_episode", "supervision", "private_receipt"}, "bundle fields are fixed")
    episode = bundle["public_episode"]
    _require(episode.get("rlds_shape") == RLDS_SHAPE, "bundle must use the declared RLDS-shaped envelope")
    _require(episode.get("protocol_version") == PROTOCOL_VERSION, "unsupported protocol version")
    _require(isinstance(episode.get("steps"), list) and episode["steps"], "episode needs steps")
    _require(episode["steps"][0].get("is_first") is True, "first step must be marked is_first")
    _require(episode["steps"][-1].get("is_last") is True, "last step must be marked is_last")

    boundaries: dict[str, int] = {}
    goal_carriers: list[tuple[int, Mapping[str, Any]]] = []
    demonstrations: list[tuple[int, Mapping[str, Any]]] = []
    spans: dict[str, Mapping[str, Any]] = {}
    queries: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for step_index, event in _events(bundle):
        kind = event.get("kind")
        if kind == "Boundary":
            boundary = event.get("boundary")
            _require(boundary in BOUNDARIES, "unsupported boundary")
            _require(boundary not in boundaries, "each prompting boundary occurs once")
            boundaries[boundary] = step_index
        elif kind == "GoalCarrier":
            value = event.get("value")
            _require(isinstance(value, Mapping) and value.get("kind") in GOAL_CARRIER_KINDS, "unknown GoalCarrier")
            goal_carriers.append((step_index, event))
        elif kind == "ContentSpan":
            span_id = event.get("span_id")
            _require(isinstance(span_id, str) and span_id and span_id not in spans, "content spans need unique ids")
            _require(event.get("modality") in {"language", "observation"}, "unsupported content modality")
            spans[span_id] = event
        elif kind in {"Observation", "ActionExecuted"}:
            port = PortRef.from_json(event.get("port", {}))
            if kind == "ActionExecuted" and port.actor == "demonstrator":
                _require(event.get("provenance") == "demonstration", "demonstrator action must be prompt content")
                demonstrations.append((step_index, event))
        elif kind == "ActionQuery":
            query_id = event.get("query_id")
            _require(isinstance(query_id, str) and query_id and query_id not in queries, "query ids must be unique")
            _require(event.get("actor") == "target", "only target ActionQuery is supported")
            _require(isinstance(event.get("candidates"), list) and event["candidates"], "target query needs candidates")
            queries[query_id] = (step_index, event)

    _require(set(boundaries) == set(BOUNDARIES), "PromptStart, PromptEnd, and TargetStart are required")
    _require(boundaries["PromptStart"] < boundaries["PromptEnd"] < boundaries["TargetStart"], "prompt phases are out of order")
    _require(len(goal_carriers) == 1, "exactly one GoalCarrier is required")
    _require(boundaries["PromptStart"] <= goal_carriers[0][0] < boundaries["PromptEnd"], "GoalCarrier belongs in prompt")
    carrier = goal_carriers[0][1]["value"]
    if carrier["kind"] == "demonstrations":
        _require(demonstrations, "demonstrations GoalCarrier needs a demonstration")
    if carrier["kind"] in {"external_language_span", "goal_observation_span"}:
        span_id = carrier["public_content"].get("span_id")
        _require(span_id in spans, "GoalCarrier references a missing public span")
        expected_modality = "language" if carrier["kind"] == "external_language_span" else "observation"
        _require(spans[span_id].get("modality") == expected_modality, "GoalCarrier span modality mismatch")
    _require(queries, "at least one target action query is required")

    target_body = episode["public_contract"].get("target_embodiment")
    _require(isinstance(target_body, str) and target_body, "public contract needs target embodiment")
    label_map = {item.get("query_id"): item for item in bundle["supervision"].get("action_queries", [])}
    _require(set(label_map) == set(queries), "supervision must label each target query exactly once")
    for query_id, (query_step, query) in queries.items():
        _require(query_step >= boundaries["TargetStart"], "target query cannot occur before target phase")
        candidates = [PortRef.from_json(value) for value in query["candidates"]]
        _require(all(item.actor == "target" for item in candidates), "source action in target query")
        _require(all(item.namespace == "actuator" for item in candidates), "target query candidates must be actuators")
        _require(all(item.embodiment_id == target_body for item in candidates), "target query crosses embodiment")
        labels = [PortRef.from_json(value) for value in label_map[query_id].get("correct_target_actions", [])]
        _require(labels, "target query needs at least one correct action")
        _require(all(item.actor == "target" and item.embodiment_id == target_body for item in labels), "supervision crosses target embodiment")
        _require(all(item in candidates for item in labels), "supervision action must be in query candidates")

    _require("outcome_denotation" in bundle["private_receipt"], "private outcome receipt is required")
    public_json = json.dumps(episode, sort_keys=True)
    _require("outcome_denotation" not in public_json, "private outcome leaked into public episode")
    _require("source_assignment" not in public_json, "private assignment leaked into public episode")


def public_projection(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """The only learner-visible portion of a rendered bundle."""

    return copy.deepcopy(bundle["public_episode"])


def target_supervision_projection(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Loss-only labels, intentionally separate from the learner-visible episode."""

    return copy.deepcopy(bundle["supervision"])


def rename_actions(bundle: Mapping[str, Any], mapping: Mapping[str, str], *, actor: str) -> dict[str, Any]:
    """Apply a same-actor actuator rename consistently to public events and labels."""

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
    for label in result["supervision"]["action_queries"]:
        for value in label["correct_target_actions"]:
            port = PortRef.from_json(value)
            if port.actor == actor and port.namespace == "actuator" and port.local_id in mapping:
                value["local_id"] = mapping[port.local_id]
    return result


def assert_public_private_noninterference(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    """Receipts may differ, but a fixed public episode must retain its labels."""

    validate(left)
    validate(right)
    _require(public_projection(left) == public_projection(right), "noninterference comparison requires fixed public episode")
    _require(
        target_supervision_projection(left) == target_supervision_projection(right),
        "private receipt changed target supervision for identical public history",
    )
