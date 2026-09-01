"""CPU-small RDDL grounding and prompted-interface boundary conformance.

``pyRDDLGym`` remains the parser, grounder, and simulator. This module only
adapts exhaustive finite results into the isolated ``prompted-interface/0.2``
boundary. It does not import the fixture's textual goal-marker evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

from pyRDDLGym.core.env import RDDLEnv


ROOT = Path(__file__).resolve().parent
DOMAIN = ROOT / "visible-reassignment-domain.rddl"
BASE = ROOT / "visible-reassignment-base.rddl"
PERMUTED = ROOT / "visible-reassignment-permuted.rddl"
RELATION_EDIT = ROOT / "visible-reassignment-relation-edit.rddl"
HIDDEN_VARIANT = ROOT / "visible-reassignment-hidden-variant.rddl"

PROTOCOL_VERSION = "prompted-interface/0.2"
RLDS_SHAPE = "rlds/episode-step/1"
ALIGNMENT_MANIFEST_VERSION = "adapter-alignment/0.1"
ACTION_ADAPTER_VERSION = "action-adapter/0.2"
RECEIPT_VERSION = "factored-rddl-receipt/0.2"
GOAL_GROUNDING_VERSION = "goal-grounding/0.2"
TARGET_ACTOR = "target"
TARGET_EMBODIMENT = "rddl-visible-reassignment-target"
PUBLIC_OBSERVATION_KEYS = {"boundary", "channel-value___c0", "channel-value___c1"}


def normalize(value: Any) -> Any:
    """Convert pyRDDLGym/numpy values into canonical JSON values."""
    if hasattr(value, "item"):
        return normalize(value.item())
    if hasattr(value, "tolist"):
        return normalize(value.tolist())
    if isinstance(value, dict):
        return {str(key): normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    raise TypeError(f"cannot canonicalize {type(value)!r}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def environment(instance: Path) -> RDDLEnv:
    return RDDLEnv(domain=str(DOMAIN), instance=str(instance), enforce_action_constraints=True)


def grounded_action(env: RDDLEnv, fluent: str, *objects: str) -> str:
    return env.model.ground_var(fluent, objects)


def singleton_action(env: RDDLEnv, fluent: str, *objects: str) -> dict[str, bool]:
    key = grounded_action(env, fluent, *objects)
    if key not in env.action_space.spaces:
        raise AssertionError(f"missing grounded action {key!r}")
    return {key: True}


def action_name(action: Mapping[str, bool]) -> str:
    enabled = [name for name, value in sorted(action.items()) if value]
    return "noop" if not enabled else "+".join(enabled)


def action_from_name(name: str) -> dict[str, bool]:
    return {} if name == "noop" else {name: True}


def candidate_actions(env: RDDLEnv) -> list[dict[str, bool]]:
    assert_complete_candidate_surface(env)
    return [{}] + [{key: True} for key in sorted(env.action_space.spaces)]


def all_boolean_action_assignments(env: RDDLEnv) -> list[dict[str, bool]]:
    keys = sorted(env.action_space.spaces)
    return [dict(zip(keys, values, strict=True)) for values in product((False, True), repeat=len(keys))]


def assert_complete_candidate_surface(env: RDDLEnv) -> None:
    """Prove this adapter's noop/singleton surface matches the RDDL model."""
    if env.max_allowed_actions != 1:
        raise AssertionError("this adapter requires RDDL's one-nondefault-action constraint")
    if any(getattr(space, "n", None) != 2 for space in env.action_space.spaces.values()):
        raise AssertionError("this adapter requires boolean RDDL action fluents")
    assignments = all_boolean_action_assignments(env)
    legal = []
    for assignment in assignments:
        try:
            replay(env, [assignment])
        except Exception:
            continue
        legal.append(assignment)
    if len(legal) != len(env.action_space.spaces) + 1:
        raise AssertionError("one-nondefault action constraint did not yield noop plus singleton actions")
    joint = {key: True for key in env.action_space.spaces}
    if joint in legal:
        raise AssertionError("joint boolean action bypassed RDDL max-nondef-actions")
    omitted = replay(env, [{}])
    explicit_false = replay(env, [{key: False for key in env.action_space.spaces}])
    if omitted != explicit_false:
        raise AssertionError("explicit false action assignment differs from omitted noop")
    canonical = {"noop", *sorted(env.action_space.spaces)}
    if {action_name(assignment) for assignment in legal} != canonical:
        raise AssertionError("Boolean assignment surface has an unexpected legal action")


def replay(env: RDDLEnv, actions: Iterable[Mapping[str, bool]]) -> list[dict[str, Any]]:
    initial_observation, _ = env.reset(seed=0)
    if any(value is not None for value in initial_observation.values()):
        raise AssertionError("POMDP reset must not publish an undeclared observation")
    steps: list[dict[str, Any]] = []
    for action in actions:
        observation, reward, terminated, truncated, _ = env.step(dict(action))
        steps.append({
            "action": action_name(action),
            "public_observation": normalize(observation),
            "privileged_state": normalize(env.state),
            "reward": normalize(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
    return steps


def legal_actions_after(env: RDDLEnv, prefix: list[dict[str, bool]]) -> list[dict[str, bool]]:
    legal: list[dict[str, bool]] = []
    for candidate in candidate_actions(env):
        try:
            replay(env, [*prefix, candidate])
        except Exception:
            continue
        legal.append(candidate)
    return legal


def require_public_boundary(steps: Iterable[Mapping[str, Any]]) -> None:
    for step in steps:
        keys = set(step["public_observation"])
        if keys != PUBLIC_OBSERVATION_KEYS:
            raise AssertionError(f"unexpected public observation keys: {sorted(keys)}")
        if not isinstance(step["public_observation"]["boundary"], bool) or any(
            not isinstance(step["public_observation"][key], int)
            for key in sorted(PUBLIC_OBSERVATION_KEYS - {"boundary"})
        ):
            raise AssertionError("public observation does not match its exact declared schema")
        serialized = json.dumps(step["public_observation"], sort_keys=True)
        if any(token in serialized for token in ("assigned", "source-value", "s0", "s1", "target-map", "private-token")):
            raise AssertionError("a private fluent entered the public observation")


def assert_preserving_permutation() -> None:
    base, permuted = environment(BASE), environment(PERMUTED)
    base_trace = replay(base, [singleton_action(base, "pulse", "c0"), singleton_action(base, "pulse", "c1")])
    permuted_trace = replay(permuted, [singleton_action(permuted, "pulse", "c0"), singleton_action(permuted, "pulse", "c1")])
    assert [step["reward"] for step in base_trace] == [0.0, 1.0]
    assert base_trace[0]["public_observation"] == {"boundary": True, "channel-value___c0": 0, "channel-value___c1": 1}
    assert [step["reward"] for step in base_trace] == [step["reward"] for step in permuted_trace]
    assert [step["public_observation"] for step in base_trace] == [step["public_observation"] for step in permuted_trace]
    assert base_trace[-1]["privileged_state"] != permuted_trace[-1]["privileged_state"]


def assert_meaning_changing_relation_edit() -> None:
    base, edit = environment(BASE), environment(RELATION_EDIT)
    first = singleton_action(base, "pulse", "c0")
    old_channel, new_channel = singleton_action(base, "pulse", "c0"), singleton_action(base, "pulse", "c1")
    assert replay(base, [first, new_channel])[-1]["reward"] == 1.0
    assert replay(edit, [first, new_channel])[-1]["reward"] == 0.0
    assert replay(edit, [first, old_channel])[-1]["reward"] == 1.0
    assert replay(edit, [first])[0]["public_observation"] == {"boundary": True, "channel-value___c0": 1, "channel-value___c1": 0}


def assert_hidden_boundary() -> None:
    base, hidden = environment(BASE), environment(HIDDEN_VARIANT)
    actions = [singleton_action(base, "pulse", "c0"), singleton_action(base, "pulse", "c1")]
    base_trace, hidden_trace = replay(base, actions), replay(hidden, actions)
    assert [step["public_observation"] for step in base_trace] == [step["public_observation"] for step in hidden_trace]
    assert [step["reward"] for step in base_trace] == [step["reward"] for step in hidden_trace]
    assert base.model.non_fluents["private-token"] is False
    assert hidden.model.non_fluents["private-token"] is True


def assert_exhaustive_transform_matrix() -> None:
    """Retain the three transforms over all legal complete traces, not one path."""
    base, permuted = environment(BASE), environment(PERMUTED)
    relation_edit, hidden = environment(RELATION_EDIT), environment(HIDDEN_VARIANT)
    traces = {"base": traces_for_artifact(base), "permuted": traces_for_artifact(permuted), "relation_edit": traces_for_artifact(relation_edit), "hidden": traces_for_artifact(hidden)}
    for name, trace_set in traces.items():
        if len(trace_set) != 9:
            raise AssertionError(f"{name} did not retain the full legal two-step support")
    by_actions = {name: {tuple(trace["actions"]): trace for trace in trace_set} for name, trace_set in traces.items()}
    if not all(set(by_actions["base"]) == set(by_actions[name]) for name in ("permuted", "relation_edit", "hidden")):
        raise AssertionError("a transform changed legal action support")
    for actions, base_trace in by_actions["base"].items():
        permuted_trace, hidden_trace = by_actions["permuted"][actions], by_actions["hidden"][actions]
        if [step["public_observation"] for step in base_trace["steps"]] != [step["public_observation"] for step in permuted_trace["steps"]]:
            raise AssertionError("source permutation changed a public trace")
        if [step["reward"] for step in base_trace["steps"]] != [step["reward"] for step in permuted_trace["steps"]]:
            raise AssertionError("source permutation changed a reward trace")
        if [step["public_observation"] for step in base_trace["steps"]] != [step["public_observation"] for step in hidden_trace["steps"]] or [step["reward"] for step in base_trace["steps"]] != [step["reward"] for step in hidden_trace["steps"]]:
            raise AssertionError("hidden-only variant changed public or reward behavior")
    if not any(
        [step["reward"] for step in base_trace["steps"]] != [step["reward"] for step in by_actions["relation_edit"][actions]["steps"]]
        for actions, base_trace in by_actions["base"].items()
    ):
        raise AssertionError("relation edit did not change any exhaustive reward trace")


def traces_for_artifact(env: RDDLEnv) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []

    def visit(prefix: list[dict[str, bool]]) -> None:
        if len(prefix) == env.horizon:
            steps = replay(env, prefix)
            require_public_boundary(steps)
            traces.append({"actions": [action_name(action) for action in prefix], "steps": steps})
            return
        for action in legal_actions_after(env, prefix):
            visit([*prefix, action])

    visit([])
    return sorted(traces, key=lambda trace: trace["actions"])


def source_digest(instance: Path) -> str:
    return hashlib.sha256(DOMAIN.read_bytes() + b"\0" + instance.read_bytes()).hexdigest()


def _assignment_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key.startswith("assigned")}


def _snapshot(env: RDDLEnv, prefix_names: list[str]) -> dict[str, Any]:
    steps = replay(env, [action_from_name(name) for name in prefix_names])
    if not steps:
        env.reset(seed=0)
        return {"public_observation": {}, "privileged_state": normalize(env.state), "terminated": False, "truncated": False}
    return {key: steps[-1][key] for key in ("public_observation", "privileged_state", "terminated", "truncated")}


def _node_key(depth: int, state: Mapping[str, Any]) -> tuple[int, str]:
    return depth, digest(state)


def _node_id(depth: int, state: Mapping[str, Any]) -> str:
    return f"state-{depth}-{digest(state)}"


def _prefixes(env: RDDLEnv) -> dict[tuple[int, str], list[str]]:
    representatives: dict[tuple[int, str], list[str]] = {}
    queue: list[list[str]] = [[]]
    while queue:
        prefix = queue.pop(0)
        snapshot = _snapshot(env, prefix)
        key = _node_key(len(prefix), snapshot["privileged_state"])
        old = representatives.get(key)
        if old is None or tuple(prefix) < tuple(old):
            representatives[key] = prefix
        if len(prefix) >= env.horizon or snapshot["terminated"] or snapshot["truncated"]:
            continue
        for action in legal_actions_after(env, [action_from_name(name) for name in prefix]):
            queue.append([*prefix, action_name(action)])
    return representatives


def build_canonical_graph(env: RDDLEnv) -> dict[str, Any]:
    """Canonical finite graph of every reachable privileged state and legal action."""
    representatives = _prefixes(env)
    nodes: dict[tuple[int, str], dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for key, prefix in sorted(representatives.items()):
        snapshot = _snapshot(env, prefix)
        state = snapshot["privileged_state"]
        nodes[key] = {
            "node_id": _node_id(len(prefix), state), "depth": len(prefix), "representative_history": prefix,
            "privileged_state": state, "assignments": _assignment_projection(state),
            "public_observation": snapshot["public_observation"],
            "terminal": bool(snapshot["terminated"] or snapshot["truncated"] or len(prefix) == env.horizon),
            "terminated": snapshot["terminated"], "truncated": snapshot["truncated"], "legal_actions": [],
        }
    for key, node in sorted(nodes.items()):
        if node["terminal"]:
            continue
        prefix = node["representative_history"]
        for action in legal_actions_after(env, [action_from_name(name) for name in prefix]):
            node["legal_actions"].append(action_name(action))
            next_prefix = [*prefix, action_name(action)]
            step = replay(env, [action_from_name(name) for name in next_prefix])[-1]
            destination_key = _node_key(len(next_prefix), step["privileged_state"])
            if destination_key not in nodes:
                raise AssertionError("reachable destination was not enumerated")
            edges.append({"from_node_id": node["node_id"], "action": action_name(action), "to_node_id": nodes[destination_key]["node_id"], "reward": step["reward"], "terminated": step["terminated"], "truncated": step["truncated"]})
    root = _node_key(0, _snapshot(env, [])["privileged_state"])
    return {
        "version": "finite-rddl-graph/0.1",
        "scope": "reachable-privileged-states-and-legal-actions-through-fixed-rddl-horizon",
        "root_node_id": nodes[root]["node_id"],
        "nodes": sorted(nodes.values(), key=lambda node: node["node_id"]),
        "edges": sorted(edges, key=lambda edge: (edge["from_node_id"], edge["action"])),
    }


def _continuation_oracle(traces: list[Mapping[str, Any]], prefix: list[str]) -> dict[str, Any]:
    """Exact labels and private outcomes from all simulator continuations."""
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        actions = trace["actions"]
        if actions[:len(prefix)] != prefix or len(actions) <= len(prefix):
            continue
        continuation = trace["steps"][len(prefix):]
        by_action[actions[len(prefix)]].append({
            "cumulative_return": sum(float(step["reward"]) for step in continuation),
            "reward_sequence": [step["reward"] for step in continuation],
            "terminal_privileged_state": continuation[-1]["privileged_state"],
            "terminal_assignments": _assignment_projection(continuation[-1]["privileged_state"]),
        })
    if not by_action:
        raise AssertionError("a nonterminal graph node has no exhaustive continuations")
    best_cumulative_returns = {action: max(value["cumulative_return"] for value in values) for action, values in by_action.items()}
    best = max(best_cumulative_returns.values())
    return {
        "best_cumulative_return": best,
        "correct_actions": sorted(action for action, value in best_cumulative_returns.items() if value == best),
        "outcomes": {
            action: {
                "continuations": sorted(values, key=canonical_bytes),
                "best_cumulative_return": best_cumulative_returns[action],
                "goal_satisfying": best_cumulative_returns[action] == best,
            }
            for action, values in sorted(by_action.items())
        },
    }


def _event(kind: str, event_id: str, phase: str, *, actor: str = "environment", embodiment_id: str = "visible-reassignment-world", modality: str = "symbolic", namespace: str = "episode", local_id: str | None = None, **fields: Any) -> dict[str, Any]:
    return {"event_id": event_id, "phase": phase, "kind": kind, "actor": actor, "embodiment_id": embodiment_id, "modality": modality, "namespace": namespace, "local_id": local_id or event_id, **fields}


def _port(local_id: str, namespace: str = "actuator") -> dict[str, str]:
    return {"actor": TARGET_ACTOR, "embodiment_id": TARGET_EMBODIMENT, "namespace": namespace, "local_id": local_id}


def _step(index: int, events: list[dict[str, Any]], *, terminal: bool = False) -> dict[str, Any]:
    return {"is_first": index == 0, "is_last": terminal, "is_terminal": terminal, "observation": [], "action": None, "reward": None, "discount": 0.0 if terminal else 1.0, "protocol_events": events}


def render_query_episode(prefix: list[str], snapshot: Mapping[str, Any], private_history: list[Mapping[str, Any]], oracle: Mapping[str, Any], graph: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    """Render one legal history: public events, loss-only labels, private receipt."""
    token = digest(prefix)[:16]
    query_id, publication_id = f"query-{token}", f"publication-{token}"
    carrier = {"carrier_id": "rddl-cumulative-return-goal", "kind": "symbolic", "public_content": {"predicate": "maximize cumulative return", "content_id": "rddl-cumulative-return-goal-v1"}}
    carrier_event = _event("GoalCarrier", f"goal-{token}", "prompt", modality="symbolic", namespace="goal", local_id="rddl-cumulative-return-goal", value=carrier)
    groups: list[list[dict[str, Any]]] = [
        [_event("Schema", f"schema-{token}", "schema", embodiment_id=TARGET_EMBODIMENT, local_id="target-interface"), _event("Boundary", f"prompt-start-{token}", "prompt", local_id="PromptStart", boundary="PromptStart")],
        [carrier_event],
        [_event("Boundary", f"prompt-end-{token}", "prompt", local_id="PromptEnd", boundary="PromptEnd")],
        [_event("Boundary", f"target-start-{token}", "target", local_id="TargetStart", boundary="TargetStart")],
    ]
    for index, action in enumerate(prefix):
        groups.append([_event("ActionExecuted", f"history-action-{index}-{token}", "target", actor=TARGET_ACTOR, embodiment_id=TARGET_EMBODIMENT, modality="action", namespace="actuator", local_id=action, port=_port(action), provenance="target-history")])
    groups.extend([
        [_event("Observation", f"observation-{token}", "target", actor=TARGET_ACTOR, embodiment_id=TARGET_EMBODIMENT, modality="observation", namespace="observation", local_id="channel-values", port=_port("channel-values", "observation"), value=snapshot["public_observation"]), _event("ActionQuery", query_id, "target", actor=TARGET_ACTOR, embodiment_id=TARGET_EMBODIMENT, modality="action", namespace="actuator", local_id=query_id, query_id=query_id, candidates=[_port(action) for action in oracle["outcomes"]])],
        [_event("EpisodeEnd", f"episode-end-{token}", "terminal", local_id="episode-end")],
    ])
    public_episode = {"rlds_shape": RLDS_SHAPE, "protocol_version": PROTOCOL_VERSION, "public_contract": {"source_embodiment": "visible-reassignment-world", "target_embodiment": TARGET_EMBODIMENT, "goal_publication": {"publication_id": publication_id, "visibility": "public", "denotation_visibility": "private", "carrier_event_id": carrier_event["event_id"]}, "action_adapters": {"target": {"version": ACTION_ADAPTER_VERSION, "actor": TARGET_ACTOR, "embodiment_id": TARGET_EMBODIMENT, "namespace": "actuator", "candidates": sorted(oracle["outcomes"]), "executable": sorted(oracle["outcomes"])}}, "adapter_alignment_manifest": {"version": ALIGNMENT_MANIFEST_VERSION, "entries": []}}, "steps": [_step(index, group, terminal=index == len(groups) - 1) for index, group in enumerate(groups)]}
    supervision = {"action_queries": [{"query_event_id": query_id, "target_actor": TARGET_ACTOR, "target_embodiment": TARGET_EMBODIMENT, "correct_target_actions": [_port(action) for action in oracle["correct_actions"]]}]}
    private_receipt = {
        "version": RECEIPT_VERSION, "source": source, "history": prefix, "privileged_history": private_history, "privileged_state": snapshot["privileged_state"], "assignments": _assignment_projection(snapshot["privileged_state"]),
        "goal_denotation": {"denotation_id": "rddl-maximize-cumulative-return", "kind": "max-cumulative-return", "horizon": 2},
        "evaluator": {"version": "factored-rddl-exhaustive-evaluator/0.2", "query_action_outcomes": {query_id: oracle["outcomes"]}},
        "goal_grounding": {"version": GOAL_GROUNDING_VERSION, "method": "exhaustive-pyrddlgym-simulation", "publication_id": publication_id, "carrier_event_id": carrier_event["event_id"], "carrier_hash": digest(carrier_event), "denotation_id": "rddl-maximize-cumulative-return", "graph_sha256": digest(graph), "best_cumulative_return": oracle["best_cumulative_return"], "grounded_target_actions": oracle["correct_actions"], "outcome_sha256": digest(oracle["outcomes"])},
    }
    return {"public_episode": public_episode, "supervision": supervision, "private_receipt": private_receipt}


def _events(episode: Mapping[str, Any]) -> Iterable[tuple[int, Mapping[str, Any]]]:
    for index, step in enumerate(episode["steps"]):
        for event in step["protocol_events"]:
            yield index, event


def derive_target_supervision(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute loss-only labels from private exhaustive query outcomes."""
    labels = []
    for _, event in _events(bundle["public_episode"]):
        if event["kind"] == "ActionQuery":
            outcomes = bundle["private_receipt"]["evaluator"]["query_action_outcomes"][event["event_id"]]
            best = max(value["best_cumulative_return"] for value in outcomes.values())
            labels.append({"query_event_id": event["event_id"], "target_actor": TARGET_ACTOR, "target_embodiment": TARGET_EMBODIMENT, "correct_target_actions": [_port(action) for action in sorted(outcomes) if outcomes[action]["best_cumulative_return"] == best]})
    return {"action_queries": labels}


def validate_rendered_episode(bundle: Mapping[str, Any]) -> None:
    """Validate the public 0.2 profile and the source-specific private receipt."""
    if set(bundle) != {"public_episode", "supervision", "private_receipt"}:
        raise AssertionError("rendered bundle fields are fixed")
    episode = bundle["public_episode"]
    if episode["rlds_shape"] != RLDS_SHAPE or episode["protocol_version"] != PROTOCOL_VERSION:
        raise AssertionError("unexpected prompted-interface envelope")
    if episode["steps"][0]["is_first"] is not True or episode["steps"][-1]["is_last"] is not True:
        raise AssertionError("invalid RLDS step endpoints")
    contract = episode["public_contract"]
    if set(contract) != {"source_embodiment", "target_embodiment", "goal_publication", "action_adapters", "adapter_alignment_manifest"}:
        raise AssertionError("public contract has trailing or missing fields")
    if not isinstance(contract["source_embodiment"], str) or not contract["source_embodiment"]:
        raise AssertionError("public contract requires a source embodiment")
    publication = contract["goal_publication"]
    if set(publication) != {"publication_id", "visibility", "denotation_visibility", "carrier_event_id"} or publication["visibility"] != "public" or publication["denotation_visibility"] != "private":
        raise AssertionError("GoalPublication contract is invalid")
    adapter = contract["action_adapters"]["target"]
    if set(contract["action_adapters"]) != {"target"} or set(adapter) != {"version", "actor", "embodiment_id", "namespace", "candidates", "executable"}:
        raise AssertionError("target action adapter has trailing or missing fields")
    if adapter["version"] != ACTION_ADAPTER_VERSION or adapter["actor"] != TARGET_ACTOR or adapter["embodiment_id"] != TARGET_EMBODIMENT or set(adapter["candidates"]) != set(adapter["executable"]):
        raise AssertionError("target action adapter is not scoped to executable target actions")
    if contract["adapter_alignment_manifest"] != {"version": ALIGNMENT_MANIFEST_VERSION, "entries": []}:
        raise AssertionError("RDDL native observations need an empty declared sidecar manifest")
    schemas = {
        "Schema": {"event_id", "phase", "kind", "actor", "embodiment_id", "modality", "namespace", "local_id"},
        "Boundary": {"event_id", "phase", "kind", "actor", "embodiment_id", "modality", "namespace", "local_id", "boundary"},
        "GoalCarrier": {"event_id", "phase", "kind", "actor", "embodiment_id", "modality", "namespace", "local_id", "value"},
        "Observation": {"event_id", "phase", "kind", "actor", "embodiment_id", "modality", "namespace", "local_id", "port", "value"},
        "ActionExecuted": {"event_id", "phase", "kind", "actor", "embodiment_id", "modality", "namespace", "local_id", "port", "provenance"},
        "ActionQuery": {"event_id", "phase", "kind", "actor", "embodiment_id", "modality", "namespace", "local_id", "query_id", "candidates"},
        "EpisodeEnd": {"event_id", "phase", "kind", "actor", "embodiment_id", "modality", "namespace", "local_id"},
    }
    ids: set[str] = set()
    queries = []
    boundaries: set[str] = set()
    carriers = []
    phase_order = {"schema": 0, "prompt": 1, "target": 2, "terminal": 3}
    last_phase = 0
    for _, event in _events(episode):
        kind = event.get("kind")
        if kind not in schemas or set(event) != schemas[kind]:
            raise AssertionError("public event field schema is closed")
        if event["event_id"] in ids or event["phase"] not in phase_order or phase_order[event["phase"]] < last_phase:
            raise AssertionError("public events need unique addresses and monotonic phases")
        last_phase = phase_order[event["phase"]]
        ids.add(event["event_id"])
        if event["kind"] == "Boundary":
            boundaries.add(event["local_id"])
        if event["kind"] == "GoalCarrier":
            value = event["value"]
            if event["phase"] != "prompt" or event["actor"] != "environment" or event["namespace"] != "goal" or set(value) != {"carrier_id", "kind", "public_content"} or value["carrier_id"] != event["local_id"] or value["kind"] != "symbolic" or set(value["public_content"]) != {"predicate", "content_id"}:
                raise AssertionError("GoalCarrier does not follow the public 0.2 symbolic schema")
            carriers.append(event)
        if event["kind"] == "Observation":
            port = event["port"]
            if set(port) != {"actor", "embodiment_id", "namespace", "local_id"} or port["actor"] != TARGET_ACTOR or port["embodiment_id"] != TARGET_EMBODIMENT or port["namespace"] != "observation":
                raise AssertionError("observation reference crosses actor, embodiment, or namespace")
        if event["kind"] in {"ActionExecuted", "ActionQuery"}:
            ports = [event["port"]] if event["kind"] == "ActionExecuted" else event["candidates"]
            for port in ports:
                if set(port) != {"actor", "embodiment_id", "namespace", "local_id"} or port["actor"] != TARGET_ACTOR or port["embodiment_id"] != TARGET_EMBODIMENT or port["namespace"] != "actuator" or port["local_id"] not in adapter["executable"]:
                    raise AssertionError("action reference is not executable by the target body")
        if event["kind"] == "ActionQuery":
            if {port["local_id"] for port in event["candidates"]} != set(adapter["executable"]):
                raise AssertionError("query candidates must equal the complete executable target surface")
            queries.append(event)
    if boundaries != {"PromptStart", "PromptEnd", "TargetStart"} or len(queries) != 1 or len(carriers) != 1 or bundle["supervision"] != derive_target_supervision(bundle):
        raise AssertionError("query labels must be exactly exhaustive and loss-only")
    carrier = carriers[0]
    if publication["carrier_event_id"] != carrier["event_id"]:
        raise AssertionError("GoalPublication contract does not bind its public carrier")
    if any(step["reward"] is not None for step in episode["steps"]):
        raise AssertionError("numeric RDDL rewards are loss-only private evaluator evidence")
    public_json = canonical_bytes(episode).decode("utf-8")
    if any(token in public_json for token in ("\"privileged_state\"", "\"assignments\"", "\"query_action_outcomes\"", "\"goal_denotation\"", "source-value", "\"assigned___")):
        raise AssertionError("private simulator data leaked into public episode")
    private = bundle["private_receipt"]
    if private["evaluator"]["version"] != "factored-rddl-exhaustive-evaluator/0.2":
        raise AssertionError("private evaluator must be the versioned RDDL source adapter")
    grounding = private["goal_grounding"]
    outcomes = private["evaluator"]["query_action_outcomes"][queries[0]["event_id"]]
    expected_actions = [port["local_id"] for port in derive_target_supervision(bundle)["action_queries"][0]["correct_target_actions"]]
    if grounding["method"] != "exhaustive-pyrddlgym-simulation" or grounding["carrier_hash"] != digest(carrier) or grounding["publication_id"] != publication["publication_id"] or grounding["carrier_event_id"] != publication["carrier_event_id"] or grounding["denotation_id"] != private["goal_denotation"]["denotation_id"] or grounding["outcome_sha256"] != digest(outcomes) or grounding["grounded_target_actions"] != expected_actions or grounding["best_cumulative_return"] != max(value["best_cumulative_return"] for value in outcomes.values()):
        raise AssertionError("goal-grounding receipt disagrees with exact public/private linkage")


def replay_equivalence(graph: Mapping[str, Any], traces: list[Mapping[str, Any]]) -> dict[str, Any]:
    nodes = {node["node_id"]: node for node in graph["nodes"]}
    edges = {(edge["from_node_id"], edge["action"]): edge for edge in graph["edges"]}
    checks = 0
    for trace in traces:
        node_id = graph["root_node_id"]
        for action, step in zip(trace["actions"], trace["steps"], strict=True):
            edge = edges.get((node_id, action))
            if edge is None:
                raise AssertionError("canonical graph omitted a replay action")
            if action not in nodes[node_id]["legal_actions"]:
                raise AssertionError("canonical graph labelled a replay action as illegal")
            node_id, node = edge["to_node_id"], nodes[edge["to_node_id"]]
            if edge["reward"] != step["reward"] or edge["terminated"] != step["terminated"] or edge["truncated"] != step["truncated"] or node["privileged_state"] != step["privileged_state"] or node["public_observation"] != step["public_observation"]:
                raise AssertionError("graph terminal, reward, or destination differs from simulator replay")
            checks += 1
    if any(
        sorted(edge["action"] for edge in graph["edges"] if edge["from_node_id"] == node["node_id"]) != node["legal_actions"]
        for node in graph["nodes"]
    ):
        raise AssertionError("graph legal action declarations and transition rows disagree")
    return {"version": "replay-equivalence/0.1", "scope": graph["scope"], "complete_traces_checked": len(traces), "transition_checks": checks, "all_replays_match": True, "trace_set_sha256": digest(traces), "graph_sha256": digest(graph)}


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Recompute the RDDL receipt so self-consistent private tampering fails."""
    if set(artifact) != {"schema", "rendered_query_episodes", "private_graph_receipt"} or artifact["schema"] != "factored-rddl-prompted-interface/0.2":
        raise AssertionError("artifact schema is invalid")
    env = environment(BASE)
    traces = traces_for_artifact(env)
    source = {"domain": DOMAIN.name, "instance": BASE.name, "sha256": source_digest(BASE), "pyRDDLGym": importlib.metadata.version("pyRDDLGym")}
    private_graph = artifact["private_graph_receipt"]
    if private_graph["source"] != source or private_graph["complete_privileged_traces"] != traces:
        raise AssertionError("private trace receipt disagrees with fresh pyRDDLGym replay")
    graph = build_canonical_graph(env)
    if private_graph["canonical_finite_graph"] != graph:
        raise AssertionError("canonical reachable graph disagrees with fresh pyRDDLGym replay")
    expected_goal = {"denotation_id": "rddl-maximize-cumulative-return", "criterion": "maximize-sum-of-simulator-rewards-from-query-through-fixed-horizon"}
    if private_graph["goal_semantics"] != expected_goal or private_graph["replay_equivalence"] != replay_equivalence(graph, traces):
        raise AssertionError("private graph goal or replay receipt is stale")
    episodes = artifact["rendered_query_episodes"]
    by_history = {tuple(bundle["private_receipt"]["history"]): bundle for bundle in episodes}
    expected_nodes = [node for node in graph["nodes"] if not node["terminal"]]
    if len(by_history) != len(episodes) or set(by_history) != {tuple(node["representative_history"]) for node in expected_nodes}:
        raise AssertionError("rendered query histories do not cover the reachable nonterminal graph")
    for node in expected_nodes:
        history = node["representative_history"]
        bundle = by_history[tuple(history)]
        validate_rendered_episode(bundle)
        private = bundle["private_receipt"]
        expected_history = replay(env, [action_from_name(name) for name in history])
        oracle = _continuation_oracle(traces, history)
        query_id = bundle["supervision"]["action_queries"][0]["query_event_id"]
        if private["source"] != source or private["goal_denotation"] != {"denotation_id": "rddl-maximize-cumulative-return", "kind": "max-cumulative-return", "horizon": 2} or private["privileged_history"] != expected_history or private["privileged_state"] != node["privileged_state"] or private["assignments"] != node["assignments"]:
            raise AssertionError("episode privileged receipt disagrees with fresh pyRDDLGym replay")
        if private["evaluator"]["query_action_outcomes"] != {query_id: oracle["outcomes"]}:
            raise AssertionError("episode query outcomes disagree with exhaustive simulator continuations")
        if bundle["supervision"] != derive_target_supervision(bundle):
            raise AssertionError("episode supervision disagrees with exhaustive query outcomes")
        grounding = private["goal_grounding"]
        if grounding["graph_sha256"] != digest(graph) or grounding["best_cumulative_return"] != oracle["best_cumulative_return"] or grounding["grounded_target_actions"] != oracle["correct_actions"] or grounding["outcome_sha256"] != digest(oracle["outcomes"]):
            raise AssertionError("episode grounding disagrees with exhaustive simulator continuations")


def build_artifact() -> dict[str, Any]:
    env = environment(BASE)
    traces = traces_for_artifact(env)
    if len(traces) != 9:
        raise AssertionError(f"expected 9 legal two-step traces, got {len(traces)}")
    graph = build_canonical_graph(env)
    source = {"domain": DOMAIN.name, "instance": BASE.name, "sha256": source_digest(BASE), "pyRDDLGym": importlib.metadata.version("pyRDDLGym")}
    episodes = []
    for node in graph["nodes"]:
        if not node["terminal"]:
            history = replay(env, [action_from_name(name) for name in node["representative_history"]])
            episode = render_query_episode(node["representative_history"], node, history, _continuation_oracle(traces, node["representative_history"]), graph, source)
            validate_rendered_episode(episode)
            episodes.append(episode)
    artifact = {"schema": "factored-rddl-prompted-interface/0.2", "rendered_query_episodes": sorted(episodes, key=lambda item: item["private_receipt"]["history"]), "private_graph_receipt": {"source": source, "canonical_finite_graph": graph, "goal_semantics": {"denotation_id": "rddl-maximize-cumulative-return", "criterion": "maximize-sum-of-simulator-rewards-from-query-through-fixed-horizon"}, "replay_equivalence": replay_equivalence(graph, traces), "complete_privileged_traces": traces}}
    validate_artifact(artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="canonical JSON artifact path")
    args = parser.parse_args()
    assert_preserving_permutation()
    assert_meaning_changing_relation_edit()
    assert_hidden_boundary()
    assert_exhaustive_transform_matrix()
    artifact = build_artifact()
    if artifact != build_artifact():
        raise AssertionError("adapter artifact is not deterministic across fresh simulator builds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(artifact) + b"\n")
    print(f"wrote {args.output} with {len(artifact['rendered_query_episodes'])} rendered query episodes")


if __name__ == "__main__":
    main()
