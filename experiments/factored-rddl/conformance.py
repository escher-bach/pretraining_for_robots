"""CPU-small conformance check for the factored RDDL source experiment.

This is intentionally not a Rust adapter.  It loads source RDDL with the
maintained pyRDDLGym parser/grounder, simulates the full horizon under every
legal singleton action, and writes a canonical JSON artifact whose learner
visible observations and privileged states are structurally separate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Iterable

from pyRDDLGym.core.env import RDDLEnv


ROOT = Path(__file__).resolve().parent
DOMAIN = ROOT / "visible-reassignment-domain.rddl"
BASE = ROOT / "visible-reassignment-base.rddl"
PERMUTED = ROOT / "visible-reassignment-permuted.rddl"
RELATION_EDIT = ROOT / "visible-reassignment-relation-edit.rddl"
HIDDEN_VARIANT = ROOT / "visible-reassignment-hidden-variant.rddl"

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


def environment(instance: Path) -> RDDLEnv:
    return RDDLEnv(
        domain=str(DOMAIN),
        instance=str(instance),
        enforce_action_constraints=True,
    )


def grounded_action(env: RDDLEnv, fluent: str, *objects: str) -> str:
    return env.model.ground_var(fluent, objects)


def singleton_action(env: RDDLEnv, fluent: str, *objects: str) -> dict[str, bool]:
    key = grounded_action(env, fluent, *objects)
    if key not in env.action_space.spaces:
        raise AssertionError(f"missing grounded action {key!r}")
    return {key: True}


def action_name(action: dict[str, bool]) -> str:
    return "noop" if not action else next(iter(action))


def candidate_actions(env: RDDLEnv) -> list[dict[str, bool]]:
    return [{}] + [{key: True} for key in sorted(env.action_space.spaces)]


def replay(env: RDDLEnv, actions: Iterable[dict[str, bool]]) -> list[dict[str, Any]]:
    initial_observation, _ = env.reset(seed=0)
    if any(value is not None for value in initial_observation.values()):
        raise AssertionError("POMDP reset must not publish an undeclared observation")
    steps: list[dict[str, Any]] = []
    for action in actions:
        observation, reward, terminated, truncated, _ = env.step(action)
        steps.append(
            {
                "action": action_name(action),
                "public_observation": normalize(observation),
                "privileged_state": normalize(env.state),
                "reward": normalize(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
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


def public_keys(observation: dict[str, Any]) -> set[str]:
    return set(observation)


def require_public_boundary(steps: Iterable[dict[str, Any]]) -> None:
    for step in steps:
        keys = public_keys(step["public_observation"])
        if not keys or any(
            not (key.startswith("channel-value") or key == "boundary") for key in keys
        ):
            raise AssertionError(f"unexpected public observation keys: {sorted(keys)}")
        serialized = json.dumps(step["public_observation"], sort_keys=True)
        if any(
            token in serialized
            for token in ("assigned", "source-value", "s0", "s1", "target-map", "private-token")
        ):
            raise AssertionError("a private fluent entered the public observation")


def assert_preserving_permutation() -> None:
    base = environment(BASE)
    permuted = environment(PERMUTED)
    base_trace = replay(
        base,
        [
            singleton_action(base, "pulse", "c0"),
            singleton_action(base, "pulse", "c1"),
        ],
    )
    permuted_trace = replay(
        permuted,
        [
            singleton_action(permuted, "pulse", "c0"),
            singleton_action(permuted, "pulse", "c1"),
        ],
    )
    assert [step["reward"] for step in base_trace] == [0.0, 1.0]
    assert [step["reward"] for step in base_trace] == [step["reward"] for step in permuted_trace]
    for left, right in zip(base_trace, permuted_trace, strict=True):
        assert left["public_observation"] == right["public_observation"]
    assert base_trace[-1]["privileged_state"] != permuted_trace[-1]["privileged_state"]


def assert_meaning_changing_relation_edit() -> None:
    base = environment(BASE)
    edit = environment(RELATION_EDIT)
    first = singleton_action(base, "pulse", "c0")
    old_channel = singleton_action(base, "pulse", "c0")
    new_channel = singleton_action(base, "pulse", "c1")
    assert replay(base, [first, new_channel])[-1]["reward"] == 1.0
    assert replay(edit, [first, new_channel])[-1]["reward"] == 0.0
    assert replay(edit, [first, old_channel])[-1]["reward"] == 1.0


def assert_hidden_boundary() -> None:
    base = environment(BASE)
    hidden = environment(HIDDEN_VARIANT)
    actions = [singleton_action(base, "pulse", "c0"), singleton_action(base, "pulse", "c1")]
    base_trace = replay(base, actions)
    hidden_trace = replay(hidden, actions)
    assert [step["public_observation"] for step in base_trace] == [step["public_observation"] for step in hidden_trace]
    assert [step["reward"] for step in base_trace] == [step["reward"] for step in hidden_trace]
    assert base.model.non_fluents["private-token"] is False
    assert hidden.model.non_fluents["private-token"] is True


def traces_for_artifact(env: RDDLEnv) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    def visit(prefix: list[dict[str, bool]]) -> None:
        if len(prefix) == env.horizon:
            steps = replay(env, prefix)
            require_public_boundary(steps)
            traces.append(
                {
                    "actions": [action_name(action) for action in prefix],
                    "steps": steps,
                }
            )
            return
        # Recompute the available actions from every prefix. This matters when
        # a relation-changing boundary alters a later action precondition.
        for action in legal_actions_after(env, prefix):
            visit([*prefix, action])

    visit([])
    return sorted(traces, key=lambda trace: trace["actions"])


def source_digest(instance: Path) -> str:
    payload = DOMAIN.read_bytes() + b"\0" + instance.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def build_artifact() -> dict[str, Any]:
    env = environment(BASE)
    traces = traces_for_artifact(env)
    if len(traces) != 9:
        raise AssertionError(f"expected 9 legal two-step traces, got {len(traces)}")
    return {
        "schema": "factored-rddl-grounded-traces-v1",
        "source": {
            "domain": DOMAIN.name,
            "instance": BASE.name,
            "sha256": source_digest(BASE),
            "pyRDDLGym": importlib.metadata.version("pyRDDLGym"),
        },
        "grounding": {
            "objects": normalize(env.model.type_to_objects),
            "action_fluents": sorted(env.action_space.spaces),
            "public_observation_fluents": sorted(env.observation_space.spaces),
            "privileged_state_fluents": sorted(env.model.state_fluents),
        },
        "public_transition_observation": traces,
        "privileged_state_is_separate": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="canonical JSON artifact path")
    args = parser.parse_args()

    assert_preserving_permutation()
    assert_meaning_changing_relation_edit()
    assert_hidden_boundary()
    artifact = build_artifact()
    # A second build catches accidental simulator state reuse and makes the
    # emitted JSON a deterministic, ground adapter candidate rather than a log.
    assert artifact == build_artifact()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} with {len(artifact['public_transition_observation'])} traces")


if __name__ == "__main__":
    main()
