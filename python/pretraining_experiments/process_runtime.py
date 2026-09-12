"""Compiler-facing composed process runtime.

The original :mod:`trajectory_world` remains frozen as the V2 receipt world.
This module is the reusable execution layer for generated process terms.  A
compiled term selects operators (reach, actuator lag, goal switches and
disturbance) and is executed through the same public event schema and
learner-facing rollout protocol.  Private dynamics are kept on the
environment; the teacher fits its predictor from the timed public prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import lsq_linear

from .trajectory_world import (
    ActionTarget,
    PublicEvent,
    PublicHistory,
    TrajectoryEpisode,
    _array_tuple,
    _event,
    _signed_permutation,
    _well_conditioned_body_map,
    _well_conditioned_sensor_matrix,
)


PROCESS_WORLD_VERSION = "composed-process-reach-v1"
PROCESS_CONTRACT_VERSION = "ComposedProcessReachV1"
PROCESS_TEACHER_VERSION = "timed-public-process-teacher-v1"


def load_compiled_processes(path: str | Path) -> tuple[dict[str, object], ...]:
    """Load the exact Rust process-export payload used by a training run."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("compiled process export must be a nonempty JSON list")
    result: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("process"), dict):
            raise ValueError("compiled process export has an invalid generated entry")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True)
class ProcessIR:
    """Small serializable IR emitted by the term compiler.

    Components are semantic process names.  The runtime deliberately rejects
    unknown names and duplicate state writers so a generated composition has
    one unambiguous transition semantics.
    """

    name: str = "reach"
    components: tuple[str, ...] = ("reach",)
    horizon: int = 16
    dt: float = 0.1
    actuator_lag: float = 0.45
    goal_switch_step: int | None = None
    disturbance_scale: float = 0.0
    disturbance_start_step: int | None = None
    plant_gain: float = 1.0
    goal_switch_magnitude: float = 0.22
    sensor_dim: int | None = None
    action_dim: int | None = None
    wiring: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        allowed = {"reach", "actuator_lag", "goal_switch", "disturbance"}
        components = tuple(dict.fromkeys(self.components))
        if not components or "reach" not in components:
            raise ValueError("a composed process must include reach")
        if set(components) - allowed:
            raise ValueError(f"unknown process component: {set(components) - allowed}")
        if self.horizon <= 0 or self.dt <= 0:
            raise ValueError("horizon and dt must be positive")
        if not np.isfinite(self.plant_gain) or self.plant_gain <= 0:
            raise ValueError("plant_gain must be finite and positive")
        if not np.isfinite(self.goal_switch_magnitude) or self.goal_switch_magnitude <= 0:
            raise ValueError("goal_switch_magnitude must be finite and positive")
        if self.sensor_dim is not None and self.sensor_dim < 2:
            raise ValueError("sensor_dim must be at least 2")
        if self.action_dim is not None and self.action_dim not in (2, 4):
            raise ValueError("action_dim must be 2 or 4")
        if "actuator_lag" in components and not 0.0 < self.actuator_lag <= 1.0:
            raise ValueError("actuator_lag must be in (0, 1]")
        if self.goal_switch_step is not None and not 0 <= self.goal_switch_step < self.horizon:
            raise ValueError("goal_switch_step must be inside the scored horizon")
        if "goal_switch" in components and self.goal_switch_step is None:
            raise ValueError("goal_switch requires goal_switch_step")
        if "disturbance" in components and self.disturbance_scale <= 0:
            raise ValueError("disturbance requires a positive disturbance_scale")
        if self.disturbance_start_step is not None and not 0 <= self.disturbance_start_step < self.horizon:
            raise ValueError("disturbance_start_step must be inside the scored horizon")
        if self.wiring:
            declared = set(self.wiring)
            required = {
                ("plant.state", "observation.state"),
                (
                    ("actuator_lag.out", "plant.action")
                    if "actuator_lag" in components
                    else ("action.out", "plant.action")
                ),
            }
            if "actuator_lag" in components:
                required.add(("action.out", "actuator_lag.command"))
            if "goal_switch" in components:
                required.update({("goal.out", "goal_switch.goal"), ("goal_switch.out", "goal_observation.goal")})
            if "disturbance" in components:
                required.add(("disturbance.out", "plant.disturbance"))
            missing = required - declared
            if missing:
                raise ValueError(f"compiled process wiring is incomplete: {sorted(missing)}")

    @classmethod
    def from_compiled_process(cls, compiled: Mapping[str, object] | object) -> "ProcessIR":
        """Adapt the Rust ``CompiledProcess`` JSON/object into this runtime.

        The adapter is intentionally narrow: the Rust compiler remains the
        authority for typed wiring and validation, while Python owns numerical
        integration and public event rendering.
        """

        # ``process-export`` emits GeneratedProcess records.  Accepting the
        # wrapper here keeps generator coordinates out of the executable IR.
        if isinstance(compiled, Mapping) and "process" in compiled and "nodes" not in compiled:
            compiled = compiled["process"]  # type: ignore[assignment]

        def get(name: str, default: object = None) -> object:
            if isinstance(compiled, Mapping):
                return compiled.get(name, default)
            return getattr(compiled, name, default)

        nodes = get("nodes", ()) or ()
        wiring_payload = get("wiring", ()) or ()
        schema_version = get("schema_version", 1)
        if int(schema_version) != 1:
            raise ValueError(f"unsupported compiled process schema_version={schema_version!r}")
        horizon = int(get("horizon", 0))
        sensor_dim = int(get("sensor_dim", 0))
        action_dim = int(get("action_dim", 0))
        if horizon <= 0 or sensor_dim < 2 or action_dim not in (2, 4):
            raise ValueError("compiled process has invalid horizon or dimensions")
        public_schema = get("public_schema", {})
        if not isinstance(public_schema, Mapping):
            raise ValueError("compiled process public_schema must be an object")
        if public_schema.get("schema") != "trajectory-public-events-v2":
            raise ValueError("compiled process uses an unsupported public schema")
        if int(public_schema.get("sensor_dim", -1)) != sensor_dim or int(public_schema.get("action_dim", -1)) != action_dim:
            raise ValueError("compiled process dimensions disagree with public schema")

        def node_get(node: object, key: str, default: object = None) -> object:
            if isinstance(node, Mapping):
                return node.get(key, default)
            return getattr(node, key, default)

        def kind_payload(node: object) -> tuple[str, Mapping[str, object]]:
            kind = node_get(node, "kind", {})
            if isinstance(kind, Mapping) and kind:
                key = next(iter(kind))
                value = kind[key]
                return str(key), value if isinstance(value, Mapping) else {}
            return type(kind).__name__, {}

        if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
            raise ValueError("compiled process nodes must be a sequence")
        node_kinds: dict[str, tuple[str, Mapping[str, object]]] = {}
        node_ports: dict[str, tuple[set[str], set[str]]] = {}
        port_specs: dict[str, tuple[str, str, str]] = {}
        for node in nodes:
            node_id = node_get(node, "id")
            if not isinstance(node_id, str) or not node_id:
                raise ValueError("compiled process node id must be a nonempty string")
            if node_id in node_kinds:
                raise ValueError(f"compiled process has duplicate node {node_id!r}")
            node_kinds[node_id] = kind_payload(node)
            inputs = node_get(node, "inputs", ()) or ()
            outputs = node_get(node, "outputs", ()) or ()
            if not isinstance(inputs, Sequence) or not isinstance(outputs, Sequence):
                raise ValueError(f"compiled process node {node_id!r} ports must be sequences")
            input_names: set[str] = set()
            output_names: set[str] = set()
            for port in inputs:
                port_name = node_get(port, "name")
                if not isinstance(port_name, str) or not port_name or port_name in input_names:
                    raise ValueError(f"compiled process node {node_id!r} has invalid duplicate input port")
                input_names.add(port_name)
                port_specs[f"{node_id}.{port_name}"] = (
                    "input",
                    str(node_get(port, "kind", "")),
                    str(node_get(port, "visibility", "")),
                )
            for port in outputs:
                port_name = node_get(port, "name")
                if not isinstance(port_name, str) or not port_name or port_name in output_names:
                    raise ValueError(f"compiled process node {node_id!r} has invalid duplicate output port")
                output_names.add(port_name)
                port_specs[f"{node_id}.{port_name}"] = (
                    "output",
                    str(node_get(port, "kind", "")),
                    str(node_get(port, "visibility", "")),
                )
            node_kind = node_kinds[node_id][0]
            if node_kind not in {"Input", "Output"}:
                if any(spec[2] == "Public" for endpoint, spec in port_specs.items() if endpoint.startswith(f"{node_id}.")):
                    raise ValueError(f"private process node {node_id!r} exposes a public port")
            node_ports[node_id] = (input_names, output_names)
        known_kinds = {"Input", "Plant", "ActuatorLag", "GoalSwitch", "Disturbance", "Output"}
        unknown_kinds = {kind for kind, _ in node_kinds.values()} - known_kinds
        if unknown_kinds:
            raise ValueError(f"compiled process has unknown node kinds: {sorted(unknown_kinds)}")
        # The compiled graph, rather than its family enum, is authoritative.
        # A family label with a missing/reordered node is rejected below.
        plant_nodes = [(node_id, payload) for node_id, payload in node_kinds.items() if payload[0] == "Plant"]
        observation_nodes = [
            (node_id, payload) for node_id, payload in node_kinds.items()
            if payload[0] == "Output" and payload[1].get("channel") == "Observation"
        ]
        if len(plant_nodes) != 1 or len(observation_nodes) != 1:
            raise ValueError("compiled process must contain exactly one plant and observation output")
        lag_nodes = [(node_id, payload) for node_id, payload in node_kinds.items() if payload[0] == "ActuatorLag"]
        switch_nodes = [(node_id, payload) for node_id, payload in node_kinds.items() if payload[0] == "GoalSwitch"]
        disturbance_nodes = [
            (node_id, payload) for node_id, payload in node_kinds.items()
            if payload[0] == "Disturbance" and float(payload[1].get("amplitude", 0.0)) != 0.0
        ]
        if len(lag_nodes) > 1 or len(switch_nodes) > 1 or len(disturbance_nodes) > 1:
            raise ValueError("compiled process has duplicate stateful process operators")
        components = ["reach"]
        lag_payload = lag_nodes[0][1][1] if lag_nodes else None
        switch_payload = switch_nodes[0][1][1] if switch_nodes else None
        disturbance_payload = disturbance_nodes[0][1][1] if disturbance_nodes else None
        if lag_payload is not None:
            components.append("actuator_lag")
        if switch_payload is not None:
            components.append("goal_switch")
        if disturbance_payload is not None:
            components.append("disturbance")
        runtime = get("runtime_private", {})
        if not isinstance(runtime, Mapping):
            runtime = {
                key: getattr(runtime, key, None)
                for key in ("lag_alpha", "goal_switch", "disturbance")
            }
        lag = runtime.get("lag_alpha")
        switch = runtime.get("goal_switch")
        disturbance = runtime.get("disturbance")
        if switch is not None and not isinstance(switch, Mapping):
            switch = {"step": getattr(switch, "step"), "magnitude": getattr(switch, "magnitude", 0.22)}
        if disturbance is not None and not isinstance(disturbance, Mapping):
            disturbance = {"start_step": getattr(disturbance, "start_step"), "amplitude": getattr(disturbance, "amplitude", 0.035)}
        if not isinstance(runtime, Mapping):
            raise ValueError("compiled process runtime_private must be an object")
        plant_gain = float(plant_nodes[0][1][1].get("gain", float("nan")))
        runtime_gain = float(runtime.get("dynamics_gain", float("nan")))
        if not np.isfinite(plant_gain) or not np.isfinite(runtime_gain) or not np.isclose(plant_gain, runtime_gain, rtol=0.0, atol=1.0e-12):
            raise ValueError("runtime_private.dynamics_gain disagrees with plant lowering")
        # Rust's lag alpha is retained-command weight; this runtime names the
        # complementary response weight to make the transition equation clear.
        node_lag = None if lag_payload is None else lag_payload.get("alpha")
        if (node_lag is None) != (lag is None) or (node_lag is not None and not np.isclose(float(node_lag), float(lag), rtol=0.0, atol=1.0e-12)):
            raise ValueError("runtime_private.lag_alpha disagrees with actuator-lag lowering")
        response_alpha = 1.0 - float(node_lag) if node_lag is not None else 1.0
        node_switch_step = None if switch_payload is None else switch_payload.get("at_step")
        node_switch_magnitude = None if switch_payload is None else switch_payload.get("magnitude")
        if (switch_payload is None) != (switch is None):
            raise ValueError("runtime_private.goal_switch disagrees with graph")
        if switch_payload is not None:
            if not isinstance(switch, Mapping) or int(switch.get("step", -1)) != int(node_switch_step) or not np.isclose(float(switch.get("magnitude", float("nan"))), float(node_switch_magnitude), rtol=0.0, atol=1.0e-12):
                raise ValueError("runtime_private.goal_switch disagrees with graph lowering")
        switch_step = None if node_switch_step is None else int(node_switch_step)
        disturbance_amp = 0.0 if disturbance_payload is None else float(disturbance_payload.get("amplitude", float("nan")))
        node_disturbance_start = None if disturbance_payload is None else disturbance_payload.get("start_step")
        if disturbance_payload is None:
            if disturbance is not None:
                raise ValueError("runtime_private.disturbance is present without a disturbance node")
        elif not isinstance(disturbance, Mapping) or int(disturbance.get("start_step", -1)) != int(node_disturbance_start) or not np.isclose(float(disturbance.get("amplitude", float("nan"))), disturbance_amp, rtol=0.0, atol=1.0e-12):
            raise ValueError("runtime_private.disturbance disagrees with graph lowering")
        expected_disturbance = None if disturbance_payload is None else int(node_disturbance_start)
        wiring: list[tuple[str, str]] = []
        if not isinstance(wiring_payload, Sequence) or isinstance(wiring_payload, (str, bytes)):
            raise ValueError("compiled process wiring must be a sequence")
        for wire in wiring_payload:
            if isinstance(wire, Mapping):
                wiring.append((str(wire.get("from")), str(wire.get("to"))))
            else:
                wiring.append((str(getattr(wire, "from")), str(getattr(wire, "to"))))
        if len(set(wiring)) != len(wiring):
            raise ValueError("compiled process contains duplicate wiring")
        edges: set[tuple[str, str]] = set()
        incoming: dict[str, int] = {}
        for source, target in wiring:
            if "." not in source or "." not in target:
                raise ValueError(f"invalid compiled process wire {source!r} -> {target!r}")
            source_node, source_port = source.split(".", 1)
            target_node, target_port = target.split(".", 1)
            if source_node not in node_ports or target_node not in node_ports:
                raise ValueError(f"compiled process wire references unknown node: {source!r} -> {target!r}")
            if source_port not in node_ports[source_node][1] or target_port not in node_ports[target_node][0]:
                raise ValueError(f"compiled process wire references the wrong port: {source!r} -> {target!r}")
            source_spec = port_specs[source]
            target_spec = port_specs[target]
            if source_spec[1] != target_spec[1]:
                raise ValueError(f"compiled process wire has incompatible port kinds: {source!r} -> {target!r}")
            if source_spec[2] == "Private" and target_spec[2] == "Public" and node_kinds[target_node][0] != "Output":
                raise ValueError(f"private process output leaks through {source!r} -> {target!r}")
            edge = (source_node, target_node)
            if edge in edges:
                raise ValueError(f"compiled process has duplicate node edge {edge!r}")
            edges.add(edge)
            incoming[target] = incoming.get(target, 0) + 1
        for node_id, (input_names, _) in node_ports.items():
            for input_name in input_names:
                endpoint = f"{node_id}.{input_name}"
                if incoming.get(endpoint, 0) != 1:
                    raise ValueError(f"compiled process input {endpoint!r} must have exactly one wire")
        for node_id, (_, output_names) in node_ports.items():
            if node_kinds[node_id][0] in {"Input", "Output"}:
                continue
            for output_name in output_names:
                endpoint = f"{node_id}.{output_name}"
                if not any(source == endpoint for source, _ in wiring):
                    raise ValueError(f"compiled process output {endpoint!r} is disconnected")
        # Reject hidden cyclic semantics at the Python boundary as well.  The
        # validated Rust graph permits recurrence only inside plant/lag state.
        indegree = {node_id: 0 for node_id in node_ports}
        adjacency: dict[str, list[str]] = {}
        for source_node, target_node in edges:
            indegree[target_node] += 1
            adjacency.setdefault(source_node, []).append(target_node)
        queue = [node_id for node_id, degree in indegree.items() if degree == 0]
        visited = 0
        while queue:
            node_id = queue.pop()
            visited += 1
            for next_node in adjacency.get(node_id, ()):
                indegree[next_node] -= 1
                if indegree[next_node] == 0:
                    queue.append(next_node)
        if visited != len(node_ports):
            raise ValueError("compiled process wiring contains a cycle")
        return cls(
            name=str(get("name", "compiled-process")),
            components=tuple(components),
            horizon=horizon,
            dt=float(get("dt", 0.1)),
            actuator_lag=response_alpha,
            goal_switch_step=(None if switch_step is None else int(switch_step)),
            disturbance_scale=float(disturbance_amp),
            disturbance_start_step=(None if expected_disturbance is None else int(expected_disturbance)),
            plant_gain=plant_gain,
            goal_switch_magnitude=(float(node_switch_magnitude) if node_switch_magnitude is not None else 0.22),
            sensor_dim=sensor_dim,
            action_dim=action_dim,
            wiring=tuple(wiring),
        )

    @property
    def uses_lag(self) -> bool:
        return "actuator_lag" in self.components

    def canonical(self) -> dict[str, object]:
        return {
            "name": self.name,
            "components": list(self.components),
            "horizon": self.horizon,
            "dt": self.dt,
            "actuator_lag": self.actuator_lag,
            "goal_switch_step": self.goal_switch_step,
            "disturbance_scale": self.disturbance_scale,
            "disturbance_start_step": self.disturbance_start_step,
            "plant_gain": self.plant_gain,
            "goal_switch_magnitude": self.goal_switch_magnitude,
            "sensor_dim": self.sensor_dim,
            "action_dim": self.action_dim,
            "wiring": [list(edge) for edge in self.wiring],
        }

    @property
    def contract_hash(self) -> str:
        encoded = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def compose_processes(*components: str, **kwargs: object) -> ProcessIR:
    """Build a validated process term from named operators."""

    return ProcessIR(components=tuple(components) or ("reach",), **kwargs)


@dataclass(frozen=True)
class ProcessWorldFactory:
    """Deterministic family sampler over one shared composed runtime.

    The factory owns only generator choices.  The returned rollout publishes
    no factory index, seed, or IR, so changing a sampled realization cannot
    become an accidental learner feature.
    """

    base: ProcessIR = field(default_factory=ProcessIR)
    sensor_dims: tuple[int, ...] = (8, 10)
    action_dims: tuple[int, ...] = (2, 4)
    goal_modes: tuple[str, ...] = ("absolute", "relative")

    def __post_init__(self) -> None:
        if not self.sensor_dims or any(width < 2 for width in self.sensor_dims):
            raise ValueError("sensor_dims must contain widths >= 2")
        if not self.action_dims or any(width not in (2, 4) for width in self.action_dims):
            raise ValueError("action_dims must contain 2 or 4")
        if not self.goal_modes or set(self.goal_modes) - {"absolute", "relative"}:
            raise ValueError("goal_modes must be absolute or relative")

    def sample(self, seed: int, index: int = 0) -> "ProcessRollout":
        selector = np.random.default_rng(np.random.SeedSequence((int(seed), int(index))))
        # A compiled program declares its embodiment dimensions.  The
        # generic factory still crosses dimensions for hand-authored IR, but
        # never silently executes an exported graph at a different width.
        sensor_dim = self.base.sensor_dim or int(selector.choice(self.sensor_dims))
        action_dim = self.base.action_dim or int(selector.choice(self.action_dims))
        goal_mode = str(selector.choice(self.goal_modes))
        return ProcessRollout.from_seed(seed + 7919 * index, ir=self.base, sensor_dim=sensor_dim, action_dim=action_dim, goal_mode=goal_mode)

    def generate(self, seed: int, count: int) -> tuple[TrajectoryEpisode, ...]:
        episodes: list[TrajectoryEpisode] = []
        for index in range(count):
            rollout = self.sample(seed, index)
            rollout.reset()
            episodes.append(_generate_rollout_episode(rollout))
        return tuple(episodes)


def _generate_rollout_episode(rollout: "ProcessRollout") -> TrajectoryEpisode:
    teacher = TimedPublicProcessTeacher()
    while not rollout.done:
        rollout.query()
        rollout.step(teacher.action_for(rollout.history, rollout.action_bounds), record_supervision=True)
    return rollout.to_episode()


@dataclass
class ProcessEnvironment:
    """Latent execution for a compiled composition."""

    ir: ProcessIR
    seed: int
    sensor_dim: int
    action_dim: int
    goal_mode: str
    sensor_matrix: np.ndarray
    sensor_offset: np.ndarray
    body_map: np.ndarray
    initial_state: np.ndarray
    target_state: np.ndarray
    dynamics_gain: float
    disturbance_start: int
    state: np.ndarray
    applied_action: np.ndarray
    scored_step: int = 0
    disturbance_rng: np.random.Generator | None = None

    @classmethod
    def from_seed(
        cls,
        seed: int,
        *,
        ir: ProcessIR,
        sensor_dim: int = 8,
        action_dim: int = 2,
        goal_mode: str = "absolute",
    ) -> "ProcessEnvironment":
        if sensor_dim < 2 or action_dim not in (2, 4):
            raise ValueError("supported embodiments are sensor>=2 and action widths 2 or 4")
        if ir.sensor_dim is not None and sensor_dim != ir.sensor_dim:
            raise ValueError(f"sensor_dim={sensor_dim} disagrees with compiled process ({ir.sensor_dim})")
        if ir.action_dim is not None and action_dim != ir.action_dim:
            raise ValueError(f"action_dim={action_dim} disagrees with compiled process ({ir.action_dim})")
        if goal_mode not in {"absolute", "relative"}:
            raise ValueError("goal_mode must be absolute or relative")
        root = np.random.SeedSequence(int(seed))
        task, sensor, body, dyn, disturb = [np.random.default_rng(s) for s in root.spawn(5)]
        sensor_base = _well_conditioned_sensor_matrix(sensor, sensor_dim)
        sensor_transform, _, _ = _signed_permutation(sensor, sensor_dim)
        body_base = _well_conditioned_body_map(body, action_dim)
        actuator_transform, _, _ = _signed_permutation(body, action_dim)
        initial = task.uniform(-0.22, 0.22, size=2)
        target = np.clip(initial + task.uniform(-0.24, 0.24, size=2), -0.45, 0.45)
        return cls(
            ir=ir,
            seed=int(seed),
            sensor_dim=sensor_dim,
            action_dim=action_dim,
            goal_mode=goal_mode,
            sensor_matrix=sensor_transform @ sensor_base,
            sensor_offset=sensor.uniform(-0.08, 0.08, size=sensor_dim),
            body_map=body_base @ actuator_transform,
            initial_state=initial.copy(),
            target_state=target.copy(),
            # The plant coefficient is lowered from the compiled graph.  It
            # is a semantic parameter, so a sampled substitute would make the
            # Python runtime diverge from the Rust program it claims to run.
            dynamics_gain=float(ir.plant_gain),
            disturbance_start=(
                int(ir.disturbance_start_step)
                if ir.disturbance_start_step is not None
                else (
                    int(disturb.integers(1, ir.horizon))
                    if "disturbance" in ir.components
                    else ir.horizon
                )
            ),
            state=initial.copy(),
            applied_action=np.zeros(action_dim, dtype=np.float64),
            disturbance_rng=disturb,
        )

    @property
    def action_bounds(self) -> np.ndarray:
        return np.full((self.action_dim, 2), (-0.5, 0.5), dtype=np.float64)

    def reset(self) -> np.ndarray:
        self.state = self.initial_state.copy()
        self.applied_action.fill(0.0)
        self.scored_step = 0
        return self.observe()

    def observe(self) -> np.ndarray:
        return self.sensor_matrix @ self.state + self.sensor_offset

    @property
    def target_sensor(self) -> np.ndarray:
        return self.sensor_matrix @ self.target_state + self.sensor_offset

    def goal_values(self, anchor: np.ndarray) -> np.ndarray:
        return self.target_sensor.copy() if self.goal_mode == "absolute" else self.target_sensor - anchor

    def calibration_step(self, command: Sequence[float]) -> np.ndarray:
        return self._advance(command, calibration=True)

    def step(self, command: Sequence[float]) -> np.ndarray:
        return self._advance(command, calibration=False)

    def _advance(self, command: Sequence[float], *, calibration: bool) -> np.ndarray:
        command_array = np.asarray(command, dtype=np.float64)
        if command_array.shape != (self.action_dim,) or not np.isfinite(command_array).all():
            raise ValueError("command has wrong shape or is non-finite")
        if np.any(command_array < -0.5) or np.any(command_array > 0.5):
            raise ValueError("command exceeds public action bounds")
        alpha = self.ir.actuator_lag if self.ir.uses_lag else 1.0
        self.applied_action = (1.0 - alpha) * self.applied_action + alpha * command_array
        disturbance = np.zeros(2, dtype=np.float64)
        if (
            not calibration
            and "disturbance" in self.ir.components
            and self.scored_step >= self.disturbance_start
        ):
            assert self.disturbance_rng is not None
            disturbance = self.disturbance_rng.uniform(
                -self.ir.disturbance_scale * self.ir.dt,
                self.ir.disturbance_scale * self.ir.dt,
                size=2,
            )
        self.state = np.clip(
            self.state + self.ir.dt * self.dynamics_gain * self.body_map @ self.applied_action + disturbance,
            -0.8,
            0.8,
        )
        if not calibration:
            self.scored_step += 1
        return self.observe()

    def switch_goal(self, rng: np.random.Generator) -> None:
        self.target_state = np.clip(
            self.state + rng.uniform(-self.ir.goal_switch_magnitude, self.ir.goal_switch_magnitude, size=2),
            -0.45,
            0.45,
        )

    def metrics(self) -> dict[str, object]:
        error = self.target_state - self.state
        return {
            "schema_version": "composed-process-physical-metrics-v1",
            "state_error_l2": float(np.linalg.norm(error)),
            "success": bool(np.linalg.norm(error) <= 0.07),
            "sensor_dim": self.sensor_dim,
            "action_dim": self.action_dim,
            "scored_steps": self.scored_step,
            "goal_switch_count": int("goal_switch" in self.ir.components),
        }

    def private_audit(self) -> dict[str, object]:
        return {
            "world_version": PROCESS_WORLD_VERSION,
            "world_contract_version": PROCESS_CONTRACT_VERSION,
            "process_ir": self.ir.canonical(),
            "contract_hash": self.ir.contract_hash,
            "seed": self.seed,
            "sensor_dim": self.sensor_dim,
            "action_dim": self.action_dim,
            "goal_mode": self.goal_mode,
            "initial_state": self.initial_state.tolist(),
            "target_state": self.target_state.tolist(),
            "sensor_matrix": self.sensor_matrix.tolist(),
            "body_map": self.body_map.tolist(),
            "disturbance_start_step": self.disturbance_start,
        }


def _calibration_design(history: PublicHistory, *, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    actions = history.of_kind("calibration_action")
    observations = history.of_kind("observation")[:1] + history.of_kind("calibration_observation")
    if len(observations) != len(actions) + 1 or not actions:
        raise ValueError("timed calibration prefix is incomplete")
    actions_array: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    for index, action_event in enumerate(actions):
        action = np.asarray(action_event.action, dtype=np.float64)
        if action_event.time != observations[index].time or observations[index + 1].time <= observations[index].time:
            raise ValueError("calibration events must carry strictly increasing physical times")
        actions_array.append(action)
        deltas.append(np.asarray(observations[index + 1].values) - np.asarray(observations[index].values))
    return np.asarray(actions_array), np.asarray(deltas)


@dataclass
class TimedPublicProcessTeacher:
    """Public-only controller for reach plus optional first-order lag."""

    control_gain: float = 0.9
    _fit_cache: tuple[tuple[object, ...], tuple[np.ndarray, np.ndarray, float]] | None = field(
        default=None, init=False, repr=False
    )

    def _fit(
        self,
        history: PublicHistory,
        action_dim: int,
        action_bounds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        commands, deltas = _calibration_design(history, action_dim=action_dim)
        # Cache only the public calibration context.  Including observed
        # deltas and event times prevents a command-only collision between
        # different bodies while allowing every later rollout query to reuse
        # the expensive identification fit.
        calibration_events = tuple(
            (event.kind, float(event.time), tuple(float(value) for value in event.values), tuple(float(value) for value in event.action))
            for event in history.events
            if event.kind in {"calibration_action", "calibration_observation"}
            or (event.kind == "observation" and event.event_id == 1)
        )
        cache_key: tuple[object, ...] = (
            action_dim,
            tuple(tuple(float(value) for value in row) for row in np.asarray(action_bounds)),
            calibration_events,
        )
        if self._fit_cache is not None and self._fit_cache[0] == cache_key:
            return self._fit_cache[1]
        best: tuple[float, np.ndarray, float] | None = None
        # Fit the public first-order state equation over a small deterministic
        # grid.  This is a numerical identification step, not a private
        # parameter read: every input row and output delta is public.
        # Search the complete public identification grid used by the audited
        # teacher.  A generated episode must never silently switch to a
        # no-lag or projected approximation for throughput.
        alpha_grid = np.linspace(0.05, 1.0, 16)
        for alpha in alpha_grid:
            applied = np.zeros(action_dim, dtype=np.float64)
            rows: list[np.ndarray] = []
            for command in commands:
                applied = (1.0 - alpha) * applied + alpha * command
                rows.append(applied.copy())
            design = np.asarray(rows)
            estimate, _, rank, _ = np.linalg.lstsq(design, deltas, rcond=None)
            if rank < action_dim:
                continue
            residual = float(np.sum((design @ estimate - deltas) ** 2))
            if best is None or residual < best[0]:
                best = (residual, estimate, float(alpha))
        if best is None:
            raise ValueError("timed calibration does not identify every actuator")
        _, total, alpha = best
        result = (alpha * total.T, (1.0 - alpha) * total.T, alpha)
        self._fit_cache = (cache_key, result)
        return result

    def _target(self, history: PublicHistory) -> np.ndarray:
        goals = history.of_kind("goal")
        if not goals:
            raise ValueError("public history has no goal")
        goal = goals[-1]
        if goal.key == "absolute_goal":
            return np.asarray(goal.values, dtype=np.float64)
        anchors = [event for event in history.events if event.kind == "observation" and event.time <= goal.time]
        if not anchors:
            raise ValueError("relative goal has no public anchor")
        return np.asarray(anchors[-1].values) + np.asarray(goal.values)

    def action_for(self, history: PublicHistory, action_bounds: np.ndarray) -> np.ndarray:
        bounds = np.asarray(action_bounds, dtype=np.float64)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("action_bounds must be [action_dim,2]")
        first, previous, alpha = self._fit(history, bounds.shape[0], bounds)
        current = np.asarray(history.latest_sensor().values, dtype=np.float64)
        desired = self._target(history) - current
        prior_events = [event for event in history.events if event.kind in {"calibration_action", "action_executed"}]
        applied = np.zeros(bounds.shape[0], dtype=np.float64)
        for event in prior_events:
            command = np.asarray(event.action, dtype=np.float64)
            applied = (1.0 - alpha) * applied + alpha * command
        # The fitted previous-action term is the public prediction of the lag
        # state; solve for the command that yields the desired next increment.
        rhs = desired - previous @ applied
        regularizer = np.sqrt(1.0e-8) * np.eye(bounds.shape[0])
        design = np.vstack((first, regularizer))
        target = np.concatenate((rhs, np.zeros(bounds.shape[0])))
        result = lsq_linear(design, target, bounds=(bounds[:, 0], bounds[:, 1]), max_iter=200)
        if not result.success:
            raise ValueError(f"public process control solve failed: {result.message}")
        command = result.x
        # For alpha<1 the fitted first block already contains alpha; no hidden
        # lag parameter is needed at execution time.  ``alpha`` is retained in
        # the calculation to make identification explicit and auditable.
        del alpha
        return np.clip(self.control_gain * command, bounds[:, 0], bounds[:, 1])


@dataclass
class ProcessRollout:
    environment: ProcessEnvironment
    task_rng: np.random.Generator
    events: list[PublicEvent] = field(default_factory=list)
    supervision: list[ActionTarget] = field(default_factory=list)
    executed_actions: list[tuple[float, ...]] = field(default_factory=list)
    _next_event_id: int = 0
    _pending_query_id: int | None = None
    _done: bool = False
    _clock: float = 0.0

    @classmethod
    def from_seed(cls, seed: int, *, ir: ProcessIR, sensor_dim: int = 8, action_dim: int = 2, goal_mode: str = "absolute") -> "ProcessRollout":
        environment = ProcessEnvironment.from_seed(seed, ir=ir, sensor_dim=sensor_dim, action_dim=action_dim, goal_mode=goal_mode)
        task_rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(7)[0])
        return cls(environment, task_rng)

    @property
    def action_bounds(self) -> np.ndarray:
        return self.environment.action_bounds.copy()

    @property
    def history(self) -> PublicHistory:
        return PublicHistory.from_events(self.events)

    @property
    def done(self) -> bool:
        return self._done

    def _append(self, event: PublicEvent) -> PublicEvent:
        if event.event_id != self._next_event_id:
            raise ValueError("event IDs must be contiguous")
        self.events.append(event)
        self._next_event_id += 1
        return event

    def reset(self) -> PublicHistory:
        observation = self.environment.reset()
        self.events.clear(); self.supervision.clear(); self.executed_actions.clear()
        self._next_event_id = 0; self._pending_query_id = None; self._done = False; self._clock = 0.0
        self._append(_event(0, "reset", 0.0, key="task_reset"))
        initial = observation.copy()
        self._append(_event(self._next_event_id, "observation", 0.0, values=initial, key="sensor", sensor_dim=self.environment.sensor_dim))
        self._append(_event(self._next_event_id, "goal", 0.0, values=self.environment.goal_values(initial), key=("absolute_goal" if self.environment.goal_mode == "absolute" else "relative_goal"), sensor_dim=self.environment.sensor_dim))
        # Every pulse has a physical interval.  This fixes the historical V2
        # zero-time calibration ambiguity while leaving V2 receipts untouched.
        for actuator in range(self.environment.action_dim):
            for sign in (1.0, -1.0):
                pulse = np.zeros(self.environment.action_dim); pulse[actuator] = sign * 0.08
                self._append(_event(self._next_event_id, "calibration_action", self._clock, action=pulse, key=f"calibration_{actuator}", action_dim=self.environment.action_dim))
                self._clock += self.environment.ir.dt
                sensor = self.environment.calibration_step(pulse)
                self._append(_event(self._next_event_id, "calibration_observation", self._clock, values=sensor, key="sensor", sensor_dim=self.environment.sensor_dim))
        return self.history

    def query(self) -> PublicHistory:
        if not self.events or self._done:
            raise RuntimeError("rollout is not ready for a query")
        if self._pending_query_id is not None:
            return self.history
        step = self.environment.scored_step
        if "goal_switch" in self.environment.ir.components and step == self.environment.ir.goal_switch_step:
            anchor = np.asarray(self.history.latest_sensor().values)
            self.environment.switch_goal(self.task_rng)
            self._append(_event(self._next_event_id, "goal", self._clock, values=self.environment.goal_values(anchor), key=("absolute_goal" if self.environment.goal_mode == "absolute" else "relative_goal"), sensor_dim=self.environment.sensor_dim))
        self._pending_query_id = self._append(_event(self._next_event_id, "action_query", self._clock, key="target_action")).event_id
        return self.history

    next_action_query = query

    def step(self, action: Sequence[float], *, record_supervision: bool = False) -> PublicHistory:
        if self._pending_query_id is None:
            raise RuntimeError("call query before step")
        query_id = self._pending_query_id
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (self.environment.action_dim,) or np.any(command < -0.5) or np.any(command > 0.5):
            raise ValueError("action exceeds public bounds")
        self._clock += self.environment.ir.dt
        sensor = self.environment.step(command)
        self.executed_actions.append(_array_tuple(command))
        self._append(_event(self._next_event_id, "action_executed", self._clock, action=command, key="target_action", action_dim=self.environment.action_dim))
        self._append(_event(self._next_event_id, "observation", self._clock, values=sensor, key="sensor", sensor_dim=self.environment.sensor_dim))
        if record_supervision:
            self.supervision.append(ActionTarget(query_id, _array_tuple(command), tuple(True for _ in command)))
        self._pending_query_id = None
        if self.environment.scored_step >= self.environment.ir.horizon:
            self._append(_event(self._next_event_id, "episode_end", self._clock, key="episode_end"))
            self._done = True
        return self.history

    def privileged_metrics(self) -> dict[str, object]:
        if not self._done:
            raise RuntimeError("rollout is incomplete")
        return self.environment.metrics()

    def to_episode(self) -> TrajectoryEpisode:
        if not self._done:
            raise RuntimeError("rollout is incomplete")
        audit = {**self.environment.private_audit(), "teacher_version": PROCESS_TEACHER_VERSION, "teacher_physical_metrics": self.privileged_metrics()}
        return TrajectoryEpisode(tuple(self.events), tuple(self.supervision), audit)


def generate_composed_episode(seed: int, *, ir: ProcessIR, sensor_dim: int = 8, action_dim: int = 2, goal_mode: str = "absolute") -> TrajectoryEpisode:
    rollout = ProcessRollout.from_seed(seed, ir=ir, sensor_dim=sensor_dim, action_dim=action_dim, goal_mode=goal_mode)
    rollout.reset()
    teacher = TimedPublicProcessTeacher()
    while not rollout.done:
        history = rollout.query()
        rollout.step(teacher.action_for(history, rollout.action_bounds), record_supervision=True)
    return rollout.to_episode()


def process_public_only() -> bool:
    return tuple(__import__("inspect").signature(TimedPublicProcessTeacher.action_for).parameters) == ("self", "history", "action_bounds")
