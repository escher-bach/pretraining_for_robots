"""CPU-small conformance tests for the standalone prompted-interface fixture."""

from __future__ import annotations

import copy
import unittest

from protocol import (
    ProtocolError,
    assert_public_private_noninterference,
    make_bundle,
    rename_actions,
    target_supervision_projection,
    validate,
)


def source_action(bundle: dict) -> str:
    for step in bundle["public_episode"]["steps"]:
        for event in step["protocol_events"]:
            if event.get("kind") == "ActionExecuted" and event["port"]["actor"] == "demonstrator":
                return event["port"]["local_id"]
    raise AssertionError("fixture should contain a demonstrator action")


class PromptedInterfaceTests(unittest.TestCase):
    def test_schema_and_all_goal_carriers(self) -> None:
        for kind in (
            "symbolic",
            "external_language_span",
            "goal_observation_span",
            "demonstrations",
        ):
            validate(make_bundle(goal_carrier_kind=kind))

    def test_missing_boundary_is_rejected(self) -> None:
        bundle = make_bundle()
        bundle["public_episode"]["steps"] = [
            step
            for step in bundle["public_episode"]["steps"]
            if not any(event.get("boundary") == "PromptEnd" for event in step["protocol_events"])
        ]
        bundle["public_episode"]["steps"][-1]["is_last"] = True
        with self.assertRaisesRegex(ProtocolError, "required"):
            validate(bundle)

    def test_source_action_in_target_query_is_rejected(self) -> None:
        bundle = make_bundle()
        query = bundle["public_episode"]["steps"][-1]["protocol_events"][0]
        query["candidates"][0]["actor"] = "demonstrator"
        with self.assertRaisesRegex(ProtocolError, "source action"):
            validate(bundle)

    def test_source_and_target_action_renames_are_disjoint(self) -> None:
        base = make_bundle()
        source_renamed = rename_actions(base, {"source-slide-left": "source-turn-clockwise"}, actor="demonstrator")
        target_renamed = rename_actions(base, {"target-roll-west": "target-drive-left"}, actor="target")
        validate(source_renamed)
        validate(target_renamed)
        self.assertEqual(
            target_supervision_projection(base),
            target_supervision_projection(source_renamed),
            "renaming demonstrated action cannot rename the target label",
        )
        self.assertNotEqual(
            target_supervision_projection(base),
            target_supervision_projection(target_renamed),
            "target namespace rename must equivary target labels",
        )

    def test_no_copy_paired_target_embodiments(self) -> None:
        wheels = make_bundle(target_embodiment="target-wheels", target_action="target-roll-west")
        arm = make_bundle(target_embodiment="target-arm", target_action="target-press-west")
        validate(wheels)
        validate(arm)
        source_action_wheels = source_action(wheels)
        source_action_arm = source_action(arm)
        self.assertEqual(source_action_wheels, source_action_arm)
        self.assertNotEqual(target_supervision_projection(wheels), target_supervision_projection(arm))

    def test_same_outcome_different_demo_preserves_target_response(self) -> None:
        first = make_bundle(source_action="source-slide-left", demo_note="source circles a barrier")
        second = make_bundle(source_action="source-pivot-right", demo_note="source uses a ramp")
        validate(first)
        validate(second)
        self.assertNotEqual(first["public_episode"], second["public_episode"])
        self.assertEqual(target_supervision_projection(first), target_supervision_projection(second))

    def test_changed_outcome_same_action_surface_changes_target_response(self) -> None:
        first = make_bundle(
            outcome="place-amber-at-bay",
            target_action="target-roll-west",
            demo_observation="amber reaches the bay",
        )
        second = make_bundle(
            outcome="place-amber-at-dock",
            target_action="target-roll-east",
            demo_observation="amber reaches the dock",
        )
        validate(first)
        validate(second)
        self.assertEqual(source_action(first), source_action(second))
        self.assertNotEqual(first["public_episode"], second["public_episode"])
        self.assertNotEqual(target_supervision_projection(first), target_supervision_projection(second))

    def test_public_private_noninterference(self) -> None:
        base = make_bundle(private_source_assignment="source-a")
        altered_receipt = copy.deepcopy(base)
        altered_receipt["private_receipt"]["generator"]["source_assignment"] = "source-b"
        altered_receipt["private_receipt"]["generator"]["seed"] = "different-seed"
        assert_public_private_noninterference(base, altered_receipt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
