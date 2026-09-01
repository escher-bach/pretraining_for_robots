"""CPU-small conformance tests for the standalone prompted-interface fixture."""

from __future__ import annotations

import copy
import json
import unittest

from protocol import (
    ProtocolError,
    assert_public_private_noninterference,
    derive_target_supervision,
    learner_input_projection,
    make_bundle,
    rename_actions,
    target_supervision_projection,
    validate,
    _events,
)


def source_action(bundle: dict) -> str:
    for step in bundle["public_episode"]["steps"]:
        for event in step["protocol_events"]:
            if event.get("kind") == "ActionExecuted" and event["port"]["actor"] == "demonstrator":
                return event["port"]["local_id"]
    raise AssertionError("fixture should contain a demonstrator action")


def target_query(bundle: dict) -> dict:
    for step in bundle["public_episode"]["steps"]:
        for event in step["protocol_events"]:
            if event.get("kind") == "ActionQuery":
                return event
    raise AssertionError("fixture should contain a target action query")


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
        query = target_query(bundle)
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
        )
        second = make_bundle(
            outcome="place-amber-at-dock",
            target_action="target-roll-west",
        )
        validate(first)
        validate(second)
        self.assertEqual(source_action(first), source_action(second))
        self.assertNotEqual(first["public_episode"], second["public_episode"])
        self.assertEqual(
            first["public_episode"]["public_contract"]["action_adapters"],
            second["public_episode"]["public_contract"]["action_adapters"],
        )
        self.assertNotEqual(target_supervision_projection(first), target_supervision_projection(second))

    def test_public_private_noninterference(self) -> None:
        base = make_bundle(private_source_assignment="source-a")
        altered_receipt = copy.deepcopy(base)
        altered_receipt["private_receipt"]["generator"]["source_assignment"] = "source-b"
        altered_receipt["private_receipt"]["generator"]["seed"] = "different-seed"
        assert_public_private_noninterference(base, altered_receipt)

    def test_event_addresses_and_alignment_manifest_are_explicit(self) -> None:
        bundle = make_bundle(goal_carrier_kind="external_language_span")
        events = [event for step in bundle["public_episode"]["steps"] for event in step["protocol_events"]]
        required = {"event_id", "phase", "actor", "embodiment_id", "modality", "namespace", "local_id"}
        self.assertTrue(all(required <= set(event) for event in events))
        manifest = bundle["public_episode"]["public_contract"]["adapter_alignment_manifest"]
        self.assertEqual({entry["event_id"] for entry in manifest["entries"]}, {"language-0"})
        self.assertNotIn("content", json.dumps(manifest))

    def test_missing_or_trailing_span_reference_is_rejected(self) -> None:
        missing = make_bundle(goal_carrier_kind="external_language_span")
        carrier = next(event for _, event in _events(missing) if event["kind"] == "GoalCarrier")
        carrier["value"]["public_content"]["span_id"] = "missing-span"
        with self.assertRaisesRegex(ProtocolError, "missing public span"):
            validate(missing)

        trailing = make_bundle(goal_carrier_kind="external_language_span")
        carrier = next(event for _, event in _events(trailing) if event["kind"] == "GoalCarrier")
        carrier["value"]["public_content"]["trailing_ref"] = "unexpected"
        with self.assertRaisesRegex(ProtocolError, "trailing"):
            validate(trailing)

    def test_missing_or_trailing_demonstration_reference_is_rejected(self) -> None:
        missing = make_bundle()
        carrier = next(event for _, event in _events(missing) if event["kind"] == "GoalCarrier")
        carrier["value"]["public_content"]["event_ids"][-1] = "missing-event"
        with self.assertRaisesRegex(ProtocolError, "missing event"):
            validate(missing)

        trailing = make_bundle()
        carrier = next(event for _, event in _events(trailing) if event["kind"] == "GoalCarrier")
        carrier["value"]["public_content"]["event_ids"].append("target-observation-0")
        with self.assertRaisesRegex(ProtocolError, "complete|content identity|trailing"):
            validate(trailing)

    def test_private_denotation_cannot_change_fixed_carrier_inputs(self) -> None:
        left = make_bundle(goal_carrier_kind="external_language_span", outcome="place-amber-at-bay")
        right = copy.deepcopy(left)
        right["private_receipt"]["goal_denotation"]["outcome"] = "place-amber-at-dock"
        right["private_receipt"]["outcome_denotation"]["predicate"] = "place-amber-at-dock"
        right["supervision"] = derive_target_supervision(right)
        self.assertEqual(learner_input_projection(left), learner_input_projection(right))
        with self.assertRaisesRegex(ProtocolError, "grounding"):
            validate(right)

    def test_private_evaluator_strings_never_enter_public_json(self) -> None:
        bundle = make_bundle()
        public_json = json.dumps(bundle["public_episode"], sort_keys=True)
        private = bundle["private_receipt"]
        self.assertNotIn(private["goal_denotation"]["outcome"], public_json)
        for outcomes in private["evaluator"]["action_outcomes"].values():
            for outcome in outcomes.values():
                self.assertNotIn(outcome, public_json)

    def test_tampered_carrier_or_grounding_receipt_is_rejected(self) -> None:
        tampered_carrier = make_bundle(goal_carrier_kind="external_language_span")
        span = next(event for _, event in _events(tampered_carrier) if event["kind"] == "ContentSpan")
        span["content"] = "a different public instruction"
        with self.assertRaisesRegex(ProtocolError, "content identity"):
            validate(tampered_carrier)

        tampered_grounding = make_bundle()
        tampered_grounding["private_receipt"]["goal_grounding"]["carrier_hash"] = "tampered"
        with self.assertRaisesRegex(ProtocolError, "carrier hash"):
            validate(tampered_grounding)

    def test_exact_evaluator_rejects_caller_supplied_labels(self) -> None:
        bundle = make_bundle()
        bundle["supervision"]["action_queries"][0]["correct_target_actions"] = []
        with self.assertRaisesRegex(ProtocolError, "candidate|exact evaluator"):
            validate(bundle)

    def test_public_carrier_change_is_visible_without_denotation_change(self) -> None:
        first = make_bundle(goal_carrier_kind="external_language_span", public_goal_text="place amber at the bay")
        second = make_bundle(goal_carrier_kind="external_language_span", public_goal_text="please place amber at the marked bay")
        self.assertNotEqual(learner_input_projection(first), learner_input_projection(second))
        self.assertEqual(target_supervision_projection(first), target_supervision_projection(second))


if __name__ == "__main__":
    unittest.main(verbosity=2)
