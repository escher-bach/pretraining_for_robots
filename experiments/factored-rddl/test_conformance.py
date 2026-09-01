"""Focused CPU conformance for the RDDL-to-prompted-interface adapter."""

from __future__ import annotations

import copy
import json
import unittest

import conformance


class FactoredRddlPromptedInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = conformance.build_artifact()

    def test_existing_permutation_relation_edit_and_hidden_controls_remain(self) -> None:
        conformance.assert_preserving_permutation()
        conformance.assert_meaning_changing_relation_edit()
        conformance.assert_hidden_boundary()
        conformance.assert_exhaustive_transform_matrix()

    def test_noop_does_not_fire_or_apply_the_reassignment_boundary(self) -> None:
        env = conformance.environment(conformance.BASE)
        step = conformance.replay(env, [{}])[0]
        self.assertFalse(step["public_observation"]["boundary"])
        self.assertTrue(step["privileged_state"]["assigned___s0__c0"])
        self.assertTrue(step["privileged_state"]["assigned___s1__c1"])

    def test_public_episodes_are_addressed_and_private_free(self) -> None:
        for bundle in self.artifact["rendered_query_episodes"]:
            conformance.validate_rendered_episode(bundle)
            public = bundle["public_episode"]
            self.assertEqual(public["protocol_version"], conformance.PROTOCOL_VERSION)
            self.assertTrue(public["public_contract"]["source_embodiment"])
            self.assertNotIn("GoalPublication", [event["kind"] for _, event in conformance._events(public)])
            self.assertTrue(all(step["reward"] is None for step in public["steps"]))
            public_json = json.dumps(public, sort_keys=True)
            self.assertNotIn("source-value", public_json)
            self.assertNotIn('"assignments"', public_json)

    def test_action_refs_are_target_scoped_and_executable(self) -> None:
        for bundle in self.artifact["rendered_query_episodes"]:
            adapter = bundle["public_episode"]["public_contract"]["action_adapters"]["target"]
            for _, event in conformance._events(bundle["public_episode"]):
                if event["kind"] == "ActionQuery":
                    ports = event["candidates"]
                elif event["kind"] == "ActionExecuted":
                    ports = [event["port"]]
                else:
                    continue
                for port in ports:
                    self.assertEqual(port["actor"], conformance.TARGET_ACTOR)
                    self.assertEqual(port["embodiment_id"], conformance.TARGET_EMBODIMENT)
                    self.assertEqual(port["namespace"], "actuator")
                    self.assertIn(port["local_id"], adapter["executable"])

    def test_supervision_and_grounding_are_exhaustively_derived(self) -> None:
        for bundle in self.artifact["rendered_query_episodes"]:
            self.assertEqual(bundle["supervision"], conformance.derive_target_supervision(bundle))
            tampered = copy.deepcopy(bundle)
            tampered["private_receipt"]["goal_grounding"]["grounded_target_actions"] = []
            with self.assertRaisesRegex(AssertionError, "goal-grounding"):
                conformance.validate_rendered_episode(tampered)

    def test_goal_publication_contract_must_bind_the_public_carrier_and_private_receipt(self) -> None:
        bundle = copy.deepcopy(self.artifact["rendered_query_episodes"][0])
        carrier = next(
            event
            for _, event in conformance._events(bundle["public_episode"])
            if event["kind"] == "GoalCarrier"
        )
        bundle["public_episode"]["public_contract"]["goal_publication"]["carrier_event_id"] = "wrong-carrier-event"
        with self.assertRaisesRegex(AssertionError, "GoalPublication"):
            conformance.validate_rendered_episode(bundle)

        self.assertEqual(carrier["value"]["public_content"].keys(), {"predicate", "content_id"})

        stale_receipt = copy.deepcopy(self.artifact["rendered_query_episodes"][0])
        stale_receipt["private_receipt"]["goal_grounding"]["publication_id"] = "wrong-publication"
        with self.assertRaisesRegex(AssertionError, "goal-grounding"):
            conformance.validate_rendered_episode(stale_receipt)

    def test_closed_public_event_fields_reject_secret_injection(self) -> None:
        bundle = copy.deepcopy(self.artifact["rendered_query_episodes"][0])
        observation = next(
            event
            for _, event in conformance._events(bundle["public_episode"])
            if event["kind"] == "Observation"
        )
        observation["secret_assignment"] = "s0-at-c0"
        with self.assertRaisesRegex(AssertionError, "field schema is closed"):
            conformance.validate_rendered_episode(bundle)

    def test_artifact_replay_rejects_self_consistent_outcome_tampering(self) -> None:
        artifact = copy.deepcopy(self.artifact)
        bundle = artifact["rendered_query_episodes"][0]
        query = bundle["supervision"]["action_queries"][0]["query_event_id"]
        outcomes = bundle["private_receipt"]["evaluator"]["query_action_outcomes"][query]
        outcomes["noop"]["continuations"][0]["cumulative_return"] = 99.0
        outcomes["noop"]["best_cumulative_return"] = 99.0
        outcomes["noop"]["goal_satisfying"] = True
        for action, outcome in outcomes.items():
            if action != "noop":
                outcome["goal_satisfying"] = False
        bundle["supervision"] = conformance.derive_target_supervision(bundle)
        grounding = bundle["private_receipt"]["goal_grounding"]
        grounding["best_cumulative_return"] = 99.0
        grounding["grounded_target_actions"] = ["noop"]
        grounding["outcome_sha256"] = conformance.digest(outcomes)
        conformance.validate_rendered_episode(bundle)
        with self.assertRaisesRegex(AssertionError, "query outcomes"):
            conformance.validate_artifact(artifact)

    def test_boolean_surface_is_complete_under_the_model_constraint(self) -> None:
        env = conformance.environment(conformance.BASE)
        conformance.assert_complete_candidate_surface(env)
        assignments = conformance.all_boolean_action_assignments(env)
        self.assertEqual(len(assignments), 4)
        self.assertEqual([conformance.action_name(action) for action in conformance.candidate_actions(env)], ["noop", "pulse___c0", "pulse___c1"])

    def test_reachable_graph_includes_legality_reward_terminal_and_replay_evidence(self) -> None:
        private = self.artifact["private_graph_receipt"]
        graph = private["canonical_finite_graph"]
        self.assertEqual(graph["scope"], "reachable-privileged-states-and-legal-actions-through-fixed-rddl-horizon")
        self.assertEqual(private["replay_equivalence"]["complete_traces_checked"], 9)
        self.assertTrue(private["replay_equivalence"]["all_replays_match"])
        self.assertEqual(private["goal_semantics"]["denotation_id"], "rddl-maximize-cumulative-return")
        self.assertTrue(all("assignments" in node for node in graph["nodes"]))
        self.assertTrue(all("reward" in edge and "truncated" in edge for edge in graph["edges"]))
        self.assertEqual(private["replay_equivalence"], conformance.replay_equivalence(graph, private["complete_privileged_traces"]))

    def test_artifact_is_canonical_and_reachable_source_values_are_bounded_by_horizon(self) -> None:
        self.assertEqual(self.artifact, conformance.build_artifact())
        values = [
            value
            for trace in self.artifact["private_graph_receipt"]["complete_privileged_traces"]
            for step in trace["steps"]
            for key, value in step["privileged_state"].items()
            if key.startswith("source-value")
        ]
        self.assertTrue(values)
        self.assertTrue(all(0 <= value <= 2 for value in values))


if __name__ == "__main__":
    unittest.main(verbosity=2)
