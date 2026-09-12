import tempfile
import unittest
from pathlib import Path

import torch

from pretraining_experiments.common_content import (
    AdapterSpec,
    CommonContentConfig,
    CommonContentBundle,
    CommonContentLlama,
    ContentAdapterRegistry,
    EmbodimentActionSpec,
    EmbodimentActionDecoder,
    G0ContentAdapter,
    GoalContentAdapter,
    PastActionContentAdapter,
    RawContentStream,
    SensorContentAdapter,
)


def tiny_config() -> CommonContentConfig:
    return CommonContentConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        attention_heads=4,
        max_position_embeddings=32,
        role_vocab_size=8,
        key_vocab_size=16,
    )


def structural(batch: int, tokens: int) -> dict[str, torch.Tensor]:
    return {
        "role_ids": torch.ones(batch, tokens, dtype=torch.long),
        "key_ids": torch.arange(tokens, dtype=torch.long).repeat(batch, 1),
        "time_values": torch.arange(tokens, dtype=torch.float32).repeat(batch, 1),
        "attention_mask": torch.ones(batch, tokens, dtype=torch.bool),
        "content_mask": torch.ones(batch, tokens, dtype=torch.bool),
        "position_ids": torch.arange(tokens, dtype=torch.long).repeat(batch, 1),
    }


class CommonContentTests(unittest.TestCase):
    def test_registry_supports_variable_width_and_token_count(self) -> None:
        torch.manual_seed(3)
        registry = ContentAdapterRegistry(hidden_size=32)
        registry.register("g0", G0ContentAdapter(32))
        registry.register("sensor10", SensorContentAdapter(10, 32))
        registry.register("goal", GoalContentAdapter(6, 32))
        registry.register("past_action", PastActionContentAdapter(7, 32))

        # Both realizations derive from a small public physical state rather
        # than unrelated opaque random vectors.  They still have different
        # widths and token counts at the adapter boundary.
        physical = torch.randn(2, 5, 4)
        g0_values = torch.cat((physical[:, :3], physical[:, :3].square()), dim=-1)
        sensor_values = torch.cat(
            (
                physical,
                physical.square(),
                physical[..., :1] + physical[..., 1:2],
                physical[..., 2:3] - physical[..., 3:4],
            ),
            dim=-1,
        )
        g0 = registry("g0", g0_values)
        sensor = registry("sensor10", sensor_values)
        goal = registry("goal", torch.randn(2, 1, 6))
        action = registry("past_action", torch.randn(2, 4, 7))

        self.assertEqual(tuple(g0.tokens.shape), (2, 3, 32))
        self.assertEqual(tuple(sensor.tokens.shape), (2, 5, 32))
        self.assertEqual(tuple(goal.tokens.shape), (2, 1, 32))
        self.assertEqual(tuple(action.tokens.shape), (2, 4, 32))
        self.assertEqual(g0.content_mask.dtype, torch.bool)

    def test_gradients_reach_two_adapters_core_and_embodiment_decoder(self) -> None:
        torch.manual_seed(4)
        registry = ContentAdapterRegistry(hidden_size=32)
        registry.register("g0", G0ContentAdapter(32))
        registry.register("sensor10", SensorContentAdapter(10, 32))
        core = CommonContentLlama(tiny_config())
        decoder = EmbodimentActionDecoder(32, action_dim=7)

        g0 = registry("g0", torch.randn(2, 3, 8))
        sensor = registry("sensor10", torch.randn(2, 2, 10))
        content = torch.cat((g0.tokens, sensor.tokens), dim=1)
        mask = torch.cat((g0.content_mask, sensor.content_mask), dim=1)
        args = structural(2, 5)
        args["content_mask"] = mask
        hidden = core(content_tokens=content, **args).last_hidden_state
        actions = decoder(hidden)
        loss = actions.square().mean()
        loss.backward()

        self.assertGreater(float(registry.adapters["g0"].projector.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(registry.adapters["sensor10"].projector.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(core.backbone.layers[0].self_attn.q_proj.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(decoder.head[-1].weight.grad.abs().sum()), 0.0)

    def test_causal_prefix_does_not_depend_on_suffix(self) -> None:
        torch.manual_seed(5)
        core = CommonContentLlama(tiny_config()).eval()
        base = torch.randn(1, 6, 32)
        changed = base.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 100
        args = structural(1, 6)
        changed_args = {name: value.clone() for name, value in args.items()}
        changed_args["role_ids"][:, 3:] = 2
        changed_args["key_ids"][:, 3:] = 7
        changed_args["time_values"][:, 3:] = 99.0
        with torch.no_grad():
            left = core(content_tokens=base, **args).last_hidden_state
            right = core(content_tokens=changed, **changed_args).last_hidden_state
        torch.testing.assert_close(left[:, :3], right[:, :3], atol=1e-5, rtol=1e-5)

    def test_padding_is_inert_and_variable_lengths_share_one_core(self) -> None:
        torch.manual_seed(6)
        core = CommonContentLlama(tiny_config()).eval()
        padded = torch.randn(2, 5, 32)
        padded[1, 3:] = 1.0e20
        args = structural(2, 5)
        args["attention_mask"][1, 3:] = False
        args["content_mask"][1, 3:] = False
        with torch.no_grad():
            batched = core(content_tokens=padded, **args).last_hidden_state
            single_args = structural(1, 3)
            single = core(content_tokens=padded[1:2, :3], **single_args).last_hidden_state
        torch.testing.assert_close(batched[1:2, :3], single, atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.equal(batched[1, 3:], torch.zeros_like(batched[1, 3:])))

    def test_core_save_reload_preserves_versioned_boundary(self) -> None:
        torch.manual_seed(7)
        core = CommonContentLlama(tiny_config()).eval()
        content = torch.randn(1, 4, 32)
        args = structural(1, 4)
        with torch.no_grad():
            expected = core(content_tokens=content, **args).last_hidden_state
        with tempfile.TemporaryDirectory() as directory:
            core.save_pretrained(directory)
            restored = CommonContentLlama.from_pretrained(directory).eval()
            self.assertEqual(restored.config.boundary_version, "common-content-1.0")
            with torch.no_grad():
                actual = restored(content_tokens=content, **args).last_hidden_state
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_action_decoder_has_declared_embodiment_width_and_mask(self) -> None:
        decoder = EmbodimentActionDecoder(32, action_dim=7)
        hidden = torch.randn(2, 4, 32)
        mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
        actions = decoder(hidden, mask)
        self.assertEqual(tuple(actions.shape), (2, 4, 7))
        self.assertTrue(torch.equal(actions[0, 2:], torch.zeros_like(actions[0, 2:])))

    def test_bundle_adapts_raw_streams_and_roundtrips_all_trainable_parts(self) -> None:
        torch.manual_seed(8)
        action_spec = EmbodimentActionSpec(
            action_dim=2,
            names=("x_velocity", "y_velocity"),
            units=("m/s", "m/s"),
            coordinate_frame="body",
            semantics=("delta", "delta"),
        )
        bundle = CommonContentBundle.from_parts(
            tiny_config(),
            [
                AdapterSpec("sensor8", "sensor", 8),
                AdapterSpec("goal10", "goal", 10),
                AdapterSpec("action2", "past_action", 2),
            ],
            action_spec,
        )
        streams = [
            RawContentStream(
                "sensor8",
                torch.randn(1, 2, 8),
                torch.tensor([[1, 1]]),
                torch.tensor([[1, 2]]),
                torch.tensor([[0.0, 0.1]]),
                torch.tensor([[0, 1]]),
            ),
            RawContentStream(
                "goal10",
                torch.randn(1, 1, 10),
                torch.tensor([[2]]),
                torch.tensor([[3]]),
                torch.tensor([[0.0]]),
                torch.tensor([[2]]),
            ),
        ]
        targets = torch.zeros(1, 3, 2)
        target_mask = torch.ones_like(targets, dtype=torch.bool)
        output = bundle.forward_streams(
            streams,
            action_mask=torch.ones(1, 3, dtype=torch.bool),
            action_targets=targets,
            action_target_mask=target_mask,
        )
        self.assertEqual(tuple(output.actions.shape), (1, 3, 2))
        self.assertIsNotNone(output.loss)
        output.loss.backward()
        self.assertGreater(float(bundle.adapters.adapters["sensor8"].projector.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(bundle.adapters.adapters["goal10"].projector.weight.grad.abs().sum()), 0.0)

        # The tensor-only route is what a standard Trainer data collator can
        # pass.  Width-8 records occupy only their declared prefix; routing is
        # consumed inside the registered adapters and never reaches the core.
        bundle.zero_grad(set_to_none=True)
        packed = torch.zeros(1, 3, 10)
        packed[:, :2, :8] = streams[0].values
        packed[:, 2:, :10] = streams[1].values
        packed_output = bundle(
            raw_values=packed,
            adapter_ids=torch.tensor([[0, 0, 1]]),
            role_ids=torch.cat((streams[0].role_ids, streams[1].role_ids), dim=1),
            key_ids=torch.cat((streams[0].key_ids, streams[1].key_ids), dim=1),
            time_values=torch.cat((streams[0].time_values, streams[1].time_values), dim=1),
            position_ids=torch.tensor([[0, 1, 2]]),
            attention_mask=torch.ones(1, 3, dtype=torch.bool),
            action_mask=torch.ones(1, 3, dtype=torch.bool),
            action_targets=targets,
            action_target_mask=target_mask,
        )
        self.assertEqual(tuple(packed_output.actions.shape), (1, 3, 2))
        self.assertIsNotNone(packed_output.loss)

        bundle.eval()
        with torch.no_grad():
            expected = bundle.forward_streams(streams).actions
        with tempfile.TemporaryDirectory() as directory:
            bundle.save_pretrained(directory)
            restored = CommonContentBundle.from_pretrained(directory).eval()
            self.assertEqual(restored.action_spec.units, ("m/s", "m/s"))
            with torch.no_grad():
                actual = restored.forward_streams(streams).actions
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
