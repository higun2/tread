"""Correctness and backward tests for fixed-subset TREAD-style routing."""

import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from models.sit_b1 import SiT as VanillaSiT
from tread_token_routing.generate import (
    parse_args as parse_generate_args,
    resolve_tread_inference_config,
)
from tread_token_routing.loss import FlowMatchingLoss
from tread_token_routing.model import (
    SiT, gather_tokens, random_subset_indices, restore_token_order,
)
from tread_token_routing.train import generate_training_preview


def tiny_kwargs(depth=6):
    return dict(
        input_size=8, patch_size=2, hidden_size=32, decoder_hidden_size=32,
        depth=depth, num_heads=4, mlp_ratio=2, num_classes=10,
        class_dropout_prob=0.1, qk_norm=False, fused_attn=False,
    )


def routed_kwargs(
    eval_mode="sparse", seed=None, fp32_endpoint=True, recursive=False,
    recursive_pattern="grouped",
):
    return dict(
        use_tread_routing=True, tread_start_block=0, tread_end_block=6,
        tread_active_ratio=0.5, tread_eval_mode=eval_mode,
        tread_recursive=recursive, tread_num_groups=3,
        tread_recursive_pattern=recursive_pattern,
        tread_fp32_endpoint=fp32_endpoint, tread_seed=seed,
    )


def activate(model):
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "adaLN_modulation"):
                nn.init.normal_(module.adaLN_modulation[-1].weight, std=0.02)
                nn.init.normal_(module.adaLN_modulation[-1].bias, std=0.02)
        nn.init.normal_(model.final_layer.linear.weight, std=0.02)


def ddp_worker(rank, store):
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{store}", rank=rank, world_size=2,
    )
    try:
        torch.manual_seed(100 + rank)
        for recursive, recursive_pattern in (
            (False, "grouped"), (True, "grouped"), (True, "interleaved"),
        ):
            model = SiT(
                **tiny_kwargs(),
                **routed_kwargs(
                    recursive=recursive, recursive_pattern=recursive_pattern,
                ),
            )
            activate(model)
            wrapped = DistributedDataParallel(model, find_unused_parameters=False)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = FlowMatchingLoss()(
                    wrapped, torch.randn(2, 4, 8, 8),
                    {"y": torch.tensor([1, 3])},
                )["total"]
            loss.backward()
            for name, parameter in wrapped.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None, name
                    assert torch.isfinite(parameter.grad).all(), name
                    synced = parameter.grad.clone()
                    dist.broadcast(synced, src=0)
                    torch.testing.assert_close(
                        synced, parameter.grad, rtol=0, atol=0,
                    )
    finally:
        dist.destroy_process_group()


class DummyVAE:
    def decode(self, latents):
        return SimpleNamespace(sample=latents[:, :3])


class TreadRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(17)
        self.x = torch.randn(2, 4, 8, 8)
        self.t = torch.tensor([0.2, 0.7])
        self.y = torch.tensor([1, 3])

    def test_random_subset_is_exact_disjoint_partition(self):
        generator = torch.Generator().manual_seed(123)
        active, bypass = random_subset_indices(
            2, 1024, 512, torch.device("cpu"), generator,
        )
        self.assertEqual(active.shape, (2, 512))
        self.assertEqual(bypass.shape, (2, 512))
        union = torch.cat((active, bypass), dim=1).sort(dim=1).values
        self.assertTrue(torch.equal(union, torch.arange(1024).expand(2, -1)))
        self.assertFalse((active.unsqueeze(2) == bypass.unsqueeze(1)).any())

    def test_disabled_matches_baseline_forward_backward_and_state_dict(self):
        torch.manual_seed(7)
        baseline = VanillaSiT(**tiny_kwargs())
        torch.manual_seed(7)
        model = SiT(**tiny_kwargs(), use_tread_routing=False)
        self.assertEqual(list(baseline.state_dict()), list(model.state_dict()))
        activate(baseline)
        model.load_state_dict(baseline.state_dict(), strict=True)
        baseline.train()
        model.train()
        torch.manual_seed(31)
        expected = baseline(self.x, self.t, self.y)[0]
        torch.manual_seed(31)
        actual = model(self.x, self.t, self.y)[0]
        self.assertTrue(torch.equal(expected, actual))
        expected.square().mean().backward()
        actual.square().mean().backward()
        for (name, left), (_, right) in zip(
            baseline.named_parameters(), model.named_parameters(),
        ):
            if left.grad is None or right.grad is None:
                self.assertIsNone(left.grad, name)
                self.assertIsNone(right.grad, name)
            else:
                self.assertTrue(torch.equal(left.grad, right.grad), name)

    def test_sparse_uses_one_fixed_half_subset_for_all_routed_blocks(self):
        model = SiT(**tiny_kwargs(), **routed_kwargs()).train()
        calls = []
        handles = [
            block.register_forward_pre_hook(
                lambda module, inputs, index=index: calls.append(
                    (index, inputs[0].shape)
                )
            )
            for index, block in enumerate(model.blocks)
        ]
        output, _, info = model(
            self.x, self.t, self.y, return_routing_info=True,
        )
        for handle in handles:
            handle.remove()
        self.assertEqual(output.shape, self.x.shape)
        self.assertEqual([shape[1] for _, shape in calls], [8] * 6)
        self.assertEqual(info["block_input_token_counts"], [8] * 6)
        self.assertEqual(info["post_block_token_counts"], [16] * 6)
        self.assertEqual(info["active_indices"].shape, (2, 8))
        self.assertEqual(info["bypass_indices"].shape, (2, 8))
        union = torch.cat(
            (info["active_indices"], info["bypass_indices"]), dim=1,
        ).sort(dim=1).values
        self.assertTrue(torch.equal(union, torch.arange(16).expand(2, -1)))
        self.assertNotIn("core_indices", info)

    def test_recursive_sparse_is_a1a1_b1b1_c1c1(self):
        model = SiT(
            **tiny_kwargs(), **routed_kwargs(recursive=True),
        ).train()
        activate(model)
        calls = []
        handles = [
            block.register_forward_pre_hook(
                lambda module, inputs, group=group: calls.append(
                    (group, id(module), inputs[0].shape)
                )
            )
            for group, block in enumerate(model.recurrent_group_blocks)
        ]
        output, _, info = model(
            self.x, self.t, self.y, return_routing_info=True,
        )
        for handle in handles:
            handle.remove()
        self.assertTrue(info["recursive"])
        self.assertEqual([group for group, _, _ in calls], [0, 0, 1, 1, 2, 2])
        self.assertEqual([shape[1] for _, _, shape in calls], [8] * 6)
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(calls[2][1], calls[3][1])
        self.assertEqual(calls[4][1], calls[5][1])
        self.assertEqual(len({calls[0][1], calls[2][1], calls[4][1]}), 3)
        self.assertFalse(hasattr(model, "blocks"))
        ordinary = SiT(**tiny_kwargs(), **routed_kwargs())
        self.assertLess(
            sum(parameter.numel() for parameter in model.parameters()),
            sum(parameter.numel() for parameter in ordinary.parameters()),
        )
        output.square().mean().backward()
        for group, block in enumerate(model.recurrent_group_blocks):
            for name, parameter in block.named_parameters():
                self.assertIsNotNone(parameter.grad, (group, name))
                self.assertTrue(torch.isfinite(parameter.grad).all(), (group, name))

    def test_recursive_dense_repeats_groups_on_all_tokens_and_strict_loads(self):
        model = SiT(
            **tiny_kwargs(),
            **routed_kwargs(eval_mode="dense", recursive=True),
        ).eval()
        calls = []
        handles = [
            block.register_forward_pre_hook(
                lambda module, inputs, group=group: calls.append(
                    (group, inputs[0].shape)
                )
            )
            for group, block in enumerate(model.recurrent_group_blocks)
        ]
        _, _, info = model(
            self.x, self.t, self.y, return_routing_info=True,
        )
        for handle in handles:
            handle.remove()
        self.assertFalse(info["routing_applied"])
        self.assertTrue(info["recursive"])
        self.assertEqual([group for group, _ in calls], [0, 0, 1, 1, 2, 2])
        self.assertEqual([shape[1] for _, shape in calls], [16] * 6)

        target = SiT(
            **tiny_kwargs(), **routed_kwargs(recursive=True),
        )
        incompatible = target.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        keys = list(model.state_dict())
        self.assertTrue(any(key.startswith("recurrent_group_blocks.") for key in keys))
        self.assertFalse(any(key.startswith("blocks.") for key in keys))

    def test_recursive_interleaved_is_a1b1c1_a1b1c1_sparse_and_dense(self):
        for eval_mode, expected_tokens in (("sparse", 8), ("dense", 16)):
            model = SiT(
                **tiny_kwargs(),
                **routed_kwargs(
                    eval_mode=eval_mode, recursive=True,
                    recursive_pattern="interleaved",
                ),
            )
            model.train(eval_mode == "sparse")
            calls = []
            handles = [
                block.register_forward_pre_hook(
                    lambda module, inputs, group=group: calls.append(
                        (group, id(module), inputs[0].shape[1])
                    )
                )
                for group, block in enumerate(model.recurrent_group_blocks)
            ]
            _, _, info = model(
                self.x, self.t, self.y, return_routing_info=True,
            )
            for handle in handles:
                handle.remove()
            self.assertEqual(
                [group for group, _, _ in calls], [0, 1, 2, 0, 1, 2],
            )
            self.assertEqual([tokens for _, _, tokens in calls], [expected_tokens] * 6)
            self.assertEqual(calls[0][1], calls[3][1])
            self.assertEqual(calls[1][1], calls[4][1])
            self.assertEqual(calls[2][1], calls[5][1])
            self.assertEqual(info["recursive"], True)

    def test_recursive_final_shared_call_is_fp32_endpoint(self):
        model = SiT(
            **tiny_kwargs(), **routed_kwargs(recursive=True),
        ).train()
        activate(model)
        dtypes = []
        handles = [
            block.attn.qkv.register_forward_hook(
                lambda module, inputs, output: dtypes.append(output.dtype)
            )
            for block in model.recurrent_group_blocks
        ]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(self.x, self.t, self.y)
        for handle in handles:
            handle.remove()
        self.assertEqual(dtypes, [torch.bfloat16] * 5 + [torch.float32])

    def test_sparse_output_matches_full_state_scatter_reference(self):
        model = SiT(**tiny_kwargs(), **routed_kwargs(seed=44)).eval()
        activate(model)
        with torch.no_grad():
            actual, _, info = model(
                self.x, self.t, self.y, return_routing_info=True,
            )
            full = model.x_embedder(self.x) + model.pos_embed
            condition = model.t_embedder(self.t) + model.y_embedder(self.y, False)
            indices = info["active_indices"]
            for block in model.blocks:
                active = block(gather_tokens(full, indices), condition)
                full = full.scatter(
                    1, indices.unsqueeze(-1).expand_as(active), active,
                )
            expected = model.unpatchify(model.final_layer(full, condition))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_training_resamples_and_seeded_eval_is_local_repeatable(self):
        model = SiT(**tiny_kwargs(), **routed_kwargs()).train()
        first = model(self.x, self.t, self.y, return_routing_info=True)[2]
        second = model(self.x, self.t, self.y, return_routing_info=True)[2]
        self.assertFalse(torch.equal(first["active_indices"], second["active_indices"]))

        model = SiT(**tiny_kwargs(), **routed_kwargs(seed=123)).eval()
        global_state = torch.get_rng_state().clone()
        first = model(self.x, self.t, self.y, return_routing_info=True)[2]
        self.assertTrue(torch.equal(global_state, torch.get_rng_state()))
        second = model(self.x, self.t, self.y, return_routing_info=True)[2]
        self.assertTrue(torch.equal(first["active_indices"], second["active_indices"]))

    def test_sparse_and_dense_eval(self):
        model = SiT(**tiny_kwargs(), **routed_kwargs()).eval()
        sparse_shapes = []
        handles = [
            block.register_forward_pre_hook(
                lambda module, inputs: sparse_shapes.append(inputs[0].shape)
            )
            for block in model.blocks
        ]
        _, _, sparse = model(
            self.x, self.t, self.y, return_routing_info=True,
        )
        for handle in handles:
            handle.remove()
        self.assertTrue(sparse["routing_applied"])
        self.assertEqual([shape[1] for shape in sparse_shapes], [8] * 6)

        dense_shapes = []
        handles = [
            block.register_forward_pre_hook(
                lambda module, inputs: dense_shapes.append(inputs[0].shape)
            )
            for block in model.blocks
        ]
        _, _, dense = model(
            self.x, self.t, self.y, tread_eval_mode="dense",
            return_routing_info=True,
        )
        for handle in handles:
            handle.remove()
        self.assertFalse(dense["routing_applied"])
        self.assertEqual([shape[1] for shape in dense_shapes], [16] * 6)

    def test_only_final_routed_block_is_fp32_during_training(self):
        routing = routed_kwargs()
        routing.update(tread_start_block=1, tread_end_block=7)
        model = SiT(**tiny_kwargs(depth=9), **routing).train()
        activate(model)
        records = []
        handles = [
            block.attn.qkv.register_forward_hook(
                lambda module, inputs, output: records.append(
                    (output.dtype, torch.is_autocast_enabled("cpu"))
                )
            )
            for block in model.blocks
        ]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, _, info = model(
                self.x, self.t, self.y, return_routing_info=True,
            )
        self.assertTrue(info["fp32_endpoint_applied"])
        self.assertEqual(
            [dtype for dtype, _ in records],
            [torch.bfloat16] * 6 + [torch.float32] + [torch.bfloat16] * 2,
        )
        self.assertEqual(
            [enabled for _, enabled in records],
            [True] * 6 + [False] + [True] * 2,
        )
        records.clear()
        model.eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            model(self.x, self.t, self.y)
        self.assertEqual([dtype for dtype, _ in records], [torch.bfloat16] * 9)
        for handle in handles:
            handle.remove()

    def test_backward_and_optimizer_dtypes(self):
        model = SiT(
            **tiny_kwargs(), **routed_kwargs(fp32_endpoint=False),
        ).train()
        activate(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = FlowMatchingLoss()(model, self.x, {"y": self.y})["total"]
            loss.backward()
            for block in model.blocks:
                self.assertTrue(all(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                    for parameter in block.parameters() if parameter.requires_grad
                ))
            optimizer.step()
            return loss

        self.assertTrue(torch.isfinite(step()))
        parameter_dtypes = {
            name: parameter.dtype for name, parameter in model.named_parameters()
        }
        state_dtypes = {
            (id(parameter), name): value.dtype
            for parameter, state in optimizer.state.items()
            for name, value in state.items() if torch.is_tensor(value)
        }
        model.tread_fp32_endpoint = True
        self.assertTrue(torch.isfinite(step()))
        self.assertEqual(
            parameter_dtypes,
            {name: parameter.dtype for name, parameter in model.named_parameters()},
        )
        self.assertEqual(state_dtypes, {
            (id(parameter), name): value.dtype
            for parameter, state in optimizer.state.items()
            for name, value in state.items() if torch.is_tensor(value)
        })

    def test_restore_is_out_of_place_and_differentiable(self):
        permutation = torch.tensor([
            [3, 1, 6, 4, 0, 7, 2, 5],
            [5, 0, 3, 7, 2, 6, 1, 4],
        ])
        grouped = torch.randn(2, 8, 3, dtype=torch.double, requires_grad=True)
        before = grouped.detach().clone()
        self.assertTrue(torch.autograd.gradcheck(
            lambda value: restore_token_order(value, permutation), (grouped,),
        ))
        restored = restore_token_order(grouped, permutation)
        self.assertTrue(torch.equal(grouped, before))
        restored.sum().backward()
        self.assertTrue(torch.equal(grouped.grad, torch.ones_like(grouped)))

    def test_training_preview_forces_dense_and_restores_mode(self):
        model = SiT(
            **tiny_kwargs(), **routed_kwargs(eval_mode="sparse"),
        ).eval()
        observed = []

        def fake_sampler(model, latents, y, **kwargs):
            observed.append(model.tread_eval_mode)
            return latents

        with patch("samplers.euler_sampler", side_effect=fake_sampler):
            images = generate_training_preview(
                model, DummyVAE(), torch.randn(2, 4, 8, 8), self.y,
                path_type="linear", cfg_scale=1.0,
            )
        self.assertEqual(images.shape, (2, 3, 8, 8))
        self.assertEqual(observed, ["dense"])
        self.assertEqual(model.tread_eval_mode, "sparse")

    def test_generate_config_and_cli_overrides(self):
        trained = dict(
            use_tread_routing=True, tread_start_block=0, tread_end_block=6,
            tread_active_ratio=0.5, tread_eval_mode="sparse",
            tread_recursive=True, tread_num_groups=3,
            tread_recursive_pattern="interleaved",
            tread_fp32_endpoint=True, tread_seed=None, tread_debug=False,
        )
        config, saved = resolve_tread_inference_config(
            {"args": trained}, "dense", 77,
        )
        self.assertEqual(saved, "sparse")
        self.assertEqual(config["tread_eval_mode"], "dense")
        self.assertEqual(config["tread_seed"], 77)
        self.assertTrue(config["tread_recursive"])
        self.assertEqual(config["tread_recursive_pattern"], "interleaved")
        args = parse_generate_args([
            "--ckpt", "dummy.pt", "--tread-eval-mode", "dense",
        ])
        self.assertEqual(args.tread_eval_mode, "dense")
        with self.assertRaises(ValueError):
            resolve_tread_inference_config({"args": {}}, "sparse", None)

    def test_two_process_ddp_bfloat16_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            store = str(Path(directory) / "store")
            mp.start_processes(
                ddp_worker, args=(store,), nprocs=2, join=True,
                start_method="spawn",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA BF16 test")
    def test_cuda_bfloat16_backward(self):
        x, t, y = self.x.cuda(), self.t.cuda(), self.y.cuda()
        for fused_attn in (False, True):
            for recursive, recursive_pattern in (
                (False, "grouped"), (True, "grouped"), (True, "interleaved"),
            ):
                kwargs = tiny_kwargs()
                kwargs["fused_attn"] = fused_attn
                model = SiT(
                    **kwargs,
                    **routed_kwargs(
                        recursive=recursive,
                        recursive_pattern=recursive_pattern,
                    ),
                ).cuda().train()
                activate(model)
                blocks = (
                    model.recurrent_group_blocks if recursive else model.blocks
                )
                dtypes = []
                handles = [
                    block.attn.qkv.register_forward_hook(
                        lambda module, inputs, output: dtypes.append(output.dtype)
                    )
                    for block in blocks
                ]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = FlowMatchingLoss()(model, x, {"y": y})["total"]
                for handle in handles:
                    handle.remove()
                self.assertEqual(
                    dtypes, [torch.bfloat16] * 5 + [torch.float32],
                    (recursive, recursive_pattern),
                )
                loss.backward()
                for index, block in enumerate(blocks):
                    gradients = [p.grad for p in block.parameters() if p.requires_grad]
                    self.assertTrue(all(
                        gradient is not None and torch.isfinite(gradient).all()
                        for gradient in gradients
                    ), (fused_attn, recursive, recursive_pattern, index))


if __name__ == "__main__":
    unittest.main()
