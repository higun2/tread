"""Correctness, autograd, precision, and DDP tests for RouteSync."""

import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from tread_token_routing.loss import FlowMatchingLoss as BaselineLoss
from tread_token_routing.model import SiT as BaselineSiT
from tread_routesync.loss import FlowMatchingLoss, routesync_relational_loss
from tread_routesync.model import SiT


def tiny_kwargs(depth=7):
    return dict(
        input_size=8, patch_size=2, hidden_size=32, decoder_hidden_size=32,
        depth=depth, num_heads=4, mlp_ratio=2, num_classes=10,
        class_dropout_prob=0.1, qk_norm=False, fused_attn=False,
    )


def route_kwargs(**overrides):
    values = dict(
        use_tread_routing=True, tread_start_block=1, tread_end_block=5,
        tread_active_ratio=0.5, tread_recursive=False, tread_num_groups=2,
        tread_recursive_pattern="grouped", tread_fp32_endpoint=True,
    )
    values.update(overrides)
    return values


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
        torch.manual_seed(40)
        model = SiT(
            **tiny_kwargs(), **route_kwargs(),
            use_routesync=True, routesync_weight=0.2,
        ).train()
        activate(model)
        wrapped = DistributedDataParallel(model, find_unused_parameters=False)
        loss_fn = FlowMatchingLoss(
            use_routesync=True, routesync_weight=0.2,
        )
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = loss_fn(
                wrapped, torch.randn(2, 4, 8, 8),
                {"y": torch.tensor([1, 3])},
            )
        result["total"].backward()
        assert result["relation_pre_mean"].dtype == torch.bfloat16
        assert torch.isfinite(result["total"])
        for name, parameter in wrapped.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
    finally:
        dist.destroy_process_group()


class RouteSyncTests(unittest.TestCase):
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

    def test_disabled_exactly_matches_existing_tread_forward_loss_and_backward(self):
        torch.manual_seed(7)
        baseline = BaselineSiT(**tiny_kwargs(), **route_kwargs()).train()
        torch.manual_seed(7)
        model = SiT(
            **tiny_kwargs(), **route_kwargs(), use_routesync=False,
        ).train()
        self.assertEqual(list(baseline.state_dict()), list(model.state_dict()))
        model.load_state_dict(baseline.state_dict(), strict=True)
        activate(baseline)
        model.load_state_dict(baseline.state_dict(), strict=True)

        torch.manual_seed(91)
        expected = BaselineLoss()(baseline, self.x, {"y": self.y})
        torch.manual_seed(91)
        actual = FlowMatchingLoss(use_routesync=False)(
            model, self.x, {"y": self.y},
        )
        self.assertTrue(torch.equal(expected["total"], actual["total"]))
        self.assertTrue(torch.equal(expected["fm_per_sample"], actual["fm_per_sample"]))
        expected["total"].backward()
        actual["total"].backward()
        for (name, left), (_, right) in zip(
            baseline.named_parameters(), model.named_parameters(),
        ):
            if left.grad is None or right.grad is None:
                self.assertIsNone(left.grad, name)
                self.assertIsNone(right.grad, name)
            else:
                self.assertTrue(torch.equal(left.grad, right.grad), name)

    def test_feature_boundaries_and_original_index_mapping(self):
        model = SiT(
            **tiny_kwargs(), **route_kwargs(), use_routesync=True,
        ).train()
        activate(model)
        captured = {}
        before = model.blocks[0].register_forward_hook(
            lambda module, inputs, output: captured.update(h_pre=output)
        )
        after = model.blocks[5].register_forward_hook(
            lambda module, inputs, output: captured.update(h_post=output)
        )
        output, features, info = model(
            self.x, self.t, self.y, return_routing_info=True,
        )
        before.remove()
        after.remove()
        self.assertEqual(output.shape, self.x.shape)
        self.assertIs(features, info["routesync_features"])
        self.assertIs(features["h_pre"], captured["h_pre"])
        self.assertIs(features["h_post"], captured["h_post"])
        self.assertTrue(torch.equal(features["routed_idx"], info["bypass_indices"]))
        self.assertTrue(torch.equal(features["processed_idx"], info["active_indices"]))
        self.assertEqual(features["h_pre"].shape, (2, 16, 32))
        self.assertEqual(features["h_post"].shape, (2, 16, 32))
        features["h_pre"].retain_grad()
        features["h_post"].retain_grad()
        route_loss, _ = routesync_relational_loss(
            features["h_pre"], features["h_post"],
            features["routed_idx"], features["processed_idx"],
            debug=True,
        )
        route_loss.backward()
        self.assertIsNotNone(features["h_pre"].grad)
        self.assertGreater(features["h_pre"].grad.abs().sum(), 0)
        self.assertIsNone(features["h_post"].grad)

    def test_relation_formula_partition_stop_gradient_and_pre_gradient(self):
        h_pre = torch.randn(2, 6, 8, requires_grad=True)
        h_post = torch.randn(2, 6, 8, requires_grad=True)
        routed = torch.tensor([[5, 1, 3], [0, 4, 2]])
        processed = torch.tensor([[0, 2, 4], [1, 3, 5]])
        loss, stats = routesync_relational_loss(
            h_pre, h_post, routed, processed, debug=True,
        )
        expected_pre = torch.bmm(
            nn.functional.normalize(h_pre.gather(1, routed[..., None].expand(-1, -1, 8)), dim=-1),
            nn.functional.normalize(h_pre.gather(1, processed[..., None].expand(-1, -1, 8)), dim=-1).transpose(1, 2),
        )
        expected_post = torch.bmm(
            nn.functional.normalize(h_post.gather(1, routed[..., None].expand(-1, -1, 8)), dim=-1),
            nn.functional.normalize(h_post.gather(1, processed[..., None].expand(-1, -1, 8)), dim=-1).transpose(1, 2),
        )
        torch.testing.assert_close(
            loss, nn.functional.l1_loss(expected_pre, expected_post.detach()),
        )
        pre_grad, post_grad = torch.autograd.grad(
            loss, (h_pre, h_post), allow_unused=True,
        )
        self.assertIsNotNone(pre_grad)
        self.assertGreater(pre_grad.abs().sum(), 0)
        self.assertIsNone(post_grad)
        self.assertTrue(torch.equal(stats["relation_abs_diff_mean"], loss.detach()))

    def test_total_loss_and_recursive_backward(self):
        model = SiT(
            **tiny_kwargs(),
            **route_kwargs(tread_recursive=True),
            use_routesync=True, routesync_weight=0.25,
        ).train()
        activate(model)
        result = FlowMatchingLoss(
            use_routesync=True, routesync_weight=0.25,
            routesync_debug=True,
        )(model, self.x, {"y": self.y})
        torch.testing.assert_close(
            result["total"],
            result["fm"] + 0.25 * result["route_sync"],
        )
        self.assertGreater(result["route_sync"], 0)
        result["total"].backward()
        groups = (model.prefix_blocks, model.recurrent_group_blocks, model.suffix_blocks)
        for blocks in groups:
            for block in blocks:
                for name, parameter in block.named_parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_recursive_logical_depth_embedding_changes_condition_and_gets_gradient(self):
        model = SiT(
            **tiny_kwargs(),
            **route_kwargs(tread_recursive=True, tread_depth_embedding=True),
            use_routesync=False,
        ).train()
        activate(model)
        self.assertEqual(tuple(model.tread_depth_embeddings.shape), (4, 32))
        with torch.no_grad():
            for offset in range(4):
                model.tread_depth_embeddings[offset].fill_(float(offset))

        conditions = []
        hook = model.recurrent_group_blocks[0].register_forward_hook(
            lambda module, inputs, output: conditions.append(inputs[1].detach().clone())
        )
        output, _ = model(self.x, self.t, self.y)
        hook.remove()

        self.assertEqual(len(conditions), 2)
        torch.testing.assert_close(
            conditions[1] - conditions[0], torch.ones_like(conditions[0]),
        )
        output.sum().backward()
        gradient = model.tread_depth_embeddings.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient.abs().sum(dim=1) > 0).all())

    def test_depth_embedding_requires_recursive(self):
        with self.assertRaisesRegex(ValueError, "requires --tread-recursive"):
            SiT(
                **tiny_kwargs(),
                **route_kwargs(tread_depth_embedding=True),
                use_routesync=False,
            )

    def test_bf16_relation_does_not_change_model_or_optimizer_dtype(self):
        model = SiT(
            **tiny_kwargs(), **route_kwargs(tread_fp32_endpoint=False),
            use_routesync=True, routesync_weight=0.1,
        ).train()
        activate(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        parameter_dtypes = {name: p.dtype for name, p in model.named_parameters()}
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = FlowMatchingLoss(
                use_routesync=True, routesync_weight=0.1,
            )(model, self.x, {"y": self.y})
        self.assertEqual(result["relation_pre_mean"].dtype, torch.bfloat16)
        self.assertEqual(result["relation_post_mean"].dtype, torch.bfloat16)
        result["total"].backward()
        optimizer.step()
        self.assertEqual(
            parameter_dtypes, {name: p.dtype for name, p in model.named_parameters()},
        )
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and value.ndim:
                    self.assertEqual(value.dtype, torch.float32)

    def test_default_routesync_follows_bf16_autocast(self):
        h_pre = torch.randn(2, 6, 8, requires_grad=True)
        h_post = torch.randn(2, 6, 8, requires_grad=True)
        routed = torch.tensor([[5, 1, 3], [0, 4, 2]])
        processed = torch.tensor([[0, 2, 4], [1, 3, 5]])
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, stats = routesync_relational_loss(
                h_pre, h_post, routed, processed,
            )
        self.assertEqual(stats["relation_pre_mean"].dtype, torch.bfloat16)
        self.assertEqual(stats["relation_post_mean"].dtype, torch.bfloat16)
        loss.backward()
        self.assertIsNotNone(h_pre.grad)
        self.assertIsNone(h_post.grad)

    def test_eval_has_no_routesync_features_in_sparse_or_dense_mode(self):
        model = SiT(
            **tiny_kwargs(), **route_kwargs(), use_routesync=True,
        ).eval()
        with torch.no_grad():
            sparse = model(self.x, self.t, self.y)
            dense = model(self.x, self.t, self.y, tread_eval_mode="dense")
        self.assertIsNone(sparse[1])
        self.assertIsNone(dense[1])

    def test_invalid_routesync_boundaries_are_rejected(self):
        with self.assertRaises(ValueError):
            SiT(
                **tiny_kwargs(),
                **route_kwargs(tread_end_block=7),
                use_routesync=True,
            )
        with self.assertRaises(ValueError):
            SiT(
                **tiny_kwargs(),
                **route_kwargs(tread_active_ratio=1.0),
                use_routesync=True,
            )

    def test_two_process_ddp_bfloat16_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            store = str(Path(directory) / "store")
            mp.start_processes(
                ddp_worker, args=(store,), nprocs=2, join=True,
                start_method="spawn",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA BF16 test")
    def test_cuda_bfloat16_routesync_backward(self):
        x, t, y = self.x.cuda(), self.t.cuda(), self.y.cuda()
        for recursive in (False, True):
            kwargs = tiny_kwargs()
            kwargs["fused_attn"] = True
            model = SiT(
                **kwargs, **route_kwargs(tread_recursive=recursive),
                use_routesync=True, routesync_weight=0.2,
            ).cuda().train()
            activate(model)
            parameter_dtypes = {
                name: parameter.dtype
                for name, parameter in model.named_parameters()
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = FlowMatchingLoss(
                    use_routesync=True, routesync_weight=0.2,
                )(model, x, {"y": y})
            self.assertEqual(result["relation_pre_mean"].dtype, torch.bfloat16)
            self.assertEqual(result["relation_post_mean"].dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(result["total"]))
            result["total"].backward()
            blocks = (
                list(model.prefix_blocks)
                + list(model.recurrent_group_blocks)
                + list(model.suffix_blocks)
                if recursive else list(model.blocks)
            )
            for index, block in enumerate(blocks):
                gradients = [
                    parameter.grad for parameter in block.parameters()
                    if parameter.requires_grad
                ]
                self.assertTrue(all(
                    gradient is not None and torch.isfinite(gradient).all()
                    for gradient in gradients
                ), (recursive, index))
            self.assertEqual(parameter_dtypes, {
                name: parameter.dtype
                for name, parameter in model.named_parameters()
            })


if __name__ == "__main__":
    unittest.main()
