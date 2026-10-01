"""Subset alignment formula, RNG behavior, stopped target and integration."""
import argparse
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from .. import loss as loss_module
from ..config import add_routesync_args, tread_kwargs
from ..loss import FlowMatchingLoss, routesync_relational_loss, sample_group_indices
from ..model import SiT


class SubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(9)
        self.pre = torch.randn(2, 12, 8, dtype=torch.float64, requires_grad=True)
        self.post = torch.randn_like(self.pre, requires_grad=True)
        self.r = torch.tensor([[0, 2, 4, 6, 8, 10]]).expand(2, -1)
        self.p = torch.tensor([[1, 3, 5, 7, 9, 11]]).expand(2, -1)

    def reference(self, r, p):
        def relation(h):
            h = nn.functional.normalize(h, dim=-1)
            a = h.gather(1, r[..., None].expand(-1, -1, 8))
            b = h.gather(1, p[..., None].expand(-1, -1, 8))
            return a @ b.transpose(1, 2)
        return nn.functional.l1_loss(relation(self.pre), relation(self.post).detach())

    def test_full_ratio_exact_formula_gradients_and_no_rng_consumption(self):
        state = torch.get_rng_state()
        actual, stats = routesync_relational_loss(self.pre, self.post, self.r, self.p, debug=True)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        expected = self.reference(self.r, self.p)
        self.assertTrue(torch.equal(actual, expected))
        actual_grad, post_grad = torch.autograd.grad(actual, (self.pre, self.post), allow_unused=True)
        reference_grad, = torch.autograd.grad(expected, (self.pre,))
        torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=0)
        self.assertIsNone(post_grad)
        self.assertEqual(stats['sampled_r_tokens'], 6)

    def test_subset_matches_selected_submatrix_and_gradients(self):
        state = torch.get_rng_state()
        r = sample_group_indices(self.r, .5)
        p = sample_group_indices(self.p, .5)
        torch.set_rng_state(state)
        # Both GEMMs must run on the smaller matrix, not mask a full matrix.
        with patch.object(loss_module.torch, 'bmm', wraps=torch.bmm) as bmm:
            actual, stats = routesync_relational_loss(self.pre, self.post, self.r, self.p, sample_ratio=.5, debug=True)
            self.assertEqual(bmm.call_count, 2)
            for call in bmm.call_args_list:
                self.assertEqual(call.args[0].shape, (2, 3, 8))
                self.assertEqual(call.args[1].shape, (2, 8, 3))
        expected = self.reference(r, p)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        grad, post_grad = torch.autograd.grad(actual, (self.pre, self.post), allow_unused=True)
        expected_grad, = torch.autograd.grad(expected, (self.pre,))
        torch.testing.assert_close(grad, expected_grad, rtol=0, atol=0)
        self.assertIsNone(post_grad)
        self.assertEqual(stats['sampled_p_tokens'], 3)
        selected = torch.zeros(2, 12, dtype=torch.bool).scatter_(1, torch.cat((r, p), 1), True)
        self.assertEqual(grad[~selected].abs().sum(), 0)

    def test_sampling_without_replacement_floor_and_minimum(self):
        indices = torch.arange(19).expand(32, -1)
        for ratio, count in ((.5, 9), (.0001, 1), (1., 19)):
            chosen = sample_group_indices(indices, ratio)
            self.assertEqual(chosen.shape, (32, count))
            for row in chosen:
                self.assertEqual(len(row.unique()), count)
                self.assertTrue(((row >= 0) & (row < 19)).all())
        a = sample_group_indices(indices, .5)
        b = sample_group_indices(indices, .5)
        self.assertFalse(torch.equal(a, b))
        self.assertFalse(torch.equal(a[0], a[1]))

    def test_recursive_bf16_loss_keeps_post_stopped(self):
        model = SiT(input_size=8, patch_size=2, hidden_size=32,
            decoder_hidden_size=32, depth=7, num_heads=4, mlp_ratio=2,
            num_classes=10, class_dropout_prob=0, qk_norm=False, fused_attn=False,
            use_tread_routing=True, tread_start_block=1, tread_end_block=5,
            tread_active_ratio=.5, tread_recursive=True, tread_num_groups=2,
            use_routesync=True, routesync_sample_ratio=.5).train()
        with torch.no_grad():
            for m in model.modules():
                if hasattr(m, 'adaLN_modulation'):
                    nn.init.normal_(m.adaLN_modulation[-1].weight, std=.02)
                    nn.init.normal_(m.adaLN_modulation[-1].bias, std=.02)
            nn.init.normal_(model.final_layer.linear.weight, std=.02)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = FlowMatchingLoss(use_routesync=True, routesync_weight=.1,
                routesync_sample_ratio=.5, routesync_debug=True)(
                    model, torch.randn(2, 4, 8, 8), {'y': torch.tensor([1, 3])})
        self.assertEqual(result['sampled_r_tokens'], 4)
        self.assertEqual(result['relation_pre_mean'].dtype, torch.bfloat16)
        params = list(model.recurrent_group_blocks.parameters()) + list(model.suffix_blocks.parameters())
        grads = torch.autograd.grad(result['route_sync'], params, allow_unused=True, retain_graph=True)
        self.assertTrue(all(g is None for g in grads))
        result['total'].backward()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)

    def test_cli_validation_and_legacy_default(self):
        parser = argparse.ArgumentParser()
        add_routesync_args(parser)
        self.assertEqual(parser.parse_args([]).routesync_sample_ratio, 1.)
        self.assertEqual(tread_kwargs(SimpleNamespace())['routesync_sample_ratio'], 1.)
        self.assertEqual(parser.parse_args(['--routesync-sample-ratio', '.5']).routesync_sample_ratio, .5)
        for value in ('0', '-.1', '1.1', 'nan', 'inf'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(['--routesync-sample-ratio', value])
            with self.assertRaises(ValueError):
                FlowMatchingLoss(routesync_sample_ratio=float(value))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(['--routesync-post-grad-scale', '.25'])


if __name__ == '__main__':
    unittest.main()
