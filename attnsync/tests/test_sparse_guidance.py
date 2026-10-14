"""Sparse Guidance: per-call active ratio override and the SG sampler combination."""
import unittest

import torch

from attnsync.model import SiT
from attnsync.sparse_guidance import (
    sg_euler_maruyama_sampler, sg_euler_sampler, validate_sparse_guidance,
)


def make():
    model = SiT(
        input_size=8, patch_size=2, hidden_size=32, decoder_hidden_size=32,
        depth=6, num_heads=4, num_classes=10, class_dropout_prob=.2,
        qk_norm=False, fused_attn=True, use_tread_routing=True,
        tread_start_block=1, tread_end_block=4, tread_active_ratio=0.5,
        tread_eval_mode="dense")
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, 'adaLN_modulation'):
                torch.nn.init.normal_(module.adaLN_modulation[-1].weight, std=.1)
        torch.nn.init.normal_(model.final_layer.linear.weight, std=.1)
    return model.eval()


class FakeModel:
    """Velocity depends on the requested active ratio, so the combination is checkable."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, t, y=None, tread_eval_mode=None, tread_active_ratio=None):
        self.calls.append((tread_eval_mode, tread_active_ratio))
        return tread_active_ratio * torch.ones_like(x), None


class ActiveRatioOverride(unittest.TestCase):
    def test_override_sets_routed_token_count(self):
        model = make()
        x, t, y = torch.randn(2, 4, 8, 8), torch.rand(2), torch.tensor([1, 2])
        tokens = model.x_embedder.num_patches  # 16
        for gamma in (0.5, 0.9):
            _, _, info = model(x, t, y, tread_eval_mode="sparse",
                               tread_active_ratio=1 - gamma, return_routing_info=True)
            self.assertEqual(info["active_indices"].shape[1], round(tokens * (1 - gamma)))
            self.assertEqual(info["block_input_token_counts"][1], round(tokens * (1 - gamma)))

    def test_default_ratio_unchanged(self):
        model = make()
        x, t, y = torch.randn(2, 4, 8, 8), torch.rand(2), torch.tensor([1, 2])
        _, _, info = model(x, t, y, tread_eval_mode="sparse", return_routing_info=True)
        self.assertEqual(info["active_indices"].shape[1], 8)

    def test_override_rejected_on_dense_path_and_in_training(self):
        model = make()
        x, t, y = torch.randn(2, 4, 8, 8), torch.rand(2), torch.tensor([1, 2])
        with self.assertRaises(ValueError):
            model(x, t, y, tread_eval_mode="dense", tread_active_ratio=0.5)
        model.train()
        with self.assertRaises(ValueError):
            model(x, t, y, tread_active_ratio=0.5)

    def test_subsets_resampled_each_call(self):
        model = make()
        x, t, y = torch.randn(2, 4, 8, 8), torch.rand(2), torch.tensor([1, 2])
        torch.manual_seed(0)
        a = model(x, t, y, tread_eval_mode="sparse", tread_active_ratio=0.5,
                  return_routing_info=True)[2]["active_indices"]
        b = model(x, t, y, tread_eval_mode="sparse", tread_active_ratio=0.5,
                  return_routing_info=True)[2]["active_indices"]
        self.assertFalse(torch.equal(a.sort(1).values, b.sort(1).values))


class Sampler(unittest.TestCase):
    def test_combination_and_branch_ratios(self):
        # v_strong = 0.5, v_weak = 0.1 -> v = 0.1 + 2.5 * 0.4 = 1.1 for every ODE step.
        fake = FakeModel()
        z = torch.zeros(2, 4, 8, 8)
        out = sg_euler_sampler(fake, z, torch.tensor([1, 2]), num_steps=4,
                               sg_scale=2.5, sg_gamma_strong=0.5, sg_gamma_weak=0.9)
        torch.testing.assert_close(out, torch.full_like(out, -1.1, dtype=torch.float64))
        self.assertEqual(len(fake.calls), 8)
        self.assertTrue(all(mode == "sparse" for mode, _ in fake.calls))
        self.assertEqual(sorted({round(r, 6) for _, r in fake.calls}), [0.1, 0.5])

    def test_outside_interval_uses_strong_only(self):
        fake = FakeModel()
        z = torch.zeros(1, 4, 8, 8)
        # t_cur in {1.0, 0.75, 0.5, 0.25}; interval [0, 0.6] guides the last two.
        out = sg_euler_sampler(fake, z, torch.tensor([1]), num_steps=4, sg_scale=2.5,
                               guidance_low=0.0, guidance_high=0.6)
        self.assertEqual(len(fake.calls), 6)
        torch.testing.assert_close(out, torch.full_like(out, -(0.5 * 0.5 + 1.1 * 0.5),
                                                       dtype=torch.float64))

    def test_scale_one_is_strong_branch_only(self):
        fake = FakeModel()
        sg_euler_maruyama_sampler(fake, torch.zeros(1, 4, 8, 8), torch.tensor([1]),
                                  num_steps=5, sg_scale=1.0)
        self.assertTrue(all(abs(r - 0.5) < 1e-9 for _, r in fake.calls))

    def test_real_model_sde_runs(self):
        model = make()
        torch.manual_seed(0)
        out = sg_euler_maruyama_sampler(model, torch.randn(2, 4, 8, 8), torch.tensor([1, 2]),
                                        num_steps=4, sg_scale=2.5)
        self.assertEqual(out.shape, (2, 4, 8, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_validation(self):
        for args in ((0.5, 0.5, 0.9), (2.5, 0.9, 0.5), (2.5, 0.5, 1.0), (2.5, -0.1, 0.5)):
            with self.assertRaises(ValueError):
                validate_sparse_guidance(*args)


if __name__ == "__main__":
    unittest.main()
