"""Focused CPU tests for the optional NC-DFM objective."""

import unittest

import torch

from loss_b8 import SILoss as BaselineSILoss
from loss_b9 import (
    SILoss,
    build_ncdfm_pairs,
    compute_ncdfm_loss,
    compute_ncdfm_schedule,
    share_pair_stochasticity,
)


class CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.7))
        self.calls = 0
        self.last_input = None
        self.last_time = None

    def forward(self, x, t, **kwargs):
        self.calls += 1
        self.last_input = x.detach().clone()
        self.last_time = t.detach().clone()
        return self.scale * x, None


class NCDFMTests(unittest.TestCase):
    def test_shared_stochastic_variables(self):
        torch.manual_seed(3)
        batch_size = 7
        time = torch.rand(batch_size, 1, 1, 1)
        noise = torch.randn(batch_size, 2, 3, 3)
        original_time, original_noise = time.clone(), noise.clone()
        pair_i, pair_j = build_ncdfm_pairs(batch_size, 0.75, time.device)
        shared_time, shared_noise = share_pair_stochasticity(time, noise, pair_i, pair_j)

        self.assertTrue(torch.equal(shared_time[pair_i], shared_time[pair_j]))
        self.assertTrue(torch.equal(shared_noise[pair_i], shared_noise[pair_j]))
        paired = torch.cat((pair_i, pair_j))
        unpaired_mask = torch.ones(batch_size, dtype=torch.bool)
        unpaired_mask[paired] = False
        self.assertTrue(torch.equal(shared_time[unpaired_mask], original_time[unpaired_mask]))
        self.assertTrue(torch.equal(shared_noise[unpaired_mask], original_noise[unpaired_mask]))

    def test_analytical_noise_cancellation(self):
        torch.manual_seed(4)
        x = torch.randn(2, 4, 3, 3)
        noise = torch.randn(1, 4, 3, 3).expand_as(x)
        t = torch.full((2, 1, 1, 1), 0.37)
        loss_fn = SILoss(path_type="linear")
        _, _, d_alpha, d_sigma = loss_fn.interpolant(t)
        target = d_alpha * x + d_sigma * noise
        self.assertTrue(torch.allclose(target[0] - target[1], d_alpha * (x[0] - x[1])))

    def test_one_model_forward(self):
        model = CountingModel()
        loss_fn = SILoss(ncdfm_enabled=True, ncdfm_pair_fraction=1.0)
        loss_fn(model, torch.randn(5, 2, 3, 3))
        self.assertEqual(model.calls, 1)

    def test_gradient_flows_through_both_pair_members(self):
        prediction = torch.randn(2, 3, 2, 2, requires_grad=True)
        target = torch.randn_like(prediction)
        pair_i = torch.tensor([0])
        pair_j = torch.tensor([1])
        schedule = torch.ones(2, 1, 1, 1)
        loss, _ = compute_ncdfm_loss(
            prediction, target, pair_i, pair_j, schedule, lambda_max=0.1
        )
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad[0].abs().sum().item(), 0.0)
        self.assertGreater(prediction.grad[1].abs().sum().item(), 0.0)

    def test_disabled_mode_matches_baseline(self):
        images = torch.randn(5, 2, 3, 3)
        baseline_model = CountingModel()
        ncdfm_model = CountingModel()
        ncdfm_model.load_state_dict(baseline_model.state_dict())

        torch.manual_seed(11)
        baseline_loss, _ = BaselineSILoss()(baseline_model, images)
        torch.manual_seed(11)
        new_loss, ncdfm_loss, _ = SILoss(ncdfm_enabled=False)(ncdfm_model, images)

        self.assertTrue(torch.equal(baseline_model.last_time, ncdfm_model.last_time))
        self.assertTrue(torch.equal(baseline_model.last_input, ncdfm_model.last_input))
        self.assertTrue(torch.equal(baseline_loss, new_loss))
        self.assertEqual(ncdfm_loss.item(), 0.0)

    def test_signal_ratio_endpoints_follow_interpolant(self):
        loss_fn = SILoss(path_type="linear")
        t = torch.tensor([0.0, 1.0]).view(2, 1, 1, 1)
        alpha, sigma, _, _ = loss_fn.interpolant(t)
        weight = compute_ncdfm_schedule(alpha, sigma, "signal_ratio", 1e-8)
        self.assertGreater(weight[0].item(), 0.999)
        self.assertLess(weight[1].item(), 1e-7)

    def test_pair_fraction_and_small_batch_edges(self):
        cases = [
            (1, 1.0, 0),
            (5, 0.0, 0),
            (5, 0.5, 1),
            (5, 1.0, 2),
            (6, 1.0, 3),
        ]
        for batch_size, fraction, expected_pairs in cases:
            with self.subTest(batch_size=batch_size, fraction=fraction):
                pair_i, pair_j = build_ncdfm_pairs(batch_size, fraction, torch.device("cpu"))
                self.assertEqual(pair_i.numel(), expected_pairs)
                self.assertEqual(pair_j.numel(), expected_pairs)
                selected = torch.cat((pair_i, pair_j))
                self.assertEqual(selected.unique().numel(), selected.numel())

    def test_cpu_bfloat16_smoke(self):
        model = CountingModel().to(dtype=torch.bfloat16)
        images = torch.randn(4, 2, 3, 3, dtype=torch.bfloat16)
        fm, ncdfm, metrics = SILoss(ncdfm_enabled=True)(model, images)
        total = fm.mean() + ncdfm
        total.backward()
        self.assertTrue(torch.isfinite(total.float()))
        self.assertTrue(torch.isfinite(model.scale.grad.float()))
        self.assertEqual(metrics["ncdfm_num_pairs"].item(), 2.0)


if __name__ == "__main__":
    unittest.main()
