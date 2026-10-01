from __future__ import annotations

import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from attention_alignment_hdiv.loss import SILoss as CKASILoss
from attention_alignment_row_hdiv.loss import (
    SILoss,
    cosine_token_relation_matrix,
    row_relational_l1_distance,
    row_relational_l1_diversity_loss,
)
from attention_alignment_row_hdiv.train import parse_args, resolve_hdiv_mode


def manual_row_l1_distance(shallow, deep, eps=1e-6, include_diagonal=False):
    shallow = F.normalize(shallow.float(), dim=-1, eps=eps)
    deep = F.normalize(deep.detach().float(), dim=-1, eps=eps)
    relation_shallow = shallow @ shallow.transpose(-1, -2)
    relation_deep = deep @ deep.transpose(-1, -2)
    difference = (relation_shallow - relation_deep).abs()
    if include_diagonal:
        return difference.mean(dim=-1)
    tokens = shallow.shape[1]
    off_diagonal = 1.0 - torch.eye(tokens, dtype=difference.dtype)
    return (difference * off_diagonal.unsqueeze(0)).sum(-1) / (tokens - 1)


class RowRelationalL1Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)

    def test_relation_matrix_shape_range_and_finiteness(self):
        hidden = torch.randn(2, 7, 5)
        relation = cosine_token_relation_matrix(hidden)
        self.assertEqual(relation.shape, (2, 7, 7))
        self.assertTrue(torch.isfinite(relation).all())
        self.assertTrue((relation <= 1.0 + 1e-6).all())
        self.assertTrue((relation >= -1.0 - 1e-6).all())

    def test_exact_l1_formula_diagonal_exclusion_and_single_normalization(self):
        shallow = torch.randn(2, 7, 5)
        deep = torch.randn(2, 7, 5)
        with mock.patch(
            "attention_alignment_row_hdiv.loss.F.normalize", wraps=F.normalize
        ) as normalize:
            actual = row_relational_l1_distance(shallow, deep)
        expected = manual_row_l1_distance(shallow, deep)
        including_diagonal = manual_row_l1_distance(
            shallow, deep, include_diagonal=True
        )
        self.assertEqual(normalize.call_count, 2)  # hs once and hd once only
        self.assertEqual(actual.shape, (2, 7))
        torch.testing.assert_close(actual, expected)
        # The included version divides by T and is not the requested formula.
        self.assertGreater((actual - including_diagonal).abs().max().item(), 1e-5)

    def test_identical_relations_give_zero_distance_and_margin_loss(self):
        shallow = torch.randn(3, 9, 6)
        margin = 0.2
        result = row_relational_l1_diversity_loss(
            shallow, shallow.clone(), margin=margin
        )
        torch.testing.assert_close(
            result["row_distance"], torch.zeros_like(result["row_distance"]),
            atol=2e-7, rtol=0,
        )
        torch.testing.assert_close(
            result["loss"], torch.full_like(result["loss"], margin),
            atol=2e-7, rtol=0,
        )
        self.assertEqual(result["active"].mean().item(), 1.0)

    def test_relations_beyond_margin_have_zero_loss(self):
        # Shallow has orthogonal tokens (off-diagonal relation 0), while every
        # deep token is identical (off-diagonal relation 1): row distance is 1.
        shallow = torch.eye(4).unsqueeze(0)
        deep = torch.ones(1, 4, 4)
        result = row_relational_l1_diversity_loss(shallow, deep, margin=0.5)
        torch.testing.assert_close(
            result["row_distance"], torch.ones_like(result["row_distance"])
        )
        self.assertEqual(result["loss"].abs().sum().item(), 0.0)
        self.assertEqual(result["active"].abs().sum().item(), 0.0)

    def test_margin_and_active_definition(self):
        shallow = torch.randn(3, 9, 6)
        deep = shallow + 0.4 * torch.randn_like(shallow)
        margin = 0.15
        result = row_relational_l1_diversity_loss(shallow, deep, margin=margin)
        expected_loss = torch.relu(margin - result["row_distance"])
        expected_active = (result["row_distance"] < margin).float()
        torch.testing.assert_close(result["loss"], expected_loss)
        torch.testing.assert_close(result["active"], expected_active)

    def test_row_loss_updates_shallow_but_never_deep(self):
        base = torch.randn(2, 12, 8)
        shallow = (base + 0.3 * torch.randn_like(base)).requires_grad_()
        deep = base.clone().requires_grad_()
        result = row_relational_l1_diversity_loss(shallow, deep, margin=0.5)
        self.assertGreater(result["loss"].mean().item(), 0.0)
        result["loss"].mean().backward()
        self.assertIsNotNone(shallow.grad)
        self.assertGreater(shallow.grad.abs().sum().item(), 0.0)
        self.assertIsNone(deep.grad)

    def test_batched_images_do_not_mix(self):
        shallow = torch.randn(4, 11, 7)
        deep = torch.randn(4, 11, 7)
        batched = row_relational_l1_distance(shallow, deep)
        individual = torch.cat([
            row_relational_l1_distance(shallow[i:i + 1], deep[i:i + 1])
            for i in range(4)
        ])
        torch.testing.assert_close(batched, individual)

    def test_bfloat16_autocast_returns_finite_fp32(self):
        shallow = torch.randn(2, 16, 12, requires_grad=True)
        deep = torch.randn(2, 16, 12, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = row_relational_l1_diversity_loss(shallow, deep, margin=0.5)
        self.assertEqual(result["loss"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(result["loss"]).all())
        self.assertTrue(torch.isfinite(result["row_distance"]).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA BF16 test")
    def test_cuda_bfloat16_backward(self):
        base = torch.randn(2, 32, 16, device="cuda")
        shallow = (base + 0.3 * torch.randn_like(base)).requires_grad_()
        deep = base.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = row_relational_l1_diversity_loss(shallow, deep, margin=0.5)
        result["loss"].mean().backward()
        self.assertTrue(torch.isfinite(result["loss"]).all())
        self.assertTrue(torch.isfinite(shallow.grad).all())
        self.assertIsNone(deep.grad)


class ModeIntegrationTests(unittest.TestCase):
    class DummyModel:
        def __call__(self, x, t, **kwargs):
            return x * 0.25, None

    def setUp(self):
        torch.manual_seed(41)
        self.images = torch.randn(2, 4, 8, 8)
        self.hidden = {
            "shallow": [
                types.SimpleNamespace(saved_hidden=torch.randn(2, 10, 6)),
                types.SimpleNamespace(saved_hidden=torch.randn(2, 10, 6)),
            ],
            "deep": types.SimpleNamespace(saved_hidden=torch.randn(2, 10, 6)),
        }

    def test_none_mode_matches_existing_disabled_loss_exactly(self):
        old = CKASILoss(enable_hdiv=False, attn_loss_type="kl")
        new = SILoss(hdiv_mode="none", attn_loss_type="kl")
        attention = {
            "shallow": torch.randn(2, 3, 5, 5).softmax(-1),
            "deep": torch.randn(2, 3, 5, 5).softmax(-1),
        }
        torch.manual_seed(123)
        old_result = old(self.DummyModel(), self.images, attn_maps_dict=attention)
        torch.manual_seed(123)
        new_result = new(self.DummyModel(), self.images, attn_maps_dict=attention)
        torch.testing.assert_close(new_result[0], old_result[0], rtol=0, atol=0)
        torch.testing.assert_close(new_result[1], old_result[1], rtol=0, atol=0)
        self.assertEqual(new_result[2]["loss"].abs().sum().item(), 0.0)

    def test_cka_mode_matches_existing_cka_exactly(self):
        old = CKASILoss(
            enable_hdiv=True, hdiv_cka_threshold=0.5, hdiv_detach_deep=True
        )
        new = SILoss(
            hdiv_mode="cka", hdiv_cka_threshold=0.5, hdiv_detach_deep=True
        )
        torch.manual_seed(321)
        old_result = old(
            self.DummyModel(), self.images, hidden_states_dict=self.hidden
        )
        torch.manual_seed(321)
        new_result = new(
            self.DummyModel(), self.images, hidden_states_dict=self.hidden
        )
        for index in (0, 1):
            torch.testing.assert_close(new_result[index], old_result[index], rtol=0, atol=0)
        for key in ("loss", "cka_mean", "distance_mean", "active_ratio"):
            torch.testing.assert_close(new_result[2][key], old_result[2][key], rtol=0, atol=0)

    def test_row_l1_mode_averages_all_shallow_sources(self):
        criterion = SILoss(hdiv_mode="row_l1", row_hdiv_margin=0.3)
        _, _, hdiv = criterion(
            self.DummyModel(), self.images, hidden_states_dict=self.hidden
        )
        expected = torch.stack([
            row_relational_l1_diversity_loss(
                source.saved_hidden,
                self.hidden["deep"].saved_hidden,
                margin=0.3,
            )["loss"]
            for source in self.hidden["shallow"]
        ]).mean(0)
        self.assertEqual(hdiv["loss"].shape, (2, 10))
        torch.testing.assert_close(hdiv["loss"], expected)

    def test_parser_defaults_and_zero_weight_disable(self):
        base = ["--exp-name", "test", "--data-dir", "/tmp/unused"]
        default_args = parse_args(base)
        self.assertEqual(default_args.hdiv_mode, "none")
        row_args = parse_args(
            base + ["--hdiv-mode", "row_l1", "--hdiv-weight", "0"]
        )
        self.assertEqual(resolve_hdiv_mode(row_args), "none")


if __name__ == "__main__":
    unittest.main()
