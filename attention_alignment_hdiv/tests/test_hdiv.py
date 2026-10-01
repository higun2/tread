from __future__ import annotations

import types
import unittest

import torch

from attention_alignment_hdiv.loss import SILoss, hdiv_margin_loss, linear_cka_per_sample
from attention_alignment_hdiv.train import create_hidden_capture_hook
from loss_b8 import SILoss as ExistingSILoss
from models.sit_b1 import SiT


class HDiversityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_identical_features_have_high_cka_and_active_margin(self):
        shallow = torch.randn(3, 32, 16)
        result = hdiv_margin_loss(shallow, shallow.clone(), cka_threshold=0.8)
        self.assertTrue(torch.all(result["cka"] > 0.999))
        self.assertGreater(result["loss"].mean().item(), 0.0)
        self.assertEqual(result["active"].mean().item(), 1.0)

    def test_orthogonal_token_patterns_have_zero_margin_loss(self):
        shallow = torch.zeros(1, 4, 2)
        deep = torch.zeros(1, 4, 2)
        shallow[0, :, 0] = torch.tensor([1.0, -1.0, 0.0, 0.0])
        deep[0, :, 0] = torch.tensor([0.0, 0.0, 1.0, -1.0])
        result = hdiv_margin_loss(shallow, deep, cka_threshold=0.8)
        self.assertLess(result["cka"].item(), 1e-4)
        self.assertEqual(result["loss"].item(), 0.0)
        self.assertEqual(result["active"].item(), 0.0)

    def test_gradient_reaches_shallow_but_not_deep(self):
        base = torch.randn(2, 24, 12)
        shallow = (base + 0.15 * torch.randn_like(base)).requires_grad_()
        deep = base.clone().requires_grad_()
        result = hdiv_margin_loss(shallow, deep, cka_threshold=0.5)
        self.assertGreater(result["loss"].mean().item(), 0.0)
        result["loss"].mean().backward()
        self.assertIsNotNone(shallow.grad)
        self.assertGreater(shallow.grad.abs().sum().item(), 0.0)
        self.assertIsNone(deep.grad)

    def test_detach_deep_false_updates_both_representations(self):
        base = torch.randn(2, 24, 12)
        shallow = (base + 0.15 * torch.randn_like(base)).requires_grad_()
        deep = base.clone().requires_grad_()
        result = hdiv_margin_loss(
            shallow, deep, cka_threshold=0.5, detach_deep=False
        )
        self.assertGreater(result["loss"].mean().item(), 0.0)
        result["loss"].mean().backward()
        self.assertGreater(shallow.grad.abs().sum().item(), 0.0)
        self.assertGreater(deep.grad.abs().sum().item(), 0.0)

    def test_per_sample_cka_matches_individual_calls(self):
        shallow = torch.randn(4, 20, 10)
        deep = torch.randn(4, 20, 10)
        batched = linear_cka_per_sample(shallow, deep)
        individual = torch.cat([
            linear_cka_per_sample(shallow[index:index + 1], deep[index:index + 1])
            for index in range(4)
        ])
        torch.testing.assert_close(batched, individual)

    @staticmethod
    def covariance_cka_reference(shallow, deep, eps=1e-6):
        """Previous [D,D] implementation retained only as a test oracle."""
        x = shallow.float() - shallow.float().mean(dim=1, keepdim=True)
        y = deep.float() - deep.float().mean(dim=1, keepdim=True)
        cross = x.transpose(1, 2) @ y
        self_x = x.transpose(1, 2) @ x
        self_y = y.transpose(1, 2) @ y
        numerator = cross.square().sum(dim=(-2, -1))
        denominator = (
            self_x.square().sum(dim=(-2, -1)).sqrt()
            * self_y.square().sum(dim=(-2, -1)).sqrt()
        )
        return numerator / (denominator + eps)

    def test_token_gram_matches_covariance_values_loss_and_gradients(self):
        shallow_base = torch.randn(2, 17, 31)
        deep_base = shallow_base + 0.4 * torch.randn_like(shallow_base)
        shallow_cov = shallow_base.clone().requires_grad_()
        deep_cov = deep_base.clone().requires_grad_()
        shallow_gram = shallow_base.clone().requires_grad_()
        deep_gram = deep_base.clone().requires_grad_()

        cka_cov = self.covariance_cka_reference(shallow_cov, deep_cov)
        cka_gram = linear_cka_per_sample(
            shallow_gram, deep_gram, detach_deep=False
        )
        torch.testing.assert_close(cka_gram, cka_cov, rtol=2e-6, atol=2e-7)

        threshold = 0.5
        margin = torch.acos(cka_cov.new_tensor(threshold))
        loss_cov = torch.relu(margin - torch.acos(cka_cov.clamp(1e-6, 1 - 1e-6)))
        result_gram = hdiv_margin_loss(
            shallow_gram, deep_gram, cka_threshold=threshold, detach_deep=False
        )
        torch.testing.assert_close(result_gram["loss"], loss_cov, rtol=2e-6, atol=2e-7)

        loss_cov.mean().backward()
        result_gram["loss"].mean().backward()
        torch.testing.assert_close(
            shallow_gram.grad, shallow_cov.grad, rtol=2e-5, atol=2e-8
        )
        torch.testing.assert_close(
            deep_gram.grad, deep_cov.grad, rtol=2e-5, atol=2e-8
        )

    def test_bfloat16_autocast_is_fp32_and_finite(self):
        shallow = torch.randn(2, 32, 16, requires_grad=True)
        deep = torch.randn(2, 32, 16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = hdiv_margin_loss(shallow, deep, cka_threshold=0.8)
        self.assertEqual(result["loss"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(result["loss"]).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA BF16 test")
    def test_cuda_bfloat16_backward(self):
        base = torch.randn(2, 32, 16, device="cuda")
        shallow = (base + 0.2 * torch.randn_like(base)).requires_grad_()
        deep = base.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = hdiv_margin_loss(shallow, deep, cka_threshold=0.5)
        self.assertTrue(torch.isfinite(result["loss"]).all())
        result["loss"].mean().backward()
        self.assertIsNotNone(shallow.grad)
        self.assertIsNone(deep.grad)

    def test_no_projector_or_parameters_are_introduced(self):
        criterion = SILoss(enable_hdiv=True)
        self.assertFalse(isinstance(criterion, torch.nn.Module))
        self.assertFalse(hasattr(criterion, "projector"))


class ExistingBehaviorTests(unittest.TestCase):
    class DummyModel:
        def __call__(self, x, t, **kwargs):
            return x * 0.25, None

    @staticmethod
    def attention(batch=2, heads=3, tokens=5):
        logits = torch.randn(batch, heads, tokens, tokens)
        return logits.softmax(dim=-1)

    def test_hdiv_disabled_matches_loss_b8_exactly(self):
        images = torch.randn(2, 4, 8, 8)
        shallow, deep = self.attention(), self.attention()
        existing = ExistingSILoss(attn_loss_type="kl")
        extended = SILoss(attn_loss_type="kl", enable_hdiv=False)
        kwargs = {"shallow": shallow, "deep": deep}
        torch.manual_seed(1234)
        old_fm, old_attn = existing(
            self.DummyModel(), images, {"y": torch.tensor([1, 2])}, attn_maps_dict=kwargs
        )
        torch.manual_seed(1234)
        new_fm, new_attn, hdiv = extended(
            self.DummyModel(), images, {"y": torch.tensor([1, 2])}, attn_maps_dict=kwargs
        )
        torch.testing.assert_close(new_fm, old_fm, rtol=0, atol=0)
        torch.testing.assert_close(new_attn, old_attn, rtol=0, atol=0)
        self.assertEqual(hdiv["loss"].abs().sum().item(), 0.0)

    def test_hidden_hook_captures_only_complete_selected_block_outputs(self):
        model = SiT(
            input_size=8, patch_size=2, hidden_size=32, decoder_hidden_size=32,
            depth=3, num_heads=4, mlp_ratio=2.0, num_classes=10,
            class_dropout_prob=0.0, qk_norm=False, fused_attn=True,
        )
        selected = (0, 2)
        hooks = []
        for index in selected:
            model.blocks[index].saved_hidden = None
            hooks.append(model.blocks[index].register_forward_hook(create_hidden_capture_hook()))
        model(torch.randn(2, 4, 8, 8), torch.rand(2), torch.tensor([1, 2]))
        self.assertEqual(model.blocks[0].saved_hidden.shape, (2, 16, 32))
        self.assertEqual(model.blocks[2].saved_hidden.shape, (2, 16, 32))
        self.assertFalse(hasattr(model.blocks[1], "saved_hidden"))
        for hook in hooks:
            hook.remove()


if __name__ == "__main__":
    unittest.main()
