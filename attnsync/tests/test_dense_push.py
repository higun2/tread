"""DensePush: margin loss, dense/sparse feature correspondence and gradients."""
import itertools
import unittest

import torch
import torch.nn.functional as F

from attnsync.loss import FlowMatchingLoss, dense_push_loss
from attnsync.model import SiT


def make(recursive=False, **kwargs):
    model = SiT(
        input_size=8, patch_size=2, hidden_size=32, decoder_hidden_size=32,
        depth=5, num_heads=4, num_classes=10, class_dropout_prob=.2,
        qk_norm=False, fused_attn=True, use_tread_routing=True,
        tread_start_block=1, tread_end_block=3, tread_recursive=recursive,
        tread_num_groups=1, use_dense_push=True, **kwargs)
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, 'adaLN_modulation'):
                torch.nn.init.normal_(module.adaLN_modulation[-1].weight, std=.1)
        torch.nn.init.normal_(model.final_layer.linear.weight, std=.1)
    return model


class DensePushLossTests(unittest.TestCase):
    def test_formula_and_margin(self):
        sparse = torch.randn(4, 6, 8, dtype=torch.float64, requires_grad=True)
        dense = torch.randn(4, 6, 8, dtype=torch.float64)
        loss, cosine = dense_push_loss(sparse, dense, margin=0.0)
        expected_cos = F.cosine_similarity(sparse.mean(1), dense.mean(1), dim=-1)
        torch.testing.assert_close(cosine, expected_cos.detach())
        torch.testing.assert_close(loss, F.relu(expected_cos).square().mean())

    def test_zero_gradient_below_margin(self):
        sparse = torch.randn(3, 5, 8, requires_grad=True)
        dense = -sparse.detach()  # cosine -1, below any valid margin
        loss, _ = dense_push_loss(sparse, dense, margin=0.5)
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(sparse.grad.abs().sum().item(), 0.0)

    def test_gradient_reduces_cosine(self):
        sparse = torch.randn(2, 5, 8, requires_grad=True)
        dense = sparse.detach() + 0.01 * torch.randn(2, 5, 8)
        loss, before = dense_push_loss(sparse, dense, margin=0.5)
        loss.backward()
        with torch.no_grad():
            _, after = dense_push_loss(sparse - 1.0 * sparse.grad, dense, margin=0.5)
        self.assertTrue((after < before).all())


class DensePushModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_dense_matches_reference_and_sparse_is_captured(self):
        for recursive, block, tokens in itertools.product(
                (False, True), (2, 3, 4), ("all", "active", "routed")):
            with self.subTest(recursive=recursive, block=block, tokens=tokens):
                torch.manual_seed(0)
                model = make(recursive, dense_push_block=block, dense_push_tokens=tokens,
                             dense_push_ratio=.5).train()
                prefix = []
                first = model.prefix_blocks[0] if recursive else model.blocks[0]
                hook = first.register_forward_hook(
                    lambda m, a, o: prefix.append((o.detach(), a[1].detach())))
                _, features, info = model(torch.randn(6, 4, 8, 8), torch.rand(6),
                                          torch.arange(6) % 10, return_routing_info=True)
                hook.remove()
                f = features['dense_push']
                self.assertEqual(f['dense'].shape, f['sparse'].shape)
                self.assertFalse(f['dense'].requires_grad)
                self.assertTrue(f['sparse'].requires_grad)
                idx = f['sample_indices']
                self.assertEqual(idx.numel(), 3)
                # Recompute the dense path from the saved prefix output.
                with torch.no_grad():
                    h, c = prefix[0][0][idx], prefix[0][1][idx]
                    for b in range(1, block + 1):
                        if recursive and b < 3:
                            module = model.recurrent_group_blocks[0]
                        elif recursive:
                            module = model.suffix_blocks[b - 3]
                        else:
                            module = model.blocks[b]
                        h = module(h, c)
                    if tokens != "all":
                        key = "active_indices" if tokens == "active" else "bypass_indices"
                        pos = info[key][idx]
                        h = h.gather(1, pos.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
                torch.testing.assert_close(f['dense'], h, rtol=1e-4, atol=1e-5)

    def test_grad_modes_and_training_step(self):
        for grad_mode in ("sparse", "both"):
            with self.subTest(grad=grad_mode):
                model = make(dense_push_grad=grad_mode, dense_push_margin=-1.0).train()
                criterion = FlowMatchingLoss(use_dense_push=True, dense_push_weight=1.0,
                                             dense_push_margin=-1.0)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    result = criterion(model, torch.randn(4, 4, 8, 8), {'y': torch.tensor([1, 2, 3, 4])})
                self.assertGreater(result['dense_push'].item(), 0)
                self.assertTrue(torch.isfinite(result['total']))
                # Gradient only from the push term must still reach the prefix.
                model.zero_grad(set_to_none=True)
                result['dense_push'].backward()
                first = model.blocks[0]
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in first.parameters()))

    def test_both_mode_backprops_dense_path(self):
        model = make(dense_push_grad="both", dense_push_ratio=1.0).train()
        _, features = model(torch.randn(2, 4, 8, 8), torch.rand(2), torch.tensor([1, 2]))
        self.assertTrue(features['dense_push']['dense'].requires_grad)

    def test_eval_and_disabled_paths_emit_nothing(self):
        model = make().eval()
        out = model(torch.randn(2, 4, 8, 8), torch.rand(2), torch.tensor([1, 2]))
        self.assertIsNone(out[1])
        model = make(dense_push_weight=0.0).train()
        _, features = model(torch.randn(2, 4, 8, 8), torch.rand(2), torch.tensor([1, 2]))
        self.assertIsNone(features)

    def test_validation(self):
        with self.assertRaises(ValueError):
            make(dense_push_block=0)
        with self.assertRaises(ValueError):
            make(dense_push_block=5)
        with self.assertRaises(ValueError):
            make(dense_push_margin=1.0)


if __name__ == '__main__':
    unittest.main()
