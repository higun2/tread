"""Routed attention sync: map recomputation, divergences, gradients and validation."""
import itertools
import math
import unittest

import torch
import torch.nn.functional as F

from attnsync.loss import FlowMatchingLoss, attention_sync_loss
from attnsync.model import SiT, attention_log_probs


def make(recursive=False, correction=False, **kwargs):
    kwargs.setdefault('attn_sync_student_blocks', [1, 2])
    kwargs.setdefault('attn_sync_teacher_block', 3)
    model = SiT(
        input_size=8, patch_size=2, hidden_size=32, decoder_hidden_size=32,
        depth=6, num_heads=4, num_classes=10, class_dropout_prob=.2,
        qk_norm=False, fused_attn=True, use_tread_routing=True,
        tread_start_block=1, tread_end_block=4, tread_recursive=recursive,
        tread_num_groups=3, tread_attn_correction=correction,
        use_attn_sync=True, **kwargs)
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, 'adaLN_modulation'):
                torch.nn.init.normal_(module.adaLN_modulation[-1].weight, std=.1)
        torch.nn.init.normal_(model.final_layer.linear.weight, std=.1)
    return model


def reference_divergence(s, t, heads, kind):
    """Direct probability-space reference."""
    ps, pt = s.exp(), t.exp()
    if heads == 'mean':
        ps, pt = ps.mean(1), pt.mean(1)
    if kind == 'l1':
        return (ps - pt).abs().sum(-1).mean()
    kl = lambda a, b: (a * (a.log() - b.log())).sum(-1)
    if kind == 'kl':
        return kl(pt, ps).mean()
    m = (ps + pt) / 2
    return (0.5 * kl(ps, m) + 0.5 * kl(pt, m)).mean()


class AttentionSyncLossTests(unittest.TestCase):
    def test_matches_reference_and_stops_teacher(self):
        for heads, kind in itertools.product(('mean', 'per-head'), ('l1', 'js', 'kl')):
            with self.subTest(heads=heads, kind=kind):
                s = torch.randn(2, 3, 5, 5, dtype=torch.float64).log_softmax(-1).requires_grad_()
                t = torch.randn(2, 3, 5, 5, dtype=torch.float64).log_softmax(-1).requires_grad_()
                loss = attention_sync_loss(s, t, heads, kind)
                torch.testing.assert_close(loss, reference_divergence(s.detach(), t.detach(), heads, kind))
                loss.backward()
                self.assertIsNone(t.grad)
                self.assertGreater(s.grad.abs().sum().item(), 0)

    def test_zero_when_equal_and_js_bounded(self):
        s = torch.randn(2, 3, 6, 6).log_softmax(-1)
        for heads, kind in itertools.product(('mean', 'per-head'), ('l1', 'js', 'kl')):
            self.assertAlmostEqual(attention_sync_loss(s, s.clone(), heads, kind).item(), 0, places=5)
        one_hot = lambda i: torch.full((1, 1, 1, 4), -1e4).index_fill(-1, torch.tensor([i]), 0).log_softmax(-1)
        js = attention_sync_loss(one_hot(0), one_hot(1), 'per-head', 'js')
        self.assertAlmostEqual(js.item(), math.log(2), places=4)

    def test_head_mean_ignores_head_permutation(self):
        s = torch.randn(2, 4, 6, 6).log_softmax(-1)
        permuted = s[:, torch.tensor([2, 0, 3, 1])]
        self.assertAlmostEqual(attention_sync_loss(s, permuted, 'mean', 'js').item(), 0, places=5)
        self.assertGreater(attention_sync_loss(s, permuted, 'per-head', 'js').item(), 1e-3)


class AttentionSyncModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_log_probs_match_explicit_attention(self):
        for correction in (False, True):
            with self.subTest(correction=correction):
                model = make(correction=correction).train()
                block = model.blocks[2]
                hidden, cond = torch.randn(3, 8, 32), torch.randn(3, 32)
                captured = []
                def hook(module, args, output):
                    captured.append(args[0])
                handle = block.attn.register_forward_hook(hook)
                block(hidden, cond)
                handle.remove()
                attn = block.attn
                x = captured[0]
                q, k, v = attn.qkv(x).reshape(3, 8, 3, 4, 8).permute(2, 0, 3, 1, 4).unbind(0)
                log_probs = attention_log_probs(block, hidden, cond)
                # Applying the recomputed probabilities to V must reproduce the real attention output.
                out = (log_probs.exp() @ v).transpose(1, 2).reshape(3, 8, 32)
                torch.testing.assert_close(attn.proj(out), attn(x), rtol=1e-4, atol=1e-5)

    def test_features_gradients_and_training_step(self):
        for recursive, heads, kind in itertools.product((False, True), ('mean', 'per-head'), ('l1', 'js', 'kl')):
            with self.subTest(recursive=recursive, heads=heads, kind=kind):
                model = make(recursive, attn_sync_heads=heads, attn_sync_loss=kind).train()
                _, features = model(torch.randn(4, 4, 8, 8), torch.rand(4), torch.tensor([1, 2, 3, 4]))
                f = features['attn_sync']
                self.assertEqual(set(f['students']), {1, 2})
                self.assertEqual(f['teacher'].shape, (4, 4, 8, 8))
                self.assertFalse(f['teacher'].requires_grad)
                self.assertTrue(all(s.requires_grad for s in f['students'].values()))
                criterion = FlowMatchingLoss(use_attn_sync=True, attn_sync_weight=1.0,
                                             attn_sync_heads=heads, attn_sync_loss=kind)
                model.zero_grad(set_to_none=True)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    result = criterion(model, torch.randn(4, 4, 8, 8), {'y': torch.tensor([1, 2, 3, 4])})
                self.assertGreater(result['attn_sync'].item(), 0)
                result['total'].backward()
                # The teacher block's own forward must still receive FM gradients (no autocast cache leak).
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters() if p.requires_grad))

    def test_sync_gradient_only_reaches_student_and_upstream(self):
        model = make().train()
        _, features = model(torch.randn(4, 4, 8, 8), torch.rand(4), torch.tensor([1, 2, 3, 4]))
        f = features['attn_sync']
        loss = attention_sync_loss(f['students'][1], f['teacher'])
        loss.backward()
        grad = lambda m: sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
        self.assertGreater(grad(model.blocks[1].attn.qkv), 0)
        self.assertGreater(grad(model.blocks[0]), 0)       # prefix (upstream)
        self.assertEqual(grad(model.blocks[3]), 0)          # teacher block
        self.assertEqual(grad(model.blocks[2]), 0)          # not a student in this loss

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA autocast weight-cache regression')
    def test_cuda_autocast_teacher_keeps_block_gradients(self):
        # Teacher off the FP32 endpoint block so its weights go through autocast casts.
        model = make(attn_sync_student_blocks=[1, 3], attn_sync_teacher_block=2).cuda().train()
        criterion = FlowMatchingLoss(use_attn_sync=True, attn_sync_weight=1.0)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = criterion(model, torch.randn(4, 4, 8, 8, device='cuda'),
                               {'y': torch.tensor([1, 2, 3, 4], device='cuda')})
        result['total'].backward()
        teacher = model.blocks[2]
        self.assertTrue(all(p.grad is not None for p in teacher.parameters()))

    def test_ratio_subsamples_batch(self):
        model = make(attn_sync_ratio=0.5).train()
        _, features = model(torch.randn(4, 4, 8, 8), torch.rand(4), torch.tensor([1, 2, 3, 4]))
        self.assertEqual(features['attn_sync']['teacher'].shape[0], 2)

    def test_eval_emits_nothing_and_validation(self):
        out = make().eval()(torch.randn(2, 4, 8, 8), torch.rand(2), torch.tensor([1, 2]))
        self.assertIsNone(out[1])
        for bad in (dict(attn_sync_student_blocks=[0]), dict(attn_sync_teacher_block=4),
                    dict(attn_sync_student_blocks=[3]), dict(attn_sync_student_blocks=[1, 1]),
                    dict(attn_sync_teacher_block=None)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                make(**bad)


if __name__ == '__main__':
    unittest.main()
