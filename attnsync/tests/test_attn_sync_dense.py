"""Attention sync without TREAD: plain dense SiT forward plus full-token attention capture."""
import itertools
import unittest

import torch

from attnsync.loss import FlowMatchingLoss, attention_sync_loss
from attnsync.model import SiT
from models.sit_b1 import SiT as VanillaSiT


def make_dense(**kwargs):
    kwargs.setdefault('attn_sync_student_blocks', [1])
    kwargs.setdefault('attn_sync_teacher_block', 4)
    torch.manual_seed(0)
    model = SiT(
        input_size=8, patch_size=2, hidden_size=32, decoder_hidden_size=32,
        depth=6, num_heads=4, num_classes=10, class_dropout_prob=.2,
        qk_norm=False, fused_attn=True, use_tread_routing=False,
        use_attn_sync=True, **kwargs)
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, 'adaLN_modulation'):
                torch.nn.init.normal_(module.adaLN_modulation[-1].weight, std=.1)
        torch.nn.init.normal_(model.final_layer.linear.weight, std=.1)
    return model


def inputs(batch=4):
    torch.manual_seed(1)
    return torch.randn(batch, 4, 8, 8), torch.rand(batch), torch.arange(1, batch + 1)


class DenseAttentionSyncTests(unittest.TestCase):
    def test_output_matches_plain_sit_forward(self):
        model = make_dense().train()
        x, t, y = inputs()
        drop = torch.zeros(4, dtype=torch.bool)  # same labels in both calls
        out, features = model(x, t, y, force_drop_ids=drop)
        ref, _ = VanillaSiT.forward(model, x, t, y, force_drop_ids=drop)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        self.assertIn('attn_sync', features)

    def test_captures_full_token_maps_that_match_real_attention(self):
        model = make_dense(attn_sync_student_blocks=[0, 2]).train()
        x, t, y = inputs()
        drop = torch.zeros(4, dtype=torch.bool)
        captured = {}
        hooks = [model.blocks[i].attn.register_forward_hook(
            lambda m, a, o, i=i: captured.__setitem__(i, (a[0], o))) for i in (0, 2, 4)]
        _, features = model(x, t, y, force_drop_ids=drop)
        for h in hooks:
            h.remove()
        f = features['attn_sync']
        tokens = model.x_embedder.num_patches
        self.assertEqual(set(f['students']), {0, 2})
        self.assertEqual(f['teacher_block'], 4)
        self.assertEqual(f['teacher'].shape, (4, 4, tokens, tokens))
        self.assertFalse(f['teacher'].requires_grad)
        for i, log_probs in [*f['students'].items(), (4, f['teacher'])]:
            attn_in, attn_out = captured[i]
            attn = model.blocks[i].attn
            v = attn.qkv(attn_in).reshape(4, tokens, 3, 4, 8).permute(2, 0, 3, 1, 4)[2]
            recon = attn.proj((log_probs.exp() @ v).transpose(1, 2).reshape(4, tokens, 32))
            torch.testing.assert_close(recon, attn_out, rtol=1e-4, atol=1e-5)

    def test_any_block_order_allowed_without_tread(self):
        # Teacher before the student and blocks outside any TREAD route are valid without TREAD.
        for students, teacher in (([0], 5), ([5], 0), ([1, 3], 2)):
            with self.subTest(students=students, teacher=teacher):
                model = make_dense(attn_sync_student_blocks=students,
                                   attn_sync_teacher_block=teacher).train()
                _, features = model(*inputs())
                self.assertEqual(set(features['attn_sync']['students']), set(students))
        for bad in (dict(attn_sync_teacher_block=6), dict(attn_sync_student_blocks=[-1]),
                    dict(attn_sync_student_blocks=[4]), dict(attn_sync_student_blocks=[1, 1]),
                    dict(attn_sync_teacher_block=None)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                make_dense(**bad)

    def test_tread_still_requires_routed_blocks(self):
        with self.assertRaises(ValueError):
            SiT(input_size=8, patch_size=2, hidden_size=32, decoder_hidden_size=32, depth=6,
                num_heads=4, num_classes=10, qk_norm=False, fused_attn=True,
                use_tread_routing=True, tread_start_block=1,
                tread_end_block=4, use_attn_sync=True, attn_sync_student_blocks=[1],
                attn_sync_teacher_block=5)

    def test_sync_gradient_only_reaches_student_and_upstream(self):
        model = make_dense(attn_sync_student_blocks=[2], attn_sync_teacher_block=4).train()
        _, features = model(*inputs())
        f = features['attn_sync']
        attention_sync_loss(f['students'][2], f['teacher']).backward()
        grad = lambda m: sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
        self.assertGreater(grad(model.blocks[2].attn.qkv), 0)
        self.assertGreater(grad(model.blocks[0]), 0)
        self.assertEqual(grad(model.blocks[4]), 0)   # teacher is stopped
        self.assertEqual(grad(model.blocks[3]), 0)   # downstream of the student only via FM

    def test_training_step_all_losses(self):
        for heads, kind in itertools.product(('mean', 'per-head'), ('l1', 'js', 'kl')):
            with self.subTest(heads=heads, kind=kind):
                model = make_dense(attn_sync_heads=heads, attn_sync_loss=kind).train()
                criterion = FlowMatchingLoss(use_attn_sync=True, attn_sync_weight=1.0,
                                             attn_sync_heads=heads, attn_sync_loss=kind)
                x, _, y = inputs()
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    result = criterion(model, x, {'y': y})
                self.assertGreater(result['attn_sync'].item(), 0)
                self.assertEqual(set(result['attn_sync_per_block']), {1})
                result['total'].backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters() if p.requires_grad))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA autocast weight-cache regression')
    def test_cuda_autocast_teacher_keeps_block_gradients(self):
        model = make_dense(attn_sync_student_blocks=[1], attn_sync_teacher_block=3).cuda().train()
        criterion = FlowMatchingLoss(use_attn_sync=True, attn_sync_weight=1.0)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result = criterion(model, torch.randn(4, 4, 8, 8, device='cuda'),
                               {'y': torch.tensor([1, 2, 3, 4], device='cuda')})
        result['total'].backward()
        self.assertTrue(all(p.grad is not None for p in model.blocks[3].parameters()))

    def test_ratio_eval_and_zero_weight(self):
        model = make_dense(attn_sync_ratio=0.5).train()
        _, features = model(*inputs())
        self.assertEqual(features['attn_sync']['teacher'].shape[0], 2)
        out = model.eval()(*inputs(2))
        self.assertIsNone(out[1])
        off = make_dense(attn_sync_weight=0.0).train()
        self.assertIsNone(off(*inputs())[1])


if __name__ == '__main__':
    unittest.main()
