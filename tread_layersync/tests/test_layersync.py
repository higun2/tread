import argparse
import unittest

import torch

from tread_layersync.loss import FlowMatchingLayerSyncLoss, align_pair, layersync_loss
from tread_layersync.model import SiT_models
from tread_layersync.train import layersync_pairs
from attnsync.loss import FlowMatchingLoss
from attnsync.model import SiT_models as TreadModels, gather_tokens

KWARGS = dict(input_size=8, num_classes=10, use_cfg=True, class_dropout_prob=0.1,
              use_tread_routing=True, tread_start_block=3, tread_end_block=9,
              fused_attn=False, qk_norm=False, tread_fp32_endpoint=False)


def models():
    torch.manual_seed(0)
    ls = SiT_models["SiT-B/2"](**KWARGS).double()
    for p in ls.parameters():  # zero-init heads would hide differences
        p.data.normal_(0, 0.05)
    tread = TreadModels["SiT-B/2"](**KWARGS).double()
    tread.load_state_dict(ls.state_dict(), strict=True)
    return ls.train(), tread.train()


class LayerSyncTest(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(4, 4, 8, 8, dtype=torch.float64)
        self.t = torch.rand(4, dtype=torch.float64)
        self.y = torch.tensor([1, 2, 3, 4])
        self.drop = torch.zeros(4, dtype=torch.long)

    def test_capture_forward_matches_tread(self):
        ls, tread = models()
        torch.manual_seed(3)
        a, _, feats, routing = ls(self.x, self.t, self.y, force_drop_ids=self.drop,
                                  capture_blocks=[2, 5, 8, 10], return_routing=True)
        torch.manual_seed(3)
        b, _, info = tread(self.x, self.t, self.y, force_drop_ids=self.drop, return_routing_info=True)
        torch.testing.assert_close(a, b, rtol=0, atol=1e-12)
        self.assertTrue(torch.equal(routing["active_indices"], info["active_indices"]))
        self.assertIsNone(feats[2][1])
        self.assertIsNone(feats[10][1])
        self.assertTrue(torch.equal(feats[5][1], info["active_indices"]))
        self.assertEqual(feats[5][0].shape[1], info["active_indices"].shape[1])

    def test_alignment_of_pairs(self):
        ls, _ = models()
        _, _, feats = ls(self.x, self.t, self.y, force_drop_ids=self.drop, capture_blocks=[2, 5, 8, 10])
        idx = feats[5][1]
        s, t, used = align_pair(feats[2], feats[5], "all", None)  # prefix -> routed
        self.assertEqual(used, "active")
        torch.testing.assert_close(s, gather_tokens(feats[2][0], idx), rtol=0, atol=0)
        s, t, used = align_pair(feats[5], feats[8], "all", None)  # routed -> routed
        self.assertEqual(used, "active")
        s, t, used = align_pair(feats[5], feats[10], "all", None)  # routed -> suffix
        torch.testing.assert_close(t, gather_tokens(feats[10][0], idx), rtol=0, atol=0)
        s, t, used = align_pair(feats[2], feats[10], "all", None)  # prefix -> suffix, all tokens
        self.assertEqual((used, s.shape[1]), ("all", 16))

    def test_zero_weight_equals_tread_loss(self):
        ls, tread = models()
        torch.manual_seed(9)
        a = FlowMatchingLayerSyncLoss(pairs=[(4, 8)], weight=0.0)(ls, self.x, {"y": self.y})
        torch.manual_seed(9)
        b = FlowMatchingLoss()(tread, self.x, {"y": self.y})
        torch.testing.assert_close(a["total"], b["total"], rtol=0, atol=1e-12)

    def test_weighted_total_and_fm_unchanged(self):
        ls, tread = models()
        torch.manual_seed(9)
        a = FlowMatchingLayerSyncLoss(pairs=[(4, 8)], weight=0.5)(ls, self.x, {"y": self.y})
        torch.manual_seed(9)
        b = FlowMatchingLoss()(tread, self.x, {"y": self.y})
        torch.testing.assert_close(a["fm"], b["fm"], rtol=0, atol=1e-12)
        torch.testing.assert_close(a["total"], a["fm"] + 0.5 * a["layersync"], rtol=0, atol=1e-12)

    def test_teacher_is_stop_gradient(self):
        ls, _ = models()
        _, _, feats = ls(self.x, self.t, self.y, force_drop_ids=self.drop, capture_blocks=[4, 8])
        loss, _ = layersync_loss(feats, [(4, 8)])
        loss.backward()
        for index in range(5, 12):  # blocks after the student receive no LayerSync gradient
            self.assertTrue(all(p.grad is None or p.grad.abs().max() == 0
                                for p in ls.blocks[index].parameters()), index)
        self.assertTrue(any(p.grad is not None and p.grad.abs().max() > 0
                            for p in ls.blocks[4].parameters()))

    def test_pair_validation(self):
        base = dict(model="SiT-B/2", tread_start_block=3, tread_end_block=9, layersync_tokens="all")
        ok = layersync_pairs(argparse.Namespace(**base, layersync_student_blocks=[4, 2],
                                                layersync_teacher_blocks=[8, 10]))
        self.assertEqual(ok, [(4, 8), (2, 10)])
        for students, teachers in (([8], [4]), ([4], [4]), ([4], [12]), ([4, 5], [8])):
            with self.assertRaises(ValueError):
                layersync_pairs(argparse.Namespace(**base, layersync_student_blocks=students,
                                                   layersync_teacher_blocks=teachers))


if __name__ == "__main__":
    unittest.main()
