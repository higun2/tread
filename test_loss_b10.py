"""CPU unit tests for the ParityFlow prototype."""

import unittest

import torch
import torch.nn as nn

from loss_b10 import (
    SILoss,
    compute_clean_parity,
    compute_syndrome,
    create_orthonormal_parity_matrix,
    patchify_state,
    project_to_parity_constraint,
    unpatchify_state,
)
from train_b4 import (
    ParityFlowWrapper,
    parityflow_euler_maruyama_sampler,
    parityflow_euler_sampler,
)


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = (2, 2)
        self.num_patches = 4
        self.proj = nn.Conv2d(2, 8, 2, 2)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class YEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(5, 8)

    def forward(self, y, training, force_drop_ids=None):
        return self.table(y)


class Block(nn.Module):
    def forward(self, x, c):
        return x + c.unsqueeze(1)


class Final(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x, c):
        return self.linear(x + c.unsqueeze(1))


class TinySiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = self.out_channels = 2
        self.num_classes = 4
        self.x_embedder = PatchEmbed()
        self.pos_embed = nn.Parameter(torch.zeros(1, 4, 8), requires_grad=False)
        self.t_linear = nn.Linear(1, 8)
        self.y_embedder = YEmbed()
        self.blocks = nn.ModuleList([Block()])
        self.final_layer = Final()

    def t_embedder(self, t):
        return self.t_linear(t[:, None])

    def unpatchify(self, tokens):
        return unpatchify_state(tokens, 2, 2, 4, 4)


class CountingParityWrapper(ParityFlowWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return super().forward(*args, **kwargs)


class ParityFlowTests(unittest.TestCase):
    def test_parity_matrix_is_deterministic_and_orthonormal(self):
        a = create_orthonormal_parity_matrix(3, 7, 19)
        b = create_orthonormal_parity_matrix(3, 7, 19)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.allclose(a @ a.T, torch.eye(3), atol=1e-6))

    def test_patchify_clean_parity_and_unpatchify(self):
        x = torch.randn(2, 2, 4, 4)
        z = patchify_state(x, 2)
        a = create_orthonormal_parity_matrix(2, 4, 0)
        p = compute_clean_parity(a, z)
        self.assertEqual(z.shape, (2, 4, 8))
        self.assertEqual(p.shape, (2, 2, 8))
        self.assertTrue(torch.equal(x, unpatchify_state(z, 2, 2, 4, 4)))

    def test_half_projection_enforces_syndrome(self):
        a = create_orthonormal_parity_matrix(2, 5, 1)
        z, p = torch.randn(3, 5, 4), torch.randn(3, 2, 4)
        before = compute_syndrome(p, z, a).square().sum()
        zc, pc, _ = project_to_parity_constraint(z, p, a, torch.full((3,), 0.5))
        after = compute_syndrome(pc, zc, a).square().sum()
        self.assertLess(after.item(), 1e-10)
        self.assertLessEqual(after.item(), before.item())

    def test_model_shapes_one_forward_and_gradients(self):
        model = CountingParityWrapper(TinySiT(), 2, matrix_seed=3)
        loss_fn = SILoss(parityflow_enabled=True)
        images = torch.randn(3, 2, 4, 4)
        labels = torch.tensor([0, 1, 2])
        image_loss, parity_loss, syndrome_loss, _ = loss_fn(model, images, {"y": labels})
        total = image_loss.mean() + parity_loss + syndrome_loss
        total.backward()
        self.assertEqual(model.calls, 1)
        self.assertTrue(torch.isfinite(total))
        self.assertIsNotNone(model.parity_embed.weight.grad)

    def test_buffer_and_checkpoint_round_trip(self):
        model = ParityFlowWrapper(TinySiT(), 2, matrix_seed=5)
        self.assertIn("parity_matrix", dict(model.named_buffers()))
        self.assertNotIn("parity_matrix", dict(model.named_parameters()))
        clone = ParityFlowWrapper(TinySiT(), 2, matrix_seed=99)
        clone.load_state_dict(model.state_dict())
        self.assertTrue(torch.equal(model.parity_matrix, clone.parity_matrix))

    def test_signal_ratio_endpoints(self):
        loss_fn = SILoss(parityflow_enabled=True)
        t = torch.tensor([0.0, 1.0]).view(2, 1, 1, 1)
        alpha, sigma, _, _ = loss_fn.interpolant(t)
        weight = loss_fn.schedule(alpha, sigma)
        self.assertGreater(weight[0].item(), 0.999)
        self.assertLess(weight[1].item(), 1e-7)

    def test_joint_sampler_cfg_and_correction_smoke(self):
        model = ParityFlowWrapper(TinySiT(), 2, matrix_seed=7).eval()
        loss_fn = SILoss(parityflow_enabled=True)
        output = parityflow_euler_sampler(
            model,
            torch.randn(2, 2, 4, 4),
            torch.tensor([0, 1]),
            loss_fn,
            num_steps=2,
            heun=True,
            cfg_scale=1.5,
            correction_enabled=True,
        )
        self.assertEqual(output.shape, (2, 2, 4, 4))
        self.assertTrue(torch.isfinite(output).all())

    def test_joint_sde_cfg_correction_smoke(self):
        model = CountingParityWrapper(TinySiT(), 2, matrix_seed=8).eval()
        loss_fn = SILoss(parityflow_enabled=True)
        output = parityflow_euler_maruyama_sampler(
            model,
            torch.randn(2, 2, 4, 4),
            torch.tensor([0, 1]),
            loss_fn,
            num_steps=3,
            cfg_scale=1.5,
            correction_enabled=True,
        )
        self.assertEqual(output.shape, (2, 2, 4, 4))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(model.calls, 3)


if __name__ == "__main__":
    unittest.main()
