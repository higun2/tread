"""Dummy tests with optional CUDA coverage; no dataset or downloads required."""

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import json
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

from models.sit_b1 import SiT as VanillaSiT
from mixture_of_recursions.model import SiT, gather_tokens, scatter_tokens
from mixture_of_recursions.loss import FlowMatchingLoss


class DummyDecoder(nn.Module):
    def decode(self, latents):
        return SimpleNamespace(sample=latents[:, :3].tanh())


def tiny_kwargs():
    return dict(input_size=32, patch_size=2, hidden_size=32, decoder_hidden_size=32,
                depth=12, num_heads=4, mlp_ratio=2, num_classes=10,
                class_dropout_prob=0.1, qk_norm=False, fused_attn=False)


def activate(model):
    """Emulate post-startup weights without altering production zero init."""
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "adaLN_modulation"):
                nn.init.normal_(module.adaLN_modulation[-1].weight, std=0.02)
                nn.init.normal_(module.adaLN_modulation[-1].bias, std=0.02)
        nn.init.normal_(model.final_layer.linear.weight, std=0.02)


def ddp_worker(rank, store):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=2)
    try:
        for conditioning in (True, False):
            torch.manual_seed(17 + rank)
            model = SiT(**tiny_kwargs(), use_mor=True, use_recursion_conditioning=conditioning)
            activate(model)
            wrapped = DistributedDataParallel(model, find_unused_parameters=False)
            optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    loss = FlowMatchingLoss()(wrapped, torch.randn(2, 4, 32, 32), {"y": torch.tensor([1, 3])})["total"]
                loss.backward()
                for name, parameter in wrapped.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                    reference = parameter.grad.clone()
                    dist.broadcast(reference, src=0)
                    torch.testing.assert_close(reference, parameter.grad, rtol=0, atol=0)
                optimizer.step()
    finally:
        dist.destroy_process_group()


class MoRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(17)
        self.x = torch.randn(2, 4, 32, 32)
        self.t = torch.tensor([0.2, 0.7])
        self.y = torch.tensor([1, 3])

    def test_disabled_exact_initialization_checkpoint_forward_and_backward(self):
        torch.manual_seed(7)
        vanilla = VanillaSiT(**tiny_kwargs())
        rng = torch.get_rng_state()
        torch.manual_seed(7)
        baseline = SiT(**tiny_kwargs(), use_mor=False)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(list(vanilla.state_dict()), list(baseline.state_dict()))
        for key, value in vanilla.state_dict().items():
            self.assertTrue(torch.equal(value, baseline.state_dict()[key]), key)
        activate(vanilla)  # Avoid a vacuous comparison of two zero outputs.
        baseline.load_state_dict(vanilla.state_dict(), strict=True)
        for training in (False, True):
            vanilla.train(training)
            baseline.train(training)
            torch.manual_seed(53)
            a = vanilla(self.x, self.t, self.y)[0]
            torch.manual_seed(53)
            b = baseline(self.x, self.t, self.y)[0]
            self.assertTrue(torch.equal(a, b))
        a.square().mean().backward()
        b.square().mean().backward()
        for (name, p), (_, q) in zip(vanilla.named_parameters(), baseline.named_parameters()):
            if p.requires_grad:
                self.assertTrue(torch.equal(p.grad, q.grad), name)
        self.assertIsNone(baseline(self.x, self.t, self.y)[1])

    def test_nested_shapes_sharing_and_conditioning(self):
        for conditioning in (True, False):
            model = SiT(**tiny_kwargs(), use_mor=True, use_recursion_conditioning=conditioning).eval()
            self.assertEqual(len(model.routers), 2)
            self.assertTrue(all(type(router) is nn.Linear for router in model.routers))
            self.assertIsNot(model.routers[0].weight, model.routers[1].weight)
            activate(model)
            calls, ordinary = [], []
            handles = []
            for block in model.recurrent_blocks:
                handles.append(block.register_forward_pre_hook(
                    lambda module, inputs: calls.append((id(module), inputs[0].shape, inputs[1].detach()))))
            for block in [*model.prefix_blocks, *model.suffix_blocks, model.final_layer]:
                handles.append(block.register_forward_pre_hook(
                    lambda module, inputs: ordinary.append((inputs[0].shape, inputs[1].detach()))))
            output, aux = model(self.x, self.t, self.y, return_aux=True)
            self.assertEqual(output.shape, self.x.shape)
            self.assertEqual(aux["active_token_counts"], [256, 128, 64])
            self.assertEqual([entry[1] for entry in calls],
                             [torch.Size([2, k, 32]) for k in (256, 128, 64) for _ in range(2)])
            self.assertEqual(len(set(entry[0] for entry in calls)), 2)
            self.assertEqual([entry[0] for entry in calls[:2]], [entry[0] for entry in calls[2:4]])
            self.assertEqual([entry[0] for entry in calls[:2]], [entry[0] for entry in calls[4:]])
            self.assertTrue(all(entry[0] == (2, 256, 32) for entry in ordinary))
            c_base = model.t_embedder(self.t) + model.y_embedder(self.y, False)
            self.assertTrue(all(torch.equal(c_base, entry[1]) for entry in ordinary))
            for r in range(3):
                expected = c_base + model.recursion_embed.weight[r] if conditioning else c_base
                self.assertTrue(torch.equal(calls[2*r][2], expected))
                self.assertTrue(torch.equal(calls[2*r][2], calls[2*r+1][2]))
            if conditioning:
                self.assertFalse(torch.equal(calls[0][2], calls[2][2]))
                self.assertFalse(torch.equal(calls[2][2], calls[4][2]))
            else:
                self.assertIsNone(model.recursion_embed)
            for r, scores in enumerate(aux["router_scores"]):
                previous, selected = aux["selected_indices"][r:r+2]
                expected = previous.gather(1, scores.topk(aux["active_token_counts"][r+1], dim=1).indices)
                self.assertTrue(torch.equal(selected, expected))
                self.assertTrue((selected.unsqueeze(-1) == previous.unsqueeze(1)).any(-1).all())
                self.assertFalse(scores.requires_grad)
                self.assertTrue(((scores >= 0) & (scores <= 1)).all())
            for handle in handles:
                handle.remove()
            unique = sum(p.numel() for p in model.parameters())
            dense = sum(p.numel() for p in VanillaSiT(**tiny_kwargs()).parameters())
            self.assertLess(unique, dense)

    def test_global_topk_reranks_the_full_sequence(self):
        model = SiT(**tiny_kwargs(), use_mor=True, mor_global_topk=True).eval()
        router_input_shapes = []
        handles = []
        for router in model.routers:
            handles.append(router.register_forward_pre_hook(
                lambda module, inputs: router_input_shapes.append(inputs[0].shape)
            ))

        def fixed_logits(ascending):
            def hook(module, inputs, output):
                batch, tokens, _ = inputs[0].shape
                values = torch.linspace(-10, 10, tokens, device=output.device, dtype=output.dtype)
                if not ascending:
                    values = values.flip(0)
                return values.view(1, tokens, 1).expand(batch, -1, -1)
            return hook

        handles.append(model.routers[0].register_forward_hook(fixed_logits(ascending=False)))
        handles.append(model.routers[1].register_forward_hook(fixed_logits(ascending=True)))
        with torch.no_grad():
            _, aux = model(self.x, self.t, self.y, return_aux=True)
        self.assertEqual(router_input_shapes, [torch.Size([2, 256, 32])] * 2)
        self.assertEqual([scores.shape for scores in aux["router_scores"]],
                         [torch.Size([2, 256]), torch.Size([2, 256])])
        for recursion, count in ((1, 128), (2, 64)):
            expected = aux["router_scores"][recursion - 1].topk(count, dim=1).indices
            self.assertTrue(torch.equal(aux["selected_indices"][recursion], expected))
        selected_r2, selected_r3 = aux["selected_indices"][1:]
        self.assertFalse((selected_r3.unsqueeze(-1) == selected_r2.unsqueeze(1)).any(-1).any())
        for handle in handles:
            handle.remove()

        # Global selection remains differentiable through the selected scores.
        model = SiT(**tiny_kwargs(), use_mor=True, mor_global_topk=True)
        activate(model)
        FlowMatchingLoss()(model, self.x, {"y": self.y})["total"].backward()
        self.assert_routing_gradients(model)

    def test_scatter_gradcheck_and_inactive_preservation(self):
        indices = torch.tensor([[1, 3], [0, 2]])
        base = torch.randn(2, 5, 3, dtype=torch.double, requires_grad=True)
        updates = torch.randn(2, 2, 3, dtype=torch.double, requires_grad=True)
        original = base.detach().clone()
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: scatter_tokens(a, indices, b), (base, updates)))
        result = scatter_tokens(base, indices, updates)
        mask = torch.zeros(2, 5, dtype=torch.bool).scatter(1, indices, True)
        self.assertTrue(torch.equal(result[~mask], base[~mask]))
        self.assertTrue(torch.equal(base, original))
        result.sum().backward()
        self.assertTrue(torch.equal(base.grad, (~mask).unsqueeze(-1).expand_as(base).double()))
        self.assertTrue(torch.equal(updates.grad, torch.ones_like(updates)))

    def test_full_state_matches_direct_hard_routing_reference(self):
        model = SiT(**tiny_kwargs(), use_mor=True, use_recursion_conditioning=True).eval()
        activate(model)
        suffix_inputs = []
        handle = model.suffix_blocks[0].register_forward_pre_hook(
            lambda module, inputs: suffix_inputs.append(inputs[0].detach()))
        with torch.no_grad():
            output, aux = model(self.x, self.t, self.y, return_aux=True)
            full = model.x_embedder(self.x) + model.pos_embed
            c = model.t_embedder(self.t) + model.y_embedder(self.y, False)
            for block in model.prefix_blocks:
                full = block(full, c)
            for r, indices in enumerate(aux["selected_indices"]):
                active = gather_tokens(full, indices)
                before = active
                if r:
                    # Recompute the gate from the previous full state, avoiding
                    # dependence on implementation-provided diagnostic scores.
                    gate = model.routers[r - 1](before).float().sigmoid()
                for block in model.recurrent_blocks:
                    active = block(active, c + model.recursion_embed.weight[r])
                if r:
                    active = before + gate * active
                previous = full
                full = scatter_tokens(full, indices, active)
                mask = torch.zeros(2, 256, dtype=torch.bool).scatter(1, indices, True)
                self.assertTrue(torch.equal(previous[~mask], full[~mask]))
            self.assertTrue(torch.equal(full, suffix_inputs[0]))
        # Gradient-enabled forward and no_grad sampling use the same actual gate.
        torch.testing.assert_close(model(self.x, self.t, self.y)[0], output, rtol=0, atol=0)
        handle.remove()

    def assert_routing_gradients(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
        for group in [*model.recurrent_blocks, *model.routers]:
            for name, p in group.named_parameters():
                self.assertGreater(p.grad.abs().sum().item(), 0, name)
        if model.recursion_embed is not None:
            self.assertTrue((model.recursion_embed.weight.grad.abs().sum(-1) > 0).all())

    def test_outer_residual_gates_full_unit_output_including_sampling(self):
        model = SiT(**tiny_kwargs(), use_mor=True).eval()
        with torch.no_grad():
            for router, probability in zip(model.routers, (0.25, 0.75)):
                router.weight.zero_()
                router.bias.copy_(torch.logit(torch.tensor([probability])))
        # Each block adds 1: f(x)=x+2. The outer update must be
        # x + g*(x+2), including the unit's identity contribution.
        handles = [block.register_forward_hook(lambda module, inputs, output: inputs[0] + 1)
                   for block in model.recurrent_blocks]
        states = []
        handles.append(model.suffix_blocks[0].register_forward_pre_hook(
            lambda module, inputs: states.append(inputs[0].detach())))
        with torch.no_grad():
            _, aux = model(self.x, self.t, self.y, return_aux=True)
            base = model.x_embedder(self.x) + model.pos_embed
            expected = base + 2  # Prefix is identity at adaLN-Zero initialization.
            for r, probability in ((1, 0.25), (2, 0.75)):
                indices = aux["selected_indices"][r]
                selected = gather_tokens(expected, indices)
                expected = scatter_tokens(expected, indices, selected + probability * (selected + 2))
            torch.testing.assert_close(states[0], expected)
            torch.testing.assert_close(aux["router_scores"][0], torch.full((2, 256), 0.25))
            torch.testing.assert_close(aux["router_scores"][1], torch.full((2, 128), 0.75))
        model(self.x, self.t, self.y)
        torch.testing.assert_close(states[1], states[0], rtol=0, atol=0)
        for handle in handles:
            handle.remove()

    def test_no_gating_uses_full_outer_update_and_skips_routers(self):
        model = SiT(
            **tiny_kwargs(), use_mor=True, mor_capacity_ratios=(1, 1, 1),
            mor_gating=False,
        ).eval()
        self.assertTrue(all(not parameter.requires_grad for parameter in model.routers.parameters()))
        handles = [
            router.register_forward_hook(
                lambda module, inputs, output: self.fail("router must not run without gating")
            )
            for router in model.routers
        ]
        handles.extend(
            block.register_forward_hook(lambda module, inputs, output: inputs[0] + 1)
            for block in model.recurrent_blocks
        )
        states = []
        handles.append(model.suffix_blocks[0].register_forward_pre_hook(
            lambda module, inputs: states.append(inputs[0].detach())
        ))
        with torch.no_grad():
            _, aux = model(self.x, self.t, self.y, return_aux=True)
            base = model.x_embedder(self.x) + model.pos_embed
            expected = base + 2
            expected = expected + (expected + 2)
            expected = expected + (expected + 2)
            torch.testing.assert_close(states[0], expected)
        self.assertEqual(aux["active_token_counts"], [256, 256, 256])
        self.assertEqual(aux["router_scores"], [])
        self.assertEqual([indices.shape for indices in aux["selected_indices"]],
                         [torch.Size([2, 256])] * 3)
        for handle in handles:
            handle.remove()

        model = SiT(
            **tiny_kwargs(), use_mor=True, mor_capacity_ratios=(1, 1, 1),
            mor_gating=False,
        )
        activate(model)
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group(
                "gloo", init_method=f"file://{directory}/store", rank=0, world_size=1
            )
            try:
                wrapped = DistributedDataParallel(model, find_unused_parameters=False)
                for _ in range(2):
                    wrapped.zero_grad(set_to_none=True)
                    FlowMatchingLoss()(wrapped, self.x, {"y": self.y})["total"].backward()
                    for name, parameter in model.recurrent_blocks.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
            finally:
                dist.destroy_process_group()
        self.assertTrue(all(parameter.grad is None for parameter in model.routers.parameters()))

    def test_backward_fp32_and_bfloat16(self):
        for fused in (False, True):
            for bf16 in (False, True):
                kwargs = dict(tiny_kwargs(), fused_attn=fused)
                model = SiT(**kwargs, use_mor=True, use_recursion_conditioning=True)
                activate(model)
                context = torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else nullcontext()
                with torch.autograd.detect_anomaly(), context:
                    result = FlowMatchingLoss()(model, self.x, {"y": self.y})
                    result["total"].backward()
                self.assert_routing_gradients(model)

    def test_zero_initialization_warms_up_with_real_optimizer_steps(self):
        model = SiT(**tiny_kwargs(), use_mor=True, use_recursion_conditioning=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.assertEqual(torch.count_nonzero(model.final_layer.linear.weight), 0)
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            with torch.autograd.detect_anomaly():
                loss = FlowMatchingLoss()(model, self.x, {"y": self.y})["total"]
                loss.backward()
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
            if step == 0:
                self.assertEqual(model.recursion_embed.weight.grad.abs().sum(), 0)
            optimizer.step()
        self.assert_routing_gradients(model)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_backward_and_sampling(self):
        device = torch.device("cuda:0")
        x, t, y = self.x.to(device), self.t.to(device), self.y.to(device)
        for conditioning in (False, True):
            for fused in (False, True):
                for bf16 in (False, True):
                    with self.subTest(conditioning=conditioning, fused=fused, bf16=bf16):
                        model = SiT(**dict(tiny_kwargs(), fused_attn=fused), use_mor=True,
                                    use_recursion_conditioning=conditioning).to(device).eval()
                        activate(model)
                        context = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
                        with torch.autograd.detect_anomaly(), context:
                            output, aux = model(x, t, y, return_aux=True)
                            (output.float() - x).square().mean().backward()
                            with torch.no_grad():
                                sampled = model(x, t, y)[0]
                        self.assertEqual(aux["active_token_counts"], [256, 128, 64])
                        torch.testing.assert_close(output, sampled, rtol=0, atol=0)
                        self.assert_routing_gradients(model)

    def test_ddp_repeated_backward_without_unused_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group("gloo", init_method=f"file://{directory}/store", rank=0, world_size=1)
            try:
                for conditioning in (True, False):
                    model = SiT(**tiny_kwargs(), use_mor=True, use_recursion_conditioning=conditioning)
                    activate(model)
                    wrapped = DistributedDataParallel(model, find_unused_parameters=False)
                    for _ in range(2):
                        wrapped.zero_grad(set_to_none=True)
                        FlowMatchingLoss()(wrapped, self.x, {"y": self.y})["total"].backward()
                        self.assert_routing_gradients(model)
            finally:
                dist.destroy_process_group()

    def test_two_rank_bfloat16_ddp(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ddp_worker, args=(str(Path(directory) / "store"),), nprocs=2, join=True)

    def test_training_checkpoint_resume_and_config(self):
        from mixture_of_recursions.train import main, parse_args
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"use_mor": True, "batch_size": 256}))
            argv = ["--config", str(config), "--exp-name", "smoke", "--data-dir", directory,
                    "--output-dir", directory, "--batch-size", "2", "--max-train-steps", "4",
                    "--checkpointing-steps", "2", "--num-workers", "0", "--report-to", "wandb",
                    "--mixed-precision", "no", "--num-classes", "10", "--sampling-steps", "2",
                    "--sampling-batch-size", "4"]
            args = parse_args(argv)
            self.assertTrue(args.use_mor)
            self.assertFalse(args.use_recursion_conditioning)
            self.assertEqual(args.batch_size, 2)  # CLI overrides JSON.
            self.assertFalse(parse_args(argv + ["--no-use-mor"]).use_mor)
            dataset = torch.utils.data.TensorDataset(torch.randn(8, 8, 32, 32), torch.arange(8))
            def build_tiny(*unused, **kwargs):
                return SiT(**tiny_kwargs(), use_mor=kwargs["use_mor"])
            with patch("mixture_of_recursions.train.build_mor_sit", side_effect=build_tiny), \
                    patch("mixture_of_recursions.train.CustomDataset", return_value=dataset), \
                    patch("diffusers.models.AutoencoderKL.from_pretrained", return_value=DummyDecoder()) as load_vae, \
                    patch("accelerate.Accelerator.init_trackers"), \
                    patch("accelerate.Accelerator.log") as log, \
                    patch("wandb.Image", side_effect=lambda image: image):
                main(args)
                self.assertEqual(load_vae.call_count, 1)
                previews = [call for call in log.call_args_list if "samples" in call.args[0]]
                self.assertEqual([call.kwargs["step"] for call in previews], [1, 2, 4])
                self.assertEqual(previews[0].args[0]["samples"].shape, (70, 70, 3))
                self.assertEqual(previews[0].args[0]["samples"].dtype, np.uint8)
                self.assertFalse((Path(directory) / "smoke" / "samples").exists())
                self.assertFalse(list(Path(directory).rglob("*.png")))
                checkpoint_path = Path(directory) / "smoke" / "checkpoints" / "0000004.pt"
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                self.assertEqual(checkpoint["steps"], 4)
                model = build_tiny(use_mor=True).eval()
                self.assertIsNone(model.recursion_embed)
                self.assertNotIn("recursion_embed.weight", model.state_dict())
                model.load_state_dict(checkpoint["model"], strict=True)
                copy = deepcopy(model)  # EMA copies whole model, not individual recursions.
                torch.testing.assert_close(copy(self.x, self.t, self.y)[0], model(self.x, self.t, self.y)[0])
                self.assertEqual(len(copy.recurrent_blocks), 2)
                main(parse_args(argv + ["--resume-step", "4", "--max-train-steps", "6"]))
                resumed = torch.load(checkpoint_path.with_name("0000006.pt"), weights_only=False)
                self.assertEqual(resumed["steps"], 6)
                previews = [call for call in log.call_args_list if "samples" in call.args[0]]
                self.assertEqual([call.kwargs["step"] for call in previews], [1, 2, 4, 6])
                load_vae.reset_mock()
                log.reset_mock()
                main(parse_args(argv + ["--exp-name", "disabled", "--sampling-steps", "0"]))
                load_vae.assert_not_called()
                self.assertFalse(any("samples" in call.args[0] for call in log.call_args_list))
                self.assertFalse((Path(directory) / "disabled" / "samples").exists())
                disabled = torch.load(Path(directory) / "disabled" / "checkpoints" / "0000004.pt",
                                      weights_only=False)
                for name, value in checkpoint["model"].items():
                    torch.testing.assert_close(value, disabled["model"][name], rtol=0, atol=0)
                main(parse_args(argv + ["--exp-name", "no_report", "--report-to", "none"]))
                load_vae.assert_not_called()
                self.assertFalse(any("samples" in call.args[0] for call in log.call_args_list))
                self.assertFalse(list(Path(directory).rglob("*.png")))

    def test_preview_inputs_repeat_without_consuming_training_rng(self):
        from mixture_of_recursions.train import prepare_training_sample
        args = SimpleNamespace(seed=17, resolution=256, num_classes=1000)
        def load(*unused):
            torch.rand(3)  # Emulate RNG consumed during decoder construction.
            return DummyDecoder()
        with patch("diffusers.models.AutoencoderKL.from_pretrained", side_effect=load):
            state = torch.get_rng_state()
            vae, noise, labels = prepare_training_sample(
                args, torch.device("cpu"), local_batch_size=4, process_index=2
            )
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            _, repeated_noise, repeated_labels = prepare_training_sample(
                args, torch.device("cpu"), local_batch_size=4, process_index=2
            )
            self.assertTrue(torch.equal(noise, repeated_noise))
            self.assertTrue(torch.equal(labels, repeated_labels))
            self.assertEqual(noise.shape, (4, 4, 32, 32))
            _, other_noise, _ = prepare_training_sample(
                args, torch.device("cpu"), local_batch_size=4, process_index=3
            )
            self.assertFalse(torch.equal(noise, other_noise))
            self.assertFalse(vae.training)

    def test_configuration_and_single_recursion(self):
        invalid = [dict(mor_max_recursions=0), dict(mor_capacity_ratios=(1, 0.5)),
                   dict(mor_capacity_ratios=(1, 0.25, 0.5)),
                   dict(mor_capacity_ratios=(0.5, 0.5, 0.25)), dict(mor_capacity_ratios=(1, 0, 0)),
                   dict(mor_capacity_ratios=(1, float("nan"), 0.25))]
        for kwargs in invalid:
            with self.assertRaises(ValueError):
                SiT(**tiny_kwargs(), use_mor=True, **kwargs)
        with self.assertRaises(ValueError):
            SiT(**tiny_kwargs(), use_mor=True, mor_gating=False)
        kwargs = dict(tiny_kwargs(), depth=2)
        model = SiT(**kwargs, use_mor=True, mor_prefix_blocks=0, mor_suffix_blocks=0,
                    mor_max_recursions=1, mor_capacity_ratios=(1,))
        self.assertEqual(len(model.routers), 0)
        self.assertEqual(model(self.x, self.t, self.y, return_aux=True)[1]["active_token_counts"], [256])
        model = SiT(**tiny_kwargs(), use_mor=True, mor_capacity_ratios=(1, 0.003, 0.001))
        self.assertEqual(model(self.x, self.t, self.y, return_aux=True)[1]["active_token_counts"], [256, 1, 1])

    def test_custom_effective_depth_can_differ_from_named_base(self):
        model = SiT(
            **tiny_kwargs(), use_mor=True, mor_global_topk=True,
            use_recursion_conditioning=True, mor_prefix_blocks=1,
            mor_recurrent_blocks=4, mor_suffix_blocks=1, mor_max_recursions=3,
        )
        self.assertEqual(model.base_depth, 12)
        self.assertEqual(model.effective_depth, 14)
        self.assertEqual(len(model.prefix_blocks), 1)
        self.assertEqual(len(model.recurrent_blocks), 4)
        self.assertEqual(len(model.suffix_blocks), 1)
        activate(model)
        output, aux = model(self.x, self.t, self.y, return_aux=True)
        self.assertEqual(output.shape, self.x.shape)
        self.assertEqual(aux["active_token_counts"], [256, 128, 64])
        output.square().mean().backward()
        self.assert_routing_gradients(model)

    def test_generation_can_override_capacity_without_changing_checkpoint_shapes(self):
        from mixture_of_recursions.generate import parse_args, resolve_mor_inference_config

        checkpoint = {"args": {
            "use_mor": True,
            "mor_prefix_blocks": 3,
            "mor_recurrent_blocks": 2,
            "mor_suffix_blocks": 3,
            "mor_max_recursions": 3,
            "mor_capacity_ratios": [1.0, 0.5, 0.25],
            "use_recursion_conditioning": False,
            "mor_global_topk": False,
        }}
        args = parse_args(["--ckpt", "dummy.pt", "--mor-capacity-ratios", "1", "1", "1"])
        config, trained = resolve_mor_inference_config(checkpoint, args.mor_capacity_ratios)
        self.assertEqual(trained, (1.0, 0.5, 0.25))
        self.assertEqual(config["mor_capacity_ratios"], (1.0, 1.0, 1.0))

        trained_model = SiT(**tiny_kwargs(), **resolve_mor_inference_config(checkpoint)[0])
        inference_model = SiT(**tiny_kwargs(), **config)
        inference_model.load_state_dict(trained_model.state_dict(), strict=True)
        self.assertEqual(trained_model.active_token_counts, (256, 128, 64))
        self.assertEqual(inference_model.active_token_counts, (256, 256, 256))
        _, aux = inference_model(self.x, self.t, self.y, return_aux=True)
        self.assertEqual(aux["active_token_counts"], [256, 256, 256])
        self.assertEqual([scores.shape for scores in aux["router_scores"]],
                         [torch.Size([2, 256]), torch.Size([2, 256])])

        vanilla_checkpoint = {"args": {"use_mor": False}}
        with self.assertRaises(ValueError):
            resolve_mor_inference_config(vanilla_checkpoint, (1.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
