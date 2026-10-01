# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models.sit_b1 import SiT_models
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from datetime import timedelta
from samplers import euler_sampler, euler_maruyama_sampler
from loss_b10 import SILoss
from train_b4 import (
    ParityFlowWrapper,
    parityflow_euler_maruyama_sampler,
    parityflow_euler_sampler,
)


def checkpoint_arg(checkpoint, name, default=None):
    """Read a saved argparse Namespace or dict without requiring either format."""
    saved_args = checkpoint.get("args")
    if isinstance(saved_args, dict):
        return saved_args.get(name, default)
    return getattr(saved_args, name, default) if saved_args is not None else default

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:cd
    dist.init_process_group("nccl", timeout=timedelta(minutes=args.dist_timeout_min))
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Recreate exactly the architecture recorded by train_b4.py.
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.state_key not in checkpoint:
        raise KeyError(f"Checkpoint has no '{args.state_key}' state. Available: {list(checkpoint)}")
    state_dict = checkpoint[args.state_key]
    model_name = args.model or checkpoint_arg(checkpoint, "model")
    resolution = args.resolution or checkpoint_arg(checkpoint, "resolution", 256)
    num_classes = args.num_classes or checkpoint_arg(checkpoint, "num_classes", 1000)
    cfg_prob = checkpoint_arg(checkpoint, "cfg_prob", 0.1)
    fused_attn = args.fused_attn if args.fused_attn is not None else checkpoint_arg(checkpoint, "fused_attn", True)
    qk_norm = args.qk_norm if args.qk_norm is not None else checkpoint_arg(checkpoint, "qk_norm", False)
    path_type = args.path_type or checkpoint_arg(checkpoint, "path_type", "linear")
    parityflow = checkpoint_arg(
        checkpoint, "parityflow", any(key.startswith("base_model.") for key in state_dict)
    )
    parity_ablation = args.parity_ablation or checkpoint_arg(checkpoint, "parity_ablation", "parity")
    num_parity_tokens = args.num_parity_tokens or checkpoint_arg(checkpoint, "num_parity_tokens", 8)
    matrix_seed = args.parity_matrix_seed
    if matrix_seed is None:
        matrix_seed = checkpoint_arg(checkpoint, "parity_matrix_seed", 0)
    schedule_eps = args.parity_schedule_eps
    if schedule_eps is None:
        schedule_eps = checkpoint_arg(checkpoint, "parity_schedule_eps", 1e-8)
    if model_name not in SiT_models:
        raise ValueError(f"Unknown or missing model name: {model_name}")
    if args.cfg_scale > 1.0 and cfg_prob <= 0:
        raise ValueError("CFG requires a checkpoint trained with cfg_prob > 0")
    if args.parity_correction and (not parityflow or parity_ablation != "parity"):
        raise ValueError("Parity correction requires a stochastic ParityFlow checkpoint")

    block_kwargs = {"fused_attn": fused_attn, "qk_norm": qk_norm}
    latent_size = resolution // 8
    model = SiT_models[model_name](
        input_size=latent_size,
        num_classes=num_classes,
        use_cfg=(cfg_prob > 0),
        class_dropout_prob=cfg_prob,
        **block_kwargs,
    )
    if parityflow:
        model = ParityFlowWrapper(
            model,
            num_parity_tokens=num_parity_tokens,
            matrix_seed=matrix_seed,
            control_only=parity_ablation == "learned_tokens",
        )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()  # important!
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"

    # Create folder to save samples:
    model_string_name = model_name.replace("/", "-")
    if args.ckpt:
        ckpt_basename = os.path.basename(args.ckpt).replace(".pt", "")
        ckpt_parent = os.path.basename(os.path.dirname(args.ckpt))
        ckpt_grandparent = os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))
        if ckpt_parent == "checkpoints" and ckpt_grandparent:
            ckpt_string_name = f"{ckpt_grandparent}-{ckpt_basename}"
        else:
            ckpt_string_name = ckpt_basename
    else:
        ckpt_string_name = "pretrained"
    method_name = f"parity-{num_parity_tokens}-{parity_ablation}" if parityflow else "baseline"
    correction_name = f"corr-{args.parity_correction_gamma}" if args.parity_correction else "nocorr"
    folder_name = f"{model_string_name}-{ckpt_string_name}-{method_name}-{correction_name}-size-{resolution}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-glow-{args.guidance_low}-ghigh-{args.guidance_high}-" \
                  f"seed-{args.global_seed}-{args.mode}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    # sample_folder_dir = f"/mnt/GH/{args.sample_dir}/{folder_name}"
    # sample_folder_dir = f"/v/mnt/GH/{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, num_classes, (n,), device=device)

        # Sample images:
        sampling_kwargs = dict(
            model=model,
            latents=z,
            y=y,
            num_steps=args.num_steps,
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=path_type,
        )
        with torch.no_grad():
            if parityflow and parity_ablation == "parity":
                loss_fn = SILoss(
                    path_type=path_type,
                    parityflow_enabled=True,
                    schedule_eps=schedule_eps,
                )
                parity_kwargs = dict(
                    model=model, latents=z, y=y, loss_fn=loss_fn,
                    num_steps=args.num_steps, cfg_scale=args.cfg_scale,
                    guidance_low=args.guidance_low,
                    guidance_high=args.guidance_high,
                    correction_enabled=args.parity_correction,
                    correction_gamma_max=args.parity_correction_gamma,
                    correction_schedule=args.parity_correction_schedule,
                )
                if args.mode == "sde":
                    samples = parityflow_euler_maruyama_sampler(
                        **parity_kwargs
                    ).to(torch.float32)
                else:
                    samples = parityflow_euler_sampler(
                        **parity_kwargs, heun=args.heun
                    ).to(torch.float32)
            elif args.mode == "sde":
                samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
            elif args.mode == "ode":
                samples = euler_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError()

            latents_scale = torch.tensor(
                [0.18215, 0.18215, 0.18215, 0.18215, ]
                ).view(1, 4, 1, 1).to(device)
            latents_bias = -torch.tensor(
                [0., 0., 0., 0.,]
                ).view(1, 4, 1, 1).to(device)
            samples = vae.decode((samples -  latents_bias) / latents_scale).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a train_b4 checkpoint.")
    parser.add_argument("--state-key", choices=["ema", "model"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default=None)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=None)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)

    # vae
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")

    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    # sampling related hyperparameters
    parser.add_argument("--mode", choices=["ode", "sde"], default="ode")
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)

    # ParityFlow architecture values default to the checkpoint's saved args.
    parser.add_argument("--num-parity-tokens", type=int, default=None)
    parser.add_argument("--parity-matrix-seed", type=int, default=None)
    parser.add_argument("--parity-ablation", choices=["parity", "learned_tokens"], default=None)
    parser.add_argument("--parity-correction", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--parity-correction-gamma", type=float, default=0.5)
    parser.add_argument("--parity-correction-schedule", choices=["constant", "signal_ratio"], default="signal_ratio")
    parser.add_argument("--parity-schedule-eps", type=float, default=None)
    parser.add_argument("--dist-timeout-min", type=int, default=120,
                        help="Process group timeout in minutes. Increase for slow filesystems or long post-processing.")


    args = parser.parse_args()
    if not 0 <= args.parity_correction_gamma <= 0.5:
        parser.error("--parity-correction-gamma must be in [0, 0.5]")
    if args.num_parity_tokens is not None and args.num_parity_tokens <= 0:
        parser.error("--num-parity-tokens must be positive")
    main(args)

'''
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --standalone generate3.py \
  --ckpt /v/mnt/GH/SiT/parityflow-v1/checkpoints/0400000.pt \
  --sample-dir /v/mnt/GH/samples \
  --num-fid-samples 50000 \
  --per-proc-batch-size 32 \
  --mode ode \
  --num-steps 250 \
  --cfg-scale 1.0


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --standalone generate3.py \
  --ckpt /v/mnt/GH/SiT/parityflow-v1/checkpoints/0400000.pt \
  --sample-dir /v/mnt/GH/samples \
  --num-fid-samples 50000 \
  --per-proc-batch-size 32 \
  --mode ode \
  --num-steps 250 \
  --cfg-scale 1.0 \
  --parity-correction \
  --parity-correction-gamma 0.5

'''
