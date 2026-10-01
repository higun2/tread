# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Generates paired samples from a pre-trained SiT model using DDP.
For each pair:
1) sample from base noise z
2) sample from slightly perturbed noise z + eps, where eps ~ N(0, perturb_std^2)
"""
import torch
import torch.distributed as dist
from models.sit import SiT_models
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from samplers import euler_sampler, euler_maruyama_sampler
from utils import load_legacy_checkpoints, download_model

def decode_to_uint8(vae, latents, latents_scale, latents_bias):
    samples = vae.decode((latents - latents_bias) / latents_scale).sample
    samples = (samples + 1) / 2.
    samples = torch.clamp(255. * samples, 0, 255)
    samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return samples


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:cd
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = True,
        z_dims = [int(z_dim) for z_dim in args.projector_embed_dims.split(',')],
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    ).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt
    if ckpt_path is None:
        args.ckpt = 'SiT-XL-2-256x256.pt'
        assert args.model == 'SiT-XL/2'
        assert len(args.projector_embed_dims.split(',')) == 1
        assert int(args.projector_embed_dims.split(',')[0]) == 768
        state_dict = download_model('last.pt')
    else:
        state_dict = torch.load(ckpt_path, map_location=f'cuda:{device}')['ema']
    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
            )
    model.load_state_dict(state_dict)
    model.eval()  # important!
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.resolution}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.mode}-pairs-{args.num_pairs}-pstd-{args.perturb_std}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many pairs we need to generate on each GPU:
    total_pairs = args.num_pairs
    world_size = dist.get_world_size()
    pairs_needed_this_gpu = int(math.ceil(total_pairs / world_size))
    total_pairs_rounded = pairs_needed_this_gpu * world_size

    if rank == 0:
        print(f"Total number of pairs that will be sampled: {total_pairs}")
        print(f"Total number of images that will be sampled: {total_pairs * 2}")
        if total_pairs_rounded != total_pairs:
            print(f"Rounded pairs for DDP balance: {total_pairs_rounded} (extra pairs are discarded)")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"projector Parameters: {sum(p.numel() for p in model.projectors.parameters()):,}")

    pbar = range(pairs_needed_this_gpu)
    pbar = tqdm(pbar) if rank == 0 else pbar

    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215, ]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = -torch.tensor(
        [0., 0., 0., 0.,]
        ).view(1, 4, 1, 1).to(device)

    for i in pbar:
        pair_index = i * world_size + rank
        if pair_index >= total_pairs:
            continue

        # Sample inputs:
        z = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)

        # z_perturbed_mean = torch.abs(z)
        # z_perturbed_mean = torch.clamp(z_perturbed_mean, min=0, max=3)
        # delta = z_perturbed_mean * args.perturb_std
        # z_perturbed = z + torch.randn(z.shape, device=z.device) * delta

        # z_perturbed = z

        z_perturbed = (1 - args.perturb_std) * z + args.perturb_std * torch.randn_like(z)

        if args.class_id is None:
            y = torch.randint(0, args.num_classes, (1,), device=device)
        else:
            y = torch.tensor([args.class_id], device=device)

        # Sample images:
        sampling_kwargs = dict(
            model=model,
            y=y,
            num_steps=args.num_steps,
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
        )
        with torch.no_grad():
            sampling_kwargs["latents"] = z
            if args.mode == "sde":
                sample_base = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
            elif args.mode == "ode":
                sample_base = euler_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError()

            sampling_kwargs["latents"] = z_perturbed
            if args.mode == "sde":
                sample_perturbed = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
            elif args.mode == "ode":
                sample_perturbed = euler_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError()

            image_base = decode_to_uint8(vae, sample_base, latents_scale, latents_bias)[0]
            image_perturbed = decode_to_uint8(vae, sample_perturbed, latents_scale, latents_bias)[0]
            image_pair = np.concatenate([image_base, image_perturbed], axis=1)

            Image.fromarray(image_base).save(f"{sample_folder_dir}/{pair_index:06d}_base.png")
            Image.fromarray(image_perturbed).save(f"{sample_folder_dir}/{pair_index:06d}_perturbed.png")
            Image.fromarray(image_pair).save(f"{sample_folder_dir}/{pair_index:06d}_pair.png")

    # Make sure all processes have finished saving their samples.
    dist.barrier()
    if rank == 0:
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)

    # vae
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")

    # number of pairs (each pair = base image + perturbed image)
    parser.add_argument("--num-pairs", type=int, default=1)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--perturb-std", type=float, default=1e-3)

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--projector-embed-dims", type=str, default="768,1024")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)

    # will be deprecated
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False) # only for ode


    args = parser.parse_args()
    main(args)


'''
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 generate_single_noise_perturb.py \
  --model SiT-XL/2 \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --mode=ode \
  --num-steps=250 \
  --cfg-scale=1.8 \
  --guidance-high=0.7 \
  --num-pairs=2 \
  --sample-dir=./aaa \
  --perturb-std=0.1
'''

'''
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 generate_single_noise_perturb.py \
  --model SiT-XL/2 \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --mode=ode \
  --num-steps=10 \
  --cfg-scale=1.8 \
  --guidance-high=0.7 \
  --num-pairs=2 \
  --sample-dir=./aaa \
  --perturb-std=0.1
'''
