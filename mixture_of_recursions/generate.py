"""Distributed sampling of MoR SiT checkpoints, restoring saved architecture."""

from __future__ import annotations

import argparse
from datetime import timedelta
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from tqdm import tqdm

from .model import SiT_models
from .config import MOR_DEFAULTS
from samplers import euler_maruyama_sampler, euler_sampler
from utils import load_legacy_checkpoints


def create_npz_from_sample_folder(sample_dir: Path, num: int = 50_000) -> Path:
    """Build the ADM-compatible NPZ used by the repository's FID pipeline."""
    samples = []
    for index in tqdm(range(num), desc="Building .npz file from samples"):
        with Image.open(sample_dir / f"{index:06d}.png") as image:
            samples.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    samples = np.stack(samples)
    if samples.shape != (num, samples.shape[1], samples.shape[2], 3):
        raise RuntimeError(f"Unexpected sample array shape: {samples.shape}")
    # Append instead of Path.with_suffix(): folder names contain decimal dots
    # such as ``cfg-1.0`` and do not have a real filename suffix.
    npz_path = Path(f"{sample_dir}.npz")
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def _saved_arg(checkpoint: dict, name: str, fallback):
    """Read an argument saved either as a dict or an argparse Namespace."""
    saved = checkpoint.get("args", {})
    if isinstance(saved, dict):
        return saved.get(name, fallback)
    return getattr(saved, name, fallback)


def _resolve(value, checkpoint: dict, name: str, fallback):
    return _saved_arg(checkpoint, name, fallback) if value is None else value


def _checkpoint_name(path: Path) -> str:
    if path.parent.name == "checkpoints" and path.parent.parent.name:
        return f"{path.parent.parent.name}-{path.stem}"
    return path.stem


def resolve_mor_inference_config(checkpoint: dict, capacity_ratios=None):
    """Restore MoR topology while allowing an inference-only capacity override."""
    config = {
        name: _saved_arg(checkpoint, name, default)
        for name, default in MOR_DEFAULTS.items()
    }
    trained_ratios = tuple(float(value) for value in config["mor_capacity_ratios"])
    if capacity_ratios is not None:
        if not config["use_mor"]:
            raise ValueError("--mor-capacity-ratios requires a checkpoint trained with MoR")
        config["mor_capacity_ratios"] = tuple(float(value) for value in capacity_ratios)
    else:
        config["mor_capacity_ratios"] = trained_ratios
    return config, trained_ratios


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed generation requires at least one CUDA GPU")
    if args.cfg_scale < 1.0:
        raise ValueError("--cfg-scale must be >= 1.0")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    dist.init_process_group("nccl", timeout=timedelta(minutes=args.dist_timeout_min))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Load on CPU first. Each rank constructs the model directly on its own GPU.
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if args.state_key not in checkpoint:
        raise KeyError(
            f"Checkpoint has no {args.state_key!r} state; available keys: "
            f"{sorted(checkpoint.keys())}"
        )

    model_name = _resolve(args.model, checkpoint, "model", "SiT-B/2")
    resolution = _resolve(args.resolution, checkpoint, "resolution", 256)
    num_classes = _resolve(args.num_classes, checkpoint, "num_classes", 1000)
    fused_attn = _resolve(args.fused_attn, checkpoint, "fused_attn", True)
    qk_norm = _resolve(args.qk_norm, checkpoint, "qk_norm", False)
    cfg_prob = float(_saved_arg(checkpoint, "cfg_prob", 0.1))
    path_type = _resolve(args.path_type, checkpoint, "path_type", "linear")
    mor_config, trained_capacity_ratios = resolve_mor_inference_config(
        checkpoint, args.mor_capacity_ratios
    )

    if model_name not in SiT_models:
        raise ValueError(f"Unknown model {model_name!r}; choose from {list(SiT_models)}")
    if resolution not in (256, 512):
        raise ValueError(f"Unsupported resolution: {resolution}")
    if args.cfg_scale > 1.0 and cfg_prob <= 0:
        raise ValueError(
            "CFG sampling requires a checkpoint trained with --cfg-prob > 0 "
            "so that the null-class embedding exists"
        )

    if args.cfg_scale > 1 and num_classes != 1000:
        raise ValueError("The repository CFG samplers require num_classes=1000")

    latent_size = resolution // 8
    model = SiT_models[model_name](
        input_size=latent_size,
        num_classes=num_classes,
        use_cfg=cfg_prob > 0,
        class_dropout_prob=cfg_prob,
        fused_attn=fused_attn,
        qk_norm=qk_norm,
        **mor_config,
    ).to(device)
    state_dict = checkpoint[args.state_key]
    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
        )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    del checkpoint, state_dict

    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{args.vae}"
    ).to(device).eval()

    model_string = model_name.replace("/", "-")
    folder_name = (
        f"{model_string}-{_checkpoint_name(checkpoint_path)}-size-{resolution}-"
        f"vae-{args.vae}-cfg-{args.cfg_scale}-glow-{args.guidance_low}-"
        f"ghigh-{args.guidance_high}-seed-{args.global_seed}-{args.mode}"
    )
    if args.mor_capacity_ratios is not None:
        capacity_tag = "-".join(f"{ratio:g}" for ratio in mor_config["mor_capacity_ratios"])
        folder_name += f"-mor-cap-{capacity_tag}"
    sample_folder = Path(args.sample_dir).expanduser() / folder_name
    if rank == 0:
        sample_folder.mkdir(parents=True, exist_ok=True)
        print(f"Saving .png samples at {sample_folder}")
        print(
            "Resolved checkpoint config: "
            f"model={model_name}, resolution={resolution}, classes={num_classes}, "
            f"fused_attn={fused_attn}, qk_norm={qk_norm}, path_type={path_type}, "
            f"state={args.state_key}, trained_mor_capacity={trained_capacity_ratios}, "
            f"inference_mor_capacity={mor_config['mor_capacity_ratios']}"
        )
    dist.barrier()

    local_batch = args.per_proc_batch_size
    global_batch = local_batch * world_size
    total_samples = math.ceil(args.num_fid_samples / global_batch) * global_batch
    iterations = total_samples // global_batch
    if rank == 0:
        print(
            f"Generating {args.num_fid_samples} requested images "
            f"({total_samples} distributed slots)."
        )
        print(f"SiT parameters: {sum(p.numel() for p in model.parameters()):,}")
    iterator = tqdm(range(iterations), desc="Sampling") if rank == 0 else range(iterations)

    latent_scale = torch.full((1, 4, 1, 1), 0.18215, device=device)
    for iteration in iterator:
        noise = torch.randn(
            local_batch, model.in_channels, latent_size, latent_size, device=device
        )
        labels = torch.randint(0, num_classes, (local_batch,), device=device)
        sampling_kwargs = {
            "model": model,
            "latents": noise,
            "y": labels,
            "num_steps": args.num_steps,
            "heun": args.heun,
            "cfg_scale": args.cfg_scale,
            "guidance_low": args.guidance_low,
            "guidance_high": args.guidance_high,
            "path_type": path_type,
        }
        if args.mode == "sde":
            latents = euler_maruyama_sampler(**sampling_kwargs)
        else:
            latents = euler_sampler(**sampling_kwargs)

        images = vae.decode(latents.float() / latent_scale).sample
        images = images.add(1).div(2).mul(255).clamp(0, 255)
        images = images.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
        global_offset = iteration * global_batch
        for local_index, image in enumerate(images):
            index = global_offset + local_index * world_size + rank
            # The final distributed batch can be larger than the requested count.
            if index < args.num_fid_samples:
                Image.fromarray(image).save(sample_folder / f"{index:06d}.png")

    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder, args.num_fid_samples)
        print("Done.")
    dist.destroy_process_group()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Training checkpoint path")
    parser.add_argument("--state-key", choices=["ema", "model"], default="ema",
                        help="Sample EMA weights by default")
    parser.add_argument("--sample-dir", default="samples")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)

    # None means: restore the value saved by mixture_of_recursions.train.
    parser.add_argument("--model", choices=list(SiT_models), default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=None)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--mor-capacity-ratios", type=float, nargs="+", default=None,
        help="Inference-only capacity ratios; defaults to the values saved in the checkpoint",
    )

    parser.add_argument("--vae", choices=["ema", "mse"], default="ema")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--mode", choices=["ode", "sde"], default="ode")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--path-type", choices=["linear", "cosine"], default=None)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)

    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--encoder-depth", type=int, default=8,
                        help="Only used by the repository's legacy checkpoint converter")
    parser.add_argument("--dist-timeout-min", type=int, default=120)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())

'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --standalone --nproc_per_node=8 \
  -m mixture_of_recursions.generate \
  --ckpt /v/mnt/GH/SiT/mor2a/checkpoints/0400000.pt \
  --sample-dir /v/mnt/GH/samples \
  --num-fid-samples 50000 --per-proc-batch-size 32 \
  --mode sde --num-steps 250 --cfg-scale 1.0 --mor-capacity-ratios 1.0 1.0 1.0

'''
