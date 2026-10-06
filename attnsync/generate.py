"""Distributed generation for TREAD + RouteSync checkpoints."""

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

from samplers import euler_maruyama_sampler, euler_sampler
from utils import load_legacy_checkpoints
from .config import TREAD_DEFAULTS
from .model import SiT_models


def create_npz_from_sample_folder(sample_dir: Path, num: int = 50_000) -> Path:
    samples = []
    for index in tqdm(range(num), desc="Building .npz file from samples"):
        with Image.open(sample_dir / f"{index:06d}.png") as image:
            samples.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    samples = np.stack(samples)
    npz_path = Path(f"{sample_dir}.npz")
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def _saved_arg(checkpoint, name, fallback):
    saved = checkpoint.get("args", {})
    if isinstance(saved, dict):
        return saved.get(name, fallback)
    return getattr(saved, name, fallback)


def _resolve(value, checkpoint, name, fallback):
    return _saved_arg(checkpoint, name, fallback) if value is None else value


def _checkpoint_name(path: Path) -> str:
    if path.parent.name == "checkpoints" and path.parent.parent.name:
        return f"{path.parent.parent.name}-{path.stem}"
    return path.stem


def resolve_tread_inference_config(checkpoint, eval_mode=None, routing_seed=None):
    """Restore training topology with inference-only mode and seed overrides."""
    config = {
        name: _saved_arg(checkpoint, name, default)
        for name, default in TREAD_DEFAULTS.items()
    }
    trained_eval_mode = config["tread_eval_mode"]
    if eval_mode is not None:
        if not config["use_tread_routing"] and eval_mode == "sparse":
            raise ValueError(
                "--tread-eval-mode sparse requires a checkpoint trained with "
                "--use-tread-routing"
            )
        config["tread_eval_mode"] = eval_mode
    if routing_seed is not None:
        config["tread_seed"] = routing_seed
    return config, trained_eval_mode


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
    path_type = _resolve(args.path_type, checkpoint, "path_type", "linear")
    cfg_prob = float(_saved_arg(checkpoint, "cfg_prob", 0.1))
    routing_config, trained_eval_mode = resolve_tread_inference_config(
        checkpoint, args.tread_eval_mode, args.tread_seed,
    )

    if model_name not in SiT_models:
        raise ValueError(f"Unknown model {model_name!r}; choose from {list(SiT_models)}")
    if args.cfg_scale > 1.0 and cfg_prob <= 0:
        raise ValueError("CFG requires a checkpoint trained with --cfg-prob greater than zero")
    if args.cfg_scale > 1.0 and num_classes != 1000:
        raise ValueError("The repository CFG samplers require num_classes=1000")

    latent_size = resolution // 8
    model = SiT_models[model_name](
        input_size=latent_size, num_classes=num_classes,
        use_cfg=cfg_prob > 0, class_dropout_prob=cfg_prob,
        fused_attn=fused_attn, qk_norm=qk_norm, path_type=path_type,
        **routing_config,
    ).to(device)
    state_dict = checkpoint[args.state_key]
    if args.legacy:
        state_dict = load_legacy_checkpoints(state_dict, encoder_depth=args.encoder_depth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    del checkpoint, state_dict

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device).eval()
    route_mode = routing_config["tread_eval_mode"]
    folder_name = (
        f"{model_name.replace('/', '-')}-{_checkpoint_name(checkpoint_path)}-size-{resolution}-"
        f"vae-{args.vae}-cfg-{args.cfg_scale}-glow-{args.guidance_low}-"
        f"ghigh-{args.guidance_high}-seed-{args.global_seed}-{args.mode}-"
        f"tread-{route_mode}-ratio-{routing_config['tread_active_ratio']}-"
        f"recursive-{int(routing_config['tread_recursive'])}-"
        f"pattern-{routing_config['tread_recursive_pattern']}-"
        f"depth-emb-{int(routing_config['tread_depth_embedding'])}"
    )
    sample_folder = Path(args.sample_dir).expanduser() / folder_name
    if rank == 0:
        sample_folder.mkdir(parents=True, exist_ok=True)
        print(f"Saving .png samples at {sample_folder}")
        print(
            "Resolved config: "
            f"model={model_name}, resolution={resolution}, path_type={path_type}, "
            f"route_range=[{routing_config['tread_start_block']}, "
            f"{routing_config['tread_end_block']}), "
            f"active_ratio={routing_config['tread_active_ratio']}, "
            f"recursive={routing_config['tread_recursive']}, "
            f"groups={routing_config['tread_num_groups']}, "
            f"recursive_pattern={routing_config['tread_recursive_pattern']}, "
            f"depth_embedding={routing_config['tread_depth_embedding']}, "
            f"checkpoint_eval_mode={trained_eval_mode}, inference_eval_mode={route_mode}"
            f", routing_seed={routing_config['tread_seed']}"
        )
    dist.barrier()

    local_batch = args.per_proc_batch_size
    global_batch = local_batch * world_size
    total_samples = math.ceil(args.num_fid_samples / global_batch) * global_batch
    iterations = total_samples // global_batch
    iterator = tqdm(range(iterations), desc="Sampling") if rank == 0 else range(iterations)
    latent_scale = torch.full((1, 4, 1, 1), 0.18215, device=device)

    for iteration in iterator:
        noise = torch.randn(local_batch, model.in_channels, latent_size, latent_size, device=device)
        labels = torch.randint(0, num_classes, (local_batch,), device=device)
        sampling_kwargs = dict(
            model=model, latents=noise, y=labels, num_steps=args.num_steps,
            heun=args.heun, cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low, guidance_high=args.guidance_high,
            path_type=path_type,
        )
        latents = (
            euler_maruyama_sampler(**sampling_kwargs)
            if args.mode == "sde" else euler_sampler(**sampling_kwargs)
        )
        images = vae.decode(latents.float() / latent_scale).sample
        images = images.add(1).div(2).mul(255).clamp(0, 255)
        images = images.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
        global_offset = iteration * global_batch
        for local_index, image in enumerate(images):
            index = global_offset + local_index * world_size + rank
            if index < args.num_fid_samples:
                Image.fromarray(image).save(sample_folder / f"{index:06d}.png")

    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder, args.num_fid_samples)
        print("Done.")
    dist.destroy_process_group()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--state-key", choices=["ema", "model"], default="ema")
    parser.add_argument("--sample-dir", default="samples")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", choices=list(SiT_models), default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=None)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--tread-eval-mode", choices=["sparse", "dense"], default=None,
        help="Inference policy override; omitted means use the checkpoint setting",
    )
    parser.add_argument(
        "--tread-seed", type=int, default=None,
        help="Inference-only deterministic active-subset seed override",
    )
    parser.add_argument("--vae", choices=["ema", "mse"], default="ema")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--mode", choices=["ode", "sde"], default="sde")
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--path-type", choices=["linear", "cosine"], default=None)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--dist-timeout-min", type=int, default=120)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())


'''
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone --nproc_per_node=4 \
  -m attnsync.generate \
  --ckpt /v/mnt/GH/SiT/b16/checkpoints/0400000.pt \
  --sample-dir /v/mnt/GH/samples \
  --tread-eval-mode dense \
  --num-fid-samples 50000 --per-proc-batch-size 32 \
  --mode sde --num-steps 250 --cfg-scale 1.0


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --standalone --nproc_per_node=4 \
  -m attnsync.generate \
  --ckpt /v/mnt/GH/SiT/b17/checkpoints/0400000.pt \
  --sample-dir /v/mnt/GH/samples \
  --tread-eval-mode dense \
  --num-fid-samples 50000 --per-proc-batch-size 32 \
  --mode sde --num-steps 250 --cfg-scale 1.0

torchrun \
  --standalone --nproc_per_node=8 \
  -m attnsync.generate \
  --ckpt /v/mnt/GH/SiT/dense_sync/checkpoints/0400000.pt \
  --sample-dir /root/samples \
  --tread-eval-mode dense \
  --num-fid-samples 50000 --per-proc-batch-size 32 \
  --mode sde --num-steps 250 --cfg-scale 1.0
'''
