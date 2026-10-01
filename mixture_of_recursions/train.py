"""Flow-matching training for Mixture-of-Recursions SiT."""

from __future__ import annotations

import argparse
import copy
from copy import deepcopy
import json
import logging
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision.utils import make_grid

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from dataset import CustomDataset
from .loss import FlowMatchingLoss
from .model import build_mor_sit
from .config import MOR_DEFAULTS, add_mor_args, mor_kwargs


def sample_posterior(moments, scale, bias):
    mean, std = torch.chunk(moments, 2, dim=1)
    return (mean + std * torch.randn_like(mean)) * scale + bias


def array2grid(x):
    """Build the same approximately square uint8 grid used by existing trainers."""
    nrow = round(math.sqrt(x.size(0)))
    grid = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    return grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


def prepare_training_sample(args, device, local_batch_size=1, process_index=0):
    """Load a local decoder and fixed rank-specific inputs without changing training RNG."""
    from diffusers.models import AutoencoderKL

    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
        vae.requires_grad_(False)
    generator = torch.Generator(device=device).manual_seed((args.seed or 0) + process_index)
    noise = torch.randn(local_batch_size, 4, args.resolution // 8, args.resolution // 8,
                        generator=generator, device=device)
    labels = torch.randint(args.num_classes, (local_batch_size,), generator=generator, device=device)
    return vae, noise, labels


@torch.no_grad()
def generate_training_samples(ema, vae, noise, labels, path_type, cfg_scale):
    """Return a local image batch using the existing 50-step Euler sampler."""
    from samplers import euler_sampler

    latents = euler_sampler(
        ema, noise, labels, num_steps=50, cfg_scale=cfg_scale,
        guidance_low=0.0, guidance_high=1.0, path_type=path_type, heun=False,
    ).float()
    return vae.decode(latents / 0.18215).sample.float().add(1).div(2).clamp(0, 1)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    parameters = dict(model.named_parameters())
    for name, ema_parameter in ema_model.named_parameters():
        ema_parameter.mul_(decay).add_(parameters[name].detach(), alpha=1 - decay)


def set_requires_grad(model, flag):
    for parameter in model.parameters():
        parameter.requires_grad_(flag)


def random_state_dict():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def create_logger(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(directory / "log.txt")], force=True,
    )
    return logging.getLogger("mixture_of_recursions")


def main(args):
    run_dir = Path(args.output_dir) / args.exp_name
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=ProjectConfiguration(project_dir=str(run_dir), logging_dir=str(run_dir / "logs")),
    )
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    logger = create_logger(run_dir) if accelerator.is_main_process else logging.getLogger(__name__)
    checkpoint_dir = run_dir / "checkpoints"
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "args.json").open("w") as handle:
            json.dump(vars(args), handle, indent=2)

    model = build_mor_sit(
        args.model, input_size=args.resolution // 8, num_classes=args.num_classes,
        class_dropout_prob=args.cfg_prob, use_cfg=args.cfg_prob > 0,
        qk_norm=args.qk_norm, fused_attn=args.fused_attn,
        path_type=args.path_type, **mor_kwargs(args),
    ).to(accelerator.device)
    ema = deepcopy(model).to(accelerator.device).eval()
    set_requires_grad(ema, False)
    update_ema(ema, model, decay=0)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    criterion = FlowMatchingLoss(args.prediction, args.path_type, args.weighting)
    dataset = CustomDataset(args.data_dir, num_classes=args.num_classes)
    if args.batch_size % accelerator.num_processes:
        raise ValueError("global --batch-size must be divisible by the number of processes")
    local_batch = args.batch_size // accelerator.num_processes
    if args.report_to == "wandb" and args.sampling_steps > 0:
        if args.sampling_batch_size < 1:
            raise ValueError("--sampling-batch-size must be positive")
        if args.sampling_batch_size % accelerator.num_processes:
            raise ValueError("--sampling-batch-size must be divisible by the number of processes")
    dataloader = DataLoader(
        dataset, batch_size=local_batch, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )

    global_step = 0
    resume_random_state = None
    if args.resume_step > 0:
        checkpoint = torch.load(checkpoint_dir / f"{args.resume_step:07d}.pt", map_location="cpu", weights_only=False)
        # Capacity ratios/conditioning affect behavior without necessarily
        # changing state-dict shapes. Do not silently resume a different model.
        saved = checkpoint["args"]
        for name, value in mor_kwargs(args).items():
            previous = (
                saved.get(name, MOR_DEFAULTS[name])
                if isinstance(saved, dict)
                else getattr(saved, name, MOR_DEFAULTS[name])
            )
            if name == "mor_capacity_ratios":
                previous, value = tuple(previous), tuple(value)
            if previous != value:
                raise ValueError(f"Resume configuration differs for {name}: {previous} vs {value}")
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["opt"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = int(checkpoint["steps"])
        resume_random_state = checkpoint.get("random_state")

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    if resume_random_state is not None:
        restore_random_state(resume_random_state)
    if accelerator.is_main_process and args.report_to != "none":
        accelerator.init_trackers(
            args.project_name, config=vars(copy.deepcopy(args)),
            init_kwargs={"wandb": {"name": args.exp_name}},
        )
    logger.info("Parameters: %s", f"{sum(p.numel() for p in accelerator.unwrap_model(model).parameters()):,}")
    logger.info("MoR configuration: %s", mor_kwargs(args))
    if accelerator.unwrap_model(model).use_mor:
        raw_model = accelerator.unwrap_model(model)
        logger.info(
            "MoR depth: base=%d effective=%d unique=%d",
            raw_model.base_depth, raw_model.effective_depth,
            len(raw_model.prefix_blocks) + len(raw_model.recurrent_blocks) + len(raw_model.suffix_blocks),
        )

    scale = torch.full((1, 4, 1, 1), 0.18215, device=accelerator.device)
    bias = torch.zeros_like(scale)
    progress = tqdm(total=args.max_train_steps, initial=global_step, disable=not accelerator.is_local_main_process)
    grad_norm = torch.tensor(0.0, device=accelerator.device)
    preview = None
    for _epoch in range(args.epochs):
        if global_step >= args.max_train_steps:
            break
        model.train()
        for moments, labels in dataloader:
            moments = moments.squeeze(1).to(accelerator.device)
            labels = labels.to(accelerator.device)
            with torch.no_grad():
                images = sample_posterior(moments, scale, bias)
            with accelerator.accumulate(model):
                result = criterion(
                    model, images, {"y": labels},
                )
                accelerator.backward(result["total"])
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    update_ema(ema, accelerator.unwrap_model(model), decay=args.ema_decay)
            if not accelerator.sync_gradients:
                continue

            global_step += 1
            progress.update(1)
            logs = {
                "loss/total": result["total"].detach().item(),
                "loss/fm": result["fm"].detach().item(),
                "optimization/grad_norm": float(grad_norm),
                "optimization/lr": scheduler.get_last_lr()[0],
            }
            raw_model = accelerator.unwrap_model(model)
            if raw_model.use_mor:
                logs.update({f"mor/active_tokens_r{r + 1}": count
                             for r, count in enumerate(raw_model.active_token_counts)})
            progress.set_postfix(loss=f"{logs['loss/total']:.4f}")
            accelerator.log(logs, step=global_step)

            if global_step % args.checkpointing_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    torch.save({
                        "model": accelerator.unwrap_model(model).state_dict(), "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "steps": global_step, "args": vars(args), "random_state": random_state_dict(),
                    }, checkpoint_dir / f"{global_step:07d}.pt")
            if (args.report_to == "wandb" and args.sampling_steps > 0
                    and (global_step == 1 or global_step % args.sampling_steps == 0)):
                # Every rank generates a fixed local subset with its EMA copy;
                # gather creates the same global grid layout at every interval.
                accelerator.wait_for_everyone()
                if preview is None:
                    preview = prepare_training_sample(
                        args, accelerator.device,
                        local_batch_size=args.sampling_batch_size // accelerator.num_processes,
                        process_index=accelerator.process_index,
                    )
                vae, noise, sample_labels = preview
                # The repository sampler's CFG null class is fixed at 1000.
                cfg_scale = 4.0 if args.cfg_prob > 0 and args.num_classes == 1000 else 1.0
                samples = generate_training_samples(
                    ema, vae, noise, sample_labels, args.path_type, cfg_scale,
                )
                samples = accelerator.gather(samples)
                if accelerator.is_main_process:
                    import wandb
                    accelerator.log({"samples": wandb.Image(array2grid(samples))}, step=global_step)
                accelerator.wait_for_everyone()
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break
    progress.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done at step %d", global_step)
    accelerator.end_training()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON defaults; explicit CLI flags take precedence")
    parser.add_argument("--output-dir", default="exps_mor")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--report-to", choices=["wandb", "tensorboard", "none"], default="wandb")
    parser.add_argument("--project-name", default="SiT")
    parser.add_argument("--model", choices=["SiT-B/2", "SiT-L/2", "SiT-XL/2"], default="SiT-B/2")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    add_mor_args(parser)
    parser.add_argument("--path-type", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--prediction", choices=["v"], default="v")
    parser.add_argument("--weighting", choices=["uniform", "lognormal"], default="uniform")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--sampling-steps", type=int, default=10000,
                        help="Log an EMA image grid to W&B at step 1 and every N optimizer steps; <= 0 disables")
    parser.add_argument("--sampling-batch-size", type=int, default=64,
                        help="Global number of generated images in each W&B preview grid")
    parser.add_argument("--resume-step", type=int, default=0)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args(argv)
    if config_args.config is not None:
        with config_args.config.open() as handle:
            defaults = json.load(handle)
        unknown = set(defaults) - {action.dest for action in parser._actions}
        if unknown:
            parser.error(f"Unknown config fields: {sorted(unknown)}")
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())


'''
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --multi_gpu --num_processes 8 --mixed_precision bf16 \
  -m mixture_of_recursions.train \
  --model SiT-B/2 --exp-name mor2a \
  --data-dir /v/mnt/GH/imagenet_256 \
  --output-dir /v/mnt/GH/SiT \
  --use-mor \
  --use-recursion-conditioning \
  --mor-prefix-blocks 3 --mor-recurrent-blocks 2 --mor-suffix-blocks 3 \
  --mor-max-recursions 3 --mor-capacity-ratios 1.0 1.0 1.0 --no-mor-gating \
  --batch-size 256 --max-train-steps 400000 \
  --mixed-precision bf16 --allow-tf32

  --mor-global-topk
'''
