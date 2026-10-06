"""Flow-matching training for TREAD SiT with optional DensePush loss."""

from __future__ import annotations

from .attention_correction import log_resume_correction

import argparse
import copy
from copy import deepcopy
import logging
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, gather_object, set_seed

from dataset import CustomDataset
from .config import (
    ROUTESYNC_DEFAULTS, TREAD_DEFAULTS, add_routesync_args, add_tread_args,
    tread_kwargs,
)
from .loss import FlowMatchingLoss
from .model import build_tread_sit


def sample_posterior(moments, scale, bias):
    mean, std = torch.chunk(moments, 2, dim=1)
    return (mean + std * torch.randn_like(mean)) * scale + bias


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    grid = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    return grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


def prepare_training_preview(args, device, local_batch_size, process_index):
    """Load the VAE and fixed rank-specific preview inputs without changing training RNG."""
    from diffusers.models import AutoencoderKL

    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
        vae.requires_grad_(False)
    generator = torch.Generator(device=device).manual_seed((args.seed or 0) + process_index)
    noise = torch.randn(
        local_batch_size, 4, args.resolution // 8, args.resolution // 8,
        generator=generator, device=device,
    )
    labels = torch.randint(
        args.num_classes, (local_batch_size,), generator=generator, device=device,
    )
    return vae, noise, labels


@torch.no_grad()
def generate_training_preview(ema, vae, noise, labels, path_type, cfg_scale):
    from samplers import euler_sampler

    # Training previews always use the dense evaluation path so image quality
    # monitoring is independent of a sampled routing subset.
    previous_mode = ema.tread_eval_mode
    ema.tread_eval_mode = "dense"
    try:
        latents = euler_sampler(
            ema, noise, labels, num_steps=50, cfg_scale=cfg_scale,
            guidance_low=0.0, guidance_high=1.0, path_type=path_type, heun=False,
        ).float()
    finally:
        ema.tread_eval_mode = previous_mode
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
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_random_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        if isinstance(state["cuda"], (list, tuple)):  # Older checkpoint compatibility.
            torch.cuda.set_rng_state_all(state["cuda"])
        else:
            torch.cuda.set_rng_state(state["cuda"])


def create_logger(directory):
    directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(directory / "log.txt")], force=True,
    )
    return logging.getLogger("densepush")


def main(args):
    run_dir = Path(args.output_dir) / args.exp_name
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=ProjectConfiguration(project_dir=str(run_dir), logging_dir=str(run_dir / "logs")),
    )
    # Construct identical model/EMA weights on every rank. Reseed by rank after
    # construction so random token partitions remain independent under DDP.
    if args.seed is not None:
        set_seed(args.seed)
    logger = create_logger(run_dir) if accelerator.is_main_process else logging.getLogger(__name__)
    checkpoint_dir = run_dir / "checkpoints"
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        import json
        with (run_dir / "args.json").open("w") as handle:
            json.dump(vars(args), handle, indent=2)

    model = build_tread_sit(
        args.model, input_size=args.resolution // 8, num_classes=args.num_classes,
        class_dropout_prob=args.cfg_prob, use_cfg=args.cfg_prob > 0,
        qk_norm=args.qk_norm, fused_attn=args.fused_attn,
        path_type=args.path_type, **tread_kwargs(args),
    ).to(accelerator.device)
    ema = deepcopy(model).to(accelerator.device).eval()
    set_requires_grad(ema, False)
    update_ema(ema, model, decay=0)
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    criterion = FlowMatchingLoss(
        args.prediction, args.path_type, args.weighting,
        use_routesync=args.use_routesync,
        routesync_weight=args.routesync_weight,
        routesync_sample_ratio=args.routesync_sample_ratio,
        routesync_loss_type=args.routesync_loss_type,
        routesync_debug=args.routesync_debug,
        use_dense_push=args.use_dense_push,
        dense_push_weight=args.dense_push_weight,
        dense_push_margin=args.dense_push_margin,
    )
    dataset = CustomDataset(args.data_dir, num_classes=args.num_classes)
    if args.batch_size % accelerator.num_processes:
        raise ValueError("global --batch-size must be divisible by the number of processes")
    local_batch = args.batch_size // accelerator.num_processes
    if args.report_to == "wandb" and args.sampling_steps > 0:
        if args.sampling_batch_size < 1 or args.sampling_batch_size % accelerator.num_processes:
            raise ValueError("--sampling-batch-size must be positive and divisible by the process count")
    dataloader = DataLoader(
        dataset, batch_size=local_batch, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )

    global_step = 0
    resume_random_state = None
    if args.resume_step > 0:
        checkpoint = torch.load(
            checkpoint_dir / f"{args.resume_step:07d}.pt", map_location="cpu", weights_only=False,
        )
        saved = checkpoint["args"]
        log_resume_correction(logger, saved, args)
        # Only fields that change the training graph must match. Evaluation and
        # one-shot debug policy can safely change when resuming.
        for name in (
            "use_tread_routing", "tread_start_block", "tread_end_block",
            "tread_active_ratio", "tread_active_ratios", "tread_recursive", "tread_num_groups",
            "tread_recursive_pattern", "tread_depth_embedding",
            "tread_fp32_endpoint",
            "use_dense_push", "dense_push_ratio", "dense_push_weight",
            "dense_push_margin", "dense_push_block", "dense_push_tokens",
            "dense_push_grad",
            "use_routesync", "routesync_weight", "routesync_sample_ratio",
            "routesync_loss_type",
            "routesync_target_blocks",
        ):
            value = getattr(args, name)
            defaults = {**TREAD_DEFAULTS, **ROUTESYNC_DEFAULTS}
            previous = (
                saved.get(name, defaults[name])
                if isinstance(saved, dict)
                else getattr(saved, name, defaults[name])
            )
            if name == "routesync_target_blocks":
                saved_end = saved.get("tread_end_block", 9) if isinstance(saved, dict) else getattr(saved, "tread_end_block", 9)
                previous = tuple(previous) if previous is not None else (saved_end,)
                value = tuple(value) if value is not None else (args.tread_end_block,)
            if previous != value:
                raise ValueError(f"Resume configuration differs for {name}: {previous} vs {value}")
        model.load_state_dict(checkpoint["model"], strict=True)
        ema.load_state_dict(checkpoint["ema"], strict=True)
        optimizer.load_state_dict(checkpoint["opt"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = int(checkpoint["steps"])
        saved_random_states = checkpoint.get("random_states")
        if saved_random_states is not None:
            if len(saved_random_states) != accelerator.num_processes:
                raise ValueError(
                    "Checkpoint process count differs from this run: "
                    f"{len(saved_random_states)} vs {accelerator.num_processes}"
                )
            resume_random_state = saved_random_states[accelerator.process_index]
        else:
            resume_random_state = checkpoint.get("random_state")

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    if isinstance(model, DDP):
        model.register_comm_hook(state=None, hook=bf16_compress_hook)
        logger.info("DDP gradient communication: BF16 compression enabled")
    if resume_random_state is not None:
        restore_random_state(resume_random_state)
    if accelerator.is_main_process and args.report_to != "none":
        accelerator.init_trackers(
            args.project_name, config=vars(copy.deepcopy(args)),
            init_kwargs={"wandb": {"name": args.exp_name,
                                #    "dir": str(Path(args.output_dir) / "run_dir")}}
                                   "dir": str(Path('/root') / "run_dir")}}
            )
    logger.info("Parameters: %s", f"{sum(p.numel() for p in accelerator.unwrap_model(model).parameters()):,}")
    logger.info("TREAD fixed-subset routing: %s", tread_kwargs(args))

    scale = torch.full((1, 4, 1, 1), 0.18215, device=accelerator.device)
    bias = torch.zeros_like(scale)
    progress = tqdm(
        total=args.max_train_steps, initial=global_step,
        disable=not accelerator.is_local_main_process,
    )
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
                result = criterion(model, images, {"y": labels})
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
                "loss/route_sync": result["route_sync"].detach().item(),
                "loss/route_sync_weighted": result["route_sync_weighted"].detach().item(),
                "optimization/grad_norm": float(grad_norm),
                "optimization/lr": scheduler.get_last_lr()[0],
            }
            if args.use_dense_push:
                logs.update({
                    "loss/dense_push": result["dense_push"].detach().item(),
                    "loss/dense_push_weighted": result["dense_push_weighted"].detach().item(),
                    "dense_push/cosine": result["dense_push_cosine"].item(),
                    "dense_push/above_margin_fraction": result["dense_push_active_fraction"].item(),
                    "dense_push/samples_per_rank": result["dense_push_samples"],
                    "dense_push/actual_fraction": result["dense_push_fraction"],
                })
            if args.use_routesync:
                logs["routesync/target_block"] = accelerator.unwrap_model(model).last_routesync_target_block
            if args.tread_active_ratios is not None:
                routed_model = accelerator.unwrap_model(model)
                logs.update({
                    "routing/active_ratio": routed_model.last_tread_active_ratio,
                    "routing/active_fraction": routed_model.last_tread_active_fraction,
                })
            if args.use_routesync and args.routesync_loss_type == "feature-cosine":
                logs.update({
                    "routesync/feature_cosine_mean": result["feature_cosine_mean"].item(),
                    "routesync/feature_cosine_r_mean": result["feature_cosine_r_mean"].item(),
                    "routesync/feature_cosine_p_mean": result["feature_cosine_p_mean"].item(),
                })
            if args.routesync_debug and args.routesync_loss_type == "relational":
                logs.update({
                    "routesync/relation_pre_mean": result["relation_pre_mean"].item(),
                    "routesync/relation_post_mean": result["relation_post_mean"].item(),
                    "routesync/relation_abs_diff_mean": result["relation_abs_diff_mean"].item(),
                })
            progress.set_postfix(loss=f"{logs['loss/total']:.4f}")
            accelerator.log(logs, step=global_step)

            if global_step % args.checkpointing_steps == 0:
                accelerator.wait_for_everyone()
                random_states = gather_object([random_state_dict()])
                if accelerator.is_main_process:
                    torch.save({
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(), "opt": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "steps": global_step,
                        "args": vars(args), "random_states": random_states,
                    }, checkpoint_dir / f"{global_step:07d}.pt")

            if (args.report_to == "wandb" and args.sampling_steps > 0
                    and (global_step == 1 or global_step % args.sampling_steps == 0)):
                accelerator.wait_for_everyone()
                if preview is None:
                    preview = prepare_training_preview(
                        args, accelerator.device,
                        args.sampling_batch_size // accelerator.num_processes,
                        accelerator.process_index,
                    )
                vae, noise, sample_labels = preview
                cfg_scale = 4.0 if args.cfg_prob > 0 and args.num_classes == 1000 else 1.0
                devices = [accelerator.device] if accelerator.device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    preview_seed = (args.seed or 0) + 100_000 + accelerator.process_index
                    torch.manual_seed(preview_seed)
                    samples = generate_training_preview(
                        ema, vae, noise, sample_labels, args.path_type, cfg_scale,
                    )
                samples = accelerator.gather(samples)
                if accelerator.is_main_process:
                    import wandb
                    accelerator.log({"samples": wandb.Image(array2grid(samples))}, step=global_step)
                accelerator.wait_for_everyone()

            if global_step >= args.max_train_steps:
                break
    progress.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done at step %d", global_step)
    accelerator.end_training()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="exps_tread")
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
    add_tread_args(parser)
    add_routesync_args(parser)
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
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--sampling-batch-size", type=int, default=64)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())

'''
WANDB_API_KEY=${WANDB_API_KEY} \
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
  --multi_gpu --num_processes 4 --mixed_precision bf16 \
  -m densepush.train \
  --model SiT-B/2 --exp-name dense_push2 \
  --data-dir /root/imagenet_256 \
  --output-dir /v/mnt/GH/SiT \
  --use-tread-routing \
  --tread-start-block 2 --tread-end-block 9 --tread-active-ratio 0.5 \
  --use-dense-push --dense-push-margin 0.9 --dense-push-tokens routed \
  --dense-push-ratio 0.1 --dense-push-weight 1 \
  --batch-size 256 --max-train-steps 400000 \
  --allow-tf32
'''
