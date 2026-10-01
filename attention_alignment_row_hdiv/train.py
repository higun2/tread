"""SiT training with attention alignment and selectable CKA/row-L1 H-diversity."""

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
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision.utils import make_grid

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.models import AutoencoderKL
import wandb

from dataset import CustomDataset
from models.sit_b1 import SiT_models
from .loss import SILoss


def sample_posterior(moments, scale, bias):
    mean, std = torch.chunk(moments, 2, dim=1)
    return (mean + std * torch.randn_like(mean)) * scale + bias


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    return x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    parameters = dict(model.named_parameters())
    for name, ema_parameter in ema_model.named_parameters():
        ema_parameter.mul_(decay).add_(parameters[name].detach(), alpha=1 - decay)


def set_requires_grad(module, flag):
    for parameter in module.parameters():
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
    return logging.getLogger("attention_alignment_row_hdiv")


def resolve_attention_depths(args):
    """Resolve existing 1-based depth inputs into a unique sorted list."""
    depths = sorted(set(args.encoder_depths))
    if len(depths) < 2:
        raise ValueError("Provide at least two depths via --encoder-depths")
    if min(depths) < 1:
        raise ValueError("Depth values must be >= 1 (1-based)")
    return depths


def resolve_hdiv_mode(args):
    """Disable all H-div hooks/computation when its effective weight is zero."""
    return args.hdiv_mode if args.hdiv_weight != 0.0 else "none"


def create_attention_capture_hook():
    """The same attention reconstruction used by train_b2.py."""
    def hook_fn(module, inputs, output):
        x = inputs[0]
        batch, tokens, channels = x.shape
        if hasattr(module, "qkv") and hasattr(module, "num_heads"):
            qkv = module.qkv(x).reshape(
                batch, tokens, 3, module.num_heads, channels // module.num_heads
            ).permute(2, 0, 3, 1, 4)
            query, key, _value = qkv[0], qkv[1], qkv[2]
            module.saved_attn = ((query @ key.transpose(-2, -1)) * module.scale).softmax(-1)
    return hook_fn


def create_hidden_capture_hook():
    """Capture the final [B,T,D] output of a selected complete block."""
    def hook_fn(module, inputs, output):
        # Sampling under no_grad should not retain diagnostic hidden tensors.
        module.saved_hidden = output if torch.is_grad_enabled() else None
    return hook_fn


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

    depths = resolve_attention_depths(args)
    model = SiT_models[args.model](
        input_size=args.resolution // 8, num_classes=args.num_classes,
        use_cfg=args.cfg_prob > 0, class_dropout_prob=args.cfg_prob,
        qk_norm=args.qk_norm, fused_attn=args.fused_attn,
    ).to(accelerator.device)
    indices = [depth - 1 for depth in depths]
    invalid = [index for index in indices if index < 0 or index >= len(model.blocks)]
    if invalid:
        raise ValueError(f"capture indices {invalid} outside 0..{len(model.blocks) - 1}")

    # EMA is copied before training-only hooks, so its inference path is clean.
    ema = deepcopy(model).to(accelerator.device).eval()
    set_requires_grad(ema, False)
    update_ema(ema, model, decay=0)
    # Preserve train_b2.py's periodic sample-generation pipeline.
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(accelerator.device).eval()

    # A zero auxiliary weight must install no hidden hooks and perform no H-div
    # computation, independently of the selected ablation mode.
    effective_hdiv_mode = resolve_hdiv_mode(args)
    hdiv_enabled = effective_hdiv_mode != "none"
    hooks = []
    for index in indices:
        attention = model.blocks[index].attn
        attention.saved_attn = None
        hooks.append(attention.register_forward_hook(create_attention_capture_hook()))
        if hdiv_enabled:
            block = model.blocks[index]
            block.saved_hidden = None
            hooks.append(block.register_forward_hook(create_hidden_capture_hook()))
    logger.info(
        "Attention depths (1-based): %s; deep target=%d; H-div mode=%s "
        "weight=%g cka_threshold=%.4f row_margin=%.4f cka_detach_deep=%s",
        depths, depths[-1], effective_hdiv_mode, args.hdiv_weight,
        args.hdiv_cka_threshold, args.row_hdiv_margin, args.hdiv_detach_deep,
    )

    criterion = SILoss(
        prediction=args.prediction, path_type=args.path_type, weighting=args.weighting,
        attn_loss_type=args.attn_loss_type, accelerator=accelerator,
        hdiv_mode=effective_hdiv_mode, hdiv_cka_threshold=args.hdiv_cka_threshold,
        row_hdiv_margin=args.row_hdiv_margin,
        hdiv_eps=args.hdiv_eps, hdiv_detach_deep=args.hdiv_detach_deep,
        row_hdiv_log_quantiles=args.row_hdiv_log_quantiles,
    )
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,
    )
    dataset = CustomDataset(args.data_dir, num_classes=args.num_classes)
    if args.batch_size % accelerator.num_processes:
        raise ValueError("global --batch-size must be divisible by the process count")
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size // accelerator.num_processes,
        shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    global_step = 0
    resume_random_state = None
    if args.resume_step > 0:
        checkpoint = torch.load(
            checkpoint_dir / f"{args.resume_step:07d}.pt", map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        ema.load_state_dict(checkpoint["ema"], strict=True)
        optimizer.load_state_dict(checkpoint["opt"])
        global_step = int(checkpoint["steps"])
        resume_random_state = checkpoint.get("random_state")

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    if resume_random_state is not None:
        restore_random_state(resume_random_state)
    if accelerator.is_main_process and args.report_to != "none":
        accelerator.init_trackers(
            args.project_name, config=vars(copy.deepcopy(args)),
            init_kwargs={"wandb": {"name": args.exp_name}},
        )
    logger.info("SiT parameters: %s", f"{sum(p.numel() for p in accelerator.unwrap_model(model).parameters()):,}")

    scale = torch.full((1, 4, 1, 1), 0.18215, device=accelerator.device)
    bias = torch.zeros_like(scale)
    sample_batch_size = 64 // accelerator.num_processes
    diagnostic_moments, _ = next(iter(dataloader))
    diagnostic_moments = diagnostic_moments[:sample_batch_size].squeeze(1).to(accelerator.device)
    with torch.no_grad():
        diagnostic_latents = sample_posterior(diagnostic_moments, scale, bias)
    diagnostic_labels = torch.randint(
        args.num_classes, (sample_batch_size,), device=accelerator.device
    )
    diagnostic_noise = torch.randn(
        sample_batch_size, 4, args.resolution // 8, args.resolution // 8,
        device=accelerator.device,
    )
    progress = tqdm(
        total=args.max_train_steps, initial=global_step,
        disable=not accelerator.is_local_main_process,
    )
    grad_norm = torch.tensor(0.0, device=accelerator.device)
    for _epoch in range(args.epochs):
        model.train()
        for moments, labels in dataloader:
            moments = moments.squeeze(1).to(accelerator.device)
            labels = labels.to(accelerator.device)
            if args.legacy:
                drop_ids = torch.rand(labels.shape[0], device=labels.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, labels)
            with torch.no_grad():
                images = sample_posterior(moments, scale, bias)

            with accelerator.accumulate(model):
                raw_model = accelerator.unwrap_model(model)
                shallow_attention = [raw_model.blocks[index].attn for index in indices[:-1]]
                deep_attention = raw_model.blocks[indices[-1]].attn
                for attention in (*shallow_attention, deep_attention):
                    attention.saved_attn = None

                hidden_dict = None
                if hdiv_enabled:
                    shallow_blocks = [raw_model.blocks[index] for index in indices[:-1]]
                    deep_block = raw_model.blocks[indices[-1]]
                    for block in (*shallow_blocks, deep_block):
                        block.saved_hidden = None
                    hidden_dict = {"shallow": shallow_blocks, "deep": deep_block}

                fm_per_sample, attn_per_sample, hdiv = criterion(
                    model, images, {"y": labels},
                    attn_maps_dict={"shallow": shallow_attention, "deep": deep_attention},
                    hidden_states_dict=hidden_dict,
                )
                loss_fm = fm_per_sample.mean()
                loss_attn = attn_per_sample.mean()
                loss_hdiv = hdiv["loss"].mean()
                weighted_attn = args.kl_coeff * loss_attn
                weighted_hdiv = args.hdiv_weight * loss_hdiv
                total_loss = loss_fm + weighted_attn + weighted_hdiv

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    update_ema(ema, accelerator.unwrap_model(model), decay=args.ema_decay)
            if not accelerator.sync_gradients:
                continue

            global_step += 1
            progress.update(1)
            def reduced_item(value):
                value = value.detach() if torch.is_tensor(value) else torch.tensor(
                    value, device=accelerator.device
                )
                return accelerator.reduce(value.float(), reduction="mean").item()

            def reduced_sum(value):
                value = value.detach() if torch.is_tensor(value) else torch.tensor(
                    value, device=accelerator.device
                )
                return accelerator.reduce(value.float(), reduction="sum")

            def reduced_extreme(value, op):
                value = value.detach().float().clone()
                if accelerator.num_processes > 1:
                    dist.all_reduce(value, op=op)
                return value.item()

            logs = {
                # Requested canonical names.
                "loss_total": reduced_item(total_loss),
                "loss_fm": reduced_item(loss_fm),
                "loss_attn": reduced_item(loss_attn),
                "loss_hdiv": reduced_item(loss_hdiv),
                "hdiv_cka_mean": reduced_item(hdiv["cka_mean"]),
                "hdiv_distance_mean": reduced_item(hdiv["distance_mean"]),
                "hdiv_active_ratio": reduced_item(hdiv["active_ratio"]),
                # Weighted contributions make total-loss accounting explicit.
                "loss_attn_weighted": reduced_item(weighted_attn),
                "loss_hdiv_weighted": reduced_item(weighted_hdiv),
                "optimization/grad_norm": reduced_item(grad_norm),
                "optimization/lr": optimizer.param_groups[0]["lr"],
            }
            if effective_hdiv_mode == "row_l1":
                row_sum = reduced_sum(hdiv["row_dist_sum"])
                row_sq_sum = reduced_sum(hdiv["row_dist_sq_sum"])
                row_active_sum = reduced_sum(hdiv["row_active_sum"])
                row_count = reduced_sum(hdiv["row_dist_count"]).clamp_min(1.0)
                row_mean = row_sum / row_count
                row_variance = (row_sq_sum / row_count - row_mean.square()).clamp_min(0.0)
                logs.update({
                    "loss_row_hdiv": logs["loss_hdiv"],
                    "row_dist_mean": row_mean.item(),
                    "row_dist_std": row_variance.sqrt().item(),
                    "row_dist_min": reduced_extreme(hdiv["row_dist_min"], dist.ReduceOp.MIN),
                    "row_dist_max": reduced_extreme(hdiv["row_dist_max"], dist.ReduceOp.MAX),
                    "row_active_ratio": (row_active_sum / row_count).item(),
                })
                if args.row_hdiv_log_quantiles:
                    global_row_dist = accelerator.gather(
                        hdiv["row_dist_values"].contiguous()
                    ).float()
                    quantiles = torch.quantile(
                        global_row_dist, global_row_dist.new_tensor([0.25, 0.5, 0.75])
                    )
                    logs.update({
                        "row_dist_p25": quantiles[0].item(),
                        "row_dist_median": quantiles[1].item(),
                        "row_dist_p75": quantiles[2].item(),
                    })

            postfix = {
                "total": f"{logs['loss_total']:.4f}",
                "fm": f"{logs['loss_fm']:.4f}",
                "attn": f"{logs['loss_attn_weighted']:.4f}",
                "hdiv": f"{logs['loss_hdiv_weighted']:.4f}",
            }
            if effective_hdiv_mode == "cka":
                postfix.update(
                    cka=f"{logs['hdiv_cka_mean']:.3f}",
                    active=f"{logs['hdiv_active_ratio']:.2f}",
                )
            elif effective_hdiv_mode == "row_l1":
                postfix.update(
                    row_dist=f"{logs['row_dist_mean']:.3f}",
                    row_active=f"{logs['row_active_ratio']:.2f}",
                )
            progress.set_postfix(**postfix)
            accelerator.log(logs, step=global_step)

            if global_step % args.checkpointing_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    torch.save({
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(), "opt": optimizer.state_dict(),
                        "args": vars(args), "steps": global_step,
                        "random_state": random_state_dict(),
                    }, checkpoint_dir / f"{global_step:07d}.pt")

            if global_step == 1 or (
                args.sampling_steps > 0 and global_step % args.sampling_steps == 0
            ):
                from samplers import euler_sampler
                with torch.no_grad():
                    samples = euler_sampler(
                        model, diagnostic_noise, diagnostic_labels, num_steps=50,
                        cfg_scale=4.0, guidance_low=0.0, guidance_high=1.0,
                        path_type=args.path_type, heun=False,
                    ).float()
                    samples = vae.decode(samples / scale).sample.add(1).div(2)
                    references = vae.decode(diagnostic_latents / scale).sample.add(1).div(2)
                samples = accelerator.gather(samples.float())
                references = accelerator.gather(references.float())
                accelerator.log({
                    "samples": wandb.Image(array2grid(samples)),
                    "gt_samples": wandb.Image(array2grid(references)),
                }, step=global_step)
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break
    for hook in hooks:
        hook.remove()
    progress.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done at step %d", global_step)
    accelerator.end_training()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="exps_attention_row_hdiv")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--report-to", choices=["wandb", "tensorboard", "none"], default="wandb")
    parser.add_argument("--project-name", default="SiT")
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--model", choices=list(SiT_models.keys()), default="SiT-B/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depths", type=int, nargs="+", default=[3, 5, 7],
                        help="1-based attention/H-div depths; final value is the shared deep target")
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--path-type", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--prediction", choices=["v"], default="v")
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--kl-coeff", type=float, default=0.1,
                        help="Existing attention-alignment coefficient")
    parser.add_argument("--attn-loss-type", choices=["kl", "js", "l1", "l2"], default="kl")
    parser.add_argument("--weighting", choices=["uniform", "lognormal"], default="uniform")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hdiv-mode", choices=SILoss.MODES, default="none",
                        help="Independent H-div ablation; default preserves FM+attention")
    parser.add_argument("--hdiv-weight", type=float, default=0.1)
    parser.add_argument("--hdiv-cka-threshold", type=float, default=0.8)
    parser.add_argument("--row-hdiv-margin", type=float, default=0.1,
                        help="Minimum desired off-diagonal relational L1 distance")
    parser.add_argument("--hdiv-eps", type=float, default=1e-6)
    parser.add_argument("--hdiv-detach-deep", action=argparse.BooleanOptionalAction, default=True,
                        help="Existing CKA-only deep detach option; row_l1 always detaches deep")
    parser.add_argument("--row-hdiv-log-quantiles", action=argparse.BooleanOptionalAction,
                        default=False, help="Gather row distances and log p25/median/p75")

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())

'''
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --num_processes 4 \
  -m attention_alignment_row_hdiv.train \
  --exp-name n1 \
  --model SiT-B/2 \
  --data-dir /v/mnt/GH/imagenet_256 \
  --output-dir /v/mnt/GH/SiT \
  --batch-size 256 \
  --mixed-precision bf16 \
  --allow-tf32 \
  --max-train-steps 400000 \
  --encoder-depths 3 5 7 \
  --attn-loss-type kl \
  --kl-coeff 0.1 \
  --hdiv-mode row_l1 \
  --hdiv-weight 0.1 \
  --row-hdiv-margin 0.2

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
  --num_processes 4 \
  -m attention_alignment_row_hdiv.train \
  --exp-name n2 \
  --model SiT-B/2 \
  --data-dir /v/mnt/GH/imagenet_256 \
  --output-dir /v/mnt/GH/SiT \
  --batch-size 256 \
  --mixed-precision bf16 \
  --allow-tf32 \
  --max-train-steps 400000 \
  --encoder-depths 3 5 7 \
  --attn-loss-type kl \
  --kl-coeff 0.1 \
  --hdiv-mode row_l1 \
  --hdiv-weight 0.1 \
  --row-hdiv-margin 0.3




accelerate launch --multi_gpu --num_processes 8 \
  -m attention_alignment_row_hdiv.train \
  --exp-name n1 \
  --model SiT-B/2 \
  --data-dir /v/mnt/GH/imagenet_256 \
  --output-dir /v/mnt/GH/SiT \
  --batch-size 256 \
  --mixed-precision bf16 \
  --allow-tf32 \
  --max-train-steps 400000 \
  --encoder-depths 3 5 7 \
  --attn-loss-type kl \
  --kl-coeff 0.1 \
  --hdiv-mode row_l1 \
  --hdiv-weight 0.1 \
  --row-hdiv-margin 0.2
'''
