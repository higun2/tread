import argparse
import copy
from copy import deepcopy
import logging
import os
import time
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs

from models.sit_b1 import SiT_models
from loss_b10 import (
    SILoss,
    create_orthonormal_parity_matrix,
    patchify_state,
    project_to_parity_constraint,
    unpatchify_state,
)

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from samplers import get_score_from_velocity

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device

    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


class ParityFlowWrapper(nn.Module):
    """Inject stochastic parity-state tokens into an existing SiT sequence."""

    def __init__(self, base_model, num_parity_tokens, matrix_seed=0, control_only=False):
        super().__init__()
        self.base_model = base_model
        self.num_parity_tokens = num_parity_tokens
        self.num_image_tokens = base_model.x_embedder.num_patches
        self.patch_size = base_model.x_embedder.patch_size[0]
        self.in_channels = base_model.in_channels
        self.out_channels = base_model.out_channels
        self.control_only = control_only
        hidden_size = base_model.pos_embed.shape[-1]
        state_dim = self.patch_size * self.patch_size * self.in_channels
        matrix = create_orthonormal_parity_matrix(
            num_parity_tokens, self.num_image_tokens, matrix_seed
        )
        self.register_buffer("parity_matrix", matrix)
        self.parity_embed = nn.Linear(state_dim, hidden_size)
        self.parity_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.parity_index_embedding = nn.Parameter(
            torch.zeros(1, num_parity_tokens, hidden_size)
        )
        self.learned_control_tokens = (
            nn.Parameter(torch.zeros(1, num_parity_tokens, hidden_size))
            if control_only else None
        )
        nn.init.xavier_uniform_(self.parity_embed.weight)
        nn.init.zeros_(self.parity_embed.bias)
        nn.init.normal_(self.parity_type_embedding, std=0.02)
        nn.init.normal_(self.parity_index_embedding, std=0.02)
        if self.learned_control_tokens is not None:
            nn.init.normal_(self.learned_control_tokens, std=0.02)

    def forward(self, x, t, y, parity_t=None, return_parity=True, force_drop_ids=None, **_kwargs):
        base = self.base_model
        image_hidden = base.x_embedder(x) + base.pos_embed
        if self.control_only:
            parity_hidden = self.learned_control_tokens.expand(x.shape[0], -1, -1)
        else:
            if parity_t is None:
                raise ValueError("parity_t is required for stochastic ParityFlow")
            parity_hidden = self.parity_embed(parity_t)
        parity_hidden = parity_hidden + self.parity_type_embedding + self.parity_index_embedding
        hidden = torch.cat((image_hidden, parity_hidden), dim=1)
        conditioning = base.t_embedder(t) + base.y_embedder(
            y, self.training, force_drop_ids=force_drop_ids
        )
        for block in base.blocks:
            hidden = block(hidden, conditioning)
        prediction_tokens = base.final_layer(hidden, conditioning)
        image_tokens = prediction_tokens[:, : self.num_image_tokens]
        parity_prediction = prediction_tokens[:, self.num_image_tokens :]
        image_prediction = base.unpatchify(image_tokens)
        return image_prediction, parity_prediction if return_parity else None


def _velocity_from_clean(state, clean, alpha, sigma, d_alpha, d_sigma, eps=1e-8):
    safe_sigma = sigma.clamp_min(eps)
    noise = (state - alpha * clean) / safe_sigma
    return d_alpha * clean + d_sigma * noise


@torch.no_grad()
def parityflow_euler_sampler(
    model,
    latents,
    y,
    loss_fn,
    num_steps=50,
    heun=False,
    cfg_scale=1.0,
    guidance_low=0.0,
    guidance_high=1.0,
    correction_enabled=False,
    correction_gamma_max=0.5,
    correction_schedule="signal_ratio",
):
    """Joint Euler/Heun evolution with CFG and optional post-CFG syndrome correction."""
    dtype, device = latents.dtype, latents.device
    model_ref = model.module if hasattr(model, "module") else model
    x_next = latents.to(torch.float64)
    state_dim = model_ref.patch_size * model_ref.patch_size * model_ref.in_channels
    parity_next = torch.randn(
        latents.shape[0], model_ref.num_parity_tokens, state_dim, device=device, dtype=dtype
    ).to(torch.float64)
    y_null = torch.full_like(y, model_ref.base_model.num_classes) if cfg_scale > 1 else None
    t_steps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float64, device=device)

    def evaluate(x_state, parity_state, t_scalar):
        use_cfg = cfg_scale > 1 and guidance_low <= float(t_scalar) <= guidance_high
        x_input = torch.cat((x_state, x_state), 0) if use_cfg else x_state
        p_input = torch.cat((parity_state, parity_state), 0) if use_cfg else parity_state
        y_input = torch.cat((y, y_null), 0) if use_cfg else y
        time_value = float(t_scalar)
        time = torch.full((x_input.shape[0],), time_value, device=device, dtype=dtype)
        image_v, parity_v = model(
            x_input.to(dtype), time, y=y_input, parity_t=p_input.to(dtype)
        )
        image_v, parity_v = image_v.to(torch.float64), parity_v.to(torch.float64)
        if use_cfg:
            ic, iu = image_v.chunk(2)
            pc, pu = parity_v.chunk(2)
            image_v = iu + cfg_scale * (ic - iu)
            parity_v = pu + cfg_scale * (pc - pu)

        if correction_enabled:
            t4 = torch.full((x_state.shape[0], 1, 1, 1), time_value, device=device, dtype=torch.float64)
            alpha, sigma, da, ds = loss_fn.interpolant(t4)
            p_alpha = alpha.flatten(1).mean(1).view(-1, 1, 1)
            p_sigma = sigma.flatten(1).mean(1).view(-1, 1, 1)
            p_da = da.flatten(1).mean(1).view(-1, 1, 1) if torch.is_tensor(da) else da
            p_ds = ds.flatten(1).mean(1).view(-1, 1, 1) if torch.is_tensor(ds) else ds
            image_clean = loss_fn.clean_from_velocity(x_state, image_v, alpha, sigma, da, ds)
            parity_clean = loss_fn.clean_from_velocity(
                parity_state, parity_v, p_alpha, p_sigma, p_da, p_ds
            )
            image_tokens = patchify_state(image_clean, model_ref.patch_size)
            schedule = (
                torch.ones_like(alpha)
                if correction_schedule == "constant"
                else alpha.square() / (alpha.square() + sigma.square() + loss_fn.schedule_eps)
            )
            gamma = correction_gamma_max * schedule.flatten(1).mean(1)
            corrected_tokens, corrected_parity, _ = project_to_parity_constraint(
                image_tokens, parity_clean, model_ref.parity_matrix, gamma
            )
            corrected_image = unpatchify_state(
                corrected_tokens, model_ref.patch_size, model_ref.in_channels,
                x_state.shape[-2], x_state.shape[-1]
            )
            image_v = _velocity_from_clean(x_state, corrected_image, alpha, sigma, da, ds)
            parity_v = _velocity_from_clean(
                parity_state, corrected_parity, p_alpha, p_sigma, p_da, p_ds
            )
        return image_v, parity_v

    for index, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur, parity_cur = x_next, parity_next
        image_v, parity_v = evaluate(x_cur, parity_cur, t_cur)
        dt = t_next - t_cur
        x_next = x_cur + dt * image_v
        parity_next = parity_cur + dt * parity_v
        if heun and index < num_steps - 1:
            image_prime, parity_prime = evaluate(x_next, parity_next, t_next)
            x_next = x_cur + dt * 0.5 * (image_v + image_prime)
            parity_next = parity_cur + dt * 0.5 * (parity_v + parity_prime)
    return x_next


@torch.no_grad()
def parityflow_euler_maruyama_sampler(
    model,
    latents,
    y,
    loss_fn,
    num_steps=50,
    cfg_scale=1.0,
    guidance_low=0.0,
    guidance_high=1.0,
    correction_enabled=False,
    correction_gamma_max=0.5,
    correction_schedule="signal_ratio",
):
    """Joint reverse-SDE sampling with independent image/parity Brownian noise."""
    if num_steps < 2:
        raise ValueError("ParityFlow SDE sampling requires num_steps >= 2")
    dtype, device = latents.dtype, latents.device
    model_ref = model.module if hasattr(model, "module") else model
    x_next = latents.to(torch.float64)
    state_dim = model_ref.patch_size * model_ref.patch_size * model_ref.in_channels
    parity_next = torch.randn(
        latents.shape[0], model_ref.num_parity_tokens, state_dim,
        device=device, dtype=dtype,
    ).to(torch.float64)
    y_null = torch.full_like(y, model_ref.base_model.num_classes) if cfg_scale > 1 else None
    t_steps = torch.linspace(1.0, 0.04, num_steps, dtype=torch.float64, device=device)
    t_steps = torch.cat((t_steps, torch.zeros(1, dtype=torch.float64, device=device)))

    def guided_velocity(x_state, parity_state, t_scalar):
        use_cfg = cfg_scale > 1 and guidance_low <= float(t_scalar) <= guidance_high
        x_input = torch.cat((x_state, x_state), 0) if use_cfg else x_state
        p_input = torch.cat((parity_state, parity_state), 0) if use_cfg else parity_state
        y_input = torch.cat((y, y_null), 0) if use_cfg else y
        time_value = float(t_scalar)
        time_input = torch.full(
            (x_input.shape[0],), time_value, device=device, dtype=dtype
        )
        image_v, parity_v = model(
            x_input.to(dtype), time_input, y=y_input, parity_t=p_input.to(dtype)
        )
        image_v, parity_v = image_v.to(torch.float64), parity_v.to(torch.float64)
        if use_cfg:
            image_cond, image_uncond = image_v.chunk(2)
            parity_cond, parity_uncond = parity_v.chunk(2)
            image_v = image_uncond + cfg_scale * (image_cond - image_uncond)
            parity_v = parity_uncond + cfg_scale * (parity_cond - parity_uncond)

        if correction_enabled:
            t4 = torch.full(
                (x_state.shape[0], 1, 1, 1), time_value,
                device=device, dtype=torch.float64,
            )
            alpha, sigma, d_alpha, d_sigma = loss_fn.interpolant(t4)
            p_alpha = alpha.flatten(1).mean(1).view(-1, 1, 1)
            p_sigma = sigma.flatten(1).mean(1).view(-1, 1, 1)
            p_d_alpha = (
                d_alpha.flatten(1).mean(1).view(-1, 1, 1)
                if torch.is_tensor(d_alpha) else d_alpha
            )
            p_d_sigma = (
                d_sigma.flatten(1).mean(1).view(-1, 1, 1)
                if torch.is_tensor(d_sigma) else d_sigma
            )
            image_clean = loss_fn.clean_from_velocity(
                x_state, image_v, alpha, sigma, d_alpha, d_sigma
            )
            parity_clean = loss_fn.clean_from_velocity(
                parity_state, parity_v, p_alpha, p_sigma, p_d_alpha, p_d_sigma
            )
            image_tokens = patchify_state(image_clean, model_ref.patch_size)
            schedule = (
                torch.ones_like(alpha)
                if correction_schedule == "constant"
                else alpha.square()
                / (alpha.square() + sigma.square() + loss_fn.schedule_eps)
            )
            gamma = correction_gamma_max * schedule.flatten(1).mean(1)
            corrected_tokens, corrected_parity, _ = project_to_parity_constraint(
                image_tokens, parity_clean, model_ref.parity_matrix, gamma
            )
            corrected_image = unpatchify_state(
                corrected_tokens,
                model_ref.patch_size,
                model_ref.in_channels,
                x_state.shape[-2],
                x_state.shape[-1],
            )
            image_v = _velocity_from_clean(
                x_state, corrected_image, alpha, sigma, d_alpha, d_sigma
            )
            parity_v = _velocity_from_clean(
                parity_state, corrected_parity,
                p_alpha, p_sigma, p_d_alpha, p_d_sigma,
            )
        return image_v, parity_v

    # Match the repository sampler: stochastic steps down to t=.04, followed
    # by one deterministic final step to t=0.
    for t_cur, t_next in zip(t_steps[:-2], t_steps[1:-1]):
        x_cur, parity_cur = x_next, parity_next
        image_v, parity_v = guided_velocity(x_cur, parity_cur, t_cur)
        batch_time = torch.full(
            (x_cur.shape[0],), float(t_cur), device=device, dtype=torch.float64
        )
        image_score = get_score_from_velocity(
            image_v, x_cur, batch_time, path_type=loss_fn.path_type
        )
        parity_score = get_score_from_velocity(
            parity_v, parity_cur, batch_time, path_type=loss_fn.path_type
        )
        diffusion = 2 * t_cur
        image_drift = image_v - 0.5 * diffusion * image_score
        parity_drift = parity_v - 0.5 * diffusion * parity_score
        dt = t_next - t_cur
        # These two draws are intentionally independent.
        image_brownian = torch.randn_like(x_cur) * torch.sqrt(torch.abs(dt))
        parity_brownian = torch.randn_like(parity_cur) * torch.sqrt(torch.abs(dt))
        x_next = x_cur + image_drift * dt + torch.sqrt(diffusion) * image_brownian
        parity_next = (
            parity_cur + parity_drift * dt + torch.sqrt(diffusion) * parity_brownian
        )

    t_cur, t_next = t_steps[-2], t_steps[-1]
    x_cur, parity_cur = x_next, parity_next
    image_v, parity_v = guided_velocity(x_cur, parity_cur, t_cur)
    batch_time = torch.full(
        (x_cur.shape[0],), float(t_cur), device=device, dtype=torch.float64
    )
    image_score = get_score_from_velocity(
        image_v, x_cur, batch_time, path_type=loss_fn.path_type
    )
    parity_score = get_score_from_velocity(
        parity_v, parity_cur, batch_time, path_type=loss_fn.path_type
    )
    diffusion = 2 * t_cur
    dt = t_next - t_cur
    image_drift = image_v - 0.5 * diffusion * image_score
    parity_drift = parity_v - 0.5 * diffusion * parity_score
    x_next = x_cur + image_drift * dt
    parity_next = parity_cur + parity_drift * dt
    return x_next


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        class_dropout_prob=args.cfg_prob,
        **block_kwargs
    )
    if args.parityflow:
        model = ParityFlowWrapper(
            model,
            num_parity_tokens=args.num_parity_tokens,
            matrix_seed=args.parity_matrix_seed,
            control_only=args.parity_ablation == "learned_tokens",
        )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)

    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting,
        parityflow_enabled=args.parityflow,
        parity_fm_weight=args.parity_fm_weight,
        syndrome_loss_weight=args.syndrome_loss_weight,
        syndrome_schedule=args.syndrome_schedule,
        schedule_eps=args.parity_schedule_eps,
        control_only=args.parity_ablation == "learned_tokens",
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Setup data:
    train_dataset = CustomDataset(args.data_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="SiT",
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    gt_xs, _ = next(iter(train_dataloader))
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
        )
    ys = torch.randint(1000, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_dataloader:
            iteration_start = time.perf_counter()
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            z = None
            if args.legacy:
                # In our early experiments, we accidentally apply label dropping twice:
                # once in train.py and once in sit.py.
                # We keep this option for exact reproducibility with previous runs.
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, y)
            else:
                labels = y

            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)

            grad_norm = torch.zeros((), device=device)
            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)

                image_loss, parity_fm_loss, syndrome_loss, parity_metrics = loss_fn(
                    model,
                    x,
                    model_kwargs,
                )

                image_loss_mean = image_loss.mean()
                total_loss = image_loss_mean + parity_fm_loss + syndrome_loss

                ## optimization
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function

            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                from samplers import euler_sampler
                with torch.no_grad():
                    if args.parityflow and args.parity_ablation == "parity":
                        samples = parityflow_euler_sampler(
                            model, xT, ys, loss_fn,
                            num_steps=50,
                            cfg_scale=4.0,
                            guidance_low=0.,
                            guidance_high=1.,
                            correction_enabled=args.parity_correction,
                            correction_gamma_max=args.parity_correction_gamma,
                            correction_schedule=args.parity_correction_schedule,
                        ).to(torch.float32)
                    else:
                        samples = euler_sampler(
                            model, xT, ys, num_steps=50, cfg_scale=4.0,
                            guidance_low=0., guidance_high=1.,
                            path_type=args.path_type, heun=False,
                        ).to(torch.float32)
                    samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                 "gt_samples": wandb.Image(array2grid(gt_samples))})
                logging.info("Generating EMA samples done.")

            logs = {
                "loss": accelerator.gather(image_loss_mean).mean().detach().item(),
                # "train/image_fm_loss": accelerator.gather(image_loss_mean).mean().detach().item(),
                "train/parity_fm_loss_weighted": accelerator.gather(parity_fm_loss).mean().detach().item(),
                "train/syndrome_loss_weighted": accelerator.gather(syndrome_loss).mean().detach().item(),
                # "total_loss": accelerator.gather(total_loss).mean().detach().item(),
                # "train/total_loss": accelerator.gather(total_loss).mean().detach().item(),
                # "train/iteration_time": time.perf_counter() - iteration_start,
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            if args.parityflow and global_step % args.parity_log_interval == 0:
                for name, value in parity_metrics.items():
                    logs[f"train/{name}"] = accelerator.gather(value).float().mean().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000) # 80 epochs
    parser.add_argument("--checkpointing-steps", type=int, default=50000) # 10 epcohs
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    # Error-correcting stochastic parity-token flow (disabled by default).
    parser.add_argument("--parityflow", action="store_true")
    parser.add_argument("--num-parity-tokens", type=int, default=8)
    parser.add_argument("--parity-matrix-seed", type=int, default=0)
    parser.add_argument("--parity-fm-weight", type=float, default=0.1)
    parser.add_argument("--syndrome-loss-weight", type=float, default=0.1)
    parser.add_argument("--syndrome-schedule", choices=["constant", "signal_ratio"], default="signal_ratio")
    parser.add_argument("--parity-correction", action="store_true")
    parser.add_argument("--parity-correction-gamma", type=float, default=0.5)
    parser.add_argument("--parity-correction-schedule", choices=["constant", "signal_ratio"], default="signal_ratio")
    parser.add_argument("--parity-schedule-eps", type=float, default=1e-8)
    parser.add_argument("--parity-log-interval", type=int, default=100)
    parser.add_argument("--parity-ablation", choices=["parity", "learned_tokens"], default="parity")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.num_parity_tokens <= 0:
        parser.error("--num-parity-tokens must be positive")
    if args.parity_fm_weight < 0 or args.syndrome_loss_weight < 0:
        parser.error("ParityFlow loss weights must be non-negative")
    if not 0 <= args.parity_correction_gamma <= 0.5:
        parser.error("--parity-correction-gamma must be in [0, 0.5]")
    if args.parity_schedule_eps <= 0 or args.parity_log_interval <= 0:
        parser.error("ParityFlow epsilon and log interval must be positive")
    if args.parity_correction and not args.parityflow:
        parser.error("--parity-correction requires --parityflow")
    if args.parity_correction and args.parity_ablation != "parity":
        parser.error("Parity correction is not defined for learned-token control")

    return args

if __name__ == "__main__":
    args = parse_args()

    main(args)



'''
accelerate launch train_b4.py \
  --model "SiT-B/2" \
  --exp-name "parityflow-v1" \
  --data-dir /v/mnt/GH/imagenet_256 \
  --output-dir /v/mnt/GH/SiT \
  --parityflow \
  --num-parity-tokens 8 \
  --parity-fm-weight 0.1 \
  --syndrome-loss-weight 0.1 \
  --syndrome-schedule signal_ratio
'''
