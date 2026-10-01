"""SiT flow-matching loss with optional Noise-Cancelled DFM (NC-DFM)."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def mean_flat(x: torch.Tensor) -> torch.Tensor:
    """Average over every dimension except the leading batch/pair dimension."""
    return x.mean(dim=tuple(range(1, x.ndim)))


def build_ncdfm_pairs(
    batch_size: int,
    pair_fraction: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly select disjoint local-batch pairs without duplicating samples."""
    if not 0.0 <= pair_fraction <= 1.0:
        raise ValueError("pair_fraction must be in [0, 1]")
    paired_count = min(batch_size, int(pair_fraction * batch_size))
    paired_count -= paired_count % 2
    if paired_count == 0:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty
    selected = torch.randperm(batch_size, device=device)[:paired_count]
    return selected[: paired_count // 2], selected[paired_count // 2 :]


def share_pair_stochasticity(
    time_input: torch.Tensor,
    noises: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Make each j member use its i member's timestep and Gaussian noise."""
    if pair_i.numel() == 0:
        return time_input, noises
    # Out-of-place clones avoid autograd/version-counter surprises and keep the
    # initially sampled tensors available to tests and future diagnostics.
    shared_time = time_input.clone()
    shared_noise = noises.clone()
    shared_time[pair_j] = time_input[pair_i]
    shared_noise[pair_j] = noises[pair_i]
    return shared_time, shared_noise


def compute_ncdfm_schedule(
    alpha_t: torch.Tensor,
    sigma_t: torch.Tensor,
    schedule: str,
    eps: float,
) -> torch.Tensor:
    """Return a schedule derived from the repository's actual interpolant."""
    if schedule == "constant":
        return torch.ones_like(alpha_t)
    if schedule == "signal_ratio":
        return alpha_t.square() / (alpha_t.square() + sigma_t.square() + eps)
    raise ValueError(f"Unsupported NC-DFM schedule: {schedule}")


def compute_ncdfm_loss(
    model_output: torch.Tensor,
    model_target: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    schedule_weight: torch.Tensor,
    lambda_max: float,
    normalize_target: bool = False,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute raw/weighted NC-DFM and detached scalar diagnostics."""
    zero = model_output.new_zeros(())
    if pair_i.numel() == 0:
        metrics = {
            "ncdfm_loss_raw": zero.detach(),
            "ncdfm_loss_weighted": zero.detach(),
            "ncdfm_schedule_mean": zero.detach(),
            "ncdfm_schedule_min": zero.detach(),
            "ncdfm_schedule_max": zero.detach(),
            "ncdfm_pred_diff_norm": zero.detach(),
            "ncdfm_target_diff_norm": zero.detach(),
            "ncdfm_pred_target_cosine": zero.detach(),
        }
        return zero, metrics

    pred_diff = model_output[pair_i] - model_output[pair_j]
    target_diff = model_target[pair_i] - model_target[pair_j]
    if normalize_target:
        # Optional scale ablation. The default exactly follows the definition.
        scale = mean_flat(target_diff.float().square()).sqrt().clamp_min(eps)
        view_shape = (scale.shape[0],) + (1,) * (target_diff.ndim - 1)
        pred_diff = pred_diff / scale.view(view_shape).to(pred_diff.dtype)
        target_diff = target_diff / scale.view(view_shape).to(target_diff.dtype)

    pair_loss = mean_flat((pred_diff - target_diff).square())
    pair_schedule = schedule_weight[pair_i].reshape(pair_i.numel(), -1).mean(dim=1)
    weighted_loss = (float(lambda_max) * pair_schedule * pair_loss).mean()

    pred_flat = pred_diff.float().flatten(1)
    target_flat = target_diff.float().flatten(1)
    cosine = F.cosine_similarity(pred_flat, target_flat, dim=1, eps=eps).mean()
    metrics = {
        "ncdfm_loss_raw": pair_loss.mean().detach(),
        "ncdfm_loss_weighted": weighted_loss.detach(),
        "ncdfm_schedule_mean": pair_schedule.mean().detach(),
        "ncdfm_schedule_min": pair_schedule.min().detach(),
        "ncdfm_schedule_max": pair_schedule.max().detach(),
        "ncdfm_pred_diff_norm": pred_flat.norm(dim=1).mean().detach(),
        "ncdfm_target_diff_norm": target_flat.norm(dim=1).mean().detach(),
        "ncdfm_pred_target_cosine": cosine.detach(),
    }
    return weighted_loss, metrics


class SILoss:
    """Standard SiT flow-matching loss plus an optional NC-DFM term."""

    def __init__(
        self,
        prediction: str = "v",
        path_type: str = "linear",
        weighting: str = "uniform",
        accelerator=None,
        latents_scale=None,
        latents_bias=None,
        ncdfm_enabled: bool = False,
        ncdfm_lambda: float = 0.1,
        ncdfm_pair_fraction: float = 1.0,
        ncdfm_pairing: str = "random",
        ncdfm_schedule: str = "signal_ratio",
        ncdfm_schedule_eps: float = 1e-8,
        ncdfm_normalize_target: bool = False,
    ):
        if not 0.0 <= ncdfm_pair_fraction <= 1.0:
            raise ValueError("ncdfm_pair_fraction must be in [0, 1]")
        if ncdfm_lambda < 0.0:
            raise ValueError("ncdfm_lambda must be non-negative")
        if ncdfm_pairing != "random":
            raise ValueError("Only random NC-DFM pairing is currently supported")
        if ncdfm_schedule not in {"constant", "signal_ratio"}:
            raise ValueError("ncdfm_schedule must be 'constant' or 'signal_ratio'")
        if ncdfm_schedule_eps <= 0.0:
            raise ValueError("ncdfm_schedule_eps must be positive")

        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.ncdfm_enabled = ncdfm_enabled
        self.ncdfm_lambda = ncdfm_lambda
        self.ncdfm_pair_fraction = ncdfm_pair_fraction
        self.ncdfm_schedule = ncdfm_schedule
        self.ncdfm_schedule_eps = ncdfm_schedule_eps
        self.ncdfm_normalize_target = ncdfm_normalize_target

    def interpolant(self, t: torch.Tensor):
        if self.path_type == "linear":
            return 1 - t, t, -1, 1
        if self.path_type == "cosine":
            angle = t * np.pi / 2
            return (
                torch.cos(angle),
                torch.sin(angle),
                -np.pi / 2 * torch.sin(angle),
                np.pi / 2 * torch.cos(angle),
            )
        raise NotImplementedError()

    def _sample_time(self, images: torch.Tensor) -> torch.Tensor:
        shape = (images.shape[0], 1, 1, 1)
        if self.weighting == "uniform":
            time_input = torch.rand(shape)
        elif self.weighting == "lognormal":
            sigma = torch.randn(shape).exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
            else:
                raise NotImplementedError()
        else:
            raise ValueError(f"Unsupported timestep weighting: {self.weighting}")
        return time_input.to(device=images.device, dtype=images.dtype)

    def __call__(
        self,
        model,
        images: torch.Tensor,
        model_kwargs: Optional[dict] = None,
        class_labels=None,
    ):
        del class_labels  # Pairing is deliberately class-agnostic in this version.
        model_kwargs = {} if model_kwargs is None else model_kwargs

        # Disabled mode intentionally executes the exact baseline RNG sequence:
        # timestep sampling followed immediately by Gaussian-noise sampling.
        time_input = self._sample_time(images)
        noises = torch.randn_like(images)
        pair_i = torch.empty(0, device=images.device, dtype=torch.long)
        pair_j = torch.empty(0, device=images.device, dtype=torch.long)
        if self.ncdfm_enabled:
            pair_i, pair_j = build_ncdfm_pairs(
                images.shape[0], self.ncdfm_pair_fraction, images.device
            )
            time_input, noises = share_pair_stochasticity(time_input, noises, pair_i, pair_j)

        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction != "v":
            raise NotImplementedError("NC-DFM currently follows the repository's v-prediction path")
        model_target = d_alpha_t * images + d_sigma_t * noises

        # The baseline model exposes exactly the four velocity channels. This is
        # the sole model forward for both FM and NC-DFM.
        model_output, _ = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target).square())

        schedule_weight = compute_ncdfm_schedule(
            alpha_t, sigma_t, self.ncdfm_schedule, self.ncdfm_schedule_eps
        )
        ncdfm_weighted, metrics = compute_ncdfm_loss(
            model_output,
            model_target,
            pair_i,
            pair_j,
            schedule_weight,
            self.ncdfm_lambda,
            self.ncdfm_normalize_target,
            self.ncdfm_schedule_eps,
        )
        
        num_pairs = pair_i.numel()
        metrics.update(
            {
                "ncdfm_num_pairs": model_output.new_tensor(float(num_pairs)).detach(),
                "ncdfm_pair_fraction_actual": model_output.new_tensor(
                    float(2 * num_pairs) / max(images.shape[0], 1)
                ).detach(),
            }
        )
        return denoising_loss, ncdfm_weighted, metrics
