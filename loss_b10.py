"""Flow-matching objectives and state utilities for the ParityFlow prototype."""

from typing import Dict, Tuple

import numpy as np
import torch


def mean_flat(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=tuple(range(1, x.ndim)))


def patchify_state(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert BCHW latent states to BND raw state patches."""
    b, c, h, w = x.shape
    if h % patch_size or w % patch_size:
        raise ValueError("Latent spatial dimensions must be divisible by patch_size")
    return (
        x.reshape(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(b, (h // patch_size) * (w // patch_size), patch_size * patch_size * c)
    )


def unpatchify_state(
    tokens: torch.Tensor, patch_size: int, channels: int, height: int, width: int
) -> torch.Tensor:
    """Convert BND raw state patches back to BCHW latent states."""
    b = tokens.shape[0]
    gh, gw = height // patch_size, width // patch_size
    if tokens.shape[1:] != (gh * gw, patch_size * patch_size * channels):
        raise ValueError("Token shape is incompatible with requested latent shape")
    return (
        tokens.reshape(b, gh, gw, patch_size, patch_size, channels)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(b, channels, height, width)
    )


def create_orthonormal_parity_matrix(
    num_parity_tokens: int, num_image_tokens: int, seed: int, dtype=torch.float32
) -> torch.Tensor:
    """Create deterministic row-orthonormal A[K,N] using a stable QR factorization."""
    if not 0 < num_parity_tokens <= num_image_tokens:
        raise ValueError("num_parity_tokens must be in [1, num_image_tokens]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    basis = torch.randn(num_image_tokens, num_parity_tokens, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(basis, mode="reduced")
    return q.T.contiguous().to(dtype=dtype)


def compute_clean_parity(parity_matrix: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
    return torch.einsum("kn,bnd->bkd", parity_matrix.to(image_tokens.dtype), image_tokens)


def compute_syndrome(
    parity_clean: torch.Tensor, image_clean: torch.Tensor, parity_matrix: torch.Tensor
) -> torch.Tensor:
    return parity_clean - compute_clean_parity(parity_matrix, image_clean)


def project_to_parity_constraint(
    image_clean: torch.Tensor,
    parity_clean: torch.Tensor,
    parity_matrix: torch.Tensor,
    gamma: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the symmetric soft projection; gamma=.5 is exact for orthonormal rows."""
    residual = compute_syndrome(parity_clean, image_clean, parity_matrix)
    while gamma.ndim < residual.ndim:
        gamma = gamma.unsqueeze(-1)
    image_delta = torch.einsum(
        "nk,bkd->bnd", parity_matrix.T.to(residual.dtype), residual
    )
    return image_clean + gamma * image_delta, parity_clean - gamma * residual, residual


class SILoss:
    """Standard image FM plus optional parity FM and clean-syndrome consistency."""

    def __init__(
        self,
        prediction="v",
        path_type="linear",
        weighting="uniform",
        parityflow_enabled=False,
        parity_fm_weight=0.1,
        syndrome_loss_weight=0.1,
        syndrome_schedule="signal_ratio",
        schedule_eps=1e-8,
        control_only=False,
        **_unused,
    ):
        if prediction != "v":
            raise NotImplementedError("This repository currently supports v-prediction only")
        if parity_fm_weight < 0 or syndrome_loss_weight < 0:
            raise ValueError("ParityFlow loss weights must be non-negative")
        if syndrome_schedule not in {"constant", "signal_ratio"}:
            raise ValueError("syndrome_schedule must be constant or signal_ratio")
        self.prediction = prediction
        self.path_type = path_type
        self.weighting = weighting
        self.parityflow_enabled = parityflow_enabled
        self.parity_fm_weight = parity_fm_weight
        self.syndrome_loss_weight = syndrome_loss_weight
        self.syndrome_schedule = syndrome_schedule
        self.schedule_eps = schedule_eps
        self.control_only = control_only

    def interpolant(self, t):
        if self.path_type == "linear":
            return 1 - t, t, -1, 1
        if self.path_type == "cosine":
            angle = t * np.pi / 2
            return torch.cos(angle), torch.sin(angle), -np.pi / 2 * torch.sin(angle), np.pi / 2 * torch.cos(angle)
        raise NotImplementedError(self.path_type)

    def sample_time(self, images):
        shape = (images.shape[0], 1, 1, 1)
        if self.weighting == "uniform":
            t = torch.rand(shape)
        elif self.weighting == "lognormal":
            sigma = torch.randn(shape).exp()
            t = sigma / (1 + sigma) if self.path_type == "linear" else 2 / np.pi * torch.atan(sigma)
        else:
            raise ValueError(f"Unsupported weighting: {self.weighting}")
        return t.to(device=images.device, dtype=images.dtype)

    def schedule(self, alpha, sigma):
        if self.syndrome_schedule == "constant":
            return torch.ones_like(alpha)
        return alpha.square() / (alpha.square() + sigma.square() + self.schedule_eps)

    def clean_from_velocity(self, state, velocity, alpha, sigma, d_alpha, d_sigma):
        """Solve the affine 2x2 system for x_0 without assuming a t orientation."""
        determinant = alpha * d_sigma - sigma * d_alpha
        return (d_sigma * state - sigma * velocity) / determinant

    def __call__(self, model, images, model_kwargs=None):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        model_ref = model.module if hasattr(model, "module") else model
        t = self.sample_time(images)
        image_noise = torch.randn_like(images)
        alpha, sigma, d_alpha, d_sigma = self.interpolant(t)
        image_t = alpha * images + sigma * image_noise
        image_target = d_alpha * images + d_sigma * image_noise

        zero = images.new_zeros(())
        if not self.parityflow_enabled:
            image_pred, _ = model(image_t, t.flatten(), **model_kwargs)
            return mean_flat((image_pred - image_target).square()), zero, zero, {}

        if self.control_only:
            image_pred, _ = model(image_t, t.flatten(), parity_t=None, **model_kwargs)
            return mean_flat((image_pred - image_target).square()), zero, zero, {
                "extra_token_ratio": image_pred.new_tensor(model_ref.num_parity_tokens / model_ref.num_image_tokens),
                "estimated_attention_flops_ratio": image_pred.new_tensor(((model_ref.num_image_tokens + model_ref.num_parity_tokens) / model_ref.num_image_tokens) ** 2),
            }

        image_tokens = patchify_state(images, model_ref.patch_size)
        parity_0 = compute_clean_parity(model_ref.parity_matrix, image_tokens)
        parity_noise = torch.randn_like(parity_0)  # independent of image_noise
        parity_alpha = alpha.flatten(1).mean(1).view(-1, 1, 1)
        parity_sigma = sigma.flatten(1).mean(1).view(-1, 1, 1)
        parity_d_alpha = (
            d_alpha.flatten(1).mean(1).view(-1, 1, 1)
            if torch.is_tensor(d_alpha) else d_alpha
        )
        parity_d_sigma = (
            d_sigma.flatten(1).mean(1).view(-1, 1, 1)
            if torch.is_tensor(d_sigma) else d_sigma
        )
        parity_t = parity_alpha * parity_0 + parity_sigma * parity_noise
        parity_target = parity_d_alpha * parity_0 + parity_d_sigma * parity_noise

        image_pred, parity_pred = model(image_t, t.flatten(), parity_t=parity_t, **model_kwargs)
        image_loss = mean_flat((image_pred - image_target).square())
        parity_fm_raw = mean_flat((parity_pred - parity_target).square())
        parity_fm = self.parity_fm_weight * parity_fm_raw.mean()

        image_0_pred = self.clean_from_velocity(image_t, image_pred, alpha, sigma, d_alpha, d_sigma)
        parity_0_pred = self.clean_from_velocity(
            parity_t, parity_pred, parity_alpha, parity_sigma, parity_d_alpha, parity_d_sigma
        )
        syndrome = compute_syndrome(
            parity_0_pred, patchify_state(image_0_pred, model_ref.patch_size), model_ref.parity_matrix
        )
        syndrome_per_sample = mean_flat(syndrome.square())
        weight = self.schedule(alpha, sigma).flatten(1).mean(1)
        syndrome_weighted = self.syndrome_loss_weight * (weight * syndrome_per_sample).mean()

        metrics: Dict[str, torch.Tensor] = {
            "parity_fm_loss": parity_fm_raw.mean().detach(),
            "syndrome_loss_raw": syndrome_per_sample.mean().detach(),
            "syndrome_loss_weighted": syndrome_weighted.detach(),
            "syndrome_norm": syndrome.float().flatten(1).norm(dim=1).mean().detach(),
            "parity_target_norm": parity_target.float().flatten(1).norm(dim=1).mean().detach(),
            "parity_prediction_norm": parity_pred.float().flatten(1).norm(dim=1).mean().detach(),
            "parity_clean_norm": parity_0_pred.float().flatten(1).norm(dim=1).mean().detach(),
            "syndrome_weight_mean": weight.mean().detach(),
            "extra_token_ratio": image_pred.new_tensor(model_ref.num_parity_tokens / model_ref.num_image_tokens),
            "estimated_attention_flops_ratio": image_pred.new_tensor(((model_ref.num_image_tokens + model_ref.num_parity_tokens) / model_ref.num_image_tokens) ** 2),
        }
        return image_loss, parity_fm, syndrome_weighted, metrics
