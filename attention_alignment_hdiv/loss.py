"""Existing attention-alignment objective extended with H-diversity."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def mean_flat(x):
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def _validate_hidden_pair(h_shallow: torch.Tensor, h_deep: torch.Tensor) -> None:
    if h_shallow.ndim != 3 or h_deep.ndim != 3:
        raise ValueError(
            f"H-div expects [B, T, D], got {tuple(h_shallow.shape)} and {tuple(h_deep.shape)}"
        )
    if h_shallow.shape != h_deep.shape:
        raise ValueError(
            f"shallow/deep hidden shapes must match, got {tuple(h_shallow.shape)} "
            f"and {tuple(h_deep.shape)}"
        )
    if h_shallow.shape[1] <= 1:
        raise ValueError(f"linear CKA requires T > 1, got T={h_shallow.shape[1]}")


def linear_cka_per_sample(h_shallow: torch.Tensor, h_deep: torch.Tensor,
                          eps: float = 1e-6,
                          detach_deep: bool = True) -> torch.Tensor:
    """Differentiable token-Gram linear CKA for each image.

    By default the deep representation is detached, so gradients only update
    the shallow-side graph. ``detach_deep=False`` enables gradients through
    both representations. Computation is explicitly FP32 under AMP. No
    cross-image or flattened ``[B*T, B*T]`` Gram matrix is constructed.

    For centered X and Y, the identities

    ``||X.T @ Y||_F^2 = <X @ X.T, Y @ Y.T>_F`` and
    ``||X.T @ X||_F = ||X @ X.T||_F``

    make this mathematically equivalent to covariance-form linear CKA. SiT-B/2
    has T=256 and D=768, so materializing per-sample ``[T,T]`` matrices is much
    cheaper than the previous three ``[D,D]`` matrices.
    """
    _validate_hidden_pair(h_shallow, h_deep)
    if eps <= 0:
        raise ValueError(f"hdiv_eps must be positive, got {eps}")
    # A surrounding autocast context can downcast FP32 matmuls again, so turn
    # it off for the complete numerically sensitive CKA region.
    with torch.autocast(device_type=h_shallow.device.type, enabled=False):
        x = h_shallow.float()                          # [B, T, D]
        deep_source = h_deep.detach() if detach_deep else h_deep
        y = deep_source.float()                        # [B, T, D]
        x = x - x.mean(dim=1, keepdim=True)            # center over tokens
        y = y - y.mean(dim=1, keepdim=True)            # center over tokens

        gram_x = x @ x.transpose(1, 2)                 # [B, T, T]
        gram_y = y @ y.transpose(1, 2)                 # [B, T, T]
        numerator = (gram_x * gram_y).sum(dim=(-2, -1))  # [B]
        norm_x = gram_x.square().sum(dim=(-2, -1)).sqrt()  # [B]
        norm_y = gram_y.square().sum(dim=(-2, -1)).sqrt()  # [B]
        return numerator / (norm_x * norm_y + eps)     # [B]


def hdiv_margin_loss(h_shallow: torch.Tensor, h_deep: torch.Tensor,
                     cka_threshold: float = 0.8, eps: float = 1e-6,
                     detach_deep: bool = True) -> dict:
    """Angular CKA margin loss and detached analysis statistics."""
    if not 0.0 < cka_threshold < 1.0:
        raise ValueError(
            f"hdiv_cka_threshold must be strictly between 0 and 1, got {cka_threshold}"
        )
    if eps >= 0.5:
        raise ValueError(f"hdiv_eps must be smaller than 0.5, got {eps}")
    cka_raw = linear_cka_per_sample(
        h_shallow, h_deep, eps=eps, detach_deep=detach_deep
    )                                                   # [B]
    cka = cka_raw.clamp(eps, 1.0 - eps)                # [B]
    distance = torch.acos(cka)                         # [B]
    margin = torch.acos(cka.new_tensor(cka_threshold).clamp(eps, 1.0 - eps))
    loss = F.relu(margin - distance)                   # [B]
    return {
        "loss": loss,
        "cka": cka.detach(),
        "distance": distance.detach(),
        "active": (cka_raw.detach() > cka_threshold).float(),
    }


class SILoss:
    """Original FM/attention loss plus optional hidden-state diversity."""

    def __init__(self, prediction="v", path_type="linear", weighting="uniform",
                 attn_loss_type="kl", accelerator=None, latents_scale=None,
                 latents_bias=None, enable_hdiv=False,
                 hdiv_cka_threshold=0.8, hdiv_eps=1e-6,
                 hdiv_detach_deep=True):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.attn_loss_type = attn_loss_type
        self.enable_hdiv = bool(enable_hdiv)
        self.hdiv_cka_threshold = float(hdiv_cka_threshold)
        self.hdiv_eps = float(hdiv_eps)
        self.hdiv_detach_deep = bool(hdiv_detach_deep)
        if self.enable_hdiv:
            if not 0 < self.hdiv_cka_threshold < 1:
                raise ValueError("hdiv_cka_threshold must be in (0, 1)")
            if not 0 < self.hdiv_eps < 0.5:
                raise ValueError("hdiv_eps must be in (0, 0.5)")

    def interpolant(self, t):
        if self.path_type == "linear":
            return 1 - t, t, -1, 1
        if self.path_type == "cosine":
            return (
                torch.cos(t * np.pi / 2), torch.sin(t * np.pi / 2),
                -np.pi / 2 * torch.sin(t * np.pi / 2),
                np.pi / 2 * torch.cos(t * np.pi / 2),
            )
        raise NotImplementedError()

    def _attn_loss(self, attn_shallow, attn_deep, eps=1e-8):
        """Kept equivalent to loss_b8.SILoss._attn_loss."""
        attn_shallow = attn_shallow.float()
        attn_deep = attn_deep.float()
        if self.attn_loss_type == "kl":
            log_prob_shallow = attn_shallow.clamp_min(eps).log()
            log_prob_deep = attn_deep.clamp_min(eps).log()
            return F.kl_div(
                log_prob_shallow, log_prob_deep, reduction="none", log_target=True,
            ).sum(dim=-1).mean(dim=(1, 2))
        if self.attn_loss_type == "js":
            log_prob_shallow = attn_shallow.clamp_min(eps).log()
            log_prob_deep = attn_deep.clamp_min(eps).log()
            mixture = 0.5 * (attn_shallow + attn_deep)
            log_mixture = mixture.clamp_min(eps).log()
            loss_deep = F.kl_div(
                log_mixture, log_prob_deep, reduction="none", log_target=True,
            ).sum(dim=-1).mean(dim=(1, 2))
            loss_shallow = F.kl_div(
                log_mixture, log_prob_shallow, reduction="none", log_target=True,
            ).sum(dim=-1).mean(dim=(1, 2))
            return (loss_deep + loss_shallow) / 2
        if self.attn_loss_type == "l1":
            return F.l1_loss(attn_shallow, attn_deep, reduction="none").sum(-1).mean((1, 2))
        if self.attn_loss_type == "l2":
            return F.mse_loss(attn_shallow, attn_deep, reduction="none").sum(-1).mean((1, 2))
        raise ValueError(f"unknown attn_loss_type {self.attn_loss_type!r}")

    @staticmethod
    def _extract(obj, attribute):
        return getattr(obj, attribute) if hasattr(obj, attribute) else obj

    @staticmethod
    def _as_list(obj):
        return list(obj) if isinstance(obj, (list, tuple)) else [obj]

    def __call__(self, model, images, model_kwargs=None, class_labels=None,
                 attn_maps_dict=None, hidden_states_dict=None):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            sigma = torch.randn((images.shape[0], 1, 1, 1)).exp()
            time_input = sigma / (1 + sigma) if self.path_type == "linear" else 2 / np.pi * torch.atan(sigma)
        else:
            raise NotImplementedError(self.weighting)
        time_input = time_input.to(device=images.device, dtype=images.dtype)
        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
        model_input = alpha_t * images + sigma_t * noises
        if self.prediction != "v":
            raise NotImplementedError()
        model_target = d_alpha_t * images + d_sigma_t * noises
        model_output, _ = model(model_input, time_input.flatten(), **model_kwargs)
        denoising_loss = mean_flat((model_output - model_target) ** 2)

        # Existing attention alignment: every shallow source targets the final
        # selected deep attention, with deep attention detached exactly as before.
        attn_loss = torch.zeros_like(denoising_loss)
        valid_shallow_count = 0
        if attn_maps_dict is not None and "shallow" in attn_maps_dict and "deep" in attn_maps_dict:
            shallow_objs = self._as_list(attn_maps_dict["shallow"])
            attn_deep = self._extract(attn_maps_dict["deep"], "saved_attn")
            if attn_deep is not None and shallow_objs:
                for shallow_obj in shallow_objs:
                    attn_shallow = self._extract(shallow_obj, "saved_attn")
                    attn_loss += self._attn_loss(attn_shallow, attn_deep.detach())
                    valid_shallow_count += 1
        if valid_shallow_count > 0:
            attn_loss = attn_loss / valid_shallow_count

        zero = denoising_loss.new_zeros(())
        hdiv = {
            "loss": torch.zeros_like(denoising_loss),
            "cka_mean": zero,
            "distance_mean": zero,
            "active_ratio": zero,
        }
        if self.enable_hdiv:
            if hidden_states_dict is None or "shallow" not in hidden_states_dict or "deep" not in hidden_states_dict:
                raise RuntimeError("H-div is enabled but selected block outputs were not supplied")
            shallow_hidden = self._as_list(hidden_states_dict["shallow"])
            h_deep = self._extract(hidden_states_dict["deep"], "saved_hidden")
            per_source = []
            for shallow_obj in shallow_hidden:
                h_shallow = self._extract(shallow_obj, "saved_hidden")
                if h_shallow is None or h_deep is None:
                    raise RuntimeError("an H-div block-output hook did not capture a tensor")
                per_source.append(hdiv_margin_loss(
                    h_shallow, h_deep,
                    cka_threshold=self.hdiv_cka_threshold, eps=self.hdiv_eps,
                    detach_deep=self.hdiv_detach_deep,
                ))
            if not per_source:
                raise RuntimeError("H-div requires at least one shallow block")
            hdiv = {
                "loss": torch.stack([item["loss"] for item in per_source]).mean(dim=0),
                "cka_mean": torch.stack([item["cka"].mean() for item in per_source]).mean(),
                "distance_mean": torch.stack([item["distance"].mean() for item in per_source]).mean(),
                "active_ratio": torch.stack([item["active"].mean() for item in per_source]).mean(),
            }
        return denoising_loss, attn_loss, hdiv


__all__ = ["SILoss", "hdiv_margin_loss", "linear_cka_per_sample"]
