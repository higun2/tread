"""Sparse Guidance (SG_FLOPS) samplers.

Krause et al., "Guiding Token-Sparse Diffusion Models" (arXiv 2601.01608), Eq. 7:
    D_SG = w * D(c; gamma_strong) + (1 - w) * D(c; gamma_weak)
Both branches are the same class-conditional checkpoint run through the sparse
TREAD path; gamma is the routed (dropped) token fraction, so the branch keeps
1 - gamma of the tokens. Each forward draws a fresh random subset, as in the
official ``RF.sample_tread`` (github.com/CompVis/tread, rf.py).

The loops mirror ``samplers.euler_sampler`` / ``euler_maruyama_sampler`` with
the CFG combination replaced by SG. Inside [guidance_low, guidance_high] both
branches run; outside it only the strong branch runs (w = 1).
"""

import torch

from samplers import compute_diffusion, get_score_from_velocity


def validate_sparse_guidance(scale, gamma_strong, gamma_weak):
    if scale < 1.0:
        raise ValueError("--sg-scale must be >= 1.0")
    if not 0.0 <= gamma_strong < gamma_weak < 1.0:
        raise ValueError("Sparse Guidance needs 0 <= gamma_strong < gamma_weak < 1")


def _sg_velocity(model, x, t, y, scale, gamma_strong, gamma_weak, guided):
    def branch(gamma):
        return model(x, t, y=y, tread_eval_mode="sparse",
                     tread_active_ratio=1.0 - gamma)[0].to(torch.float64)

    v_strong = branch(gamma_strong)
    if not guided:
        return v_strong
    v_weak = branch(gamma_weak)
    return v_weak + scale * (v_strong - v_weak)


@torch.no_grad()
def sg_euler_maruyama_sampler(
        model, latents, y, num_steps=20, sg_scale=2.5, sg_gamma_strong=0.5,
        sg_gamma_weak=0.9, guidance_low=0.0, guidance_high=1.0, path_type="linear",
        **_unused,
        ):
    validate_sparse_guidance(sg_scale, sg_gamma_strong, sg_gamma_weak)
    _dtype = latents.dtype
    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    def drift(x_cur, t_cur):
        guided = sg_scale > 1.0 and guidance_low <= t_cur <= guidance_high
        time_input = torch.ones(x_cur.size(0), device=device, dtype=torch.float64) * t_cur
        v_cur = _sg_velocity(model, x_cur.to(_dtype), time_input.to(_dtype), y,
                             sg_scale, sg_gamma_strong, sg_gamma_weak, guided)
        s_cur = get_score_from_velocity(v_cur, x_cur, time_input, path_type=path_type)
        return v_cur - 0.5 * compute_diffusion(t_cur) * s_cur

    for t_cur, t_next in zip(t_steps[:-2], t_steps[1:-1]):
        dt = t_next - t_cur
        x_cur = x_next
        deps = torch.randn_like(x_cur) * torch.sqrt(torch.abs(dt))
        x_next = x_cur + drift(x_cur, t_cur) * dt + torch.sqrt(compute_diffusion(t_cur)) * deps

    # last step: deterministic mean
    t_cur, t_next = t_steps[-2], t_steps[-1]
    return x_next + (t_next - t_cur) * drift(x_next, t_cur)


@torch.no_grad()
def sg_euler_sampler(
        model, latents, y, num_steps=20, sg_scale=2.5, sg_gamma_strong=0.5,
        sg_gamma_weak=0.9, guidance_low=0.0, guidance_high=1.0, heun=False,
        **_unused,
        ):
    validate_sparse_guidance(sg_scale, sg_gamma_strong, sg_gamma_weak)
    if heun:
        raise NotImplementedError("Sparse Guidance sampler supports Euler only")
    _dtype = latents.dtype
    t_steps = torch.linspace(1, 0, num_steps + 1, dtype=torch.float64)
    x_next = latents.to(torch.float64)
    device = x_next.device
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        x_cur = x_next
        guided = sg_scale > 1.0 and guidance_low <= t_cur <= guidance_high
        time_input = torch.ones(x_cur.size(0), device=device, dtype=torch.float64) * t_cur
        v_cur = _sg_velocity(model, x_cur.to(_dtype), time_input.to(_dtype), y,
                             sg_scale, sg_gamma_strong, sg_gamma_weak, guided)
        x_next = x_cur + (t_next - t_cur) * v_cur
    return x_next
