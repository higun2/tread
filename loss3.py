import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import smooth_l1_loss

# simple loss function
class Simpleloss(nn.Module):
    def __init__(self):
        super(Simpleloss, self).__init__()

    def forward(self, a, b, loss_type="sml1"):
        if loss_type == "sml1":
            align_loss = smooth_l1_loss(a, b, beta=0.05)
        elif loss_type == "l2":
            align_loss = F.mse_loss(a, b)
        elif loss_type == "l1":
            align_loss = F.l1_loss(a, b)
        else:
            raise NotImplementedError()
        return align_loss


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class SILoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            loss_type="sml1",
            accelerator=None,
            latents_scale=None,
            latents_bias=None,
            y_disp_tau=0.5,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.loss_type = loss_type
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.y_disp_tau = y_disp_tau

        self.criterion = Simpleloss()

    def _masked_dispersive_loss(self, z, labels):
        bsz = z.shape[0]
        if bsz < 2:
            return z.new_zeros(())

        dist2 = torch.cdist(z, z).pow(2) / z.shape[1]

        labels = labels.to(device=z.device).reshape(-1)
        same_class = labels[:, None] == labels[None, :]
        diagonal = torch.eye(bsz, device=z.device, dtype=torch.bool)
        valid_mask = (~same_class) & (~diagonal)

        if not torch.any(valid_mask):
            return z.new_zeros(())

        valid_dist = dist2[valid_mask]
        return torch.logsumexp(-valid_dist / self.y_disp_tau, dim=0) - torch.log(
            torch.tensor(valid_dist.numel(), device=z.device, dtype=z.dtype)
        )


    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, model_kwargs=None, class_labels=None):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1, 1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)

        time_input = time_input.to(device=images.device, dtype=images.dtype)

        noises = torch.randn_like(images)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)

        model_input = alpha_t * images + sigma_t * noises
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction

        model_output, reg_info = model(
            model_input,
            time_input.flatten(),
            return_y_embeddings=True,
            **model_kwargs,
        )

        denoising_loss = mean_flat((model_output - model_target) ** 2)

        y_ortho_loss = model_output.new_zeros(())
        y_disp_loss = model_output.new_zeros(())

        if isinstance(reg_info, dict) and "y_cond" in reg_info and "y_uncond" in reg_info:
            y_cond = reg_info["y_cond"]
            y_uncond = reg_info["y_uncond"]

            y_residual = y_cond - y_uncond
            y_residual_norm = F.normalize(y_residual, dim=-1)
            y_uncond_norm = F.normalize(y_uncond, dim=-1)
            cos_val = (y_residual_norm * y_uncond_norm).sum(dim=-1)
            y_ortho_loss = cos_val.pow(2).mean()

            z = y_cond.reshape((y_cond.shape[0], -1))
            labels_for_disp = class_labels if class_labels is not None else model_kwargs.get("y", None)
            if labels_for_disp is not None:
                y_disp_loss = self._masked_dispersive_loss(z, labels_for_disp)

        return denoising_loss, y_ortho_loss, y_disp_loss
