"""Checkpoint-compatible diagonal correction for sparse routed attention."""
from functools import lru_cache
import math

import torch
import torch.nn.functional as F
from timm.models.vision_transformer import Attention


@lru_cache(maxsize=1)
def _compiled_flex():
    from torch.nn.attention.flex_attention import flex_attention
    def run(q, k, v, score_mod, scale):
        # Explicit guards unify symbolic dimensions in PyTorch 2.5's lowering.
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert q.shape[-2] == k.shape[-2] == v.shape[-2]
        return flex_attention(q,k,v,score_mod=score_mod,scale=scale)
    # PyTorch 2.5 FlexAttention cannot reliably lower symbolic QKV strides.
    return torch.compile(run, fullgraph=True, dynamic=False)


@lru_cache(maxsize=64)
def _score_mod(bias):
    def modify(score, batch, head, query, key):
        return score + torch.where(query == key, bias, 0.)
    return modify


def corrected_attention(q, k, v, bias, backend='sdpa', scale=None):
    """Add bias only to self logits, with full autograd through Q/K/V."""
    if not bias or q.shape[-2] == 1:
        return F.scaled_dot_product_attention(q, k, v, scale=scale)
    if backend == 'flex':
        if q.device.type != 'cuda':
            raise RuntimeError('Flex correction requires CUDA; use --tread-attn-correction-backend sdpa on CPU')
        return _compiled_flex()(q, k, v, score_mod=_score_mod(bias), scale=scale)
    if backend != 'sdpa':
        raise ValueError(f'Unknown correction backend: {backend}')
    # Shared KxK bias; never allocate a BxHxKxK tensor explicitly.
    mask = torch.zeros((q.shape[-2], k.shape[-2]), device=q.device, dtype=q.dtype)
    mask.diagonal().fill_(bias)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


class CorrectedAttention(Attention):
    def forward(self, x, attn_mask=None):
        batch, count, _ = x.shape
        # Dense execution and zero-strength ablation use the original exact path.
        if count >= self.correction_full_tokens or self.correction_strength == 0 or count == 1:
            return super().forward(x, attn_mask=attn_mask)
        if attn_mask is not None:
            raise ValueError('TREAD diagonal correction does not support additional attention masks')
        if self.training and self.attn_drop.p:
            raise ValueError('TREAD diagonal correction requires attention dropout=0')
        q,k,v = self.qkv(x).reshape(batch,count,3,self.num_heads,self.head_dim).permute(2,0,3,1,4).unbind(0)
        q,k = self.q_norm(q),self.k_norm(k)
        bias = self.correction_strength * math.log((count-1)/(self.correction_full_tokens-1))
        output = corrected_attention(q,k,v,bias,self.correction_backend,self.scale)
        output = output.transpose(1,2).reshape(batch,count,self.attn_dim)
        return self.proj_drop(self.proj(self.norm(output)))


def configure_correction(model, enabled, strength, backend):
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('tread_attn_correction_strength must be finite and nonnegative')
    if backend not in ('flex','sdpa'):
        raise ValueError('tread_attn_correction_backend must be flex or sdpa')
    if enabled and not model.use_tread_routing:
        raise ValueError('--tread-attn-correction requires --use-tread-routing')
    model.tread_attn_correction = enabled
    model.tread_attn_correction_strength = strength
    model.tread_attn_correction_backend = backend
    if not enabled:
        return
    blocks = model.recurrent_group_blocks if model.tread_recursive else model.blocks[model.tread_start_block:model.tread_end_block]
    for block in blocks:
        # Preserve every parameter object, name, initializer, and state_dict key.
        block.attn.__class__ = CorrectedAttention
        block.attn.correction_full_tokens = model.x_embedder.num_patches
        block.attn.correction_strength = strength
        block.attn.correction_backend = backend


def log_resume_correction(logger, saved, args):
    """An explicit CLI change is allowed for the requested continuation ablation."""
    defaults = dict(tread_attn_correction=False,tread_attn_correction_strength=1.,tread_attn_correction_backend='sdpa')
    for key,default in defaults.items():
        previous = saved.get(key,default) if isinstance(saved,dict) else getattr(saved,key,default)
        current = getattr(args,key,default)
        if previous != current:
            logger.warning('Attention correction continuation override: %s: %s -> %s',key,previous,current)
