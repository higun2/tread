"""TREAD SiT that can expose selected block outputs for LayerSync.

The forward pass is the TREAD forward of ``tread_token_routing`` unchanged; when
``capture_blocks`` is given, the output of each listed block (zero-based) is
returned together with the original token positions it holds:

- prefix / suffix blocks: all tokens in original order (``indices=None``),
- routed blocks: only the active tokens, in ``active_indices`` order.

No parameters are added, so checkpoints share TREAD's state-dict keys and
inference is the ordinary dense SiT forward.
"""

from functools import partial

import torch

from tread_token_routing.model import SiT as TreadSiT
from tread_token_routing.model import gather_tokens, random_subset_indices, restore_token_order


class SiT(TreadSiT):
    def _block_for(self, index):
        if not self.tread_recursive:
            return self.blocks[index]
        if index < self.tread_start_block:
            return self.prefix_blocks[index]
        if index >= self.tread_end_block:
            return self.suffix_blocks[index - self.tread_end_block]
        return self.recurrent_group_blocks[self._recursive_group_for_offset(index - self.tread_start_block)]

    def forward(self, x, t, y, return_logvar=False, force_drop_ids=None,
                capture_blocks=None, return_routing=False, **kwargs):
        if not capture_blocks:
            return super().forward(x, t, y, return_logvar=return_logvar,
                                   force_drop_ids=force_drop_ids, **kwargs)
        capture = set(capture_blocks)
        features = {}
        x = self.x_embedder(x) + self.pos_embed
        batch, tokens, _ = x.shape
        condition = self.t_embedder(t) + self.y_embedder(y, self.training, force_drop_ids=force_drop_ids)
        sparse = self._sparse_for_forward(kwargs.get("tread_eval_mode"))

        for index in range(self.tread_start_block if sparse else 0):
            x = self._block_for(index)(x, condition)
            if index in capture:
                features[index] = (x, None)
        if not sparse:  # dense path (evaluation or routing disabled)
            for index in range(self.base_depth):
                x = self._block_for(index)(x, condition)
                if index in capture:
                    features[index] = (x, None)
            out = (self.unpatchify(self.final_layer(x, condition)), None, features)
            return out + (None,) if return_routing else out

        active_tokens = max(1, min(tokens, int(round(tokens * self.tread_active_ratio))))
        active_indices, bypass_indices = random_subset_indices(
            batch, tokens, active_tokens, x.device, generator=self._subset_generator(x.device))
        permutation = torch.cat((active_indices, bypass_indices), dim=1)
        grouped = gather_tokens(x, permutation)
        active, bypass = grouped[:, :active_tokens], grouped[:, active_tokens:]
        for index in range(self.tread_start_block, self.tread_end_block):
            active = self._run_routed_block(self._block_for(index), active, condition,
                                            endpoint=index == self.tread_end_block - 1)
            if index in capture:
                features[index] = (active, active_indices)
        x = restore_token_order(torch.cat((active, bypass), dim=1), permutation)
        for index in range(self.tread_end_block, self.base_depth):
            x = self._block_for(index)(x, condition)
            if index in capture:
                features[index] = (x, None)
        out = (self.unpatchify(self.final_layer(x, condition)), None, features)
        routing = {"active_indices": active_indices, "bypass_indices": bypass_indices}
        return out + (routing,) if return_routing else out


def build_layersync_sit(model_name="SiT-B/2", **kwargs):
    size, patch = model_name.removeprefix("SiT-").split("/")
    depth, hidden_size, heads = {
        "S": (12, 384, 6), "B": (12, 768, 12),
        "L": (24, 1024, 16), "XL": (28, 1152, 16),
    }[size]
    decoder_hidden_size = 768 if size == "S" else hidden_size
    return SiT(depth=depth, hidden_size=hidden_size, decoder_hidden_size=decoder_hidden_size,
               num_heads=heads, patch_size=int(patch), **kwargs)


SiT_models = {
    f"SiT-{size}/{patch}": partial(build_layersync_sit, f"SiT-{size}/{patch}")
    for size in ("S", "B", "L", "XL") for patch in (2, 4, 8)
}
