# SiT attention alignment + relational-L1 diversity

This experiment is isolated from `attention_alignment_hdiv`. It preserves the
existing Flow Matching, attention-alignment, and CKA H-div implementations and
adds direct row-wise relational L1 diversity as an independent ablation. The
model remains `models/sit_b1.py`; no trainable parameter or inference
architecture is added.

## Selected blocks

Depths use the existing 1-based convention. The final depth is the shared deep
target and all preceding depths are shallow sources.

```text
--encoder-depths 3 5 9
attention:  mean(block 3 -> 9, block 5 -> 9)
row-L1:     mean(output 3 vs stopgrad(output 9),
                 output 5 vs stopgrad(output 9))
```

Hidden features are captured by forward hooks on the complete selected blocks,
after their attention, MLP, and residual updates.

## Row-L1 objective

For each `[B,T,D]` shallow/deep pair, computation is explicitly FP32:

```text
hs = L2_normalize(h_shallow, dim=D)
hd = L2_normalize(stopgrad(h_deep), dim=D)

Rs = hs @ hs^T                                      # [B,T,T]
Rd = hd @ hd^T                                      # [B,T,T]
diff = abs(Rs - Rd) * off_diagonal_mask
row_dist = sum(diff, dim=-1) / (T - 1)              # [B,T]
L_row = mean_B,T(relu(row_hdiv_margin - row_dist))
```

Token features are L2-normalized exactly once. Relation rows are not normalized
again, and no row cosine similarity is computed. The diagonal self-similarity
is excluded before the off-diagonal mean.

`row_hdiv_margin` is the minimum desired relational distance. Therefore:

```text
row_dist < margin   -> active, loss = margin - row_dist
row_dist >= margin  -> inactive, loss = 0
```

Row-L1 always detaches the deep branch only for this auxiliary loss. FM and
attention alignment continue to update the model normally.

The total objective remains:

```text
L_total = L_FM + kl_coeff * L_attn + hdiv_weight * L_hdiv
```

`L_hdiv` is selected by `--hdiv-mode {none,cka,row_l1}`. Multiple shallow
sources are averaged, so adding another shallow depth does not simply multiply
the loss.

## Training

```bash
accelerate launch --multi_gpu --num_processes 8 \
  -m attention_alignment_row_hdiv.train \
  --exp-name sit_b2_attn_row_l1 \
  --model SiT-B/2 \
  --data-dir /v/mnt/GH/imagenet_256 \
  --output-dir /v/mnt/GH/SiT \
  --batch-size 256 \
  --mixed-precision bf16 \
  --allow-tf32 \
  --max-train-steps 400000 \
  --encoder-depths 3 5 9 \
  --attn-loss-type kl \
  --kl-coeff 0.1 \
  --hdiv-mode row_l1 \
  --hdiv-weight 0.1 \
  --row-hdiv-margin 0.1
```

Ablations:

```text
--hdiv-mode none      # original FM + attention; no hidden hooks
--hdiv-mode cka       # unchanged existing CKA H-div
--hdiv-mode row_l1    # direct off-diagonal relational L1 margin
--hdiv-weight 0       # effective mode=none; skips hidden hooks/computation
```

For CKA mode, `--hdiv-cka-threshold` and
`--hdiv-detach-deep/--no-hdiv-detach-deep` retain their previous meanings.
`--hdiv-detach-deep` does not control row-L1 because row-L1 always detaches the
deep reference.

## Logging

Existing metrics remain:

```text
loss_total, loss_fm, loss_attn, loss_hdiv
loss_attn_weighted, loss_hdiv_weighted
hdiv_cka_mean, hdiv_distance_mean, hdiv_active_ratio
```

Row-L1 additionally logs exact global DDP statistics:

```text
loss_row_hdiv
row_dist_mean, row_dist_std, row_dist_min, row_dist_max
row_active_ratio
```

`row_active_ratio` is the fraction of all shallow-pair/image/token distances
below `--row-hdiv-margin`. Add `--row-hdiv-log-quantiles` to gather distances
and log `row_dist_p25`, `row_dist_median`, and `row_dist_p75`. It is disabled by
default to avoid a distributed gather and quantile sort every training step.

## Generation

All auxiliary losses and hooks are training-only. Generation uses the ordinary
SiT model and EMA weight by default:

```bash
torchrun --nnodes=1 --nproc_per_node=8 --master_port=10005 \
  -m attention_alignment_row_hdiv.generate \
  --ckpt /v/mnt/GH/SiT/sit_b2_attn_row_l1/checkpoints/0400000.pt \
  --sample-dir /v/mnt/GH/samples \
  --num-fid-samples 50000 \
  --per-proc-batch-size 32 \
  --mode sde \
  --num-steps 250 \
  --cfg-scale 1.0
```

## Tests

```bash
python -m unittest -v attention_alignment_row_hdiv.tests.test_row_hdiv
```

Tests cover the exact direct-L1 formula, one-time feature normalization,
diagonal exclusion, both hinge cases, per-image isolation, FP32/BF16 stability,
shallow-only row-L1 gradient, multiple-shallow averaging, zero-weight
disabling, and exact none/CKA compatibility with the existing loss.
