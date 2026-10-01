# SiT attention alignment + H-diversity

This directory extends `loss_b8.py` / `train_b2.py` without modifying them.
The model remains `models/sit_b1.py`; no projector or inference parameter is
introduced.

## Selected blocks

`--encoder-depths` preserves the existing 1-based convention. The final depth
is the deep target and every preceding depth is a shallow source.

```text
--encoder-depths 3 9
  attention: block 3 -> block 9
  H-div:     output(block 3) vs stopgrad(output(block 9))

--encoder-depths 3 5 9
  attention: mean(block 3 -> 9, block 5 -> 9)
  H-div:     mean(output 3 vs 9, output 5 vs 9)
```

Attention maps retain the existing attention-module forward hook and KL/JS/L1/L2
implementation. H-div uses a separate hook on the complete selected block, so
`saved_hidden` is captured after attention, MLP, and both residual updates. Only
selected block outputs are retained.

## H-div objective

For every image independently, the selected `[T,D]` features are centered over
tokens and linear CKA is computed in FP32 using token Gram matrices:

```text
CKA = ||Xc^T Yc||F^2 / (||Xc^T Xc||F ||Yc^T Yc||F + eps)
    = <Xc Xc^T, Yc Yc^T>F / (||Xc Xc^T||F ||Yc Yc^T||F + eps)
Y = stopgrad(H_deep)  # default; configurable
distance = acos(clamp(CKA, eps, 1-eps))
margin = acos(hdiv_cka_threshold)
L_hdiv = relu(margin - distance)
```

Thus H-div is active exactly when CKA exceeds the configured threshold and
becomes zero after sufficient diversity. By default gradients flow to the
shallow hidden state only. Add `--no-hdiv-detach-deep` to allow H-div gradients
through both selected block outputs. Existing attention alignment continues to
detach its deep attention target regardless of this option.

The total loss is:

```text
L_total = L_FM + kl_coeff * L_attn + hdiv_weight * L_hdiv
```

## Training

One shallow/deep pair on SiT-B/2:

```bash
accelerate launch --multi_gpu --num_processes 8 \
  -m attention_alignment_hdiv.train \
  --exp-name sit_b2_attn3_hdiv9 \
  --model SiT-B/2 \
  --data-dir /v/mnt/GH/imagenet_256 \
  --output-dir /v/mnt/GH/SiT \
  --batch-size 256 \
  --mixed-precision bf16 \
  --allow-tf32 \
  --max-train-steps 400000 \
  --encoder-depths 3 9 \
  --attn-loss-type kl \
  --kl-coeff 0.3 \
  --enable-hdiv \
  --hdiv-weight 0.1 \
  --hdiv-cka-threshold 0.8 \
  --hdiv-detach-deep
```

Use `--no-enable-hdiv` or `--hdiv-weight 0` for the original FM + attention
behavior. In either case, no hidden-output hooks are installed.

To update both shallow and deep representations from H-div:

```bash
--no-hdiv-detach-deep
```

Logged metrics are global means across DDP ranks:

```text
loss_total, loss_fm, loss_attn, loss_hdiv
loss_attn_weighted, loss_hdiv_weighted
hdiv_cka_mean, hdiv_distance_mean, hdiv_active_ratio
```

## Generation

Generation uses the ordinary `models.sit_b1` inference path. Attention
alignment and H-div hooks are training-only and are not installed. By default,
the generator loads `ema` and restores the model, resolution, class count,
attention options, and path type from the checkpoint's saved training args.

```bash
torchrun --nnodes=1 --nproc_per_node=8 --master_port=10005 \
  -m attention_alignment_hdiv.generate \
  --ckpt /v/mnt/GH/SiT/sit_b2_attn3_hdiv9/checkpoints/0400000.pt \
  --sample-dir /v/mnt/GH/samples \
  --num-fid-samples 50000 \
  --per-proc-batch-size 32 \
  --mode sde \
  --num-steps 250 \
  --cfg-scale 1.0
```

Explicit `--model SiT-B/2 --resolution 256 --path-type linear` arguments may
be supplied, but are unnecessary for checkpoints produced by this folder.
Use `--state-key model` only when intentionally evaluating non-EMA weights.

## Memory and distributed behavior

The implementation uses two per-sample `[B_local,T,T]` FP32 token Gram tensors
per shallow/deep CKA pair. This is mathematically equivalent to the original
covariance form but avoids its three `[B_local,D,D]` tensors. For SiT-B/2 with
local batch 32, `T=256`, and `D=768`, these explicit forward intermediates drop
from approximately 216 MiB to 16 MiB per pair, before autograd storage.
Multiple shallow depths still add proportional work. It never constructs a
cross-image `[B*T,B*T]` Gram matrix.

CKA and margin loss are computed independently on each rank's local samples;
DDP averages gradients normally. Logged scalar statistics are explicitly
reduced across ranks.

## Tests

```bash
python -m unittest -v attention_alignment_hdiv.tests.test_hdiv
```

Tests verify exact disabled equivalence with `loss_b8`, per-sample CKA,
threshold behavior, shallow-only gradient, deep stop-gradient, block-output
capture, no projector, and CPU/CUDA BF16 stability.
