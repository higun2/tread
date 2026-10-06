# TREAD + DensePush (margin path repulsion)

DensePush is the mirror image of DenseSync (`tread_routesync --use-dense-sparse-sync`).
DenseSync pulls the sparse path toward the dense path (`h_S -> h_D`). DensePush instead
keeps a minimum amount of difference between the dense and sparse paths for the same `x_t`:

```text
z_D = normalize(mean_tokens(h_D)),   z_S = normalize(mean_tokens(h_S))
L_push = mean_over_samples( [cos(z_D, z_S) - tau]_+^2 )
L      = L_FM + lambda_push * L_push
```

There is no gradient once `cos <= tau`, so the paths are only stopped from collapsing into
the same representation; they are never pushed apart without limit (the hinge-margin
variant of Dispersive Loss). The features are pooled globally, so local per-token detail
is not pushed apart.

The package is a copy of `tread_routesync` with DenseSync replaced by DensePush. TREAD
routing, recursion, attention correction and RouteSync are unchanged. DensePush adds no
parameters, and it runs only during sparse training forwards. Evaluation (dense or sparse)
and `densepush.generate` work the same way as in `tread_routesync`.

## Feature location

The routed interval is `[tread_start_block, tread_end_block)`. For a random
`--dense-push-ratio` fraction of the local batch (floor, at least 1), the extra dense pass
starts from the shared prefix output and runs every token through blocks
`tread_start_block .. dense_push_block`. The sparse feature comes from the normal TREAD
forward at the same block:

- `dense_push_block >= tread_end_block` (default `tread_end_block`, the same place as
  DenseSync): the sparse path's full-token state after that suffix block.
- `dense_push_block < tread_end_block` (e.g. 7 or 8 for `[2,9)`): the sparse state restored
  to spatial order. Bypass tokens still hold their pre-route values here.

`--dense-push-tokens` picks the tokens in the mean pool: `all` (default), `active` or
`routed`.

## Gradient

- `--dense-push-grad sparse` (default): the dense pass runs under `no_grad` and is a fixed
  anchor, as in DenseSync. Its cost is one extra no-grad pass over the routed segment for
  the selected samples.
- `--dense-push-grad both`: also backpropagates through the dense pass and the shared
  prefix. Note that this changes the dense path that is used at `--tread-eval-mode dense`.

## Options

| flag | default | meaning |
|---|---|---|
| `--use-dense-push` | off | enable |
| `--dense-push-weight` | 0.1 | lambda_push |
| `--dense-push-margin` | 0.95 | tau, in [-1, 1) |
| `--dense-push-ratio` | 0.1 | fraction of the local batch that gets the dense pass |
| `--dense-push-block` | `tread_end_block` | zero-based block whose output is pooled |
| `--dense-push-tokens` | all | all / active / routed |
| `--dense-push-grad` | sparse | sparse / both |

Logged values: `loss/dense_push`, `dense_push/cosine` (mean pooled cosine), and
`dense_push/above_margin_fraction` (fraction of samples that receive gradient). First
watch `dense_push/cosine` to see the natural gap, then set tau a little below it.

## Usage

```bash
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
  -m densepush.train --model SiT-B/2 --exp-name dense_push \
  --data-dir /root/imagenet_256 --output-dir /v/mnt/GH/SiT \
  --use-tread-routing --tread-start-block 2 --tread-end-block 9 --tread-active-ratio 0.5 \
  --use-dense-push --dense-push-margin 0.95 --dense-push-weight 0.1 --dense-push-ratio 0.1 \
  --batch-size 256 --max-train-steps 400000 --allow-tf32

torchrun --standalone --nproc_per_node=4 -m densepush.generate \
  --ckpt /v/mnt/GH/SiT/dense_push/checkpoints/0400000.pt --sample-dir /root/samples \
  --tread-eval-mode dense --num-fid-samples 50000 --per-proc-batch-size 32 \
  --mode sde --num-steps 250 --cfg-scale 1.0
```

Tests: `python -m unittest discover -s densepush/tests -t .`
