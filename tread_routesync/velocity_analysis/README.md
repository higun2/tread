# Sparse velocity analysis

Checkpoint inference only: the analysis never trains, changes checkpoint weights,
or evaluates a dense teacher. `R` denotes bypass tokens, `P` processed tokens.
Random token routing preserves original spatial indices, so adjacency is measurable.

## Reproduce

Run from the repository root (choose an appropriate GPU):

```bash
CUDA_VISIBLE_DEVICES=4 python -m tread_routesync.velocity_analysis.run \
  --checkpoint /v/mnt/GH/SiT/tread-routesync-b/checkpoints/0400000.pt \
  --output tread_routesync/velocity_analysis/results_0400000 \
  --samples 256 --batch-size 8 --route-repeats 2 \
  --rollout-samples 32 --rollout-steps 50
python -m tread_routesync.velocity_analysis.summarize \
  tread_routesync/velocity_analysis/results_0400000
```

The first command uses the checkpoint's dataset path unless `--data-dir` is given.
EMA, BF16 inference and seed 20260922 are defaults. `--precision fp32` changes
inference precision. The reference run used one RTX 6000 Ada GPU.

## Protocol

- **Noised data:** 256 randomly selected training latent files, posterior sampled
  exactly as the trainer (`(mean + std * noise) * 0.18215`), seven timesteps
  `[0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95]`, two routing masks per image.
  `x_t = (1-t)*x0 + t*eps`, conditional flow-matching target `eps-x0`.
  Images, latent samples, noise and corresponding routing masks are reused across
  timesteps; only t changes. This is evaluation-mode inference on training data,
  not an independent held-out generalization estimate.
- **Generated trajectories:** 32 noise-to-latent Euler ODE trajectories, 50 steps,
  CFG=1. One random routing partition stays fixed throughout each trajectory,
  following seeded sparse generation behavior. Six trajectory locations are
  measured. These states have no known ground-truth velocity target. No VAE
  decoding, FID, or image-quality evaluation is performed.
- `t=0` is clean, `t=1` is noise. All calls use `eval()` and sparse execution.
- **Prediction:** final model velocity. **Target:** sampled conditional FM target.
  **Error:** prediction minus that target. This residual includes conditional
  target uncertainty and is not a direct observation of error against the true
  marginal velocity field.

## Pair definitions

All-pair comparisons use unordered, distinct token pairs (no diagonal and no
reverse duplicates). A 16x16 grid with 128 R and 128 P tokens gives 8,128 R-R,
16,384 R-P, and 8,128 P-P pairs per image/forward. Each token vector contains
2x2x4=16 velocity components with identical within-patch ordering.

`adjacent_patches` compares the same 16-component vectors for horizontally or
vertically adjacent patches; diagonal neighbors are excluded. `boundary_pixels`
compares 4-channel latent vectors at actual touching pixels across those patch
boundaries. This is distinct from comparing corresponding offsets in neighboring
patches. Distance bands are measured in Euclidean token-grid units.

Metrics: mean componentwise L1, mean squared component difference, its square
root (RMSE), cosine similarity. Zero-norm pairs are excluded from cosine averages.
Pair counts and valid cosine counts are retained. Absolute cosine levels across
16-dimensional patch and 4-dimensional boundary-pixel metrics are not equivalent.

## Aggregation and uncertainty

Average pairs within image/mask, then masks within image, then images equally.
RMSE is computed within each image/mask before averaging. A contrast is
`R-P - (R-R + P-P)/2`, paired within the same image. 95% CIs use 2,000 image-level
bootstrap resamples. Pair counts are NOT treated as independent sample counts.
Intervals are exploratory/pointwise, with no multiple-comparison correction.
Different timesteps share images/noise and are not independent replicates.

## Artifacts

- `results_0400000/report.md`: Korean findings and limitations.
- `results_0400000/dashboard.html`: standalone interactive plots and contrast table.
- `summary.csv`, `paired_contrasts.csv`, `fm_summary.csv`, `summary.json`: aggregates.
- `per_image_pairs.csv`, `per_image_fm.csv`: image/mask-level measurements.
- `sample_manifest.csv`, `metadata.json`: input identities and reproducibility settings.

The inference code requires torch and numpy; summary and HTML generation need no
additional plotting libraries.
