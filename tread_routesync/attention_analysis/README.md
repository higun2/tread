# TREAD diagonal-attention audit

Read-only inference analysis; original model/training files and checkpoints are
not modified. Results for the user-provided **TREAD-only** checkpoint live in
`results_tread_v_0400000/`. Open `report.md` or the self-contained interactive
`dashboard.html` (head, layer, and timestep selectors; no internet required).

Run from the repository root:

```bash
python -m tread_routesync.attention_analysis.run \
  --checkpoint /v/mnt/GH/SiT/tread-v/checkpoints/0400000.pt \
  --output tread_routesync/attention_analysis/results_tread_v_0400000 \
  --samples 128 --batch-size 4 --route-repeats 2 --device cuda:0
python -m tread_routesync.attention_analysis.summarize \
  tread_routesync/attention_analysis/results_tread_v_0400000
```

Uses EMA by default (`--state-key model` selects non-EMA). FP32 with TF32 disabled.
Inputs are posterior-sampled ImageNet **training** VAE latents from the checkpoint's
dataset path. These are noised real-image inputs, not generated trajectories.
The masks are deterministic for the recorded seed **and batch size**, and are
saved in `measurements.npz`. Images/noise/masks are paired across all modes and
timesteps. Original file identities are in `sample_manifest.csv`.

The five modes are:

- `dense`: full network, evaluated at the same active query positions.
- `local_sparse`: each dense layer's fixed Q/K/V, restricted to the route subset.
- `local_corrected`: same fixed Q/K/V plus diagonal log((K-1)/(N-1)).
- `full_sparse`: ordinary sparse network with one fixed subset across the route.
- `full_corrected`: sparse network with the correction applied to every routed block.

Self mass is mean A_ii. Output error is headwise **AV before output projection**;
it is not an error of the projected/gated residual update. Outputs are compared
at the same original token positions. Full final-velocity comparisons include
both dense-reference MSE and the conditional FM target MSE.

The run verifies instrumentation against original fused-attention dense/sparse
outputs, verifies exact subset matching, and checks local/full equivalence at
the first routed layer. `metadata.json` records numerical tolerances observed.
Checkpoint weights load strictly; no model parameters are trained or edited.

`measurements.npz` axes: image, timestep, mask repeat, logical layer, mode, head,
metric. Aggregate confidence intervals use 2,000 paired image bootstrap resamples
after averaging mask/head measurements within image. They are exploratory,
pointwise intervals without multiple-comparison correction.

Dependencies: torch, timm, numpy (already used by the training repository).
Reporting and standalone SVG/HTML generation need no additional plotting library.
No FID, held-out quality evaluation, or retraining is performed by this analysis.
