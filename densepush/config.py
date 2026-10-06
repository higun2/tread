"""Arguments for TREAD routing with DensePush margin path repulsion."""

import argparse


TREAD_DEFAULTS = {
    "use_tread_routing": False,
    "tread_start_block": 3,
    "tread_end_block": 9,
    "tread_active_ratio": 0.5,
    "tread_attn_correction": False,
    "tread_attn_correction_strength": 1.0,
    "tread_attn_correction_backend": "sdpa",
    "tread_active_ratios": None,
    "tread_recursive": False,
    "tread_num_groups": 3,
    "tread_recursive_pattern": "grouped",
    "tread_depth_embedding": False,
    "tread_eval_mode": "sparse",
    "tread_fp32_endpoint": True,
    "tread_seed": None,
    "tread_debug": False,
}

ROUTESYNC_DEFAULTS = {
    "use_dense_push": False,
    "dense_push_ratio": 0.1,
    "dense_push_weight": 0.1,
    "dense_push_margin": 0.95,
    "dense_push_block": None,
    "dense_push_tokens": "all",
    "dense_push_grad": "sparse",
    "use_routesync": False,
    "routesync_weight": 0.0,
    "routesync_loss_type": "relational",
    "routesync_sample_ratio": 1.0,
    "routesync_target_blocks": None,
    "routesync_debug": False,
}


def add_tread_args(parser):
    parser.add_argument("--tread-attn-correction", action=argparse.BooleanOptionalAction,
        default=False, help="Correct self-key inclusion bias in sparse routed attention; dense inference unchanged")
    parser.add_argument("--tread-attn-correction-strength", type=float, default=1.0,
        help="Multiply diagonal log((K-1)/(N-1)); 0 is the exact original path")
    parser.add_argument("--tread-attn-correction-backend", choices=["flex", "sdpa"], default="sdpa",
        help="flex: compiled fused score modification; sdpa: additive-mask SDPA")
    parser.add_argument(
        "--use-tread-routing", action=argparse.BooleanOptionalAction,
        default=TREAD_DEFAULTS["use_tread_routing"],
    )
    parser.add_argument(
        "--tread-start-block", type=int,
        default=TREAD_DEFAULTS["tread_start_block"],
    )
    parser.add_argument(
        "--tread-end-block", type=int,
        default=TREAD_DEFAULTS["tread_end_block"],
    )
    parser.add_argument(
        "--tread-active-ratio", type=float,
        default=TREAD_DEFAULTS["tread_active_ratio"],
    )
    parser.add_argument(
        "--tread-active-ratios", type=float, nargs="+",
        default=TREAD_DEFAULTS["tread_active_ratios"],
        help="Uniformly sample one active ratio per training forward, e.g. 0.3 0.5 0.7; shared across ranks. Evaluation uses --tread-active-ratio",
    )
    parser.add_argument(
        "--tread-recursive", action=argparse.BooleanOptionalAction,
        default=TREAD_DEFAULTS["tread_recursive"],
        help="Share one block across each contiguous routed block group",
    )
    parser.add_argument(
        "--tread-num-groups", type=int,
        default=TREAD_DEFAULTS["tread_num_groups"],
    )
    parser.add_argument(
        "--tread-recursive-pattern", choices=["grouped", "interleaved"],
        default=TREAD_DEFAULTS["tread_recursive_pattern"],
        help="Use AABBCC-style grouped or ABCABC-style interleaved sharing",
    )
    parser.add_argument(
        "--tread-depth-embedding", action=argparse.BooleanOptionalAction,
        default=TREAD_DEFAULTS["tread_depth_embedding"],
        help="Add a learned logical-depth embedding to the AdaLN condition in recursive blocks",
    )
    parser.add_argument(
        "--tread-eval-mode", choices=["sparse", "dense"],
        default=TREAD_DEFAULTS["tread_eval_mode"],
    )
    parser.add_argument(
        "--tread-fp32-endpoint", action=argparse.BooleanOptionalAction,
        default=TREAD_DEFAULTS["tread_fp32_endpoint"],
        help="Run only the final routed block in FP32 during training",
    )
    parser.add_argument(
        "--tread-seed", type=int, default=TREAD_DEFAULTS["tread_seed"],
        help="Local deterministic subset seed used only during sparse evaluation",
    )
    parser.add_argument(
        "--tread-debug", action=argparse.BooleanOptionalAction,
        default=TREAD_DEFAULTS["tread_debug"],
    )


def sample_ratio(value):
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("sample ratio must be in (0, 1]")
    return value


def add_routesync_args(parser):
    parser.add_argument("--use-dense-push", action=argparse.BooleanOptionalAction,
        default=False, help="Training-only margin repulsion between pooled dense and sparse path features")
    parser.add_argument("--dense-push-ratio", type=sample_ratio, default=0.1,
        help="Random fraction of local batch receiving an extra dense routed-segment pass; floor, minimum one")
    parser.add_argument("--dense-push-weight", type=float, default=0.1,
        help="lambda_push for mean [cos(z_D, z_S) - tau]_+^2 over selected samples")
    parser.add_argument("--dense-push-margin", type=float, default=0.95,
        help="tau: zero gradient once pooled dense/sparse cosine is at or below this value")
    parser.add_argument("--dense-push-block", type=int, default=None,
        help="Zero-based block whose output is pooled, in [tread_start_block, depth). "
             "Default: tread_end_block (first full-token suffix block, as in DenseSync)")
    parser.add_argument("--dense-push-tokens", choices=["all", "active", "routed"],
        default="all", help="Spatial tokens entering the mean pool before L2 normalization")
    parser.add_argument("--dense-push-grad", choices=["sparse", "both"], default="sparse",
        help="sparse: stopped no-grad dense anchor; both: also backprop through the dense path")
    parser.add_argument(
        "--routesync-loss-type", choices=["relational", "feature-cosine"],
        default="relational", help="relational: R-P relation L1; feature-cosine: negative mean cosine between corresponding pre/post tokens with stopped post target",
    )
    parser.add_argument(
        "--routesync-target-blocks", type=int, nargs="+", default=None,
        help="Zero-based dense suffix block output(s) used as stopped target; one fixes the target, multiple choose uniformly per training forward, synchronized across ranks. Default: tread_end_block",
    )
    parser.add_argument(
        "--use-routesync", action=argparse.BooleanOptionalAction,
        default=ROUTESYNC_DEFAULTS["use_routesync"],
        help="Add the training-only alignment loss selected by --routesync-loss-type",
    )
    parser.add_argument(
        "--routesync-weight", type=float,
        default=ROUTESYNC_DEFAULTS["routesync_weight"],
    )
    parser.add_argument(
        "--routesync-sample-ratio", type=sample_ratio,
        default=ROUTESYNC_DEFAULTS["routesync_sample_ratio"],
        help="Fraction of each R/P group sampled for alignment in (0, 1]; 1 uses all tokens",
    )
    parser.add_argument(
        "--routesync-debug", action=argparse.BooleanOptionalAction,
        default=ROUTESYNC_DEFAULTS["routesync_debug"],
        help="Print and validate RouteSync tensors on the first training forward",
    )


def tread_kwargs(args):
    values = {
        name: getattr(args, name, default)
        for name, default in TREAD_DEFAULTS.items()
    }
    values.update({
        name: getattr(args, name, default)
        for name, default in ROUTESYNC_DEFAULTS.items()
    })
    return values
