"""Arguments for TREAD routing and routed attention sync (AttnSync)."""

import argparse


TREAD_DEFAULTS = {
    "use_tread_routing": False,
    "tread_start_block": 3,
    "tread_end_block": 9,
    "tread_active_ratio": 0.5,
    "tread_eval_mode": "sparse",
    "tread_fp32_endpoint": True,
    "tread_seed": None,
}

ATTN_SYNC_DEFAULTS = {
    "use_attn_sync": False,
    "attn_sync_student_blocks": None,
    "attn_sync_teacher_block": None,
    "attn_sync_heads": "mean",
    "attn_sync_loss": "js",
    "attn_sync_weight": 0.1,
    "attn_sync_ratio": 1.0,
    "attn_sync_mask_ratio": 0.0,
    "attn_sync_mask_unit": "element",
}

# Options of earlier experiments that this code no longer implements, with the value
# that means "off". A checkpoint trained with any of them on cannot be rebuilt here.
REMOVED_OPTIONS = {
    "tread_attn_correction": False,
    "tread_active_ratios": None,
    "tread_recursive": False,
    "tread_depth_embedding": False,
    "use_dense_push": False,
    "use_routesync": False,
}


def add_tread_args(parser):
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


def sample_ratio(value):
    value = float(value)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("sample ratio must be in (0, 1]")
    return value


def mask_ratio(value):
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise argparse.ArgumentTypeError("mask ratio must be in [0, 1)")
    return value


def add_attn_sync_args(parser):
    parser.add_argument("--use-attn-sync", action=argparse.BooleanOptionalAction,
        default=False, help="Training-only routed attention sync: student blocks follow a stopped teacher block")
    parser.add_argument("--attn-sync-student-blocks", type=int, nargs="+", default=None,
        help="Zero-based routed block indices trained to follow the teacher, e.g. 3 5")
    parser.add_argument("--attn-sync-teacher-block", type=int, default=None,
        help="Zero-based routed block index whose attention is the stop-gradient target, e.g. 7")
    parser.add_argument("--attn-sync-heads", choices=["mean", "per-head"], default="mean",
        help="mean: compare head-averaged attention rows; per-head: match head h to head h")
    parser.add_argument("--attn-sync-loss", choices=["l1", "js", "kl"], default="js",
        help="Per-query-row divergence; kl is KL(teacher || student)")
    parser.add_argument("--attn-sync-weight", type=float, default=0.1,
        help="Weight of the attention sync loss averaged over student blocks")
    parser.add_argument("--attn-sync-ratio", type=sample_ratio, default=1.0,
        help="Random fraction of local batch whose attention maps are computed; floor, minimum one")
    parser.add_argument("--attn-sync-mask-ratio", type=mask_ratio, default=0.0,
        help="Random fraction of attention-map entries excluded from the sync loss, resampled each step; 0 disables")
    parser.add_argument("--attn-sync-mask-unit", choices=["element", "query", "key"], default="element",
        help="element: drop single (query, key) entries; query: drop whole rows; key: drop whole columns")


def tread_kwargs(args):
    return {
        name: getattr(args, name, default)
        for name, default in {**TREAD_DEFAULTS, **ATTN_SYNC_DEFAULTS}.items()
    }


def check_removed_options(saved):
    """Raise if saved checkpoint args enable an option this code no longer implements."""
    for name, off in REMOVED_OPTIONS.items():
        value = saved.get(name, off) if isinstance(saved, dict) else getattr(saved, name, off)
        if value != off:
            raise ValueError(f"checkpoint was trained with {name}={value!r}, which this code no longer supports")
