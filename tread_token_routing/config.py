"""Shared arguments for fixed-subset TREAD-style token routing."""

import argparse


TREAD_DEFAULTS = {
    "use_tread_routing": False,
    "tread_start_block": 3,
    "tread_end_block": 9,
    "tread_active_ratio": 0.5,
    "tread_attn_correction": False,
    "tread_attn_correction_strength": 1.0,
    "tread_attn_correction_backend": "sdpa",
    "tread_recursive": False,
    "tread_num_groups": 3,
    "tread_recursive_pattern": "grouped",
    "tread_eval_mode": "sparse",
    "tread_fp32_endpoint": True,
    "tread_seed": None,
    "tread_debug": False,
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


def tread_kwargs(args):
    return {
        name: getattr(args, name, default)
        for name, default in TREAD_DEFAULTS.items()
    }
