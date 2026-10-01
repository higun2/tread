"""Shared constructor/CLI/checkpoint fields for training and generation."""

import argparse


MOR_DEFAULTS = {
    "use_mor": False,
    "mor_prefix_blocks": 3,
    "mor_recurrent_blocks": 2,
    "mor_suffix_blocks": 3,
    "mor_max_recursions": 3,
    "mor_capacity_ratios": (1.0, 0.5, 0.25),
    "use_recursion_conditioning": False,
    "mor_global_topk": False,
    "mor_gating": True,
}


def add_mor_args(parser):
    for name, default in MOR_DEFAULTS.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(default, tuple):
            parser.add_argument(flag, type=float, nargs="+", default=default)
        else:
            parser.add_argument(flag, type=int, default=default)


def mor_kwargs(args):
    return {name: getattr(args, name, default) for name, default in MOR_DEFAULTS.items()}
