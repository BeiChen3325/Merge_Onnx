#!/usr/bin/env python3
from __future__ import annotations

import argparse

from merge_onnx_tools.config import (
    DEFAULT_LOCOMOTION_MODEL,
    DEFAULT_OUTPUT_MODEL,
    DEFAULT_TRACKING_MODEL,
    SwitchInput,
)
from merge_onnx_tools.merger import make_if_model
from merge_onnx_tools.quick_test import quick_test


def parse_args() -> argparse.Namespace:
    """Parse CLI options for building the merged ONNX policy."""

    parser = argparse.ArgumentParser(
        description="Merge tracking and locomotion ONNX policies with an external switch-controlled ONNX If node."
    )
    parser.add_argument("--tracking-model", default=DEFAULT_TRACKING_MODEL)
    parser.add_argument("--locomotion-model", default=DEFAULT_LOCOMOTION_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_MODEL)
    parser.add_argument("--obs-name", default="obs")
    parser.add_argument("--output-name", default="actions")
    parser.add_argument("--switch-name", default="switch")
    parser.add_argument(
        "--switch-input-type",
        choices=["bool", "int64"],
        default="bool",
        help="bool follows ONNX If directly; int64 lets callers feed 0/1 and casts to bool inside the graph.",
    )
    parser.add_argument(
        "--tracking-indices",
        default=None,
        help="Comma-separated feature indices from merged obs to feed the tracking model. "
        "Default is prefix slice obs[:, :tracking_input_size].",
    )
    parser.add_argument(
        "--single-frame-size",
        type=int,
        default=53,
        help="Used only for diagnostics of locomotion history observation size.",
    )
    parser.add_argument(
        "--command-size",
        type=int,
        default=3,
        help="Used only for diagnostics of command fields inside one locomotion observation frame.",
    )
    parser.add_argument("--skip-shape-inference", action="store_true")
    parser.add_argument("--quick-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build the merged model, then optionally compare both branches with ONNX Runtime."""

    args = parse_args()
    switch = SwitchInput(name=args.switch_name, elem_type=args.switch_input_type)

    make_if_model(
        tracking_model_path=args.tracking_model,
        locomotion_model_path=args.locomotion_model,
        output_path=args.output,
        outer_obs_name=args.obs_name,
        final_output_name=args.output_name,
        switch=switch,
        tracking_indices=args.tracking_indices,
        single_frame_size=args.single_frame_size,
        command_size=args.command_size,
        skip_shape_inference=args.skip_shape_inference,
    )

    if args.quick_test:
        quick_test(
            merged_path=args.output,
            tracking_model_path=args.tracking_model,
            locomotion_model_path=args.locomotion_model,
            switch=switch,
            tracking_indices=args.tracking_indices,
        )


if __name__ == "__main__":
    main()
