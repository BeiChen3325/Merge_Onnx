from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from .config import TrackingObsMapping
from .graph_utils import int64_initializer


def parse_indices(raw: str | None) -> list[int] | None:
    """Parse comma-separated feature indices for the tracking branch input."""

    if raw is None or raw.strip() == "":
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def build_tracking_mapping(
    tracking_input_size: int,
    union_input_size: int,
    raw_indices: str | None,
) -> TrackingObsMapping:
    """Choose the default prefix mapping or validate an explicit index mapping."""

    indices = parse_indices(raw_indices)
    if indices is None:
        if tracking_input_size > union_input_size:
            raise ValueError(
                "tracking input is larger than the merged observation; pass --tracking-indices explicitly"
            )
        return TrackingObsMapping(mode="prefix", size=tracking_input_size)

    if len(indices) != tracking_input_size:
        raise ValueError(
            f"--tracking-indices length must equal tracking input size {tracking_input_size}, got {len(indices)}"
        )
    if min(indices) < 0 or max(indices) >= union_input_size:
        raise ValueError(f"--tracking-indices must be in [0, {union_input_size - 1}]")
    return TrackingObsMapping(mode="indices", size=tracking_input_size, indices=indices)


def add_tracking_observation_adapter(
    graph: onnx.GraphProto,
    outer_obs_name: str,
    branch_input_name: str,
    mapping: TrackingObsMapping,
) -> None:
    """Insert branch-local Slice/Gather nodes that build tracking obs from merged obs."""

    if mapping.mode == "prefix":
        graph.initializer.extend(
            [
                int64_initializer("tracking_slice_starts", [0]),
                int64_initializer("tracking_slice_ends", [mapping.size]),
                int64_initializer("tracking_slice_axes", [1]),
                int64_initializer("tracking_slice_steps", [1]),
            ]
        )
        adapter_node = helper.make_node(
            "Slice",
            inputs=[
                outer_obs_name,
                "tracking_slice_starts",
                "tracking_slice_ends",
                "tracking_slice_axes",
                "tracking_slice_steps",
            ],
            outputs=[branch_input_name],
            name="BuildTrackingObsByPrefixSlice",
        )
    elif mapping.mode == "indices":
        if not mapping.indices:
            raise ValueError("indices mapping requires at least one index")
        graph.initializer.append(int64_initializer("tracking_gather_indices", mapping.indices))
        adapter_node = helper.make_node(
            "Gather",
            inputs=[outer_obs_name, "tracking_gather_indices"],
            outputs=[branch_input_name],
            name="BuildTrackingObsByGather",
            axis=1,
        )
    else:
        raise ValueError(f"Unsupported tracking observation mapping mode: {mapping.mode}")

    graph.node.insert(0, adapter_node)


def build_tracking_obs_numpy(obs: np.ndarray, mapping: TrackingObsMapping) -> np.ndarray:
    """Numpy equivalent of the ONNX tracking observation adapter."""

    if mapping.mode == "prefix":
        return obs[:, : mapping.size].copy()
    if mapping.mode == "indices" and mapping.indices is not None:
        return obs[:, mapping.indices].copy()
    raise ValueError(f"Unsupported tracking observation mapping mode: {mapping.mode}")
