from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnx


DEFAULT_TRACKING_MODEL = "model/input/beyondmimic_tracking_only_v3.onnx"
DEFAULT_LOCOMOTION_MODEL = "model/input/lateral_motion_v9.onnx"
DEFAULT_OUTPUT_MODEL = "model/output/merged_tracking_locomotion_if.onnx"


@dataclass(frozen=True)
class ModelInfo:
    """A loaded single-input/single-output ONNX policy and its tensor metadata."""

    path: Path
    model: onnx.ModelProto
    input_name: str
    output_name: str
    input_shape: list[int | str]
    output_shape: list[int | str]
    input_elem_type: int
    output_elem_type: int


@dataclass(frozen=True)
class TrackingObsMapping:
    """How to derive the tracking model input from the merged outer observation."""

    mode: str
    size: int
    indices: list[int] | None = None


@dataclass(frozen=True)
class SwitchInput:
    """External switch input that selects the ONNX If branch."""

    name: str = "switch"
    elem_type: str = "bool"
