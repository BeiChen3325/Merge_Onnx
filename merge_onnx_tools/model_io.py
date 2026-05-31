from __future__ import annotations

from pathlib import Path
from typing import Sequence

import onnx
from onnx import TensorProto, checker

from .config import ModelInfo


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    """Return a compact Python representation of a tensor shape."""

    dims: list[int | str] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def fixed_last_dim(shape: Sequence[int | str], model_name: str) -> int:
    """Read the static feature dimension from a [batch, features] tensor."""

    if len(shape) < 2 or not isinstance(shape[-1], int):
        raise ValueError(f"{model_name} input must have a static last dimension, got {shape}")
    return int(shape[-1])


def load_single_io_model(path: str | Path, role: str) -> ModelInfo:
    """Load and validate a policy model with exactly one float input and output."""

    model_path = Path(path)
    model = onnx.load(model_path)
    checker.check_model(model)

    if len(model.graph.input) != 1:
        raise ValueError(f"{role} model must have exactly 1 graph input, got {len(model.graph.input)}")
    if len(model.graph.output) != 1:
        raise ValueError(f"{role} model must have exactly 1 graph output, got {len(model.graph.output)}")

    graph_input = model.graph.input[0]
    graph_output = model.graph.output[0]
    input_type = graph_input.type.tensor_type
    output_type = graph_output.type.tensor_type

    if input_type.elem_type != TensorProto.FLOAT:
        raise ValueError(f"{role} input must be FLOAT, got {TensorProto.DataType.Name(input_type.elem_type)}")
    if output_type.elem_type != TensorProto.FLOAT:
        raise ValueError(f"{role} output must be FLOAT, got {TensorProto.DataType.Name(output_type.elem_type)}")

    return ModelInfo(
        path=model_path,
        model=model,
        input_name=graph_input.name,
        output_name=graph_output.name,
        input_shape=tensor_shape(graph_input),
        output_shape=tensor_shape(graph_output),
        input_elem_type=input_type.elem_type,
        output_elem_type=output_type.elem_type,
    )


def assert_compatible_models(tracking: ModelInfo, locomotion: ModelInfo) -> None:
    """Check the ONNX properties that must match between If branches."""

    tracking_opsets = [(item.domain, item.version) for item in tracking.model.opset_import]
    locomotion_opsets = [(item.domain, item.version) for item in locomotion.model.opset_import]
    if tracking_opsets != locomotion_opsets:
        raise ValueError(f"opset_import mismatch: tracking={tracking_opsets}, locomotion={locomotion_opsets}")

    if tracking.output_shape != locomotion.output_shape:
        raise ValueError(
            "Branch output shapes must match: "
            f"tracking={tracking.output_shape}, locomotion={locomotion.output_shape}"
        )

    if tracking.output_elem_type != locomotion.output_elem_type:
        raise ValueError("Branch output element types must match")
