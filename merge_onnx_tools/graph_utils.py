from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import onnx
from onnx import numpy_helper


def replace_value_name_in_graph(graph: onnx.GraphProto, old_name: str, new_name: str) -> None:
    """Replace all graph-local references to a value name."""

    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == old_name:
                node.input[index] = new_name
        for index, name in enumerate(node.output):
            if name == old_name:
                node.output[index] = new_name

    for value in graph.input:
        if value.name == old_name:
            value.name = new_name
    for value in graph.output:
        if value.name == old_name:
            value.name = new_name
    for value in graph.value_info:
        if value.name == old_name:
            value.name = new_name
    for initializer in graph.initializer:
        if initializer.name == old_name:
            initializer.name = new_name


def clear_graph_inputs(graph: onnx.GraphProto) -> None:
    """Make an If branch subgraph capture outer-scope values instead of declaring inputs."""

    del graph.input[:]


def int64_initializer(name: str, values: Iterable[int]) -> onnx.TensorProto:
    """Create an int64 initializer used by Slice/Gather/Squeeze."""

    return numpy_helper.from_array(np.asarray(list(values), dtype=np.int64), name=name)


def float_initializer(name: str, values: Sequence[float] | np.ndarray) -> onnx.TensorProto:
    """Create a float initializer."""

    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name)
