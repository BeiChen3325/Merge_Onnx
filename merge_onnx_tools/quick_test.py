from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import SwitchInput
from .model_io import fixed_last_dim, load_single_io_model
from .obs_mapping import build_tracking_mapping, build_tracking_obs_numpy


def switch_value(value: bool, switch: SwitchInput) -> np.ndarray:
    """Create a scalar input value matching the configured ONNX switch input type."""

    if switch.elem_type == "bool":
        return np.asarray(value, dtype=np.bool_)
    if switch.elem_type == "int64":
        return np.asarray(1 if value else 0, dtype=np.int64)
    raise ValueError(f"Unsupported switch input type: {switch.elem_type}")


def quick_test(
    merged_path: str | Path,
    tracking_model_path: str | Path,
    locomotion_model_path: str | Path,
    switch: SwitchInput,
    tracking_indices: str | None,
) -> None:
    """Compare both merged branches against the original policies."""

    import onnxruntime as ort

    tracking = load_single_io_model(tracking_model_path, "tracking")
    locomotion = load_single_io_model(locomotion_model_path, "locomotion")
    tracking_input_size = fixed_last_dim(tracking.input_shape, "tracking")
    locomotion_input_size = fixed_last_dim(locomotion.input_shape, "locomotion")
    mapping = build_tracking_mapping(tracking_input_size, locomotion_input_size, tracking_indices)

    rng = np.random.default_rng(seed=0)
    obs = rng.normal(size=(1, locomotion_input_size)).astype(np.float32)
    tracking_obs = build_tracking_obs_numpy(obs, mapping)

    merged_sess = ort.InferenceSession(str(merged_path), providers=["CPUExecutionProvider"])
    tracking_sess = ort.InferenceSession(str(tracking_model_path), providers=["CPUExecutionProvider"])
    locomotion_sess = ort.InferenceSession(str(locomotion_model_path), providers=["CPUExecutionProvider"])

    merged_tracking = merged_sess.run(None, {"obs": obs, switch.name: switch_value(True, switch)})[0]
    expected_tracking = tracking_sess.run(None, {tracking.input_name: tracking_obs})[0]
    merged_locomotion = merged_sess.run(None, {"obs": obs, switch.name: switch_value(False, switch)})[0]
    expected_locomotion = locomotion_sess.run(None, {locomotion.input_name: obs})[0]

    tracking_diff = float(np.max(np.abs(merged_tracking - expected_tracking)))
    locomotion_diff = float(np.max(np.abs(merged_locomotion - expected_locomotion)))
    print(f"quick_test tracking branch max abs diff: {tracking_diff:.8g}")
    print(f"quick_test locomotion branch max abs diff: {locomotion_diff:.8g}")
