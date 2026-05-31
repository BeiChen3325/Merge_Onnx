# ONNX Policy Merger

English | [中文](./README.zh-CN.md)

This project merges two ONNX policy models into a single ONNX file. The merged model keeps both policy subgraphs and uses an ONNX `If` node to choose the execution path through an external `switch` input.

Default model roles:

- `beyondmimic_tracking_only_v3.onnx`: tracking / pre-step policy before climbing onto the box.
- `lateral_motion_v9.onnx`: locomotion policy after front legs are on the box, driven by remote commands.

Default merged behavior:

- `switch=True`: run the tracking branch.
- `switch=False`: run the locomotion branch.

## Project Layout

```text
.
├── README.md
├── README.zh-CN.md
├── merge.py
├── merge_onnx_tools
│   ├── __init__.py
│   ├── config.py
│   ├── graph_utils.py
│   ├── merger.py
│   ├── model_io.py
│   ├── obs_mapping.py
│   └── quick_test.py
└── model
    ├── input
    │   ├── your_model.onnx
    └── output
```

## File Responsibilities

### `merge.py`

CLI entry point. It parses arguments and calls merge logic from `merge_onnx_tools`.

Common capabilities:

- Set tracking model path.
- Set locomotion model path.
- Set output model path.
- Set `switch` input name and type.
- Configure how tracking observations are extracted from merged `obs`.
- Optionally run `--quick-test` to compare merged branch outputs with original model outputs.

### `merge_onnx_tools/config.py`

Centralized defaults and small configuration data structures.

Main items:

- `DEFAULT_TRACKING_MODEL`
- `DEFAULT_LOCOMOTION_MODEL`
- `DEFAULT_OUTPUT_MODEL`
- `ModelInfo`
- `TrackingObsMapping`
- `SwitchInput`

### `merge_onnx_tools/model_io.py`

Model loading and base validation utilities.

Main checks:

- Load ONNX models.
- Ensure each model has exactly one input and one output.
- Ensure input/output data types are `FLOAT`.
- Extract input/output shapes.
- Validate compatibility between the two models: opset, output shape, and output type.

### `merge_onnx_tools/graph_utils.py`

Common ONNX graph editing helpers.

Main utilities:

- Replace tensor/value names in a graph.
- Clear explicit inputs of `If` branch subgraphs so branches capture outer-scope values.
- Build common initializers.

### `merge_onnx_tools/obs_mapping.py`

Handles observation mapping for the tracking branch.

The merged model exposes locomotion's full observation input: `obs FLOAT [1, 265]`. The tracking model expects `FLOAT [1, 89]`, so tracking input is built inside the tracking subgraph from outer `obs`.

Default behavior:

```text
tracking_obs = obs[:, :89]
```

If you know the exact index mapping between tracking 89-dim observations and locomotion 265-dim observations, pass it via `--tracking-indices`.

### `merge_onnx_tools/merger.py`

Core merge pipeline.

Workflow:

1. Load tracking and locomotion ONNX models.
2. Validate both models can be `If` branches.
3. Prefix internal node/weight/temp tensor names to avoid collisions.
4. Insert observation adaptation nodes into the tracking subgraph.
5. Let locomotion subgraph consume outer full `obs` directly.
6. Create outer `switch` input.
7. Connect both subgraphs with an ONNX `If` node.
8. Save merged model.

### `merge_onnx_tools/quick_test.py`

Quick verification for merged model correctness.

Verification rule:

- With `switch=True`, merged output should match original tracking model output.
- With `switch=False`, merged output should match original locomotion model output.

### `model/input/`

Stores source models to merge.

Current files:

- `beyondmimic_tracking_only_v3.onnx`: tracking model, input `[1, 89]`, output `[1, 16]`.
- `lateral_motion_v9.onnx`: locomotion model, input `[1, 265]`, output `[1, 16]`.

### `model/output/`

Stores generated merged models.

Current files:

- `merged_tracking_locomotion_if.onnx`: default bool-switch merged model.
- `merged_tracking_locomotion_if_int64_switch.onnx`: merged model using int64 switch input; internally converts `0/1` to bool.

## Default Merged Model I/O

Run:

```bash
python merge.py
```

Output model:

```text
model/output/merged_tracking_locomotion_if.onnx
```

Model I/O:

```text
inputs:
  obs     FLOAT [1, 265]
  switch  BOOL  []

outputs:
  actions FLOAT [1, 16]
```

Where:

- `obs`: full locomotion observation, currently 5-frame history, `265 = 5 * 53`.
- `switch`: external branch selector, scalar bool.
- `actions`: 16-joint action output.

## Common Commands

Generate the default merged model:

```bash
python merge.py
```

Generate and run quick verification:

```bash
python merge.py --quick-test
```

Generate a model that accepts `0/1` switch input:

```bash
python merge.py --switch-input-type int64 --output model/output/merged_tracking_locomotion_if_int64_switch.onnx
```

Set a custom output path:

```bash
python merge.py --output model/output/my_merged_model.onnx
```

Set tracking observation indices:

```bash
python merge.py --tracking-indices 0,1,2,3,4
```

Note: `--tracking-indices` count must match tracking input dimension. Current tracking input dimension is 89, so 89 indices are required.

## About the 265-Dim History Observation

`lateral_motion_v9.onnx` input is `[1, 265]`, corresponding to 5-frame history:

```text
265 = 5 * 53
```

Each 53-dim frame includes:

```text
base_ang_vel       3
projected_gravity 3
command           3
joint_pos         12
joint_vel         16
prev_action       16
```

Current tracking model input is `[1, 89]`. This is not equal to `5 * (53 - 3) = 250`, so the script does not assume it is "5-frame history without command". Default behavior remains prefix slicing `obs[:, :89]`. If a more accurate mapping is confirmed, use `--tracking-indices`.

## Dependencies

Install in Python environment:

```bash
python -m pip install onnx onnxruntime numpy
```

- `onnx`: load/modify/save ONNX models.
- `onnxruntime`: run inference for `--quick-test`.
- `numpy`: build test inputs and initializers.
