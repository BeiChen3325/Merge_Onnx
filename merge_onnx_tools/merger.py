from __future__ import annotations

import copy
from pathlib import Path

import onnx
from onnx import TensorProto, checker, compose, helper, shape_inference

from .config import (
    DEFAULT_LOCOMOTION_MODEL,
    DEFAULT_OUTPUT_MODEL,
    DEFAULT_TRACKING_MODEL,
    ModelInfo,
    SwitchInput,
)
from .graph_utils import clear_graph_inputs, replace_value_name_in_graph
from .model_io import assert_compatible_models, fixed_last_dim, load_single_io_model
from .obs_mapping import add_tracking_observation_adapter, build_tracking_mapping


def prepare_branch_graph(
    info: ModelInfo,
    prefix: str,
    branch_output_name: str,
) -> tuple[onnx.GraphProto, str]:
    """Prefix a policy graph and normalize its branch output name."""

    prefixed_model = compose.add_prefix(copy.deepcopy(info.model), prefix)
    graph = copy.deepcopy(prefixed_model.graph)
    branch_input_name = graph.input[0].name
    branch_model_output_name = graph.output[0].name

    replace_value_name_in_graph(graph, branch_model_output_name, branch_output_name)
    graph.output[0].name = branch_output_name
    return graph, branch_input_name


def switch_tensor_type(elem_type: str) -> int:
    """Map CLI switch type names to ONNX tensor element types."""

    if elem_type == "bool":
        return TensorProto.BOOL
    if elem_type == "int64":
        return TensorProto.INT64
    raise ValueError(f"Unsupported switch input type: {elem_type}")


def add_switch_condition_node(
    nodes: list[onnx.NodeProto],
    switch: SwitchInput,
) -> str:
    """Return the bool scalar value name consumed by ONNX If."""

    if switch.elem_type == "bool":
        return switch.name

    condition_name = f"{switch.name}_as_bool"
    nodes.append(
        helper.make_node(
            "Cast",
            inputs=[switch.name],
            outputs=[condition_name],
            name="CastSwitchToBool",
            to=TensorProto.BOOL,
        )
    )
    return condition_name


def make_switch_input(switch: SwitchInput) -> onnx.ValueInfoProto:
    """Create the external switch input; ONNX If consumes a scalar bool condition."""

    return helper.make_tensor_value_info(
        switch.name,
        switch_tensor_type(switch.elem_type),
        [],
    )


def warn_about_history_shape(
    tracking_input_size: int,
    locomotion_input_size: int,
    single_frame_size: int,
    command_size: int,
) -> None:
    """Print a diagnostic when the observed model shapes do not match simple history math."""

    if single_frame_size <= 0:
        return
    if locomotion_input_size % single_frame_size != 0:
        return

    history_frames = locomotion_input_size // single_frame_size
    expected_without_command = history_frames * (single_frame_size - command_size)
    if tracking_input_size != expected_without_command:
        print(
            "[warn] locomotion input looks like "
            f"{history_frames} * {single_frame_size} = {locomotion_input_size}, "
            f"but tracking input is {tracking_input_size}. "
            f"If tracking was expected to be history without command, expected {expected_without_command}. "
            "The script will use the configured tracking observation mapping instead."
        )


def make_if_model(
    tracking_model_path: str | Path = DEFAULT_TRACKING_MODEL,
    locomotion_model_path: str | Path = DEFAULT_LOCOMOTION_MODEL,
    output_path: str | Path = DEFAULT_OUTPUT_MODEL,
    *,
    outer_obs_name: str = "obs",
    final_output_name: str = "actions",
    switch: SwitchInput = SwitchInput(),
    tracking_indices: str | None = None,
    single_frame_size: int = 53,
    command_size: int = 3,
    skip_shape_inference: bool = False,
) -> onnx.ModelProto:
    """Create and save a merged ONNX model with `switch` selecting the branch."""

    # 1. 读取并检查两个原始模型，确保它们满足 If 分支合并的基本要求。
    tracking = load_single_io_model(tracking_model_path, "tracking")
    locomotion = load_single_io_model(locomotion_model_path, "locomotion")
    assert_compatible_models(tracking, locomotion)

    # 2. 对外观测采用 locomotion 的完整历史观测；tracking 输入由映射规则生成。
    tracking_input_size = fixed_last_dim(tracking.input_shape, "tracking")
    locomotion_input_size = fixed_last_dim(locomotion.input_shape, "locomotion")
    warn_about_history_shape(tracking_input_size, locomotion_input_size, single_frame_size, command_size)
    mapping = build_tracking_mapping(tracking_input_size, locomotion_input_size, tracking_indices)

    # 3. 给两个子图加前缀，避免内部节点名、权重名、临时张量名相互冲突。
    branch_output_name = "branch_actions"
    tracking_graph, tracking_branch_input = prepare_branch_graph(tracking, "tracking_", branch_output_name)
    locomotion_graph, locomotion_branch_input = prepare_branch_graph(locomotion, "locomotion_", branch_output_name)

    # 4. tracking 子图内部先从外层 obs 裁剪/抽取观测；locomotion 子图直接捕获外层 obs。
    add_tracking_observation_adapter(tracking_graph, outer_obs_name, tracking_branch_input, mapping)
    replace_value_name_in_graph(locomotion_graph, locomotion_branch_input, outer_obs_name)
    clear_graph_inputs(tracking_graph)
    clear_graph_inputs(locomotion_graph)

    # 5. 外层 switch 直接决定 If 分支：True 使用 tracking，False 使用 locomotion。
    main_nodes: list[onnx.NodeProto] = []
    condition_name = add_switch_condition_node(main_nodes, switch)
    if_node = helper.make_node(
        "If",
        inputs=[condition_name],
        outputs=[final_output_name],
        name="PolicySwitch",
        then_branch=tracking_graph,
        else_branch=locomotion_graph,
    )
    main_nodes.append(if_node)

    # 6. 对外暴露两个输入：策略观测 obs 和切换信号 switch；输出仍为动作 actions。
    main_input = copy.deepcopy(locomotion.model.graph.input[0])
    main_input.name = outer_obs_name
    switch_input = make_switch_input(switch)

    main_output = copy.deepcopy(locomotion.model.graph.output[0])
    main_output.name = final_output_name

    main_graph = helper.make_graph(
        nodes=main_nodes,
        name="tracking_locomotion_if_graph",
        inputs=[main_input, switch_input],
        outputs=[main_output],
    )

    merged_model = helper.make_model(
        main_graph,
        opset_imports=locomotion.model.opset_import,
        producer_name="tracking_locomotion_if_merger",
    )
    merged_model.ir_version = min(tracking.model.ir_version, locomotion.model.ir_version)
    checker.check_model(merged_model)

    # 7. 尽量推断 shape，最后保存。推断失败不会阻止保存已通过 checker 的模型。
    if not skip_shape_inference:
        try:
            merged_model = shape_inference.infer_shapes(merged_model)
        except Exception as exc:
            print(f"[warn] shape inference failed, saved model may still run: {exc}")

    output_model_path = Path(output_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(merged_model, output_model_path)

    print(f"Saved merged model: {output_model_path}")
    print(f"tracking input: {tracking.path} {tracking.input_shape} -> output {tracking.output_shape}")
    print(f"locomotion input: {locomotion.path} {locomotion.input_shape} -> output {locomotion.output_shape}")
    print(f"outer inputs: {outer_obs_name} {locomotion.input_shape}, {switch.name} scalar {switch.elem_type}")
    print("switch semantics: True/1 -> tracking branch, False/0 -> locomotion branch")
    print(f"tracking obs mapping: {mapping.mode}" + (f" size={mapping.size}" if mapping.mode == "prefix" else ""))
    return merged_model
