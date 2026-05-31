# ONNX Policy Merger

[English](./README.md) | 中文

这个工程用于把两个 ONNX 策略模型合并到同一个 ONNX 文件中。合并后的模型会保留两个策略子图，并通过 ONNX `If` 节点根据外部输入 `switch` 选择本次推理走哪个子图。

当前默认模型用途：

- `beyondmimic_tracking_only_v3.onnx`：tracking / 上箱子前置动作策略。
- `lateral_motion_v9.onnx`：前腿搭上箱子之后，根据遥控指令移动的 locomotion 策略。

合并后的默认语义：

- `switch=True`：使用 tracking 子图。
- `switch=False`：使用 locomotion 子图。

## 文件夹结构

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

## 各文件功能

### `merge.py`

命令行入口文件。它负责解析参数，然后调用 `merge_onnx_tools` 中的合并逻辑。

常用功能：

- 指定 tracking 模型路径。
- 指定 locomotion 模型路径。
- 指定输出模型路径。
- 指定 `switch` 输入名称和类型。
- 指定 tracking 观测如何从合并后的 `obs` 中抽取。
- 可选运行 `--quick-test`，对比合并模型两个分支和原始模型的输出是否一致。

### `merge_onnx_tools/config.py`

集中保存默认路径和小型配置数据结构。

主要内容：

- `DEFAULT_TRACKING_MODEL`
- `DEFAULT_LOCOMOTION_MODEL`
- `DEFAULT_OUTPUT_MODEL`
- `ModelInfo`
- `TrackingObsMapping`
- `SwitchInput`

### `merge_onnx_tools/model_io.py`

负责 ONNX 模型读取和基础检查。

主要功能：

- 读取模型。
- 检查模型是否只有一个输入和一个输出。
- 检查输入输出是否为 `FLOAT`。
- 提取输入输出 shape。
- 检查两个模型的 opset、输出 shape、输出类型是否兼容。

### `merge_onnx_tools/graph_utils.py`

放置通用 ONNX graph 操作函数。

主要功能：

- 替换 graph 中的 tensor/value 名称。
- 清空 If 分支子图的显式输入，使子图捕获外层输入。
- 创建常用 initializer。

### `merge_onnx_tools/obs_mapping.py`

负责 tracking 分支的观测映射。

当前合并模型对外使用 locomotion 的完整观测输入，也就是 `obs FLOAT [1, 265]`。tracking 模型输入为 `FLOAT [1, 89]`，因此需要在 tracking 子图内部从外层 `obs` 中构造 tracking 输入。

默认行为：

```text
tracking_obs = obs[:, :89]
```

如果确认了 tracking 89 维观测与 locomotion 265 维观测之间的精确索引关系，可以通过 `--tracking-indices` 显式指定。

### `merge_onnx_tools/merger.py`

核心合并逻辑。

工作流程：

1. 加载 tracking 和 locomotion 两个 ONNX 模型。
2. 检查两个模型是否可以作为 `If` 的两个分支。
3. 给两个模型内部节点、权重、临时 tensor 加前缀，避免命名冲突。
4. 为 tracking 子图插入观测适配节点。
5. 让 locomotion 子图直接使用外层完整 `obs`。
6. 创建外层 `switch` 输入。
7. 用 ONNX `If` 节点连接两个子图。
8. 保存合并后的模型。

### `merge_onnx_tools/quick_test.py`

用于快速验证合并后的模型是否正确。

验证方式：

- `switch=True` 时，合并模型输出应与原始 tracking 模型输出一致。
- `switch=False` 时，合并模型输出应与原始 locomotion 模型输出一致。

### `model/input/`

存放待合并的原始模型。

当前文件：

- `beyondmimic_tracking_only_v3.onnx`：原始 tracking 模型，输入为 `[1, 89]`，输出为 `[1, 16]`。
- `lateral_motion_v9.onnx`：原始 locomotion 模型，输入为 `[1, 265]`，输出为 `[1, 16]`。

### `model/output/`

存放合并脚本生成的输出模型。

当前文件：

- `merged_tracking_locomotion_if.onnx`：默认生成的 bool switch 合并模型。
- `merged_tracking_locomotion_if_int64_switch.onnx`：使用 int64 switch 的合并模型，内部会将 0/1 转成 bool。

## 默认合并模型输入输出

默认运行：

```bash
python merge.py
```

会生成：

```text
model/output/merged_tracking_locomotion_if.onnx
```

该模型的输入输出为：

```text
inputs:
  obs     FLOAT [1, 265]
  switch  BOOL  []

outputs:
  actions FLOAT [1, 16]
```

其中：

- `obs`：locomotion 模型的完整观测，当前为 5 帧历史观测，`265 = 5 * 53`。
- `switch`：外部切换信号，标量 bool。
- `actions`：16 个关节动作输出。

## 常用命令

生成默认合并模型：

```bash
python merge.py
```

生成并进行快速验证：

```bash
python merge.py --quick-test
```

生成支持 0/1 输入的 switch 模型：

```bash
python merge.py --switch-input-type int64 --output model/output/merged_tracking_locomotion_if_int64_switch.onnx
```

指定自定义输出路径：

```bash
python merge.py --output model/output/my_merged_model.onnx
```

指定 tracking 观测索引：

```bash
python merge.py --tracking-indices 0,1,2,3,4
```

注意：`--tracking-indices` 的索引数量必须等于 tracking 模型输入维度。当前 tracking 输入为 89 维，因此需要提供 89 个索引。

## 关于 265 维历史观测

`lateral_motion_v9.onnx` 的输入为 `[1, 265]`，对应 5 帧历史观测：

```text
265 = 5 * 53
```

单帧 53 维观测包括：

```text
base_ang_vel       3
projected_gravity 3
command           3
joint_pos         12
joint_vel         16
prev_action       16
```

当前 tracking 模型输入实际为 `[1, 89]`。这个维度并不等于 `5 * (53 - 3) = 250`，因此脚本不会自动假设它是“去掉 command 后的 5 帧历史观测”。默认只做前缀裁剪 `obs[:, :89]`。如果后续确认了更准确的观测对应关系，应使用 `--tracking-indices` 显式配置。

## 依赖

需要 Python 环境中安装：

```bash
python -m pip install onnx onnxruntime numpy
```

其中：

- `onnx`：加载、修改、保存 ONNX 模型。
- `onnxruntime`：运行 `--quick-test` 时用于验证输出。
- `numpy`：构造测试输入和 initializer。
