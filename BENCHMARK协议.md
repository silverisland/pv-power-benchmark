# 光伏功率预测 Benchmark 协议

## 1. 目标

本协议用于让不同项目、不同模型在相同数据和相同指标下迭代。Benchmark 负责固定：

- 超短期与短期任务定义；
- 训练、验证和测试样本；
- 输入字段、数组长度和时间对齐方式；
- 测试样本 `row_id`；
- 装机容量和真实标签；
- 指标公式、汇总方式和协议版本；
- 数据文件 SHA-256。

模型项目只负责把标准输入转换为内部张量并输出预测，不得修改 Benchmark 的测试数据或指标实现。

## 2. 目录结构

一套 Benchmark 对应一个任务：

```text
benchmark_data/v1/
├── ultra_short/
│   ├── benchmark.json
│   ├── train.parquet
│   ├── validation.parquet
│   └── test.parquet
└── short_term/
    ├── benchmark.json
    ├── train.parquet
    ├── validation.parquet
    └── test.parquet
```

`benchmark.json` 记录协议版本、任务定义、字段、划分边界、文件哈希、行数、场站和时间范围。任何 Parquet 内容发生改变都会导致哈希校验失败。

## 3. 标准模型输入

每一行是一个场站或省级对象在一个起报时刻的样本：

| 字段 | 类型 | 长度 | 含义 |
| --- | --- | ---: | --- |
| `row_id` | string | 标量 | Benchmark 生成的稳定样本 ID |
| `task` | string | 标量 | `ultra_short` 或 `short_term` |
| `split` | string | 标量 | `train`、`validation` 或 `test` |
| `timestamp_win` | timestamp | 标量 | 起报时刻 |
| `target_start`、`target_end` | timestamp | 标量 | 评估有效时间范围 |
| `station` | string | 标量 | 场站或省级对象 ID |
| `capacity` | float | 标量 | 装机容量 |
| `power_history` | float array | 672 | 起报前 7 天历史功率 |
| `ghi_history` | float array | 672 | 起报前 7 天历史 GHI |
| `temperature_history` | float array | 672 | 起报前 7 天历史温度 |
| `ghi_forecast` | float array | 16 或 96 | 与目标时刻对齐的 GHI 预报 |
| `temperature_forecast` | float array | 16 或 96 | 温度预报 |
| `wind_speed_forecast` | float array | 16 或 96 | 风速预报 |
| `wind_direction_forecast` | float array | 16 或 96 | 风向预报 |
| `target` | float array | 16 或 96 | 真实功率标签 |

超短期使用起报后 `T0+15min` 至 `T0+4h` 的16点。短期只保留每天10:00起报样本，目标为次日00:00至23:45的96点。

模型可以只选择部分输入特征，但必须对所有模型使用同一套 Benchmark 行和标签。

## 4. 标准模型输出

模型提交的预测文件只需要：

| 字段 | 必填 | 含义 |
| --- | --- | --- |
| `row_id` | 是 | 必须与测试集完全一致且唯一 |
| `prediction` | 是 | 超短期长度16，短期长度96 |
| `model_id` | 建议 | 同一文件最多一个模型 ID |

标准提交文件只保留 `row_id`、`prediction` 和 `model_id`。评分按 `row_id` 对齐，模型输出不能携带自己的容量或真实标签来影响评分。

## 5. 公平性规则

1. 模型开发可以使用 `train.parquet` 和 `validation.parquet` 的标签。
2. 测试集只能用于最终预测和统一评分，不能拟合归一化器、阈值或后处理参数。
3. 所有预处理器只能在训练集拟合；验证集用于模型选择和早停。
4. 不允许删除难样本。预测文件的 `row_id` 必须与测试集完全一致，多一行或少一行都会拒绝评分。
5. 不允许修改测试集的 `capacity`、`target`、时间范围和指标参数。
6. 对比结果必须同时记录 `benchmark_id`、测试数据 SHA-256、模型版本和随机种子。
7. 主指标对每次起报等权平均；场站宏平均和最差场站准确率作为诊断指标。

## 6. 安装

在任意模型项目的 Python 环境中执行：

```bash
python -m pip install -e /path/to/pv-power-benchmark
```

安装后可以使用 `pv-benchmark` 命令，也可以直接导入 `pv_benchmark` 和 `pv_metrics`。

## 7. 在模型项目中使用

### 7.1 让 Codex 自动感知 Benchmark

在模型项目根目录执行一次：

```bash
pv-benchmark init-project \
  --project-dir . \
  --benchmark /path/to/pv-power-benchmark/benchmark_data/v1/ultra_short \
  --model-id my_model_v1
```

该命令会创建或更新：

- `AGENTS.md`：Codex 自动读取的约束，已有项目指令会被保留；
- `BENCHMARK.md`：当前项目的训练、验证和评分操作说明；
- `pv-benchmark.json`：任务、路径、协议版本和锁定的 `benchmark_id`。

初始化后新建一个 Codex 任务，使它从项目根目录重新加载 `AGENTS.md`。之后可以直接要求 Codex“基于当前 Benchmark 训练基线并记录 validation 指标”或“改进模型并与上一次可比实验对比”。

### 7.2 选择任务并验证 Benchmark

```bash
pv-benchmark validate \
  --benchmark /path/to/pv-power-benchmark/benchmark_data/v1/ultra_short
```

短期任务将最后一段路径替换为 `short_term`。

### 7.3 读取数据并编写 Adapter

```python
from pathlib import Path

import numpy as np
import pandas as pd

benchmark_dir = Path("/path/to/pv-power-benchmark/benchmark_data/v1/ultra_short")

train_df = pd.read_parquet(benchmark_dir / "train.parquet")
valid_df = pd.read_parquet(benchmark_dir / "validation.parquet")
test_df = pd.read_parquet(benchmark_dir / "test.parquet")


def build_x(frame: pd.DataFrame) -> np.ndarray:
    history = np.stack(frame["power_history"].to_numpy())
    ghi = np.stack(frame["ghi_forecast"].to_numpy())
    temperature = np.stack(frame["temperature_forecast"].to_numpy())
    return np.concatenate([history, ghi, temperature], axis=1)


x_train = build_x(train_df)
y_train = np.stack(train_df["target"].to_numpy())
x_valid = build_x(valid_df)
y_valid = np.stack(valid_df["target"].to_numpy())
x_test = build_x(test_df)
```

不同模型可以编写不同的 `build_x`，但不要修改行集合、标签或 `row_id`。

### 7.4 生成标准预测文件

```python
prediction = model.predict(x_test)  # ultra_short: (B, 16)

submission = test_df[["row_id", "timestamp_win", "station"]].copy()
submission["model_id"] = "my_model_v1"
submission["prediction"] = list(np.asarray(prediction, dtype=np.float32))
submission.to_parquet("predictions.parquet", index=False)
```

也可以先创建模板：

```bash
pv-benchmark template \
  --benchmark "/path/to/benchmark/ultra_short" \
  --model-id my_model_v1 \
  --output predictions_template.parquet
```

模板中的预测数组是 `NaN` 占位符，必须全部替换后才能评分。

### 7.5 统一评分

```bash
pv-benchmark score \
  --benchmark "/path/to/benchmark/ultra_short" \
  --split validation \
  --predictions predictions.parquet \
  --output metrics.json \
  --details metric_details.parquet
```

模型迭代期间使用 `--split validation`；确定模型和超参数后才使用 `--split test`。`metrics.json` 包含主准确率、场站宏平均、最差场站准确率、`benchmark_id`、当前划分数据哈希和预测文件哈希。`metric_details.parquet` 包含每个 `row_id` 的误差和准确率。

## 8. 从完整192点数据建立 Benchmark

源文件需符合 [数据说明](数据说明.md)，然后分别构造两个任务：

```bash
pv-benchmark build \
  --task ultra_short \
  --source /path/to/pv_samples_15min.parquet \
  --output-dir /path/to/benchmark/ultra_short \
  --train-end "2024-10-31 23:59:59" \
  --validation-end "2024-11-30 23:59:59"

pv-benchmark build \
  --task short_term \
  --source /path/to/pv_samples_15min.parquet \
  --output-dir /path/to/benchmark/short_term \
  --train-end "2024-10-31 23:59:59" \
  --validation-end "2024-11-30 23:59:59"
```

短期构造器只选择10:00起报行，并通过实际时间位置截取次日96点，不要求模型项目自行计算数组下标。

## 9. 版本管理

已发布的 Benchmark 目录视为不可变。数据、切分、字段或指标发生变化时创建新目录，例如 `v2/`，同时升级 `protocol_version`。模型对比报告必须标注完整的 `benchmark_id`，不同 `benchmark_id` 的分数不能直接排名。
