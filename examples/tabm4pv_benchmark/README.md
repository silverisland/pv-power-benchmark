# tabm4pv Benchmark 接入示例

这个示例从原始 `tabm4pv.py` 模型 demo 中只保留两部分：

- 特征工程：最近96点功率、当前预测时效的GHI/TEMP/WS/WD、目标小时和月份；
- 模型训推理：QuantileTransformer、TabM、AdamW、MSE、标签缩放、梯度裁剪和早停。

原脚本中的目录扫描、Parquet读取、年份切分、测试集评估和旧版月度指标均未复制。Benchmark 包负责数据加载与校验、固定划分、标准预测输出对齐和精度计算。

原脚本只预测未来第16点。本示例为了满足超短期连续4小时的协议，使用相同标量建模方式分别训练第1至第16个时效，最终拼成 `(B, 16)`。

## 文件

| 文件 | 作用 |
| --- | --- |
| `features.py` | 从 Benchmark DataFrame 构造每个时效的模型特征 |
| `model.py` | 纯训练与推理，不读取 Benchmark 文件、不计算业务指标 |
| `run.py` | 调用 Benchmark 加载、校验和评分接口 |
| `config.original.json` | 对齐原 demo 的主要模型和训练参数 |
| `config.smoke.json` | 用于小型 mock Benchmark 的快速闭环测试 |

## 快速实验

使用含有 TabM、PyTorch、scikit-learn 和 rtdl-num-embeddings 的环境，从协议项目根目录执行：

```bash
python -m examples.tabm4pv_benchmark.run \
  --benchmark benchmark_data/v1/ultra_short \
  --output-dir artifacts/benchmark \
  --config examples/tabm4pv_benchmark/config.smoke.json \
  --mode train-evaluate \
  --split validation
```

模型迭代只查看 validation。模型确定后再运行：

```bash
python -m examples.tabm4pv_benchmark.run \
  --benchmark benchmark_data/v1/ultra_short \
  --output-dir artifacts/benchmark \
  --config examples/tabm4pv_benchmark/config.smoke.json \
  --mode evaluate \
  --split test
```

输出包括：

```text
artifacts/benchmark/
├── checkpoint/
│   ├── metadata.json
│   ├── models/horizon_01.pt ... horizon_16.pt
│   └── preprocessors/horizon_01.pkl ... horizon_16.pkl
├── validation_predictions.parquet
├── validation_metrics.json
└── validation_details.parquet
```

预测文件严格只包含 `row_id`、`prediction` 和 `model_id`。真实标签、容量、时间、电站信息与指标参数均由 Benchmark 在评分时按 `row_id` 加载和对齐。
