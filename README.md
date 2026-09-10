# 光功率数据规范与评估指标

本目录包含统一的数据格式说明和可复现的 mock data。

- [数据说明](数据说明.md)
- [指标计算说明](指标计算说明.md)
- [Benchmark 协议与使用方法](BENCHMARK协议.md)
- [tabm4pv 模型接入示例](examples/tabm4pv_benchmark/README.md)
- `pv_benchmark/`：可安装的 Benchmark 构建、校验和评分包
- `pv_metrics.py`：超短期与短期指标计算接口
- `scripts/generate_mock_data.py`：固定随机种子的 mock data 生成器
- `mock_data/`：已生成的数据样例和 manifest

生成数据需要 Python、NumPy、Pandas 和 PyArrow：

```bash
python -m pip install -r requirements.txt
python scripts/generate_mock_data.py
```

生成器还会在 `benchmark_data/v1/` 下创建可直接用于接口联调的超短期和短期 Benchmark。

让另一个 Codex 模型项目接入超短期 Benchmark：

```bash
pv-benchmark init-project \
  --project-dir /path/to/model-project \
  --benchmark "$PWD/benchmark_data/v1/ultra_short" \
  --model-id my_model_v1
```

快速读取：

```python
import pandas as pd

samples = pd.read_parquet("mock_data/pv_samples_15min.parquet")
predictions = pd.read_parquet("mock_data/predictions_15min.parquet")
stations = pd.read_csv("mock_data/station_info.csv")
```

指标计算：

```python
import pandas as pd

from pv_metrics import evaluate_prediction_frame

frame = pd.read_parquet("mock_data/ultra_short_predictions.parquet")
result = evaluate_prediction_frame(frame, task="ultra_short")
print(result.as_dict())
```

命令行计算：

```bash
python scripts/calculate_metrics.py \
  --task ultra_short \
  --input mock_data/ultra_short_predictions.parquet
```

运行测试：

```bash
python -m unittest discover -s tests -v
```
