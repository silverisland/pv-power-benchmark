# Model project benchmark integration

This project uses the `ultra_short` benchmark at:

```text
../../benchmark_data/v1/ultra_short
```

Locked benchmark ID: `3da8cf0aaecb6b42921fdefe`

Protocol version: `1.0.0`
Model ID: `tabm4pv_benchmark_demo`

Validate the benchmark before work:

```bash
pv-benchmark validate --benchmark "../../benchmark_data/v1/ultra_short"
```

Read `train.parquet`, `validation.parquet`, and `test.parquet` from that directory. Keep `row_id` unchanged. Fit the model and preprocessing on `train`; select iterations with `validation`.

Create validation predictions with columns `row_id`, `prediction`, and `model_id`, then run:

```bash
pv-benchmark score \
  --benchmark "../../benchmark_data/v1/ultra_short" \
  --split validation \
  --predictions artifacts/benchmark/validation_predictions.parquet \
  --output artifacts/benchmark/validation_metrics.json \
  --details artifacts/benchmark/validation_details.parquet
```

After model selection is frozen, produce test predictions and run the same command with `--split test`. Store the training config, random seed, model commit, predictions, and metrics together under `artifacts/benchmark/`.
