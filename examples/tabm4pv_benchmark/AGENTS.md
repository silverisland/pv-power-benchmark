<!-- pv-benchmark:managed-start -->
## Photovoltaic benchmark contract

This project is evaluated against the `ultra_short` photovoltaic benchmark.

- Benchmark directory: `../../benchmark_data/v1/ultra_short`
- Locked benchmark ID: `3da8cf0aaecb6b42921fdefe`
- Project integration guide: `BENCHMARK.md`
- Before changing model or data code, read `BENCHMARK.md` and `../../benchmark_data/v1/ultra_short/benchmark.json`.
- Run benchmark validation before training or evaluation.
- Treat `row_id`, task schedule, split membership, capacity, target, and metric implementation as fixed contracts.
- Fit preprocessing and model parameters on `train`; use `validation` for iteration and model selection.
- Use `test` only for final comparison after the model choice is fixed.
- Produce predictions with exactly `row_id`, `prediction`, and one `model_id`; preserve every required row exactly once.
- Use `pv-benchmark score` for metrics. Do not reimplement or copy the metric formula into this project.
- Save benchmark predictions, metrics, and run metadata under `artifacts/benchmark/`.
- Report benchmark ID, split, accuracy, accuracy change versus the prior comparable run, and validation commands after each evaluated model change.
- Never compare scores from different benchmark IDs as if they were directly comparable.
<!-- pv-benchmark:managed-end -->
