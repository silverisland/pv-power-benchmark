"""Create durable Codex project instructions for benchmark-based model work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .protocol import validate_benchmark


START_MARKER = "<!-- pv-benchmark:managed-start -->"
END_MARKER = "<!-- pv-benchmark:managed-end -->"


def _agents_section(benchmark_path: Path, benchmark_id: str, task: str) -> str:
    return f"""{START_MARKER}
## Photovoltaic benchmark contract

This project is evaluated against the `{task}` photovoltaic benchmark.

- Benchmark directory: `{benchmark_path}`
- Locked benchmark ID: `{benchmark_id}`
- Project integration guide: `BENCHMARK.md`
- Before changing model or data code, read `BENCHMARK.md` and `{benchmark_path / 'benchmark.json'}`.
- Run benchmark validation before training or evaluation.
- Treat `row_id`, task schedule, split membership, capacity, target, and metric implementation as fixed contracts.
- Fit preprocessing and model parameters on `train`; use `validation` for iteration and model selection.
- Use `test` only for final comparison after the model choice is fixed.
- Produce predictions with exactly `row_id`, `prediction`, and one `model_id`; preserve every required row exactly once.
- Use `pv-benchmark score` for metrics. Do not reimplement or copy the metric formula into this project.
- Save benchmark predictions, metrics, and run metadata under `artifacts/benchmark/`.
- Report benchmark ID, split, accuracy, accuracy change versus the prior comparable run, and validation commands after each evaluated model change.
- Never compare scores from different benchmark IDs as if they were directly comparable.
{END_MARKER}"""


def _replace_managed_section(existing: str, section: str) -> str:
    if START_MARKER in existing or END_MARKER in existing:
        if existing.count(START_MARKER) != 1 or existing.count(END_MARKER) != 1:
            raise ValueError("AGENTS.md contains malformed pv-benchmark managed markers")
        start = existing.index(START_MARKER)
        end = existing.index(END_MARKER) + len(END_MARKER)
        return existing[:start].rstrip() + "\n\n" + section + "\n" + existing[end:].lstrip()
    if not existing.strip():
        return section + "\n"
    return existing.rstrip() + "\n\n" + section + "\n"


def initialize_codex_project(
    project_dir: str | Path,
    benchmark_dir: str | Path,
    *,
    model_id: str,
) -> dict[str, Any]:
    """Write AGENTS.md and project-local benchmark metadata without losing existing guidance."""

    project_root = Path(project_dir).expanduser().resolve()
    benchmark_root = Path(benchmark_dir).expanduser().resolve()
    if not project_root.exists() or not project_root.is_dir():
        raise FileNotFoundError(f"model project directory not found: {project_root}")
    if not model_id.strip():
        raise ValueError("model_id must not be empty")
    manifest = validate_benchmark(benchmark_root)
    task = manifest["task"]["name"]
    benchmark_id = manifest["benchmark_id"]

    config = {
        "integration_version": "1.0.0",
        "model_id": model_id,
        "task": task,
        "benchmark_dir": str(benchmark_root),
        "benchmark_id": benchmark_id,
        "protocol_version": manifest["protocol_version"],
        "iteration_split": "validation",
        "final_split": "test",
        "artifacts_dir": "artifacts/benchmark",
    }
    config_path = project_root / "pv-benchmark.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    guide = f"""# Model project benchmark integration

This project uses the `{task}` benchmark at:

```text
{benchmark_root}
```

Locked benchmark ID: `{benchmark_id}`

Protocol version: `{manifest['protocol_version']}`
Model ID: `{model_id}`

Validate the benchmark before work:

```bash
pv-benchmark validate --benchmark "{benchmark_root}"
```

Read `train.parquet`, `validation.parquet`, and `test.parquet` from that directory. Keep `row_id` unchanged. Fit the model and preprocessing on `train`; select iterations with `validation`.

Create validation predictions with columns `row_id`, `prediction`, and `model_id`, then run:

```bash
pv-benchmark score \\
  --benchmark "{benchmark_root}" \\
  --split validation \\
  --predictions artifacts/benchmark/validation_predictions.parquet \\
  --output artifacts/benchmark/validation_metrics.json \\
  --details artifacts/benchmark/validation_details.parquet
```

After model selection is frozen, produce test predictions and run the same command with `--split test`. Store the training config, random seed, model commit, predictions, and metrics together under `artifacts/benchmark/`.
"""
    guide_path = project_root / "BENCHMARK.md"
    guide_path.write_text(guide, encoding="utf-8")

    agents_path = project_root / "AGENTS.md"
    existing = agents_path.read_text(encoding="utf-8") if agents_path.exists() else ""
    section = _agents_section(benchmark_root, benchmark_id, task)
    agents_path.write_text(_replace_managed_section(existing, section), encoding="utf-8")
    return {
        "project_dir": str(project_root),
        "benchmark_dir": str(benchmark_root),
        "benchmark_id": benchmark_id,
        "task": task,
        "written": [str(agents_path), str(guide_path), str(config_path)],
    }
