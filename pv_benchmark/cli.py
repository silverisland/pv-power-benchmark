"""Command-line interface for building, validating, and scoring benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .builder import build_task_dataset
from .integration import initialize_codex_project
from .protocol import (
    load_benchmark,
    validate_benchmark,
    write_benchmark,
    write_presplit_benchmark,
)
from .scoring import score_benchmark
from .spec import TASKS, get_task_spec


def _build(args: argparse.Namespace) -> None:
    source = pd.read_parquet(args.source)
    task_frame = build_task_dataset(source, args.task)
    manifest = write_benchmark(
        task_frame,
        args.output_dir,
        task=args.task,
        train_end=args.train_end,
        validation_end=args.validation_end,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _read_parquet_input(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_parquet(path)
    if not path.is_dir():
        raise FileNotFoundError(f"split input does not exist: {path}")
    files = sorted(candidate for candidate in path.rglob("*.parquet") if candidate.is_file())
    if not files:
        raise ValueError(f"split directory contains no Parquet files: {path}")
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


def _import_splits(args: argparse.Namespace) -> None:
    sources = {
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
    }
    splits = {
        split: build_task_dataset(_read_parquet_input(path), args.task)
        for split, path in sources.items()
    }
    manifest = write_presplit_benchmark(
        splits,
        args.output_dir,
        task=args.task,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _validate(args: argparse.Namespace) -> None:
    manifest = validate_benchmark(args.benchmark)
    summary = {
        "valid": True,
        "protocol_version": manifest["protocol_version"],
        "benchmark_id": manifest["benchmark_id"],
        "task": manifest["task"]["name"],
        "rows": {split: item["rows"] for split, item in manifest["files"].items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _template(args: argparse.Namespace) -> None:
    manifest, splits = load_benchmark(args.benchmark)
    spec = get_task_spec(manifest["task"]["name"])
    selected = splits[args.split]
    template = selected[["row_id"]].copy()
    template["prediction"] = [
        np.full(spec.horizon_points, np.nan, dtype=np.float32) for _ in range(len(template))
    ]
    template["model_id"] = args.model_id
    args.output.parent.mkdir(parents=True, exist_ok=True)
    template.to_parquet(args.output, index=False)
    print(f"wrote {args.output.resolve()} rows={len(template)} horizon={spec.horizon_points}")


def _score(args: argparse.Namespace) -> None:
    summary, details = score_benchmark(
        args.benchmark,
        args.predictions,
        split=args.split,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output.resolve()}")
    if args.details is not None:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        details.to_parquet(args.details, index=False)
        print(f"wrote {args.details.resolve()}")
    print(text)


def _init_project(args: argparse.Namespace) -> None:
    result = initialize_codex_project(
        args.project_dir,
        args.benchmark,
        model_id=args.model_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pv-benchmark", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build locked task splits from source data")
    build.add_argument("--task", required=True, choices=sorted(TASKS))
    build.add_argument("--source", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument("--train-end", required=True)
    build.add_argument("--validation-end", required=True)
    build.set_defaults(handler=_build)

    import_splits = subparsers.add_parser(
        "import-splits",
        help="Build a locked benchmark while preserving predefined split membership",
    )
    import_splits.add_argument("--task", required=True, choices=sorted(TASKS))
    import_splits.add_argument("--train", required=True, type=Path)
    import_splits.add_argument("--validation", required=True, type=Path)
    import_splits.add_argument("--test", required=True, type=Path)
    import_splits.add_argument("--output-dir", required=True, type=Path)
    import_splits.set_defaults(handler=_import_splits)

    validate = subparsers.add_parser("validate", help="Validate hashes and schema")
    validate.add_argument("--benchmark", required=True, type=Path)
    validate.set_defaults(handler=_validate)

    template = subparsers.add_parser("template", help="Create a test prediction template")
    template.add_argument("--benchmark", required=True, type=Path)
    template.add_argument("--output", required=True, type=Path)
    template.add_argument("--model-id", required=True)
    template.add_argument("--split", choices=["validation", "test"], default="test")
    template.set_defaults(handler=_template)

    score = subparsers.add_parser("score", help="Score predictions against locked test labels")
    score.add_argument("--benchmark", required=True, type=Path)
    score.add_argument("--predictions", required=True, type=Path)
    score.add_argument("--output", type=Path)
    score.add_argument("--details", type=Path)
    score.add_argument("--split", choices=["validation", "test"], default="test")
    score.set_defaults(handler=_score)

    init_project = subparsers.add_parser(
        "init-project",
        help="Connect a model project to this benchmark through AGENTS.md",
    )
    init_project.add_argument("--project-dir", required=True, type=Path)
    init_project.add_argument("--benchmark", required=True, type=Path)
    init_project.add_argument("--model-id", required=True)
    init_project.set_defaults(handler=_init_project)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
