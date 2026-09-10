"""Run the extracted TabM demo against a locked benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pv_benchmark import load_benchmark, score_benchmark, validate_benchmark

from .model import predict, train


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prediction_frame(frame: pd.DataFrame, values: np.ndarray, model_id: str) -> pd.DataFrame:
    if values.shape != (len(frame), 16):
        raise ValueError(f"ultra-short prediction must have shape ({len(frame)}, 16); got {values.shape}")
    result = frame[["row_id"]].copy()
    result["prediction"] = list(np.asarray(values, dtype=np.float32))
    result["model_id"] = model_id
    return result[["row_id", "prediction", "model_id"]]


def evaluate_split(
    benchmark_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    *,
    split: str,
    model_id: str,
) -> dict:
    _, splits = load_benchmark(benchmark_dir)
    frame = splits[split]
    values = predict(frame, checkpoint_dir)
    submission = prediction_frame(frame, values, model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{split}_predictions.parquet"
    metrics_path = output_dir / f"{split}_metrics.json"
    details_path = output_dir / f"{split}_details.parquet"
    submission.to_parquet(prediction_path, index=False)
    metrics, details = score_benchmark(
        benchmark_dir,
        prediction_path,
        split=split,
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    details.to_parquet(details_path, index=False)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-id", default="tabm4pv_benchmark_demo")
    parser.add_argument(
        "--mode",
        choices=["train", "train-evaluate", "evaluate"],
        default="train-evaluate",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="Evaluation split. Use validation during iteration and test only after model selection.",
    )
    args = parser.parse_args()

    manifest = validate_benchmark(args.benchmark)
    if manifest["task"]["name"] != "ultra_short":
        raise ValueError("this extracted tabm4pv demo supports the ultra_short benchmark")
    _, splits = load_benchmark(args.benchmark)
    checkpoint_dir = args.output_dir / "checkpoint"
    if args.mode in {"train", "train-evaluate"}:
        metadata = train(
            splits["train"],
            splits["validation"],
            checkpoint_dir,
            config=load_config(args.config),
        )
        metadata.update(
            {
                "benchmark_id": manifest["benchmark_id"],
                "protocol_version": manifest["protocol_version"],
                "model_id": args.model_id,
            }
        )
        (checkpoint_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.mode in {"train-evaluate", "evaluate"}:
        evaluate_split(
            args.benchmark,
            checkpoint_dir,
            args.output_dir,
            split=args.split,
            model_id=args.model_id,
        )


if __name__ == "__main__":
    main()
