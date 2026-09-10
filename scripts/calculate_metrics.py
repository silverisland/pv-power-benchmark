"""Calculate photovoltaic forecast accuracy from a Parquet prediction table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pv_metrics import evaluate_prediction_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["ultra_short", "short_term"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--capacity-floor-ratio", type=float, default=0.2)
    parser.add_argument(
        "--skip-schedule-validation",
        action="store_true",
        help="Skip issue-time and target-interval validation.",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    result = evaluate_prediction_frame(
        frame,
        task=args.task,
        capacity_floor_ratio=args.capacity_floor_ratio,
        validate_schedule=not args.skip_schedule_validation,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
