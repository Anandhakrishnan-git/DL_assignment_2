"""2.8 Meta-analysis helper: collect key metrics from prior runs.

This script uses the W&B API to pull runs from a project and logs a summary table
back into the same project (as a new run).

Example:
  python wandb_experiments/exp_2_8_meta_analysis.py --entity <your_entity> --project dl-assignment-2
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--entity", type=str, required=True)
    p.add_argument("--project", type=str, required=True)
    p.add_argument("--name_contains", type=str, default="2.", help="Only include runs whose name contains this.")
    p.add_argument("--max_runs", type=int, default=200)
    p.add_argument("--mode", type=str, choices=("online", "offline", "disabled"), default="online")
    return p.parse_args()


def _get(d: Dict[str, Any], key: str, default=None):
    try:
        return d.get(key, default)
    except Exception:
        return default


def main() -> None:
    args = parse_args()

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")

    rows: List[List[Any]] = []
    for run in runs[: args.max_runs]:
        name = getattr(run, "name", "") or ""
        if args.name_contains and args.name_contains not in name:
            continue

        summary = {}
        try:
            summary = run.summary._json_dict  # type: ignore[attr-defined]
        except Exception:
            summary = {}

        config = {}
        try:
            config = {k: v for k, v in run.config.items() if not str(k).startswith("_")}
        except Exception:
            config = {}

        rows.append(
            [
                run.id,
                name,
                ",".join(getattr(run, "tags", []) or []),
                _get(config, "task"),
                _get(config, "batchnorm"),
                _get(config, "dropout_p"),
                _get(config, "strategy"),
                _get(summary, "best_val_acc"),
                _get(summary, "best_val_dice"),
                _get(summary, "last_stable_lr"),
            ]
        )

    wandb.init(
        project=args.project,
        entity=args.entity,
        name="2.8-meta-analysis",
        mode=args.mode,
        config={
            "name_contains": args.name_contains,
            "max_runs": args.max_runs,
        },
    )

    table = wandb.Table(
        columns=[
            "run_id",
            "name",
            "tags",
            "task",
            "batchnorm",
            "dropout_p",
            "strategy",
            "best_val_acc",
            "best_val_dice",
            "last_stable_lr",
        ]
    )
    for r in rows:
        table.add_data(*r)

    wandb.log({"runs_summary": table}, step=0)
    wandb.finish()


if __name__ == "__main__":
    main()

