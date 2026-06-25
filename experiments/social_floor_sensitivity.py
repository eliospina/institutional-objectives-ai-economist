"""Run a sensitivity analysis across multiple minimum social floor thresholds."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from social_floor_policy_comparison import POLICIES, run_episode


METRICS = [
    "productivity",
    "share_below_floor",
    "floor_gap",
    "floor_security",
    "minimum_coin",
    "gini",
    "mean_tax_rate",
    "government_utility",
]


def parse_floors(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    groups: dict[tuple[float, str], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        groups[(float(row["social_floor"]), str(row["policy"]))].append(row)

    summary = []
    for (social_floor, policy), group_rows in sorted(groups.items()):
        summary_row: dict[str, float | str] = {
            "social_floor": social_floor,
            "policy": policy,
            "seeds": len(group_rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group_rows]
            summary_row[f"mean_{metric}"] = sum(values) / len(values)
        summary.append(summary_row)
    return summary


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--social-floors", default="50,100,150,200")
    parser.add_argument("--tax-period-length", type=int, default=20)
    parser.add_argument("--out", default="results/social_floor_sensitivity.csv")
    parser.add_argument(
        "--summary-out", default="results/social_floor_sensitivity_summary.csv"
    )
    args = parser.parse_args()

    floors = parse_floors(args.social_floors)
    rows = [
        run_episode(
            policy_name=policy_name,
            social_floor=social_floor,
            seed=seed,
            steps=args.steps,
            tax_period_length=args.tax_period_length,
        )
        for social_floor in floors
        for policy_name in POLICIES
        for seed in range(args.seeds)
    ]
    summary = summarize(rows)

    write_csv(Path(args.out), rows)
    write_csv(Path(args.summary_out), summary)
    print(f"wrote {args.out} ({len(rows)} rows)")
    print(f"wrote {args.summary_out} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
