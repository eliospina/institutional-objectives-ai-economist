"""Generate SVG figures from experiment result CSV files.

The figure focuses on treatment effects relative to the no-tax baseline. The
absolute lines are nearly flat, so plotting deltas makes the small but measurable
tradeoff visible without adding plotting dependencies.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


POLICIES = ["low_progressive_tax", "high_progressive_tax", "social_floor_feedback"]
COLORS = {
    "low_progressive_tax": "#F58518",
    "high_progressive_tax": "#54A24B",
    "social_floor_feedback": "#B279A2",
}
LABELS = {
    "low_progressive_tax": "Low progressive",
    "high_progressive_tax": "High progressive",
    "social_floor_feedback": "Social-floor feedback",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def compute_deltas(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    by_floor_policy = {
        (float(row["social_floor"]), row["policy"]): row for row in rows
    }
    floors = sorted({float(row["social_floor"]) for row in rows})
    deltas = []
    for floor in floors:
        baseline = by_floor_policy[(floor, "no_tax")]
        baseline_gap = float(baseline["mean_floor_gap"])
        baseline_productivity = float(baseline["mean_productivity"])
        baseline_minimum_coin = float(baseline["mean_minimum_coin"])
        for policy in POLICIES:
            row = by_floor_policy[(floor, policy)]
            floor_gap = float(row["mean_floor_gap"])
            productivity = float(row["mean_productivity"])
            minimum_coin = float(row["mean_minimum_coin"])
            deltas.append(
                {
                    "social_floor": floor,
                    "policy": policy,
                    "floor_gap_reduction_pct": 100
                    * (baseline_gap - floor_gap)
                    / baseline_gap,
                    "productivity_delta_pct": 100
                    * (productivity - baseline_productivity)
                    / baseline_productivity,
                    "minimum_coin_gain": minimum_coin - baseline_minimum_coin,
                }
            )
    return deltas


def scale(value: float, domain: tuple[float, float], range_: tuple[float, float]) -> float:
    low, high = domain
    start, end = range_
    if high == low:
        return (start + end) / 2
    return start + (value - low) * (end - start) / (high - low)


def bar_panel(
    deltas: list[dict[str, float | str]],
    metric: str,
    title: str,
    subtitle: str,
    domain: tuple[float, float],
    unit: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> str:
    floors = sorted({float(row["social_floor"]) for row in deltas})
    left = x + 68
    right = x + width - 30
    top = y + 58
    bottom = y + height - 38
    zero = scale(0, domain, (left, right))
    group_height = (bottom - top) / len(floors)
    bar_height = 14

    parts = [
        f'<text x="{x}" y="{y + 18}" class="panel-title">{title}</text>',
        f'<text x="{x}" y="{y + 38}" class="panel-subtitle">{subtitle}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis" />',
        f'<line x1="{zero:.1f}" y1="{top}" x2="{zero:.1f}" y2="{bottom}" class="zero" />',
    ]

    for tick in [domain[0], 0, domain[1]]:
        tx = scale(tick, domain, (left, right))
        parts.append(
            f'<line x1="{tx:.1f}" y1="{bottom}" x2="{tx:.1f}" y2="{bottom + 5}" class="tick" />'
        )
        parts.append(
            f'<text x="{tx:.1f}" y="{bottom + 20}" class="tick-label" text-anchor="middle">{tick:g}{unit}</text>'
        )

    for floor_index, floor in enumerate(floors):
        group_y = top + floor_index * group_height + 14
        parts.append(
            f'<text x="{x + 8}" y="{group_y + 18}" class="floor-label">floor {floor:g}</text>'
        )
        floor_rows = [
            row for row in deltas if float(row["social_floor"]) == floor
        ]
        for policy_index, row in enumerate(floor_rows):
            value = float(row[metric])
            color = COLORS[str(row["policy"])]
            y_bar = group_y + policy_index * (bar_height + 5)
            end = scale(value, domain, (left, right))
            bar_x = min(zero, end)
            bar_width = max(abs(end - zero), 1)
            label_x = end + 5 if value >= 0 else end - 5
            anchor = "start" if value >= 0 else "end"
            parts.append(
                f'<rect x="{bar_x:.1f}" y="{y_bar:.1f}" width="{bar_width:.1f}" height="{bar_height}" fill="{color}" rx="2" />'
            )
            parts.append(
                f'<text x="{label_x:.1f}" y="{y_bar + 11:.1f}" class="value-label" text-anchor="{anchor}">{value:.2f}{unit}</text>'
            )

    return "\n".join(parts)


def legend_svg(x: int, y: int) -> str:
    parts = []
    offset = 0
    for policy in POLICIES:
        color = COLORS[policy]
        parts.append(
            f'<rect x="{x + offset}" y="{y - 10}" width="14" height="14" fill="{color}" rx="2" />'
        )
        parts.append(
            f'<text x="{x + offset + 22}" y="{y + 2}" class="legend">{LABELS[policy]}</text>'
        )
        offset += 205
    return "\n".join(parts)


def build_svg(rows: list[dict[str, str]]) -> str:
    deltas = compute_deltas(rows)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="860" viewBox="0 0 1120 860" role="img" aria-label="Social floor treatment effects relative to no tax">
<style>
  .title {{ font: 700 25px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #18202b; }}
  .subtitle {{ font: 400 14px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #526070; }}
  .panel-title {{ font: 700 16px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #18202b; }}
  .panel-subtitle {{ font: 400 12px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #526070; }}
  .legend {{ font: 12px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #253043; }}
  .floor-label {{ font: 700 12px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #39475a; }}
  .tick-label {{ font: 11px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #526070; }}
  .value-label {{ font: 600 10px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #253043; }}
  .axis {{ stroke: #39475a; stroke-width: 1; }}
  .tick {{ stroke: #39475a; stroke-width: 1; }}
  .zero {{ stroke: #18202b; stroke-width: 1.25; }}
  .note {{ font: 600 13px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #18202b; }}
</style>
<rect width="1120" height="860" fill="#ffffff" />
<text x="36" y="42" class="title">Minimum Social Floor: Treatment Effects vs No Tax</text>
<text x="36" y="68" class="subtitle">The absolute outcomes are nearly flat, so this figure plots changes relative to the no-tax baseline. Positive floor-gap reduction and minimum-coin gain are better; productivity deltas are the cost.</text>
{legend_svg(36, 104)}
{bar_panel(deltas, "floor_gap_reduction_pct", "Floor-Gap Reduction", "Percent reduction in mean shortfall vs no tax", (0, 5), "%", 36, 135, 520, 310)}
{bar_panel(deltas, "productivity_delta_pct", "Productivity Change", "Percent change in productivity vs no tax", (-0.35, 0), "%", 585, 135, 500, 310)}
{bar_panel(deltas, "minimum_coin_gain", "Minimum-Coin Gain", "Additional minimum coins vs no tax", (0, 2.5), "", 300, 495, 520, 310)}
<text x="36" y="832" class="note">Reading: the social-floor feedback rule has small productivity costs, but it improves floor security most clearly when the target floor is 100 or higher.</text>
</svg>
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="results/social_floor_sensitivity_summary.csv",
    )
    parser.add_argument(
        "--out",
        default="figures/social_floor_treatment_effects.svg",
    )
    args = parser.parse_args()

    rows = load_rows(Path(args.summary))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_svg(rows))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
