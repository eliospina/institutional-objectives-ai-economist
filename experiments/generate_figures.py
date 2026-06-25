"""Generate SVG figures from experiment result CSV files.

The script intentionally uses only the Python standard library so the figures
can be regenerated without adding plotting dependencies to the simulation repo.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


COLORS = {
    "no_tax": "#4C78A8",
    "low_progressive_tax": "#F58518",
    "high_progressive_tax": "#54A24B",
    "social_floor_feedback": "#B279A2",
}

LABELS = {
    "no_tax": "No tax",
    "low_progressive_tax": "Low progressive",
    "high_progressive_tax": "High progressive",
    "social_floor_feedback": "Social-floor feedback",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def group_by_policy(rows: list[dict[str, str]], metric: str) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["policy"]].append(
            (float(row["social_floor"]), float(row[f"mean_{metric}"]))
        )
    return {policy: sorted(values) for policy, values in grouped.items()}


def scale(value: float, domain: tuple[float, float], range_: tuple[float, float]) -> float:
    low, high = domain
    start, end = range_
    if high == low:
        return (start + end) / 2
    return start + (value - low) * (end - start) / (high - low)


def nice_domain(values: list[float], pad_ratio: float = 0.08) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    pad = (high - low) * pad_ratio or max(abs(high), 1) * pad_ratio
    return low - pad, high + pad


def panel_svg(
    rows: list[dict[str, str]],
    metric: str,
    title: str,
    y_label: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> str:
    grouped = group_by_policy(rows, metric)
    floors = sorted({float(row["social_floor"]) for row in rows})
    values = [point[1] for points in grouped.values() for point in points]
    x_domain = (min(floors), max(floors))
    y_domain = nice_domain(values)

    left = x + 52
    right = x + width - 18
    top = y + 42
    bottom = y + height - 42

    parts = [
        f'<text x="{x}" y="{y + 18}" class="panel-title">{title}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis" />',
        f'<text x="{left}" y="{y + height - 8}" class="axis-label">Social floor</text>',
        f'<text x="{x + 2}" y="{top - 12}" class="axis-label">{y_label}</text>',
    ]

    for floor in floors:
        px = scale(floor, x_domain, (left, right))
        parts.append(f'<line x1="{px:.1f}" y1="{bottom}" x2="{px:.1f}" y2="{bottom + 5}" class="tick" />')
        parts.append(f'<text x="{px:.1f}" y="{bottom + 20}" class="tick-label" text-anchor="middle">{floor:g}</text>')

    for ratio in [0, 0.5, 1]:
        value = y_domain[0] + ratio * (y_domain[1] - y_domain[0])
        py = scale(value, y_domain, (bottom, top))
        parts.append(f'<line x1="{left - 5}" y1="{py:.1f}" x2="{left}" y2="{py:.1f}" class="tick" />')
        parts.append(f'<line x1="{left}" y1="{py:.1f}" x2="{right}" y2="{py:.1f}" class="grid" />')
        parts.append(f'<text x="{left - 8}" y="{py + 4:.1f}" class="tick-label" text-anchor="end">{value:.1f}</text>')

    for policy, points in grouped.items():
        path = " ".join(
            f"{'M' if index == 0 else 'L'} "
            f"{scale(floor, x_domain, (left, right)):.1f} "
            f"{scale(value, y_domain, (bottom, top)):.1f}"
            for index, (floor, value) in enumerate(points)
        )
        color = COLORS[policy]
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5" />')
        for floor, value in points:
            px = scale(floor, x_domain, (left, right))
            py = scale(value, y_domain, (bottom, top))
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="{color}" />')

    return "\n".join(parts)


def legend_svg(x: int, y: int) -> str:
    parts = []
    offset = 0
    for policy, color in COLORS.items():
        parts.append(f'<line x1="{x + offset}" y1="{y}" x2="{x + offset + 24}" y2="{y}" stroke="{color}" stroke-width="3" />')
        parts.append(f'<text x="{x + offset + 32}" y="{y + 4}" class="legend">{LABELS[policy]}</text>')
        offset += 180
    return "\n".join(parts)


def build_svg(rows: list[dict[str, str]]) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="760" viewBox="0 0 1100 760" role="img" aria-label="Social floor sensitivity results">
<style>
  .title {{ font: 700 24px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #18202b; }}
  .subtitle {{ font: 400 14px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #526070; }}
  .panel-title {{ font: 700 16px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #18202b; }}
  .axis-label {{ font: 600 12px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #526070; }}
  .tick-label {{ font: 11px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #526070; }}
  .legend {{ font: 12px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #253043; }}
  .axis {{ stroke: #39475a; stroke-width: 1; }}
  .tick {{ stroke: #39475a; stroke-width: 1; }}
  .grid {{ stroke: #d9dee7; stroke-width: 1; }}
</style>
<rect width="1100" height="760" fill="#ffffff" />
<text x="36" y="42" class="title">Minimum Social Floor Sensitivity</text>
<text x="36" y="68" class="subtitle">Deterministic tax-rule baselines across social-floor thresholds. Lower floor gap is better; higher floor security and productivity are better.</text>
{legend_svg(36, 106)}
{panel_svg(rows, "floor_gap", "Floor Gap", "Mean shortfall", 36, 145, 500, 260)}
{panel_svg(rows, "floor_security", "Floor Security", "Security score", 575, 145, 490, 260)}
{panel_svg(rows, "productivity", "Productivity", "Mean coin", 305, 455, 500, 260)}
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
        default="figures/social_floor_sensitivity.svg",
    )
    args = parser.parse_args()

    rows = load_rows(Path(args.summary))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_svg(rows))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
