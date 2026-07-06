"""Generate final SVG figures and the static research walkthrough."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from html import escape
import json
from pathlib import Path
from statistics import mean

from ears_q_learning.constants import ACTIONS
from ears_q_learning.state_space import state_to_bits
from ears_q_learning.types import CountryYearRow


NAVY = "#142936"
IVORY = "#f4efe3"
RUST = "#a44a2d"
TEAL = "#2f6f6a"
GOLD = "#c6923d"
MUTED = "#667277"
GRID = "#d9d0bf"


def _svg(title: str, subtitle: str, body: list[str], width: int = 760) -> str:
    identifier = "".join(
        character if character.isalnum() else "-" for character in title.lower()
    ).strip("-")
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="{identifier}-title {identifier}-desc" width="{width}" height="420" viewBox="0 0 {width} 420">',
            f'<title id="{identifier}-title">{escape(title)}</title>',
            f'<desc id="{identifier}-desc">{escape(subtitle)}</desc>',
            f'<rect width="{width}" height="420" rx="18" fill="{IVORY}"/>',
            f'<text x="42" y="42" font-family="Georgia, serif" font-size="22" font-weight="bold" fill="{NAVY}">{escape(title)}</text>',
            f'<text x="42" y="66" font-family="Georgia, serif" font-size="13" fill="{MUTED}">{escape(subtitle)}</text>',
            *body,
            "</svg>",
        ]
    )


def _line_chart(
    title: str,
    subtitle: str,
    series: list[tuple[str, str, list[tuple[int, float]]]],
    y_min: float,
    y_max: float,
    y_format: str = ".0f",
) -> str:
    left, top, width, height = 70, 95, 640, 250
    years = sorted({year for _, _, points in series for year, _ in points})
    x_min, x_max = min(years), max(years)

    def x(value: int) -> float:
        return left + (value - x_min) / max(1, x_max - x_min) * width

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * height

    body: list[str] = []
    for step in range(5):
        value = y_min + (y_max - y_min) * step / 4
        yy = y(value)
        body.extend(
            [
                f'<line x1="{left}" y1="{yy:.1f}" x2="{left + width}" y2="{yy:.1f}" stroke="{GRID}"/>',
                f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end" font-family="Georgia, serif" font-size="11" fill="{MUTED}">{format(value, y_format)}</text>',
            ]
        )
    for year in years:
        body.append(
            f'<text x="{x(year):.1f}" y="{top + height + 24}" text-anchor="middle" font-family="Georgia, serif" font-size="11" fill="{MUTED}">{year}</text>'
        )
    for index, (label, color, points) in enumerate(series):
        coordinates = " ".join(f"{x(year):.1f},{y(value):.1f}" for year, value in points)
        body.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for year, value in points:
            body.append(
                f'<circle cx="{x(year):.1f}" cy="{y(value):.1f}" r="4" fill="{color}" stroke="{IVORY}" stroke-width="2"/>'
            )
        legend_x = 72 + index * 205
        body.extend(
            [
                f'<line x1="{legend_x}" y1="388" x2="{legend_x + 24}" y2="388" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 32}" y="392" font-family="Georgia, serif" font-size="12" fill="{NAVY}">{escape(label)}</text>',
            ]
        )
    return _svg(title, subtitle, body)


def _resistance_trends(rows: list[CountryYearRow]) -> str:
    by_year: dict[int, list[CountryYearRow]] = defaultdict(list)
    for row in rows:
        by_year[row.year].append(row)
    definitions = (
        ("Third-generation cephalosporins", RUST, "resistance_3gc"),
        ("Fluoroquinolones", GOLD, "resistance_fq"),
        ("Carbapenems", TEAL, "resistance_carb"),
    )
    series = [
        (
            label,
            color,
            [
                (year, mean(getattr(row, attribute) for row in by_year[year]))
                for year in sorted(by_year)
            ],
        )
        for label, color, attribute in definitions
    ]
    maximum = max(value for _, _, points in series for _, value in points)
    return _line_chart(
        "Resistance trends",
        "Equal-country mean resistance among eligible EARS-Net country-years (%)",
        series,
        0.0,
        max(30.0, float(int(maximum / 10 + 1) * 10)),
    )


def _data_coverage(rows: list[CountryYearRow]) -> str:
    counts = Counter(row.year for row in rows)
    left, top, chart_width, chart_height = 65, 100, 650, 240
    years = sorted(counts)
    maximum = max(counts.values())
    gap = 9
    bar_width = (chart_width - gap * (len(years) - 1)) / len(years)
    body: list[str] = []
    for index, year in enumerate(years):
        value = counts[year]
        height = value / maximum * chart_height
        xx = left + index * (bar_width + gap)
        yy = top + chart_height - height
        color = NAVY if year <= 2019 else RUST
        body.extend(
            [
                f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{bar_width:.1f}" height="{height:.1f}" rx="4" fill="{color}"/>',
                f'<text x="{xx + bar_width / 2:.1f}" y="{yy - 7:.1f}" text-anchor="middle" font-family="Georgia, serif" font-size="11" fill="{NAVY}">{value}</text>',
                f'<text x="{xx + bar_width / 2:.1f}" y="{top + chart_height + 22}" text-anchor="middle" font-family="Georgia, serif" font-size="10" fill="{MUTED}">{year}</text>',
            ]
        )
    body.extend(
        [
            f'<rect x="450" y="378" width="12" height="12" rx="2" fill="{NAVY}"/><text x="470" y="389" font-family="Georgia, serif" font-size="12" fill="{NAVY}">Training</text>',
            f'<rect x="550" y="378" width="12" height="12" rx="2" fill="{RUST}"/><text x="570" y="389" font-family="Georgia, serif" font-size="12" fill="{NAVY}">Evaluation</text>',
        ]
    )
    return _svg(
        "Data coverage",
        "Complete country-year observations retained after transition filtering",
        body,
    )


def _drift_figure(transition_model: dict[str, object]) -> str:
    points = [
        (int(item["to_year"]), float(item["distance"]))
        for item in transition_model["annual_distances"]
    ]
    epsilon = float(transition_model["epsilon_star"])
    chart = _line_chart(
        "Historical Wasserstein drift",
        f"Annual W1 distance; median calibration radius = {epsilon:.4f}",
        [("Observed annual distance", RUST, points)],
        0.0,
        max(0.10, max(value for _, value in points) * 1.1),
        ".3f",
    )
    return chart.replace(
        "</svg>",
        f'<line x1="70" y1="{95 + (max(0.10, max(v for _, v in points) * 1.1) - epsilon) / max(0.10, max(v for _, v in points) * 1.1) * 250:.1f}" x2="710" y2="{95 + (max(0.10, max(v for _, v in points) * 1.1) - epsilon) / max(0.10, max(v for _, v in points) * 1.1) * 250:.1f}" stroke="{TEAL}" stroke-width="2" stroke-dasharray="7 6"/><text x="708" y="108" text-anchor="end" font-family="Georgia, serif" font-size="11" fill="{TEAL}">epsilon-star</text></svg>',
    )


def _selected_policies(evaluation: dict[str, object], epsilon_star: float):
    classical = next(
        item for item in evaluation["learned_policies"] if item["algorithm"] == "classical"
    )
    robust = min(
        (
            item
            for item in evaluation["learned_policies"]
            if item["algorithm"] == "wasserstein_robust"
        ),
        key=lambda item: abs(float(item["robust_epsilon"]) - epsilon_star),
    )
    return classical, robust


def _policy_figure(evaluation: dict[str, object], epsilon_star: float) -> str:
    classical, robust = _selected_policies(evaluation, epsilon_star)
    policies = (
        ("Classical", classical["exact_bellman_policy_metrics"]["policy"]),
        ("Robust at epsilon-star", robust["exact_bellman_policy_metrics"]["policy"]),
    )
    body: list[str] = []
    action_colors = {0: RUST, 1: GOLD, 2: TEAL}
    for row_index, (label, policy) in enumerate(policies):
        yy = 130 + row_index * 115
        body.append(
            f'<text x="42" y="{yy - 20}" font-family="Georgia, serif" font-size="14" font-weight="bold" fill="{NAVY}">{label}</text>'
        )
        for state, action in enumerate(policy):
            xx = 42 + state * 86
            bits = "".join(str(value) for value in state_to_bits(state))
            body.extend(
                [
                    f'<rect x="{xx}" y="{yy}" width="74" height="72" rx="9" fill="{action_colors[int(action)]}"/>',
                    f'<text x="{xx + 37}" y="{yy + 25}" text-anchor="middle" font-family="Georgia, serif" font-size="11" fill="white">state {bits}</text>',
                    f'<text x="{xx + 37}" y="{yy + 51}" text-anchor="middle" font-family="Georgia, serif" font-size="13" font-weight="bold" fill="white">{escape(ACTIONS[int(action)].label.replace("Third-generation cephalosporin", "3GC"))}</text>',
                ]
            )
    body.append(
        f'<text x="42" y="382" font-family="Georgia, serif" font-size="12" fill="{MUTED}">State bits: 3GC resistance / fluoroquinolone resistance / carbapenem resistance</text>'
    )
    return _svg(
        "Policy by resistance state",
        "Exact Bellman actions; robust optimization selected carbapenems in every state",
        body,
    )


def _yearly_performance(evaluation: dict[str, object], epsilon_star: float) -> str:
    classical, robust = _selected_policies(evaluation, epsilon_star)
    series = []
    for label, color, item in (
        ("Classical adjusted coverage", NAVY, classical),
        ("Robust adjusted coverage", RUST, robust),
    ):
        points = [
            (int(row["outcome_year"]), 100 * float(row["mean_adjusted_coverage"]))
            for row in item["exact_bellman_policy_metrics"]["yearly_performance"]
        ]
        series.append((label, color, points))
    return _line_chart(
        "Performance by outcome year",
        "Mean adjusted coverage for exact policies (%)",
        series,
        89.5,
        91.0,
        ".1f",
    )


def _tradeoff_figure(evaluation: dict[str, object]) -> str:
    left, top, width, height = 80, 100, 610, 240

    def x(value: float) -> float:
        return left + (value - 0.60) / 0.40 * width

    def y(value: float) -> float:
        return top + (1.00 - value) / 0.04 * height

    body: list[str] = []
    for value in (0.6, 0.7, 0.8, 0.9, 1.0):
        xx = x(value)
        body.extend(
            [
                f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top + height}" stroke="{GRID}"/>',
                f'<text x="{xx:.1f}" y="{top + height + 24}" text-anchor="middle" font-family="Georgia, serif" font-size="11" fill="{MUTED}">{value:.0%}</text>',
            ]
        )
    for value in (0.96, 0.97, 0.98, 0.99, 1.0):
        yy = y(value)
        body.extend(
            [
                f'<line x1="{left}" y1="{yy:.1f}" x2="{left + width}" y2="{yy:.1f}" stroke="{GRID}"/>',
                f'<text x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end" font-family="Georgia, serif" font-size="11" fill="{MUTED}">{value:.0%}</text>',
            ]
        )
    for item in evaluation["learned_policies"]:
        metrics = item["exact_bellman_policy_metrics"]
        carb = float(metrics["carbapenem_use_rate"])
        susceptibility = float(metrics["mean_raw_susceptibility"])
        classical = item["algorithm"] == "classical"
        color = NAVY if classical else RUST
        radius = float(item["robust_epsilon"])
        label = "Classical" if classical else f"Robust {radius:.3f}"
        body.extend(
            [
                f'<circle cx="{x(carb):.1f}" cy="{y(susceptibility):.1f}" r="7" fill="{color}" stroke="{IVORY}" stroke-width="2"/>',
                f'<text x="{x(carb) - 8:.1f}" y="{y(susceptibility) - 12:.1f}" text-anchor="end" font-family="Georgia, serif" font-size="11" fill="{color}">{label}</text>',
            ]
        )
    body.extend(
        [
            f'<text x="{left + width / 2}" y="395" text-anchor="middle" font-family="Georgia, serif" font-size="12" fill="{NAVY}">Carbapenem-use rate</text>',
            f'<text x="20" y="{top + height / 2}" transform="rotate(-90 20 {top + height / 2})" text-anchor="middle" font-family="Georgia, serif" font-size="12" fill="{NAVY}">Raw susceptibility</text>',
        ]
    )
    return _svg(
        "Coverage-breadth trade-off",
        "Primary exact policies: greater susceptibility coincided with broader use",
        body,
    )


def write_walkthrough(
    *,
    site_dir: Path,
    rows: list[CountryYearRow],
    transition_model: dict[str, object],
    evaluation: dict[str, object],
    processed_dir: Path | None = None,
) -> dict[str, str]:
    """Write six figures and a self-contained static HTML walkthrough."""
    assets = site_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    epsilon_star = float(transition_model["epsilon_star"])
    figures = {
        "resistance-trends.svg": _resistance_trends(rows),
        "data-coverage.svg": _data_coverage(rows),
        "wasserstein-drift.svg": _drift_figure(transition_model),
        "policy-by-state.svg": _policy_figure(evaluation, epsilon_star),
        "performance-by-year.svg": _yearly_performance(evaluation, epsilon_star),
        "coverage-tradeoff.svg": _tradeoff_figure(evaluation),
    }
    for name, content in figures.items():
        (assets / name).write_text(content + "\n", encoding="ascii")

    classical, robust = _selected_policies(evaluation, epsilon_star)
    classical_metrics = classical["exact_bellman_policy_metrics"]
    robust_metrics = robust["exact_bellman_policy_metrics"]
    adjusted_difference = 100 * (
        float(robust_metrics["mean_adjusted_coverage"])
        - float(classical_metrics["mean_adjusted_coverage"])
    )
    susceptibility_difference = 100 * (
        float(robust_metrics["mean_raw_susceptibility"])
        - float(classical_metrics["mean_raw_susceptibility"])
    )
    carb_difference = 100 * (
        float(robust_metrics["carbapenem_use_rate"])
        - float(classical_metrics["carbapenem_use_rate"])
    )
    details: dict[str, object] = {}
    if processed_dir is not None:
        artifact_names = (
            "preprocessing_report.json",
            "state_assignments.json",
            "training_summary.json",
            "economic_full_training.json",
        )
        for name in artifact_names:
            path = processed_dir / name
            if path.exists():
                details[name] = json.loads(path.read_text(encoding="utf-8"))
        table_path = processed_dir / "final_results_table.csv"
        if table_path.exists():
            with table_path.open("r", encoding="utf-8", newline="") as handle:
                details["final_results_table.csv"] = list(csv.DictReader(handle))
    html = _html_document(
        country_count=len({row.country for row in rows}),
        country_year_count=len(rows),
        transition_count=int(evaluation["transition_count"]),
        epsilon_star=epsilon_star,
        adjusted_difference=adjusted_difference,
        susceptibility_difference=susceptibility_difference,
        carb_difference=carb_difference,
        figures=figures,
        details=details,
        baselines=evaluation.get("baselines", []),
        oracle=evaluation.get("hindsight_oracle", {}),
    )
    index = site_dir / "index.html"
    index.write_text(html, encoding="utf-8")
    return {
        "index": str(index),
        "assets": str(assets),
        "figure_count": str(len(figures)),
    }


def _html_document(
    *,
    country_count: int,
    country_year_count: int,
    transition_count: int,
    epsilon_star: float,
    adjusted_difference: float,
    susceptibility_difference: float,
    carb_difference: float,
    figures: dict[str, str],
    details: dict[str, object],
    baselines: list[dict[str, object]],
    oracle: dict[str, object],
) -> str:
    preprocessing = details.get("preprocessing_report.json", {})
    assignments = details.get("state_assignments.json", {})
    training = details.get("training_summary.json", {})
    thresholds = assignments.get("thresholds", {})
    initial_country_count = preprocessing.get("country_count", country_count)
    country_year_rows = preprocessing.get("country_year_row_count", country_year_count)
    classical_config = training.get("classical", {}).get("configuration", {})
    robust_config = training.get("robust", {}).get("selected_configuration", {})
    economic = details.get("economic_full_training.json", {})
    cost_data = economic.get("cost_data", {})
    costs = cost_data.get("cost_per_ddd_gbp", {})
    state_rows = "".join(
        f"<tr><td>{state}</td><td><code>{''.join(str(bit) for bit in state_to_bits(state))}</code></td><td>{' / '.join(label for bit, label in zip(state_to_bits(state), ('3GC high' if state_to_bits(state)[0] else '3GC low', 'FQ high' if state_to_bits(state)[1] else 'FQ low', 'Carb &gt; 0' if state_to_bits(state)[2] else 'Carb = 0'), strict=True))}</td></tr>"
        for state in range(8)
    )
    baseline_rows = "".join(
        f"<tr><td>{escape(str(row['name']).replace('_', ' ').title())}</td><td>{100 * float(row['mean_adjusted_coverage']):.2f}%</td><td>{100 * float(row.get('mean_raw_susceptibility', 0)):.2f}%</td><td>{100 * float(row.get('carbapenem_use_rate', 0)):.2f}%</td></tr>"
        for row in baselines
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="A reproducible study of Wasserstein-robust Q-learning under European E. coli resistance drift.">
  <title>EARS-Net Q-Learning | Research Walkthrough</title>
  <style>
    :root {{ --navy:{NAVY}; --ivory:{IVORY}; --paper:#fffaf0; --rust:{RUST}; --teal:{TEAL}; --gold:{GOLD}; --muted:{MUTED}; }}
    * {{ box-sizing:border-box; }} html {{ scroll-behavior:smooth; }}
    body {{ margin:0; color:var(--navy); background:var(--ivory); font-family:Georgia,'Times New Roman',serif; line-height:1.65; }}
    .progress {{ position:fixed; inset:0 auto auto 0; height:4px; width:0; background:var(--rust); z-index:20; }}
    .skip {{ position:absolute; left:-999px; }} .skip:focus {{ left:1rem; top:1rem; background:white; padding:.7rem; z-index:30; }}
    nav {{ position:sticky; top:0; z-index:10; display:flex; justify-content:space-between; align-items:center; padding:1rem 5vw; background:rgba(20,41,54,.96); color:white; }}
    nav a {{ color:white; text-decoration:none; margin-left:1.2rem; font-size:.88rem; letter-spacing:.04em; }} .mark {{ font-weight:bold; letter-spacing:.08em; }}
    header {{ min-height:86vh; display:grid; grid-template-columns:1.25fr .75fr; gap:4rem; align-items:center; padding:8vw; background:radial-gradient(circle at 85% 20%,rgba(164,74,45,.22),transparent 28%),linear-gradient(135deg,var(--navy),#203f4e); color:white; }}
    .eyebrow {{ color:#e4b272; text-transform:uppercase; letter-spacing:.16em; font-size:.76rem; }}
    h1 {{ font-size:clamp(3rem,7vw,6.6rem); line-height:.94; margin:.4rem 0 1.5rem; max-width:12ch; font-weight:normal; }}
    h2 {{ font-size:clamp(2rem,4vw,3.6rem); line-height:1.05; font-weight:normal; margin:0 0 1rem; }}
    h3 {{ font-size:1.35rem; margin:.2rem 0 .7rem; }} .lede {{ font-size:1.22rem; max-width:54ch; color:#dce5e7; }}
    .hero-card {{ border:1px solid rgba(255,255,255,.24); padding:2rem; background:rgba(255,255,255,.06); backdrop-filter:blur(8px); }}
    .hero-card strong {{ display:block; font-size:3.4rem; color:#f0c58b; line-height:1; }}
    main section {{ padding:7rem max(6vw,calc((100vw - 1180px)/2)); }} .paper {{ background:var(--paper); }}
    .section-grid {{ display:grid; grid-template-columns:.7fr 1.3fr; gap:5rem; align-items:start; }}
    .kicker {{ color:var(--rust); text-transform:uppercase; letter-spacing:.14em; font-size:.76rem; font-weight:bold; }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin:2rem 0; }}
    .metric {{ padding:1.5rem; border-top:4px solid var(--rust); background:white; box-shadow:0 12px 34px rgba(20,41,54,.08); }}
    .metric strong {{ display:block; font-size:2.1rem; line-height:1.1; }} .metric span {{ color:var(--muted); font-size:.88rem; }}
    .figure-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1.4rem; margin-top:2.5rem; }}
    figure {{ margin:0; padding:1rem; background:white; border:1px solid #e3dac9; }} .figure-art svg {{ width:100%; height:auto; display:block; }} figcaption {{ padding:.7rem .4rem .2rem; color:var(--muted); font-size:.87rem; }}
    .equation {{ padding:1.5rem 2rem; margin:2rem 0; border-left:5px solid var(--rust); background:#efe5d4; font-size:1.15rem; overflow-wrap:anywhere; }}
    .decision-trail {{ counter-reset:step; display:grid; gap:1rem; }} .decision {{ position:relative; padding:1.4rem 1.4rem 1.4rem 4.4rem; background:white; border:1px solid #ddd3c1; }}
    .decision:before {{ counter-increment:step; content:counter(step); position:absolute; left:1.3rem; top:1.15rem; width:2rem; height:2rem; display:grid; place-items:center; border-radius:50%; background:var(--navy); color:white; }}
    .result-band {{ background:var(--rust); color:white; }} .result-band .kicker {{ color:#f5cf9b; }}
    .result-band blockquote {{ font-size:clamp(1.8rem,4vw,3.6rem); line-height:1.15; margin:2rem 0; max-width:22ch; }}
    table {{ width:100%; border-collapse:collapse; background:white; margin-top:2rem; }} th,td {{ padding:.9rem; text-align:left; border-bottom:1px solid #ddd3c1; }} th {{ background:var(--navy); color:white; }}
    .limit {{ border-top:1px solid #cfc4b1; padding:1rem 0; }} footer {{ padding:3rem 6vw; background:var(--navy); color:#dce5e7; display:flex; justify-content:space-between; gap:2rem; }}
    .plain-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1rem; margin:2rem 0; }} .plain-card {{ padding:1.4rem; border:1px solid #d8cdbb; background:rgba(255,255,255,.7); }}
    .plain-card b {{ display:block; color:var(--rust); text-transform:uppercase; letter-spacing:.09em; font-size:.74rem; margin-bottom:.45rem; }}
    .note {{ padding:1.4rem 1.6rem; border-left:5px solid var(--teal); background:#e5eee9; margin:1.5rem 0; }}
    .code-block {{ padding:1rem 1.2rem; background:#10232e; color:#edf2ef; overflow-x:auto; font-family:Consolas,'Courier New',monospace; font-size:.86rem; }}
    .compact td,.compact th {{ padding:.62rem .75rem; font-size:.88rem; }} code {{ font-family:Consolas,'Courier New',monospace; }}
    .conclusion {{ font-size:1.25rem; padding:1.7rem; border:1px solid #d8cdbb; background:white; }}
    @media (max-width:800px) {{ header,.section-grid {{ grid-template-columns:1fr; }} .section-grid>* {{ min-width:0; }} header {{ gap:2rem; }} .figure-grid,.metric-grid,.plain-grid {{ grid-template-columns:1fr; }} nav .links {{ display:none; }} main section {{ padding:4.5rem 1.25rem; }} table {{ display:block; max-width:100%; overflow-x:auto; }} footer {{ flex-direction:column; }} }}
    @media (prefers-reduced-motion:reduce) {{ html {{ scroll-behavior:auto; }} }}
  </style>
</head>
<body>
<div class="progress" aria-hidden="true"></div><a class="skip" href="#main">Skip to content</a>
<nav aria-label="Primary navigation"><span class="mark">EARS / RL</span><span class="links"><a href="#overview">Overview</a><a href="#data">Data</a><a href="#model">Model</a><a href="#results">Results</a><a href="#decisions">Decisions</a><a href="#reproduce">Run</a></span></nav>
<header>
  <div><p class="eyebrow">Distributionally robust reinforcement learning</p><h1>When robustness chooses breadth.</h1><p class="lede">A country-level EARS-Net study of one-year-ahead antibiotic coverage under resistance drift. The strongest result is a trade-off, not a clinical recommendation.</p></div>
  <aside class="hero-card"><span>Primary finding</span><strong>{adjusted_difference:+.2f} pp</strong><p>change in adjusted coverage at epsilon-star. Raw susceptibility increased by {susceptibility_difference:.2f} points, while carbapenem use increased by {carb_difference:.2f} points.</p></aside>
</header>
<main id="main">
  <section id="overview" class="paper"><p class="kicker">01 / Project map</p><div class="section-grid"><div><h2>The complete project in one page.</h2><p>The question is whether Wasserstein-robust Q-learning preserves one-year-ahead adjusted antibiotic coverage when national resistance patterns drift. The answer from this formulation is no: robustness raised raw susceptibility but selected broader treatment and reduced the prespecified endpoint.</p></div><div class="plain-grid"><div class="plain-card"><b>Decision maker</b>Population-informed policy selecting one antibiotic class for a country-year resistance state.</div><div class="plain-card"><b>Comparison</b>Classical tabular Q-learning versus Wasserstein-robust tabular Q-learning.</div><div class="plain-card"><b>Evaluation</b>2020-2023 decisions scored against observed 2021-2024 outcomes.</div><div class="plain-card"><b>Primary endpoint</b>Mean next-year susceptibility minus 0.10 for carbapenem use.</div><div class="plain-card"><b>Main finding</b>Robust adjusted coverage was 0.36 percentage points lower at epsilon-star.</div><div class="plain-card"><b>Scope</b>Country-level methodological study, not patient-level prescribing guidance.</div></div></div><h3>Key terms before you continue</h3><div class="plain-grid"><div class="plain-card"><b>Resistance</b>The percentage of tested isolates classified as resistant to an antibiotic class. Lower is preferable.</div><div class="plain-card"><b>Susceptibility</b>The modeled probability that the class remains effective, calculated here as 1 - resistance/100. Higher is preferable.</div><div class="plain-card"><b>Policy</b>A table assigning one antibiotic action to each of the eight resistance states.</div><div class="plain-card"><b>Adjusted coverage</b>Observed next-year susceptibility after subtracting the carbapenem-use penalty.</div><div class="plain-card"><b>Exact policy</b>The deterministic solution of the estimated finite MDP's Bellman equation.</div><div class="plain-card"><b>Modal policy</b>The most frequent learned action in each state across 30 seeded Q-learning runs.</div></div></section>

  <section id="data"><p class="kicker">02 / Data and cohort</p><div class="section-grid"><div><h2>What was observed.</h2><p>The source is public EARS-Net surveillance for invasive <em>Escherichia coli</em> isolates, primarily blood and cerebrospinal-fluid isolates reported through national systems. Three antibiotic-class exports provide resistance percentages and tested-isolate counts from 2015 through 2024.</p><p>Countries required at least three complete training transitions and two complete evaluation transitions. Missing values were not imputed. Of {initial_country_count} observed countries, {country_count} met the rule, producing {country_year_rows} retained country-year rows and {transition_count} held-out transitions.</p><p>Equal-country weighting is primary: one complete country transition receives one vote regardless of laboratory volume. Tested-isolate weighting is a sensitivity analysis because larger surveillance systems may otherwise dominate the European estimate.</p></div><div><div class="note"><strong>Why this matters:</strong> the data describe national surveillance populations. They do not contain diagnoses, doses, contraindications, clinical severity, or individual treatment outcomes. Differences between countries may also reflect sampling, laboratory, and reporting practices.</div><h3>How to read the figures</h3><p>The resistance-trend figure averages countries equally. It shows that fluoroquinolone and third-generation cephalosporin resistance were much higher than carbapenem resistance. The coverage bars show where complete observations support training and evaluation; they are not patient counts.</p></div></div><div class="figure-grid"><figure><div class="figure-art">{figures["resistance-trends.svg"]}</div><figcaption><strong>Figure 1.</strong> Equal-country mean resistance. The vertical level is resistance prevalence, not reward and not the success rate of prescriptions.</figcaption></figure><figure><div class="figure-art">{figures["data-coverage.svg"]}</div><figcaption><strong>Figure 2.</strong> Complete country-year records. Navy years train the model; rust years support held-out evaluation. A lower bar means fewer complete countries, not fewer isolates.</figcaption></figure></div></section>

  <section id="model" class="paper"><p class="kicker">03 / State, action, transition, reward</p><div class="section-grid"><div><h2>The eight-state MDP.</h2><p>Each state contains three binary indicators in this order: third-generation cephalosporin resistance, fluoroquinolone resistance, and carbapenem resistance. The first two thresholds are training-only medians ({float(thresholds.get("resistance_3gc_median", 0)):.2f}% and {float(thresholds.get("resistance_fq_median", 0)):.2f}%); carbapenem is encoded as zero versus greater than zero. “High” therefore means above the training median, not clinically high.</p><div class="equation">Primary reward r(x,a,x') = susceptibility(x',a) - 0.10 x I(a = carbapenem)</div><p>Actions are `0` third-generation cephalosporin, `1` fluoroquinolone, and `2` carbapenem. A policy such as <code>20222222</code> lists one action for each state from 0 to 7.</p><div class="note"><strong>Worked reward example:</strong> if next-year carbapenem susceptibility is 95%, its adjusted reward is 0.95 - 0.10 = 0.85. A third-generation cephalosporin with 85% susceptibility also receives 0.85. The penalty can therefore make a narrower class preferable even when its raw susceptibility is lower.</div></div><div><table class="compact"><thead><tr><th>State</th><th>Bits</th><th>Interpretation</th></tr></thead><tbody>{state_rows}</tbody></table></div></div><div class="note"><strong>Critical modeling boundary:</strong> the transition kernel is action-independent. Choosing an antibiotic does not change the modeled next resistance state. Therefore, future-resistance effects are not estimated causally, and the environment behaves close to a state-wise decision problem. Q-learning is retained as a controlled application of the reference robust algorithm under observed drift; it is not evidence that a sequential RL model was necessary for this dataset.</div></section>

  <section id="algorithm"><p class="kicker">04 / Learning algorithm</p><div class="section-grid"><div><h2>Classical versus Wasserstein-robust Q-learning.</h2><p>A Q-value answers: “if the system is in this resistance state and chooses this antibiotic class, what discounted reward should be expected now and later?” Classical Q-learning samples a next state and moves the current estimate toward immediate reward plus the best next-state value. Epsilon-greedy exploration sometimes tries a non-greedy action; the learning rate 1/(1 + visits) shrinks as evidence accumulates.</p><div class="equation">Q(s,a) &larr; Q(s,a) + learning rate x [target - Q(s,a)]</div><p>The robust learner replaces the nominal expected target with the worst target inside a Wasserstein ball. Intuitively, Wasserstein distance is an earth-mover cost: an adversary may shift probability from one next state to another, paying more when more resistance bits must change. The policy chooses the action with the best value after this unfavorable redistribution.</p></div><div><div class="plain-card"><b>Ambiguity center</b>Pooled training transition kernel. Total smoothing gamma = 0.10 gives every next state a small probability, preventing unobserved training transitions from being treated as impossible.</div><div class="plain-card"><b>Ground cost</b>Normalized Hamming distance: changing one bit costs 1/3, two bits costs 2/3, and all three costs 1. Wasserstein order q = 1.</div><div class="plain-card"><b>Radius</b>Median historical annual W1 drift, epsilon-star = {epsilon_star:.4f}. Larger radii permit more adversarial redistribution; 0.5x, 1x, 1.5x, and 2x test sensitivity.</div><div class="plain-card"><b>Verification</b>The robust dual was checked against finite linear-program primal examples. Epsilon = 0 recovers the classical comparator, and exact Bellman solutions provide a finite-state reference.</div></div></div><div class="figure-grid"><figure><div class="figure-art">{figures["wasserstein-drift.svg"]}</div><figcaption><strong>Figure 3.</strong> Each point is the transport distance between consecutive annual state distributions. The dashed median is selected before evaluation; it is not tuned to make a policy look favorable.</figcaption></figure><figure><div class="figure-art">{figures["policy-by-state.svg"]}</div><figcaption><strong>Figure 4.</strong> The exact classical policy uses 3GC only in state 1. Under robustness, the adversarial transition calculation makes carbapenem highest-valued in every state.</figcaption></figure></div></section>

  <section id="training" class="paper"><p class="kicker">05 / Training and validation</p><div class="section-grid"><div><h2>How parameters were selected.</h2><p>Two rolling-origin folds respect time: each fold fits thresholds, rewards, and transitions using only earlier years, then scores the next observed transition. This prevents future resistance information from leaking into model selection. Discount factors 0.3, 0.45, 0.7, and 0.9 and exploration rates 0.05, 0.1, and 0.2 were tuned separately for each algorithm over ten fixed seeds.</p><p>The selected classical configuration was discount {float(classical_config.get("discount", 0)):.2f}, exploration {float(classical_config.get("exploration_rate", 0)):.2f}; the robust configuration was discount {float(robust_config.get("discount", 0)):.2f}, exploration {float(robust_config.get("exploration_rate", 0)):.2f}. Separate tuning prevents parameters chosen for one algorithm from disadvantaging the other.</p></div><div><div class="metric-grid"><div class="metric"><strong>50,000</strong><span>updates per final learner</span></div><div class="metric"><strong>10</strong><span>fixed tuning seeds</span></div><div class="metric"><strong>30</strong><span>fixed final seeds</span></div></div><p>Ties were broken by lower seed variability, then proximity to the paper defaults (0.45, 0.10). The <strong>exact policy</strong> is the Bellman optimum of the estimated model. The <strong>modal seeded policy</strong> uses the most frequent learned action in each state. Exact-policy outcomes compare models; seed distributions show whether finite Q-learning reliably recovers those policies.</p></div></div></section>

  <section id="decisions"><p class="kicker">06 / Decision trail</p><h2>Why the project was designed this way.</h2><div class="decision-trail"><div class="decision"><h3>Population level, not patient level</h3><p>EARS-Net lacks patient covariates and treatment assignments. The project therefore studies country-year empiric selection and avoids patient-specific claims.</p></div><div class="decision"><h3>Eight states rather than continuous resistance</h3><p>Three binary indicators keep the tabular problem auditable and match Setting 1 in the reference formulation. Thresholds use training data only to prevent leakage.</p></div><div class="decision"><h3>Action-independent transitions</h3><p>Surveillance does not identify how choosing a class changes future national resistance. The project does not invent that causal mechanism.</p></div><div class="decision"><h3>Carbapenem penalty in the primary reward</h3><p>Susceptibility alone would favor the broadest class. A fixed 0.10 penalty makes broad use visible in the endpoint while retaining raw susceptibility as a separate outcome.</p></div><div class="decision"><h3>Historical calibration of robustness</h3><p>The radius comes from observed 2015-2019 state-distribution drift, not from searching for a desirable policy.</p></div><div class="decision"><h3>Secondary reward rather than retrospective replacement</h3><p>Breadth, representative acquisition cost, and future-pressure interactions are reported as normative sensitivity analyses, not used to rewrite the primary result.</p></div></div></section>

  <section id="results" class="result-band"><p class="kicker">07 / Primary result</p><blockquote>Robustness increased susceptibility, but not the endpoint that penalized broad use.</blockquote><p>The classical exact policy <code>20222222</code> achieved 90.13% adjusted coverage, 96.58% raw susceptibility, and 64.55% carbapenem use. At epsilon-star, the robust exact policy <code>22222222</code> achieved 89.76% adjusted coverage, 99.76% raw susceptibility, and 100% carbapenem use.</p><p>Raw susceptibility asks only whether the chosen class is likely to cover the organism. Adjusted coverage additionally charges 0.10 whenever that class is a carbapenem. The robust policy gained {susceptibility_difference:.2f} susceptibility points by always choosing the class with near-universal observed activity, but paid the carbapenem penalty on every transition. Its adjusted coverage therefore changed by {adjusted_difference:+.2f} points.</p><p>The all-carbapenem result follows the model: the adversary shifts probability toward resistant next states, while carbapenem resistance remains rare in the data. The fixed 0.10 penalty was not large enough to offset carbapenem's worst-case efficacy advantage. Robustness was conservative about coverage, not stewardship.</p><p>The robust policy also had greater regret to the hindsight oracle. The primary hypothesis was therefore not supported.</p><div class="figure-grid"><figure><div class="figure-art">{figures["performance-by-year.svg"]}</div><figcaption><strong>Figure 5.</strong> Each point averages complete country transitions for one outcome year. Classical adjusted coverage exceeded robust coverage in all four years; this is descriptive, not a significance test.</figcaption></figure><figure><div class="figure-art">{figures["coverage-tradeoff.svg"]}</div><figcaption><strong>Figure 6.</strong> The horizontal axis is carbapenem-use rate and the vertical axis is unpenalized susceptibility. Robust radii overlap at the upper-right because they produce the same all-carbapenem policy.</figcaption></figure></div></section>

  <section id="baselines" class="paper"><p class="kicker">08 / Baselines and sensitivities</p><div class="section-grid"><div><h2>What the policies were compared against.</h2><table class="compact"><thead><tr><th>Baseline</th><th>Adjusted coverage</th><th>Raw susceptibility</th><th>Carbapenem use</th></tr></thead><tbody>{baseline_rows}</tbody></table><p>The myopic baseline chooses the highest immediate modeled reward in each state. It equals the classical exact policy because actions share the same transition kernel, so future-state value does not distinguish actions. That agreement is expected and exposes the limited sequential role of RL here.</p><p>The hindsight oracle chooses the best class after observing each next-year outcome. Its adjusted coverage was {100 * float(oracle.get("mean_adjusted_coverage", 0)):.2f}%. It is an unattainable upper bound, not a deployable policy. Regret is the gap between a policy and this oracle. Worst-country reward was also recorded to reveal poor geographic performance, but remains descriptive because countries are heterogeneous.</p></div><div><h3>Prespecified sensitivity findings</h3><div class="equation">Secondary reward = susceptibility - beta x breadth - gamma x normalized cost - delta x breadth x next-state severity</div><ul><li>Tested-isolate weighting selected carbapenems in every state for both algorithms, so larger surveillance volumes strengthened rather than removed the broad-policy result.</li><li>Adding breadth and future-pressure penalties reduced carbapenem use for some classical policies, but robust policies remained broader. The pressure term is normative, not a causal estimate of resistance caused by treatment.</li><li>Representative one-DDD acquisition costs came from the English NHS eMIT database: about GBP {float(costs.get("3gc", 0)):.2f} for ceftriaxone, GBP {float(costs.get("fq", 0)):.2f} for ciprofloxacin, and GBP {float(costs.get("carb", 0)):.2f} for meropenem. Log normalization limits scale dominance.</li><li>With beta = 0.15, gamma = 0.025, and delta = 0.10, the cost-adjusted classical policy <code>00022222</code> reduced carbapenem use to 49.09%; the robust epsilon-star policy increased it to 64.55%.</li><li>Fluoroquinolones were not selected by any reported exact policy. They were dominated under the estimated efficacy and penalties; the algorithm was not required to use every action.</li></ul><div class="note">Rewards from different definitions are not numerically comparable. Sensitivities show how normative objectives change policy composition; they do not retrospectively replace the primary endpoint.</div></div></div></section>

  <section id="convergence"><p class="kicker">09 / Convergence and uncertainty</p><div class="section-grid"><div><h2>What “not converged” means here.</h2><p>The strict criterion required every one of the 30 seeded greedy policies to match the exact Bellman policy in all eight states. No final training group met that criterion. Primary mean state agreement was 90.00% for classical Q-learning and approximately 96.67% at epsilon-star for robust Q-learning.</p><p>The classical mismatch was concentrated in state 1, where only 6 of 30 seeds selected the exact 3GC action and the modal learned action was carbapenem. This signals a small or difficult action-value margin, not broad disagreement across the whole policy.</p><p>The exact Bellman solution remains deterministic for the estimated model. However, sampled Q-learning retained finite-run error after 50,000 updates, so sampled policies must not be described as fully converged.</p></div><div><div class="plain-card"><b>Policy agreement</b>The fraction of eight states where a seeded greedy policy matches the exact policy, averaged across seeds.</div><div class="plain-card"><b>Bellman sup-norm error</b>The largest absolute difference between learned and exact Q-values. It can remain nonzero even when greedy actions agree.</div><div class="plain-card"><b>Reported together</b>Exact policy, modal policy, state agreement, Q-value error, visits, and seed outcomes prevent reward stability alone from being called convergence.</div><div class="plain-card"><b>Not claimed</b>Asymptotic convergence, statistical significance, confidence intervals, causal effects, or clinical safety.</div></div></div></section>

  <section id="conclusion" class="paper"><p class="kicker">10 / Conclusions</p><h2>How to interpret the evidence.</h2><div class="plain-grid"><div class="conclusion"><strong>1. Evidence</strong><p>Robust Q-learning did not improve the prespecified adjusted-coverage endpoint and shifted the policy toward universal carbapenem use.</p></div><div class="conclusion"><strong>2. Assumption</strong><p>This comparison is valid for the constructed country-year MDP, its discretization, estimated rewards, and action-independent transition model.</p></div><div class="conclusion"><strong>3. Not established</strong><p>The study does not show which antibiotic an individual patient should receive or how prescribing changes future resistance.</p></div></div><p>The methodological contribution is the reproducible application and audit: it shows that distributional robustness can protect modeled efficacy while worsening a stewardship-adjusted objective. Negative results remain informative when the endpoint, calibration, and limitations are prespecified.</p></section>

  <section id="limits"><p class="kicker">11 / Limitations</p><h2>What could change the answer.</h2><div class="section-grid"><div><div class="limit"><h3>Aggregate surveillance</h3><p>Country-level percentages obscure hospitals, patients, infection sources, severity, treatment histories, and laboratory practice differences.</p></div><div class="limit"><h3>Partial observability</h3><p>Three binary indicators are not a complete Markov description of antimicrobial resistance dynamics.</p></div><div class="limit"><h3>Reward specification</h3><p>The carbapenem, breadth, cost, and pressure coefficients express study objectives. Different justified values can change policies.</p></div></div><div><div class="limit"><h3>Action-independent dynamics</h3><p>The formulation cannot estimate stewardship effects on future resistance and largely reduces long-run action differences to immediate modeled rewards.</p></div><div class="limit"><h3>External validity</h3><p>Results apply to included European surveillance years and should not be generalized to local formularies or patient care.</p></div><div class="limit"><h3>Pandemic-era shift</h3><p>2020-2024 is treated as a temporal shift period without attributing observed resistance changes solely to COVID-19.</p></div></div></div></section>

  <section id="reproduce" class="paper"><p class="kicker">12 / Reproduce the project</p><div class="section-grid"><div><h2>Two commands, two purposes.</h2><p>The demo uses one fixed seed and selected parameters for a short execution walkthrough. The primary configuration performs all tuning, 30-seed training, weighting sensitivity, stewardship scenarios, and cost analysis used in the report.</p><h3>Fast demo</h3><pre class="code-block">uv run --python 3.11 python -m ears_q_learning run --config configs/demo.yaml</pre><h3>Complete experiment</h3><pre class="code-block">uv run --python 3.11 python -m ears_q_learning run --config configs/primary.yaml</pre></div><div><h3>Generated evidence</h3><p>Machine-readable outputs include raw-data validation, preprocessing reports, state assignments, transition kernels, tuning traces, all seeded policies, exact Bellman references, endpoint metrics, sensitivity tables, figures, and run metadata.</p><p>The full run took 2 hours 37 minutes on the development machine; the demo took about two minutes. Python 3.11 is required. The model is CPU-based and does not require a GPU.</p><div class="note">For the precise environment setup, standard-Python commands, and output paths, consult the repository README.</div></div></div></section>
</main>
<footer><span>EARS-Net Q-Learning</span><span>Idan Dunsky and Yaniv Kaveh Shtul · Afeka Academic College of Engineering</span></footer>
<script>const bar=document.querySelector('.progress');addEventListener('scroll',()=>{{const d=document.documentElement;bar.style.width=(100*d.scrollTop/(d.scrollHeight-d.clientHeight))+'%';}},{{passive:true}});</script>
</body></html>"""
