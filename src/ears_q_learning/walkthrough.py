"""Generate final SVG figures and the static research walkthrough."""

from __future__ import annotations

from collections import Counter, defaultdict
from html import escape
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
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" width="{width}" height="420" viewBox="0 0 {width} 420">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(subtitle)}</desc>',
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
    html = _html_document(
        country_count=len({row.country for row in rows}),
        transition_count=int(evaluation["transition_count"]),
        epsilon_star=epsilon_star,
        adjusted_difference=adjusted_difference,
        susceptibility_difference=susceptibility_difference,
        carb_difference=carb_difference,
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
    transition_count: int,
    epsilon_star: float,
    adjusted_difference: float,
    susceptibility_difference: float,
    carb_difference: float,
) -> str:
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
    figure {{ margin:0; padding:1rem; background:white; border:1px solid #e3dac9; }} figure img {{ width:100%; display:block; }} figcaption {{ padding:.7rem .4rem .2rem; color:var(--muted); font-size:.87rem; }}
    .equation {{ padding:1.5rem 2rem; margin:2rem 0; border-left:5px solid var(--rust); background:#efe5d4; font-size:1.15rem; overflow-wrap:anywhere; }}
    .decision-trail {{ counter-reset:step; display:grid; gap:1rem; }} .decision {{ position:relative; padding:1.4rem 1.4rem 1.4rem 4.4rem; background:white; border:1px solid #ddd3c1; }}
    .decision:before {{ counter-increment:step; content:counter(step); position:absolute; left:1.3rem; top:1.15rem; width:2rem; height:2rem; display:grid; place-items:center; border-radius:50%; background:var(--navy); color:white; }}
    .result-band {{ background:var(--rust); color:white; }} .result-band .kicker {{ color:#f5cf9b; }}
    .result-band blockquote {{ font-size:clamp(1.8rem,4vw,3.6rem); line-height:1.15; margin:2rem 0; max-width:22ch; }}
    table {{ width:100%; border-collapse:collapse; background:white; margin-top:2rem; }} th,td {{ padding:.9rem; text-align:left; border-bottom:1px solid #ddd3c1; }} th {{ background:var(--navy); color:white; }}
    .limit {{ border-top:1px solid #cfc4b1; padding:1rem 0; }} footer {{ padding:3rem 6vw; background:var(--navy); color:#dce5e7; display:flex; justify-content:space-between; gap:2rem; }}
    @media (max-width:800px) {{ header,.section-grid {{ grid-template-columns:1fr; }} .section-grid>* {{ min-width:0; }} header {{ gap:2rem; }} .figure-grid,.metric-grid {{ grid-template-columns:1fr; }} nav .links {{ display:none; }} main section {{ padding:4.5rem 1.25rem; }} table {{ display:block; max-width:100%; overflow-x:auto; }} footer {{ flex-direction:column; }} }}
    @media (prefers-reduced-motion:reduce) {{ html {{ scroll-behavior:auto; }} }}
  </style>
</head>
<body>
<div class="progress" aria-hidden="true"></div><a class="skip" href="#main">Skip to content</a>
<nav aria-label="Primary navigation"><span class="mark">EARS / RL</span><span class="links"><a href="#question">Question</a><a href="#method">Method</a><a href="#results">Results</a><a href="#limits">Limits</a></span></nav>
<header>
  <div><p class="eyebrow">Distributionally robust reinforcement learning</p><h1>When robustness chooses breadth.</h1><p class="lede">A country-level EARS-Net study of one-year-ahead antibiotic coverage under resistance drift. The strongest result is a trade-off, not a clinical recommendation.</p></div>
  <aside class="hero-card"><span>Primary finding</span><strong>{adjusted_difference:+.2f} pp</strong><p>change in adjusted coverage at epsilon-star. Raw susceptibility increased by {susceptibility_difference:.2f} points, while carbapenem use increased by {carb_difference:.2f} points.</p></aside>
</header>
<main id="main">
  <section id="question" class="paper"><div class="section-grid"><div><p class="kicker">01 / Research question</p><h2>Does robustness preserve useful coverage under drift?</h2></div><div><p>The project compares classical and Wasserstein-robust tabular Q-learning for population-informed empiric antibiotic selection. It uses invasive <em>Escherichia coli</em> surveillance, not patient records.</p><div class="metric-grid"><div class="metric"><strong>{country_count}</strong><span>eligible countries</span></div><div class="metric"><strong>{transition_count}</strong><span>held-out transitions</span></div><div class="metric"><strong>8 x 3</strong><span>states and actions</span></div></div></div></div><div class="figure-grid"><figure><img src="assets/resistance-trends.svg" alt="Resistance trend line chart"><figcaption>Country-level resistance means provide the empirical context.</figcaption></figure><figure><img src="assets/data-coverage.svg" alt="Country-year coverage bar chart"><figcaption>No missing resistance values were imputed.</figcaption></figure></div></section>
  <section id="method"><p class="kicker">02 / Decision model</p><div class="section-grid"><div><h2>A small MDP with an explicit boundary.</h2><p>Three binary resistance indicators define each state. Actions select an antibiotic class. Resistance transitions are action-independent, so the model does not claim that recommendations change future resistance.</p></div><div><div class="equation">Primary reward = next-year susceptibility - 0.10 x I(carbapenem)</div><div class="decision-trail"><div class="decision"><h3>Temporal separation</h3><p>Transitions ending by 2019 train the model; decisions from 2020-2023 are evaluated on 2021-2024 outcomes.</p></div><div class="decision"><h3>Radius calibration</h3><p>The Wasserstein radius is the median historical annual W1 distance: {epsilon_star:.4f}.</p></div><div class="decision"><h3>Independent tuning</h3><p>Classical and robust learners use rolling-origin validation, followed by 30 fixed seeds and exact Bellman checks.</p></div></div></div></div><div class="figure-grid"><figure><img src="assets/wasserstein-drift.svg" alt="Annual Wasserstein drift line chart"><figcaption>The ambiguity radius is calibrated independently of policy performance.</figcaption></figure><figure><img src="assets/policy-by-state.svg" alt="Classical and robust policy actions by state"><figcaption>The primary robust policy collapses to carbapenems in all eight states.</figcaption></figure></div></section>
  <section id="results" class="result-band"><p class="kicker">03 / Result</p><blockquote>Robustness increased susceptibility, but not the endpoint that penalized broad use.</blockquote><p>The classical exact policy achieved 90.13% adjusted coverage with 64.55% carbapenem use. At epsilon-star, the robust exact policy achieved 89.76% adjusted coverage with 100% carbapenem use. No primary training group met the strict all-seed convergence criterion.</p><div class="figure-grid"><figure><img src="assets/performance-by-year.svg" alt="Adjusted coverage by outcome year"><figcaption>Classical adjusted coverage remained above the robust policy in each outcome year.</figcaption></figure><figure><img src="assets/coverage-tradeoff.svg" alt="Susceptibility versus carbapenem use scatter plot"><figcaption>Higher raw susceptibility coincided with universal carbapenem selection.</figcaption></figure></div></section>
  <section class="paper"><p class="kicker">04 / Reward sensitivity</p><div class="section-grid"><div><h2>Changing the objective changed the policy.</h2><p>A prespecified secondary reward added breadth, representative acquisition cost, and an action-severity pressure interaction. These are normative penalties, not estimated causal effects.</p></div><div><table><thead><tr><th>Scenario</th><th>Classical policy</th><th>Carbapenem use</th><th>Scalar reward</th></tr></thead><tbody><tr><td>Primary</td><td><code>20222222</code></td><td>64.55%</td><td>90.13%</td></tr><tr><td>Breadth + pressure</td><td><code>20222222</code></td><td>64.55%</td><td>79.92%</td></tr><tr><td>Breadth + cost + pressure</td><td><code>00022222</code></td><td>49.09%</td><td>78.12%</td></tr></tbody></table><p><small>Scalar rewards across different definitions are not directly comparable.</small></p></div></div></section>
  <section id="limits"><p class="kicker">05 / Interpretation limits</p><h2>What this evidence does not establish.</h2><div class="section-grid"><div><div class="limit"><h3>Not patient-specific</h3><p>Country-year surveillance cannot represent individual diagnosis, contraindications, severity, dose, route, or local formulary constraints.</p></div><div class="limit"><h3>Not causal stewardship</h3><p>Actions do not alter the transition kernel. Future-pressure terms express a normative preference rather than an estimated intervention effect.</p></div></div><div><div class="limit"><h3>Not fully converged</h3><p>Policy agreement was high, but every final training group failed the strict all-seed exact-policy criterion.</p></div><div class="limit"><h3>Not clinical guidance</h3><p>The project evaluates a modeling hypothesis under documented surveillance limitations; it does not recommend antibiotics for care.</p></div></div></div></section>
</main>
<footer><span>EARS-Net Q-Learning</span><span>Idan Dunsky and Yaniv Kaveh Shtul · Afeka Academic College of Engineering</span></footer>
<script>const bar=document.querySelector('.progress');addEventListener('scroll',()=>{{const d=document.documentElement;bar.style.width=(100*d.scrollTop/(d.scrollHeight-d.clientHeight))+'%';}},{{passive:true}});</script>
</body></html>"""
