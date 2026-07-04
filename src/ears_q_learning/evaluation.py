"""Out-of-time policy evaluation and descriptive comparator metrics."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, median, pstdev

import numpy as np

from ears_q_learning.constants import ACTIONS, ACTION_INDEX, STATE_COUNT
from ears_q_learning.mdp import observed_reward
from ears_q_learning.types import CountryYearRow


def _susceptibility(row: CountryYearRow, action_code: str) -> float:
    resistance = {
        "3gc": row.resistance_3gc,
        "fq": row.resistance_fq,
        "carb": row.resistance_carb,
    }[action_code]
    return 1.0 - resistance / 100.0


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "standard_deviation": float(pstdev(values)) if len(values) > 1 else 0.0,
    }


def _evaluation_transitions(
    rows: list[CountryYearRow],
    decision_year_start: int,
    outcome_year_end: int,
) -> list[tuple[CountryYearRow, CountryYearRow]]:
    by_key = {(row.country, row.year): row for row in rows}
    transitions: list[tuple[CountryYearRow, CountryYearRow]] = []
    for current in rows:
        if not decision_year_start <= current.year < outcome_year_end:
            continue
        next_row = by_key.get((current.country, current.year + 1))
        if next_row is not None:
            transitions.append((current, next_row))
    return sorted(transitions, key=lambda pair: (pair[0].year, pair[0].country))


def evaluate_policy(
    name: str,
    policy: np.ndarray,
    rows: list[CountryYearRow],
    state_lookup: dict[tuple[str, int], int],
    carbapenem_penalty: float,
    decision_year_start: int = 2020,
    outcome_year_end: int = 2024,
) -> dict[str, object]:
    """Evaluate a state-wise policy on complete one-year-ahead transitions."""
    if policy.shape != (STATE_COUNT,):
        raise ValueError(f"Policy must contain {STATE_COUNT} state actions.")
    if np.any((policy < 0) | (policy >= len(ACTIONS))):
        raise ValueError("Policy contains an invalid action index.")

    observations: list[dict[str, object]] = []
    for current, next_row in _evaluation_transitions(
        rows, decision_year_start, outcome_year_end
    ):
        state = state_lookup[(current.country, current.year)]
        action = ACTIONS[int(policy[state])]
        adjusted_reward = observed_reward(next_row, action.code, carbapenem_penalty)
        oracle_reward = max(
            observed_reward(next_row, candidate.code, carbapenem_penalty)
            for candidate in ACTIONS
        )
        observations.append(
            {
                "country": current.country,
                "decision_year": current.year,
                "outcome_year": next_row.year,
                "state": state,
                "action": action.code,
                "adjusted_reward": adjusted_reward,
                "raw_susceptibility": _susceptibility(next_row, action.code),
                "oracle_reward": oracle_reward,
                "regret_to_oracle": oracle_reward - adjusted_reward,
            }
        )
    if not observations:
        raise ValueError("No complete evaluation transitions were found.")

    yearly: list[dict[str, object]] = []
    country_rewards: dict[str, list[float]] = defaultdict(list)
    for observation in observations:
        country_rewards[str(observation["country"])].append(
            float(observation["adjusted_reward"])
        )
    for year in sorted({int(item["decision_year"]) for item in observations}):
        subset = [item for item in observations if item["decision_year"] == year]
        yearly.append(
            {
                "decision_year": year,
                "outcome_year": year + 1,
                "transition_count": len(subset),
                "mean_adjusted_coverage": float(
                    mean(float(item["adjusted_reward"]) for item in subset)
                ),
                "mean_raw_susceptibility": float(
                    mean(float(item["raw_susceptibility"]) for item in subset)
                ),
                "carbapenem_use_rate": float(
                    mean(item["action"] == "carb" for item in subset)
                ),
                "mean_regret_to_oracle": float(
                    mean(float(item["regret_to_oracle"]) for item in subset)
                ),
            }
        )
    country_means = {
        country: float(mean(rewards))
        for country, rewards in sorted(country_rewards.items())
    }
    worst_country = min(country_means, key=lambda country: (country_means[country], country))
    return {
        "name": name,
        "policy": policy.astype(int).tolist(),
        "transition_count": len(observations),
        "mean_adjusted_coverage": float(
            mean(float(item["adjusted_reward"]) for item in observations)
        ),
        "mean_raw_susceptibility": float(
            mean(float(item["raw_susceptibility"]) for item in observations)
        ),
        "carbapenem_use_rate": float(
            mean(item["action"] == "carb" for item in observations)
        ),
        "mean_regret_to_oracle": float(
            mean(float(item["regret_to_oracle"]) for item in observations)
        ),
        "worst_country": worst_country,
        "worst_country_reward": country_means[worst_country],
        "yearly_performance": yearly,
        "country_mean_rewards": country_means,
    }


def _modal_policy(training_output: dict[str, object]) -> np.ndarray:
    summaries = training_output["state_summaries"]
    return np.array([row["modal_action"] for row in summaries], dtype=int)


def _exact_policy(training_output: dict[str, object]) -> np.ndarray:
    return np.asarray(
        training_output["exact_bellman_reference"]["policy"], dtype=int
    )


def _seed_endpoint_summary(
    training_output: dict[str, object],
    rows: list[CountryYearRow],
    state_lookup: dict[tuple[str, int], int],
    carbapenem_penalty: float,
    decision_year_start: int,
    outcome_year_end: int,
) -> dict[str, object]:
    seed_metrics: list[dict[str, object]] = []
    for result in training_output["seed_results"]:
        evaluated = evaluate_policy(
            name=f"seed_{result['seed']}",
            policy=np.asarray(result["policy"], dtype=int),
            rows=rows,
            state_lookup=state_lookup,
            carbapenem_penalty=carbapenem_penalty,
            decision_year_start=decision_year_start,
            outcome_year_end=outcome_year_end,
        )
        seed_metrics.append(
            {
                "seed": result["seed"],
                "mean_adjusted_coverage": evaluated["mean_adjusted_coverage"],
                "mean_raw_susceptibility": evaluated["mean_raw_susceptibility"],
                "carbapenem_use_rate": evaluated["carbapenem_use_rate"],
                "mean_regret_to_oracle": evaluated["mean_regret_to_oracle"],
                "worst_country_reward": evaluated["worst_country_reward"],
            }
        )
    endpoints = (
        "mean_adjusted_coverage",
        "mean_raw_susceptibility",
        "carbapenem_use_rate",
        "mean_regret_to_oracle",
        "worst_country_reward",
    )
    return {
        "seed_metrics": seed_metrics,
        "endpoint_summaries": {
            endpoint: _summary([float(item[endpoint]) for item in seed_metrics])
            for endpoint in endpoints
        },
    }


def run_policy_evaluation(
    rows: list[CountryYearRow],
    state_lookup: dict[tuple[str, int], int],
    training_summary: dict[str, object],
    myopic_policy: np.ndarray,
    carbapenem_penalty: float,
    decision_year_start: int = 2020,
    outcome_year_end: int = 2024,
) -> dict[str, object]:
    """Evaluate learned policies and prespecified descriptive comparators."""
    def evaluate(name: str, policy: np.ndarray) -> dict[str, object]:
        return evaluate_policy(
            name=name,
            policy=policy,
            rows=rows,
            state_lookup=state_lookup,
            carbapenem_penalty=carbapenem_penalty,
            decision_year_start=decision_year_start,
            outcome_year_end=outcome_year_end,
        )

    def learned_evaluation(
        algorithm: str,
        epsilon: float,
        training_output: dict[str, object],
    ) -> dict[str, object]:
        convergence = training_output["convergence"]
        return {
            "algorithm": algorithm,
            "robust_epsilon": epsilon,
            "sampled_policy_role": (
                "descriptive_only"
                if convergence["status"] != "policy_converged"
                else "converged_policy"
            ),
            "convergence": convergence,
            "modal_policy_metrics": evaluate(
                f"{algorithm}_modal_epsilon_{epsilon:g}",
                _modal_policy(training_output),
            ),
            "exact_bellman_policy_metrics": evaluate(
                f"{algorithm}_exact_bellman_epsilon_{epsilon:g}",
                _exact_policy(training_output),
            ),
            **_seed_endpoint_summary(
                training_output,
                rows,
                state_lookup,
                carbapenem_penalty,
                decision_year_start,
                outcome_year_end,
            ),
        }

    classical_training = training_summary["classical"]
    learned = [
        learned_evaluation(
            algorithm="classical",
            epsilon=0.0,
            training_output=classical_training,
        )
    ]
    for radius_output in training_summary["robust"]["radii"]:
        epsilon = float(radius_output["robust_epsilon"])
        learned.append(
            learned_evaluation(
                algorithm="wasserstein_robust",
                epsilon=epsilon,
                training_output=radius_output,
            )
        )

    baselines = [evaluate("myopic", myopic_policy)]
    baselines.extend(
        evaluate(
            f"fixed_{action.code}",
            np.full(STATE_COUNT, ACTION_INDEX[action.code], dtype=int),
        )
        for action in ACTIONS
    )
    transitions = _evaluation_transitions(rows, decision_year_start, outcome_year_end)
    oracle_rewards = [
        max(
            observed_reward(next_row, action.code, carbapenem_penalty)
            for action in ACTIONS
        )
        for _current, next_row in transitions
    ]
    return {
        "status": "evaluation_completed",
        "interpretation": "descriptive_population_informed_non_causal",
        "policy_reporting_rule": (
            "Exact Bellman policies represent model optima. Sampled modal policies are "
            "descriptive only unless all final seeds match the Bellman policy."
        ),
        "decision_years": list(range(decision_year_start, outcome_year_end)),
        "outcome_years": list(range(decision_year_start + 1, outcome_year_end + 1)),
        "transition_count": len(transitions),
        "learned_policies": learned,
        "baselines": baselines,
        "hindsight_oracle": {
            "name": "hindsight_oracle_upper_bound",
            "deployable_policy": False,
            "mean_adjusted_coverage": float(mean(oracle_rewards)),
        },
    }
