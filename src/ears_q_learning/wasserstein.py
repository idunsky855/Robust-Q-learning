"""Finite-state Wasserstein utilities."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustDualSolution:
    """Optimizer and transformed values for a finite Wasserstein dual."""

    multiplier: float
    transformed_values: np.ndarray
    lower_expectation: float


@dataclass
class _FlowEdge:
    """One mutable residual-network edge."""

    destination: int
    reverse_index: int
    capacity: float
    cost: float


def _add_flow_edge(
    graph: list[list[_FlowEdge]],
    source: int,
    destination: int,
    capacity: float,
    cost: float,
) -> None:
    forward = _FlowEdge(destination, len(graph[destination]), capacity, cost)
    reverse = _FlowEdge(source, len(graph[source]), 0.0, -cost)
    graph[source].append(forward)
    graph[destination].append(reverse)


def _validate_transport_inputs(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
) -> None:
    """Validate a balanced finite optimal-transport problem."""
    if source.ndim != 1 or target.ndim != 1 or source.shape != target.shape:
        raise ValueError("Source and target must be one-dimensional and equally sized.")
    if cost_matrix.shape != (len(source), len(source)):
        raise ValueError("Cost matrix shape must match the distributions.")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("Distributions must contain only finite values.")
    if np.any(source < 0) or np.any(target < 0):
        raise ValueError("Distribution values must be non-negative.")
    if not np.isclose(source.sum(), 1.0) or not np.isclose(target.sum(), 1.0):
        raise ValueError("Source and target must each sum to one.")
    if not np.all(np.isfinite(cost_matrix)) or np.any(cost_matrix < 0):
        raise ValueError("Transport costs must be finite and non-negative.")


def wasserstein_distance(
    source: np.ndarray,
    target: np.ndarray,
    cost_matrix: np.ndarray,
) -> float:
    """Solve a small finite Wasserstein-1 transport problem exactly.

    Successive shortest augmenting paths are sufficient here because the
    project state space has only eight source and eight destination nodes.
    Bellman-Ford permits the negative reverse edges in the residual network.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    _validate_transport_inputs(source, target, cost_matrix)

    state_count = len(source)
    source_node = 0
    origin_offset = 1
    destination_offset = origin_offset + state_count
    sink_node = destination_offset + state_count
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink_node + 1)]

    for state, mass in enumerate(source):
        _add_flow_edge(graph, source_node, origin_offset + state, float(mass), 0.0)
    for origin in range(state_count):
        for destination in range(state_count):
            _add_flow_edge(
                graph,
                origin_offset + origin,
                destination_offset + destination,
                1.0,
                float(cost_matrix[origin, destination]),
            )
    for state, mass in enumerate(target):
        _add_flow_edge(graph, destination_offset + state, sink_node, float(mass), 0.0)

    total_flow = 0.0
    total_cost = 0.0
    tolerance = 1e-12
    while total_flow < 1.0 - tolerance:
        distances = [float("inf")] * len(graph)
        previous: list[tuple[int, int] | None] = [None] * len(graph)
        distances[source_node] = 0.0

        for _ in range(len(graph) - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distances[node] == float("inf"):
                    continue
                for edge_index, edge in enumerate(edges):
                    candidate = distances[node] + edge.cost
                    if edge.capacity > tolerance and candidate < distances[edge.destination] - tolerance:
                        distances[edge.destination] = candidate
                        previous[edge.destination] = (node, edge_index)
                        changed = True
            if not changed:
                break

        if previous[sink_node] is None:
            raise RuntimeError("The balanced transport problem has no augmenting path.")

        augmentation = 1.0 - total_flow
        node = sink_node
        while node != source_node:
            prior_node, edge_index = previous[node]  # type: ignore[misc]
            augmentation = min(augmentation, graph[prior_node][edge_index].capacity)
            node = prior_node

        node = sink_node
        while node != source_node:
            prior_node, edge_index = previous[node]  # type: ignore[misc]
            edge = graph[prior_node][edge_index]
            edge.capacity -= augmentation
            graph[node][edge.reverse_index].capacity += augmentation
            total_cost += augmentation * edge.cost
            node = prior_node
        total_flow += augmentation

    return float(max(0.0, total_cost))


def _candidate_lambdas(values: np.ndarray, cost_row: np.ndarray) -> list[float]:
    candidates = {0.0}
    for i, j in itertools.combinations(range(len(values)), 2):
        denominator = cost_row[i] - cost_row[j]
        if abs(denominator) < 1e-12:
            continue
        candidate = (values[j] - values[i]) / denominator
        if candidate >= 0:
            candidates.add(float(candidate))
    return sorted(candidates)


def robust_lower_expectation_dual(
    reference_distribution: np.ndarray,
    values: np.ndarray,
    epsilon: float,
    cost_matrix: np.ndarray,
) -> float:
    """Compute the Wasserstein-robust lower expectation by dual search.

    The one-dimensional dual objective is piecewise linear. Its maximum is
    attained at zero or where two destination-state affine terms intersect.
    Evaluating all such breakpoints in arrays avoids Python work inside every
    Q-learning update while preserving the exact finite candidate search.
    """
    return robust_lower_expectation_dual_solution(
        reference_distribution=reference_distribution,
        values=values,
        epsilon=epsilon,
        cost_matrix=cost_matrix,
    ).lower_expectation


def robust_lower_expectation_dual_solution(
    reference_distribution: np.ndarray,
    values: np.ndarray,
    epsilon: float,
    cost_matrix: np.ndarray,
) -> RobustDualSolution:
    """Return the optimal multiplier and sampled-update transform values."""
    reference_distribution = np.asarray(reference_distribution, dtype=float)
    values = np.asarray(values, dtype=float)
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    if epsilon < 0:
        raise ValueError("Wasserstein epsilon must be non-negative.")
    if reference_distribution.ndim != 1 or values.ndim != 1:
        raise ValueError("Reference distribution and values must be vectors.")
    if len(reference_distribution) != len(values):
        raise ValueError("Reference distribution and values must have equal length.")
    if cost_matrix.shape != (len(values), len(values)):
        raise ValueError("Cost matrix shape must match the value vector.")

    cost_differences = cost_matrix[:, :, None] - cost_matrix[:, None, :]
    value_differences = values[None, None, :] - values[None, :, None]
    valid = np.abs(cost_differences) >= 1e-12
    intersections = np.divide(
        value_differences,
        cost_differences,
        out=np.full_like(cost_differences, -1.0),
        where=valid,
    )
    candidates = np.unique(
        np.concatenate(([0.0], intersections[intersections >= 0.0]))
    )
    affine_values = (
        values[None, None, :]
        + candidates[:, None, None] * cost_matrix[None, :, :]
    )
    row_terms = np.min(affine_values, axis=2)
    objectives = row_terms @ reference_distribution - candidates * epsilon
    best_index = int(np.argmax(objectives))
    return RobustDualSolution(
        multiplier=float(candidates[best_index]),
        transformed_values=row_terms[best_index].copy(),
        lower_expectation=float(objectives[best_index]),
    )


def robust_lower_expectation_primal_lp(
    reference_distribution: np.ndarray,
    values: np.ndarray,
    epsilon: float,
    cost_matrix: np.ndarray,
) -> float:
    """Solve the finite robust lower-expectation primal by LP vertex search.

    This exact helper is intended for small mathematical checks. It enumerates
    basic feasible solutions of the transport LP with an added budget slack:
    transport rows sum to the reference distribution, transport cost plus slack
    equals the Wasserstein radius, and the objective is destination-state value.
    """
    reference_distribution = np.asarray(reference_distribution, dtype=float)
    values = np.asarray(values, dtype=float)
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    if epsilon < 0:
        raise ValueError("Wasserstein epsilon must be non-negative.")
    if reference_distribution.ndim != 1 or values.ndim != 1:
        raise ValueError("Reference distribution and values must be vectors.")
    if len(reference_distribution) != len(values):
        raise ValueError("Reference distribution and values must have equal length.")
    _validate_transport_inputs(
        reference_distribution,
        reference_distribution.copy(),
        cost_matrix,
    )

    state_count = len(reference_distribution)
    transport_variable_count = state_count * state_count
    slack_index = transport_variable_count
    variable_count = transport_variable_count + 1
    equality_count = state_count + 1

    constraints = np.zeros((equality_count, variable_count), dtype=float)
    rhs = np.zeros(equality_count, dtype=float)
    for origin in range(state_count):
        rhs[origin] = reference_distribution[origin]
        for destination in range(state_count):
            constraints[origin, origin * state_count + destination] = 1.0
            constraints[-1, origin * state_count + destination] = cost_matrix[
                origin, destination
            ]
    constraints[-1, slack_index] = 1.0
    rhs[-1] = epsilon

    objective = np.zeros(variable_count, dtype=float)
    for origin in range(state_count):
        for destination in range(state_count):
            objective[origin * state_count + destination] = values[destination]

    best = float("inf")
    tolerance = 1e-10
    for basis in itertools.combinations(range(variable_count), equality_count):
        basis_matrix = constraints[:, basis]
        try:
            solution = np.linalg.solve(basis_matrix, rhs)
        except np.linalg.LinAlgError:
            continue
        if np.any(solution < -tolerance):
            continue
        full_solution = np.zeros(variable_count, dtype=float)
        full_solution[list(basis)] = np.maximum(solution, 0.0)
        if not np.allclose(constraints @ full_solution, rhs, atol=1e-8):
            continue
        best = min(best, float(objective @ full_solution))

    if best == float("inf"):
        raise RuntimeError("The robust lower-expectation primal LP is infeasible.")
    return best


def robust_lower_expectation_primal_binary(
    reference_distribution: np.ndarray,
    values: np.ndarray,
    epsilon: float,
) -> float:
    """Closed-form two-state primal used in tests."""
    if len(reference_distribution) != 2 or len(values) != 2:
        raise ValueError("This helper only supports two-state examples.")
    if values[0] <= values[1]:
        low, high = 0, 1
    else:
        low, high = 1, 0
    shift = min(reference_distribution[high], epsilon)
    stressed = reference_distribution.copy()
    stressed[low] += shift
    stressed[high] -= shift
    return float(stressed @ values)
