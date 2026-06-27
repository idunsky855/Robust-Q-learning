"""Finite-state Wasserstein utilities."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


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
    """Compute the Wasserstein-robust lower expectation by dual search."""
    candidates = {0.0}
    for row in range(cost_matrix.shape[0]):
        candidates.update(_candidate_lambdas(values, cost_matrix[row, :]))
    best = float("-inf")
    for lam in sorted(candidates):
        row_terms = []
        for row in range(cost_matrix.shape[0]):
            row_terms.append(np.min(values + lam * cost_matrix[row, :]))
        objective = float(reference_distribution @ np.array(row_terms) - lam * epsilon)
        if objective > best:
            best = objective
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
