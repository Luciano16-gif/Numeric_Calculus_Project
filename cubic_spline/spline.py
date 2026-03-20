from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplineResult:
    x_nodes: np.ndarray
    y_nodes: np.ndarray
    second_derivatives: np.ndarray
    dense_x: np.ndarray
    dense_y: np.ndarray


def _solve_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    lower = lower.astype(np.float64).copy()
    diagonal = diagonal.astype(np.float64).copy()
    upper = upper.astype(np.float64).copy()
    rhs = rhs.astype(np.float64).copy()

    size = diagonal.size
    for index in range(1, size):
        factor = lower[index] / diagonal[index - 1]
        diagonal[index] -= factor * upper[index - 1]
        rhs[index] -= factor * rhs[index - 1]

    solution = np.zeros(size, dtype=np.float64)
    solution[-1] = rhs[-1] / diagonal[-1]
    for index in range(size - 2, -1, -1):
        solution[index] = (rhs[index] - upper[index] * solution[index + 1]) / diagonal[index]

    return solution


def natural_cubic_second_derivatives(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)

    if x_values.size < 3:
        raise ValueError("Se necesitan al menos 3 puntos para construir el spline cubico natural.")
    if np.any(np.diff(x_values) <= 0):
        raise ValueError("Los valores de x deben ser estrictamente crecientes.")

    size = x_values.size
    h_values = np.diff(x_values)

    lower = np.zeros(size, dtype=np.float64)
    diagonal = np.ones(size, dtype=np.float64)
    upper = np.zeros(size, dtype=np.float64)
    rhs = np.zeros(size, dtype=np.float64)

    for index in range(1, size - 1):
        lower[index] = h_values[index - 1]
        diagonal[index] = 2.0 * (h_values[index - 1] + h_values[index])
        upper[index] = h_values[index]
        rhs[index] = 6.0 * (
            (y_values[index + 1] - y_values[index]) / h_values[index]
            - (y_values[index] - y_values[index - 1]) / h_values[index - 1]
        )

    return _solve_tridiagonal(lower, diagonal, upper, rhs)


def evaluate_spline(
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    second_derivatives: np.ndarray,
    x_eval: np.ndarray,
) -> np.ndarray:
    x_nodes = np.asarray(x_nodes, dtype=np.float64)
    y_nodes = np.asarray(y_nodes, dtype=np.float64)
    second_derivatives = np.asarray(second_derivatives, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)

    interval_indices = np.searchsorted(x_nodes, x_eval, side="right") - 1
    interval_indices = np.clip(interval_indices, 0, x_nodes.size - 2)

    x_left = x_nodes[interval_indices]
    x_right = x_nodes[interval_indices + 1]
    y_left = y_nodes[interval_indices]
    y_right = y_nodes[interval_indices + 1]
    m_left = second_derivatives[interval_indices]
    m_right = second_derivatives[interval_indices + 1]
    h_values = x_right - x_left

    left_term = x_right - x_eval
    right_term = x_eval - x_left

    return (
        m_left * left_term**3 / (6.0 * h_values)
        + m_right * right_term**3 / (6.0 * h_values)
        + (y_left - m_left * h_values**2 / 6.0) * (left_term / h_values)
        + (y_right - m_right * h_values**2 / 6.0) * (right_term / h_values)
    )


def fit_natural_cubic_spline(
    x_values: np.ndarray,
    y_values: np.ndarray,
    samples_per_pixel: float = 3.0,
) -> SplineResult:
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)

    second_derivatives = natural_cubic_second_derivatives(x_values, y_values)
    x_span = x_values[-1] - x_values[0]
    dense_count = max(500, int(x_span * samples_per_pixel), x_values.size * 30)
    dense_x = np.linspace(x_values[0], x_values[-1], dense_count)
    dense_y = evaluate_spline(x_values, y_values, second_derivatives, dense_x)

    return SplineResult(
        x_nodes=x_values,
        y_nodes=y_values,
        second_derivatives=second_derivatives,
        dense_x=dense_x,
        dense_y=dense_y,
    )
