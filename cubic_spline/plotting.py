from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from image_processing import ContourSeries
from spline import SplineResult


def _style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("x (pixeles)")
    ax.set_ylabel("y (pixeles)")


def save_figure(fig: plt.Figure, output_path: Path, show: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if not show:
        plt.close(fig)


def plot_original_image(rgb: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb.astype(np.uint8), origin="upper")
    _style_axes(ax, "Imagen original")
    return fig


def plot_grabcut_diagnostic(gray_image: np.ndarray, grabcut_labels: np.ndarray) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    axes[0].imshow(gray_image, cmap="gray", origin="upper")
    _style_axes(axes[0], "Mapa de intensidades")

    image = axes[1].imshow(grabcut_labels, cmap="viridis", origin="upper", vmin=0, vmax=3)
    _style_axes(axes[1], "Clases iniciales de GrabCut")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="Clase GrabCut")
    fig.suptitle("Diagnostico previo de segmentacion")
    return fig


def plot_mask_and_silhouette(raw_mask: np.ndarray, final_mask: np.ndarray) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    axes[0].imshow(raw_mask, cmap="gray", origin="upper")
    _style_axes(axes[0], "Mascara binaria inicial")
    axes[1].imshow(final_mask, cmap="gray", origin="upper")
    _style_axes(axes[1], "Silueta final del objeto")
    fig.suptitle("Segmentacion del objeto principal")
    return fig


def plot_raw_contours(
    final_mask: np.ndarray,
    upper_contour: ContourSeries,
    lower_contour: ContourSeries,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.imshow(final_mask, cmap="gray", origin="upper")
    ax.plot(
        upper_contour.raw_x,
        upper_contour.raw_y,
        color="cyan",
        linewidth=1.2,
        label="Contorno superior crudo",
    )
    ax.plot(
        lower_contour.raw_x,
        lower_contour.raw_y,
        color="gold",
        linewidth=1.2,
        label="Contorno inferior crudo",
    )
    _style_axes(ax, "Contornos crudos extraidos de la silueta")
    ax.legend(loc="upper right")
    return fig


def plot_sampled_points(
    upper_contour: ContourSeries,
    lower_contour: ContourSeries,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(upper_contour.raw_x, upper_contour.smooth_y, color="deepskyblue", linewidth=1.2, alpha=0.6)
    ax.scatter(
        upper_contour.sampled_x,
        upper_contour.sampled_y,
        color="navy",
        s=18,
        label="Puntos muestreados superiores",
    )
    ax.plot(lower_contour.raw_x, lower_contour.smooth_y, color="orange", linewidth=1.2, alpha=0.6)
    ax.scatter(
        lower_contour.sampled_x,
        lower_contour.sampled_y,
        color="darkred",
        s=18,
        label="Puntos muestreados inferiores",
    )
    _style_axes(ax, "Puntos usados para la interpolacion")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return fig


def plot_spline_vs_points(
    upper_contour: ContourSeries,
    lower_contour: ContourSeries,
    upper_spline: SplineResult,
    lower_spline: SplineResult,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    axes[0].scatter(upper_contour.sampled_x, upper_contour.sampled_y, color="navy", s=18, label="Puntos")
    axes[0].plot(upper_spline.dense_x, upper_spline.dense_y, color="deepskyblue", linewidth=2.0, label="Spline")
    _style_axes(axes[0], "Contorno superior")
    axes[0].invert_yaxis()
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].scatter(lower_contour.sampled_x, lower_contour.sampled_y, color="darkred", s=18, label="Puntos")
    axes[1].plot(lower_spline.dense_x, lower_spline.dense_y, color="orange", linewidth=2.0, label="Spline")
    _style_axes(axes[1], "Contorno inferior")
    axes[1].invert_yaxis()
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle("Spline cubico natural vs puntos muestreados")
    return fig


def plot_final_overlay(
    rgb: np.ndarray,
    upper_spline: SplineResult,
    lower_spline: SplineResult,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.imshow(rgb.astype(np.uint8), origin="upper")
    ax.plot(upper_spline.dense_x, upper_spline.dense_y, color="cyan", linewidth=2.2, label="Spline superior")
    ax.plot(lower_spline.dense_x, lower_spline.dense_y, color="yellow", linewidth=2.2, label="Spline inferior")
    _style_axes(ax, "Superposicion final sobre la imagen original")
    ax.legend(loc="upper right")
    return fig


def plot_linear_vs_spline(
    upper_contour: ContourSeries,
    lower_contour: ContourSeries,
    upper_spline: SplineResult,
    lower_spline: SplineResult,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    upper_linear = np.interp(upper_spline.dense_x, upper_contour.sampled_x, upper_contour.sampled_y)
    lower_linear = np.interp(lower_spline.dense_x, lower_contour.sampled_x, lower_contour.sampled_y)

    axes[0].scatter(upper_contour.sampled_x, upper_contour.sampled_y, color="black", s=15, label="Puntos")
    axes[0].plot(upper_spline.dense_x, upper_linear, color="gray", linewidth=1.4, linestyle="--", label="Interpolacion lineal")
    axes[0].plot(upper_spline.dense_x, upper_spline.dense_y, color="deepskyblue", linewidth=2.0, label="Spline cubico")
    _style_axes(axes[0], "Superior: lineal vs spline")
    axes[0].invert_yaxis()
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].scatter(lower_contour.sampled_x, lower_contour.sampled_y, color="black", s=15, label="Puntos")
    axes[1].plot(lower_spline.dense_x, lower_linear, color="gray", linewidth=1.4, linestyle="--", label="Interpolacion lineal")
    axes[1].plot(lower_spline.dense_x, lower_spline.dense_y, color="orange", linewidth=2.0, label="Spline cubico")
    _style_axes(axes[1], "Inferior: lineal vs spline")
    axes[1].invert_yaxis()
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle("Comparacion entre interpolacion lineal y spline cubico")
    return fig
