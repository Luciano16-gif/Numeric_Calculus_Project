from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .image_processing import ContourSeries
    from .spline import SplineResult
except ImportError:
    from image_processing import ContourSeries
    from spline import SplineResult


def _as_list(item):
    if isinstance(item, list):
        return item
    if isinstance(item, tuple):
        return list(item)
    return [item]


def _style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("x (pixeles)")
    ax.set_ylabel("y (pixeles)")


def _set_equal_data_view(ax: plt.Axes, *series_groups) -> None:
    x_values = []
    y_values = []

    for group in series_groups:
        for item in _as_list(group):
            if isinstance(item, ContourSeries):
                x_values.append(np.asarray(item.sampled_x, dtype=np.float64))
                y_values.append(np.asarray(item.sampled_y, dtype=np.float64))
            else:
                x_values.append(np.asarray(item.dense_x, dtype=np.float64))
                y_values.append(np.asarray(item.dense_y, dtype=np.float64))

    if not x_values or not y_values:
        return

    all_x = np.concatenate(x_values)
    all_y = np.concatenate(y_values)
    x_min = float(np.min(all_x))
    x_max = float(np.max(all_x))
    y_min = float(np.min(all_y))
    y_max = float(np.max(all_y))
    pad_x = max(12.0, 0.04 * max(1.0, x_max - x_min))
    pad_y = max(12.0, 0.04 * max(1.0, y_max - y_min))

    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_max + pad_y, y_min - pad_y)
    ax.set_aspect("equal", adjustable="box")


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
    upper_contours = _as_list(upper_contour)
    lower_contours = _as_list(lower_contour)
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.imshow(final_mask, cmap="gray", origin="upper")
    for index, segment in enumerate(upper_contours):
        ax.plot(
            segment.raw_x,
            segment.raw_y,
            color="cyan",
            linewidth=1.2,
            label="Contorno superior crudo" if index == 0 else None,
        )
    for index, segment in enumerate(lower_contours):
        ax.plot(
            segment.raw_x,
            segment.raw_y,
            color="gold",
            linewidth=1.2,
            label="Contorno inferior crudo" if index == 0 else None,
        )
    _style_axes(ax, "Contornos crudos extraidos de la silueta")
    ax.legend(loc="upper right")
    return fig


def plot_sampled_points(
    upper_contour: ContourSeries,
    lower_contour: ContourSeries,
) -> plt.Figure:
    upper_contours = _as_list(upper_contour)
    lower_contours = _as_list(lower_contour)
    fig, ax = plt.subplots(figsize=(10, 7))
    for index, segment in enumerate(upper_contours):
        ax.plot(segment.smooth_x, segment.smooth_y, color="deepskyblue", linewidth=1.2, alpha=0.6)
        ax.scatter(
            segment.sampled_x,
            segment.sampled_y,
            color="navy",
            s=18,
            label="Puntos muestreados superiores" if index == 0 else None,
        )
    for index, segment in enumerate(lower_contours):
        ax.plot(segment.smooth_x, segment.smooth_y, color="orange", linewidth=1.2, alpha=0.6)
        ax.scatter(
            segment.sampled_x,
            segment.sampled_y,
            color="darkred",
            s=18,
            label="Puntos muestreados inferiores" if index == 0 else None,
        )
    _style_axes(ax, "Puntos usados para la interpolacion")
    _set_equal_data_view(ax, upper_contours, lower_contours)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return fig


def plot_spline_vs_points(
    upper_contour: ContourSeries,
    lower_contour: ContourSeries,
    upper_spline: SplineResult,
    lower_spline: SplineResult,
) -> plt.Figure:
    upper_contours = _as_list(upper_contour)
    upper_splines = _as_list(upper_spline)
    lower_contours = _as_list(lower_contour)
    lower_splines = _as_list(lower_spline)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for index, (segment, spline_segment) in enumerate(zip(upper_contours, upper_splines)):
        axes[0].scatter(
            segment.sampled_x,
            segment.sampled_y,
            color="navy",
            s=18,
            label="Puntos" if index == 0 else None,
        )
        axes[0].plot(
            spline_segment.dense_x,
            spline_segment.dense_y,
            color="deepskyblue",
            linewidth=2.0,
            label="Spline" if index == 0 else None,
        )
    _style_axes(axes[0], "Contorno superior")
    _set_equal_data_view(axes[0], upper_contours, upper_splines)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    for index, (segment, spline_segment) in enumerate(zip(lower_contours, lower_splines)):
        axes[1].scatter(
            segment.sampled_x,
            segment.sampled_y,
            color="darkred",
            s=18,
            label="Puntos" if index == 0 else None,
        )
        axes[1].plot(
            spline_segment.dense_x,
            spline_segment.dense_y,
            color="orange",
            linewidth=2.0,
            label="Spline" if index == 0 else None,
        )
    _style_axes(axes[1], "Contorno inferior")
    _set_equal_data_view(axes[1], lower_contours, lower_splines)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle("Spline cubico natural vs puntos muestreados")
    return fig


def plot_final_overlay(
    rgb: np.ndarray,
    upper_spline: SplineResult,
    lower_spline: SplineResult,
) -> plt.Figure:
    upper_splines = _as_list(upper_spline)
    lower_splines = _as_list(lower_spline)
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.imshow(rgb.astype(np.uint8), origin="upper")
    for index, spline_segment in enumerate(upper_splines):
        ax.plot(
            spline_segment.dense_x,
            spline_segment.dense_y,
            color="cyan",
            linewidth=2.4,
            solid_capstyle="round",
            solid_joinstyle="round",
            label="Spline superior" if index == 0 else None,
        )
    for index, spline_segment in enumerate(lower_splines):
        ax.plot(
            spline_segment.dense_x,
            spline_segment.dense_y,
            color="yellow",
            linewidth=2.6,
            solid_capstyle="round",
            solid_joinstyle="round",
            label="Spline inferior" if index == 0 else None,
        )
    _style_axes(ax, "Superposicion final sobre la imagen original")
    ax.legend(loc="upper right")
    return fig


def plot_linear_vs_spline(
    upper_contour: ContourSeries,
    lower_contour: ContourSeries,
    upper_spline: SplineResult,
    lower_spline: SplineResult,
) -> plt.Figure:
    upper_contours = _as_list(upper_contour)
    upper_splines = _as_list(upper_spline)
    lower_contours = _as_list(lower_contour)
    lower_splines = _as_list(lower_spline)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for index, (segment, spline_segment) in enumerate(zip(upper_contours, upper_splines)):
        if segment.parameter_axis == "x":
            upper_linear_x = spline_segment.dense_x
            upper_linear_y = np.interp(spline_segment.dense_x, segment.sampled_x, segment.sampled_y)
        else:
            upper_linear_y = spline_segment.dense_y
            upper_linear_x = np.interp(spline_segment.dense_y, segment.sampled_y, segment.sampled_x)
        axes[0].scatter(
            segment.sampled_x,
            segment.sampled_y,
            color="black",
            s=15,
            label="Puntos" if index == 0 else None,
        )
        axes[0].plot(
            upper_linear_x,
            upper_linear_y,
            color="gray",
            linewidth=1.4,
            linestyle="--",
            label="Interpolacion lineal" if index == 0 else None,
        )
        axes[0].plot(
            spline_segment.dense_x,
            spline_segment.dense_y,
            color="deepskyblue",
            linewidth=2.0,
            label="Spline cubico" if index == 0 else None,
        )
    _style_axes(axes[0], "Superior: lineal vs spline")
    _set_equal_data_view(axes[0], upper_contours, upper_splines)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    for index, (segment, spline_segment) in enumerate(zip(lower_contours, lower_splines)):
        if segment.parameter_axis == "x":
            lower_linear_x = spline_segment.dense_x
            lower_linear_y = np.interp(spline_segment.dense_x, segment.sampled_x, segment.sampled_y)
        else:
            lower_linear_y = spline_segment.dense_y
            lower_linear_x = np.interp(spline_segment.dense_y, segment.sampled_y, segment.sampled_x)
        axes[1].scatter(
            segment.sampled_x,
            segment.sampled_y,
            color="black",
            s=15,
            label="Puntos" if index == 0 else None,
        )
        axes[1].plot(
            lower_linear_x,
            lower_linear_y,
            color="gray",
            linewidth=1.4,
            linestyle="--",
            label="Interpolacion lineal" if index == 0 else None,
        )
        axes[1].plot(
            spline_segment.dense_x,
            spline_segment.dense_y,
            color="orange",
            linewidth=2.0,
            label="Spline cubico" if index == 0 else None,
        )
    _style_axes(axes[1], "Inferior: lineal vs spline")
    _set_equal_data_view(axes[1], lower_contours, lower_splines)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle("Comparacion entre interpolacion lineal y spline cubico")
    return fig


def plot_book_style_superior(
    upper_contour: ContourSeries,
    upper_spline: SplineResult,
    image_height: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6.5))

    step = max(1, upper_contour.sampled_x.size // 28)
    display_x = upper_contour.sampled_x[::step]
    display_y = upper_contour.sampled_y[::step]
    if display_x[-1] != upper_contour.sampled_x[-1]:
        display_x = np.append(display_x, upper_contour.sampled_x[-1])
        display_y = np.append(display_y, upper_contour.sampled_y[-1])

    points_y = image_height - display_y
    spline_y = image_height - upper_spline.dense_y

    ax.scatter(
        display_x,
        points_y,
        color="#0b84f3",
        s=34,
        label="Puntos refinados",
        zorder=3,
    )
    ax.plot(
        upper_spline.dense_x,
        spline_y,
        color="#14a3dc",
        linewidth=2.6,
        label="Spline cubico natural",
        zorder=2,
    )
    ax.set_title("Contorno superior refinado de G6")
    ax.set_xlabel("x (pixeles)")
    ax.set_ylabel("f(x)")
    ax.grid(True, which="both", linewidth=0.8, alpha=0.45)
    ax.legend(loc="best")

    x_min = max(0.0, np.floor(display_x.min() / 20.0) * 20.0 - 20.0)
    x_max = np.ceil(display_x.max() / 20.0) * 20.0 + 20.0
    y_min = max(0.0, np.floor(points_y.min() / 20.0) * 20.0 - 20.0)
    y_max = np.ceil(points_y.max() / 20.0) * 20.0 + 20.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    return fig
