from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy import ndimage

try:
    import cv2
except ImportError:  # pragma: no cover - handled with a user-facing error in segment_subject
    cv2 = None


@dataclass
class ImageData:
    path: Path
    rgb: np.ndarray
    gray: np.ndarray


@dataclass
class SegmentationResult:
    grabcut_labels: np.ndarray
    raw_mask: np.ndarray
    final_mask: np.ndarray
    contour_area: float


@dataclass
class ContourSeries:
    label: str
    parameter_axis: str
    raw_x: np.ndarray
    raw_y: np.ndarray
    smooth_x: np.ndarray
    smooth_y: np.ndarray
    sampled_x: np.ndarray
    sampled_y: np.ndarray


def load_image(image_path: Path) -> ImageData:
    if not image_path.exists():
        raise FileNotFoundError(f"No se encontro la imagen: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError as error:
        raise ValueError(f"El archivo no es una imagen valida: {image_path}") from error

    rgb = np.asarray(image, dtype=np.uint8)
    gray = np.asarray(image.convert("L"), dtype=np.float64)
    return ImageData(path=image_path, rgb=rgb, gray=gray)


def load_control_points(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo de puntos refinados: {csv_path}")

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    data = np.atleast_2d(data)
    if data.shape[1] != 2 or data.shape[0] < 3:
        raise ValueError(f"El archivo de puntos refinados es invalido: {csv_path}")

    x_values = data[:, 0].astype(np.float64)
    y_values = data[:, 1].astype(np.float64)
    if np.any(np.diff(x_values) <= 0):
        raise ValueError("Los puntos refinados deben tener coordenadas x estrictamente crecientes.")

    return x_values, y_values


def segment_subject(
    rgb: np.ndarray,
    grabcut_iters: int = 5,
    grabcut_margin: float = 0.05,
    min_component_size: int = 1500,
) -> SegmentationResult:
    if cv2 is None:
        raise ImportError("OpenCV no esta instalado. Instala opencv-python-headless para usar GrabCut.")

    if grabcut_iters < 1:
        raise ValueError("--grabcut-iters debe ser un entero positivo.")
    if not 0.0 < grabcut_margin < 0.45:
        raise ValueError("--grabcut-margin debe estar entre 0 y 0.45.")

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    margin_x = max(1, int(round(width * grabcut_margin)))
    margin_y = max(1, int(round(height * grabcut_margin)))
    rect_width = width - 2 * margin_x
    rect_height = height - 2 * margin_y
    if rect_width <= 0 or rect_height <= 0:
        raise ValueError("El margen configurado deja un rectangulo invalido para GrabCut.")

    rectangle = (margin_x, margin_y, rect_width, rect_height)
    grabcut_labels = np.zeros((height, width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            bgr,
            grabcut_labels,
            rectangle,
            background_model,
            foreground_model,
            int(grabcut_iters),
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error as error:
        raise ValueError(f"GrabCut no pudo segmentar la imagen: {error}") from error

    raw_mask = np.isin(grabcut_labels, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    cleaned_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_close)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel_open)
    cleaned_mask = ndimage.binary_fill_holes(cleaned_mask.astype(bool)).astype(np.uint8)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No se encontro un contorno externo valido despues de GrabCut.")

    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(largest_contour))
    if contour_area < float(min_component_size):
        raise ValueError("El contorno principal detectado es demasiado pequeno.")

    final_mask = np.zeros_like(cleaned_mask, dtype=np.uint8)
    cv2.drawContours(final_mask, [largest_contour], -1, color=1, thickness=cv2.FILLED)
    final_mask = ndimage.binary_fill_holes(final_mask.astype(bool))

    return SegmentationResult(
        grabcut_labels=grabcut_labels,
        raw_mask=raw_mask.astype(bool),
        final_mask=final_mask.astype(bool),
        contour_area=contour_area,
    )


def extract_contours(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_columns = np.flatnonzero(mask.any(axis=0))
    if valid_columns.size == 0:
        raise ValueError("No se encontraron columnas validas en la silueta final.")

    column_mask = mask[:, valid_columns]
    upper_y = np.argmax(column_mask, axis=0)
    lower_y = mask.shape[0] - 1 - np.argmax(column_mask[::-1, :], axis=0)

    x_values = valid_columns.astype(np.float64)
    return x_values, upper_y.astype(np.float64), x_values, lower_y.astype(np.float64)


def extract_lower_outline(
    mask: np.ndarray,
    min_segment_points: int = 2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    return _extract_outline(mask, prefer_lower=True, min_segment_points=min_segment_points)


def extract_upper_outline(
    mask: np.ndarray,
    min_segment_points: int = 2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    return _extract_outline(mask, prefer_lower=False, min_segment_points=min_segment_points)


def _extract_outline(
    mask: np.ndarray,
    prefer_lower: bool,
    min_segment_points: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if cv2 is None:
        raise ImportError("OpenCV no esta instalado.")

    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No se encontro contorno en la mascara.")

    largest = max(contours, key=cv2.contourArea)
    pts = largest.squeeze().astype(np.float64)

    left_idx = int(np.argmin(pts[:, 0]))
    right_idx = int(np.argmax(pts[:, 0]))
    n = len(pts)

    def _build_path(start: int, end: int) -> np.ndarray:
        if start <= end:
            return pts[start : end + 1]
        return np.vstack([pts[start:], pts[: end + 1]])

    path_a = _build_path(left_idx, right_idx)
    path_b = _build_path(right_idx, left_idx)

    if prefer_lower:
        selected_path = path_a if np.mean(path_a[:, 1]) >= np.mean(path_b[:, 1]) else path_b
    else:
        selected_path = path_a if np.mean(path_a[:, 1]) <= np.mean(path_b[:, 1]) else path_b

    if selected_path[0, 0] > selected_path[-1, 0]:
        selected_path = selected_path[::-1]

    monotonic_runs = _split_monotonic_x_runs(selected_path)
    segments: list[tuple[np.ndarray, np.ndarray]] = []

    for run in monotonic_runs:
        if run.shape[0] >= min_segment_points:
            segments.append((run[:, 0].copy(), run[:, 1].copy()))

    return segments


def _split_monotonic_x_runs(path_points: np.ndarray) -> list[np.ndarray]:
    if path_points.shape[0] < 2:
        return []

    delta_x = np.diff(path_points[:, 0])
    if delta_x.size == 0:
        return [path_points.copy()]

    runs: list[np.ndarray] = []
    run_start = 0
    previous_sign = 1 if delta_x[0] >= 0 else -1

    for index, delta in enumerate(delta_x):
        current_sign = previous_sign if delta == 0 else (1 if delta > 0 else -1)
        if current_sign != previous_sign:
            run = path_points[run_start : index + 1].copy()
            if previous_sign < 0:
                run = run[::-1]
            runs.append(run)
            run_start = index
        previous_sign = current_sign

    final_run = path_points[run_start:].copy()
    if previous_sign < 0:
        final_run = final_run[::-1]
    runs.append(final_run)

    return runs


def smooth_contour(y_values: np.ndarray, window_size: int = 5) -> np.ndarray:
    if int(window_size) <= 1:
        return np.asarray(y_values, dtype=np.float64).copy()

    window_size = max(3, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    return ndimage.median_filter(y_values, size=window_size, mode="nearest")


def sample_contour(x_values: np.ndarray, y_values: np.ndarray, step: int = 8) -> tuple[np.ndarray, np.ndarray]:
    if x_values.size == 0:
        raise ValueError("No hay puntos para muestrear.")
    if int(step) < 1:
        raise ValueError("El paso de muestreo debe ser un entero positivo.")

    step = int(step)
    selected_indices = [0]
    last_x = x_values[0]

    for index in range(1, x_values.size - 1):
        if x_values[index] - last_x >= step:
            selected_indices.append(index)
            last_x = x_values[index]

    if selected_indices[-1] != x_values.size - 1:
        selected_indices.append(x_values.size - 1)

    sampled_x = x_values[selected_indices]
    sampled_y = y_values[selected_indices]
    return sampled_x, sampled_y


def _dedupe_monotonic_parameter(
    parameter_values: np.ndarray,
    companion_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if parameter_values.size == 0:
        return parameter_values, companion_values

    kept_parameters = [float(parameter_values[0])]
    kept_companions = [float(companion_values[0])]

    for parameter, companion in zip(parameter_values[1:], companion_values[1:]):
        parameter = float(parameter)
        companion = float(companion)
        if parameter > kept_parameters[-1]:
            kept_parameters.append(parameter)
            kept_companions.append(companion)
        else:
            kept_parameters[-1] = parameter
            kept_companions[-1] = companion

    return (
        np.asarray(kept_parameters, dtype=np.float64),
        np.asarray(kept_companions, dtype=np.float64),
    )


def build_contour_series(
    label: str,
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    sample_step: int,
    smooth_window: int = 3,
    parameter_axis: str = "x",
) -> ContourSeries:
    raw_x = np.asarray(raw_x, dtype=np.float64)
    raw_y = np.asarray(raw_y, dtype=np.float64)

    if parameter_axis == "x":
        parameter_values, smooth_y = _dedupe_monotonic_parameter(raw_x, raw_y)
        smooth_y = smooth_contour(smooth_y, window_size=smooth_window)
        sampled_x, sampled_y = sample_contour(parameter_values, smooth_y, step=sample_step)
        return ContourSeries(
            label=label,
            parameter_axis="x",
            raw_x=raw_x,
            raw_y=raw_y,
            smooth_x=parameter_values.copy(),
            smooth_y=smooth_y,
            sampled_x=sampled_x,
            sampled_y=sampled_y,
        )

    if parameter_axis != "y":
        raise ValueError("parameter_axis debe ser 'x' o 'y'.")

    if raw_y[0] > raw_y[-1]:
        raw_x = raw_x[::-1]
        raw_y = raw_y[::-1]

    parameter_values, smooth_x = _dedupe_monotonic_parameter(raw_y, raw_x)
    smooth_x = smooth_contour(smooth_x, window_size=smooth_window)
    sampled_y, sampled_x = sample_contour(parameter_values, smooth_x, step=sample_step)
    return ContourSeries(
        label=label,
        parameter_axis="y",
        raw_x=raw_x,
        raw_y=raw_y,
        smooth_x=smooth_x,
        smooth_y=parameter_values.copy(),
        sampled_x=sampled_x,
        sampled_y=sampled_y,
    )


def build_auto_oriented_contour_series(
    label: str,
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    sample_step: int,
    smooth_window: int = 3,
) -> ContourSeries:
    def _build_three_point_series(
        x_values: np.ndarray,
        y_values: np.ndarray,
        axis: str,
    ) -> ContourSeries:
        return ContourSeries(
            label=label,
            parameter_axis=axis,
            raw_x=x_values.copy(),
            raw_y=y_values.copy(),
            smooth_x=x_values.copy(),
            smooth_y=y_values.copy(),
            sampled_x=x_values.copy(),
            sampled_y=y_values.copy(),
        )

    def _ensure_three_points(series: ContourSeries, axis: str) -> ContourSeries:
        if series.sampled_x.size >= 3:
            return series

        midpoint_x = np.array([(series.sampled_x[0] + series.sampled_x[-1]) / 2.0], dtype=np.float64)
        midpoint_y = np.array([(series.sampled_y[0] + series.sampled_y[-1]) / 2.0], dtype=np.float64)
        extended_x = np.concatenate((series.sampled_x[:1], midpoint_x, series.sampled_x[-1:]))
        extended_y = np.concatenate((series.sampled_y[:1], midpoint_y, series.sampled_y[-1:]))
        return _build_three_point_series(extended_x, extended_y, axis)

    raw_x = np.asarray(raw_x, dtype=np.float64)
    raw_y = np.asarray(raw_y, dtype=np.float64)
    unique_x = np.unique(raw_x).size
    unique_y = np.unique(raw_y).size

    if unique_x >= 3:
        return _ensure_three_points(build_contour_series(
            label=label,
            raw_x=raw_x,
            raw_y=raw_y,
            sample_step=sample_step,
            smooth_window=smooth_window,
            parameter_axis="x",
        ), "x")

    if unique_y >= 3:
        return _ensure_three_points(build_contour_series(
            label=label,
            raw_x=raw_x,
            raw_y=raw_y,
            sample_step=sample_step,
            smooth_window=smooth_window,
            parameter_axis="y",
        ), "y")

    if raw_x.size < 2:
        raise ValueError("El tramo no tiene suficientes puntos para interpolar.")

    midpoint_x = np.array([(raw_x[0] + raw_x[-1]) / 2.0], dtype=np.float64)
    midpoint_y = np.array([(raw_y[0] + raw_y[-1]) / 2.0], dtype=np.float64)
    extended_x = np.concatenate((raw_x[:1], midpoint_x, raw_x[-1:]))
    extended_y = np.concatenate((raw_y[:1], midpoint_y, raw_y[-1:]))
    parameter_axis = "x" if abs(raw_x[-1] - raw_x[0]) >= abs(raw_y[-1] - raw_y[0]) else "y"

    return _build_three_point_series(extended_x, extended_y, parameter_axis)


def build_bridge_contour_series(
    label: str,
    start_point: np.ndarray,
    end_point: np.ndarray,
) -> ContourSeries:
    start_point = np.asarray(start_point, dtype=np.float64)
    end_point = np.asarray(end_point, dtype=np.float64)
    midpoint = (start_point + end_point) / 2.0

    if abs(end_point[0] - start_point[0]) >= abs(end_point[1] - start_point[1]):
        x_values = np.array([start_point[0], midpoint[0], end_point[0]], dtype=np.float64)
        y_values = np.array([start_point[1], midpoint[1], end_point[1]], dtype=np.float64)
        order = np.argsort(x_values)
        x_values = x_values[order]
        y_values = y_values[order]
        axis = "x"
    else:
        y_values = np.array([start_point[1], midpoint[1], end_point[1]], dtype=np.float64)
        x_values = np.array([start_point[0], midpoint[0], end_point[0]], dtype=np.float64)
        order = np.argsort(y_values)
        y_values = y_values[order]
        x_values = x_values[order]
        axis = "y"

    return ContourSeries(
        label=label,
        parameter_axis=axis,
        raw_x=x_values.copy(),
        raw_y=y_values.copy(),
        smooth_x=x_values.copy(),
        smooth_y=y_values.copy(),
        sampled_x=x_values.copy(),
        sampled_y=y_values.copy(),
    )


def split_at_discontinuities(
    x_values: np.ndarray,
    y_values: np.ndarray,
    threshold: float = 30.0,
    min_segment_size: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if x_values.size == 0:
        return []

    diffs = np.abs(np.diff(y_values))
    jump_indices = np.where(diffs > threshold)[0]

    split_points = np.concatenate(([0], jump_indices + 1, [x_values.size]))

    segments = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        seg_x = x_values[start:end]
        seg_y = y_values[start:end]
        if seg_x.size >= min_segment_size:
            segments.append((seg_x, seg_y))

    return segments


def build_lower_segments_auto(
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    sample_step: int,
    threshold: float = 30.0,
    smooth_window: int = 3,
    min_points_for_spline: int = 3,
) -> list[ContourSeries]:
    segments_data = split_at_discontinuities(raw_x, raw_y, threshold=threshold)

    if not segments_data:
        raise ValueError("No se encontraron segmentos validos en el contorno inferior.")

    segments = []
    for index, (seg_x, seg_y) in enumerate(segments_data, start=1):
        series = build_contour_series(
            label=f"inferior tramo {index}",
            raw_x=seg_x,
            raw_y=seg_y,
            sample_step=sample_step,
            smooth_window=smooth_window,
        )
        if series.sampled_x.size >= min_points_for_spline:
            segments.append(series)

    if not segments:
        raise ValueError("Ningun segmento del contorno inferior tiene suficientes puntos para interpolar.")

    return segments


def build_refined_contour_series(
    label: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> ContourSeries:
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    if x_values.size != y_values.size or x_values.size < 3:
        raise ValueError("Los puntos refinados deben contener al menos 3 pares x,y.")
    if np.any(np.diff(x_values) <= 0):
        raise ValueError("Los puntos refinados deben estar ordenados en x de forma estrictamente creciente.")

    return ContourSeries(
        label=label,
        parameter_axis="x",
        raw_x=x_values,
        raw_y=y_values,
        smooth_x=x_values.copy(),
        smooth_y=y_values.copy(),
        sampled_x=x_values.copy(),
        sampled_y=y_values.copy(),
    )
