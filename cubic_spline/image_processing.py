from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
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
    raw_x: np.ndarray
    raw_y: np.ndarray
    smooth_y: np.ndarray
    sampled_x: np.ndarray
    sampled_y: np.ndarray


def load_image(image_path: Path) -> ImageData:
    if not image_path.exists():
        raise FileNotFoundError(f"No se encontro la imagen: {image_path}")

    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    gray = np.asarray(image.convert("L"), dtype=np.float64)
    return ImageData(path=image_path, rgb=rgb, gray=gray)


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


def smooth_contour(y_values: np.ndarray, window_size: int = 5) -> np.ndarray:
    window_size = max(3, int(window_size))
    if window_size % 2 == 0:
        window_size += 1
    return ndimage.median_filter(y_values, size=window_size, mode="nearest")


def sample_contour(x_values: np.ndarray, y_values: np.ndarray, step: int = 8) -> tuple[np.ndarray, np.ndarray]:
    if x_values.size == 0:
        raise ValueError("No hay puntos para muestrear.")

    step = max(1, int(step))
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


def build_contour_series(
    label: str,
    raw_x: np.ndarray,
    raw_y: np.ndarray,
    sample_step: int,
    smooth_window: int = 5,
) -> ContourSeries:
    smooth_y = smooth_contour(raw_y, window_size=smooth_window)
    sampled_x, sampled_y = sample_contour(raw_x, smooth_y, step=sample_step)
    return ContourSeries(
        label=label,
        raw_x=raw_x,
        raw_y=raw_y,
        smooth_y=smooth_y,
        sampled_x=sampled_x,
        sampled_y=sampled_y,
    )
