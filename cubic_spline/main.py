from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from image_processing import (
    build_contour_series,
    extract_contours,
    load_image,
    segment_subject,
)
from plotting import (
    plot_final_overlay,
    plot_grabcut_diagnostic,
    plot_linear_vs_spline,
    plot_mask_and_silhouette,
    plot_original_image,
    plot_raw_contours,
    plot_sampled_points,
    plot_spline_vs_points,
    save_figure,
)
from spline import SplineResult, fit_natural_cubic_spline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detecta la silueta principal de una imagen y la interpola con splines cubicos naturales."
    )
    parser.add_argument(
        "--input",
        default="img/G6.jpg",
        help="Ruta de la imagen de entrada. Por defecto usa img/G6.jpg.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directorio donde se guardaran datos y graficos.")
    parser.add_argument(
        "--grabcut-iters",
        type=int,
        default=5,
        help="Numero de iteraciones de GrabCut para estimar la silueta.",
    )
    parser.add_argument(
        "--grabcut-margin",
        type=float,
        default=0.05,
        help="Margen relativo que se deja libre en los bordes para inicializar el rectangulo de GrabCut.",
    )
    parser.add_argument("--sample-step", type=int, default=8, help="Separacion horizontal minima entre puntos muestreados del contorno.")
    parser.add_argument(
        "--compare-scipy",
        action="store_true",
        help="Compara el spline manual con scipy.interpolate.CubicSpline.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Muestra las figuras ademas de guardarlas en disco.",
    )
    return parser.parse_args()


def save_xy_csv(output_path: Path, x_values: np.ndarray, y_values: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((x_values, y_values))
    np.savetxt(output_path, data, delimiter=",", header="x,y", comments="")


def validate_with_scipy(spline_result: SplineResult) -> float | None:
    try:
        from scipy.interpolate import CubicSpline
    except ImportError:
        return None

    scipy_spline = CubicSpline(
        spline_result.x_nodes,
        spline_result.y_nodes,
        bc_type="natural",
    )
    scipy_values = scipy_spline(spline_result.dense_x)
    return float(np.max(np.abs(scipy_values - spline_result.dense_y)))


def build_output_paths(base_output_dir: Path) -> tuple[Path, Path]:
    plots_dir = base_output_dir / "plots"
    data_dir = base_output_dir / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, data_dir


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    image_data = load_image(input_path)
    segmentation = segment_subject(
        image_data.rgb,
        grabcut_iters=args.grabcut_iters,
        grabcut_margin=args.grabcut_margin,
    )
    plots_dir, data_dir = build_output_paths(output_dir)

    upper_x, upper_y, lower_x, lower_y = extract_contours(segmentation.final_mask)
    upper_contour = build_contour_series("superior", upper_x, upper_y, sample_step=args.sample_step)
    lower_contour = build_contour_series("inferior", lower_x, lower_y, sample_step=args.sample_step)

    if upper_contour.sampled_x.size < 3 or lower_contour.sampled_x.size < 3:
        raise ValueError("Al menos uno de los contornos no tiene suficientes puntos para interpolar.")

    upper_spline = fit_natural_cubic_spline(upper_contour.sampled_x, upper_contour.sampled_y)
    lower_spline = fit_natural_cubic_spline(lower_contour.sampled_x, lower_contour.sampled_y)

    save_xy_csv(data_dir / "upper_raw.csv", upper_contour.raw_x, upper_contour.raw_y)
    save_xy_csv(data_dir / "lower_raw.csv", lower_contour.raw_x, lower_contour.raw_y)
    save_xy_csv(data_dir / "upper_sampled.csv", upper_contour.sampled_x, upper_contour.sampled_y)
    save_xy_csv(data_dir / "lower_sampled.csv", lower_contour.sampled_x, lower_contour.sampled_y)
    save_xy_csv(data_dir / "upper_spline.csv", upper_spline.dense_x, upper_spline.dense_y)
    save_xy_csv(data_dir / "lower_spline.csv", lower_spline.dense_x, lower_spline.dense_y)

    figures = [
        ("01_imagen_original.png", plot_original_image(image_data.rgb)),
        ("02_diferencia_fondo.png", plot_grabcut_diagnostic(image_data.gray, segmentation.grabcut_labels)),
        ("03_mascara_y_silueta.png", plot_mask_and_silhouette(segmentation.raw_mask, segmentation.final_mask)),
        ("04_contornos_crudos.png", plot_raw_contours(segmentation.final_mask, upper_contour, lower_contour)),
        ("05_puntos_muestreados.png", plot_sampled_points(upper_contour, lower_contour)),
        ("06_spline_vs_puntos.png", plot_spline_vs_points(upper_contour, lower_contour, upper_spline, lower_spline)),
        ("07_superposicion_final.png", plot_final_overlay(image_data.rgb, upper_spline, lower_spline)),
        ("08_lineal_vs_spline.png", plot_linear_vs_spline(upper_contour, lower_contour, upper_spline, lower_spline)),
    ]

    for filename, figure in figures:
        save_figure(figure, plots_dir / filename, show=args.show)

    print("Proceso completado correctamente.")
    print(f"Imagen procesada: {input_path}")
    print(f"Directorio de salida: {output_dir.resolve()}")
    print(f"Iteraciones de GrabCut: {args.grabcut_iters}")
    print(f"Margen relativo de GrabCut: {args.grabcut_margin:.3f}")
    print(f"Area del contorno principal: {segmentation.contour_area:.2f}")
    print(f"Columnas validas detectadas: {upper_contour.raw_x.size}")
    print(f"Puntos muestreados contorno superior: {upper_contour.sampled_x.size}")
    print(f"Puntos muestreados contorno inferior: {lower_contour.sampled_x.size}")

    if args.compare_scipy:
        upper_diff = validate_with_scipy(upper_spline)
        lower_diff = validate_with_scipy(lower_spline)
        if upper_diff is None or lower_diff is None:
            print("No fue posible comparar con SciPy porque scipy.interpolate no esta disponible.")
        else:
            print(f"Error maximo spline superior vs SciPy: {upper_diff:.8f}")
            print(f"Error maximo spline inferior vs SciPy: {lower_diff:.8f}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ImportError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
