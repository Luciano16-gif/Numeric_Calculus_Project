from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .image_processing import (
        build_auto_oriented_contour_series,
        build_bridge_contour_series,
        build_contour_series,
        build_refined_contour_series,
        extract_contours,
        extract_lower_outline,
        extract_upper_outline,
        load_control_points,
        load_image,
        segment_subject,
    )
    from .plotting import (
        plot_book_style_superior,
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
    from .spline import SplineResult, fit_natural_cubic_spline
except ImportError:
    from image_processing import (
        build_auto_oriented_contour_series,
        build_bridge_contour_series,
        build_contour_series,
        build_refined_contour_series,
        extract_contours,
        extract_lower_outline,
        extract_upper_outline,
        load_control_points,
        load_image,
        segment_subject,
    )
    from plotting import (
        plot_book_style_superior,
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


PACKAGE_DIR = Path(__file__).resolve().parent
G6_POINTS_PATH = PACKAGE_DIR / "data" / "g6_upper_points.csv"
G6_SPECIAL_OUTPUT_DIR = Path("output_g6")


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
    parser.add_argument("--sample-step", type=int, default=4, help="Separacion horizontal minima entre puntos muestreados del contorno.")
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
    args = parser.parse_args()
    if args.sample_step < 1:
        parser.error("--sample-step debe ser un entero positivo.")
    return args


def save_xy_csv(output_path: Path, x_values: np.ndarray, y_values: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack((x_values, y_values))
    np.savetxt(output_path, data, delimiter=",", header="x,y", comments="")


def save_spline_coefficients_csv(output_path: Path, spline_result: SplineResult) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_path,
        spline_result.interval_coefficients,
        delimiter=",",
        header="x_left,x_right,a,b,c,d",
        comments="",
    )


def validate_with_scipy(spline_result: SplineResult) -> float | None:
    try:
        from scipy.interpolate import CubicSpline
    except ImportError:
        return None

    scipy_spline = CubicSpline(
        spline_result.parameter_nodes,
        spline_result.value_nodes,
        bc_type="natural",
    )
    scipy_values = scipy_spline(spline_result.dense_parameter)
    return float(np.max(np.abs(scipy_values - spline_result.dense_values)))


def build_output_paths(base_output_dir: Path) -> tuple[Path, Path]:
    plots_dir = base_output_dir / "plots"
    data_dir = base_output_dir / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir, data_dir


def _as_list(item):
    if isinstance(item, list):
        return item
    if isinstance(item, tuple):
        return list(item)
    return [item]


def _cleanup_contour_data_files(data_dir: Path, prefix: str, segmented: bool) -> None:
    standard_files = (
        f"{prefix}_raw.csv",
        f"{prefix}_sampled.csv",
        f"{prefix}_spline.csv",
        f"{prefix}_piecewise_coefficients.csv",
    )
    segmented_patterns = (
        f"{prefix}_segment_*_raw.csv",
        f"{prefix}_segment_*_sampled.csv",
        f"{prefix}_segment_*_spline.csv",
        f"{prefix}_segment_*_piecewise_coefficients.csv",
    )

    if segmented:
        for filename in standard_files:
            file_path = data_dir / filename
            if file_path.exists():
                file_path.unlink()
        return

    for pattern in segmented_patterns:
        for file_path in data_dir.glob(pattern):
            file_path.unlink()


def save_bundle_outputs(
    output_dir: Path,
    image_data,
    segmentation,
    upper_contour,
    lower_contour,
    upper_spline: SplineResult,
    lower_spline,
    show: bool,
    extra_figures: list[tuple[str, plt.Figure]] | None = None,
    extra_data: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> None:
    plots_dir, data_dir = build_output_paths(output_dir)
    upper_contours = _as_list(upper_contour)
    upper_splines = _as_list(upper_spline)
    lower_contours = _as_list(lower_contour)
    lower_splines = _as_list(lower_spline)
    segmented_upper = len(upper_contours) > 1 or len(upper_splines) > 1
    segmented_lower = len(lower_contours) > 1 or len(lower_splines) > 1

    _cleanup_contour_data_files(data_dir, prefix="upper", segmented=segmented_upper)
    _cleanup_contour_data_files(data_dir, prefix="lower", segmented=segmented_lower)

    if len(upper_contours) == 1 and len(upper_splines) == 1:
        save_xy_csv(data_dir / "upper_raw.csv", upper_contours[0].raw_x, upper_contours[0].raw_y)
        save_xy_csv(data_dir / "upper_sampled.csv", upper_contours[0].sampled_x, upper_contours[0].sampled_y)
        save_xy_csv(data_dir / "upper_spline.csv", upper_splines[0].dense_x, upper_splines[0].dense_y)
        save_spline_coefficients_csv(data_dir / "upper_piecewise_coefficients.csv", upper_splines[0])
    else:
        for index, (segment, spline_segment) in enumerate(zip(upper_contours, upper_splines), start=1):
            save_xy_csv(data_dir / f"upper_segment_{index}_raw.csv", segment.raw_x, segment.raw_y)
            save_xy_csv(data_dir / f"upper_segment_{index}_sampled.csv", segment.sampled_x, segment.sampled_y)
            save_xy_csv(data_dir / f"upper_segment_{index}_spline.csv", spline_segment.dense_x, spline_segment.dense_y)
            save_spline_coefficients_csv(
                data_dir / f"upper_segment_{index}_piecewise_coefficients.csv",
                spline_segment,
            )

    if len(lower_contours) == 1 and len(lower_splines) == 1:
        save_xy_csv(data_dir / "lower_raw.csv", lower_contours[0].raw_x, lower_contours[0].raw_y)
        save_xy_csv(data_dir / "lower_sampled.csv", lower_contours[0].sampled_x, lower_contours[0].sampled_y)
        save_xy_csv(data_dir / "lower_spline.csv", lower_splines[0].dense_x, lower_splines[0].dense_y)
        save_spline_coefficients_csv(data_dir / "lower_piecewise_coefficients.csv", lower_splines[0])
    else:
        for index, (segment, spline_segment) in enumerate(zip(lower_contours, lower_splines), start=1):
            save_xy_csv(data_dir / f"lower_segment_{index}_raw.csv", segment.raw_x, segment.raw_y)
            save_xy_csv(data_dir / f"lower_segment_{index}_sampled.csv", segment.sampled_x, segment.sampled_y)
            save_xy_csv(data_dir / f"lower_segment_{index}_spline.csv", spline_segment.dense_x, spline_segment.dense_y)
            save_spline_coefficients_csv(
                data_dir / f"lower_segment_{index}_piecewise_coefficients.csv",
                spline_segment,
            )

    if extra_data:
        for filename, x_values, y_values in extra_data:
            save_xy_csv(data_dir / filename, x_values, y_values)

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
    if extra_figures:
        figures.extend(extra_figures)

    for filename, figure in figures:
        save_figure(figure, plots_dir / filename, show=show)


def print_summary(
    label: str,
    output_dir: Path,
    segmentation,
    upper_contour,
    lower_contour,
    upper_spline: SplineResult,
    lower_spline,
    compare_scipy: bool,
) -> None:
    upper_contours = _as_list(upper_contour)
    upper_splines = _as_list(upper_spline)
    lower_contours = _as_list(lower_contour)
    lower_splines = _as_list(lower_spline)

    print(f"{label}: {output_dir.resolve()}")
    print(f"Area del contorno principal: {segmentation.contour_area:.2f}")
    print(f"Puntos crudos contorno superior: {sum(segment.raw_x.size for segment in upper_contours)}")
    print(f"Puntos usados contorno superior: {sum(segment.sampled_x.size for segment in upper_contours)}")
    print(f"Puntos usados contorno inferior: {sum(segment.sampled_x.size for segment in lower_contours)}")
    print(
        "Funciones por tramos del spline superior: "
        f"{sum(segment.interval_coefficients.shape[0] for segment in upper_splines)}"
    )
    print(
        "Funciones por tramos del spline inferior: "
        f"{sum(segment.interval_coefficients.shape[0] for segment in lower_splines)}"
    )
    if len(upper_contours) > 1:
        print(f"Tramos superiores: {len(upper_contours)}")
    if len(lower_contours) > 1:
        print(f"Tramos inferiores: {len(lower_contours)}")

    if compare_scipy:
        upper_diffs = [validate_with_scipy(spline_segment) for spline_segment in upper_splines]
        lower_diffs = [validate_with_scipy(spline_segment) for spline_segment in lower_splines]
        upper_diff = None if any(diff is None for diff in upper_diffs) else float(max(upper_diffs))
        lower_diff = None if any(diff is None for diff in lower_diffs) else float(max(lower_diffs))
        if upper_diff is None or lower_diff is None:
            print("No fue posible comparar con SciPy porque scipy.interpolate no esta disponible.")
        else:
            print(f"Error maximo spline superior vs SciPy: {upper_diff:.8f}")
            print(f"Error maximo spline inferior vs SciPy: {lower_diff:.8f}")


def is_g6_image(input_path: Path) -> bool:
    return input_path.name.lower() == "g6.jpg"


def fit_spline_segments(contours) -> list[SplineResult]:
    spline_segments: list[SplineResult] = []
    for segment in contours:
        if segment.parameter_axis == "x":
            spline_segments.append(
                fit_natural_cubic_spline(
                    segment.sampled_x,
                    segment.sampled_y,
                    parameter_axis="x",
                )
            )
        else:
            spline_segments.append(
                fit_natural_cubic_spline(
                    segment.sampled_y,
                    segment.sampled_x,
                    parameter_axis="y",
                )
            )
    return spline_segments


def _endpoint_key(start_point: np.ndarray, end_point: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    start = (int(round(float(start_point[0]))), int(round(float(start_point[1]))))
    end = (int(round(float(end_point[0]))), int(round(float(end_point[1]))))
    return tuple(sorted((start, end)))


def build_bridge_segments(
    contours,
    max_gap: float = 50.0,
    max_horizontal_gap: float = 12.0,
) -> list:
    bridges = []
    seen_keys = set()
    for index in range(len(contours) - 1):
        current = contours[index]
        following = contours[index + 1]
        endpoint_pairs = [
            (
                np.array([current.sampled_x[-1], current.sampled_y[-1]], dtype=np.float64),
                np.array([following.sampled_x[0], following.sampled_y[0]], dtype=np.float64),
                max_gap,
                max_horizontal_gap,
            ),
            (
                np.array([current.sampled_x[0], current.sampled_y[0]], dtype=np.float64),
                np.array([following.sampled_x[0], following.sampled_y[0]], dtype=np.float64),
                55.0,
                1.0,
            ),
        ]

        for pair_start, pair_end, pair_max_gap, pair_max_horizontal_gap in endpoint_pairs:
            distance = float(np.linalg.norm(pair_end - pair_start))
            horizontal_gap = float(abs(pair_end[0] - pair_start[0]))
            if (
                distance < 6.0
                or distance > pair_max_gap
                or horizontal_gap > pair_max_horizontal_gap
            ):
                continue
            key = _endpoint_key(pair_start, pair_end)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            bridges.append(
                build_bridge_contour_series(
                    label=f"puente contorno G6 {index + 1}",
                    start_point=pair_start,
                    end_point=pair_end,
                )
            )
    return bridges


def build_silhouette_endpoint_bridges(
    upper_contour,
    lower_contours,
    max_gap: float = 20.0,
    max_horizontal_gap: float = 10.0,
) -> list:
    if not lower_contours:
        return []

    upper_start = np.array([upper_contour.sampled_x[0], upper_contour.sampled_y[0]], dtype=np.float64)
    upper_end = np.array([upper_contour.sampled_x[-1], upper_contour.sampled_y[-1]], dtype=np.float64)
    lower_endpoints = []

    for segment in lower_contours:
        lower_endpoints.append(np.array([segment.sampled_x[0], segment.sampled_y[0]], dtype=np.float64))
        lower_endpoints.append(np.array([segment.sampled_x[-1], segment.sampled_y[-1]], dtype=np.float64))

    bridges = []
    seen_keys = set()
    for label, upper_point in (("inicio", upper_start), ("fin", upper_end)):
        best_point = None
        best_distance = None
        for endpoint in lower_endpoints:
            distance = float(np.linalg.norm(endpoint - upper_point))
            horizontal_gap = float(abs(endpoint[0] - upper_point[0]))
            if (
                distance < 1.0
                or distance > max_gap
                or horizontal_gap > max_horizontal_gap
            ):
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_point = endpoint

        if best_point is None:
            continue
        key = _endpoint_key(upper_point, best_point)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        bridges.append(
            build_bridge_contour_series(
                label=f"puente borde superior G6 {label}",
                start_point=upper_point,
                end_point=best_point,
            )
        )
    return bridges


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

    upper_x, upper_y, lower_x, lower_y = extract_contours(segmentation.final_mask)
    automatic_upper = build_contour_series("superior", upper_x, upper_y, sample_step=args.sample_step)
    automatic_lower = build_contour_series("inferior", lower_x, lower_y, sample_step=args.sample_step)

    if automatic_upper.sampled_x.size < 3 or automatic_lower.sampled_x.size < 3:
        raise ValueError("Al menos uno de los contornos generales no tiene suficientes puntos para interpolar.")

    automatic_upper_spline = fit_natural_cubic_spline(automatic_upper.sampled_x, automatic_upper.sampled_y)
    automatic_lower_spline = fit_natural_cubic_spline(automatic_lower.sampled_x, automatic_lower.sampled_y)

    save_bundle_outputs(
        output_dir=output_dir,
        image_data=image_data,
        segmentation=segmentation,
        upper_contour=automatic_upper,
        lower_contour=automatic_lower,
        upper_spline=automatic_upper_spline,
        lower_spline=automatic_lower_spline,
        show=args.show,
    )

    print("Proceso completado correctamente.")
    print(f"Imagen procesada: {input_path}")
    print(f"Iteraciones de GrabCut: {args.grabcut_iters}")
    print(f"Margen relativo de GrabCut: {args.grabcut_margin:.3f}")
    print_summary(
        label="Salida general",
        output_dir=output_dir,
        segmentation=segmentation,
        upper_contour=automatic_upper,
        lower_contour=automatic_lower,
        upper_spline=automatic_upper_spline,
        lower_spline=automatic_lower_spline,
        compare_scipy=args.compare_scipy,
    )

    if is_g6_image(input_path):
        book_x, book_y = load_control_points(G6_POINTS_PATH)
        book_upper = build_refined_contour_series("superior tipo libro G6", book_x, book_y)
        book_upper_spline = fit_natural_cubic_spline(
            book_upper.sampled_x,
            book_upper.sampled_y,
            parameter_axis="x",
        )
        upper_outline_segments_g6 = extract_upper_outline(segmentation.final_mask)
        refined_upper_segments = [
            build_auto_oriented_contour_series(
                f"superior G6 tramo {i+1}",
                sx,
                sy,
                sample_step=1,
                smooth_window=1,
            )
            for i, (sx, sy) in enumerate(upper_outline_segments_g6)
            if sx.size >= 2
        ]
        refined_upper_segments.extend(build_bridge_segments(refined_upper_segments))
        refined_upper_splines = fit_spline_segments(refined_upper_segments)

        outline_segments_g6 = extract_lower_outline(segmentation.final_mask)
        refined_lower_segments = [
            build_auto_oriented_contour_series(
                f"inferior G6 tramo {i+1}",
                sx,
                sy,
                sample_step=1,
                smooth_window=1,
            )
            for i, (sx, sy) in enumerate(outline_segments_g6)
            if sx.size >= 2
        ]
        refined_lower_segments.extend(build_bridge_segments(refined_lower_segments))
        refined_lower_splines = fit_spline_segments(refined_lower_segments)

        g6_extra_figures = [
            (
                "09_g6_superior_tipo_libro.png",
                plot_book_style_superior(
                    upper_contour=book_upper,
                    upper_spline=book_upper_spline,
                    image_height=image_data.rgb.shape[0],
                ),
            )
        ]
        g6_extra_data = [
            ("upper_refined_points.csv", book_upper.sampled_x, book_upper.sampled_y),
            ("upper_refined_spline.csv", book_upper_spline.dense_x, book_upper_spline.dense_y),
        ]

        save_bundle_outputs(
            output_dir=G6_SPECIAL_OUTPUT_DIR,
            image_data=image_data,
            segmentation=segmentation,
            upper_contour=refined_upper_segments,
            lower_contour=refined_lower_segments,
            upper_spline=refined_upper_splines,
            lower_spline=refined_lower_splines,
            show=args.show,
            extra_figures=g6_extra_figures,
            extra_data=g6_extra_data,
        )

        print_summary(
            label="Salida refinada G6",
            output_dir=G6_SPECIAL_OUTPUT_DIR,
            segmentation=segmentation,
            upper_contour=refined_upper_segments,
            lower_contour=refined_lower_segments,
            upper_spline=refined_upper_splines,
            lower_spline=refined_lower_splines,
            compare_scipy=args.compare_scipy,
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ImportError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
