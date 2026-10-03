"""
postprocess.py

This module contains CSV export and plotting utilities for the transient
2D stepped-domain heat conduction project.

Outputs:
- Long-format CSV files: x, y, temperature
- Summary CSV files
- Temperature contour plots
- Heatmap plots
- 1D temperature profile plots
- Comparison plots for parametric studies

Author: AERO 242 Project
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from geometry import Grid
from solver import SimulationResult


def ensure_output_directory(output_directory: str) -> Path:
    """
    Create output directory if it does not exist.

    Parameters
    ----------
    output_directory : str
        Output directory path.

    Returns
    -------
    pathlib.Path
        Path object for the output directory.
    """

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path


def sanitize_name(name: str) -> str:
    """
    Convert a case name into a safe filename component.

    Parameters
    ----------
    name : str
        Input name.

    Returns
    -------
    str
        Safe filename string.
    """

    safe_characters = []

    for character in name.strip().lower():
        if character.isalnum():
            safe_characters.append(character)
        elif character in ["_", "-", "."]:
            safe_characters.append(character)
        else:
            safe_characters.append("_")

    safe_name = "".join(safe_characters)

    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")

    return safe_name.strip("_")


def format_float_for_filename(value: float) -> str:
    """
    Format a float so it is safe for filenames.

    Examples
    --------
    0.125 -> 0p125
    5000.0 -> 5000
    """

    if abs(value - round(value)) < 1.0e-12:
        return str(int(round(value)))

    return f"{value:g}".replace(".", "p").replace("-", "m")


def format_time_for_filename(time_value: float) -> str:
    """
    Format time value for filenames.

    Parameters
    ----------
    time_value : float
        Time in seconds.

    Returns
    -------
    str
        Formatted time label.
    """

    return format_float_for_filename(time_value)


def get_case_output_directory(
    result: SimulationResult,
    output_directory: Optional[str] = None,
) -> Path:
    """
    Return output directory for a result.

    Parameters
    ----------
    result : SimulationResult
        Simulation result.
    output_directory : str, optional
        Output directory override.

    Returns
    -------
    pathlib.Path
        Output path.
    """

    if output_directory is None:
        output_directory = result.parameters.simulation.output_directory

    return ensure_output_directory(output_directory)


def find_saved_time(
    result: SimulationResult,
    requested_time: float,
    tolerance: float = 1.0e-9,
) -> float:
    """
    Find a saved time key that matches the requested time.

    Parameters
    ----------
    result : SimulationResult
        Simulation result.
    requested_time : float
        Requested saved time.
    tolerance : float
        Time comparison tolerance.

    Returns
    -------
    float
        Matching saved time key.

    Raises
    ------
    KeyError
        If requested time is not saved.
    """

    for saved_time in result.saved_temperature_fields.keys():
        if abs(saved_time - requested_time) <= tolerance:
            return saved_time

    available_times = sorted(result.saved_temperature_fields.keys())

    raise KeyError(
        f"Requested time {requested_time} s was not saved. "
        f"Available times are: {available_times}"
    )


def temperature_field_to_dataframe(
    grid: Grid,
    temperature_field: np.ndarray,
) -> pd.DataFrame:
    """
    Convert a 2D temperature field into long-format dataframe.

    Inactive nodes are ignored because they are stored as NaN.

    Parameters
    ----------
    grid : Grid
        Grid object.
    temperature_field : np.ndarray
        2D temperature field with NaN inactive nodes.

    Returns
    -------
    pandas.DataFrame
        Columns: x, y, temperature
    """

    if temperature_field.shape != grid.shape:
        raise ValueError(
            "Temperature field shape does not match grid shape. "
            f"temperature_field.shape = {temperature_field.shape}, "
            f"grid.shape = {grid.shape}"
        )

    active_mask = ~np.isnan(temperature_field)

    dataframe = pd.DataFrame(
        {
            "x": grid.x_mesh[active_mask].ravel(),
            "y": grid.y_mesh[active_mask].ravel(),
            "temperature": temperature_field[active_mask].ravel(),
        }
    )

    dataframe = dataframe.sort_values(["y", "x"]).reset_index(drop=True)

    return dataframe


def export_temperature_csv(
    result: SimulationResult,
    save_time: float,
    output_directory: Optional[str] = None,
) -> Path:
    """
    Export one saved temperature field as long-format CSV.

    CSV columns:
    x, y, temperature

    Parameters
    ----------
    result : SimulationResult
        Simulation result.
    save_time : float
        Saved time to export.
    output_directory : str, optional
        Output directory override.

    Returns
    -------
    pathlib.Path
        Written CSV path.
    """

    output_path = get_case_output_directory(result, output_directory)

    matching_time = find_saved_time(result, save_time)
    temperature_field = result.saved_temperature_fields[matching_time]

    dataframe = temperature_field_to_dataframe(
        grid=result.grid,
        temperature_field=temperature_field,
    )

    case_name = sanitize_name(result.case_name)
    time_label = format_time_for_filename(matching_time)

    file_path = output_path / f"{case_name}_t{time_label}.csv"

    dataframe.to_csv(file_path, index=False)

    return file_path


def export_all_temperature_csvs(
    result: SimulationResult,
    output_directory: Optional[str] = None,
) -> List[Path]:
    """
    Export all saved temperature fields to CSV.

    Parameters
    ----------
    result : SimulationResult
        Simulation result.
    output_directory : str, optional
        Output directory override.

    Returns
    -------
    list of pathlib.Path
        Written CSV paths.
    """

    written_files = []

    for save_time in sorted(result.saved_temperature_fields.keys()):
        written_file = export_temperature_csv(
            result=result,
            save_time=save_time,
            output_directory=output_directory,
        )
        written_files.append(written_file)

    return written_files


def create_summary_row(result: SimulationResult) -> Dict[str, float]:
    """
    Create one summary row for a simulation result.

    Parameters
    ----------
    result : SimulationResult
        Simulation result.

    Returns
    -------
    dict
        Summary row data.
    """

    final_time = result.parameters.simulation.t_final
    matching_final_time = find_saved_time(result, final_time)

    final_vector = result.saved_temperature_vectors[matching_final_time]

    row = {
        "case_name": result.case_name,
        "material_name": result.parameters.material.name,
        "k": result.parameters.material.thermal_conductivity,
        "rho": result.parameters.material.density,
        "cp": result.parameters.material.specific_heat,
        "alpha": result.parameters.material.thermal_diffusivity,
        "dx": result.parameters.simulation.dx,
        "dy": result.parameters.simulation.dy,
        "dt": result.parameters.simulation.dt,
        "t_final": result.parameters.simulation.t_final,
        "min_temperature": float(np.min(final_vector)),
        "max_temperature": float(np.max(final_vector)),
        "average_temperature": float(np.mean(final_vector)),
        "active_node_count": result.grid.active_node_count,
    }

    return row


def export_summary_csv(
    results: Sequence[SimulationResult],
    output_directory: str,
    filename: str = "summary.csv",
) -> Path:
    """
    Export summary CSV for one or more results.

    Parameters
    ----------
    results : sequence of SimulationResult
        Simulation results.
    output_directory : str
        Output directory.
    filename : str
        Summary CSV filename.

    Returns
    -------
    pathlib.Path
        Written CSV path.
    """

    output_path = ensure_output_directory(output_directory)

    rows = [create_summary_row(result) for result in results]

    dataframe = pd.DataFrame(rows)

    file_path = output_path / filename
    dataframe.to_csv(file_path, index=False)

    return file_path


def save_figure(
    figure: plt.Figure,
    file_path: Path,
    dpi: int = 300,
) -> Path:
    """
    Save a matplotlib figure and close it.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure object.
    file_path : pathlib.Path
        Output file path.
    dpi : int
        Plot resolution.

    Returns
    -------
    pathlib.Path
        Written file path.
    """

    figure.tight_layout()
    figure.savefig(file_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    return file_path


def get_masked_temperature_field(
    result: SimulationResult,
    save_time: float,
) -> Tuple[float, np.ma.MaskedArray]:
    """
    Return masked temperature field for plotting.

    Parameters
    ----------
    result : SimulationResult
        Simulation result.
    save_time : float
        Saved time.

    Returns
    -------
    tuple
        matching_time, masked_temperature_field
    """

    matching_time = find_saved_time(result, save_time)
    temperature_field = result.saved_temperature_fields[matching_time]

    masked_temperature = np.ma.masked_invalid(temperature_field)

    return matching_time, masked_temperature


def plot_temperature_contour(
    result: SimulationResult,
    save_time: float,
    output_directory: Optional[str] = None,
    levels: int = 20,
    show: bool = False,
) -> Path:
    """
    Plot temperature contour for a saved time.

    Parameters
    ----------
    result : SimulationResult
        Simulation result.
    save_time : float
        Saved time.
    output_directory : str, optional
        Output directory override.
    levels : int
        Number of contour levels.
    show : bool
        If True, display plot interactively.

    Returns
    -------
    pathlib.Path
        Written image path.
    """

    output_path = get_case_output_directory(result, output_directory)

    matching_time, masked_temperature = get_masked_temperature_field(
        result=result,
        save_time=save_time,
    )

    figure, axis = plt.subplots(figsize=(8.0, 5.0))

    contour = axis.contourf(
        result.grid.x_mesh,
        result.grid.y_mesh,
        masked_temperature,
        levels=levels,
    )

    contour_lines = axis.contour(
        result.grid.x_mesh,
        result.grid.y_mesh,
        masked_temperature,
        levels=levels,
        linewidths=0.4,
    )

    axis.clabel(contour_lines, inline=True, fontsize=7)

    colorbar = figure.colorbar(contour, ax=axis)
    colorbar.set_label("Temperature [°C]")

    axis.set_title(
        f"{result.case_name} Temperature Contour at t = {matching_time:g} s"
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linewidth=0.3)

    case_name = sanitize_name(result.case_name)
    time_label = format_time_for_filename(matching_time)

    file_path = output_path / f"{case_name}_contour_t{time_label}.png"

    if show:
        plt.show()

    return save_figure(figure, file_path)


def plot_temperature_heatmap(
    result: SimulationResult,
    save_time: float,
    output_directory: Optional[str] = None,
    show: bool = False,
) -> Path:
    """
    Plot temperature heatmap for a saved time.

    Parameters
    ----------
    result : SimulationResult
        Simulation result.
    save_time : float
        Saved time.
    output_directory : str, optional
        Output directory override.
    show : bool
        If True, display plot interactively.

    Returns
    -------
    pathlib.Path
        Written image path.
    """

    output_path = get_case_output_directory(result, output_directory)

    matching_time, masked_temperature = get_masked_temperature_field(
        result=result,
        save_time=save_time,
    )

    figure, axis = plt.subplots(figsize=(8.0, 5.0))

    heatmap = axis.pcolormesh(
        result.grid.x_mesh,
        result.grid.y_mesh,
        masked_temperature,
        shading="auto",
    )

    colorbar = figure.colorbar(heatmap, ax=axis)
    colorbar.set_label("Temperature [°C]")

    axis.set_title(
        f"{result.case_name} Temperature Heatmap at t = {matching_time:g} s"
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linewidth=0.3)

    case_name = sanitize_name(result.case_name)
    time_label = format_time_for_filename(matching_time)

    file_path = output_path / f"{case_name}_heatmap_t{time_label}.png"

    if show:
        plt.show()

    return save_figure(figure, file_path)


def export_profile_csv(
    result: SimulationResult,
    save_time: float,
    profile_y: float = 0.5,
    output_directory: Optional[str] = None,
) -> Path:
    """
    Export 1D temperature profile along a horizontal line y = profile_y.

    Parameters
    ----------
    result : SimulationResult
        Simulation result.
    save_time : float
        Saved time.
    profile_y : float
        Requested horizontal line location.
    output_directory : str, optional
        Output directory override.

    Returns
    -------
    pathlib.Path
        Written CSV path.
    """

    output_path = get_case_output_directory(result, output_directory)

    matching_time, x_values, temperature_values, actual_y = extract_horizontal_profile(
        result=result,
        save_time=save_time,
        profile_y=profile_y,
    )

    dataframe = pd.DataFrame(
        {
            "x": x_values,
            "y": np.full_like(x_values, actual_y, dtype=float),
            "temperature": temperature_values,
        }
    )

    case_name = sanitize_name(result.case_name)
    time_label = format_time_for_filename(matching_time)
    y_label = format_float_for_filename(actual_y)

    file_path = output_path / f"{case_name}_profile_y{y_label}_t{time_label}.csv"

    dataframe.to_csv(file_path, index=False)

    return file_path


def extract_horizontal_profile(
    result: SimulationResult,
    save_time: float,
    profile_y: float = 0.5,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Extract temperature profile along the nearest available horizontal grid line.

    Parameters
    ----------
    result : SimulationResult
        Simulation result.
    save_time : float
        Saved time.
    profile_y : float
        Requested y-location.

    Returns
    -------
    tuple
        matching_time, x_values, temperature_values, actual_y
    """

    matching_time = find_saved_time(result, save_time)
    temperature_field = result.saved_temperature_fields[matching_time]

    y_coordinates = result.grid.y_coordinates

    nearest_y_index = int(np.argmin(np.abs(y_coordinates - profile_y)))
    actual_y = float(y_coordinates[nearest_y_index])

    profile_temperatures = temperature_field[nearest_y_index, :]
    active_mask = ~np.isnan(profile_temperatures)

    x_values = result.grid.x_coordinates[active_mask]
    temperature_values = profile_temperatures[active_mask]

    return matching_time, x_values, temperature_values, actual_y


def plot_horizontal_profile(
    result: SimulationResult,
    save_time: float,
    profile_y: float = 0.5,
    output_directory: Optional[str] = None,
    show: bool = False,
) -> Path:
    """
    Plot 1D temperature profile along a horizontal line.

    Parameters
    ----------
    result : SimulationResult
        Simulation result.
    save_time : float
        Saved time.
    profile_y : float
        Requested y-location.
    output_directory : str, optional
        Output directory override.
    show : bool
        If True, display plot interactively.

    Returns
    -------
    pathlib.Path
        Written image path.
    """

    output_path = get_case_output_directory(result, output_directory)

    matching_time, x_values, temperature_values, actual_y = extract_horizontal_profile(
        result=result,
        save_time=save_time,
        profile_y=profile_y,
    )

    figure, axis = plt.subplots(figsize=(8.0, 5.0))

    axis.plot(x_values, temperature_values, marker="o", linewidth=1.5)

    axis.set_title(
        f"{result.case_name} Temperature Profile at y = {actual_y:g} m, "
        f"t = {matching_time:g} s"
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("Temperature [°C]")
    axis.grid(True, linewidth=0.3)

    case_name = sanitize_name(result.case_name)
    time_label = format_time_for_filename(matching_time)
    y_label = format_float_for_filename(actual_y)

    file_path = output_path / f"{case_name}_profile_y{y_label}_t{time_label}.png"

    if show:
        plt.show()

    return save_figure(figure, file_path)


def plot_temperature_history(
    result: SimulationResult,
    output_directory: Optional[str] = None,
    history_type: str = "maximum",
    show: bool = False,
) -> Path:
    """
    Plot maximum or average temperature versus time.

    Parameters
    ----------
    result : SimulationResult
        Simulation result.
    output_directory : str, optional
        Output directory override.
    history_type : str
        Either "maximum" or "average".
    show : bool
        If True, display plot interactively.

    Returns
    -------
    pathlib.Path
        Written image path.
    """

    output_path = get_case_output_directory(result, output_directory)

    history_type = history_type.lower()

    if history_type == "maximum":
        history = result.maximum_temperature_history
        ylabel = "Maximum Temperature [°C]"
        title_name = "Maximum Temperature"
        filename_part = "maximum_temperature_history"

    elif history_type == "average":
        history = result.average_temperature_history
        ylabel = "Average Temperature [°C]"
        title_name = "Average Temperature"
        filename_part = "average_temperature_history"

    else:
        raise ValueError("history_type must be either 'maximum' or 'average'.")

    time_values = np.array([item[0] for item in history], dtype=float)
    temperature_values = np.array([item[1] for item in history], dtype=float)

    figure, axis = plt.subplots(figsize=(8.0, 5.0))

    axis.plot(time_values, temperature_values, linewidth=1.8)

    axis.set_title(f"{result.case_name} {title_name} versus Time")
    axis.set_xlabel("Time [s]")
    axis.set_ylabel(ylabel)
    axis.grid(True, linewidth=0.3)

    case_name = sanitize_name(result.case_name)

    file_path = output_path / f"{case_name}_{filename_part}.png"

    if show:
        plt.show()

    return save_figure(figure, file_path)


def plot_comparison_profiles(
    results: Sequence[SimulationResult],
    labels: Sequence[str],
    save_time: float,
    profile_y: float,
    output_directory: str,
    filename: str,
    title: str,
    show: bool = False,
) -> Path:
    """
    Plot horizontal temperature profiles from multiple cases on one graph.

    Parameters
    ----------
    results : sequence of SimulationResult
        Results to compare.
    labels : sequence of str
        Plot labels.
    save_time : float
        Saved time to compare.
    profile_y : float
        Horizontal line location.
    output_directory : str
        Output directory.
    filename : str
        Output PNG filename.
    title : str
        Plot title.
    show : bool
        If True, display plot interactively.

    Returns
    -------
    pathlib.Path
        Written image path.
    """

    if len(results) != len(labels):
        raise ValueError("results and labels must have the same length.")

    output_path = ensure_output_directory(output_directory)

    figure, axis = plt.subplots(figsize=(8.0, 5.0))

    actual_y_values = []

    for result, label in zip(results, labels):
        (
            matching_time,
            x_values,
            temperature_values,
            actual_y,
        ) = extract_horizontal_profile(
            result=result,
            save_time=save_time,
            profile_y=profile_y,
        )

        actual_y_values.append(actual_y)

        axis.plot(
            x_values,
            temperature_values,
            marker="o",
            linewidth=1.5,
            label=label,
        )

    average_actual_y = float(np.mean(actual_y_values))

    axis.set_title(
        f"{title}\nProfile near y = {average_actual_y:g} m, "
        f"t = {save_time:g} s"
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("Temperature [°C]")
    axis.grid(True, linewidth=0.3)
    axis.legend()

    file_path = output_path / filename

    if show:
        plt.show()

    return save_figure(figure, file_path)


def plot_case_metric_comparison(
    results: Sequence[SimulationResult],
    labels: Sequence[str],
    metric: str,
    output_directory: str,
    filename: str,
    title: str,
    show: bool = False,
) -> Path:
    """
    Create a simple comparison plot using final-time scalar metrics.

    Valid metrics:
    - "min_temperature"
    - "max_temperature"
    - "average_temperature"

    Parameters
    ----------
    results : sequence of SimulationResult
        Results to compare.
    labels : sequence of str
        Case labels.
    metric : str
        Metric name.
    output_directory : str
        Output directory.
    filename : str
        Output PNG filename.
    title : str
        Plot title.
    show : bool
        If True, display plot interactively.

    Returns
    -------
    pathlib.Path
        Written image path.
    """

    if len(results) != len(labels):
        raise ValueError("results and labels must have the same length.")

    valid_metrics = {
        "min_temperature": "Minimum Temperature [°C]",
        "max_temperature": "Maximum Temperature [°C]",
        "average_temperature": "Average Temperature [°C]",
    }

    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {list(valid_metrics.keys())}")

    output_path = ensure_output_directory(output_directory)

    metric_values = []

    for result in results:
        summary_row = create_summary_row(result)
        metric_values.append(summary_row[metric])

    x_positions = np.arange(len(labels))

    figure, axis = plt.subplots(figsize=(8.0, 5.0))

    axis.plot(x_positions, metric_values, marker="o", linewidth=1.8)

    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=20, ha="right")

    axis.set_title(title)
    axis.set_xlabel("Case")
    axis.set_ylabel(valid_metrics[metric])
    axis.grid(True, linewidth=0.3)

    file_path = output_path / filename

    if show:
        plt.show()

    return save_figure(figure, file_path)


def plot_all_saved_contours(
    result: SimulationResult,
    output_directory: Optional[str] = None,
) -> List[Path]:
    """
    Plot contour figures for all saved times.

    Parameters
    ----------
    result : SimulationResult
        Simulation result.
    output_directory : str, optional
        Output directory override.

    Returns
    -------
    list of pathlib.Path
        Written plot paths.
    """

    written_files = []

    for save_time in sorted(result.saved_temperature_fields.keys()):
        file_path = plot_temperature_contour(
            result=result,
            save_time=save_time,
            output_directory=output_directory,
        )
        written_files.append(file_path)

    return written_files


def plot_all_saved_heatmaps(
    result: SimulationResult,
    output_directory: Optional[str] = None,
) -> List[Path]:
    """
    Plot heatmap figures for all saved times.

    Parameters
    ----------
    result : SimulationResult
        Simulation result.
    output_directory : str, optional
        Output directory override.

    Returns
    -------
    list of pathlib.Path
        Written plot paths.
    """

    written_files = []

    for save_time in sorted(result.saved_temperature_fields.keys()):
        file_path = plot_temperature_heatmap(
            result=result,
            save_time=save_time,
            output_directory=output_directory,
        )
        written_files.append(file_path)

    return written_files


def process_single_result(
    result: SimulationResult,
    output_directory: Optional[str] = None,
    profile_y: float = 0.5,
) -> List[Path]:
    """
    Export standard CSV files and plots for one simulation result.

    This function is useful for baseline and individual parametric cases.

    Generated outputs:
    - CSV files for all saved times
    - Contour plots for all saved times
    - Heatmap plots for all saved times
    - Final horizontal profile CSV
    - Final horizontal profile plot
    - Maximum temperature history plot
    - Average temperature history plot

    Parameters
    ----------
    result : SimulationResult
        Simulation result.
    output_directory : str, optional
        Output directory override.
    profile_y : float
        Horizontal profile location.

    Returns
    -------
    list of pathlib.Path
        All written files.
    """

    if output_directory is None:
        output_directory = result.parameters.simulation.output_directory

    written_files: List[Path] = []

    written_files.extend(
        export_all_temperature_csvs(
            result=result,
            output_directory=output_directory,
        )
    )

    written_files.extend(
        plot_all_saved_contours(
            result=result,
            output_directory=output_directory,
        )
    )

    written_files.extend(
        plot_all_saved_heatmaps(
            result=result,
            output_directory=output_directory,
        )
    )

    final_time = result.parameters.simulation.t_final

    written_files.append(
        export_profile_csv(
            result=result,
            save_time=final_time,
            profile_y=profile_y,
            output_directory=output_directory,
        )
    )

    written_files.append(
        plot_horizontal_profile(
            result=result,
            save_time=final_time,
            profile_y=profile_y,
            output_directory=output_directory,
        )
    )

    written_files.append(
        plot_temperature_history(
            result=result,
            output_directory=output_directory,
            history_type="maximum",
        )
    )

    written_files.append(
        plot_temperature_history(
            result=result,
            output_directory=output_directory,
            history_type="average",
        )
    )

    return written_files


def print_written_files(written_files: Iterable[Path]) -> None:
    """
    Print written output files.

    Parameters
    ----------
    written_files : iterable of pathlib.Path
        Written file paths.
    """

    print("\nWritten output files")
    print("-" * 70)

    for file_path in written_files:
        print(f"  {file_path}")

    print("-" * 70)