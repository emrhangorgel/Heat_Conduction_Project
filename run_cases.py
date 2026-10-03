"""
run_cases.py

This module defines automatic case-running utilities for the AERO 242
transient 2D heat conduction project.

It runs:
1. Baseline Ti-6Al-4V case
2. Thermal conductivity study
3. Grid resolution study
4. Boundary condition studies:
   - h_bottom study
   - q_right study
   - T_inf_bottom study

Author: AERO 242 Project
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from parameters import ProjectParameters, create_baseline_parameters
from postprocess import (
    export_summary_csv,
    format_float_for_filename,
    plot_case_metric_comparison,
    plot_comparison_profiles,
    print_written_files,
    process_single_result,
)
from solver import SimulationResult, run_single_case


@dataclass
class CaseRunner:
    """
    Runs baseline and parametric study cases.

    Attributes
    ----------
    output_directory : str
        Output folder path.
    linear_solver : str
        Linear solver option: "scipy", "gauss_seidel", or "sor".
    verbose : bool
        If True, print detailed solver progress.
    process_outputs : bool
        If True, export CSV files and plots.
    profile_y : float
        y-location for horizontal profile plots.
    baseline_parameters : ProjectParameters
        Baseline parameter object.
    """

    output_directory: str = "outputs"
    linear_solver: str = "scipy"
    verbose: bool = True
    process_outputs: bool = True
    profile_y: float = 0.5
    baseline_parameters: ProjectParameters = field(
        default_factory=create_baseline_parameters
    )

    def create_case_parameters(self) -> ProjectParameters:
        """
        Create a fresh copy of baseline parameters for a new case.

        Returns
        -------
        ProjectParameters
            Deep copy of baseline parameters.
        """

        parameters = deepcopy(self.baseline_parameters)
        parameters.simulation.output_directory = self.output_directory

        return parameters

    def run_case(
        self,
        parameters: ProjectParameters,
        case_name: str,
    ) -> SimulationResult:
        """
        Run one case and optionally process its outputs.

        Parameters
        ----------
        parameters : ProjectParameters
            Case parameters.
        case_name : str
            Case name.

        Returns
        -------
        SimulationResult
            Simulation result.
        """

        print("\n" + "=" * 80)
        print(f"Running case: {case_name}")
        print("=" * 80)

        result = run_single_case(
            parameters=parameters,
            case_name=case_name,
            linear_solver=self.linear_solver,
            verbose=self.verbose,
        )

        if self.process_outputs:
            written_files = process_single_result(
                result=result,
                output_directory=self.output_directory,
                profile_y=self.profile_y,
            )

            if self.verbose:
                print_written_files(written_files)

        return result

    def run_baseline_case(self) -> SimulationResult:
        """
        Run the baseline Ti-6Al-4V case.

        Returns
        -------
        SimulationResult
            Baseline simulation result.
        """

        parameters = self.create_case_parameters()

        parameters.material.thermal_conductivity = 6.7

        parameters.simulation.dx = 0.125
        parameters.simulation.dy = 0.125
        parameters.simulation.dt = 10.0
        parameters.simulation.t_final = 5000.0
        parameters.simulation.save_times = [0.0, 500.0, 1000.0, 2500.0, 5000.0]

        result = self.run_case(
            parameters=parameters,
            case_name="baseline",
        )

        return result

    def run_thermal_conductivity_study(self) -> List[SimulationResult]:
        """
        Run thermal conductivity study.

        Cases:
        k = [5.0, 25.0, 75.0] W/(m*K)

        Returns
        -------
        list of SimulationResult
            Thermal conductivity study results.
        """

        k_values = [5.0, 25.0, 75.0]

        results = []

        for k_value in k_values:
            parameters = self.create_case_parameters()
            parameters.material.thermal_conductivity = k_value

            case_name = f"k_{format_float_for_filename(k_value)}"

            result = self.run_case(
                parameters=parameters,
                case_name=case_name,
            )

            results.append(result)

        self.create_thermal_conductivity_comparison_plots(results, k_values)

        if self.process_outputs:
            export_summary_csv(
                results=results,
                output_directory=self.output_directory,
                filename="k_study_summary.csv",
            )

        return results

    def create_thermal_conductivity_comparison_plots(
        self,
        results: Sequence[SimulationResult],
        k_values: Sequence[float],
    ) -> None:
        """
        Create comparison plots for thermal conductivity study.

        Parameters
        ----------
        results : sequence of SimulationResult
            Study results.
        k_values : sequence of float
            Thermal conductivity values.
        """

        if not self.process_outputs:
            return

        final_time = self.baseline_parameters.simulation.t_final

        labels = [f"k = {k_value:g} W/mK" for k_value in k_values]

        plot_comparison_profiles(
            results=results,
            labels=labels,
            save_time=final_time,
            profile_y=self.profile_y,
            output_directory=self.output_directory,
            filename="k_study_profile_comparison_t5000.png",
            title="Thermal Conductivity Study",
        )

        plot_case_metric_comparison(
            results=results,
            labels=labels,
            metric="max_temperature",
            output_directory=self.output_directory,
            filename="k_study_max_temperature_comparison.png",
            title="Thermal Conductivity Study: Final Maximum Temperature",
        )

        plot_case_metric_comparison(
            results=results,
            labels=labels,
            metric="average_temperature",
            output_directory=self.output_directory,
            filename="k_study_average_temperature_comparison.png",
            title="Thermal Conductivity Study: Final Average Temperature",
        )

    def run_grid_resolution_study(self) -> List[SimulationResult]:
        """
        Run grid resolution study.

        Cases:
        dx = dy = [0.25, 0.125, 0.0625] m

        Returns
        -------
        list of SimulationResult
            Grid resolution study results.
        """

        grid_sizes = [0.25, 0.125, 0.0625]

        results = []

        for grid_size in grid_sizes:
            parameters = self.create_case_parameters()

            parameters.simulation.dx = grid_size
            parameters.simulation.dy = grid_size

            case_name = f"grid_{format_float_for_filename(grid_size)}"

            result = self.run_case(
                parameters=parameters,
                case_name=case_name,
            )

            results.append(result)

        self.create_grid_resolution_comparison_plots(results, grid_sizes)

        if self.process_outputs:
            export_summary_csv(
                results=results,
                output_directory=self.output_directory,
                filename="grid_resolution_study_summary.csv",
            )

        return results

    def create_grid_resolution_comparison_plots(
        self,
        results: Sequence[SimulationResult],
        grid_sizes: Sequence[float],
    ) -> None:
        """
        Create comparison plots for grid resolution study.

        Parameters
        ----------
        results : sequence of SimulationResult
            Study results.
        grid_sizes : sequence of float
            Grid sizes.
        """

        if not self.process_outputs:
            return

        final_time = self.baseline_parameters.simulation.t_final

        labels = [f"dx = dy = {grid_size:g} m" for grid_size in grid_sizes]

        plot_comparison_profiles(
            results=results,
            labels=labels,
            save_time=final_time,
            profile_y=self.profile_y,
            output_directory=self.output_directory,
            filename="grid_resolution_profile_comparison_t5000.png",
            title="Grid Resolution Study",
        )

        plot_case_metric_comparison(
            results=results,
            labels=labels,
            metric="max_temperature",
            output_directory=self.output_directory,
            filename="grid_resolution_max_temperature_comparison.png",
            title="Grid Resolution Study: Final Maximum Temperature",
        )

        plot_case_metric_comparison(
            results=results,
            labels=labels,
            metric="average_temperature",
            output_directory=self.output_directory,
            filename="grid_resolution_average_temperature_comparison.png",
            title="Grid Resolution Study: Final Average Temperature",
        )

    def run_h_bottom_study(self) -> List[SimulationResult]:
        """
        Run bottom convection coefficient study.

        One-variable-at-a-time:
        h_bottom = [10, 20, 40] W/(m^2*K)

        Other variables stay at baseline.

        Returns
        -------
        list of SimulationResult
            h_bottom study results.
        """

        h_values = [10.0, 20.0, 40.0]

        results = []

        for h_value in h_values:
            parameters = self.create_case_parameters()
            parameters.boundary.h_bottom = h_value

            case_name = f"h_bottom_{format_float_for_filename(h_value)}"

            result = self.run_case(
                parameters=parameters,
                case_name=case_name,
            )

            results.append(result)

        labels = [f"h_bottom = {h_value:g} W/m²K" for h_value in h_values]

        self.create_boundary_study_comparison_plots(
            results=results,
            labels=labels,
            study_name="h_bottom_study",
            title_prefix="Bottom Convection Coefficient Study",
        )

        if self.process_outputs:
            export_summary_csv(
                results=results,
                output_directory=self.output_directory,
                filename="h_bottom_study_summary.csv",
            )

        return results

    def run_q_right_study(self) -> List[SimulationResult]:
        """
        Run right heat flux study.

        One-variable-at-a-time:
        q_right = [75, 150, 300] W/m^2

        Other variables stay at baseline.

        Returns
        -------
        list of SimulationResult
            q_right study results.
        """

        q_values = [75.0, 150.0, 300.0]

        results = []

        for q_value in q_values:
            parameters = self.create_case_parameters()
            parameters.boundary.q_right = q_value

            case_name = f"q_right_{format_float_for_filename(q_value)}"

            result = self.run_case(
                parameters=parameters,
                case_name=case_name,
            )

            results.append(result)

        labels = [f"q_right = {q_value:g} W/m²" for q_value in q_values]

        self.create_boundary_study_comparison_plots(
            results=results,
            labels=labels,
            study_name="q_right_study",
            title_prefix="Right Heat Flux Study",
        )

        if self.process_outputs:
            export_summary_csv(
                results=results,
                output_directory=self.output_directory,
                filename="q_right_study_summary.csv",
            )

        return results

    def run_t_inf_bottom_study(self) -> List[SimulationResult]:
        """
        Run bottom ambient temperature study.

        One-variable-at-a-time:
        T_inf_bottom = [50, 100, 150] C

        Other variables stay at baseline.

        Returns
        -------
        list of SimulationResult
            T_inf_bottom study results.
        """

        t_inf_values = [50.0, 100.0, 150.0]

        results = []

        for t_inf_value in t_inf_values:
            parameters = self.create_case_parameters()
            parameters.boundary.t_inf_bottom = t_inf_value

            case_name = f"t_inf_bottom_{format_float_for_filename(t_inf_value)}"

            result = self.run_case(
                parameters=parameters,
                case_name=case_name,
            )

            results.append(result)

        labels = [f"T_inf_bottom = {t_inf_value:g} °C" for t_inf_value in t_inf_values]

        self.create_boundary_study_comparison_plots(
            results=results,
            labels=labels,
            study_name="t_inf_bottom_study",
            title_prefix="Bottom Ambient Temperature Study",
        )

        if self.process_outputs:
            export_summary_csv(
                results=results,
                output_directory=self.output_directory,
                filename="t_inf_bottom_study_summary.csv",
            )

        return results

    def create_boundary_study_comparison_plots(
        self,
        results: Sequence[SimulationResult],
        labels: Sequence[str],
        study_name: str,
        title_prefix: str,
    ) -> None:
        """
        Create comparison plots for one boundary condition study.

        Parameters
        ----------
        results : sequence of SimulationResult
            Study results.
        labels : sequence of str
            Plot labels.
        study_name : str
            Filename prefix.
        title_prefix : str
            Plot title prefix.
        """

        if not self.process_outputs:
            return

        final_time = self.baseline_parameters.simulation.t_final

        plot_comparison_profiles(
            results=results,
            labels=labels,
            save_time=final_time,
            profile_y=self.profile_y,
            output_directory=self.output_directory,
            filename=f"{study_name}_profile_comparison_t5000.png",
            title=title_prefix,
        )

        plot_case_metric_comparison(
            results=results,
            labels=labels,
            metric="max_temperature",
            output_directory=self.output_directory,
            filename=f"{study_name}_max_temperature_comparison.png",
            title=f"{title_prefix}: Final Maximum Temperature",
        )

        plot_case_metric_comparison(
            results=results,
            labels=labels,
            metric="average_temperature",
            output_directory=self.output_directory,
            filename=f"{study_name}_average_temperature_comparison.png",
            title=f"{title_prefix}: Final Average Temperature",
        )

    def run_boundary_condition_studies(self) -> Dict[str, List[SimulationResult]]:
        """
        Run all one-variable-at-a-time boundary condition studies.

        Returns
        -------
        dict
            Dictionary containing boundary study result lists.
        """

        boundary_results = {
            "h_bottom_study": self.run_h_bottom_study(),
            "q_right_study": self.run_q_right_study(),
            "t_inf_bottom_study": self.run_t_inf_bottom_study(),
        }

        return boundary_results

    def run_all_cases(self) -> Dict[str, object]:
        """
        Run baseline and all parametric studies.

        Returns
        -------
        dict
            Dictionary containing all result groups.
        """

        baseline_result = self.run_baseline_case()

        k_study_results = self.run_thermal_conductivity_study()

        grid_study_results = self.run_grid_resolution_study()

        boundary_study_results = self.run_boundary_condition_studies()

        all_results: List[SimulationResult] = []

        all_results.append(baseline_result)
        all_results.extend(k_study_results)
        all_results.extend(grid_study_results)

        for study_results in boundary_study_results.values():
            all_results.extend(study_results)

        if self.process_outputs:
            export_summary_csv(
                results=all_results,
                output_directory=self.output_directory,
                filename="all_cases_summary.csv",
            )

        print("\n" + "=" * 80)
        print("All cases completed.")
        print(f"Output directory: {self.output_directory}")
        print("=" * 80)

        return {
            "baseline": baseline_result,
            "k_study": k_study_results,
            "grid_resolution_study": grid_study_results,
            "boundary_studies": boundary_study_results,
            "all_results": all_results,
        }


def run_project_cases(
    output_directory: str = "outputs",
    linear_solver: str = "scipy",
    verbose: bool = True,
    process_outputs: bool = True,
    profile_y: float = 0.5,
) -> Dict[str, object]:
    """
    Convenience function to run the complete project.

    Parameters
    ----------
    output_directory : str
        Output folder path.
    linear_solver : str
        Linear solver option.
    verbose : bool
        If True, print progress.
    process_outputs : bool
        If True, create CSV and plot files.
    profile_y : float
        Horizontal profile y-location.

    Returns
    -------
    dict
        Dictionary containing all result groups.
    """

    runner = CaseRunner(
        output_directory=output_directory,
        linear_solver=linear_solver,
        verbose=verbose,
        process_outputs=process_outputs,
        profile_y=profile_y,
    )

    results = runner.run_all_cases()

    return results