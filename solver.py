"""
solver.py

This module contains the transient 2D heat conduction solver for the
stepped domain.

Numerical method:
- Finite difference / finite volume style node balance
- Backward Euler implicit time integration
- Sparse matrix system: A * T_new = b

Important:
- This solver does NOT use direct matrix inverse.
- It does NOT use numpy.linalg.inv().
- Default linear solver is scipy.sparse.linalg.spsolve.
- A custom Gauss-Seidel / SOR fallback is included for educational use.

Author: AERO 242 Project
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from parameters import ProjectParameters
from geometry import (
    BoundaryType,
    Direction,
    FaceClassification,
    Grid,
    create_grid,
    create_temperature_field_with_nan,
    get_neighbor_information,
    print_grid_summary,
)


@dataclass
class MatrixAssemblyResult:
    """
    Stores the assembled linear system.

    Attributes
    ----------
    matrix : csr_matrix
        Sparse coefficient matrix A.
    rhs : np.ndarray
        Right-hand side vector b.
    """

    matrix: csr_matrix
    rhs: np.ndarray


@dataclass
class TimeStepStatistics:
    """
    Stores useful statistics for a single saved time step.

    Attributes
    ----------
    time : float
        Simulation time in seconds.
    minimum_temperature : float
        Minimum active-node temperature in Celsius.
    maximum_temperature : float
        Maximum active-node temperature in Celsius.
    average_temperature : float
        Average active-node temperature in Celsius.
    """

    time: float
    minimum_temperature: float
    maximum_temperature: float
    average_temperature: float


@dataclass
class SimulationResult:
    """
    Stores all important results from a transient simulation.

    Attributes
    ----------
    case_name : str
        Name of the simulation case.
    parameters : ProjectParameters
        Project parameter object used for this run.
    grid : Grid
        Computational grid.
    saved_temperature_vectors : dict
        Dictionary mapping saved time to active temperature vector.
    saved_temperature_fields : dict
        Dictionary mapping saved time to 2D field with NaN inactive region.
    statistics : list of TimeStepStatistics
        Temperature statistics at saved times.
    maximum_temperature_history : list of tuple
        List of (time, maximum_temperature).
    average_temperature_history : list of tuple
        List of (time, average_temperature).
    """

    case_name: str
    parameters: ProjectParameters
    grid: Grid
    saved_temperature_vectors: Dict[float, np.ndarray] = field(default_factory=dict)
    saved_temperature_fields: Dict[float, np.ndarray] = field(default_factory=dict)
    statistics: List[TimeStepStatistics] = field(default_factory=list)
    maximum_temperature_history: List[Tuple[float, float]] = field(default_factory=list)
    average_temperature_history: List[Tuple[float, float]] = field(default_factory=list)


class HeatConductionSolver:
    """
    Transient 2D heat conduction solver for the stepped domain.
    """

    def __init__(
        self,
        parameters: ProjectParameters,
        case_name: str = "baseline",
        linear_solver: str = "scipy",
        sor_omega: float = 1.3,
        iterative_tolerance: float = 1.0e-8,
        maximum_iterations: int = 10000,
        verbose: bool = True,
    ):
        """
        Initialize the solver.

        Parameters
        ----------
        parameters : ProjectParameters
            Complete parameter object.
        case_name : str
            Case name used for output labels.
        linear_solver : str
            Linear solver option. Valid options:
            - "scipy"
            - "gauss_seidel"
            - "sor"
        sor_omega : float
            Relaxation factor for SOR. Must be 0 < omega < 2.
            omega = 1 gives Gauss-Seidel.
        iterative_tolerance : float
            Convergence tolerance for custom iterative solvers.
        maximum_iterations : int
            Maximum iterations for custom iterative solvers.
        verbose : bool
            If True, print progress and sanity checks.
        """

        self.parameters = parameters
        self.case_name = case_name
        self.linear_solver = linear_solver.lower()
        self.sor_omega = sor_omega
        self.iterative_tolerance = iterative_tolerance
        self.maximum_iterations = maximum_iterations
        self.verbose = verbose

        self.grid = create_grid(
            geometry=self.parameters.geometry,
            simulation=self.parameters.simulation,
        )

        self._validate_solver_settings()

        if self.verbose:
            self.parameters.print_summary()
            print_grid_summary(self.grid)

    def _validate_solver_settings(self) -> None:
        """
        Validate linear solver settings.
        """

        valid_solvers = {"scipy", "gauss_seidel", "sor"}

        if self.linear_solver not in valid_solvers:
            raise ValueError(
                f"Unknown linear solver '{self.linear_solver}'. "
                f"Valid options are: {sorted(valid_solvers)}"
            )

        if not (0.0 < self.sor_omega < 2.0):
            raise ValueError("SOR omega must satisfy 0 < omega < 2.")

        if self.iterative_tolerance <= 0.0:
            raise ValueError("Iterative tolerance must be positive.")

        if self.maximum_iterations <= 0:
            raise ValueError("Maximum iterations must be positive.")

    def create_initial_temperature_vector(self) -> np.ndarray:
        """
        Create the initial active-node temperature vector.

        Returns
        -------
        np.ndarray
            Active-node temperature vector at t = 0.
        """

        initial_temperature = self.parameters.simulation.initial_temperature

        temperature_vector = np.full(
            self.grid.active_node_count,
            initial_temperature,
            dtype=float,
        )

        self._validate_temperature_vector(temperature_vector)

        return temperature_vector

    def _validate_temperature_vector(self, temperature_vector: np.ndarray) -> None:
        """
        Run basic sanity checks on the temperature vector.

        Parameters
        ----------
        temperature_vector : np.ndarray
            Active-node temperature vector.
        """

        if len(temperature_vector) != self.grid.active_node_count:
            raise ValueError(
                "Temperature vector length does not match active node count. "
                f"Expected {self.grid.active_node_count}, got {len(temperature_vector)}."
            )

        if np.any(np.isnan(temperature_vector)):
            raise ValueError("Temperature vector contains NaN values.")

        if np.any(np.isinf(temperature_vector)):
            raise ValueError("Temperature vector contains infinite values.")

    def get_face_area_and_distance(self, direction: Direction) -> Tuple[float, float]:
        """
        Return face area and neighbor distance for a given direction.

        Parameters
        ----------
        direction : Direction
            Direction of the face.

        Returns
        -------
        tuple of float
            face_area, distance

        Notes
        -----
        Unit thickness is used by default through geometry.thickness.

        East/west face area = dy * thickness
        North/south face area = dx * thickness
        """

        dx = self.parameters.simulation.dx
        dy = self.parameters.simulation.dy
        thickness = self.parameters.geometry.thickness

        if direction in (Direction.EAST, Direction.WEST):
            face_area = dy * thickness
            distance = dx

        elif direction in (Direction.NORTH, Direction.SOUTH):
            face_area = dx * thickness
            distance = dy

        else:
            raise ValueError(f"Unknown direction: {direction}")

        return face_area, distance

    def get_boundary_convection_data(
        self,
        boundary_type: BoundaryType,
    ) -> Tuple[float, float]:
        """
        Return convection coefficient and ambient temperature for a boundary.

        Parameters
        ----------
        boundary_type : BoundaryType
            Boundary type.

        Returns
        -------
        tuple of float
            h, T_inf
        """

        boundary = self.parameters.boundary

        if boundary_type == BoundaryType.BOTTOM_CONVECTION:
            return boundary.h_bottom, boundary.t_inf_bottom

        if boundary_type == BoundaryType.LOWER_TOP_CONVECTION:
            return boundary.h_top_lower, boundary.t_inf_top_lower

        if boundary_type == BoundaryType.UPPER_TOP_CONVECTION:
            return boundary.h_top_upper, boundary.t_inf_top_upper

        raise ValueError(
            f"Boundary type {boundary_type.value} is not a convection boundary."
        )

    def get_boundary_heat_flux(
        self,
        boundary_type: BoundaryType,
    ) -> float:
        """
        Return heat flux for a heat flux boundary.

        Sign convention:
        Positive heat flux means heat enters the solid domain.

        Parameters
        ----------
        boundary_type : BoundaryType
            Boundary type.

        Returns
        -------
        float
            Heat flux in W/m^2.
        """

        boundary = self.parameters.boundary

        if boundary_type == BoundaryType.LEFT_HEAT_FLUX:
            return boundary.q_left

        if boundary_type == BoundaryType.RIGHT_HEAT_FLUX:
            return boundary.q_right

        raise ValueError(
            f"Boundary type {boundary_type.value} is not a heat flux boundary."
        )

    def apply_boundary_contribution(
        self,
        matrix: lil_matrix,
        rhs: np.ndarray,
        row_index: int,
        face: FaceClassification,
    ) -> None:
        """
        Apply missing-neighbor boundary contribution to the matrix equation.

        Parameters
        ----------
        matrix : lil_matrix
            Sparse matrix in LIL format during assembly.
        rhs : np.ndarray
            Right-hand side vector.
        row_index : int
            Current matrix row index.
        face : FaceClassification
            Boundary face classification.
        """

        boundary_type = face.boundary_type
        face_area, _ = self.get_face_area_and_distance(face.direction)

        if boundary_type in (
            BoundaryType.BOTTOM_CONVECTION,
            BoundaryType.LOWER_TOP_CONVECTION,
            BoundaryType.UPPER_TOP_CONVECTION,
        ):
            h, t_inf = self.get_boundary_convection_data(boundary_type)

            matrix[row_index, row_index] += h * face_area
            rhs[row_index] += h * face_area * t_inf

        elif boundary_type in (
            BoundaryType.LEFT_HEAT_FLUX,
            BoundaryType.RIGHT_HEAT_FLUX,
        ):
            heat_flux = self.get_boundary_heat_flux(boundary_type)

            rhs[row_index] += heat_flux * face_area

        elif boundary_type in (
            BoundaryType.STEP_INSULATED,
            BoundaryType.INSULATED,
        ):
            # Insulated boundary: dT/dn = 0
            # No heat transfer contribution.
            pass

        else:
            raise ValueError(f"Unsupported boundary type: {boundary_type.value}")

    def assemble_system(
        self,
        old_temperature_vector: np.ndarray,
    ) -> MatrixAssemblyResult:
        """
        Assemble the sparse linear system for one Backward Euler time step.

        The equation has the form:

            A * T_new = b

        For each active node p:

            transient_coefficient * T_new_p
            + sum(conduction terms using T_new)
            + convection terms using T_new
            =
            transient_coefficient * T_old_p
            + heat flux source terms
            + convection ambient source terms

        Parameters
        ----------
        old_temperature_vector : np.ndarray
            Active-node temperature vector from the previous time step.

        Returns
        -------
        MatrixAssemblyResult
            Sparse matrix and RHS vector.
        """

        self._validate_temperature_vector(old_temperature_vector)

        number_of_unknowns = self.grid.active_node_count

        matrix = lil_matrix((number_of_unknowns, number_of_unknowns), dtype=float)
        rhs = np.zeros(number_of_unknowns, dtype=float)

        material = self.parameters.material
        simulation = self.parameters.simulation
        geometry = self.parameters.geometry

        dx = simulation.dx
        dy = simulation.dy
        dt = simulation.dt
        thickness = geometry.thickness

        volume = dx * dy * thickness

        transient_coefficient = (
            material.density * material.specific_heat * volume / dt
        )

        thermal_conductivity = material.thermal_conductivity

        directions = [
            Direction.EAST,
            Direction.WEST,
            Direction.NORTH,
            Direction.SOUTH,
        ]

        for i, j, row_index, _, _ in self.grid.iter_active_nodes():
            # Transient term
            matrix[row_index, row_index] += transient_coefficient
            rhs[row_index] += transient_coefficient * old_temperature_vector[row_index]

            for direction in directions:
                (
                    has_active_neighbor,
                    neighbor_i,
                    neighbor_j,
                    neighbor_matrix_index,
                    face_classification,
                ) = get_neighbor_information(
                    grid=self.grid,
                    i=i,
                    j=j,
                    direction=direction,
                    geometry=geometry,
                )

                face_area, distance = self.get_face_area_and_distance(direction)

                if has_active_neighbor:
                    conductance = thermal_conductivity * face_area / distance

                    matrix[row_index, row_index] += conductance
                    matrix[row_index, neighbor_matrix_index] -= conductance

                else:
                    self.apply_boundary_contribution(
                        matrix=matrix,
                        rhs=rhs,
                        row_index=row_index,
                        face=face_classification,
                    )

        return MatrixAssemblyResult(
            matrix=matrix.tocsr(),
            rhs=rhs,
        )

    def solve_linear_system(
        self,
        matrix: csr_matrix,
        rhs: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solve A * x = b without using direct inverse.

        Parameters
        ----------
        matrix : csr_matrix
            Sparse coefficient matrix A.
        rhs : np.ndarray
            Right-hand side vector b.
        initial_guess : np.ndarray, optional
            Initial guess for iterative solvers.

        Returns
        -------
        np.ndarray
            Solution vector.
        """

        if self.linear_solver == "scipy":
            solution = spsolve(matrix, rhs)

        elif self.linear_solver == "gauss_seidel":
            solution = self.solve_with_sor(
                matrix=matrix,
                rhs=rhs,
                initial_guess=initial_guess,
                omega=1.0,
            )

        elif self.linear_solver == "sor":
            solution = self.solve_with_sor(
                matrix=matrix,
                rhs=rhs,
                initial_guess=initial_guess,
                omega=self.sor_omega,
            )

        else:
            raise ValueError(f"Unknown linear solver: {self.linear_solver}")

        self._validate_temperature_vector(solution)

        return solution

    def solve_with_sor(
        self,
        matrix: csr_matrix,
        rhs: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        omega: float = 1.3,
    ) -> np.ndarray:
        """
        Solve A * x = b using Successive Over-Relaxation.

        This is included as an educational fallback. For the final project,
        scipy.sparse.linalg.spsolve is usually faster and more robust.

        Parameters
        ----------
        matrix : csr_matrix
            Sparse coefficient matrix.
        rhs : np.ndarray
            Right-hand side vector.
        initial_guess : np.ndarray, optional
            Initial guess for the solution.
        omega : float
            Relaxation parameter.
            omega = 1.0 gives Gauss-Seidel.
            1.0 < omega < 2.0 may accelerate convergence.

        Returns
        -------
        np.ndarray
            Solution vector.

        Raises
        ------
        RuntimeError
            If the solver does not converge.
        """

        if not (0.0 < omega < 2.0):
            raise ValueError("SOR omega must satisfy 0 < omega < 2.")

        matrix_csr = matrix.tocsr()

        number_of_unknowns = matrix_csr.shape[0]

        if initial_guess is None:
            x = np.zeros(number_of_unknowns, dtype=float)
        else:
            x = initial_guess.copy().astype(float)

        diagonal = matrix_csr.diagonal()

        if np.any(np.abs(diagonal) < 1.0e-30):
            raise ValueError("Matrix contains a zero diagonal entry.")

        for iteration in range(self.maximum_iterations):
            x_old = x.copy()

            for row in range(number_of_unknowns):
                row_start = matrix_csr.indptr[row]
                row_end = matrix_csr.indptr[row + 1]

                sigma = 0.0

                for data_index in range(row_start, row_end):
                    column = matrix_csr.indices[data_index]
                    value = matrix_csr.data[data_index]

                    if column != row:
                        sigma += value * x[column]

                gauss_seidel_value = (rhs[row] - sigma) / diagonal[row]

                x[row] = (1.0 - omega) * x[row] + omega * gauss_seidel_value

            difference_norm = np.linalg.norm(x - x_old, ord=np.inf)
            solution_norm = max(np.linalg.norm(x, ord=np.inf), 1.0)

            relative_change = difference_norm / solution_norm

            if relative_change < self.iterative_tolerance:
                if self.verbose:
                    print(
                        f"  Iterative solver converged in {iteration + 1} iterations. "
                        f"Relative change = {relative_change:.3e}"
                    )
                return x

        residual = matrix_csr.dot(x) - rhs
        residual_norm = np.linalg.norm(residual, ord=np.inf)

        raise RuntimeError(
            "SOR/Gauss-Seidel solver did not converge. "
            f"Maximum iterations = {self.maximum_iterations}, "
            f"final residual infinity norm = {residual_norm:.6e}"
        )

    def compute_statistics(
        self,
        time: float,
        temperature_vector: np.ndarray,
    ) -> TimeStepStatistics:
        """
        Compute min, max, and average active-node temperature.

        Parameters
        ----------
        time : float
            Current time in seconds.
        temperature_vector : np.ndarray
            Active-node temperature vector.

        Returns
        -------
        TimeStepStatistics
            Temperature statistics.
        """

        self._validate_temperature_vector(temperature_vector)

        return TimeStepStatistics(
            time=time,
            minimum_temperature=float(np.min(temperature_vector)),
            maximum_temperature=float(np.max(temperature_vector)),
            average_temperature=float(np.mean(temperature_vector)),
        )

    def should_save_time(
        self,
        current_time: float,
        save_times: List[float],
    ) -> Optional[float]:
        """
        Check whether current time is one of the requested save times.

        Parameters
        ----------
        current_time : float
            Current simulation time.
        save_times : list of float
            Requested save times.

        Returns
        -------
        float or None
            Matching save time if current time should be saved.
            Otherwise None.
        """

        dt = self.parameters.simulation.dt
        tolerance = 0.5 * dt + 1.0e-9

        for save_time in save_times:
            if abs(current_time - save_time) <= tolerance:
                return float(save_time)

        return None

    def store_saved_result(
        self,
        result: SimulationResult,
        save_time: float,
        temperature_vector: np.ndarray,
    ) -> None:
        """
        Store temperature vector, 2D temperature field, and statistics.

        Parameters
        ----------
        result : SimulationResult
            Simulation result object.
        save_time : float
            Time to be saved.
        temperature_vector : np.ndarray
            Active-node temperature vector.
        """

        temperature_copy = temperature_vector.copy()

        result.saved_temperature_vectors[save_time] = temperature_copy

        result.saved_temperature_fields[save_time] = create_temperature_field_with_nan(
            grid=self.grid,
            active_temperature_vector=temperature_copy,
        )

        statistics = self.compute_statistics(
            time=save_time,
            temperature_vector=temperature_copy,
        )

        result.statistics.append(statistics)

        if self.verbose:
            print(
                f"  Saved t = {save_time:8.2f} s | "
                f"T_min = {statistics.minimum_temperature:10.4f} C | "
                f"T_max = {statistics.maximum_temperature:10.4f} C | "
                f"T_avg = {statistics.average_temperature:10.4f} C"
            )

    def run_transient_simulation(self) -> SimulationResult:
        """
        Run the full transient simulation.

        Returns
        -------
        SimulationResult
            Complete simulation result.
        """

        simulation = self.parameters.simulation

        dt = simulation.dt
        t_final = simulation.t_final
        save_times = sorted(float(time) for time in simulation.save_times)

        if t_final < 0.0:
            raise ValueError("Final time must be non-negative.")

        if dt <= 0.0:
            raise ValueError("Time step must be positive.")

        number_of_steps_float = t_final / dt
        number_of_steps = int(round(number_of_steps_float))

        if abs(number_of_steps * dt - t_final) > 1.0e-8:
            raise ValueError(
                "Final time must be an integer multiple of dt for this project code. "
                f"t_final = {t_final}, dt = {dt}"
            )

        temperature_vector = self.create_initial_temperature_vector()

        result = SimulationResult(
            case_name=self.case_name,
            parameters=self.parameters,
            grid=self.grid,
        )

        if self.verbose:
            print("\nStarting transient simulation")
            print("-" * 70)
            print(f"  Case name        : {self.case_name}")
            print(f"  Linear solver    : {self.linear_solver}")
            print(f"  Matrix size      : {self.grid.active_node_count} x {self.grid.active_node_count}")
            print(f"  Number of steps  : {number_of_steps}")
            print("-" * 70)

        # Save initial condition if requested
        initial_save_time = self.should_save_time(0.0, save_times)

        if initial_save_time is not None:
            self.store_saved_result(
                result=result,
                save_time=initial_save_time,
                temperature_vector=temperature_vector,
            )

        # Store history at t = 0
        initial_statistics = self.compute_statistics(
            time=0.0,
            temperature_vector=temperature_vector,
        )
        result.maximum_temperature_history.append(
            (0.0, initial_statistics.maximum_temperature)
        )
        result.average_temperature_history.append(
            (0.0, initial_statistics.average_temperature)
        )

        for step in range(1, number_of_steps + 1):
            current_time = step * dt

            assembly = self.assemble_system(
                old_temperature_vector=temperature_vector,
            )

            temperature_vector_new = self.solve_linear_system(
                matrix=assembly.matrix,
                rhs=assembly.rhs,
                initial_guess=temperature_vector,
            )

            temperature_vector = temperature_vector_new

            statistics = self.compute_statistics(
                time=current_time,
                temperature_vector=temperature_vector,
            )

            result.maximum_temperature_history.append(
                (current_time, statistics.maximum_temperature)
            )
            result.average_temperature_history.append(
                (current_time, statistics.average_temperature)
            )

            save_time = self.should_save_time(current_time, save_times)

            if save_time is not None:
                if save_time not in result.saved_temperature_vectors:
                    self.store_saved_result(
                        result=result,
                        save_time=save_time,
                        temperature_vector=temperature_vector,
                    )

            if self.verbose:
                progress_interval = max(number_of_steps // 10, 1)

                if step % progress_interval == 0 or step == number_of_steps:
                    print(
                        f"  Step {step:5d}/{number_of_steps:5d} | "
                        f"t = {current_time:9.2f} s | "
                        f"T_max = {statistics.maximum_temperature:10.4f} C | "
                        f"T_avg = {statistics.average_temperature:10.4f} C"
                    )

        self.run_final_sanity_checks(result)

        if self.verbose:
            print("-" * 70)
            print("Simulation completed.")
            print("-" * 70)

        return result

    def run_final_sanity_checks(self, result: SimulationResult) -> None:
        """
        Run final sanity checks after the simulation.

        Parameters
        ----------
        result : SimulationResult
            Simulation result.
        """

        if len(result.saved_temperature_vectors) == 0:
            raise RuntimeError("No saved temperature results were produced.")

        final_time = self.parameters.simulation.t_final

        if final_time not in result.saved_temperature_vectors:
            raise RuntimeError(
                f"Final time {final_time} s was not saved. "
                "Check save_times in SimulationParameters."
            )

        final_temperature = result.saved_temperature_vectors[final_time]

        self._validate_temperature_vector(final_temperature)

        final_minimum = float(np.min(final_temperature))
        final_maximum = float(np.max(final_temperature))
        final_average = float(np.mean(final_temperature))

        if self.verbose:
            print("\nFinal sanity checks")
            print("-" * 70)
            print(f"  Active node count       : {self.grid.active_node_count}")
            print(f"  Final minimum temp      : {final_minimum:.6f} C")
            print(f"  Final maximum temp      : {final_maximum:.6f} C")
            print(f"  Final average temp      : {final_average:.6f} C")

        # These are not hard physical laws, only practical warnings.
        if final_maximum < self.parameters.simulation.initial_temperature:
            print(
                "Warning: Final maximum temperature is below the initial temperature. "
                "This may be possible only if cooling dominates strongly."
            )

        if final_average < -100.0 or final_average > 1000.0:
            print(
                "Warning: Final average temperature looks unusual. "
                "Check boundary conditions, units, and time step."
            )

        if self.verbose:
            print("-" * 70)


def run_single_case(
    parameters: ProjectParameters,
    case_name: str = "baseline",
    linear_solver: str = "scipy",
    verbose: bool = True,
) -> SimulationResult:
    """
    Convenience function to run a single simulation case.

    Parameters
    ----------
    parameters : ProjectParameters
        Complete project parameter object.
    case_name : str
        Case name.
    linear_solver : str
        Linear solver option.
    verbose : bool
        If True, print progress.

    Returns
    -------
    SimulationResult
        Simulation result.
    """

    solver = HeatConductionSolver(
        parameters=parameters,
        case_name=case_name,
        linear_solver=linear_solver,
        verbose=verbose,
    )

    result = solver.run_transient_simulation()

    return result