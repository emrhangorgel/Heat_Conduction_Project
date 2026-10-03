"""
parameters.py

This module contains all input parameters for the transient 2D heat
conduction project.

All units are SI units unless otherwise stated.

Author: AERO 242 Project
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class MaterialProperties:
    """
    Material properties for the solid domain.

    Attributes
    ----------
    name : str
        Material name.
    density : float
        Density in kg/m^3.
    specific_heat : float
        Specific heat capacity in J/(kg*K).
    thermal_conductivity : float
        Thermal conductivity in W/(m*K).
    """

    name: str = "Ti-6Al-4V Titanium Alloy"
    density: float = 4430.0
    specific_heat: float = 526.3
    thermal_conductivity: float = 6.7

    @property
    def thermal_diffusivity(self) -> float:
        """
        Return thermal diffusivity alpha = k / (rho * cp).

        Returns
        -------
        float
            Thermal diffusivity in m^2/s.
        """
        return self.thermal_conductivity / (self.density * self.specific_heat)


@dataclass
class BoundaryConditions:
    """
    Boundary condition values for the stepped domain.

    Sign convention:
    Heat flux values are positive when heat enters the solid domain.

    Attributes
    ----------
    q_left : float
        Heat flux into the domain on the left boundary in W/m^2.
    q_right : float
        Heat flux into the domain on the right boundary in W/m^2.
    h_bottom : float
        Convection coefficient on the bottom boundary in W/(m^2*K).
    t_inf_bottom : float
        Ambient temperature for bottom convection in degrees Celsius.
    h_top_lower : float
        Convection coefficient on lower top boundary in W/(m^2*K).
    t_inf_top_lower : float
        Ambient temperature for lower top boundary in degrees Celsius.
    h_top_upper : float
        Convection coefficient on upper top boundary in W/(m^2*K).
    t_inf_top_upper : float
        Ambient temperature for upper top boundary in degrees Celsius.
    """

    q_left: float = 50.0
    q_right: float = 150.0

    h_bottom: float = 20.0
    t_inf_bottom: float = 100.0

    h_top_lower: float = 40.0
    t_inf_top_lower: float = 25.0

    h_top_upper: float = 40.0
    t_inf_top_upper: float = 25.0


@dataclass
class GeometryParameters:
    """
    Geometry parameters for the stepped 2D domain.

    The domain is formed by two rectangular parts:
    - Lower-left rectangle: 0 <= x <= step_x, 0 <= y <= lower_height
    - Right rectangle: step_x <= x <= total_length, 0 <= y <= total_height

    Attributes
    ----------
    total_length : float
        Total bottom length in meters.
    lower_height : float
        Height of the lower-left part in meters.
    total_height : float
        Total right-side height in meters.
    step_x : float
        x-location of the vertical step in meters.
    thickness : float
        Out-of-plane thickness in meters.
    tolerance : float
        Coordinate comparison tolerance.
    """

    total_length: float = 2.5
    lower_height: float = 1.0
    total_height: float = 2.0
    step_x: float = 1.25
    thickness: float = 1.0
    tolerance: float = 1.0e-10


@dataclass
class SimulationParameters:
    """
    Main simulation settings.

    Attributes
    ----------
    dx : float
        Grid spacing in x-direction in meters.
    dy : float
        Grid spacing in y-direction in meters.
    dt : float
        Time step in seconds.
    t_final : float
        Final simulation time in seconds.
    initial_temperature : float
        Initial temperature in degrees Celsius.
    save_times : list of float
        Times at which temperature fields should be saved.
    output_directory : str
        Folder where output files are written.
    """

    dx: float = 0.125
    dy: float = 0.125
    dt: float = 10.0
    t_final: float = 5000.0
    initial_temperature: float = 25.0

    save_times: List[float] = field(
        default_factory=lambda: [0.0, 500.0, 1000.0, 2500.0, 5000.0]
    )

    output_directory: str = "outputs"


@dataclass
class ProjectParameters:
    """
    Container class that holds all parameter groups together.
    """

    material: MaterialProperties = field(default_factory=MaterialProperties)
    boundary: BoundaryConditions = field(default_factory=BoundaryConditions)
    geometry: GeometryParameters = field(default_factory=GeometryParameters)
    simulation: SimulationParameters = field(default_factory=SimulationParameters)

    def print_summary(self) -> None:
        """
        Print a readable summary of the project parameters.
        """

        print("=" * 70)
        print("AERO 242 Transient 2D Heat Conduction Project")
        print("=" * 70)

        print("\nMaterial")
        print(f"  Name                  : {self.material.name}")
        print(f"  Density               : {self.material.density:.3f} kg/m^3")
        print(f"  Specific heat         : {self.material.specific_heat:.3f} J/(kg*K)")
        print(
            f"  Thermal conductivity  : "
            f"{self.material.thermal_conductivity:.3f} W/(m*K)"
        )
        print(
            f"  Thermal diffusivity   : "
            f"{self.material.thermal_diffusivity:.6e} m^2/s"
        )

        print("\nGeometry")
        print(f"  Total length          : {self.geometry.total_length:.3f} m")
        print(f"  Lower height          : {self.geometry.lower_height:.3f} m")
        print(f"  Total height          : {self.geometry.total_height:.3f} m")
        print(f"  Step x-location       : {self.geometry.step_x:.3f} m")
        print(f"  Thickness             : {self.geometry.thickness:.3f} m")

        print("\nBoundary Conditions")
        print(f"  Left heat flux        : {self.boundary.q_left:.3f} W/m^2")
        print(f"  Right heat flux       : {self.boundary.q_right:.3f} W/m^2")
        print(
            f"  Bottom convection     : h = {self.boundary.h_bottom:.3f} W/(m^2*K), "
            f"T_inf = {self.boundary.t_inf_bottom:.3f} C"
        )
        print(
            f"  Lower top convection  : h = {self.boundary.h_top_lower:.3f} W/(m^2*K), "
            f"T_inf = {self.boundary.t_inf_top_lower:.3f} C"
        )
        print(
            f"  Upper top convection  : h = {self.boundary.h_top_upper:.3f} W/(m^2*K), "
            f"T_inf = {self.boundary.t_inf_top_upper:.3f} C"
        )
        print("  Vertical step         : insulated")

        print("\nSimulation")
        print(f"  dx                    : {self.simulation.dx:.6f} m")
        print(f"  dy                    : {self.simulation.dy:.6f} m")
        print(f"  dt                    : {self.simulation.dt:.6f} s")
        print(f"  Final time            : {self.simulation.t_final:.3f} s")
        print(f"  Initial temperature   : {self.simulation.initial_temperature:.3f} C")
        print(f"  Save times            : {self.simulation.save_times}")
        print("=" * 70)


def create_baseline_parameters() -> ProjectParameters:
    """
    Create the baseline parameter set.

    Returns
    -------
    ProjectParameters
        Baseline project parameters.
    """

    return ProjectParameters()