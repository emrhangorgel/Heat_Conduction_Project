"""
main.py

Main entry point for the AERO 242 transient 2D stepped-domain heat
conduction project.

Run this file to execute:
1. Baseline case
2. Thermal conductivity study
3. Grid resolution study
4. Boundary condition studies

Author: AERO 242 Project
"""

from run_cases import run_project_cases


def main() -> None:
    """
    Run all project simulations and generate outputs.
    """

    results = run_project_cases(
        output_directory="outputs",
        linear_solver="scipy",
        verbose=True,
        process_outputs=True,
        profile_y=0.5,
    )

    baseline = results["baseline"]

    final_time = baseline.parameters.simulation.t_final
    final_temperature_vector = baseline.saved_temperature_vectors[final_time]

    print("\nBaseline final result")
    print("-" * 70)
    print(f"  Final time              : {final_time:.2f} s")
    print(f"  Minimum temperature     : {final_temperature_vector.min():.4f} C")
    print(f"  Maximum temperature     : {final_temperature_vector.max():.4f} C")
    print(f"  Average temperature     : {final_temperature_vector.mean():.4f} C")
    print("-" * 70)


if __name__ == "__main__":
    main()