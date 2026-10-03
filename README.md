AERO 242 · Computational Heat Transfer Project

Transient 2D Heat Conduction Analysis in a Stepped Domain

Finite Difference Method with Implicit (Backward Euler) Time Marching

👥 Project Team (Group 31)

Mert Bardakcı — 230601012045

Emirhan Görgel — 230601012022

🎯 Problem Definition & Scope

This repository contains a complete Python-based computational solver for evaluating the transient temperature field 
T
(
x
,
y
,
t
)
 over an L-shaped (stepped) solid plate made of Ti-6Al-4V Titanium Alloy.

Initial State: The plate starts uniformly at 
25
 
∘
C
.

Boundary Conditions (6 Distinct Zones acting simultaneously):

Left Edge (Heat Flux): 
q
″
=
50
 
W/m
2
 (Heat in)

Right Edge (Heat Flux): 
q
″
=
150
 
W/m
2
 (Strong heat in)

Bottom Edge (Convection): 
h
=
20
 
W/m
2
K
, 
T
∞
=
100
 
∘
C
 (Main thermal driver)

Lower Top Edge (Convection): 
h
=
40
 
W/m
2
K
, 
T
∞
=
25
 
∘
C
 (Cooling)

Upper Top Edge (Convection): 
h
=
40
 
W/m
2
K
, 
T
∞
=
25
 
∘
C
 (Cooling)

Vertical Step Face (Insulation): 
∂
T
∂
n
=
0
 (Adiabatic wall forcing heat to flow around the corner)

Time Horizon: 
5000
 
s
 (
Δ
t
=
10
 
s
, 500 implicit time steps).

⚙️ Methodology & Numerical Formulation

Governing Equation:

ρ
c
p
∂
T
∂
t
=
k
(
∂
2
T
∂
x
2
+
∂
2
T
∂
y
2
)

Discretization: Control-volume formulation using a standard 5-point stencil with uniform grid spacing (
Δ
x
=
Δ
y
).

Time Integration: Fully implicit Backward Euler method. Unconditionally stable, allowing robust time-marching without stability constraints.

Active Node Handling: The upper-left quadrant is cut out from the domain. An active-node mask filters out inactive cells so that only the 277 active solid nodes (for the baseline grid) are assigned matrix rows.

Direct Solver Rule Compliance: Sparse linear systems 
A
⋅
T
n
+
1
=
b
 are solved directly using SciPy's sparse solver (spsolve). No matrix inverse (
A
−
1
) is ever formed.

📂 Code Architecture

The repository is modularly organized as follows:

parameters.py — Dataclasses containing material properties, geometry specs, boundary conditions, and simulation settings.

geometry.py — Constructs the stepped grid, active-node mask, index map, and face classifier.

solver.py — Handles sparse matrix assembly and Backward-Euler time marching.

postprocess.py — Generates CSV files, contour plots, heatmaps, 1D profiles, and historical trajectory curves.

run_cases.py — Driver script running baseline and parametric studies.

main.py — Project entry point executing all simulations end-to-end.

🚀 Getting Started & Installation

Clone the repository:

git clone https://github.com/your-username/aero242-heat-conduction.git cd aero242-heat-conduction

Install dependencies: Ensure you have Python installed along with the required scientific packages:

pip install numpy scipy matplotlib

Run the simulation suite:

python main.py

📊 Key Results & Findings

Baseline Peak Temperature: Reaches 
42.6
 
∘
C
 near the bottom-right corners by 
t
=
5000
 
s
.

Effect of Thermal Conductivity (
k
): Higher conductivity (
k
=
75
 
W/m
⋅
K
) flattens internal temperature gradients and reduces local peaks, whereas lower conductivity (
k
=
5
 
W/m
⋅
K
) accentuates hot spots up to 
43.7
 
∘
C
.

Grid Refinement: Coarse grids (
Δ
=
0.25
 
m
) smear out steep boundary-layer gradients (
36.8
 
∘
C
 peak), while refined grids (
Δ
=
0.0625
 
m
, 1033 nodes) capture sharper local peaks (
46.4
 
∘
C
).

Dominant Driver: Bottom convection parameters (
h
bottom
 and 
T
∞
,
bottom
) serve as the most powerful leverage variables affecting maximum plate temperatures.

📄 License

This project is developed as part of the AERO 242 Computational Heat Transfer course curriculum. Feel free to use and reference code snippets with proper attribution.
