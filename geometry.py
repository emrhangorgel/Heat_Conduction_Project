"""
geometry.py

This module creates the stepped 2D finite difference grid and provides
helper functions for active node detection and boundary face classification.

The solver will use this module to:
1. Build active/inactive node masks.
2. Assign matrix indices to active nodes only.
3. Classify missing-neighbor faces as flux, convection, or insulated boundaries.

Author: AERO 242 Project
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from parameters import GeometryParameters, SimulationParameters


class Direction(Enum):
    """
    Coordinate directions used for neighbor searching.
    """

    EAST = "east"
    WEST = "west"
    NORTH = "north"
    SOUTH = "south"


class BoundaryType(Enum):
    """
    Boundary face types used by the matrix assembly.
    """

    LEFT_HEAT_FLUX = "left_heat_flux"
    RIGHT_HEAT_FLUX = "right_heat_flux"
    BOTTOM_CONVECTION = "bottom_convection"
    LOWER_TOP_CONVECTION = "lower_top_convection"
    UPPER_TOP_CONVECTION = "upper_top_convection"
    STEP_INSULATED = "step_insulated"
    INSULATED = "insulated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FaceClassification:
    """
    Result of a missing-neighbor face classification.

    Attributes
    ----------
    boundary_type : BoundaryType
        Type of boundary face.
    direction : Direction
        Direction of the missing neighbor.
    x : float
        x-coordinate of the current active node.
    y : float
        y-coordinate of the current active node.
    """

    boundary_type: BoundaryType
    direction: Direction
    x: float
    y: float


@dataclass
class Grid:
    """
    Finite difference grid for the stepped domain.

    Attributes
    ----------
    x_coordinates : np.ndarray
        1D array of x-node coordinates.
    y_coordinates : np.ndarray
        1D array of y-node coordinates.
    x_mesh : np.ndarray
        2D meshgrid of x-coordinates.
    y_mesh : np.ndarray
        2D meshgrid of y-coordinates.
    active_mask : np.ndarray
        Boolean array where True means active solid node.
    index_map : np.ndarray
        Integer array mapping grid nodes to linear matrix indices.
        Inactive nodes are marked by -1.
    active_node_count : int
        Number of active nodes.
    dx : float
        Grid spacing in x-direction.
    dy : float
        Grid spacing in y-direction.
    """

    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    x_mesh: np.ndarray
    y_mesh: np.ndarray
    active_mask: np.ndarray
    index_map: np.ndarray
    active_node_count: int
    dx: float
    dy: float

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Return grid shape as (number_of_y_nodes, number_of_x_nodes).
        """

        return self.active_mask.shape

    @property
    def number_of_x_nodes(self) -> int:
        """
        Return number of nodes in x-direction.
        """

        return len(self.x_coordinates)

    @property
    def number_of_y_nodes(self) -> int:
        """
        Return number of nodes in y-direction.
        """

        return len(self.y_coordinates)

    def is_inside_array(self, i: int, j: int) -> bool:
        """
        Check whether array indices are inside the full rectangular grid.

        Parameters
        ----------
        i : int
            y-index.
        j : int
            x-index.

        Returns
        -------
        bool
            True if indices are inside the array.
        """

        ny, nx = self.shape
        return 0 <= i < ny and 0 <= j < nx

    def is_active_index(self, i: int, j: int) -> bool:
        """
        Check whether a node index is active.

        Parameters
        ----------
        i : int
            y-index.
        j : int
            x-index.

        Returns
        -------
        bool
            True if node is inside the active stepped domain.
        """

        if not self.is_inside_array(i, j):
            return False

        return bool(self.active_mask[i, j])

    def get_matrix_index(self, i: int, j: int) -> int:
        """
        Return the matrix index of an active node.

        Parameters
        ----------
        i : int
            y-index.
        j : int
            x-index.

        Returns
        -------
        int
            Matrix index. Returns -1 for inactive nodes.
        """

        if not self.is_inside_array(i, j):
            return -1

        return int(self.index_map[i, j])

    def get_coordinates(self, i: int, j: int) -> Tuple[float, float]:
        """
        Return physical coordinates of a node.

        Parameters
        ----------
        i : int
            y-index.
        j : int
            x-index.

        Returns
        -------
        tuple of float
            x and y coordinates.
        """

        return float(self.x_mesh[i, j]), float(self.y_mesh[i, j])

    def iter_active_nodes(self):
        """
        Iterate over active nodes.

        Yields
        ------
        tuple
            i, j, matrix_index, x, y
        """

        active_indices = np.argwhere(self.active_mask)

        for i, j in active_indices:
            matrix_index = self.get_matrix_index(i, j)
            x, y = self.get_coordinates(i, j)
            yield int(i), int(j), matrix_index, x, y


def approximately_equal(value_a: float, value_b: float, tolerance: float) -> bool:
    """
    Compare two floating point values using an absolute tolerance.

    Parameters
    ----------
    value_a : float
        First value.
    value_b : float
        Second value.
    tolerance : float
        Absolute tolerance.

    Returns
    -------
    bool
        True if the two values are approximately equal.
    """

    return abs(value_a - value_b) <= tolerance


def is_less_or_equal(value_a: float, value_b: float, tolerance: float) -> bool:
    """
    Tolerance-safe check for value_a <= value_b.
    """

    return value_a <= value_b + tolerance


def is_greater_or_equal(value_a: float, value_b: float, tolerance: float) -> bool:
    """
    Tolerance-safe check for value_a >= value_b.
    """

    return value_a >= value_b - tolerance


def is_between(
    value: float,
    lower_bound: float,
    upper_bound: float,
    tolerance: float,
) -> bool:
    """
    Tolerance-safe interval check.

    Returns True if lower_bound <= value <= upper_bound.
    """

    return (
        is_greater_or_equal(value, lower_bound, tolerance)
        and is_less_or_equal(value, upper_bound, tolerance)
    )


def is_active_node(
    x: float,
    y: float,
    geometry: GeometryParameters,
) -> bool:
    """
    Check whether a coordinate belongs to the active stepped solid domain.

    Active domain rule:
    A node is active if:

    (x <= step_x and y <= lower_height)
    OR
    (x >= step_x and x <= total_length and y <= total_height)

    Parameters
    ----------
    x : float
        x-coordinate.
    y : float
        y-coordinate.
    geometry : GeometryParameters
        Geometry settings.

    Returns
    -------
    bool
        True if the node is active.
    """

    tol = geometry.tolerance

    inside_lower_left_part = (
        is_between(x, 0.0, geometry.step_x, tol)
        and is_between(y, 0.0, geometry.lower_height, tol)
    )

    inside_right_part = (
        is_between(x, geometry.step_x, geometry.total_length, tol)
        and is_between(y, 0.0, geometry.total_height, tol)
    )

    return inside_lower_left_part or inside_right_part


def is_left_boundary_face(
    x: float,
    y: float,
    direction: Direction,
    geometry: GeometryParameters,
) -> bool:
    """
    Detect the left heat flux boundary face.

    Boundary:
    x = 0, 0 <= y <= lower_height

    This face is reached when the missing neighbor is WEST.
    """

    tol = geometry.tolerance

    return (
        direction == Direction.WEST
        and approximately_equal(x, 0.0, tol)
        and is_between(y, 0.0, geometry.lower_height, tol)
    )


def is_right_boundary_face(
    x: float,
    y: float,
    direction: Direction,
    geometry: GeometryParameters,
) -> bool:
    """
    Detect the right heat flux boundary face.

    Boundary:
    x = total_length, 0 <= y <= total_height

    This face is reached when the missing neighbor is EAST.
    """

    tol = geometry.tolerance

    return (
        direction == Direction.EAST
        and approximately_equal(x, geometry.total_length, tol)
        and is_between(y, 0.0, geometry.total_height, tol)
    )


def is_bottom_boundary_face(
    x: float,
    y: float,
    direction: Direction,
    geometry: GeometryParameters,
) -> bool:
    """
    Detect the bottom convection boundary face.

    Boundary:
    y = 0, 0 <= x <= total_length

    This face is reached when the missing neighbor is SOUTH.
    """

    tol = geometry.tolerance

    return (
        direction == Direction.SOUTH
        and approximately_equal(y, 0.0, tol)
        and is_between(x, 0.0, geometry.total_length, tol)
    )


def is_lower_top_convection_face(
    x: float,
    y: float,
    direction: Direction,
    geometry: GeometryParameters,
) -> bool:
    """
    Detect the lower top horizontal convection boundary.

    Boundary:
    y = lower_height, 0 <= x < step_x

    This face is reached when the missing neighbor is NORTH.

    Note:
    At x = step_x, the node belongs to the vertical connection line of the
    right part. Its north neighbor may be active, so normally it is not a
    missing-neighbor boundary face.
    """

    tol = geometry.tolerance

    return (
        direction == Direction.NORTH
        and approximately_equal(y, geometry.lower_height, tol)
        and is_between(x, 0.0, geometry.step_x, tol)
    )


def is_upper_top_convection_face(
    x: float,
    y: float,
    direction: Direction,
    geometry: GeometryParameters,
) -> bool:
    """
    Detect the upper top horizontal convection boundary.

    Boundary:
    y = total_height, step_x <= x <= total_length

    This face is reached when the missing neighbor is NORTH.
    """

    tol = geometry.tolerance

    return (
        direction == Direction.NORTH
        and approximately_equal(y, geometry.total_height, tol)
        and is_between(x, geometry.step_x, geometry.total_length, tol)
    )


def is_step_insulated_face(
    x: float,
    y: float,
    direction: Direction,
    geometry: GeometryParameters,
) -> bool:
    """
    Detect the vertical insulated step boundary.

    Boundary:
    x = step_x, lower_height < y <= total_height

    This face is reached when the missing neighbor is WEST.

    Explanation:
    For nodes on x = step_x and y > lower_height, the west neighbor is
    outside the solid because the upper-left rectangular region is missing.
    Therefore that west face is the vertical step boundary.
    """

    tol = geometry.tolerance

    return (
        direction == Direction.WEST
        and approximately_equal(x, geometry.step_x, tol)
        and is_greater_or_equal(y, geometry.lower_height, tol)
        and is_less_or_equal(y, geometry.total_height, tol)
    )


def classify_missing_neighbor_face(
    x: float,
    y: float,
    direction: Direction,
    geometry: GeometryParameters,
) -> FaceClassification:
    """
    Classify a face where the neighbor node is inactive or outside the array.

    Parameters
    ----------
    x : float
        x-coordinate of the current active node.
    y : float
        y-coordinate of the current active node.
    direction : Direction
        Direction of the missing neighbor.
    geometry : GeometryParameters
        Geometry settings.

    Returns
    -------
    FaceClassification
        Boundary classification information.

    Raises
    ------
    ValueError
        If the missing face cannot be classified.
    """

    if is_left_boundary_face(x, y, direction, geometry):
        boundary_type = BoundaryType.LEFT_HEAT_FLUX

    elif is_right_boundary_face(x, y, direction, geometry):
        boundary_type = BoundaryType.RIGHT_HEAT_FLUX

    elif is_bottom_boundary_face(x, y, direction, geometry):
        boundary_type = BoundaryType.BOTTOM_CONVECTION

    elif is_upper_top_convection_face(x, y, direction, geometry):
        boundary_type = BoundaryType.UPPER_TOP_CONVECTION

    elif is_step_insulated_face(x, y, direction, geometry):
        boundary_type = BoundaryType.STEP_INSULATED

    elif is_lower_top_convection_face(x, y, direction, geometry):
        boundary_type = BoundaryType.LOWER_TOP_CONVECTION

    else:
        boundary_type = BoundaryType.UNKNOWN

    classification = FaceClassification(
        boundary_type=boundary_type,
        direction=direction,
        x=x,
        y=y,
    )

    if boundary_type == BoundaryType.UNKNOWN:
        message = (
            "Could not classify missing-neighbor face. "
            f"x = {x:.6f}, y = {y:.6f}, direction = {direction.value}"
        )
        raise ValueError(message)

    return classification


def get_neighbor_offset(direction: Direction) -> Tuple[int, int]:
    """
    Return array-index offset for a neighbor direction.

    Array convention:
    - i is y-index
    - j is x-index
    - increasing i means increasing y because y_coordinates are stored ascending

    Parameters
    ----------
    direction : Direction
        Neighbor direction.

    Returns
    -------
    tuple of int
        di, dj offsets.
    """

    if direction == Direction.EAST:
        return 0, 1

    if direction == Direction.WEST:
        return 0, -1

    if direction == Direction.NORTH:
        return 1, 0

    if direction == Direction.SOUTH:
        return -1, 0

    raise ValueError(f"Unknown direction: {direction}")


def create_coordinate_array(
    minimum_value: float,
    maximum_value: float,
    spacing: float,
    tolerance: float,
) -> np.ndarray:
    """
    Create a coordinate array that includes the maximum endpoint.

    Parameters
    ----------
    minimum_value : float
        Minimum coordinate value.
    maximum_value : float
        Maximum coordinate value.
    spacing : float
        Grid spacing.
    tolerance : float
        Floating point tolerance.

    Returns
    -------
    np.ndarray
        Coordinate array.
    """

    if spacing <= 0.0:
        raise ValueError("Grid spacing must be positive.")

    number_of_intervals_float = (maximum_value - minimum_value) / spacing
    number_of_intervals = int(round(number_of_intervals_float))

    reconstructed_maximum = minimum_value + number_of_intervals * spacing

    if abs(reconstructed_maximum - maximum_value) > tolerance:
        raise ValueError(
            "Grid spacing does not divide the geometry length exactly. "
            f"minimum = {minimum_value}, maximum = {maximum_value}, "
            f"spacing = {spacing}, reconstructed maximum = {reconstructed_maximum}"
        )

    coordinates = minimum_value + spacing * np.arange(number_of_intervals + 1)

    return coordinates


def create_grid(
    geometry: GeometryParameters,
    simulation: SimulationParameters,
) -> Grid:
    """
    Create the full rectangular grid and mark active stepped-domain nodes.

    Parameters
    ----------
    geometry : GeometryParameters
        Geometry settings.
    simulation : SimulationParameters
        Simulation settings.

    Returns
    -------
    Grid
        Grid object containing coordinates, active mask, and index map.
    """

    dx = simulation.dx
    dy = simulation.dy
    tol = geometry.tolerance

    x_coordinates = create_coordinate_array(
        minimum_value=0.0,
        maximum_value=geometry.total_length,
        spacing=dx,
        tolerance=tol,
    )

    y_coordinates = create_coordinate_array(
        minimum_value=0.0,
        maximum_value=geometry.total_height,
        spacing=dy,
        tolerance=tol,
    )

    x_mesh, y_mesh = np.meshgrid(x_coordinates, y_coordinates)

    active_mask = np.zeros_like(x_mesh, dtype=bool)

    for i in range(y_mesh.shape[0]):
        for j in range(x_mesh.shape[1]):
            x = float(x_mesh[i, j])
            y = float(y_mesh[i, j])
            active_mask[i, j] = is_active_node(x, y, geometry)

    index_map = -np.ones_like(active_mask, dtype=int)

    active_indices = np.argwhere(active_mask)

    for matrix_index, index_pair in enumerate(active_indices):
        i, j = index_pair
        index_map[i, j] = matrix_index

    active_node_count = int(np.sum(active_mask))

    grid = Grid(
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        x_mesh=x_mesh,
        y_mesh=y_mesh,
        active_mask=active_mask,
        index_map=index_map,
        active_node_count=active_node_count,
        dx=dx,
        dy=dy,
    )

    return grid


def get_direction_data() -> Dict[Direction, Tuple[int, int]]:
    """
    Return direction-to-index-offset dictionary.

    Returns
    -------
    dict
        Dictionary mapping Direction to (di, dj).
    """

    return {
        Direction.EAST: get_neighbor_offset(Direction.EAST),
        Direction.WEST: get_neighbor_offset(Direction.WEST),
        Direction.NORTH: get_neighbor_offset(Direction.NORTH),
        Direction.SOUTH: get_neighbor_offset(Direction.SOUTH),
    }


def get_neighbor_information(
    grid: Grid,
    i: int,
    j: int,
    direction: Direction,
    geometry: GeometryParameters,
) -> Tuple[bool, Optional[int], Optional[int], Optional[int], Optional[FaceClassification]]:
    """
    Return information about a neighbor in a given direction.

    Parameters
    ----------
    grid : Grid
        Grid object.
    i : int
        Current y-index.
    j : int
        Current x-index.
    direction : Direction
        Neighbor direction.
    geometry : GeometryParameters
        Geometry settings.

    Returns
    -------
    tuple
        has_active_neighbor, neighbor_i, neighbor_j, neighbor_matrix_index, face_classification

    Notes
    -----
    If the neighbor is active:
        has_active_neighbor = True
        face_classification = None

    If the neighbor is missing:
        has_active_neighbor = False
        neighbor indices and matrix index are None
        face_classification describes the boundary face
    """

    di, dj = get_neighbor_offset(direction)

    neighbor_i = i + di
    neighbor_j = j + dj

    if grid.is_active_index(neighbor_i, neighbor_j):
        neighbor_matrix_index = grid.get_matrix_index(neighbor_i, neighbor_j)

        return (
            True,
            neighbor_i,
            neighbor_j,
            neighbor_matrix_index,
            None,
        )

    x, y = grid.get_coordinates(i, j)

    face_classification = classify_missing_neighbor_face(
        x=x,
        y=y,
        direction=direction,
        geometry=geometry,
    )

    return (
        False,
        None,
        None,
        None,
        face_classification,
    )


def create_temperature_field_with_nan(
    grid: Grid,
    active_temperature_vector: np.ndarray,
) -> np.ndarray:
    """
    Convert active-node temperature vector into a 2D array with NaN inactive cells.

    Parameters
    ----------
    grid : Grid
        Grid object.
    active_temperature_vector : np.ndarray
        1D temperature vector containing active-node temperatures only.

    Returns
    -------
    np.ndarray
        2D temperature field. Inactive nodes are NaN.
    """

    if len(active_temperature_vector) != grid.active_node_count:
        raise ValueError(
            "Temperature vector length does not match active node count. "
            f"Expected {grid.active_node_count}, got {len(active_temperature_vector)}."
        )

    temperature_field = np.full(grid.shape, np.nan, dtype=float)

    for i, j, matrix_index, _, _ in grid.iter_active_nodes():
        temperature_field[i, j] = active_temperature_vector[matrix_index]

    return temperature_field


def print_grid_summary(grid: Grid) -> None:
    """
    Print basic grid information and active-node sanity checks.

    Parameters
    ----------
    grid : Grid
        Grid object.
    """

    total_node_count = grid.number_of_x_nodes * grid.number_of_y_nodes
    inactive_node_count = total_node_count - grid.active_node_count

    print("\nGrid Summary")
    print("-" * 70)
    print(f"  Number of x nodes       : {grid.number_of_x_nodes}")
    print(f"  Number of y nodes       : {grid.number_of_y_nodes}")
    print(f"  Total rectangular nodes : {total_node_count}")
    print(f"  Active solid nodes      : {grid.active_node_count}")
    print(f"  Inactive nodes          : {inactive_node_count}")
    print(f"  dx                      : {grid.dx:.6f} m")
    print(f"  dy                      : {grid.dy:.6f} m")
    print("-" * 70)