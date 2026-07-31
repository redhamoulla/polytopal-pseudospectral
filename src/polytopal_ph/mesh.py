#!/usr/bin/env python3
"""Power-preserving assembly of compatible polygonal pseudospectral cells.

Each cell is the local model built in :mod:`polytopal_ph`.  Shared edges are
interconnected through a common pressure effort and the constraint that the
sum of outward fluxes vanishes.  Eliminating that constraint with an
orthonormal null-space basis produces a reduced skew-symmetric operator.

The module validates:

* exact cancellation of interface power;
* maximality through the graph of a square skew matrix;
* an acoustic plane wave on a pentagon split into two cells;
* p-convergence of the state and interface pressure;
* the loss of exponential p-convergence for a re-entrant-corner singularity.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-multicell-ph")

import matplotlib
import numpy as np
import scipy.linalg as la
from numpy.polynomial.legendre import Legendre

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .cell import (
    BoundaryQuadrature,
    PlaneWave,
    PolygonComplex,
    acoustic_matrices,
    build_nested_frames,
    build_polygon_complex,
    ensure_ccw,
    exact_semidiscrete_plane_wave,
    plane_wave_errors,
    polygon_quadrature,
    project_plane_wave,
    regular_test_pentagon,
)


Array = np.ndarray


def split_test_pentagon() -> list[Array]:
    """Split the reference pentagon along the diagonal from vertex 0 to 2."""

    vertices = regular_test_pentagon()
    return [
        vertices[[0, 1, 2]],
        vertices[[0, 2, 3, 4]],
    ]


def edge_key(start: Array, end: Array, decimals: int = 12) -> tuple[tuple[float, float], tuple[float, float]]:
    first = tuple(np.round(start, decimals=decimals))
    second = tuple(np.round(end, decimals=decimals))
    return tuple(sorted((first, second)))  # type: ignore[return-value]


@dataclass
class EdgePort:
    cell_index: int
    local_edge: int
    key: tuple[tuple[float, float], tuple[float, float]]
    basis_values: Array
    weights: Array
    sample_points: Array
    local_trace: Array
    flux_projection: Array
    local_input: Array

    @property
    def dimension(self) -> int:
        return self.flux_projection.shape[0]

    def project_scalar_samples(self, samples: Array) -> Array:
        return self.basis_values.T @ (self.weights * samples)


def canonical_edge_basis(
    points: Array,
    start: Array,
    end: Array,
    degree: int,
) -> Array:
    """Orthonormal Legendre basis using a geometry-defined global orientation."""

    if tuple(start) <= tuple(end):
        origin, target = start, end
    else:
        origin, target = end, start
    vector = target - origin
    length = float(np.linalg.norm(vector))
    coordinate = 2.0 * ((points - origin[None, :]) @ vector) / length**2 - 1.0
    return np.column_stack(
        [
            np.sqrt((2 * order + 1) / length)
            * Legendre.basis(order)(coordinate)
            for order in range(degree + 1)
        ]
    )


def build_edge_port(
    complex_: PolygonComplex,
    cell_index: int,
    local_edge: int,
) -> EdgePort:
    mask = complex_.boundary.edge_ids == local_edge
    points = complex_.boundary.points[mask]
    weights = complex_.boundary.weights[mask]
    trace = complex_.trace1_modal[mask]
    start = complex_.vertices[local_edge]
    end = complex_.vertices[(local_edge + 1) % len(complex_.vertices)]
    basis = canonical_edge_basis(
        points, start, end, degree=complex_.degree - 1
    )
    flux_projection = basis.T @ (weights[:, None] * trace)
    local_input = np.vstack(
        (
            np.zeros((complex_.n2, complex_.degree)),
            -flux_projection.T,
        )
    )
    return EdgePort(
        cell_index=cell_index,
        local_edge=local_edge,
        key=edge_key(start, end),
        basis_values=basis,
        weights=weights,
        sample_points=points,
        local_trace=trace,
        flux_projection=flux_projection,
        local_input=local_input,
    )


@dataclass
class Interface:
    key: tuple[tuple[float, float], tuple[float, float]]
    occurrences: tuple[EdgePort, EdgePort]
    column_slice: slice


@dataclass
class MultiCellComplex:
    degree: int
    cells: list[PolygonComplex]
    offsets: Array
    edge_ports: list[EdgePort]
    interfaces: list[Interface]
    boundary_ports: list[EdgePort]
    internal_operator: Array
    interface_input: Array
    boundary_input: Array
    constraint_basis: Array
    reduced_operator: Array
    reduced_boundary_input: Array

    @property
    def full_dimension(self) -> int:
        return self.internal_operator.shape[0]

    @property
    def reduced_dimension(self) -> int:
        return self.reduced_operator.shape[0]

    @property
    def interface_dimension(self) -> int:
        return self.interface_input.shape[1]

    @property
    def boundary_dimension(self) -> int:
        return self.boundary_input.shape[1]


def insert_local_block(
    global_matrix: Array,
    local_matrix: Array,
    row_start: int,
    column_start: int,
) -> None:
    rows, columns = local_matrix.shape
    global_matrix[
        row_start : row_start + rows,
        column_start : column_start + columns,
    ] += local_matrix


def assemble_cells(
    polygons: list[Array],
    degree: int,
) -> MultiCellComplex:
    """Assemble equal-order cells through conservative hybrid interfaces."""

    cells = [
        build_polygon_complex(degree, vertices=ensure_ccw(vertices))
        for vertices in polygons
    ]
    state_sizes = np.array([cell.n2 + cell.n1 for cell in cells], dtype=int)
    offsets = np.concatenate(([0], np.cumsum(state_sizes)))
    full_dimension = int(offsets[-1])
    internal_operator = la.block_diag(
        *(acoustic_matrices(cell)[0] for cell in cells)
    )

    occurrences: dict[
        tuple[tuple[float, float], tuple[float, float]], list[EdgePort]
    ] = {}
    edge_ports: list[EdgePort] = []
    for cell_index, cell in enumerate(cells):
        for local_edge in range(len(cell.vertices)):
            port = build_edge_port(cell, cell_index, local_edge)
            edge_ports.append(port)
            occurrences.setdefault(port.key, []).append(port)

    interface_groups = [
        (key, ports) for key, ports in occurrences.items() if len(ports) == 2
    ]
    boundary_ports = [
        ports[0] for ports in occurrences.values() if len(ports) == 1
    ]
    invalid = {key: len(ports) for key, ports in occurrences.items() if len(ports) > 2}
    if invalid:
        raise ValueError(f"non-manifold edges: {invalid}")

    interface_dimension = degree * len(interface_groups)
    boundary_dimension = degree * len(boundary_ports)
    interface_input = np.zeros((full_dimension, interface_dimension))
    boundary_input = np.zeros((full_dimension, boundary_dimension))
    interfaces: list[Interface] = []

    for interface_index, (key, ports) in enumerate(interface_groups):
        column_slice = slice(
            interface_index * degree, (interface_index + 1) * degree
        )
        for port in ports:
            insert_local_block(
                interface_input,
                port.local_input,
                int(offsets[port.cell_index]),
                column_slice.start,
            )
        interfaces.append(
            Interface(
                key=key,
                occurrences=(ports[0], ports[1]),
                column_slice=column_slice,
            )
        )

    for boundary_index, port in enumerate(boundary_ports):
        column_start = boundary_index * degree
        insert_local_block(
            boundary_input,
            port.local_input,
            int(offsets[port.cell_index]),
            column_start,
        )

    if interface_dimension:
        column_norms = np.linalg.norm(interface_input, axis=0)
        if np.any(column_norms == 0.0):
            raise RuntimeError("interface input matrix has a zero column")
        equilibrated_input = interface_input / column_norms[None, :]
        singular_values = la.svdvals(equilibrated_input)
        relative_tolerance = (
            1e3
            * max(equilibrated_input.shape)
            * np.finfo(float).eps
        )
        tolerance = relative_tolerance * singular_values[0]
        if int(np.sum(singular_values > tolerance)) != interface_dimension:
            raise RuntimeError("interface input matrix is not full column rank")
        constraint_basis = la.null_space(
            equilibrated_input.T,
            rcond=relative_tolerance,
        )
    else:
        constraint_basis = np.eye(full_dimension)

    reduced_operator = constraint_basis.T @ internal_operator @ constraint_basis
    reduced_boundary_input = constraint_basis.T @ boundary_input
    return MultiCellComplex(
        degree=degree,
        cells=cells,
        offsets=offsets,
        edge_ports=edge_ports,
        interfaces=interfaces,
        boundary_ports=boundary_ports,
        internal_operator=internal_operator,
        interface_input=interface_input,
        boundary_input=boundary_input,
        constraint_basis=constraint_basis,
        reduced_operator=reduced_operator,
        reduced_boundary_input=reduced_boundary_input,
    )


def project_multicell_plane_wave(
    mesh: MultiCellComplex,
    wave: PlaneWave,
    time: float,
) -> tuple[Array, Array]:
    """Return raw local projections and constrained reduced coordinates."""

    raw_state = np.concatenate(
        [project_plane_wave(cell, wave, time=time) for cell in mesh.cells]
    )
    reduced = mesh.constraint_basis.T @ raw_state
    return raw_state, reduced


def harmonic_boundary_coefficients(
    mesh: MultiCellComplex,
    wave: PlaneWave,
) -> tuple[Array, Array]:
    cosine_blocks: list[Array] = []
    sine_blocks: list[Array] = []
    for port in mesh.boundary_ports:
        phase = port.sample_points @ wave.wave_vector
        cosine_blocks.append(port.project_scalar_samples(np.cos(phase)))
        sine_blocks.append(port.project_scalar_samples(np.sin(phase)))
    return np.concatenate(cosine_blocks), np.concatenate(sine_blocks)


def boundary_coefficients_at_time(
    mesh: MultiCellComplex,
    wave: PlaneWave,
    time: float,
) -> Array:
    return np.concatenate(
        [
            port.project_scalar_samples(
                wave.pressure(port.sample_points, time)
            )
            for port in mesh.boundary_ports
        ]
    )


def exact_multicell_plane_wave(
    mesh: MultiCellComplex,
    wave: PlaneWave,
    final_time: float,
) -> tuple[Array, Array]:
    """Exact-in-time integration of the reduced forced linear system."""

    _, initial = project_multicell_plane_wave(mesh, wave, time=0.0)
    cosine, sine = harmonic_boundary_coefficients(mesh, wave)
    dimension = mesh.reduced_dimension
    augmented = np.zeros((dimension + 2, dimension + 2))
    augmented[:dimension, :dimension] = mesh.reduced_operator
    augmented[:dimension, dimension] = (
        mesh.reduced_boundary_input @ cosine
    )
    augmented[:dimension, dimension + 1] = (
        mesh.reduced_boundary_input @ sine
    )
    augmented[dimension, dimension + 1] = -wave.omega
    augmented[dimension + 1, dimension] = wave.omega
    initial_augmented = np.concatenate((initial, [1.0, 0.0]))
    final_augmented = la.expm(final_time * augmented) @ initial_augmented
    reduced_final = final_augmented[:dimension]
    full_final = mesh.constraint_basis @ reduced_final
    return reduced_final, full_final


def multicell_plane_wave_errors(
    mesh: MultiCellComplex,
    wave: PlaneWave,
    full_state: Array,
    final_time: float,
) -> dict[str, float]:
    pressure_error_sq = 0.0
    pressure_norm_sq = 0.0
    flux_error_sq = 0.0
    flux_norm_sq = 0.0
    for cell_index, cell in enumerate(mesh.cells):
        start = int(mesh.offsets[cell_index])
        stop = int(mesh.offsets[cell_index + 1])
        state = full_state[start:stop]
        points, weights = polygon_quadrature(
            cell.vertices, max(2 * mesh.degree + 10, 30)
        )
        frame1 = cell.frames[mesh.degree - 1]
        frame2 = cell.frames[mesh.degree - 2]
        q = state[: cell.n2]
        beta_x = state[cell.n2 : cell.n2 + cell.scalar_n1]
        beta_y = state[cell.n2 + cell.scalar_n1 :]
        numerical_pressure = frame2.values(points) @ q
        numerical_x = frame1.values(points) @ beta_x
        numerical_y = frame1.values(points) @ beta_y
        exact_pressure = wave.pressure(points, final_time)
        exact_x, exact_y = wave.flux_form(points, final_time)
        pressure_error_sq += float(
            np.sum(weights * (numerical_pressure - exact_pressure) ** 2)
        )
        pressure_norm_sq += float(np.sum(weights * exact_pressure**2))
        flux_error_sq += float(
            np.sum(
                weights
                * (
                    (numerical_x - exact_x) ** 2
                    + (numerical_y - exact_y) ** 2
                )
            )
        )
        flux_norm_sq += float(
            np.sum(weights * (exact_x**2 + exact_y**2))
        )
    return {
        "pressure_l2_relative": float(
            np.sqrt(pressure_error_sq / pressure_norm_sq)
        ),
        "flux_l2_relative": float(np.sqrt(flux_error_sq / flux_norm_sq)),
    }


def recover_interface_effort(
    mesh: MultiCellComplex,
    full_state: Array,
    boundary_effort: Array,
) -> Array:
    """Recover the common interface pressure multiplier by scaled least squares."""

    if mesh.interface_dimension == 0:
        return np.empty(0)
    right_hand_side = mesh.internal_operator @ full_state
    right_hand_side += mesh.boundary_input @ boundary_effort
    column_norms = np.linalg.norm(mesh.interface_input, axis=0)
    if np.any(column_norms == 0.0):
        raise RuntimeError("interface input matrix has a zero column")
    equilibrated_input = mesh.interface_input / column_norms[None, :]
    relative_tolerance = (
        1e3
        * max(equilibrated_input.shape)
        * np.finfo(float).eps
    )
    equilibrated_multiplier, _, rank, _ = la.lstsq(
        equilibrated_input,
        -right_hand_side,
        cond=relative_tolerance,
        lapack_driver="gelsd",
    )
    if rank != mesh.interface_dimension:
        raise RuntimeError("equilibrated interface solve lost column rank")
    return equilibrated_multiplier / column_norms


def exact_interface_pressure(
    mesh: MultiCellComplex,
    wave: PlaneWave,
    time: float,
) -> Array:
    blocks = []
    for interface in mesh.interfaces:
        reference = interface.occurrences[0]
        blocks.append(
            reference.project_scalar_samples(
                wave.pressure(reference.sample_points, time)
            )
        )
    return np.concatenate(blocks) if blocks else np.empty(0)


def interface_diagnostics(
    mesh: MultiCellComplex,
    wave: PlaneWave,
    full_state: Array,
    final_time: float,
) -> dict[str, float]:
    """Return interface-effort error and null-space assembly residuals.

    The historical ``interface_flux_jump_*`` keys below measure
    ``G.T @ full_state`` in coefficient space.  They are normalized
    interface-constraint residuals, not independently sampled edge-L2
    flux-jump errors.  The names are retained for snapshot compatibility.
    """

    boundary_effort = boundary_coefficients_at_time(mesh, wave, final_time)
    multiplier = recover_interface_effort(
        mesh, full_state, boundary_effort
    )
    exact_multiplier = exact_interface_pressure(
        mesh, wave, final_time
    )
    constraint_residual = mesh.interface_input.T @ full_state
    multiplier_error = np.linalg.norm(multiplier - exact_multiplier)
    multiplier_norm = np.linalg.norm(exact_multiplier)
    interface_power = float(multiplier @ constraint_residual)
    return {
        "interface_flux_jump_abs": float(
            np.linalg.norm(constraint_residual)
        ),
        "interface_flux_jump_relative": float(
            np.linalg.norm(constraint_residual)
            / max(1.0, np.linalg.norm(full_state))
        ),
        "interface_pressure_l2_relative": float(
            multiplier_error / max(np.finfo(float).eps, multiplier_norm)
        ),
        "interface_power_abs": abs(interface_power),
    }


def multicell_structural_diagnostics(
    mesh: MultiCellComplex,
) -> dict[str, float]:
    skew_defect = mesh.reduced_operator + mesh.reduced_operator.T
    interface_constraint = (
        mesh.interface_input.T @ mesh.constraint_basis
    )
    if mesh.boundary_dimension:
        open_dirac = np.block(
            [
                [mesh.reduced_operator, mesh.reduced_boundary_input],
                [
                    -mesh.reduced_boundary_input.T,
                    np.zeros(
                        (mesh.boundary_dimension, mesh.boundary_dimension)
                    ),
                ],
            ]
        )
        open_skew = open_dirac + open_dirac.T
        open_scale = np.linalg.norm(open_dirac, ord=np.inf)
    else:
        open_skew = skew_defect
        open_scale = np.linalg.norm(mesh.reduced_operator, ord=np.inf)
    return {
        "degree": float(mesh.degree),
        "cell_count": float(len(mesh.cells)),
        "full_state_dimension": float(mesh.full_dimension),
        "interface_dimension": float(mesh.interface_dimension),
        "reduced_state_dimension": float(mesh.reduced_dimension),
        "boundary_port_dimension": float(mesh.boundary_dimension),
        "constraint_rank": float(
            np.linalg.matrix_rank(mesh.interface_input)
        ),
        "constraint_basis_rel": float(
            np.linalg.norm(interface_constraint, ord=np.inf)
            / max(1.0, np.linalg.norm(mesh.interface_input, ord=np.inf))
        ),
        "reduced_skew_rel": float(
            np.linalg.norm(skew_defect, ord=np.inf)
            / max(1.0, np.linalg.norm(mesh.reduced_operator, ord=np.inf))
        ),
        "open_dirac_skew_rel": float(
            np.linalg.norm(open_skew, ord=np.inf)
            / max(1.0, open_scale)
        ),
    }


def midpoint_boundary_power(
    mesh: MultiCellComplex,
    wave: PlaneWave,
    final_time: float = 1.0,
    time_step: float = 0.002,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    _, state = project_multicell_plane_wave(mesh, wave, time=0.0)
    steps = int(round(final_time / time_step))
    time_step = final_time / steps
    identity = np.eye(mesh.reduced_dimension)
    left = identity - 0.5 * time_step * mesh.reduced_operator
    right = identity + 0.5 * time_step * mesh.reduced_operator
    lu, piv = la.lu_factor(left)
    initial_energy = 0.5 * float(state @ state)
    work = 0.0
    rows = [
        {
            "time": 0.0,
            "energy": initial_energy,
            "boundary_work": 0.0,
            "balance_defect": 0.0,
        }
    ]
    for step in range(steps):
        midpoint_time = (step + 0.5) * time_step
        effort = boundary_coefficients_at_time(
            mesh, wave, midpoint_time
        )
        next_state = la.lu_solve(
            (lu, piv),
            right @ state
            + time_step * (mesh.reduced_boundary_input @ effort),
        )
        midpoint_state = 0.5 * (state + next_state)
        output = mesh.reduced_boundary_input.T @ midpoint_state
        work += time_step * float(effort @ output)
        state = next_state
        energy = 0.5 * float(state @ state)
        rows.append(
            {
                "time": (step + 1) * time_step,
                "energy": energy,
                "boundary_work": work,
                "balance_defect": energy - initial_energy - work,
            }
        )
    defect = rows[-1]["balance_defect"]
    return {
        "initial_energy": initial_energy,
        "final_energy": rows[-1]["energy"],
        "cumulative_boundary_work": work,
        "absolute_balance_defect": abs(defect),
        "relative_balance_defect": abs(defect)
        / max(1.0, initial_energy, abs(work)),
        "time_step": time_step,
        "steps": float(steps),
    }, rows


def l_shape_cells() -> list[Array]:
    return [
        np.array([[-1.0, -1.0], [0.0, -1.0], [0.0, 0.0], [-1.0, 0.0]]),
        np.array([[-1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [-1.0, 1.0]]),
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    ]


def corner_singular_function(points: Array) -> Array:
    radius = np.linalg.norm(points, axis=1)
    theta = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2.0 * np.pi)
    values = np.zeros_like(radius)
    nonzero = radius > 0
    values[nonzero] = (
        radius[nonzero] ** (2.0 / 3.0)
        * np.sin(2.0 * theta[nonzero] / 3.0)
    )
    return values


def analytic_test_function(points: Array) -> Array:
    x = points[:, 0]
    y = points[:, 1]
    return np.exp(0.24 * x - 0.17 * y) * np.cos(0.9 * x + 0.6 * y)


def piecewise_projection_error(
    polygons: list[Array],
    degree: int,
    function,
    quadrature_order: int,
) -> float:
    error_sq = 0.0
    norm_sq = 0.0
    for polygon in polygons:
        frames, _, _ = build_nested_frames(polygon, degree)
        frame = frames[degree]
        points, weights = polygon_quadrature(polygon, quadrature_order)
        values = function(points)
        phi = frame.values(points)
        coefficients = phi.T @ (weights * values)
        numerical = phi @ coefficients
        error_sq += float(np.sum(weights * (numerical - values) ** 2))
        norm_sq += float(np.sum(weights * values**2))
    return float(np.sqrt(error_sq / norm_sq))


def corner_convergence(
    degrees: range,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    cells = l_shape_cells()
    rows = []
    for degree in degrees:
        order = max(4 * degree + 36, 72)
        rows.append(
            {
                "degree": float(degree),
                "analytic_l2_relative": piecewise_projection_error(
                    cells, degree, analytic_test_function, order
                ),
                "corner_l2_relative": piecewise_projection_error(
                    cells, degree, corner_singular_function, order
                ),
            }
        )
    fit_rows = [row for row in rows if row["degree"] >= 5]
    analytic_fit_rows = [
        row for row in rows if 5 <= row["degree"] <= 10
    ]
    degree_values = np.array([row["degree"] for row in fit_rows])
    corner_values = np.array(
        [row["corner_l2_relative"] for row in fit_rows]
    )
    analytic_degree_values = np.array(
        [row["degree"] for row in analytic_fit_rows]
    )
    analytic_values = np.array(
        [row["analytic_l2_relative"] for row in analytic_fit_rows]
    )
    algebraic_slope, _ = np.polyfit(
        np.log(degree_values), np.log(corner_values), 1
    )
    exponential_slope, _ = np.polyfit(
        analytic_degree_values, np.log10(analytic_values), 1
    )
    return rows, {
        "corner_algebraic_order": float(-algebraic_slope),
        "analytic_digits_per_degree": float(-exponential_slope),
    }


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_mesh(mesh: MultiCellComplex, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    colors = ["#dbeafe", "#dcfce7", "#fef3c7", "#fce7f3"]
    for index, cell in enumerate(mesh.cells):
        closed = np.vstack((cell.vertices, cell.vertices[0]))
        axis.fill(
            closed[:, 0],
            closed[:, 1],
            color=colors[index % len(colors)],
            alpha=0.8,
        )
        axis.plot(closed[:, 0], closed[:, 1], color="#1f2937", lw=1.5)
        center = np.mean(cell.vertices, axis=0)
        axis.text(center[0], center[1], rf"$K_{index + 1}$", ha="center")
    for interface in mesh.interfaces:
        endpoints = np.asarray(interface.key)
        axis.plot(
            endpoints[:, 0],
            endpoints[:, 1],
            color="#dc2626",
            lw=3,
            label="interface interne",
        )
    axis.set_aspect("equal")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("Pentagone décomposé en deux cellules")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles[:1], labels[:1], frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_multicell_convergence(
    rows: list[dict[str, float]], output: Path
) -> None:
    degree = np.array([row["degree"] for row in rows])
    figure, axis = plt.subplots(figsize=(7.4, 4.7), constrained_layout=True)
    series = [
        ("multicell_pressure", "Pression, deux cellules", "o"),
        ("single_pressure", "Pression, cellule unique", "s"),
        ("multicell_flux", "Flux, deux cellules", "^"),
        ("interface_pressure", "Pression d’interface", "D"),
    ]
    for key, label, marker in series:
        values = np.maximum(np.array([row[key] for row in rows]), 5e-16)
        axis.semilogy(degree, values, marker=marker, lw=1.6, label=label)
    axis.set_xlabel("Degré polynomial N")
    axis.set_ylabel("Erreur relative L²")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("Convergence après assemblage port-Hamiltonien")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_corner_convergence(
    rows: list[dict[str, float]], output: Path
) -> None:
    degree = np.array([row["degree"] for row in rows])
    analytic = np.array([row["analytic_l2_relative"] for row in rows])
    corner = np.array([row["corner_l2_relative"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), constrained_layout=True)
    axes[0].semilogy(degree, analytic, "o-", label="fonction analytique")
    axes[0].semilogy(degree, corner, "s-", label="coin rentrant")
    axes[0].set_xlabel("Degré polynomial N")
    axes[0].set_ylabel("Erreur relative L²")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].set_title("Échelle semi-logarithmique")
    axes[1].loglog(degree, corner, "s-", color="#b91c1c")
    axes[1].set_xlabel("Degré polynomial N")
    axes[1].set_ylabel("Erreur du mode singulier")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].set_title("Comportement algébrique")
    figure.suptitle("Effet d’un coin rentrant sur la convergence en p")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def run_multicell_validation(
    output_directory: Path,
    degrees: range = range(2, 12),
    final_time: float = 0.65,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    figures = output_directory / "figures"
    wave = PlaneWave(np.array([1.35, -0.92]))
    convergence_rows: list[dict[str, float]] = []
    structure_rows: list[dict[str, float]] = []
    selected_mesh: MultiCellComplex | None = None

    for degree in degrees:
        mesh = assemble_cells(split_test_pentagon(), degree)
        structure = multicell_structural_diagnostics(mesh)
        _, full_state = exact_multicell_plane_wave(
            mesh, wave, final_time=final_time
        )
        errors = multicell_plane_wave_errors(
            mesh, wave, full_state, final_time=final_time
        )
        interface = interface_diagnostics(
            mesh, wave, full_state, final_time=final_time
        )
        single = build_polygon_complex(degree)
        single_state = exact_semidiscrete_plane_wave(
            single, wave, final_time=final_time
        )
        single_errors = plane_wave_errors(
            single, wave, single_state, final_time=final_time
        )
        convergence_rows.append(
            {
                "degree": float(degree),
                "multicell_pressure": errors["pressure_l2_relative"],
                "multicell_flux": errors["flux_l2_relative"],
                "single_pressure": single_errors["pressure_l2_relative"],
                "single_flux": single_errors["flux_l2_relative"],
                "interface_pressure": interface[
                    "interface_pressure_l2_relative"
                ],
                "interface_flux_jump": interface[
                    "interface_flux_jump_relative"
                ],
                "interface_power_abs": interface["interface_power_abs"],
            }
        )
        structure_rows.append({**structure, **interface})
        selected_mesh = mesh

    if selected_mesh is None:
        raise RuntimeError("empty degree range")

    power_mesh = assemble_cells(split_test_pentagon(), min(8, max(degrees)))
    power_metrics, power_rows = midpoint_boundary_power(
        power_mesh, wave, final_time=1.0, time_step=0.002
    )
    corner_rows, corner_fit = corner_convergence(range(2, 15))

    save_csv(output_directory / "multicell_convergence.csv", convergence_rows)
    save_csv(output_directory / "multicell_structure.csv", structure_rows)
    save_csv(output_directory / "multicell_power.csv", power_rows)
    save_csv(output_directory / "corner_convergence.csv", corner_rows)
    plot_mesh(selected_mesh, figures / "multicell_mesh.png")
    plot_multicell_convergence(
        convergence_rows, figures / "multicell_convergence.png"
    )
    plot_corner_convergence(
        corner_rows, figures / "corner_convergence.png"
    )

    fit_rows = [row for row in convergence_rows if row["degree"] >= 5]
    fit_degree = np.array([row["degree"] for row in fit_rows])
    fits = {}
    for key in (
        "multicell_pressure",
        "multicell_flux",
        "interface_pressure",
    ):
        slope, _ = np.polyfit(
            fit_degree,
            np.log10([row[key] for row in fit_rows]),
            1,
        )
        fits[f"{key}_digits_per_degree"] = float(-slope)

    summary = {
        "cell_polygons": [polygon.tolist() for polygon in split_test_pentagon()],
        "degrees": [int(row["degree"]) for row in convergence_rows],
        "final_time": final_time,
        "highest_degree_convergence": convergence_rows[-1],
        "highest_degree_structure": structure_rows[-1],
        "empirical_convergence": fits,
        "power_test": power_metrics,
        "corner_convergence": corner_fit,
        "highest_corner_errors": corner_rows[-1],
    }
    with (output_directory / "multicell_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results_multicell",
    )
    parser.add_argument("--degree-min", type=int, default=2)
    parser.add_argument("--degree-max", type=int, default=11)
    parser.add_argument("--final-time", type=float, default=0.65)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_multicell_validation(
        args.output,
        degrees=range(args.degree_min, args.degree_max + 1),
        final_time=args.final_time,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
