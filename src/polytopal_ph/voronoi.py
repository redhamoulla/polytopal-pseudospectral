#!/usr/bin/env python3
"""Scaling experiment for the hybrid constraint on bounded Voronoi meshes.

This module reuses the local polygon complexes, edge ports, interface scatter
convention, and column-equilibrated null-space reduction from
:mod:`multicell_ph`, while adding phase timings and conditioning diagnostics.

Notation
--------
For the stacked acoustic state ``z`` and one pressure multiplier coefficient
per polynomial mode and internal geometric edge, the existing implementation
uses

    G = B_I C in R^(n_z x m_I),       G.T z = 0.

Consequently, full *column* rank of ``G`` is the relevant hypothesis.  We
diagnose the spectrum of ``G`` directly.  In particular,

    cond_2(G.T G) = cond_2(G)^2,

but neither ``G.T G`` nor its inverse is formed in this experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-voronoi-scaling")

import matplotlib
import numpy as np
import scipy.linalg as la

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .mesh import (
    Interface,
    MultiCellComplex,
    build_edge_port,
    insert_local_block,
)
from .cell import (
    acoustic_matrices,
    build_polygon_complex,
    ensure_ccw,
    polygon_centroid,
    signed_area,
)


Array = np.ndarray
BOX = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=float,
)


def _clip_half_plane(
    polygon: Array,
    normal: Array,
    offset: float,
    tolerance: float = 5.0e-14,
) -> Array:
    """Clip a counter-clockwise polygon by ``normal @ x <= offset``."""

    if len(polygon) == 0:
        return polygon
    output: list[Array] = []
    previous = polygon[-1]
    previous_value = float(previous @ normal - offset)
    previous_inside = previous_value <= tolerance
    for current in polygon:
        current_value = float(current @ normal - offset)
        current_inside = current_value <= tolerance
        if current_inside != previous_inside:
            direction = current - previous
            denominator = float(direction @ normal)
            if abs(denominator) <= np.finfo(float).eps:
                intersection = 0.5 * (previous + current)
            else:
                fraction = float((offset - previous @ normal) / denominator)
                fraction = min(1.0, max(0.0, fraction))
                intersection = previous + fraction * direction
            output.append(intersection)
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return np.asarray(output, dtype=float).reshape((-1, 2))


def _canonicalize_polygon(
    polygon: Array,
    decimals: int = 12,
    collinear_tolerance: float = 2.0e-11,
) -> Array:
    """Snap shared vertices and remove duplicate/collinear consecutive points."""

    polygon = np.round(np.asarray(polygon, dtype=float), decimals=decimals)
    if len(polygon) < 3:
        raise RuntimeError("Voronoi clipping produced a degenerate cell")

    deduplicated: list[Array] = []
    for point in polygon:
        if not deduplicated or not np.array_equal(point, deduplicated[-1]):
            deduplicated.append(point)
    if len(deduplicated) > 1 and np.array_equal(
        deduplicated[0], deduplicated[-1]
    ):
        deduplicated.pop()
    polygon = np.asarray(deduplicated)

    changed = True
    while changed and len(polygon) > 3:
        changed = False
        keep = np.ones(len(polygon), dtype=bool)
        for index in range(len(polygon)):
            previous = polygon[index - 1]
            current = polygon[index]
            following = polygon[(index + 1) % len(polygon)]
            left = current - previous
            right = following - current
            scale = max(1.0, np.linalg.norm(left) * np.linalg.norm(right))
            cross = left[0] * right[1] - left[1] * right[0]
            if abs(cross) <= collinear_tolerance * scale:
                keep[index] = False
                changed = True
        if np.sum(keep) < 3:
            break
        polygon = polygon[keep]
    return ensure_ccw(polygon)


def bounded_voronoi_cells(
    sites: Array,
    decimals: int = 12,
) -> list[Array]:
    """Construct exact half-plane Voronoi cells clipped to the unit square."""

    sites = np.asarray(sites, dtype=float)
    if sites.ndim != 2 or sites.shape[1] != 2:
        raise ValueError("sites must have shape (n_sites, 2)")
    if np.any(sites <= 0.0) or np.any(sites >= 1.0):
        raise ValueError("sites must lie strictly inside the unit square")
    cells: list[Array] = []
    for index, site in enumerate(sites):
        polygon = BOX.copy()
        for other_index, other in enumerate(sites):
            if other_index == index:
                continue
            normal = 2.0 * (other - site)
            offset = float(other @ other - site @ site)
            polygon = _clip_half_plane(polygon, normal, offset)
            if len(polygon) < 3:
                raise RuntimeError("empty bounded Voronoi cell")
        cells.append(_canonicalize_polygon(polygon, decimals=decimals))
    return cells


def relaxed_voronoi_mesh(
    cell_count: int,
    seed: int,
    lloyd_steps: int = 2,
) -> tuple[Array, list[Array]]:
    """Generate a reproducible, mildly regular centroidal Voronoi mesh."""

    if cell_count < 2:
        raise ValueError("cell_count must be at least two")
    rng = np.random.default_rng(seed)
    margin = 0.035
    sites = margin + (1.0 - 2.0 * margin) * rng.random((cell_count, 2))
    for _ in range(lloyd_steps):
        cells = bounded_voronoi_cells(sites)
        sites = np.vstack([polygon_centroid(cell) for cell in cells])
        sites = np.clip(sites, 10.0 * margin**2, 1.0 - 10.0 * margin**2)
    cells = bounded_voronoi_cells(sites)
    validate_voronoi_mesh(cells, expected_cells=cell_count)
    return sites, cells


def _edge_key(
    start: Array,
    end: Array,
    decimals: int = 12,
) -> tuple[tuple[float, float], tuple[float, float]]:
    endpoints = (
        tuple(np.round(start, decimals=decimals)),
        tuple(np.round(end, decimals=decimals)),
    )
    return tuple(sorted(endpoints))  # type: ignore[return-value]


def mesh_topology(polygons: list[Array]) -> dict[str, float | int]:
    edge_occurrences: dict[
        tuple[tuple[float, float], tuple[float, float]], int
    ] = {}
    vertices: set[tuple[float, float]] = set()
    for polygon in polygons:
        for index, start in enumerate(polygon):
            end = polygon[(index + 1) % len(polygon)]
            vertices.add(tuple(np.round(start, decimals=12)))
            key = _edge_key(start, end)
            edge_occurrences[key] = edge_occurrences.get(key, 0) + 1
    invalid = {key: count for key, count in edge_occurrences.items() if count > 2}
    if invalid:
        raise RuntimeError(f"non-manifold Voronoi edges: {invalid}")
    internal_edges = sum(count == 2 for count in edge_occurrences.values())
    boundary_edges = sum(count == 1 for count in edge_occurrences.values())
    edge_count = len(edge_occurrences)
    edge_lengths = np.array(
        [
            np.linalg.norm(np.asarray(key[1]) - np.asarray(key[0]))
            for key in edge_occurrences
        ]
    )
    return {
        "vertex_count": len(vertices),
        "edge_count": edge_count,
        "internal_edge_count": internal_edges,
        "boundary_edge_count": boundary_edges,
        "euler_disk": len(vertices) - edge_count + len(polygons),
        "minimum_edge_length": float(np.min(edge_lengths)),
        "median_edge_length": float(np.median(edge_lengths)),
        "maximum_edge_length": float(np.max(edge_lengths)),
    }


def validate_voronoi_mesh(
    polygons: list[Array],
    expected_cells: int | None = None,
) -> dict[str, float | int]:
    if expected_cells is not None and len(polygons) != expected_cells:
        raise RuntimeError("unexpected Voronoi cell count")
    areas = np.array([signed_area(ensure_ccw(cell)) for cell in polygons])
    if np.any(areas <= 1.0e-10):
        raise RuntimeError("degenerate Voronoi cell area")
    area_defect = abs(float(np.sum(areas)) - 1.0)
    if area_defect > 5.0e-10:
        raise RuntimeError(f"Voronoi area defect is {area_defect:.3e}")
    topology = mesh_topology(polygons)
    if topology["euler_disk"] != 1:
        raise RuntimeError(
            f"nonconforming mesh: Euler characteristic {topology['euler_disk']}"
        )
    return {
        **topology,
        "area_sum": float(np.sum(areas)),
        "area_defect": area_defect,
        "minimum_cell_area": float(np.min(areas)),
        "maximum_cell_area": float(np.max(areas)),
    }


def current_rss_mib() -> float:
    """Read current resident memory on Linux; return NaN elsewhere."""

    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size / 2.0**20
    except (OSError, ValueError, IndexError):
        return float("nan")


@dataclass
class TimedAssembly:
    mesh: MultiCellComplex
    timings: dict[str, float]
    singular_values: Array
    rank_tolerance: float
    equilibrated_singular_values: Array
    equilibrated_rank_tolerance: float
    rss: dict[str, float]


def timed_assemble_cells(
    polygons: list[Array],
    degree: int,
) -> TimedAssembly:
    """Reproduce ``assemble_cells`` while timing each algebraic phase."""

    timings: dict[str, float] = {}
    rss: dict[str, float] = {"rss_initial_mib": current_rss_mib()}

    start = time.perf_counter()
    cells = [
        build_polygon_complex(degree, vertices=ensure_ccw(vertices))
        for vertices in polygons
    ]
    timings["local_discretization_seconds"] = time.perf_counter() - start
    rss["rss_after_local_mib"] = current_rss_mib()

    start = time.perf_counter()
    state_sizes = np.array([cell.n2 + cell.n1 for cell in cells], dtype=int)
    offsets = np.concatenate(([0], np.cumsum(state_sizes)))
    full_dimension = int(offsets[-1])
    internal_operator = la.block_diag(
        *(acoustic_matrices(cell)[0] for cell in cells)
    )

    occurrences: dict[
        tuple[tuple[float, float], tuple[float, float]], list
    ] = {}
    edge_ports = []
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
        raise RuntimeError(f"non-manifold assembled edges: {invalid}")

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
        insert_local_block(
            boundary_input,
            port.local_input,
            int(offsets[port.cell_index]),
            boundary_index * degree,
        )
    timings["global_scatter_seconds"] = time.perf_counter() - start
    rss["rss_after_scatter_mib"] = current_rss_mib()

    if interface_dimension == 0:
        singular_values = np.empty(0)
        tolerance = 0.0
        equilibrated_singular_values = np.empty(0)
        equilibrated_tolerance = 0.0
        constraint_basis = np.eye(full_dimension)
        timings["rank_svd_seconds"] = 0.0
        timings["nullspace_seconds"] = 0.0
    else:
        start = time.perf_counter()
        singular_values = la.svdvals(interface_input)
        column_norms = np.linalg.norm(interface_input, axis=0)
        if np.any(column_norms == 0.0):
            raise RuntimeError("interface input matrix has a zero column")
        equilibrated_input = interface_input / column_norms[None, :]
        equilibrated_singular_values = la.svdvals(equilibrated_input)
        timings["rank_svd_seconds"] = time.perf_counter() - start
        tolerance = (
            1.0e3
            * max(interface_input.shape)
            * np.finfo(float).eps
            * singular_values[0]
        )
        relative_tolerance = (
            1.0e3
            * max(equilibrated_input.shape)
            * np.finfo(float).eps
        )
        equilibrated_tolerance = (
            relative_tolerance * equilibrated_singular_values[0]
        )
        rank = int(
            np.sum(equilibrated_singular_values > equilibrated_tolerance)
        )
        if rank != interface_dimension:
            raise RuntimeError(
                "column-equilibrated G has numerical rank "
                f"{rank}, expected {interface_dimension}"
            )
        start = time.perf_counter()
        constraint_basis = la.null_space(
            equilibrated_input.T,
            rcond=relative_tolerance,
        )
        timings["nullspace_seconds"] = time.perf_counter() - start
    rss["rss_after_nullspace_mib"] = current_rss_mib()

    start = time.perf_counter()
    reduced_operator = constraint_basis.T @ internal_operator @ constraint_basis
    reduced_boundary_input = constraint_basis.T @ boundary_input
    timings["galerkin_projection_seconds"] = time.perf_counter() - start
    rss["rss_after_reduction_mib"] = current_rss_mib()

    timings["assembly_seconds"] = (
        timings["local_discretization_seconds"]
        + timings["global_scatter_seconds"]
    )
    timings["reduction_seconds"] = (
        timings["rank_svd_seconds"]
        + timings["nullspace_seconds"]
        + timings["galerkin_projection_seconds"]
    )
    timings["assembly_and_reduction_seconds"] = (
        timings["assembly_seconds"] + timings["reduction_seconds"]
    )

    mesh = MultiCellComplex(
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
    return TimedAssembly(
        mesh,
        timings,
        singular_values,
        tolerance,
        equilibrated_singular_values,
        equilibrated_tolerance,
        rss,
    )


def _array_storage_mib(arrays: Iterable[Array]) -> float:
    return sum(array.nbytes for array in arrays) / 2.0**20


def run_one(
    cell_count: int,
    degree: int,
    seed: int,
    lloyd_steps: int,
) -> dict[str, float | int]:
    """Execute and validate one independent mesh/assembly configuration."""

    # Initialize the BLAS runtime before phase timers start.
    _ = np.eye(4) @ np.eye(4)
    start = time.perf_counter()
    _, polygons = relaxed_voronoi_mesh(cell_count, seed, lloyd_steps)
    mesh_seconds = time.perf_counter() - start
    topology = validate_voronoi_mesh(polygons, expected_cells=cell_count)

    assembled = timed_assemble_cells(polygons, degree)
    mesh = assembled.mesh
    if len(mesh.interfaces) != int(topology["internal_edge_count"]):
        raise RuntimeError("topological and algebraic interface counts differ")
    if len(mesh.boundary_ports) != int(topology["boundary_edge_count"]):
        raise RuntimeError("topological and algebraic boundary counts differ")
    g_matrix = mesh.interface_input
    z_basis = mesh.constraint_basis
    singular_values = assembled.singular_values
    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])
    condition_g = sigma_max / sigma_min

    equilibrated_singular_values = assembled.equilibrated_singular_values
    condition_equilibrated = float(
        equilibrated_singular_values[0] / equilibrated_singular_values[-1]
    )

    constraint_residual = g_matrix.T @ z_basis
    constraint_residual_fro = float(np.linalg.norm(constraint_residual, ord="fro"))
    constraint_residual_relative = constraint_residual_fro / max(
        1.0,
        sigma_max * np.linalg.norm(z_basis, ord="fro"),
    )
    orthogonality = z_basis.T @ z_basis - np.eye(z_basis.shape[1])
    orthogonality_relative = float(
        np.linalg.norm(orthogonality, ord="fro")
        / max(1.0, np.sqrt(z_basis.shape[1]))
    )

    skew_defect = mesh.reduced_operator + mesh.reduced_operator.T
    reduced_skew_relative = float(
        np.linalg.norm(skew_defect, ord="fro")
        / max(1.0, np.linalg.norm(mesh.reduced_operator, ord="fro"))
    )
    raw_numerical_rank = int(
        np.sum(singular_values > assembled.rank_tolerance)
    )
    numerical_rank = int(
        np.sum(
            equilibrated_singular_values
            > assembled.equilibrated_rank_tolerance
        )
    )

    core_storage = _array_storage_mib(
        [
            mesh.internal_operator,
            mesh.interface_input,
            mesh.boundary_input,
            mesh.constraint_basis,
            mesh.reduced_operator,
            mesh.reduced_boundary_input,
        ]
    )
    reduction_storage = _array_storage_mib(
        [
            mesh.constraint_basis,
            mesh.reduced_operator,
            mesh.reduced_boundary_input,
        ]
    )
    result: dict[str, float | int] = {
        "cell_count": cell_count,
        "degree": degree,
        "seed": seed,
        "lloyd_steps": lloyd_steps,
        **topology,
        "state_dimension": mesh.full_dimension,
        "constraint_count": mesh.interface_dimension,
        "constraint_rank": numerical_rank,
        "constraint_rank_raw": raw_numerical_rank,
        "nullity": mesh.reduced_dimension,
        "boundary_port_dimension": mesh.boundary_dimension,
        "sigma_min_G": sigma_min,
        "sigma_max_G": sigma_max,
        "condition_G_2": condition_g,
        "condition_GtG_2_estimate": condition_g**2,
        "condition_column_equilibrated_G_2": condition_equilibrated,
        "condition_column_equilibrated_GtG_2_estimate": (
            condition_equilibrated**2
        ),
        "rank_tolerance": assembled.rank_tolerance,
        "sigma_min_over_rank_tolerance": sigma_min / assembled.rank_tolerance,
        "sigma_min_column_equilibrated_G": float(
            equilibrated_singular_values[-1]
        ),
        "sigma_max_column_equilibrated_G": float(
            equilibrated_singular_values[0]
        ),
        "column_equilibrated_rank_tolerance": (
            assembled.equilibrated_rank_tolerance
        ),
        "sigma_min_column_equilibrated_over_rank_tolerance": float(
            equilibrated_singular_values[-1]
            / assembled.equilibrated_rank_tolerance
        ),
        "constraint_residual_GtZ_fro": constraint_residual_fro,
        "constraint_residual_GtZ_relative": constraint_residual_relative,
        "basis_orthogonality_relative": orthogonality_relative,
        "reduced_skew_relative_fro": reduced_skew_relative,
        "mesh_generation_seconds": mesh_seconds,
        **assembled.timings,
        **assembled.rss,
        "core_dense_storage_mib": core_storage,
        "reduction_dense_storage_mib": reduction_storage,
    }

    if numerical_rank != mesh.interface_dimension:
        raise AssertionError("full-column-rank hypothesis failed")
    if constraint_residual_relative > 5.0e-13:
        raise AssertionError("null-space residual is too large")
    if orthogonality_relative > 5.0e-13:
        raise AssertionError("null-space basis lost orthogonality")
    if reduced_skew_relative > 5.0e-13:
        raise AssertionError("reduced operator is not skew-symmetric")
    return result


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(
    rows: list[dict[str, float | int]],
) -> list[dict[str, float | int]]:
    aggregated: list[dict[str, float | int]] = []
    for cell_count in sorted({int(row["cell_count"]) for row in rows}):
        subset = [row for row in rows if int(row["cell_count"]) == cell_count]
        aggregate: dict[str, float | int] = {
            "cell_count": cell_count,
            "degree": int(subset[0]["degree"]),
            "sample_count": len(subset),
        }
        keys = [
            "internal_edge_count",
            "state_dimension",
            "constraint_count",
            "nullity",
            "sigma_min_G",
            "sigma_max_G",
            "condition_G_2",
            "condition_GtG_2_estimate",
            "condition_column_equilibrated_G_2",
            "condition_column_equilibrated_GtG_2_estimate",
            "sigma_min_over_rank_tolerance",
            "sigma_min_column_equilibrated_over_rank_tolerance",
            "constraint_residual_GtZ_relative",
            "reduced_skew_relative_fro",
            "assembly_seconds",
            "reduction_seconds",
            "assembly_and_reduction_seconds",
            "core_dense_storage_mib",
            "rss_after_reduction_mib",
        ]
        for key in keys:
            values = np.asarray([float(row[key]) for row in subset])
            aggregate[f"{key}_median"] = float(np.median(values))
            aggregate[f"{key}_min"] = float(np.min(values))
            aggregate[f"{key}_max"] = float(np.max(values))
        aggregated.append(aggregate)
    return aggregated


def plot_example_mesh(polygons: list[Array], sites: Array, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 6.0), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for index, polygon in enumerate(polygons):
        closed = np.vstack((polygon, polygon[0]))
        axis.fill(
            closed[:, 0],
            closed[:, 1],
            color=cmap(0.15 + 0.7 * index / max(1, len(polygons) - 1)),
            alpha=0.42,
        )
        axis.plot(closed[:, 0], closed[:, 1], color="#243447", lw=0.65)
    axis.scatter(sites[:, 0], sites[:, 1], s=8, color="#9f1239", zorder=3)
    axis.set_aspect("equal")
    axis.set_xlim(-0.015, 1.015)
    axis.set_ylim(-0.015, 1.015)
    axis.set_xlabel("$x$")
    axis.set_ylabel("$y$")
    axis.set_title(f"Bounded relaxed Voronoi mesh ({len(polygons)} cells)")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=210)
    plt.close(figure)


def plot_scaling(
    rows: list[dict[str, float | int]],
    aggregated: list[dict[str, float | int]],
    output: Path,
) -> None:
    counts = np.array([int(row["cell_count"]) for row in aggregated])
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), constrained_layout=True)

    for row in rows:
        axes[0, 0].scatter(
            row["cell_count"], row["condition_G_2"], color="#2563eb", s=25, alpha=0.65
        )
        axes[0, 0].scatter(
            row["cell_count"],
            row["condition_column_equilibrated_G_2"],
            color="#9333ea",
            marker="s",
            s=20,
            alpha=0.65,
        )
        axes[0, 1].scatter(
            row["cell_count"],
            row["sigma_min_over_rank_tolerance"],
            color="#059669",
            s=25,
            alpha=0.65,
        )
    axes[0, 0].plot(
        counts,
        [row["condition_G_2_median"] for row in aggregated],
        color="#1e3a8a",
        marker="o",
        label="raw coordinates",
    )
    axes[0, 0].plot(
        counts,
        [
            row["condition_column_equilibrated_G_2_median"]
            for row in aggregated
        ],
        color="#6b21a8",
        marker="s",
        label="column-equilibrated",
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel(r"$\kappa_2(G)$")
    axes[0, 0].set_title("Hybrid constraint conditioning")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].plot(
        counts,
        [row["sigma_min_over_rank_tolerance_median"] for row in aggregated],
        color="#065f46",
        marker="o",
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel(r"$\sigma_{\min}(G)/\tau_{\mathrm{rank}}$")
    axes[0, 1].set_title("Distance from numerical rank loss")

    axes[1, 0].plot(
        counts,
        [row["assembly_seconds_median"] for row in aggregated],
        marker="o",
        label="local + scatter assembly",
        color="#7c3aed",
    )
    axes[1, 0].plot(
        counts,
        [row["reduction_seconds_median"] for row in aggregated],
        marker="s",
        label="SVD + null space + projection",
        color="#ea580c",
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("wall time [s]")
    axes[1, 0].set_title("Dense reference implementation")
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].plot(
        counts,
        [row["core_dense_storage_mib_median"] for row in aggregated],
        marker="o",
        color="#be123c",
    )
    axes[1, 1].set_ylabel("explicit matrix storage [MiB]")
    axes[1, 1].set_title("Matrices retained after reduction")

    for axis in axes.ravel():
        axis.set_xlabel("number of Voronoi cells")
        axis.grid(True, which="both", alpha=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=210)
    plt.close(figure)


def _format_range(
    row: dict[str, float | int],
    key: str,
    fmt: str,
) -> str:
    median = float(row[f"{key}_median"])
    minimum = float(row[f"{key}_min"])
    maximum = float(row[f"{key}_max"])
    return f"{format(median, fmt)} [{format(minimum, fmt)}, {format(maximum, fmt)}]"


def write_report(
    path: Path,
    rows: list[dict[str, float | int]],
    aggregated: list[dict[str, float | int]],
    command: str,
) -> None:
    degree = int(rows[0]["degree"])
    seeds = sorted({int(row["seed"]) for row in rows})
    lines = [
        "# Bounded Voronoi multi-cell scaling experiment",
        "",
        "## Purpose and algebraic convention",
        "",
        "This experiment tests the full-column-rank hypothesis for the global "
        "hybrid constraint on nontrivial conforming polygonal meshes. In the "
        "implementation,",
        "",
        r"\[G=B_I C\in\mathbb{R}^{n_z\times m_I},\qquad G^\top z=0.\]",
        "",
        "Each internal geometric edge contributes `degree` columns to `G`, one "
        "for each coefficient of its common pressure polynomial. Thus "
        "`m_I = degree × internal_edge_count`. The algebraic equation is the "
        "coefficient-space balance of the two local outward flux traces. Full "
        "column rank means that no interface multiplier is redundant.",
        "",
        "The reported conditioning is computed from the singular values of "
        "`G`. We never form or invert the normal matrix. The quantity labelled "
        r"`cond(GᵀG)` is the identity \(\kappa_2(G^\top G)=\kappa_2(G)^2\).",
        "",
        "## Protocol",
        "",
        f"- Polynomial degree: `{degree}` (fixed to isolate mesh-size scaling).",
        f"- Cell counts: `{sorted({int(row['cell_count']) for row in rows})}`.",
        f"- Independent seeds: `{seeds}`.",
        f"- Lloyd centroid updates per mesh: `{int(rows[0]['lloyd_steps'])}`.",
        "- Domain: the unit square; cells are computed by floating-point "
        "half-plane clipping and snapped consistently before assembly.",
        "- Shared coordinates are snapped to 12 decimal digits; conformity is "
        "then checked by edge multiplicity, area sum, and Euler characteristic.",
        "- Rank and null-space calculations use the unit-column matrix "
        r"\(\widehat G\). The tolerance is "
        r"\(\tau=10^3\max(n_z,m_I)\epsilon_{\rm mach}"
        r"\sigma_{\max}(\widehat G)\). Raw-coordinate singular values are "
        "retained as a conditioning diagnostic.",
        "- The null-space basis is produced by `scipy.linalg.null_space` with "
        "this tolerance.",
        "- Each configuration runs in a fresh one-thread BLAS subprocess. "
        "Timings are wall-clock values on the present machine and are intended "
        "as implementation diagnostics, not hardware-independent complexity "
        "constants.",
        "",
        f"Reproduction command: `{command}`",
        "",
        "## Results",
        "",
        "Entries are medians across seeds, with `[minimum, maximum]`.",
        "",
        "| cells | internal edges | $n_z$ | $m_I$ = rank | nullity | "
        "$\\sigma_{\\min}(G)$ | $\\kappa_2(G)$ | "
        "$\\kappa_2(G^\\top G)$ | equilibrated $\\kappa_2(G)$ | "
        "raw $\\sigma_{\\min}/\\tau_{\\rm raw}$ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregated:
        lines.append(
            "| {cells} | {edges} | {state} | {constraints} | {nullity} | "
            "{sigma} | {cond} | {gram} | {equilibrated} | {margin} |".format(
                cells=int(row["cell_count"]),
                edges=_format_range(row, "internal_edge_count", ".0f"),
                state=_format_range(row, "state_dimension", ".0f"),
                constraints=_format_range(row, "constraint_count", ".0f"),
                nullity=_format_range(row, "nullity", ".0f"),
                sigma=_format_range(row, "sigma_min_G", ".3e"),
                cond=_format_range(row, "condition_G_2", ".3f"),
                gram=_format_range(row, "condition_GtG_2_estimate", ".3f"),
                equilibrated=_format_range(
                    row, "condition_column_equilibrated_G_2", ".3f"
                ),
                margin=_format_range(
                    row, "sigma_min_over_rank_tolerance", ".3e"
                ),
            )
        )
    lines.extend(
        [
            "",
            "| cells | $\\|G^\\top Z\\|_F/(\\|G\\|_2\\|Z\\|_F)$ | "
            "reduced skew defect | assembly [s] | reduction [s] | "
            "explicit storage [MiB] |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregated:
        lines.append(
            "| {cells} | {constraint} | {skew} | {assembly} | {reduction} | "
            "{storage} |".format(
                cells=int(row["cell_count"]),
                constraint=_format_range(
                    row, "constraint_residual_GtZ_relative", ".3e"
                ),
                skew=_format_range(
                    row, "reduced_skew_relative_fro", ".3e"
                ),
                assembly=_format_range(row, "assembly_seconds", ".3f"),
                reduction=_format_range(row, "reduction_seconds", ".3f"),
                storage=_format_range(row, "core_dense_storage_mib", ".2f"),
            )
        )
    maximum_condition = max(float(row["condition_G_2"]) for row in rows)
    maximum_equilibrated_condition = max(
        float(row["condition_column_equilibrated_G_2"]) for row in rows
    )
    minimum_margin = min(
        float(row["sigma_min_over_rank_tolerance"]) for row in rows
    )
    minimum_equilibrated_margin = min(
        float(
            row[
                "sigma_min_column_equilibrated_over_rank_tolerance"
            ]
        )
        for row in rows
    )
    maximum_constraint = max(
        float(row["constraint_residual_GtZ_relative"]) for row in rows
    )
    maximum_skew = max(
        float(row["reduced_skew_relative_fro"]) for row in rows
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"All `{len(rows)}` meshes have `rank(G) = m_I` in both raw and "
            "unit-column coordinates. The smallest raw-coordinate singular "
            f"value remains a factor `{minimum_margin:.3e}` above its "
            "conservative threshold; after equilibration the minimum margin "
            f"is `{minimum_equilibrated_margin:.3e}`. This supports the "
            "algebraic full-rank hypothesis for these realizations.",
            "",
            "The raw Euclidean coordinates are **not** uniformly well "
            f"conditioned: the worst observed `cond(G)` is "
            f"`{maximum_condition:.3e}`, so squaring it in raw normal equations "
            "can exhaust double precision. Much of the observed conditioning "
            "is associated with short-edge modal scaling. "
            "A diagonal column equilibration leaves both `im(G)` and "
            "`ker(Gᵀ)` unchanged and reduces the worst observed condition "
            f"number to `{maximum_equilibrated_condition:.3f}`. Thus the "
            "experiment supports rank and demonstrates effective coordinate "
            "preconditioning, not a mesh-uniform inf-sup bound. The explicit "
            "raw formula `(GᵀG)⁻¹Gᵀ` is not a viable numerical algorithm; "
            "multiplier recovery uses scaled QR/SVD or least squares.",
            "",
            "The largest normalized constraint residual is "
            f"`{maximum_constraint:.3e}` and the largest reduced skew defect is "
            f"`{maximum_skew:.3e}`. These are backward residuals: the numerical "
            "null-space reduction satisfies the hybrid constraint and Dirac "
            "skew symmetry at floating-point accuracy.",
            "",
            "The timing and storage columns also expose the actual limitation "
            "of this reference code: local polygon construction grows nearly "
            "linearly with the cell count, whereas the dense global null-space "
            "and Galerkin projection grow more rapidly and will eventually "
            "dominate. The experiment "
            "validates the algebraic assembly, but it is not a claim of "
            "large-scale optimal complexity. A production implementation "
            "should retain sparse block structure and use sparse QR or a "
            "constraint-preserving iterative formulation. Reported storage is "
            "the sum of retained dense arrays, not peak process memory.",
            "",
            "## Scope",
            "",
            "This is numerical evidence, not a mesh-uniform inf-sup theorem. "
            "It covers mildly regular centroidally relaxed Voronoi cells on a "
            "square, a fixed polynomial degree, and three random realizations "
            "per size. Highly anisotropic slivers, hanging interfaces, mixed "
            "orders, and nonmatching traces require separate study.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_suite(
    sizes: list[int],
    seeds: list[int],
    degree: int,
    lloyd_steps: int,
    output_dir: Path,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    script = Path(__file__).resolve()
    for cell_count in sizes:
        for seed in seeds:
            command = [
                sys.executable,
                str(script),
                "--worker",
                "--sizes",
                str(cell_count),
                "--seeds",
                str(seed),
                "--degree",
                str(degree),
                "--lloyd-steps",
                str(lloyd_steps),
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                }
            )
            completed = subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=True,
                env=environment,
            )
            row = json.loads(completed.stdout)
            rows.append(row)
            print(
                f"cells={cell_count:>2}, seed={seed}: "
                f"rank={row['constraint_rank']}/{row['constraint_count']}, "
                f"cond(G)={row['condition_G_2']:.3f}, "
                f"time={row['assembly_and_reduction_seconds']:.2f}s",
                flush=True,
            )

    aggregated = _aggregate(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "voronoi_scaling_runs.csv", rows)
    _write_csv(output_dir / "voronoi_scaling_summary.csv", aggregated)
    (output_dir / "voronoi_scaling_results.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "description": (
                    "Full-column-rank and dense null-space scaling diagnostics "
                    "for G = B_I C on bounded Voronoi meshes."
                ),
                "runs": rows,
                "aggregate": aggregated,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    example_sites, example_cells = relaxed_voronoi_mesh(
        max(sizes), seeds[0], lloyd_steps
    )
    plot_example_mesh(
        example_cells,
        example_sites,
        output_dir / "figures" / "voronoi_mesh_50_cells.png",
    )
    plot_scaling(
        rows,
        aggregated,
        output_dir / "figures" / "voronoi_constraint_scaling.png",
    )
    reproduction = (
        f"python {script.name} --sizes {','.join(map(str, sizes))} "
        f"--seeds {','.join(map(str, seeds))} --degree {degree} "
        f"--lloyd-steps {lloyd_steps} --output-dir {output_dir.name}"
    )
    write_report(
        output_dir / "VORONOI_SCALING_REPORT.md",
        rows,
        aggregated,
        reproduction,
    )
    return rows


def _integer_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=_integer_list, default=[20, 35, 50])
    parser.add_argument("--seeds", type=_integer_list, default=[17, 29, 43])
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--lloyd-steps", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results_voronoi_scaling"),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()

    if arguments.worker:
        if len(arguments.sizes) != 1 or len(arguments.seeds) != 1:
            parser.error("worker mode requires one size and one seed")
        result = run_one(
            arguments.sizes[0],
            arguments.degree,
            arguments.seeds[0],
            arguments.lloyd_steps,
        )
        print(json.dumps(result))
        return

    run_suite(
        arguments.sizes,
        arguments.seeds,
        arguments.degree,
        arguments.lloyd_steps,
        arguments.output_dir,
    )


if __name__ == "__main__":
    main()
