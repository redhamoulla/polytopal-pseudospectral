#!/usr/bin/env python3
"""Compatible nodal pseudospectral discretization on one convex polygon.

The implementation realizes the total-degree polynomial de Rham complex

    P_N --d--> [P_{N-1}]^2 --d--> P_{N-2}

on a non-tensorial convex polygon.  Modal bases are orthonormalized by an
exact-enough polygonal cubature and then converted to cardinal bases at
approximate Fekete points.  Both representations satisfy the same discrete
Stokes identity.

The executable validation covers:

* exactness, D1 D0 = 0;
* the 2-D Stokes/SBP identity;
* the quotient induced by the rectangular 0--2 pairing;
* compression of oversampled boundary traces to minimal power ports;
* a boundary-driven acoustic plane wave;
* exact-in-time spatial convergence and midpoint power balance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-polytopal-ph")

import matplotlib
import numpy as np
import scipy.linalg as la
from numpy.polynomial.legendre import leggauss
from scipy.stats import qmc

matplotlib.use("Agg")
import matplotlib.pyplot as plt


Array = np.ndarray


def regular_test_pentagon() -> Array:
    """Return a deliberately non-affine, counter-clockwise convex pentagon."""

    return np.array(
        [
            [-1.00, -0.55],
            [0.45, -0.88],
            [1.12, 0.05],
            [0.38, 1.02],
            [-0.82, 0.72],
        ],
        dtype=float,
    )


def robustness_polygons() -> dict[str, Array]:
    """Convex geometries used to check that the construction is not pentagon-specific."""

    pentagon = regular_test_pentagon()
    angle = 0.43
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    transformed = 1.7 * (pentagon @ rotation.T) + np.array([2.1, -0.8])
    return {
        "triangle": np.array(
            [[-1.0, -0.62], [1.05, -0.38], [-0.18, 1.02]]
        ),
        "quadrilateral": np.array(
            [[-1.0, -0.72], [0.72, -0.86], [1.08, 0.62], [-0.58, 1.01]]
        ),
        "pentagon": pentagon,
        "hexagon": np.array(
            [
                [-1.02, -0.18],
                [-0.58, -0.82],
                [0.48, -0.76],
                [1.04, 0.02],
                [0.46, 0.91],
                [-0.62, 0.76],
            ]
        ),
        "pentagon_transformed": transformed,
    }


def signed_area(vertices: Array) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def ensure_ccw(vertices: Array) -> Array:
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError("vertices must have shape (n_vertices, 2)")
    if abs(signed_area(vertices)) < 100.0 * np.finfo(float).eps:
        raise ValueError("degenerate polygon")
    return vertices.copy() if signed_area(vertices) > 0 else vertices[::-1].copy()


def polygon_centroid(vertices: Array) -> Array:
    """Area centroid of a simple counter-clockwise polygon."""

    vertices = ensure_ccw(vertices)
    x = vertices[:, 0]
    y = vertices[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area6 = 3.0 * np.sum(cross)
    cx = np.sum((x + np.roll(x, -1)) * cross) / area6
    cy = np.sum((y + np.roll(y, -1)) * cross) / area6
    return np.array([cx, cy])


def polygon_quadrature(vertices: Array, order: int) -> tuple[Array, Array]:
    """Positive quadrature obtained by a centroid fan and Duffy maps."""

    vertices = ensure_ccw(vertices)
    center = polygon_centroid(vertices)
    g, wg = leggauss(order)
    r = 0.5 * (g + 1.0)
    wr = 0.5 * wg
    rr, ss = np.meshgrid(r, r, indexing="ij")
    wrr, wss = np.meshgrid(wr, wr, indexing="ij")
    rr = rr.ravel()
    ss = ss.ravel()
    base_weight = (wrr * wss).ravel()

    point_blocks: list[Array] = []
    weight_blocks: list[Array] = []
    for index, vertex in enumerate(vertices):
        next_vertex = vertices[(index + 1) % len(vertices)]
        edge_a = vertex - center
        edge_b = next_vertex - center
        # x = center + r*(vertex-center) + (1-r)*s*(next-center)
        points = (
            center[None, :]
            + rr[:, None] * edge_a[None, :]
            + ((1.0 - rr) * ss)[:, None] * edge_b[None, :]
        )
        jacobian = abs(np.linalg.det(np.column_stack((edge_a, edge_b))))
        weights = base_weight * jacobian * (1.0 - rr)
        point_blocks.append(points)
        weight_blocks.append(weights)
    return np.vstack(point_blocks), np.concatenate(weight_blocks)


@dataclass
class BoundaryQuadrature:
    points: Array
    weights: Array
    tangents: Array
    normals: Array
    edge_ids: Array


def boundary_quadrature(vertices: Array, order: int) -> BoundaryQuadrature:
    """Gauss quadrature on the counter-clockwise oriented polygon boundary."""

    vertices = ensure_ccw(vertices)
    g, wg = leggauss(order)
    point_blocks: list[Array] = []
    weight_blocks: list[Array] = []
    tangent_blocks: list[Array] = []
    normal_blocks: list[Array] = []
    edge_blocks: list[Array] = []
    for edge_id, start in enumerate(vertices):
        end = vertices[(edge_id + 1) % len(vertices)]
        vector = end - start
        length = float(np.linalg.norm(vector))
        tangent = vector / length
        normal = np.array([tangent[1], -tangent[0]])
        points = (
            0.5 * (1.0 - g)[:, None] * start[None, :]
            + 0.5 * (1.0 + g)[:, None] * end[None, :]
        )
        point_blocks.append(points)
        weight_blocks.append(0.5 * length * wg)
        tangent_blocks.append(np.repeat(tangent[None, :], order, axis=0))
        normal_blocks.append(np.repeat(normal[None, :], order, axis=0))
        edge_blocks.append(np.full(order, edge_id, dtype=int))
    return BoundaryQuadrature(
        points=np.vstack(point_blocks),
        weights=np.concatenate(weight_blocks),
        tangents=np.vstack(tangent_blocks),
        normals=np.vstack(normal_blocks),
        edge_ids=np.concatenate(edge_blocks),
    )


def monomial_powers(degree: int) -> list[tuple[int, int]]:
    if degree < 0:
        return []
    return [
        (px, total - px)
        for total in range(degree + 1)
        for px in range(total + 1)
    ]


def polynomial_dimension(degree: int) -> int:
    return (degree + 1) * (degree + 2) // 2 if degree >= 0 else 0


@dataclass
class PolynomialFrame:
    degree: int
    center: Array
    scale: float
    powers: list[tuple[int, int]]
    raw_to_modal_basis: Array

    @property
    def dimension(self) -> int:
        return len(self.powers)

    def raw_values(self, points: Array) -> Array:
        points = np.asarray(points)
        xi = (points[:, 0] - self.center[0]) / self.scale
        eta = (points[:, 1] - self.center[1]) / self.scale
        return np.column_stack(
            [xi**px * eta**py for px, py in self.powers]
        )

    def values(self, points: Array) -> Array:
        return self.raw_values(points) @ self.raw_to_modal_basis


def build_nested_frames(
    vertices: Array, max_degree: int, quadrature_order: int | None = None
) -> tuple[dict[int, PolynomialFrame], Array, Array]:
    """Build nested, numerically orthonormal polynomial bases."""

    if max_degree < 0:
        raise ValueError("max_degree must be nonnegative")
    vertices = ensure_ccw(vertices)
    center = polygon_centroid(vertices)
    scale = float(np.max(np.linalg.norm(vertices - center[None, :], axis=1)))
    order = quadrature_order or max(max_degree + 5, 14)
    points, weights = polygon_quadrature(vertices, order)
    powers = monomial_powers(max_degree)
    template = PolynomialFrame(
        degree=max_degree,
        center=center,
        scale=scale,
        powers=powers,
        raw_to_modal_basis=np.eye(len(powers)),
    )
    raw = template.raw_values(points)
    weighted = np.sqrt(weights)[:, None] * raw
    _, upper = la.qr(weighted, mode="economic")
    transform = la.solve_triangular(
        upper, np.eye(upper.shape[0]), lower=False
    )
    frames: dict[int, PolynomialFrame] = {}
    for degree in range(max_degree + 1):
        dim = polynomial_dimension(degree)
        frames[degree] = PolynomialFrame(
            degree=degree,
            center=center,
            scale=scale,
            powers=powers[:dim],
            raw_to_modal_basis=transform[:dim, :dim].copy(),
        )
    return frames, points, weights


def raw_derivative_matrix(
    source: PolynomialFrame, target: PolynomialFrame, axis: int
) -> Array:
    """Derivative in normalized monomial coordinates, mapped to physical x/y."""

    lookup = {power: index for index, power in enumerate(target.powers)}
    derivative = np.zeros((target.dimension, source.dimension))
    for column, (px, py) in enumerate(source.powers):
        exponent = px if axis == 0 else py
        if exponent == 0:
            continue
        target_power = (px - 1, py) if axis == 0 else (px, py - 1)
        derivative[lookup[target_power], column] = exponent / source.scale
    return derivative


def modal_derivative(
    source: PolynomialFrame, target: PolynomialFrame, axis: int
) -> Array:
    raw_derivative = raw_derivative_matrix(source, target, axis)
    right_hand_side = raw_derivative @ source.raw_to_modal_basis
    return la.solve_triangular(
        target.raw_to_modal_basis,
        right_hand_side,
        lower=False,
    )


def points_in_convex_polygon(points: Array, vertices: Array, tol: float = 1e-12) -> Array:
    vertices = ensure_ccw(vertices)
    inside = np.ones(len(points), dtype=bool)
    for index, start in enumerate(vertices):
        edge = vertices[(index + 1) % len(vertices)] - start
        relative = points - start[None, :]
        cross = edge[0] * relative[:, 1] - edge[1] * relative[:, 0]
        inside &= cross >= -tol
    return inside


def approximate_fekete_nodes(
    vertices: Array,
    frame: PolynomialFrame,
    candidate_factor: int = 45,
) -> tuple[Array, float]:
    """Select cardinal nodes by pivoted QR from boundary and Halton candidates."""

    vertices = ensure_ccw(vertices)
    dimension = frame.dimension
    edge_count = max(2 * frame.degree + 5, 12)
    theta = np.linspace(0.0, np.pi, edge_count)
    edge_parameter = 0.5 * (1.0 - np.cos(theta))
    boundary_candidates = []
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % len(vertices)]
        boundary_candidates.append(
            (1.0 - edge_parameter)[:, None] * start[None, :]
            + edge_parameter[:, None] * end[None, :]
        )

    minimum_candidates = max(candidate_factor * dimension, 1200)
    sampler = qmc.Halton(d=2, scramble=False)
    raw = sampler.random(n=4 * minimum_candidates + 97)
    lower = np.min(vertices, axis=0)
    upper = np.max(vertices, axis=0)
    box_points = lower[None, :] + raw * (upper - lower)[None, :]
    interior = box_points[points_in_convex_polygon(box_points, vertices)]
    if len(interior) < minimum_candidates:
        raise RuntimeError("insufficient Halton candidates inside polygon")
    candidates = np.vstack(
        [np.vstack(boundary_candidates), interior[:minimum_candidates]]
    )
    vandermonde = frame.values(candidates)
    _, _, pivots = la.qr(vandermonde.T, mode="economic", pivoting=True)
    nodes = candidates[pivots[:dimension]]
    condition = float(np.linalg.cond(frame.values(nodes)))
    order = np.lexsort((nodes[:, 0], nodes[:, 1]))
    return nodes[order], condition


@dataclass
class PortCompression:
    trace: Array
    left_vectors: Array
    singular_values: Array
    rank: int

    def effort_from_samples(self, samples: Array, sqrt_weights: Array) -> Array:
        return self.left_vectors.T @ (sqrt_weights * samples)


@dataclass
class PolygonComplex:
    degree: int
    vertices: Array
    frames: dict[int, PolynomialFrame]
    volume_points: Array
    volume_weights: Array
    boundary: BoundaryQuadrature
    d0_modal: Array
    d1_modal: Array
    pair02_modal: Array
    wedge11_modal: Array
    boundary_pair_modal: Array
    trace0_modal: Array
    trace1_modal: Array
    nodes0: Array
    nodes1: Array
    nodes2: Array
    vandermonde0: Array
    vandermonde1: Array
    vandermonde2: Array
    nodal_condition0: float
    nodal_condition1: float
    nodal_condition2: float
    d0_nodal: Array
    d1_nodal: Array
    pair02_nodal: Array
    wedge11_nodal: Array
    boundary_pair_nodal: Array
    port_compression: PortCompression

    @property
    def n0(self) -> int:
        return self.frames[self.degree].dimension

    @property
    def scalar_n1(self) -> int:
        return self.frames[self.degree - 1].dimension

    @property
    def n1(self) -> int:
        return 2 * self.scalar_n1

    @property
    def n2(self) -> int:
        return self.frames[self.degree - 2].dimension


def build_polygon_complex(
    degree: int,
    vertices: Array | None = None,
    quadrature_order: int | None = None,
    boundary_order: int | None = None,
) -> PolygonComplex:
    """Build modal and cardinal representations of the polygonal complex."""

    if degree < 2:
        raise ValueError("degree must be at least 2")
    vertices = ensure_ccw(regular_test_pentagon() if vertices is None else vertices)
    frames, volume_points, volume_weights = build_nested_frames(
        vertices, degree, quadrature_order=quadrature_order
    )
    frame0 = frames[degree]
    frame1 = frames[degree - 1]
    frame2 = frames[degree - 2]

    dx0 = modal_derivative(frame0, frame1, axis=0)
    dy0 = modal_derivative(frame0, frame1, axis=1)
    dx1 = modal_derivative(frame1, frame2, axis=0)
    dy1 = modal_derivative(frame1, frame2, axis=1)
    d0_modal = np.vstack((dx0, dy0))
    d1_modal = np.hstack((-dy1, dx1))

    phi0 = frame0.values(volume_points)
    phi1 = frame1.values(volume_points)
    phi2 = frame2.values(volume_points)
    pair02_modal = phi0.T @ (volume_weights[:, None] * phi2)
    scalar_mass1 = phi1.T @ (volume_weights[:, None] * phi1)
    zeros = np.zeros_like(scalar_mass1)
    wedge11_modal = np.block(
        [[zeros, scalar_mass1], [-scalar_mass1, zeros]]
    )

    boundary = boundary_quadrature(
        vertices, boundary_order or max(degree + 3, 10)
    )
    trace0_modal = frame0.values(boundary.points)
    trace_scalar1 = frame1.values(boundary.points)
    trace1_modal = np.hstack(
        (
            boundary.tangents[:, 0, None] * trace_scalar1,
            boundary.tangents[:, 1, None] * trace_scalar1,
        )
    )
    boundary_pair_modal = trace0_modal.T @ (
        boundary.weights[:, None] * trace1_modal
    )

    nodes0, condition0 = approximate_fekete_nodes(vertices, frame0)
    nodes1, condition1 = approximate_fekete_nodes(vertices, frame1)
    nodes2, condition2 = approximate_fekete_nodes(vertices, frame2)
    vandermonde0 = frame0.values(nodes0)
    vandermonde1 = frame1.values(nodes1)
    vandermonde2 = frame2.values(nodes2)
    inverse0 = la.solve(vandermonde0, np.eye(frame0.dimension))
    inverse1 = la.solve(vandermonde1, np.eye(frame1.dimension))
    inverse2 = la.solve(vandermonde2, np.eye(frame2.dimension))
    block_inverse1 = la.block_diag(inverse1, inverse1)
    block_vandermonde1 = la.block_diag(vandermonde1, vandermonde1)

    d0_nodal = block_vandermonde1 @ d0_modal @ inverse0
    d1_nodal = vandermonde2 @ d1_modal @ block_inverse1
    pair02_nodal = inverse0.T @ pair02_modal @ inverse2
    wedge11_nodal = block_inverse1.T @ wedge11_modal @ block_inverse1
    trace0_nodal = trace0_modal @ inverse0
    trace1_nodal = trace1_modal @ block_inverse1
    boundary_pair_nodal = trace0_nodal.T @ (
        boundary.weights[:, None] * trace1_nodal
    )

    sqrt_weights = np.sqrt(boundary.weights)
    weighted_trace = sqrt_weights[:, None] * trace1_modal
    left, singular_values, right_t = la.svd(
        weighted_trace, full_matrices=False
    )
    tolerance = (
        1.0e3 * max(weighted_trace.shape)
        * np.finfo(float).eps
        * singular_values[0]
    )
    rank = int(np.sum(singular_values > tolerance))
    left = left[:, :rank]
    singular_values = singular_values[:rank]
    trace_minimal = singular_values[:, None] * right_t[:rank]
    compression = PortCompression(
        trace=trace_minimal,
        left_vectors=left,
        singular_values=singular_values,
        rank=rank,
    )

    return PolygonComplex(
        degree=degree,
        vertices=vertices,
        frames=frames,
        volume_points=volume_points,
        volume_weights=volume_weights,
        boundary=boundary,
        d0_modal=d0_modal,
        d1_modal=d1_modal,
        pair02_modal=pair02_modal,
        wedge11_modal=wedge11_modal,
        boundary_pair_modal=boundary_pair_modal,
        trace0_modal=trace0_modal,
        trace1_modal=trace1_modal,
        nodes0=nodes0,
        nodes1=nodes1,
        nodes2=nodes2,
        vandermonde0=vandermonde0,
        vandermonde1=vandermonde1,
        vandermonde2=vandermonde2,
        nodal_condition0=condition0,
        nodal_condition1=condition1,
        nodal_condition2=condition2,
        d0_nodal=d0_nodal,
        d1_nodal=d1_nodal,
        pair02_nodal=pair02_nodal,
        wedge11_nodal=wedge11_nodal,
        boundary_pair_nodal=boundary_pair_nodal,
        port_compression=compression,
    )


def structural_diagnostics(complex_: PolygonComplex) -> dict[str, float]:
    """Return scale-aware structural defects in modal and nodal coordinates."""

    def relative(defect: Array, *terms: Array) -> float:
        scale = max(1.0, *(np.linalg.norm(term, ord=np.inf) for term in terms))
        return float(np.linalg.norm(defect, ord=np.inf) / scale)

    modal_stokes_left = (
        complex_.pair02_modal @ complex_.d1_modal
        + complex_.d0_modal.T @ complex_.wedge11_modal
    )
    nodal_stokes_left = (
        complex_.pair02_nodal @ complex_.d1_nodal
        + complex_.d0_nodal.T @ complex_.wedge11_nodal
    )
    rank_d0 = np.linalg.matrix_rank(complex_.d0_modal)
    rank_d1 = np.linalg.matrix_rank(complex_.d1_modal)
    rank_pair = np.linalg.matrix_rank(complex_.pair02_modal)
    quotient_projector = complex_.pair02_modal @ la.pinv(
        complex_.pair02_modal
    )
    quotient_idempotence = quotient_projector @ quotient_projector - quotient_projector

    rng = np.random.default_rng(20260730 + complex_.degree)
    effort0 = rng.standard_normal(complex_.n0)
    effort1 = rng.standard_normal(complex_.n1)
    flow2 = -complex_.d1_modal @ effort1
    flow1 = complex_.d0_modal @ effort0
    boundary_flow = complex_.trace0_modal @ effort0
    boundary_effort = complex_.trace1_modal @ effort1
    volume_power = (
        effort0 @ complex_.pair02_modal @ flow2
        + effort1 @ complex_.wedge11_modal @ flow1
    )
    boundary_power = np.sum(
        complex_.boundary.weights * boundary_flow * boundary_effort
    )

    port = complex_.port_compression
    full_trace_gram = (
        complex_.trace1_modal.T
        @ (complex_.boundary.weights[:, None] * complex_.trace1_modal)
    )
    compressed_trace_gram = port.trace.T @ port.trace
    return {
        "degree": float(complex_.degree),
        "n0": float(complex_.n0),
        "n1": float(complex_.n1),
        "n2": float(complex_.n2),
        "euler_characteristic": float(
            complex_.n0 - complex_.n1 + complex_.n2
        ),
        "d_squared_modal_rel": relative(
            complex_.d1_modal @ complex_.d0_modal,
            complex_.d1_modal,
            complex_.d0_modal,
        ),
        "d_squared_nodal_rel": relative(
            complex_.d1_nodal @ complex_.d0_nodal,
            complex_.d1_nodal,
            complex_.d0_nodal,
        ),
        "stokes_modal_rel": relative(
            modal_stokes_left - complex_.boundary_pair_modal,
            modal_stokes_left,
            complex_.boundary_pair_modal,
        ),
        "stokes_nodal_rel": relative(
            nodal_stokes_left - complex_.boundary_pair_nodal,
            nodal_stokes_left,
            complex_.boundary_pair_nodal,
        ),
        "wedge_skew_rel": relative(
            complex_.wedge11_modal + complex_.wedge11_modal.T,
            complex_.wedge11_modal,
        ),
        "rank_d0": float(rank_d0),
        "rank_d1": float(rank_d1),
        "betti0": float(complex_.n0 - rank_d0),
        "betti1": float(complex_.n1 - rank_d0 - rank_d1),
        "betti2": float(complex_.n2 - rank_d1),
        "rank_pair02": float(rank_pair),
        "pair02_left_nullity": float(complex_.n0 - rank_pair),
        "quotient_projector_rel": relative(
            quotient_idempotence, quotient_projector
        ),
        "random_dirac_power_abs": float(abs(volume_power + boundary_power)),
        "random_dirac_power_rel": float(
            abs(volume_power + boundary_power)
            / max(1.0, abs(volume_power), abs(boundary_power))
        ),
        "boundary_sample_count": float(len(complex_.boundary.weights)),
        "minimal_port_dimension": float(port.rank),
        "port_compression_rel": relative(
            full_trace_gram - compressed_trace_gram,
            full_trace_gram,
        ),
        "cond_vandermonde_0": complex_.nodal_condition0,
        "cond_vandermonde_1": complex_.nodal_condition1,
        "cond_vandermonde_2": complex_.nodal_condition2,
    }


@dataclass
class PlaneWave:
    wave_vector: Array

    @property
    def omega(self) -> float:
        return float(np.linalg.norm(self.wave_vector))

    def phase(self, points: Array, time: float) -> Array:
        return points @ self.wave_vector - self.omega * time

    def pressure(self, points: Array, time: float) -> Array:
        return np.cos(self.phase(points, time))

    def flux_form(self, points: Array, time: float) -> tuple[Array, Array]:
        """Return coefficients (a,b) of beta=a dx+b dy=*v^flat."""

        value = np.cos(self.phase(points, time))
        kx, ky = self.wave_vector / self.omega
        return -ky * value, kx * value


def l2_project_scalar(
    frame: PolynomialFrame,
    points: Array,
    weights: Array,
    values: Array,
) -> Array:
    return frame.values(points).T @ (weights * values)


def project_plane_wave(
    complex_: PolygonComplex,
    wave: PlaneWave,
    time: float,
    quadrature_order: int | None = None,
) -> Array:
    points, weights = polygon_quadrature(
        complex_.vertices,
        quadrature_order or max(2 * complex_.degree + 8, 28),
    )
    pressure = wave.pressure(points, time)
    flux_x, flux_y = wave.flux_form(points, time)
    frame1 = complex_.frames[complex_.degree - 1]
    frame2 = complex_.frames[complex_.degree - 2]
    q = l2_project_scalar(frame2, points, weights, pressure)
    beta_x = l2_project_scalar(frame1, points, weights, flux_x)
    beta_y = l2_project_scalar(frame1, points, weights, flux_y)
    return np.concatenate((q, beta_x, beta_y))


def acoustic_matrices(complex_: PolygonComplex) -> tuple[Array, Array]:
    """Return skew internal matrix J and minimal boundary input matrix B."""

    zeros_q = np.zeros((complex_.n2, complex_.n2))
    zeros_v = np.zeros((complex_.n1, complex_.n1))
    internal = np.block(
        [
            [zeros_q, -complex_.d1_modal],
            [complex_.d1_modal.T, zeros_v],
        ]
    )
    boundary_input = np.vstack(
        (
            np.zeros((complex_.n2, complex_.port_compression.rank)),
            -complex_.port_compression.trace.T,
        )
    )
    return internal, boundary_input


def exact_semidiscrete_plane_wave(
    complex_: PolygonComplex,
    wave: PlaneWave,
    final_time: float,
) -> Array:
    """Integrate the linear forced semi-discretization by one matrix exponential."""

    internal, boundary_input = acoustic_matrices(complex_)
    initial = project_plane_wave(complex_, wave, time=0.0)
    boundary_phase = complex_.boundary.points @ wave.wave_vector
    cosine_samples = np.cos(boundary_phase)
    sine_samples = np.sin(boundary_phase)
    sqrt_weights = np.sqrt(complex_.boundary.weights)
    cosine_effort = complex_.port_compression.effort_from_samples(
        cosine_samples, sqrt_weights
    )
    sine_effort = complex_.port_compression.effort_from_samples(
        sine_samples, sqrt_weights
    )
    state_dimension = len(initial)
    augmented = np.zeros((state_dimension + 2, state_dimension + 2))
    augmented[:state_dimension, :state_dimension] = internal
    augmented[:state_dimension, state_dimension] = (
        boundary_input @ cosine_effort
    )
    augmented[:state_dimension, state_dimension + 1] = (
        boundary_input @ sine_effort
    )
    augmented[state_dimension, state_dimension + 1] = -wave.omega
    augmented[state_dimension + 1, state_dimension] = wave.omega
    initial_augmented = np.concatenate((initial, [1.0, 0.0]))
    final_augmented = la.expm(final_time * augmented) @ initial_augmented
    return final_augmented[:state_dimension]


def plane_wave_errors(
    complex_: PolygonComplex,
    wave: PlaneWave,
    state: Array,
    final_time: float,
    quadrature_order: int | None = None,
) -> dict[str, float]:
    points, weights = polygon_quadrature(
        complex_.vertices,
        quadrature_order or max(2 * complex_.degree + 10, 30),
    )
    frame1 = complex_.frames[complex_.degree - 1]
    frame2 = complex_.frames[complex_.degree - 2]
    q = state[: complex_.n2]
    beta_x = state[complex_.n2 : complex_.n2 + complex_.scalar_n1]
    beta_y = state[complex_.n2 + complex_.scalar_n1 :]
    numerical_pressure = frame2.values(points) @ q
    numerical_x = frame1.values(points) @ beta_x
    numerical_y = frame1.values(points) @ beta_y
    exact_pressure = wave.pressure(points, final_time)
    exact_x, exact_y = wave.flux_form(points, final_time)

    pressure_error = np.sqrt(
        np.sum(weights * (numerical_pressure - exact_pressure) ** 2)
    )
    pressure_norm = np.sqrt(np.sum(weights * exact_pressure**2))
    flux_error = np.sqrt(
        np.sum(
            weights
            * (
                (numerical_x - exact_x) ** 2
                + (numerical_y - exact_y) ** 2
            )
        )
    )
    flux_norm = np.sqrt(np.sum(weights * (exact_x**2 + exact_y**2)))
    return {
        "pressure_l2_relative": float(pressure_error / pressure_norm),
        "flux_l2_relative": float(flux_error / flux_norm),
        "state_energy": float(0.5 * state @ state),
    }


def interpolation_errors(
    complex_: PolygonComplex,
    function: Callable[[Array], Array],
    quadrature_order: int | None = None,
) -> dict[str, float]:
    """Cardinal interpolation error and differentiated interpolation error."""

    frame0 = complex_.frames[complex_.degree]
    frame1 = complex_.frames[complex_.degree - 1]
    nodal_values = function(complex_.nodes0)
    modal_coefficients = la.solve(complex_.vandermonde0, nodal_values)
    points, weights = polygon_quadrature(
        complex_.vertices,
        quadrature_order or max(2 * complex_.degree + 10, 30),
    )
    exact = function(points)
    numerical = frame0.values(points) @ modal_coefficients
    error = np.sqrt(np.sum(weights * (numerical - exact) ** 2))
    norm = np.sqrt(np.sum(weights * exact**2))

    # The test function is fixed below; its exact gradient is supplied here.
    x = points[:, 0]
    y = points[:, 1]
    exponential = np.exp(0.28 * x - 0.19 * y)
    phase = 1.1 * x + 0.7 * y
    exact_dx = exponential * (0.28 * np.sin(phase) + 1.1 * np.cos(phase))
    exact_dy = exponential * (-0.19 * np.sin(phase) + 0.7 * np.cos(phase))
    derivative_coefficients = complex_.d0_modal @ modal_coefficients
    count = complex_.scalar_n1
    numerical_dx = frame1.values(points) @ derivative_coefficients[:count]
    numerical_dy = frame1.values(points) @ derivative_coefficients[count:]
    derivative_error = np.sqrt(
        np.sum(
            weights
            * (
                (numerical_dx - exact_dx) ** 2
                + (numerical_dy - exact_dy) ** 2
            )
        )
    )
    derivative_norm = np.sqrt(
        np.sum(weights * (exact_dx**2 + exact_dy**2))
    )
    return {
        "interpolation_l2_relative": float(error / norm),
        "gradient_l2_relative": float(derivative_error / derivative_norm),
    }


def midpoint_power_simulation(
    complex_: PolygonComplex,
    wave: PlaneWave,
    final_time: float = 1.0,
    time_step: float = 0.002,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Run midpoint integration and verify exact discrete boundary work."""

    internal, boundary_input = acoustic_matrices(complex_)
    state = project_plane_wave(complex_, wave, time=0.0)
    identity = np.eye(len(state))
    left = identity - 0.5 * time_step * internal
    right = identity + 0.5 * time_step * internal
    lu, piv = la.lu_factor(left)
    steps = int(round(final_time / time_step))
    time_step = final_time / steps
    if abs(time_step - final_time / steps) > 1e-15:
        raise AssertionError("time-step normalization failure")

    # Re-factor if final_time changed the requested step.
    left = identity - 0.5 * time_step * internal
    right = identity + 0.5 * time_step * internal
    lu, piv = la.lu_factor(left)
    initial_energy = 0.5 * float(state @ state)
    cumulative_work = 0.0
    rows: list[dict[str, float]] = [
        {
            "time": 0.0,
            "energy": initial_energy,
            "boundary_work": 0.0,
            "balance_defect": 0.0,
        }
    ]
    sqrt_weights = np.sqrt(complex_.boundary.weights)
    for step in range(steps):
        midpoint_time = (step + 0.5) * time_step
        samples = wave.pressure(complex_.boundary.points, midpoint_time)
        effort = complex_.port_compression.effort_from_samples(
            samples, sqrt_weights
        )
        next_state = la.lu_solve(
            (lu, piv), right @ state + time_step * (boundary_input @ effort)
        )
        midpoint_state = 0.5 * (state + next_state)
        output = boundary_input.T @ midpoint_state
        step_work = time_step * float(effort @ output)
        cumulative_work += step_work
        state = next_state
        energy = 0.5 * float(state @ state)
        rows.append(
            {
                "time": (step + 1) * time_step,
                "energy": energy,
                "boundary_work": cumulative_work,
                "balance_defect": energy - initial_energy - cumulative_work,
            }
        )
    final_defect = rows[-1]["balance_defect"]
    metrics = {
        "initial_energy": initial_energy,
        "final_energy": rows[-1]["energy"],
        "cumulative_boundary_work": cumulative_work,
        "absolute_balance_defect": abs(final_defect),
        "relative_balance_defect": abs(final_defect)
        / max(1.0, initial_energy, abs(cumulative_work)),
        "time_step": time_step,
        "steps": float(steps),
    }
    return metrics, rows


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_nodes(complex_: PolygonComplex, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    groups = [
        (complex_.nodes0, rf"$V^0=P_{{{complex_.degree}}}$"),
        (complex_.nodes1, rf"$V^1=[P_{{{complex_.degree - 1}}}]^2$"),
        (complex_.nodes2, rf"$V^2=P_{{{complex_.degree - 2}}}$"),
    ]
    closed = np.vstack((complex_.vertices, complex_.vertices[0]))
    for axis, (nodes, title) in zip(axes, groups):
        axis.plot(closed[:, 0], closed[:, 1], color="#1f2937", lw=1.5)
        axis.scatter(nodes[:, 0], nodes[:, 1], s=18, color="#2563eb")
        axis.set_aspect("equal")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[:].set_visible(False)
    figure.suptitle("Nœuds de collocation de type Fekete sur le pentagone")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_convergence(rows: list[dict[str, float]], output: Path) -> None:
    degree = np.array([row["degree"] for row in rows])
    figure, axis = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    series = [
        ("pressure_l2_relative", "Pression, onde acoustique", "o"),
        ("flux_l2_relative", "Flux, onde acoustique", "s"),
        ("interpolation_l2_relative", "Interpolation cardinale", "^"),
        ("gradient_l2_relative", "Gradient interpolé", "D"),
    ]
    for key, label, marker in series:
        values = np.maximum(
            np.array([row[key] for row in rows]), 5e-16
        )
        axis.semilogy(degree, values, marker=marker, lw=1.7, label=label)
    axis.set_xlabel("Degré polynomial N")
    axis.set_ylabel("Erreur relative L²")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("Convergence en p sur un polygone non tensoriel")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_structure(rows: list[dict[str, float]], output: Path) -> None:
    degree = np.array([row["degree"] for row in rows])
    figure, axis = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    keys = [
        ("d_squared_modal_rel", r"$D_1D_0$ modal", "o"),
        ("stokes_modal_rel", "Stokes modal", "s"),
        ("d_squared_nodal_rel", r"$D_1D_0$ nodal", "^"),
        ("stokes_nodal_rel", "Stokes nodal", "D"),
        ("port_compression_rel", "Compression des ports", "v"),
    ]
    for key, label, marker in keys:
        values = np.maximum(
            np.array([row[key] for row in rows]), 1e-18
        )
        axis.semilogy(degree, values, marker=marker, lw=1.5, label=label)
    axis.axhline(np.finfo(float).eps, color="#6b7280", ls="--", lw=1)
    axis.set_xlabel("Degré polynomial N")
    axis.set_ylabel("Défaut relatif")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    axis.set_title("Invariants de structure")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_power(rows: list[dict[str, float]], output: Path) -> None:
    time = np.array([row["time"] for row in rows])
    energy = np.array([row["energy"] for row in rows])
    work = np.array([row["boundary_work"] for row in rows])
    defect = np.abs(np.array([row["balance_defect"] for row in rows]))
    initial = energy[0]
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.4, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )
    axes[0].plot(time, energy - initial, label=r"$H(t)-H(0)$", lw=2)
    axes[0].plot(time, work, "--", label="Travail cumulé au bord", lw=1.7)
    axes[0].set_ylabel("Énergie / travail")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].set_title("Bilan de puissance du système ouvert")
    axes[1].semilogy(time, np.maximum(defect, 1e-18), color="#b91c1c")
    axes[1].set_xlabel("Temps")
    axes[1].set_ylabel("Défaut")
    axes[1].grid(True, which="both", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def plot_final_field(
    complex_: PolygonComplex,
    wave: PlaneWave,
    state: Array,
    final_time: float,
    output: Path,
) -> None:
    vertices = complex_.vertices
    lower = np.min(vertices, axis=0)
    upper = np.max(vertices, axis=0)
    gx = np.linspace(lower[0], upper[0], 230)
    gy = np.linspace(lower[1], upper[1], 230)
    xx, yy = np.meshgrid(gx, gy)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    inside = points_in_convex_polygon(points, vertices, tol=1e-10)
    numerical = np.full(len(points), np.nan)
    exact = np.full(len(points), np.nan)
    frame2 = complex_.frames[complex_.degree - 2]
    numerical[inside] = frame2.values(points[inside]) @ state[: complex_.n2]
    exact[inside] = wave.pressure(points[inside], final_time)
    error = numerical - exact
    numerical = numerical.reshape(xx.shape)
    error = error.reshape(xx.shape)
    closed = np.vstack((vertices, vertices[0]))

    figure, axes = plt.subplots(1, 2, figsize=(9.7, 4.2), constrained_layout=True)
    first = axes[0].contourf(xx, yy, numerical, levels=24, cmap="viridis")
    axes[0].plot(closed[:, 0], closed[:, 1], color="#111827", lw=1)
    axes[0].set_title("Pression numérique")
    second = axes[1].contourf(xx, yy, error, levels=24, cmap="coolwarm")
    axes[1].plot(closed[:, 0], closed[:, 1], color="#111827", lw=1)
    axes[1].set_title("Erreur ponctuelle")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    figure.colorbar(first, ax=axes[0], shrink=0.82)
    figure.colorbar(second, ax=axes[1], shrink=0.82)
    figure.suptitle(
        rf"Solution acoustique manufacturée, $N={complex_.degree}$, "
        rf"$t={final_time:g}$"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190)
    plt.close(figure)


def run_validation(
    output_directory: Path,
    degrees: range = range(2, 14),
    final_time: float = 0.65,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory = output_directory / "figures"
    wave = PlaneWave(np.array([1.35, -0.92]))
    convergence_rows: list[dict[str, float]] = []
    structure_rows: list[dict[str, float]] = []
    selected_complex: PolygonComplex | None = None
    selected_state: Array | None = None

    analytic_function = lambda points: np.exp(
        0.28 * points[:, 0] - 0.19 * points[:, 1]
    ) * np.sin(1.1 * points[:, 0] + 0.7 * points[:, 1])

    for degree in degrees:
        complex_ = build_polygon_complex(degree)
        structure = structural_diagnostics(complex_)
        structure_rows.append(structure)
        final_state = exact_semidiscrete_plane_wave(
            complex_, wave, final_time=final_time
        )
        errors = plane_wave_errors(
            complex_, wave, final_state, final_time=final_time
        )
        interpolation = interpolation_errors(complex_, analytic_function)
        convergence_rows.append(
            {
                "degree": float(degree),
                **errors,
                **interpolation,
                "cond_vandermonde_0": complex_.nodal_condition0,
            }
        )
        if degree == max(degrees):
            selected_complex = complex_
            selected_state = final_state

    if selected_complex is None or selected_state is None:
        raise RuntimeError("empty degree range")
    power_degree = min(8, max(degrees))
    power_complex = build_polygon_complex(power_degree)
    power_metrics, power_rows = midpoint_power_simulation(
        power_complex, wave, final_time=1.0, time_step=0.002
    )
    internal, boundary_input = acoustic_matrices(power_complex)
    full_dirac = np.block(
        [
            [internal, boundary_input],
            [-boundary_input.T, np.zeros(
                (
                    power_complex.port_compression.rank,
                    power_complex.port_compression.rank,
                )
            )],
        ]
    )
    dirac_skew_defect = float(
        np.linalg.norm(full_dirac + full_dirac.T, ord=np.inf)
        / max(1.0, np.linalg.norm(full_dirac, ord=np.inf))
    )
    power_metrics["dirac_skew_relative"] = dirac_skew_defect
    power_metrics["degree"] = float(power_degree)

    geometry_rows: list[dict[str, float | str]] = []
    for name, polygon in robustness_polygons().items():
        geometry_complex = build_polygon_complex(8, vertices=polygon)
        diagnostics = structural_diagnostics(geometry_complex)
        geometry_rows.append(
            {
                "geometry": name,
                "edge_count": float(len(polygon)),
                "stokes_modal_rel": diagnostics["stokes_modal_rel"],
                "stokes_nodal_rel": diagnostics["stokes_nodal_rel"],
                "d_squared_nodal_rel": diagnostics["d_squared_nodal_rel"],
                "random_dirac_power_rel": diagnostics[
                    "random_dirac_power_rel"
                ],
                "cond_vandermonde_0": diagnostics["cond_vandermonde_0"],
            }
        )

    fit_rows = [row for row in convergence_rows if row["degree"] >= 5]
    degrees_for_fit = np.array([row["degree"] for row in fit_rows])
    convergence_fits: dict[str, float] = {}
    for key in (
        "pressure_l2_relative",
        "flux_l2_relative",
        "interpolation_l2_relative",
        "gradient_l2_relative",
    ):
        logarithm = np.log10([row[key] for row in fit_rows])
        slope, _ = np.polyfit(degrees_for_fit, logarithm, 1)
        convergence_fits[f"{key}_digits_per_degree"] = float(-slope)

    save_csv(output_directory / "convergence.csv", convergence_rows)
    save_csv(output_directory / "structure.csv", structure_rows)
    save_csv(output_directory / "power_balance.csv", power_rows)
    save_csv(output_directory / "geometry_robustness.csv", geometry_rows)
    plot_nodes(selected_complex, figure_directory / "polygon_nodes.png")
    plot_convergence(convergence_rows, figure_directory / "convergence.png")
    plot_structure(structure_rows, figure_directory / "structure.png")
    plot_power(power_rows, figure_directory / "power_balance.png")
    plot_final_field(
        selected_complex,
        wave,
        selected_state,
        final_time,
        figure_directory / "final_field.png",
    )

    summary = {
        "polygon_vertices": selected_complex.vertices.tolist(),
        "degrees": [int(row["degree"]) for row in convergence_rows],
        "wave_vector": wave.wave_vector.tolist(),
        "wave_frequency": wave.omega,
        "final_time_for_convergence": final_time,
        "highest_degree": int(max(degrees)),
        "highest_degree_structure": structure_rows[-1],
        "highest_degree_errors": convergence_rows[-1],
        "empirical_convergence": convergence_fits,
        "geometry_robustness": geometry_rows,
        "power_test": power_metrics,
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="output directory",
    )
    parser.add_argument("--degree-min", type=int, default=2)
    parser.add_argument("--degree-max", type=int, default=13)
    parser.add_argument("--final-time", type=float, default=0.65)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_validation(
        args.output,
        degrees=range(args.degree_min, args.degree_max + 1),
        final_time=args.final_time,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
