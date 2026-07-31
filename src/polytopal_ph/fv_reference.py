#!/usr/bin/env python3
"""Independent structured-grid reference for the porous-acoustic duct.

This module deliberately does not import the polygonal pseudospectral code.
It discretises

    p_t + div(v) = 0,
    v_t + grad(p) + r(x, v) = 0

on a staggered finite-volume grid.  Pressure is cell centred, horizontal
velocity lives on vertical faces, and vertical velocity lives on horizontal
faces.  The top and bottom walls are rigid.  The characteristic impedance
conditions are

    u(0, y, t) = 2 p_inc(y, t) - p(0, y, t),
    u(L, y, t) = p(L, y, t).

They give the exact semi-discrete boundary power

    P_b = integral_y [2 p_inc p_left - p_left**2 - p_right**2] dy.

The nonlinear anisotropic drag is evaluated from a cell-centred velocity.
The interpolation from dynamic face velocities to cell centres and the
projection of drag back to faces are Euclidean adjoints.  Consequently, the
semi-discrete identity

    dH_h/dt = P_b - D_h

holds to roundoff, including for the nonlinear closure.

The porous insert is the fixed rectangle [0.72, 1.34] x [0.07, 0.93].
Cut cells use their exact area fraction, so its represented area is invariant
under grid refinement.  The constitutive law is an independent transcription
of the analytic passive teacher used by ``run_physical_experiment.py``.  A
non-negative porous-strength parameter scales only the positive-semidefinite
porous part of its tensor; isotropic air damping is never scaled.  Callers may
instead supply any pointwise object with ``predict(features, velocity)``—in
particular, a reloaded learned closure from the polygonal experiment.

The time integrator is classical RK4.  The returned ``time_balance_defect``
therefore measures temporal error, whereas ``power_identity_defect`` verifies
the exact spatial energy identity at every recorded state.

The default final time is 2.4 rather than 1.65.  With unit wave speed, a burst
whose window becomes appreciable after t=0 needs roughly 1.58 additional time
units to reach the downstream sensor.  At t=1.65 that sensor sees mainly a
small dispersive precursor, which is not a meaningful transmission metric.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

import numpy as np


Array = np.ndarray
DUCT_LENGTH = 2.0
DUCT_HEIGHT = 1.0
AIR_DAMPING = 2.0e-3
INSERT_X = (0.72, 1.34)
INSERT_Y = (0.07, 0.93)
SENSOR_POINTS = {
    "upstream": (0.58, 0.48),
    "downstream": (1.58, 0.52),
}


class PointClosure(Protocol):
    """Minimal interface shared by analytic and saved learned closures."""

    def predict(self, features: Array, velocity: Array) -> Array:
        ...


ClosureCallback = PointClosure | Callable[[Array, Array], Array]


@dataclass(frozen=True)
class Excitation:
    amplitude: float = 1.35
    omega: float = 10.5
    burst_duration: float = 0.72


@dataclass(frozen=True)
class StructuredGrid:
    nx: int
    ny: int
    dx: float
    dy: float
    x_centres: Array
    y_centres: Array
    x_field: Array
    y_field: Array
    porous_fraction: Array
    material: Array
    cos2angle: Array
    sin2angle: Array

    @property
    def pressure_size(self) -> int:
        return self.nx * self.ny

    @property
    def horizontal_velocity_size(self) -> int:
        return (self.nx - 1) * self.ny

    @property
    def vertical_velocity_size(self) -> int:
        return self.nx * (self.ny - 1)

    @property
    def state_size(self) -> int:
        return (
            self.pressure_size
            + self.horizontal_velocity_size
            + self.vertical_velocity_size
        )

    @property
    def represented_porous_area(self) -> float:
        return float(
            self.dx * self.dy * np.sum(self.porous_fraction)
        )


@dataclass
class ReferenceSimulation:
    nx: int
    ny: int
    excitation: Excitation
    final_time: float
    time_step: float
    step_count: int
    times: Array
    sensor_pressure: dict[str, Array]
    energy: Array
    boundary_power: Array
    dissipation: Array
    power_identity_defect: Array
    time_balance_defect: Array
    minimum_power_density: Array
    porous_area: float
    porous_strength: float = 1.0

    def summary(self) -> dict[str, float | int]:
        energy_scale = max(
            1.0e-15,
            float(np.max(self.energy)),
            float(
                np.max(
                    np.abs(self.boundary_power - self.dissipation)
                )
                * self.final_time
            ),
        )
        return {
            "nx": self.nx,
            "ny": self.ny,
            "state_dimension": (
                self.nx * self.ny
                + (self.nx - 1) * self.ny
                + self.nx * (self.ny - 1)
            ),
            "step_count": self.step_count,
            "time_step": self.time_step,
            "porous_area": self.porous_area,
            "porous_strength": self.porous_strength,
            "maximum_energy": float(np.max(self.energy)),
            "final_energy": float(self.energy[-1]),
            "maximum_power_identity_defect": float(
                np.max(self.power_identity_defect)
            ),
            "relative_time_balance_defect": float(
                np.max(np.abs(self.time_balance_defect))
                / energy_scale
            ),
            "minimum_power_density": float(
                np.min(self.minimum_power_density)
            ),
            "upstream_peak_pressure": float(
                np.max(np.abs(self.sensor_pressure["upstream"]))
            ),
            "downstream_peak_pressure": float(
                np.max(np.abs(self.sensor_pressure["downstream"]))
            ),
        }


def _interval_overlap(
    cell_left: Array,
    cell_right: Array,
    interval: tuple[float, float],
) -> Array:
    """Length of the intersection of cells with one fixed interval."""
    return np.maximum(
        0.0,
        np.minimum(cell_right, interval[1])
        - np.maximum(cell_left, interval[0]),
    )


def build_grid(nx: int, ny: int) -> StructuredGrid:
    if nx < 4 or ny < 4:
        raise ValueError("nx and ny must both be at least four")
    dx = DUCT_LENGTH / nx
    dy = DUCT_HEIGHT / ny
    x_centres = (np.arange(nx, dtype=float) + 0.5) * dx
    y_centres = (np.arange(ny, dtype=float) + 0.5) * dy
    x_field, y_field = np.meshgrid(x_centres, y_centres)

    x_left = np.arange(nx, dtype=float) * dx
    y_bottom = np.arange(ny, dtype=float) * dy
    x_fraction = _interval_overlap(
        x_left, x_left + dx, INSERT_X
    ) / dx
    y_fraction = _interval_overlap(
        y_bottom, y_bottom + dy, INSERT_Y
    ) / dy
    porous_fraction = y_fraction[:, None] * x_fraction[None, :]

    x_normalised = x_field / DUCT_LENGTH
    y_normalised = y_field / DUCT_HEIGHT
    material = (
        0.60 * np.sin(2.0 * np.pi * y_normalised)
        + 0.25 * np.cos(2.0 * np.pi * x_normalised)
    )
    angle = (
        0.33 * np.sin(np.pi * y_normalised)
        + 0.17 * np.cos(2.0 * np.pi * x_normalised)
    )
    return StructuredGrid(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        x_centres=x_centres,
        y_centres=y_centres,
        x_field=x_field,
        y_field=y_field,
        porous_fraction=porous_fraction,
        material=material,
        cos2angle=np.cos(2.0 * angle),
        sin2angle=np.sin(2.0 * angle),
    )


def incident_pressure(
    y: Array, time_value: float, excitation: Excitation
) -> Array:
    if time_value < 0.0 or time_value > excitation.burst_duration:
        temporal = 0.0
    else:
        window = np.sin(
            np.pi * time_value / excitation.burst_duration
        ) ** 2
        temporal = (
            excitation.amplitude
            * np.sin(excitation.omega * time_value)
            * window
        )
    profile = (
        1.0
        + 0.24 * np.cos(np.pi * y / DUCT_HEIGHT)
        + 0.12 * np.sin(2.0 * np.pi * y / DUCT_HEIGHT)
    )
    return temporal * profile


def passive_reference_drag(
    grid: StructuredGrid,
    velocity_x: Array,
    velocity_y: Array,
    porous_strength: float = 1.0,
) -> tuple[Array, Array, Array]:
    """Evaluate the passive nonlinear teacher at cell centres.

    Returns ``(drag_x, drag_y, power_density)``.  The lower-triangular
    factorisation used below makes non-negative power structural rather than
    a posteriori clipped.  ``porous_strength`` acts on the complete porous
    tensor while the isotropic ``AIR_DAMPING`` contribution is unchanged:

        A(strength) = AIR_DAMPING I
                      + strength (A(1) - AIR_DAMPING I).

    For the teacher coefficients and material range, the second term is
    positive semidefinite.  Hence every non-negative strength is passive and
    strength zero is precisely air damping.
    """
    if not np.isfinite(porous_strength) or porous_strength < 0.0:
        raise ValueError("porous_strength must be finite and non-negative")
    speed = np.sqrt(velocity_x * velocity_x + velocity_y * velocity_y)
    angle = 0.5 * np.arctan2(grid.sin2angle, grid.cos2angle)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    principal1 = cosine * velocity_x + sine * velocity_y
    principal2 = -sine * velocity_x + cosine * velocity_y
    gate = grid.porous_fraction
    material = grid.material
    saturation = speed * speed / (1.0 + 0.65 * speed * speed)

    diagonal1_sq = (
        AIR_DAMPING
        + gate
        * (
            0.080
            + 0.030 * (material + 1.0) / 2.0
            + 0.220 * speed
            + 0.095 * saturation
            + 0.045 * speed * speed / (1.0 + speed * speed)
        )
    )
    diagonal2_sq = (
        AIR_DAMPING
        + gate
        * (
            0.110
            + 0.040 * (1.0 - material) / 2.0
            + 0.140 * speed
            + 0.125 * saturation
            + 0.055 * speed * speed / (1.0 + speed * speed)
        )
    )
    diagonal1 = np.sqrt(diagonal1_sq)
    diagonal2 = np.sqrt(diagonal2_sq)
    lower21 = (
        gate
        * (0.10 + 0.025 * material)
        * np.tanh(1.4 * speed)
        * diagonal2
    )
    drag1 = (
        diagonal1 * diagonal1 * principal1
        + diagonal1 * lower21 * principal2
    )
    drag2 = (
        diagonal1 * lower21 * principal1
        + (lower21 * lower21 + diagonal2 * diagonal2) * principal2
    )
    nominal_drag_x = cosine * drag1 - sine * drag2
    nominal_drag_y = sine * drag1 + cosine * drag2
    drag_x = (
        AIR_DAMPING * velocity_x
        + porous_strength
        * (nominal_drag_x - AIR_DAMPING * velocity_x)
    )
    drag_y = (
        AIR_DAMPING * velocity_y
        + porous_strength
        * (nominal_drag_y - AIR_DAMPING * velocity_y)
    )
    power_density = velocity_x * drag_x + velocity_y * drag_y
    return drag_x, drag_y, power_density


def closure_features(
    grid: StructuredGrid,
    velocity_x: Array,
    velocity_y: Array,
) -> tuple[Array, Array]:
    """Build the feature/velocity matrices expected by learned closures."""
    speed = np.sqrt(velocity_x * velocity_x + velocity_y * velocity_y)
    features = np.column_stack(
        (
            np.log1p(speed).ravel(),
            grid.porous_fraction.ravel(),
            grid.material.ravel(),
            grid.cos2angle.ravel(),
            grid.sin2angle.ravel(),
        )
    )
    velocity = np.column_stack(
        (velocity_x.ravel(), velocity_y.ravel())
    )
    return features, velocity


def evaluate_cell_drag(
    grid: StructuredGrid,
    velocity_x: Array,
    velocity_y: Array,
    *,
    porous_strength: float = 1.0,
    closure: ClosureCallback | None = None,
) -> tuple[Array, Array, Array]:
    """Evaluate either the analytic teacher or a supplied local closure.

    A callback can be a callable ``closure(features, velocity)`` or any object
    exposing ``predict(features, velocity)``.  This is the same pointwise API
    as the saved neural models.  A supplied callback replaces the analytic
    teacher and is responsible for its own constitutive parameters; therefore
    ``porous_strength`` is intentionally rejected for callbacks unless it is
    one.  Passivity of a generic callback is diagnosed through the returned
    power density but cannot be guaranteed by the finite-volume wrapper.
    """
    if closure is None:
        return passive_reference_drag(
            grid,
            velocity_x,
            velocity_y,
            porous_strength=porous_strength,
        )
    if porous_strength != 1.0:
        raise ValueError(
            "porous_strength must equal one with a custom closure; "
            "encode material-strength changes in the callback itself"
        )
    features, velocity = closure_features(
        grid, velocity_x, velocity_y
    )
    predictor = getattr(closure, "predict", None)
    if predictor is None:
        prediction = closure(features, velocity)
    else:
        prediction = predictor(features, velocity)
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != velocity.shape:
        raise ValueError(
            "closure must return an array of shape "
            f"{velocity.shape}, received {prediction.shape}"
        )
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("closure returned non-finite drag")
    drag_x = prediction[:, 0].reshape(grid.ny, grid.nx)
    drag_y = prediction[:, 1].reshape(grid.ny, grid.nx)
    power_density = velocity_x * drag_x + velocity_y * drag_y
    return drag_x, drag_y, power_density


def _unpack_state(
    grid: StructuredGrid, state: Array
) -> tuple[Array, Array, Array]:
    if state.shape != (grid.state_size,):
        raise ValueError(
            f"expected state of shape {(grid.state_size,)}, "
            f"received {state.shape}"
        )
    p_stop = grid.pressure_size
    u_stop = p_stop + grid.horizontal_velocity_size
    pressure = state[:p_stop].reshape(grid.ny, grid.nx)
    horizontal = state[p_stop:u_stop].reshape(
        grid.ny, grid.nx - 1
    )
    vertical = state[u_stop:].reshape(grid.ny - 1, grid.nx)
    return pressure, horizontal, vertical


def _pack_state(
    pressure: Array, horizontal: Array, vertical: Array
) -> Array:
    return np.concatenate(
        (pressure.ravel(), horizontal.ravel(), vertical.ravel())
    )


def _dynamic_velocity_at_cells(
    grid: StructuredGrid, horizontal: Array, vertical: Array
) -> tuple[Array, Array]:
    """Interpolate only the dynamic face velocities to cell centres.

    The characteristic boundary velocities are algebraic port variables, not
    kinetic-energy states.  Excluding them here is what makes interpolation
    and drag projection exact adjoints.  The porous insert is separated from
    the inlet and outlet, and this boundary-layer convention converges to the
    same continuum drag law under refinement.
    """
    horizontal_full = np.zeros((grid.ny, grid.nx + 1))
    horizontal_full[:, 1:-1] = horizontal
    vertical_full = np.zeros((grid.ny + 1, grid.nx))
    vertical_full[1:-1, :] = vertical
    velocity_x = 0.5 * (
        horizontal_full[:, :-1] + horizontal_full[:, 1:]
    )
    velocity_y = 0.5 * (
        vertical_full[:-1, :] + vertical_full[1:, :]
    )
    return velocity_x, velocity_y


def spatial_rhs(
    grid: StructuredGrid,
    state: Array,
    time_value: float,
    excitation: Excitation,
    *,
    porous_strength: float = 1.0,
    closure: ClosureCallback | None = None,
) -> tuple[Array, dict[str, float]]:
    pressure, horizontal, vertical = _unpack_state(grid, state)

    horizontal_full = np.empty((grid.ny, grid.nx + 1))
    horizontal_full[:, 1:-1] = horizontal
    incoming = incident_pressure(
        grid.y_centres, time_value, excitation
    )
    horizontal_full[:, 0] = 2.0 * incoming - pressure[:, 0]
    horizontal_full[:, -1] = pressure[:, -1]
    vertical_full = np.zeros((grid.ny + 1, grid.nx))
    vertical_full[1:-1, :] = vertical

    pressure_rate = -(
        (horizontal_full[:, 1:] - horizontal_full[:, :-1]) / grid.dx
        + (vertical_full[1:, :] - vertical_full[:-1, :]) / grid.dy
    )
    horizontal_rate = -(
        pressure[:, 1:] - pressure[:, :-1]
    ) / grid.dx
    vertical_rate = -(
        pressure[1:, :] - pressure[:-1, :]
    ) / grid.dy

    velocity_x, velocity_y = _dynamic_velocity_at_cells(
        grid, horizontal, vertical
    )
    drag_x, drag_y, power_density = evaluate_cell_drag(
        grid,
        velocity_x,
        velocity_y,
        porous_strength=porous_strength,
        closure=closure,
    )
    # The following averages are the exact adjoints of the preceding
    # face-to-cell averages for the uniform mass matrices.
    horizontal_rate -= 0.5 * (
        drag_x[:, :-1] + drag_x[:, 1:]
    )
    vertical_rate -= 0.5 * (
        drag_y[:-1, :] + drag_y[1:, :]
    )

    derivative = _pack_state(
        pressure_rate, horizontal_rate, vertical_rate
    )
    boundary_power = float(
        grid.dy
        * np.sum(
            pressure[:, 0] * horizontal_full[:, 0]
            - pressure[:, -1] * horizontal_full[:, -1]
        )
    )
    dissipation = float(
        grid.dx * grid.dy * np.sum(power_density)
    )
    energy_rate = float(
        grid.dx
        * grid.dy
        * (
            np.sum(pressure * pressure_rate)
            + np.sum(horizontal * horizontal_rate)
            + np.sum(vertical * vertical_rate)
        )
    )
    scale = max(
        1.0,
        abs(energy_rate),
        abs(boundary_power),
        abs(dissipation),
    )
    identity_defect = abs(
        energy_rate - (boundary_power - dissipation)
    ) / scale
    return derivative, {
        "boundary_power": boundary_power,
        "dissipation": dissipation,
        "energy_rate": energy_rate,
        "power_identity_defect": identity_defect,
        "minimum_power_density": float(np.min(power_density)),
    }


def discrete_energy(grid: StructuredGrid, state: Array) -> float:
    pressure, horizontal, vertical = _unpack_state(grid, state)
    return float(
        0.5
        * grid.dx
        * grid.dy
        * (
            np.sum(pressure * pressure)
            + np.sum(horizontal * horizontal)
            + np.sum(vertical * vertical)
        )
    )


def _bilinear_cell_value(
    grid: StructuredGrid,
    field: Array,
    point: tuple[float, float],
) -> float:
    """Bilinear reconstruction from the cell-centred pressure field."""
    x, y = point
    xi = x / grid.dx - 0.5
    eta = y / grid.dy - 0.5
    left = int(np.floor(xi))
    bottom = int(np.floor(eta))
    left = max(0, min(grid.nx - 2, left))
    bottom = max(0, min(grid.ny - 2, bottom))
    tx = min(1.0, max(0.0, xi - left))
    ty = min(1.0, max(0.0, eta - bottom))
    return float(
        (1.0 - tx) * (1.0 - ty) * field[bottom, left]
        + tx * (1.0 - ty) * field[bottom, left + 1]
        + (1.0 - tx) * ty * field[bottom + 1, left]
        + tx * ty * field[bottom + 1, left + 1]
    )


def simulate_reference(
    nx: int,
    ny: int,
    *,
    excitation: Excitation = Excitation(),
    final_time: float = 2.4,
    cfl: float = 0.45,
    output_interval: float = 0.01,
    porous_strength: float = 1.0,
    closure: ClosureCallback | None = None,
) -> ReferenceSimulation:
    """Run one independent finite-volume reference simulation."""
    if final_time <= 0.0:
        raise ValueError("final_time must be positive")
    if not 0.0 < cfl <= 0.8:
        raise ValueError("cfl must lie in (0, 0.8]")
    if not np.isfinite(porous_strength) or porous_strength < 0.0:
        raise ValueError("porous_strength must be finite and non-negative")
    if closure is not None and porous_strength != 1.0:
        raise ValueError(
            "porous_strength must equal one with a custom closure"
        )
    grid = build_grid(nx, ny)
    stable_step = cfl / np.sqrt(
        1.0 / grid.dx**2 + 1.0 / grid.dy**2
    )
    step_count = int(np.ceil(final_time / stable_step))
    time_step = final_time / step_count
    save_stride = max(
        1, int(np.round(output_interval / time_step))
    )

    state = np.zeros(grid.state_size)
    saved_times: list[float] = []
    saved_energy: list[float] = []
    saved_boundary: list[float] = []
    saved_dissipation: list[float] = []
    saved_identity: list[float] = []
    saved_time_defect: list[float] = []
    saved_minimum_power: list[float] = []
    saved_sensors = {name: [] for name in SENSOR_POINTS}
    cumulative_power = 0.0
    initial_energy = discrete_energy(grid, state)

    def record(time_value: float, diagnostic: dict[str, float]) -> None:
        pressure, _, _ = _unpack_state(grid, state)
        energy = discrete_energy(grid, state)
        saved_times.append(time_value)
        saved_energy.append(energy)
        saved_boundary.append(diagnostic["boundary_power"])
        saved_dissipation.append(diagnostic["dissipation"])
        saved_identity.append(diagnostic["power_identity_defect"])
        saved_time_defect.append(
            energy - initial_energy - cumulative_power
        )
        saved_minimum_power.append(
            diagnostic["minimum_power_density"]
        )
        for name, point in SENSOR_POINTS.items():
            saved_sensors[name].append(
                _bilinear_cell_value(grid, pressure, point)
            )

    _, initial_diagnostic = spatial_rhs(
        grid,
        state,
        0.0,
        excitation,
        porous_strength=porous_strength,
        closure=closure,
    )
    record(0.0, initial_diagnostic)

    for step in range(step_count):
        time_value = step * time_step
        k1, d1 = spatial_rhs(
            grid,
            state,
            time_value,
            excitation,
            porous_strength=porous_strength,
            closure=closure,
        )
        k2, d2 = spatial_rhs(
            grid,
            state + 0.5 * time_step * k1,
            time_value + 0.5 * time_step,
            excitation,
            porous_strength=porous_strength,
            closure=closure,
        )
        k3, d3 = spatial_rhs(
            grid,
            state + 0.5 * time_step * k2,
            time_value + 0.5 * time_step,
            excitation,
            porous_strength=porous_strength,
            closure=closure,
        )
        k4, d4 = spatial_rhs(
            grid,
            state + time_step * k3,
            time_value + time_step,
            excitation,
            porous_strength=porous_strength,
            closure=closure,
        )
        state += (
            time_step
            * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            / 6.0
        )
        stage_net_power = (
            d1["boundary_power"]
            - d1["dissipation"]
            + 2.0
            * (d2["boundary_power"] - d2["dissipation"])
            + 2.0
            * (d3["boundary_power"] - d3["dissipation"])
            + d4["boundary_power"]
            - d4["dissipation"]
        ) / 6.0
        cumulative_power += time_step * stage_net_power
        if not np.all(np.isfinite(state)):
            raise FloatingPointError(
                f"non-finite state at RK4 step {step + 1}"
            )
        if (step + 1) % save_stride == 0 or step + 1 == step_count:
            new_time = (step + 1) * time_step
            _, diagnostic = spatial_rhs(
                grid,
                state,
                new_time,
                excitation,
                porous_strength=porous_strength,
                closure=closure,
            )
            record(new_time, diagnostic)

    return ReferenceSimulation(
        nx=nx,
        ny=ny,
        excitation=excitation,
        final_time=final_time,
        time_step=time_step,
        step_count=step_count,
        times=np.asarray(saved_times),
        sensor_pressure={
            name: np.asarray(values)
            for name, values in saved_sensors.items()
        },
        energy=np.asarray(saved_energy),
        boundary_power=np.asarray(saved_boundary),
        dissipation=np.asarray(saved_dissipation),
        power_identity_defect=np.asarray(saved_identity),
        time_balance_defect=np.asarray(saved_time_defect),
        minimum_power_density=np.asarray(saved_minimum_power),
        porous_area=grid.represented_porous_area,
        porous_strength=porous_strength,
    )


def _relative_time_l2(
    candidate_times: Array,
    candidate: Array,
    reference_times: Array,
    reference: Array,
) -> float:
    interpolated = np.interp(
        candidate_times, reference_times, reference
    )
    numerator = np.trapezoid(
        (candidate - interpolated) ** 2, candidate_times
    )
    denominator = np.trapezoid(
        interpolated**2, candidate_times
    )
    return float(
        np.sqrt(numerator / max(denominator, 1.0e-30))
    )


def run_refinement_study(
    levels: Iterable[tuple[int, int]] = (
        (40, 20),
        (80, 40),
        (160, 80),
    ),
    *,
    excitation: Excitation = Excitation(),
    final_time: float = 2.4,
    cfl: float = 0.45,
    porous_strength: float = 1.0,
    closure: ClosureCallback | None = None,
) -> tuple[list[ReferenceSimulation], dict[str, object]]:
    """Run a three-level reference study and estimate observed sensor rates."""
    level_list = list(levels)
    if len(level_list) < 2:
        raise ValueError("at least two refinement levels are required")
    for (nx0, ny0), (nx1, ny1) in zip(
        level_list[:-1], level_list[1:]
    ):
        if nx1 <= nx0 or ny1 <= ny0:
            raise ValueError("levels must be strictly refined")

    simulations: list[ReferenceSimulation] = []
    level_wall_seconds: list[float] = []
    for nx, ny in level_list:
        before = time.perf_counter()
        simulations.append(
            simulate_reference(
            nx,
            ny,
            excitation=excitation,
            final_time=final_time,
            cfl=cfl,
            porous_strength=porous_strength,
            closure=closure,
        )
        )
        level_wall_seconds.append(time.perf_counter() - before)
    pair_errors: dict[str, list[float]] = {
        name: [] for name in SENSOR_POINTS
    }
    for coarse, fine in zip(simulations[:-1], simulations[1:]):
        for name in SENSOR_POINTS:
            pair_errors[name].append(
                _relative_time_l2(
                    coarse.times,
                    coarse.sensor_pressure[name],
                    fine.times,
                    fine.sensor_pressure[name],
                )
            )

    observed_rates: dict[str, list[float]] = {
        name: [] for name in SENSOR_POINTS
    }
    if len(simulations) >= 3:
        for name, errors in pair_errors.items():
            for index in range(len(errors) - 1):
                # This assumes the standard factor-two refinement used by the
                # default study.  It remains a useful diagnostic otherwise.
                observed_rates[name].append(
                    float(
                        np.log(
                            max(errors[index], 1.0e-30)
                            / max(errors[index + 1], 1.0e-30)
                        )
                        / np.log(2.0)
                    )
                )

    exact_area = (
        (INSERT_X[1] - INSERT_X[0])
        * (INSERT_Y[1] - INSERT_Y[0])
    )
    report: dict[str, object] = {
        "levels": [
            {
                **simulation.summary(),
                "wall_seconds": elapsed,
            }
            for simulation, elapsed in zip(
                simulations, level_wall_seconds
            )
        ],
        "sensor_pair_relative_l2": pair_errors,
        "sensor_observed_rates": observed_rates,
        "exact_insert_area": exact_area,
        "porous_strength": porous_strength,
        "maximum_insert_area_error": float(
            max(
                abs(simulation.porous_area - exact_area)
                for simulation in simulations
            )
        ),
    }
    return simulations, report


def verify_reference_implementation() -> dict[str, float | int]:
    """Fast deterministic algebraic and short-rollout verification."""
    grid = build_grid(18, 9)
    rng = np.random.default_rng(20260730)
    state = 0.12 * rng.standard_normal(grid.state_size)
    excitation = Excitation(amplitude=0.71, omega=8.3)
    _, diagnostic = spatial_rhs(
        grid,
        state,
        0.31,
        excitation,
        porous_strength=3.0,
    )

    velocity_x = rng.standard_normal((grid.ny, grid.nx))
    velocity_y = rng.standard_normal((grid.ny, grid.nx))
    drag0_x, drag0_y, power0 = passive_reference_drag(
        grid, velocity_x, velocity_y, porous_strength=0.0
    )
    drag1_x, drag1_y, power1 = passive_reference_drag(
        grid, velocity_x, velocity_y, porous_strength=1.0
    )
    drag4_x, drag4_y, power4 = passive_reference_drag(
        grid, velocity_x, velocity_y, porous_strength=4.0
    )
    air_error = max(
        float(
            np.max(
                np.abs(drag0_x - AIR_DAMPING * velocity_x)
            )
        ),
        float(
            np.max(
                np.abs(drag0_y - AIR_DAMPING * velocity_y)
            )
        ),
    )
    affine_strength_error = max(
        float(
            np.max(
                np.abs(
                    drag4_x
                    - (
                        drag0_x
                        + 4.0 * (drag1_x - drag0_x)
                    )
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    drag4_y
                    - (
                        drag0_y
                        + 4.0 * (drag1_y - drag0_y)
                    )
                )
            )
        ),
    )

    def air_callback(features: Array, velocity: Array) -> Array:
        if features.shape[1] != 5:
            raise AssertionError("callback received incorrect features")
        return AIR_DAMPING * velocity

    short = simulate_reference(
        24,
        12,
        excitation=excitation,
        final_time=0.18,
        cfl=0.4,
        output_interval=0.01,
    )
    callback_short = simulate_reference(
        12,
        6,
        excitation=excitation,
        final_time=0.05,
        cfl=0.4,
        output_interval=0.01,
        closure=air_callback,
    )
    exact_area = (
        (INSERT_X[1] - INSERT_X[0])
        * (INSERT_Y[1] - INSERT_Y[0])
    )
    checks: dict[str, float | int] = {
        "random_state_power_identity_defect": diagnostic[
            "power_identity_defect"
        ],
        "minimum_random_drag_power_strength_0": float(
            np.min(power0)
        ),
        "minimum_random_drag_power_strength_1": float(
            np.min(power1)
        ),
        "minimum_random_drag_power_strength_4": float(
            np.min(power4)
        ),
        "air_only_scaling_error": air_error,
        "affine_strength_scaling_error": affine_strength_error,
        "short_rollout_power_identity_defect": float(
            np.max(short.power_identity_defect)
        ),
        "callback_rollout_power_identity_defect": float(
            np.max(callback_short.power_identity_defect)
        ),
        "short_rollout_relative_time_balance_defect": short.summary()[
            "relative_time_balance_defect"
        ],
        "insert_area_error": abs(
            grid.represented_porous_area - exact_area
        ),
        "short_rollout_steps": short.step_count,
    }
    if checks["random_state_power_identity_defect"] > 2.0e-13:
        raise AssertionError("spatial power identity is not exact")
    for key in (
        "minimum_random_drag_power_strength_0",
        "minimum_random_drag_power_strength_1",
        "minimum_random_drag_power_strength_4",
    ):
        if checks[key] < -2.0e-13:
            raise AssertionError(
                f"reference drag lost passivity: {key}"
            )
    if checks["air_only_scaling_error"] > 2.0e-15:
        raise AssertionError("zero strength changed air damping")
    if checks["affine_strength_scaling_error"] > 2.0e-14:
        raise AssertionError("porous tensor scaling is not affine")
    if checks["short_rollout_power_identity_defect"] > 2.0e-13:
        raise AssertionError("rollout power identity is not exact")
    if checks["callback_rollout_power_identity_defect"] > 2.0e-13:
        raise AssertionError("callback broke the power identity")
    if checks["insert_area_error"] > 2.0e-14:
        raise AssertionError("cut-cell insert area is not invariant")
    return checks


def _parse_levels(text: str) -> list[tuple[int, int]]:
    levels: list[tuple[int, int]] = []
    for block in text.split(","):
        try:
            nx_text, ny_text = block.lower().split("x", maxsplit=1)
            levels.append((int(nx_text), int(ny_text)))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                "levels must look like 40x20,80x40,160x80"
            ) from exc
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Independent energy-compatible structured FV reference for "
            "the nonlinear porous-acoustic duct."
        )
    )
    parser.add_argument(
        "--levels",
        type=_parse_levels,
        default=[(40, 20), (80, 40), (160, 80)],
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=2.4,
        help=(
            "2.4 captures the transmitted pulse; at 1.65 the main burst "
            "has not yet reached the downstream sensor at x=1.58."
        ),
    )
    parser.add_argument("--amplitude", type=float, default=1.35)
    parser.add_argument("--omega", type=float, default=10.5)
    parser.add_argument("--cfl", type=float, default=0.45)
    parser.add_argument(
        "--porous-strength",
        type=float,
        default=1.0,
        help=(
            "non-negative multiplier of the passive porous tensor; "
            "the isotropic air damping remains unchanged"
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="run only the fast algebraic and short-rollout checks",
    )
    arguments = parser.parse_args()
    verification = verify_reference_implementation()
    if arguments.verify_only:
        print(json.dumps({"verification": verification}, indent=2))
        return
    excitation = Excitation(
        amplitude=arguments.amplitude,
        omega=arguments.omega,
    )
    _, report = run_refinement_study(
        arguments.levels,
        excitation=excitation,
        final_time=arguments.final_time,
        cfl=arguments.cfl,
        porous_strength=arguments.porous_strength,
    )
    print(
        json.dumps(
            {"verification": verification, "refinement": report},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
