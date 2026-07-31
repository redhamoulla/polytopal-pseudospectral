import numpy as np

from polytopal_ph.mesh import (
    assemble_cells,
    boundary_coefficients_at_time,
    corner_convergence,
    exact_multicell_plane_wave,
    interface_diagnostics,
    midpoint_boundary_power,
    multicell_plane_wave_errors,
    multicell_structural_diagnostics,
    split_test_pentagon,
)
from polytopal_ph.cell import PlaneWave


def test_interface_assembly_is_power_preserving() -> None:
    mesh = assemble_cells(split_test_pentagon(), 6)
    diagnostics = multicell_structural_diagnostics(mesh)
    assert mesh.interface_dimension == 6
    assert diagnostics["constraint_rank"] == 6
    assert diagnostics["constraint_basis_rel"] < 2e-12
    assert diagnostics["reduced_skew_rel"] < 2e-12
    assert diagnostics["open_dirac_skew_rel"] < 2e-12


def test_interface_flux_and_multiplier_converge() -> None:
    wave = PlaneWave(np.array([1.35, -0.92]))
    pressure_errors = []
    for degree in (3, 5, 7):
        mesh = assemble_cells(split_test_pentagon(), degree)
        _, full_state = exact_multicell_plane_wave(mesh, wave, 0.3)
        diagnostics = interface_diagnostics(mesh, wave, full_state, 0.3)
        pressure_errors.append(
            diagnostics["interface_pressure_l2_relative"]
        )
        assert diagnostics["interface_flux_jump_relative"] < 2e-12
        assert diagnostics["interface_power_abs"] < 2e-12
    assert pressure_errors[2] < pressure_errors[1] < pressure_errors[0]


def test_multicell_plane_wave_converges() -> None:
    wave = PlaneWave(np.array([1.35, -0.92]))
    errors = []
    for degree in (3, 5, 7):
        mesh = assemble_cells(split_test_pentagon(), degree)
        _, full_state = exact_multicell_plane_wave(mesh, wave, 0.3)
        errors.append(
            multicell_plane_wave_errors(
                mesh, wave, full_state, 0.3
            )["pressure_l2_relative"]
        )
    assert errors[2] < errors[1] < errors[0]


def test_multicell_midpoint_power_balance() -> None:
    wave = PlaneWave(np.array([1.35, -0.92]))
    mesh = assemble_cells(split_test_pentagon(), 5)
    metrics, _ = midpoint_boundary_power(
        mesh, wave, final_time=0.08, time_step=0.004
    )
    assert metrics["relative_balance_defect"] < 2e-12


def test_reentrant_corner_breaks_exponential_p_convergence() -> None:
    rows, fit = corner_convergence(range(4, 11))
    final = rows[-1]
    assert final["analytic_l2_relative"] < 1e-11
    assert final["corner_l2_relative"] > 1e-4
    assert final["corner_l2_relative"] < rows[0]["corner_l2_relative"]
    assert 1.0 < fit["corner_algebraic_order"] < 5.0
    assert fit["analytic_digits_per_degree"] > 0.8
