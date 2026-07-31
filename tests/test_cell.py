import numpy as np

from polytopal_ph.cell import (
    PlaneWave,
    acoustic_matrices,
    build_polygon_complex,
    exact_semidiscrete_plane_wave,
    midpoint_power_simulation,
    plane_wave_errors,
    robustness_polygons,
    structural_diagnostics,
)


def test_exact_complex_and_stokes() -> None:
    complex_ = build_polygon_complex(6)
    diagnostics = structural_diagnostics(complex_)
    assert diagnostics["euler_characteristic"] == 1.0
    assert diagnostics["betti0"] == 1.0
    assert diagnostics["betti1"] == 0.0
    assert diagnostics["betti2"] == 0.0
    assert diagnostics["d_squared_modal_rel"] < 2e-12
    assert diagnostics["d_squared_nodal_rel"] < 2e-10
    assert diagnostics["stokes_modal_rel"] < 2e-12
    assert diagnostics["stokes_nodal_rel"] < 2e-10
    assert diagnostics["random_dirac_power_abs"] < 2e-10


def test_pairing_quotient_and_minimal_ports() -> None:
    complex_ = build_polygon_complex(7)
    diagnostics = structural_diagnostics(complex_)
    assert diagnostics["rank_pair02"] == complex_.n2
    assert diagnostics["pair02_left_nullity"] == complex_.n0 - complex_.n2
    assert diagnostics["minimal_port_dimension"] <= complex_.n1
    assert diagnostics["port_compression_rel"] < 2e-12


def test_open_dirac_matrix_and_midpoint_power() -> None:
    complex_ = build_polygon_complex(5)
    internal, boundary_input = acoustic_matrices(complex_)
    assert np.linalg.norm(internal + internal.T, ord=np.inf) < 2e-12
    assert boundary_input.shape[0] == complex_.n1 + complex_.n2
    wave = PlaneWave(np.array([1.35, -0.92]))
    metrics, _ = midpoint_power_simulation(
        complex_, wave, final_time=0.08, time_step=0.004
    )
    assert metrics["relative_balance_defect"] < 2e-12


def test_plane_wave_error_decreases() -> None:
    wave = PlaneWave(np.array([1.35, -0.92]))
    errors = []
    for degree in (3, 5, 7):
        complex_ = build_polygon_complex(degree)
        state = exact_semidiscrete_plane_wave(complex_, wave, final_time=0.3)
        errors.append(
            plane_wave_errors(
                complex_, wave, state, final_time=0.3
            )["pressure_l2_relative"]
        )
    assert errors[2] < errors[1] < errors[0]


def test_geometry_and_affine_robustness() -> None:
    for polygon in robustness_polygons().values():
        diagnostics = structural_diagnostics(
            build_polygon_complex(5, vertices=polygon)
        )
        assert diagnostics["d_squared_nodal_rel"] < 2e-10
        assert diagnostics["stokes_nodal_rel"] < 2e-10
        assert diagnostics["betti0"] == 1.0
        assert diagnostics["betti1"] == 0.0
        assert diagnostics["betti2"] == 0.0
