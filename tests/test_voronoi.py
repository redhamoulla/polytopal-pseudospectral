import numpy as np

from polytopal_ph.voronoi import (
    relaxed_voronoi_mesh,
    run_one,
    validate_voronoi_mesh,
)


def test_bounded_voronoi_mesh_is_conforming() -> None:
    _, polygons = relaxed_voronoi_mesh(8, seed=11, lloyd_steps=1)
    diagnostics = validate_voronoi_mesh(polygons, expected_cells=8)
    assert abs(diagnostics["area_sum"] - 1.0) < 5.0e-10
    assert diagnostics["euler_disk"] == 1
    assert diagnostics["internal_edge_count"] > 0


def test_small_voronoi_hybrid_constraint_has_full_rank() -> None:
    diagnostics = run_one(
        cell_count=6,
        degree=2,
        seed=7,
        lloyd_steps=1,
    )
    assert diagnostics["constraint_rank"] == diagnostics["constraint_count"]
    assert diagnostics["constraint_rank_raw"] == diagnostics["constraint_count"]
    assert diagnostics["sigma_min_G"] > diagnostics["rank_tolerance"]
    assert (
        diagnostics["sigma_min_column_equilibrated_G"]
        > diagnostics["column_equilibrated_rank_tolerance"]
    )
    assert diagnostics["constraint_residual_GtZ_relative"] < 5.0e-13
    assert diagnostics["basis_orthogonality_relative"] < 5.0e-13
    assert diagnostics["reduced_skew_relative_fro"] < 5.0e-13
    assert np.isfinite(diagnostics["condition_G_2"])
