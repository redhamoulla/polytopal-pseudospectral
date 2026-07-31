"""Short, deterministic validations used by the public release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .closures import DirectDragClosure, StructuredSPDClosure
from .fv_reference import Excitation, ReferenceSimulation, simulate_reference


def _relative_signal_error(
    prediction: ReferenceSimulation,
    reference: ReferenceSimulation,
    sensor: str,
) -> float:
    target = np.interp(
        prediction.times,
        reference.times,
        reference.sensor_pressure[sensor],
    )
    return float(
        np.linalg.norm(prediction.sensor_pressure[sensor] - target)
        / max(np.linalg.norm(target), 1.0e-30)
    )


def evaluate_pretrained_closures(
    asset_directory: str | Path,
    *,
    nx: int = 80,
    ny: int = 40,
    final_time: float = 2.4,
    cfl: float = 0.45,
) -> dict[str, Any]:
    """Evaluate the released SPD and direct checkpoints on an independent FV grid."""

    assets = Path(asset_directory)
    excitation = Excitation(amplitude=1.35, omega=9.0, burst_duration=0.72)
    reference = simulate_reference(
        nx,
        ny,
        excitation=excitation,
        final_time=final_time,
        cfl=cfl,
        porous_strength=4.0,
    )
    models = {
        "structured-spd": StructuredSPDClosure.load(
            assets / "structured_spd.npz"
        ),
        "direct-mlp": DirectDragClosure.load(assets / "direct_mlp.npz"),
    }

    results: dict[str, Any] = {
        "configuration": {
            "nx": nx,
            "ny": ny,
            "final_time": final_time,
            "cfl": cfl,
            "excitation_amplitude": excitation.amplitude,
            "excitation_omega": excitation.omega,
            "burst_duration": excitation.burst_duration,
        },
        "models": {},
    }
    for name, model in models.items():
        prediction = simulate_reference(
            nx,
            ny,
            excitation=excitation,
            final_time=final_time,
            cfl=cfl,
            closure=model,
        )
        results["models"][name] = {
            "upstream_pressure_nrmse": _relative_signal_error(
                prediction, reference, "upstream"
            ),
            "downstream_pressure_nrmse": _relative_signal_error(
                prediction, reference, "downstream"
            ),
            "energy_nrmse": float(
                np.linalg.norm(prediction.energy - reference.energy)
                / max(np.linalg.norm(reference.energy), 1.0e-30)
            ),
            "minimum_power_density": float(
                np.min(prediction.minimum_power_density)
            ),
            "maximum_power_identity_defect": float(
                np.max(prediction.power_identity_defect)
            ),
        }
    return results


def assert_reference_metrics(
    observed: dict[str, Any],
    reference_path: str | Path,
    *,
    relative_tolerance: float = 2.0e-8,
    absolute_tolerance: float = 2.0e-10,
) -> None:
    """Check the canonical 80x40 result without requiring bitwise BLAS identity."""

    expected = json.loads(Path(reference_path).read_text(encoding="utf-8"))[
        "fv80x40"
    ]["models"]
    if observed["configuration"]["nx"] != 80 or observed["configuration"]["ny"] != 40:
        raise ValueError("reference metrics apply only to the canonical 80x40 grid")

    fields = (
        "upstream_pressure_nrmse",
        "downstream_pressure_nrmse",
        "energy_nrmse",
        "minimum_power_density",
        "maximum_power_identity_defect",
    )
    failures: list[str] = []
    for model_name, expected_metrics in expected.items():
        observed_metrics = observed["models"][model_name]
        for field in fields:
            if not np.isclose(
                observed_metrics[field],
                expected_metrics[field],
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            ):
                failures.append(
                    f"{model_name}.{field}: observed={observed_metrics[field]:.16g}, "
                    f"expected={expected_metrics[field]:.16g}"
                )
    if failures:
        raise AssertionError("reference regression failed:\n" + "\n".join(failures))

    if observed["models"]["structured-spd"]["minimum_power_density"] < -1.0e-12:
        raise AssertionError("the structured SPD closure violated passivity")
    if observed["models"]["direct-mlp"]["minimum_power_density"] >= 0.0:
        raise AssertionError("the direct baseline no longer exposes the recorded violation")
