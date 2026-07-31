from pathlib import Path

import numpy as np

from polytopal_ph.closures import DirectDragClosure, StructuredSPDClosure


ASSETS = Path(__file__).resolve().parents[1] / "examples" / "assets"


def test_released_spd_checkpoint_is_passive() -> None:
    model = StructuredSPDClosure.load(ASSETS / "structured_spd.npz")
    rng = np.random.default_rng(23)
    features = rng.normal(size=(512, model.feature_dim))
    velocity = rng.normal(size=(512, 2))
    drag = model.predict(features, velocity)
    power = np.einsum("ij,ij->i", velocity, drag)
    assert np.min(power) >= -1.0e-12


def test_released_models_return_finite_drag() -> None:
    rng = np.random.default_rng(29)
    features = rng.normal(size=(64, 5))
    velocity = rng.normal(size=(64, 2))
    for model in (
        StructuredSPDClosure.load(ASSETS / "structured_spd.npz"),
        DirectDragClosure.load(ASSETS / "direct_mlp.npz"),
    ):
        prediction = model.predict(features, velocity)
        assert prediction.shape == velocity.shape
        assert np.all(np.isfinite(prediction))
