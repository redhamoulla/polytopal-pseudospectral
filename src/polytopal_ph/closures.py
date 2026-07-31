"""Small NumPy neural closures for nonlinear porous-acoustic drag.

The module deliberately has no machine-learning dependency.  It provides two
models with the same training interface:

``StructuredSPDClosure``
    maps local features to a lower-triangular matrix ``L`` and predicts
    ``drag = L L.T velocity``.  Positive dissipation is therefore guaranteed.

``DirectDragClosure``
    maps local features and velocity directly to drag and is used as an
    unconstrained baseline.

Both models use a two-hidden-layer tanh MLP, mini-batch Adam, feature
standardisation, scalar physical-unit scaling for velocity and drag, and
optional ``.npz`` serialisation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


Array = np.ndarray


def _as_matrix(value: Array, columns: int | None = None, name: str = "array") -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if columns is not None and result.shape[1] != columns:
        raise ValueError(f"{name} must have shape (n, {columns}).")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains a non-finite value.")
    return result


def _softplus(value: Array) -> Array:
    return np.maximum(value, 0.0) + np.log1p(np.exp(-np.abs(value)))


def _sigmoid(value: Array) -> Array:
    result = np.empty_like(value)
    positive = value >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


@dataclass
class _FeatureStandardizer:
    mean: Array | None = None
    scale: Array | None = None

    def fit(self, values: Array) -> "_FeatureStandardizer":
        self.mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        self.scale = np.where(scale > 1.0e-12, scale, 1.0)
        return self

    def transform(self, values: Array) -> Array:
        if self.mean is None or self.scale is None:
            raise RuntimeError("The feature standardizer has not been fitted.")
        return (values - self.mean) / self.scale


class _TanhMLP:
    """Two-hidden-layer tanh MLP with manually differentiated NumPy kernels."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        seed: int,
    ) -> None:
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must be positive.")
        if len(hidden_dims) != 2 or any(width < 1 for width in hidden_dims):
            raise ValueError("hidden_dims must contain exactly two positive widths.")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = (int(hidden_dims[0]), int(hidden_dims[1]))
        rng = np.random.default_rng(seed)
        sizes = (self.input_dim, *self.hidden_dims, self.output_dim)
        self.parameters: list[Array] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            bound = np.sqrt(6.0 / (fan_in + fan_out))
            weight = rng.uniform(-bound, bound, size=(fan_in, fan_out))
            bias = np.zeros(fan_out, dtype=float)
            self.parameters.extend((weight, bias))

    @property
    def parameter_count(self) -> int:
        return int(sum(parameter.size for parameter in self.parameters))

    def forward(self, values: Array, cache: bool = False) -> Array | tuple[Array, tuple[Array, ...]]:
        w1, b1, w2, b2, w3, b3 = self.parameters
        h1 = np.tanh(values @ w1 + b1)
        h2 = np.tanh(h1 @ w2 + b2)
        output = h2 @ w3 + b3
        if cache:
            return output, (values, h1, h2)
        return output

    def backward(self, gradient: Array, cache: tuple[Array, ...]) -> list[Array]:
        values, h1, h2 = cache
        w1, _, w2, _, w3, _ = self.parameters
        grad_w3 = h2.T @ gradient
        grad_b3 = np.sum(gradient, axis=0)
        grad_h2 = (gradient @ w3.T) * (1.0 - h2 * h2)
        grad_w2 = h1.T @ grad_h2
        grad_b2 = np.sum(grad_h2, axis=0)
        grad_h1 = (grad_h2 @ w2.T) * (1.0 - h1 * h1)
        grad_w1 = values.T @ grad_h1
        grad_b1 = np.sum(grad_h1, axis=0)
        return [grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3]


class _Adam:
    def __init__(
        self,
        parameters: Sequence[Array],
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1.0e-8,
    ) -> None:
        self.parameters = parameters
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.first_moments = [np.zeros_like(parameter) for parameter in parameters]
        self.second_moments = [np.zeros_like(parameter) for parameter in parameters]
        self.step_number = 0

    def step(self, gradients: Sequence[Array]) -> None:
        self.step_number += 1
        correction1 = 1.0 - self.beta1**self.step_number
        correction2 = 1.0 - self.beta2**self.step_number
        for parameter, gradient, moment1, moment2 in zip(
            self.parameters, gradients, self.first_moments, self.second_moments
        ):
            moment1 *= self.beta1
            moment1 += (1.0 - self.beta1) * gradient
            moment2 *= self.beta2
            moment2 += (1.0 - self.beta2) * gradient * gradient
            estimate1 = moment1 / correction1
            estimate2 = moment2 / correction2
            parameter -= self.learning_rate * estimate1 / (np.sqrt(estimate2) + self.epsilon)


class _ClosureBase:
    """Shared data validation, optimisation loop, and persistence."""

    _model_name = "base"

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: Sequence[int] = (32, 32),
        seed: int = 0,
    ) -> None:
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        self.feature_dim = int(feature_dim)
        self.hidden_dims = (int(hidden_dims[0]), int(hidden_dims[1]))
        self.seed = int(seed)
        self.feature_scaler = _FeatureStandardizer()
        self.velocity_scale = 1.0
        self.drag_scale = 1.0
        self.history: list[dict[str, float]] = []
        self._fitted = False

    @property
    def parameter_count(self) -> int:
        return self.network.parameter_count

    def _validate_triplet(
        self, features: Array, velocity: Array, drag: Array | None = None
    ) -> tuple[Array, Array, Array | None]:
        x = _as_matrix(features, self.feature_dim, "features")
        v = _as_matrix(velocity, 2, "velocity")
        if x.shape[0] != v.shape[0]:
            raise ValueError("features and velocity must have the same number of rows.")
        y = None if drag is None else _as_matrix(drag, 2, "drag")
        if y is not None and y.shape[0] != x.shape[0]:
            raise ValueError("features and drag must have the same number of rows.")
        return x, v, y

    def _fit_scalers(self, features: Array, velocity: Array, drag: Array) -> None:
        self.feature_scaler.fit(features)
        velocity_rms = float(np.sqrt(np.mean(velocity * velocity)))
        drag_rms = float(np.sqrt(np.mean(drag * drag)))
        self.velocity_scale = max(velocity_rms, 1.0e-12)
        self.drag_scale = max(drag_rms, 1.0e-12)

    def _normalise_inputs(self, features: Array, velocity: Array) -> tuple[Array, Array]:
        return (
            self.feature_scaler.transform(features),
            velocity / self.velocity_scale,
        )

    def fit(
        self,
        features: Array,
        velocity: Array,
        drag: Array,
        *,
        epochs: int = 500,
        batch_size: int = 128,
        learning_rate: float = 2.0e-3,
        weight_decay: float = 0.0,
        validation_data: tuple[Array, Array, Array] | None = None,
        verbose: bool = False,
    ) -> "_ClosureBase":
        x, v, y_optional = self._validate_triplet(features, velocity, drag)
        assert y_optional is not None
        y = y_optional
        if x.shape[0] == 0:
            raise ValueError("At least one training sample is required.")
        if epochs < 1 or batch_size < 1 or learning_rate <= 0.0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive.")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")

        self._fit_scalers(x, v, y)
        x_scaled, v_scaled = self._normalise_inputs(x, v)
        y_scaled = y / self.drag_scale

        validation_scaled: tuple[Array, Array, Array] | None = None
        if validation_data is not None:
            x_val, v_val, y_val_optional = self._validate_triplet(*validation_data)
            assert y_val_optional is not None
            x_val_scaled, v_val_scaled = self._normalise_inputs(x_val, v_val)
            validation_scaled = (x_val_scaled, v_val_scaled, y_val_optional / self.drag_scale)

        optimizer = _Adam(self.network.parameters, learning_rate)
        rng = np.random.default_rng(self.seed + 104729)
        n_samples = x.shape[0]
        effective_batch = min(int(batch_size), n_samples)
        self.history = []

        for epoch in range(1, int(epochs) + 1):
            permutation = rng.permutation(n_samples)
            for start in range(0, n_samples, effective_batch):
                indices = permutation[start : start + effective_batch]
                prediction, cache = self._forward_scaled(
                    x_scaled[indices], v_scaled[indices], cache=True
                )
                difference = prediction - y_scaled[indices]
                output_gradient = 2.0 * difference / difference.size
                gradients = self._backward_scaled(output_gradient, cache)
                if weight_decay:
                    gradients = [
                        gradient + weight_decay * parameter
                        if parameter.ndim == 2
                        else gradient
                        for parameter, gradient in zip(self.network.parameters, gradients)
                    ]
                optimizer.step(gradients)

            train_prediction = self._forward_scaled(x_scaled, v_scaled)
            record = {
                "epoch": float(epoch),
                "train_loss": float(np.mean((train_prediction - y_scaled) ** 2)),
            }
            if validation_scaled is not None:
                x_val_scaled, v_val_scaled, y_val_scaled = validation_scaled
                val_prediction = self._forward_scaled(x_val_scaled, v_val_scaled)
                record["validation_loss"] = float(
                    np.mean((val_prediction - y_val_scaled) ** 2)
                )
            self.history.append(record)
            if verbose and (epoch == 1 or epoch == epochs or epoch % max(epochs // 10, 1) == 0):
                suffix = (
                    f", val={record['validation_loss']:.4e}"
                    if "validation_loss" in record
                    else ""
                )
                print(f"epoch {epoch:5d}: train={record['train_loss']:.4e}{suffix}")

        self._fitted = True
        return self

    def predict(self, features: Array, velocity: Array) -> Array:
        if not self._fitted:
            raise RuntimeError("Call fit before predict.")
        x, v, _ = self._validate_triplet(features, velocity)
        x_scaled, v_scaled = self._normalise_inputs(x, v)
        return self.drag_scale * self._forward_scaled(x_scaled, v_scaled)

    def _metadata(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "feature_dim": self.feature_dim,
            "hidden_dims": list(self.hidden_dims),
            "seed": self.seed,
            "velocity_scale": self.velocity_scale,
            "drag_scale": self.drag_scale,
            "fitted": self._fitted,
        }

    def save(self, path: str | Path) -> None:
        if self.feature_scaler.mean is None or self.feature_scaler.scale is None:
            raise RuntimeError("Call fit before save.")
        payload: dict[str, Array] = {
            "metadata": np.asarray(json.dumps(self._metadata())),
            "feature_mean": self.feature_scaler.mean,
            "feature_scale": self.feature_scaler.scale,
        }
        payload.update(
            {f"parameter_{index}": parameter for index, parameter in enumerate(self.network.parameters)}
        )
        np.savez_compressed(Path(path), **payload)

    def _restore(self, archive: Any, metadata: dict[str, Any]) -> None:
        for index, parameter in enumerate(self.network.parameters):
            saved = np.asarray(archive[f"parameter_{index}"], dtype=float)
            if saved.shape != parameter.shape:
                raise ValueError("Saved parameter shape is incompatible with the model.")
            parameter[...] = saved
        self.feature_scaler.mean = np.asarray(archive["feature_mean"], dtype=float)
        self.feature_scaler.scale = np.asarray(archive["feature_scale"], dtype=float)
        self.velocity_scale = float(metadata["velocity_scale"])
        self.drag_scale = float(metadata["drag_scale"])
        self._fitted = bool(metadata["fitted"])

    def _forward_scaled(
        self, features: Array, velocity: Array, cache: bool = False
    ) -> Array | tuple[Array, Any]:
        raise NotImplementedError

    def _backward_scaled(self, gradient: Array, cache: Any) -> list[Array]:
        raise NotImplementedError


class StructuredSPDClosure(_ClosureBase):
    """Learn a symmetric positive-definite 2-D drag tensor.

    The network outputs ``(a, b, c)`` and constructs

    ``L = [[softplus(a) + eps, 0], [b, softplus(c) + eps]]``.

    Consequently ``velocity.T @ predict_matrix(features) @ velocity`` is
    non-negative by construction, independently of the fitted weights.
    """

    _model_name = "structured_spd"

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: Sequence[int] = (32, 32),
        seed: int = 0,
        min_diagonal: float = 1.0e-4,
    ) -> None:
        super().__init__(feature_dim, hidden_dims, seed)
        if min_diagonal <= 0.0:
            raise ValueError("min_diagonal must be positive.")
        self.min_diagonal = float(min_diagonal)
        self.network = _TanhMLP(feature_dim, 3, self.hidden_dims, seed)

    def _scaled_matrix_and_raw(self, features: Array) -> tuple[Array, Array]:
        raw = self.network.forward(features)
        diagonal1 = _softplus(raw[:, 0]) + self.min_diagonal
        lower21 = raw[:, 1]
        diagonal2 = _softplus(raw[:, 2]) + self.min_diagonal
        matrix = np.empty((features.shape[0], 2, 2), dtype=float)
        matrix[:, 0, 0] = diagonal1 * diagonal1
        matrix[:, 0, 1] = diagonal1 * lower21
        matrix[:, 1, 0] = matrix[:, 0, 1]
        matrix[:, 1, 1] = lower21 * lower21 + diagonal2 * diagonal2
        return matrix, raw

    def _forward_scaled(
        self, features: Array, velocity: Array, cache: bool = False
    ) -> Array | tuple[Array, Any]:
        if cache:
            raw, network_cache = self.network.forward(features, cache=True)
            diagonal1 = _softplus(raw[:, 0]) + self.min_diagonal
            lower21 = raw[:, 1]
            diagonal2 = _softplus(raw[:, 2]) + self.min_diagonal
            prediction = np.column_stack(
                (
                    diagonal1 * diagonal1 * velocity[:, 0]
                    + diagonal1 * lower21 * velocity[:, 1],
                    diagonal1 * lower21 * velocity[:, 0]
                    + (lower21 * lower21 + diagonal2 * diagonal2) * velocity[:, 1],
                )
            )
            closure_cache = (network_cache, raw, diagonal1, lower21, diagonal2, velocity)
            return prediction, closure_cache

        matrix, _ = self._scaled_matrix_and_raw(features)
        return np.einsum("nij,nj->ni", matrix, velocity)

    def _backward_scaled(self, gradient: Array, cache: Any) -> list[Array]:
        network_cache, raw, diagonal1, lower21, diagonal2, velocity = cache
        velocity1, velocity2 = velocity[:, 0], velocity[:, 1]
        grad1, grad2 = gradient[:, 0], gradient[:, 1]
        grad_diagonal1 = (
            grad1 * (2.0 * diagonal1 * velocity1 + lower21 * velocity2)
            + grad2 * lower21 * velocity1
        )
        grad_lower21 = (
            grad1 * diagonal1 * velocity2
            + grad2 * (diagonal1 * velocity1 + 2.0 * lower21 * velocity2)
        )
        grad_diagonal2 = grad2 * (2.0 * diagonal2 * velocity2)
        grad_raw = np.column_stack(
            (
                grad_diagonal1 * _sigmoid(raw[:, 0]),
                grad_lower21,
                grad_diagonal2 * _sigmoid(raw[:, 2]),
            )
        )
        return self.network.backward(grad_raw, network_cache)

    def predict_matrix(self, features: Array) -> Array:
        """Return the physical-unit SPD tensor for every feature row."""
        if not self._fitted:
            raise RuntimeError("Call fit before predict_matrix.")
        x = _as_matrix(features, self.feature_dim, "features")
        x_scaled = self.feature_scaler.transform(x)
        scaled_matrix, _ = self._scaled_matrix_and_raw(x_scaled)
        return (self.drag_scale / self.velocity_scale) * scaled_matrix

    def dissipation(self, features: Array, velocity: Array) -> Array:
        """Return ``v.T R v`` sample by sample (always non-negative up to roundoff)."""
        matrix = self.predict_matrix(features)
        v = _as_matrix(velocity, 2, "velocity")
        if matrix.shape[0] != v.shape[0]:
            raise ValueError("features and velocity must have the same number of rows.")
        return np.einsum("ni,nij,nj->n", v, matrix, v)

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        metadata["min_diagonal"] = self.min_diagonal
        return metadata

    @classmethod
    def load(cls, path: str | Path) -> "StructuredSPDClosure":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("model_name") != cls._model_name:
                raise ValueError("The archive does not contain a StructuredSPDClosure.")
            model = cls(
                feature_dim=int(metadata["feature_dim"]),
                hidden_dims=metadata["hidden_dims"],
                seed=int(metadata["seed"]),
                min_diagonal=float(metadata["min_diagonal"]),
            )
            model._restore(archive, metadata)
        return model


class DirectDragClosure(_ClosureBase):
    """Unconstrained MLP baseline mapping ``(features, velocity)`` to drag.

    ``odd_symmetry=True`` antisymmetrises the saved model at inference:
    ``r_odd(v) = (r(v) - r(-v)) / 2``.  The flag is serialised so that a
    reloaded model reproduces the closure used in a rollout.
    """

    _model_name = "direct_drag"

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: Sequence[int] = (32, 32),
        seed: int = 0,
        odd_symmetry: bool = False,
    ) -> None:
        super().__init__(feature_dim, hidden_dims, seed)
        self.odd_symmetry = bool(odd_symmetry)
        self.network = _TanhMLP(feature_dim + 2, 2, self.hidden_dims, seed)

    def _forward_scaled(
        self, features: Array, velocity: Array, cache: bool = False
    ) -> Array | tuple[Array, Any]:
        inputs = np.column_stack((features, velocity))
        return self.network.forward(inputs, cache=cache)

    def _backward_scaled(self, gradient: Array, cache: Any) -> list[Array]:
        return self.network.backward(gradient, cache)

    def predict(self, features: Array, velocity: Array) -> Array:
        if not self._fitted:
            raise RuntimeError("Call fit before predict.")
        x, v, _ = self._validate_triplet(features, velocity)
        x_scaled, v_scaled = self._normalise_inputs(x, v)
        prediction = self.drag_scale * self._forward_scaled(x_scaled, v_scaled)
        if not self.odd_symmetry:
            return prediction
        reversed_prediction = self.drag_scale * self._forward_scaled(
            x_scaled, -v_scaled
        )
        return 0.5 * (prediction - reversed_prediction)

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        metadata["odd_symmetry"] = self.odd_symmetry
        return metadata

    @classmethod
    def load(cls, path: str | Path) -> "DirectDragClosure":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("model_name") != cls._model_name:
                raise ValueError("The archive does not contain a DirectDragClosure.")
            model = cls(
                feature_dim=int(metadata["feature_dim"]),
                hidden_dims=metadata["hidden_dims"],
                seed=int(metadata["seed"]),
                odd_symmetry=bool(metadata.get("odd_symmetry", False)),
            )
            model._restore(archive, metadata)
        return model


def _self_test() -> None:
    rng = np.random.default_rng(17)
    features = rng.normal(size=(600, 3))
    velocity = rng.normal(size=(600, 2))
    diagonal1 = 0.75 + 0.15 * np.tanh(features[:, 0])
    lower21 = 0.12 * np.tanh(features[:, 1])
    diagonal2 = 0.55 + 0.10 * np.tanh(features[:, 2])
    true_matrix = np.empty((features.shape[0], 2, 2))
    true_matrix[:, 0, 0] = diagonal1**2
    true_matrix[:, 0, 1] = diagonal1 * lower21
    true_matrix[:, 1, 0] = true_matrix[:, 0, 1]
    true_matrix[:, 1, 1] = lower21**2 + diagonal2**2
    drag = np.einsum("nij,nj->ni", true_matrix, velocity)

    train = np.arange(500)
    test = np.arange(500, 600)
    structured = StructuredSPDClosure(3, hidden_dims=(16, 16), seed=3).fit(
        features[train],
        velocity[train],
        drag[train],
        epochs=500,
        batch_size=64,
        learning_rate=3.0e-3,
    )
    baseline = DirectDragClosure(3, hidden_dims=(16, 16), seed=3).fit(
        features[train],
        velocity[train],
        drag[train],
        epochs=500,
        batch_size=64,
        learning_rate=3.0e-3,
    )
    structured_error = float(
        np.mean((structured.predict(features[test], velocity[test]) - drag[test]) ** 2)
    )
    baseline_error = float(
        np.mean((baseline.predict(features[test], velocity[test]) - drag[test]) ** 2)
    )
    minimum_dissipation = float(
        np.min(structured.dissipation(features[test], velocity[test]))
    )
    assert structured_error < 2.0e-3
    assert baseline_error < 1.0e-2
    assert minimum_dissipation >= -1.0e-12
    print(
        "self-test passed:",
        f"structured_mse={structured_error:.3e},",
        f"direct_mse={baseline_error:.3e},",
        f"min_dissipation={minimum_dissipation:.3e}",
    )


if __name__ == "__main__":
    _self_test()
