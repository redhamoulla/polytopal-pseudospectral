# Released closure checkpoints

The two NumPy checkpoints were trained with seed `20260730` on the same local
porous-drag teacher and feature convention.

- `structured_spd.npz` predicts a lower-triangular factor and applies
  `drag = L L.T velocity`; non-negative dissipated power is guaranteed by
  construction.
- `direct_mlp.npz` predicts drag directly and is retained only as the
  unconstrained scientific control.

The canonical evaluation is an independently implemented staggered
finite-volume solver on an `80 x 40` grid. Run:

```bash
python examples/reproduce_porous_duct.py --check
```

The checkpoints and reference metrics are research artefacts, not universal
surrogates. They are valid for the geometry, nondimensionalisation, feature
map and excitation recorded in `reference_metrics.json`.

`h48p5_observables.npz` is separate from these learned checkpoints. It stores
the 1,025 observable samples of the corrected, post-confirmatory H48P5 rollout
with the analytic passive drag law. It is included to document the energy
ledger and signal provenance; it is not a training set.
