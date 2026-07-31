# Contributing

This repository is intentionally smaller than the research workspace from
which it was released. Contributions should preserve that boundary.

## Numerical changes

- Add or update a test for every change to numerical behaviour.
- Record random seeds, geometry parameters and tolerances explicitly.
- Motivate tolerances by an invariant or a discretization error, not by one
  machine's last digits.
- Use equilibrated rank-revealing factorizations; do not form normal-equation
  inverses.
- Do not commit caches, complete campaign directories, logs, notebooks or
  large generated arrays.
- A benchmark must record its environment and protocol and must not be
  presented as an accuracy ranking unless errors are matched.

## Local checks

```bash
python -m pip install -e ".[test]"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python examples/reproduce_porous_duct.py --check
python tools/validate_demo.py --check-only
python tools/verify_manifest.py
```

Open a focused pull request and describe the mathematical or physical
invariant affected by the change. The preferred merge strategy is squash.
