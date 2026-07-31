#!/usr/bin/env python3
"""Verify checksums of the small numerical artefacts in the public release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "examples" / "assets"


def main() -> None:
    manifest = json.loads((ASSETS / "MANIFEST.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    for relative, expected in manifest["sha256"].items():
        path = (ASSETS / relative).resolve()
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            failures.append(f"{relative}: {observed} != {expected}")
    if failures:
        raise SystemExit("checksum verification failed:\n" + "\n".join(failures))
    print(f"PASS: {len(manifest['sha256'])} release artefacts verified")


if __name__ == "__main__":
    main()
