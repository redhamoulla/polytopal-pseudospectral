#!/usr/bin/env python3
"""Reproduce the canonical passive-closure comparison in a few seconds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from polytopal_ph.validation import (
    assert_reference_metrics,
    evaluate_pretrained_closures,
)


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=80)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--final-time", type=float, default=2.4)
    parser.add_argument("--cfl", type=float, default=0.45)
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the canonical 80x40 values against the released reference",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    result = evaluate_pretrained_closures(
        HERE / "assets",
        nx=arguments.nx,
        ny=arguments.ny,
        final_time=arguments.final_time,
        cfl=arguments.cfl,
    )
    if arguments.check:
        assert_reference_metrics(result, HERE / "assets" / "reference_metrics.json")
        result["reference_check"] = "PASS"

    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
