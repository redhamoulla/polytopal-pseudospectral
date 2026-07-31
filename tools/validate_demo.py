#!/usr/bin/env python3
"""Validate the self-contained demo and render its actual pressure preview."""

from __future__ import annotations

import argparse
import base64
import json
import re
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HTML = ROOT / "docs" / "index.html"
DEFAULT_PREVIEW = ROOT / "docs" / "preview.svg"


def load_data(path: Path) -> dict[str, object]:
    source = path.read_text(encoding="utf-8")
    match = re.search(r"const DATA = (\{.*?\});\n  const root", source, re.S)
    if match is None:
        raise RuntimeError("embedded demo dataset not found")
    return json.loads(match.group(1))


def validate(data: dict[str, object]) -> dict[str, float | int]:
    meta = data["meta"]
    pressure = data["pressure"]
    series = data["series"]
    integrals = data["integrals"]
    assert isinstance(meta, dict)
    assert isinstance(pressure, dict)
    assert isinstance(series, dict)
    assert isinstance(integrals, dict)

    frames = int(meta["frames"])
    triangles = int(meta["triangle_count"])
    encoded = base64.b64decode(str(pressure["data"]), validate=True)
    if len(encoded) != 2 * frames * triangles:
        raise AssertionError("pressure payload shape does not match its metadata")
    if len(data["polygons"]) != int(meta["cell_count"]):
        raise AssertionError("polygon count does not match its metadata")
    if len(data["triangles"]) != triangles:
        raise AssertionError("triangle count does not match its metadata")
    if len(series["time"]) != frames:
        raise AssertionError("time and pressure frame counts differ")

    incident = float(integrals["incident_energy"])
    reflected = float(integrals["reflected_energy"])
    transmitted = float(integrals["finite_window_transmitted_energy"])
    stored = float(integrals["final_stored_energy"])
    dissipated = float(series["cumulative_volumetric_dissipation"][-1])
    ledger = (reflected + transmitted + stored + dissipated) / incident
    if abs(ledger - 1.0) > 2.0e-5:
        raise AssertionError(f"terminal energy ledger is {ledger:.12f}, expected 1")

    return {
        "frames": frames,
        "cells": int(meta["cell_count"]),
        "triangles": triangles,
        "wall_rank": int(meta["wall_rank"]),
        "wall_constraints": int(meta["wall_constraint_count"]),
        "terminal_energy_ledger_percent": 100.0 * ledger,
    }


def _mix(background: tuple[int, int, int], colour: tuple[int, int, int], alpha: float) -> str:
    channels = [round((1.0 - alpha) * a + alpha * b) for a, b in zip(background, colour)]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def render_preview(data: dict[str, object], output: Path, frame: int = 35) -> None:
    meta = data["meta"]
    pressure = data["pressure"]
    assert isinstance(meta, dict) and isinstance(pressure, dict)
    triangle_count = int(meta["triangle_count"])
    raw = base64.b64decode(str(pressure["data"]), validate=True)
    values = struct.unpack(f"<{len(raw) // 2}h", raw)
    if not 0 <= frame < int(meta["frames"]):
        raise ValueError("preview frame is outside the recorded rollout")

    width, height = 1000, 500
    background = (251, 250, 246)
    positive = (8, 127, 140)
    negative = (223, 107, 79)

    def point(x: float, y: float) -> str:
        return f"{500.0 * x:.2f},{height * (1.0 - y):.2f}"

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 500" role="img" aria-labelledby="title desc">',
        '<title id="title">Computed pressure field on a 48-cell polygonal mesh</title>',
        '<desc id="desc">A recorded H48P5 pressure snapshot with the porous insert highlighted.</desc>',
        '<rect width="1000" height="500" fill="#fbfaf6"/>',
    ]
    start = frame * triangle_count
    for index, triangle in enumerate(data["triangles"]):
        value = values[start + index] / 32767.0
        magnitude = min(1.0, abs(value))
        alpha = 0.0 if magnitude < 0.005 else 0.08 + 0.92 * magnitude**0.68
        fill = _mix(background, positive if value >= 0.0 else negative, alpha)
        points = " ".join(
            point(float(triangle[offset]), float(triangle[offset + 1]))
            for offset in (0, 2, 4)
        )
        elements.append(f'<polygon points="{points}" fill="{fill}"/>')

    for polygon in data["polygons"]:
        points = " ".join(point(float(x), float(y)) for x, y in polygon)
        elements.append(
            f'<polygon points="{points}" fill="none" stroke="#9da9aa" stroke-width="0.8"/>'
        )
    x0, x1, y0, y1 = (float(value) for value in data["porous_insert"])
    elements.extend(
        [
            f'<rect x="{500*x0:.2f}" y="{height*(1-y1):.2f}" width="{500*(x1-x0):.2f}" height="{height*(y1-y0):.2f}" fill="#d99a36" fill-opacity="0.10" stroke="#6a7477" stroke-width="1.5"/>',
            '<rect x="18" y="18" width="288" height="39" fill="#fbfaf6" fill-opacity="0.88"/>',
            '<text x="32" y="44" fill="#17252b" font-family="system-ui,sans-serif" font-size="18" font-weight="650">H48P5 · computed pressure field</text>',
            '<rect x="694" y="443" width="288" height="39" fill="#fbfaf6" fill-opacity="0.88"/>',
            '<text x="709" y="469" fill="#647176" font-family="system-ui,sans-serif" font-size="16">48 polygons · row rank 70/70</text>',
            "</svg>",
        ]
    )
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--check-only", action="store_true")
    arguments = parser.parse_args()
    data = load_data(arguments.html)
    summary = validate(data)
    if not arguments.check_only:
        render_preview(data, arguments.preview)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
