from pathlib import Path

from polytopal_ph.validation import (
    assert_reference_metrics,
    evaluate_pretrained_closures,
)


ASSETS = Path(__file__).resolve().parents[1] / "examples" / "assets"


def test_canonical_fv_closure_comparison() -> None:
    result = evaluate_pretrained_closures(ASSETS)
    assert_reference_metrics(result, ASSETS / "reference_metrics.json")
