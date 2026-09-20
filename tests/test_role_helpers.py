"""Offline tests for the public validation-stage helpers and the package surface."""

from __future__ import annotations

import cfnet_change_detection_pipeline as pkg
from cfnet_change_detection_pipeline import CHANGE_THRESHOLD, IGNORE_INDEX, INPUT_SCHEMA, SIDE_MULTIPLE, validate_inputs
from conftest import synthetic_records


def test_input_schema_names_the_contract():
    assert INPUT_SCHEMA["ignore_index"] == IGNORE_INDEX == -1 and SIDE_MULTIPLE == 32
    assert INPUT_SCHEMA["decision_rule"] == f"changed where the tanh change map exceeds {CHANGE_THRESHOLD} (upstream)"
    assert "any two same-sized RGB images are compared" in INPUT_SCHEMA["validation"]
    assert INPUT_SCHEMA["classes"] == {"0": "unchanged", "1": "changed"}


def test_validate_inputs_reports_the_record():
    report = validate_inputs(synthetic_records(1)[0])
    assert report["id"] == "pair-000" and report["shape"] == (256, 256, 3) and report["has_label"]
    assert report["ignored_pixels"] == 64 and 0.05 < report["change_fraction"] < 0.2


def test_public_surface_is_exported():
    for name in pkg.__all__:
        assert hasattr(pkg, name), name
    assert "CFNetChangePipeline" in pkg.__all__ and "audit_pickle" in pkg.__all__ and "unchanged_baseline" in pkg.__all__
