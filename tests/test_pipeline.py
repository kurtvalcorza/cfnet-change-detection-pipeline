"""Offline tests for the snapshot manifest, staging, the static pickle audit, the pair contract, the metrics and
the artifact-manifest rejections. No model library is imported."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

from cfnet_change_detection_pipeline import (
    CHANGE_THRESHOLD,
    CKPT_ALLOWED_GLOBALS,
    CLASS_NAMES,
    INPUT_SCHEMA,
    MODEL_ID,
    MODEL_REVISION,
    NUM_CLASSES,
    SAMPLE_SIZE,
    CFNetChangePipeline,
    audit_pickle,
    change_metrics,
    dataset_digest,
    read_image,
    read_mask,
    stage_missing_files,
    unchanged_baseline,
    validate_dataset,
    validate_inputs,
    verify_converted,
    verify_snapshot,
)
from cfnet_change_detection_pipeline import pipeline as pl
from conftest import synthetic_pair, synthetic_records

ROOT = Path(__file__).resolve().parents[1]


def _write_snapshot(root: Path, model_id: str, revision: str, files: dict[str, bytes], *, pin_source: bool = True) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
        entries.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    if pin_source and pl.SOURCE_CKPT_NAME not in files:
        entries.append({"path": pl.SOURCE_CKPT_NAME, "bytes": pl.SOURCE_CKPT_BYTES, "sha256": pl.SOURCE_CKPT_SHA256})
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": pl.MODEL_KEY,
        "modelId": model_id,
        "revision": revision,
        "files": entries,
        "totalBytes": sum(e["bytes"] for e in entries),
    }
    (root / pl.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


# --- identity and the committed manifest ------------------------------------------------------------------


def test_identity_is_immutable_and_the_manifest_agrees():
    assert len(MODEL_REVISION) == 40 and len(pl.UPSTREAM_CODE_COMMIT) == 40
    manifest = json.loads((ROOT / "weights" / pl.MODEL_KEY / pl.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert (manifest["modelId"], manifest["revision"], manifest["modelKey"]) == (MODEL_ID, MODEL_REVISION, pl.MODEL_KEY)
    assert manifest["totalBytes"] == sum(e["bytes"] for e in manifest["files"])
    source = [e for e in manifest["files"] if e["path"] == pl.SOURCE_CKPT_NAME]
    assert source and (source[0]["bytes"], source[0]["sha256"]) == (pl.SOURCE_CKPT_BYTES, pl.SOURCE_CKPT_SHA256)
    assert len(pl.CONVERTED_SHA256) == 64 and pl.CONVERTED_BYTES > 0 and len(pl.PICKLE_AUDIT_SHA256) == 64
    assert pl.PARAMETER_COUNT == 3_838_563 and pl.STATE_TENSORS == 776 and pl.ENCODER_TENSORS == 428
    assert len(CLASS_NAMES) == NUM_CLASSES == 2 and CHANGE_THRESHOLD == 0.5


# --- snapshot verification and staging --------------------------------------------------------------------


def test_verify_snapshot_refuses_mismatches(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"README.md": b"# x", pl.SOURCE_CKPT_NAME: b"ckpt"}, pin_source=False)
    with pytest.raises(ValueError, match="disagrees with the package constant"):
        verify_snapshot(root)
    (root / pl.SOURCE_CKPT_NAME).write_bytes(b"ckpt-longer")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(root)
    (root / pl.SOURCE_CKPT_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        verify_snapshot(root)
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"README.md": b"# x"}, pin_source=False)
    with pytest.raises(ValueError, match="does not list"):
        verify_snapshot(root)
    _write_snapshot(root, "someone/else", MODEL_REVISION, {"README.md": b"# x"})
    with pytest.raises(ValueError, match="modelId"):
        verify_snapshot(root)
    _write_snapshot(root, MODEL_ID, "0" * 40, {"README.md": b"# x"})
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(root)
    with pytest.raises(FileNotFoundError, match="manifest"):
        verify_snapshot(tmp_path / "nowhere")


def test_verify_converted_checks_the_pinned_digest(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="converted file missing"):
        verify_converted(tmp_path)
    (tmp_path / pl.CONVERTED_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="size|sha256"):
        verify_converted(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries(tmp_path, forbid_model_imports):
    root = tmp_path / "snap"
    _write_snapshot(root, MODEL_ID, MODEL_REVISION, {"README.md": b"# x"})
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(root)
    calls = []

    def downloader(rel, dst):
        calls.append(rel)
        (dst / rel).write_bytes(b"fetched")

    assert stage_missing_files(root, allow_download=True, downloader=downloader) == [pl.SOURCE_CKPT_NAME]
    assert calls == [pl.SOURCE_CKPT_NAME]
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == []
    _write_snapshot(root, "someone/else", MODEL_REVISION, {"README.md": b"# x"})
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(root, allow_download=True, downloader=downloader)


# --- static pickle audit ----------------------------------------------------------------------------------


class _Evil:
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


def _torch_like_archive(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("archive/data.pkl", payload)
        archive.writestr("archive/version", b"3\n")
    return buffer.getvalue()


def test_audit_pickle_lists_globals_and_refuses_code(tmp_path, forbid_model_imports):
    benign = tmp_path / "benign.pth"
    benign.write_bytes(_torch_like_archive(pickle.dumps({"a": 1, "b": [2, 3]})))
    report = audit_pickle(benign)
    assert report["torch_archive"] and report["pickles"] == 1 and report["globals"] == [] and report["violations"] == []
    evil = tmp_path / "evil.pth"
    evil.write_bytes(_torch_like_archive(pickle.dumps({"state_dict": _Evil()})))
    with pytest.raises(ValueError, match="pickle audit failed.*os.system|posix.system|nt.system"):
        audit_pickle(evil)
    plain = tmp_path / "plain.pickle"
    plain.write_bytes(pickle.dumps(complex(1, 2)))
    with pytest.raises(ValueError, match="pickle audit failed"):
        audit_pickle(plain)
    stream = b"".join(
        [
            pickle.PROTO + b"\x02",
            pickle.GLOBAL + b"collections\nOrderedDict\n",
            pickle.GLOBAL + b"torch._utils\n_rebuild_tensor_v2\n",
            pickle.GLOBAL + b"torch\nFloatStorage\n",
            pickle.STOP,
        ]
    )
    state = tmp_path / "state.pth"
    state.write_bytes(_torch_like_archive(stream))
    report = audit_pickle(state)
    assert set(report["globals"]) < CKPT_ALLOWED_GLOBALS and report["violations"] == []
    assert len(CKPT_ALLOWED_GLOBALS) == 4
    with pytest.raises(FileNotFoundError):
        audit_pickle(tmp_path / "missing.pth")


# --- pair contract ----------------------------------------------------------------------------------------


def test_validate_dataset_reports_balance_and_digest(forbid_model_imports):
    records = synthetic_records(6)
    report = validate_dataset(records)
    assert report["n_records"] == 6 and report["n_labelled"] == 6 and report["sizes"] == [[256, 256]]
    assert 0.05 < report["change_fraction"] < 0.2 and report["ignored_pixels"] == 6 * 64
    assert report["digest"] == dataset_digest(records) and len(report["digest"]) == 64
    assert INPUT_SCHEMA["classes"] == {"0": "unchanged", "1": "changed"} and SAMPLE_SIZE == 256
    assert "any two same-sized RGB images are compared" in INPUT_SCHEMA["validation"]


def test_validate_dataset_refusals_name_the_rule(forbid_model_imports):
    records = synthetic_records(6)
    with pytest.raises(ValueError, match="records must be a list"):
        validate_dataset({"id": "x"})
    with pytest.raises(ValueError, match="4..2000 are required"):
        validate_dataset(records[:3])
    with pytest.raises(ValueError, match="missing 'after'"):
        validate_dataset([{"id": "a", "before": records[0]["before"], "label": records[0]["label"]}, *records[1:]])
    with pytest.raises(ValueError, match="must have shape \\(H, W, 3\\)"):
        validate_dataset([{**records[0], "before": records[0]["before"][..., :2]}, *records[1:]])
    with pytest.raises(ValueError, match="same shape"):
        validate_dataset([{**records[0], "after": records[0]["after"][:224, :224]}, *records[1:]])
    with pytest.raises(ValueError, match="multiples of 32"):
        validate_dataset(
            [
                {
                    **records[0],
                    "before": records[0]["before"][:200, :200],
                    "after": records[0]["after"][:200, :200],
                    "label": records[0]["label"][:200, :200],
                },
                *records[1:],
            ]
        )
    with pytest.raises(ValueError, match="sides must be in"):
        validate_dataset(
            [
                {
                    **records[0],
                    "before": records[0]["before"][:32, :32],
                    "after": records[0]["after"][:32, :32],
                    "label": records[0]["label"][:32, :32],
                },
                *records[1:],
            ]
        )
    with pytest.raises(ValueError, match="uint8 RGB or numeric"):
        validate_dataset([{**records[0], "before": records[0]["before"].astype(np.float32) * 2.0}, *records[1:]])
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset([records[0], *records])
    with pytest.raises(ValueError, match="has no label"):
        validate_dataset([{"id": "u", "before": records[0]["before"], "after": records[0]["after"]}, *records[1:]])
    with pytest.raises(ValueError, match="label values \\[2\\] outside"):
        validate_dataset([{**records[0], "label": records[0]["label"] + 1}, *records[1:]])
    with pytest.raises(ValueError, match="label must have shape"):
        validate_dataset([{**records[0], "label": records[0]["label"][:100]}, *records[1:]])
    with pytest.raises(ValueError, match="no changed pixel"):
        validate_dataset([{**r, "label": np.zeros_like(r["label"])} for r in records])
    assert (
        validate_dataset([{"id": r["id"], "before": r["before"], "after": r["after"]} for r in records], require_labels=False)[
            "n_labelled"
        ]
        == 0
    )


def test_float_images_in_range_are_accepted(forbid_model_imports):
    before, after, label = synthetic_pair()
    report = validate_inputs({"id": "s", "before": before.astype(np.float32), "after": after, "label": label})
    assert report["shape"] == (256, 256, 3) and report["ignored_pixels"] == 64 and 0.05 < report["change_fraction"] < 0.2


def test_png_round_trip(tmp_path, forbid_model_imports):
    from PIL import Image

    before, after, label = synthetic_pair()
    Image.fromarray(before).save(tmp_path / "a.png")
    Image.fromarray(after).save(tmp_path / "b.jpg", quality=95)
    Image.fromarray(np.where(label == 1, 255, 0).astype(np.uint8)).save(tmp_path / "label.png")
    Image.fromarray(before).convert("L").save(tmp_path / "grey.png")
    assert np.array_equal(read_image(tmp_path / "a.png"), before)
    assert read_image(tmp_path / "b.jpg").shape == (256, 256, 3) and read_image(tmp_path / "grey.png").shape == (256, 256, 3)
    mask = read_mask(tmp_path / "label.png")
    assert mask.dtype == np.int64 and np.array_equal(mask, np.where(label == -1, 0, label))
    report = validate_inputs(
        {"id": "p", "before": str(tmp_path / "a.png"), "after": str(tmp_path / "b.jpg"), "label": str(tmp_path / "label.png")}
    )
    assert report["has_label"] and report["ignored_pixels"] == 0
    with pytest.raises(ValueError, match="before file not found"):
        validate_inputs({"id": "p", "before": str(tmp_path / "none.png"), "after": str(tmp_path / "a.png")})


# --- metrics ----------------------------------------------------------------------------------------------


def test_change_metrics_and_baseline(forbid_model_imports):
    _, _, label = synthetic_pair()
    perfect = change_metrics([np.where(label < 0, 0, label)], [label])
    assert perfect["f1"] == 1.0 and perfect["iou"] == 1.0 and perfect["accuracy"] == 1.0 and perfect["pixels"] == 256 * 256 - 64
    baseline = unchanged_baseline([label])
    assert baseline["f1"] == 0.0 and baseline["iou"] == 0.0 and baseline["recall"] == 0.0 and baseline["precision"] is None
    assert baseline["accuracy"] == round(1 - baseline["change_fraction_label"], 4) and baseline["note"].startswith(
        "predicts 'unchanged'"
    )
    flipped = change_metrics([np.where(label < 0, 0, 1 - label)], [label])
    assert flipped["accuracy"] == 0.0 and flipped["f1"] == 0.0 and flipped["confusion"]["tp"] == 0
    with pytest.raises(ValueError, match="argument 2 is shorter|shape"):
        change_metrics([label], [label[:10]])


# --- artifact manifest (static checks, no weights) -------------------------------------------------------


def _good_manifest(root: Path) -> dict:
    (root / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    return {
        "format": pl.ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": pl.MODEL_KEY, "converted_sha256": pl.CONVERTED_SHA256},
        "adapter": {"trainable": "change_decoder"},
        "tensors": ["change_decoder.x"],
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}],
    }


def test_artifact_manifest_static_checks(tmp_path, forbid_model_imports):
    good = _good_manifest(tmp_path)
    path, mode = CFNetChangePipeline.check_artifact_manifest(tmp_path, good)
    assert path == (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).resolve() and mode == "change_decoder"
    cases = {
        "format": ({**good, "format": "other"}, "artifact format"),
        "version": ({**good, "format_version": "2.0"}, "format_version"),
        "base": ({**good, "base_model": {**good["base_model"], "revision": "0" * 40}}, "different base model"),
        "digest": ({**good, "base_model": {**good["base_model"], "converted_sha256": "0" * 64}}, "converted-base digest"),
        "two files": ({**good, "files": good["files"] * 2}, "exactly one weights file"),
        "other name": ({**good, "files": [{**good["files"][0], "path": "weights.safetensors"}]}, "must be named"),
        "mode": ({**good, "adapter": {"trainable": "everything"}}, "adapter.trainable"),
        "tensors": ({**good, "tensors": "all"}, "list its tensors"),
    }
    for name, (manifest, message) in cases.items():
        with pytest.raises(ValueError, match=message):
            CFNetChangePipeline.check_artifact_manifest(tmp_path, manifest)
        del name
