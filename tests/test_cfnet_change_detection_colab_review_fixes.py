"""Regression tests for the 2026-10-02 notebook review of cfnet_change_detection_colab (CFN-M1..M5, CFN-m1..m5).

Most run without torch (CI's install budget). The adaptation tests (CFN-M2 / CFN-M3) need torch and use a stub model
with no weights; they are skipped where torch is absent.
"""
# ruff: noqa: E501

from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

with contextlib.suppress(ImportError):  # Windows DLL load-order trap: import torch before any NumPy linear algebra
    import torch  # noqa: F401

from cfnet_change_detection_pipeline import (  # noqa: E402
    BYOD_TEST_FRACTION,
    BYOD_VAL_FRACTION,
    INPUT_SCHEMA,
    SAMPLE_LABEL_SOURCE,
    dataset_manifest,
    load_byod_dataset,
    minimum_pairs,
    split_dataset,
    validate_dataset,
    validate_inputs,
    write_dataset_csv,
    write_sample_pair,
)
from cfnet_change_detection_pipeline import samples as sm  # noqa: E402
from conftest import synthetic_records  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"_cfn_{name}", ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEMPLATE = _load_tool("notebook_template").TEMPLATE
NOTEBOOK = json.loads((ROOT / "tutorials" / TEMPLATE["notebook_name"]).read_text(encoding="utf-8"))
CELLS = NOTEBOOK["cells"]
MARKDOWN = "\n".join(c["source"] for c in CELLS if c["cell_type"] == "markdown")
CODE = [c["source"] for c in CELLS if c["cell_type"] == "code"]
KERNEL = [s for s in CODE if "# dimer: kernel cell" in s]
EMBEDDED = [c["source"] for c in CELLS if c["cell_type"] == "code" and c["metadata"].get("dimer", {}).get("embedded_module")]
LEARNER = [s for s in CODE if s not in KERNEL and s not in EMBEDDED]


def _section_code(number: int) -> str:
    index = next(i for i, c in enumerate(CELLS) if c["cell_type"] == "markdown" and f"## {number}. " in c["source"])
    return next(c["source"] for c in CELLS[index + 1 :] if c["cell_type"] == "code")


# --- CFN-M1: uv isolated environment, no in-kernel install, Candidate records ------------------------------------


def test_isolated_runtime_replaces_the_in_kernel_install():
    build = _load_tool("build_notebook")
    assert NOTEBOOK["metadata"]["dimer"]["notebook_spec"] == "2.2"
    assert NOTEBOOK["metadata"]["dimer"]["generated_from"]["generator"] == "build_notebook.py/2.1"
    assert len(KERNEL) == 2
    install = next(k for k in KERNEL if "LOCK_TEXT = r" in k)
    for needed in ('"--managed-python"', '"--require-hashes"', '"--only-binary"', '":all:"', "UV_SHA256", "LOCK_SHA256", 'platform.machine() != "x86_64"'):
        assert needed in install
    assert "MANAGED_PYTHON = '3.12.12'" in install
    lock = (ROOT / TEMPLATE["lock"]).read_text(encoding="utf-8")
    build.check_lock(build._pins(ROOT), lock)  # every pin locked at its version, every entry hashed
    assert "_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)" in "\n".join(KERNEL)
    assert "Restart the runtime" not in MARKDOWN and "restart instruction" not in MARKDOWN
    assert "never needs a runtime restart" in MARKDOWN and "**Linux x86_64 only**" in MARKDOWN


def test_release_records_are_candidate_and_do_not_call_the_restart_run_a_pass():
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    assert "Current status: **Candidate**" in status
    verification = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "restart after the install is expected" not in verification
    assert "not a one-pass Run all" in verification and "not promotion evidence" in verification
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.split("## Release status", 1)[1].lstrip().startswith("**Candidate")


# --- CFN-M2 / CFN-M3: deterministic selection, reset before use, no bare asserts ---------------------------------


def test_sections_5_and_6_reset_to_the_pretrained_model_first():
    for number in (5, 6):
        code = _section_code(number)
        lines = [line for line in code.splitlines() if "pipe." in line and not line.lstrip().startswith("#")]
        assert lines[0].startswith("pipe.reset_to_pretrained()"), number
    assert "draw_seed=s" in _section_code(6) and "adapt_result['selection']" in _section_code(6)


def test_learner_cells_raise_with_a_cause_instead_of_asserting():
    learner = "\n".join(LEARNER)
    assert "\nassert " not in "\n" + learner
    assert "raise RuntimeError(f'Reload parity failed:" in learner
    assert "_result.json').unlink(missing_ok=True)" in _section_code(8)
    assert "run_history = globals().get('run_history', [])" in _section_code(7)


def test_selection_prose_does_not_read_noise_as_improvement():
    assert "the sign that a small learning rate and validation selection are doing their job" not in MARKDOWN
    assert "is not, on its own, evidence that anything improved" in MARKDOWN
    assert "while the frozen model keeps the kept epoch" not in MARKDOWN


# --- CFN-M5: the guided layer --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marker, least",
    [
        ("**Who this is for.**", 1),
        ("**Input → Model → Output.**", 1),
        ("**How to use this notebook.**", 1),
        ("**Roadmap:**", 1),
        ("**Predict before running:**", 5),
        ("**What to notice:**", 5),
        ("<summary>Check your reasoning</summary>", 6),
        ("## 9. Your turn — change one thing: the learning rate", 1),
        ("## Troubleshooting", 1),
        ("## Glossary", 1),
        ("## Conclusion (your notes)", 1),
        ("> **Infrastructure.**", 3),
    ],
)
def test_guided_layer_markers(marker, least):
    assert MARKDOWN.count(marker) >= least


def test_carried_modules_are_collapsed():
    embedded = [c for c in CELLS if c["cell_type"] == "code" and c["metadata"].get("dimer", {}).get("embedded_module")]
    assert len(embedded) == 4 and all(c["metadata"].get("jupyter", {}).get("source_hidden") for c in embedded)


# --- CFN-m1 / CFN-m2 / CFN-m3 ---------------------------------------------------------------------------------------


def test_no_doubled_braces_and_timings_name_their_environment():
    assert "{{" not in MARKDOWN and "}}" not in MARKDOWN
    assert "`{id, before, after, label}`" in MARKDOWN
    for stale in ("a CPU a few minutes", "about a minute on a CPU", "in the build record 0.9352 and 0.8782, precision 0.9424"):
        assert stale not in MARKDOWN


def test_every_external_access_statement_names_the_tarball():
    header = CELLS[0]["source"]
    prereq = next(c["source"] for c in CELLS if c["cell_type"] == "markdown" and c["source"].startswith("## Prerequisites"))
    access = next(line for line in prereq.splitlines() if line.startswith("- **External access:**"))
    for text in (header.split("**This notebook is standalone.**", 1)[1].split("\n\n", 1)[0], access):
        assert "LEVIR-CD-processed.tar.gz" in text and "3.8 GB" in text
    assert "the Hugging Face Hub only" not in MARKDOWN


# --- CFN-M4 / CFN-m5: BYOD ------------------------------------------------------------------------------------------


def _write_byod(folder: Path, records, *, sub: str = "") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    rows = ["id,before,after,label"]
    for record in records:
        rid = record["id"]
        (folder / sub).mkdir(parents=True, exist_ok=True)
        write_sample_pair(record, folder / sub / f"{rid}_A.png", folder / sub / f"{rid}_B.png", folder / sub / f"{rid}_label.png")
        rows.append(f"{rid},{sub}{rid}_A.png,{sub}{rid}_B.png,{sub}{rid}_label.png")
    (folder / "pairs.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return folder


def _zip(folder: Path, path: Path, *, prefix: str = "", skip: str | None = None) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for f in sorted(folder.rglob("*")):
            rel = f.relative_to(folder).as_posix()
            if f.is_file() and rel != skip:
                zf.write(f, prefix + rel)
    return path


def test_minimum_pairs_matches_the_split(forbid_model_imports):
    assert (BYOD_TEST_FRACTION, BYOD_VAL_FRACTION) == (0.25, 0.2)
    assert minimum_pairs() == 7
    with pytest.raises(ValueError, match="needs at least 7 distinct pairs"):
        split_dataset(synthetic_records(6))
    splits = split_dataset(synthetic_records(7))
    assert [len(splits[k]) for k in ("test", "validation", "train")] == [2, 1, 4]
    with pytest.raises(ValueError, match="after removing 1 duplicate"):
        split_dataset([*synthetic_records(6), {**synthetic_records(1)[0], "id": "dup"}])


def test_byod_records_carry_a_file_safe_unique_source_id(tmp_path, forbid_model_imports):
    records = synthetic_records(3)
    records[1]["id"] = "a b/c"
    records[2]["id"] = "a b?c"
    folder = tmp_path / "byod"
    folder.mkdir()
    rows = ["id,before,after,label"]
    for i, record in enumerate(records):
        write_sample_pair(record, folder / f"p{i}_A.png", folder / f"p{i}_B.png", folder / f"p{i}_label.png")
        rows.append(f"\"{record['id']}\",p{i}_A.png,p{i}_B.png,p{i}_label.png")
    (folder / "pairs.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    loaded = load_byod_dataset(folder)
    ids = [r["source_id"] for r in loaded]
    assert ids[0] == "pair-000" and ids[1] == "a_b_c" and ids[2] == "a_b_c-0002"
    assert all(set(i) <= set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") for i in ids)


def test_byod_zip_paths_are_relative_to_pairs_csv(tmp_path, forbid_model_imports):
    records = synthetic_records(3)
    folder = _write_byod(tmp_path / "byod", records, sub="imgs/")
    assert len(load_byod_dataset(_zip(folder, tmp_path / "nested.zip", prefix="mydata/"))) == 3
    assert len(load_byod_dataset(folder)) == 3
    missing = _zip(folder, tmp_path / "missing.zip", skip="imgs/pair-001_B.png")
    with pytest.raises(ValueError, match=r"row 'pair-001': after file 'imgs/pair-001_B.png' matches no file"):
        load_byod_dataset(missing)
    (folder / "imgs" / "pair-002_A.png").unlink()
    with pytest.raises(ValueError, match="not in the BYOD directory"):
        load_byod_dataset(folder)
    (folder / "pairs.csv").write_text("id,before,after,label\nx,../outside.png,b.png,c.png\n", encoding="utf-8")
    (tmp_path / "outside.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="not in the BYOD directory"):
        load_byod_dataset(folder)


def test_byod_ambiguous_bare_name_is_refused(tmp_path, forbid_model_imports):
    records = synthetic_records(1)
    folder = _write_byod(tmp_path / "byod", records)
    path = tmp_path / "ambiguous.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for f in folder.iterdir():
            zf.write(f, f"a/{f.name}" if f.name != "pairs.csv" else "pairs.csv")
            if f.name.endswith("_A.png"):
                zf.write(f, f"b/{f.name}")
    with pytest.raises(ValueError, match="matches several files"):
        load_byod_dataset(path)


def test_byod_zip_ceilings_refuse_before_decoding(tmp_path, monkeypatch, forbid_model_imports):
    folder = _write_byod(tmp_path / "byod", synthetic_records(2))
    archive = _zip(folder, tmp_path / "byod.zip")
    decoded = []
    monkeypatch.setattr(sm, "read_image", lambda path: decoded.append(path))
    monkeypatch.setattr(sm, "BYOD_MAX_MEMBERS", 3)
    with pytest.raises(ValueError, match="7 files; at most 3 are accepted"):
        load_byod_dataset(archive)
    monkeypatch.setattr(sm, "BYOD_MAX_MEMBERS", 6_001)
    monkeypatch.setattr(sm, "BYOD_MAX_EXPANDED_BYTES", 1024)
    with pytest.raises(ValueError, match="at most 1,024 bytes"):
        load_byod_dataset(archive)
    assert decoded == []


def _section4_namespace(tmp_path, monkeypatch, *, byod_path: str = "", upload=None):
    monkeypatch.chdir(tmp_path)
    if upload is not None:
        google = types.ModuleType("google")
        colab = types.ModuleType("google.colab")
        files = types.ModuleType("google.colab.files")
        files.upload = upload
        colab.files = files
        google.colab = colab
        monkeypatch.setitem(sys.modules, "google", google)
        monkeypatch.setitem(sys.modules, "google.colab", colab)
        monkeypatch.setitem(sys.modules, "google.colab.files", files)
    source = _section_code(4).replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param")
    source = source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {byod_path!r}  # @param")
    namespace = {
        "load_byod_dataset": load_byod_dataset,
        "split_dataset": split_dataset,
        "fetch_sample_dataset": None,
        "SAMPLE_LABEL_SOURCE": SAMPLE_LABEL_SOURCE,
        "dataset_manifest": dataset_manifest,
        "validate_inputs": validate_inputs,
        "write_sample_pair": write_sample_pair,
        "write_dataset_csv": write_dataset_csv,
        "INPUT_SCHEMA": INPUT_SCHEMA,
        "minimum_pairs": minimum_pairs,
        "BYOD_TEST_FRACTION": BYOD_TEST_FRACTION,
        "BYOD_VAL_FRACTION": BYOD_VAL_FRACTION,
        "validate_dataset": validate_dataset,
    }
    return source, namespace


def test_section4_byod_minimum_set_passes_and_probes_name_their_condition(tmp_path, monkeypatch, capsys, forbid_model_imports):
    folder = _write_byod(tmp_path / "data", synthetic_records(7), sub="imgs/")
    archive = _zip(folder, tmp_path / "byod.zip", prefix="upload/")
    source, namespace = _section4_namespace(tmp_path, monkeypatch, byod_path=str(archive))
    exec(compile(source, "<section 4>", "exec"), namespace)
    out = capsys.readouterr().out
    assert "'splits': {'train': 4, 'validation': 1, 'test': 2}" in out and "'byod_minimum_pairs': 7" in out
    assert "before and after must have the same shape" in out
    assert "sides must be multiples of 32" in out
    assert "label values [2] outside" in out
    assert "records; 4..2000 are required" not in out and "'verdict': 'accepted'" not in out
    assert all("source_id" in r for r in namespace["test_records"])


def test_section4_upload_guard_and_missing_path(tmp_path, monkeypatch, forbid_model_imports):
    source, namespace = _section4_namespace(tmp_path, monkeypatch, upload=lambda: {})
    with pytest.raises(ValueError, match=r"Upload exactly one \.zip file \(received 0\)"):
        exec(compile(source, "<section 4>", "exec"), namespace)
    source, namespace = _section4_namespace(tmp_path, monkeypatch, byod_path=str(tmp_path / "nope.zip"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        exec(compile(source, "<section 4>", "exec"), namespace)
    source, namespace = _section4_namespace(tmp_path, monkeypatch, byod_path=str(_zip(_write_byod(tmp_path / "six", synthetic_records(6)), tmp_path / "six.zip")))
    with pytest.raises(ValueError, match="needs at least 7 distinct pairs"):
        exec(compile(source, "<section 4>", "exec"), namespace)


# --- CFN-M2 / CFN-M3 on a stub model (torch required) ---------------------------------------------------------------


@pytest.fixture
def stub_pipeline(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    from cfnet_change_detection_pipeline import CFNetChangePipeline
    from cfnet_change_detection_pipeline import pipeline as pl
    from test_adaptation import _StubModel

    torch.manual_seed(0)
    model = _StubModel()
    with torch.no_grad():
        model.change_decoder.weight.copy_(torch.tensor([0.5, -0.2, 0.1]))
    save_file({k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}, str(tmp_path / pl.CONVERTED_WEIGHTS_NAME))
    monkeypatch.setattr(pl, "verify_converted", lambda path=None: {"files": []})
    return CFNetChangePipeline(model=model, device="cpu", weights_dir=tmp_path, source="stub")


def test_validation_loss_is_deterministic_and_is_epoch_0(stub_pipeline):
    import torch

    val = synthetic_records(2, seed=50)
    first = stub_pipeline.validation_loss(val)
    assert first == stub_pipeline.validation_loss(val)
    others = {round(stub_pipeline.validation_loss(val, draw_seed=s), 8) for s in range(4)}
    assert len(others) > 1  # different draws do move the number: that is the noise Section 6 shows
    result = stub_pipeline.adapt(synthetic_records(4), val, epochs=2, lr=1e-2, batch_size=2)
    assert result["history"][0]["val_loss"] == first and result["validation_draw_seed"] == 0
    assert "re-seeded identically for every epoch" in result["selection"]
    kept = result["history"][result["best_epoch"]]["val_loss"]
    assert stub_pipeline.validation_loss(val) == pytest.approx(kept, abs=1e-7)
    assert isinstance(torch.get_num_threads(), int)


def test_adapt_refuses_an_adapted_pipeline_and_reset_restores_the_base(stub_pipeline):
    import torch

    base = {k: v.clone() for k, v in stub_pipeline.model.state_dict().items()}
    val = synthetic_records(2, seed=50)
    frozen = stub_pipeline.validation_loss(val)
    stub_pipeline.adapt(synthetic_records(4), val, epochs=2, lr=1e-2, batch_size=2, trainable="decoders")
    with pytest.raises(ValueError, match="already adapted; call reset_to_pretrained"):
        stub_pipeline.adapt(synthetic_records(4), val, epochs=1, lr=1e-2)
    assert stub_pipeline.reset_to_pretrained() is stub_pipeline
    assert stub_pipeline.adapter is None
    assert all(torch.equal(base[k], v) for k, v in stub_pipeline.model.state_dict().items())
    assert all(not p.requires_grad for p in stub_pipeline.model.parameters())
    again = stub_pipeline.adapt(synthetic_records(4), val, epochs=1, lr=1e-2, batch_size=2)
    assert again["history"][0]["val_loss"] == frozen and again["history"][0]["note"] == "frozen model"
