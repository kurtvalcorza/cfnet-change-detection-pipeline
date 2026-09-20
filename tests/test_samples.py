"""Offline tests for the pinned LEVIR-CD sample (tarball + member pins), role assignment, BYOD loaders and sample
export."""

from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile

import numpy as np
import pytest

from cfnet_change_detection_pipeline import (
    SAMPLE_RECORDS,
    TAR_SHA256,
    check_split_disjoint,
    dataset_manifest,
    extract_pinned_members,
    fetch_corpus,
    fetch_sample_dataset,
    fetch_tarball,
    load_byod_dataset,
    read_corpus,
    split_dataset,
    write_dataset_csv,
    write_sample_pair,
)
from cfnet_change_detection_pipeline import samples as sm
from conftest import synthetic_pair, synthetic_records


def test_pinned_records_are_consistent(forbid_model_imports):
    assert len(SAMPLE_RECORDS) == 64 and len(TAR_SHA256) == 64 and sm.TAR_BYTES > 3_000_000_000
    roles: dict[str, int] = {}
    pairs: dict[str, set[int]] = {}
    for key, role, pair, before, b_bytes, b_sha, after, a_bytes, a_sha, label, l_bytes, l_sha in SAMPLE_RECORDS:
        roles[role] = roles.get(role, 0) + 1
        split = {"train": "train", "validation": "val", "test": "test"}[role]
        assert key.startswith(f"{split}_{pair}_") and role in sm.ROLES
        assert before == f"LEVIR-CD-processed/{split}/A/{key}.png" and after == f"LEVIR-CD-processed/{split}/B/{key}.png"
        assert label == f"LEVIR-CD-processed/{split}/label/{key}.png" and ".." not in before and not before.startswith("/")
        assert b_bytes > 0 and a_bytes > 0 and 0 < l_bytes < b_bytes and len(b_sha) == len(a_sha) == len(l_sha) == 64
        assert pair not in pairs.setdefault(role, set()), f"two crops of pair {pair} in {role}"
        pairs[role].add(pair)
    assert roles == {"train": 32, "validation": 8, "test": 24}
    assert len({r[0] for r in SAMPLE_RECORDS}) == 64 and len({r[5] for r in SAMPLE_RECORDS}) == 64
    assert sum(r[4] + r[7] + r[10] for r in SAMPLE_RECORDS) == sm.CORPUS_BYTES
    assert sm.CORPUS_BASE_URL.startswith("https://huggingface.co/datasets/") and sm.DATASET_REVISION in sm.CORPUS_BASE_URL
    assert "academic purposes only" in sm.CORPUS_LICENSE


def _png_bytes(array) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _fake_tarball(monkeypatch, n_per_role=(4, 2, 2)):
    """Replace the record table and the tarball pins with a small synthetic gzipped tarball (plus a decoy member),
    with the archive's `./` member prefix."""
    records, members = [], {}
    index = 0
    for role, count in zip(sm.ROLES, n_per_role, strict=True):
        split = {"train": "train", "validation": "val", "test": "test"}[role]
        for _ in range(count):
            before, after, label = synthetic_pair(seed=index)
            key = f"{split}_{index + 1}_{index % 25}"
            triple = {}
            for folder, array in (("A", before), ("B", after), ("label", np.where(label == 1, 255, 0).astype(np.uint8))):
                member = f"LEVIR-CD-processed/{split}/{folder}/{key}.png"
                data = _png_bytes(array)
                members["./" + member] = data
                triple[folder] = (member, len(data), hashlib.sha256(data).hexdigest())
            records.append((key, role, index + 1, *triple["A"], *triple["B"], *triple["label"]))
            index += 1
    members["./LEVIR-CD-processed/decoy.py"] = b"print('not pinned')"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    tar_bytes = buffer.getvalue()
    monkeypatch.setattr(sm, "SAMPLE_RECORDS", tuple(records))
    monkeypatch.setattr(sm, "TAR_BYTES", len(tar_bytes))
    monkeypatch.setattr(sm, "TAR_SHA256", hashlib.sha256(tar_bytes).hexdigest())
    return tar_bytes


def test_fetch_tarball_verifies_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    tar_bytes = _fake_tarball(monkeypatch)
    calls = []

    def fetcher(url):
        calls.append(url)
        return tar_bytes

    path = fetch_tarball(cache_dir=tmp_path, fetcher=fetcher)
    assert path.name == sm.TAR_NAME and len(calls) == 1 and calls[0].endswith(sm.TAR_NAME)
    fetch_tarball(cache_dir=tmp_path, fetcher=fetcher)
    assert len(calls) == 1  # cached
    with pytest.raises(ValueError, match="pinned"):
        fetch_tarball(cache_dir=tmp_path / "other", fetcher=lambda url: b"tampered")


def test_extract_pinned_members_only(tmp_path, monkeypatch, forbid_model_imports):
    tar_bytes = _fake_tarball(monkeypatch)
    tar_path = tmp_path / "fake.tar.gz"
    tar_path.write_bytes(tar_bytes)
    out = extract_pinned_members(tar_path, cache_dir=tmp_path)
    assert len(out) == 24 and not (tmp_path / "crops" / "decoy.py").exists()
    assert sorted(p.name for p in (tmp_path / "crops").iterdir()) == sorted(sm._cache_name(m) for m in out)
    assert (tmp_path / "crops" / "train_A_train_1_0.png").is_file()
    extra = (
        "test_99_0",
        "test",
        99,
        "LEVIR-CD-processed/test/A/test_99_0.png",
        1,
        "0" * 64,
        "LEVIR-CD-processed/test/B/test_99_0.png",
        1,
        "0" * 64,
        "LEVIR-CD-processed/test/label/test_99_0.png",
        1,
        "0" * 64,
    )
    monkeypatch.setattr(sm, "SAMPLE_RECORDS", (*sm.SAMPLE_RECORDS, extra))
    with pytest.raises(ValueError, match="does not contain"):
        extract_pinned_members(tar_path, cache_dir=tmp_path / "again")


def test_sample_dataset_roles_and_manifest(tmp_path, monkeypatch, forbid_model_imports):
    tar_bytes = _fake_tarball(monkeypatch)
    splits = fetch_sample_dataset(cache_dir=tmp_path, fetcher=lambda url: tar_bytes)
    assert {k: len(v) for k, v in splits.items()} == {"train": 4, "validation": 2, "test": 2}
    assert splits["test"][0]["id"] == "test-000" and splits["test"][0]["before"].shape == (256, 256, 3)
    assert splits["test"][0]["before"].dtype == np.uint8 and set(np.unique(splits["test"][0]["label"])) <= {0, 1}
    manifest = dataset_manifest(splits)
    assert manifest["disjoint"] == {"train": 4, "validation": 2, "test": 2} and len(manifest["digest"]) == 64
    assert manifest["splits"]["train"]["regions"] == ["train-pair-1", "train-pair-2", "train-pair-3", "train-pair-4"]
    assert 0.05 < manifest["splits"]["test"]["change_fraction"] < 0.2
    files = fetch_corpus(cache_dir=tmp_path, fetcher=lambda url: (_ for _ in ()).throw(AssertionError(url)))
    assert len(files) == 8  # served from the extracted cache, nothing fetched
    assert read_corpus(files)["train"][0]["source"].startswith(sm.CORPUS_BASE_URL)


def test_check_split_disjoint_detects_leakage(forbid_model_imports):
    records = synthetic_records(4)
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint({"train": records[:2], "test": [records[0]]})
    grouped = [{**r, "region": "train-pair-7"} for r in records]
    with pytest.raises(ValueError, match="source pair 'train-pair-7' has crops in both"):
        check_split_disjoint({"train": grouped[:2], "test": grouped[2:]})


def test_split_dataset(forbid_model_imports):
    records = synthetic_records(12)
    splits = split_dataset(records, seed=1)
    assert sum(len(v) for v in splits.values()) == 12 and len(splits["train"]) >= 4 and splits["test"]
    check_split_disjoint(splits)
    assert sum(len(v) for v in split_dataset(records + [{**records[0], "id": "dup"}]).values()) == 12
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, test_fraction=0.9)


def test_byod_directory_and_zip_loaders(tmp_path, forbid_model_imports):
    records = synthetic_records(5)
    folder = tmp_path / "byod"
    folder.mkdir()
    for record in records:
        write_sample_pair(
            record, folder / f"{record['id']}_A.png", folder / f"{record['id']}_B.png", folder / f"{record['id']}_label.png"
        )
    write_dataset_csv(records, folder / "pairs.csv")
    loaded = load_byod_dataset(folder)
    assert [r["id"] for r in loaded] == [r["id"] for r in records]
    assert np.array_equal(loaded[0]["before"], records[0]["before"]) and np.array_equal(loaded[0]["after"], records[0]["after"])
    assert np.array_equal(loaded[0]["label"], np.where(records[0]["label"] == -1, 0, records[0]["label"]))
    archive = tmp_path / "byod.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in folder.iterdir():
            zf.write(path, f"nested/{path.name}")
    assert len(load_byod_dataset(archive)) == 5
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("readme.txt", "no table")
    with pytest.raises(ValueError, match="pairs.csv"):
        load_byod_dataset(bad)
    with pytest.raises(ValueError, match="directory or a .zip"):
        load_byod_dataset(tmp_path / "missing.tar")
    (folder / "pairs.csv").write_text("id,before\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_byod_dataset(folder)
