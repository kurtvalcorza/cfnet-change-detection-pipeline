"""CFNet bi-temporal change detection (`wifibk/CFNet`, the LEVIR-CD checkpoint) DIMER pipeline: verified snapshot,
one-time conversion of the pickled state dict into safetensors, binary building-change maps for pairs of
co-registered RGB patches, held-out evaluation against an all-unchanged baseline, and bounded fine-tuning of the
change decoder to a user's labelled pairs with a portable adapter.

CFNet (Wu et al., 2025) is a content-focuser network: a shared EfficientNet-B5 encoder over both dates, one
content decoder per date, a focuser that turns the cosine distance between the dates' content maps into per-scale
change weights, and a change decoder that fuses the two dates with 3-D convolutions and decodes one change map.
The checkpoint packaged here, `levir-cd.pth`, is the authors' model trained on LEVIR-CD — 0.5 m Google Earth
patches of Texas cities with building-change labels (Chen and Shi, 2020) — which they report at F1 92.18 / IoU
85.49 on that dataset's test split.

The upstream asset is a torch zip archive whose pickle references only `collections.OrderedDict`,
`torch._utils._rebuild_tensor_v2` and two storage classes (verified statically by `audit_pickle`). Under the fleet
asset specification (§11) that is executable serialization, so this package converts it once —
`torch.load(weights_only=True)`, a strict load into the vendored architecture — into safetensors with a pinned
digest, and serves only the converted file. The architecture is `modeling.py`, plain PyTorch plus the stem and four
stages of torchvision's EfficientNet-B5 built with `weights=None`; nothing is fetched from the Hub at load time
except the manifest-listed files.

Everything model-related is imported lazily so that snapshot verification, the pickle audit and input validation
run (and can refuse) before `torch` or `torchvision` are imported (fleet RTM-001). `numpy` and `PIL` are used for
images and are imported freely.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import pickletools
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "wifibk/CFNet"
MODEL_REVISION = "c23279428d14186d67ce199b3db358038bf37585"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "cfnet-levir-cd"
ARTIFACT_FORMAT = "org.valcorza.cfnet-change-detection.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Immutable upstream source asset (a torch pickle of the state dict; see docs/WEIGHTS.md).
SOURCE_CKPT_NAME = "levir-cd.pth"
SOURCE_CKPT_BYTES = 15_805_666
SOURCE_CKPT_SHA256 = "22ab286b2138ab1082e04290b27cbff5abd3802375a6b2f01d4ba78ca8b0c1aa"
# Code-free serving file produced deterministically by `convert_model` (asset spec §11.2).
CONVERTED_WEIGHTS_NAME = "cfnet-levir-cd.safetensors"
CONVERTED_SHA256 = "b348186e5003a7447ab5eb5874204c50e63ad71aa61aca3ff0a80746116963d9"
CONVERTED_BYTES = 15_598_980
# Static-audit digest of the source pickle (sorted global names), see `audit_pickle`.
PICKLE_AUDIT_SHA256 = "5b9f0ba08490293d6c17b9cef219991e1a6edda31609429679f8dca1af5a7b10"
CKPT_ALLOWED_GLOBALS = frozenset(
    {"collections.OrderedDict", "torch._utils._rebuild_tensor_v2", "torch.FloatStorage", "torch.LongStorage"}
)

# Architecture and data-contract facts (upstream `model/` and `dataset/dataset.py` at the vendored commit).
UPSTREAM_CODE_COMMIT = "54acadab23b9d9395ec6814386c2d4a1253eac5a"
STATE_TENSORS = 776
STATE_NUMEL = 3_877_781
PARAMETER_COUNT = 3_838_563  # nn.Parameters; the rest are BatchNorm buffers
ENCODER_TENSORS = 428  # torchvision efficientnet_b5.features[0:5]
NUM_CLASSES = 2
CLASS_NAMES: tuple[str, ...] = ("unchanged", "changed")
IGNORE_INDEX = -1
CHANGE_THRESHOLD = 0.5  # upstream: a pixel is changed where the tanh change map exceeds 0.5
# Upstream standardises each date separately, after `cv2.imread` (BGR channel order) and division by 255, with
# the LEVIR-CD statistics of `dataset/dataset.py`; the tuples below are in BGR order, as the code applied them.
MEANS_BEFORE: tuple[float, ...] = (0.45028868, 0.44673658, 0.38134101)
STDS_BEFORE: tuple[float, ...] = (0.17450373, 0.16485656, 0.15314709)
MEANS_AFTER: tuple[float, ...] = (0.34578751, 0.33841163, 0.28902323)
STDS_AFTER: tuple[float, ...] = (0.1292925, 0.12592704, 0.11870409)
MIN_SIDE = 64
MAX_SIDE = 2048
SIDE_MULTIPLE = 32  # four stride-2 stages plus the stem
SAMPLE_SIZE = 256  # the LEVIR-CD crops of the tutorial
MIN_RECORDS = 4
MAX_RECORDS = 2_000
ADAPTATION_MODES = ("change_decoder", "decoders")  # the only scopes an adapter may declare
TRAINABLE_PREFIXES: dict[str, tuple[str, ...]] = {
    "change_decoder": ("change_decoder.",),
    "decoders": ("change_decoder.", "content_decoder_1.", "content_decoder_2."),
}
CONTENT_LOSS_WEIGHT = 0.1  # upstream `beta`; the change loss weight `alpha` is 1


# --------------------------------------------------------------------------------------------------
# manifest, staging, static pickle audit and conversion
# --------------------------------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_manifest(root: Path, model_id: str, revision: str) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no snapshot manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("modelId") != model_id:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {model_id!r}")
    if manifest.get("revision") != revision:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {revision!r}")
    listed = {entry["path"] for entry in manifest["files"]}
    if SOURCE_CKPT_NAME not in listed:
        raise ValueError(f"manifest does not list {SOURCE_CKPT_NAME}; refusing to proceed")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256_file(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
        if entry["path"] == SOURCE_CKPT_NAME and (size, digest) != (SOURCE_CKPT_BYTES, SOURCE_CKPT_SHA256):
            raise ValueError(f"{entry['path']}: manifest digest disagrees with the package constant")
    return manifest


def verify_converted(path: str | Path | None = None) -> dict[str, Any]:
    """Check the converted serving file (safetensors) against the pinned digest."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    file_path = root / CONVERTED_WEIGHTS_NAME
    if not file_path.is_file():
        raise FileNotFoundError(f"converted file missing: {file_path}")
    size = file_path.stat().st_size
    if size != CONVERTED_BYTES:
        raise ValueError(f"{CONVERTED_WEIGHTS_NAME}: size {size} != pinned {CONVERTED_BYTES}")
    digest = _sha256_file(file_path)
    if digest != CONVERTED_SHA256:
        raise ValueError(f"{CONVERTED_WEIGHTS_NAME}: sha256 {digest} != pinned {CONVERTED_SHA256}")
    return {"files": [{"path": CONVERTED_WEIGHTS_NAME, "bytes": size, "sha256": digest}]}


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check the snapshot against its DIMER manifest (size + SHA-256 of every listed Hub file) and, when the
    converted serving file is present, that against the pinned digest."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _verify_manifest(root, MODEL_ID, MODEL_REVISION)
    converted = (root / CONVERTED_WEIGHTS_NAME).is_file()
    if converted:
        verify_converted(root)
    return {**manifest, "converted": converted}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at the pinned revision straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest entries that are absent locally (a fresh clone commits the manifest and git-ignores the
    checkpoint and the safetensors it converts to)."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _pickle_globals(data: bytes) -> dict[str, int]:
    """Every global a pickle stream would import, collected with `pickletools.genops` (no execution)."""
    found: dict[str, int] = {}
    stack: list[Any] = []
    for op, arg, _pos in pickletools.genops(io.BytesIO(data)):
        if op.name == "GLOBAL":  # pickletools renders the (module, name) pair space-separated
            key = arg.replace("\n", " ").replace(" ", ".", 1)
            found[key] = found.get(key, 0) + 1
        elif op.name == "STACK_GLOBAL":
            key = f"{stack[-2]}.{stack[-1]}"
            found[key] = found.get(key, 0) + 1
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "SHORT_BINSTRING", "BINSTRING"):
            stack.append(arg)
        elif op.name in ("MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"):
            pass
        else:
            stack.append(None)
    return found


def audit_pickle(path: str | Path, *, allowed: frozenset[str] = CKPT_ALLOWED_GLOBALS) -> dict[str, Any]:
    """Statically list the globals a pickle (plain, or inside a torch zip archive) would import and refuse any
    outside `allowed`. Executes nothing. Returns the sorted globals and their digest."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"file not found: {file_path}")
    data = file_path.read_bytes()
    found: dict[str, int] = {}
    nested = 0
    if data[:4] == b"PK\x03\x04":
        archive = zipfile.ZipFile(io.BytesIO(data))
        for name in archive.namelist():
            if name.endswith(".pkl"):
                nested += 1
                for key, count in _pickle_globals(archive.read(name)).items():
                    found[key] = found.get(key, 0) + count
    else:
        found = _pickle_globals(data)
    violations = sorted(name for name in found if name not in allowed)
    summary = {
        "file": file_path.name,
        "torch_archive": data[:4] == b"PK\x03\x04",
        "pickles": nested if nested else 1,
        "globals": sorted(found),
        "violations": violations,
        "audit_sha256": hashlib.sha256("\n".join(sorted(found)).encode("utf-8")).hexdigest(),
    }
    if violations:
        raise ValueError(f"{file_path.name}: pickle audit failed, globals outside the allow-list: {violations}")
    return summary


def _check_pinned_source(root: Path) -> dict[str, Any]:
    source = root / SOURCE_CKPT_NAME
    if not source.is_file():
        raise FileNotFoundError(f"source file not found: {source}")
    size = source.stat().st_size
    if size != SOURCE_CKPT_BYTES:
        raise ValueError(f"{SOURCE_CKPT_NAME}: size {size} != pinned {SOURCE_CKPT_BYTES}")
    digest = _sha256_file(source)
    if digest != SOURCE_CKPT_SHA256:
        raise ValueError(f"{SOURCE_CKPT_NAME}: sha256 {digest} != pinned {SOURCE_CKPT_SHA256}")
    audit = audit_pickle(source)
    if audit["audit_sha256"] != PICKLE_AUDIT_SHA256:
        raise ValueError(f"{SOURCE_CKPT_NAME}: pickle audit digest {audit['audit_sha256']} != pinned {PICKLE_AUDIT_SHA256}")
    return {"path": SOURCE_CKPT_NAME, "bytes": size, "sha256": digest, "audit": audit}


def build_model() -> Any:
    """Instantiate the architecture from the vendored module (the EfficientNet-B5 stages from the installed
    torchvision package with `weights=None`; no download)."""
    from .modeling import CFNet

    return CFNet()


def convert_model(path: str | Path | None = None) -> dict[str, Any]:
    """Convert the pinned checkpoint into safetensors, deterministically, after size, digest and static-audit
    checks: torch's weights-only unpickler, a strict load into the vendored architecture, and the model's own
    state dict saved."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    source = _check_pinned_source(root)
    import torch
    from safetensors.torch import save_file

    started = time.perf_counter()
    checkpoint = root / SOURCE_CKPT_NAME
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"{SOURCE_CKPT_NAME} did not unpickle to a state dict of tensors")
    if len(state) != STATE_TENSORS:
        raise ValueError(f"{SOURCE_CKPT_NAME}: state dict has {len(state)} tensors, expected {STATE_TENSORS}")
    model = build_model()
    model.load_state_dict(state, strict=True)
    canonical = {k: v.contiguous() for k, v in model.state_dict().items()}
    n_elements = sum(v.numel() for v in canonical.values())
    if len(canonical) != STATE_TENSORS or n_elements != STATE_NUMEL:
        raise ValueError(
            f"converted state dict has {len(canonical)} tensors / {n_elements} elements; expected {STATE_TENSORS} / {STATE_NUMEL}"
        )
    save_file(canonical, str(root / CONVERTED_WEIGHTS_NAME), metadata={"format": "pt"})
    report = verify_converted(root)
    return {
        "source": {k: v for k, v in source.items() if k != "audit"},
        "audit": source["audit"],
        "checkpoint": {
            "entries": "a plain state dict (no optimizer, no metadata)",
            "state_dict_tensors": len(state),
            "encoder_tensors": sum(1 for k in state if k.startswith("encoder.")),
        },
        "converted": report["files"],
        "seconds": round(time.perf_counter() - started, 2),
    }


# --------------------------------------------------------------------------------------------------
# image pairs, labels and validation (no model import)
# --------------------------------------------------------------------------------------------------

INPUT_SCHEMA: dict[str, Any] = {
    "record": (
        "{id, before, after, label?}: before / after = (H, W, 3) uint8 RGB images of the same co-registered footprint "
        "at two dates (or PNG / JPEG paths); label = (H, W) int mask with 0 = unchanged, 1 = changed, -1 = no data "
        "(or a PNG path with 0 / 255), optional"
    ),
    "image_size": (
        f"H and W in [{MIN_SIDE}, {MAX_SIDE}], both multiples of {SIDE_MULTIPLE}; the tutorial uses {SAMPLE_SIZE} × {SAMPLE_SIZE}"
    ),
    "value_units": (
        "8-bit RGB (the pipeline reorders to BGR and standardises each date with the LEVIR-CD statistics, as upstream)"
    ),
    "classes": {str(i): name for i, name in enumerate(CLASS_NAMES)},
    "ignore_index": IGNORE_INDEX,
    "decision_rule": f"changed where the tanh change map exceeds {CHANGE_THRESHOLD} (upstream)",
    "records": [MIN_RECORDS, MAX_RECORDS],
    "validation": (
        "record shape, dtype range, equal sizes of the two dates, side multiples and label values only. Nothing checks "
        "that the two images show the same place, that they are co-registered, that the resolution is about 0.5 m, "
        "or that the label was drawn for this pair -- any two same-sized RGB images are compared without complaint"
    ),
}


def read_image(path: str | Path) -> Any:
    """Load an RGB image from a PNG / JPEG as uint8 (H, W, 3); greyscale and RGBA are converted."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))


def read_mask(path: str | Path) -> Any:
    """Load a label raster (0 = unchanged, 255 = changed; upstream thresholds at 127) as int64 (H, W) with 0 / 1."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    return np.ascontiguousarray((array > 127).astype(np.int64))


def _check_image(image: Any, label_name: str, what: str) -> Any:
    import numpy as np

    if isinstance(image, str | Path):
        if not Path(image).is_file():
            raise ValueError(f"{label_name}: {what} file not found: {image}")
        image = read_image(image)
    try:
        array = np.asarray(image)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label_name}: {what} must be an image array") from exc
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{label_name}: {what} must have shape (H, W, 3), got {array.shape}")
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.number) or float(array.min()) < 0 or float(array.max()) > 255:
            raise ValueError(f"{label_name}: {what} must be uint8 RGB or numeric in [0, 255]")
        array = np.rint(array).astype(np.uint8)
    height, width = array.shape[:2]
    if not (MIN_SIDE <= height <= MAX_SIDE and MIN_SIDE <= width <= MAX_SIDE):
        raise ValueError(f"{label_name}: {what} sides must be in [{MIN_SIDE}, {MAX_SIDE}], got {(height, width)}")
    if height % SIDE_MULTIPLE or width % SIDE_MULTIPLE:
        raise ValueError(f"{label_name}: {what} sides must be multiples of {SIDE_MULTIPLE}, got {(height, width)}")
    return np.ascontiguousarray(array)


def _check_record(record: Any, index: int) -> dict[str, Any]:
    import numpy as np

    label_name = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label_name} must be a mapping with id/before/after[/label]")
    for key in ("id", "before", "after"):
        if key not in record:
            raise ValueError(f"{label_name} is missing {key!r}")
    rid = record["id"]
    if not isinstance(rid, str) or not rid or len(rid) > 128:
        raise ValueError(f"{label_name}: id must be a non-empty string of at most 128 characters")
    before = _check_image(record["before"], label_name, "before")
    after = _check_image(record["after"], label_name, "after")
    if before.shape != after.shape:
        raise ValueError(f"{label_name}: before and after must have the same shape, got {before.shape} and {after.shape}")
    item: dict[str, Any] = {"id": rid, "before": before, "after": after}
    label = record.get("label")
    if label is not None:
        if isinstance(label, str | Path):
            if not Path(label).is_file():
                raise ValueError(f"{label_name}: label file not found: {label}")
            label = read_mask(label)
        try:
            mask = np.asarray(label)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label_name}: label must be an integer array") from exc
        if mask.shape != before.shape[:2]:
            raise ValueError(f"{label_name}: label must have shape {before.shape[:2]}, got {mask.shape}")
        if not np.issubdtype(mask.dtype, np.integer) and not np.all(mask == np.round(mask)):
            raise ValueError(f"{label_name}: label values must be integers")
        allowed = {0, 1, IGNORE_INDEX}
        found = set(np.unique(mask).astype(int).tolist())
        if not found <= allowed:
            raise ValueError(f"{label_name}: label values {sorted(found - allowed)} outside {sorted(allowed)}")
        item["label"] = np.ascontiguousarray(mask.astype(np.int64))
    for key in ("split", "region", "source", "source_id"):
        if key in record:
            item[key] = record[key]
    return item


def check_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one record and return its normalised copy (uint8 RGB dates, int64 label)."""
    return _check_record(record, 0)


def pair_digest(record: Mapping[str, Any]) -> str:
    checked = _check_record(record, 0)
    digest = hashlib.sha256(checked["before"].tobytes())
    digest.update(checked["after"].tobytes())
    if "label" in checked:
        digest.update(checked["label"].tobytes())
    return digest.hexdigest()


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [[r["id"], pair_digest(r)] for r in records]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_RECORDS,
    max_records: int = MAX_RECORDS,
    require_labels: bool = True,
) -> dict[str, Any]:
    """Structural validation of a pair dataset; raises ValueError before any model import."""

    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, before, after, label} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked = []
    ids: set[str] = set()
    changed = unchanged = ignored = 0
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        if require_labels and "label" not in item:
            raise ValueError(f"records[{index}] has no label; every record of a labelled dataset needs one")
        if "label" in item:
            changed += int((item["label"] == 1).sum())
            unchanged += int((item["label"] == 0).sum())
            ignored += int((item["label"] == IGNORE_INDEX).sum())
        checked.append(item)
    labelled = sum("label" in r for r in checked)
    if require_labels and labelled and changed == 0:
        raise ValueError("no changed pixel in the dataset; nothing to learn or evaluate")
    total = changed + unchanged
    sizes = sorted({tuple(r["before"].shape[:2]) for r in checked})
    return {
        "records": checked,
        "n_records": len(checked),
        "n_labelled": labelled,
        "sizes": [list(s) for s in sizes],
        "change_fraction": round(changed / total, 4) if total else None,
        "ignored_pixels": ignored,
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def validate_inputs(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one record; returns its id, size, and label change fraction."""
    item = _check_record(record, 0)
    report = {"id": item["id"], "shape": tuple(item["before"].shape), "has_label": "label" in item}
    if "label" in item:
        label = item["label"]
        valid = int((label != IGNORE_INDEX).sum())
        report["change_fraction"] = round(float((label == 1).sum()) / max(valid, 1), 4)
        report["ignored_pixels"] = int((label == IGNORE_INDEX).sum())
    return report


# --------------------------------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------------------------------


def _normalise(images: Any, *, which: str) -> Any:
    """(B, H, W, 3) uint8 RGB -> (B, 3, H, W) float32 standardised as the upstream loader did: `cv2.imread`
    delivers BGR, so the channels are reversed, divided by 255 and standardised per date with the LEVIR-CD
    statistics (`which` = 'before' or 'after')."""
    import numpy as np

    mean, std = (MEANS_BEFORE, STDS_BEFORE) if which == "before" else (MEANS_AFTER, STDS_AFTER)
    batch = np.asarray(images, dtype=np.float32)[..., ::-1] / 255.0  # RGB -> BGR
    batch = np.transpose(batch, (0, 3, 1, 2))
    mean_a = np.asarray(mean, dtype=np.float32)[None, :, None, None]
    std_a = np.asarray(std, dtype=np.float32)[None, :, None, None]
    return np.ascontiguousarray((batch - mean_a) / std_a)


@dataclass
class CFNetChangePipeline:
    """Building-change maps and bounded change-decoder fine-tuning on top of the verified CFNet LEVIR-CD model."""

    model: Any
    device: str
    weights_dir: Path
    source: str
    adapter: dict[str, Any] | None = None

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        require_source: bool = True,
        report: Callable[[dict[str, Any]], None] | None = None,
    ) -> CFNetChangePipeline:
        """Verify, convert if needed, rebuild from the vendored module and strictly load. With
        `require_source=False` the checkpoint may be absent (the DIMER-hosted case) as long as the converted file
        verifies. `report` receives the audit and conversion records when a conversion happens."""
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if require_source:
            stage_missing_files(root, allow_download=allow_download)
            snapshot = verify_snapshot(root)
            if not snapshot["converted"]:
                conversion = convert_model(root)
                if report is not None:
                    report({"conversion": conversion})
                snapshot = verify_snapshot(root)
            elif report is not None:
                report({"conversion": "converted file already present and digest-verified"})
            source = "converted from the manifest-verified source checkpoint"
        else:
            verify_converted(root)
            source = "converted file, pinned digest (source checkpoint not required)"
        import torch
        from safetensors.torch import load_file

        chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if chosen.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("device='cuda' requested but CUDA is not available")
        model = build_model()
        state = load_file(str(root / CONVERTED_WEIGHTS_NAME))
        model.load_state_dict(state, strict=True)
        n_params = sum(p.numel() for p in model.parameters())
        if n_params != PARAMETER_COUNT:
            raise ValueError(f"rebuilt model has {n_params} parameters, expected {PARAMETER_COUNT}")
        model.to(torch.device(chosen)).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return cls(model=model, device=chosen, weights_dir=root, source=source)

    # ---- forward ---------------------------------------------------------------------------------------

    def _inputs(self, before: Any, after: Any) -> tuple[Any, Any]:
        import torch

        x1 = torch.from_numpy(_normalise(before, which="before")).to(self.device)
        x2 = torch.from_numpy(_normalise(after, which="after")).to(self.device)
        return x1, x2

    def _change_map(self, before: Any, after: Any, *, grad: bool = False, with_content: bool = False) -> Any:
        """(B, H, W, 3) uint8 pairs -> (B, H, W) float32 change map in (-1, 1) [and the content / focus maps]."""
        import torch

        x1, x2 = self._inputs(before, after)
        use_amp = self.device.startswith("cuda")
        context = torch.enable_grad() if grad else torch.inference_mode()
        with context, torch.autocast(device_type=self.device.split(":")[0], dtype=torch.float16, enabled=use_amp):
            if with_content:
                change, maps_1, maps_2, focuses = self.model.forward_with_content(x1, x2)
                return change.float(), maps_1, maps_2, focuses
            return self.model(x1, x2).float()

    # ---- inference -------------------------------------------------------------------------------------

    def predict(self, records: Sequence[Mapping[str, Any]], *, batch_size: int = 4) -> dict[str, Any]:
        """Detect change: per record the binary mask (H, W) uint8 (1 = changed, the upstream threshold on the
        change map), the change map (H, W) float32 in (-1, 1) — the network's output, not a calibrated
        probability — and the changed fraction."""
        import numpy as np

        checked = validate_dataset(records, min_records=1, require_labels=False)["records"]
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be an int in 1..32")
        started = time.perf_counter()
        predictions = []
        groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
        for record in checked:  # batches hold one size at a time
            groups.setdefault(tuple(record["before"].shape), []).append(record)
        by_id: dict[str, dict[str, Any]] = {}
        for group in groups.values():
            for start in range(0, len(group), batch_size):
                batch = group[start : start + batch_size]
                maps = (
                    self._change_map(np.stack([r["before"] for r in batch]), np.stack([r["after"] for r in batch])).cpu().numpy()
                )
                for record, change_map in zip(batch, maps, strict=True):
                    mask = (change_map > CHANGE_THRESHOLD).astype(np.uint8)
                    by_id[record["id"]] = {
                        "id": record["id"],
                        "mask": mask,
                        "change_map": change_map.astype(np.float32),
                        "changed_fraction": round(float(mask.mean()), 4),
                    }
        predictions = [by_id[r["id"]] for r in checked]
        return {
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "adapted": self.adapter is not None},
            "classes": list(CLASS_NAMES),
            "decision_rule": INPUT_SCHEMA["decision_rule"],
            "predictions": predictions,
            "seconds": round(time.perf_counter() - started, 3),
        }

    def evaluate(self, records: Sequence[Mapping[str, Any]], *, batch_size: int = 4) -> dict[str, Any]:
        """Pixel-level change metrics on labelled pairs (ignore index excluded): F1, IoU, precision and recall of
        the changed class as upstream scores them, plus accuracy and the change fractions, with the all-unchanged
        baseline scored on the same pixels."""
        from .metrics import change_metrics, unchanged_baseline

        checked = validate_dataset(records, min_records=1)["records"]
        started = time.perf_counter()
        result = self.predict(checked, batch_size=batch_size)
        masks = [p["mask"] for p in result["predictions"]]
        labels = [r["label"] for r in checked]
        return {
            "n_records": len(checked),
            "metric": (
                "pixel F1 / IoU of the changed class over the labelled pixels of the held-out pairs (ignore index excluded)"
            ),
            "model": change_metrics(masks, labels, ignore_index=IGNORE_INDEX),
            "baseline_unchanged": unchanged_baseline(labels, ignore_index=IGNORE_INDEX),
            "adapted": self.adapter is not None,
            "seconds": round(time.perf_counter() - started, 3),
        }

    # ---- adaptation ------------------------------------------------------------------------------------

    def _trainable(self, mode: str) -> list[str]:
        if mode not in ADAPTATION_MODES:
            raise ValueError(f"trainable must be one of {ADAPTATION_MODES}")
        prefixes = TRAINABLE_PREFIXES[mode]
        return sorted(name for name, _param in self.model.named_parameters() if name.startswith(prefixes))

    @staticmethod
    def _content_loss(x1: Any, x2: Any, generator: Any, *, mode: str) -> Any:
        """Upstream `ContentLoss`, vectorised: for `n = W` random pixel pairs, the cosine similarity between the
        two pixels' feature vectors is computed within each date; the loss is the mean absolute difference between
        the dates (`unchange`) or one minus it (`change`)."""
        import torch

        batch, channels, height, width = x1.shape
        flat_1 = x1.reshape(batch, channels, -1)
        flat_2 = x2.reshape(batch, channels, -1)
        n = width
        first = torch.randint(0, height * width, (n,), generator=generator, device="cpu").to(x1.device)
        second = torch.randint(0, height * width, (n,), generator=generator, device="cpu").to(x1.device)
        sim_1 = torch.nn.functional.cosine_similarity(flat_1[:, :, first], flat_1[:, :, second], dim=1)
        sim_2 = torch.nn.functional.cosine_similarity(flat_2[:, :, first], flat_2[:, :, second], dim=1)
        value = torch.mean(torch.abs(sim_1 - sim_2))
        return value if mode == "unchange" else 1.0 - value

    def _loss(self, change: Any, maps_1: Any, maps_2: Any, focuses: Any, target: Any, generator: Any) -> Any:
        """Upstream `Loss`: MSE between the tanh change map and the 0 / 1 target over the labelled pixels, plus
        `CONTENT_LOSS_WEIGHT / n` times the summed change- and unchange-content losses over the n scales."""
        import torch

        valid = target != IGNORE_INDEX
        main = torch.nn.functional.mse_loss(change[valid], target.float()[valid])
        content = change.new_zeros(())
        for map_1, map_2, focus in zip(maps_1, maps_2, focuses, strict=True):
            weight = focus.unsqueeze(1)
            content = content + self._content_loss(map_1 * weight, map_2 * weight, generator, mode="change")
            content = content + self._content_loss(map_1 * (1 - weight), map_2 * (1 - weight), generator, mode="unchange")
        return main + CONTENT_LOSS_WEIGHT / len(maps_1) * content

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 6,
        lr: float = 1e-4,
        batch_size: int = 4,
        trainable: str = "change_decoder",
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning of the change decoder (`trainable="change_decoder"`; `"decoders"` also unfreezes the
        two content decoders) on labelled pairs: the upstream loss (MSE on the change map plus the content terms),
        AdamW at a fixed learning rate, seeded horizontal/vertical flips applied to both dates and the label,
        float16 autocast with loss scaling on CUDA, BatchNorm statistics frozen. Epoch 0 records the frozen model;
        the epoch with the lowest validation loss is kept."""
        if not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be an int in 1..50")
        if not (0.0 < lr <= 1e-2):
            raise ValueError("lr must be in (0, 1e-2]")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 16:
            raise ValueError("batch_size must be an int in 1..16")
        names = self._trainable(trainable)
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val is not None else None
        sizes = {tuple(r["before"].shape) for r in train_checked} | {tuple(r["before"].shape) for r in val_checked or []}
        if len(sizes) != 1:
            raise ValueError(f"adaptation needs pairs of one size, got {sorted(sizes)}")
        import numpy as np
        import torch

        torch.manual_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        started = time.perf_counter()
        model = self.model
        name_set = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in name_set)
        params = [p for n, p in model.named_parameters() if n in name_set]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        use_amp = self.device.startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        rng = np.random.default_rng(seed)

        def batch_loss(batch: list[dict[str, Any]], *, grad: bool) -> Any:
            before = np.stack([r["before"] for r in batch])
            after = np.stack([r["after"] for r in batch])
            labels = np.stack([r["label"] for r in batch])
            if grad:
                if rng.random() < 0.5:
                    before, after, labels = before[:, :, ::-1], after[:, :, ::-1], labels[:, :, ::-1]
                if rng.random() < 0.5:
                    before, after, labels = before[:, ::-1], after[:, ::-1], labels[:, ::-1]
            change, maps_1, maps_2, focuses = self._change_map(
                np.ascontiguousarray(before), np.ascontiguousarray(after), grad=grad, with_content=True
            )
            target = torch.from_numpy(np.ascontiguousarray(labels)).to(self.device)
            return self._loss(change, maps_1, maps_2, focuses, target, generator)

        def val_loss() -> float | None:
            if val_checked is None:
                return None
            model.eval()
            losses = []
            for start in range(0, len(val_checked), batch_size):
                losses.append(float(batch_loss(val_checked[start : start + batch_size], grad=False)))
            return sum(losses) / len(losses)

        initial_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
        try:
            history: list[dict[str, Any]] = []
            entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val_loss": val_loss(), "note": "frozen model"}
            if val_checked is not None:
                entry["val"] = self.evaluate(val_checked, batch_size=batch_size)["model"]
            history.append(entry)
            best_val = entry["val_loss"] if entry["val_loss"] is not None else math.inf
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
            best_epoch = 0
            if progress:
                progress(entry)
            n_steps = 0
            for epoch in range(1, epochs + 1):
                model.train()
                for module in model.modules():  # BatchNorm statistics stay frozen: tiny batches would corrupt them
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
                order = rng.permutation(len(train_checked)).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    batch = [train_checked[i] for i in order[start : start + batch_size]]
                    loss = batch_loss(batch, grad=True)
                    optimiser.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    scaler.step(optimiser)
                    scaler.update()
                    losses.append(float(loss.detach()))
                    n_steps += 1
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val_loss": val_loss()}
                if val_checked is not None:
                    entry["val"] = self.evaluate(val_checked, batch_size=batch_size)["model"]
                history.append(entry)
                if progress:
                    progress(entry)
                if entry["val_loss"] is None or entry["val_loss"] < best_val:
                    best_val = entry["val_loss"] if entry["val_loss"] is not None else best_val
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
                    best_epoch = epoch
        except BaseException:
            # Transactional: a failure in training, validation or the progress callback leaves the model as it
            # was before adapt() (trained tensors restored), frozen, with no adapter attached.
            restore = dict(model.state_dict())
            restore.update(initial_state)
            model.load_state_dict(restore, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable": trainable,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "lr": lr,
            "batch_size": batch_size,
            "loss": "upstream loss: MSE on the tanh change map + 0.1 × the content-consistency terms, ignore index excluded",
            "augmentation": "seeded horizontal/vertical flips of both dates and the label",
            "batchnorm": "running statistics frozen (eval mode) during adaptation",
            "precision": "float16 autocast + GradScaler" if use_amp else "float32",
            "n_train_records": len(train_checked),
            "n_steps": n_steps,
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts -------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted tensors as safetensors with a manifest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "key": MODEL_KEY, "converted_sha256": CONVERTED_SHA256},
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {"path": ARTIFACT_WEIGHTS_NAME, "bytes": weights_path.stat().st_size, "sha256": _sha256_file(weights_path)}
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out

    @staticmethod
    def check_artifact_manifest(root: Path, manifest: Mapping[str, Any]) -> tuple[Path, str]:
        """Static checks on an adapter manifest, before any model or weights work: format and version, the pinned
        base and converted digest, exactly one weights entry named `adapter.safetensors` inside the artifact
        directory, and an adaptation mode that is one of the declared scopes. Returns the weights path and mode."""
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {manifest.get('format_version')!r} is not supported "
                f"(expected {ARTIFACT_FORMAT_VERSION!r})"
            )
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("artifact was adapted from a different base model or revision")
        if base.get("converted_sha256") != CONVERTED_SHA256:
            raise ValueError("artifact records a different converted-base digest")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must list exactly one weights file")
        entry = files[0]
        if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact weights file must be named {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / entry["path"]).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weights file must sit inside the artifact directory")
        adapter = manifest.get("adapter")
        mode = adapter.get("trainable") if isinstance(adapter, Mapping) else None
        if mode not in ADAPTATION_MODES:
            raise ValueError(f"artifact adapter.trainable must be one of {ADAPTATION_MODES}")
        if not isinstance(manifest.get("tensors"), list):
            raise ValueError("artifact manifest must list its tensors")
        return weights_path, mode

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, scope and digest, then overwrite exactly the tensors the scope allows."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path, mode = self.check_artifact_manifest(root, manifest)
        expected = self._trainable(mode)
        if sorted(manifest["tensors"]) != expected:
            raise ValueError(
                f"artifact tensor list does not match the {len(expected)} tensors that trainable={mode!r} may change"
            )
        entry = manifest["files"][0]
        if _sha256_file(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from the validated manifest")
        state = self.model.state_dict()
        for key, value in tensors.items():
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(f"artifact tensor {key} has shape {tuple(value.shape)}, model has {tuple(state[key].shape)}")
        merged = dict(state)
        merged.update({k: v.to(state[k].device, state[k].dtype) for k, v in tensors.items()})
        self.model.load_state_dict(merged, strict=True)
        self.model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": manifest["tensors"], "history": manifest.get("history", [])}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        require_source: bool = True,
    ) -> CFNetChangePipeline:
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        cls.check_artifact_manifest(root, manifest)
        pipeline = cls.from_pretrained(
            device=device, weights_dir=weights_dir, allow_download=allow_download, require_source=require_source
        )
        pipeline.load_artifact(artifact_dir)
        return pipeline
