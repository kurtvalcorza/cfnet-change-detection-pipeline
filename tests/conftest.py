import builtins

import numpy as np
import pytest

MODEL_LIBRARIES = {"torch", "torchvision", "safetensors", "huggingface_hub", "timm", "albumentations", "cv2"}


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing or initializing model libraries (fleet RTM-001)."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in MODEL_LIBRARIES:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def synthetic_pair(*, seed: int = 0, size: int = 256, change_fraction: float = 0.1):
    """Two smooth RGB 'aerial' images that share a background; the second carries a bright rectangular 'building'
    whose footprint (about `change_fraction` of the pixels) is the change label."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / size
    base = np.stack([90 + 60 * np.sin(5 * x + seed), 110 + 50 * np.cos(4 * y), 70 + 40 * np.sin(3 * x * y + 1)], axis=-1)
    before = np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    after = np.clip(base * 0.9 + 10 + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    side = int(np.sqrt(change_fraction) * size)
    top, left = rng.integers(0, size - side), rng.integers(0, size - side)
    after[top : top + side, left : left + side] = (235, 230, 220)
    label = np.zeros((size, size), dtype=np.int64)
    label[top : top + side, left : left + side] = 1
    label[:8, :8] = -1  # a no-data corner
    return before, after, label


def synthetic_records(n: int = 6, *, seed: int = 0, labels: bool = True):
    out = []
    for i in range(n):
        before, after, label = synthetic_pair(seed=seed + i)
        record = {"id": f"pair-{i:03d}", "before": before, "after": after}
        if labels:
            record["label"] = label
        out.append(record)
    return out
