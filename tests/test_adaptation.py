"""Adaptation, evaluation and artifact tests on a stub model (torch required, no weights): the scope of the
trainable tensors, epoch selection, the transactional guarantee and the artifact round trip. Plus the vendored
architecture's shape and key contract on a randomly initialised instance (torchvision required)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cfnet_change_detection_pipeline import ADAPTATION_MODES, CFNetChangePipeline, build_model  # noqa: E402
from cfnet_change_detection_pipeline import pipeline as pl  # noqa: E402
from conftest import synthetic_records  # noqa: E402


class _StubModel(torch.nn.Module):
    """Parameter names follow the vendored layout; the change map depends on the change-decoder tensors so
    training moves them. Frozen, it predicts 'unchanged' everywhere (map = 0 < threshold)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.w = torch.nn.Parameter(torch.ones(1))
        self.content_decoder_1 = torch.nn.Module()
        self.content_decoder_1.scale = torch.nn.Parameter(torch.ones(1))
        self.content_decoder_2 = torch.nn.Module()
        self.content_decoder_2.scale = torch.nn.Parameter(torch.ones(1))
        self.change_decoder = torch.nn.Module()
        self.change_decoder.weight = torch.nn.Parameter(torch.zeros(3))
        self.change_decoder.bias = torch.nn.Parameter(torch.zeros(1))

    def _maps(self, x1, x2):
        d1 = (x1 * self.content_decoder_1.scale * self.encoder.w)[:, :, ::16, ::16]
        d2 = (x2 * self.content_decoder_2.scale * self.encoder.w)[:, :, ::16, ::16]
        return [d1], [d2]

    def forward(self, x1, x2):
        diff = 10.0 * ((x2 - x1) * self.change_decoder.weight[None, :, None, None]).sum(dim=1) + self.change_decoder.bias
        return torch.tanh(diff)

    def forward_with_content(self, x1, x2):
        maps_1, maps_2 = self._maps(x1, x2)
        focus = [
            torch.tanh(1.0 - torch.nn.functional.cosine_similarity(a, b, dim=1)) for a, b in zip(maps_1, maps_2, strict=True)
        ]
        return self.forward(x1, x2), maps_1, maps_2, focus


def _pipeline() -> CFNetChangePipeline:
    return CFNetChangePipeline(model=_StubModel(), device="cpu", weights_dir=Path("unused"), source="stub")


def test_trainable_scopes():
    pipe = _pipeline()
    assert pipe._trainable("change_decoder") == ["change_decoder.bias", "change_decoder.weight"]
    assert pipe._trainable("decoders") == [
        "change_decoder.bias",
        "change_decoder.weight",
        "content_decoder_1.scale",
        "content_decoder_2.scale",
    ]
    assert ADAPTATION_MODES == ("change_decoder", "decoders")
    with pytest.raises(ValueError, match="trainable must be one of"):
        pipe._trainable("everything")


def test_predict_and_evaluate_shapes():
    pipe = _pipeline()
    records = synthetic_records(3)
    result = pipe.predict(records, batch_size=2)
    assert len(result["predictions"]) == 3 and result["predictions"][0]["mask"].shape == (256, 256)
    assert result["predictions"][0]["change_map"].shape == (256, 256) and result["decision_rule"].startswith("changed where")
    assert result["predictions"][0]["changed_fraction"] == 0.0
    report = pipe.evaluate(records)
    assert set(report["model"]) >= {"f1", "iou", "precision", "recall", "accuracy", "confusion"}
    assert report["baseline_unchanged"]["f1"] == 0.0 and report["model"]["f1"] == 0.0
    with pytest.raises(ValueError, match="batch_size"):
        pipe.predict(records, batch_size=0)
    mixed = [
        *records,
        {
            **records[0],
            "id": "big",
            "before": np.tile(records[0]["before"], (2, 2, 1)),
            "after": np.tile(records[0]["after"], (2, 2, 1)),
            "label": np.tile(records[0]["label"], (2, 2)),
        },
    ]
    result = pipe.predict(mixed, batch_size=4)  # sizes are batched separately, order preserved
    assert [p["id"] for p in result["predictions"]] == [r["id"] for r in mixed] and result["predictions"][-1]["mask"].shape == (
        512,
        512,
    )


def test_adapt_trains_only_the_scope_and_keeps_the_best_epoch():
    pipe = _pipeline()
    train, val = synthetic_records(6), synthetic_records(2, seed=50)
    encoder_before = pipe.model.encoder.w.clone()
    before = pipe.evaluate(val)["model"]["f1"]
    seen = []
    result = pipe.adapt(train, val, epochs=3, lr=1e-2, batch_size=2, seed=0, progress=seen.append)
    assert [e["epoch"] for e in seen] == [0, 1, 2, 3] and seen[0]["note"] == "frozen model" and seen[0]["val"]["f1"] == before
    assert result["best_epoch"] == min(range(4), key=lambda i: result["history"][i]["val_loss"])
    assert result["n_trainable"] == 4 and result["n_steps"] == 9 and result["precision"] == "float32"
    assert result["loss"].startswith("upstream loss: MSE on the tanh change map")
    assert torch.equal(pipe.model.encoder.w, encoder_before) and torch.equal(pipe.model.content_decoder_1.scale, torch.ones(1))
    assert pipe.adapter is not None and all(not p.requires_grad for p in pipe.model.parameters())
    assert pipe.evaluate(val)["model"] == result["history"][result["best_epoch"]]["val"]
    assert result["history"][-1]["train_loss"] < result["history"][1]["train_loss"] or result["best_epoch"] >= 1


def test_adapt_is_transactional_when_the_progress_callback_raises():
    pipe = _pipeline()
    initial = {k: v.clone() for k, v in pipe.model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        pipe.adapt(synthetic_records(4), None, epochs=2, lr=1e-2, progress=boom)
    assert pipe.adapter is None
    assert all(torch.equal(initial[k], v) for k, v in pipe.model.state_dict().items())
    assert all(not p.requires_grad for p in pipe.model.parameters())


def test_adapt_refusals():
    pipe = _pipeline()
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(synthetic_records(4), epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(synthetic_records(4), lr=1.0)
    with pytest.raises(ValueError, match="4..2000"):
        pipe.adapt(synthetic_records(3))
    with pytest.raises(ValueError, match="trainable must be one of"):
        pipe.adapt(synthetic_records(4), trainable="all")
    records = synthetic_records(4)
    big = {
        **records[0],
        "id": "big",
        "before": np.tile(records[0]["before"], (2, 2, 1)),
        "after": np.tile(records[0]["after"], (2, 2, 1)),
        "label": np.tile(records[0]["label"], (2, 2)),
    }
    with pytest.raises(ValueError, match="pairs of one size"):
        pipe.adapt([*records, big])
    with pytest.raises(ValueError, match="nothing to save"):
        pipe.save_artifact("unused")


def test_artifact_round_trip_and_refusals(tmp_path):
    pipe = _pipeline()
    records = synthetic_records(4)
    pipe.adapt(records, epochs=1, lr=1e-2, trainable="decoders")
    adapted = pipe.evaluate(records)["model"]
    out = pipe.save_artifact(tmp_path / "adapter", metadata={"tutorial": "test"})
    manifest = json.loads((out / pl.ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["tensors"] == [
        "change_decoder.bias",
        "change_decoder.weight",
        "content_decoder_1.scale",
        "content_decoder_2.scale",
    ]
    assert manifest["adapter"]["trainable"] == "decoders" and manifest["metadata"] == {"tutorial": "test"}
    fresh = _pipeline()
    assert fresh.evaluate(records)["model"] != adapted
    fresh.load_artifact(out)
    assert fresh.evaluate(records)["model"] == adapted and fresh.adapter["best_epoch"] == 1
    narrowed = dict(manifest, adapter={**manifest["adapter"], "trainable": "change_decoder"})
    (out / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(narrowed), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _pipeline().load_artifact(out)
    (out / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    (out / pl.ARTIFACT_WEIGHTS_NAME).write_bytes((out / pl.ARTIFACT_WEIGHTS_NAME).read_bytes() + b"\0")
    with pytest.raises(ValueError, match="digest or size"):
        _pipeline().load_artifact(out)
    assert np.isfinite(adapted["f1"])


# --- the vendored architecture (random weights) ----------------------------------------------------------


def test_vendored_architecture_matches_the_checkpoint_contract():
    pytest.importorskip("torchvision")
    from cfnet_change_detection_pipeline.modeling import check_shapes

    model = build_model()
    assert check_shapes(model) == {"tensors": pl.STATE_TENSORS, "elements": pl.STATE_NUMEL, "parameters": pl.PARAMETER_COUNT}
    state = model.state_dict()
    prefixes = {k.split(".")[0] for k in state}
    assert prefixes == {"encoder", "content_decoder_1", "content_decoder_2", "change_decoder"}
    assert sum(1 for k in state if k.startswith("encoder._backbone.")) == pl.ENCODER_TENSORS
    assert tuple(state["change_decoder.upconv.0.weight"].shape) == (24, 1, 3, 3)
    assert tuple(state["change_decoder.fuseconv3ds.4.conv.0.weight"].shape) == (3, 3, 2, 3, 3)  # present, never called
    assert tuple(state["content_decoder_1.cbam.0.channel_attention.fc1.weight"].shape) == (8, 128, 1, 1)  # present, never applied
    assert not any(k.startswith("focuser.") for k in state)
    with torch.inference_mode():
        x = torch.zeros(1, 3, 256, 256)
        change = model.eval()(x, x)
        assert tuple(change.shape) == (1, 256, 256) and float(change.abs().max()) < 1.0
        _change, maps_1, maps_2, focuses = model.forward_with_content(x, x)
        assert [tuple(m.shape) for m in maps_1] == [(1, 128, 16, 16), (1, 64, 32, 32), (1, 40, 64, 64), (1, 24, 128, 128)]
        assert len(maps_2) == 4 and [tuple(f.shape) for f in focuses] == [(1, 16, 16), (1, 32, 32), (1, 64, 64), (1, 128, 128)]
    with pytest.raises(ValueError, match="multiples of 32"):
        model(torch.zeros(1, 3, 200, 200), torch.zeros(1, 3, 200, 200))
    with pytest.raises(ValueError, match="same shape"):
        model(torch.zeros(1, 3, 256, 256), torch.zeros(1, 3, 224, 224))


def test_normalise_reorders_to_bgr_with_per_date_statistics():
    """Upstream read images with cv2 (BGR) and standardised each date with its own statistics."""
    image = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    image[..., 0] = 255  # pure red in RGB -> the third BGR channel
    before = pl._normalise(image, which="before")
    after = pl._normalise(image, which="after")
    assert before.shape == (1, 3, 4, 4)
    assert np.allclose(before[0, 2], (1.0 - pl.MEANS_BEFORE[2]) / pl.STDS_BEFORE[2], atol=1e-5)
    assert np.allclose(before[0, 0], (0.0 - pl.MEANS_BEFORE[0]) / pl.STDS_BEFORE[0], atol=1e-5)
    assert np.allclose(after[0, 2], (1.0 - pl.MEANS_AFTER[2]) / pl.STDS_AFTER[2], atol=1e-5)
