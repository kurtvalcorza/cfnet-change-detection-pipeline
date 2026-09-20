"""Model-backed smoke on the real converted weights: skipped unless the digest-verified safetensors file is staged
(`weights/cfnet-levir-cd/`). Loads strictly, detects change on synthetic pairs, runs one adaptation epoch on the
change decoder and checks reload parity — evidence that the vendored architecture and the converted file agree,
not a quality claim. Runs on the CPU."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from cfnet_change_detection_pipeline import DEFAULT_WEIGHTS_DIR, PARAMETER_COUNT, CFNetChangePipeline  # noqa: E402
from cfnet_change_detection_pipeline import pipeline as pl  # noqa: E402
from conftest import synthetic_records  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (DEFAULT_WEIGHTS_DIR / pl.CONVERTED_WEIGHTS_NAME).is_file(), reason="converted weights are not staged"
)


def test_converted_weights_load_detect_adapt_and_reload(tmp_path):
    pipe = CFNetChangePipeline.from_pretrained(device="cpu", require_source=False)
    assert sum(p.numel() for p in pipe.model.parameters()) == PARAMETER_COUNT
    records = synthetic_records(4)
    result = pipe.predict(records[:2], batch_size=2)
    masks = [p["mask"] for p in result["predictions"]]
    assert masks[0].shape == (256, 256) and masks[0].dtype == np.uint8 and set(np.unique(masks[0])) <= {0, 1}
    change_map = result["predictions"][0]["change_map"]
    assert float(change_map.min()) > -1.0 and float(change_map.max()) < 1.0
    frozen = pipe.evaluate(records[2:])["model"]
    adapt = pipe.adapt(records, records[2:], epochs=1, lr=1e-5, batch_size=2, trainable="change_decoder")
    assert adapt["n_trainable"] == 852_867 and adapt["n_steps"] == 2 and adapt["history"][0]["val"] == frozen
    out = pipe.save_artifact(tmp_path / "adapter")
    again = CFNetChangePipeline.from_artifact(out, device="cpu", require_source=False)
    assert again.evaluate(records[2:])["model"] == pipe.evaluate(records[2:])["model"]
