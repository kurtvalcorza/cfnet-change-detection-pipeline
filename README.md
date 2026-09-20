# CFNet Change Detection Pipeline

DIMER-oriented pipeline for **CFNet bi-temporal remote-sensing change detection** (`wifibk/CFNet`, the LEVIR-CD checkpoint — Wu, Dong and Meng's content-focuser network trained on 0.5 m Google Earth building-change pairs), pinned to an immutable Hugging Face revision. The repository exposes binary change maps for pairs of co-registered RGB patches, changed-class F1 / IoU / precision / recall against an all-unchanged baseline, a labelled-pair contract with explicit ceilings, a bounded fine-tuning contract for the change decoder with a portable safetensors adapter, a `MODEL_CARD.md` at DIMER Model Card Specification 1.1, and a standalone `E2E` tutorial at DIMER Notebook Specification 2.0.

## Upstream alignment

- Model: `wifibk/CFNet`
- Revision: `c23279428d14186d67ce199b3db358038bf37585`
- Source asset: `levir-cd.pth` (15,805,666 bytes, SHA-256 `22ab286b…`) — a torch pickle of the state dict, converted once to safetensors and never served (see below)
- Upstream code: `wifiBlack/CFNet` at `54acadab23b9d9395ec6814386c2d4a1253eac5a` (`model/*.py`), vendored in plain PyTorch as `src/cfnet_change_detection_pipeline/modeling.py`; the encoder backbone is torchvision's EfficientNet-B5 stem and first four stages, built with `weights=None`
- Upstream weight license: Apache-2.0
- Upstream task: bi-temporal building-change detection — a shared EfficientNet-B5 encoder, a content decoder per date, a cosine-distance focuser and a 3-D-convolution change decoder, trained on LEVIR-CD with a change loss plus content-consistency terms (Wu et al., 2025; F1 92.18 / IoU 85.49 on the LEVIR-CD test split)
- Runtime: `torch==2.14.0` + `torchvision==0.29.0` + `numpy` + `pillow` + `safetensors` + `huggingface-hub` — **no Hub-hosted code, no served pickle**; runs on a CPU
- Repository adaptation: **E2E** (bounded fine-tuning of the change decoder — optionally the content decoders — on labelled pairs with the upstream loss, with a portable safetensors adapter)

## Three things to know before you start

**The checkpoint is a pickle, and the pipeline converts it once.** The upstream file is a torch zip archive whose pickle references only the four state-dict globals. `audit_pickle` lists them with `pickletools` (no execution) and refuses anything else; `convert_model` unpickles the file once through `torch.load(weights_only=True)`, loads the 776 tensors strictly into the vendored network and writes `cfnet-levir-cd.safetensors` (15,598,980 bytes, SHA-256 `b348186e…`), the only file the model is ever loaded from.

**The two dates are not interchangeable, and the channels are BGR inside.** The upstream loader read images with `cv2` (BGR), divided by 255 and standardised each date with its own LEVIR-CD statistics; the pipeline reproduces that exactly (`_normalise`), so `before` must be the earlier image and `after` the later. The output is the network's `tanh` change map in (−1, 1), thresholded at 0.5 as upstream does.

**The tutorial data are academic-use only, and the model trained on them.** LEVIR-CD "can only be used for academic purposes" and its imagery is subject to Google Earth's terms; the tutorial fetches the CFNet authors' Hub mirror at run time and this repository redistributes nothing. Roles are the dataset's own splits — the checkpoint was trained on the training split — so the frozen model's held-out F1 of 0.9352 / IoU 0.8782 on 24 test crops (against 0 / 0 for the all-unchanged baseline) is what a model fitted to this dataset scores on crops chosen for having change, and the bounded adaptation leaves it where it was. That is the contract demonstrated, not a claim that the sample improves the model.

## Quick start

```python
from cfnet_change_detection_pipeline import CFNetChangePipeline, fetch_sample_dataset

pipe = CFNetChangePipeline.from_pretrained(allow_download=True)  # stages + verifies the snapshot, audits + converts the pickle once, loads safetensors
splits = fetch_sample_dataset()                                    # 32 / 8 / 24 labelled crops extracted from the pinned LEVIR-CD tarball (3.8 GB)
print(pipe.evaluate(splits["test"])["model"]["f1"])                # frozen model (the all-unchanged baseline is under ["baseline_unchanged"])
pipe.adapt(splits["train"], splits["validation"], epochs=4, lr=1e-5)  # bounded fine-tuning of the change decoder, epoch selected by validation loss
print(pipe.evaluate(splits["test"])["model"]["f1"])                # adapted model, same crops
maps = pipe.predict([{"id": "mine", "before": "t1.png", "after": "t2.png"}])["predictions"]  # mask (H, W) uint8 + change_map (H, W) float32
pipe.save_artifact("outputs/adapter")
```

`predict()` takes records `{id, before, after}` with two (H, W, 3) uint8 RGB arrays of the same size (sides in [64, 2048], multiples of 32) or PNG / JPEG paths; `evaluate()` and `adapt()` take `{id, before, after, label}` records with an (H, W) label of 0 / 1 and −1 for no data (or a PNG path with 0 / 255). Validation is structural: nothing checks that the two images show the same place, that they are co-registered, or that a label belongs to its pair.

## Weights layout

```
weights/cfnet-levir-cd/   levir-cd.pth                 (git-ignored, the pinned source)
                          README.md                    (the upstream card)
                          dimer-base-manifest.json
                          cfnet-levir-cd.safetensors   (git-ignored, converted)
weights/levir-cd/         LEVIR-CD-processed.tar.gz + crops/   (git-ignored, the pinned dataset tarball and the 192 extracted members)
```

`from_pretrained()` calls `stage_missing_files()` (fetches only absent manifest entries, only at the pinned revision, only with `allow_download=True`) then `verify_snapshot()` (byte size + SHA-256 of every manifest entry and of the converted file when present), converts the pickle when the safetensors file is absent, and refuses on the first mismatch. With `require_source=False` the digest-verified converted file is accepted without the checkpoint — the DIMER-hosted shape. `docs/WEIGHTS.md` records the provenance, the audit, the conversion, the vendored code, the data pins and terms, and the DIMER hosting notes.

## Sample data

`fetch_sample_dataset()` fetches `LEVIR-CD-processed.tar.gz` (3.8 GB) from the authors' Hub mirror at an immutable revision, verifies it by size and SHA-256, streams through it once and extracts exactly the 192 pinned members (each verified again) into `weights/levir-cd/crops/`, then reads them as `{id, before, after, label}` records with roles from the dataset's own splits (32 train / 8 validation / 24 test crops, one per source pair, each with at least 3 % change). `dataset_manifest` validates the splits, refuses a crop or a source pair in two splits and records a digest. Nothing is vendored under `weights/`; the data terms are academic use only.

## Adapter artifacts

`save_artifact(dir)` writes `adapter.safetensors` (the trained tensors — the change decoder, about 3.4 MB; with `trainable="decoders"` also the two content decoders, about 7 MB) and a `manifest.json` recording the artifact format, the exact base model id and revision, the converted-base digest, the adaptation scope, the tensor names, the file size and SHA-256, the training configuration and the epoch history. `CFNetChangePipeline.from_artifact(dir)` re-verifies the base file, checks the manifest, scope and digest before deserialising, rebuilds the network and overlays the tensors.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline and run on a CPU in seconds: crafted pickles, temporary manifests, synthetic pairs, a synthetic tarball with a decoy member, a stub model with the vendored parameter layout, and the vendored network at random initialisation (shapes, keys, the unused CBAM and fusion blocks, the BGR / per-date normalisation), never the weights; `tests/test_model_backed.py` runs the real converted weights when they are staged (strict load, detection, one adaptation epoch, reload parity) and skips otherwise. The model-backed smoke and the CPU sweep are recorded in `MODEL_CARD.md` (*Runtime*).

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/cfnet-change-detection-pipeline/blob/main/tutorials/cfnet_change_detection_colab.ipynb)

`tutorials/cfnet_change_detection_colab.ipynb` is declared `E2E` and is **standalone** (DIMER Notebook Specification 2.0 §4): it is generated by `tools/build_notebook.py` from `tools/notebook_template.py` and embeds the 4 package modules (`modeling.py`, `metrics.py`, `pipeline.py`, `samples.py`) verbatim in dependency order, the pinned model identity, the snapshot manifest and the exact runtime pins, so the exported `.ipynb` keeps working without this repository being reachable. It runs on a GPU or a CPU. It converts the pinned checkpoint in the runtime (the audit and conversion records are printed before the model loads), extracts the pinned crops from the digest-verified tarball, and runs the sample path: validation, the frozen model against the all-unchanged baseline, bounded fine-tuning of the change decoder, held-out evaluation, change maps, adapter export and reload parity. Do not edit the notebook by hand; regenerate it (`python tools/build_notebook.py`; `--check` is enforced by the validator and CI).

## Release status

**Candidate** — the `E2E` notebook has not yet executed in a supported runtime; by the maintainer's rule no local GPU pre-flight is run for this row, so the clean-runtime Kaggle execution is its first execution and the evidence that promotes it, to be recorded in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks and the CPU model-backed test — are necessary but are not the evidence; the hosted run is.

## Licensing

- Upstream weights: Apache-2.0 (`wifibk/CFNet`), staged from the pinned Hub revision and converted, not modified, into the served safetensors.
- Upstream code: Apache-2.0 (`wifiBlack/CFNet`), vendored as `modeling.py` with the commit recorded in `docs/WEIGHTS.md`; torchvision (BSD-3) provides the EfficientNet-B5 class.
- Tutorial data: LEVIR-CD is for **academic purposes only** (commercial use prohibited; Google Earth terms apply); the tutorial fetches the CFNet authors' mirror at run time and this repository redistributes no image.
- This repository's code and documentation: Apache-2.0 (`LICENSE`).
- The upstream licence governs your use of the weights, including commercial use and redistribution; this repository grants no rights beyond it, and none over the data.

## AI Assistance Disclosure

This repository's code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
