---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: image-segmentation
task: "Segmentation - Satellite Change Detection (bi-temporal, binary building change)"
base_model: wifibk/CFNet
date_published: "2025-03-11"
date_published_source: "Hugging Face Hub commit `8a359cbd` (\"Initial commit of my model\", 2025-03-11) that published `levir-cd.pth`, the same day as the arXiv preprint (2503.08505) and the code release; the pinned revision `c2327942…` (2025-03-16, \"Add figures\") carries the identical LFS object (SHA-256 `22ab286b…`). The previously recorded `2025-03-11` names the same release."
---

# CFNet — Bi-Temporal Building-Change Detection (LEVIR-CD Checkpoint & Bounded Change-Decoder Fine-Tuning)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-wifibk%2FCFNet-ffcc4d?style=flat)](https://huggingface.co/wifibk/CFNet)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-wifiBlack%2FCFNet-181717?style=flat&logo=github&logoColor=white)](https://github.com/wifiBlack/CFNet)
[![Paper](https://img.shields.io/badge/arXiv-2503.08505-b31b1b.svg)](https://arxiv.org/abs/2503.08505)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

> [!IMPORTANT]
> **The upstream asset is a pickle, the network is vendored, and the tutorial data are academic-use only.** `levir-cd.pth` is a torch zip archive whose pickle references only `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2` and two storage classes. `src/cfnet_change_detection_pipeline/pipeline.py` statically audits it, unpickles it **once** through `torch.load(weights_only=True)`, loads the 776 tensors strictly into the network vendored in `modeling.py` (plain PyTorch; the EfficientNet-B5 stem and first four stages come from torchvision built with `weights=None`), and serves only the resulting 15.6 MB safetensors, whose digest is pinned. The tutorial's labelled pairs are LEVIR-CD crops fetched at run time from the authors' Hub mirror: LEVIR-CD "can only be used for academic purposes, but [is] prohibited for any commercial use", and its imagery is subject to Google Earth's terms — this repository redistributes none of it. On 24 held-out test crops the frozen model reaches a changed-class F1 of 0.9352 / IoU 0.8782 against an all-unchanged baseline that scores 0 (accuracy 0.8615) — sample-sanity evidence on crops chosen for having change, not the benchmark (the authors report F1 92.18 / IoU 85.49 on the full test split). The bounded adaptation is a fine-tune of the change decoder (852,867 parameters) with the upstream loss, selected by validation loss; since the model trained on this dataset, it leaves the held-out numbers where they were.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — stage and digest-verify the pickled checkpoint, audit and convert it into safetensors, extract and validate pinned labelled pairs from a digest-verified tarball, score the all-unchanged baseline and the frozen model, fine-tune the change decoder, evaluate on held-out pairs, write change maps, and export and reload the adapter:

- **End-to-End Change-Detection Fine-Tuning Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/cfnet-change-detection-pipeline/blob/main/tutorials/cfnet_change_detection_colab.ipynb) [`cfnet_change_detection_colab.ipynb`](https://github.com/kurtvalcorza/cfnet-change-detection-pipeline/blob/main/tutorials/cfnet_change_detection_colab.ipynb)  
  *End-to-end use of the CFNet LEVIR-CD checkpoint: the pickle audited and converted once, 64 labelled 256 × 256 bi-temporal crops extracted from the digest-pinned LEVIR-CD tarball with the dataset's own splits, structural validation with refusal probes, the all-unchanged baseline, the frozen model's F1 / IoU / precision / recall, a bounded fine-tuning of the change decoder with the upstream loss and validation-loss epoch selection, a paired comparison on held-out crops, change maps of two crops, and safetensors adapter export with verified reload parity.*

---

#### Description

`wifibk/CFNet` at revision `c23279428d14186d67ce199b3db358038bf37585` is the checkpoint release of CFNet — the Content Focuser Network for bi-temporal remote-sensing change detection of Wu, Dong and Meng (2025) — and the file packaged here, `levir-cd.pth` (15,805,666 bytes, SHA-256 `22ab286b…`), is the authors' model trained on LEVIR-CD, the 0.5 m Google Earth building-change dataset of Chen and Shi (2020): 637 pairs of 1024 × 1024 patches of Texas cities taken 5–14 years apart, with 31,333 hand-drawn building-change instances. The network is small — 3,838,563 parameters in 776 tensors — and is built from four parts: a shared encoder (the stem and first four stages of torchvision's EfficientNet-B5, 2.09 M parameters, applied to both dates; the outputs at 1/2, 1/4, 1/8 and 1/16 resolution with 24, 40, 64 and 128 channels are kept), one content decoder per date (a top-down path of three aggregation blocks — transposed-convolution upsampling, 1 × 1 fusion, two residual blocks — that produces four content maps per date), a parameter-free focuser (the cosine distance between the two dates' content maps through `tanh`, per scale, is the change-focus weight), and a change decoder (per scale a (2, 3, 3) 3-D convolution fuses the dates, the focus weight scales the result, the scales are aggregated top-down, and one stride-2 transposed convolution with `tanh` decodes the change map at the input resolution). The authors trained it with a mean-squared-error change loss plus content-consistency terms that make the content maps insensitive to acquisition style, and report F1 92.18 % / IoU 85.49 % on the LEVIR-CD test split (F1 81.41 % on CLCD and 82.89 % on SYSU-CD with the sibling checkpoints, which this repository does not package).

What this repository adds is the `CFNetChangePipeline` class in `src/cfnet_change_detection_pipeline/pipeline.py`: manifest verification of the 2-file Hub snapshot before any model library is imported (`verify_snapshot`, which also checks the converted file against its pinned digest when present), fresh-clone staging at the pinned revision (`stage_missing_files`), a **static audit of the pickle** that lists every global it would import with `pickletools` and refuses anything outside the allow-list (`audit_pickle`, digest pinned), a **one-time conversion** through torch's weights-only unpickler into a strictly loaded model whose state dict is saved as safetensors (`convert_model`), a labelled-pair contract with explicit ceilings and refusal probes (`validate_dataset`, `validate_inputs`), the upstream preprocessing reproduced exactly (`cv2`'s BGR channel order and per-date standardisation), pixel-level change metrics scored as upstream scores them against an all-unchanged baseline (`evaluate`), a bounded fine-tuning contract for the change decoder — optionally the content decoders too — with the upstream loss and validation-loss epoch selection (`adapt`), and a portable safetensors adapter with a verified manifest (`save_artifact`, `from_artifact`). `samples.py` pins 192 members of the authors' LEVIR-CD mirror by path, size and SHA-256 and extracts exactly those; `modeling.py` is the network, vendored from `wifiBlack/CFNet` at commit `54acadab23b9d9395ec6814386c2d4a1253eac5a` without its training scaffolding.

#### Intended Use and Limitations

###### Primary Intended Uses

Binary building-change maps for pairs of co-registered RGB patches at about 0.5 m resolution (sides multiples of 32, 64–2048 pixels), evaluation of those maps on labelled pairs against an all-unchanged baseline, and bounded fine-tuning of the change decoder to a user's labelled pairs with a portable adapter — as a tutorial and evaluation contract for a DIMER model profile, and as the runtime that profile would serve. The default tutorial path downloads the checkpoint from the Hub and the LEVIR-CD tarball from the authors' mirror at immutable revisions, for academic use, and nothing else.

###### Primary Intended Users

Researchers, students and engineers evaluating bi-temporal change detection on very-high-resolution imagery; Operators publishing the model profile; Maintainers who need a reference for a vendored network with a torchvision backbone and for tutorial data under academic-only terms.

###### Out-of-scope use cases

Any commercial use of the LEVIR-CD data or of maps derived from them (the dataset's terms forbid it; the model weights are Apache-2.0, the data are not); operational damage assessment, insurance, planning-enforcement or regulatory decisions taken from these maps without independent validation; pairs that are not co-registered, that mix sensors or seasons in ways the training data did not, or whose resolution is far from 0.5 m; change other than buildings (the training labels are buildings only); multi-class or multi-date change; images larger than a patch without tiling by the caller; any claim that a 64-crop sample stands in for an evaluation.

#### Factors

###### Groups

Performance varies with building size and density (small or partial buildings at crop edges are the usual misses), with the magnitude of style change between the dates (season, illumination, sensor — what the content-aware training targets), with the change fraction of a crop (the full LEVIR-CD test split is more than half change-free, which lowers the pooled F1 relative to the tutorial's change-rich sample), and with geography: the training data are suburban Texas.

###### Instrumentation

Very-high-resolution (about 0.5 m) three-band RGB imagery from Google Earth, as 8-bit PNG or JPEG, two co-registered dates of the same footprint. The pipeline reorders RGB to BGR and standardises each date with the LEVIR-CD channel statistics the upstream loader used (`before` with the first date's statistics, `after` with the second's), so the two dates are not interchangeable. Other sensors, resolutions or band orders are outside the contract.

###### Environment

The pipeline runs on a CUDA GPU in float16 autocast or on the CPU in float32; the network is small enough that a 256 × 256 pair takes well under a second on a CPU and the default adaptation about a minute. No network access is needed after the snapshot and the tarball are staged.

#### Metrics

###### Performance Measures

F1, IoU, precision and recall of the changed class over the labelled pixels of the scored pairs, pooled into one confusion matrix with the upstream rule (`map > 0.5`) — the four numbers the upstream evaluation reports — plus the overall pixel accuracy and the change fractions of the labels and of the predictions, with the all-unchanged baseline scored on the same pixels. Adaptation is judged by the same numbers on held-out pairs beside the frozen model and by the validation loss, which selects the epoch. Every number in this card is tutorial sample-sanity evidence from 24 test crops.

###### Decision thresholds

The mask is the `tanh` change map thresholded at 0.5, the upstream rule; the map is the network's output, not a calibrated probability. Any other threshold, minimum-area filter or post-processing is the deployment's to set and to validate.

###### Approaches to uncertainty and variability

No dispersion estimate: one seeded run, 24 test crops chosen for having at least 3 % change, pixel-pooled metrics that let large buildings dominate. The upstream loss's content terms sample random pixel pairs, so the reported losses depend on the seed even for the frozen model (the pipeline seeds them). Adaptation results also depend on the shuffling and flips, the batch size, the learning rate, the epoch count and the validation set that selects the epoch; the tutorial fixes all of them and records the per-epoch history.

#### Ethical considerations and biases

###### Data

The tutorial's labelled data are 64 crops of LEVIR-CD: Google Earth imagery of Texas suburbs with hand-drawn building-change labels — no personal data, no imagery of people, but imagery under Google Earth's terms and a dataset under academic-only terms, which the card, the README and the notebook state. The training data are the authors'; their coverage (Texas cities, 2002–2018, buildings) bounds where the model is meaningful. Users' own pairs may be commercial imagery under licence; the tutorial says so and uploads nothing by default.

###### Human Life

Change maps do not act on people directly, but building-change detection feeds decisions — disaster damage assessment, planning enforcement, insurance — that affect residents and owners; a detector validated on suburban Texas buildings is not a basis for such decisions elsewhere without validation against ground truth, and this card claims none.

###### Mitigations

The snapshot is pinned to an immutable revision and verified by size and SHA-256 before any model library is imported. The pickle is statically audited against an allow-list with a pinned digest, unpickled once through torch's weights-only loader, and converted into safetensors whose digest is pinned; the network is vendored, and the only third-party model code is torchvision's EfficientNet-B5 from PyPI at a pinned version, built without weights. Inputs are validated structurally before inference (shape, dtype range, equal sizes, side multiples, label values) and every refusal names its rule. Adaptation is bounded to declared scopes, keeps BatchNorm statistics frozen, is transactional on failure, and its artifact is verified — format, base identity, converted-base digest, scope, tensor names, size and SHA-256 — before deserialising. The tarball is pinned and streamed without `extractall`; members are written under flat names, never at archive paths. The tutorial states the all-unchanged baseline beside every model number and the data terms beside the data.

###### Risks and harms

A swapped date order, uncalibrated imagery or a different resolution produces confident, plausible-looking, wrong maps. Building change in other regions, denser cities or other sensors will score lower than the tutorial's numbers suggest. The change-rich sample overstates F1 relative to a realistic change-free majority of scenes. Using the LEVIR-CD data or derived maps commercially breaches the dataset's terms. The head-only adaptation cannot correct a domain shift in the encoder's features.

###### Use cases

Tutorial and evaluation of CFNet building-change detection on LEVIR-CD-like pairs; a DIMER model profile serving the converted checkpoint; a reference for vendoring a torchvision-backboned network and for handling tutorial data under academic-only terms; a starting point for fine-tuning the change decoder on a user's labelled pairs, with the caveats above.

## Immutable provenance

- Model: `wifibk/CFNet`
- Revision: `c23279428d14186d67ce199b3db358038bf37585`
- Manifest: `weights/cfnet-levir-cd/dimer-base-manifest.json`, format `dimer_hf_snapshot` v1, 2 files, `totalBytes` 15808640
- Source asset `levir-cd.pth` (15,805,666 bytes) SHA-256: `22ab286b2138ab1082e04290b27cbff5abd3802375a6b2f01d4ba78ca8b0c1aa` (first published in Hub commit `8a359cbd` on 2025-03-11). Torch zip archive (`archive/data.pkl`) holding a plain state dict of 776 tensors (668 float32, 108 int64 `num_batches_tracked`) under the module names `encoder._backbone.*` (428), `content_decoder_1.*` (108), `content_decoder_2.*` (108) and `change_decoder.*` (132); no optimizer, no metadata. **Never loaded by the runtime path.**
- Static audit (`audit_pickle`): globals `collections.OrderedDict`, `torch.FloatStorage`, `torch.LongStorage`, `torch._utils._rebuild_tensor_v2`; audit SHA-256 `5b9f0ba08490293d6c17b9cef219991e1a6edda31609429679f8dca1af5a7b10`; 0 violations.
- Converted serving file (asset spec §11.2, `derived_from_sha256` = the source digest above; git-ignored, regenerated deterministically by `convert_model`): `cfnet-levir-cd.safetensors` (15,598,980 bytes, 776 tensors, 3,877,781 elements of which 3,838,563 are parameters) SHA-256 `b348186e5003a7447ab5eb5874204c50e63ad71aa61aca3ff0a80746116963d9`
- Also staged: upstream `README.md` (2,974 bytes); not staged: `clcd.pth`, `sysu-cd.pth` (the sibling checkpoints) and the five figures.
- Vendored network: `src/cfnet_change_detection_pipeline/modeling.py`, rewritten from `model/{CFNet,encoder,content_decoder,change_decoder,focuser,utils}.py` of `wifiBlack/CFNet` at commit `54acadab23b9d9395ec6814386c2d4a1253eac5a` (Apache-2.0); the encoder is `torchvision.models.efficientnet_b5(weights=None).features[0:5]` from `torchvision==0.29.0`. Two upstream quirks are reproduced as they are: the content decoders' CBAM modules and the change decoder's fifth fusion block carry weights but are never used by the forward pass.
- Tutorial data: `LEVIR-CD-processed.tar.gz` of the Hugging Face dataset `wifibk/CFNet_Datasets` at revision `ba68aa9a54ae15fe32ce9b02c380eb384fae528e` (3,831,872,824 bytes, SHA-256 `6515dd451c159b9ed5bd53b3fb6e15188dd114b296ff9169ca21fbcabcd1d109`); 192 pinned members (64 crops × before / after / label PNGs, 16,921,921 bytes in total), each pinned by member path, byte size and SHA-256 in `samples.py`, with roles from the dataset's own splits (32 train, 8 validation, 24 test; one crop per source pair, drawn with a fixed seed on 2026-09-20 from the crops with a clean 0 / 255 mask and at least 3 % change). LEVIR-CD terms: academic purposes only, commercial use prohibited, Google Earth terms apply.
- Upstream reference: https://huggingface.co/wifibk/CFNet · https://github.com/wifiBlack/CFNet · https://huggingface.co/datasets/wifibk/CFNet_Datasets · https://justchenhao.github.io/LEVIR/

## Input/output contract

- `CFNetChangePipeline.from_pretrained(device=None, weights_dir=None, allow_download=False, require_source=True, report=None)`: stages and verifies the snapshot, audits and converts the checkpoint when the safetensors file is absent (reporting through `report`), builds the vendored network and loads the safetensors strictly. `require_source=False` accepts the digest-verified converted file without the checkpoint.
- `predict(records, *, batch_size=4) -> dict` with `model` (id, revision, key, `adapted`), `classes`, `decision_rule`, `predictions` (per record: `id`, `mask` (H, W) uint8 with 1 = changed, `change_map` (H, W) float32 in (−1, 1), `changed_fraction`), `seconds`. Records of different sizes are batched separately; the output order follows the input.
- `evaluate(records, *, batch_size=4) -> dict`: `n_records`, `metric`, `model` (`f1`, `iou`, `precision`, `recall`, `accuracy`, `change_fraction_label`, `change_fraction_predicted`, `confusion`, `pixels`), `baseline_unchanged` (the same fields plus `note`), `adapted`, `seconds`.
- `adapt(train, val=None, *, epochs=6, lr=1e-4, batch_size=4, trainable="change_decoder", seed=0, progress=None) -> dict`: bounded AdamW fine-tuning with the upstream loss (MSE on the change map + 0.1 × the content-consistency terms); returns the trainable set and counts, hyperparameters, `best_epoch`, per-epoch `history` (epoch 0 = the frozen model, each with `train_loss`, `val_loss` and the validation metrics) and `seconds`. `trainable="decoders"` also unfreezes the two content decoders. Pairs of one size per call. The tutorial uses `epochs=4, lr=1e-5`.
- `save_artifact(output_dir, metadata=None) -> Path` writes `adapter.safetensors` (the trained tensors) + `manifest.json` (format, base model id/revision/key and converted digest, adaptation record, history, tensor names, file size and SHA-256); `from_artifact(...)` / `load_artifact(...)` verify all of that before deserialising.
- `validate_inputs(record) -> dict` (`id`, `shape`, `has_label`, `change_fraction`, `ignored_pixels`) and `validate_dataset(records, *, min_records=4, max_records=2000, require_labels=True) -> dict` (normalised `records`, counts, `sizes`, `change_fraction`, `ignored_pixels`, `digest`) raise exactly what the execution path would raise; `check_record` returns one normalised record. `read_image(path)` reads a PNG / JPEG as uint8 RGB; `read_mask(path)` reads a 0 / 255 label as 0 / 1; `fetch_sample_dataset(cache_dir=None)` / `fetch_corpus` / `fetch_tarball` / `extract_pinned_members` are the pinned-data primitives; `load_byod_dataset(path)`, `write_sample_pair`, `write_dataset_csv` the BYOD I/O; `split_dataset`, `check_split_disjoint`, `dataset_manifest` the split helpers; `change_metrics`, `confusion_counts`, `metrics_from_counts`, `unchanged_baseline` the metrics.
- `audit_pickle(path, allowed=…)`, `convert_model(path=None)`, `verify_converted(path=None)` and `build_model()` are the serialization primitives.
- Constants: `MEANS_BEFORE`, `STDS_BEFORE`, `MEANS_AFTER`, `STDS_AFTER` (BGR order), `CHANGE_THRESHOLD = 0.5`, `MIN_SIDE = 64`, `MAX_SIDE = 2048`, `SIDE_MULTIPLE = 32`, `SAMPLE_SIZE = 256`, `NUM_CLASSES = 2`, `CLASS_NAMES = ("unchanged", "changed")`, `IGNORE_INDEX = -1`, `MIN_RECORDS = 4`, `MAX_RECORDS = 2000`, `PARAMETER_COUNT = 3838563`, `STATE_TENSORS = 776`, `ENCODER_TENSORS = 428`, `ADAPTATION_MODES = ("change_decoder", "decoders")`, `CONTENT_LOSS_WEIGHT = 0.1`, `ARTIFACT_FORMAT = "org.valcorza.cfnet-change-detection.adapter.v1"`, `TAR_SHA256`, `DATASET_REVISION`, `UPSTREAM_CODE_COMMIT`.

## DIMER deployment notes

| Field | Status |
|---|---|
| **DIMER status** | **Planned / conditional** — the `.pth` asset-format and deserialization-trust review DIMER requires is what this repository implements; the review's acceptance is the maintainer's call |
| Licence | Apache-2.0 (weights and the upstream `wifiBlack/CFNet` code; torchvision BSD-3; this repository's code Apache-2.0) — use, modification, redistribution and commercial use of the **model** permitted with the licence and notices preserved. The tutorial **data** (LEVIR-CD) are academic-use only and are never redistributed |
| Weights | Would be redistributed converted, not unmodified: the served artifact is the deterministic safetensors derived from the pinned checkpoint, with both identities recorded (asset spec §11.2); this repository redistributes neither |
| Remote code | **Not required** — no Hub-hosted module is imported; the network is `modeling.py` in this repository plus torchvision's EfficientNet-B5 class from PyPI |
| Executable serialization | One pickle, unpickled **once** at conversion through torch's weights-only loader after a digest check and a static audit; a DIMER profile should carry the safetensors file and never the `.pth` |
| Runtime | `torch==2.14.0` + `torchvision==0.29.0` + `numpy` + `pillow` + `safetensors` + `huggingface-hub`; float16 autocast on CUDA; a CPU serves a 256 × 256 pair in well under a second |
| Upload format | `cfnet-levir-cd.safetensors` (15,598,980 bytes, SHA-256 `b348186e…`); **the `.pth` file must not be uploaded** |
| Input contract | two co-registered 8-bit RGB images of the same size (sides in [64, 2048], multiples of 32) as arrays or PNG / JPEG; the earlier date as `before`, the later as `after`; labels 0 / 1 / −1 for adaptation (files: 0 / 255) |
| Sample data | the CFNet authors' LEVIR-CD mirror (academic use only) fetched at run time from the Hub at an immutable revision, 192 pinned members extracted, never vendored |

**One thing is open, and it is neither the licence nor the code:** whether a one-time unpickle through torch's weights-only loader, after a static audit with a pinned digest — in the build and in the tutorial runtime, where the notebook converts what it downloads — meets the bar for redistribution, or whether only the safetensors converted and verified once by the maintainer should be published. The served artifact is the same file either way. A second, data-side point for the profile: a DIMER tutorial that fetches LEVIR-CD is bound by its academic-only terms; the profile's own inference needs no LEVIR-CD data.

## Runtime

- Pins (`pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `numpy==2.5.3`, `pillow==11.3.0`, `safetensors==0.8.0`, `huggingface-hub==1.32.0`; dev `pytest==8.4.2`, `ruff==0.16.6`. Python 3.12; every local run of this profile was **CPU-only** (Windows venv, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`) — no local GPU is used for this profile by the maintainer's rule; the GPU execution is the clean-runtime Kaggle T4 run recorded below and in `docs/release-verification.md`.
- Executed 2026-09-20: `python -m pytest -q -o addopts= tests` — 38 passed in about 10 s on the CPU (33 offline including the stub-model adaptation, the synthetic-tarball tests, the randomly initialised vendored network and the BGR / per-date normalisation check, + 5 notebook-parity tests; the model-backed smoke on the converted weights included), exit 0; `ruff check src tests tools` clean; `tools/validate_release_assets.py` PASS.
- Executed 2026-09-20, build-time conversion (CPU, 4.4 s): the static audit found exactly the four state-dict globals; `torch.load(weights_only=True)` returned a plain state dict of 776 tensors with no `module.` prefix; `load_state_dict(strict=True)` on the vendored network matched every key with 0 missing, 0 unexpected and 0 shape mismatches; the safetensors reproduced the pinned digest on two consecutive conversions. Vendoring check: the network built with `torchvision.models.efficientnet_b5(weights=None).features[0:5]` has exactly the checkpoint's 428 encoder tensors; identical dates give a change map with maximum 0.023, far below the 0.5 threshold.
- Executed 2026-09-20, dataset pinning: the 3.83 GB tarball was downloaded, its digest checked against the Hub LFS pointer before use, and indexed once on the CPU in 30 s (47,781 file members: 11,125 / 1,600 / 3,200 crop triples in `train` / `val` / `test`, three list files, the authors' `1to255.py` and two stray PNGs at the root; labels uint8 0 / 255 with 24 test crops carrying an intermediate value 156 or 254; 1,716 of the 3,200 test crops have no change, mean change fraction 5.2 %); 64 crops were drawn with a fixed seed from the clean-label crops with ≥ 3 % change, one per source pair (355 / 48 / 103 candidate pairs).
- Executed 2026-09-20 on the CPU, the package smoke (`smoke.py`): load 3.8 s; the 192 pinned members streamed out of the tarball in 16.9 s (change fractions train 0.103, validation 0.128, test 0.139); 24 test crops scored in 1.5 s — **all-unchanged baseline** accuracy 0.8615, F1 0, IoU 0; **frozen model** F1 0.9352, IoU 0.8782, precision 0.9424, recall 0.9281, accuracy 0.9822; validation F1 0.9152, IoU 0.8437. Adaptation sweep of the change decoder (852,867 of 3,838,563 parameters, batch 4, flips, BatchNorm frozen; validation loss = the upstream loss with seeded content terms): lr 10⁻⁵ × 6 epochs lowered the validation loss 0.1162 → 0.1141 at epoch 2 (test F1 0.9352 → 0.9352, IoU 0.8782 → 0.8783; 22 s); lr 10⁻⁴ × 4 → 0.1156 at epoch 2 (test F1 0.9343; 15 s); lr 10⁻³ × 6 → 0.1161 at epoch 2 (test F1 0.9333, validation F1 dipped to 0.889 at epoch 3); `decoders` (1,746,995 parameters) at lr 10⁻⁴ × 6 → 0.1154 at epoch 2 (test F1 0.9344; 35 s). Default chosen: `change_decoder`, lr 10⁻⁵, 4 epochs. Adapter 3,419,684 bytes (132 tensors); reload parity exact.
- Not executed locally: the notebook itself (no local pre-flight by rule — the Kaggle run is the first execution), any GPU path, the `allow_download=True` path through the pipeline (the Hub downloads were exercised by the build with `huggingface_hub`), any published benchmark, and any measurement of latency or memory beyond the times above.
- Executed 2026-09-20, the **committed notebook blob** (`b92e01f` / `9e515a82`) run top-to-bottom on a clean Kaggle Tesla T4 kernel (`kurtvalcorza/dimer-nb2-cfnet-change-detection` v1, `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `pillow 11.3.0`, `numpy 2.5.3`, Python 3.12.13, `cuda`, empty Hugging Face cache, no repository checkout, blob SHA-1 verified against GitHub before execution): 11/11 ok (1 restart after install cell), 271.0 s, 204 files, 3880 MB fetched and digest-verified inside the notebook, conversion performed in the notebook; comparison test changed-class F1 / IoU (all-unchanged baseline 0 / 0): frozen 0.9352 / 0.8782 → adapted 0.9352 / 0.8783, precision 0.9423 → 0.9424, recall 0.9281 → 0.9282, accuracy 0.9822 → 0.9822 vs baseline 0.8615 (best epoch 2, validation loss 0.1162 → 0.1141, validation F1 0.9151 → 0.9137); reload parity f1_diff: 0.0, metrics_identical: True, max_abs_map_diff: 0.0. Recorded in `docs/release-verification.md`.
- Not executed: the CLCD and SYSU-CD checkpoints of the upstream repository.

## References

- Wu, F., Dong, S., Meng, X. (2025). CFNet: Optimizing remote sensing change detection through content-aware enhancement. arXiv:2503.08505. https://arxiv.org/abs/2503.08505
- Upstream code: https://github.com/wifiBlack/CFNet; pinned Hub repository: https://huggingface.co/wifibk/CFNet; the authors' dataset mirror: https://huggingface.co/datasets/wifibk/CFNet_Datasets
- Chen, H., Shi, Z. (2020). A spatial-temporal attention-based method and a new dataset for remote sensing image change detection. Remote Sensing, 12(10), 1662. LEVIR-CD (academic use only): https://justchenhao.github.io/LEVIR/
- Tan, M., Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML (the encoder backbone, from torchvision)
