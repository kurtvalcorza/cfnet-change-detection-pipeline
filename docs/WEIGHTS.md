# Weight provenance, the pickle audit, the conversion, the vendored network, the pinned dataset tarball and DIMER hosting

This repository pins **one** model snapshot with its own `dimer-base-manifest.json` and **one** dataset tarball. The checkpoint is a pickle, which this pipeline audits and converts but never serves; the network is vendored in plain PyTorch with a torchvision backbone; the tarball is streamed once for exactly the pinned members and never `extractall`-ed; the data are under academic-only terms and are never redistributed.

## CFNet LEVIR-CD weights

- Upstream: `wifibk/CFNet`
- Immutable revision: `c23279428d14186d67ce199b3db358038bf37585` (2025-03-16, "Add figures", the repository head at pinning time); the checkpoint was first published in commit `8a359cbd` (2025-03-11, "Initial commit of my model"); the LFS object is identical at both revisions.
- Source format: `levir-cd.pth` — torch zip archive (`archive/data.pkl`) holding a plain `state_dict` of 776 tensors (668 float32, 108 int64 `num_batches_tracked`) saved by `torch.save(model.state_dict(), …)`: no `module.` prefix, no optimizer, no metadata.
- Upstream weight license: Apache-2.0 (`license: apache-2.0` in the pinned upstream README front matter).
- Local layout: `weights/cfnet-levir-cd/` holds the 2 manifest entries (the checkpoint and the upstream `README.md`; 15,808,640 bytes total) with byte size and SHA-256 for each, plus the converted file described below. `verify_snapshot()` in `src/cfnet_change_detection_pipeline/pipeline.py` re-hashes every entry and, when present, the converted file.
- Cross-check: the manifest's checkpoint digest `22ab286b2138ab1082e04290b27cbff5abd3802375a6b2f01d4ba78ca8b0c1aa` equals the `oid sha256` of the Hub LFS pointer at the pinned revision (size 15805666).

## What the pickle would execute, and how it is audited

Under the fleet asset specification (§11) a pickle is executable serialization. `audit_pickle()` disassembles the file with `pickletools.genops` — every `.pkl` inside the torch zip archive — collects every `GLOBAL` / `STACK_GLOBAL` it would import, and refuses anything outside the allow-list, executing nothing:

| File | Globals found | Allow-list | Audit SHA-256 |
|---|---|---|---|
| `levir-cd.pth` | `collections.OrderedDict`, `torch.FloatStorage`, `torch.LongStorage`, `torch._utils._rebuild_tensor_v2` | exactly those four | `5b9f0ba08490293d6c17b9cef219991e1a6edda31609429679f8dca1af5a7b10` |

The audit reports 0 violations and its digest is pinned in `PICKLE_AUDIT_SHA256` (the digest covers the sorted set of global names, so it equals the digest of every fleet checkpoint that names the same four); `convert_model()` refuses a file whose audit digest differs. Tests craft a torch archive carrying `os.system`, a plain pickle of a `complex` number and a stream naming three of the four allowed globals, and assert that the audit refuses the first two and accepts the third.

An allow-list bounds what the unpickler can name; the loader below bounds what it can construct. The digest pins tie the audited bytes to the loaded bytes, and the unpickle happens once, in the operator's environment.

## The conversion (asset spec §11.2)

`convert_model()` runs size check → SHA-256 check against the package constant → static audit and audit-digest check, and only then:

- `torch.load(map_location="cpu", weights_only=True)` — torch's restricted unpickler, which constructs tensors and containers and nothing else — must return a dict of exactly 776 tensors;
- the tensors are loaded with `strict=True` into `CFNet()` from `modeling.py` — 0 missing, 0 unexpected, 0 shape mismatches;
- the network's own state dict (776 tensors, 3,877,781 elements, of which 3,838,563 are parameters and 39,218 are BatchNorm buffers) is saved with `safetensors.torch.save_file`.

Serving file (both identities recorded, `derived_from_sha256` = the source digest above):

| File | Bytes | Tensors | SHA-256 | In Git |
|---|---|---|---|---|
| `cfnet-levir-cd.safetensors` | 15,598,980 | 776 (3,877,781 elements; 3,838,563 parameters) | `b348186e5003a7447ab5eb5874204c50e63ad71aa61aca3ff0a80746116963d9` | no (regenerated) |

The conversion is deterministic: the digest was reproduced on two consecutive build conversions and on the executed tutorial notebook, which converts the file it downloads. `verify_converted()` checks size and digest; `from_pretrained()` loads the safetensors with `strict=True` and asserts the parameter count.

## The vendored network

`modeling.py` reimplements, in one plain-PyTorch module, `Encoder`, `ContentDecoder`, `Focuser`, `ChangeDecoder` and the building blocks (`CBA1x1`, `BasicBlock`, `Aggregation`, `FuseConv3d`, `CBAM` with its channel and spatial attention) from `model/*.py` of `wifiBlack/CFNet` at commit `54acadab23b9d9395ec6814386c2d4a1253eac5a` (Apache-2.0), without the training scaffolding (`autocast` inside `forward`, the `is_RM` / `is_content` switches, the feature-map dumps, the argument parser). The encoder backbone is `torchvision.models.efficientnet_b5(weights=None).features` sliced to its first five children (the stem and stages 1–4; 428 tensors, 2,091,568 parameters) — the same `getattr(torchvision.models, 'efficientnet_b5')(...).features` slice upstream builds, minus the ImageNet download, because the fine-tuned checkpoint carries those weights. Parameter and buffer names reproduce the checkpoint's exactly (`encoder._backbone.*`, `content_decoder_1.*`, `content_decoder_2.*`, `change_decoder.*`), which is what lets the conversion load with `strict=True`. Nothing from `cv2`, `albumentations`, `mmcv` or `timm` is imported.

Two upstream quirks are kept rather than fixed, because the checkpoint was trained with them: `ContentDecoder.forward` computes the CBAM attention of each content map and discards it (a loop that reassigns a local), so the 24 CBAM tensors carry weights that never affect the output, and `ChangeDecoder` constructs five fusion blocks (one per channel width including the 3-channel input level) of which the forward pass calls four. The port keeps the modules so the state dict matches and leaves them unused so the outputs match.

Preprocessing is the upstream loader's: `cv2.imread` delivers BGR, so `_normalise` reverses the RGB channels, divides by 255 and standardises each date with its own LEVIR-CD statistics (`Normalize(mean=[0.4503, 0.4467, 0.3813], std=[0.1745, 0.1649, 0.1531])` for the first date, `Normalize(mean=[0.3458, 0.3384, 0.2890], std=[0.1293, 0.1259, 0.1187])` for the second, in BGR order); the output is the network's `tanh` change map, thresholded at 0.5 as `predict.py` and `confuse_matrix.py` do. `tests/test_adaptation.py::test_normalise_reorders_to_bgr_with_per_date_statistics` pins the reorder.

## Fidelity

No upstream regression fixture is published for this checkpoint. The evidence is the strict key-and-shape match, the checkpoint's 428 encoder tensors matching the torchvision slice exactly, identical dates giving a change map that never exceeds 0.023 (far below the threshold), and the agreement with the authors' report on the dataset the model was trained on: the authors report F1 92.18 % / IoU 85.49 % on the full LEVIR-CD test split (3,200 crops, more than half change-free); on the 24 pinned test crops (all from that split, each with at least 3 % change) the vendored network reaches F1 0.9352 / IoU 0.8782 with precision 0.9424 and recall 0.9281. Sample-sanity evidence, not a reproduction of the benchmark.

## Runtime facts

- The model is float32 as shipped; `predict` runs under `torch.inference_mode()` with float16 autocast on CUDA and moves results to the CPU; `adapt` trains with gradients only on the selected tensors, with BatchNorm running statistics frozen. The network is small (3.8 M parameters): a 256 × 256 pair takes well under a second on a CPU, and the default adaptation about a minute.
- The loss reproduced by `adapt` is upstream's `Loss`: the mean squared error between the `tanh` change map and the 0 / 1 target over the labelled pixels, plus 0.1 / n times the summed content-consistency terms over the n = 4 scales (cosine similarities of `W` random pixel pairs within each date's focused and unfocused content maps, compared between dates); the random pairs are drawn from a seeded generator, so the frozen model's validation loss is reproducible.
- The package depends on `torch`, `torchvision`, `numpy`, `pillow`, `safetensors` and `huggingface-hub` only; images are read with Pillow.

## The tutorial data: the pinned LEVIR-CD tarball and its terms

`samples.py` fetches `LEVIR-CD-processed.tar.gz` from the Hugging Face dataset `wifibk/CFNet_Datasets` — the CFNet authors' mirror — at the immutable revision `ba68aa9a54ae15fe32ce9b02c380eb384fae528e` (3,831,872,824 bytes, SHA-256 `6515dd451c159b9ed5bd53b3fb6e15188dd114b296ff9169ca21fbcabcd1d109`, hashed once per `fetch_tarball` call and refused on a mismatch; the digest was checked against the Hub LFS pointer before it was pinned), then streams through it **once** with `extract_pinned_members`, copying out exactly the 192 members pinned in `SAMPLE_RECORDS` (64 crops × `A` / `B` / `label` PNGs: 256 × 256 RGB and 256 × 256 greyscale 0 / 255), each refused on a size or digest mismatch and written under a flat name (`<split>_<folder>_<key>.png`) in `weights/levir-cd/crops/`. Nothing else in the archive is written: not the other 47,589 members (11,125 / 1,600 / 3,200 crop triples in `train` / `val` / `test`, three list files, the authors' `1to255.py` and two stray root PNGs).

Roles are the dataset's own splits: one crop per source pair from 32 training pairs, 8 validation pairs and 24 test pairs, drawn with a fixed seed on 2026-09-20 from the crops whose label is a clean 0 / 255 mask (24 test crops carry an intermediate value, 156 or 254, and were excluded) with at least 3 % change (1,716 of the 3,200 test crops have none). `check_split_disjoint` refuses a crop (by pixel digest) or a source pair (by `region`) in two splits. The checkpoint was trained on the training split and selected on the validation split, so the test crops are a true hold-out and the adaptation's training crops are ones the model already learned from.

**Terms.** LEVIR-CD (Chen and Shi, 2020): "All images and annotations in LEVIR-CD can only be used for academic purposes, but are prohibited for any commercial use", and the imagery originates from Google Earth, whose terms of use apply. The mirror's repository card says `apache-2.0`, which cannot relicense the data; the dataset's own terms govern. This repository redistributes no image, fetches the mirror at run time for the academic purpose of the tutorial, states the terms in the card, the README and the notebook, and never vendors the crops under `weights/`.

## Files deliberately not staged

The upstream repository at the pinned revision also carries `clcd.pth` and `sysu-cd.pth` (the CLCD and SYSU-CD checkpoints) and five figures; none is listed in the manifest. The ImageNet weights of EfficientNet-B5 are not fetched: the fine-tuned checkpoint carries the backbone. The dataset mirror's `LEVIR-CD.tar.gz` (the raw 1024 × 1024 pairs), `CLCD*.tar.gz` and `SYSU-CD.tar.gz` are not fetched. Of the processed tarball, only the 192 pinned members are ever written to disk.

## DIMER hosting

- Apache-2.0 permits use, modification, redistribution and commercial use of the model subject to preservation of the licence and notices. DIMER may host the converted safetensors in its model store under those terms; it is derived from, and recorded beside, the unmodified upstream checkpoint.
- Upload set: `cfnet-levir-cd.safetensors`. **The `.pth` file must not be uploaded** — a profile that carries it would reintroduce the executable-serialization boundary this conversion removes.
- Loader trust boundary: no `trust_remote_code`, no Hub-hosted code, no pickle on the serving path; the network is this repository's `modeling.py` plus torchvision's EfficientNet-B5 class from PyPI, the served state dict is safetensors, and `from_pretrained(require_source=False)` accepts the digest-verified file without the manifest or the checkpoint.
- Serving shape: a pair needs the 15.6 MB weights and two same-sized RGB arrays; a CPU serves a 256 × 256 pair in well under a second, a T4 in a few milliseconds. An adapted profile needs the weights plus a 3.4 MB adapter (the change decoder) or 7 MB (with the content decoders).
- Data: the profile's inference needs no LEVIR-CD data. A DIMER tutorial that fetches LEVIR-CD is bound by its academic-only terms; the profile must not vendor or redistribute the crops.
- One review item is open: whether the one-time restricted unpickle (in the build and, for the tutorial, in the runtime) meets the DIMER deserialization-trust bar or whether DIMER hosts only maintainer-converted files. The served artifact is the same file either way.
- Line endings: `.gitattributes` carries `weights/** -text`, so a Windows checkout cannot rewrite a snapshot file's newlines and break its recorded digest.
