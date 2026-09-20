"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (modeling.py, metrics.py, pipeline.py, samples.py), and the model pin/stage/verify cells are produced
by the generator from repository sources so they cannot drift from the package.

This template configures an E2E change-detection workflow: the pinned CFNet LEVIR-CD checkpoint (a torch pickle
of the state dict) is digest-verified, statically audited and converted once into safetensors, 64 labelled
bi-temporal crops are extracted from the digest-pinned LEVIR-CD tarball, validated and assigned the dataset's own
splits, the frozen model is scored against the all-unchanged baseline, a bounded fine-tuning of the change decoder
runs in the kernel, the held-out pairs are scored again, change maps are written, and the adapter is exported and
reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "cfnet-change-detection-pipeline"

BADGES = [
    (
        "GitHub",
        "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
        f"https://github.com/kurtvalcorza/{REPO}",
    ),
    (
        "Open In Colab",
        "https://colab.research.google.com/assets/colab-badge.svg",
        f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/cfnet_change_detection_colab.ipynb",
    ),
    (
        "Hugging Face",
        "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-wifibk%2FCFNet-ffcc4d?style=flat",
        "https://huggingface.co/wifibk/CFNet",
    ),
    (
        "Upstream",
        "https://img.shields.io/badge/Upstream-wifiBlack%2FCFNet-181717?style=flat&logo=github&logoColor=white",
        "https://github.com/wifiBlack/CFNet",
    ),
    ("Paper", "https://img.shields.io/badge/arXiv-2503.08505-b31b1b.svg", "https://arxiv.org/abs/2503.08505"),
]

TEMPLATE = {
    "package": "cfnet_change_detection_pipeline",
    "repo_name": REPO,
    "stem": "cfnet_change_detection",
    "notebook_name": "cfnet_change_detection_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh runtime (GPU recommended, CPU works) installs the pinned dependencies (torch, torchvision, "
        "numpy, pillow, safetensors, huggingface-hub — the network is carried in this notebook, the EfficientNet-B5 stages come "
        "from the installed torchvision with no download), stages and digest-verifies the pinned CFNet LEVIR-CD checkpoint (15.8 MB) "
        "from the Hub, statically audits the pickle against an allow-list, converts it once into safetensors with a pinned digest, "
        "rebuilds the network from the carried module and loads it strictly, fetches the digest-pinned LEVIR-CD tarball (3.8 GB, "
        "no credential) and extracts exactly the 192 pinned before / after / label members, validates them and assigns the "
        "dataset's own splits (32 training, 8 validation, 24 test crops), detects change on the held-out pairs with the frozen "
        "model and scores them against the all-unchanged baseline, runs a bounded fine-tuning of the change decoder, scores the "
        "same pairs again, writes change maps for two held-out pairs beside their images and labels, exports the adapter as "
        "safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify prediction parity. The default "
        "path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.0 §5). On a T4 the whole path takes under a minute of model time after the downloads; the 3.8 GB "
        "tarball is the slowest step."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "labelled pairs as a zip holding `pairs.csv` (columns `id`, `before`, `after`, `label`) beside co-registered RGB PNG / "
        "JPEG images of the same size (sides multiples of 32) and single-band label PNGs (0 = unchanged, 255 = changed); at least "
        "four pairs with some change. Your pairs are split by seed into training, validation and test sets and flow through the "
        "same contract — validation, frozen baseline, adaptation, held-out evaluation, change maps, artifact export and reload "
        "parity. The expected schema, the ceilings and the privacy guidance are stated in the Prerequisites and in Section 4, and "
        "uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "CFNetChangePipeline",
    "model_load": "CFNetChangePipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=('cuda' if torch.cuda.is_available() else 'cpu'), report=print)",
    "weights_key": "cfnet-levir-cd",
    "modules": ["modeling.py", "metrics.py", "pipeline.py", "samples.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "torchvision", "PIL"],
    "title": "CFNet bi-temporal change detection — DIMER E2E fine-tuning tutorial (standalone)",
    "badges": BADGES,
    "capability": "binary building-change maps for pairs of co-registered 0.5 m RGB patches with CFNet's content-focuser network, held-out F1/IoU against an all-unchanged baseline, and bounded fine-tuning of the change decoder to labelled pairs",
    "intro": (
        "CFNet (Wu et al., 2025) is a content-focuser network for bi-temporal remote-sensing change detection: a shared "
        "EfficientNet-B5 encoder over both dates, a content decoder per date trained to keep what is intrinsic to the scene "
        "rather than to its acquisition, a focuser that turns the cosine distance between the two dates' content maps into "
        "per-scale change weights, and a change decoder that fuses the dates with 3-D convolutions and decodes one change map. "
        "The checkpoint packaged here is the authors' LEVIR-CD model — trained on 0.5 m Google Earth patches of Texas cities "
        "taken 5–14 years apart with building-change labels (Chen and Shi, 2020) — which they report at F1 92.18 / IoU 85.49 "
        "on that dataset's test split.\n\n"
        "Three things about this row are handled in the open. **The upstream asset is a pickle** — a torch state dict saved "
        "with `torch.save`. Section 3 downloads and digest-verifies it, statically lists every global the pickle would import "
        "(a state dict of tensors and nothing else), refuses anything outside that allow-list, unpickles it exactly once "
        "through torch's weights-only loader, and writes a safetensors file whose digest is pinned in the carried module; the "
        "network you run is rebuilt from `modeling.py`, carried in this notebook, with the EfficientNet-B5 stages taken from "
        "the installed torchvision package (`weights=None`, no download) and loads that file strictly. **The dataset ships as "
        "one 3.8 GB tarball under academic-only terms**: LEVIR-CD may be used for academic purposes only, commercial use is "
        "prohibited, and the imagery is subject to Google Earth's terms — so Section 4 pins the authors' mirror of the tarball "
        "by size and digest, streams through it once and copies out exactly the 192 pinned members (each pinned again by size "
        "and digest, no `extractall`, no paths taken from the archive), leaves the other 47,589 alone, and redistributes "
        "nothing. **The model was trained on this dataset**, so the bounded adaptation in Section 6 is a demonstration of the "
        "contract on the training split's own crops, selected by validation loss with the frozen model as epoch 0, and the "
        "test split — which the model never trained on — is the held-out set; the point of the contract is the same recipe "
        "applied to *your* labelled pairs."
    ),
    "learning_objectives": (
        "install the pinned runtime; inspect the carried network, pipeline, dataset and metrics modules; stage and digest-verify "
        "a pickled checkpoint, read its static audit and see it converted into safetensors; extract pinned members from a "
        "digest-verified tarball and validate real labelled bi-temporal pairs with an ignore class; read F1, IoU, precision "
        "and recall of the changed class against an all-unchanged baseline; run a bounded fine-tuning of the change decoder "
        "with the upstream loss, explicit hyperparameters and frozen BatchNorm statistics; compare the adapted and frozen "
        "models on the same held-out pairs; write change maps; and export a safetensors adapter that reloads against the "
        "pinned base with verified parity."
    ),
    "exclusions": (
        "multi-class or semantic change, change between more than two dates, images that are not co-registered, resolutions "
        "far from 0.5 m, tiling of scenes larger than a patch, the published benchmark scores, the CLCD and SYSU-CD checkpoints "
        "of the same repository, and any claim that a 64-crop sample stands in for an operational evaluation. The repository "
        "exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab, or a Jupyter kernel with Python 3.12); a GPU (T4 or better) makes the model time a few seconds, a CPU a few minutes — the network has 3.8 M parameters. About 4 GB of disk is needed for the tarball; the checkpoint and its conversion are 31 MB together.",
        "- **Knowledge:** what a co-registered bi-temporal image pair is, what a binary change mask and an ignore class are, and how F1, IoU, precision and recall of a rare class are read against a majority baseline.",
        "- **Executable serialization handled explicitly:** the pinned checkpoint is a pickle. It is digest-verified, statically audited against an allow-list (audit digest pinned) and unpickled **once** through torch's weights-only loader to produce the safetensors the network is actually loaded from. No Hub-hosted Python module is imported; the network is the carried `modeling.py` plus torchvision's EfficientNet-B5 stages built without weights.",
        "- **Data contract:** a record is `{{id, before, after, label}}` — two (H, W, 3) uint8 RGB images (or PNG / JPEG paths) of the same size, sides in [64, 2048] and multiples of 32, and an (H, W) mask with 0 / 1 / −1 (or a PNG with 0 / 255). The pipeline reorders the channels to BGR and standardises each date with its own LEVIR-CD statistics, exactly as the upstream loader did. Validation is structural: nothing checks that the two images show the same place, that they are co-registered, that the resolution is about 0.5 m, or that the label belongs to the pair.",
        "- **Privacy and terms:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — commercial imagery under licence is exactly that. The default path uploads nothing. LEVIR-CD, which the default path fetches, may be used **for academic purposes only**; commercial use is prohibited and the imagery is subject to Google Earth's terms of use.",
        "- **External access (data):** besides the model snapshot, the default path fetches one pinned object — the 3.8 GB `LEVIR-CD-processed.tar.gz` of the Hugging Face dataset `wifibk/CFNet_Datasets` (the CFNet authors' mirror) at an immutable revision — over HTTPS, digest-verified before any member is read.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample pairs, validation and roles\n\n"
                "The default dataset is 64 labelled 256 × 256 crops of LEVIR-CD as the CFNet authors processed them (25 "
                "overlapping crops of every 1024 × 1024 pair): one crop from each of 32 training pairs, 8 validation pairs and "
                "24 test pairs, drawn with a fixed seed from the crops whose label is a clean 0 / 255 mask with at least 3 % "
                "change, so every crop can be scored. Roles are the dataset's own splits — the checkpoint was trained on the "
                "training split, selected on the validation split and reported on the test split. `fetch_corpus` downloads the "
                "tarball from the Hub at its immutable revision, refuses it on any size or SHA-256 mismatch, streams through it "
                "once and copies out exactly the 192 pinned members — each refused on its own size or digest mismatch and "
                "written under a flat name, never at a path taken from the archive — then reads the RGB PNGs and the masks, "
                "which become 0 / 1. `dataset_manifest` validates every split, checks that no crop and no source pair appears "
                "twice and records a digest.\n\n"
                "Look for: 32 / 8 / 24 crops with change fractions around 0.10–0.14, one source pair per crop, a written sample "
                "pair (`outputs/{stem}_sample_before.png`, `_sample_after.png`, `_sample_label.png` — the BYOD shape), and three "
                "refusal probes — dates of different sizes, a side that is not a multiple of 32, a mask with an unknown value — "
                "each rejected before the model runs. The tarball takes a few minutes to fetch and about a minute to stream."
            ),
            "code": (
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "import numpy as np\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    splits = split_dataset(load_byod_dataset(byod_path), seed=0)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "else:\n"
                "    splits = fetch_sample_dataset(cache_dir='weights/levir-cd')\n"
                "    data_source = SAMPLE_LABEL_SOURCE\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n\n"
                "dataset_report = dataset_manifest({{'train': train_records, 'validation': val_records, 'test': test_records}})\n"
                "print({{'data_source': data_source, 'splits': {{k: v['n_records'] for k, v in dataset_report['splits'].items()}}, 'disjoint': dataset_report['disjoint'], 'digest': dataset_report['digest'][:16] + '...'}})\n"
                "for name, part in dataset_report['splits'].items():\n"
                "    print({{name: {{'change_fraction': part['change_fraction'], 'sizes': part['sizes'], 'ignored_pixels': part['ignored_pixels'], 'source_pairs': len(part['regions'])}}}})\n"
                "print({{'first_test_pair': validate_inputs(test_records[0])}})\n"
                "sample_pair = write_sample_pair(test_records[0], 'outputs/{stem}_sample_before.png', 'outputs/{stem}_sample_after.png', 'outputs/{stem}_sample_label.png')\n"
                "print({{'sample_pair': sample_pair, 'pairs_csv': str(write_dataset_csv(test_records, 'outputs/{stem}_sample_pairs.csv'))}})\n\n"
                "print({{'validation': INPUT_SCHEMA['validation']}})\n"
                "probes = {{\n"
                "    'dates of different sizes': [{{**test_records[0], 'after': test_records[0]['after'][:128, :128]}}, *test_records[1:4]],\n"
                "    'side not a multiple of 32': [{{**test_records[0], 'before': test_records[0]['before'][:200, :200], 'after': test_records[0]['after'][:200, :200], 'label': test_records[0]['label'][:200, :200]}}, *test_records[1:4]],\n"
                "    'unknown label value': [{{**test_records[0], 'label': np.where(test_records[0]['label'] == 1, 2, test_records[0]['label'])}}, *test_records[1:4]],\n"
                "}}\n"
                "for name, records in probes.items():\n"
                "    try:\n"
                "        validate_dataset(records)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. The frozen model against the all-unchanged baseline\n\n"
                "`pipe.predict` reorders each image to BGR, divides by 255 and standardises each date with its own LEVIR-CD "
                "statistics — exactly the upstream loader's preprocessing — runs the shared encoder, the two content decoders, "
                "the focuser and the change decoder (in float16 autocast on a GPU), and returns the change map (the network's "
                "`tanh` output in (−1, 1), not a calibrated probability), the binary mask (changed where the map exceeds 0.5, "
                "the upstream rule) and the changed fraction per pair. `pipe.evaluate` pools the labelled pixels of every "
                "held-out pair into one confusion matrix (−1 pixels excluded) and reports the F1, IoU, precision and recall of "
                "the changed class exactly as the upstream evaluation does, plus the overall accuracy; the **all-unchanged "
                "baseline** — every pixel predicted unchanged — is scored on the same pixels, so its accuracy is exactly the "
                "unchanged fraction and its F1 and IoU are 0.\n\n"
                "Look for: a changed-class F1 near 0.93 and an IoU near 0.88 on the test crops (in the build record 0.9352 and "
                "0.8782, precision 0.9424, recall 0.9281, against an all-unchanged accuracy of 0.8615 — the authors report F1 "
                "92.18 / IoU 85.49 on the full test split, whose crops are mostly change-free) and a validation F1 near 0.92. "
                "These are sample-sanity numbers on 24 crops chosen for having change, not the benchmark."
            ),
            "code": (
                "import time\n\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "frozen_val = pipe.evaluate(val_records)\n"
                "print({{'seconds': round(time.perf_counter() - t0, 1), 'metric': frozen_test['metric']}})\n"
                "print({{'baseline_unchanged_test': {{k: frozen_test['baseline_unchanged'][k] for k in ('f1', 'iou', 'accuracy')}}}})\n"
                "print({{'frozen_test': {{k: frozen_test['model'][k] for k in ('f1', 'iou', 'precision', 'recall', 'accuracy', 'change_fraction_label', 'change_fraction_predicted')}}}})\n"
                "print({{'frozen_validation': {{k: frozen_val['model'][k] for k in ('f1', 'iou')}}}})\n"
                "frozen_predictions = pipe.predict(test_records)\n"
                "for record, pred in list(zip(test_records, frozen_predictions['predictions']))[:6]:\n"
                "    labelled = record['label'] >= 0\n"
                "    print({{'pair': record['source_id'], 'change_label': round(float((record['label'] == 1).sum() / labelled.sum()), 3), 'change_predicted': pred['changed_fraction'], 'map_range': [round(float(pred['change_map'].min()), 3), round(float(pred['change_map'].max()), 3)]}})\n"
                "print({{'decision_rule': frozen_predictions['decision_rule'], 'map_shape': frozen_predictions['predictions'][0]['change_map'].shape}})"
            ),
        },
        {
            "md": (
                "## 6. Bounded fine-tuning of the change decoder\n\n"
                "`pipe.adapt` trains the 132 tensors of the change decoder (852,867 parameters — 22 % of the model) and nothing "
                "else: the encoder and the two content decoders are frozen (no gradient is stored for them), and every "
                "BatchNorm layer keeps its running statistics, because batches of four crops would corrupt them. Each step "
                "takes four pairs with a seeded horizontal or vertical flip applied to both dates and the label, computes the "
                "upstream loss — the mean squared error between the `tanh` change map and the 0 / 1 target over the labelled "
                "pixels, plus 0.1 times the content-consistency terms that compare cosine similarities of random pixel pairs "
                "within each date's focused and unfocused content maps — and takes an AdamW step at a small fixed learning rate "
                "with gradient-norm clipping and float16 loss scaling. Epoch 0 records the frozen model's validation loss and "
                "metrics; the epoch with the lowest validation loss is kept — which can be epoch 0, since the packaged model "
                "already trained on this split.\n\n"
                "Watch the validation loss: in the build record it fell from 0.1162 to 0.1141 at epoch 2 and drifted up "
                "afterwards, while the validation F1 stayed within 0.005 of the frozen model's — the sign that a small "
                "learning rate and validation selection are doing their job on a model that has little left to learn from "
                "32 crops of a dataset it trained on. Four epochs (32 steps) take seconds on a T4 and about a minute on a CPU. "
                "`TRAINABLE = 'decoders'` also unfreezes the two content decoders (1.75 M parameters)."
            ),
            "code": (
                "EPOCHS = 4  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 4  # @param {{type:\"integer\"}}\n"
                "TRAINABLE = 'change_decoder'  # @param [\"change_decoder\", \"decoders\"]\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4), 'val_loss': round(entry['val_loss'], 4)}}\n"
                "    if 'val' in entry:\n"
                "        row['val_f1'] = entry['val']['f1']\n"
                "        row['val_iou'] = entry['val']['iou']\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable=TRAINABLE, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'steps': adapt_result['n_steps'], 'best_epoch': adapt_result['best_epoch'], 'loss': adapt_result['loss'], 'precision': adapt_result['precision'], 'batchnorm': adapt_result['batchnorm'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 7. Held-out evaluation: the paired comparison\n\n"
                "The test crops were never used for training or epoch selection — neither here nor by the upstream authors, "
                "whose model was trained on the training split and selected on the validation split. The adapted model is "
                "scored exactly as the frozen model was in Section 5, and the table puts the baseline, the frozen and the "
                "adapted numbers side by side. The cell asserts what the procedure guarantees — the kept epoch's validation "
                "loss is no higher than the frozen model's, and re-scoring the validation crops reproduces the kept epoch's F1 "
                "within 0.01 (float16 kernels are not bit-reproducible across batch sizes) — and prints the test numbers "
                "without asserting a direction: on this sample the changed-class F1 moved from 0.9352 to 0.9352 and the IoU "
                "from 0.8782 to 0.8783 in the build record, a sample-sanity observation on 24 crops with no dispersion "
                "estimate, not a quality claim. With your own pairs from another city, sensor or season, the gap between "
                "frozen and adapted is the number to watch."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "comparison = {{}}\n"
                "for key in ('f1', 'iou', 'precision', 'recall', 'accuracy'):\n"
                "    comparison[key] = {{'baseline_unchanged': frozen_test['baseline_unchanged'][key], 'frozen': frozen_test['model'][key], 'adapted': adapted_test['model'][key]}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "print({{'validation_f1': {{'frozen': frozen_val['model']['f1'], 'adapted': adapted_val['model']['f1']}}, 'validation_loss': {{'frozen': adapt_result['history'][0]['val_loss'], 'kept_epoch': adapt_result['history'][adapt_result['best_epoch']]['val_loss']}}}})\n"
                "evaluation_report = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset': dataset_report,\n"
                "    'frozen': {{'test': frozen_test, 'validation': frozen_val}},\n"
                "    'adapted': {{'test': adapted_test, 'validation': adapted_val}},\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report, f, indent=2)\n"
                "assert adapt_result['history'][adapt_result['best_epoch']]['val_loss'] <= adapt_result['history'][0]['val_loss']\n"
                "assert abs(adapted_val['model']['f1'] - adapt_result['history'][adapt_result['best_epoch']]['val']['f1']) < 1e-2\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 8. Change maps, artifact export and fresh reload\n\n"
                "The adapted model's change maps of two held-out pairs are written as PNGs beside the two dates and the "
                "reference mask — the binary mask (0 / 255, the BYOD label convention) and the raw `tanh` map rescaled to "
                "0–255 — so they can be opened side by side: the agreement per pair printed here is a sanity check, not an "
                "evaluation.\n\n"
                "`pipe.save_artifact` writes the trained tensors (about 3.4 MB) as `adapter.safetensors`, with a "
                "`manifest.json` recording the artifact format, the base model id and revision, the digest of the converted "
                "base file, the adaptation scope, the tensor names, the file size and SHA-256, the training configuration and "
                "the epoch history (OUT8). `CFNetChangePipeline.from_artifact` re-verifies the base file, checks the artifact "
                "manifest, scope and digest **before** deserialising, rebuilds the network and overlays the tensors — a fresh "
                "object from files, not the in-memory model (VER2). The cell asserts the same held-out F1 within 0.001 and "
                "change maps within 0.01 (VER4: float16 tolerances; on one device they are usually identical)."
            ),
            "code": (
                "import platform\n"
                "import shutil\n\n"
                "from PIL import Image\n\n"
                "shown_records = test_records[:2]\n"
                "shown_predictions = pipe.predict(shown_records)\n"
                "for record, pred in zip(shown_records, shown_predictions['predictions']):\n"
                "    tag = record['source_id']\n"
                "    Image.fromarray(record['before']).save(f'outputs/{stem}_before_' + tag + '.png')\n"
                "    Image.fromarray(record['after']).save(f'outputs/{stem}_after_' + tag + '.png')\n"
                "    Image.fromarray(np.where(record['label'] == 1, 255, 0).astype(np.uint8)).save(f'outputs/{stem}_label_' + tag + '.png')\n"
                "    Image.fromarray((pred['mask'] * 255).astype(np.uint8)).save(f'outputs/{stem}_change_adapted_' + tag + '.png')\n"
                "    Image.fromarray(np.clip((pred['change_map'] + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8)).save(f'outputs/{stem}_map_adapted_' + tag + '.png')\n"
                "    labelled = record['label'] >= 0\n"
                "    print({{'pair': tag, 'agreement': round(float((pred['mask'] == record['label'])[labelled].mean()), 3), 'change_label': round(float((record['label'] == 1).sum() / labelled.sum()), 3), 'change_predicted': pred['changed_fraction'], 'note': 'sanity check on two pairs'}})\n"
                "with open('outputs/{stem}_predictions.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump({{'model': shown_predictions['model'], 'classes': shown_predictions['classes'], 'decision_rule': shown_predictions['decision_rule'], 'predictions': [{{'id': p['id'], 'changed_fraction': p['changed_fraction']}} for p in shown_predictions['predictions']]}}, f, indent=2)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'trainable': artifact_manifest['adapter']['trainable'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = CFNetChangePipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded_test = reloaded.evaluate(test_records)\n"
                "before = pipe.predict(test_records[:2])['predictions']\n"
                "after = reloaded.predict(test_records[:2])['predictions']\n"
                "parity = {{'f1_diff': round(abs(reloaded_test['model']['f1'] - adapted_test['model']['f1']), 6), 'metrics_identical': reloaded_test['model'] == adapted_test['model'], 'max_abs_map_diff': max(float(np.abs(a['change_map'] - b['change_map']).max()) for a, b in zip(before, after))}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['f1_diff'] < 1e-3 and parity['max_abs_map_diff'] < 1e-2\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model': {{**evaluation_report['model'], 'model_license': MODEL_LICENSE, 'device': pipe.device, 'source': pipe.source}},\n"
                "    'provenance': {{\n"
                "        'source_asset': [e for e in MANIFEST['files'] if e['path'] == SOURCE_CKPT_NAME],\n"
                "        'pickle_audit_sha256': PICKLE_AUDIT_SHA256,\n"
                "        'converted': verify_converted(WEIGHTS_DIR)['files'],\n"
                "        'pickle_unpickled_once_for_conversion': True,\n"
                "        'served_from_pickle': False,\n"
                "        'remote_code_executed': False,\n"
                "        'network_source': 'modeling.py carried in this notebook; EfficientNet-B5 stages from torchvision ' + torchvision.__version__ + ' with weights=None',\n"
                "        'data_tarball': {{'name': TAR_NAME, 'sha256': TAR_SHA256, 'pinned_members': 3 * len(SAMPLE_RECORDS)}},\n"
                "        'data_base_url': CORPUS_BASE_URL,\n"
                "        'data_license': CORPUS_LICENSE,\n"
                "    }},\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'torchvision': torchvision.__version__, 'pillow': PIL.__version__, 'numpy': np.__version__}},\n"
                "    'data_source': data_source,\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes']}},\n"
                "    'reload_parity': parity,\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_payload, f, indent=2)\n\n"
                "print('outputs/:')\n"
                "for path in sorted(Path('outputs').rglob('*')):\n"
                "    if path.is_file():\n"
                "        print(f'  - {{path.as_posix()}} ({{path.stat().st_size / 1024:.1f}} KB)')"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "On 24 held-out crops the packaged change detector finds building change with an F1 near 0.93 and an IoU near 0.88, "
        "against an all-unchanged baseline that scores 0 on both; a bounded fine-tuning of its change decoder on 32 crops of "
        "the training split, selected by validation loss with the frozen model as a candidate, leaves those numbers where "
        "they were (0.9352 → 0.9352 F1 in the build record). That is the claim: the adaptation contract runs end to end on "
        "real labelled bi-temporal pairs drawn from a digest-verified tarball, the pickle is audited and converted rather "
        "than served, the network is carried in plain PyTorch, and the artifact that carries the change is 3.4 MB and "
        "reloads with the same outputs. It is not a claim that this sample improves the model — the model already trained on "
        "this dataset — nor that 24 crops measure its skill.\n\n"
        "The numbers are sample-sanity evidence: one seeded run, 24 crops chosen for having change (the full test split is "
        "mostly change-free, which is why the authors' F1 is lower), no dispersion estimate, pixel-pooled metrics that let "
        "large buildings dominate, and labels drawn by hand with their own uncertainty at building edges. Nothing here "
        "measures the model outside Texas suburbs, outside 0.5 m Google Earth imagery, on pairs that are not co-registered, "
        "or on change that is not a building.\n\n"
        "Three things to carry to real data. **The two dates are not interchangeable:** the network standardises each date "
        "with its own statistics and its content decoders are separate, so `before` and `after` must be the earlier and the "
        "later image; a swapped or uncalibrated pair is compared without complaint and silently wrong. **Split by scene, not "
        "by crop:** overlapping crops of one scene are near-duplicates, and a random split makes memorisation look like "
        "skill. **Read the baseline first:** on a crop with 5 % change the all-unchanged baseline is 95 % accurate; only the "
        "changed-class F1, IoU, precision and recall say whether the detector did anything.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify a pickled upstream checkpoint, audit and convert it into safetensors without "
        "executing anything outside the audited allow-list, rebuild the network from the carried module, fetch a digest-pinned "
        "tarball and extract exactly the pinned labelled pairs, execute bounded fine-tuning, evaluate against a baseline and "
        "the frozen model on held-out pairs, and emit the shown machine-readable artifacts — without the repository being "
        "reachable. It does **not** establish benchmark superiority, production fitness, or change-detection skill beyond "
        "the checks shown.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE = 'decoders'`; raise `EPOCHS` and "
        "watch the validation loss drift; try `LEARNING_RATE = 1e-3` to see the validation F1 fall while the frozen model "
        "keeps the kept epoch; or bring your own labelled pairs through BYOD and read the baseline before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/cfnet-change-detection-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/cfnet-change-detection-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights and conversion notes: https://github.com/kurtvalcorza/cfnet-change-detection-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Hugging Face model repository: https://huggingface.co/wifibk/CFNet (revision `{MODEL_REVISION}`)\n"
        "- LEVIR-CD dataset (academic use only): https://justchenhao.github.io/LEVIR/ · the CFNet authors' mirror: https://huggingface.co/datasets/wifibk/CFNet_Datasets\n"
        "- Wu, F., Dong, S., Meng, X. (2025). CFNet: Optimizing remote sensing change detection through content-aware enhancement. arXiv:2503.08505: https://arxiv.org/abs/2503.08505\n"
        "- Chen, H., Shi, Z. (2020). A spatial-temporal attention-based method and a new dataset for remote sensing image change detection. Remote Sensing 12(10), 1662 (LEVIR-CD)\n"
        "- Upstream code: https://github.com/wifiBlack/CFNet (the network is vendored in `modeling.py`)\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)\n"
    ),
}
