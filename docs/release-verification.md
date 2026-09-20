# Release verification

`tutorials/cfnet_change_detection_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`modeling.py`, `metrics.py`, `pipeline.py`, `samples.py`), each equal to its
  source after the generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest
  and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions; the dataset-mirror
  revision `ba68aa9a…` and the vendored upstream commit `54acadab…` are the only other 40-hex commits the documents
  may name;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `CFNetChangePipeline.from_pretrained(weights_dir=..., device=..., report=print)` so the pickle audit and the
  conversion are printed before the model loads, `fetch_sample_dataset` from the pinned cache path,
  `load_byod_dataset`, `dataset_manifest`, `write_sample_pair`, `validate_dataset` with the refusal probes,
  `pipe.evaluate` on the frozen model with the all-unchanged baseline and after the adaptation with the procedural
  assertions, `pipe.adapt` with its explicit hyperparameters, `pipe.predict` change maps written beside the dates and
  the labels, `pipe.save_artifact`, `CFNetChangePipeline.from_artifact` and the reload-parity assertion, and the
  provenance fields `served_from_pickle: False`, `remote_code_executed: False`, `data_license` and the
  `data_tarball` record), the eight expected `outputs/` paths, the learner-facing statements (the asset is a pickle
  unpickled once, the data are academic-use only under Google Earth's terms, the model was trained on this dataset,
  the all-unchanged baseline, the tarball is streamed with no `extractall`, split by scene) and the gated-off BYOD
  default; forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary
  path, a mutable `revision='main'`, direct `huggingface_hub` / `safetensors` / `urllib` / `tarfile.open(` /
  `Unpickler` / `CFNet(` / `efficientnet_b5(` / cv2 / albumentations / mmcv use or `torch.load(` / `pickle.load`
  **outside the carried module cells**, `trust_remote_code=True`, `pickle.load` or `torch.load(` without
  `weights_only=True` anywhere, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit suite
(`tests/test_pipeline.py`, `tests/test_samples.py`, `tests/test_adaptation.py` (stub model and the randomly
initialised vendored network, skipped without torch / torchvision), `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`, `tests/test_model_backed.py` (skipped without the
staged converted weights); crafted pickles, temporary manifests, synthetic pairs, a synthetic tarball with a decoy
member and an injected fetcher, no weights). These are source/provenance and unit checks. They are **not** execution
evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab runtime (a GPU makes the model time seconds; a CPU works) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence — and, for this row, the **first** execution of the notebook |
| Local harness (pre-flight only) | WSL workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence. Not used for this row: by the maintainer's rule of 2026-09-20 no local GPU job is run, and the package was smoke-tested on the CPU instead (`MODEL_CARD.md`, *Runtime*) |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/cfnet-levir-cd/` or the data cache `weights/levir-cd/` (the standalone path writes the manifest
   itself, stages both listed files from the Hub, audits and converts the checkpoint, fetches the 3.8 GB tarball
   from the Hub dataset at its immutable revision, hashes it and streams out the 192 pinned members, so neither
   directory may be seeded); the runtime needs about 4 GB of free disk for the tarball;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `EPOCHS = 4`, `LEARNING_RATE = 1e-5`, `BATCH_SIZE = 4`, `TRAINABLE = 'change_decoder'`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `numpy==2.5.3`, `pillow==11.3.0`,
   `safetensors==0.8.0`, `huggingface-hub==1.32.0` (an interpreter restart after the install is expected where the
   runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the four carried module cells execute (defining `CFNet`, `CFNetChangePipeline`, `audit_pickle`,
     `convert_model`, `build_model`, `verify_snapshot`, `verify_converted`, `stage_missing_files`, `validate_inputs`,
     `validate_dataset`, `read_image`, `read_mask`, `fetch_tarball`, `extract_pinned_members`,
     `fetch_sample_dataset`, `load_byod_dataset`, `write_sample_pair`, `write_dataset_csv`, `dataset_manifest`,
     `check_split_disjoint`, `change_metrics`, `unchanged_baseline` and the constants) with no repository import;
   - the model cell writing `weights/cfnet-levir-cd/dimer-base-manifest.json`, staging the two files from the Hub
     at the pinned revision and `verify_snapshot` reporting 2 verified files;
   - the model cell printing the **conversion record** with the static audit (the four torch globals, 0 violations,
     audit digest `5b9f0ba0…`), the checkpoint record (a plain state dict of 776 tensors, 428 of them the encoder)
     and the converted file (`b348186e…`, 15,598,980 bytes), then the load report with source "converted from the
     manifest-verified source checkpoint";
   - the tarball fetched and hashed (`6515dd45…`, 3,831,872,824 bytes), the 192 pinned members extracted under
     `weights/levir-cd/crops/`, the dataset manifest with 32 / 8 / 24 crops from as many source pairs and change
     fractions about 0.10 / 0.13 / 0.14, the written sample pair and `outputs/cfnet_change_detection_sample_pairs.csv`,
     and three refusals (dates of different sizes, a side not a multiple of 32, an unknown label value);
   - the all-unchanged baseline and the frozen model on the test crops (on the sample: baseline accuracy ≈ 0.86,
     F1 0; frozen F1 ≈ 0.935, IoU ≈ 0.878) and the validation crops (F1 ≈ 0.915);
   - `pipe.adapt` printing epoch 0 as the frozen model, 852,867 trainable of 3,838,563 parameters, 32 steps, the
     upstream loss, frozen BatchNorm statistics, and a four-epoch history with validation loss ≈ 0.116 → ≈ 0.114 at
     the kept epoch;
   - `pipe.evaluate` on the test crops with the three-way comparison and
     `outputs/cfnet_change_detection_evaluation_report.json` written (the cell asserts the kept epoch's validation
     loss is no higher than the frozen model's and that the validation F1 matches the history within 0.01);
   - the change maps of two held-out crops written beside their dates and labels with
     `outputs/cfnet_change_detection_predictions.json`;
   - `pipe.save_artifact` writing `outputs/cfnet_change_detection_adapter/{adapter.safetensors,manifest.json}`
     (132 tensors, about 3.4 MB), and `CFNetChangePipeline.from_artifact` reloading it with held-out metrics and
     change maps matching the adapted model (the cell asserts an F1 difference below 10⁻³ and a maximum map
     difference below 10⁻²);
   - `outputs/cfnet_change_detection_result.json` written with `NOTEBOOK_SOURCE`, the model identity, the
     provenance block (`served_from_pickle: false`, `remote_code_executed: false`, the audit digest, the converted
     digest, the `data_tarball` record, the data terms), the runtime versions, the comparison and the reload parity;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, device), the model identifier and
   immutable revision, whether the model cache, the weights directory and the data cache were clean, outcome,
   produced outputs, the observed metrics (as observations, not a benchmark) and any warning or applicable `SHOULD`
   deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `cfnet_change_detection_colab.ipynb` (`E2E`) | `b92e01f` / `9e515a82` | 2026-09-20 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-cfnet-change-detection` v1; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `pillow 11.3.0`, `numpy 2.5.3` after, Python 3.12.13, `cuda`) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 204 files, 3880 MB fetched into a clean runtime (the Hub snapshot, the 3.8 GB tarball and the 192 extracted members); comparison test changed-class F1 / IoU (all-unchanged baseline 0 / 0): frozen 0.9352 / 0.8782 → adapted 0.9352 / 0.8783, precision 0.9423 → 0.9424, recall 0.9281 → 0.9282, accuracy 0.9822 → 0.9822 vs baseline 0.8615 (best epoch 2, validation loss 0.1162 → 0.1141, validation F1 0.9151 → 0.9137); reload parity f1_diff: 0.0, metrics_identical: True, max_abs_map_diff: 0.0; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-cfnet-change-detection/v1/evidence/` in the workspace |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/cfnet_change_detection_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/cfnet_change_detection_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-20 | `b92e01f` / `9e515a82` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-cfnet-change-detection` v1; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `pillow 11.3.0`, `numpy 2.5.3` after, Python 3.12.13, `cuda`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution); the checkpoint audited and converted in the notebook, the tarball fetched and the pinned members extracted by the notebook — the notebook's first execution anywhere | 271.0 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 204 files, 3880 MB fetched into a clean runtime (the Hub snapshot, the 3.8 GB tarball and the 192 extracted members); comparison test changed-class F1 / IoU (all-unchanged baseline 0 / 0): frozen 0.9352 / 0.8782 → adapted 0.9352 / 0.8783, precision 0.9423 → 0.9424, recall 0.9281 → 0.9282, accuracy 0.9822 → 0.9822 vs baseline 0.8615 (best epoch 2, validation loss 0.1162 → 0.1141, validation F1 0.9151 → 0.9137); reload parity f1_diff: 0.0, metrics_identical: True, max_abs_map_diff: 0.0; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-cfnet-change-detection/v1/evidence/` in the workspace |

## Current status

**Release-grade.** The `E2E` notebook blob `9e515a82` (committed at `b92e01f`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-20 (11/11 ok (1 restart after install cell), 271.0 s, 204 files, 3880 MB fetched and digest-verified inside the notebook, the checkpoint converted in the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.
