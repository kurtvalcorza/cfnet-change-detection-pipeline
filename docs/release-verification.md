# Release verification

`tutorials/cfnet_change_detection_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate record.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`modeling.py`, `metrics.py`, `pipeline.py`, `samples.py`), each equal to its
  source after the generator's documented rewrites; the inline `MANIFEST` equal to the committed snapshot manifest
  and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; exactly two kernel cells — the uv isolated install (pinned
  `uv` wheel by size and SHA-256, managed CPython, `--require-hashes --only-binary :all:`, Linux x86_64 check) and the
  router to the isolated worker; `NOTEBOOK_SOURCE` recorded in exports; the guided layer (audience, how to use,
  roadmap, predictions, worked answers, the Section 9 activity, troubleshooting, glossary, conclusion), the reset to
  the pretrained model before Sections 5 and 6 use `pipe`, no bare `assert` in learner cells, and a list of stale
  learner-facing text that must not return (review CFN-M1..M5, CFN-m1..m3);
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
| Local harness (pre-flight only) | Workstation CPU, the learner cells executed verbatim from the notebook JSON in one namespace (the two kernel cells skipped, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), pre-staged pins and files | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence. No local GPU job is run (maintainer's rule of 2026-09-20) |

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
   `USE_BYOD = False`, `BYOD_PATH = ''`, `EPOCHS = 4`, `LEARNING_RATE = 1e-5`, `BATCH_SIZE = 4`,
   `TRAINABLE = 'change_decoder'`) in **one pass, with no runtime restart** — a run that needs a restart is a failed
   `Run all` (RUN1, REL11), whatever the second pass does;
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `numpy==2.5.3`, `pillow==11.3.0`,
   `safetensors==0.8.0`, `huggingface-hub==1.32.0`, imported in the isolated environment (Python 3.12.12) that Section 1
   built from the hash-locked `tutorials/requirements-colab.lock.txt`;
5. verify every default-path stage completes:
   - the isolated environment built from the carried lock (the `uv` wheel and the lock verified by digest) with no
     GitHub access, and every later cell routed to it;
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
   - Section 6 printing the frozen model's validation loss under five draw seeds (≈ 0.115–0.116), then `pipe.adapt`
     printing epoch 0 as the frozen model (≈ 0.1162), 852,867 trainable of 3,838,563 parameters, 32 steps, the upstream
     loss, frozen BatchNorm statistics, a four-epoch history and the deterministic selection rule; on the CPU the
     trained epochs score slightly worse and epoch 0 is kept (on a GPU a trained epoch within float16 noise of epoch 0
     may be kept instead);
   - `pipe.evaluate` on the test crops with the three-way comparison, the selection margin beside the draw-noise span,
     the run summary and run history, and `outputs/cfnet_change_detection_evaluation_report.json` written (the cell
     stops with a named cause if the kept epoch's validation loss is higher than the frozen model's or the validation
     F1 differs from the history by 0.01 or more);
   - the change maps of two held-out crops written beside their dates and labels with
     `outputs/cfnet_change_detection_predictions.json`;
   - `pipe.save_artifact` writing `outputs/cfnet_change_detection_adapter/{adapter.safetensors,manifest.json}`
     (72 tensors, about 3.4 MB), and `CFNetChangePipeline.from_artifact` reloading it with held-out metrics and
     change maps matching the adapted model (the cell stops with a named cause unless the F1 difference is below 10⁻³
     and the maximum map difference below 10⁻²), the previous `result.json` removed first and the run summary printed
     last;
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
| `cfnet_change_detection_colab.ipynb` (`E2E`) | `b92e01f` / `9e515a82` | 2026-09-20 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-cfnet-change-detection` v1; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `pillow 11.3.0`, `numpy 2.5.3` after, Python 3.12.13, `cuda`) | **PASSED after a manual restart — not a one-pass Run all, not promotion evidence** (notebook review CFN-M1, 2026-10-02): pass 1 stopped in cell 3 with the restart `RuntimeError` (`cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3`) after 165.9 s; pass 2, after a restart, 11/11 code cells ok in 105.1 s; 204 files, 3880 MB fetched into a clean runtime (the Hub snapshot, the 3.8 GB tarball and the 192 extracted members); comparison test changed-class F1 / IoU (all-unchanged baseline 0 / 0): frozen 0.9352 / 0.8782 → adapted 0.9352 / 0.8783, precision 0.9423 → 0.9424, recall 0.9281 → 0.9282, accuracy 0.9822 → 0.9822 vs baseline 0.8615 (best epoch 2, validation loss 0.1162 → 0.1141, validation F1 0.9151 → 0.9137 (that epoch choice was within the validation loss's draw noise — the criterion's pixel-pair draws advanced from epoch to epoch; review CFN-M2)); reload parity f1_diff: 0.0, metrics_identical: True, max_abs_map_diff: 0.0; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-cfnet-change-detection/v1/evidence/` in the workspace |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/cfnet_change_detection_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/cfnet_change_detection_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-20 | `b92e01f` / `9e515a82` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-cfnet-change-detection` v1; image `torch 2.10.0+cu128` before the pinned install, `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `pillow 11.3.0`, `numpy 2.5.3` after, Python 3.12.13, `cuda`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution); the checkpoint audited and converted in the notebook, the tarball fetched and the pinned members extracted by the notebook — the notebook's first execution anywhere | 271.0 s | **PASSED after a manual restart — not a one-pass Run all, not promotion evidence** (notebook review CFN-M1, 2026-10-02): pass 1 stopped in cell 3 with the restart `RuntimeError` (`cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3`) after 165.9 s; pass 2, after a restart, 11/11 code cells ok in 105.1 s; 204 files, 3880 MB fetched into a clean runtime (the Hub snapshot, the 3.8 GB tarball and the 192 extracted members); comparison test changed-class F1 / IoU (all-unchanged baseline 0 / 0): frozen 0.9352 / 0.8782 → adapted 0.9352 / 0.8783, precision 0.9423 → 0.9424, recall 0.9281 → 0.9282, accuracy 0.9822 → 0.9822 vs baseline 0.8615 (best epoch 2, validation loss 0.1162 → 0.1141, validation F1 0.9151 → 0.9137 (that epoch choice was within the validation loss's draw noise — the criterion's pixel-pair draws advanced from epoch to epoch; review CFN-M2)); reload parity f1_diff: 0.0, metrics_identical: True, max_abs_map_diff: 0.0; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-cfnet-change-detection/v1/evidence/` in the workspace |
| 2026-10-03 | review-fix commit on `review/cfnet_change_detection_colab-2026-10-02` / `90325e06` | Local CPU pre-flight — **not clean-runtime evidence, not promotion evidence**: Windows workstation (shared, 24 threads), CPython 3.12.10, `torch 2.14.0+cpu`, `torchvision 0.29.0+cpu`, `numpy 2.5.3`, `pillow 11.3.0`; the learner cells executed verbatim from the notebook JSON in one namespace, the two kernel cells not run (the isolated environment was not built; `DIMER_NOTEBOOK_CI_PREINSTALLED=1`); checkpoint and the 192 pinned crops pre-staged (tarball not fetched), the conversion done by the notebook. The executed blob `72532038` differs from `90325e06` only in one Prerequisites sentence (these timings) | Default path; then **Run after** from Section 6 with `TRAINABLE = 'decoders'` and then with `LEARNING_RATE = 1e-3`; then BYOD: a 7-pair zip of real LEVIR-CD crops (paths in a sub-folder, relative to `pairs.csv`) through Sections 4–8, a 6-pair zip and a zip missing one file | 106.2–127.7 s (default path) | **PASSED (pre-flight)** — default learner cells all ok: frozen test F1 / IoU 0.9352 / 0.8782 vs all-unchanged accuracy 0.8615; frozen validation loss over five draw seeds 0.11518–0.11621 (span 0.00103); epochs 0.11621 → 0.11626 → 0.11637 → 0.11649 → 0.11658, **epoch 0 kept**, adapted = frozen (ΔF1 0, ΔIoU 0); adapter 72 tensors, 3,419,684 bytes; reload parity f1_diff 0.0, max_abs_map_diff 0.0. Each re-run printed epoch 0 = 0.11621 (the pretrained model), kept epoch 0, passed reload parity and rewrote `result.json` (`decoders`: 198 tensors; `1e-3`: validation loss up to 0.12181, validation F1 down to 0.8888). BYOD 7 pairs → 4 / 1 / 2, Sections 4–8 complete, parity exact; the 6-pair zip refused naming the minimum of 7; the missing file refused naming its row and path. |
| 2026-10-03 | `ee1708b` / `90325e06` (PR #7 head; blob verified before the VM was allocated) | Google Colab CLI 0.7.4 sequential execution on a fresh Colab VM, Tesla T4 (`colab new --gpu T4`, `colab exec -f`, `colab stop`, workspace `colab-cli-serial-test-suite`); kernel Python 3.13.15, isolated uv environment CPython 3.12.12 with 45 locked packages, `torch 2.14.0+cu130`, `torchvision 0.29.0+cu130`, `cuda`; not a browser `Run all` and no execution counts — order is evidenced by the CLI's `Executing cell k/13` log | Default settings only (`TRAINABLE = 'change_decoder'`, `LEARNING_RATE = 1e-5`, `EPOCHS = 4`, built-in LEVIR-CD sample, BYOD off), no repository checkout, empty cache: the Hub snapshot (2 files) fetched and verified at `c232794`, the checkpoint audited and converted in the notebook, the tarball fetched and the 64 pinned crops extracted (59.6 s) | 176.3 s for the whole session including VM allocation (environment setup 57 s, adaptation 5.6 s) | **PASSED — one pass, no restart, 0 errors; 13/13 code cells** (cells 4–7 are the carried `metrics` / `modeling` / `pipeline` / `samples` module definitions and print nothing by design). Split 32 / 8 / 24; refusal probes refused as designed. Test changed-class F1 / IoU: all-unchanged baseline 0 / 0 (accuracy 0.8615), frozen 0.9352 / 0.8782 (precision 0.9423, recall 0.9281, accuracy 0.9822), frozen validation F1 0.9151. Frozen validation loss over five draw seeds 0.11519–0.11621 (span 0.00102); epochs 0.11621 (epoch 0) → 0.11625 → 0.11635 → 0.11640 → 0.11642, **epoch 0 kept** (float16 autocast did not let a trained epoch win), adapted = frozen (ΔF1 0, ΔIoU 0, margin 0); adapter 72 tensors, 3,419,684 bytes, `best_epoch` 0; reload parity f1_diff 0.0, metrics_identical True, max_abs_map_diff 0.0. **Differences from the worked answers** (quoted from the 2026-10-03 local CPU pre-flight): precision 0.9423 vs 0.9424, validation F1 0.9151 vs 0.9152, draw-seed minimum 0.11519 vs 0.11518, trained-epoch validation losses 0.11625 / 0.11635 / 0.11640 / 0.11642 vs 0.11626 / 0.11637 / 0.11649 / 0.11658 — GPU float16 differences of the size the Troubleshooting section names; the kept epoch, the test F1 / IoU and the conclusions are unchanged, and the notebook text was not edited. Evidence, byte-for-byte copies: [`execution-evidence/2026-10-03/cfnet_change_detection_colab_ee1708b_colab-cli-t4.ipynb`](execution-evidence/2026-10-03/cfnet_change_detection_colab_ee1708b_colab-cli-t4.ipynb) SHA-256 `36109de977fde492a752ef4d0c03aba76ed0858460976aa2a2281aca6b680f15`, `…_run_summary.json` `1f9fdf42a336beaceb26b978563b845213d9d102a9b9d2606dd818186b3e65b0`, `…_exec.log` `37d3855f3faed1d111e814fae8ad486b3a2d805df0c0c21d913ce00926b78d7c`. Not exercised: BYOD (positive and negative), the Section 9 experiments and any non-default settings |

## Current status

**Candidate.** The notebook was regenerated on 2026-10-03 for the notebook review of 2026-10-02 (CFN-M1..M5, CFN-m1..m5): a uv isolated environment instead of the in-kernel install, a deterministic epoch-selection criterion, a reset to the pretrained model before Sections 5 and 6, BYOD fixes and the guided layer. Its blob `90325e06` (commit `ee1708b`) passed one hosted Colab CLI sequential execution on a fresh Tesla T4 on 2026-10-03 — 13/13 code cells, one pass, no restart, 0 errors, epoch 0 kept (see "Recorded executions"); that is not a browser `Run all`, and BYOD was not exercised, so the status stays **Candidate**. Promotion needs a one-pass hosted `Run all` of the current blob recorded above, plus a BYOD run through Section 8 with at least one refusal. The 2026-09-20 Kaggle T4 run of blob `9e515a82` (committed at `b92e01f`) needed a manual restart after the install cell, so it is not a one-pass `Run all` and not promotion evidence.
