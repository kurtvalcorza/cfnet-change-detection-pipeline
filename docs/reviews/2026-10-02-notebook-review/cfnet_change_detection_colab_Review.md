# CFNet Bi-temporal Change Detection E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 2 October 2026  
**Repository:** `kurtvalcorza/cfnet-change-detection-pipeline`  
**Notebook:** `tutorials/cfnet_change_detection_colab.ipynb`  
**Reviewed commit:** `bb6be09f8f0eaeb749ad11e163a401553d5b86fc` (`main`, confirmed with `gh api repos/kurtvalcorza/cfnet-change-detection-pipeline/commits/main`)  
**Notebook Git blob:** `9e515a82c796270c70049dd9db177ee07c6b74d8`. This is the blob committed at `b92e01f` (merged as `5c2b195`) and executed in the recorded Kaggle run of 2026-09-20. The later commits on `main` touch only the model card, the README and the validator/tests (`9ee55de`: `tests/test_weight_facts.py`). Generator `--check` and `tools/validate_release_assets.py` both exit 0 at the reviewed commit.  
**Finding prefix:** `CFN`

## Executive assessment

On the default path the engineering is careful and the run reproduces. The notebook carries its four modules byte for byte, digest-verifies a two-file snapshot, statically audits the pickled checkpoint (four torch globals, 0 violations, audit digest `5b9f0ba0…`), converts it once through the weights-only loader into a pinned safetensors file, streams exactly 192 pinned members out of a digest-pinned 3.8 GB tarball, keeps the dataset's own splits (32 / 8 / 24 crops from as many source pairs), scores an all-unchanged baseline on the same pixels, fine-tunes the change decoder with frozen BatchNorm statistics and reloads the safetensors adapter with asserted parity. The limits prose is unusually honest: it says the model trained on this dataset, that the crops were chosen for having change, and that the adaptation is a contract demonstration, not an improvement.

A direct CPU execution of all 11 code cells, verbatim, with the exact pins and the real checkpoint, reproduced the recorded comparison:

| Measure | This review (CPU) | Kaggle T4 record |
|---|---|---|
| Test F1 / IoU, frozen → adapted (baseline 0 / 0) | 0.9352 / 0.8782 → 0.9352 / 0.8783 | 0.9352 / 0.8782 → 0.9352 / 0.8783 |
| Test precision, frozen → adapted | 0.9424 → 0.9421 | 0.9423 → 0.9424 |
| All-unchanged accuracy | 0.8615 | 0.8615 |
| Validation loss by epoch (kept) | 0.1162 → 0.1165 → **0.1141** → 0.1158 → 0.1162 (epoch 2) | 0.1162 → 0.1165 → **0.1141** → 0.1158 → 0.1161 (epoch 2) |
| Validation F1, frozen → kept epoch | 0.9152 → 0.9136 | 0.9151 → 0.9137 |
| Reload parity | f1_diff 0.0, max map diff 0.0 | f1_diff 0.0, max map diff 0.0 |
| Cell time | 27 s (crops from cache, no tarball) | 271 s over two passes |

Five problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (CFN-M1).** The recorded qualification run of this exact blob stopped in cell 3 with the restart `RuntimeError` (`cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3`) after 165.9 s and passed only on a second attempt. The release records call it a PASS and the procedure calls the restart "expected".
2. **The epoch-selection criterion is random, and the notebook teaches its noise as signal (CFN-M2).** Validation loss includes the content-consistency terms, which sample random pixel pairs from a generator that advances through training. Re-scored with the same 10 generator seeds, the kept epoch's validation loss is **higher** than the frozen model's on every seed (+0.00016), and its deterministic change term is higher too (0.01614 vs 0.01598). The "fall from 0.1162 to 0.1141" that the notebook calls "the sign that … validation selection [is] doing its job" is smaller than the seed-to-seed spread of the frozen model alone (0.11421–0.11686). The documented `LEARNING_RATE = 1e-3` experiment promises that "the frozen model keeps the kept epoch"; in this review it kept epoch 2, with validation F1 0.9036 vs 0.9152 and test F1 0.9333 vs 0.9352.
3. **Reruns reuse the adapted model and call it "frozen" (CFN-M3).** Verified: the documented `TRAINABLE = 'decoders'` rerun starts from the adapted weights and labels them `frozen model`; switching back to the default scope afterwards exports an adapter that omits the modified content decoders, and Section 8 ends in a bare `AssertionError` (max map difference 0.026 > 0.01), leaving the previous run's `result.json` in `outputs/`.
4. **BYOD cannot get past Section 5 (CFN-M4).** Verified: every BYOD upload stops in cell 17 with `KeyError: 'source_id'` (and cell 23 has the same lookup). The stated minimum is four pairs; the enforced minimum is seven. With a test split under four pairs, all three "refusal probes" are rejected for the record count, not the condition they name. A missing or subfolder member raises a bare `KeyError`.
5. **Guided layer missing (CFN-M5).** The notebook is declared `GUIDED`, but there is no audience statement, how-to-use, roadmap, glossary, prediction, checkpoint, troubleshooting section or conclusion template, and 2,538 lines of carried code are not labelled as infrastructure.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** (2026-09-26), `ml-worker` `origin/main` |
| Intended audience | Not stated. Prerequisites: co-registered bi-temporal pairs, binary change masks with an ignore class, F1 / IoU / precision / recall of a rare class against a majority baseline |
| Supported runtime | "Google Colab, or a Jupyter kernel with Python 3.12"; GPU (T4) or CPU; float16 autocast on CUDA; about 4 GB of disk for the tarball |
| Promised outcomes | One-pass `Run all` with no configuration edit; a pinned install; four carried modules; a digest-verified snapshot; static pickle audit and one-time conversion; 192 pinned members from the digest-pinned tarball; 32 / 8 / 24 crops with three refusals; frozen model vs all-unchanged baseline; bounded change-decoder fine-tuning with validation-loss epoch selection; a paired held-out comparison; change maps beside the dates and labels; a safetensors adapter with reload parity; `outputs/` exports; four optional experiments; BYOD "through the same contract — validation, frozen baseline, adaptation, held-out evaluation, change maps, artifact export and reload parity" with "at least four pairs" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; carried modules from `src/cfnet_change_detection_pipeline/`, recorded revision `71983b0` |

### Evidence actually obtained

- **Source inspection.** All 25 cells (11 code). The carried `pipeline.py` (`adapt`, `_loss`, `_content_loss`, `evaluate`, `predict`, `save_artifact`, `from_artifact`, `validate_dataset`), `samples.py` (`fetch_corpus`, `extract_pinned_members`, `split_dataset`, `load_byod_dataset`, `read_corpus`), `metrics.py`, the generator and template, `README.md`, `STATUS.md`, `MODEL_CARD.md`, `tutorials/README.md` and `docs/release-verification.md`. `docs/execution-evidence/` does not exist in this repository.
- **Documented execution evidence.**
  - Sources: `docs/release-verification.md` and the archived executor summary `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-cfnet-change-detection/v1/evidence/run_summary.json` (workspace).
  - The run: Kaggle Tesla T4, 2026-09-20, on **blob `9e515a82`, the reviewed blob** (`fetched_blob_verified: true`, clean HF cache).
  - Attempt 1 failed in cell 3 with the restart `RuntimeError` after 165.9 s.
  - Attempt 2 ran 11/11 cells in 105.1 s (`restarted_after_install_cell: true`).
  - No Colab run, no optional-experiment run and no BYOD run of this blob are recorded.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py` on Windows, CPython 3.12, in an existing workspace venv holding the notebook's exact pins with CPU builds (`torch 2.14.0+cpu`, `torchvision 0.29.0+cpu`, `numpy 2.5.3`, `pillow 11.3.0`, `safetensors 0.8.0`, `huggingface-hub 1.32.0`). CPU only, `HF_HUB_OFFLINE=1`. Nothing installed.
  - **Not a clean runtime:** the pinned checkpoint, its README and the 192 already-extracted pinned crops were hard-linked from the maintainer's cache into a scratch working directory. The tarball was **not** linked, so the tarball fetch, hash and stream were not exercised; `fetch_corpus` re-hashed every crop and used the cache. The notebook wrote its own manifest and converted the checkpoint itself (`fetched: []`, conversion record printed, `b348186e…`).
  - **Install skipped:** cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`.
  - **Execution method:** every code cell ran **verbatim from the notebook JSON** in one namespace. Only form-field literals were substituted, and `google.colab.files.upload` was replaced by a fake returning a zip built from real LEVIR-CD crops written in the documented BYOD shape.
  - **Probes run (total wall time about 7 minutes over three runs):**
    - P1, the default path, 27 s of cell time;
    - P2, the documented `TRAINABLE = 'decoders'` rerun of cells 19 / 21 / 23 after P1;
    - P3, the documented `LEARNING_RATE = 1e-3` rerun after P2, with `TRAINABLE` back at its default;
    - P4, `LEARNING_RATE = 1e-3` from a fresh base (cell 13 rerun);
    - P5, BYOD: the enforced minimum pair count, four invalid uploads, then cell 15 with `USE_BYOD = True` and cell 17;
    - P6, the selection criterion: the frozen model and the default kept epoch re-scored with 10 generator seeds, with the deterministic change term split off.
  - **Static checks:** JSON parse, blob id, generator `--check` (exit 0), `tools/validate_release_assets.py` (exit 0).
- **Not verified:** a Colab run of any kind; a one-pass hosted `Run all`; CUDA and float16 autocast in this review; the 3.8 GB tarball fetch, hash and stream; BYOD past Section 5; the real upload widget; learner understanding.

## 2. Separate judgments

- **Technical correctness:** sound on the default path. Identity, digests, the pickle audit and conversion, the pinned-member extraction, split integrity, the baseline, export and reload parity all hold, and direct execution matched the record. Defects:
  - the restart-dependent install (CFN-M1);
  - a stochastic selection criterion (CFN-M2);
  - state that survives a rerun, is mislabelled, and breaks export parity (CFN-M3);
  - the BYOD path (CFN-M4).
- **Scientific validity:**
  - **Sound:** held-out test split never used for selection; the all-unchanged baseline on the same pixels; one crop per source pair; explicit statements that the model trained on this dataset, that the crops were chosen for having change, and that 24 crops measure nothing beyond sanity.
  - **Not supported:** the reading of the training curve. The validation-loss "improvement" that selects the exported epoch is sampling noise in the content-consistency term; the kept epoch is slightly worse than the frozen model on the same draws and on the deterministic change term, and its validation F1 is lower (CFN-M2).
- **Promise fulfilment:**
  - **Delivered:** the default capability list.
  - **Not delivered:** one-pass `Run all` (CFN-M1).
  - **Not delivered as written:** the optional experiments (CFN-M2, CFN-M3).
  - **Not delivered:** BYOD past Section 4 (CFN-M4).
- **Learner experience:** clear stage prose, three "Look for" notes, explicit limits and three "things to carry to real data". There is no prediction, checkpoint, worked answer, troubleshooting section or conclusion template, and 2,538 carried lines sit unlabelled (CFN-M5). The guidance the learner is given on the training curve leads to a wrong conclusion (CFN-M2).
- **Spec conformance (2.2):** these applicable `MUST`s fail:

  | Finding | Failed `MUST`s |
  |---|---|
  | CFN-M1 | RUN1, RUN10, ENV6, REL2, REL11 |
  | CFN-M2 | ENV8, EVAL14 (selection basis), ART8 (selection basis recorded as a loss improvement that is not one) |
  | CFN-M3 | UX7, OUT8, ART8, VER4 (parity failure on a documented path) |
  | CFN-M4 | DAT12, DAT14, DAT19, REL12 |
  | CFN-m2 | UX12 |

  GDL1–GDL15 are largely unmet (SHOULD). The declared spec is 2.0 (CFN-S1).

## 3. Promise and objective tracing

| Claim (cell) | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass `Run all` in a fresh runtime (0) | Cell 3 pip-installs six pins into the kernel and raises on a changed loaded distribution | Kaggle attempt 1 stops in cell 3; attempt 2 passes after a restart | The opening promises no intervention; the release record calls it a PASS | **Fails** (CFN-M1) |
| Pickle audited and converted once, never served (0, 3) | `audit_pickle`, `convert_model`, strict load | Four globals, 0 violations, 776 tensors, `b348186e…` (P1 and record) | Printed before the model loads | Delivered |
| Exactly 192 pinned members from a pinned tarball (0, 4) | `fetch_tarball`, `extract_pinned_members` | Record: 3,831,872,824 bytes fetched and hashed; P1 used the crop cache | Clear | Delivered per record; tarball path not run here |
| Dataset's own splits, three refusals (4) | `read_corpus`, `dataset_manifest`, probes | 32 / 8 / 24, three rejections naming the condition (P1) | "Look for" note matches | Delivered on the sample; not under BYOD (CFN-M4) |
| Frozen model vs all-unchanged baseline (5) | `pipe.evaluate` | F1 0.9352 vs 0 (P1) | Baseline-first reading explained well | Delivered |
| Validation-loss epoch selection "doing its job" (6) | `adapt` keeps the lowest `val_loss()`; `_content_loss` draws random pixel pairs | The 0.1162 → 0.1141 drop is within the frozen model's own seed spread; the kept epoch is worse on the same draws (P6) | Learner reads noise as a working selection | **Misleading** (CFN-M2) |
| Paired held-out comparison (7) | `pipe.evaluate` + two assertions | 0.9352 → 0.9352 (P1) | Reported as flat, correctly | Delivered |
| Adapter export and reload parity (8) | `save_artifact`, `from_artifact`, assertion | Parity 0.0 on the default path (P1); fails after a documented rerun sequence (P3) | — | Default only (CFN-M3) |
| Optional experiments do not affect the default path (24) | Rerun of cells 19 → 23 on the live `pipe` | Starts from the adapted model labelled `frozen model` (P2); parity `AssertionError` (P3); `1e-3` keeps a degraded epoch (P4) | Learner told the opposite of what happens | **Fails** (CFN-M2, CFN-M3) |
| BYOD through the same contract, at least four pairs (0, 4) | Cell 15 BYOD branch, `split_dataset`, cells 17 / 23 | `KeyError: 'source_id'` in cell 17; minimum seven pairs (P5) | — | **Fails** (CFN-M4) |

| Objective (cell 0) | Learner activity | Evidence it is exercised |
|---|---|---|
| Install the pinned runtime | Run cell 3 | Needs a restart (CFN-M1) |
| Inspect the carried modules | Scroll 2,538 lines | No guidance on what to look at (CFN-M5) |
| Stage, verify, audit and convert a pickled checkpoint | Read the conversion record | Printed; no checkpoint question |
| Extract pinned members and validate pairs with an ignore class | Read the manifest and three refusals | Printed; the sample has 0 ignored pixels, so the ignore class is never exercised |
| Read F1 / IoU / precision / recall against the baseline | Read Section 5 output | Well explained; no question asked of the learner |
| Run bounded fine-tuning | Run cell 19, optionally change a form field | Runs; the reading of its output is wrong (CFN-M2); reruns are inconsistent (CFN-M3) |
| Compare adapted and frozen on the same pairs | Read Section 7 | Delivered |
| Write change maps; export and reload | Open the PNGs; read parity | Delivered on the default path |

## 4. Journeys

- **First-time learner (source inspection).** The opening is long but informative and the limits are stated before the experiment. A learner new to change detection meets "content decoder", "focuser", "tanh change map", "ignore index", "content-consistency terms" and "BatchNorm statistics" with brief explanations but no glossary. Nothing marks the 2,538 carried lines as safe to skip, there is no roadmap, no prediction before the comparison, and no conclusion template (CFN-M5). The Section 6 note tells the learner how to read the validation loss, and that reading is wrong (CFN-M2).
- **Clean default.** *Documented:* Kaggle T4 on the reviewed blob, attempt 1 failed in cell 3 with the restart `RuntimeError`, attempt 2 passed 11/11 (CFN-M1). *Direct (CPU, not clean):* 11/11 cells, 27 s of cell time, metrics equal to the record to four decimals except precision at the fourth decimal (CFN-m2). The final summary and `outputs/` (19 files) belong to the run. No Colab run.
- **Active learning (direct).** P2: after the default run, the documented `TRAINABLE = 'decoders'` change reruns from the adapted model; epoch 0 is printed and exported as `frozen model` with validation loss 0.11637, the seed-0 value of the previously kept weights (P6), not the frozen model's 0.11621. P3: the next documented change, `LEARNING_RATE = 1e-3` with the scope back at `change_decoder`, ends cell 23 in `AssertionError` (max map difference 0.0262), so the new `result.json` is never written and the previous one remains. P4: `1e-3` from a fresh base keeps epoch 2 (validation loss 0.11606 vs 0.11621), with validation F1 0.9036 and test F1 0.9333 / IoU 0.8749, against the promised "frozen model keeps the kept epoch". `EPOCHS` changes were not run.
- **Reuse and recovery (direct, Sections 4–5 only).** P5: the enforced minimum is seven pairs (4–6 refused with "split leaves N training pairs; at least 4 are required"). A 12-pair upload of real LEVIR-CD crops in the documented shape passes cell 15 with a 7 / 2 / 3 split, but all three refusal probes print `3 records; 4..2000 are required`, and cell 17 stops with `KeyError: 'source_id'`. Missing `pairs.csv` and a missing column give actionable messages; a missing member gives `KeyError: 'p0_after.png'`; subfolder paths give `KeyError: 'imgs/p0_before.png'`. Sections 6–9 under BYOD, the upload widget and non-LEVIR imagery were not verified.

## 5. Findings

### Major

#### CFN-M1 — `Run all` needs a manual restart after the install cell, and the release record calls it a pass

- **Cell/section:** Section 1, cell 3 (generated by `tools/build_notebook.py` lines 48–70 and the Section 1 prose at line 409); release records in `README.md` *Release status*, `STATUS.md`, `tutorials/README.md` and `docs/release-verification.md` (procedure step 4 and both evidence tables).
- **Observed issue:** cell 3 pip-installs the six pins into the running kernel. In a hosted image whose preloaded NumPy or CUDA bindings differ from the pins, it raises `RuntimeError: Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.`
- **Consequence:** the opening promises that "Selecting **Run all** in a fresh runtime" completes with no intervention. In the supported runtime class it does not. The release records nevertheless report `**PASSED** — 11/11 code cells ok (1 restart after install cell)`, and the procedure calls the restart "expected".
- **Evidence (documented):** `run_summary.json` for blob `9e515a82`: attempt 1 `ok: false` after 165.9 s with `cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3`; attempt 2 `ok: true` after a restart; `restarted_after_install_cell: true`.
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in the repository's notebook generator, regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:** a hosted Colab or Kaggle `Run all` of the regenerated blob, from a fresh runtime with an empty cache, completes all code cells in **one pass** (`restarted_after_install_cell: false`); the release records name that blob and no longer describe a restart as expected.
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11.

#### CFN-M2 — The epoch-selection criterion is stochastic, and the notebook teaches its noise as a working selection

- **Cell/section:** Section 6 prose ("Watch the validation loss … the sign that a small learning rate and validation selection are doing their job", template lines 234–237), cell 19, Section 7 prose and cell 21 assertion (template line 265), the `LEARNING_RATE = 1e-3` experiment (template line 404); carried `pipeline.py` `adapt` (`val_loss`), `_loss` and `_content_loss`.
- **Observed issue:** `val_loss()` is the full training loss: the change-map MSE plus `0.1 / n` times content-consistency terms computed over `n = W` **random pixel pairs** drawn from a generator seeded once per `adapt()` call and advanced by every training step. The same weights therefore score differently at different epochs. The kept epoch is the one whose draw happened to be lowest.
- **Consequence:** the exported adapter is chosen by sampling noise, and the notebook tells the learner the 0.1162 → 0.1141 drop shows the selection working. The learner draws the wrong conclusion about what validation selection did, and the documented `1e-3` experiment does not behave as promised.
- **Evidence (direct, P6 / P4):**
  - Frozen model, validation loss over 10 generator seeds: **0.11421–0.11686** (span 0.0027); deterministic change term **0.01598**.
  - Default kept epoch (epoch 2), same 10 seeds: **0.11437–0.11702**, higher than the frozen model on every seed (by 0.00016); change term **0.01614**; validation F1 0.9136 vs 0.9152.
  - The recorded selection margin, 0.1162 − 0.1141 = 0.0021, is inside the frozen model's own seed spread.
  - P4, `LEARNING_RATE = 1e-3` from a fresh base: epoch 2 kept on 0.11606 < 0.11621, with validation F1 0.9036 (frozen 0.9152) and test F1 / IoU 0.9333 / 0.8749 (frozen 0.9352 / 0.8782). The notebook promises "the frozen model keeps the kept epoch".
  - The cell 21 assertion (kept loss ≤ epoch-0 loss) is true by construction and so cannot catch this.
- **Recommended correction:** in `pipeline.py` `adapt`, compute the selection criterion deterministically — either the change-map term alone on the validation crops, or the full loss with the content-term pixel pairs drawn once from a fixed validation generator and reused at every epoch (including epoch 0). Record which in the adapter manifest. Then rewrite the Section 6 note and the `1e-3` experiment from a regenerated run, and say that a validation-loss difference smaller than its evaluation noise is not evidence of improvement.
- **Acceptance check:** calling the selection criterion twice on the same weights returns the same value; on the default sample, `history[0].val_loss` equals the criterion re-computed on the base model; the Section 6 and Interpretation prose quote numbers from a run of the regenerated blob and do not describe a within-noise change as selection working; the `1e-3` experiment description matches what a run of it prints.
- **Spec:** ENV8, EVAL14, ART8; GDL8.

#### CFN-M3 — Reruns of Section 6 start from the adapted model, label it "frozen", and can break export parity

- **Cell/section:** cells 19 / 21 / 23 and the Optional experiments paragraph (template line 403–404); BYOD "re-run from that cell" (template line 65); carried `pipeline.py` `adapt` (trains `self.model` as it is) and `save_artifact` (writes only the current scope's tensors).
- **Observed issue:** `adapt()` never restores the base weights. A rerun of Section 6 trains on top of the previous adaptation, and its epoch 0 is printed and exported as `frozen model`. Because `save_artifact` writes only the current scope's tensors, a `decoders` run followed by a `change_decoder` run exports an adapter that drops the changed content decoders.
- **Consequence:** each documented experiment silently compounds the previous one; the evaluation report mislabels the starting point; after the documented sequence, Section 8 fails and leaves the previous run's `result.json` in `outputs/`. A BYOD rerun "from that cell" (Section 4) would score the adapted model as the "frozen" model in Section 5 (inferred from source; BYOD did not reach Section 5, CFN-M4).
- **Evidence (direct):** P2 (`TRAINABLE = 'decoders'` after the default run): epoch 0 `note: frozen model`, validation loss 0.11637 — the previously kept weights at seed 0 (P6), not the frozen 0.11621; the evaluation report's `history[0]` carries the same label and value. P3 (`LEARNING_RATE = 1e-3`, scope back at `change_decoder`): cell 23 `AssertionError`, reload parity `max_abs_map_diff 0.0262`, `metrics_identical: False`.
- **Recommended correction:** in `pipeline.py`, have `adapt()` start from the verified base (reload the converted safetensors, or snapshot the base state once in `from_pretrained` and restore it at the start of every `adapt()`), or refuse to adapt an already adapted pipeline with an actionable message. State the rerun range for every optional experiment and for BYOD in the template prose.
- **Acceptance check:** running the documented experiments in the listed order, each from the stated rerun cell, gives an epoch-0 validation result equal to the frozen model's, completes Section 8 with the parity assertion passing, and rewrites `result.json`; a BYOD rerun from Section 4 reports the base model as frozen in Section 5.
- **Spec:** UX7, OUT8, ART8, VER4; GDL10.

#### CFN-M4 — BYOD stops in Section 5, and its stated contract does not match what is enforced

- **Cell/section:** cell 15 BYOD branch (template line 161) and refusal probes (template lines 175–178); cells 17 and 23 (`record['source_id']`, template lines 217 and 322); opening and Section 4 BYOD text ("at least four pairs", template lines 65–68); carried `samples.py` `split_dataset`, `load_byod_dataset`.
- **Observed issue:**
  1. BYOD records carry no `source_id` (only `read_corpus` sets it), and cells 17 and 23 index it unconditionally.
  2. `split_dataset` takes 25 % test and 20 % validation and requires four training pairs, so the effective minimum is seven pairs, not four.
  3. The refusal probes reuse `test_records[1:4]`; with fewer than four test pairs every probe is rejected for the record count, while the printout still names the condition being probed.
  4. A pair whose file is missing from the zip, or a `pairs.csv` that names subfolder paths, raises a bare `KeyError` (`load_byod_dataset` indexes members by basename).
- **Consequence:** the promised BYOD flow ("validation, frozen baseline, adaptation, held-out evaluation, change maps, artifact export and reload parity") cannot pass Section 5 for any upload. A learner who follows the stated minimum is refused, and a small upload shows refusals that teach the wrong reason.
- **Evidence (direct, P5):** pair counts 4, 5, 6 refused (`split leaves 2 / 3 / 3 training pairs; at least 4 are required`), 7 accepted (2 / 1 / 4). A 12-pair zip of real LEVIR-CD crops in the documented shape: cell 15 passes (test 3 / validation 2 / train 7); all three probes print `3 records; 4..2000 are required`; cell 17 `KeyError: 'source_id'`. Missing member → `KeyError: 'p0_after.png'`; subfolder paths → `KeyError: 'imgs/p0_before.png'`.
- **Recommended correction:** in the template, use `record.get('source_id', record['id'])` (or set `source_id` in `load_byod_dataset`); build the refusal probes so they do not depend on the test split's size (for example, validate the probed record alone with `min_records=1`); state the real minimum and the split fractions, or derive the minimum from them in `samples.py`; resolve members by their full path and raise a `ValueError` naming the missing file.
- **Acceptance check:** a BYOD zip with the documented minimum number of pairs, in the documented layout, runs Sections 4–8 to the parity assertion; each refusal probe prints the condition it names; a zip with a missing image and one with subfolder paths are each either accepted or refused with a message naming the file.
- **Spec:** DAT12, DAT14, DAT19, REL12; UX10.

#### CFN-M5 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** whole notebook; generator `tools/build_notebook.py` (opening and section scaffolding) and `tools/notebook_template.py`.
- **Observed issue:** no intended-learner statement (GDL1), no "How to use this notebook" (GDL2), no roadmap (GDL3), no Input → Model → Output contract (GDL4), no glossary (GDL6), no prediction before the frozen-vs-adapted comparison (GDL7), no interpretation checkpoints or worked answers (GDL9), no Predict → Change → Run → Observe → Explain activity (GDL10), the four carried-module cells (2,538 lines) not labelled as infrastructure (GDL11), no troubleshooting section (GDL13), no conclusion template (GDL14), no per-section synthesis (UX8). "Look for" notes exist in three places (GDL8, partially met).
- **Consequence:** a learner new to change detection gets a careful technical record but little help deciding what matters, what to predict, or how to state a conclusion; the optional experiments have no rerun instructions (CFN-M3).
- **Evidence (source inspection):** keyword scan of all 14 markdown cells: 0 occurrences of troubleshooting, prediction, checkpoint, glossary, how-to-use, infrastructure, roadmap or audience; 3 "Look for".
- **Recommended correction:** add the guided layer in the generator and template: audience and how-to-use in the opening, a roadmap, an Infrastructure label on the Section 1–3 cells saying they may be run without reading, a short glossary, a prediction prompt before Section 7, a checkpoint with a collapsible sample answer after Sections 5 and 7, one Predict → Change → Run → Observe → Explain activity with its rerun range, a troubleshooting section (restart, Hub download, 4 GB disk, digest mismatch, BYOD refusals), and a conclusion template.
- **Acceptance check:** each of GDL1–GDL4, GDL6, GDL7, GDL9–GDL11, GDL13 and GDL14 can be pointed to in the regenerated notebook.
- **Spec:** GDL1–GDL15 (SHOULD), UX5, UX8, UX9.

### Minor

#### CFN-m1 — Doubled braces in the data contract

- **Cell/section:** Prerequisites, cell 1 (template line 124).
- **Observed issue:** the record shape renders as `{{id, before, after, label}}`.
- **Consequence:** the one place that states the record contract shows a malformed literal.
- **Evidence (source inspection):** cell 1 text.
- **Recommended correction:** single braces in the template string (it is not a format string at that point).
- **Acceptance check:** the regenerated cell 1 shows `{id, before, after, label}`.
- **Spec:** DAT12.

#### CFN-m2 — Runtime and "build record" figures are unlabelled or do not match the record

- **Cell/section:** cells 0, 1, 16, 18 (template lines 121, 201, 237).
- **Observed issue:** "a CPU a few minutes" and four epochs "about a minute on a CPU" name no environment and are not labelled as estimates; this review measured 27 s of total cell time and 15 s for the four epochs on a CPU. Cell 16 gives the "build record" frozen precision as 0.9424; the recorded Kaggle run printed 0.9423 (this review's CPU run printed 0.9424).
- **Consequence:** small, but the learner cannot tell which run the quoted numbers come from.
- **Evidence:** direct (P1 timings) and documented (`run_summary.json`, cell 17).
- **Recommended correction:** label estimates as estimates and name the environment of each measured figure; quote "build record" figures from the recorded run.
- **Acceptance check:** every time and metric quoted in the prose names its environment or says "estimate", and quoted build-record figures match `docs/release-verification.md`.
- **Spec:** UX12.

#### CFN-m3 — "Hub only, ~16 MB" statements contradict the 3.8 GB data fetch

- **Cell/section:** cell 0 standalone paragraph ("Its only external dependencies are … the Hugging Face Hub at the immutable revision `c23279…` (~16 MB …)", `tools/build_notebook.py` line 384) and the last Prerequisites bullet ("the Hugging Face Hub only … (~16 MB in total)", line 396).
- **Observed issue:** both generator-appended sentences describe only the model snapshot; the bullet just above (template line 126) correctly states the 3.8 GB dataset tarball.
- **Consequence:** a learner planning bandwidth or disk reads two contradictory figures in the Prerequisites.
- **Evidence (source inspection).**
- **Recommended correction:** let the template supply the data dependency to the generator's standalone paragraph and External access bullet, or merge the two External access bullets.
- **Acceptance check:** every statement of the notebook's external dependencies includes the dataset tarball and its size.
- **Spec:** RUN14 (access conditions stated), UX1.

#### CFN-m4 — Recorded generating revision is not in `main`'s history

- **Cell/section:** `NOTEBOOK_SOURCE.repository_revision`, `metadata.dimer.generated_from.revision`, Section 2 heading (`71983b0`).
- **Observed issue:** `71983b0` is a commit of the squash-merged PR #1; it is fetchable from GitHub but is not an ancestor of `main` (`git branch -r --contains` returns nothing). The carried modules are identical to `main`'s (per-module SHA-256 match).
- **Consequence:** provenance points at a revision that a reader cannot reach from `main`, and could be lost if the PR ref were garbage-collected.
- **Evidence (direct):** `git branch -r --contains 71983b0` (empty); `gh api …/commits/71983b0` resolves.
- **Recommended correction:** regenerate on `main` after the CFN-M1 change so the recorded revision is a `main` commit (the generator's `--check` already ties the notebook to its recorded revision).
- **Acceptance check:** `git merge-base --is-ancestor <recorded revision> origin/main` exits 0.
- **Spec:** OUT6, OUT7.

#### CFN-m5 — BYOD zip has no expanded-size or member-count ceiling

- **Cell/section:** carried `samples.py` `load_byod_dataset`; cell 15.
- **Observed issue:** the zip is read with `ZipFile.read` for each named member with no limit on the archive's expanded size or member count; only the record count (2,000) and image sides (≤ 2048) are bounded, and 2,000 pairs at 2048² would exceed hosted memory well before validation finishes. (Members are decoded from bytes, never extracted, so path traversal is not possible.)
- **Consequence:** an oversized upload fails late with an out-of-memory crash rather than an actionable refusal.
- **Evidence (source inspection).**
- **Recommended correction:** check `sum(info.file_size)` and the member count against stated ceilings before reading, and state the ceilings in Section 4.
- **Acceptance check:** a zip above the stated expanded-size ceiling is refused with a message naming the ceiling before any image is decoded.
- **Spec:** VAL6, DAT19.

### Suggestions

- **CFN-S1 — Update the declared spec.** The notebook, `tutorials/README.md` and the validator declare NOTEBOOK_SPEC 2.0; the current baseline is 2.2.
- **CFN-S2 — Exercise the ignore class.** The objectives promise validating pairs "with an ignore class", but every sample crop has 0 ignored pixels; one probe or one sample crop with `-1` pixels would make the objective observable.
- **CFN-S3 — Show dispersion for the paired comparison.** A per-crop bootstrap of the frozen-minus-adapted F1 would let the learner see that 0.9352 → 0.9352 is within noise rather than having to take it on trust.
- **CFN-S4 — Record per-stage wall times in `result.json`.** The tarball fetch dominates the run; recording it separately would make the release record's 271 s interpretable.

## 6. Readiness

**Needs revision.** Five Major findings are open, and the applicable `MUST`s RUN1, RUN10, ENV6, REL2, REL11 (CFN-M1), ENV8, EVAL14, ART8 (CFN-M2), UX7, OUT8, VER4 (CFN-M3) and DAT12, DAT14, DAT19, REL12 (CFN-M4) are not met. The recorded execution evidence is for the reviewed blob but documents a restart, so it does not satisfy a one-pass `Run all`.

Remaining gates after the fixes: a one-pass hosted `Run all` of the regenerated blob recorded in `docs/release-verification.md`; a BYOD run through Section 8 with representative pairs and at least one clear refusal (REL12); the documented experiments re-run from their stated cells.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, not a clean runtime):** the default path's metrics and parity; the conversion record; the rerun mislabelling and the parity failure after the documented sequence; the stochastic selection criterion and its size; the `1e-3` outcome; the BYOD minimum, the probe mislabelling, the `KeyError: 'source_id'` and the bare `KeyError`s.
- **Verified from documented evidence:** the restart in the recorded Kaggle run of the reviewed blob.
- **Inferred from source:** BYOD Sections 6–9 behaviour after the `source_id` fix; the BYOD rerun scoring the adapted model as frozen; the zip-size failure mode; GPU/float16 behaviour.
- **Not verified:** any Colab run; the tarball fetch and stream in this review; learner understanding.
- **Finding most likely to be wrong:** CFN-M2's severity. The seed-spread numbers are robust (10 seeds, the same weights, a deterministic MSE term that moved the wrong way), but on a GPU with float16 autocast the noise and the kept epoch may differ; the claim that the selection is noise-driven rests on CPU float32 scoring of the sample's eight validation crops.
