# VulGuard Lite

A lightweight toolkit for **Just-in-Time Vulnerability Prediction (JIT-VP)** —
training, evaluating, and running full experiments with effort-aware threshold
calibration on commit-level vulnerability classifiers.

---

## Table of contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Sub-commands](#sub-commands)
   - [experiment](#experiment)
   - [training](#training)
   - [evaluating](#evaluating)
4. [Argument reference](#argument-reference)
   - [Global flags](#global-flags)
   - [Common arguments](#common-arguments)
   - [experiment arguments](#experiment-arguments)
   - [training arguments](#training-arguments)
   - [evaluating arguments](#evaluating-arguments)
5. [Supported models](#supported-models)
6. [Dataset conventions](#dataset-conventions)
7. [Output layout](#output-layout)

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick start

### Run a full experiment (train → calibrate → test)

```bash
python -m vulguard_lite experiment \
  -repo_name linux \
  -repo_language C \
  -model lapredict \
  -device cpu \
  -dg_save_folder /path/to/output \
  -hf_repo_id TheSyx/vulguard_lite \
  -hf_revision main \
  -hf_split_path linux/linux_3_1 \
  -hf_upload_result True \
  -runs 3 \
  -epochs 30 \
  -sampling True \
  -sampling_seed 0 \
  -budget 0.05 0.1 0.2 \
  -calibration_range 0 1 10001
```

### Train only

```bash
python -m vulguard_lite training \
  -repo_name linux \
  -repo_language C \
  -model tlel \
  -train_set /data/train_Kamei_features_linux.jsonl \
  -val_set   /data/val_Kamei_features_linux.jsonl \
  -epochs 30
```

### Evaluate only

```bash
python -m vulguard_lite evaluating \
  -repo_name linux \
  -repo_language C \
  -model tlel \
  -test_set /data/test_tlel_linux.jsonl \
  -threshold 0.5
```

---

## Sub-commands

| Sub-command | Description |
|---|---|
| `experiment` | Full pipeline: training → validation calibration → test |
| `training` | Fit a model and save the best checkpoint |
| `evaluating` | Run inference on a test (or val) set and compute metrics |

---

## Argument reference

### Global flags

These flags apply to the top-level `vulguard_lite` command (before the
sub-command name).

| Flag | Type | Default | Description |
|---|---|---|---|
| `-version` | — | — | Print version and exit |
| `-debug` | flag | off | Enable verbose debug logging |
| `-log_to_file` | flag | off | Write logs to `<src>/logs/logs.log` instead of stdout |

---

### Common arguments

Shared by **all** three sub-commands.

| Argument | Type | Default | Required | Description |
|---|---|---|---|---|
| `-repo_name` | str | `None` | yes | Short name of the repository (used for file naming and directory structure) |
| `-repo_language` | str | — | yes | Primary language of the repository. Choices: `Python`, `Java`, `C++`, `C`, `C#`, `JavaScript`, `TypeScript`, `Ruby`, `PHP`, `Go`, `Swift` |
| `-dg_save_folder` | str | `.` | no | Root directory for all cache and output files |
| `-seed` | int ≥ 0 | `42` | no | Global random seed applied to Python `random`, NumPy, and PyTorch at startup |
| `-mode` | str | `local` | no | Extractor mode. Choices: `local`, `remote` |
| `-repo_path` | str | `None` | no | Path to a local git repository (for data extraction) |
| `-repo_clone_url` | str | `None` | no | Remote URL to clone a repository from |
| `-repo_clone_path` | str | `None` | no | Local path where the remote repository should be cloned |
| `-hf_repo_id` | str | `None` | no | Hugging Face dataset repo ID to download train/val/test files from (e.g. `TheSyx/vulguard_lite`) |
| `-hf_revision` | str | `main` | no | Branch, tag, or commit hash in the HF dataset repo |
| `-hf_split_path` | str | `None` | no | Sub-directory within the HF repo that identifies a specific data split or cross-validation fold (e.g. `linux/linux_3_1`). The last path component becomes the split tag used in output filenames. |
| `-hf_output_repo_id` | str | `None` | no | HF dataset repo to upload results to. Falls back to `-hf_repo_id` when omitted and `-hf_upload_result True` is set. |
| `-hf_upload_result` | bool | `False` | no | Push the experiment output folder to the HF dataset repo after all runs complete. Requires `-hf_repo_id` or `-hf_output_repo_id`. |

---

### `experiment` arguments

All **common arguments** plus:

| Argument | Type | Default | Required | Description |
|---|---|---|---|---|
| `-model` | str | — | yes | Model to train. See [Supported models](#supported-models). |
| `-device` | str | `cpu` | no | PyTorch device string: `cpu`, `cuda`, `cuda:0`, etc. |
| `-runs` | int ≥ 1 | `1` | no | Number of independent train → calibrate → test repetitions. Each run produces its own artifacts inside `run_<N>/`. |
| `-epochs` | int | `1` | no | Training epochs per run. |
| `-model_path` | str | `None` | no | Path to pre-trained model weights to warm-start training. |
| `-train_set` | str | `None` | no | Explicit path to the training JSONL file(s). When omitted, files are resolved from `-hf_repo_id`. Multi-file models (JITFine) accept a comma-separated pair: `features.jsonl,code.jsonl`. |
| `-val_set` | str | `None` | no | Validation set path (used for threshold calibration). Required when `-calibrated True` and `-hf_repo_id` is not set. |
| `-test_set` | str | `None` | no | Test set path for final evaluation. Required when `-hf_repo_id` is not set. |
| `-hyperparameters` | str | auto | no | Path to a `hyperparameters.json` file. Defaults to the model's bundled file inside the package. |
| `-dictionary` | str | `None` | no | Path to a token dictionary file (required for `deepjit`, `simcom`; not used by `tlel`, `lapredict`, `lr`, `jitfine`). |
| `-sampling` | bool | `False` | no | Enable random undersampling of the majority class in the training set. |
| `-sampling_seed` | int ≥ 0 | `None` | no | Fixed random seed for undersampling. When `None`, the global `-seed` value is used. The **same** seed is applied to every run (reproducible data splits; runs differ only in model initialisation). |
| `-calibrated` | bool | `True` | no | Run threshold calibration on the validation set before testing. Set to `False` to use a fixed threshold. |
| `-threshold` | float | `0.5` | no | Starting / fixed decision threshold (used when `-calibrated False` or as the initial value before calibration). |
| `-budget` | float(s) in [0,1] | `1` | no | Inspection budget(s) for threshold calibration: the fraction of commits an inspector can review. Pass multiple values to sweep (e.g. `-budget 0.05 0.1 0.2`). |
| `-calibration_range` | START END STEPS | `None` | no | Grid search range for the decision threshold. Three integers/floats: `START END STEPS` (e.g. `-calibration_range 0 1 10001` searches 10 001 evenly-spaced values in [0, 1]). |
| `-resume_from_checkpoint` | bool | `False` | no | If `True` and a completed run's metric file already exists, skip that run. Also resumes training from the latest epoch checkpoint when the model supports it. |
| `-checkpoint_dir` | str | auto | no | Directory for epoch checkpoints. Defaults to `<experiment_root>/run_<N>/checkpoints/`. |

---

### `training` arguments

All **common arguments** plus:

| Argument | Type | Default | Required | Description |
|---|---|---|---|---|
| `-model` | str | — | yes | Model to train. |
| `-device` | str | `cpu` | no | PyTorch device string. |
| `-epochs` | int | `1` | no | Number of training epochs. |
| `-threshold` | float | `None` | no | Decision threshold (stored with the model for later use). |
| `-model_path` | str | `None` | no | Path to pre-trained weights. |
| `-train_set` | str | `None` | no | Training JSONL file(s). Resolved from HF when omitted. |
| `-val_set` | str | `None` | no | Validation JSONL file(s). Resolved from HF when omitted. |
| `-hyperparameters` | str | auto | no | Path to hyperparameters JSON. |
| `-dictionary` | str | `None` | no | Token dictionary path (required for `deepjit`, `simcom`). |
| `-sampling` | bool | `False` | no | Enable random undersampling on the training set. |
| `-resume_from_checkpoint` | bool | `False` | no | Resume training from the latest checkpoint. |
| `-checkpoint_dir` | str | auto | no | Epoch checkpoint directory. |

---

### `evaluating` arguments

All **common arguments** plus:

| Argument | Type | Default | Required | Description |
|---|---|---|---|---|
| `-model` | str | — | yes | Model to evaluate. |
| `-device` | str | `cpu` | no | PyTorch device string. |
| `-threshold` | float | `None` | no | Fixed decision threshold. |
| `-model_path` | str | auto | no | Path to saved model weights. Defaults to `<dg_cache>/save/<repo_name>/models/best_epoch`. |
| `-test_set` | str | `None` | no | Test (or validation) JSONL file(s). Resolved from HF when omitted. |
| `-size_set` | str | `None` | no | JSONL file with `added_lines` / `deleted_lines` columns per commit — required for Effort@20, Recall@20, and Popt metrics. |
| `-hyperparameters` | str | auto | no | Path to hyperparameters JSON. |
| `-dictionary` | str | `None` | no | Token dictionary path. |
| `-calibrated` | bool | `False` | no | Search for the best threshold and write it to `predict_scores/<model>_selected_threshold.json`. |
| `-budget` | float in [0,1] | `1` | no | Inspection budget used during threshold calibration. |
| `-runs` | int ≥ 1 | `1` | no | Number of evaluation runs (averages metrics across runs). |
| `-calibration_range` | START END STEPS | `None` | no | Threshold grid-search range (see `experiment` description above). |
| `-hf_output_folder` | str | `None` | no | Explicit remote folder path inside the HF dataset repo for uploading results (e.g. `output/linux/deepjit/sampling/deepjit_linux_1_3_sampling`). When omitted the path is derived automatically from model, repo, split, and sampling settings. |

---

## Supported models

| Model name | Type | Training data |
|---|---|---|
| `tlel` | Sklearn (tree ensemble) | TLEL / Kamei features (`train_Kamei_features_<repo>.jsonl`) |
| `lapredict` | Sklearn (logistic regression) | Kamei features (same as TLEL) |
| `lr` | Sklearn (logistic regression) | Kamei features (same as TLEL) |
| `deepjit` | Deep learning (CNN) | DeepJIT merge (`train_merge_<repo>.jsonl`) |
| `jitfine` | Deep learning (RoBERTa + features) | **Both** Kamei features + DeepJIT merge |
| `simcom` | Deep learning | Kamei features + patch |

---

## Dataset conventions

Files are expected in JSONL format (one JSON object per line).  The naming
convention used when resolving files automatically from a Hugging Face
repository is:

| Split | Kamei / TLEL features | Code changes (merge) |
|---|---|---|
| Train | `out_train_Kamei_features_<repo>.jsonl` | `out_train_merge_<repo>.jsonl` |
| Val | `out_val_Kamei_features_<repo>.jsonl` | `out_val_merge_<repo>.jsonl` |
| Test (features) | `out_test_tlel_<repo>.jsonl` | — |
| Test (code) | — | `out_test_deepjit_<repo>.jsonl` |

Files are downloaded once and cached locally under
`<dg_save_folder>/dg_cache/dataset/<repo_name>/hf/`.

### Undersampling cache

When `-sampling True`, balanced subsets are written to:

```
<dg_save_folder>/dg_cache/dataset/<repo_name>/sampled/
    out_train_Kamei_features_<repo>_undersampled_run_<N>.jsonl
    out_train_merge_<repo>_undersampled_run_<N>.jsonl   # JITFine only
```

For multi-file models (JITFine), both files contain **exactly the same
commit IDs** to prevent feature / code mismatch.

---

## Output layout

```
<dg_save_folder>/dg_cache/
  dataset/<repo_name>/
    hf/                          ← downloaded HF files
    sampled/                     ← undersampled training files
  save/<repo_name>/
    predict_scores/              ← raw prediction CSVs (per run)
    results/                     ← metric CSVs (per run)
    models/
      best_epoch/                ← best model weights
      last_epoch/
      checkpoints/
    experiments/<slug>/
      run_1/
        <model>_<budget>_val_scores.csv
        <model>_<budget>_val_metrics.csv
        <model>_<budget>_val_threshold_calibration.csv
        <model>_<budget>_selected_threshold.json
        <model>_<budget>_test_scores.csv
        <model>_<budget>_test_metrics.csv
        <model>_test_metrics.csv
      run_2/ …
      <slug>_test_all_run.csv    ← single-budget final summary
      <slug>_test_all_budget.csv ← multi-budget final summary
```

where `<slug>` = `<model>_<repo_name>_<split_tag>_<sampling_tag>`.
