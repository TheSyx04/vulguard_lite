# VulGuard Lite

A lightweight toolkit for **Just-in-Time Vulnerability Prediction (JIT-VP)** —
training, evaluating, and running full experiments with effort-aware threshold
calibration on commit-level vulnerability classifiers.

---

## Table of Contents

1. [Installation](#installation)
2. [How to Run](#how-to-run)
   - [Full Experiment Pipeline](#1-full-experiment-pipeline)
   - [Train Only](#2-train-only)
   - [Evaluate Only](#3-evaluate-only)
3. [Sub-commands](#sub-commands)
4. [Argument Reference](#argument-reference)
   - [Global Flags](#global-flags)
   - [Common Arguments](#common-arguments)
   - [experiment Arguments](#experiment-arguments)
   - [training Arguments](#training-arguments)
   - [evaluating Arguments](#evaluating-arguments)
5. [Supported Models](#supported-models)
6. [Dataset Format](#dataset-format)
7. [Output Layout](#output-layout)

---

## Installation

**Prerequisites:** Python 3.8+

```bash
# 1. Clone the repo
git clone https://github.com/TheSyx04/vulguard_lite.git
cd vulguard_lite

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the package in editable mode
pip install -e .
```

After installation the package can be invoked as a module:

```bash
python -m vulguard_lite --help
```

---

## How to Run

All commands follow this pattern:

```
python -m vulguard_lite <sub-command> [arguments...]
```

### 1. Full Experiment Pipeline

The `experiment` sub-command runs the complete **train → calibrate → test** loop
one or more times and writes aggregated results to CSV.

```bash
python -m vulguard_lite experiment \
  -repo_name   linux \
  -repo_language C \
  -model       lapredict \
  -device      cpu \
  -dg_save_folder ./output \
  -hf_repo_id  TheSyx/vulguard_lite \
  -hf_split_path linux/linux_3_1 \
  -runs        3 \
  -sampling    True \
  -sampling_seed 0 \
  -budget      0.05 0.1 0.2 \
  -calibration_range 0 1 10001
```

**With local dataset files (no Hugging Face):**

```bash
python -m vulguard_lite experiment \
  -repo_name   myrepo \
  -repo_language Python \
  -model       tlel \
  -train_set   /data/train_Kamei_features_myrepo.jsonl \
  -val_set     /data/val_Kamei_features_myrepo.jsonl \
  -test_set    /data/test_tlel_myrepo.jsonl \
  -runs        3 \
  -budget      0.2 \
  -calibration_range 0 1 10001
```

**With GPU (deep learning models):**

```bash
python -m vulguard_lite experiment \
  -repo_name   linux \
  -repo_language C \
  -model       jitfine \
  -device      cuda \
  -dg_save_folder ./output \
  -hf_repo_id  TheSyx/vulguard_lite \
  -hf_split_path linux/linux_3_1 \
  -epochs      50 \
  -runs        5 \
  -sampling    True \
  -sampling_seed 42 \
  -budget      0.05 0.1 0.2 0.5 1.0 \
  -calibration_range 0 1 10001
```

**Resume an interrupted experiment:**

```bash
python -m vulguard_lite experiment \
  -repo_name   linux \
  -repo_language C \
  -model       deepjit \
  -device      cuda \
  -hf_repo_id  TheSyx/vulguard_lite \
  -hf_split_path linux/linux_3_1 \
  -runs        5 \
  -epochs      30 \
  -budget      0.1 0.2 \
  -resume_from_checkpoint True
```

> If a run already has a metric file it is skipped entirely.
> If training was interrupted mid-epoch, it resumes from the last saved checkpoint.

---

### 2. Train Only

```bash
python -m vulguard_lite training \
  -repo_name   linux \
  -repo_language C \
  -model       tlel \
  -train_set   /data/train_Kamei_features_linux.jsonl \
  -val_set     /data/val_Kamei_features_linux.jsonl \
  -epochs      30
```

The best model checkpoint is saved to
`<dg_save_folder>/dg_cache/save/<repo_name>/models/best_epoch/`.

---

### 3. Evaluate Only

```bash
python -m vulguard_lite evaluating \
  -repo_name   linux \
  -repo_language C \
  -model       tlel \
  -test_set    /data/test_tlel_linux.jsonl \
  -threshold   0.5
```

**With threshold calibration:**

```bash
python -m vulguard_lite evaluating \
  -repo_name   linux \
  -repo_language C \
  -model       tlel \
  -test_set    /data/val_Kamei_features_linux.jsonl \
  -calibrated  True \
  -budget      0.2 \
  -calibration_range 0 1 10001
```

---

## Sub-commands

| Sub-command | Description |
|---|---|
| `experiment` | Full pipeline: training → validation calibration → test. Supports multiple runs and budget sweeps. |
| `training` | Fit a model on the training set and save the best checkpoint. |
| `evaluating` | Run inference on a test (or val) set, compute metrics, and optionally calibrate the decision threshold. |

---

## Argument Reference

### Global Flags

Applied to the top-level command (before the sub-command name).

| Flag | Type | Default | Description |
|---|---|---|---|
| `-version` | — | — | Print version and exit |
| `-debug` | flag | off | Enable verbose debug logging |
| `-log_to_file` | flag | off | Write logs to `<src>/logs/logs.log` instead of stdout |

---

### Common Arguments

Shared by **all** three sub-commands.

| Argument | Type | Default | Required | Description |
|---|---|---|---|---|
| `-repo_name` | str | `None` | **yes** | Short name of the repository (used for file naming and directory structure) |
| `-repo_language` | str | — | **yes** | Primary language. Choices: `Python`, `Java`, `C++`, `C`, `C#`, `JavaScript`, `TypeScript`, `Ruby`, `PHP`, `Go`, `Swift` |
| `-dg_save_folder` | str | `.` | no | Root directory for all cache and output files |
| `-seed` | int ≥ 0 | `42` | no | Global random seed applied to Python `random`, NumPy, and PyTorch at startup |
| `-mode` | str | `local` | no | Extractor mode. Choices: `local`, `remote` |
| `-repo_path` | str | `None` | no | Path to a local git repository (for data extraction) |
| `-repo_clone_url` | str | `None` | no | Remote URL to clone a repository from |
| `-repo_clone_path` | str | `None` | no | Local path where the repository should be cloned |
| `-hf_repo_id` | str | `None` | no | Hugging Face dataset repo ID to download train/val/test files (e.g. `TheSyx/vulguard_lite`) |
| `-hf_revision` | str | `main` | no | Branch, tag, or commit hash in the HF dataset repo |
| `-hf_split_path` | str | `None` | no | Sub-directory in the HF repo identifying a specific data split or cross-validation fold (e.g. `linux/linux_3_1`). The last path component becomes the split tag in output filenames. |
| `-hf_output_repo_id` | str | `None` | no | HF dataset repo to upload results to. Falls back to `-hf_repo_id` when omitted. |
| `-hf_upload_result` | bool | `False` | no | Push the experiment output folder to the HF dataset repo after all runs complete. |

---

### `experiment` Arguments

All **common arguments** plus:

| Argument | Type | Default | Required | Description |
|---|---|---|---|---|
| `-model` | str | — | **yes** | Model to train. See [Supported Models](#supported-models). |
| `-device` | str | `cpu` | no | PyTorch device: `cpu`, `cuda`, `cuda:0`, etc. |
| `-runs` | int ≥ 1 | `1` | no | Number of independent train → calibrate → test repetitions. |
| `-epochs` | int | `1` | no | Training epochs per run. |
| `-model_path` | str | `None` | no | Path to pre-trained model weights to warm-start training. |
| `-train_set` | str | `None` | no | Training JSONL file(s). When omitted, files are resolved from `-hf_repo_id`. Multi-file models (JITFine) accept a comma-separated pair: `features.jsonl,code.jsonl`. |
| `-val_set` | str | `None` | no | Validation set path (used for threshold calibration). Required when `-calibrated True` and `-hf_repo_id` is not set. |
| `-test_set` | str | `None` | no | Test set path for final evaluation. Required when `-hf_repo_id` is not set. |
| `-hyperparameters` | str | auto | no | Path to a `hyperparameters.json` file. Defaults to the model's bundled file. |
| `-dictionary` | str | `None` | no | Token dictionary file (required for `deepjit`, `simcom`; not used by `tlel`, `lapredict`, `lr`, `jitfine`). |
| `-sampling` | bool | `False` | no | Enable random undersampling of the majority class in the training set. |
| `-sampling_seed` | int ≥ 0 | `None` | no | Fixed seed for undersampling. When `None`, the global `-seed` is used. The **same** seed applies to every run (reproducible data splits). |
| `-sampling_seeds` | int(s) | `None` | no | List of undersampling seeds for multi-seed mode. Total runs = `len(seeds) × -runs`. Example: `-sampling_seeds 1 2 3 4 5`. Overrides `-sampling_seed`. |
| `-calibrated` | bool | `True` | no | Run threshold calibration on the validation set before testing. |
| `-threshold` | float | `0.5` | no | Fixed decision threshold used when `-calibrated False`. |
| `-budget` | float(s) in [0,1] | `1` | no | Inspection budget(s) for calibration: fraction of commits an inspector can review. Accepts multiple values: `-budget 0.05 0.1 0.2`. |
| `-calibration_range` | START END STEPS | `None` | no | Grid search range: three values `START END STEPS` (e.g. `-calibration_range 0 1 10001`). |
| `-resume_from_checkpoint` | bool | `False` | no | Skip completed runs; resume training from the latest epoch checkpoint for interrupted runs. |
| `-checkpoint_dir` | str | auto | no | Directory for epoch checkpoints. Defaults to `<experiment_root>/run_<N>/checkpoints/`. |
| `-hf_output_folder` | str | `None` | no | Custom remote folder inside the HF dataset repo for uploading results. Derived automatically when omitted. |

---

### `training` Arguments

All **common arguments** plus:

| Argument | Type | Default | Required | Description |
|---|---|---|---|---|
| `-model` | str | — | **yes** | Model to train. |
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

### `evaluating` Arguments

All **common arguments** plus:

| Argument | Type | Default | Required | Description |
|---|---|---|---|---|
| `-model` | str | — | **yes** | Model to evaluate. |
| `-device` | str | `cpu` | no | PyTorch device string. |
| `-threshold` | float | `None` | no | Fixed decision threshold. |
| `-model_path` | str | auto | no | Path to saved model weights. Defaults to `<dg_cache>/save/<repo_name>/models/best_epoch`. |
| `-test_set` | str | `None` | no | Test (or validation) JSONL file(s). Resolved from HF when omitted. |
| `-size_set` | str | `None` | no | JSONL file with `la` / `ld` columns per commit — required for Effort@20, Recall@20, and Popt metrics. |
| `-hyperparameters` | str | auto | no | Path to hyperparameters JSON. |
| `-dictionary` | str | `None` | no | Token dictionary path. |
| `-calibrated` | bool | `False` | no | Search for the best threshold and write it to `predict_scores/<model>_selected_threshold.json`. |
| `-budget` | float in [0,1] | `1` | no | Inspection budget used during threshold calibration. |
| `-runs` | int ≥ 1 | `1` | no | Number of evaluation runs. |
| `-calibration_range` | START END STEPS | `None` | no | Threshold grid-search range (same format as `experiment`). |

---

## Supported Models

| Model | Type | Needs dictionary | GPU | Primary input |
|---|---|---|---|---|
| `lapredict` | Sklearn (logistic regression) | No | No | Kamei features (`la` column only) |
| `lr` | Sklearn (logistic regression + scaler) | No | No | Kamei features (all 14 columns) |
| `tlel` | Sklearn (tree ensemble) | No | No | Kamei features |
| `deepjit` | Deep learning (CNN) | Yes | Yes | DeepJIT merge file |
| `jitfine` | Deep learning (RoBERTa + features) | No | Yes | Kamei features + merge file |
| `simcom` | Deep learning (Sim + Com fusion) | Yes | Yes | Kamei features + patch file |

**Kamei features** are 14 commit-level metrics: `ns`, `nd`, `nf`, `entropy`, `la`, `ld`, `lt`, `fix`, `ndev`, `age`, `nuc`, `exp`, `rexp`, `sexp`.

---

## Dataset Format

All dataset files are in **JSONL** format (one JSON object per line).

**Kamei features file** (`train_Kamei_features_<repo>.jsonl`):

```json
{"commit_id": "abc123", "label": 1, "ns": 2, "nd": 3, "nf": 5, "la": 100, "ld": 20, ...}
{"commit_id": "def456", "label": 0, "ns": 1, "nd": 1, "nf": 2, "la": 10,  "ld": 5,  ...}
```

**Merge / code file** (`train_merge_<repo>.jsonl`):

```json
{"commit_id": "abc123", "label": 1, "message": "fix buffer overflow", "diff": "..."}
```

### File naming on Hugging Face

| Split | File(s) |
|---|---|
| Train (Kamei) | `out_train_Kamei_features_<repo>.jsonl` |
| Train (code)  | `out_train_merge_<repo>.jsonl` |
| Train (patch) | `out_train_patch_<repo>.jsonl` |
| Val (Kamei)   | `out_val_Kamei_features_<repo>.jsonl` |
| Val (code)    | `out_val_merge_<repo>.jsonl` |
| Test (features) | `out_test_tlel_<repo>.jsonl` |
| Test (code)   | `out_test_deepjit_<repo>.jsonl` / `out_test_jitfine_<repo>.jsonl` |
| Dictionary    | `dict_<repo>.jsonl` |

Files are downloaded once and cached at `<dg_save_folder>/dg_cache/dataset/<repo_name>/hf/`.

### Undersampling cache

When `-sampling True`, balanced subsets are written to:

```
<dg_save_folder>/dg_cache/dataset/<repo_name>/sampled/
    <file>_undersampled_run_<N>.jsonl
```

For multi-file models (JITFine, SimCom), all paired files contain **exactly the same commit IDs** to prevent feature/code mismatch.

---

## Output Layout

```
<dg_save_folder>/dg_cache/
  dataset/<repo_name>/
    hf/                          <- downloaded HF dataset files (cache)
    sampled/                     <- undersampled training files
  save/<repo_name>/
    predict_scores/              <- raw prediction scores per phase
    results/                     <- metric CSVs per phase
    models/
      best_epoch/                <- best model weights (used for test)
      last_epoch/                <- model after the final epoch
      checkpoints/               <- per-epoch checkpoints (deep learning)
    experiments/<slug>/
      run_1/
        <model>_<budget>_val_scores.csv
        <model>_<budget>_val_metrics.csv
        <model>_<budget>_val_threshold_calibration.csv
        <model>_<budget>_selected_threshold.json
        <model>_<budget>_test_scores.csv
        <model>_<budget>_test_metrics.csv
        <model>_test_metrics.csv          <- aggregated summary for this run
      run_2/ ...
      <slug>_test_all_run.csv             <- single-budget final summary
      <slug>_test_all_budget.csv          <- multi-budget final summary
      <slug>_timing.log                   <- per-run timing log
```

`<slug>` = `<model>_<repo_name>_<split_tag>_<sampling_tag>`

Example: `lapredict_linux_linux_3_1_sampling`
