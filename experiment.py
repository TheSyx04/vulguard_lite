"""experiment.py — Full train → calibrate → test pipeline for VulGuard Lite.

Public entry point
------------------
    run_experiment(params)
        Orchestrates N repeated experiment runs for a single model on a single
        repository split.  Each run executes three phases:

        1. **Training** — fit the model on the (optionally undersampled) training
           set and save the best checkpoint to disk.
        2. **Calibration** (optional) — run inference on the validation set and
           choose the probability threshold that maximises the chosen effort
           metric under a given inspection *budget*.
        3. **Testing** — run inference on the test set with the calibrated (or
           fixed) threshold and record all evaluation metrics.

        Results for every run / budget combination are saved as CSV files inside
        an *experiment root* directory and, optionally, uploaded to a Hugging
        Face dataset repository.

Directory layout produced
-------------------------
    <dg_save_folder>/dg_cache/save/<repo_name>/experiments/<slug>/
        run_1/
            <model>_<budget>_val_scores.csv
            <model>_<budget>_val_metrics.csv
            <model>_<budget>_val_threshold_calibration.csv
            <model>_<budget>_selected_threshold.json
            <model>_<budget>_test_scores.csv
            <model>_<budget>_test_metrics.csv
            <model>_test_metrics.csv          ← aggregated per-run summary
        run_2/ …
        <slug>_test_all_run.csv               ← single-budget final table
        <slug>_test_all_budget.csv            ← multi-budget final table

where ``<slug>`` is ``<model>_<repo>_<split>_<sampling_tag>``.
"""

import json
import os
import shutil
from argparse import Namespace

import pandas as pd

from .evaluating import evaluating
from .training import training
from .utils.hf_upload import upload_folder_to_hf_dataset
from .utils.utils import create_dg_cache
from .utils.reproducibility import seed_everything


def _clone_params(params, overrides):
    """Return a shallow copy of *params* (an argparse Namespace) with *overrides* applied.

    Parameters
    ----------
    params : argparse.Namespace
        Original parameter namespace produced by the CLI parser.
    overrides : dict
        Key/value pairs that replace or extend the namespace for a sub-phase
        (e.g. setting ``model_path=None`` before training, or injecting a
        per-run ``sampling_run_id``).

    Returns
    -------
    argparse.Namespace
        New namespace — does **not** mutate *params*.
    """
    data = vars(params).copy()
    data.update(overrides)
    return Namespace(**data)


def _safe_remove(path):
    """Delete *path* if it exists; silently do nothing otherwise."""
    if os.path.exists(path):
        os.remove(path)


def _normalize_budgets(budget_value):
    """Coerce the CLI *budget* value to a list of floats.

    The experiment parser accepts ``-budget`` with ``nargs='+'``, so
    *budget_value* may already be a list when multiple budgets are given, or
    a bare scalar when only one is specified.

    Parameters
    ----------
    budget_value : float | list[float]
        Raw value from ``params.budget``.

    Returns
    -------
    list[float]
    """
    if isinstance(budget_value, (list, tuple)):
        budgets = list(budget_value)
    else:
        budgets = [budget_value]

    return [float(budget) for budget in budgets]


def _budget_tag(budget):
    """Return a filesystem-safe string label for a budget value.

    Examples::

        _budget_tag(0.05)   → "budget_0p05"
        _budget_tag(1.0)    → "budget_1"
        _budget_tag(0.1)    → "budget_0p1"
    """
    return f"budget_{budget:.4f}".replace(".", "p").rstrip("0").rstrip("p")


def _split_tag(params):
    """Derive a short tag that identifies the data split used for this run.

    Resolution order:

    1. If ``-hf_split_path`` is set, use its last path component
       (e.g. ``linux/linux_3_1`` → ``"linux_3_1"``).
    2. If ``-test_set`` is provided manually, return ``"manual"``.
    3. Otherwise return ``"default"``.
    """
    hf_split_path = getattr(params, "hf_split_path", None)
    if hf_split_path:
        return hf_split_path.rstrip("/").split("/")[-1]

    test_set = getattr(params, "test_set", None)
    if test_set:
        return "manual"

    return "default"


def _sampling_tag(params):
    """Return ``"sampling"`` if undersampling is enabled, else ``"no_sampling"``."""
    return "sampling" if getattr(params, "sampling", False) else "no_sampling"


def _experiment_slug(params):
    """Return a unique, human-readable identifier for this experiment configuration.

    Format: ``<model>_<repo_name>_<split_tag>_<sampling_tag>``

    Used as the name of the experiment root directory and as a prefix for
    summary CSV files.
    """
    return f"{params.model}_{params.repo_name}_{_split_tag(params)}_{_sampling_tag(params)}"


def _hf_output_path(params):
    """Return the remote path within the HF dataset repo where results are uploaded.

    If ``-hf_output_folder`` is supplied it is used as-is (leading/trailing
    slashes are stripped).  Otherwise the path is derived automatically:

        ``output/<repo_name>/<model>/<sampling_tag>/<slug>``
    """
    custom = getattr(params, "hf_output_folder", None)
    if custom:
        return custom.strip("/")
    return f"output/{params.repo_name}/{params.model}/{_sampling_tag(params)}/{_experiment_slug(params)}"


def _collect_metric_row(metrics_file, model_name, run_idx, budget, threshold, threshold_payload):
    """Read a per-run metrics CSV and attach run-level metadata columns.

    Parameters
    ----------
    metrics_file : str
        Path to the ``<model>.csv`` produced by ``evaluating()``.
    model_name : str
        Name of the model (written into the ``"model"`` column).
    run_idx : int
        1-based run index.
    budget : float
        Inspection budget for this calibration sweep (0–1).
    threshold : float
        Decision threshold used during the test evaluation.
    threshold_payload : dict | None
        Full threshold selection payload written by calibration, or ``None``
        when calibration was skipped.

    Returns
    -------
    dict
        One row suitable for appending to the cross-run summary DataFrame.
    """
    metrics_df = pd.read_csv(metrics_file, index_col=0).reset_index()
    metrics_df = metrics_df.rename(columns={"index": "model"})
    metrics_row = metrics_df.iloc[0].to_dict()
    metrics_row.update(
        {
            "model": model_name,
            "run": run_idx,
            "budget": float(budget),
            "threshold": float(threshold),
        }
    )

    return metrics_row


# ---------------------------------------------------------------------------
# Model-group constants (shared with training.py and evaluating.py)
# ---------------------------------------------------------------------------

# Models that do not use a dictionary file.
_NO_DICT_MODELS = {"tlel", "lapredict", "lr", "jitfine"}
# Models that do not use a hyperparameters.json file at all.
_NO_HYPERPARAMS_MODELS = {"lapredict", "lr"}
# Sklearn-based models: no GPU device, no epoch checkpoints.
_SKLEARN_MODELS = {"lapredict", "lr"}


def _resolve_hyperparameters(params):
    """Return the resolved hyperparameters path for the current model.

    - ``lapredict`` / ``lr``  → ``None`` (no hyperparameter file).
    - all others              → ``<SRC_PATH>/models/<model>/hyperparameters.json``.

    If ``params.hyperparameters`` is already set (e.g. passed via ``-hyperparameters``
    on the CLI), it is returned unchanged.
    """
    from .utils.utils import SRC_PATH

    if params.hyperparameters is not None:
        return params.hyperparameters
    if params.model in _NO_HYPERPARAMS_MODELS:
        return None
    return f"{SRC_PATH}/models/{params.model}/hyperparameters.json"


def run_experiment(params):
    """Run the full train → calibrate → test pipeline for one model / split.

    This function is registered as the ``func`` for the ``experiment`` CLI
    sub-command (see ``cli.py``).

    Pipeline overview (per run)
    ---------------------------
    1. **Training** — calls :func:`training` with the (optionally undersampled)
       training dataset.  The best model checkpoint is saved to
       ``<experiment_root>/run_<N>/checkpoints/``.
    2. **Calibration** (when ``-calibrated True``) — calls :func:`evaluating`
       on the validation set, searches for the optimal decision threshold over
       ``-calibration_range`` at the given ``-budget``, and writes the selected
       threshold to a JSON file inside ``run_<N>/``.
    3. **Testing** — calls :func:`evaluating` on the test set with the
       calibrated threshold (or a fixed 0.5 when calibration is disabled) and
       records all evaluation metrics.

    After all runs complete, per-budget average metrics are computed and the
    full summary table is written to ``<experiment_root>/<slug>_test_all_*.csv``.
    If ``-hf_upload_result True`` is set, the entire experiment directory is
    pushed to the specified Hugging Face dataset repository.

    Parameters
    ----------
    params : argparse.Namespace
        Parsed CLI arguments.  See the **Argument reference** section in
        ``README.md`` for the full list.

    Raises
    ------
    ValueError
        If required arguments are missing (e.g. ``-val_set`` when calibration
        is enabled and no ``-hf_repo_id`` is provided, or
        ``-dictionary`` for models that need one).
    FileNotFoundError
        If the threshold JSON file is not produced after calibration.
    """
    hf_repo_id = getattr(params, "hf_repo_id", None)
    if params.model not in _NO_DICT_MODELS and params.dictionary is None and hf_repo_id is None:
        raise ValueError(
            f"-dictionary is required for experiment mode when -model is {params.model}, "
            "unless -hf_repo_id is provided."
        )

    # Pre-resolve hyperparameters so training and evaluating both use the
    # correct path regardless of where the model package lives.
    params.hyperparameters = _resolve_hyperparameters(params)

    dg_cache_path = create_dg_cache(params.dg_save_folder)
    base_save_path = f"{dg_cache_path}/save/{params.repo_name}"
    experiment_slug = _experiment_slug(params)
    predict_score_path = f"{base_save_path}/predict_scores"
    result_path = f"{base_save_path}/results"
    experiment_root = f"{base_save_path}/experiments/{experiment_slug}"
    os.makedirs(experiment_root, exist_ok=True)

    use_calibration = getattr(params, "calibrated", True)

    if use_calibration and params.val_set is None and hf_repo_id is None:
        raise ValueError("-val_set is required for experiment mode when -calibrated is True.")
    if params.test_set is None and hf_repo_id is None:
        raise ValueError("-test_set is required for experiment mode to run final test.")

    base_sampling_seed = getattr(params, "sampling_seed", None)
    base_checkpoint_dir = getattr(params, "checkpoint_dir", None)
    base_seed = getattr(params, "seed", 42)
    total_runs = params.runs
    all_test_metrics = []
    budgets = _normalize_budgets(getattr(params, "budget", [1]))

    for run_idx in range(1, total_runs + 1):
        print(f"================ Experiment Run {run_idx}/{total_runs} ================")
        # Reset random state per run so repeated runs are directly comparable.
        seed_everything(base_seed)
        run_dir = f"{experiment_root}/run_{run_idx}"
        os.makedirs(run_dir, exist_ok=True)
        model_name = params.model
        # Sklearn models (lapredict, lr) save via pickle and have no checkpoint concept.
        if model_name not in _SKLEARN_MODELS:
            run_checkpoint_dir = (
                f"{base_checkpoint_dir}/run_{run_idx}"
                if base_checkpoint_dir
                else f"{run_dir}/checkpoints"
            )
            os.makedirs(run_checkpoint_dir, exist_ok=True)
        else:
            run_checkpoint_dir = None

        run_test_metric_file = f"{run_dir}/{model_name}_test_metrics.csv"
        if getattr(params, "resume_from_checkpoint", False) and os.path.exists(run_test_metric_file):
            print(f"Run {run_idx} already completed. Skip this run: {run_test_metric_file}")
            test_metrics_df = pd.read_csv(run_test_metric_file)
            all_test_metrics.append(test_metrics_df)
            continue

        # Keep the same sampling seed across runs for reproducible undersampling.
        run_sampling_seed = base_seed if base_sampling_seed is None else base_sampling_seed

        train_params = _clone_params(
            params,
            {
                "model_path": None,
                "sampling_run_id": run_idx,
                "sampling_seed": run_sampling_seed,
                "checkpoint_dir": run_checkpoint_dir,
                # Propagate the pre-resolved hyperparameters path.
                "hyperparameters": params.hyperparameters,
            },
        )
        print("[1/3] Training...")
        training(train_params)

        run_rows = []
        for budget_idx, budget in enumerate(budgets, start=1):
            budget_label = _budget_tag(budget)
            print(f"[2/3] Budget {budget_idx}/{len(budgets)}: calibration with budget={budget}")

            if use_calibration:
                val_eval_params = _clone_params(
                    params,
                    {
                        "test_set": params.val_set,
                        "calibrated": True,
                        "budget": float(budget),
                        "runs": 1,
                    },
                )
                evaluating(val_eval_params)

                selected_threshold_file = f"{predict_score_path}/{model_name}_selected_threshold.json"
                if not os.path.exists(selected_threshold_file):
                    raise FileNotFoundError(
                        f"Selected threshold file not found: {selected_threshold_file}. "
                        "Ensure validation evaluating ran with calibration enabled."
                    )

                with open(selected_threshold_file, "r", encoding="utf-8") as f:
                    threshold_payload = json.load(f)
                selected_threshold = float(threshold_payload["threshold"])
                print(f"Selected threshold for run {run_idx}, budget {budget}: {selected_threshold}")

                val_score_file = f"{predict_score_path}/{model_name}.csv"
                val_metrics_file = f"{result_path}/{model_name}.csv"
                val_calibration_file = f"{predict_score_path}/{model_name}_threshold_calibration.csv"

                if os.path.exists(val_score_file):
                    shutil.copy2(val_score_file, f"{run_dir}/{model_name}_{budget_label}_val_scores.csv")
                if os.path.exists(val_metrics_file):
                    shutil.copy2(val_metrics_file, f"{run_dir}/{model_name}_{budget_label}_val_metrics.csv")
                if os.path.exists(val_calibration_file):
                    shutil.copy2(val_calibration_file, f"{run_dir}/{model_name}_{budget_label}_val_threshold_calibration.csv")
                shutil.copy2(selected_threshold_file, f"{run_dir}/{model_name}_{budget_label}_selected_threshold.json")

                _safe_remove(f"{predict_score_path}/{model_name}.csv")
                _safe_remove(f"{predict_score_path}/{model_name}_run_1.csv")
                _safe_remove(f"{predict_score_path}/{model_name}_threshold_calibration.csv")
                _safe_remove(f"{predict_score_path}/{model_name}_threshold_calibration_run_1.csv")
                _safe_remove(f"{predict_score_path}/{model_name}_selected_threshold.json")
                _safe_remove(f"{predict_score_path}/{model_name}_selected_threshold_run_1.json")
                _safe_remove(f"{result_path}/{model_name}.csv")
                _safe_remove(f"{result_path}/{model_name}_run_1.csv")
                print("[3/3] Final test evaluating with fixed threshold...")
            else:
                selected_threshold = 0.5 if params.threshold is None else float(params.threshold)
                threshold_payload = {
                    "threshold": selected_threshold,
                    "source": "fixed",
                    "run": int(run_idx),
                    "budget": float(budget),
                }
                print(f"[2/2] Skip calibration, use fixed threshold: {selected_threshold}")
                with open(f"{run_dir}/{model_name}_{budget_label}_selected_threshold.json", "w", encoding="utf-8") as f:
                    json.dump(threshold_payload, f, indent=2)

            test_eval_params = _clone_params(
                params,
                {
                    "test_set": params.test_set,
                    "calibrated": False,
                    "budget": float(budget),
                    "threshold": selected_threshold,
                    "runs": 1,
                },
            )
            evaluating(test_eval_params)

            test_score_file = f"{predict_score_path}/{model_name}.csv"
            test_metrics_file = f"{result_path}/{model_name}.csv"
            if os.path.exists(test_score_file):
                shutil.copy2(test_score_file, f"{run_dir}/{model_name}_{budget_label}_test_scores.csv")
            if os.path.exists(test_metrics_file):
                shutil.copy2(test_metrics_file, f"{run_dir}/{model_name}_{budget_label}_test_metrics.csv")
                run_rows.append(
                    _collect_metric_row(
                        test_metrics_file,
                        model_name,
                        run_idx,
                        budget,
                        selected_threshold,
                        threshold_payload if use_calibration else None,
                    )
                )

            _safe_remove(f"{predict_score_path}/{model_name}.csv")
            _safe_remove(f"{predict_score_path}/{model_name}_run_1.csv")
            _safe_remove(f"{result_path}/{model_name}.csv")
            _safe_remove(f"{result_path}/{model_name}_run_1.csv")

        if run_rows:
            run_summary_df = pd.DataFrame(run_rows)
            run_summary_df.to_csv(run_test_metric_file, index=False)
            all_test_metrics.append(run_summary_df)

        print(f"Run {run_idx} artifacts saved to: {run_dir}")

    if all_test_metrics:
        combined_test_metrics = pd.concat(all_test_metrics, ignore_index=True)
        combined_test_metrics = combined_test_metrics.sort_values(["budget", "run"], kind="stable")

        metric_columns = [
            column
            for column in combined_test_metrics.columns
            if column not in {"model", "run", "budget"} and pd.api.types.is_numeric_dtype(combined_test_metrics[column])
        ]

        summary_frames = []
        for budget in sorted(combined_test_metrics["budget"].dropna().unique()):
            budget_frame = combined_test_metrics[combined_test_metrics["budget"] == budget].copy()
            average_row = {"model": "average", "run": "average", "budget": float(budget)}
            for column in metric_columns:
                average_row[column] = budget_frame[column].mean()
            summary_frames.append(budget_frame)
            summary_frames.append(pd.DataFrame([average_row]))

        final_summary = pd.concat(summary_frames, ignore_index=True)
        if len(budgets) == 1:
            output_name = f"{experiment_slug}_test_all_run.csv"
        else:
            output_name = f"{experiment_slug}_test_all_budget.csv"
        final_summary_path = f"{experiment_root}/{output_name}"
        final_summary.to_csv(final_summary_path, index=False)

        if getattr(params, "hf_upload_result", False):
            output_repo_id = getattr(params, "hf_output_repo_id", None) or hf_repo_id
            if not output_repo_id:
                raise ValueError("-hf_output_repo_id or -hf_repo_id is required when -hf_upload_result is True.")

            remote_path = _hf_output_path(params)
            print(f"Uploading experiment results to HF dataset repo: {output_repo_id}/{remote_path}")
            upload_folder_to_hf_dataset(
                local_folder=experiment_root,
                repo_id=output_repo_id,
                path_in_repo=remote_path,
                commit_message=f"Upload experiment results for {experiment_slug}",
            )
