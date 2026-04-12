import json
import os
import shutil
import io
from contextlib import redirect_stdout
from argparse import Namespace

import pandas as pd

from .evaluating import evaluating
from .training import training
from .utils.utils import create_dg_cache, seed_torch


def _clone_params(params, overrides):
    data = vars(params).copy()
    data.update(overrides)
    return Namespace(**data)


def _run_silently(func, params):
    # Suppress verbose stdout from training/evaluating and keep experiment logs readable.
    with redirect_stdout(io.StringIO()):
        return func(params)


def _safe_remove(path):
    if os.path.exists(path):
        os.remove(path)


def run_experiment(params):
    if params.model == "deepjit" and params.dictionary is None:
        raise ValueError("-dictionary is required for experiment mode when -model is deepjit.")

    dg_cache_path = create_dg_cache(params.dg_save_folder)
    base_save_path = f"{dg_cache_path}/save/{params.repo_name}"
    predict_score_path = f"{base_save_path}/predict_scores"
    result_path = f"{base_save_path}/results"
    experiment_root = f"{base_save_path}/experiments"
    os.makedirs(experiment_root, exist_ok=True)

    if params.val_set is None:
        raise ValueError("-val_set is required for experiment mode to calibrate threshold.")
    if params.test_set is None:
        raise ValueError("-test_set is required for experiment mode to run final test.")

    base_sampling_seed = getattr(params, "sampling_seed", None)
    total_runs = params.runs
    all_test_metrics = []

    for run_idx in range(1, total_runs + 1):
        seed_torch()  # Reset seed at the start of every run so all runs are identical
        print(f"================ Experiment Run {run_idx}/{total_runs} ================")
        run_dir = f"{experiment_root}/run_{run_idx}"
        os.makedirs(run_dir, exist_ok=True)

        train_params = _clone_params(
            params,
            {
                "model_path": None,
                "sampling_run_id": run_idx,
                "sampling_seed": base_sampling_seed,
            },
        )
        print("[1/3] Training...")
        _run_silently(training, train_params)

        print("[2/3] Validation evaluating with calibration...")
        val_eval_params = _clone_params(
            params,
            {
                "test_set": params.val_set,
                "calibrated": True,
                "runs": 1,
            },
        )
        _run_silently(evaluating, val_eval_params)

        model_name = params.model
        selected_threshold_file = f"{predict_score_path}/{model_name}_selected_threshold.json"
        if not os.path.exists(selected_threshold_file):
            raise FileNotFoundError(
                f"Selected threshold file not found: {selected_threshold_file}. "
                "Ensure validation evaluating ran with calibration enabled."
            )

        with open(selected_threshold_file, "r", encoding="utf-8") as f:
            threshold_payload = json.load(f)
        selected_threshold = float(threshold_payload["threshold"])
        print(f"Selected threshold for run {run_idx}: {selected_threshold}")

        val_score_file = f"{predict_score_path}/{model_name}.csv"
        val_metrics_file = f"{result_path}/{model_name}.csv"
        val_calibration_file = f"{predict_score_path}/{model_name}_threshold_calibration.csv"

        if os.path.exists(val_score_file):
            shutil.copy2(val_score_file, f"{run_dir}/{model_name}_val_scores.csv")
        if os.path.exists(val_metrics_file):
            shutil.copy2(val_metrics_file, f"{run_dir}/{model_name}_val_metrics.csv")
        if os.path.exists(val_calibration_file):
            shutil.copy2(val_calibration_file, f"{run_dir}/{model_name}_val_threshold_calibration.csv")
        shutil.copy2(selected_threshold_file, f"{run_dir}/{model_name}_selected_threshold.json")

        # Remove validation intermediate outputs from shared folders.
        _safe_remove(f"{predict_score_path}/{model_name}.csv")
        _safe_remove(f"{predict_score_path}/{model_name}_run_1.csv")
        _safe_remove(f"{predict_score_path}/{model_name}_threshold_calibration.csv")
        _safe_remove(f"{predict_score_path}/{model_name}_threshold_calibration_run_1.csv")
        _safe_remove(f"{predict_score_path}/{model_name}_selected_threshold.json")
        _safe_remove(f"{predict_score_path}/{model_name}_selected_threshold_run_1.json")
        _safe_remove(f"{result_path}/{model_name}.csv")
        _safe_remove(f"{result_path}/{model_name}_run_1.csv")

        print("[3/3] Final test evaluating with fixed threshold...")
        test_eval_params = _clone_params(
            params,
            {
                "test_set": params.test_set,
                "calibrated": False,
                "threshold": selected_threshold,
                "runs": 1,
            },
        )
        _run_silently(evaluating, test_eval_params)

        test_score_file = f"{predict_score_path}/{model_name}.csv"
        test_metrics_file = f"{result_path}/{model_name}.csv"
        if os.path.exists(test_score_file):
            shutil.copy2(test_score_file, f"{run_dir}/{model_name}_test_scores.csv")
        if os.path.exists(test_metrics_file):
            shutil.copy2(test_metrics_file, f"{run_dir}/{model_name}_test_metric_run_{run_idx}.csv")
            test_metrics_df = pd.read_csv(test_metrics_file, index_col=0).reset_index()
            test_metrics_df = test_metrics_df.rename(columns={"index": "model"})
            test_metrics_df.insert(1, "run", run_idx)
            all_test_metrics.append(test_metrics_df)

        # Remove test intermediate outputs from shared folders.
        _safe_remove(f"{predict_score_path}/{model_name}.csv")
        _safe_remove(f"{predict_score_path}/{model_name}_run_1.csv")
        _safe_remove(f"{result_path}/{model_name}.csv")
        _safe_remove(f"{result_path}/{model_name}_run_1.csv")

        print(f"Run {run_idx} artifacts saved to: {run_dir}")

    if all_test_metrics:
        combined_test_metrics = pd.concat(all_test_metrics, ignore_index=True)
        metric_columns = [
            column
            for column in combined_test_metrics.columns
            if column not in {"model", "run"} and pd.api.types.is_numeric_dtype(combined_test_metrics[column])
        ]

        average_row = {"model": "average", "run": "average"}
        for column in metric_columns:
            average_row[column] = combined_test_metrics[column].mean()

        combined_test_metrics = pd.concat(
            [combined_test_metrics, pd.DataFrame([average_row])],
            ignore_index=True,
        )

        output_name = "deepjit_test_all_run.csv" if params.model == "deepjit" else f"{params.model}_test_all_run.csv"
        combined_test_metrics.to_csv(f"{experiment_root}/{output_name}", index=False)
