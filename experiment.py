import json
import os
import shutil
from argparse import Namespace

from .evaluating import evaluating
from .training import training
from .utils.utils import create_dg_cache


def _clone_params(params, overrides):
    data = vars(params).copy()
    data.update(overrides)
    return Namespace(**data)


def run_experiment(params):
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

    for run_idx in range(1, total_runs + 1):
        print(f"================ Experiment Run {run_idx}/{total_runs} ================")
        run_dir = f"{experiment_root}/run_{run_idx}"
        os.makedirs(run_dir, exist_ok=True)

        # Keep the same sampling seed across runs for reproducible undersampling.
        run_sampling_seed = base_sampling_seed

        train_params = _clone_params(
            params,
            {
                "model_path": None,
                "sampling_run_id": run_idx,
                "sampling_seed": run_sampling_seed,
            },
        )
        print("[1/3] Training...")
        training(train_params)

        print("[2/3] Validation evaluating with calibration...")
        val_eval_params = _clone_params(
            params,
            {
                "test_set": params.val_set,
                "calibrated": True,
                "runs": 1,
            },
        )
        evaluating(val_eval_params)

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
        evaluating(test_eval_params)

        test_score_file = f"{predict_score_path}/{model_name}.csv"
        test_metrics_file = f"{result_path}/{model_name}.csv"
        if os.path.exists(test_score_file):
            shutil.copy2(test_score_file, f"{run_dir}/{model_name}_test_scores.csv")
        if os.path.exists(test_metrics_file):
            shutil.copy2(test_metrics_file, f"{run_dir}/{model_name}_test_metrics.csv")

        with open(f"{run_dir}/summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run": run_idx,
                    "selected_threshold": selected_threshold,
                    "sampling_enabled": bool(getattr(params, "sampling", False)),
                    "sampling_seed": run_sampling_seed,
                    "budget": float(params.budget),
                    "calibration_range": getattr(params, "calibration_range", None),
                },
                f,
                indent=2,
            )

        print(f"Run {run_idx} artifacts saved to: {run_dir}")
