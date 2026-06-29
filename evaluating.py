import os
import json
import shutil

import numpy as np
import pandas as pd

from .models.init_model import init_model
from .utils.utils import create_dg_cache
from .utils.hf_dataset import prepare_hf_dataset_paths
from .utils.metrics import get_metrics


def _compute_ratios(result_df, threshold):
    pred = (result_df["probability"] > threshold).astype(int)
    y = result_df["label"].astype(int)

    vuln = int((y == 1).sum())
    marked = int((pred == 1).sum())
    all_commit = len(y)
    marked_vuln = int(((y == 1) & (pred == 1)).sum())

    vuln_detection_ratio = marked_vuln / vuln if vuln > 0 else 0.0
    marked_function_ratio = marked / all_commit if all_commit > 0 else 0.0
    return vuln_detection_ratio, marked_function_ratio


def _parse_calibration_range(calibration_range):
    if calibration_range is None:
        return 0.0, 1.0, 10001

    start_raw, end_raw, steps_raw = calibration_range
    try:
        start = float(start_raw)
        end = float(end_raw)
        steps = int(steps_raw)
    except (TypeError, ValueError):
        raise ValueError("calibration_range must be: START END STEPS, where START/END are floats and STEPS is an integer")

    if start < 0.0 or end > 1.0:
        raise ValueError("calibration_range START and END must be within [0, 1]")
    if start > end:
        raise ValueError("calibration_range START must be <= END")
    if steps < 2:
        raise ValueError("calibration_range STEPS must be >= 2")

    return start, end, steps


def _select_calibrated_threshold(result_df, budget, calibration_range=None):
    start, end, steps = _parse_calibration_range(calibration_range)
    thresholds = np.round(np.linspace(start, end, steps), 4)
    rows = []
    for threshold in thresholds:
        vuln_detection_ratio, marked_function_ratio = _compute_ratios(result_df, threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "vuln_detection_ratio": vuln_detection_ratio,
                "marked_function_ratio": marked_function_ratio,
            }
        )

    calibration_df = pd.DataFrame(rows)
    feasible = calibration_df[calibration_df["marked_function_ratio"] <= budget].copy()

    if len(feasible) > 0:
        best = feasible.sort_values(
            ["vuln_detection_ratio", "marked_function_ratio", "threshold"],
            ascending=[False, True, False],
        ).iloc[0]
    else:
        best = calibration_df.sort_values(
            ["marked_function_ratio", "vuln_detection_ratio", "threshold"],
            ascending=[True, False, False],
        ).iloc[0]

    return float(best["threshold"]), best, calibration_df, start, end, steps
          
def evaluating(params):
    dg_cache_path = create_dg_cache(params.dg_save_folder)
    predict_score_path = f'{dg_cache_path}/save/{params.repo_name}/predict_scores'
    result_path = f'{dg_cache_path}/save/{params.repo_name}/results'
    os.makedirs(predict_score_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    
    threshold = 0.5 if params.threshold is None else params.threshold 
    runs = getattr(params, "runs", 1)

    for run_idx in range(1, runs + 1):
        print(f"========== Run {run_idx}/{runs} ==========")

        # Init model per run so each run starts fresh.
        model = init_model(params.model, params.repo_language, params.device)
        hf_paths = prepare_hf_dataset_paths(
            dg_cache_path,
            params.repo_name,
            model.model_name,
            getattr(params, "hf_repo_id", None),
            revision=getattr(params, "hf_revision", "main"),
            split_path=getattr(params, "hf_split_path", None),
        )

        NO_DICT_MODELS = {"tlel", "lapredict", "lr", "jitfine"}
        if model.model_name in NO_DICT_MODELS:
            hf_paths["dictionary"] = None

        dictionary = params.dictionary if params.dictionary is not None else hf_paths.get("dictionary")
        if dictionary is None and model.model_name not in NO_DICT_MODELS:
            dictionary = f'{dg_cache_path}/dataset/{params.repo_name}/dict_{params.repo_name}.jsonl'
        hyperparameters = params.hyperparameters
        model_path = f'{dg_cache_path}/save/{params.repo_name}/models/best_epoch' if params.model_path is None else params.model_path
        print(f"Init model: {model.model_name}")
        model.initialize(model_path=model_path, dictionary=dictionary, hyperparameters=hyperparameters)


        if params.test_set:
            test_df_path = params.test_set
        elif hf_paths.get("test_set"):
            test_df_path = hf_paths["test_set"]
        else:
            test_df_path = ','.join([f'{dg_cache_path}/dataset/{params.repo_name}/data/test_{t}_{params.repo_name}.jsonl' for t in getattr(model, 'default_test_input', model.default_input).split(",")])

        current_threshold = threshold
        result_df = model.inference(infer_df=test_df_path, threshold=current_threshold, params=params)

        if getattr(params, "calibrated", False):
            if "label" not in result_df.columns:
                raise ValueError("Calibration requires labeled evaluation data with a 'label' column.")

            selected_threshold, best_row, calibration_df, start, end, steps = _select_calibrated_threshold(
                result_df,
                params.budget,
                getattr(params, "calibration_range", None),
            )
            current_threshold = selected_threshold
            result_df["prediction"] = (result_df["probability"] > current_threshold).astype(float)

            calibration_file = f'{predict_score_path}/{model.model_name}_threshold_calibration.csv'
            calibration_run_file = f'{predict_score_path}/{model.model_name}_threshold_calibration_run_{run_idx}.csv'
            calibration_df.to_csv(calibration_file, index=False)
            calibration_df.to_csv(calibration_run_file, index=False)

            selected_threshold_file = f'{predict_score_path}/{model.model_name}_selected_threshold.json'
            selected_threshold_run_file = f'{predict_score_path}/{model.model_name}_selected_threshold_run_{run_idx}.json'
            selected_threshold_payload = {
                "threshold": float(best_row["threshold"]),
                "vuln_detection_ratio": float(best_row["vuln_detection_ratio"]),
                "marked_function_ratio": float(best_row["marked_function_ratio"]),
                "budget": float(params.budget),
                "calibration_range": {
                    "start": float(start),
                    "end": float(end),
                    "steps": int(steps),
                },
                "run": int(run_idx),
            }
            with open(selected_threshold_file, "w", encoding="utf-8") as f:
                json.dump(selected_threshold_payload, f, indent=2)
            with open(selected_threshold_run_file, "w", encoding="utf-8") as f:
                json.dump(selected_threshold_payload, f, indent=2)

            print(f"Calibration points saved to: {calibration_file}")
            print(f"Calibration points saved to: {calibration_run_file}")
            print(f"Selected threshold file saved to: {selected_threshold_file}")
            print(f"Selected threshold file saved to: {selected_threshold_run_file}")
            print(f"Selected threshold: {best_row['threshold']}")
            print(f"vuln_detection_ratio: {best_row['vuln_detection_ratio']}")
            print(f"marked_function_ratio: {best_row['marked_function_ratio']}")
            print(f"calibration_range: start={start}, end={end}, steps={steps}")

        score_file = f'{predict_score_path}/{model.model_name}.csv'
        score_run_file = f'{predict_score_path}/{model.model_name}_run_{run_idx}.csv'
        result_df.to_csv(score_file, index=False, columns=["commit_id", "label", "prediction", "probability"])
        shutil.copy2(score_file, score_run_file)
        print(f"Predict scores saved to: {score_file}")
        print(f"Predict scores saved to: {score_run_file}")

        metrics_df = get_metrics(result_df, model.model_name, None)
        metrics_file = f'{result_path}/{model.model_name}.csv'
        metrics_run_file = f'{result_path}/{model.model_name}_run_{run_idx}.csv'
        metrics_df.to_csv(metrics_file, index=True)
        shutil.copy2(metrics_file, metrics_run_file)
        print(f"Metrics saved to: {metrics_file}")
        print(f"Metrics saved to: {metrics_run_file}")