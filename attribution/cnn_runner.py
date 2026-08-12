"""Inference-only row attribution for DeepJIT and SimCom's Com component."""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import torch

from ..models.deepjit.dataset import CustomDataset as DeepJITDataset
from ..models.deepjit.warper import DeepJIT
from ..models.simcom.com.dataset import CustomDataset as ComDataset
from ..models.simcom.com.warper import Com
from ..models.simcom.warper import SimCom
from .export import (file_fingerprint, load_jsonl, write_json, write_jsonl,
                     write_line_csv, write_row_csv)
from .hierarchical_gradcam import hierarchical_row_gradcam, rank_observed_rows
from .line_ranking import (aggregate_line_scores, load_provenance_index,
                           merge_token_line_ids, patch_token_line_ids,
                           verify_serialization)


def _checkpoint_path(model_name, model_path):
    if os.path.isfile(model_path):
        return model_path
    if model_name == "simcom":
        return os.path.join(model_path, "com.pth")
    latest = os.path.join(model_path, "deepjit_checkpoint_last.pth")
    return latest if os.path.exists(latest) else os.path.join(model_path, "deepjit.pth")


def _split_inputs(model_name, test_set):
    parts = [part.strip() for part in test_set.split(",")]
    if model_name == "deepjit":
        if len(parts) != 1 or not parts[0]:
            raise ValueError("DeepJIT -test_set must be one merge JSONL file")
        return None, parts[0]
    if len(parts) == 1 and parts[0]:
        return None, parts[0]
    if len(parts) == 2 and all(parts):
        return parts[0], parts[1]
    raise ValueError("SimCom -test_set must be patch.jsonl or features.jsonl,patch.jsonl")


def _metadata(args, feature_path, code_path):
    checkpoint = _checkpoint_path(args.model, args.model_path)
    fingerprints = {
        "checkpoint": file_fingerprint(checkpoint),
        "hyperparameters": file_fingerprint(args.hyperparameters),
        "dictionary": file_fingerprint(args.dictionary),
        "code_file": file_fingerprint(code_path),
    }
    if feature_path:
        fingerprints["feature_file"] = file_fingerprint(feature_path)
    if getattr(args, "line_provenance", None):
        fingerprints["line_provenance"] = file_fingerprint(args.line_provenance)
    return {
        "schema_version": 1,
        "model": args.model,
        "attribution_method": "hierarchical_gradcam",
        "target_class": args.target_class,
        "checkpoint_path": os.path.abspath(checkpoint),
        "feature_file": os.path.abspath(feature_path) if feature_path else None,
        "code_file": os.path.abspath(code_path),
        "fingerprints": fingerprints,
        "device": args.device,
        "threshold": args.threshold,
        "top_k": args.top_k,
        "commit_id": args.commit_id,
        "only_predicted_vulnerable": args.only_predicted_vulnerable,
        "line_aggregation": getattr(args, "line_aggregation", "sum"),
        "seed": args.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _validate_resume(path, current):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        previous = json.load(handle)
    keys = ("model", "fingerprints", "target_class", "threshold", "top_k", "commit_id",
            "only_predicted_vulnerable", "line_aggregation")
    if any(previous.get(key) != current.get(key) for key in keys):
        raise ValueError("resume_configuration_mismatch")


def _component_diagnostics(sim_score, com_score, threshold):
    final_score = (sim_score + com_score) / 2.0
    sim_class = int(sim_score > threshold)
    com_class = int(com_score > threshold)
    final_class = int(final_score > threshold)
    return {
        "sim_score": sim_score,
        "com_score": com_score,
        "final_score": final_score,
        "component_agreement": sim_class == com_class,
        "com_supports_final_class": com_class == final_class,
    }


def attribute_cnn(args):
    if not args.dictionary:
        raise ValueError("-dictionary is required for DeepJIT and SimCom attribution")
    feature_path, code_path = _split_inputs(args.model, args.test_set)
    os.makedirs(args.output_dir, exist_ok=True)
    metadata_path = os.path.join(args.output_dir, "run_metadata.json")
    jsonl_path = os.path.join(args.output_dir, "row_attributions.jsonl")
    csv_path = os.path.join(args.output_dir, "row_attributions.csv")
    line_csv_path = os.path.join(args.output_dir, "line_attributions.csv")
    summary_path = os.path.join(args.output_dir, "summary.json")
    outputs = (metadata_path, jsonl_path, csv_path, summary_path)
    if any(os.path.exists(path) for path in outputs) and not (args.overwrite or args.resume):
        raise FileExistsError("output exists; use -overwrite or -resume")

    run_metadata = _metadata(args, feature_path, code_path)
    if args.resume:
        _validate_resume(metadata_path, run_metadata)
    write_json(metadata_path, run_metadata)
    records = load_jsonl(jsonl_path) if args.resume else []
    completed = {str(record.get("commit_id")) for record in records}
    provenance_index = (
        load_provenance_index(args.line_provenance)
        if getattr(args, "line_provenance", None) else None
    )

    frame = pd.read_json(code_path, orient="records", lines=True)
    if args.model == "deepjit":
        wrapper = DeepJIT(args.repo_language, args.device)
        wrapper.initialize(args.dictionary, args.hyperparameters, args.model_path, inference_only=True)
        dataset = DeepJITDataset(frame, wrapper.hyperparameters, wrapper.code_dictionary,
                                 wrapper.message_dictionary)
        cnn_model = wrapper.model
        sim_scores = None
        scope = "deepjit_code_change_hierarchical_cnn_rows"
        component = None
    else:
        sim_checkpoint = None if os.path.isfile(args.model_path) else os.path.join(args.model_path, "sim.pkl")
        has_full_simcom = feature_path is not None and os.path.exists(sim_checkpoint)
        if has_full_simcom:
            wrapper = SimCom(args.repo_language, args.device)
            wrapper.initialize(args.dictionary, args.hyperparameters, args.model_path, inference_only=True)
            com_wrapper = wrapper.com
        else:
            com_wrapper = Com(args.repo_language, args.device)
            com_wrapper.initialize(args.dictionary, args.hyperparameters, args.model_path, inference_only=True)
            wrapper = None
        dataset = ComDataset(frame, com_wrapper.hyperparameters, com_wrapper.code_dictionary,
                             com_wrapper.message_dictionary)
        cnn_model = com_wrapper.model
        if has_full_simcom:
            sim_predictions = wrapper.sim.inference(feature_path, args.threshold)
            sim_scores = dict(zip(sim_predictions["commit_id"].astype(str), sim_predictions["probability"]))
        else:
            sim_scores = None
        scope = "simcom_com_code_change_hierarchical_cnn_rows"
        component = "Com"

    cnn_model.eval()
    for index in range(len(dataset)):
        row = frame.iloc[index]
        commit_id = str(row["commit_id"])
        if commit_id in completed or (args.commit_id and commit_id != str(args.commit_id)):
            continue
        try:
            sample = dataset[index]
            code = sample["code"].unsqueeze(0).to(args.device)
            message = sample["message"].unsqueeze(0).to(args.device)
            with torch.no_grad():
                normal_score = float(cnn_model(message, code)[0].item())
            gradcam = hierarchical_row_gradcam(cnn_model, message, code, args.target_class)
            com_score = float(gradcam["probability"][0].item())
            tolerance = 1e-5 if str(args.device).startswith("cuda") else 1e-6
            if abs(normal_score - com_score) >= tolerance:
                raise ValueError("prediction_preservation_failed")

            diagnostics = None
            prediction_score = com_score
            if sim_scores is not None:
                if commit_id not in sim_scores:
                    raise ValueError("missing_sim_score")
                diagnostics = _component_diagnostics(float(sim_scores[commit_id]), com_score, args.threshold)
                prediction_score = diagnostics["final_score"]
            if args.only_predicted_vulnerable and prediction_score <= args.threshold:
                record = {"commit_id": commit_id, "status": "skipped",
                          "reason": "filtered_not_predicted_vulnerable"}
            else:
                all_rows = str(row["code_change"]).split("\n")
                code_line = wrapper.hyperparameters["code_line"] if args.model == "deepjit" else com_wrapper.hyperparameters["code_line"]
                observed_rows = all_rows[:code_line]
                ranked = rank_observed_rows(
                    observed_rows, gradcam["row_scores"][0].cpu().tolist(),
                    "mean_cam_over_exact_kernel_receptive_fields", args.top_k,
                )
                record = {
                    "commit_id": commit_id,
                    "status": "succeeded",
                    "model_name": args.model,
                    "prediction_score": prediction_score,
                    "predicted_label": int(prediction_score > args.threshold),
                    "threshold": args.threshold,
                    "attribution_method": "hierarchical_gradcam",
                    "target_class": args.target_class,
                    "attribution_is_class_specific": True,
                    "explanation_scope": scope,
                    "localization_component": component,
                    "full_model_explanation": False,
                    "ranked_rows": [item.to_dict() for item in ranked],
                    "input_rows": len(all_rows),
                    "observed_rows": len(observed_rows),
                    "truncated_rows": max(0, len(all_rows) - len(observed_rows)),
                    "truncated": len(all_rows) > len(observed_rows),
                    "metadata": {
                        "row_semantics": "serialized_code_change_rows",
                        "source_line_provenance_verified": False,
                        "branch_activation_shapes": gradcam["branch_activation_shapes"],
                        "prediction_preservation_delta": abs(normal_score - com_score),
                        "component_scores": diagnostics if diagnostics is not None else (
                            {"sim_score": None, "com_score": com_score, "final_score": None,
                             "component_agreement": None, "com_supports_final_class": None}
                            if args.model == "simcom" else None
                        ),
                        "prediction_scope": (
                            "full_simcom_mean" if diagnostics is not None else
                            "com_component_only" if args.model == "simcom" else
                            "deepjit_commit_probability"
                        ),
                        "unexplained_components": ["Sim"] if args.model == "simcom" else ["message_branch"],
                    },
                }
                if provenance_index is not None:
                    if commit_id not in provenance_index:
                        raise ValueError("missing_line_provenance")
                    provenance = provenance_index[commit_id]
                    kind = "merge" if args.model == "deepjit" else "patch"
                    verify_serialization(provenance, str(row["code_change"]), kind)
                    position_scores = []
                    token_scores = gradcam["token_scores"][0]
                    if args.model == "deepjit":
                        mapping = merge_token_line_ids(provenance)
                        for token_position, line_id in mapping.items():
                            if token_position < token_scores.shape[1]:
                                position_scores.append((line_id, token_scores[0, token_position].item()))
                    else:
                        mapping = patch_token_line_ids(provenance)
                        for (row_position, token_position), line_id in mapping.items():
                            if row_position < token_scores.shape[0] and token_position < token_scores.shape[1]:
                                position_scores.append(
                                    (line_id, token_scores[row_position, token_position].item())
                                )
                    line_result = aggregate_line_scores(
                        provenance, position_scores,
                        getattr(args, "line_aggregation", "sum"), args.top_k,
                        model_input_kind=kind,
                    )
                    record.update(line_result)
                    record["metadata"]["source_line_provenance_verified"] = True
                    record["metadata"]["serialization_alignment_status"] = "exact"
                    record["metadata"]["token_branch_activation_shapes"] = gradcam[
                        "token_branch_activation_shapes"
                    ]
        except Exception as exc:
            record = {"commit_id": commit_id, "status": "failed",
                      "reason": f"{type(exc).__name__}:{exc}"}
        records.append(record)
        completed.add(commit_id)
        write_jsonl(jsonl_path, records)

    if args.commit_id and str(args.commit_id) not in completed:
        raise ValueError(f"commit_id_not_found:{args.commit_id}")
    write_jsonl(jsonl_path, records)
    write_row_csv(csv_path, records)
    if provenance_index is not None:
        write_line_csv(line_csv_path, records)
    summary = {
        "processed": len(records),
        "succeeded": sum(r.get("status") == "succeeded" for r in records),
        "skipped": sum(r.get("status") == "skipped" for r in records),
        "failed": sum(r.get("status") == "failed" for r in records),
    }
    write_json(summary_path, summary)
    print(f"Row attribution complete: {summary['succeeded']} succeeded, "
          f"{summary['skipped']} skipped, {summary['failed']} failed")
    return summary
