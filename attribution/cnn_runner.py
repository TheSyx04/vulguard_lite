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
        "simcom_chunk_size": getattr(args, "simcom_chunk_size", 10),
        "seed": args.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _validate_resume(path, current):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        previous = json.load(handle)
    keys = ("model", "fingerprints", "target_class", "threshold", "top_k", "commit_id",
            "only_predicted_vulnerable", "line_aggregation", "simcom_chunk_size")
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


def split_simcom_row(row, chunk_size=10):
    """Split one historical SimCom patch while preserving commit-level fields."""
    if chunk_size < 1:
        raise ValueError("simcom_chunk_size_must_be_positive")
    patch_rows = str(row["code_change"]).split("\n")
    chunks = []
    for chunk_id, start in enumerate(range(0, len(patch_rows), chunk_size)):
        end = min(start + chunk_size, len(patch_rows))
        chunk_row = row.copy()
        chunk_row["code_change"] = "\n".join(patch_rows[start:end])
        chunks.append({
            "chunk_id": chunk_id,
            "row_start": start,
            "row_end_exclusive": end,
            "rows": patch_rows[start:end],
            "data": chunk_row,
        })
    return chunks


def _chunk_patch_position_scores(provenance, token_scores, row_start, row_end):
    """Map local chunk tensor positions back to canonical changed-line IDs."""
    position_scores = []
    for (global_row, token_position), line_id in patch_token_line_ids(provenance).items():
        if row_start <= global_row < row_end:
            local_row = global_row - row_start
            if local_row < token_scores.shape[0] and token_position < token_scores.shape[1]:
                position_scores.append((line_id, token_scores[local_row, token_position].item()))
    return position_scores


def _with_chunk_fields(items, chunk):
    return [dict(item, chunk_id=chunk["chunk_id"],
                 chunk_row_start=chunk["row_start"],
                 chunk_row_end_exclusive=chunk["row_end_exclusive"])
            for item in items]


def _attribute_simcom_commit(args, row, commit_id, cnn_model, com_wrapper,
                             sim_scores, provenance_index, scope, component):
    chunk_size = getattr(args, "simcom_chunk_size", 10)
    model_rows = int(com_wrapper.hyperparameters["code_line"])
    if chunk_size > model_rows:
        raise ValueError(
            f"simcom_chunk_size_exceeds_model_code_line:{chunk_size}>{model_rows}"
        )
    provenance = None
    if provenance_index is not None:
        if commit_id not in provenance_index:
            raise ValueError("missing_line_provenance")
        provenance = provenance_index[commit_id]
        verify_serialization(provenance, str(row["code_change"]), "patch")

    chunks, flat_rows, chunk_top_lines, all_position_scores = [], [], [], []
    prediction_scores = []
    for chunk in split_simcom_row(row, chunk_size):
        chunk_frame = pd.DataFrame([chunk["data"]])
        chunk_dataset = ComDataset(
            chunk_frame, com_wrapper.hyperparameters,
            com_wrapper.code_dictionary, com_wrapper.message_dictionary,
        )
        sample = chunk_dataset[0]
        code = sample["code"].unsqueeze(0).to(args.device)
        message = sample["message"].unsqueeze(0).to(args.device)
        with torch.no_grad():
            normal_score = float(cnn_model(message, code)[0].item())
        gradcam = hierarchical_row_gradcam(cnn_model, message, code, args.target_class)
        com_score = float(gradcam["probability"][0].item())
        tolerance = 1e-5 if str(args.device).startswith("cuda") else 1e-6
        if abs(normal_score - com_score) >= tolerance:
            raise ValueError(f"prediction_preservation_failed:chunk_{chunk['chunk_id']}")

        diagnostics = None
        chunk_prediction = com_score
        if sim_scores is not None:
            if commit_id not in sim_scores:
                raise ValueError("missing_sim_score")
            diagnostics = _component_diagnostics(
                float(sim_scores[commit_id]), com_score, args.threshold,
            )
            chunk_prediction = diagnostics["final_score"]
        prediction_scores.append(chunk_prediction)

        ranked_rows = rank_observed_rows(
            chunk["rows"], gradcam["row_scores"][0].cpu().tolist(),
            "mean_cam_over_exact_kernel_receptive_fields", args.top_k,
        )
        row_dicts = _with_chunk_fields([item.to_dict() for item in ranked_rows], chunk)
        for item in row_dicts:
            item["row_position"] += chunk["row_start"]
        flat_rows.extend(row_dicts)

        chunk_record = {
            "chunk_id": chunk["chunk_id"],
            "row_start": chunk["row_start"],
            "row_end_exclusive": chunk["row_end_exclusive"],
            "row_count": len(chunk["rows"]),
            "commit_message_repeated": True,
            "com_score": com_score,
            "prediction_score": chunk_prediction,
            "predicted_label": int(chunk_prediction > args.threshold),
            "ranked_rows": row_dicts,
            "top_line": None,
            "metadata": {
                "component_scores": diagnostics,
                "prediction_preservation_delta": abs(normal_score - com_score),
                "branch_activation_shapes": gradcam["branch_activation_shapes"],
                "token_branch_activation_shapes": gradcam["token_branch_activation_shapes"],
            },
        }
        if provenance is not None:
            position_scores = _chunk_patch_position_scores(
                provenance, gradcam["token_scores"][0],
                chunk["row_start"], chunk["row_end_exclusive"],
            )
            all_position_scores.extend(position_scores)
            line_result = aggregate_line_scores(
                provenance, position_scores,
                getattr(args, "line_aggregation", "sum"), args.top_k,
                model_input_kind="patch",
            )
            ranked_lines = _with_chunk_fields(line_result["ranked_lines"], chunk)
            chunk_record["ranked_lines"] = ranked_lines
            if ranked_lines:
                chunk_record["top_line"] = ranked_lines[0]
                chunk_top_lines.append(ranked_lines[0])
        chunks.append(chunk_record)

    prediction_score = max(prediction_scores)
    if args.only_predicted_vulnerable and prediction_score <= args.threshold:
        return {"commit_id": commit_id, "status": "skipped",
                "reason": "filtered_not_predicted_vulnerable"}

    all_rows = str(row["code_change"]).split("\n")
    record = {
        "commit_id": commit_id,
        "status": "succeeded",
        "model_name": "simcom",
        "prediction_score": prediction_score,
        "predicted_label": int(prediction_score > args.threshold),
        "threshold": args.threshold,
        "attribution_method": "hierarchical_gradcam",
        "target_class": args.target_class,
        "attribution_is_class_specific": True,
        "explanation_scope": scope,
        "localization_component": component,
        "full_model_explanation": False,
        "ranked_rows": flat_rows,
        "ranked_lines": chunk_top_lines,
        "chunk_top_lines": chunk_top_lines,
        "chunks": chunks,
        "input_rows": len(all_rows),
        "observed_rows": len(all_rows),
        "truncated_rows": 0,
        "truncated": False,
        "metadata": {
            "row_semantics": "serialized_code_change_rows",
            "source_line_provenance_verified": provenance is not None,
            "chunking": {
                "strategy": "contiguous_patch_rows",
                "chunk_size": chunk_size,
                "chunk_count": len(chunks),
                "commit_message_repeated_per_chunk": True,
                "prediction_aggregation": "max_chunk_probability",
                "line_selection": "top_1_per_chunk",
            },
            "prediction_scope": (
                "max_chunk_full_simcom_mean" if sim_scores is not None
                else "max_chunk_com_component_only"
            ),
            "unexplained_components": ["Sim"],
        },
    }
    if provenance is not None:
        coverage = aggregate_line_scores(
            provenance, all_position_scores,
            getattr(args, "line_aggregation", "sum"), None,
            model_input_kind="patch",
        )
        for key in ("uncovered_lines", "total_changed_lines", "covered_changed_lines",
                    "coverage_ratio", "line_aggregation"):
            record[key] = coverage[key]
        record["metadata"]["serialization_alignment_status"] = "exact"
    return record


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
        dataset = None  # SimCom samples are built per chunk below.
        cnn_model = com_wrapper.model
        if has_full_simcom:
            sim_predictions = wrapper.sim.inference(feature_path, args.threshold)
            sim_scores = dict(zip(sim_predictions["commit_id"].astype(str), sim_predictions["probability"]))
        else:
            sim_scores = None
        scope = "simcom_com_code_change_hierarchical_cnn_rows"
        component = "Com"

    cnn_model.eval()
    for index in range(len(frame)):
        row = frame.iloc[index]
        commit_id = str(row["commit_id"])
        if commit_id in completed or (args.commit_id and commit_id != str(args.commit_id)):
            continue
        try:
            if args.model == "simcom":
                record = _attribute_simcom_commit(
                    args, row, commit_id, cnn_model, com_wrapper,
                    sim_scores, provenance_index, scope, component,
                )
                records.append(record)
                completed.add(commit_id)
                write_jsonl(jsonl_path, records)
                continue
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
