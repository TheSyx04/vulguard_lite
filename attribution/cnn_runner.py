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
                     write_hunk_report, write_line_csv, write_row_csv)
from .hierarchical_gradcam import hierarchical_row_gradcam, rank_observed_rows
from .line_ranking import (aggregate_line_scores, load_provenance_index,
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
        "hunk_chunk_size": getattr(args, "hunk_chunk_size", 10),
        "seed": args.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _validate_resume(path, current):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        previous = json.load(handle)
    keys = ("model", "fingerprints", "target_class", "threshold", "top_k", "commit_id",
            "only_predicted_vulnerable", "line_aggregation", "hunk_chunk_size")
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


def _serialize_chunk_rows(provenance, line_ids, model_name):
    """Serialize one hunk slice and retain exact line IDs for every model row."""
    by_id = {int(line["line_id"]): line for line in provenance["lines"]}
    selected = set(line_ids)
    row_groups = []
    if model_name == "deepjit":
        added = [line_id for line_id in line_ids
                 if by_id[line_id]["change_type"] == "added"]
        deleted = [line_id for line_id in line_ids
                   if by_id[line_id]["change_type"] == "deleted"]
        text = "<ADD> " + " ".join(by_id[x]["normalized_text"] for x in added)
        text += " <REMOVE> " + " ".join(by_id[x]["normalized_text"] for x in deleted)
        row_groups.append({"text": text.strip(), "first_line_ids": added,
                           "second_line_ids": deleted})
    else:
        for block in provenance["blocks"]:
            deleted = [int(x) for x in block["deleted_line_ids"] if int(x) in selected]
            added = [int(x) for x in block["added_line_ids"] if int(x) in selected]
            if not deleted and not added:
                continue
            parts = ["<ADD>"]
            parts.extend(by_id[x]["normalized_text"] for x in deleted)
            parts.append("<REMOVE>")
            parts.extend(by_id[x]["normalized_text"] for x in added)
            row_groups.append({"text": " ".join(parts), "first_line_ids": deleted,
                               "second_line_ids": added})
    if not row_groups:
        raise ValueError("hunk_chunk_has_no_serialized_rows")
    return row_groups


def build_hunk_chunks(provenance, row, model_name, max_changed_lines=10):
    """Create chunks that never cross a Git hunk and contain at most N changed lines."""
    if max_changed_lines < 1:
        raise ValueError("hunk_chunk_size_must_be_positive")
    grouped = []
    current_key = None
    for line in provenance["lines"]:
        key = (line["file_path"], int(line["hunk_id"]))
        if key != current_key:
            grouped.append({"key": key, "lines": []})
            current_key = key
        grouped[-1]["lines"].append(line)

    chunks = []
    for hunk_index, group in enumerate(grouped):
        lines = group["lines"]
        part_count = (len(lines) + max_changed_lines - 1) // max_changed_lines
        for part_index, start in enumerate(range(0, len(lines), max_changed_lines)):
            selected_lines = lines[start:start + max_changed_lines]
            line_ids = [int(line["line_id"]) for line in selected_lines]
            row_groups = _serialize_chunk_rows(provenance, line_ids, model_name)
            chunk_row = row.copy()
            chunk_row["code_change"] = "\n".join(group["text"] for group in row_groups)
            chunks.append({
                "chunk_id": len(chunks), "hunk_index": hunk_index,
                "file_path": group["key"][0], "hunk_id": group["key"][1],
                "hunk_header": selected_lines[0]["hunk_header"],
                "hunk_part": part_index, "hunk_part_count": part_count,
                "hunk_line_start": start,
                "hunk_line_end_exclusive": start + len(selected_lines),
                "changed_line_count": len(selected_lines), "line_ids": line_ids,
                "rows": [group["text"] for group in row_groups],
                "row_groups": row_groups, "data": chunk_row,
            })
    return chunks


def _chunk_token_position_scores(provenance, chunk, token_scores):
    by_id = {int(line["line_id"]): line for line in provenance["lines"]}
    position_scores = []
    for row_position, group in enumerate(chunk["row_groups"]):
        position = 1  # first model marker
        for line_id in group["first_line_ids"]:
            for _ in by_id[line_id]["normalized_text"].split():
                if row_position < token_scores.shape[0] and position < token_scores.shape[1]:
                    position_scores.append((line_id, token_scores[row_position, position].item()))
                position += 1
        position += 1  # second model marker
        for line_id in group["second_line_ids"]:
            for _ in by_id[line_id]["normalized_text"].split():
                if row_position < token_scores.shape[0] and position < token_scores.shape[1]:
                    position_scores.append((line_id, token_scores[row_position, position].item()))
                position += 1
    return position_scores


def _with_chunk_fields(items, chunk):
    return [dict(item, chunk_id=chunk["chunk_id"], file_path=chunk["file_path"],
                 hunk_id=chunk["hunk_id"], hunk_part=chunk["hunk_part"],
                 hunk_part_count=chunk["hunk_part_count"])
            for item in items]


def _attribute_hunk_chunked_commit(args, row, commit_id, model_name, cnn_model,
                                   model_wrapper, sim_scores, provenance_index,
                                   scope, component):
    if provenance_index is None:
        raise ValueError("hunk_chunking_requires_line_provenance")
    if commit_id not in provenance_index:
        raise ValueError("missing_line_provenance")
    provenance = provenance_index[commit_id]
    kind = "merge" if model_name == "deepjit" else "patch"
    verify_serialization(provenance, str(row["code_change"]), kind)
    chunk_size = getattr(args, "hunk_chunk_size", 10)
    model_line_limit = int(model_wrapper.hyperparameters["code_line"])
    if chunk_size > model_line_limit:
        raise ValueError(
            f"hunk_chunk_size_exceeds_model_code_line:{chunk_size}>{model_line_limit}"
        )
    chunks = build_hunk_chunks(provenance, row, model_name, chunk_size)
    if not chunks:
        raise ValueError("commit_has_no_rankable_hunk_chunks")

    dataset_class = DeepJITDataset if model_name == "deepjit" else ComDataset
    chunk_records, flat_rows, chunk_top_lines, all_position_scores = [], [], [], []
    prediction_scores = []
    for chunk in chunks:
        if len(chunk["rows"]) > model_line_limit:
            raise ValueError(
                f"serialized_hunk_rows_exceed_model_code_line:chunk_{chunk['chunk_id']}"
            )
        chunk_dataset = dataset_class(
            pd.DataFrame([chunk["data"]]), model_wrapper.hyperparameters,
            model_wrapper.code_dictionary, model_wrapper.message_dictionary,
        )
        sample = chunk_dataset[0]
        code = sample["code"].unsqueeze(0).to(args.device)
        message = sample["message"].unsqueeze(0).to(args.device)
        with torch.no_grad():
            normal_score = float(cnn_model(message, code)[0].item())
        gradcam = hierarchical_row_gradcam(cnn_model, message, code, args.target_class)
        component_score = float(gradcam["probability"][0].item())
        tolerance = 1e-5 if str(args.device).startswith("cuda") else 1e-6
        if abs(normal_score - component_score) >= tolerance:
            raise ValueError(f"prediction_preservation_failed:chunk_{chunk['chunk_id']}")

        diagnostics = None
        chunk_prediction = component_score
        if sim_scores is not None:
            if commit_id not in sim_scores:
                raise ValueError("missing_sim_score")
            diagnostics = _component_diagnostics(
                float(sim_scores[commit_id]), component_score, args.threshold,
            )
            chunk_prediction = diagnostics["final_score"]
        prediction_scores.append(chunk_prediction)

        ranked_rows = rank_observed_rows(
            chunk["rows"], gradcam["row_scores"][0].cpu().tolist(),
            "mean_cam_over_exact_kernel_receptive_fields", args.top_k,
        )
        row_dicts = _with_chunk_fields([item.to_dict() for item in ranked_rows], chunk)
        flat_rows.extend(row_dicts)
        position_scores = _chunk_token_position_scores(
            provenance, chunk, gradcam["token_scores"][0],
        )
        all_position_scores.extend(position_scores)
        line_result = aggregate_line_scores(
            provenance, position_scores, getattr(args, "line_aggregation", "sum"),
            args.top_k, model_input_kind=kind,
        )
        ranked_lines = _with_chunk_fields(line_result["ranked_lines"], chunk)
        top_line = ranked_lines[0] if ranked_lines else None
        if top_line is not None:
            chunk_top_lines.append(top_line)
        chunk_records.append({
            "chunk_id": chunk["chunk_id"], "file_path": chunk["file_path"],
            "hunk_id": chunk["hunk_id"], "hunk_header": chunk["hunk_header"],
            "hunk_part": chunk["hunk_part"],
            "hunk_part_count": chunk["hunk_part_count"],
            "hunk_line_start": chunk["hunk_line_start"],
            "hunk_line_end_exclusive": chunk["hunk_line_end_exclusive"],
            "changed_line_count": chunk["changed_line_count"],
            "line_ids": chunk["line_ids"], "model_row_count": len(chunk["rows"]),
            "commit_message_repeated": True,
            "component_score": component_score, "prediction_score": chunk_prediction,
            "predicted_label": int(chunk_prediction > args.threshold),
            "ranked_rows": row_dicts, "ranked_lines": ranked_lines,
            "top_line": top_line,
            "metadata": {
                "component_scores": diagnostics,
                "prediction_preservation_delta": abs(normal_score - component_score),
                "branch_activation_shapes": gradcam["branch_activation_shapes"],
                "token_branch_activation_shapes": gradcam["token_branch_activation_shapes"],
            },
        })

    prediction_score = max(prediction_scores)
    if args.only_predicted_vulnerable and prediction_score <= args.threshold:
        return {"commit_id": commit_id, "status": "skipped",
                "reason": "filtered_not_predicted_vulnerable"}
    coverage = aggregate_line_scores(
        provenance, all_position_scores, getattr(args, "line_aggregation", "sum"),
        None, model_input_kind=kind,
    )
    record = {
        "commit_id": commit_id, "status": "succeeded", "model_name": model_name,
        "prediction_score": prediction_score,
        "predicted_label": int(prediction_score > args.threshold),
        "threshold": args.threshold, "attribution_method": "hierarchical_gradcam",
        "target_class": args.target_class, "attribution_is_class_specific": True,
        "explanation_scope": scope, "localization_component": component,
        "full_model_explanation": False, "ranked_rows": flat_rows,
        "ranked_lines": chunk_top_lines, "chunk_top_lines": chunk_top_lines,
        "chunks": chunk_records,
        "input_changed_lines": len(provenance["lines"]),
        "observed_changed_lines": coverage["covered_changed_lines"],
        # Backward-compatible aliases; these now count canonical changed lines.
        "input_rows": len(provenance["lines"]),
        "observed_rows": coverage["covered_changed_lines"],
        "truncated_rows": len(coverage["uncovered_lines"]),
        "truncated": bool(coverage["uncovered_lines"]),
        "metadata": {
            "row_semantics": "hunk_bounded_model_rows",
            "source_line_provenance_verified": True,
            "serialization_alignment_status": "exact_full_input_before_chunking",
            "chunking": {
                "strategy": "git_hunk_then_changed_line_limit",
                "max_changed_lines_per_chunk": chunk_size,
                "chunk_count": len(chunks),
                "commit_message_repeated_per_chunk": True,
                "prediction_aggregation": "max_chunk_probability",
                "line_selection": "top_1_per_chunk",
            },
            "prediction_scope": (
                "max_chunk_full_simcom_mean" if sim_scores is not None else
                "max_chunk_com_component_only" if model_name == "simcom" else
                "max_chunk_deepjit_probability"
            ),
            "unexplained_components": ["Sim"] if model_name == "simcom" else ["message_branch"],
        },
    }
    for key in ("uncovered_lines", "total_changed_lines", "covered_changed_lines",
                "coverage_ratio", "line_aggregation"):
        record[key] = coverage[key]
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
    hunk_report_path = os.path.join(args.output_dir, "hunk_line_report.jsonl")
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
        dataset = None  # DeepJIT samples are built per hunk chunk below.
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
            model_wrapper = wrapper if args.model == "deepjit" else com_wrapper
            record = _attribute_hunk_chunked_commit(
                args, row, commit_id, args.model, cnn_model, model_wrapper,
                sim_scores, provenance_index, scope, component,
            )
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
        report_records = write_hunk_report(
            hunk_report_path, records, provenance_index,
            getattr(args, "hunk_chunk_size", 10),
        )
    else:
        report_records = []
    summary = {
        "processed": len(records),
        "succeeded": sum(r.get("status") == "succeeded" for r in records),
        "skipped": sum(r.get("status") == "skipped" for r in records),
        "failed": sum(r.get("status") == "failed" for r in records),
        "hunk_report_records": len(report_records),
    }
    write_json(summary_path, summary)
    print(f"Row attribution complete: {summary['succeeded']} succeeded, "
          f"{summary['skipped']} skipped, {summary['failed']} failed")
    return summary
