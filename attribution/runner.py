import importlib.metadata
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List

import torch

from ..models.jitfine.dataset import TextDataset
from ..models.jitfine.warper import JITFine
from .export import (
    file_fingerprint,
    load_jsonl,
    write_json,
    write_jsonl,
    write_line_csv,
    write_token_csv,
)
from .jitfine_attention import aggregate_cls_attention, rank_code_tokens
from .line_ranking import (aggregate_line_scores, jitfine_position_line_ids,
                           load_provenance_index, verify_serialization)
from .schemas import TokenAttributionResult


EXPLANATION_SCOPE = "jitfine_code_tokens_within_joint_message_code_encoder_input"


def _checkpoint_path(model_path: str) -> str:
    return model_path if os.path.isfile(model_path) else os.path.join(model_path, "jitfine.pth")


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _build_run_metadata(args, feature_path: str, code_path: str) -> Dict[str, Any]:
    checkpoint_path = _checkpoint_path(args.model_path)
    fingerprints = {
        "checkpoint": file_fingerprint(checkpoint_path),
        "hyperparameters": file_fingerprint(args.hyperparameters),
        "feature_file": file_fingerprint(feature_path),
        "code_file": file_fingerprint(code_path),
    }
    if getattr(args, "line_provenance", None):
        fingerprints["line_provenance"] = file_fingerprint(args.line_provenance)
    selection_mode = "all_commits"
    if args.commit_id:
        selection_mode = "commit_id"
    elif args.only_predicted_vulnerable:
        selection_mode = "only_predicted_vulnerable"
    return {
        "schema_version": 1,
        "model": "jitfine",
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "hyperparameters_path": os.path.abspath(args.hyperparameters),
        "feature_file": os.path.abspath(feature_path),
        "code_file": os.path.abspath(code_path),
        "fingerprints": fingerprints,
        "device": args.device,
        "threshold": args.threshold,
        "attention_strategy": args.attention_strategy,
        "top_k": args.top_k,
        "selection_mode": selection_mode,
        "commit_id": args.commit_id,
        "line_aggregation": getattr(args, "line_aggregation", "sum"),
        "seed": args.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "versions": {
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
        },
    }


def _validate_resume_metadata(path: str, current: Dict[str, Any]) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        previous = json.load(handle)
    comparable = (
        "fingerprints", "threshold", "attention_strategy", "top_k",
        "selection_mode", "commit_id", "line_aggregation",
    )
    if any(previous.get(key) != current.get(key) for key in comparable):
        raise ValueError("resume_configuration_mismatch")


def _failure(commit_id: str, reason: str, status: str = "failed") -> Dict[str, Any]:
    return {"commit_id": str(commit_id), "status": status, "reason": reason}


def _attribute_example(model: JITFine, example, args, checkpoint_id: str, provenance=None):
    input_ids = torch.tensor(example.input_ids, dtype=torch.long, device=model.device).unsqueeze(0)
    attention_mask = torch.tensor(example.input_mask, dtype=torch.long, device=model.device).unsqueeze(0)
    manual_features = torch.tensor(example.manual_features, device=model.device).unsqueeze(0)

    with torch.no_grad():
        normal_probability = model.model(input_ids, attention_mask, manual_features)
        prediction_score = float(normal_probability.reshape(-1)[0].item())

        if args.only_predicted_vulnerable and prediction_score <= args.threshold:
            return _failure(
                example.commit_id,
                "filtered_not_predicted_vulnerable",
                status="skipped",
            )

        attribution_output = model.model(
            input_ids,
            attention_mask,
            manual_features,
            output_attentions=True,
            return_attribution_data=True,
        )

    attribution_probability = float(attribution_output["probability"].reshape(-1)[0].item())
    tolerance = 1e-5 if str(model.device).startswith("cuda") else 1e-6
    if abs(prediction_score - attribution_probability) >= tolerance:
        raise ValueError(
            "prediction_preservation_failed:"
            f"normal={prediction_score},attention={attribution_probability},tolerance={tolerance}"
        )

    scores = aggregate_cls_attention(
        attribution_output["attentions"],
        args.attention_strategy,
    )[0].detach().cpu().tolist()
    metadata = example.attribution_metadata
    ranked_tokens = rank_code_tokens(
        input_tokens=example.input_tokens,
        input_ids=example.input_ids,
        sequence_regions=metadata["sequence_regions"],
        token_scores=scores,
        strategy=args.attention_strategy,
        top_k=args.top_k,
    )
    observed_added = metadata["sequence_regions"].count("added")
    observed_removed = metadata["sequence_regions"].count("removed")
    result = TokenAttributionResult(
        commit_id=str(example.commit_id),
        model_name="jitfine",
        checkpoint_id=checkpoint_id,
        prediction_score=prediction_score,
        predicted_label=int(prediction_score > args.threshold),
        threshold=args.threshold,
        attribution_method="attention",
        attention_strategy=args.attention_strategy,
        explanation_scope=EXPLANATION_SCOPE,
        attribution_is_class_specific=False,
        ranked_tokens=ranked_tokens,
        observed_sequence_tokens=int(sum(example.input_mask)),
        observed_code_tokens=observed_added + observed_removed,
        observed_added_tokens=observed_added,
        observed_removed_tokens=observed_removed,
        pre_truncation_content_tokens=metadata["pre_truncation_content_tokens"],
        truncated_content_tokens=metadata["truncated_content_tokens"],
        truncated=metadata["truncated"],
        metadata={
            "full_model_explanation": False,
            "excluded_model_inputs": ["manual_features"],
            "excluded_sequence_regions": [
                "commit_message", "special_tokens", "padding",
            ],
            "add_marker_position": metadata["add_marker_position"],
            "remove_marker_position": metadata["remove_marker_position"],
            "truncated_added_tokens": metadata["truncated_added_tokens"],
            "truncated_removed_tokens": metadata["truncated_removed_tokens"],
            "prediction_preservation_delta": abs(prediction_score - attribution_probability),
        },
    )
    record = result.to_dict()
    if provenance is not None:
        verify_serialization(provenance, example.code_change, "merge")
        position_to_line = jitfine_position_line_ids(model.tokenizer, example, provenance)
        line_result = aggregate_line_scores(
            provenance,
            ((line_id, scores[position]) for position, line_id in position_to_line.items()),
            getattr(args, "line_aggregation", "sum"), args.top_k,
            model_input_kind="merge",
        )
        record.update(line_result)
        record["metadata"]["source_line_provenance_verified"] = True
        record["metadata"]["serialization_alignment_status"] = "exact"
    return record


def attribute_jitfine(args):
    """CLI entry point for the JITFine token-attention baseline."""
    paths = [part.strip() for part in args.test_set.split(",", 1)]
    if len(paths) != 2 or not all(paths):
        raise ValueError("-test_set must be features.jsonl,code.jsonl")
    feature_path, code_path = paths
    os.makedirs(args.output_dir, exist_ok=True)

    metadata_path = os.path.join(args.output_dir, "run_metadata.json")
    jsonl_path = os.path.join(args.output_dir, "token_attributions.jsonl")
    csv_path = os.path.join(args.output_dir, "token_attributions.csv")
    line_csv_path = os.path.join(args.output_dir, "line_attributions.csv")
    summary_path = os.path.join(args.output_dir, "summary.json")
    output_paths = (metadata_path, jsonl_path, csv_path, summary_path)
    if any(os.path.exists(path) for path in output_paths) and not (args.overwrite or args.resume):
        raise FileExistsError("output exists; use -overwrite or -resume")

    run_metadata = _build_run_metadata(args, feature_path, code_path)
    if args.resume:
        _validate_resume_metadata(metadata_path, run_metadata)
    write_json(metadata_path, run_metadata)

    loaded_records = load_jsonl(jsonl_path) if args.resume else []
    # A failed commit is not complete: retry it on resume while retaining
    # successful and deliberately skipped records.
    records: List[Dict[str, Any]] = [
        record for record in loaded_records if record.get("status") != "failed"
    ]
    completed_ids = {record.get("commit_id") for record in records}
    provenance_index = (
        load_provenance_index(args.line_provenance)
        if getattr(args, "line_provenance", None) else None
    )

    model = JITFine(language=args.repo_language, device=args.device)
    model.initialize(
        hyperparameters=args.hyperparameters,
        model_path=args.model_path,
        inference_only=True,
        attention_implementation="eager",
    )
    model.model.eval()
    dataset = TextDataset(
        tokenizer=model.tokenizer,
        hyperparameters=model.hyperparameters,
        changes_filename=code_path,
        features_filename=feature_path,
        mode="test",
        return_metadata=True,
        skip_invalid=True,
    )

    selected_examples = dataset.examples
    if args.commit_id:
        selected_examples = [
            example for example in selected_examples
            if str(example.commit_id) == str(args.commit_id)
        ]
        if not selected_examples and str(args.commit_id) not in completed_ids:
            raise ValueError(f"commit_id_not_found:{args.commit_id}")

    for failure in dataset.failures:
        commit_id = str(failure["commit_id"])
        if commit_id not in completed_ids and (not args.commit_id or commit_id == str(args.commit_id)):
            reason = failure["reason"]
            status = "skipped" if reason == "missing_manual_features" else "failed"
            records.append(_failure(commit_id, reason, status=status))
            completed_ids.add(commit_id)

    checkpoint_id = run_metadata["fingerprints"]["checkpoint"]
    for example in selected_examples:
        commit_id = str(example.commit_id)
        if commit_id in completed_ids:
            continue
        try:
            provenance = provenance_index.get(commit_id) if provenance_index is not None else None
            if provenance_index is not None and provenance is None:
                raise ValueError("missing_line_provenance")
            record = _attribute_example(model, example, args, checkpoint_id, provenance)
        except Exception as exc:
            record = _failure(commit_id, f"{type(exc).__name__}:{exc}")
        records.append(record)
        completed_ids.add(commit_id)
        # Atomic per-commit checkpoint makes resume useful after interruption.
        write_jsonl(jsonl_path, records)

    write_jsonl(jsonl_path, records)
    write_token_csv(csv_path, records)
    if provenance_index is not None:
        write_line_csv(line_csv_path, records)
    succeeded = sum(record.get("status") == "succeeded" for record in records)
    skipped = sum(record.get("status") == "skipped" for record in records)
    failed = sum(record.get("status") == "failed" for record in records)
    summary = {
        "processed": len(records),
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": failed,
        "failures": [
            {"commit_id": record.get("commit_id"), "reason": record.get("reason")}
            for record in records if record.get("status") == "failed"
        ],
    }
    write_json(summary_path, summary)
    print(f"Token attribution complete: {succeeded} succeeded, {skipped} skipped, {failed} failed")
    print(f"Results: {args.output_dir}")
    return summary


def attribute(args):
    """Dispatch to the attribution implementation matching the selected model."""
    if args.model == "jitfine":
        return attribute_jitfine(args)
    if args.model in {"deepjit", "simcom"}:
        from .cnn_runner import attribute_cnn
        return attribute_cnn(args)
    raise ValueError(f"unsupported attribution model:{args.model}")
