"""Verified mappings from historical model positions to canonical Git lines."""

import math
from typing import Dict, Iterable, List, Sequence

from .export import load_jsonl


def load_provenance_index(path: str) -> Dict[str, Dict]:
    return {str(record["commit_id"]): record for record in load_jsonl(path)}


def verify_serialization(provenance: Dict, code_change: str, kind: str) -> None:
    expected = provenance["historical_serialization"][kind]
    if expected != code_change:
        raise ValueError(f"{kind}_serialization_mismatch")


def _tokenize_segments(tokenizer, segments: List[Dict], model_region: str):
    selected = [segment for segment in segments if segment["model_region"] == model_region]
    full_text = "".join(" " + segment["text"] for segment in selected)
    expected = tokenizer.tokenize(full_text)
    tokens, line_ids = [], []
    for segment in selected:
        current = tokenizer.tokenize(" " + segment["text"])
        tokens.extend(current)
        line_ids.extend([segment["line_id"]] * len(current))
    if tokens != expected:
        raise ValueError(f"{model_region}_tokenizer_segment_alignment_mismatch")
    return tokens, line_ids


def jitfine_position_line_ids(tokenizer, example, provenance: Dict) -> Dict[int, int]:
    segments = provenance["historical_serialization"]["merge_segments"]
    mapping = {}
    for region in ("added", "removed"):
        expected_tokens, line_ids = _tokenize_segments(tokenizer, segments, region)
        positions = [index for index, value in
                     enumerate(example.attribution_metadata["sequence_regions"])
                     if value == region]
        observed_expected = expected_tokens[:len(positions)]
        observed_actual = [example.input_tokens[position] for position in positions]
        if observed_expected != observed_actual:
            raise ValueError(f"{region}_observed_token_alignment_mismatch")
        mapping.update({position: line_id for position, line_id in
                        zip(positions, line_ids[:len(positions)])})
    return mapping


def merge_token_line_ids(provenance: Dict) -> Dict[int, int]:
    """Whitespace-token positions in the legacy flattened merge row."""
    segments = provenance["historical_serialization"]["merge_segments"]
    mapping = {}
    position = 1  # <ADD>
    for segment in (x for x in segments if x["model_region"] == "added"):
        for _ in segment["text"].split():
            mapping[position] = segment["line_id"]
            position += 1
    position += 1  # <REMOVE>
    for segment in (x for x in segments if x["model_region"] == "removed"):
        for _ in segment["text"].split():
            mapping[position] = segment["line_id"]
            position += 1
    expected = provenance["historical_serialization"]["merge"].splitlines()[0].split()
    if position != len(expected):
        raise ValueError("merge_whitespace_token_alignment_mismatch")
    return mapping


def patch_token_line_ids(provenance: Dict) -> Dict[tuple, int]:
    """Whitespace-token positions for every historical patch row."""
    by_id = {line["line_id"]: line for line in provenance["lines"]}
    mapping = {}
    for row in provenance["historical_serialization"]["patch_rows"]:
        position = 1  # <ADD>
        for line_id in row["deleted_line_ids"]:
            for _ in by_id[line_id]["normalized_text"].split():
                mapping[(row["row_position"], position)] = line_id
                position += 1
        position += 1  # <REMOVE>
        for line_id in row["added_line_ids"]:
            for _ in by_id[line_id]["normalized_text"].split():
                mapping[(row["row_position"], position)] = line_id
                position += 1
        if position != len(row["text"].split()):
            raise ValueError(f"patch_whitespace_token_alignment_mismatch:{row['row_position']}")
    return mapping


def aggregate_line_scores(
        provenance: Dict, position_scores: Iterable, strategy: str,
        top_k=None, model_input_kind="merge") -> Dict:
    if strategy not in {"sum", "mean", "max"}:
        raise ValueError(f"unsupported_line_aggregation:{strategy}")
    grouped: Dict[int, List[float]] = {}
    for line_id, score in position_scores:
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("non_finite_line_source_score")
        grouped.setdefault(int(line_id), []).append(value)
    by_id = {int(line["line_id"]): line for line in provenance["lines"]}
    candidates = []
    for line_id, scores in grouped.items():
        if strategy == "sum": value = sum(scores)
        elif strategy == "mean": value = sum(scores) / len(scores)
        else: value = max(scores)
        candidates.append((line_id, value, scores))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    values = [item[1] for item in candidates]
    minimum, maximum = (min(values), max(values)) if values else (0.0, 0.0)
    span = maximum - minimum
    ranked = []
    for rank, (line_id, value, scores) in enumerate(candidates, 1):
        line = by_id[line_id]
        ranked.append({
            "rank": rank, "line_id": line_id, "file_path": line["file_path"],
            "hunk_id": line["hunk_id"], "diff_position": line["diff_position"],
            "old_line_no": line["old_line_no"], "new_line_no": line["new_line_no"],
            "change_type": line["change_type"],
            "model_marker": line["model_markers"][model_input_kind],
            "model_markers": line["model_markers"],
            "text": line["text"], "normalized_text": line["normalized_text"],
            "raw_score": value,
            "normalized_score": 0.0 if span <= 1e-12 else (value - minimum) / span,
            "attributed_position_count": len(scores), "covered": True,
        })
    uncovered = [{
        "line_id": line_id, "file_path": line["file_path"],
        "old_line_no": line["old_line_no"], "new_line_no": line["new_line_no"],
        "change_type": line["change_type"], "text": line["text"],
        "raw_score": None, "normalized_score": None, "covered": False,
    } for line_id, line in by_id.items() if line_id not in grouped]
    return {
        "ranked_lines": ranked if top_k is None else ranked[:top_k],
        "uncovered_lines": uncovered,
        "total_changed_lines": len(by_id), "covered_changed_lines": len(grouped),
        "coverage_ratio": len(grouped) / len(by_id) if by_id else 0.0,
        "line_aggregation": strategy,
    }
