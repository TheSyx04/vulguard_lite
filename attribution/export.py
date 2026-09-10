import csv
import hashlib
import io
import json
import os
import tempfile
from typing import Any, Dict, Iterable, List


def file_fingerprint(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: str, content: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix=".tmp-", dir=directory, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path: str, value: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    atomic_write_text(path, content)


CSV_COLUMNS = [
    "commit_id", "model", "prediction_score", "predicted_label",
    "attention_strategy", "rank", "sequence_position", "token", "token_id",
    "change_type", "raw_score", "normalized_score", "truncated",
]


def write_token_csv(path: str, records: Iterable[Dict[str, Any]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        if record.get("status") != "succeeded":
            continue
        for token in record.get("ranked_tokens", []):
            writer.writerow({
                "commit_id": record["commit_id"],
                "model": record["model_name"],
                "prediction_score": record["prediction_score"],
                "predicted_label": record["predicted_label"],
                "attention_strategy": record["attention_strategy"],
                "rank": token["rank"],
                "sequence_position": token["sequence_position"],
                "token": token["token"],
                "token_id": token["token_id"],
                "change_type": token["change_type"],
                "raw_score": token["raw_score"],
                "normalized_score": token["normalized_score"],
                "truncated": record["truncated"],
            })
    atomic_write_text(path, output.getvalue())


ROW_CSV_COLUMNS = [
    "commit_id", "model", "prediction_score", "predicted_label", "target_class",
    "chunk_id", "file_path", "hunk_id", "hunk_part", "hunk_part_count", "rank",
    "row_position", "text", "raw_score", "normalized_score", "truncated",
]


def write_row_csv(path: str, records: Iterable[Dict[str, Any]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=ROW_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        if record.get("status") != "succeeded":
            continue
        for row in record.get("ranked_rows", []):
            writer.writerow({
                "commit_id": record["commit_id"], "model": record["model_name"],
                "prediction_score": record["prediction_score"],
                "predicted_label": record["predicted_label"],
                "target_class": record["target_class"],
                "chunk_id": row.get("chunk_id"),
                "file_path": row.get("file_path"), "hunk_id": row.get("hunk_id"),
                "hunk_part": row.get("hunk_part"),
                "hunk_part_count": row.get("hunk_part_count"),
                "rank": row["rank"],
                "row_position": row["row_position"], "text": row["text"],
                "raw_score": row["raw_score"], "normalized_score": row["normalized_score"],
                "truncated": record["truncated"],
            })
    atomic_write_text(path, output.getvalue())


LINE_CSV_COLUMNS = [
    "commit_id", "model", "prediction_score", "predicted_label", "rank",
    "chunk_id", "hunk_part", "hunk_part_count",
    "line_id", "file_path", "hunk_id", "diff_position", "old_line_no",
    "new_line_no", "change_type", "model_marker", "text", "raw_score",
    "normalized_score", "attributed_position_count", "coverage_ratio",
]


def write_line_csv(path: str, records: Iterable[Dict[str, Any]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=LINE_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        if record.get("status") != "succeeded":
            continue
        for line in record.get("ranked_lines", []):
            writer.writerow({
                "commit_id": record["commit_id"], "model": record["model_name"],
                "prediction_score": record["prediction_score"],
                "predicted_label": record["predicted_label"], "rank": line["rank"],
                "chunk_id": line.get("chunk_id"),
                "hunk_part": line.get("hunk_part"),
                "hunk_part_count": line.get("hunk_part_count"),
                "line_id": line["line_id"], "file_path": line["file_path"],
                "hunk_id": line["hunk_id"], "diff_position": line["diff_position"],
                "old_line_no": line["old_line_no"], "new_line_no": line["new_line_no"],
                "change_type": line["change_type"], "model_marker": line["model_marker"],
                "text": line["text"], "raw_score": line["raw_score"],
                "normalized_score": line["normalized_score"],
                "attributed_position_count": line["attributed_position_count"],
                "coverage_ratio": record.get("coverage_ratio"),
            })
    atomic_write_text(path, output.getvalue())


def build_hunk_report_records(records, provenance_index, chunk_limit=10):
    """Build review-oriented JSONL records: one commit summary and one record per hunk."""
    output = []
    by_commit = {str(record.get("commit_id")): record for record in records}
    for commit_id, provenance in provenance_index.items():
        attribution = by_commit.get(str(commit_id))
        if attribution is None or attribution.get("status") != "succeeded":
            output.append({
                "record_type": "commit_summary", "commit_id": str(commit_id),
                "status": attribution.get("status", "missing") if attribution else "missing",
                "reason": attribution.get("reason") if attribution else "missing_attribution",
                "hunk_count": 0, "oversized_hunks": [],
            })
            continue

        hunk_groups = []
        group_index = {}
        for line in provenance["lines"]:
            key = (line["file_path"], int(line["hunk_id"]))
            if key not in group_index:
                group_index[key] = len(hunk_groups)
                hunk_groups.append({"key": key, "header": line["hunk_header"], "lines": []})
            hunk_groups[group_index[key]]["lines"].append(line)

        chunks_by_hunk = {}
        for chunk in attribution.get("chunks", []):
            key = (chunk["file_path"], int(chunk["hunk_id"]))
            chunks_by_hunk.setdefault(key, []).append(chunk)
        oversized = []
        for group in hunk_groups:
            if len(group["lines"]) > chunk_limit:
                oversized.append({
                    "file_path": group["key"][0], "hunk_id": group["key"][1],
                    "changed_line_count": len(group["lines"]),
                    "subchunk_count": len(chunks_by_hunk.get(group["key"], [])),
                })
        output.append({
            "record_type": "commit_summary", "commit_id": str(commit_id),
            "status": "succeeded", "prediction_score": attribution["prediction_score"],
            "predicted_label": attribution["predicted_label"],
            "hunk_count": len(hunk_groups), "chunk_count": len(attribution.get("chunks", [])),
            "chunk_limit_changed_lines": chunk_limit,
            "oversized_hunk_count": len(oversized), "oversized_hunks": oversized,
            "coverage_ratio": attribution.get("coverage_ratio"),
        })

        for hunk_index, group in enumerate(hunk_groups):
            chunks = sorted(chunks_by_hunk.get(group["key"], []),
                            key=lambda item: item["hunk_part"])
            scored_lines = {}
            for chunk in chunks:
                for ranked in chunk.get("ranked_lines", []):
                    scored_lines[int(ranked["line_id"])] = ranked
            lines = []
            for line in group["lines"]:
                ranked = scored_lines.get(int(line["line_id"]), {})
                lines.append({
                    "line_id": int(line["line_id"]),
                    "diff_position": line["diff_position"],
                    "old_line_no": line["old_line_no"], "new_line_no": line["new_line_no"],
                    "change_type": line["change_type"], "text": line["text"],
                    "normalized_text": line["normalized_text"],
                    "chunk_id": ranked.get("chunk_id"),
                    "hunk_part": ranked.get("hunk_part"),
                    "rank_in_chunk": ranked.get("rank"),
                    "raw_score": ranked.get("raw_score"),
                    "normalized_score": ranked.get("normalized_score"),
                    "attributed_position_count": ranked.get("attributed_position_count", 0),
                    "covered": bool(ranked),
                })
            output.append({
                "record_type": "hunk", "commit_id": str(commit_id),
                "hunk_index": hunk_index, "file_path": group["key"][0],
                "hunk_id": group["key"][1], "hunk_header": group["header"],
                "changed_line_count": len(group["lines"]),
                "exceeded_chunk_limit": len(group["lines"]) > chunk_limit,
                "subchunk_count": len(chunks),
                "subchunks": [{
                    "chunk_id": chunk["chunk_id"], "hunk_part": chunk["hunk_part"],
                    "changed_line_count": chunk["changed_line_count"],
                    "line_ids": chunk["line_ids"], "top_line": chunk.get("top_line"),
                } for chunk in chunks],
                "lines": lines,
            })
    return output


def write_hunk_report(path, records, provenance_index, chunk_limit=10):
    report_records = build_hunk_report_records(records, provenance_index, chunk_limit)
    write_jsonl(path, report_records)
    return report_records
