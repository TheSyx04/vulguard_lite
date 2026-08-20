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
    "chunk_id", "chunk_row_start", "chunk_row_end_exclusive", "rank",
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
                "chunk_row_start": row.get("chunk_row_start"),
                "chunk_row_end_exclusive": row.get("chunk_row_end_exclusive"),
                "rank": row["rank"],
                "row_position": row["row_position"], "text": row["text"],
                "raw_score": row["raw_score"], "normalized_score": row["normalized_score"],
                "truncated": record["truncated"],
            })
    atomic_write_text(path, output.getvalue())


LINE_CSV_COLUMNS = [
    "commit_id", "model", "prediction_score", "predicted_label", "rank",
    "chunk_id", "chunk_row_start", "chunk_row_end_exclusive",
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
                "chunk_row_start": line.get("chunk_row_start"),
                "chunk_row_end_exclusive": line.get("chunk_row_end_exclusive"),
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
