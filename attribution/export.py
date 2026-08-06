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
