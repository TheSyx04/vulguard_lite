"""Prepare provenance-rich changed lines while preserving legacy model inputs."""

import json
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple

from .export import write_jsonl


HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
)
COMMIT_URL = re.compile(r"https?://github\.com/[^/]+/[^/]+/commit/([0-9a-fA-F]{7,40})")

LANGUAGE_EXTENSIONS = {
    "C": {"c", "h"}, "C++": {"cc", "cpp", "cxx", "h", "hh", "hpp"},
    "C#": {"cs"}, "Python": {"py"}, "Java": {"java"},
    "JavaScript": {"js", "jsx"}, "TypeScript": {"ts", "tsx"},
    "Ruby": {"rb"}, "PHP": {"php"}, "Go": {"go"}, "Swift": {"swift"},
}


def normalize_model_text(value: str) -> str:
    """Reproduce VulGuard's historical split_sentence/get_std_str path."""
    value = value.replace("_", " ")
    for character in ".@-~%^&*()+={}|\\[]:;,<>?/":
        value = value.replace(character, f" {character} ")
    return " ".join(value.strip().split()).lower()


def _git(repo_path: str, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", repo_path, *arguments], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout.decode("utf-8", errors="replace")


def resolve_commit(repo_path: str, reference: str) -> str:
    match = COMMIT_URL.fullmatch(reference.strip())
    candidate = match.group(1) if match else reference.strip()
    resolved = _git(repo_path, "rev-parse", "--verify", f"{candidate}^{{commit}}").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise ValueError(f"invalid_resolved_commit:{reference}")
    return resolved


def load_commit_references(path: Optional[str], commit_id: Optional[str]) -> List[Dict]:
    if commit_id:
        return [{"reference": commit_id}]
    if not path:
        raise ValueError("provide -commit_id or -commit_urls")
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                value = json.loads(line)
                reference = value.get("commit_url") or value.get("url") or value.get("commit_id")
                if not reference:
                    raise ValueError(f"missing_commit_reference_at_line:{line_number}")
                records.append({**value, "reference": reference})
            else:
                records.append({"reference": line})
    return records


def _display_path(header: str) -> Optional[str]:
    value = header[4:].strip()
    if value == "/dev/null":
        return None
    if value.startswith(('"a/', '"b/')) and value.endswith('"'):
        value = value[3:-1]
    elif value.startswith(("a/", "b/")):
        value = value[2:]
    return value


def parse_unified_diff(diff_text: str, language: str) -> Tuple[List[Dict], List[Dict]]:
    """Parse changed lines and historical blocks separated by context lines."""
    allowed = LANGUAGE_EXTENSIONS.get(language, set())
    lines: List[Dict] = []
    blocks: List[Dict] = []
    old_path = new_path = None
    old_line = new_line = None
    hunk_id = -1
    diff_position = 0
    current_block = None
    eligible_file = False

    def finish_block():
        nonlocal current_block
        if current_block and (current_block["deleted_line_ids"] or current_block["added_line_ids"]):
            current_block["block_id"] = len(blocks)
            blocks.append(current_block)
        current_block = None

    for raw in diff_text.splitlines():
        if raw.startswith("diff --git "):
            finish_block()
            old_path = new_path = None
            eligible_file = False
            hunk_id = -1
            diff_position = 0
            continue
        if raw.startswith("--- "):
            old_path = _display_path(raw)
            continue
        if raw.startswith("+++ "):
            new_path = _display_path(raw)
            selected = new_path or old_path or ""
            extension = selected.rsplit(".", 1)[-1].lower() if "." in selected else ""
            # Historical Miner skipped newly created files because it required
            # a parent-side file for blame. Preserve that selection here so
            # reconstructed checkpoint inputs remain comparable.
            eligible_file = old_path is not None and (not allowed or extension in allowed)
            continue
        match = HUNK_HEADER.match(raw)
        if match:
            finish_block()
            hunk_id += 1
            old_line, new_line = int(match.group(1)), int(match.group(3))
            hunk_header = raw
            continue
        if old_line is None or new_line is None or not raw or raw[0] not in " +-":
            continue
        action, text = raw[0], raw[1:]
        diff_position += 1
        if action == " ":
            finish_block()
            old_line += 1
            new_line += 1
            continue
        if not eligible_file:
            if action == "-": old_line += 1
            if action == "+": new_line += 1
            continue
        if current_block is None:
            current_block = {"file_path": new_path or old_path, "hunk_id": hunk_id,
                             "hunk_header": hunk_header, "deleted_line_ids": [],
                             "added_line_ids": []}
        change_type = "deleted" if action == "-" else "added"
        line = {
            "line_id": len(lines), "file_path": new_path or old_path,
            "old_file_path": old_path, "new_file_path": new_path,
            "hunk_id": hunk_id, "hunk_header": hunk_header,
            "diff_position": diff_position,
            "old_line_no": old_line if action == "-" else None,
            "new_line_no": new_line if action == "+" else None,
            "change_type": change_type, "text": text,
            "normalized_text": normalize_model_text(text),
            # Merge uses semantic markers; legacy patch rows reversed them.
            "model_markers": {
                "merge": "REMOVE" if action == "-" else "ADD",
                "patch": "ADD" if action == "-" else "REMOVE",
            },
            "eligible_for_ranking": True,
        }
        lines.append(line)
        current_block[f"{change_type}_line_ids"].append(line["line_id"])
        if action == "-": old_line += 1
        else: new_line += 1
    finish_block()
    return lines, blocks


def serialize_historical(lines: List[Dict], blocks: List[Dict]) -> Dict:
    by_id = {line["line_id"]: line for line in lines}
    deleted = [line for line in lines if line["change_type"] == "deleted"]
    added = [line for line in lines if line["change_type"] == "added"]
    merge = "<ADD> " + " ".join(line["normalized_text"] for line in added)
    merge += " <REMOVE> " + " ".join(line["normalized_text"] for line in deleted) + "\n"
    patch_rows = []
    for block in blocks:
        deleted_text = " ".join(by_id[i]["normalized_text"] for i in block["deleted_line_ids"])
        added_text = " ".join(by_id[i]["normalized_text"] for i in block["added_line_ids"])
        row_parts = ["<ADD>"]
        if deleted_text:
            row_parts.append(deleted_text)
        row_parts.append("<REMOVE>")
        if added_text:
            row_parts.append(added_text)
        patch_rows.append({
            "row_position": len(patch_rows),
            "text": " ".join(row_parts),
            "deleted_line_ids": block["deleted_line_ids"],
            "added_line_ids": block["added_line_ids"],
        })
    return {
        "merge": merge, "patch": "\n".join(row["text"] for row in patch_rows),
        "merge_segments": [
            *[{"line_id": x["line_id"], "model_region": "added", "text": x["normalized_text"]} for x in added],
            *[{"line_id": x["line_id"], "model_region": "removed", "text": x["normalized_text"]} for x in deleted],
        ],
        "patch_rows": patch_rows,
    }


def prepare_commit(repo_path: str, reference: str, language: str, extra=None) -> Dict:
    commit_id = resolve_commit(repo_path, reference)
    parents = _git(repo_path, "show", "-s", "--format=%P", commit_id).strip().split()
    if not parents:
        raise ValueError(f"root_commit_not_supported:{commit_id}")
    parent_id = parents[0]
    message = normalize_model_text(_git(repo_path, "show", "-s", "--format=%B", commit_id))
    diff = _git(repo_path, "diff", "--no-color", "--no-ext-diff", "--unified=3",
                "--find-renames", parent_id, commit_id, "--")
    lines, blocks = parse_unified_diff(diff, language)
    serialization = serialize_historical(lines, blocks)
    return {
        "schema_version": 1, "commit_id": commit_id, "input_reference": reference,
        "parent_id": parent_id, "parent_selection": "first_parent",
        "message": message, "lines": lines, "blocks": blocks,
        "historical_serialization": serialization,
        "metadata": {"merge_parent_count": len(parents), "language": language,
                     **({k: v for k, v in (extra or {}).items() if k != "reference"})},
    }


def prepare_lines(args):
    if not args.repo_path or not os.path.isdir(os.path.join(args.repo_path, ".git")):
        raise ValueError("-repo_path must point to a local Git clone")
    references = load_commit_references(args.commit_urls, args.commit_id)
    records = [prepare_commit(args.repo_path, item["reference"], args.repo_language, item)
               for item in references]
    provenance_path = os.path.join(args.output_dir, "line_provenance.jsonl")
    merge_path = os.path.join(args.output_dir, "merge.jsonl")
    patch_path = os.path.join(args.output_dir, "patch.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    write_jsonl(provenance_path, records)
    write_jsonl(merge_path, [{"commit_id": r["commit_id"], "messages": r["message"],
                              "code_change": r["historical_serialization"]["merge"]}
                             for r in records])
    write_jsonl(patch_path, [{"commit_id": r["commit_id"], "messages": r["message"],
                              "code_change": r["historical_serialization"]["patch"]}
                             for r in records])
    print(f"Prepared {len(records)} commits with line provenance: {args.output_dir}")
    return {"processed": len(records), "provenance": provenance_path,
            "merge": merge_path, "patch": patch_path}
