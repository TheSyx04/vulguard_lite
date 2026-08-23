"""End-to-end ground-truth hunk localization pipeline.

The pipeline reads vulnerability-introducing commits and vulnerable source lines
from an XLSX worksheet, reconstructs checkpoint-compatible inputs from a local
Git clone, runs DeepJIT/SimCom hunk attribution, and retains only hunks containing
eligible ground-truth lines. Ground truth in newly added files is reported and
excluded because the historical miner did not admit those files.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import subprocess
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET
import zipfile

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"m": MAIN_NS, "r": OFFICE_REL_NS}

HUNK_HEADER_RE = re.compile(
    r"^@@\s+-(?P<old_start>\d+)(?:,(?P<old_count>\d+))?\s+"
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))?\s+@@"
)
VUL_LINE_RE = re.compile(
    r"^(?P<file>.+?)_+line_+(?P<lines>all|\d+(?:_\d+)*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GroundTruthTarget:
    file_path: str
    line_numbers: frozenset[int]
    all_lines: bool
    source_value: str


@dataclass(frozen=True)
class GroundTruthRow:
    excel_row: int
    vul_commit: str
    patch_commit: str
    cve: str
    targets: Tuple[GroundTruthTarget, ...]


def _column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference.upper())
    if letters is None:
        raise ValueError(f"invalid_xlsx_cell_reference:{cell_reference}")
    result = 0
    for char in letters.group():
        result = result * 26 + ord(char) - ord("A") + 1
    return result - 1


def _shared_strings(archive: zipfile.ZipFile) -> List[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [
        "".join(node.text or "" for node in item.findall(".//m:t", NS))
        for item in root.findall("m:si", NS)
    ]


def _worksheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    relation_targets = {
        relation.attrib["Id"]: relation.attrib["Target"]
        for relation in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
    }
    for sheet in workbook.findall(".//m:sheet", NS):
        if sheet.attrib["name"].casefold() != sheet_name.casefold():
            continue
        relation_id = sheet.attrib[f"{{{OFFICE_REL_NS}}}id"]
        target = relation_targets[relation_id].replace("\\", "/").lstrip("/")
        return target if target.startswith("xl/") else f"xl/{target}"
    available = [sheet.attrib["name"] for sheet in workbook.findall(".//m:sheet", NS)]
    raise ValueError(f"sheet_not_found:{sheet_name};available={available}")


def read_xlsx_rows(path: Path, sheet_name: str) -> List[Dict[str, str]]:
    """Read one worksheet without requiring openpyxl."""
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        root = ET.fromstring(archive.read(_worksheet_path(archive, sheet_name)))

    raw_rows = []
    for row in root.findall(".//m:sheetData/m:row", NS):
        values = {}
        for cell in row.findall("m:c", NS):
            cell_type = cell.attrib.get("t")
            value_node = cell.find("m:v", NS)
            value = "" if value_node is None or value_node.text is None else value_node.text
            if cell_type == "s" and value:
                value = shared[int(value)]
            elif cell_type == "inlineStr":
                value = "".join(
                    node.text or "" for node in cell.findall(".//m:t", NS)
                )
            values[_column_index(cell.attrib["r"])] = value
        raw_rows.append((int(row.attrib["r"]), values))

    if not raw_rows:
        return []
    headers = {
        index: value.strip()
        for index, value in raw_rows[0][1].items()
        if value.strip()
    }
    return [
        {
            **{header: values.get(index, "").strip() for index, header in headers.items()},
            "_excel_row": str(row_number),
        }
        for row_number, values in raw_rows[1:]
        if any(values.get(index, "").strip() for index in headers)
    ]


def normalize_path(value: str) -> str:
    value = value.strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return str(PurePosixPath(value))


def parse_vul_lines(value: str) -> Tuple[GroundTruthTarget, ...]:
    try:
        entries = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"invalid_vul_lines:{value}") from error
    if isinstance(entries, str):
        entries = [entries]
    if not isinstance(entries, (list, tuple)):
        raise ValueError(f"vul_lines_must_be_string_list:{value}")

    targets = []
    for entry in entries:
        if not isinstance(entry, str):
            raise ValueError(f"vul_lines_entry_must_be_string:{entry!r}")
        match = VUL_LINE_RE.match(entry.strip())
        if match is None:
            raise ValueError(f"unparseable_vul_lines_entry:{entry}")
        line_spec = match.group("lines").casefold()
        targets.append(
            GroundTruthTarget(
                file_path=normalize_path(match.group("file")),
                line_numbers=(
                    frozenset()
                    if line_spec == "all"
                    else frozenset(int(number) for number in line_spec.split("_"))
                ),
                all_lines=line_spec == "all",
                source_value=entry,
            )
        )
    return tuple(targets)


def load_ground_truth(path: Path, sheet_name: str) -> List[GroundTruthRow]:
    result = []
    seen_commits = set()
    for row in read_xlsx_rows(path, sheet_name):
        if not row.get("Vul_commit") or not row.get("Vul_lines"):
            continue
        vul_commit = row["Vul_commit"].strip().lower()
        if vul_commit in seen_commits:
            raise ValueError(f"duplicate_vul_commit:{vul_commit}")
        seen_commits.add(vul_commit)
        result.append(
            GroundTruthRow(
                excel_row=int(row["_excel_row"]),
                vul_commit=vul_commit,
                patch_commit=row.get("Patch_commit", "").strip().lower(),
                cve=row.get("CVE", "").strip(),
                targets=parse_vul_lines(row["Vul_lines"]),
            )
        )
    if not result:
        raise ValueError(f"no_ground_truth_rows:{sheet_name}")
    return result


def write_commit_manifest(path: Path, rows: Iterable[GroundTruthRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps({
                "commit_id": row.vul_commit,
                "excel_row": row.excel_row,
                "cve": row.cve,
                "patch_commit": row.patch_commit,
            }, ensure_ascii=False) + "\n")


def commit_matches(full_commit: str, abbreviated_commit: str) -> bool:
    full_commit = full_commit.strip().lower()
    abbreviated_commit = abbreviated_commit.strip().lower()
    return bool(full_commit and abbreviated_commit) and (
        full_commit.startswith(abbreviated_commit)
        or abbreviated_commit.startswith(full_commit)
    )


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid_json:{path}:{line_number}:{error}") from error


def _load_jsonl(path: Path) -> List[Dict]:
    return list(_iter_jsonl(path))


def _new_files(repo_path: Path, parent_id: str, commit_id: str) -> Set[str]:
    completed = subprocess.run(
        [
            "git", "-C", str(repo_path), "diff", "--name-only",
            "--find-renames", "--diff-filter=A", "-z", parent_id, commit_id, "--",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        normalize_path(path.decode("utf-8", errors="surrogateescape"))
        for path in completed.stdout.split(b"\0")
        if path
    }


def find_new_file_exclusions(
    repo_path: Path,
    ground_truth: Iterable[GroundTruthRow],
    provenance_path: Path,
) -> Tuple[Set[Tuple[int, str]], List[Dict]]:
    """Identify ground-truth targets in files added by their vulnerable commit."""
    provenance = _load_jsonl(provenance_path)
    excluded_keys: Set[Tuple[int, str]] = set()
    exclusions = []
    for record in provenance:
        commit_id = str(record["commit_id"])
        new_files = _new_files(repo_path, str(record["parent_id"]), commit_id)
        for row in ground_truth:
            if not commit_matches(commit_id, row.vul_commit):
                continue
            for target in row.targets:
                if target.file_path not in new_files:
                    continue
                excluded_keys.add((row.excel_row, target.source_value))
                exclusions.append({
                    "excel_row": row.excel_row,
                    "vul_commit": row.vul_commit,
                    "resolved_commit": commit_id,
                    "cve": row.cve,
                    "target": target.source_value,
                    "file_path": target.file_path,
                    "excluded_line_numbers": sorted(target.line_numbers),
                    "all_lines": target.all_lines,
                    "reason": "new_file_excluded_by_historical_miner",
                })
    return excluded_keys, exclusions


def _hunk_interval(header: str) -> Tuple[int, int]:
    match = HUNK_HEADER_RE.match(header)
    if match is None:
        raise ValueError(f"invalid_hunk_header:{header}")
    start = int(match.group("new_start"))
    count_value = match.group("new_count")
    count = 1 if count_value is None else int(count_value)
    return start, start + count


def _target_match(target: GroundTruthTarget, hunk: Dict) -> Optional[List[int]]:
    if normalize_path(hunk.get("file_path", "")) != target.file_path:
        return None
    if target.all_lines:
        return []
    start, end = _hunk_interval(hunk["hunk_header"])
    matched = sorted(line for line in target.line_numbers if start <= line < end)
    return matched or None


def filter_ground_truth_hunks(
    report_path: Path,
    ground_truth: Iterable[GroundTruthRow],
    excluded_keys: Set[Tuple[int, str]],
    exclusions: List[Dict],
) -> Tuple[List[Dict], Dict]:
    ground_truth = list(ground_truth)
    matched_target_keys: Set[Tuple[int, str]] = set()
    matched_lines_by_target: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
    report_commits = set()
    report_files_by_commit: Dict[str, Set[str]] = defaultdict(set)
    output_hunks = []
    report_hunks = 0

    for record in _iter_jsonl(report_path):
        commit_id = str(record.get("commit_id", "")).lower()
        if commit_id:
            report_commits.add(commit_id)
        if record.get("record_type") != "hunk":
            continue
        report_hunks += 1
        report_files_by_commit[commit_id].add(normalize_path(record.get("file_path", "")))
        matches = []
        for row in ground_truth:
            if not commit_matches(commit_id, row.vul_commit):
                continue
            for target in row.targets:
                key = (row.excel_row, target.source_value)
                if key in excluded_keys:
                    continue
                matched_lines = _target_match(target, record)
                if matched_lines is None:
                    continue
                matched_target_keys.add(key)
                matched_lines_by_target[key].update(matched_lines)
                changed_matches = [
                    changed_line
                    for changed_line in record.get("lines", [])
                    if changed_line.get("new_line_no") in matched_lines
                ]
                matches.append({
                    "excel_row": row.excel_row,
                    "vul_commit": row.vul_commit,
                    "patch_commit": row.patch_commit,
                    "cve": row.cve,
                    "target": target.source_value,
                    "matched_line_numbers": matched_lines,
                    "matched_changed_lines": changed_matches,
                    "matched_by_all_lines": target.all_lines,
                    "line_side": "new",
                })
        if matches:
            output_hunks.append({**record, "ground_truth_matches": matches})

    eligible_targets = []
    unmatched_targets = []
    fully_matched_targets = 0
    eligible_numeric_lines = 0
    matched_numeric_lines = 0
    for row in ground_truth:
        commit_files = set()
        for commit_id, files in report_files_by_commit.items():
            if commit_matches(commit_id, row.vul_commit):
                commit_files.update(files)
        for target in row.targets:
            key = (row.excel_row, target.source_value)
            if key in excluded_keys:
                continue
            eligible_targets.append(key)
            matched_lines = matched_lines_by_target[key]
            eligible_numeric_lines += len(target.line_numbers)
            matched_numeric_lines += len(matched_lines)
            fully_matched = (
                key in matched_target_keys
                if target.all_lines
                else matched_lines == set(target.line_numbers)
            )
            if fully_matched:
                fully_matched_targets += 1
                continue
            unmatched_targets.append({
                "excel_row": row.excel_row,
                "vul_commit": row.vul_commit,
                "cve": row.cve,
                "target": target.source_value,
                "unmatched_line_numbers": sorted(target.line_numbers - matched_lines),
                "reason": (
                    "line_not_in_any_hunk"
                    if target.file_path in commit_files
                    else "file_not_present_for_commit"
                ),
            })

    ground_truth_targets = sum(len(row.targets) for row in ground_truth)
    ground_truth_numeric_lines = sum(
        len(target.line_numbers) for row in ground_truth for target in row.targets
    )
    excluded_numeric_lines = sum(len(item["excluded_line_numbers"]) for item in exclusions)
    summary = {
        "line_side": "new",
        "ground_truth_rows": len(ground_truth),
        "ground_truth_commits_found_in_report": sum(
            any(commit_matches(commit, row.vul_commit) for commit in report_commits)
            for row in ground_truth
        ),
        "report_commits": len(report_commits),
        "report_hunks": report_hunks,
        "matched_hunks": len(output_hunks),
        "ground_truth_targets_total": ground_truth_targets,
        "excluded_new_file_targets": len(exclusions),
        "eligible_ground_truth_targets": len(eligible_targets),
        "matched_ground_truth_targets": fully_matched_targets,
        "ground_truth_line_numbers_total": ground_truth_numeric_lines,
        "excluded_new_file_line_numbers": excluded_numeric_lines,
        "eligible_ground_truth_line_numbers": eligible_numeric_lines,
        "matched_ground_truth_line_numbers": matched_numeric_lines,
        "new_file_exclusions": exclusions,
        "unmatched_eligible_ground_truth_targets": unmatched_targets,
    }
    return output_hunks, summary


def _write_json(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def _write_jsonl(path: Path, values: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for value in values:
            output.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_provenance_hunk_report(provenance_path: Path, output_path: Path) -> List[Dict]:
    """Export model-independent changed-line hunks from line provenance."""
    records = []
    for provenance in _iter_jsonl(provenance_path):
        groups = []
        by_key = {}
        for line in provenance.get("lines", []):
            key = (normalize_path(line["file_path"]), int(line["hunk_id"]))
            if key not in by_key:
                group = {
                    "record_type": "hunk",
                    "commit_id": str(provenance["commit_id"]),
                    "parent_id": str(provenance["parent_id"]),
                    "file_path": key[0],
                    "hunk_id": key[1],
                    "hunk_header": line["hunk_header"],
                    "lines": [],
                }
                groups.append(group)
                by_key[key] = group
            by_key[key]["lines"].append(line)
        for group in groups:
            group["changed_line_count"] = len(group["lines"])
            records.append(group)
    _write_jsonl(output_path, records)
    return records


def prepare_ground_truth_hunks(args):
    """Create reusable ground-truth hunks without loading or running a model."""
    from .attribution.source_provenance import prepare_lines

    workbook = Path(args.ground_truth).resolve()
    repo_path = Path(args.repo_path).resolve()
    output_root = Path(args.output_dir).resolve()
    inputs_dir = output_root / "inputs"
    manifest_path = inputs_dir / "commit_references.jsonl"
    provenance_path = inputs_dir / "line_provenance.jsonl"
    report_path = inputs_dir / "all_eligible_hunks.jsonl"
    filtered_path = output_root / "ground_truth_hunks.jsonl"
    summary_path = output_root / "ground_truth_hunks_summary.json"

    if not workbook.is_file():
        raise ValueError(f"ground_truth_workbook_not_found:{workbook}")
    if not (repo_path / ".git").is_dir():
        raise ValueError(f"repo_path_is_not_git_clone:{repo_path}")

    ground_truth = load_ground_truth(workbook, args.sheet)
    write_commit_manifest(manifest_path, ground_truth)
    print(f"Ground truth: {len(ground_truth)} commits from sheet {args.sheet!r}")
    print(f"Commit manifest: {manifest_path}")

    prepare_lines(SimpleNamespace(
        repo_path=str(repo_path),
        commit_id=None,
        commit_urls=str(manifest_path),
        repo_language=args.repo_language,
        output_dir=str(inputs_dir),
    ))

    excluded_keys, exclusions = find_new_file_exclusions(
        repo_path, ground_truth, provenance_path,
    )
    print(f"New-file ground-truth targets excluded: {len(exclusions)}")

    write_provenance_hunk_report(provenance_path, report_path)
    hunks, summary = filter_ground_truth_hunks(
        report_path, ground_truth, excluded_keys, exclusions,
    )
    summary.update({
        "sheet": args.sheet,
        "stage": "prepare-ground-truth-hunks",
        "model_independent": True,
        "paths": {
            "commit_manifest": str(manifest_path),
            "line_provenance": str(provenance_path),
            "all_eligible_hunks": str(report_path),
            "ground_truth_hunks": str(filtered_path),
        },
    })
    _write_jsonl(filtered_path, hunks)
    _write_json(summary_path, summary)
    print(f"Ground-truth hunks: {len(hunks)}")
    print(f"Filtered output: {filtered_path}")
    print(f"Summary: {summary_path}")
    return summary


def _hunk_keys(path: Path) -> Set[Tuple[str, str, int]]:
    return {
        (
            str(record["commit_id"]).lower(),
            normalize_path(record["file_path"]),
            int(record["hunk_id"]),
        )
        for record in _iter_jsonl(path)
    }


def prepare_cnn_ranking_inputs(prepared_root: Path, output_dir: Path) -> Dict[str, Path]:
    """Restrict CNN inputs to common ground-truth hunks without altering a hunk."""
    from .attribution.source_provenance import serialize_historical

    common_hunks = prepared_root / "ground_truth_hunks.jsonl"
    source_provenance = prepared_root / "inputs" / "line_provenance.jsonl"
    selected_keys = _hunk_keys(common_hunks)
    selected_records = []
    for record in _iter_jsonl(source_provenance):
        commit_id = str(record["commit_id"]).lower()
        lines = [
            line for line in record.get("lines", [])
            if (
                commit_id,
                normalize_path(line["file_path"]),
                int(line["hunk_id"]),
            ) in selected_keys
        ]
        if not lines:
            continue
        selected_line_ids = {int(line["line_id"]) for line in lines}
        blocks = []
        for block in record.get("blocks", []):
            deleted = [
                int(line_id) for line_id in block.get("deleted_line_ids", [])
                if int(line_id) in selected_line_ids
            ]
            added = [
                int(line_id) for line_id in block.get("added_line_ids", [])
                if int(line_id) in selected_line_ids
            ]
            if deleted or added:
                blocks.append({
                    **block,
                    "deleted_line_ids": deleted,
                    "added_line_ids": added,
                    "block_id": len(blocks),
                })
        selected_records.append({
            **record,
            "lines": lines,
            "blocks": blocks,
            "historical_serialization": serialize_historical(lines, blocks),
            "metadata": {
                **record.get("metadata", {}),
                "ranking_scope": "ground_truth_hunks_only",
            },
        })

    inputs_dir = output_dir / "inputs"
    provenance_path = inputs_dir / "line_provenance.jsonl"
    merge_path = inputs_dir / "merge.jsonl"
    patch_path = inputs_dir / "patch.jsonl"
    _write_jsonl(provenance_path, selected_records)
    _write_jsonl(merge_path, [
        {
            "commit_id": record["commit_id"],
            "messages": record["message"],
            "code_change": record["historical_serialization"]["merge"],
        }
        for record in selected_records
    ])
    _write_jsonl(patch_path, [
        {
            "commit_id": record["commit_id"],
            "messages": record["message"],
            "code_change": record["historical_serialization"]["patch"],
        }
        for record in selected_records
    ])
    return {"provenance": provenance_path, "merge": merge_path, "patch": patch_path}


def _ranking_key(record: Dict) -> Tuple[str, str, int]:
    return (
        str(record["commit_id"]).lower(),
        normalize_path(record["file_path"]),
        int(record["hunk_id"]),
    )


def merge_model_rankings(
    common_hunks_path: Path,
    attribution_dir: Path,
    model: str,
) -> Tuple[List[Dict], Dict]:
    """Attach one model's ranking to the reusable common hunk records."""
    ranking_by_hunk = {}
    commit_status = {}
    if model in {"deepjit", "simcom"}:
        report_path = attribution_dir / "hunk_line_report.jsonl"
        for record in _iter_jsonl(report_path):
            if record.get("record_type") == "commit_summary":
                commit_status[str(record.get("commit_id", "")).lower()] = record
            elif record.get("record_type") == "hunk":
                ranking_by_hunk[_ranking_key(record)] = record
    else:
        attribution_path = attribution_dir / "token_attributions.jsonl"
        for record in _iter_jsonl(attribution_path):
            commit_id = str(record.get("commit_id", "")).lower()
            commit_status[commit_id] = {
                key: record.get(key)
                for key in ("commit_id", "status", "prediction_score", "predicted_label")
            }
            commit_status[commit_id]["ranked_line_count"] = len(
                record.get("ranked_lines", [])
            )
            commit_status[commit_id]["total_changed_lines"] = record.get(
                "total_changed_lines"
            )
            grouped = defaultdict(list)
            for line in record.get("ranked_lines", []):
                grouped[(normalize_path(line["file_path"]), int(line["hunk_id"]))].append(line)
            for (file_path, hunk_id), lines in grouped.items():
                ranking_by_hunk[(commit_id, file_path, hunk_id)] = {
                    "record_type": "hunk_ranking",
                    "commit_id": record.get("commit_id"),
                    "file_path": file_path,
                    "hunk_id": hunk_id,
                    "prediction_score": record.get("prediction_score"),
                    "predicted_label": record.get("predicted_label"),
                    "lines": lines,
                }

    output = []
    unranked = []
    for hunk in _iter_jsonl(common_hunks_path):
        key = _ranking_key(hunk)
        ranking = ranking_by_hunk.get(key)
        output.append({
            **hunk,
            "model_ranking": {
                "model": model,
                "status": "ranked" if ranking is not None else "unranked",
                "ranking": ranking,
                "commit_attribution": commit_status.get(key[0]),
            },
        })
        if ranking is None:
            unranked.append({
                "commit_id": hunk["commit_id"],
                "file_path": hunk["file_path"],
                "hunk_id": hunk["hunk_id"],
            })
    return output, {
        "model": model,
        "ground_truth_hunks": len(output),
        "ranked_ground_truth_hunks": len(output) - len(unranked),
        "unranked_ground_truth_hunks": unranked,
    }


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.fmean(values) if values else None


def _median(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.median(values) if values else None


def _average_precision(relevant_ranks: List[int], relevant_count: int) -> float:
    if relevant_count <= 0:
        return 0.0
    return sum(
        index / rank for index, rank in enumerate(sorted(relevant_ranks), start=1)
    ) / relevant_count


def _ndcg_at_k(relevant_ranks: List[int], relevant_count: int, cutoff: int) -> float:
    if relevant_count <= 0 or cutoff <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in relevant_ranks
        if rank <= cutoff
    )
    ideal_count = min(relevant_count, cutoff)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_ranking_metrics(
    ranked_hunks: Iterable[Dict],
    model: str,
    top_k: Iterable[int] = (1, 3, 5, 10),
    effort_fraction: float = 0.2,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate vulnerable-line rankings with explicit native ranking scopes.

    CNN rankings are native to a hunk chunk. JITFine line ranks are native to a
    complete commit. Unranked ground-truth lines count as misses in absolute
    line metrics, while unit metrics describe only units with ranked candidates.
    """
    top_k = sorted(set(int(value) for value in top_k))
    if not top_k or top_k[0] < 1:
        raise ValueError("metric_top_k_must_contain_positive_integers")
    if not 0.0 < effort_fraction <= 1.0:
        raise ValueError("effort_fraction_must_be_in_(0,1]")

    ranked_hunks = list(ranked_hunks)
    candidates_by_unit: Dict[Tuple, Dict] = {}
    ground_truth_entries = []
    seen_ground_truth = set()

    for hunk in ranked_hunks:
        commit_id = str(hunk["commit_id"]).lower()
        file_path = normalize_path(hunk["file_path"])
        hunk_id = int(hunk["hunk_id"])
        numeric_lines = {
            int(line_number)
            for match in hunk.get("ground_truth_matches", [])
            for line_number in match.get("matched_line_numbers", [])
        }
        all_lines = any(
            match.get("matched_by_all_lines")
            for match in hunk.get("ground_truth_matches", [])
        )
        common_lines = hunk.get("lines", [])
        relevant_common_lines = [
            line for line in common_lines
            if line.get("new_line_no") is not None
            and (all_lines or int(line["new_line_no"]) in numeric_lines)
        ]

        # A ground-truth line can be context inside a hunk and therefore absent
        # from the changed-line list. Keep it as an explicit unranked entry.
        represented_numbers = {
            int(line["new_line_no"]) for line in relevant_common_lines
        }
        for line_number in sorted(numeric_lines - represented_numbers):
            relevant_common_lines.append({
                "line_id": None,
                "new_line_no": line_number,
                "text": None,
            })

        ranking_wrapper = hunk.get("model_ranking", {})
        ranking = ranking_wrapper.get("ranking") or {}
        ranking_lines = ranking.get("lines", [])
        ranking_by_line_id = {
            int(line["line_id"]): line
            for line in ranking_lines
            if line.get("line_id") is not None
        }
        ranking_by_new_number = {
            int(line["new_line_no"]): line
            for line in ranking_lines
            if line.get("new_line_no") is not None
        }

        if model == "jitfine":
            unit_key = (commit_id,)
            commit_info = ranking_wrapper.get("commit_attribution") or {}
            candidate_count = int(commit_info.get("ranked_line_count") or 0)
            candidates_by_unit.setdefault(unit_key, {
                "unit_type": "commit",
                "commit_id": commit_id,
                "file_path": None,
                "hunk_id": None,
                "chunk_id": None,
                "candidate_count": candidate_count,
                "relevant_ranks": [],
                "relevant_count": 0,
            })
        else:
            for line in ranking_lines:
                if line.get("rank_in_chunk") is None or line.get("chunk_id") is None:
                    continue
                unit_key = (commit_id, file_path, hunk_id, int(line["chunk_id"]))
                unit = candidates_by_unit.setdefault(unit_key, {
                    "unit_type": "chunk",
                    "commit_id": commit_id,
                    "file_path": file_path,
                    "hunk_id": hunk_id,
                    "chunk_id": int(line["chunk_id"]),
                    "candidate_count": 0,
                    "relevant_ranks": [],
                    "relevant_count": 0,
                })
                unit["candidate_count"] += 1

        for common_line in relevant_common_lines:
            identity = (
                commit_id,
                file_path,
                hunk_id,
                common_line.get("line_id"),
                int(common_line["new_line_no"]),
            )
            if identity in seen_ground_truth:
                continue
            seen_ground_truth.add(identity)
            ranked_line = None
            if common_line.get("line_id") is not None:
                ranked_line = ranking_by_line_id.get(int(common_line["line_id"]))
            if ranked_line is None:
                ranked_line = ranking_by_new_number.get(int(common_line["new_line_no"]))

            rank = None
            candidate_count = None
            native_unit_key = None
            if ranked_line is not None:
                if model == "jitfine" and ranked_line.get("rank") is not None:
                    rank = int(ranked_line["rank"])
                    native_unit_key = (commit_id,)
                elif (
                    model != "jitfine"
                    and ranked_line.get("rank_in_chunk") is not None
                    and ranked_line.get("chunk_id") is not None
                ):
                    rank = int(ranked_line["rank_in_chunk"])
                    native_unit_key = (
                        commit_id, file_path, hunk_id, int(ranked_line["chunk_id"]),
                    )
                if native_unit_key in candidates_by_unit:
                    candidate_count = candidates_by_unit[native_unit_key]["candidate_count"]
                    candidates_by_unit[native_unit_key]["relevant_ranks"].append(rank)
                    candidates_by_unit[native_unit_key]["relevant_count"] += 1

            ground_truth_entries.append({
                "commit_id": commit_id,
                "file_path": file_path,
                "hunk_id": hunk_id,
                "line_id": common_line.get("line_id"),
                "new_line_no": int(common_line["new_line_no"]),
                "rank": rank,
                "candidate_count": candidate_count,
                "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
                "rank_percentile": (
                    None
                    if rank is None or not candidate_count
                    else rank / candidate_count
                ),
            })

    unit_metrics = []
    for unit in candidates_by_unit.values():
        relevant_ranks = sorted(unit["relevant_ranks"])
        if not relevant_ranks:
            continue
        relevant_count = unit["relevant_count"]
        candidate_count = unit["candidate_count"]
        first_rank = relevant_ranks[0]
        required_for_effort = max(1, math.ceil(effort_fraction * relevant_count))
        effort_rank = (
            relevant_ranks[required_for_effort - 1]
            if len(relevant_ranks) >= required_for_effort
            else None
        )
        metrics = {
            **unit,
            "first_relevant_rank": first_rank,
            "reciprocal_rank": 1.0 / first_rank,
            "average_precision": _average_precision(relevant_ranks, relevant_count),
            "initial_false_alarms": first_rank - 1,
            "exam": first_rank / candidate_count if candidate_count else None,
            f"recall_at_{effort_fraction:g}_loc": (
                sum(
                    rank <= max(1, math.ceil(effort_fraction * candidate_count))
                    for rank in relevant_ranks
                ) / relevant_count
                if candidate_count else None
            ),
            f"effort_at_{effort_fraction:g}_recall": (
                effort_rank / candidate_count
                if effort_rank is not None and candidate_count else None
            ),
        }
        for cutoff in top_k:
            hits = sum(rank <= cutoff for rank in relevant_ranks)
            metrics[f"hit_at_{cutoff}"] = int(hits > 0)
            metrics[f"recall_at_{cutoff}"] = hits / relevant_count
            metrics[f"ndcg_at_{cutoff}"] = _ndcg_at_k(
                relevant_ranks, relevant_count, cutoff,
            )
        unit_metrics.append(metrics)

    total_lines = len(ground_truth_entries)
    ranked_entries = [entry for entry in ground_truth_entries if entry["rank"] is not None]
    absolute = {
        "ground_truth_line_count": total_lines,
        "ranked_ground_truth_line_count": len(ranked_entries),
        "unranked_ground_truth_line_count": total_lines - len(ranked_entries),
        "ranking_coverage": len(ranked_entries) / total_lines if total_lines else None,
        "line_mrr_unranked_as_zero": _mean(
            entry["reciprocal_rank"] for entry in ground_truth_entries
        ),
        "mean_rank_ranked_lines": _mean(entry["rank"] for entry in ranked_entries),
        "median_rank_ranked_lines": _median(entry["rank"] for entry in ranked_entries),
        "mean_rank_percentile_ranked_lines": _mean(
            entry["rank_percentile"]
            for entry in ranked_entries
            if entry["rank_percentile"] is not None
        ),
    }
    for cutoff in top_k:
        hit_count = sum(
            entry["rank"] is not None and entry["rank"] <= cutoff
            for entry in ground_truth_entries
        )
        absolute[f"absolute_hit_count_at_{cutoff}"] = hit_count
        absolute[f"absolute_hit_rate_at_{cutoff}"] = (
            hit_count / total_lines if total_lines else None
        )
        absolute[f"recall_at_{cutoff}_among_ranked_lines"] = (
            hit_count / len(ranked_entries) if ranked_entries else None
        )

    aggregate_units = {
        "ranking_unit": "commit" if model == "jitfine" else "hunk_chunk",
        "evaluated_unit_count": len(unit_metrics),
        "mrr": _mean(unit["reciprocal_rank"] for unit in unit_metrics),
        "map": _mean(unit["average_precision"] for unit in unit_metrics),
        "mean_first_relevant_rank": _mean(
            unit["first_relevant_rank"] for unit in unit_metrics
        ),
        "median_first_relevant_rank": _median(
            unit["first_relevant_rank"] for unit in unit_metrics
        ),
        "mean_initial_false_alarms": _mean(
            unit["initial_false_alarms"] for unit in unit_metrics
        ),
        "mean_exam": _mean(
            unit["exam"] for unit in unit_metrics if unit["exam"] is not None
        ),
        f"mean_recall_at_{effort_fraction:g}_loc": _mean(
            unit[f"recall_at_{effort_fraction:g}_loc"]
            for unit in unit_metrics
            if unit[f"recall_at_{effort_fraction:g}_loc"] is not None
        ),
        f"mean_effort_at_{effort_fraction:g}_recall": _mean(
            unit[f"effort_at_{effort_fraction:g}_recall"]
            for unit in unit_metrics
            if unit[f"effort_at_{effort_fraction:g}_recall"] is not None
        ),
    }
    for cutoff in top_k:
        aggregate_units[f"hit_rate_at_{cutoff}"] = _mean(
            unit[f"hit_at_{cutoff}"] for unit in unit_metrics
        )
        aggregate_units[f"mean_recall_at_{cutoff}"] = _mean(
            unit[f"recall_at_{cutoff}"] for unit in unit_metrics
        )
        aggregate_units[f"mean_ndcg_at_{cutoff}"] = _mean(
            unit[f"ndcg_at_{cutoff}"] for unit in unit_metrics
        )

    return {
        "model": model,
        "top_k": top_k,
        "effort_fraction": effort_fraction,
        "absolute_line_metrics": absolute,
        "native_unit_metrics": aggregate_units,
        "definitions": {
            "absolute_hit_rate": "ground-truth lines at rank <= K / all eligible ground-truth lines; unranked lines are misses",
            "line_mrr": "mean reciprocal native rank per eligible ground-truth line; unranked lines contribute zero",
            "mrr": "mean reciprocal rank of the first relevant line per native ranking unit",
            "map": "mean average precision with binary vulnerable-line relevance per native ranking unit",
            "ifa": "number of ranked non-vulnerable lines before the first vulnerable line",
            "exam": "first relevant rank divided by ranked candidate count",
        },
        "ground_truth_line_results": ground_truth_entries,
    }, unit_metrics


def rank_ground_truth(args):
    """Run one model and attach its ranking to prepared common ground-truth hunks."""
    from .attribution.runner import attribute

    prepared_root = Path(args.prepared_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    common_hunks_path = prepared_root / "ground_truth_hunks.jsonl"
    full_inputs = prepared_root / "inputs"
    if not common_hunks_path.is_file():
        raise ValueError(f"prepared_ground_truth_hunks_not_found:{common_hunks_path}")

    if args.model in {"deepjit", "simcom"}:
        inputs = prepare_cnn_ranking_inputs(prepared_root, output_root)
        provenance_path = inputs["provenance"]
        code_file = inputs["merge" if args.model == "deepjit" else "patch"]
        test_set = str(code_file)
    else:
        if not args.features:
            raise ValueError("-features_is_required_for_jitfine")
        provenance_path = full_inputs / "line_provenance.jsonl"
        code_file = full_inputs / "merge.jsonl"
        test_set = f"{Path(args.features).resolve()},{code_file}"

    attribution_dir = output_root / "attribution"
    attribution_summary = attribute(SimpleNamespace(
        model=args.model,
        repo_language=args.repo_language,
        device=args.device,
        model_path=args.model_path,
        test_set=test_set,
        hyperparameters=args.hyperparameters,
        dictionary=args.dictionary,
        line_provenance=str(provenance_path),
        line_aggregation=args.line_aggregation,
        hunk_chunk_size=args.hunk_chunk_size,
        output_dir=str(attribution_dir),
        threshold=args.threshold,
        target_class=args.target_class,
        attention_strategy=args.attention_strategy,
        top_k=None,
        commit_id=None,
        only_predicted_vulnerable=False,
        all_commits=True,
        overwrite=args.overwrite,
        resume=args.resume,
        seed=args.seed,
    ))

    ranked_hunks, summary = merge_model_rankings(
        common_hunks_path, attribution_dir, args.model,
    )
    ranking_metrics, unit_metrics = evaluate_ranking_metrics(
        ranked_hunks,
        args.model,
        top_k=args.metric_top_k,
        effort_fraction=args.effort_fraction,
    )
    summary.update({
        "stage": "rank-ground-truth",
        "attribution": attribution_summary,
        "prepared_dir": str(prepared_root),
        "ranking_metrics": ranking_metrics,
    })
    ranked_path = output_root / "ranked_ground_truth_hunks.jsonl"
    summary_path = output_root / "ranked_ground_truth_summary.json"
    unit_metrics_path = output_root / "ranking_metrics_by_unit.jsonl"
    _write_jsonl(ranked_path, ranked_hunks)
    _write_jsonl(unit_metrics_path, unit_metrics)
    _write_json(summary_path, summary)
    print(f"Ranked ground-truth hunks: {summary['ranked_ground_truth_hunks']}/{summary['ground_truth_hunks']}")
    native_metrics = ranking_metrics["native_unit_metrics"]
    absolute_metrics = ranking_metrics["absolute_line_metrics"]
    print(
        "Ranking metrics: "
        f"MRR={native_metrics['mrr']}, MAP={native_metrics['map']}, "
        f"coverage={absolute_metrics['ranking_coverage']}"
    )
    print(
        "Absolute hit rates: "
        + ", ".join(
            f"Hit@{cutoff}={absolute_metrics[f'absolute_hit_rate_at_{cutoff}']}"
            for cutoff in ranking_metrics["top_k"]
        )
    )
    print(f"Ranked output: {ranked_path}")
    print(f"Summary: {summary_path}")
    if attribution_summary.get("failed"):
        raise RuntimeError(
            f"attribution_failed_for_{attribution_summary['failed']}_commits;see={attribution_dir / 'summary.json'}"
        )
    return summary
