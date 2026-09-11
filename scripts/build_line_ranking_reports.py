#!/usr/bin/env python3
"""Build concise, separate line-level and chunk-level ranking reports.

DeepJIT/SimCom summaries are reused directly. JITFine's stored full-commit line
scores are re-ranked post-hoc inside <=N-line hunk chunks. No model inference is
performed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import quote
from urllib.request import Request, urlopen

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ground_truth_pipeline import (
    coverage_aware_line_effort_metrics,
    evaluate_ranking_metrics,
    normalize_path,
    rerank_jitfine_hunk_chunks,
)


MODELS = ("deepjit", "simcom", "jitfine")
MODEL_ORDER = {model: index for index, model in enumerate(MODELS)}
LINE_COLUMNS = (
    "model", "config", "seed",
    "gt_hunks", "ranked_hunks",
    "gt_lines", "ranked_lines", "unranked_lines", "coverage",
    "recall_1", "recall_3", "recall_5", "recall_10", "line_mrr",
    "mean_rank", "median_rank", "mean_rank_pct",
    "recall_20pct_loc", "effort_20pct_recall", "mean_line_exam",
)
CHUNK_COLUMNS = (
    "model", "config", "seed", "gt_chunks", "ranked_chunks",
    "hit_1", "hit_3", "hit_5", "hit_10",
    "recall_1", "recall_3", "recall_5", "recall_10",
    "ndcg_1", "ndcg_3", "ndcg_5", "ndcg_10",
    "mrr", "map", "mean_first_rank", "median_first_rank", "mean_ifa",
    "mean_exam", "recall_20pct_loc", "effort_20pct_recall",
)
REPORT_README = """# Line-ranking reports

`line_metrics/` is the primary, coverage-aware report. Every eligible
ground-truth line is retained; an unranked line is a miss, contributes zero to
`line_mrr`, and receives a `mean_line_exam` penalty of 1.0.

- `recall_20pct_loc`: fraction of all ground-truth lines found after inspecting
  the top 20% of each local candidate list.
- `effort_20pct_recall`: uniform local-list fraction needed to recover 20% of
  all ground-truth lines; empty when coverage is insufficient.
- `mean_line_exam`: mean `rank / candidate_count` over all ground-truth lines,
  with unranked lines set to 1.0.

`chunk_metrics/` is secondary diagnostic output. Its rows contain macro means
over hunk chunks. `hit_K` is coverage-aware: its denominator is every chunk
containing at least one ground-truth line, and an entirely unranked chunk is a
miss. The remaining macro ranking metrics are conditional on `ranked_chunks`.

JITFine values reuse stored full-commit attention line scores and re-rank them
inside maximum-10-line hunk chunks. No model inference is rerun.
"""


def natural_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def parse_remote_path(path, dataset):
    parts = PurePosixPath(path).parts
    if len(parts) != 6 or parts[:2] != ("line_ranking", dataset):
        return None
    model, config, seed_part, filename = parts[2:]
    if model not in MODELS or not seed_part.startswith("seed_"):
        return None
    if filename != "ranked_ground_truth_hunks.jsonl":
        return None
    seed_text = seed_part[5:]
    seed = int(seed_text) if re.fullmatch(r"\d+", seed_text) else seed_text
    return model, config, seed


def discover(args):
    api = HfApi()
    sources = []
    for dataset in args.datasets:
        for model in MODELS:
            root = f"line_ranking/{dataset}/{model}"
            for entry in api.list_repo_tree(
                repo_id=args.hf_repo_id,
                path_in_repo=root,
                recursive=True,
                revision=args.revision,
                repo_type="dataset",
            ):
                path = getattr(entry, "rfilename", None) or getattr(entry, "path", None)
                if not path:
                    continue
                identity = parse_remote_path(path, dataset)
                if identity is not None:
                    sources.append((dataset, path, identity))
    return sources


def read_remote_text(args, path):
    repo_id = quote(args.hf_repo_id, safe="/")
    revision = quote(args.revision, safe="")
    filename = quote(path, safe="/")
    url = (
        f"https://huggingface.co/datasets/{repo_id}/resolve/"
        f"{revision}/{filename}?download=true"
    )
    request = Request(url, headers={"User-Agent": "vulguard-lite-report-builder"})
    with urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def read_jsonl(args, path):
    return [
        json.loads(line)
        for line in read_remote_text(args, path).splitlines()
        if line.strip()
    ]


def load_summary(args, path, identity):
    model, _, _ = identity
    hunks = read_jsonl(args, path)
    if model == "jitfine":
        hunks, summary, _ = rerank_jitfine_hunk_chunks(
            hunks,
            chunk_size=args.chunk_size,
            top_k=args.metric_top_k,
            effort_fraction=args.effort_fraction,
        )
        return summary, hunks
    ranking_metrics, _ = evaluate_ranking_metrics(
        hunks,
        model,
        top_k=args.metric_top_k,
        effort_fraction=args.effort_fraction,
    )
    summary = {
        "ground_truth_hunks": len(hunks),
        "ranked_ground_truth_hunks": sum(
            hunk.get("model_ranking", {}).get("status") == "ranked"
            for hunk in hunks
        ),
        "ranking_metrics": ranking_metrics,
    }
    return summary, hunks


def metric_at(metrics, name, cutoff):
    return metrics.get(f"{name}_at_{cutoff}")


def coverage_aware_chunk_hits(hunks, line_results, chunk_size, top_k):
    locations_by_id = {}
    locations_by_number = {}
    for hunk in hunks:
        commit_id = str(hunk.get("commit_id", "")).lower()
        file_path = normalize_path(hunk["file_path"])
        hunk_id = int(hunk["hunk_id"])
        for position, line in enumerate(hunk.get("lines", [])):
            unit = (commit_id, file_path, hunk_id, position // chunk_size)
            if line.get("line_id") is not None:
                locations_by_id[(commit_id, file_path, hunk_id, int(line["line_id"]))] = unit
            if line.get("new_line_no") is not None:
                locations_by_number[
                    (commit_id, file_path, hunk_id, int(line["new_line_no"]))
                ] = unit

    units = {}
    for entry in line_results:
        commit_id = str(entry.get("commit_id", "")).lower()
        file_path = normalize_path(entry["file_path"])
        hunk_id = int(entry["hunk_id"])
        unit = None
        if entry.get("line_id") is not None:
            unit = locations_by_id.get(
                (commit_id, file_path, hunk_id, int(entry["line_id"]))
            )
        if unit is None:
            unit = locations_by_number.get(
                (commit_id, file_path, hunk_id, int(entry["new_line_no"]))
            )
        if unit is None:
            unit = (commit_id, file_path, hunk_id, "unrepresented")
        units.setdefault(unit, []).append(entry.get("rank"))

    total = len(units)
    rates = {
        cutoff: (
            sum(
                any(rank is not None and rank <= cutoff for rank in ranks)
                for ranks in units.values()
            ) / total
            if total else None
        )
        for cutoff in top_k
    }
    ranked = sum(any(rank is not None for rank in ranks) for ranks in units.values())
    return total, ranked, rates


def report_rows(args, dataset, identity, summary, hunks):
    model, config, seed = identity
    metrics = summary["ranking_metrics"]
    lines = metrics["absolute_line_metrics"]
    chunks = metrics["native_unit_metrics"]
    line_effort = coverage_aware_line_effort_metrics(
        metrics["ground_truth_line_results"], args.effort_fraction,
    )
    gt_chunks, ranked_chunks, chunk_hits = coverage_aware_chunk_hits(
        hunks,
        metrics["ground_truth_line_results"],
        args.chunk_size,
        args.metric_top_k,
    )
    fraction = f"{args.effort_fraction:g}"
    line_row = {
        "model": model,
        "config": config,
        "seed": seed,
        "gt_hunks": summary["ground_truth_hunks"],
        "ranked_hunks": summary["ranked_ground_truth_hunks"],
        "gt_lines": lines["ground_truth_line_count"],
        "ranked_lines": lines["ranked_ground_truth_line_count"],
        "unranked_lines": lines["unranked_ground_truth_line_count"],
        "coverage": lines["ranking_coverage"],
        "recall_1": metric_at(lines, "absolute_hit_rate", 1),
        "recall_3": metric_at(lines, "absolute_hit_rate", 3),
        "recall_5": metric_at(lines, "absolute_hit_rate", 5),
        "recall_10": metric_at(lines, "absolute_hit_rate", 10),
        "line_mrr": lines["line_mrr_unranked_as_zero"],
        "mean_rank": lines["mean_rank_ranked_lines"],
        "median_rank": lines["median_rank_ranked_lines"],
        "mean_rank_pct": lines["mean_rank_percentile_ranked_lines"],
        "recall_20pct_loc": line_effort[f"recall_at_{fraction}_loc"],
        "effort_20pct_recall": line_effort[f"effort_at_{fraction}_recall"],
        "mean_line_exam": line_effort["mean_line_exam_unranked_as_one"],
    }
    chunk_row = {
        "model": model,
        "config": config,
        "seed": seed,
        "gt_chunks": gt_chunks,
        "ranked_chunks": ranked_chunks,
        "hit_1": chunk_hits[1],
        "hit_3": chunk_hits[3],
        "hit_5": chunk_hits[5],
        "hit_10": chunk_hits[10],
        "recall_1": metric_at(chunks, "mean_recall", 1),
        "recall_3": metric_at(chunks, "mean_recall", 3),
        "recall_5": metric_at(chunks, "mean_recall", 5),
        "recall_10": metric_at(chunks, "mean_recall", 10),
        "ndcg_1": metric_at(chunks, "mean_ndcg", 1),
        "ndcg_3": metric_at(chunks, "mean_ndcg", 3),
        "ndcg_5": metric_at(chunks, "mean_ndcg", 5),
        "ndcg_10": metric_at(chunks, "mean_ndcg", 10),
        "mrr": chunks["mrr"],
        "map": chunks["map"],
        "mean_first_rank": chunks["mean_first_relevant_rank"],
        "median_first_rank": chunks["median_first_relevant_rank"],
        "mean_ifa": chunks["mean_initial_false_alarms"],
        "mean_exam": chunks["mean_exam"],
        "recall_20pct_loc": chunks[f"mean_recall_at_{fraction}_loc"],
        "effort_20pct_recall": chunks[f"mean_effort_at_{fraction}_recall"],
    }
    return dataset, line_row, chunk_row


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def validate_matrix(dataset, line_rows):
    identities = {
        model: {(row["config"], str(row["seed"])) for row in line_rows if row["model"] == model}
        for model in MODELS
    }
    reference = identities[MODELS[0]]
    mismatched = [model for model in MODELS[1:] if identities[model] != reference]
    if mismatched:
        details = "; ".join(
            f"{model}: missing={sorted(reference - identities[model], key=str)}, "
            f"extra={sorted(identities[model] - reference, key=str)}"
            for model in mismatched
        )
        raise ValueError(f"incomplete experiment matrix for {dataset}: {details}")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo-id", default="TheSyx/vulguard_lite")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--datasets", nargs="+", default=["linux", "openssl"])
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--metric-top-k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--effort-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("line_ranking_reports"))
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.chunk_size < 1 or args.workers < 1:
        raise SystemExit("--chunk-size and --workers must be at least 1")
    if args.metric_top_k != [1, 3, 5, 10]:
        raise SystemExit("concise reports currently require --metric-top-k 1 3 5 10")
    if args.effort_fraction != 0.2:
        raise SystemExit("concise report column names currently require --effort-fraction 0.2")

    print(f"Discovering ranking artifacts in {args.hf_repo_id}@{args.revision} ...")
    sources = discover(args)
    print(f"Loading {len(sources)} experiment(s); no inference will run ...")
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(load_summary, args, path, identity):
            (dataset, path, identity)
            for dataset, path, identity in sources
        }
        for future in as_completed(futures):
            dataset, path, identity = futures[future]
            try:
                summary, hunks = future.result()
                results.append(report_rows(args, dataset, identity, summary, hunks))
            except Exception as error:
                errors.append((path, error))
    if errors:
        for path, error in sorted(errors):
            print(f"ERROR {path}: {error}", file=sys.stderr)
        raise SystemExit(f"Failed to process {len(errors)} experiment(s)")

    for dataset in args.datasets:
        line_rows = [line for name, line, _ in results if name == dataset]
        chunk_rows = [chunk for name, _, chunk in results if name == dataset]
        sort_key = lambda row: (
            MODEL_ORDER[row["model"]], natural_key(row["config"]), natural_key(row["seed"]),
        )
        line_rows.sort(key=sort_key)
        chunk_rows.sort(key=sort_key)
        validate_matrix(dataset, line_rows)
        write_csv(args.output_dir / "line_metrics" / f"{dataset}.csv", LINE_COLUMNS, line_rows)
        write_csv(args.output_dir / "chunk_metrics" / f"{dataset}.csv", CHUNK_COLUMNS, chunk_rows)
        print(
            f"{dataset}: {len(line_rows)} line rows and {len(chunk_rows)} chunk rows"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "README.md").write_text(REPORT_README, encoding="utf-8")


if __name__ == "__main__":
    main()
