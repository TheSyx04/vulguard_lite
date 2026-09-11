#!/usr/bin/env python3
"""Re-rank stored JITFine line scores in <=N-line Git-hunk chunks.

The command reads existing ``ranked_ground_truth_hunks.jsonl`` artifacts from
Hugging Face.  It does not load a checkpoint or run model inference.  Outputs
are written to a separate tree so the native commit-level results stay intact.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path, PurePosixPath
import re
import sys

from huggingface_hub import HfApi, HfFileSystem


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ground_truth_pipeline import rerank_jitfine_hunk_chunks


SOURCE_FILENAME = "ranked_ground_truth_hunks.jsonl"
SUMMARY_FILENAME = "ranked_ground_truth_summary_hunk_chunk.json"
HUNKS_FILENAME = "ranked_ground_truth_hunks_hunk_chunk.jsonl"
UNITS_FILENAME = "ranking_metrics_by_unit_hunk_chunk.jsonl"


def parse_source_path(path, datasets):
    parts = PurePosixPath(path).parts
    if (
        len(parts) != 6
        or parts[0] != "line_ranking"
        or parts[1] not in datasets
        or parts[2] != "jitfine"
        or parts[4].startswith("seed_") is False
        or parts[5] != SOURCE_FILENAME
    ):
        return None
    seed_text = parts[4][5:]
    seed = int(seed_text) if re.fullmatch(r"\d+", seed_text) else seed_text
    return parts[1], parts[3], seed


def read_remote_jsonl(filesystem, repo_id, revision, remote_path):
    path = f"datasets/{repo_id}/{remote_path}"
    with filesystem.open(path, "r", revision=revision, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )


def flatten(value, prefix=""):
    output = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            if child_prefix in {
                "ranking_metrics.definitions",
                "ranking_metrics.ground_truth_line_results",
            }:
                continue
            output.update(flatten(child, child_prefix))
    elif not isinstance(value, list):
        output[prefix] = value
    return output


def rerank_one(args, filesystem, remote_path, identity):
    dataset, config, seed = identity
    hunks = read_remote_jsonl(
        filesystem, args.hf_repo_id, args.revision, remote_path,
    )
    reranked, summary, units = rerank_jitfine_hunk_chunks(
        hunks,
        chunk_size=args.chunk_size,
        top_k=args.metric_top_k,
        effort_fraction=args.effort_fraction,
    )
    summary["artifact_source"] = {
        "repo_id": args.hf_repo_id,
        "revision": args.revision,
        "path": remote_path,
    }
    run_dir = args.output_dir / dataset / "jitfine_hunk_chunk" / config / f"seed_{seed}"
    write_jsonl(run_dir / HUNKS_FILENAME, reranked)
    write_jsonl(run_dir / UNITS_FILENAME, units)
    write_json(run_dir / SUMMARY_FILENAME, summary)
    record = {
        "dataset": dataset,
        "model": "jitfine_hunk_chunk",
        "config": config,
        "seed": seed,
        "source_file": remote_path,
        "summary_file": str(run_dir / SUMMARY_FILENAME),
    }
    record.update(flatten(summary))
    return record


def discover_sources(args):
    api = HfApi()
    sources = []
    for dataset in args.datasets:
        root = f"line_ranking/{dataset}/jitfine"
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
            identity = parse_source_path(path, set(args.datasets))
            if identity is not None:
                sources.append((path, identity))
    sources.sort(key=lambda item: (item[1][0], item[1][1], str(item[1][2])))
    return sources


def write_csv(path, records):
    identifiers = ["dataset", "model", "config", "seed", "source_file", "summary_file"]
    remaining = sorted({key for record in records for key in record} - set(identifiers))
    columns = identifiers + remaining
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def write_report(records, output_path, csv_dir):
    csv_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, records)
    for dataset in sorted({record["dataset"] for record in records}):
        write_csv(
            csv_dir / f"{dataset}.csv",
            [record for record in records if record["dataset"] == dataset],
        )


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-repo-id", default="TheSyx/vulguard_lite")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--datasets", nargs="+", default=["openssl", "linux"])
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--metric-top-k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--effort-fraction", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("jitfine_hunk_chunk_results"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("jitfine_hunk_chunk_results.csv"),
    )
    parser.add_argument(
        "--csv-dir", type=Path, default=Path("jitfine_hunk_chunk_csv"),
    )
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.chunk_size < 1 or args.workers < 1:
        raise SystemExit("--chunk-size and --workers must be at least 1")
    if not 0 < args.effort_fraction <= 1:
        raise SystemExit("--effort-fraction must be in (0, 1]")

    print(f"Discovering JITFine artifacts in {args.hf_repo_id}@{args.revision} ...")
    sources = discover_sources(args)
    if not sources:
        raise SystemExit("No JITFine ranked_ground_truth_hunks.jsonl artifacts found")
    print(f"Re-ranking {len(sources)} existing experiment(s); no inference will run ...")

    filesystem = HfFileSystem()
    records = []
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(rerank_one, args, filesystem, path, identity): path
            for path, identity in sources
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                records.append(future.result())
            except Exception as error:
                errors.append((path, error))
    if errors:
        for path, error in sorted(errors):
            print(f"ERROR {path}: {error}", file=sys.stderr)
        raise SystemExit(f"Failed to process {len(errors)} experiment(s)")

    records.sort(key=lambda row: (row["dataset"], row["config"], str(row["seed"])))
    write_report(records, args.report, args.csv_dir)
    print(f"Wrote {len(records)} re-ranked summaries to {args.output_dir}")
    print(f"Combined CSV: {args.report}")
    print(f"CSV directory: {args.csv_dir}")


if __name__ == "__main__":
    main()
