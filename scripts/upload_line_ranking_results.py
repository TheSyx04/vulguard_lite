#!/usr/bin/env python3
"""Upload an already-completed line-ranking model tree to an HF dataset."""

import argparse
import json
from pathlib import Path, PurePosixPath
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vulguard_lite.utils.hf_upload import (
    list_hf_dataset_files,
    upload_folder_to_hf_dataset,
)


REQUIRED_RESULTS = {
    "ranked_ground_truth_hunks.jsonl",
    "ranked_ground_truth_summary.json",
    "ranking_metrics_by_unit.jsonl",
}


def completed_seed_directories(model_root):
    seed_directories = sorted(path for path in model_root.glob("*/seed_*") if path.is_dir())
    incomplete = []
    for directory in seed_directories:
        missing = sorted(name for name in REQUIRED_RESULTS if not (directory / name).is_file())
        attribution_summary = directory / "attribution" / "summary.json"
        if not attribution_summary.is_file():
            missing.append("attribution/summary.json")
        else:
            try:
                with attribution_summary.open("r", encoding="utf-8") as handle:
                    failed = int(json.load(handle).get("failed", 0))
                if failed:
                    missing.append(f"attribution_failed={failed}")
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                missing.append("invalid_attribution_summary")
        if missing:
            incomplete.append((directory, missing))
    return seed_directories, incomplete


def missing_upload_patterns(local_folder, path_in_repo, existing_files):
    """Return local relative paths whose destination does not exist on HF."""
    remote_root = path_in_repo.strip("/")
    missing = []
    existing_count = 0
    for local_path in sorted(path for path in local_folder.rglob("*") if path.is_file()):
        relative_path = local_path.relative_to(local_folder).as_posix()
        remote_path = (
            str(PurePosixPath(remote_root) / relative_path)
            if remote_root
            else relative_path
        )
        if remote_path in existing_files:
            existing_count += 1
        else:
            missing.append(relative_path)
    return missing, existing_count


def main():
    parser = argparse.ArgumentParser(
        description="Backfill completed line-ranking results to Hugging Face",
    )
    parser.add_argument("--output-root", required=True, help="Root containing <dataset>/<model>")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, choices=["jitfine", "deepjit", "simcom"])
    parser.add_argument("--hf-repo-id", required=True, help="Destination HF dataset repo")
    parser.add_argument("--hf-output-folder", default=None)
    parser.add_argument(
        "--skip-incomplete", action="store_true",
        help="Upload complete config/seed folders and skip incomplete ones",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_root = Path(args.output_root).expanduser().resolve() / args.dataset / args.model
    if not model_root.is_dir():
        raise SystemExit(f"Line-ranking model directory not found: {model_root}")

    seed_directories, incomplete = completed_seed_directories(model_root)
    if not seed_directories:
        raise SystemExit(f"No <config>/seed_* directories found under: {model_root}")
    if incomplete and not args.skip_incomplete:
        print("Refusing to upload a model tree containing incomplete seed outputs:", file=sys.stderr)
        for directory, missing in incomplete:
            print(f"  {directory}: missing {', '.join(missing)}", file=sys.stderr)
        raise SystemExit(2)

    remote_path = (
        args.hf_output_folder.strip("/")
        if args.hf_output_folder
        else f"line_ranking/{args.dataset}/{args.model}"
    )
    incomplete_paths = {directory for directory, _ in incomplete}
    complete_directories = [
        directory for directory in seed_directories if directory not in incomplete_paths
    ]
    if not complete_directories:
        raise SystemExit("No completed seed outputs found")

    print(f"Completed seed outputs : {len(complete_directories)}")
    print(f"Incomplete seed outputs: {len(incomplete)}")
    print(f"Local folder           : {model_root}")
    print(f"HF destination         : {args.hf_repo_id}/{remote_path}")
    if args.dry_run:
        return

    if incomplete:
        upload_jobs = [
            (
                directory,
                f"{remote_path}/{directory.relative_to(model_root).as_posix()}",
                (
                    f"Upload {args.dataset}/{args.model} line ranking for "
                    f"{directory.relative_to(model_root).as_posix()}"
                ),
            )
            for directory in complete_directories
        ]
    else:
        upload_jobs = [
            (
                model_root,
                remote_path,
                f"Upload {args.dataset}/{args.model} line-ranking results",
            )
        ]

    print("Listing files already present on Hugging Face ...")
    existing_files = list_hf_dataset_files(args.hf_repo_id)
    uploaded_count = 0
    skipped_count = 0
    for index, (directory, destination, commit_message) in enumerate(upload_jobs, start=1):
        missing_patterns, existing_count = missing_upload_patterns(
            directory, destination, existing_files
        )
        skipped_count += existing_count
        if not missing_patterns:
            print(f"Skipping {index}/{len(upload_jobs)} (all files exist): {destination}")
            continue

        print(
            f"Uploading {index}/{len(upload_jobs)}: {destination} "
            f"({len(missing_patterns)} new, {existing_count} existing)"
        )
        upload_folder_to_hf_dataset(
            local_folder=str(directory),
            repo_id=args.hf_repo_id,
            path_in_repo=destination,
            allow_patterns=missing_patterns,
            commit_message=commit_message,
        )
        uploaded_count += len(missing_patterns)
        existing_files.update(
            f"{destination.strip('/')}/{path}" if destination.strip("/") else path
            for path in missing_patterns
        )

    print(f"New files uploaded     : {uploaded_count}")
    print(f"Existing files skipped : {skipped_count}")


if __name__ == "__main__":
    main()
