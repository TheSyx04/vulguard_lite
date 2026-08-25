#!/usr/bin/env python3
"""Upload an already-completed line-ranking model tree to an HF dataset."""

import argparse
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vulguard_lite.utils.hf_upload import upload_folder_to_hf_dataset


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
        if missing:
            incomplete.append((directory, missing))
    return seed_directories, incomplete


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

    if not incomplete:
        upload_folder_to_hf_dataset(
            local_folder=str(model_root),
            repo_id=args.hf_repo_id,
            path_in_repo=remote_path,
            commit_message=f"Upload {args.dataset}/{args.model} line-ranking results",
        )
        return

    for index, directory in enumerate(complete_directories, start=1):
        relative_path = directory.relative_to(model_root).as_posix()
        destination = f"{remote_path}/{relative_path}"
        print(f"Uploading {index}/{len(complete_directories)}: {destination}")
        upload_folder_to_hf_dataset(
            local_folder=str(directory),
            repo_id=args.hf_repo_id,
            path_in_repo=destination,
            commit_message=(
                f"Upload {args.dataset}/{args.model} line ranking for {relative_path}"
            ),
        )


if __name__ == "__main__":
    main()
