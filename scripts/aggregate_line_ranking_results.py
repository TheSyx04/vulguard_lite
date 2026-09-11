#!/usr/bin/env python3
"""Download and aggregate line-ranking experiment summaries from Hugging Face.

The generated Excel workbook contains one sheet per dataset and one row per
``(model, config, seed)`` experiment. CSV cannot contain sheets, so ``--csv-dir``
optionally writes one CSV file per dataset in addition to the workbook.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path, PurePosixPath
import re
import sys

import pandas as pd
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download


DEFAULT_DATASETS = ("openssl", "linux")
DEFAULT_MODELS = ("jitfine", "deepjit", "simcom")
SUMMARY_FILENAME = "ranked_ground_truth_summary.json"
IDENTIFIER_COLUMNS = ("dataset", "model", "config", "seed", "source_file")


def parse_summary_path(path, datasets, models):
    """Parse a canonical line-ranking summary path, or return ``None``."""
    parts = PurePosixPath(path).parts
    if len(parts) != 6 or parts[0] != "line_ranking":
        return None
    _, dataset, model, config, seed_part, filename = parts
    if dataset not in datasets or model not in models or filename != SUMMARY_FILENAME:
        return None
    match = re.fullmatch(r"seed_(.+)", seed_part)
    if not match:
        return None
    seed_text = match.group(1)
    seed = int(seed_text) if re.fullmatch(r"\d+", seed_text) else seed_text
    return dataset, model, config, seed


def flatten_summary(value, prefix=""):
    """Flatten scalar summary fields while omitting bulky line-level payloads."""
    flattened = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            if child_prefix in {
                "ranking_metrics.definitions",
                "ranking_metrics.ground_truth_line_results",
            }:
                continue
            flattened.update(flatten_summary(child, child_prefix))
    elif isinstance(value, list):
        # Lists of dictionaries are detailed diagnostics, not experiment-level
        # metrics. Scalar lists such as top_k remain useful in the workbook.
        if not value or all(not isinstance(item, (dict, list)) for item in value):
            flattened[prefix] = json.dumps(value, ensure_ascii=False)
    else:
        flattened[prefix] = value
    return flattened


def download_summary(repo_id, revision, remote_path):
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="dataset",
            revision=revision,
        )
        with open(local_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as cache_error:
        # On Windows, HF's cache may occasionally fail to create a snapshot
        # symlink with WinError 1314 when Developer Mode is disabled. Reading
        # through HfFileSystem bypasses that local symlink without requiring
        # administrator privileges. If this also fails, retain both causes.
        filesystem = HfFileSystem()
        filesystem_path = f"datasets/{repo_id}/{remote_path}"
        try:
            with filesystem.open(filesystem_path, "rb", revision=revision) as handle:
                return json.load(handle)
        except Exception as direct_error:
            raise RuntimeError(
                f"cache download failed ({cache_error}); "
                f"direct HF read also failed ({direct_error})"
            ) from direct_error


def natural_key(value):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(value))]


def parse_experiment_filters(values):
    """Parse exact ``model/config/seed`` selectors from the CLI."""
    selected = set()
    for value in values or []:
        parts = PurePosixPath(value.strip("/")).parts
        if len(parts) != 3:
            raise ValueError(
                f"invalid experiment selector {value!r}; expected model/config/seed"
            )
        model, config, seed = parts
        if seed.startswith("seed_"):
            seed = seed[5:]
        if not model or not config or not seed:
            raise ValueError(
                f"invalid experiment selector {value!r}; expected model/config/seed"
            )
        selected.add((model, config, seed))
    return selected


def missing_experiments(records, dataset, models):
    """Find holes in the discovered model x config x seed matrix."""
    present = {
        (record["model"], record["config"], str(record["seed"]))
        for record in records if record["dataset"] == dataset
    }
    configs = sorted({item[1] for item in present}, key=natural_key)
    seeds = sorted({item[2] for item in present}, key=natural_key)
    return [
        (model, config, seed)
        for model in models
        for config in configs
        for seed in seeds
        if (model, config, seed) not in present
    ]


def build_frames(records, datasets):
    frames = {}
    for dataset in datasets:
        rows = [record for record in records if record["dataset"] == dataset]
        rows.sort(key=lambda row: (
            natural_key(row["model"]),
            natural_key(row["config"]),
            natural_key(row["seed"]),
        ))
        frame = pd.DataFrame(rows)
        if frame.empty:
            frames[dataset] = pd.DataFrame(columns=IDENTIFIER_COLUMNS)
            continue
        metric_columns = sorted(
            (column for column in frame.columns if column not in IDENTIFIER_COLUMNS),
            key=natural_key,
        )
        frames[dataset] = frame[list(IDENTIFIER_COLUMNS) + metric_columns]
    return frames


def write_outputs(frames, output_path, csv_dir=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for dataset, frame in frames.items():
            frame.to_excel(writer, sheet_name=dataset[:31], index=False)
            worksheet = writer.sheets[dataset[:31]]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

    if csv_dir:
        csv_dir.mkdir(parents=True, exist_ok=True)
        for dataset, frame in frames.items():
            frame.to_csv(csv_dir / f"{dataset}.csv", index=False, encoding="utf-8-sig")


def make_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate Hugging Face line-ranking summaries into a two-sheet "
            "Excel workbook."
        ),
    )
    parser.add_argument("--hf-repo-id", default="TheSyx/vulguard_lite")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--experiments", nargs="+", default=[], metavar="MODEL/CONFIG/SEED",
        help=(
            "Download only these exact experiments, for example "
            "jitfine/openssl_0_2/seed_2. Intentional matrix holes are ignored."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("line_ranking_results.xlsx"),
    )
    parser.add_argument(
        "--csv-dir", type=Path,
        help="Also write one CSV per dataset into this directory",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Write available experiments even when the model/config/seed matrix has holes",
    )
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if len(set(args.datasets)) != len(args.datasets):
        raise SystemExit("--datasets contains duplicates")
    if len(set(args.models)) != len(args.models):
        raise SystemExit("--models contains duplicates")
    try:
        selected_experiments = parse_experiment_filters(args.experiments)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    unknown_selected_models = sorted(
        {model for model, _, _ in selected_experiments} - set(args.models)
    )
    if unknown_selected_models:
        raise SystemExit(
            "--experiments contains models excluded by --models: "
            + ", ".join(unknown_selected_models)
        )

    print(f"Listing line-ranking results in {args.hf_repo_id}@{args.revision} ...")
    repo_files = HfApi().list_repo_files(
        repo_id=args.hf_repo_id,
        repo_type="dataset",
        revision=args.revision,
    )
    experiments = []
    for remote_path in repo_files:
        parsed = parse_summary_path(remote_path, set(args.datasets), set(args.models))
        if parsed and (
            not selected_experiments
            or (parsed[1], parsed[2], str(parsed[3])) in selected_experiments
        ):
            experiments.append((remote_path, parsed))
    if selected_experiments:
        discovered = {
            (model, config, str(seed))
            for _, (_, model, config, seed) in experiments
        }
        missing_selected = sorted(selected_experiments - discovered, key=natural_key)
        if missing_selected:
            formatted = ", ".join("/".join(item) for item in missing_selected)
            raise SystemExit(f"Requested experiment(s) not found: {formatted}")
    if not experiments:
        raise SystemExit(
            "No line-ranking summary found under "
            f"line_ranking/<dataset>/<model>/<config>/seed_*/{SUMMARY_FILENAME}"
        )

    print(f"Downloading {len(experiments)} experiment summaries ...")
    records = []
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_summary, args.hf_repo_id, args.revision, path):
            (path, parsed)
            for path, parsed in experiments
        }
        for future in as_completed(futures):
            remote_path, (dataset, model, config, seed) = futures[future]
            try:
                summary = future.result()
            except Exception as error:  # report every failed remote file together
                errors.append((remote_path, error))
                continue
            record = {
                "dataset": dataset,
                "model": model,
                "config": config,
                "seed": seed,
                "source_file": remote_path,
            }
            for key, value in flatten_summary(summary).items():
                # Path-derived identifiers are authoritative and stay first.
                if key not in IDENTIFIER_COLUMNS:
                    record[key] = value
            records.append(record)

    if errors:
        for path, error in sorted(errors):
            print(f"ERROR {path}: {error}", file=sys.stderr)
        raise SystemExit(f"Failed to download {len(errors)} summary file(s)")

    missing_by_dataset = {
        dataset: (
            [] if selected_experiments
            else missing_experiments(records, dataset, args.models)
        )
        for dataset in args.datasets
    }
    empty_datasets = [
        dataset for dataset in args.datasets
        if not any(record["dataset"] == dataset for record in records)
    ]
    missing_count = sum(map(len, missing_by_dataset.values()))
    if empty_datasets:
        print("No experiments found for: " + ", ".join(empty_datasets), file=sys.stderr)
    if missing_count:
        print(f"Detected {missing_count} missing model/config/seed experiment(s):", file=sys.stderr)
        for dataset, missing in missing_by_dataset.items():
            for model, config, seed in missing:
                print(f"  {dataset}/{model}/{config}/seed_{seed}", file=sys.stderr)
    if (empty_datasets or missing_count) and not args.allow_incomplete:
        raise SystemExit(
            "Result matrix is incomplete; upload the missing experiments or rerun "
            "with --allow-incomplete."
        )

    frames = build_frames(records, args.datasets)
    write_outputs(frames, args.output.resolve(), args.csv_dir.resolve() if args.csv_dir else None)
    for dataset, frame in frames.items():
        print(f"{dataset}: {len(frame)} experiment(s)")
    print(f"Workbook: {args.output.resolve()}")
    if args.csv_dir:
        print(f"CSV files: {args.csv_dir.resolve()}")


if __name__ == "__main__":
    main()
