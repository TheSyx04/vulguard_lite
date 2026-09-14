#!/usr/bin/env python3
"""Refresh the OpenSSL test set and re-run inference from completed checkpoints.

This intentionally does not call ``training()``.  For each sampling seed it
loads the current ``last_epoch`` model once, then reapplies the thresholds
already selected during the completed validation/evaluation runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


MODELS = ("tlel", "lapredict", "lr", "deepjit", "simcom", "jitfine")
MODEL_FILES = {
    "tlel": ("tlel.pkl",),
    "lapredict": ("la.pkl",),
    "lr": ("lr.pkl",),
    "deepjit": ("deepjit.pth", "deepjit_checkpoint_last.pth"),
    "simcom": ("sim.pkl", "com.pth", "simcom_checkpoint_last.pth"),
    "jitfine": ("jitfine.pth", "jitfine_checkpoint_last.pth"),
}
TEST_PREFERENCES = {
    "features": ("out_test_tlel_openssl.jsonl", "test_tlel_openssl.jsonl"),
    "deepjit": ("out_test_deepjit_openssl.jsonl", "test_deepjit_openssl.jsonl"),
    "jitfine": (
        "out_test_jitfine_openssl.jsonl",
        "test_jitfine_openssl.jsonl",
        "out_test_deepjit_openssl.jsonl",
        "test_deepjit_openssl.jsonl",
        "out_test_merge_openssl.jsonl",
        "test_merge_openssl.jsonl",
    ),
    "simcom": ("out_test_simcom_openssl.jsonl", "test_simcom_openssl.jsonl"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Existing server output root containing checkpoints and completed runs",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        required=True,
        help="Separate destination root for reinference artifacts",
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hf-repo-id", default="TheSyx/vulguard_lite")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-output-repo-id")
    parser.add_argument("--upload-results", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_commit_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            commit_id = row.get("commit_id")
            if commit_id is None:
                raise ValueError(f"Missing commit_id at {path}:{line_number}")
            ids.append(str(commit_id))
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate commit_id values in {path}")
    return ids


def download_fresh_test_dataset(args: argparse.Namespace) -> tuple[dict[str, Path], str]:
    """Force-download direct children of dataset/openssl needed for inference."""
    api = HfApi(token=os.getenv("HF_TOKEN"))
    info = api.dataset_info(args.hf_repo_id, revision=args.hf_revision)
    repo_files = api.list_repo_files(
        args.hf_repo_id, repo_type="dataset", revision=args.hf_revision
    )
    prefix = "dataset/openssl/"
    selected = [
        name
        for name in repo_files
        if name.startswith(prefix)
        and "/" not in name[len(prefix) :]
        and (
            Path(name).name.startswith(("test_", "out_test_"))
            or Path(name).name == "dict_openssl.jsonl"
        )
    ]
    if not any(Path(name).name.startswith(("test_", "out_test_")) for name in selected):
        raise FileNotFoundError(f"No OpenSSL test files found in {args.hf_repo_id}@{args.hf_revision}")

    args.stage_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    for remote_path in sorted(selected):
        local = Path(
            hf_hub_download(
                repo_id=args.hf_repo_id,
                repo_type="dataset",
                revision=args.hf_revision,
                filename=remote_path,
                local_dir=args.stage_dir,
                force_download=True,
                token=os.getenv("HF_TOKEN"),
            )
        )
        downloaded[Path(remote_path).name] = local
        print(f"Downloaded fresh: {remote_path} ({local.stat().st_size} bytes)")
    return downloaded, info.sha


def pick(downloaded: dict[str, Path], preferences: tuple[str, ...]) -> Path:
    for name in preferences:
        if name in downloaded:
            return downloaded[name]
    raise FileNotFoundError(f"None of these files exists on Hugging Face: {preferences}")


def test_paths(model_name: str, downloaded: dict[str, Path]) -> list[Path]:
    features = pick(downloaded, TEST_PREFERENCES["features"])
    if model_name in {"tlel", "lapredict", "lr"}:
        paths = [features]
    elif model_name == "deepjit":
        paths = [pick(downloaded, TEST_PREFERENCES["deepjit"])]
    elif model_name == "jitfine":
        paths = [features, pick(downloaded, TEST_PREFERENCES["jitfine"])]
    elif model_name == "simcom":
        paths = [features, pick(downloaded, TEST_PREFERENCES["simcom"])]
    else:  # pragma: no cover - argparse guards this
        raise ValueError(model_name)

    reference = jsonl_commit_ids(paths[0])
    for path in paths[1:]:
        current = jsonl_commit_ids(path)
        if current != reference:
            raise ValueError(
                f"Paired OpenSSL test files are not row-aligned: {paths[0]} vs {path}"
            )
    print(f"Fresh test rows for {model_name}: {len(reference)}")
    return paths


def atomic_copy(source: Path, destination: Path) -> tuple[int | None, int]:
    old_rows = None
    if destination.exists() and destination.suffix == ".jsonl":
        old_rows = len(jsonl_commit_ids(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    new_rows = len(jsonl_commit_ids(destination)) if destination.suffix == ".jsonl" else 0
    return old_rows, new_rows


def overwrite_split_cache(
    save_folder: Path, downloaded: dict[str, Path], args: argparse.Namespace
) -> None:
    cache_root = (
        save_folder
        / "dg_cache"
        / "dataset"
        / "openssl"
        / "hf"
        / args.hf_repo_id.replace("/", "__")
        / args.hf_revision
        / "dataset"
        / "openssl"
    )
    for name, source in sorted(downloaded.items()):
        if not name.startswith(("test_", "out_test_")):
            continue
        destination = cache_root / name
        if args.dry_run:
            print(f"DRY RUN overwrite {destination} <- {source}")
            continue
        old_rows, new_rows = atomic_copy(source, destination)
        delta = "new file" if old_rows is None else f"delta={new_rows - old_rows:+d}"
        print(f"Overwrote test cache: {destination} ({new_rows} rows, {delta})")


def find_experiment_root(save_folder: Path, model_name: str, config: str) -> Path:
    expected = f"{model_name}_openssl_{config}_sampling"
    root = save_folder / "dg_cache" / "save" / "openssl" / "experiments" / expected
    if not root.is_dir():
        raise FileNotFoundError(f"Completed experiment directory not found: {root}")
    return root


def find_save_folder(
    output_root: Path, model_name: str, config: str, seeds: list[int]
) -> Path:
    """Select the newest complete local output tree, including checkpoint_fix trees."""
    canonical = output_root / "openssl" / model_name / config
    candidates = [canonical] if canonical.is_dir() else []
    for path in output_root.rglob(config):
        if (
            path.is_dir()
            and path.parent.name == model_name
            and path.parent.parent.name == "openssl"
            and path not in candidates
        ):
            candidates.append(path)

    usable: list[tuple[float, Path]] = []
    expected_experiment = f"{model_name}_openssl_{config}_sampling"
    for path in candidates:
        experiment = path / "dg_cache" / "save" / "openssl" / "experiments" / expected_experiment
        if not experiment.is_dir():
            continue
        seed_dirs = [
            path
            / "dg_cache"
            / "save"
            / "openssl"
            / "models"
            / f"{model_name}_seed_{seed}"
            / "last_epoch"
            for seed in seeds
        ]
        if not all(directory.is_dir() for directory in seed_dirs):
            continue
        try:
            checkpoint_candidates = [
                file
                for directory in seed_dirs
                for file in checkpoint_files(directory, model_name)
            ]
        except FileNotFoundError:
            continue
        latest_mtime = max(file.stat().st_mtime for file in checkpoint_candidates)
        usable.append((latest_mtime, path))
    if not usable:
        raise FileNotFoundError(
            f"No complete output tree found for openssl/{model_name}/{config} under {output_root}"
        )
    usable.sort(key=lambda item: (item[0], str(item[1])))
    selected = usable[-1][1]
    if len(usable) > 1:
        print(f"Found {len(usable)} output trees; selected newest: {selected}")
    return selected


def checkpoint_files(checkpoint_dir: Path, model_name: str) -> list[Path]:
    existing = [checkpoint_dir / name for name in MODEL_FILES[model_name] if (checkpoint_dir / name).is_file()]
    if model_name == "simcom":
        required_groups = (("sim.pkl",), ("com.pth", "simcom_checkpoint_last.pth"))
        if any(not any((checkpoint_dir / name).is_file() for name in group) for group in required_groups):
            raise FileNotFoundError(f"Incomplete SimCom checkpoint in {checkpoint_dir}")
    if not existing:
        raise FileNotFoundError(f"Latest checkpoint not found in {checkpoint_dir}")
    return existing


def completed_run_dirs(experiment_root: Path, model_name: str, seed: int) -> list[Path]:
    completed = []
    for run_dir in sorted(experiment_root.glob(f"seed_{seed}_run_*")):
        if (run_dir / f"{model_name}_test_metrics.csv").is_file():
            completed.append(run_dir)
    return completed


def verify_checkpoint_was_evaluated(checkpoints: list[Path], run_dirs: list[Path], model_name: str) -> None:
    if not run_dirs:
        raise FileNotFoundError("No completed evaluation artifacts found for this seed")
    checkpoint_mtime = max(path.stat().st_mtime for path in checkpoints)
    evaluation_mtime = max(
        (run_dir / f"{model_name}_test_metrics.csv").stat().st_mtime for run_dir in run_dirs
    )
    if checkpoint_mtime > evaluation_mtime:
        raise RuntimeError(
            "Latest checkpoint is newer than every completed evaluation artifact; "
            "refusing to infer because it cannot be verified as evaluated"
        )


def initialize_model(
    model_name: str,
    checkpoint_dir: Path,
    downloaded: dict[str, Path],
    repo_root: Path,
    device: str,
):
    from vulguard_lite.models.init_model import init_model

    dictionary = downloaded.get("dict_openssl.jsonl")
    if model_name in {"deepjit", "simcom"} and dictionary is None:
        raise FileNotFoundError("dict_openssl.jsonl was not found in the Hugging Face dataset")
    hyperparameters = None
    if model_name not in {"lapredict", "lr"}:
        hyperparameters = repo_root / "models" / model_name / "hyperparameters.json"
        if not hyperparameters.is_file():
            raise FileNotFoundError(hyperparameters)

    model = init_model(model_name, "C", device)
    model.initialize(
        model_path=str(checkpoint_dir),
        dictionary=str(dictionary) if dictionary else None,
        hyperparameters=str(hyperparameters) if hyperparameters else None,
        inference_only=True,
    )
    return model


def update_run_artifacts(run_dir: Path, model_name: str, base_scores: pd.DataFrame) -> None:
    from vulguard_lite.utils.metrics import get_metrics

    summary_path = run_dir / f"{model_name}_test_metrics.csv"
    summary = pd.read_csv(summary_path)
    threshold_files = sorted(run_dir.glob(f"{model_name}_budget_*_selected_threshold.json"))
    if not threshold_files:
        raise FileNotFoundError(f"No saved thresholds found in completed run: {run_dir}")

    seen_budgets: set[float] = set()
    for threshold_path in threshold_files:
        with threshold_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        budget = float(payload["budget"])
        threshold = float(payload["threshold"])
        seen_budgets.add(budget)
        stem = threshold_path.name[: -len("_selected_threshold.json")]
        scores_path = run_dir / f"{stem}_test_scores.csv"
        metrics_path = run_dir / f"{stem}_test_metrics.csv"

        scores = base_scores.copy()
        scores["prediction"] = (scores["probability"] > threshold).astype(float)
        scores.to_csv(
            scores_path,
            index=False,
            columns=["commit_id", "label", "prediction", "probability"],
        )
        metrics = get_metrics(scores, model_name, None)
        metrics.to_csv(metrics_path, index=True)

        mask = (pd.to_numeric(summary["budget"], errors="coerce") - budget).abs() < 1e-12
        if not mask.any():
            raise ValueError(f"Budget {budget} is absent from {summary_path}")
        for column, value in metrics.iloc[0].items():
            summary.loc[mask, column] = value
        summary.loc[mask, "threshold"] = threshold
        print(f"Updated {run_dir.name}, budget={budget}, threshold={threshold}")

    summary_budgets = set(pd.to_numeric(summary["budget"], errors="raise").astype(float))
    if seen_budgets != summary_budgets:
        raise ValueError(
            f"Threshold/summary budget mismatch in {run_dir}: {seen_budgets} != {summary_budgets}"
        )
    summary.to_csv(summary_path, index=False)


def prepare_result_experiment(source_root: Path, result_root: Path) -> None:
    """Copy completed artifacts so reinference never mutates original results."""
    if source_root.resolve() == result_root.resolve():
        raise ValueError("Reinference result directory must differ from the source experiment")
    result_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, result_root, dirs_exist_ok=True)
    print(f"Copied source artifacts without modifying them: {source_root} -> {result_root}")


def rebuild_summary(experiment_root: Path, model_name: str) -> Path:
    run_files = sorted(experiment_root.glob(f"seed_*_run_*/{model_name}_test_metrics.csv"))
    if not run_files:
        raise FileNotFoundError(f"No completed run summaries under {experiment_root}")
    combined = pd.concat((pd.read_csv(path) for path in run_files), ignore_index=True)
    sort_columns = [name for name in ("budget", "seed", "run") if name in combined.columns]
    combined = combined.sort_values(sort_columns, kind="stable")
    metric_columns = [
        column
        for column in combined.columns
        if column not in {"model", "run", "budget", "seed"}
        and pd.api.types.is_numeric_dtype(combined[column])
    ]
    frames = []
    for budget in sorted(combined["budget"].dropna().unique()):
        budget_frame = combined[combined["budget"] == budget].copy()
        average = {"model": "average", "run": "average", "budget": float(budget)}
        average.update({column: budget_frame[column].mean() for column in metric_columns})
        frames.extend((budget_frame, pd.DataFrame([average])))
    final = pd.concat(frames, ignore_index=True)
    path = experiment_root / f"{experiment_root.name}_test_all_budget.csv"
    final.to_csv(path, index=False)
    return path


def upload_experiment(args: argparse.Namespace, model_name: str, experiment_root: Path) -> None:
    from vulguard_lite.utils.hf_upload import upload_folder_to_hf_dataset

    output_repo = args.hf_output_repo_id or args.hf_repo_id
    remote_path = f"output/openssl_reinfer/{model_name}/sampling/{experiment_root.name}"
    upload_folder_to_hf_dataset(
        local_folder=str(experiment_root),
        repo_id=output_repo,
        path_in_repo=remote_path,
        commit_message=f"Refresh OpenSSL test inference for {experiment_root.name}",
    )
    print(f"Uploaded corrected results: {output_repo}/{remote_path}")


def process_experiment(
    args: argparse.Namespace,
    model_name: str,
    config: str,
    downloaded: dict[str, Path],
    revision_sha: str,
) -> None:
    save_folder = find_save_folder(args.output_root, model_name, config, args.seeds)
    source_experiment_root = find_experiment_root(save_folder, model_name, config)
    experiment_root = args.result_root / model_name / config / source_experiment_root.name
    inputs = test_paths(model_name, downloaded)
    overwrite_split_cache(save_folder, downloaded, args)
    if not args.dry_run:
        prepare_result_experiment(source_experiment_root, experiment_root)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "inference-only",
        "source_experiment": str(source_experiment_root),
        "result_experiment": str(experiment_root),
        "dataset_repo": args.hf_repo_id,
        "dataset_revision": args.hf_revision,
        "dataset_commit": revision_sha,
        "test_rows": len(jsonl_commit_ids(inputs[0])),
        "test_files": {path.name: sha256(path) for path in inputs},
        "checkpoints": [],
    }

    for seed in args.seeds:
        checkpoint_dir = (
            save_folder
            / "dg_cache"
            / "save"
            / "openssl"
            / "models"
            / f"{model_name}_seed_{seed}"
            / "last_epoch"
        )
        checkpoints = checkpoint_files(checkpoint_dir, model_name)
        source_run_dirs = completed_run_dirs(source_experiment_root, model_name, seed)
        verify_checkpoint_was_evaluated(checkpoints, source_run_dirs, model_name)
        print(f"Verified evaluated latest checkpoint: {checkpoint_dir}")
        manifest["checkpoints"].append(
            {
                "seed": seed,
                "files": {path.name: sha256(path) for path in checkpoints},
                "completed_runs": [path.name for path in source_run_dirs],
            }
        )
        if args.dry_run:
            continue
        model = initialize_model(
            model_name, checkpoint_dir, downloaded, args.repo_root, args.device
        )
        base_scores = model.inference(
            infer_df=",".join(str(path) for path in inputs), threshold=0.5
        )
        if len(base_scores) != manifest["test_rows"]:
            raise ValueError(
                f"Inference returned {len(base_scores)} rows for {manifest['test_rows']} test commits"
            )
        for source_run_dir in source_run_dirs:
            update_run_artifacts(experiment_root / source_run_dir.name, model_name, base_scores)
        del model

    if args.dry_run:
        return
    summary_path = rebuild_summary(experiment_root, model_name)
    manifest_path = experiment_root / "openssl_test_reinference_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Rebuilt summary: {summary_path}")
    print(f"Wrote provenance: {manifest_path}")
    if args.upload_results:
        upload_experiment(args, model_name, experiment_root)


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.result_root = args.result_root.resolve()
    args.repo_root = args.repo_root.resolve()
    args.stage_dir = args.stage_dir.resolve()
    for config in args.configs:
        if not config.startswith("openssl_"):
            raise ValueError(f"Only OpenSSL configurations are allowed: {config}")

    downloaded, revision_sha = download_fresh_test_dataset(args)
    print(f"Pinned fresh dataset commit: {revision_sha}")
    failures = []
    for model_name in args.models:
        for config in args.configs:
            print(f"\n=== OpenSSL reinference: {model_name} / {config} ===")
            try:
                process_experiment(args, model_name, config, downloaded, revision_sha)
            except Exception as exc:
                failures.append((model_name, config, str(exc)))
                print(f"FAILED {model_name}/{config}: {exc}", file=sys.stderr)
    if failures:
        print("\nFailed experiments:", file=sys.stderr)
        for model_name, config, error in failures:
            print(f"  {model_name}/{config}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
