#!/usr/bin/env python3
"""Inference-only evaluation at threshold 0.5 for Linux and OpenSSL.

Local server checkpoints are preferred. Hugging Face model_config artifacts are
downloaded only when a complete local checkpoint is unavailable. This module
does not import the training or experiment entry points.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

DATASETS = ("linux", "openssl")
MODELS = ("tlel", "lapredict", "lr", "deepjit", "simcom", "jitfine")
THRESHOLD = 0.5
MODEL_FILES = {
    "tlel": (("tlel.pkl",),),
    "lapredict": (("la.pkl",),),
    "lr": (("lr.pkl",),),
    "deepjit": (("deepjit_checkpoint_last.pth", "deepjit.pth"),),
    "simcom": (("sim.pkl",), ("simcom_checkpoint_last.pth", "com.pth")),
    "jitfine": (("jitfine_checkpoint_last.pth", "jitfine.pth"),),
}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--result-root", type=Path, required=True)
    p.add_argument("--stage-dir", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, required=True)
    p.add_argument("--models", nargs="+", choices=MODELS, required=True)
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--hf-repo-id", default="TheSyx/vulguard_lite")
    p.add_argument("--hf-revision", default="main")
    p.add_argument("--hf-output-repo-id")
    p.add_argument("--upload-results", action="store_true")
    p.add_argument(
        "--train-missing-linux-cpu", action="store_true",
        help=("Train a missing checkpoint only for Linux tlel/lapredict/lr; "
              "all other missing checkpoints remain fatal"),
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a seed only when its scores, metrics, threshold and manifest all exist",
    )
    return p.parse_args()


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hf_index(args):
    api = HfApi(token=os.getenv("HF_TOKEN"))
    info = api.dataset_info(args.hf_repo_id, revision=args.hf_revision)
    return api.list_repo_files(
        args.hf_repo_id, repo_type="dataset", revision=args.hf_revision
    ), info.sha


def download(args, remote):
    return Path(hf_hub_download(
        repo_id=args.hf_repo_id, repo_type="dataset", revision=args.hf_revision,
        filename=remote, local_dir=args.stage_dir, token=os.getenv("HF_TOKEN"),
    ))


def remote_dataset_path(files, dataset, name):
    for candidate in (f"dataset/{dataset}/{name}", f"{dataset}/{name}"):
        if candidate in files:
            return candidate
    matches = [item for item in files if item.endswith(f"/{name}")]
    if len(matches) != 1:
        raise FileNotFoundError(f"Cannot uniquely locate {name} on Hugging Face: {matches}")
    return matches[0]


def local_or_hf_data(args, files, dataset, name):
    candidates = (
        args.repo_root / "datasets" / dataset / name,
        args.repo_root / "dataset" / dataset / name,
        args.repo_root / name,
    )
    local = next((path for path in candidates if path.is_file()), None)
    if local:
        print(f"Dataset local: {local}")
        return local
    remote = remote_dataset_path(files, dataset, name)
    print(f"Dataset Hugging Face: {args.hf_repo_id}/{remote}")
    return download(args, remote)


def commit_ids(path):
    ids = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            row = json.loads(line)
            if "commit_id" not in row:
                raise ValueError(f"Missing commit_id at {path}:{number}")
            ids.append(str(row["commit_id"]))
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate commit IDs in {path}")
    return ids


def inputs(args, files, dataset, model):
    feature = f"test_tlel_{dataset}.jsonl"
    names = {
        "tlel": [feature], "lapredict": [feature], "lr": [feature],
        "deepjit": [f"test_deepjit_{dataset}.jsonl"],
        "jitfine": [feature, f"test_deepjit_{dataset}.jsonl"],
        "simcom": [feature, f"test_simcom_{dataset}.jsonl"],
    }[model]
    paths = [local_or_hf_data(args, files, dataset, name) for name in names]
    reference = commit_ids(paths[0])
    if any(commit_ids(path) != reference for path in paths[1:]):
        raise ValueError(f"Paired {dataset}/{model} test inputs are not row-aligned")
    dictionary = None
    if model in {"deepjit", "simcom"}:
        dictionary = local_or_hf_data(args, files, dataset, f"dict_{dataset}.jsonl")
    return paths, dictionary


def complete(path, model):
    return path.is_dir() and all(
        any((path / name).is_file() for name in choices)
        for choices in MODEL_FILES[model]
    )


def local_checkpoint(args, dataset, model, config, seed):
    model_seed = f"{model}_seed_{seed}"
    candidates = [
        args.checkpoint_root / dataset / model / config / "dg_cache" / "save"
        / dataset / "models" / model_seed / "last_epoch",
        args.checkpoint_root / "checkpoint_fix" / dataset / model / config
        / "dg_cache" / "save" / dataset / "models" / model_seed / "last_epoch",
        args.checkpoint_root / "model_config" / dataset / model / config / f"seed_{seed}",
        args.repo_root / "model_config" / dataset / model / config / f"seed_{seed}",
    ]
    pattern = f"**/{dataset}/{model}/{config}/dg_cache/save/{dataset}/models/{model_seed}/last_epoch"
    candidates.extend(args.checkpoint_root.glob(pattern))
    candidates = list(dict.fromkeys(path for path in candidates if complete(path, model)))
    if candidates:
        chosen = max(candidates, key=lambda directory: max(
            file.stat().st_mtime for file in directory.iterdir() if file.is_file()
        ))
    elif model in {"deepjit", "jitfine"}:
        # Also support the compact local layout used by this repository.
        direct = [
            root / "checkpoint" / model / f"{config}_seed_{seed}{suffix}"
            for root in (args.checkpoint_root, args.repo_root)
            for suffix in (".pth", ".pt")
        ]
        existing = [path for path in direct if path.is_file()]
        if not existing:
            return None
        chosen = max(existing, key=lambda path: path.stat().st_mtime)
    else:
        return None
    print(f"Checkpoint local: {chosen}")
    return chosen


def select_checkpoint(args, files, dataset, model, config, seed):
    local = local_checkpoint(args, dataset, model, config, seed)
    if local:
        return local, str(local)
    prefix = f"model_config/{dataset}/{model}/{config}/seed_{seed}"
    names = {name for choices in MODEL_FILES[model] for name in choices}
    matches = [path for path in files if path.startswith(prefix + "/") and Path(path).name in names]
    groups = {prefix: matches} if matches else {}
    # Older uploads may retain checkpoints inside completed run folders rather
    # than model_config. They are a fallback after the canonical HF location.
    output_prefix = f"output/{dataset}/{model}/"
    for path in files:
        if (
            path.startswith(output_prefix) and config in path
            and f"seed_{seed}_run_" in path and "/checkpoints/" in path
            and Path(path).name in names
        ):
            groups.setdefault(path.rsplit("/", 1)[0], []).append(path)
    for remote_dir, remote_paths in groups.items():
        downloaded = [download(args, path) for path in sorted(remote_paths)]
        directory = downloaded[0].parent
        if complete(directory, model):
            source = f"hf://{args.hf_repo_id}/{remote_dir}"
            print(f"Checkpoint Hugging Face fallback: {source}")
            return directory, source
    if args.train_missing_linux_cpu and dataset == "linux" and model in {
        "tlel", "lapredict", "lr",
    }:
        if args.dry_run:
            expected = (
                args.checkpoint_root / "linux" / model / config / "dg_cache" / "save"
                / "linux" / "models" / f"{model}_seed_{seed}" / "last_epoch"
            )
            print(f"DRY RUN would train missing Linux CPU checkpoint: {expected}")
            return expected, f"would-train://linux/{model}/{config}/seed_{seed}"
        directory = train_linux_cpu_checkpoint(args, model, config, seed)
        return directory, f"trained-local://linux/{model}/{config}/seed_{seed}"
    raise FileNotFoundError(
        f"No complete local or HF checkpoint for {dataset}/{model}/{config}/seed_{seed}"
    )


def train_linux_cpu_checkpoint(args, model, config, seed):
    """Train exactly one missing Linux CPU checkpoint and return last_epoch."""
    from argparse import Namespace
    from vulguard_lite.training import training
    from vulguard_lite.utils.reproducibility import seed_everything

    if model not in {"tlel", "lapredict", "lr"} or not config.startswith("linux_"):
        raise ValueError(f"Training is forbidden outside Linux CPU models: {model}/{config}")
    save_folder = args.checkpoint_root / "linux" / model / config
    model_output = (
        save_folder / "dg_cache" / "save" / "linux" / "models" / f"{model}_seed_{seed}"
    )
    print(
        f"No local/HF checkpoint; training authorized Linux CPU model: "
        f"{model}/{config}/seed_{seed}"
    )
    seed_everything(42)
    params = Namespace(
        repo_name="linux", repo_language="C", model=model, device="cpu",
        dg_save_folder=str(save_folder), hf_repo_id=args.hf_repo_id,
        hf_revision=args.hf_revision, hf_split_path=f"dataset/linux/{config}",
        train_set=None, val_set=None, dictionary=None, hyperparameters=None,
        model_path=None, resume_from_checkpoint=False, checkpoint_dir=None,
        model_output_dir=str(model_output), sampling=True, sampling_seed=seed,
        sampling_run_id=1, seed=42, epochs=30,
    )
    result = training(params)
    directory = Path(result["last_model_dir"])
    if not complete(directory, model):
        raise RuntimeError(f"Training did not produce a complete checkpoint: {directory}")
    print(f"Trained Linux CPU checkpoint: {directory}")
    return directory


def load_model(args, name, checkpoint, dictionary):
    from vulguard_lite.models.init_model import init_model
    hyperparameters = None
    if name not in {"lapredict", "lr"}:
        hyperparameters = args.repo_root / "models" / name / "hyperparameters.json"
        if not hyperparameters.is_file():
            raise FileNotFoundError(hyperparameters)
    model = init_model(name, "C", args.device)
    model.initialize(
        model_path=str(checkpoint), dictionary=str(dictionary) if dictionary else None,
        hyperparameters=str(hyperparameters) if hyperparameters else None,
        inference_only=True,
    )
    return model


def save_result(args, revision, dataset, name, config, seed, checkpoint, source, paths, scores):
    from vulguard_lite.utils.metrics import get_metrics
    scores = scores.copy()
    scores["prediction"] = (scores["probability"] > THRESHOLD).astype(float)
    if scores["commit_id"].astype(str).tolist() != commit_ids(paths[0]):
        raise ValueError("Inference output IDs/order differ from the test input")
    destination = args.result_root / dataset / name / config / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    scores.to_csv(destination / f"{name}_test_scores.csv", index=False,
                  columns=["commit_id", "label", "prediction", "probability"])
    # Classification metrics are valid for every model input format. Effort
    # metrics need a separate Kamei feature file and are intentionally omitted.
    metrics = get_metrics(scores, name, None)
    metrics.insert(0, "model", metrics.index.astype(str))
    metrics.reset_index(drop=True, inplace=True)
    metrics.insert(0, "threshold", THRESHOLD)
    metrics.insert(0, "seed", seed)
    metrics.to_csv(destination / f"{name}_test_metrics.csv", index=False)
    (destination / f"{name}_threshold.json").write_text(json.dumps({
        "threshold": THRESHOLD, "comparison": "probability > threshold"
    }, indent=2), encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "mode": "inference-only",
        "training_called": False, "dataset": dataset, "model": name, "config": config,
        "seed": seed, "threshold": THRESHOLD, "checkpoint_source": source,
        "checkpoint_files": (
            {checkpoint.name: checksum(checkpoint)} if checkpoint.is_file() else
            {p.name: checksum(p) for p in checkpoint.iterdir() if p.is_file()}
        ),
        "dataset_repo": args.hf_repo_id, "dataset_revision": args.hf_revision,
        "dataset_commit": revision, "test_files": {p.name: checksum(p) for p in paths},
        "test_rows": len(scores),
    }
    (destination / "inference_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Saved: {destination}")


def summarize(args, dataset, model, config):
    root = args.result_root / dataset / model / config
    files = sorted(root.glob(f"seed_*/{model}_test_metrics.csv"))
    if not files:
        return
    combined = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    average = {column: combined[column].mean() for column in combined.select_dtypes("number")}
    average.update({"model": "average", "seed": "average", "threshold": THRESHOLD})
    pd.concat([combined, pd.DataFrame([average])], ignore_index=True).to_csv(
        root / f"{model}_threshold_0p5_all_seeds.csv", index=False
    )


def result_is_complete(args, dataset, model, config, seed):
    root = args.result_root / dataset / model / config / f"seed_{seed}"
    required = (
        root / f"{model}_test_scores.csv",
        root / f"{model}_test_metrics.csv",
        root / f"{model}_threshold.json",
        root / "inference_manifest.json",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        manifest = json.loads(required[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("mode") == "inference-only" and manifest.get("threshold") == THRESHOLD


def upload_config(args, dataset, model, config):
    """Upload one completed/partially completed config tree with retries."""
    from vulguard_lite.utils.hf_upload import upload_folder_to_hf_dataset

    local = args.result_root / dataset / model / config
    complete_seeds = [
        seed for seed in args.seeds
        if result_is_complete(args, dataset, model, config, seed)
    ]
    if not complete_seeds:
        print(f"No complete results to upload: {dataset}/{model}/{config}")
        return
    output_repo = args.hf_output_repo_id or args.hf_repo_id
    remote = f"output/threshold_0.5/{dataset}/{model}/{config}"
    message = (
        f"Upload threshold 0.5 inference for {dataset}/{model}/{config} "
        f"({len(complete_seeds)}/{len(args.seeds)} seeds)"
    )
    for attempt in range(1, 6):
        try:
            upload_folder_to_hf_dataset(
                local_folder=str(local), repo_id=output_repo,
                path_in_repo=remote, commit_message=message,
            )
            print(f"Uploaded to Hugging Face: {output_repo}/{remote}")
            return
        except Exception as exc:
            if attempt == 5:
                raise RuntimeError(
                    f"Hugging Face upload failed after {attempt} attempts: {exc}"
                ) from exc
            delay = 2 ** attempt
            print(f"HF upload attempt {attempt}/5 failed; retry in {delay}s: {exc}")
            time.sleep(delay)


def main():
    args = arguments()
    for attr in ("repo_root", "checkpoint_root", "result_root", "stage_dir"):
        setattr(args, attr, getattr(args, attr).resolve())
    if args.result_root == args.checkpoint_root:
        raise ValueError("Result root must differ from checkpoint root")
    args.stage_dir.mkdir(parents=True, exist_ok=True)
    files, revision = hf_index(args)
    failures = []
    for dataset in args.datasets:
        configs = [item for item in args.configs if item.startswith(dataset + "_")]
        if not configs:
            failures.append((dataset, "*", "*", "no matching config"))
            continue
        for name in args.models:
            paths, dictionary = inputs(args, files, dataset, name)
            for config in configs:
                for seed in args.seeds:
                    label = f"{dataset}/{name}/{config}/seed_{seed}"
                    try:
                        if args.skip_existing and result_is_complete(
                            args, dataset, name, config, seed
                        ):
                            print(f"SKIP complete result: {label}")
                            continue
                        checkpoint, source = select_checkpoint(args, files, dataset, name, config, seed)
                        if args.dry_run:
                            print(f"DRY RUN inference-only: {label} <- {source}")
                            continue
                        model = load_model(args, name, checkpoint, dictionary)
                        scores = model.inference(
                            infer_df=",".join(map(str, paths)), threshold=THRESHOLD
                        )
                        save_result(args, revision, dataset, name, config, seed,
                                    checkpoint, source, paths, scores)
                        del model
                    except Exception as exc:
                        failures.append((dataset, name, config, f"seed_{seed}: {exc}"))
                        print(f"FAILED {label}: {exc}", file=sys.stderr)
                if not args.dry_run:
                    summarize(args, dataset, name, config)
                    if args.upload_results:
                        try:
                            upload_config(args, dataset, name, config)
                        except Exception as exc:
                            failures.append((dataset, name, config, f"upload: {exc}"))
                            print(
                                f"FAILED upload {dataset}/{name}/{config}: {exc}",
                                file=sys.stderr,
                            )
    if failures:
        print("Failures:", file=sys.stderr)
        for dataset, name, config, error in failures:
            print(f"  {dataset}/{name}/{config}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
