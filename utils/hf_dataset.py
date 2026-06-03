import json
import os
import shutil
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


class HFDatasetError(RuntimeError):
    pass


def _normalize_repo_id(repo_id):
    return quote(repo_id, safe="/")


@lru_cache(maxsize=16)
def _list_dataset_files(repo_id, revision="main"):
    repo_id_quoted = _normalize_repo_id(repo_id)
    revision_quoted = quote(revision, safe="")
    url = f"https://huggingface.co/api/datasets/{repo_id_quoted}/tree/{revision_quoted}?recursive=1"
    try:
        with urlopen(url) as response:
            entries = json.load(response)
    except (HTTPError, URLError, json.JSONDecodeError) as exc:
        raise HFDatasetError(f"Failed to list Hugging Face dataset files from {repo_id}@{revision}: {exc}") from exc

    return [entry["path"] for entry in entries if entry.get("type") == "file" and "path" in entry]


def _download_file(repo_id, revision, remote_path, local_path):
    if os.path.exists(local_path):
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    repo_id_quoted = _normalize_repo_id(repo_id)
    revision_quoted = quote(revision, safe="")
    remote_path_quoted = quote(remote_path, safe="/")
    url = f"https://huggingface.co/datasets/{repo_id_quoted}/resolve/{revision_quoted}/{remote_path_quoted}"

    try:
        with urlopen(url) as response, open(local_path, "wb") as output_file:
            shutil.copyfileobj(response, output_file)
    except (HTTPError, URLError, OSError) as exc:
        raise HFDatasetError(
            f"Failed to download Hugging Face dataset file {remote_path} from {repo_id}@{revision}: {exc}"
        ) from exc

    return local_path


def _pick_file(repo_files, prefix, preferred_suffixes):
    for suffix in preferred_suffixes:
        candidates = sorted(
            path for path in repo_files if path.startswith(prefix) and os.path.basename(path) == suffix
        )
        if candidates:
            return candidates[0]
    return None


def prepare_hf_dataset_paths(dg_cache_path, repo_name, model_name, hf_repo_id, revision="main", split_path=None):
    if not hf_repo_id:
        return {}

    if not repo_name:
        raise HFDatasetError("repo_name is required when -hf_repo_id is set.")

    repo_files = _list_dataset_files(hf_repo_id, revision=revision)
    local_root = os.path.join(dg_cache_path, "dataset", repo_name, "hf", hf_repo_id.replace("/", "__"), revision)

    selection_prefix = f"{split_path.rstrip('/')}/" if split_path else f"{repo_name}/"
    repo_prefix = f"{repo_name}/"

    dictionary_remote = _pick_file(repo_files, repo_prefix, [f"dict_{repo_name}.jsonl"])
    train_remote = _pick_file(
        repo_files,
        selection_prefix,
        [
            f"train_merge_{repo_name}.jsonl",
            f"out_train_merge_{repo_name}.jsonl",
            f"train_patch_{repo_name}.jsonl",
            f"out_train_patch_{repo_name}.jsonl",
            f"train_Kamei_features_{repo_name}.jsonl",
            f"out_train_Kamei_features_{repo_name}.jsonl",
        ],
    )
    val_remote = _pick_file(
        repo_files,
        selection_prefix,
        [
            f"val_merge_{repo_name}.jsonl",
            f"out_val_merge_{repo_name}.jsonl",
            f"val_patch_{repo_name}.jsonl",
            f"out_val_patch_{repo_name}.jsonl",
            f"val_Kamei_features_{repo_name}.jsonl",
            f"out_val_Kamei_features_{repo_name}.jsonl",
        ],
    )
    test_remote = _pick_file(
        repo_files,
        repo_prefix,
        [
            f"test_{model_name}_{repo_name}.jsonl",
            f"out_test_{model_name}_{repo_name}.jsonl",
        ],
    )

    resolved = {}
    if dictionary_remote:
        resolved["dictionary"] = _download_file(
            hf_repo_id,
            revision,
            dictionary_remote,
            os.path.join(local_root, dictionary_remote),
        )
    if train_remote:
        resolved["train_set"] = _download_file(
            hf_repo_id,
            revision,
            train_remote,
            os.path.join(local_root, train_remote),
        )
    if val_remote:
        resolved["val_set"] = _download_file(
            hf_repo_id,
            revision,
            val_remote,
            os.path.join(local_root, val_remote),
        )
    if test_remote:
        resolved["test_set"] = _download_file(
            hf_repo_id,
            revision,
            test_remote,
            os.path.join(local_root, test_remote),
        )

    return resolved