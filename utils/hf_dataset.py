import json
import os
import re
import shutil
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


class HFDatasetError(RuntimeError):
    pass


def _normalize_repo_id(repo_id):
    return quote(repo_id, safe="/")


def _parse_next_link(link_header):
    """Extract the 'next' URL from an RFC 5988 Link header, if present."""
    if not link_header:
        return None
    for part in link_header.split(","):
        part = part.strip()
        match = re.match(r'<([^>]+)>\s*;.*\brel=["\']?next["\']?', part)
        if match:
            return match.group(1)
    return None


@lru_cache(maxsize=16)
def _list_dataset_files(repo_id, revision="main"):
    repo_id_quoted = _normalize_repo_id(repo_id)
    revision_quoted = quote(revision, safe="")
    url = f"https://huggingface.co/api/datasets/{repo_id_quoted}/tree/{revision_quoted}?recursive=1"

    all_entries = []
    while url:
        try:
            with urlopen(url) as response:
                entries = json.load(response)
                link_header = response.headers.get("Link", "")
        except (HTTPError, URLError, json.JSONDecodeError) as exc:
            raise HFDatasetError(
                f"Failed to list Hugging Face dataset files from {repo_id}@{revision}: {exc}"
            ) from exc
        all_entries.extend(entries)
        url = _parse_next_link(link_header)

    return [entry["path"] for entry in all_entries if entry.get("type") == "file" and "path" in entry]



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


def _download_files(repo_id, revision, remote_paths, local_root):
    if not remote_paths:
        raise HFDatasetError("No Hugging Face dataset files were selected for download.")

    local_paths = []
    for remote_path in remote_paths:
        local_paths.append(
            _download_file(
                repo_id,
                revision,
                remote_path,
                os.path.join(local_root, remote_path),
            )
        )

    if len(local_paths) == 1:
        return local_paths[0]

    return ",".join(local_paths)


def _download_hub_file(repo_id, revision, remote_path, local_root):
    """Download one dataset-repository artifact using the logged-in HF client."""
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=remote_path,
            local_dir=local_root,
        )
    except Exception as exc:
        raise HFDatasetError(
            f"Failed to download Hugging Face artifact {remote_path} "
            f"from {repo_id}@{revision}: {exc}"
        ) from exc


def prepare_hf_ground_truth_ranking_paths(
    cache_root,
    repo_name,
    model_name,
    hf_repo_id,
    revision="main",
    ground_truth_path=None,
    checkpoint_path=None,
    dictionary_path=None,
    features_path=None,
):
    """Resolve remote ground-truth inputs and model artifacts into an HF cache.

    ``ground_truth_path`` and ``checkpoint_path`` are paths inside a Hugging
    Face *dataset* repository. The returned values are local cache paths used
    only internally by the existing PyTorch/data-loading code.
    """
    if not hf_repo_id:
        raise HFDatasetError("-hf_repo_id is required for Hugging Face ranking inputs.")
    if not repo_name:
        raise HFDatasetError("-repo_name is required for Hugging Face ranking inputs.")
    if not checkpoint_path:
        raise HFDatasetError(
            "-hf_checkpoint_path is required because model_config contains "
            "multiple folds and seeds."
        )

    ground_truth_path = (
        ground_truth_path or f"dataset/ground_truth_hunks/{repo_name}"
    ).strip("/")
    required_ground_truth = [
        f"{ground_truth_path}/ground_truth_hunks.jsonl",
        f"{ground_truth_path}/inputs/line_provenance.jsonl",
    ]
    if model_name == "jitfine":
        required_ground_truth.append(f"{ground_truth_path}/inputs/merge.jsonl")
    checkpoint_names = {
        "deepjit": "deepjit.pth",
        "jitfine": "jitfine.pth",
        "simcom": "com.pth",
    }[model_name]
    checkpoint_path = checkpoint_path.strip("/")
    checkpoint_is_file = os.path.splitext(checkpoint_path)[1].lower() in {
        ".pth", ".pkl", ".bin", ".pt",
    }
    checkpoint_remote = (
        checkpoint_path
        if checkpoint_is_file
        else f"{checkpoint_path}/{checkpoint_names}"
    )

    cache_root = os.path.abspath(os.path.join(
        cache_root,
        "hf_ground_truth",
        hf_repo_id.replace("/", "__"),
        revision,
    ))
    for remote_path in required_ground_truth:
        _download_hub_file(hf_repo_id, revision, remote_path, cache_root)
    checkpoint_local = _download_hub_file(
        hf_repo_id, revision, checkpoint_remote, cache_root,
    )

    resolved = {
        "prepared_dir": os.path.join(cache_root, ground_truth_path),
        "model_path": checkpoint_local,
        "remote_ground_truth_path": ground_truth_path,
        "remote_checkpoint_path": checkpoint_remote,
    }
    if model_name in {"deepjit", "simcom"}:
        dictionary_remote = (
            dictionary_path or f"dataset/{repo_name}/dict_{repo_name}.jsonl"
        ).strip("/")
        resolved["dictionary"] = _download_hub_file(
            hf_repo_id, revision, dictionary_remote, cache_root,
        )
        resolved["remote_dictionary_path"] = dictionary_remote

    if model_name == "jitfine":
        if features_path:
            features_remote = features_path.strip("/")
        else:
            repo_files = _list_dataset_files(hf_repo_id, revision=revision)
            dataset_prefix = f"dataset/{repo_name}/"
            features_remote = _pick_file(
                repo_files,
                dataset_prefix,
                [
                    f"test_tlel_{repo_name}.jsonl",
                    f"out_test_tlel_{repo_name}.jsonl",
                    f"test_Kamei_features_{repo_name}.jsonl",
                    f"out_test_Kamei_features_{repo_name}.jsonl",
                ],
            )
            if not features_remote:
                raise HFDatasetError(
                    "Could not resolve the JITFine test Kamei-features file under "
                    f"{dataset_prefix}. Pass -hf_features_path explicitly."
                )
        resolved["features"] = _download_hub_file(
            hf_repo_id, revision, features_remote, cache_root,
        )
        resolved["remote_features_path"] = features_remote

    # A SimCom directory may also contain sim.pkl. Preserve full SimCom
    # inference when it exists; otherwise the downloaded com.pth runs Com-only.
    if model_name == "simcom" and not checkpoint_is_file:
        sim_remote = checkpoint_remote[:-len("com.pth")] + "sim.pkl"
        _download_hub_file(hf_repo_id, revision, sim_remote, cache_root)
        resolved["model_path"] = os.path.dirname(checkpoint_local)
        resolved["remote_sim_checkpoint_path"] = sim_remote

    return resolved


def prepare_hf_dataset_paths(dg_cache_path, repo_name, model_name, hf_repo_id, revision="main", split_path=None):
    if not hf_repo_id:
        return {}

    if not repo_name:
        raise HFDatasetError("repo_name is required when -hf_repo_id is set.")

    repo_files = _list_dataset_files(hf_repo_id, revision=revision)
    local_root = os.path.join(dg_cache_path, "dataset", repo_name, "hf", hf_repo_id.replace("/", "__"), revision)

    selection_prefix = f"{split_path.rstrip('/')}/" if split_path else f"{repo_name}/"

    # Derive repo_prefix (used for test / dictionary files) from the split_path
    # when available so that nested paths like "dataset/openssl/openssl_3_1" are
    # handled correctly (repo_prefix = "dataset/openssl/").
    if split_path:
        _parts = split_path.rstrip("/").split("/")
        if repo_name in _parts:
            _idx = _parts.index(repo_name)
            repo_prefix = "/".join(_parts[: _idx + 1]) + "/"
        else:
            repo_prefix = f"{repo_name}/"
    else:
        repo_prefix = f"{repo_name}/"

    NO_DICT_MODELS = {"tlel", "lapredict", "lr", "jitfine"}
    dictionary_remote = None if model_name in NO_DICT_MODELS else _pick_file(repo_files, repo_prefix, [f"dict_{repo_name}.jsonl"])

    if model_name == "simcom":
        train_remote = [
            _pick_file(
                repo_files,
                selection_prefix,
                [f"out_train_Kamei_features_{repo_name}.jsonl", f"train_Kamei_features_{repo_name}.jsonl"],
            ),
            _pick_file(
                repo_files,
                selection_prefix,
                [f"out_train_patch_{repo_name}.jsonl", f"train_patch_{repo_name}.jsonl"],
            ),
        ]
        val_remote = [
            _pick_file(
                repo_files,
                selection_prefix,
                [f"out_val_Kamei_features_{repo_name}.jsonl", f"val_Kamei_features_{repo_name}.jsonl"],
            ),
            _pick_file(
                repo_files,
                selection_prefix,
                [f"out_val_patch_{repo_name}.jsonl", f"val_patch_{repo_name}.jsonl"],
            ),
        ]
        test_remote = [
            _pick_file(
                repo_files,
                repo_prefix,
                [f"out_test_tlel_{repo_name}.jsonl", f"test_tlel_{repo_name}.jsonl"],
            ),
            _pick_file(
                repo_files,
                repo_prefix,
                [f"out_test_simcom_{repo_name}.jsonl", f"test_simcom_{repo_name}.jsonl"],
            ),
        ]
        if not all(train_remote):
            raise HFDatasetError(
                f"Could not resolve both simcom training files for {repo_name} under {selection_prefix}."
            )
        if not all(val_remote):
            raise HFDatasetError(
                f"Could not resolve both simcom validation files for {repo_name} under {selection_prefix}."
            )
        if not all(test_remote):
            raise HFDatasetError(
                f"Could not resolve both simcom test files for {repo_name} under {repo_prefix}."
            )
    elif model_name in {"tlel", "lapredict", "lr"}:
        # All three use only the Kamei_features file.
        train_remote = _pick_file(
            repo_files,
            selection_prefix,
            [f"out_train_Kamei_features_{repo_name}.jsonl", f"train_Kamei_features_{repo_name}.jsonl"],
        )
        val_remote = _pick_file(
            repo_files,
            selection_prefix,
            [f"out_val_Kamei_features_{repo_name}.jsonl", f"val_Kamei_features_{repo_name}.jsonl"],
        )
        test_remote = _pick_file(
            repo_files,
            repo_prefix,
            [f"out_test_tlel_{repo_name}.jsonl", f"test_tlel_{repo_name}.jsonl"],
        )
        if not train_remote or not val_remote or not test_remote:
            raise HFDatasetError(
                f"Could not resolve one or more {model_name} dataset files for {repo_name} using {selection_prefix}."
            )
    elif model_name == "jitfine":
        # JITFine uses Kamei_features (like tlel) + merge (like deepjit).
        train_remote = [
            _pick_file(
                repo_files,
                selection_prefix,
                [f"out_train_Kamei_features_{repo_name}.jsonl", f"train_Kamei_features_{repo_name}.jsonl"],
            ),
            _pick_file(
                repo_files,
                selection_prefix,
                [f"out_train_merge_{repo_name}.jsonl", f"train_merge_{repo_name}.jsonl"],
            ),
        ]
        val_remote = [
            _pick_file(
                repo_files,
                selection_prefix,
                [f"out_val_Kamei_features_{repo_name}.jsonl", f"val_Kamei_features_{repo_name}.jsonl"],
            ),
            _pick_file(
                repo_files,
                selection_prefix,
                [f"out_val_merge_{repo_name}.jsonl", f"val_merge_{repo_name}.jsonl"],
            ),
        ]
        test_remote = [
            _pick_file(
                repo_files,
                repo_prefix,
                [f"out_test_tlel_{repo_name}.jsonl", f"test_tlel_{repo_name}.jsonl"],
            ),
            _pick_file(
                repo_files,
                repo_prefix,
                [f"out_test_jitfine_{repo_name}.jsonl", f"test_jitfine_{repo_name}.jsonl",
                 f"out_test_deepjit_{repo_name}.jsonl", f"test_deepjit_{repo_name}.jsonl",
                 f"out_test_merge_{repo_name}.jsonl", f"test_merge_{repo_name}.jsonl"],
            ),
        ]
        if not all(train_remote):
            raise HFDatasetError(
                f"Could not resolve both jitfine training files (Kamei_features + merge) for {repo_name} under {selection_prefix}."
            )
        if not all(val_remote):
            raise HFDatasetError(
                f"Could not resolve both jitfine validation files (Kamei_features + merge) for {repo_name} under {selection_prefix}."
            )
        if not all(test_remote):
            raise HFDatasetError(
                f"Could not resolve both jitfine test files (Kamei_features + merge) for {repo_name} under {repo_prefix}."
            )
    else:
        # Generic fallback (e.g. deepjit): single file resolved by preferred suffix order.
        train_remote = _pick_file(
            repo_files,
            selection_prefix,
            [
                f"out_train_merge_{repo_name}.jsonl",
                f"train_merge_{repo_name}.jsonl",
                f"out_train_patch_{repo_name}.jsonl",
                f"train_patch_{repo_name}.jsonl",
                f"out_train_Kamei_features_{repo_name}.jsonl",
                f"train_Kamei_features_{repo_name}.jsonl",
            ],
        )
        val_remote = _pick_file(
            repo_files,
            selection_prefix,
            [
                f"out_val_merge_{repo_name}.jsonl",
                f"val_merge_{repo_name}.jsonl",
                f"out_val_patch_{repo_name}.jsonl",
                f"val_patch_{repo_name}.jsonl",
                f"out_val_Kamei_features_{repo_name}.jsonl",
                f"val_Kamei_features_{repo_name}.jsonl",
            ],
        )
        test_remote = _pick_file(
            repo_files,
            repo_prefix,
            [
                f"out_test_{model_name}_{repo_name}.jsonl",
                f"test_{model_name}_{repo_name}.jsonl",
            ],
        )
        if not train_remote or not val_remote or not test_remote:
            raise HFDatasetError(
                f"Could not resolve one or more {model_name} dataset files for {repo_name}."
            )

    if model_name not in NO_DICT_MODELS and not dictionary_remote:
        raise HFDatasetError(f"Could not resolve dictionary file dict_{repo_name}.jsonl for {repo_name}.")

    resolved = {}
    if dictionary_remote:
        resolved["dictionary"] = _download_files(hf_repo_id, revision, [dictionary_remote], local_root)
    if train_remote:
        remote_paths = train_remote if isinstance(train_remote, list) else [train_remote]
        remote_paths = [path for path in remote_paths if path]
        resolved["train_set"] = _download_files(hf_repo_id, revision, remote_paths, local_root)
    if val_remote:
        remote_paths = val_remote if isinstance(val_remote, list) else [val_remote]
        remote_paths = [path for path in remote_paths if path]
        resolved["val_set"] = _download_files(hf_repo_id, revision, remote_paths, local_root)
    if test_remote:
        remote_paths = test_remote if isinstance(test_remote, list) else [test_remote]
        remote_paths = [path for path in remote_paths if path]
        resolved["test_set"] = _download_files(hf_repo_id, revision, remote_paths, local_root)

    return resolved
