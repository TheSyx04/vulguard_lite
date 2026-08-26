import os

from huggingface_hub import HfApi


def _get_token(explicit_token=None):
    if explicit_token:
        return explicit_token
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def upload_folder_to_hf_dataset(
    local_folder,
    repo_id,
    path_in_repo,
    token=None,
    commit_message=None,
    allow_patterns=None,
):
    if not repo_id:
        raise ValueError("repo_id is required for Hugging Face upload.")

    if not os.path.isdir(local_folder):
        raise FileNotFoundError(f"Local folder not found: {local_folder}")

    resolved_token = _get_token(token)
    if not resolved_token:
        raise ValueError(
            "Hugging Face token not found. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN, or pass token explicitly."
        )

    api = HfApi(token=resolved_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=local_folder,
        path_in_repo=path_in_repo,
        allow_patterns=allow_patterns,
        commit_message=commit_message or f"Upload {os.path.basename(local_folder)}",
    )


def list_hf_dataset_files(repo_id, token=None):
    """Return all paths currently stored in a Hugging Face dataset repo."""
    if not repo_id:
        raise ValueError("repo_id is required for Hugging Face upload.")

    resolved_token = _get_token(token)
    if not resolved_token:
        raise ValueError(
            "Hugging Face token not found. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN, or pass token explicitly."
        )

    api = HfApi(token=resolved_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    return set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
