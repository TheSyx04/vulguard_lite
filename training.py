from .models.init_model import init_model
from .utils.utils import SRC_PATH, create_dg_cache
from .utils.hf_dataset import prepare_hf_dataset_paths
import json
import os
import random


def _undersample_jsonl(input_path, output_path, seed=None):
    if seed is not None:
        random.seed(seed)
    class_0, class_1 = [], []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            label = obj.get("label")
            if int(label) == 0:
                class_0.append(obj)
            else:
                class_1.append(obj)

    print(f"Before sampling ({os.path.basename(input_path)}):")
    print(f"Label 0: {len(class_0)}")
    print(f"Label 1: {len(class_1)}")

    minority_size = min(len(class_0), len(class_1))
    if minority_size == 0:
        print("Skip sampling because one class is empty.")
        return input_path

    class_0_sampled = random.sample(class_0, minority_size)
    class_1_sampled = random.sample(class_1, minority_size)
    balanced_data = class_0_sampled + class_1_sampled
    random.shuffle(balanced_data)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in balanced_data:
            f.write(json.dumps(item) + "\n")

    print(f"After sampling ({os.path.basename(output_path)}):")
    print(f"Total samples: {len(balanced_data)}")
    print(f"Saved to: {output_path}")
    return output_path


def _sample_commit_ids(reference_path, seed=None):
    """Pick a balanced set of commit_ids from reference_path.
    Returns the sampled set, or None if one class is empty (skip sampling).
    """
    if seed is not None:
        random.seed(seed)
    class_0_ids, class_1_ids = [], []
    with open(reference_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if int(obj.get("label", 0)) == 0:
                class_0_ids.append(obj["commit_id"])
            else:
                class_1_ids.append(obj["commit_id"])

    minority_size = min(len(class_0_ids), len(class_1_ids))
    if minority_size == 0:
        return None

    sampled = set(random.sample(class_0_ids, minority_size)) | set(random.sample(class_1_ids, minority_size))
    print(f"Paired sampling reference ({os.path.basename(reference_path)}): "
          f"class_0={len(class_0_ids)}, class_1={len(class_1_ids)} → keeping {len(sampled)} commits")
    return sampled


def _filter_jsonl_by_ids(input_path, output_path, commit_ids):
    """Write only lines whose commit_id is in commit_ids, preserving file order."""
    kept = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            if obj.get("commit_id") in commit_ids:
                fout.write(json.dumps(obj) + "\n")
                kept += 1
    print(f"Paired filter ({os.path.basename(input_path)}) → {kept} rows kept → {output_path}")
    return output_path


def _apply_undersampling_if_needed(train_df_path, params, dg_cache_path):
    if not getattr(params, "sampling", False):
        return train_df_path

    sampling_seed = getattr(params, "sampling_seed", None)
    sampling_run_id = getattr(params, "sampling_run_id", None)
    run_suffix = f"_run_{sampling_run_id}" if sampling_run_id is not None else ""

    paths = [p.strip() for p in train_df_path.split(",") if p.strip()]
    if not paths:
        return train_df_path

    sampled_dir = os.path.join(dg_cache_path, "dataset", params.repo_name, "sampled")
    os.makedirs(sampled_dir, exist_ok=True)

    def _sampled_path(clean_path):
        base_name = os.path.basename(clean_path)
        name, ext = os.path.splitext(base_name)
        return os.path.join(sampled_dir, f"{name}_undersampled{run_suffix}{ext}")

    if len(paths) == 1:
        # Single-file model (tlel, lapredict, lr, deepjit …): normal undersampling.
        result = _undersample_jsonl(paths[0], _sampled_path(paths[0]), seed=sampling_seed)
        return result

    # Multi-file model (JITFine, SimCom …): sample commit_ids ONCE from the last
    # file (the code/merge file that TextDataset uses as primary source), then
    # filter ALL files to exactly that set — so every paired file stays consistent.
    commit_ids = _sample_commit_ids(paths[-1], seed=sampling_seed)
    if commit_ids is None:
        print("Skip paired sampling because one class is empty.")
        return train_df_path

    sampled_paths = [
        _filter_jsonl_by_ids(p, _sampled_path(p), commit_ids)
        for p in paths
    ]
    return ",".join(sampled_paths)

    
def training(params):
    # create save folders
    dg_cache_path = create_dg_cache(params.dg_save_folder)
    save_path = f'{dg_cache_path}/save/{params.repo_name}'
    checkpoint_dir = params.checkpoint_dir if getattr(params, "checkpoint_dir", None) else f"{save_path}/models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = init_model(params.model, params.repo_language, params.device)

    hf_paths = prepare_hf_dataset_paths(
        dg_cache_path,
        params.repo_name,
        model.model_name,
        getattr(params, "hf_repo_id", None),
        revision=getattr(params, "hf_revision", "main"),
        split_path=getattr(params, "hf_split_path", None),
    )

    NO_DICT_MODELS = {"tlel", "lapredict", "lr", "jitfine"}
    if model.model_name in NO_DICT_MODELS:
        hf_paths["dictionary"] = None
    
    default_inputs = model.default_input.split(",")
    if params.train_set:
        train_df_path = params.train_set
    elif hf_paths.get("train_set"):
        train_df_path = hf_paths["train_set"]
    else:
        train_df_path = ','.join([f'{dg_cache_path}/dataset/{params.repo_name}/data/train_{default_input}_{params.repo_name}.jsonl' for default_input in default_inputs])

    train_df_path = _apply_undersampling_if_needed(train_df_path, params, dg_cache_path)
    
    if params.val_set:
        val_df_path = params.val_set
    elif hf_paths.get("val_set"):
        val_df_path = hf_paths["val_set"]
    else:
        val_df_path = ','.join([f'{dg_cache_path}/dataset/{params.repo_name}/data/val_{default_input}_{params.repo_name}.jsonl' for default_input in default_inputs])
        
    model_path = params.model_path
    if getattr(params, "resume_from_checkpoint", False):
        checkpoint_model_path = checkpoint_dir
        checkpoint_file = os.path.join(checkpoint_model_path, f"{params.model}_checkpoint_last.pth")
        if os.path.exists(checkpoint_file):
            model_path = checkpoint_model_path
            print(f"Resume enabled. Loading checkpoint: {checkpoint_file}")
        else:
            print(
                f"Resume enabled but checkpoint not found at {checkpoint_file}. "
                "Start training from scratch."
            )

    dictionary = params.dictionary if params.dictionary is not None else hf_paths.get("dictionary")
    if dictionary is None and model.model_name not in NO_DICT_MODELS:
        dictionary = f'{dg_cache_path}/dataset/{params.repo_name}/dict_{params.repo_name}.jsonl'
    hyperparameters = params.hyperparameters
    
    print(f"Init model: {model.model_name}")
    model.initialize(model_path=model_path, dictionary=dictionary, hyperparameters=hyperparameters)
    
    print(f"Train {model.model_name}")
    save_best_path = f"{save_path}/models/best_epoch"
    model.train(
        train_df=train_df_path,
        val_df=val_df_path,
        params=params,
        save_path=save_best_path,
        checkpoint_path=checkpoint_dir,
    )
    
    print(f"Save {model.model_name}")
    save_last_path = f"{save_path}/models/last_epoch"
    model.save(save_path=save_last_path)
    print(f"Model saved to: {save_path}")
    

    