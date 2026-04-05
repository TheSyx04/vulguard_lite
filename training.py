from .models.init_model import init_model
from .utils.utils import SRC_PATH, create_dg_cache
import json
import os
import random


def _undersample_jsonl(input_path, output_path, seed=0):
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


def _apply_undersampling_if_needed(train_df_path, params, dg_cache_path):
    if not getattr(params, "sampling", False):
        return train_df_path

    sampled_paths = []
    for path in train_df_path.split(","):
        clean_path = path.strip()
        if not clean_path:
            continue

        base_name = os.path.basename(clean_path)
        name, ext = os.path.splitext(base_name)
        sampled_dir = os.path.join(dg_cache_path, "dataset", params.repo_name, "sampled")
        os.makedirs(sampled_dir, exist_ok=True)
        sampled_path = os.path.join(sampled_dir, f"{name}_undersampled{ext}")
        sampled_paths.append(_undersample_jsonl(clean_path, sampled_path, seed=0))

    if not sampled_paths:
        return train_df_path

    return ",".join(sampled_paths)
    
def training(params):
    # create save folders
    dg_cache_path = create_dg_cache(params.dg_save_folder)
    save_path = f'{dg_cache_path}/save/{params.repo_name}'
    model = init_model(params.model, params.repo_language, params.device)
    
    default_inputs = model.default_input.split(",")
    if params.train_set:
        train_df_path = params.train_set
    else:
        train_df_path = ','.join([f'{dg_cache_path}/dataset/{params.repo_name}/data/train_{default_input}_{params.repo_name}.jsonl' for default_input in default_inputs])

    train_df_path = _apply_undersampling_if_needed(train_df_path, params, dg_cache_path)
    
    if params.val_set:
        val_df_path = params.val_set
    else:
        val_df_path = ','.join([f'{dg_cache_path}/dataset/{params.repo_name}/data/val_{default_input}_{params.repo_name}.jsonl' for default_input in default_inputs])
        
    model_path = params.model_path
    dictionary = f'{dg_cache_path}/dataset/{params.repo_name}/dict_{params.repo_name}.jsonl'  if params.dictionary is None else params.dictionary 
    hyperparameters = f"{SRC_PATH}/models/{model.model_name}/hyperparameters.json" if params.hyperparameters is None else params.hyperparameters
    
    print(f"Init model: {model.model_name}")
    model.initialize(model_path=model_path, dictionary=dictionary, hyperparameters=hyperparameters)
    
    print(f"Train {model.model_name}")
    save_best_path = f"{save_path}/models/best_epoch"
    model.train(train_df=train_df_path, val_df=val_df_path, params=params, save_path=save_best_path)
    
    print(f"Save {model.model_name}")
    save_last_path = f"{save_path}/models/last_epoch"
    model.save(save_path=save_last_path)
    print(f"Model saved to: {save_path}")
    

    