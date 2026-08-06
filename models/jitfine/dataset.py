import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
from tqdm import tqdm
import pandas as pd
import random, json, re

manual_features_columns = ['la', 'ld', 'nf', 'ns', 'nd', 'entropy', 'ndev',
                           'lt', 'nuc', 'age', 'exp', 'rexp', 'sexp', 'fix']


def convert_dtype_dataframe(df, feature_name):
    df = df.astype({i: 'float32' for i in feature_name})
    return df

def preprocess_code_line(code, remove_python_common_tokens=False):
    code = code.replace('(', ' ').replace(')', ' ').replace('{', ' ').replace('}', ' ').replace('[', ' ').replace(']',
                                                                                                                  ' ').replace(
        '.', ' ').replace(':', ' ').replace(';', ' ').replace(',', ' ').replace(' _ ', '_')

    code = re.sub('``.*``', '<STR>', code)
    code = re.sub("'.*'", '<STR>', code)
    code = re.sub('".*"', '<STR>', code)
    code = re.sub('\d+', '<NUM>', code)

    code = code.split()
    code = ' '.join(code)
    if remove_python_common_tokens:
        new_code = ''
        python_common_tokens = []
        for tok in code.split():
            if tok not in [python_common_tokens]:
                new_code = new_code + tok + ' '

        return new_code.strip()

    else:
        return code.strip()

def convert_examples_to_features(
        item,
        pad_token=0,
        mask_padding_with_zero=True,
        return_metadata=False):
    # source
    commit_id, files, msg, label, tokenizer, hyperparameters, manual_features = item
    added_tokens = []
    removed_tokens = []
    msg = msg.encode('utf-8', 'ignore').decode('utf-8')
    msg_tokens = tokenizer.tokenize(msg)
    msg_tokens = msg_tokens[:min(hyperparameters["max_msg_length"], len(msg_tokens))]

    # Keep this expression compatible with the historical preprocessing path.
    # Attribution mode validates malformed records so a batch can skip them
    # explicitly instead of failing later with an unbound local variable.
    match = re.match(r"<ADD>(.*) <REMOVE>(.*)", files)

    if not match:
        raise ValueError("missing_or_malformed_add_remove_markers")

    added_part = match.group(1).encode('utf-8', 'ignore').decode('utf-8')
    removed_part = match.group(2).encode('utf-8', 'ignore').decode('utf-8')

    added_tokens.extend(tokenizer.tokenize(added_part))
    removed_tokens.extend(tokenizer.tokenize(removed_part))
    content_tokens = msg_tokens + ['<ADD>'] + added_tokens + ['<REMOVE>'] + removed_tokens
    content_regions = (
        ['message'] * len(msg_tokens)
        + ['add_marker']
        + ['added'] * len(added_tokens)
        + ['remove_marker']
        + ['removed'] * len(removed_tokens)
    )
    max_content_length = 512 - 2
    truncated_content_tokens = max(0, len(content_tokens) - max_content_length)
    retained_content_tokens = content_tokens[:max_content_length]
    retained_content_regions = content_regions[:max_content_length]
    omitted_regions = content_regions[max_content_length:]

    input_tokens = [tokenizer.cls_token] + retained_content_tokens + [tokenizer.sep_token]
    sequence_regions = ['cls'] + retained_content_regions + ['sep']
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = 512 - len(input_ids)

    input_ids = input_ids + ([pad_token] * padding_length)
    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    sequence_regions = sequence_regions + (['padding'] * padding_length)
    assert len(input_ids) == 512
    assert len(input_mask) == 512
    assert len(sequence_regions) == 512

    attribution_metadata = None
    if return_metadata:
        attribution_metadata = {
            "tokens_before_truncation": content_tokens,
            "tokens_after_truncation": input_tokens,
            "sequence_regions": sequence_regions,
            "add_marker_position": next(
                (index for index, region in enumerate(sequence_regions) if region == 'add_marker'),
                None,
            ),
            "remove_marker_position": next(
                (index for index, region in enumerate(sequence_regions) if region == 'remove_marker'),
                None,
            ),
            "pre_truncation_content_tokens": len(content_tokens),
            "truncated_content_tokens": truncated_content_tokens,
            "truncated_added_tokens": omitted_regions.count('added'),
            "truncated_removed_tokens": omitted_regions.count('removed'),
            "truncated": truncated_content_tokens > 0,
        }

    return InputFeatures(commit_id=commit_id,
                         input_ids=input_ids,
                         input_mask=input_mask,
                         input_tokens=input_tokens,
                         manual_features=manual_features,
                         label=label,
                         attribution_metadata=attribution_metadata)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, commit_id, input_ids, input_mask, input_tokens, label,
                 manual_features, attribution_metadata=None):
        self.commit_id = commit_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_tokens = input_tokens
        self.label = label
        self.manual_features = manual_features
        self.attribution_metadata = attribution_metadata


class TextDataset(Dataset):
    def __init__(self, tokenizer, hyperparameters, changes_filename=None,
                 features_filename=None, mode='train', return_metadata=False,
                 skip_invalid=False):
        self.examples = []
        self.failures = []

        data = []
        commit_ids, labels, msgs, codes = [], [], [], []
        with open(changes_filename, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    data_point = json.loads(line)
                    commit_ids.append(data_point["commit_id"])
                    labels.append(data_point["label"] if "label" in data_point else None)
                    msgs.append(data_point["messages"])
                    codes.append(data_point["code_change"])
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    if not skip_invalid:
                        raise
                    self.failures.append({
                        "commit_id": f"<line:{line_number}>",
                        "reason": f"invalid_code_record:{exc}",
                    })
        
        features_data = pd.read_json(features_filename, lines=True)
        features_data = convert_dtype_dataframe(features_data, manual_features_columns)

        features_data = features_data[['commit_id'] + manual_features_columns]

        manual_features = preprocessing.scale(features_data[manual_features_columns].to_numpy())
        features_data[manual_features_columns] = manual_features

        for commit_id, label, msg, files in zip(commit_ids, labels, msgs, codes):
            matching_features = features_data[features_data['commit_id'] == commit_id].head(1)
            if matching_features.empty:
                if skip_invalid:
                    self.failures.append({
                        "commit_id": commit_id,
                        "reason": "missing_manual_features",
                    })
                    continue
                raise ValueError(f"missing_manual_features:{commit_id}")
            manual_features = matching_features[manual_features_columns].to_numpy().squeeze()
            data.append((commit_id, files, msg, label, tokenizer, hyperparameters, manual_features))
        # only use 20% valid data to keep best model
        # convert example to input features
        if mode == 'train':
            random.shuffle(data)

        for item in tqdm(data, total=len(data)):
            try:
                self.examples.append(convert_examples_to_features(
                    item,
                    return_metadata=return_metadata,
                ))
            except Exception as exc:
                if not skip_invalid:
                    raise
                self.failures.append({
                    "commit_id": item[0],
                    "reason": str(exc),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (
            self.examples[item].commit_id,
            torch.tensor(self.examples[item].input_ids),
            torch.tensor(self.examples[item].input_mask),
            torch.tensor(self.examples[item].manual_features),
            torch.tensor(self.examples[item].label) if self.examples[item].label is not None else None
        )
