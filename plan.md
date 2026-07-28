# Plan: Validate Line-Level Code Localization for JITFine Without Retraining

## 1. Goal and Scope

Extend the existing JITFine implementation so that an already trained checkpoint
can be inspected at inference time:

1. predict whether a commit is vulnerable;
2. measure how much CodeBERT attention is allocated to the code-change tokens;
3. rank the visible changed-code records;
4. test whether masking the top-ranked records reduces the vulnerable logit more
   than masking random or bottom-ranked records.

This is an exploratory faithfulness check. Raw Transformer attention must be
reported as `attention-based ranking`, not as proof that a line caused the
prediction.

The first implementation must:

- reuse an existing `jitfine.pth`;
- perform no retraining;
- preserve the exact model input IDs and prediction path;
- use the existing manual features when calculating the commit prediction;
- avoid changing the existing manual-feature scaling behavior;
- use attention only for ranking code tokens;
- include an occlusion test to check whether that ranking is useful;
- save machine-readable results.

Fixing per-seed checkpoint storage is intentionally out of scope for this task.
The current checkpoint represents only one retained training run, so conclusions
from this experiment apply only to that checkpoint.

---

## 2. Verified OpenSSL Test-Data Structure

The OpenSSL test directory contains three relevant files:

```text
test_tlel_openssl.jsonl
test_deepjit_openssl.jsonl
test_simcom_openssl.jsonl
```

Their roles for this experiment are:

- `test_tlel_openssl.jsonl`: the 14 manual features used by JITFine;
- `test_deepjit_openssl.jsonl`: the exact message and flattened code input used
  by JITFine;
- `test_simcom_openssl.jsonl`: line/change-record metadata used only to recover
  boundaries for localization.

Do not feed `test_simcom_openssl.jsonl` directly to the current JITFine
preprocessor. Its `code_change` contains multiple physical records:

```text
<ADD> ... <REMOVE> ...
<ADD> ... <REMOVE> ...
```

The current JITFine regex does not use `DOTALL`, so it would only consume the
first physical record.

### 2.1 Full-dataset checks already performed

For the supplied OpenSSL test data:

- `test_deepjit_openssl.jsonl`: 7,340 commits;
- `test_simcom_openssl.jsonl`: 7,340 commits;
- matching `commit_id` values: 7,340;
- average SimCom change records per commit: 11.42;
- minimum records: 1;
- maximum records: 2,393.

After splitting on whitespace, every commit satisfies this verified mapping:

```text
JITFine/DeepJIT <ADD> section
    == concatenation of the SimCom text after each <REMOVE> marker

JITFine/DeepJIT <REMOVE> section
    == concatenation of the SimCom text after each <ADD> marker
       and before its <REMOVE> marker
```

This apparent marker reversal must be preserved because it reflects the data
actually consumed during training. The output must therefore store both:

```json
{
  "model_section": "ADD",
  "metadata_marker": "REMOVE"
}
```

Do not call either side a true Git added/removed line unless it is independently
validated against a raw Git diff. In the first implementation, use the neutral
term `change_record`.

### 2.2 Required runtime dataset validation

Before localization, fail with a clear error unless:

```python
deepjit_ids == simcom_ids == feature_ids
```

For every commit, also require:

```python
deepjit_add_words == concat(simcom_remove_side_words)
deepjit_remove_words == concat(simcom_add_side_words)
```

Here `words` means the ordered non-whitespace spans from the exact strings.
Validation must run on the complete selected test set, not just a few examples.

---

## 3. Existing JITFine Input That Must Remain Unchanged

The model input remains:

```text
<commit message> <ADD> <flattened model ADD section>
<REMOVE> <flattened model REMOVE section>
```

Current preprocessing:

1. tokenize the commit message and truncate it to 64 tokens;
2. tokenize the complete DeepJIT `<ADD>` section;
3. tokenize the complete DeepJIT `<REMOVE>` section;
4. concatenate message, markers, and code tokens;
5. truncate the content to 510 tokens;
6. add CodeBERT CLS/SEP tokens;
7. pad to 512.

Must remain unchanged:

- `microsoft/codebert-base`;
- the existing `RobertaTokenizer`;
- `<ADD>` and `<REMOVE>` added-token registration;
- tokenization of each complete flattened block;
- message/code ordering;
- truncation order;
- padding;
- model layer names, dimensions, and state-dict keys;
- manual features passed to the classifier.

The localization metadata is parallel data only. It must never rebuild or
replace the input text passed to the legacy model preprocessor.

---

## 4. Files to Modify

Required:

- `models/jitfine/dataset.py`;
- `models/jitfine/model.py`;
- `models/jitfine/warper.py`;
- one small localization runner under `scripts/`;
- localization/equivalence tests.

No required change:

- `models/jitfine/hyperparameters.json`;
- checkpoint-saving logic for multiple seeds;
- manual-feature scaling.

---

## 5. Recover Change-Record Boundaries Without Changing Tokens

### 5.1 Load separate model-input and localization files

Extend the localization-only dataset construction to accept:

```python
features_filename       # test_tlel_openssl.jsonl
changes_filename        # test_deepjit_openssl.jsonl
line_metadata_filename  # test_simcom_openssl.jsonl
```

Ordinary training and inference must continue to work without
`line_metadata_filename`.

Index SimCom metadata by `commit_id`. Parse every physical record with an
anchored expression equivalent to:

```python
r"^\s*<ADD>\s*(.*?)\s*<REMOVE>\s*(.*?)\s*$"
```

Empty sides are valid and must not create a rankable record for that side.

Each non-empty side gets a globally unique `line_id` and metadata:

```python
{
    "line_id": int,
    "record_index": int,
    "model_section": "ADD" | "REMOVE",
    "metadata_marker": "REMOVE" | "ADD",
    "text": str
}
```

### 5.2 Align exact DeepJIT text to SimCom records

Do not tokenize progressively longer prefixes.

Instead:

1. parse the exact flattened DeepJIT ADD and REMOVE sections;
2. find all non-whitespace word spans in each exact DeepJIT section;
3. split each mapped SimCom side into the same ordered words;
4. validate that the word sequences are identical;
5. assign every DeepJIT word span to the corresponding SimCom `line_id`;
6. tokenize the exact full DeepJIT section with an offset-capable tokenizer;
7. assign a model token to a line when its character offset overlaps a mapped
   non-whitespace word span.

Whitespace-only tokens may be assigned `-1`. They must not be forced into an
adjacent change record.

Use `RobertaTokenizerFast` only as a parallel offset mapper. The actual model
input must still be produced by the existing tokenizer and existing code path.

For every section, require:

```python
slow_tokens = legacy_tokenizer.tokenize(exact_section)
slow_ids = legacy_tokenizer.convert_tokens_to_ids(slow_tokens)

fast = offset_tokenizer(
    exact_section,
    add_special_tokens=False,
    return_offsets_mapping=True,
)

assert fast["input_ids"] == slow_ids
```

If this assertion fails, stop localization for that commit rather than silently
using a misaligned mapping.

### 5.3 Build the parallel `token_line_ids`

Align metadata with the legacy token sequence:

```python
token_line_ids = (
    [-1] * len(msg_tokens)
    + [-1]                  # <ADD>
    + model_add_line_ids
    + [-1]                  # <REMOVE>
    + model_remove_line_ids
)
```

Apply exactly the same 510-token truncation:

```python
input_tokens = input_tokens[:510]
token_line_ids = token_line_ids[:510]
```

Then add CLS/SEP and padding entries using `-1`.

Required:

```python
assert len(input_ids) == 512
assert len(input_mask) == 512
assert len(token_line_ids) == 512
```

### 5.4 Track visibility and truncation

For each change record, report:

- total mapped model-token count;
- visible model-token count;
- `fully_visible`;
- `partially_visible`;
- `fully_truncated`.

Also report coverage separately for the model ADD and REMOVE sections:

```json
{
  "was_truncated": true,
  "model_add_token_coverage": 0.91,
  "model_remove_token_coverage": 0.12
}
```

Fully truncated records must not appear with score zero. Partially visible
records may be ranked, but must be marked as partial.

---

## 6. Dataset Return Structure

Extend `InputFeatures` with localization-only fields:

```python
token_line_ids
line_metadata
was_truncated
original_content_token_count
section_coverage
```

The DataLoader may return:

```python
(
    commit_id,
    input_ids,
    input_mask,
    manual_features,
    token_line_ids,
    label,
)
```

Keep dictionaries and strings outside default PyTorch collation:

```python
metadata_by_commit = {
    example.commit_id: example.line_metadata
    for example in dataset.examples
}
```

Ordinary train/inference behavior must remain backward compatible when
localization metadata is disabled.

---

## 7. Return Attention Without Changing the Model

Do not add, remove, rename, or resize trainable parameters.

Call CodeBERT with:

```python
outputs = self.encoder(
    input_ids=inputs_ids,
    attention_mask=attn_masks,
    output_attentions=output_attentions,
    return_dict=True,
)
```

Classification must still use:

```python
outputs.last_hidden_state
```

For the attention baseline:

```python
last_layer = outputs.attentions[-1]       # [B, H, 512, 512]
cls_by_head = last_layer[:, :, 0, :]      # [B, H, 512]
cls_attention = cls_by_head.mean(dim=1)   # [B, 512]
```

Return a stable dictionary:

```python
{
    "loss": loss,
    "probability": probability,
    "logit": logits,
    "code_attention": cls_attention,
    "code_attention_by_head": cls_by_head,  # optional diagnostic
}
```

Training and ordinary inference must use `output_attentions=False`.
Localization uses `output_attentions=True`.

---

## 8. Attention Measurements and Line Ranking

Before line aggregation, report total last-layer CLS attention mass assigned to:

- commit-message tokens;
- model ADD tokens;
- model REMOVE tokens;
- markers and special tokens;
- padding.

This is necessary to answer whether the model is looking at code at all.

For each visible change record, save:

- `sum_score`;
- `mean_score`;
- `max_score`;
- number of visible tokens.

Default ranking may use `mean_score` to reduce line-length bias, but all three
scores must be saved so the choice can be inspected later.

Normalize code-token scores for within-commit ranking, but also preserve raw
attention mass. Do not compare normalized line scores across different commits
as if they were absolute importance values.

Default localization filter:

```python
only_predicted_vulnerable = True
```

Use the same threshold and full JITFine prediction path, including the existing
manual features. Attention ranking itself is computed only from CodeBERT code
tokens; no claim is made that it explains the manual-feature branch.

---

## 9. Required Faithfulness Check

Raw attention ranking alone is not enough to decide that localization is
effective. Add an occlusion evaluation that requires no retraining.

For each selected predicted-vulnerable commit:

1. record the original pre-sigmoid vulnerable logit;
2. replace token IDs belonging to the top-1 ranked record with the CodeBERT mask
   token ID, preserving sequence length and attention mask;
3. rerun the complete model with the same message and manual features;
4. calculate the vulnerable-logit drop;
5. repeat for:
   - top-k ranked records;
   - bottom-ranked records;
   - randomly selected visible records of the same count.

Primary measurements:

```text
top_1_logit_drop
top_k_logit_drop
random_k_logit_drop_mean
bottom_k_logit_drop
top_k_minus_random_k
```

Use logit drop as the primary value because sigmoid saturation can hide changes.
Probability drop may also be saved for readability.

Run multiple random selections with a fixed evaluation seed and report their
mean and standard deviation.

Interpretation:

- attention is useful for this checkpoint if top-ranked occlusion produces a
  consistently larger vulnerable-logit drop than random/bottom occlusion;
- if it does not, report that raw attention is not a faithful localization
  method for this checkpoint;
- without ground-truth vulnerable-line annotations, this evaluates faithfulness
  to the model, not correctness against human-labelled vulnerable lines.

Integrated Gradients may be added later if the attention baseline fails, but it
is not required for this first check.

---

## 10. Checkpoint Loading

`jitfine.pth` is a custom PyTorch checkpoint, not a Hugging Face
`save_pretrained()` directory.

Allow initialization from either a checkpoint file or an existing directory:

```python
def initialize(
    self,
    hyperparameters,
    model_path=None,
    load_for_inference=False,
    **kwargs,
):
    ...
```

Inference-only loading:

```python
checkpoint = torch.load(
    resolved_checkpoint_file,
    map_location=self.device,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)
```

Do not require optimizer or scheduler restoration when
`load_for_inference=True`.

The current single retained checkpoint is sufficient for this exploratory
experiment. Per-seed checkpoint preservation will be fixed separately.

---

## 11. Output Format

Write one JSON object per commit to JSONL:

```json
{
  "commit_id": "abc123",
  "label": 1,
  "prediction": 1,
  "probability": 0.873,
  "logit": 1.927,
  "checkpoint": "jitfine.pth",
  "method": "last_layer_cls_attention",
  "ranking_aggregation": "mean",
  "attention_mass": {
    "message": 0.18,
    "model_add": 0.42,
    "model_remove": 0.21,
    "special": 0.19
  },
  "was_truncated": true,
  "model_add_token_coverage": 0.91,
  "model_remove_token_coverage": 0.12,
  "ranked_records": [
    {
      "rank": 1,
      "line_id": 4,
      "record_index": 4,
      "model_section": "ADD",
      "metadata_marker": "REMOVE",
      "text": "memcpy ( buffer , input , length ) ;",
      "visible_token_count": 8,
      "fully_visible": true,
      "partially_visible": false,
      "sum_score": 0.0871,
      "mean_score": 0.0109,
      "max_score": 0.0241
    }
  ],
  "faithfulness": {
    "top_1_logit_drop": 0.31,
    "top_5_logit_drop": 0.66,
    "random_5_logit_drop_mean": 0.12,
    "random_5_logit_drop_std": 0.04,
    "bottom_5_logit_drop": 0.03
  }
}
```

Optionally write a flattened CSV for analysis.

---

## 12. Tests and Acceptance Criteria

### 12.1 Data alignment

- all selected feature, DeepJIT, and SimCom commit IDs match;
- all SimCom physical records parse successfully;
- all DeepJIT ADD word sequences match the mapped SimCom REMOVE sides;
- all DeepJIT REMOVE word sequences match the mapped SimCom ADD sides.

### 12.2 Input equivalence

Compare legacy and localization preprocessing over the complete test set:

```python
assert old.input_ids == new.input_ids
assert old.input_mask == new.input_mask
```

Acceptance: 100% identical.

### 12.3 Checkpoint and prediction equivalence

- `strict=True` produces no missing or unexpected keys;
- ordinary probabilities before and after the code change differ by less than
  `1e-5` on the same complete test cohort.

### 12.4 Mapping

- `token_line_ids` has length 512;
- message, marker, special, padding, and whitespace-only tokens use `-1`;
- every ranked record has at least one visible non-whitespace model token;
- no fully truncated record is ranked;
- partial records are explicitly marked;
- fast offset-mapper IDs exactly equal legacy block token IDs.

### 12.5 Attention

- attention shape is `[batch_size, 512]`;
- reported attention-mass categories sum to approximately 1 over non-padding
  key positions;
- line scores are sorted correctly;
- each `(line_id, model_section)` appears at most once.

### 12.6 Faithfulness

On the selected predicted-vulnerable test commits:

- occlusion preserves sequence length and attention masks;
- top, random, and bottom comparisons use the same number of records;
- random baselines use a fixed evaluation seed;
- aggregate top-vs-random logit-drop statistics are reported;
- the final report does not claim success unless top-ranked occlusion
  consistently outperforms the baselines.

---

## 13. Deliverables

1. updated JITFine dataset, model, and wrapper code;
2. a localization runner accepting feature, DeepJIT, SimCom, and checkpoint
   paths;
3. full-test data-alignment report;
4. input- and prediction-equivalence tests;
5. attention-mass and change-record rankings;
6. top/random/bottom occlusion results;
7. sample JSONL output;
8. brief usage instructions;
9. a clear conclusion limited to the tested checkpoint.

---

## 14. Definition of Done

The experiment is complete when:

- the existing checkpoint loads without retraining;
- legacy model inputs remain 100% identical;
- ordinary predictions remain equivalent;
- all 7,340 OpenSSL test commits align across the three required files;
- visible DeepJIT tokens map back to SimCom change records;
- truncation and section coverage are reported;
- code/message attention mass is reported;
- visible change records are ranked;
- top-ranked occlusion is compared with random and bottom baselines;
- results are saved as JSONL;
- conclusions distinguish attention visualization, model faithfulness, and
  ground-truth vulnerability localization.
