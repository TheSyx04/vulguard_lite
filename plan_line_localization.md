# Implementation Plan: Provenance-Preserving Attribution and Changed-Line Ranking

## 1. Status and Scope Decision

This document intentionally separates the work into two stages.

### COMPLETED BASELINE

The inference-only JITFine token-attention baseline and hierarchical row
Grad-CAM baseline for DeepJIT and Com are implemented.

The current datasets do not preserve reliable line-level provenance such as file path, hunk, old/new line number, or a stable mapping from every model token back to an original changed line. Therefore, the MVP must not claim to localize important source lines.

The MVP output is a ranked list of observed code-change tokens and their attention scores. It must use the existing test data and existing trained checkpoints without retraining.

### IMPLEMENTED / LINE-RANKING EXTENSION

Build a provenance-preserving preparation path from a local Git repository and
an explicit commit SHA/URL list. It must retain canonical changed-line identity
while separately reproducing the historical model serialization used by the
existing checkpoints.

- **JITFine:** aggregate observed token attention by verified changed line.
- **DeepJIT:** compute within-row token Grad-CAM on the flattened merge input,
  then aggregate observed token scores by verified changed line.
- **SimCom/Com:** compute token-stage Grad-CAM inside each observed historical
  patch row, then aggregate those positions by verified changed line.
- Export file, hunk, old/new line number, coverage, truncation and alignment
  status for every ranked line.

HTML heatmaps and intervention-based faithfulness remain deferred. A line must
never be ranked unless its reconstructed model input passes exact alignment
validation against the supplied dataset record.

---

## 2. Objective

### 2.1 MVP objective

For each selected test commit:

1. load the existing JITFine checkpoint;
2. reproduce the exact normal inference input;
3. request transformer attention without changing the prediction;
4. isolate tokens located between the existing `<ADD>` and `<REMOVE>` boundaries;
5. exclude message, special, separator, and padding tokens;
6. compute a token-level attention score;
7. rank the observed code-change tokens within the commit;
8. export reproducible machine-readable results.

The output answers:

> Which observed code-change tokens received the highest attention from the JITFine CLS token?

It does not answer:

> Which source line is truly vulnerable?

### 2.2 Current line-ranking objective

Generate line provenance from Git, reproduce the model input, and aggregate
only attribution positions actually observed by the checkpoint:

```text
token attention scores
    + verified token-to-line alignment
    ↓
line score aggregation
    ↓
changed-line ranking
```

Existing checkpoints remain unchanged. DeepJIT requires token-stage Grad-CAM
because its historical `merge` input flattens all changed lines into one model
row; row-stage CAM alone cannot distinguish those lines.

---

## 3. Repository-Specific Facts

The implementation must be based on the current repository rather than a generic LineVul/DeepJIT architecture.

### 3.1 JITFine input is joint message and code input

The current preprocessing constructs one transformer sequence:

```text
[CLS] message_tokens <ADD> added_tokens <REMOVE> removed_tokens [SEP]
```

JITFine does not have an independent CodeBERT code branch. The transformer CLS representation is also combined with manual commit features by the classifier.

Use this scope metadata:

```json
{
  "explanation_scope": "jitfine_code_tokens_within_joint_message_code_encoder_input",
  "full_model_explanation": false,
  "attribution_is_class_specific": false,
  "excluded_model_inputs": ["manual_features"],
  "excluded_sequence_regions": ["commit_message", "special_tokens", "padding"]
}
```

Attention ranking is an attention-based model diagnostic. It must not be described as proof of causal importance or vulnerability.

### 3.2 Current data has token-side boundaries but not verified line provenance

The existing JITFine `code_change` value is parsed using `<ADD>` and `<REMOVE>`. These markers allow the MVP to label code tokens as:

- `added`;
- `removed`.

They do not provide a reliable mapping to:

- source file;
- diff hunk;
- old/new line number;
- original changed line.

The MVP must preserve only the information that can be established from the actual model input.

### 3.3 Manual-feature preprocessing affects prediction preservation

The current JITFine dataset scales manual features using all rows in the supplied feature file. A one-row feature file would therefore change the feature values and may change the prediction.

For existing checkpoints, token attribution must preprocess the supplied full test feature file exactly as normal evaluation does, then select commits from the resulting dataset. Do not construct a one-row temporary feature dataset for single-commit localization.

Record the feature-file identity and preprocessing context in the run metadata.

### 3.4 DeepJIT and Com are hierarchical CNNs

DeepJIT and SimCom/Com use a two-stage CNN:

```text
tokens within each input row
    ↓ line-stage convolution and max pooling
row representations
    ↓ commit-stage convolution and max pooling
commit representation
```

They are not flat Conv1d code branches. Their Grad-CAM design is retained only as a future prototype in Section 15.

---

## 4. Terminology

Use these terms consistently.

- **Commit-level prediction**: the original JITFine probability and predicted label.
- **Observed token**: a non-padding token retained in the 512-token model input.
- **Code-change token**: an observed token after `<ADD>` and before/after `<REMOVE>`, classified as added or removed.
- **Token attention score**: attention assigned from CLS to a code-change token under the selected aggregation strategy.
- **Token rank**: the within-commit ordering of code-change tokens by attention score.
- **Truncated token**: a token produced before the model's 510-content-token truncation boundary but omitted from the final input.
- **Explanation scope**: code tokens within the joint message/code transformer sequence.
- **Line attribution**: a future aggregation of verified token scores by original changed line.

Avoid the following terms for MVP output:

- vulnerable token;
- vulnerable line;
- root-cause line;
- line localization result.

Preferred task name:

```text
JITFine token-level code-change attention baseline
```

---

## 5. MVP Pipeline

```text
Existing test feature file + existing test code file
    ↓
Existing JITFine preprocessing over the full test context
    ↓
Exact input_ids, attention_mask, manual_features
    ↓
Normal prediction + optional transformer attentions
    ↓
Identify message / ADD / REMOVE / special / padding regions
    ↓
CLS-to-token attention aggregation
    ↓
Filter to observed added and removed code tokens
    ↓
Within-commit token ranking
    ↓
JSON / JSONL / CSV export
```

No raw-diff parser is required for the MVP.

---

## 6. Proposed Source Structure

Adapt names to the repository's existing layout.

```text
vulguard_lite/
├── attribution/
│   ├── __init__.py
│   ├── schemas.py
│   ├── jitfine_attention.py
│   ├── token_regions.py
│   ├── ranking.py
│   ├── export.py
│   └── validation.py
├── tests/
│   ├── test_jitfine_token_regions.py
│   ├── test_jitfine_attention.py
│   ├── test_token_ranking.py
│   ├── test_prediction_preservation.py
│   └── test_token_export.py
└── cli.py
```

Do not create the full future `localization/` framework until line provenance is available. Keep the MVP small and checkpoint-compatible.

---

## 7. MVP Data Schemas

### 7.1 Token attribution

```python
from dataclasses import dataclass


@dataclass
class TokenAttribution:
    rank: int
    sequence_position: int
    token: str
    token_id: int
    change_type: str  # added or removed
    raw_score: float
    normalized_score: float
    layer_strategy: str
    observed: bool
```

`sequence_position` is the position in the exact padded model input. It is not a source-code character offset or source-line coordinate.

### 7.2 Token attribution result

```python
@dataclass
class TokenAttributionResult:
    commit_id: str
    model_name: str
    checkpoint_id: str
    prediction_score: float
    predicted_label: int
    threshold: float
    attribution_method: str
    attention_strategy: str
    explanation_scope: str
    attribution_is_class_specific: bool
    ranked_tokens: list[TokenAttribution]
    observed_sequence_tokens: int
    observed_code_tokens: int
    observed_added_tokens: int
    observed_removed_tokens: int
    pre_truncation_content_tokens: int
    truncated_content_tokens: int
    truncated: bool
    metadata: dict
```

The implementation may use typed dictionaries or Pydantic instead of dataclasses if that better matches the repository.

---

## 8. Exact Preprocessing and Token-Region Tracking

### 8.1 Core rule

Do not introduce a second tokenization pipeline for attribution.

Extend the existing `convert_examples_to_features` path with an optional, backward-compatible argument:

```python
def convert_examples_to_features(
    item,
    pad_token=0,
    mask_padding_with_zero=True,
    return_metadata=False,
):
    ...
```

Training and normal evaluation retain the existing return behavior when `return_metadata=False`.

Attribution mode returns the same tensors plus metadata derived during the same preprocessing call.

### 8.2 Required preprocessing metadata

Record before and after truncation:

```python
{
    "tokens_before_truncation": list[str],
    "tokens_after_truncation": list[str],
    "sequence_regions": list[str],
    "add_marker_position": int | None,
    "remove_marker_position": int | None,
    "pre_truncation_content_tokens": int,
    "truncated_content_tokens": int,
    "truncated": bool,
}
```

Allowed region values:

```text
cls
message
add_marker
added
remove_marker
removed
sep
padding
```

Region assignment must happen while constructing `input_tokens`, not by searching decoded text after inference.

### 8.3 Malformed input

The current regex path can leave `added_part` or `removed_part` undefined when markers are malformed. Attribution mode must validate both markers explicitly and return a structured skip reason instead of crashing the batch.

Example:

```json
{
  "commit_id": "...",
  "status": "skipped",
  "reason": "missing_or_malformed_add_remove_markers"
}
```

Do not silently change the serialization of valid existing records.

### 8.4 Truncation

The current model keeps at most 510 content tokens plus CLS and SEP. Report:

- total content-token count before truncation;
- number of omitted tokens;
- whether the omitted region contains added or removed code tokens when determinable.

Do not emit fabricated zero scores for truncated tokens. They were not observed by the model and should not appear in `ranked_tokens`.

---

## 9. JITFine Forward API

Modify the forward path without renaming or reshaping parameters.

Preferred optional structured-output API:

```python
def forward(
    self,
    inputs_ids,
    attn_masks,
    manual_features=None,
    labels=None,
    output_attentions=False,
    return_attribution_data=False,
):
    ...
```

When `return_attribution_data=True`, return:

```python
{
    "probability": probability,
    "logit": logit,
    "attentions": outputs.attentions,
    "loss": loss_or_none,
}
```

Legacy callers must continue receiving the current return types unless they opt into the new structured output.

The encoder call should explicitly request a return dictionary in attribution mode:

```python
outputs = self.encoder(
    input_ids=inputs_ids,
    attention_mask=attn_masks,
    output_attentions=output_attentions,
    return_dict=True,
)
```

Existing state dictionaries must load without missing or unexpected model parameters.

---

## 10. Attention Strategy

### 10.1 MVP default

Use last-layer CLS-to-token attention averaged across heads:

```python
# last_attention: [batch, heads, seq, seq]
token_scores = last_attention[:, :, 0, :].mean(dim=1)
```

Default configuration:

```yaml
attribution:
  method: attention
  attention_strategy: last_layer_cls_mean
  aggregate_heads: mean
```

This is closest to the attention data already exposed by the current model and is the smallest checkpoint-compatible baseline.

### 10.2 Optional comparison strategies

Implement only after the default baseline works:

- `all_layers_cls_mean`;
- `attention_rollout`.

Store the strategy in every result. Results from different strategies must not be mixed without labeling.

### 10.3 Filtering

Rank only tokens whose region is `added` or `removed`.

Exclude:

- CLS and SEP;
- padding;
- commit-message tokens;
- `<ADD>` and `<REMOVE>` markers;
- tokens outside the attention mask.

### 10.4 Interpretation limit

CLS attention is not class-specific and is not guaranteed to be a faithful causal explanation. Documentation and exports must call it an attention diagnostic or baseline, not a vulnerability probability per token.

---

## 11. Token Score Normalization and Ranking

### 11.1 Raw score

Preserve the raw attention value for every observed code token.

### 11.2 Display normalization

For within-commit visualization and comparison, use min-max normalization over observed code tokens:

```python
normalized = (score - min_score) / (max_score - min_score + 1e-12)
```

Normalization does not replace the raw score.

### 11.3 Ranking

Sort within each commit by:

1. raw score descending;
2. sequence position ascending.

This produces deterministic results.

### 11.4 Repeated subword tokens

Do not merge repeated token strings. Each token occurrence is a separate observation identified by `sequence_position`.

Subword-to-word merging is optional and deferred. The canonical MVP artifact remains token-occurrence level.

### 11.5 Degenerate cases

Handle explicitly:

- no observed code tokens;
- only added or only removed tokens;
- all equal attention scores;
- NaN or infinite scores;
- missing attention tensors;
- malformed markers.

---

## 12. CLI

Integrate `localize` or `attribute` as a subcommand in the existing `cli.py` parser. Prefer `attribute` because the MVP does not localize source lines.

Example:

```bash
python -m vulguard_lite attribute \
  -model jitfine \
  -model_path <checkpoint-directory> \
  -test_set <features.jsonl,code.jsonl> \
  -hyperparameters models/jitfine/hyperparameters.json \
  -repo_language C \
  -device cpu \
  -output_dir results/token_attention
```

Required options:

```text
-model jitfine
-model_path
-test_set
-hyperparameters
-repo_language
-device
-threshold
-output_dir
-attention_strategy
-top_k
-commit_id
-only_predicted_vulnerable
-all_commits
-overwrite
-resume
```

Rules:

- MVP accepts only `jitfine`.
- `-commit_id` selects a commit after preprocessing the full supplied test files.
- It must not create a one-row feature file.
- `-only_predicted_vulnerable` and `-all_commits` are mutually exclusive selection modes.
- A malformed commit must not stop the full batch.
- Batch output order must follow stable dataset order.

Add deterministic sharding later if required for PBS execution:

```text
-shard_index
-num_shards
```

---

## 13. MVP Output

### 13.1 Run-level files

```text
<output-dir>/
├── run_metadata.json
├── token_attributions.jsonl
├── token_attributions.csv
└── summary.json
```

Per-commit folders and line heatmaps are deferred until line provenance is available.

### 13.2 JSONL record

```json
{
  "commit_id": "abc123",
  "status": "succeeded",
  "model": "jitfine",
  "prediction_score": 0.873,
  "predicted_label": 1,
  "threshold": 0.5,
  "attribution_method": "attention",
  "attention_strategy": "last_layer_cls_mean",
  "explanation_scope": "jitfine_code_tokens_within_joint_message_code_encoder_input",
  "full_model_explanation": false,
  "attribution_is_class_specific": false,
  "observed_code_tokens": 311,
  "observed_added_tokens": 180,
  "observed_removed_tokens": 131,
  "pre_truncation_content_tokens": 724,
  "truncated_content_tokens": 214,
  "truncated": true,
  "ranked_tokens": [
    {
      "rank": 1,
      "sequence_position": 87,
      "token": "Ġbuffer",
      "token_id": 19228,
      "change_type": "added",
      "raw_score": 0.0182,
      "normalized_score": 1.0,
      "observed": true
    }
  ]
}
```

Tokenizer-native token strings should be preserved. A decoded display form may be added as a separate field; it must not replace the canonical token string.

### 13.3 CSV columns

```text
commit_id,model,prediction_score,predicted_label,attention_strategy,rank,sequence_position,token,token_id,change_type,raw_score,normalized_score,truncated
```

### 13.4 Run metadata

Record:

- checkpoint path and fingerprint;
- hyperparameter-file fingerprint;
- feature and code input paths/fingerprints;
- device;
- threshold;
- attention strategy;
- package versions;
- random seed;
- run timestamp;
- repository commit hash when available.

Use atomic writes. Resume should validate fingerprints and configuration before reusing existing results.

---

## 14. MVP Validation and Tests

### 14.1 Prediction preservation

Attention mode must preserve the normal evaluation probability:

```python
abs(normal_probability - attention_probability) < tolerance
```

Suggested tolerance:

```text
CPU: 1e-6
GPU: 1e-5
```

The comparison must use:

- the same checkpoint;
- `model.eval()`;
- identical model tensors;
- the same full-test feature preprocessing context.

### 14.2 Golden preprocessing fixtures

Create small fixtures that assert:

- exact `input_tokens`;
- exact `input_ids`;
- exact attention mask;
- exact region per sequence position;
- exact truncation boundary;
- unchanged legacy preprocessing return value.

### 14.3 Attention tests

Verify:

- attention tensor shape is `[batch, layers, heads, seq, seq]` or the documented tuple equivalent;
- last-layer CLS extraction returns `[batch, seq]`;
- head averaging is correct;
- message, markers, special tokens, and padding are excluded;
- every exported position refers to the exact model input;
- repeated tokens remain distinct occurrences;
- deterministic output in evaluation mode;
- existing checkpoint loads without model-parameter incompatibility.

### 14.4 Export tests

Verify:

- JSONL is valid and resume-safe;
- CSV escapes tokenizer strings correctly;
- ranking is deterministic;
- raw and normalized scores are finite;
- truncated tokens are not assigned zero scores;
- skipped commits contain explicit reasons.

### 14.5 Checkpoint loading

Create an inference-only checkpoint loader that loads model weights with `map_location` and does not require optimizer or scheduler state to run attribution.

It may support both full training checkpoints and model-state-only checkpoints. Training resume behavior remains unchanged.

---

## 15. IMPLEMENTED: Provenance-Preserving Line Ranking

This section is the current implementation contract.

### 15.1 Git input and provenance contract

Add a `prepare-lines` command accepting:

```text
-repo_path <local-clone>
-commit_id <sha> | -commit_urls <txt-or-jsonl>
-output_dir
```

URLs must be normalized to full 40-character SHAs from the local clone. For
each non-binary changed file, parse the first-parent unified diff and retain:

- old and new file paths;
- hunk index and hunk header;
- diff position;
- old and new line numbers;
- exact raw line text and normalized model text;
- real Git change type (`added` or `deleted`);
- historical model marker, stored separately from real change type.

Before assigning any attribution to a line, code must:

1. parse the source diff;
2. reproduce the historical `code_change` serialization;
3. verify the reconstructed serialization against the dataset record;
4. reproduce tokenization;
5. verify exact `input_ids` equality;
6. skip commits with ambiguous or failed alignment.

Suggested future metadata:

```json
{
  "source_provenance": "raw_diff_sidecar",
  "alignment_status": "exact",
  "serialized_input_match": true,
  "model_input_ids_match": true
}
```

### 15.2 Line schema

```python
@dataclass
class DiffLine:
    line_id: int
    file_path: str
    hunk_id: int
    diff_position: int
    old_line_no: int | None
    new_line_no: int | None
    change_type: str
    text: str
    eligible_for_ranking: bool
    normalized_text: str
    model_markers: dict[str, str]
```

### 15.3 Historical serialization and exact alignment

The preparation output must include both canonical provenance and the exact
historical serializations:

```text
merge: <ADD> <all Git-added tokens> <REMOVE> <all Git-deleted tokens>\n
patch: one historical change block per newline
```

The two historical serializers are inconsistent: DeepJIT/JITFine `merge`
uses `<ADD>` for Git-added lines and `<REMOVE>` for Git-deleted lines, while
SimCom's legacy `patch` rows put Git-deleted lines after `<ADD>` and Git-added
lines after `<REMOVE>`. Preserve both directions in `model_markers` (keys
`merge` and `patch`) for checkpoint compatibility; `change_type` always follows
Git semantics. Never infer one from the other.

Alignment must be constructed while serialization is emitted, using stable
line IDs and token occurrence ranges. Never align later by token text because
repeated lines and tokens are ambiguous. Before ranking, verify the reconstructed
serialization and resulting model tensor/token IDs against the supplied JSONL.

Unobserved/truncated lines must use null scores rather than zero:

```json
{
  "covered": false,
  "raw_score": null,
  "normalized_score": null
}
```

### 15.4 Model-specific attribution and line aggregation

JITFine uses its existing code-token attention. DeepJIT hooks
`convs_code_line`, projects every kernel position through its exact token
receptive field, and combines that with gradients flowing through the
commit-stage CNN. Com uses the existing `convs_code_file` CAM and maps an
observed patch row to the changed line IDs emitted into that row.

Aggregate observed positions using configurable methods:

- `sum`;
- `mean`;
- `max`;
- `length_normalized_sum`.

Store at least sum, mean, maximum, and token count so line-length bias can be analyzed.

Suggested initial default: `sum`, with mandatory comparison against `mean`.

### 15.5 Line-level outputs

Future artifacts may include:

```text
<output-dir>/<commit-id>/
├── prediction.json
├── ranked_lines.json
├── ranked_lines.csv
└── heatmap.html
```

The HTML must show all changed lines and clearly mark lines not observed by the model.

### 15.6 Coverage and truncation

Report row and token coverage independently. DeepJIT normally observes one
flattened content row and at most `code_length` tokens. Com observes at most
`code_line` patch rows and `code_length` tokens per row. JITFine observes at
most 510 joint content tokens. Unobserved lines use null scores and never enter
the ranked list.

### 15.7 SimCom diagnostics

The current final probability is the mean of Sim and Com probabilities. Do not report a `dominant_component` based on distance from 0.5; confidence margin is not component contribution.

Instead report:

- `sim_score`;
- `com_score`;
- `final_score`;
- component agreement/disagreement;
- whether Com supports the final predicted class;
- that line/token localization covers Com only.

### 15.8 DEFERRED: intervention-based faithfulness

Line masking should be called an intervention because padding or `<NULL>` embeddings are not guaranteed to be neutral in CNN models.

Future comprehensiveness and sufficiency tests must record:

- replacement strategy;
- affected positions;
- whether sequence length was preserved;
- original and intervened probabilities;
- random/first/longest baselines;
- multiple random seeds.

---

## 16. Implementation Phases

### Phase 0: Audit and fixtures — COMPLETED

1. Record actual JITFine input construction and tensor shapes.
2. Add golden preprocessing fixtures.
3. Implement inference-only checkpoint loading.
4. Confirm prediction preservation on at least one real checkpoint and test split.

Deliverable:

```text
attribution/IMPLEMENTATION_NOTES.md
```

### Phase 1: Preprocessing metadata — COMPLETED

Implement token-region and truncation metadata inside the existing preprocessing path without changing legacy behavior.

Acceptance:

- every observed sequence position has one region;
- valid existing input produces identical tensors;
- malformed markers yield structured skips.

### Phase 2: JITFine attention baseline — COMPLETED

Implement:

- optional structured model output;
- last-layer CLS attention averaged across heads;
- code-token filtering;
- deterministic token ranking;
- prediction-preservation tests.

### Phase 3: CLI and batch export — COMPLETED

Implement:

- existing-CLI integration;
- commit selection after full-test preprocessing;
- JSONL and CSV export;
- atomic writes, resume, and summary reporting.

### Phase 4: Attention variants — OPTIONAL

Add all-layer averaging and attention rollout only after the baseline is verified.

### Phase 5: Source-line provenance — COMPLETED

Implement URL/SHA-list ingestion from a local clone, canonical diff parsing,
historical merge/patch serialization, provenance sidecars, exact alignment
verification and coverage tracking.

### Phase 6: Three-model line ranking — COMPLETED

Implement JITFine attention-to-line aggregation, DeepJIT token-stage Grad-CAM
and Com row-to-line aggregation. Export deterministic ranked and uncovered
lines for all three models.

### Phase 7: Hunk-aware chunked attribution — DEEPJIT/SIMCOM IMPLEMENTED

For DeepJIT and SimCom, never combine different Git hunks in one chunk. Treat
each hunk as one chunk when it has at most 10 changed source lines; split larger
hunks sequentially into subchunks of at most 10 changed lines. Context lines do
not count toward the limit. Repeat the unchanged commit message for every
chunk, run prediction and hierarchical Grad-CAM independently, translate local
token positions through retained canonical line IDs, and retain the top-1
ranked source line from every chunk at commit level. Record every per-chunk
score and use the maximum chunk probability only as the explicitly labelled
commit-level prediction summary.

### Phase 8: Line visualization and interventions — DEFERRED

Implement line heatmaps, comprehensiveness, sufficiency, and baselines after verified line ranking exists.

---

## 17. Acceptance Criteria

The current task is complete when:

1. an existing JITFine checkpoint loads without retraining;
2. attribution uses the exact current test preprocessing path;
3. selecting one commit does not change full-test manual-feature scaling context;
4. attention-enabled inference preserves the normal prediction within tolerance;
5. only observed added/removed code tokens are ranked;
6. message, markers, special tokens, and padding are excluded;
7. every token result contains its exact model sequence position and token ID;
8. truncation is measured and omitted tokens are not assigned zero attention;
9. ranking is deterministic;
10. results export successfully to JSONL and CSV;
11. malformed commits are skipped with explicit reasons;
12. output documentation states that attention is not class-specific and is not proof of vulnerability;
13. no source-line, file, or hunk claims are made without verified provenance;
14. HTML visualization and intervention sections remain explicitly deferred.
15. a short GitHub URL or SHA resolves to one full commit in a local clone;
16. every eligible line has stable file/hunk/old/new-line provenance;
17. real Git change type is independent from the legacy model marker;
18. reconstructed merge/patch serialization exactly matches the supplied model record;
19. JITFine, DeepJIT and Com each produce deterministic changed-line rankings;
20. DeepJIT line ranking uses token-stage attribution rather than treating its flattened merge row as one source line;
21. uncovered or truncated lines have null scores and are excluded from ranking;
22. existing checkpoint predictions are preserved within device tolerance.
23. DeepJIT/SimCom chunks never cross a Git file or hunk boundary;
24. hunks larger than 10 changed source lines are split without dropping trailing lines;
25. every chunk repeats the same commit message and exposes its file, hunk and subchunk identity;
26. the commit result exposes exactly one top-ranked source line per chunk that contains attributable source tokens with verified provenance.

---

## 18. Expected Current Deliverables

1. Backward-compatible JITFine preprocessing metadata.
2. Backward-compatible optional attention output.
3. JITFine token-level code-change attention extractor.
4. Deterministic token ranking.
5. JSONL and CSV exporters.
6. CLI integration for single-commit selection and batch attribution.
7. Prediction-preservation and preprocessing tests.
8. Inference-only checkpoint loader.
9. `attribution/IMPLEMENTATION_NOTES.md` with actual tensor shapes and repository-specific decisions.
10. README example command and interpretation limitations.

11. Git URL/SHA list preparation command and provenance sidecar.
12. Exact historical merge/patch serializer with occurrence-level alignment.
13. JITFine attention-based changed-line ranking.
14. DeepJIT token-stage Grad-CAM changed-line ranking.
15. Com hierarchical Grad-CAM changed-line ranking.
16. Ranked/uncovered line JSONL and CSV exports plus coverage metadata.
17. DeepJIT/SimCom hunk-aware 10-changed-line chunk attribution with per-chunk diagnostics and commit-level top-1 line candidates.
18. Review-oriented JSONL containing commit oversized-hunk notes and every hunk's source-line rankings.

HTML heatmaps and intervention-based faithfulness remain deferred.
