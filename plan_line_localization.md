# Implementation Plan: Post-hoc Changed-Line Ranking for JIT-VP Models

## 1. Objective

Implement an inference-only explanation pipeline that ranks changed lines inside a commit according to how strongly they contribute to a model's commit-level vulnerability prediction.

The task is not supervised line-level vulnerability localization. No line-level ground-truth labels are assumed. The system must therefore produce a ranked list of changed lines rather than predict whether each line is truly vulnerable.

Target models and attribution methods:

- **JITFine**: Transformer attention-based line ranking, following the general mechanism used by LineVul.
- **DeepJIT**: one-dimensional Grad-CAM on the CNN branch that processes code changes.
- **SimCom**: one-dimensional Grad-CAM on the CNN-based **Com** component only.

The existing trained checkpoints must be reused. The implementation must not require retraining unless the current model code makes checkpoint-compatible inference impossible.

---

## 2. Scope

### 2.1 Included

The implementation must:

1. Load existing trained checkpoints.
2. Reconstruct the exact code-change input used by each model.
3. Preserve a mapping from model tokens or CNN positions back to changed lines.
4. Compute attribution scores at token or sequence-position level.
5. Aggregate those scores into line-level scores.
6. Rank changed lines within each commit.
7. Export machine-readable and human-readable outputs.
8. Support batch inference over a dataset split.
9. Include faithfulness-oriented checks that do not require line-level ground truth.

### 2.2 Excluded

The implementation does not need to:

- train a line-level classifier;
- fine-tune any model for localization;
- use vulnerable-line labels;
- compute Top-k localization accuracy, IFA, MRR, MAP, or Recall@LOC against a line-level oracle;
- claim that the highest-ranked line is the actual vulnerability root cause;
- produce explanations for expert-feature-only components at line granularity.

---

## 3. Terminology

Use the following terms consistently in code, documentation, and output files:

- **Commit-level prediction**: the original model output indicating whether a commit is vulnerable.
- **Changed-line ranking**: the ordered list of added and deleted lines ranked by attribution score.
- **Line score**: the aggregated attribution score assigned to one changed line.
- **Explanation scope**: the part of the model being explained, such as the CodeBERT code branch or the Com CNN code branch.
- **Covered line**: a changed line that is represented in the model input after preprocessing and truncation.
- **Uncovered line**: a changed line excluded from model input, usually because of truncation.

Avoid using language that implies ground-truth localization. Preferred task names:

- `post-hoc changed-line ranking`
- `commit-level prediction explanation through changed-line ranking`

---

## 4. High-Level Pipeline

```text
Raw commit diff
    ↓
Canonical diff parsing
    ↓
Model-specific preprocessing
    ↓
Token/position-to-line alignment
    ↓
Commit-level inference
    ↓
Model-specific attribution
    ↓
Attribution aggregation by changed line
    ↓
Within-commit normalization
    ↓
Changed-line ranking
    ↓
JSON / CSV / HTML export
```

---

## 5. Proposed Source Structure

```text
vulguard_lite/
├── localization/
│   ├── __init__.py
│   ├── schemas.py
│   ├── diff_parser.py
│   ├── alignment.py
│   ├── aggregation.py
│   ├── normalization.py
│   ├── base.py
│   ├── attention.py
│   ├── gradcam_1d.py
│   ├── jitfine_localizer.py
│   ├── deepjit_localizer.py
│   ├── simcom_localizer.py
│   ├── faithfulness.py
│   ├── export.py
│   └── visualization.py
├── commands/
│   └── localize.py
└── tests/
    ├── test_diff_parser.py
    ├── test_alignment.py
    ├── test_aggregation.py
    ├── test_jitfine_attention.py
    ├── test_gradcam_1d.py
    ├── test_simcom_components.py
    ├── test_checkpoint_compatibility.py
    └── test_prediction_preservation.py
```

Adapt the paths to the current repository layout instead of forcing this exact structure if the project already has an established architecture.

---

## 6. Shared Data Schemas

Create reusable dataclasses or equivalent typed structures.

### 6.1 Diff line

```python
from dataclasses import dataclass


@dataclass
class DiffLine:
    line_id: int
    file_path: str
    hunk_id: int
    diff_position: int
    old_line_no: int | None
    new_line_no: int | None
    change_type: str  # added, deleted, context
    text: str
    eligible_for_ranking: bool
```

### 6.2 Token or model-position alignment

```python
@dataclass
class PositionAlignment:
    position: int
    line_id: int | None
    char_start: int | None
    char_end: int | None
    is_special_token: bool
    is_padding: bool
    is_code_change: bool
```

### 6.3 Ranked line

```python
@dataclass
class RankedLine:
    rank: int
    line_id: int
    file_path: str
    hunk_id: int
    old_line_no: int | None
    new_line_no: int | None
    change_type: str
    text: str
    raw_score: float
    normalized_score: float
    token_count: int
    covered: bool
```

### 6.4 Localization result

```python
@dataclass
class LocalizationResult:
    commit_id: str
    model_name: str
    attribution_method: str
    explanation_scope: str
    prediction_score: float
    predicted_label: int
    target_class: int
    ranked_lines: list[RankedLine]
    total_changed_lines: int
    covered_changed_lines: int
    coverage_ratio: float
    truncated: bool
    metadata: dict
```

---

## 7. Diff Parsing and Canonical Line Representation

### 7.1 Requirements

Implement a parser that:

1. Separates files in a multi-file diff.
2. Parses hunk headers of the form:

   ```text
   @@ -old_start,old_count +new_start,new_count @@
   ```

3. Tracks old and new line numbers independently.
4. Labels each line as:
   - `added`
   - `deleted`
   - `context`
5. Preserves the exact text used by the model preprocessing pipeline.
6. Assigns a stable `line_id` across the entire commit.
7. Marks only added and deleted lines as rankable by default.

### 7.2 Candidate-line policy

Default configuration:

```yaml
candidate_lines:
  include_added: true
  include_deleted: true
  include_context: false
```

Context lines may still receive internal attribution for debugging, but they must not appear in the main ranked list unless explicitly enabled.

### 7.3 Edge cases

Handle at least:

- empty lines;
- newline-at-end-of-file markers;
- file creation and deletion;
- renamed files;
- binary files;
- diffs with malformed or absent hunk headers;
- commits with no eligible changed lines;
- commits affecting multiple files;
- truncated model input.

Binary files and unparseable sections should be skipped with explicit metadata rather than crashing the full batch.

---

## 8. Alignment Between Model Input and Diff Lines

This is the most important shared component.

### 8.1 Core rule

Do not tokenize or preprocess the diff using a new pipeline that differs from training or normal inference.

The localization code must reuse the exact existing preprocessing functions for each model and extend them to optionally return alignment metadata.

### 8.2 Suggested API

Add an optional argument:

```python
def preprocess_commit(..., return_alignment: bool = False):
    ...
```

Normal training and evaluation must remain unchanged:

```python
return_alignment=False
```

Localization mode:

```python
return_alignment=True
```

Suggested returned structure:

```python
{
    "model_inputs": {...},
    "lines": list[DiffLine],
    "position_alignment": list[PositionAlignment],
    "covered_line_ids": list[int],
    "uncovered_line_ids": list[int],
    "truncated": bool,
}
```

### 8.3 Transformer alignment

For JITFine, prefer tokenizer offset mappings when supported:

```python
encoded = tokenizer(
    text,
    return_offsets_mapping=True,
    truncation=True,
    ...,
)
```

Map every non-special token span back to the corresponding diff line using character intervals in the exact serialized code-change string.

If the tokenizer is slow or does not return offsets, reconstruct alignment using the existing tokenization logic and explicit line separators. Add tests for subword tokens and repeated source text.

### 8.4 CNN alignment

For DeepJIT and Com, determine exactly what one sequence position represents in the current implementation:

- token;
- word;
- line;
- nested token position inside a line;
- flattened code-change sequence.

Build the alignment against the actual tensor fed into the embedding layer. Do not infer alignment from a separate reconstruction after inference.

### 8.5 Truncation

Never treat an uncovered line as score zero.

Use:

```json
{
  "covered": false,
  "raw_score": null,
  "normalized_score": null
}
```

The primary ranked list should contain only covered eligible lines. Export uncovered lines separately in metadata.

---

## 9. JITFine Attention-Based Line Ranking

### 9.1 Explanation scope

Explain only the CodeBERT branch processing code changes.

Do not claim that the line ranking fully explains the combined JITFine prediction when the final classifier also uses commit messages or expert features.

Required metadata:

```json
{
  "explanation_scope": "jitfine_codebert_code_change_branch",
  "full_model_explanation": false
}
```

### 9.2 Checkpoint compatibility

Modify the forward path only by adding optional output control. Do not rename or reshape parameters.

Suggested signature:

```python
def forward(
    self,
    ...,
    output_attentions: bool = False,
    return_localization_data: bool = False,
):
    ...
```

Call CodeBERT with:

```python
outputs = self.codebert(
    input_ids=code_input_ids,
    attention_mask=code_attention_mask,
    output_attentions=output_attentions,
    return_dict=True,
)
```

Existing checkpoints should load using the same state dictionary.

### 9.3 Baseline attention strategy

Use CLS-to-token attention averaged across heads and layers:

```python
layer_scores = []
for layer_attention in attentions:
    # [batch, heads, seq, seq]
    cls_to_tokens = layer_attention[:, :, 0, :]
    layer_scores.append(cls_to_tokens.mean(dim=1))

token_scores = torch.stack(layer_scores, dim=0).mean(dim=0)
```

Default configuration:

```yaml
jitfine:
  method: attention
  attention_strategy: all_layers_cls_mean
  aggregate_heads: mean
  aggregate_layers: mean
```

Also implement optional strategies for later comparison:

- `last_layer_cls_mean`
- `attention_rollout`

Do not make these optional variants block the baseline implementation.

### 9.4 Token filtering

Set non-code scores to zero or exclude them before line aggregation:

- CLS token;
- SEP token;
- padding;
- artificial separators;
- commit-message tokens;
- expert-feature placeholders;
- any token not mapped to a changed line.

### 9.5 Token-to-line aggregation

Implement:

- `sum`
- `mean`
- `max`
- `length_normalized_sum`

Default:

```yaml
line_aggregation: sum
```

Store at least:

- summed score;
- mean token score;
- token count.

This allows later analysis of whether long lines are favored by simple summation.

---

## 10. DeepJIT Grad-CAM Line Ranking

### 10.1 Explanation scope

Apply Grad-CAM only to the CNN branch that processes code changes.

Do not use the message CNN for the requested line ranking.

Required metadata:

```json
{
  "explanation_scope": "deepjit_code_change_cnn",
  "full_model_explanation": false
}
```

### 10.2 Target layers

Inspect the actual implementation and identify every convolution branch operating on code changes before max pooling.

Typical pattern:

```text
Embedding
├── Conv1d kernel=3
├── Conv1d kernel=4
└── Conv1d kernel=5
    ↓
Global max pooling
```

Register hooks on the convolution outputs before pooling.

### 10.3 Reusable GradCAM1D implementation

Create a class that:

1. registers a forward hook for activations;
2. registers a backward hook for gradients;
3. runs forward inference;
4. backpropagates from the vulnerable-class logit;
5. computes one-dimensional Grad-CAM;
6. interpolates the CAM to the model input sequence length;
7. removes hooks safely.

Formula implementation:

```python
weights = gradients.mean(dim=-1, keepdim=True)
cam = torch.relu((weights * activations).sum(dim=1))
```

Use the vulnerable-class logit, not the thresholded prediction and preferably not the post-sigmoid probability.

### 10.4 Multi-kernel aggregation

For each convolution branch:

1. compute a CAM;
2. resize it to the original input sequence length;
3. normalize only after all branches are aligned;
4. aggregate branches.

Default:

```yaml
deepjit:
  method: gradcam
  branch_aggregation: mean
```

Optional later variants:

- `max`
- gradient-norm-weighted mean

### 10.5 Pooling sanity check

Because global max pooling can create sparse contributions, also expose the max-pooling argmax positions for debugging.

This is not the primary explanation method. It is a validation aid to confirm that Grad-CAM highlights positions actually selected by the CNN filters.

---

## 11. SimCom Com-Component Grad-CAM

### 11.1 Explanation scope

SimCom contains:

- `Sim`: expert-feature model;
- `Com`: CNN-based component;
- final combination of their predictions.

Only the Com code-change CNN can be mapped to source lines.

Required metadata:

```json
{
  "explanation_scope": "simcom_com_code_change_cnn",
  "localization_component": "Com",
  "full_model_explanation": false
}
```

### 11.2 Expose component scores

Modify inference to optionally return:

```python
{
    "sim_score": sim_score,
    "com_score": com_score,
    "final_score": final_score,
}
```

Do not alter the original final score calculation.

### 11.3 Grad-CAM implementation

Reuse the DeepJIT GradCAM1D module against the Com code-change CNN.

Do not run Grad-CAM against the full SimCom wrapper if the wrapper simply combines Sim and Com predictions. This would only scale the gradient and would not add line-level information.

### 11.4 Component dominance metadata

Report whether the final prediction is mostly supported by Sim or Com.

A simple initial diagnostic:

```python
sim_relevance = abs(sim_score - 0.5)
com_relevance = abs(com_score - 0.5)
dominant_component = "Sim" if sim_relevance > com_relevance else "Com"
```

Output:

```json
{
  "sim_score": 0.41,
  "com_score": 0.83,
  "final_score": 0.62,
  "dominant_component": "Com",
  "line_ranking_reliability": "normal"
}
```

If Sim is dominant, set:

```json
{
  "line_ranking_reliability": "limited_due_to_sim_component"
}
```

This diagnostic is heuristic and must be documented as such.

---

## 12. Line-Score Normalization and Ranking

### 12.1 Raw aggregation

For every covered eligible line, aggregate all mapped token or position scores.

### 12.2 Within-commit normalization

Normalize scores independently inside each commit:

```python
normalized = (score - min_score) / (max_score - min_score + 1e-12)
```

Do not normalize across the full dataset because the task is to rank lines within one commit.

### 12.3 Ranking

Sort by normalized score descending.

Tie-breaking order:

1. higher raw score;
2. lower diff position;
3. lower line ID.

This guarantees deterministic output.

### 12.4 Degenerate cases

Handle:

- all raw scores equal;
- all raw scores zero;
- one eligible covered line;
- no eligible covered lines;
- NaN or infinite attribution values.

For all-equal scores, set normalized scores consistently, for example to `0.0`, and preserve deterministic diff order.

---

## 13. CLI Requirements

Implement a command similar to:

```bash
python -m vulguard_lite localize \
  --model jitfine \
  --checkpoint <checkpoint_or_hf_path> \
  --dataset linux \
  --split test \
  --output-dir results/localization/jitfine/linux
```

Required options:

```text
--model {jitfine,deepjit,simcom}
--checkpoint
--dataset
--split
--output-dir
--device
--batch-size
--target-class
--line-aggregation
--only-predicted-vulnerable
--only-true-positive
--all-commits
--top-k
--overwrite
--resume
```

Rules:

- `--only-true-positive` uses commit-level ground-truth labels only.
- `--only-predicted-vulnerable` filters by the model's commit-level prediction.
- `--all-commits` runs localization for every commit.
- Make these three selection modes mutually exclusive.
- Batch mode must continue after one malformed commit and log the failure.

Also support one-commit inference by commit ID or input JSON if practical in the current repository.

---

## 14. Output Format

For each commit:

```text
<output-dir>/<commit-id>/
├── prediction.json
├── ranked_lines.json
├── ranked_lines.csv
└── heatmap.html
```

### 14.1 `prediction.json`

```json
{
  "commit_id": "abc123",
  "model": "jitfine",
  "prediction_score": 0.873,
  "predicted_label": 1,
  "target_class": 1,
  "attribution_method": "attention",
  "explanation_scope": "jitfine_codebert_code_change_branch",
  "total_changed_lines": 52,
  "eligible_changed_lines": 47,
  "covered_changed_lines": 21,
  "coverage_ratio": 0.4468,
  "truncated": true,
  "uncovered_line_ids": [22, 23, 24]
}
```

### 14.2 `ranked_lines.json`

```json
{
  "commit_id": "abc123",
  "ranked_lines": [
    {
      "rank": 1,
      "line_id": 8,
      "file_path": "crypto/x509/x509_vfy.c",
      "hunk_id": 0,
      "old_line_no": null,
      "new_line_no": 242,
      "change_type": "added",
      "text": "if (ctx == NULL) return 0;",
      "raw_score": 0.182,
      "normalized_score": 1.0,
      "token_count": 7,
      "covered": true
    }
  ]
}
```

### 14.3 CSV columns

```text
commit_id,model,rank,line_id,file_path,hunk_id,old_line_no,new_line_no,change_type,text,raw_score,normalized_score,token_count,covered
```

### 14.4 HTML heatmap

Show:

- commit ID;
- model;
- commit-level prediction score;
- attribution method;
- explanation scope;
- coverage and truncation information;
- file and hunk boundaries;
- diff markers;
- line rank;
- normalized score;
- clear marker for uncovered lines.

Do not hide uncovered lines. Render them with a label such as `not observed by model`.

---

## 15. Faithfulness Checks Without Line-Level Ground Truth

These checks do not prove that ranked lines are truly vulnerable. They test whether the ranking reflects the model's own prediction behavior.

### 15.1 Prediction preservation

Attribution mode must not change normal inference output.

```python
abs(original_score - localization_score) < tolerance
```

Suggested tolerance:

```text
1e-6 on CPU
1e-5 on GPU
```

### 15.2 Comprehensiveness

Mask the top-k ranked lines and rerun inference:

```text
Comprehensiveness@k = original_score - score_without_top_k
```

Larger positive values indicate that top-ranked lines were important to the prediction.

Evaluate for configurable values such as:

```text
k ∈ {1, 3, 5, 10}
```

### 15.3 Sufficiency

Retain only top-k ranked lines, mask other eligible changed lines, and rerun inference:

```text
Sufficiency@k = original_score - score_with_only_top_k
```

Smaller values indicate that top-ranked lines preserve more of the original prediction.

### 15.4 Masking policy

Do not delete source lines and rebuild an arbitrary new diff unless that matches the original model preprocessing.

Prefer masking at the model-input level:

- replace token IDs with the tokenizer mask token where supported;
- otherwise use padding or unknown tokens while preserving sequence length;
- preserve special tokens and segment boundaries;
- for CNN inputs, replace positions with the model's padding index or a documented neutral representation.

The masking strategy must be model-specific and recorded in output metadata.

### 15.5 Baselines

Compare attribution ranking against:

- random changed lines;
- first-k changed lines;
- longest-k changed lines;
- optionally max-pooling activation positions for CNN models.

Use several random repeats and a fixed seed.

### 15.6 Stability

Provide optional utilities for:

- top-k overlap;
- Spearman correlation;
- Kendall's tau;
- comparison across checkpoints or repeated runs.

Do not block the core implementation on stability analysis.

---

## 16. Configuration

Add a localization section to the existing config system.

```yaml
localization:
  target_class: 1
  candidate_lines:
    include_added: true
    include_deleted: true
    include_context: false

  line_aggregation: sum
  normalization: minmax_within_commit
  top_k: 20

  jitfine:
    method: attention
    attention_strategy: all_layers_cls_mean
    aggregate_heads: mean
    aggregate_layers: mean

  deepjit:
    method: gradcam
    target_branch: code
    branch_aggregation: mean

  simcom:
    method: gradcam
    component: com
    target_branch: code
    branch_aggregation: mean
    report_component_scores: true

  faithfulness:
    enabled: false
    k_values: [1, 3, 5, 10]
    random_repeats: 20
    seed: 42
```

---

## 17. Testing Requirements

### 17.1 Diff parser tests

Test:

- one file, one hunk;
- multiple files;
- multiple hunks;
- added-only file;
- deleted-only file;
- blank changed lines;
- no-newline marker;
- malformed hunk;
- binary file;
- rename.

### 17.2 Alignment tests

Verify:

- every covered non-special code token maps to a valid line;
- subword tokens map to the same source line;
- special and padding tokens are excluded;
- repeated identical text on different lines maps correctly;
- truncation produces uncovered lines rather than zero-scored lines;
- multi-file diffs do not mix line IDs.

### 17.3 JITFine tests

Verify:

- checkpoint loads without missing or unexpected trainable parameters;
- normal prediction equals attention-enabled prediction within tolerance;
- attention output shape is valid;
- padding and special-token scores are excluded;
- line-score sum matches mapped token-score sum;
- deterministic inference in evaluation mode.

### 17.4 Grad-CAM tests

Verify:

- activations and gradients are captured;
- gradients are not `None`;
- one CAM is produced per convolution branch;
- each branch CAM is resized correctly;
- merged CAM length equals input sequence length;
- hooks are removed after inference;
- repeated calls do not accumulate hooks;
- prediction remains unchanged by hook registration;
- target class selection works.

### 17.5 SimCom tests

Verify:

- Sim, Com, and final scores are returned correctly;
- final score matches the existing combination logic;
- Grad-CAM is computed only from Com;
- component dominance metadata is emitted;
- checkpoint compatibility is preserved.

### 17.6 Export tests

Verify:

- valid JSON;
- deterministic ranking;
- CSV escaping for commas, quotes, and newlines;
- HTML generation for multi-file commits;
- null scores for uncovered lines;
- output directories are resume-safe.

---

## 18. Logging and Error Handling

Use structured logging where possible.

Log at least:

- model and checkpoint;
- dataset and split;
- commit ID;
- number of parsed files and lines;
- input length before and after truncation;
- coverage ratio;
- attribution runtime;
- export paths;
- skipped commits and reasons.

Batch execution must write a summary file:

```json
{
  "processed": 1000,
  "succeeded": 973,
  "skipped": 12,
  "failed": 15,
  "failures": [
    {
      "commit_id": "...",
      "reason": "..."
    }
  ]
}
```

---

## 19. Performance and Batch Processing

### 19.1 Inference

Use:

```python
model.eval()
```

For JITFine attention extraction, inference may use `torch.no_grad()`.

For Grad-CAM, gradients are required. Do not wrap the relevant forward pass in `torch.no_grad()`.

Before Grad-CAM backward:

```python
model.zero_grad(set_to_none=True)
```

### 19.2 Caching

Cache reusable preprocessing outputs when practical:

- parsed diff lines;
- serialized model input;
- token IDs;
- attention masks;
- alignment metadata.

Do not cache computation graphs or gradients.

### 19.3 Resume support

If all required output files for a commit already exist and pass basic validation, skip it unless `--overwrite` is enabled.

### 19.4 PBS compatibility

Keep the CLI stateless enough to support dataset sharding:

```bash
--shard-index 0
--num-shards 10
```

Sharding should be deterministic by ordered commit ID list or dataset index.

---

## 20. Implementation Phases

### Phase 1: Repository inspection

Before coding:

1. Locate model definitions for JITFine, DeepJIT, SimCom, and Com.
2. Locate dataset and preprocessing code.
3. Identify checkpoint loading logic.
4. Document actual tensor shapes.
5. Identify where truncation occurs.
6. Identify how code changes are serialized.
7. Confirm which JITFine input corresponds to code changes.
8. Confirm the exact SimCom score-combination rule.

Deliverable:

```text
localization/IMPLEMENTATION_NOTES.md
```

This file must record actual class names, module paths, tensor shapes, and chosen hook layers.

### Phase 2: Diff parser and alignment

Implement:

- canonical diff parser;
- line schemas;
- model-input alignment;
- truncation coverage tracking;
- unit tests.

Acceptance criteria:

- every covered model position can be traced to a diff line or explicitly marked non-code;
- uncovered lines are identified correctly;
- existing training and evaluation paths remain unchanged.

### Phase 3: JITFine localizer

Implement:

- optional attention output;
- CLS-based token scoring;
- token filtering;
- line aggregation;
- ranking and export;
- checkpoint compatibility tests.

Acceptance criteria:

- existing checkpoint loads;
- prediction is preserved;
- one commit produces valid ranked-line output without retraining.

### Phase 4: DeepJIT localizer

Implement:

- reusable GradCAM1D;
- hooks on code convolution branches;
- multi-kernel CAM aggregation;
- line aggregation and export;
- hook lifecycle tests.

Acceptance criteria:

- valid CAM is produced for each code branch;
- prediction is preserved;
- no hooks leak between samples.

### Phase 5: SimCom localizer

Implement:

- component score exposure;
- Com-only Grad-CAM;
- dominance and reliability metadata;
- output export.

Acceptance criteria:

- final score remains identical to existing inference;
- ranked lines come only from Com;
- output clearly states that Sim is not explained at line level.

### Phase 6: Batch CLI and visualization

Implement:

- dataset split iteration;
- selection modes;
- resume and overwrite;
- JSON, CSV, HTML;
- error summary;
- optional sharding.

### Phase 7: Faithfulness utilities

Implement:

- prediction preservation;
- comprehensiveness@k;
- sufficiency@k;
- random, first-line, and longest-line baselines;
- summary CSV.

This phase is secondary to producing correct rankings.

---

## 21. Acceptance Criteria

The task is complete when all of the following hold:

1. Existing checkpoints for JITFine, DeepJIT, and SimCom load successfully.
2. No retraining is required.
3. Normal model predictions are preserved when localization is enabled.
4. Every ranked line corresponds to an actual added or deleted line in the original diff.
5. Lines omitted by truncation are marked uncovered rather than assigned zero attribution.
6. JITFine produces attention-based changed-line rankings from its code branch.
7. DeepJIT produces Grad-CAM-based changed-line rankings from its code CNN.
8. SimCom produces Grad-CAM-based changed-line rankings from Com only.
9. SimCom output includes Sim, Com, and final scores.
10. Each commit can be exported to JSON and CSV.
11. HTML heatmaps clearly display files, hunks, scores, ranks, and uncovered lines.
12. Batch execution can resume after interruption.
13. Unit tests cover parser, alignment, attribution, checkpoint compatibility, and prediction preservation.
14. Documentation explicitly states that the output is a model-derived ranking, not verified vulnerable-line ground truth.

---

## 22. Important Constraints for the Coding Agent

- Inspect the real repository before assuming class names or tensor shapes.
- Reuse existing preprocessing rather than reimplementing it independently.
- Keep all model changes backward-compatible with existing checkpoints.
- Do not modify training behavior unless strictly necessary.
- Do not retrain models as part of this task.
- Do not silently assign zero attribution to truncated lines.
- Do not describe attention or Grad-CAM scores as proof that a line is vulnerable.
- Do not claim that JITFine or SimCom is fully explained when only one branch is localized.
- Prefer small, testable changes over rewriting model code.
- Add comments around tensor dimensions and hook locations.
- Record any mismatch between the intended plan and the actual repository architecture in `IMPLEMENTATION_NOTES.md`.

---

## 23. Expected Final Deliverables

The coding agent should produce:

1. Source code for the shared localization framework.
2. JITFine attention localizer.
3. DeepJIT Grad-CAM localizer.
4. SimCom Com-component Grad-CAM localizer.
5. Diff parser and model-position alignment.
6. CLI for single-commit and batch localization.
7. JSON and CSV exporters.
8. HTML heatmap generator.
9. Faithfulness utilities.
10. Unit and integration tests.
11. `IMPLEMENTATION_NOTES.md` describing repository-specific decisions.
12. A concise README section with example commands.

