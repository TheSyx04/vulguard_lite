# Ground-truth hunk pipeline

The workflow has two independent stages:

1. `prepare-ground-truth-hunks` creates a reusable, model-independent ground-
   truth hunk dataset from XLSX metadata and a local Git clone.
2. `rank-ground-truth` optionally ranks that common dataset with JITFine,
   DeepJIT, or SimCom/Com.

Ground-truth line numbers are matched against the new (`+`) side of each diff,
because `Vul_commit` identifies the vulnerability-introducing commit. Targets
inside newly added files are detected using Git status `A`, reported separately,
and excluded from the eligible evaluation denominator.

## Installation

```bash
cd /work/vulguard_lite
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Stage 1: prepare common ground-truth hunks

This stage requires only the workbook and a local Git clone. It does not load a
model, checkpoint, dictionary, Torch inference, or Grad-CAM.

### Linux

```bash
python -m vulguard_lite prepare-ground-truth-hunks \
  -repo_language C \
  -repo_path /data/repos/linux \
  -ground_truth /work/vulguard_lite/Data_line_vul.xlsx \
  -sheet Linux \
  -output_dir /data/results/linux_ground_truth
```

### OpenSSL

```bash
python -m vulguard_lite prepare-ground-truth-hunks \
  -repo_language C \
  -repo_path /data/repos/openssl \
  -ground_truth /work/vulguard_lite/Data_line_vul.xlsx \
  -sheet OpenSSL \
  -output_dir /data/results/openssl_ground_truth
```

The commit manifest is generated from `Vul_commit`; no manually maintained
commit text file is needed.

Stage 1 output:

```text
linux_ground_truth/
├── inputs/
│   ├── commit_references.jsonl
│   ├── line_provenance.jsonl
│   ├── merge.jsonl
│   ├── patch.jsonl
│   └── all_eligible_hunks.jsonl
├── ground_truth_hunks.jsonl
└── ground_truth_hunks_summary.json
```

`ground_truth_hunks.jsonl` is the common file used by every model. It contains
Git paths, hunk headers, old/new line numbers, changed text, and
`ground_truth_matches`, but no model score or ranking.

The summary separates total, excluded-new-file, eligible, matched, and remaining
unmatched ground truth.

## Stage 2: rank with one model

Use a separate output directory per model. DeepJIT and SimCom rank only the
prepared ground-truth hunks, rather than every hunk in each commit.

### SimCom/Com

```bash
python -m vulguard_lite rank-ground-truth \
  -repo_language C \
  -prepared_dir /data/results/linux_ground_truth \
  -model simcom \
  -device cuda:0 \
  -model_path /data/checkpoints/simcom/linux_checkpoint.pth \
  -hyperparameters /work/vulguard_lite/models/simcom/hyperparameters.json \
  -dictionary /data/models/dict_linux.jsonl \
  -output_dir /data/results/linux_rankings/simcom \
  -hunk_chunk_size 10 \
  -line_aggregation sum \
  -overwrite
```

A standalone Com `.pth` with `-model simcom` runs Com-only inference, matching
the existing OpenSSL configuration.

### DeepJIT

Ground-truth inputs, dictionary, and checkpoint can be resolved directly from a
Hugging Face dataset repository. The checkpoint path is explicit because
`model_config` contains multiple folds and seeds. Downloads are cached under
`<dg_save_folder>/dg_cache/hf_ground_truth`; no manual download step is needed.

```bash
python -m vulguard_lite rank-ground-truth \
  -repo_language C \
  -repo_name openssl \
  -model deepjit \
  -device cpu \
  -hf_repo_id TheSyx/vulguard_lite \
  -hf_ground_truth_path dataset/ground_truth_hunks/openssl \
  -hf_checkpoint_path model_config/openssl/deepjit/openssl_0_1/seed_1 \
  -hyperparameters /work/vulguard_lite/models/deepjit/hyperparameters.json \
  -output_dir /data/results/openssl_rankings/deepjit \
  -metric_top_k 1 3 5 10 \
  -overwrite
```

When `-hf_ground_truth_path` is omitted it defaults to
`dataset/ground_truth_hunks/<repo_name>`. For CNN models, the dictionary also
defaults to `dataset/<repo_name>/dict_<repo_name>.jsonl`; override it with
`-hf_dictionary_path` only for a different repository layout. Local
`-prepared_dir`, `-model_path`, and `-dictionary` remain supported for offline
runs. Do not combine local prepared/checkpoint paths with `-hf_repo_id`.

### JITFine

JITFine uses the full commit input plus its 14 manual commit features, then the
pipeline retains rankings belonging to the common ground-truth hunks. For a
local run, `-features` is required and must align with the prepared commit IDs.

```bash
python -m vulguard_lite rank-ground-truth \
  -repo_language C \
  -prepared_dir /data/results/linux_ground_truth \
  -model jitfine \
  -device cuda:0 \
  -model_path /data/checkpoints/jitfine \
  -hyperparameters /work/vulguard_lite/models/jitfine/hyperparameters.json \
  -features /data/datasets/linux_features.jsonl \
  -output_dir /data/results/linux_rankings/jitfine \
  -overwrite
```

JITFine can also resolve all ranking artifacts from the Hugging Face dataset
repository. Its full test Kamei-feature file is used so feature scaling stays
consistent with normal test-time preprocessing; only the prepared commit IDs
are selected after scaling.

```bash
python -m vulguard_lite rank-ground-truth \
  -repo_language C \
  -repo_name openssl \
  -model jitfine \
  -device cpu \
  -hf_repo_id TheSyx/vulguard_lite \
  -hf_ground_truth_path dataset/ground_truth_hunks/openssl \
  -hf_checkpoint_path model_config/openssl/jitfine/openssl_0_1/seed_1 \
  -hyperparameters /work/vulguard_lite/models/jitfine/hyperparameters.json \
  -output_dir /data/results/openssl_rankings/jitfine \
  -metric_top_k 1 3 5 10 \
  -overwrite
```

The default JITFine feature lookup searches `dataset/<repo_name>` for the test
Kamei/TLEL JSONL. Use `-hf_features_path` when the repository uses a different
layout.

Stage 2 output contains the model's normal attribution exports plus:

```text
ranked_ground_truth_hunks.jsonl
ranked_ground_truth_summary.json
ranking_metrics_by_unit.jsonl
```

Each ranked record preserves the common hunk and adds `model_ranking`, so outputs
from different models can be compared without changing ground-truth membership.

### Upload ranking outputs to Hugging Face

Pass `-hf_upload_result True` to upload a successful ranking directory. The
destination dataset defaults to `-hf_repo_id`; override it with
`-hf_output_repo_id`. For HF checkpoint inputs, the default remote path is:

```text
line_ranking/<repo_name>/<model>/<config>/seed_<seed>
```

Use `-hf_output_folder` to override that path. Existing completed results can
be uploaded without rerunning attribution:

```bash
python scripts/upload_line_ranking_results.py \
  --output-root /data/output/line_ranking \
  --dataset linux \
  --model simcom \
  --hf-repo-id TheSyx/vulguard_lite
```

The backfill command refuses to upload when any discovered seed directory is
missing its ranked hunks, summary, or per-unit metrics file. Pass
`--skip-incomplete` to upload only complete config/seed directories.
Before uploading, it lists the destination dataset repo and skips every local
file whose destination path already exists. It therefore adds only missing
files and never overwrites an existing uploaded result.

Aggregate every uploaded experiment into one Excel workbook with one sheet for
each dataset:

```bash
python scripts/aggregate_line_ranking_results.py \
  --hf-repo-id TheSyx/vulguard_lite \
  --output line_ranking_results.xlsx
```

The default sheets are `openssl` and `linux`. Each row represents one
model/config/seed experiment and contains the flattened aggregate fields from
`ranked_ground_truth_summary.json`. The command checks that the discovered
three-model config/seed matrix is complete before writing. Use
`--allow-incomplete` for an in-progress upload. Because CSV files do not support
multiple sheets, pass `--csv-dir line_ranking_csv` to additionally create one
CSV file per dataset.

For JITFine, a ground-truth commit absent from the supplied manual-feature file
is recorded as skipped and remains unranked; it contributes a miss to absolute
coverage metrics rather than failing the whole job. Attribution records with
other failures are retried by `-resume` and block result upload until resolved.

### Ranking metrics

The rank stage reports two metric scopes. This distinction is required because
DeepJIT and SimCom run Grad-CAM independently for each hunk chunk; raw scores
from different chunks do not share a calibrated global scale.

| Scope | DeepJIT/SimCom unit | JITFine unit | Treatment of unranked ground truth |
|---|---|---|---|
| Absolute line metrics | All eligible ground-truth lines in the dataset | Same | Retained and counted as misses |
| Native-unit metrics | One Git-hunk chunk | One complete commit | Only units with at least one ranked relevant line are evaluated |

For the definitions below, `N` is the number of ranked candidates in a native
unit, `R` is the number of ranked relevant (vulnerable) lines in that unit, and
`r_i` is the native rank of relevant line `i`. Ranks are one-based. By default,
the pipeline evaluates `K = 1, 3, 5, 10`.

#### Absolute line metrics

These metrics use every eligible ground-truth line as the denominator and are
therefore the primary metrics for reporting localization coverage.

| JSON field | Definition |
|---|---|
| `ground_truth_line_count` | Total number of unique eligible vulnerable lines. |
| `ranked_ground_truth_line_count` | Vulnerable lines that received a native rank. |
| `unranked_ground_truth_line_count` | Eligible vulnerable lines with no rank, including lines not observed because of serialization or truncation. |
| `ranking_coverage` | `ranked_ground_truth_line_count / ground_truth_line_count`. |
| `line_mrr_unranked_as_zero` | Mean of `1 / r_i` across all eligible vulnerable lines; an unranked line contributes `0`. |
| `absolute_hit_count_at_K` | Number of all eligible vulnerable lines whose native rank is at most `K`. |
| `absolute_hit_rate_at_K` | `absolute_hit_count_at_K / ground_truth_line_count`; unranked lines are misses. |
| `recall_at_K_among_ranked_lines` | `absolute_hit_count_at_K / ranked_ground_truth_line_count`; this excludes unranked lines and must be read together with coverage. |
| `mean_rank_ranked_lines` / `median_rank_ranked_lines` | Mean/median native rank over ranked vulnerable lines only. |
| `mean_rank_percentile_ranked_lines` | Mean of `r_i / N` over ranked vulnerable lines for which the native candidate count is known. Lower is better. |

#### Native-unit metrics

Per-unit values are written to `ranking_metrics_by_unit.jsonl`. Aggregate
fields in `ranked_ground_truth_summary.json` are arithmetic means over evaluated
native units. A native unit is evaluated only when it contains at least one
ranked relevant line, so these observed-only metrics must not be interpreted as
dataset-wide coverage.

| Metric / JSON field | Per-unit definition | Better |
|---|---|---|
| Reciprocal rank / `mrr` | `1 / min(r_i)`; aggregate MRR is its mean over units. | Higher |
| Average precision / `map` | `AP = (1 / R) * sum(j / r_(j))`, where relevant ranks are sorted and `j` starts at 1; MAP is mean AP. | Higher |
| `hit_at_K` / `hit_rate_at_K` | Per unit: `1` if any `r_i <= K`, otherwise `0`; aggregate is the fraction of evaluated units hit at K. | Higher |
| `recall_at_K` / `mean_recall_at_K` | Per unit: `count(r_i <= K) / R`; aggregate is the mean over units. | Higher |
| `ndcg_at_K` / `mean_ndcg_at_K` | Binary-relevance DCG at K divided by ideal DCG for `min(R, K)` relevant lines. | Higher |
| `first_relevant_rank` | `min(r_i)`. The summary reports its mean and median. | Lower |
| IFA / `initial_false_alarms` | `first_relevant_rank - 1`: non-vulnerable candidates inspected before the first vulnerable line. | Lower |
| EXAM / `exam` | `first_relevant_rank / N`: fraction of the native candidate list inspected before the first vulnerable line is found. | Lower |
| `recall_at_F_loc` | Recall after inspecting the first `max(1, ceil(F * N))` candidates. With the default `F=0.2`, this is Recall@20% LOC. | Higher |
| `effort_at_F_recall` | Rank required to find `max(1, ceil(F * R))` relevant lines, divided by `N`. With `F=0.2`, this is Effort@20% Recall. | Lower |

The summary prefixes aggregated effort-aware metrics with `mean_`, for example
`mean_recall_at_0.2_loc` and `mean_effort_at_0.2_recall`. It also records
`evaluated_unit_count`, which should always accompany native-unit results.

#### Post-hoc JITFine hunk-chunk evaluation

JITFine natively ranks observed changed lines across a complete commit, while
DeepJIT and SimCom rank inside Git-hunk chunks. Existing JITFine artifacts can
be evaluated on the same maximum-10-line chunk scope without running inference
again. The post-processor retains each stored full-commit attention
`raw_score`, reconstructs chunks from canonical hunk line order, and assigns a
new `rank_in_chunk`:

```bash
python scripts/rerank_jitfine_hunk_chunks.py \
  --hf-repo-id TheSyx/vulguard_lite \
  --datasets linux openssl \
  --chunk-size 10 \
  --output-dir jitfine_hunk_chunk_results \
  --report jitfine_hunk_chunk_results.csv
```

The separate output filenames end in `_hunk_chunk`; native commit-level files
are never overwritten. Summaries record `inference_reused: true`,
`score_source_scope: full_commit_attention`, and
`ranking_scope: posthoc_hunk_chunk`. Coverage does not increase: skipped,
truncated, or otherwise unscored JITFine lines remain unranked.

For publication-oriented output, build separate concise reports:

```bash
python scripts/build_line_ranking_reports.py \
  --output-dir line_ranking_reports
```

`line_metrics/<dataset>.csv` is the primary report. Its effort-aware fields are
micro metrics over all eligible ground-truth lines: unranked lines remain
misses, contribute zero to `line_mrr`, and receive a `mean_line_exam` penalty
of `1.0`. `chunk_metrics/<dataset>.csv` contains secondary macro diagnostics
over hunk chunks. Its `hit_K` columns are coverage-aware binary success rates
over every chunk containing ground truth, including entirely unranked chunks as
misses; the other chunk macro metrics remain conditional on `ranked_chunks`.
CSV column names stay short because scope is encoded by the parent directory
instead of a flattened JSON prefix.

Customize the cutoffs and effort fraction with:

```bash
-metric_top_k 1 3 5 10 20 -effort_fraction 0.2
```

Aggregate metrics and their definitions are embedded in
`ranked_ground_truth_summary.json`. Per-unit diagnostics are written to
`ranking_metrics_by_unit.jsonl`, while per-ground-truth-line rank and coverage
details are stored under `ranking_metrics.ground_truth_line_results` in the
summary. Each line result contains its native `rank`, `candidate_count`,
`reciprocal_rank`, and `rank_percentile`; unranked lines have `rank: null` and
`reciprocal_rank: 0.0`.

Use `-resume` instead of `-overwrite` after an interrupted rank run. Do not pass
both flags.

## CPU, GPU, and network

- Stage 1 is CPU-only. A normal CPU is sufficient; no GPU is used.
- Stage 2 supports `-device cpu`, so a GPU is not mandatory. A CUDA GPU is
  recommended for Linux because attribution is neural-model inference repeated
  across hunk chunks.
- Neither stage calls GitHub APIs or crawls the web.
- Stage 1 can run offline when the Git clone contains every vulnerable commit
  and its first parent. Stage 2 can run offline when model artifacts and Python
  dependencies are local. HF-backed rank commands need network on their first
  run and then reuse the local Hugging Face cache.
- A shallow or partial clone may need network access for missing history or Git
  objects. Check representative commits before starting:

```bash
git -C /data/repos/linux cat-file -e 6019ce07^{commit}
git -C /data/repos/linux rev-parse 6019ce07^
```
