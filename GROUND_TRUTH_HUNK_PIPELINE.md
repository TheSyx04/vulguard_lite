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
pipeline retains rankings belonging to the common ground-truth hunks. Therefore
`-features` is required and must align with the prepared commit IDs.

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

Stage 2 output contains the model's normal attribution exports plus:

```text
ranked_ground_truth_hunks.jsonl
ranked_ground_truth_summary.json
ranking_metrics_by_unit.jsonl
```

Each ranked record preserves the common hunk and adds `model_ranking`, so outputs
from different models can be compared without changing ground-truth membership.

### Ranking metrics

By default the rank stage calculates metrics at `K = 1, 3, 5, 10`:

- absolute vulnerable-line hit count and hit rate at K;
- ranking coverage and line-level MRR, with unranked ground-truth lines counted
  as reciprocal rank zero;
- native-unit MRR and MAP;
- Hit@K, Recall@K, and NDCG@K;
- mean/median first vulnerable-line rank;
- IFA (non-vulnerable lines before the first vulnerable line);
- EXAM (first vulnerable rank divided by ranked candidate count);
- Recall@20% LOC and Effort@20% Recall.

The native ranking unit is a hunk chunk for DeepJIT/SimCom and a complete commit
for JITFine. This avoids pretending that independently attributed CNN chunks
share one comparable score scale. Native-unit metrics are observed-only;
absolute hit rates retain every eligible ground-truth line and treat an
unranked line as a miss.

Customize the cutoffs and effort fraction with:

```bash
-metric_top_k 1 3 5 10 20 -effort_fraction 0.2
```

Aggregate metrics and their definitions are embedded in
`ranked_ground_truth_summary.json`. Per-unit diagnostics are written to
`ranking_metrics_by_unit.jsonl`, while per-ground-truth-line rank and coverage
details are stored under `ranking_metrics.ground_truth_line_results` in the
summary.

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
