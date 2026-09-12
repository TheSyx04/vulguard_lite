# Implementation Plan: VIC-first Qualitative Analysis for Classification and Prioritization

## 1. Scope and research questions

The primary analysis uses VIC cases for **both Linux and OpenSSL**. VFC-only and paired VIC–VFC analyses are optional supplements, not prerequisites or required outputs of the primary pipeline.

Research question: How do labeling strategies change the detection and inspection priority of independently identified vulnerability-introducing commits?

Evaluate where `data0-1` is reliable, practically tied, or unsuccessful. Do not assume its superiority as a required finding. VFC strategy remains an experimental factor even when every inspected case is a VIC: hold VIC strategy fixed and compare the same commit across VFC configurations.

Primary views:

- Classification: detection/misses at the stored budget-selected threshold for label-consistent VICs; retain variation across seeds, models, and configurations.
- Prioritization: full-test-set rank/percentile, selection across budgets, and first reliable detection budget.

These views share the same selection event when derived from the same threshold. They are not independent evidence. VIC-only cases cannot measure false-positive behavior, specificity, full-test MCC, or PR-AUC. Full-test metrics may provide separate context, but must not be reconstructed from this positive case set.

This document describes planned work; implementation and human annotation remain separate tasks.

## 2. Local evidence and provisional population

The following findings describe the locally inspected artifacts, not a newly queried remote revision.

| Project | Workbook VICs | VICs with predictions in extracted data | Label-consistent provisional primary VICs |
|---|---:|---:|---:|
| Linux | 28 | 26 | 24 |
| OpenSSL | 12 | 12 | 12 |

Linux VIC prefixes `622ceadd` and `e8c08826` lack predictions in the inspected extracted files. Linux VIC prefixes `f18ec422` and `8a4cd82d` have experimental label 0 despite their workbook VIC role. Resolve these against full SHA identifiers, source relations, and common test inputs before freezing the population.

Do not relabel these two commits automatically. Until resolved, keep them in a label-discrepancy audit and out of primary TP/FN denominators. Any separately reported prioritization observations must explicitly identify the label discrepancy. Do not silently discard them.

The provisional primary population is **24 Linux + 12 OpenSSL = 36 VICs**, subject to validation. There are 38 VICs with scores if the two discrepant cases are counted. These are distinct denominators, not interchangeable counts.

Expected design remains:

- Linux: 16 configurations, VFC0–3 × VIC0–3.
- OpenSSL: 12 configurations, VFC0–3 × VIC1–3; VIC0 is structurally unavailable.
- Six models: TLEL, LR, LAPredict, DeepJIT, SimCom, JITFine.
- Five seeds, only `run_1`; no fallback to another run.
- Budgets: 0.05, 0.075, 0.10, 0.15, 0.20.

If the provisional 36-case population is confirmed and coverage is complete:

- Unique commit-config cells: `24×16 + 12×12 = 528`.
- Unique score observations: `528×6×5 = 15,840`.
- Budget observations: `15,840×5 = 79,200`.

If both discrepant Linux VICs are subsequently verified eligible, the corresponding maxima become 560, 16,800, and 84,000. Calculate expected counts from the frozen eligibility manifest, rather than hard-coding either outcome.

### Supplemental VFC coverage

OpenSSL is not currently suitable for a full-grid paired analysis. Inspection of cached `run_1`, budget 5%, all five seeds showed:

- DeepJIT `openssl_0_1` and `openssl_2_1`: 7,340 output commit IDs, including all 12 workbook VFC roles (11 unique VFCs).
- Other model/config groups: 7,291 output commit IDs, including all 12 VICs but none of those VFCs.
- The larger set contains the smaller set plus 49 commits.

The extracted all-budget CSV confirms corresponding VFC missingness. This is an output-population difference, not simply an extraction failure. The historical cause remains unresolved. Audit population membership across all budgets before comparing full-test ranks or metrics. Even VICs shared by both populations have ranks and thresholds influenced by the differing populations.

Report native-population ranks with population identifiers and sizes. A common-intersection rank may be an explicitly separate sensitivity analysis, never substituted for original experimental selection or threshold. Flag affected comparisons and show whether conclusions persist when the affected groups are omitted.

## 3. Input strategy: local first, network only for identified gaps

Source precedence:

1. `Data_line_vul.xlsx`: VIC/VFC role mapping, CVE, workbook row, and vulnerable-line references.
2. Existing extracted CSVs in `output/`: quick coverage checks and reconciliation.
3. Existing Hugging Face cache: authoritative cached full score files for full SHAs, source provenance, and full-population ranking.
4. Narrow downloads only for a documented missing input.

The inspected cache snapshot is `532708085bc199fad101d8ea293d03caa5eb43cf` for dataset `TheSyx/vulguard_lite`. It contains 12,600 score files, including 4,200 `run_1` files (2,400 Linux; 1,800 OpenSSL). Counts alone do not establish schema or population consistency.

Default to this immutable cached snapshot. Do not require querying the newest remote revision or listing the entire remote tree. Record local file hashes and the snapshot revision. An explicitly chosen snapshot refresh is a separate operation; never silently mix revisions.

Useful existing files:

- `output/linux_all_models_scores_labeled.csv`
- `output/openssl_all_models.csv`
- `output/linux_*_scores_full.csv`
- `scripts/extract_line_vul_scores.py`
- `scripts/check_prediction_labels.py`

Despite their names, the extracted `*_scores_full.csv` files contain workbook-selected cases, not the complete test population. They omit full commit IDs, original source path/revision, rank, and threshold. Never derive full-test rank from them. Do not concatenate combined and per-model CSVs as independent observations.

Resolve each workbook prefix uniquely within the project, using cached full scores and verified metadata. Preserve full hashes internally. In particular, historical exclusion prefix `abc7b3f` appears as `abc7b3f1` in the workbook-derived CSV; use resolved identity, not literal short-string equality.

Potential missing inputs include threshold JSONs or summary metrics, common test inputs for label verification, and selected-case metadata/diffs. First inspect all configured local roots. Download exact paths only when needed, record the reason and revision, and reuse downloaded material. Never delete pre-existing caches or user files. Cleanup applies only to explicitly tracked temporary files created by the current run.

## 4. Configuration and data contracts

Use a versioned configuration with projects/config availability, model aliases, seeds, budgets, selected run, source roots, pinned revision, population policy, pattern cutoffs, practical effect margins, and numeric tolerances.

Required settings include:

```yaml
analysis_scope: vic_primary
selected_run: 1
decision_mode: budget_threshold
input_policy: local_first
refresh_remote: false
population_policy: label_consistent_vic
supplements:
  vfc: false
  paired_vic_vfc: false
```

### Eligibility manifest

One row per unique project/VIC, with full SHA, source workbook row, CVE if available, expected benchmark label, observed experiment label, coverage status, eligibility status, reason code, and provenance. Retain all candidate VICs in this audit manifest; downstream primary tables contain only eligible cases. Distinguish missing prediction, label discrepancy, and structural unavailability.

The older prohibition on keeping exclusion rows is superseded: auditable eligibility and discrepancy records are required. Do not include VFC rows in the primary case manifest.

### Score table

Key: `project × commit_id × config_id × model × seed × run_id`, with `run_id=1`.

Store raw score, experiment label, benchmark label, VFC/VIC strategy IDs, source file, source row, source revision, and source hash. Validate that score is invariant across budgets before collapsing budget files. If it varies, stop and resolve the evaluation contract; do not average it away.

### Budget table

Key: score key plus budget. Store observed prediction/selection, score, threshold if verified, rank, rank percentile, population size/identifier, and separate provenance for selection, threshold, and rank.

Missing thresholds or ranks may be null during discovery; they block only downstream outputs requiring them. Stored predictions can support preliminary coverage/detection checks. Do not claim threshold verification until threshold metadata is available.

### Full-run metric table

Keep metrics separate from case observations. PR-AUC has no budget dimension after verifying repeated source values agree. Preserve population identity and provenance. Do not calculate case-set MCC for the VIC-only primary analysis.

## 5. Evaluation and aggregation rules

Repository code in `evaluating.py` uses strict `probability > threshold`. Its calibrated branch selects the threshold maximizing VDR subject to marked-commit fraction ≤ budget, breaking ties by lower marked fraction and then higher threshold. It does not implement simple top-k truncation.

Reuse stored decisions and verified thresholds. Do not replace experimental selection with top-k, boundary truncation, or a new decision mode. Document actual calibration provenance: the current calibrated code uses labels from the evaluated population. Verify whether the saved artifacts used this branch before characterizing their protocol.

For rank, use full corresponding score populations, with an explicit tie convention and percentile definition. Preserve experimental selection independently of the ranking tie convention.

Aggregate in this order:

1. Seeds within each model: detection rate, score dispersion, rank summaries, denominator.
2. Equal-weight model-level rates: mean detection rate, model vote rate, model heterogeneity, actual model/seed coverage.
3. Cross-budget summaries only after retaining all five budget points.

Do not weight a model more heavily because it has more available seeds. Match observations before contrasts. Under partial coverage, use a documented common support and report exclusions; never compare independently averaged unmatched cells.

Prioritize detection rates and within-model rank contrasts. Raw probabilities across different models are not assumed calibrated to the same scale. Keep score gaps within model and seed before summarizing their direction/distribution.

Define reliable detection, seed-majority votes, and practical parity explicitly. Numerical epsilon is not the practical-effect margin. Normalize trapezoidal budget AUC over the observed interval [0.05, 0.20], not an invented [0, 1] curve.

## 6. Sensitivity and comparisons

Scan every valid configuration before selecting cases.

- VFC sensitivity: change VFC with VIC fixed.
- VIC sensitivity: change VIC with VFC fixed.
- Preserve model/seed pairing and budget.
- Compare `0-1` with `1-1`, `2-1`, and `3-1` for fixed VIC1.
- Compare `0-1` with `0-3` for a fixed-VFC contrast.
- Compare `0-1` with the median valid alternative descriptively, including parity and failures.

Retain range-based sensitivity as descriptive. For OpenSSL, explicitly acknowledge four VFC levels versus three VIC levels; supplement ranges with mean absolute pairwise differences and a documented equal-cardinality sensitivity check. Do not claim that an unadjusted range comparison alone establishes greater VFC influence.

Report each project separately. Any pooled number must declare commit or project weighting. Score/rank comparisons involving differing test populations require the population checks in Section 2.

## 7. Behavioral screening and manual case selection

Screening labels are quantitative, not inferred code mechanisms:

- stable detection;
- persistent miss;
- VFC-sensitive detection;
- VIC-sensitive detection;
- VFC2-favorable exception;
- `data0-1` advantage, practical parity, or failure;
- model/seed-dependent behavior.

Freeze thresholds and selection scores before viewing code. Report cutoff sensitivity. Allow multiple screening labels with deterministic primary strata for sampling.

Select typical, strong, boundary, and deviant cases where available, with representation from both projects. Set a configurable annotation sample size and deterministic deduplication/fill rules; do not require a distinct case for every overlapping stratum. Do not select only `data0-1` wins.

Packets contain blinded case IDs, project, commit message, diff/context, CVE references when available, and evidence fields. Hide model/config performance during initial coding. Fixing-commit material may provide context without requiring its prediction.

Pilot open coding, then freeze a codebook. Use independent annotators where feasible, report agreement, and retain original annotations and adjudication history. Regeneration must never overwrite completed fields. Unblind only after coding is frozen.

Mechanism counts describe the annotated sample. They are not prevalence estimates for all VICs when sampling deliberately favors strata or extreme cases.

## 8. Primary outputs

- Input inventory, pinned-source manifest, eligibility/label-discrepancy audit, and coverage report.
- Normalized eligible VIC scores and budget observations.
- Per-model, per-config, and per-budget VIC profiles.
- Matched VFC/VIC sensitivity and `data0-1` contrasts.
- All-config VIC heatmap, with structural and experimental missingness distinguished.
- Detection-versus-budget curves for typical and deviant cases, with seed/model variation identified.
- VIC sensitivity comparison and `data0-1` advantage distribution.
- Selected-case table, blinded packets, versioned annotations, and evidence-backed mechanism descriptions.

Every figure has companion data and a caption stating population, denominator, weighting, and uncertainty unit. Rank-based figures identify their full-test population. No paired score-gap table, VFC false-positive curve, or 37-pair figure is required for primary completion.

Supplemental Linux VFC or paired analysis requires separate eligibility/label/coverage checks. OpenSSL paired results are restricted to verified available groups and cannot support full-grid claims. Shared VFCs must never inflate unique-commit denominators.

## 9. Implementation sequence and acceptance gates

1. Read-only local/cache discovery; inventory exact missing inputs without mandatory remote access.
2. Resolve identities, Linux label discrepancies, and output-population differences; freeze primary eligibility policy.
3. Build configuration and audit manifests.
4. Normalize cached scores and stored decisions; obtain only missing required metadata.
5. Validate coverage, labels, source populations, budget invariance, thresholds, and ranking.
6. Aggregate seeds then models; calculate matched contrasts and sensitivity.
7. Freeze screening rules and select blinded VIC cases.
8. Generate annotation packets; pause for human coding.
9. Ingest/adjudicate annotations and generate primary report assets.
10. Optionally assess supplemental VFC/paired analysis as separately scoped work.

Unknown fields block dependent analyses, not unrelated discovery. `--allow-partial` cannot bypass ambiguous identity, unresolved label handling, or an unknown decision interpretation. Save explicit actual denominators whenever partial coverage is allowed.

Use deterministic, restartable commands with configured input/output roots. Never overwrite raw inputs or annotations. Preserve production validation logic and run appropriate checks for uniqueness, prefix resolution, population membership, score invariance, strict threshold comparison, monotonic selection where applicable, matched support, model weighting, deterministic selection, and annotation preservation.

Definition of done: the eligible VIC analysis for both projects is reproducible, every case/contrast traces to source data, all unresolved limitations are explicit, and qualitative claims remain within the selected case evidence. Supplemental VFC coverage does not block primary completion.
