# LoCoMo Formal External Validation Results

## Execution

- Dataset: official LoCoMo release
- Conversations: 10
- Questions: 1,986
- Answerable questions: 1,542
- Predictions: 7,944
- Methods: no memory, simple retrieval, all memory and C3
- Backbone: Qwen/Qwen3-8B
- Quantisation: 4-bit NF4
- Formal Slurm array job: 36180
- Controller tuning after Dataset A: none
- Controller tuning after LoCoMo smoke or pilot: none

## Overall metrics

| Method | Answer F1 | Source Evidence Precision | Source Evidence Recall | Source Evidence F1 | Mean selected evidence |
|---|---:|---:|---:|---:|---:|
| No memory | 0.0276 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| Simple retrieval | 0.1424 | 0.1421 | 0.5177 | 0.2162 | 5.00 |
| All memory | 0.1354 | 0.0431 | 0.6002 | 0.0783 | 19.87 |
| C3 | 0.1236 | 0.2176 | 0.3899 | 0.2712 | 2.03 |

## Paired bootstrap analysis

The analysis used 50,000 paired bootstrap resamples. Two-sided
percentile confidence intervals were reported. Holm-Bonferroni
correction was applied separately to the pre-specified quality and
efficiency families.

### Answer Token F1

- C3 versus simple retrieval:
  delta = -0.0187,
  95% CI [-0.0248, -0.0127],
  Holm-adjusted p < .001.

- C3 versus all memory:
  delta = -0.0118,
  95% CI [-0.0182, -0.0053],
  Holm-adjusted p < .001.

- C3 versus no memory:
  delta = +0.0960,
  95% CI [+0.0886, +0.1036],
  Holm-adjusted p < .001.

### Source-equivalent Evidence F1

- C3 versus simple retrieval:
  delta = +0.0551,
  95% CI [+0.0448, +0.0652],
  Holm-adjusted p < .001.

- C3 versus all memory:
  delta = +0.1929,
  95% CI [+0.1800, +0.2062],
  Holm-adjusted p < .001.

- C3 versus no memory:
  delta = +0.2712,
  95% CI [+0.2570, +0.2852],
  Holm-adjusted p < .001.

### Memory budget

- C3 selected 2.9663 fewer memories than simple retrieval,
  95% CI [-2.9753, -2.9567],
  Holm-adjusted p < .001.

- C3 selected 17.8338 fewer memories than all-memory prompting,
  95% CI [-17.8741, -17.7895],
  Holm-adjusted p < .001.

## Interpretation

C3 generalised successfully as a selective evidence-orchestration
mechanism. It achieved the highest source-equivalent Evidence F1 and
precision while using substantially fewer memories.

However, C3 did not outperform the broader retrieval baselines on
lexical Answer Token F1. The results therefore demonstrate a
precision-recall-generation trade-off rather than unconditional
end-to-end superiority.

The confidence gate also showed limited transfer to LoCoMo's
unanswerable questions and should be treated as an external-validation
limitation.

## Gold evidence repair

Six official QA records required composite-reference expansion or
numeric identifier canonicalisation. Four records contained genuinely
empty evidence lists, leaving 1,982 records for evidence-level
evaluation.

The repair changed only evaluation annotations and source-equivalence
mappings. The frozen model predictions from Slurm Job 36180 were not
regenerated.
