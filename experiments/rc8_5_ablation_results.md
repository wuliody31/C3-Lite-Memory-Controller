# RC8.5 Component Ablation Results

## Frozen execution

- Job: 36152
- Commit: cbff8bb4cef9f244a15564be480a119d0af2ba03
- Model: Qwen/Qwen3-8B
- Quantisation: 4-bit NF4
- Dataset: Dataset A RC8.4, 60 questions
- Variants: 4
- Total predictions: 240
- Bootstrap resamples: 50,000
- Correction: Holm-Bonferroni across eight primary comparisons

## Main findings

### Route planner

Removing the route planner reduced Answer Token F1 from 0.4184 to
0.3899 and Evidence F1 from 0.5071 to 0.3806.

The Evidence F1 reduction was statistically significant:

- Mean difference: 0.1265
- 95% CI: [0.0804, 0.1768]
- Holm-adjusted p: 0.000160

The Answer Token F1 difference was not significant after correction:

- Mean difference: 0.0285
- 95% CI: [0.0066, 0.0514]
- Holm-adjusted p: 0.076918

### Evidence selector

Removing the structured evidence selector reduced Answer Token F1
from 0.4184 to 0.3567.

The reduction was statistically significant:

- Mean difference: 0.0617
- 95% CI: [0.0272, 0.0976]
- Holm-adjusted p: 0.004900

Without the selector, evidence precision decreased from 0.5514 to
0.4100, evidence recall increased from 0.5106 to 0.6411, and mean
evidence usage increased from 2.65 to 4.65.

### Conflict handling

Removing conflict handling did not significantly change aggregate
Answer Token F1 or Evidence F1. Most paired observations were
unchanged. A separate conflict mechanism audit is required before
drawing conclusions about detector and resolver effectiveness.

### Coverage and confidence gate

Bypassing the gate did not reduce lexical answer quality, but
abstention correctness decreased from 0.9500 to 0.9333. The component
is therefore treated as a reliability and abstention-control
mechanism rather than a general answer-quality mechanism.

## Interpretation

The route planner primarily improves evidence focus. The evidence
selector primarily improves the usefulness of the final prompt
context and generated answer. Conflict handling and confidence
control perform specialised reliability functions that are not fully
represented by aggregate lexical metrics.
