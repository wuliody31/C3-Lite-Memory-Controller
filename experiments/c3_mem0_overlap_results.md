# C3–Mem0 Pilot-200 Candidate Complementarity

## Experimental scope

This analysis compares the evidence coverage of the existing C3
retrieval pipeline with a provenance-preserving Mem0 dense retrieval
backend on the same fixed LoCoMo Pilot-200 question set.

All evidence was evaluated using the corrected source-equivalent
LoCoMo v0.2 annotations.

## Main results

| Comparison | C3 Hit | Union Hit | Delta Hit | C3 Recall | Union Recall | Delta Recall | Mem0-only Hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| C3 selected + Mem0 top-5 | 0.3800 | 0.5900 | +0.2100 | 0.3073 | 0.5044 | +0.1971 | 42 |
| C3 selected + Mem0 top-20 | 0.3800 | 0.7150 | +0.3350 | 0.3073 | 0.6130 | +0.3057 | 67 |
| C3 ranked-10 + Mem0 top-5 | 0.5450 | 0.6450 | +0.1000 | 0.4475 | 0.5546 | +0.1071 | 20 |
| C3 ranked-10 + Mem0 top-20 | 0.5450 | 0.7250 | +0.1800 | 0.4475 | 0.6237 | +0.1762 | 36 |
| C3 raw-30 + Mem0 top-5 | 0.6250 | 0.7000 | +0.0750 | 0.5193 | 0.6059 | +0.0867 | 15 |
| C3 raw-30 + Mem0 top-20 | 0.6250 | 0.7500 | +0.1250 | 0.5193 | 0.6608 | +0.1416 | 25 |

## Interpretation

Mem0 is complementary even to the original C3 raw retrieval pool.
Mem0 top-20 recovered relevant evidence for 25 questions for which
the C3 raw-30 pool contained no gold-equivalent evidence.

This demonstrates that Mem0 is not merely duplicating candidates
already retrieved by C3. A hybrid candidate-generation architecture
is therefore justified.

The union pool should not be passed directly to the language model.
Mem0 top-20 has high recall but low precision, so the combined
candidates must still pass through the existing C3 ranking, gating,
budgeting and evidence-selection stages.

## Next experiment

The next experiment will construct a hybrid candidate provider:

1. retrieve the existing C3 raw candidate pool;
2. append source-distinct Mem0 top-20 candidates;
3. record candidate provenance as C3, Mem0 or both;
4. reuse the frozen C3 ranker, gate and evidence selector;
5. retain the existing evidence-count and token-budget constraints.

The first evaluation will be controller-only on Pilot-200. Qwen3
answer generation will be run only if hybrid evidence selection
improves over the frozen C3 controller.
