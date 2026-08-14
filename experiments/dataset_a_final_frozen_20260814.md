# Dataset A Final Frozen Evaluation

Status: FINAL / FROZEN

Method comparison:

- RC8.3: `c3-rc8.3-paper-baseline`
- C3-v3: `c3-v3-active-final-candidate`
- Backbone: Qwen3-8B
- Questions: 60
- Generations: 120
- GPU job: 39861

Primary result:

- RC8.3 Answer F1: 0.35544
- C3-v3 Answer F1: 0.35539
- Mean paired delta: -0.00005
- Input tokens: -10.9%
- Output tokens: -14.0%
- Latency: -15.9%

Conditional analysis:

- Changed evidence without annotated-gold removal:
  Delta F1 = +0.04387, not statistically significant.
- Annotated-gold removal:
  Delta F1 = -0.06824, 95% CI [-0.11372, -0.02406],
  paired sign-flip p = 0.00788.

Dataset A is frozen after this evaluation and must not be used
for further controller tuning. Subsequent validation is performed
on LoCoMo.
