# LoCoMo Gold Evidence Mapping Repair

## Scope

Five official LoCoMo QA records contained multiple dialogue IDs inside
a single string or used a zero-padded turn identifier.

Affected records:

- q_locomo_conv_26_0038:
  `D8:6; D9:17`
- q_locomo_conv_49_0032:
  `D9:1 D4:4 D4:6`
- q_locomo_conv_49_0039:
  `D22:1 D22:2 D9:10 D9:11`
- q_locomo_conv_49_0047:
  `D21:18 D21:22 D11:15 D11:19`
- q_locomo_conv_50_0070:
  `D30:05`, canonicalised to `D30:5`

## Repair

The adapter was updated to extract all dialogue identifiers from each
evidence string and canonicalise numeric session and turn components.

The repair changed only `supporting_memory_ids`. It did not alter:

- question text;
- gold answers;
- answerability labels;
- memory text;
- memory IDs;
- controller behaviour;
- retrieval;
- generation prompts;
- model outputs.

Therefore, the frozen 7,944 formal predictions were re-scored without
rerunning generation.

## Final scoring population

Four QA records contain genuinely empty official evidence annotations.
The corrected evidence-level evaluation therefore contains 1,982
questions.
