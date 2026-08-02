# LoCoMo Gold Evidence Mapping Repair

## Problem

The initial LoCoMo adapter did not fully support all formatting
variants used by the official evidence annotations.

The official dataset included:

- multiple dialogue IDs separated by semicolons inside one string;
- multiple dialogue IDs separated by spaces inside one string;
- zero-padded turn identifiers such as `D30:05`;
- a partially mapped QA record in which some evidence was retained
  while another non-canonical identifier was omitted.

## Affected QA records

Six QA records changed between adapter v0.1 and v0.2:

- q_locomo_conv_26_0038
- q_locomo_conv_49_0032
- q_locomo_conv_49_0039
- q_locomo_conv_49_0047
- q_locomo_conv_50_0006
- q_locomo_conv_50_0070

Five records were previously missing all mapped evidence. One record
was only partially mapped and was therefore not detected by the
initial empty-evidence audit.

## Repair

The adapter now:

1. extracts every `D<session>:<turn>` identifier from an evidence
   string;
2. supports semicolon, whitespace and other separator formats;
3. canonicalises zero-padded numerical components;
4. preserves source order;
5. removes duplicate identifiers;
6. fails strict validation when a non-empty reference cannot be
   resolved.

The same parser also repaired official semantic-observation source
annotations.

## Generation invariance

The repair did not alter:

- question text;
- gold answers;
- answerability labels;
- memory identifiers;
- memory text;
- memory types;
- model prompts already used in Job 36180;
- selected memory IDs stored in the frozen predictions;
- generated answers.

Only gold evidence mappings and source-equivalence annotations changed.

Consequently, the 7,944 frozen Qwen3 predictions were re-scored
without rerunning generation.

## Final scoring population

Four official QA records contain genuinely empty evidence lists.

The corrected evidence evaluation therefore contains:

- total QA records: 1,986;
- evidence-scored records: 1,982;
- answerable records: 1,542.
