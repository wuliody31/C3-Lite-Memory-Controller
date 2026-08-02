# LoCoMo Gold Evidence Mapping Repair

## Problem

The initial LoCoMo adapter did not fully support all formatting
variants used by the official evidence and observation-source
annotations.

The official release included:

- multiple dialogue IDs separated by semicolons inside one string;
- multiple dialogue IDs separated by whitespace inside one string;
- zero-padded turn identifiers such as `D30:05`;
- duplicate evidence identifiers inside one QA annotation.

## Affected QA records

Six QA records changed between adapter v0.1 and v0.2:

- q_locomo_conv_26_0038
- q_locomo_conv_49_0032
- q_locomo_conv_49_0039
- q_locomo_conv_49_0047
- q_locomo_conv_50_0006
- q_locomo_conv_50_0070

The first four records contained composite reference strings. The
fifth contained the duplicate evidence sequence
`D4:5, D4:5, D5:5`. The sixth used the zero-padded identifier
`D30:05`, which was canonicalised to `D30:5`.

## Semantic observation annotations

Five semantic observations also contained composite source-reference
strings. Adapter v0.2 correctly expanded these references into
namespaced episodic source IDs.

The repaired semantic observations were:

- s_locomo_conv_44_obs_s26_andrew_004
- s_locomo_conv_48_obs_s22_deborah_005
- s_locomo_conv_48_obs_s22_jolene_003
- s_locomo_conv_49_obs_s04_sam_005
- s_locomo_conv_50_obs_s12_dave_004

## Repair behaviour

The adapter now:

1. extracts every `D<session>:<turn>` identifier from a reference
   string;
2. supports semicolon and whitespace-separated IDs;
3. canonicalises zero-padded numerical components;
4. preserves source order;
5. removes duplicate identifiers;
6. reports unresolved non-empty references during strict validation.

## Generation invariance

The audit confirmed:

- generation-field changes: zero;
- memory core-field changes: zero;
- memory identifier changes: zero;
- memory text changes: zero;
- other metadata changes: zero.

Only gold evidence mappings, semantic `source_ids` and their associated
raw-source metadata changed.

Therefore, the frozen 7,944 predictions from Slurm Job 36180 were
re-scored without rerunning Qwen3 generation.

## Final evaluation population

- Total questions: 1,986
- Answerable questions: 1,542
- Genuinely empty official evidence annotations: 4
- Evidence-scored questions: 1,982
