# RC8.6 Conflict Mechanism Audit

## Frozen inputs

- Full C3 job: 36131
- No-conflict ablation job: 36152
- Dataset: Dataset A RC8.4
- Conflict-labelled questions: 7

## Audit summary

- Questions with detected conflicts: 1/7
- Questions with changed selected evidence: 0/7
- Questions with changed answers: 0/7
- Questions with changed decisions: 0/7

## Interpretation

Conflict handling did not produce a measurable aggregate improvement on Dataset A. The audit is used to distinguish between detection failure, functional overlap with ranking and selection, and generator insensitivity. No controller parameters were changed after the formal evaluation.

## Per-question audit

| Question | Detected groups | Evidence changed | Answer changed | Decision changed |
|---|---:|---:|---:|---:|
| q_user01_003 | 0 | False | False | False |
| q_user01_007 | 2 | False | False | False |
| q_user01_020 | 0 | False | False | False |
| q_user02_003 | 0 | False | False | False |
| q_user02_017 | 0 | False | False | False |
| q_user03_003 | 0 | False | False | False |
| q_user03_019 | 0 | False | False | False |
