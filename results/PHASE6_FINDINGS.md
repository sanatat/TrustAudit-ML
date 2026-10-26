# Phase 6 — Unseen combined mutations

- Two small public tabular datasets; initialization seeds 11, 23, 47.
- Six previously untested mutation recipes in each dataset/seed, comprising controlled combinations and an unknown input perturbation.
- Training sensitivity observations come exclusively from single-component Phase 2 runs on the other dataset; test labels are not used to decide selection.
- 36 held-out scenario runs × 4 measured metrics = 144 full executions.

| Policy | Tests selected | Reduction | Missed observed changes |
|---|---:|---:|---:|
| Full | 144 | 0% | 0 |
| Declared | 102 | 29.2% | 42 |
| Fail-Closed | 144 | 0% | 0 |
| Learned | 102 | 29.2% | 42 |
| Learned-Guarded | 144 | 0% | 0 |

This is a **negative result** for generalization and cost-saving: the learned policy does not improve on declared dependencies, and a guarded strategy requires full testing here. Missing observed metric changes do not equal security breaches; likewise zero misses is not proof of model safety. Development had previously used the same two datasets; independence is limited.

Future work: wider mutation families, more independent datasets, confidence-calibrated evidence invalidation and a nontrivial risk-aware partial-retention rule with explicit limitations.
