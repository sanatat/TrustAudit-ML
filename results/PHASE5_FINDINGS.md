# Phase 5 exploratory findings

Protocol: leave-one-dataset-out evaluation using 2 public tabular datasets.
Only training-dataset observations are used to learn dependency links;
held-out observed metric differences are read only for retrospective scoring.
All six mutation labels recur across datasets: this does NOT test unknown
mutation-family generalization.

| Policy | Executions of 48 | Reduction | Missed observable metric events |
|---|---:|---:|---:|
| Full | 48 | 0% | 0 |
| Declared | 22 | 54.2% | 8 |
| Fail-closed | 30 | 37.5% | 0 |
| Learned | 30 | 37.5% | 0 |
| Learned guarded | 30 | 37.5% | 0 |

**Interpretation:** empirical links recovered previously missing test
relationships across two dataset groups. They did not reduce test executions
relative to the conservative baseline on this benchmark. No evidence of
statistical generalization or assurance guarantees. Hashes identify an input
file, but do not prove it was produced honestly.
