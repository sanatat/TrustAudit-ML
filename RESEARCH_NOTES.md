# Research interpretation and limitations

## Scope
The repository is an exploratory, CPU-only demonstration of metric-based change detection and selective reassessment. It is not a security product, formal verification tool, privacy/fairness certificate, reproduction of Atlas, or generalizable new algorithm.

## Primary outcomes, held-out combined mutations (Phase 6)
On 2 small public tabular datasets, 3 initialization seeds, 36 combined mutation cases (144 candidate assessment executions):

| Policy | Selected assessments | Reduction in count | Missed observed metric-change events |
|---|---:|---:|---:|
| Full | 144 | 0% | 0 |
| Declared | 102 | 29.2% | 42 |
| Fail-closed | 144 | 0% | 0 |
| Learned | 102 | 29.2% | 42 |
| Learned + guarded | 144 | 0% | 0 |

Phase 7 is an **offline replay on Phase 6 observations**, not independent new experimental validation. Thresholds 0.7 and 0.9 select 117/144 and 114/144 assessments and miss 6 and 9 observed events respectively. Threshold <=0.5 selects all assessments with 0 misses; at 1.01 no tests are selected and 109 observed events are missed. Do not interpret these as real-world failure rates.

## Validity threats
- Only two small datasets; overlapping recipes between folds; combination recipes held out versus single-component training examples.
- Ground truth is observable differences in four *finite-sample proxy metrics*; changes to those metrics do not prove real security vulnerabilities, and unchanged metrics do not prove safety.
- Subgroup gap is feature-based, not demographic fairness; Gaussian-noise accuracy is not certified robustness.
- Test-selection savings count assessments, not measured wall-clock savings.
- Wilson upper confidence estimates are exploratory under small and correlated samples; no formal risk bound.
- Recorded SHA256 hashes do not authenticate the provenance of original files or model execution.

## Suggested next research work
Independent datasets/recipe families; formalized validity conditions of evidence; safety-aware baselines; stronger, domain-specific assessment claims; confidence calibration, intervals and uncertainty propagation; explicit cost and false-negative tolerances under shift.
