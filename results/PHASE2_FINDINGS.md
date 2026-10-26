# Phase 2 findings — TrustAudit-ML

Date: 2026-10-26. Code and generated results are in this repository.

## Experimental setup

Two datasets from scikit-learn (`breast_cancer` and `wine`), one small PyTorch
MLP per dataset, random seed 42 and six controlled mutations per dataset.
Metrics: accuracy, accuracy under fixed random noise (not certified robustness),
feature-based subgroup accuracy gap (not demographic fairness), and fraction
of predictions exceeding a confidence threshold.

### Mutations

1. Noise parameter adjustment
2. Confidence cutoff adjustment
3. Subgroup boundary adjustment
4. Test-input preprocessing scale adjustment
5. Trained-model weight perturbation
6. An untracked change to model input values

Changes are scored against a baseline model using *full re-evaluation*. We
compare the observed metric differences with the tests chosen using manually
specified dependencies. Any unrecognized changed component triggers all tests
under a separate conservative, fail-closed policy.

## Observed results

- 12 dataset × mutation runs
- 21 individual observed metric-change events
- 8 metric-change events were not covered by declared-only selection
- 0 such events were missed by a fail-closed selection policy
- Declared-only selection chose 45.83% of all potential tests by count
- 9 unit tests passed

## Interpretation

The deliberately untracked changes caused observed model-metric differences
while the dependency selector scheduled no tests. Treating unrecognized
changes as reasons to rerun all tests avoids this failure in this experiment.
The benefit is therefore principally a **negative result about incomplete
provenance**, not evidence of superior selective verification.

## Important limitations

- Small datasets and toy MLPs; no robustness, privacy or security guarantee.
- Ground truth is a finite-sample metric comparison, **not** a complete oracle
  for invalid assurance claims.
- Dependent tests and safety claims cannot generally be inferred from metric
  changes alone.
- No dependency-learning algorithm or novel cryptographic protocol exists here.
- The safety fallback necessarily increases work for unknown changes.
- Timings are measured, but at micro/millisecond scales and without repeated
  controlled benchmarking, so no credible measured speedup is claimed.
- A real proof-carrying audit would require signed attestations and verified
  claims, neither of which is implemented in this prototype.

## Proposed next research question

Can we infer enough missing dependencies to avoid full re-testing while
conservatively bounding the risk of reusing stale evidence? This is a proposed
research question, not a demonstrated solution.
