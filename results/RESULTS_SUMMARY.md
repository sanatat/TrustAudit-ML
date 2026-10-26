# Pilot results (executed)

Two PyTorch MLP classifiers were trained on CPU and evaluated with seed 42.

| Dataset | Accuracy | Accuracy under synthetic Gaussian noise | Feature-based subgroup accuracy gap | Fraction confidence >= 0.8 |
| --- | ---: | ---: | ---: | ---: |
| Breast cancer | 0.97203 | 0.97203 | 0.05556 | 0.96503 |
| Wine (binary) | 0.97778 | 0.97778 | 0.04348 | 0.95556 |

For eight manually defined synthetic change scenarios, the dependency-based selector chose 13 of 32 possible test executions, representing 59.38% fewer selected tests than always re-running all four. In one intentionally constructed hidden-dependency scenario, it missed one oracle-required assessment. The oracle is synthetic and this does not establish actual model safety or measured runtime savings.

All six automated tests passed in the initial environment. The raw, generated report and versions are in `pilot.json`.

**Next research step:** real pipeline changes, empirical risk outcomes, conservative handling of uncertain dependencies, measured costs, independent evidence validation, and comparisons with prior academic baselines.
