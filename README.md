# TrustAudit-ML

**Change-aware ML assurance auditing — reproducible CPU-only feasibility study (Phases 1–7).**

This repository tests when prior machine-learning assessment evidence may be inappropriately reused after pipeline changes. It compares full reassessment with declared dependencies, a fail-closed policy, learned associations, and an uncertainty-aware threshold policy. **This is an exploratory pilot, not a validated security or safety solution.**

## Main finding

In held-out *combinations* of changes, simple learned/declared dependency selectors missed observed metric changes (42 events out of 144 candidate metric evaluations), while fail-closed reassessment selected all 144 assessments. An offline uncertainty-aware replay exposes an explicit count–miss trade-off, but no tested threshold achieved both fewer assessments and zero missed observable events. These outcomes are metric proxies, not security breaches.

| Stage | Key observation |
|---|---|
| Phase 2 | 12 actual mutation cases; 8 missed observable metric events under declared-only selection |
| Phase 3–4 | Fail-closed selection: 30 of 48 tests; 0 missed observable events in that small set |
| Phase 5 | Dataset-held-out learning matched fail-closed: 30 of 48 assessments |
| Phase 6 | 36 combined mutation cases, 3 seeds, 2 datasets; learned and declared each missed 42 events |
| Phase 7 | Offline replay: at threshold 0.70, 117/144 tests selected, 6 observable events missed; at 0.90, 114/144, 9 missed |

## Setup (Linux / CPU)

Python 3.10+ recommended. A CPU-only PyTorch installation avoids the large NVIDIA CUDA dependencies frequently pulled in by the default Linux wheel. Follow the [official PyTorch install selector](https://pytorch.org/get-started/locally/) for your OS.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the experiments

Run from the repository root:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python trustaudit.py --output results/pilot.json
python real_mutations.py
python phase3_audit.py
python phase4_uncertainty.py
python phase5_discovery.py
python phase6_unseen.py
python phase7_adaptive.py
python -m pytest -q
```

All experimental outputs are JSON files under `results/`. Saved outputs are included as illustrative reference runs; generated output can vary across environments. **The user independently reported 31/31 tests passing on Linux Mint for v7.** The results for Phase 7 replay are derived from Phase 6 measurements, not a new prospective study.

For details, see [RESEARCH_NOTES.md](RESEARCH_NOTES.md) and the phase findings in [`results/`](results/).

## Methods at a glance

- Datasets: scikit-learn breast cancer and binarized wine classifications; small PyTorch MLPs.
- Metrics: accuracy, Gaussian-noise accuracy, feature-based subgroup accuracy gap, and proportion of confident predictions (>=0.8).
- Mutations: actual input, weight, preprocessing/evaluation parameter changes; later combined changes and an unseen component.
- Policies: full testing, declared dependency links, fail-closed fallback, empirically learned links, and uncertainty-aware test selection.
- Evaluation: select tests from training evidence, then score retrospectively using fully observed metrics on the held-out cases.

## Limitations and responsible interpretation

No method here proves privacy, fairness, security, robustness or the validity of reused assurance claims. The metrics are **finite-sample proxies**, the datasets are small, confidence scores are exploratory, and an assessment-count reduction is **not** a measured runtime reduction. There is no third-party attestation or formal verification. This repo should be presented as *preliminary experimental research* only.

## Files

- `trustaudit.py` — baseline toy models and synthetic scenarios
- `real_mutations.py` — real controlled changes and empirical metrics
- `phase3_audit.py` — post-hoc auditing report
- `phase4_uncertainty.py` — declared uncertainty fallback
- `phase5_discovery.py` — empirical dependency selection across datasets
- `phase6_unseen.py` — combined mutation experiments with three seeds
- `phase7_adaptive.py` — threshold sweep, offline held-out replay
- `tests/` — regression tests
- `results/` — recorded outputs and findings

## License and citation

MIT License. See `CITATION.cff` for software citation metadata. Author: Sana Tat.
