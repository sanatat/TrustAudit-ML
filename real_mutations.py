"""Real, controlled model/config mutations and measured selective test execution.

WARNING: A passed metric comparison does not prove safety. 'Observable change' is
only a comparison of these four finite-sample metrics, with a fixed tolerance.
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from trustaudit import (SEED, TESTS, DEPENDENCIES, make_dataset, train_model,
                        predictions, select_tests, sha256_bytes)


def measure_one(name, model, x, y, *, noise_scale=.12, conf_threshold=.8,
             group_cut=0.5):
    """Compute exactly one metric; wall time includes model inference."""
    start = time.perf_counter()
    if name == 'robustness':
        noise = np.random.default_rng(SEED).normal(0, noise_scale, x.shape).astype(np.float32)
        p = predictions(model, x + noise)
        value = float(np.mean(p.argmax(1) == y))
    else:
        p = predictions(model, x)
        if name == 'accuracy':
            value = float(np.mean(p.argmax(1) == y))
        elif name == 'confidence':
            value = float(np.mean(p.max(axis=1) >= conf_threshold))
        elif name == 'subgroup_gap':
            threshold = float(np.quantile(x[:, 0], group_cut))
            group = x[:, 0] >= threshold
            if group.all() or (~group).all():
                group = np.arange(len(x)) % 2 == 0
            value = abs(float(np.mean(p.argmax(1)[group] == y[group])) -
                        float(np.mean(p.argmax(1)[~group] == y[~group])))
        else:
            raise ValueError(name)
    return round(value, 8), time.perf_counter() - start


def evaluate(model, x, y, names=TESTS, **kw):
    metrics, seconds = {}, 0.0
    for name in names:
        metrics[name], elapsed = measure_one(name, model, x, y, **kw)
        seconds += elapsed
    return metrics, seconds


def model_digest(model):
    return sha256_bytes(b''.join(w.detach().cpu().numpy().tobytes() for w in model.parameters()))


def make_changed_model(xtr, ytr, epochs=45):
    # A real changed trained model with a different optimization trajectory.
    model = train_model(xtr, ytr, epochs=epochs)
    with torch.no_grad():
        # Deterministic perturbation to a real model artifact.
        model[-1].bias[0] += 1.5
        model[-1].bias[1] -= 1.5
    return model


def run(output=Path('results/real_mutations.json'), epochs=45):
    rows = []
    for dataset in ('breast_cancer', 'wine'):
        xtr, ytr, xte, yte = make_dataset(dataset)
        base = train_model(xtr, ytr, epochs=epochs)
        base_scores, base_cost = evaluate(base, xte, yte)
        changed = make_changed_model(xtr, ytr, epochs=epochs)
        # Each mutation is actually applied to an evaluation configuration or weights.
        cases = [
            ('noise_scale_change', {'noise_config'}, base, xte, {'noise_scale': 0.8}),
            ('confidence_threshold_change', {'threshold_config'}, base, xte, {'conf_threshold': 0.95}),
            ('group_boundary_change', {'group_config'}, base, xte, {'group_cut': 0.75}),
            ('preprocessing_scale_change', {'preprocess'}, base, xte * np.float32(1.25), {}),
            ('weight_perturbation', {'weights'}, changed, xte, {}),
            ('unknown_dependency', {'untracked_pipeline_change'}, base, xte * np.float32(0), {}),
        ]
        for label, components, model, xx, config in cases:
            full_metrics, full_elapsed = evaluate(model, xx, yte, **config)
            selected = sorted(select_tests(components))
            selective_metrics, selective_elapsed = evaluate(model, xx, yte, names=selected, **config)
            changed_metrics = sorted(k for k in TESTS if abs(full_metrics[k] - base_scores[k]) > 1e-7)
            missed = sorted(set(changed_metrics) - set(selected))
            # Conservative strategy: unknown component => rerun everything.
            conservative = sorted(set(TESTS) if components - set().union(*DEPENDENCIES.values()) else selected)
            conservative_missed = sorted(set(changed_metrics) - set(conservative))
            rows.append({
                'dataset': dataset, 'scenario': label, 'changed_components': sorted(components),
                'baseline_metrics': base_scores, 'mutated_metrics': full_metrics,
                'observably_changed_metrics': changed_metrics,
                'declared_selected_tests': selected, 'declared_missed_observable_changes': missed,
                'conservative_selected_tests': conservative,
                'conservative_missed_observable_changes': conservative_missed,
                'baseline_weights_sha256': model_digest(base), 'scenario_weights_sha256': model_digest(model),
                'baseline_test_inputs_sha256': sha256_bytes(xte.tobytes()),
                'scenario_test_inputs_sha256': sha256_bytes(xx.tobytes()),
                'full_evaluation_seconds': round(full_elapsed, 8),
                'selected_evaluation_seconds': round(selective_elapsed, 8),
                'selected_test_count': len(selected),
                'total_test_count': len(TESTS),
                'note': 'Observed metric changes are NOT a complete oracle of assurance invalidation.'
            })
    summary = {
        'datasets': 2, 'scenario_runs': len(rows),
        'changed_metric_events': sum(len(r['observably_changed_metrics']) for r in rows),
        'missed_metric_events_declared': sum(len(r['declared_missed_observable_changes']) for r in rows),
        'missed_metric_events_conservative': sum(len(r['conservative_missed_observable_changes']) for r in rows),
        'selected_test_fraction': round(sum(r['selected_test_count'] for r in rows)/(len(rows)*len(TESTS)), 4),
        'methodology': 'Real mutations, measured finite-sample test metrics; test timings noisy at millisecond scale.'
    }
    result = {'title': 'TrustAudit-ML controlled mutation study', 'seed': SEED,
              'runtime': {'python': platform.python_version(), 'torch': torch.__version__},
              'summary': summary, 'runs': rows,
              'limitations': [
                  'Metrics are observational proxies, not certification of robustness, privacy or fairness.',
                  'The subgroup uses a numeric feature, not a protected characteristic.',
                  'Only 2 small tabular datasets and 4 cheap tests; no meaningful runtime speedup claim.',
                  'No hidden-dependency discovery algorithm is implemented.',
                  'Metric equality does not establish reusable assurance evidence.',
                  'Preprocessing/input mutation is evaluated without retraining by design.',
              ]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print('Wrote', output)
    return result


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('results/real_mutations.json'))
    ap.add_argument('--epochs', type=int, default=45)
    args = ap.parse_args()
    run(args.output, args.epochs)
