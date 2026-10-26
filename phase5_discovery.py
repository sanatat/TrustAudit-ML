"""Dataset-held-out empirical dependency discovery.

The policies select using only training-dataset observations and metadata.
The held-out dataset's outcome labels are consulted ONLY after selection.
A learned edge means 'observed metric sensitivity', not a proven dependency.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from trustaudit import DEPENDENCIES, TESTS
from phase4_uncertainty import choose


def learned_edges(train_rows, min_support=1):
    """Count per-component observed sensitivity. Only single-component interventions."""
    counts = defaultdict(lambda: defaultdict(int))
    trials = defaultdict(int)
    for row in train_rows:
        changed = row['changed_components']
        if len(changed) != 1:
            continue  # multi-component observations cannot attribute a unique edge
        component = changed[0]
        trials[component] += 1
        for metric in row['observably_changed_metrics']:
            if metric not in TESTS:
                raise ValueError(f'Unexpected metric: {metric}')
            counts[component][metric] += 1
    return {c: {'trials': trials[c],
                'positive_counts': {m: counts[c].get(m, 0) for m in TESTS},
                'suggested_tests': sorted(m for m in TESTS if counts[c].get(m, 0) >= min_support)}
            for c in sorted(trials)}


def select_policy(changes, policy, learned, min_trials=1):
    changes = set(changes)
    if policy in ('full', 'declared', 'fail_closed'):
        return choose(changes, policy)
    if policy not in ('learned', 'learned_guarded'):
        raise ValueError(policy)
    if policy == 'learned_guarded' and any(c not in learned or learned[c]['trials'] < min_trials for c in changes):
        return set(TESTS)  # fail closed on never-observed components
    selected = choose(changes, 'declared')
    for c in changes:
        selected.update(learned.get(c, {}).get('suggested_tests', []))
    return selected


def score(rows, learned, policy, min_trials=1):
    total = len(rows) * len(TESTS)
    executions, missed, cases = 0, 0, 0
    detailed = []
    for row in rows:
        selected = select_policy(row['changed_components'], policy, learned, min_trials)
        observed = set(row['observably_changed_metrics'])
        misses = sorted(observed - selected)
        executions += len(selected)
        missed += len(misses)
        cases += bool(misses)
        detailed.append({'scenario': row['scenario'], 'selected_tests': sorted(selected),
                         'heldout_observable_changes': sorted(observed), 'misses': misses})
    return {'executed_tests': executions, 'full_tests': total,
            'test_reduction_fraction': round(1 - executions / total, 4) if total else None,
            'missed_observable_metric_events': missed, 'scenarios_with_misses': cases,
            'decisions': detailed}


def cross_dataset_evaluation(rows, min_support=1, min_trials=1):
    datasets = sorted(set(r['dataset'] for r in rows))
    if len(datasets) < 2:
        raise ValueError('At least two independent dataset groups required')
    folds = []
    for heldout in datasets:
        train = [r for r in rows if r['dataset'] != heldout]
        test = [r for r in rows if r['dataset'] == heldout]
        learned = learned_edges(train, min_support=min_support)
        policies = {p: score(test, learned, p, min_trials) for p in
                    ('full', 'declared', 'fail_closed', 'learned', 'learned_guarded')}
        folds.append({'train_datasets': sorted(set(r['dataset'] for r in train)),
                      'test_dataset': heldout, 'train_scenarios': len(train),
                      'heldout_scenarios': len(test), 'learned_edges': learned, 'policies': policies})
    aggregate = {}
    for p in ('full', 'declared', 'fail_closed', 'learned', 'learned_guarded'):
        executed = sum(f['policies'][p]['executed_tests'] for f in folds)
        full = sum(f['policies'][p]['full_tests'] for f in folds)
        aggregate[p] = {'executed_tests': executed, 'full_tests': full,
                        'test_reduction_fraction': round(1 - executed/full, 4),
                        'missed_observable_metric_events': sum(f['policies'][p]['missed_observable_metric_events'] for f in folds),
                        'scenarios_with_misses': sum(f['policies'][p]['scenarios_with_misses'] for f in folds)}
    return {'protocol': 'leave-one-dataset-out; train on other dataset(s), select blind to held-out metric labels',
            'min_support': min_support, 'min_trials': min_trials, 'folds': folds,
            'aggregate': aggregate, 'limitations': [
                'Only two datasets and six mutation types: tiny exploratory pilot with reused scenario labels.',
                'The same mutation recipes occur in both datasets; this is not unseen-mutation generalization.',
                'Observed metric changes are not a security, fairness, or privacy validity oracle.',
                'Single-component co-occurrence is correlation, not causal identification or a guarantee.',
                'All full-test outcomes were precomputed for retrospective scoring; selection cannot access held-out outcome labels.',
                'No claimed wall-clock performance or statistical robustness guarantees.'
            ]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('results/real_mutations.json'))
    parser.add_argument('--output', type=Path, default=Path('results/phase5_discovery.json'))
    args = parser.parse_args()
    raw = args.input.read_bytes()
    source = json.loads(raw)
    report = cross_dataset_evaluation(source['runs'])
    report['source_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for name, result in report['aggregate'].items():
        print(f'{name:16s} tests={result["executed_tests"]:2d}/{result["full_tests"]} '
              f'reduction={result["test_reduction_fraction"]:.1%} '
              f'missed_observable_events={result["missed_observable_metric_events"]}')
    print('Saved', args.output)

if __name__ == '__main__':
    main()
