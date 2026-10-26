"""Prospective selection from metadata, followed by retrospective evaluation.

No metric outcomes are accessed when selecting tests. This is NOT a dependency
learning method; it evaluates explicit assumptions under incomplete metadata.
"""
import argparse
import json
from pathlib import Path
from trustaudit import TESTS, DEPENDENCIES

KNOWN = set().union(*DEPENDENCIES.values())


def choose(changes, mode='declared', uncertain=()):
    """Select tests before inspecting their observed results."""
    changes = set(changes)
    uncertain = set(uncertain)
    declared = {name for name, deps in DEPENDENCIES.items() if changes & deps}
    if mode == 'full':
        return set(TESTS)
    if mode == 'declared':
        return declared
    if mode == 'fail_closed':
        return set(TESTS) if changes - KNOWN or changes & uncertain else declared
    raise ValueError(mode)


def evaluate(rows, uncertain=()):
    policy_names = ('full', 'declared', 'fail_closed')
    results = {p: {'executed': 0, 'missed_events': 0, 'cases_with_misses': 0} for p in policy_names}
    for row in rows:
        changes = row['changed_components']
        observed = set(row['observably_changed_metrics'])
        for p in policy_names:
            selected = choose(changes, p, uncertain)
            misses = observed - selected
            results[p]['executed'] += len(selected)
            results[p]['missed_events'] += len(misses)
            results[p]['cases_with_misses'] += bool(misses)
    total = len(rows) * len(TESTS)
    for p in policy_names:
        results[p]['execution_fraction'] = round(results[p]['executed']/total, 4) if total else None
        results[p]['reduction_vs_full'] = round(1-results[p]['executed']/total, 4) if total else None
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('results/real_mutations.json'))
    ap.add_argument('--output', type=Path, default=Path('results/phase4_uncertainty.json'))
    args = ap.parse_args()
    source = json.loads(args.input.read_text())
    runs = source['runs']
    # Distinct uncertainty assumptions: useful for sensitivity analysis, not learned.
    experiments = {
        'unknown_only': [],
        'unknown_plus_preprocess': ['preprocess'],
        'unknown_plus_weights': ['weights'],
        'unknown_plus_all_config': ['noise_config', 'group_config', 'threshold_config'],
        'all_components_uncertain': sorted(KNOWN),
    }
    report = {'status': 'selection_uses_metadata_only; scoring_uses_exhaustive_posthoc_metrics',
              'scenario_runs': len(runs), 'experiments': {label: {
                  'uncertain_components': comps,
                  'policies': evaluate(runs, comps),
              } for label, comps in experiments.items()},
              'limitations': [
                  'The uncertainty sets are chosen manually rather than inferred.',
                  'Observable metric changes are not complete ground truth of invalidated assurances.',
                  'Results are retrospective scoring of a prospective selection rule.',
                  'Unknown components trigger all tests; known hidden dependencies remain possible.',
                  'No statistical guarantee on unseen pipelines or threat models.'
              ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    for name, result in report['experiments'].items():
        print(name, '=>', result['policies']['fail_closed'])
    print('Saved', args.output)

if __name__ == '__main__':
    main()
