"""Retrospective change-impact audit for phase 2 data.

IMPORTANT: changed metrics come from FULL evaluation. This is a post-hoc
benchmark/diagnostic, NOT a prospective detector or security certification.
"""
import argparse
import hashlib
import json
from pathlib import Path

from trustaudit import TESTS


def digest(data):
    return hashlib.sha256(data).hexdigest()


def audit(source):
    rows = source['runs']
    report = []
    for row in rows:
        observed = set(row['observably_changed_metrics'])
        declared = set(row['declared_selected_tests'])
        conservative = set(row['conservative_selected_tests'])
        # Exhaustive observed-change oracle exists only because all tests ran.
        report.append({
            'dataset': row['dataset'], 'scenario': row['scenario'],
            'changed_components': row['changed_components'],
            'observed_changed_metrics': sorted(observed),
            'declared_missed': sorted(observed - declared),
            'conservative_missed': sorted(observed - conservative),
            'declared_tests': sorted(declared),
            'conservative_tests': sorted(conservative),
            'full_tests': list(TESTS),
            'evidence_reuse_allowed': False,
            'reason': 'Metric observations do not prove assurance evidence remains valid.',
        })
    n = len(rows) * len(TESTS)
    d = sum(len(x['declared_tests']) for x in report)
    c = sum(len(x['conservative_tests']) for x in report)
    return {
        'status': 'retrospective_experimental_audit_only',
        'summary': {
            'scenario_runs': len(rows), 'full_test_executions': n,
            'declared_test_executions': d, 'conservative_test_executions': c,
            'declared_omitted_observable_changes': sum(len(x['declared_missed']) for x in report),
            'conservative_omitted_observable_changes': sum(len(x['conservative_missed']) for x in report),
            'conservative_fraction_vs_full': round(c/n, 4) if n else None,
        },
        'cases': report,
        'caveat': 'This is a post-hoc evaluation using full-run metric observations, not a proof of safety or evidence validity.'
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('results/real_mutations.json'))
    p.add_argument('--output', type=Path, default=Path('results/phase3_audit.json'))
    args = p.parse_args()
    content = args.input.read_bytes()
    result = audit(json.loads(content))
    result['source_sha256'] = digest(content)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))
    print('Audit saved:', args.output)
    print('Input SHA256:', result['source_sha256'])


if __name__ == '__main__':
    main()
