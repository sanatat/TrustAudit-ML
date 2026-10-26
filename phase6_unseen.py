"""Phase 6: truly applied, held-out COMBINATIONS under multiple model seeds.

Learned sensitivities: v2 single-component interventions on OTHER dataset only.
Test recipes: unseen combinations and unseen shift component. Full metrics are
computed as post-hoc assessment labels, NEVER exposed to the selection policies.
No guarantees about security, future data, or causal dependency identification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from trustaudit import TESTS, make_dataset, train_model
from real_mutations import evaluate
from phase5_discovery import learned_edges, select_policy

SEEDS = (11, 23, 47)
DATASETS = ('breast_cancer', 'wine')
EPS = 1e-7
POLICIES = ('full', 'declared', 'fail_closed', 'learned', 'learned_guarded')


def heldout_cases(model, x):
    """Changes are applied, and descriptions match the interventions."""
    shifted = np.asarray(x * np.float32(1.25), dtype=np.float32)
    zeroed = np.zeros_like(x, dtype=np.float32)
    import copy
    altered = copy.deepcopy(model)
    with torch.no_grad():
        altered[-1].bias[0] += 1.5
        altered[-1].bias[1] -= 1.5
    return [
      ('preprocess_plus_noise', ['preprocess','noise_config'], model, shifted, {'noise_scale':.8}),
      ('weights_plus_threshold', ['weights','threshold_config'], altered, x, {'conf_threshold':.95}),
      ('preprocess_plus_group', ['preprocess','group_config'], model, shifted, {'group_cut':.75}),
      ('weights_plus_preprocess_plus_noise', ['weights','preprocess','noise_config'], altered, shifted, {'noise_scale':.8}),
      ('unseen_input_zero_shift', ['input_shift_v2'], model, zeroed, {}),
      ('unseen_shift_plus_threshold', ['input_shift_v2','threshold_config'], model, zeroed, {'conf_threshold':.95}),
    ]


def run(source_path=Path('results/real_mutations.json'), output=Path('results/phase6_unseen.json'), epochs=45, seeds=SEEDS):
    raw = source_path.read_bytes()
    training_rows = json.loads(raw)['runs']
    records = []
    edges_by_fold = {}
    for heldout in DATASETS:
        train = [r for r in training_rows if r['dataset'] != heldout]
        learned = learned_edges(train)
        edges_by_fold[heldout] = learned
        xtr, ytr, xte, yte = make_dataset(heldout)
        for seed in seeds:
            model = train_model(xtr, ytr, epochs=epochs, seed=int(seed))
            reference, _ = evaluate(model, xte, yte)
            for name, comps, target_model, xx, kw in heldout_cases(model, xte):
                scores, _ = evaluate(target_model, xx, yte, **kw)
                observed = sorted(k for k in TESTS if abs(scores[k]-reference[k]) > EPS)
                choices = {}
                for policy in POLICIES:
                    selected = sorted(select_policy(comps, policy, learned))
                    choices[policy] = {'selected':selected,'missed_observed':sorted(set(observed)-set(selected))}
                records.append({'dataset':heldout,'seed':int(seed),'scenario':name,
                                'changed_components':comps,'baseline_metrics':reference,
                                'observed_metrics':scores,'changed_observed_metrics':observed,
                                'selection':choices})
    aggregates = {}
    for policy in POLICIES:
        executed = sum(len(r['selection'][policy]['selected']) for r in records)
        misses = sum(len(r['selection'][policy]['missed_observed']) for r in records)
        total = len(records)*len(TESTS)
        aggregates[policy] = {'executions':executed,'full_executions':total,
                              'reduction_fraction':round(1-executed/total,4),
                              'missed_observed_metric_events':misses,
                              'cases_with_misses':sum(bool(r['selection'][policy]['missed_observed']) for r in records)}
    per_seed = {str(seed): {p: {'executions':sum(len(r['selection'][p]['selected']) for r in records if r['seed']==seed),
                               'missed_events':sum(len(r['selection'][p]['missed_observed']) for r in records if r['seed']==seed)}
                            for p in POLICIES} for seed in seeds}
    report={'protocol':'Single-component training outcomes from OTHER dataset; heldout dataset with NEW combination recipes; 3 initialization seeds; outcomes used only for scoring',
            'train_source_sha256':hashlib.sha256(raw).hexdigest(),'seeds':list(seeds),
            'train_source':'Phase 2 measured single-component interventions, no Phase 6 label in training',
            'test_recipes_not_seen_in_training':sorted(set(r['scenario'] for r in records)),
            'aggregation':aggregates,'per_seed':per_seed,'learned_by_fold':edges_by_fold,'runs':records,
            'limitations':['The held-out combinations reuse components known to the training procedure, except input_shift_v2.',
                           'Both datasets were used in earlier exploratory project development; this is a pilot, not independent validation.',
                           'Seeds change initialization but do not constitute independent datasets or demonstrate statistical significance.',
                           'Observed metric changes are finite-sample differences, not a complete assurance invalidation oracle.',
                           'The learned rule uses co-occurrence, not causal proof; unknown components fail closed only in guarded policy.',
                           'No runtime speedup or security/fairness guarantee is implied by fewer selected tests.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    for key,item in aggregates.items():
        print(f'{key:16} selected={item["executions"]}/{item["full_executions"]} '
              f'reduction={item["reduction_fraction"]:.1%} missed={item["missed_observed_metric_events"]}')
    print('Saved',output)
    return report

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=Path('results/real_mutations.json'))
    ap.add_argument('--output',type=Path,default=Path('results/phase6_unseen.json'))
    ap.add_argument('--epochs',type=int,default=45)
    opts=ap.parse_args()
    run(opts.input,opts.output,opts.epochs)
