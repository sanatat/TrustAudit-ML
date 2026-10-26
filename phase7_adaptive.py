"""Phase 7: retrospective, dataset-held-out uncertainty-aware audit policies.

Calibrate empirical metric-change frequency using ONLY the other dataset's
Phase 6 labels. Decisions use component names and trained estimates, never
held-out model outcomes. This is exploratory policy evaluation, not certified
coverage, security assurance, or novel unseen-recipe generalization.
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from trustaudit import TESTS

THRESHOLDS = (0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90, 1.01)


def wilson_upper(positive, trials, z=1.96):
    if trials == 0:
        return 1.0
    phat = positive / trials
    denom = 1 + z*z/trials
    centre = phat + z*z/(2*trials)
    span = z * math.sqrt(phat*(1-phat)/trials + z*z/(4*trials*trials))
    return min(1., (centre+span)/denom)


def recipe_key(row):
    return tuple(sorted(set(row['changed_components'])))


def fit(train):
    counts = defaultdict(lambda: defaultdict(int))
    trials = defaultdict(int)
    for row in train:
        key = recipe_key(row)
        trials[key] += 1
        for metric in row['changed_observed_metrics']:
            if metric not in TESTS:
                raise ValueError('Unknown metric')
            counts[key][metric] += 1
    return {key: {'trials': n, 'counts': dict(counts[key])} for key,n in trials.items()}


def select(row, training, threshold, min_support=3):
    if threshold <= 0:
        return set(TESTS)
    key = recipe_key(row)
    stat = training.get(key)
    if stat is None or stat['trials'] < min_support:
        return set(TESTS)  # no known recipe: conservative, do not extrapolate
    return {metric for metric in TESTS if wilson_upper(stat['counts'].get(metric,0), stat['trials']) >= threshold}


def evaluate(rows, thresholds=THRESHOLDS, min_support=3):
    groups = sorted({r['dataset'] for r in rows})
    if len(groups) < 2:
        raise ValueError('Need two or more dataset groups')
    folds = []
    all_choices = {str(t): [] for t in thresholds}
    for heldout in groups:
        train = [r for r in rows if r['dataset'] != heldout]
        test = [r for r in rows if r['dataset'] == heldout]
        model = fit(train)
        for row in test:
            for t in thresholds:
                chosen = select(row, model, t, min_support)
                observed = set(row['changed_observed_metrics'])
                all_choices[str(t)].append({'dataset':heldout,'seed':row['seed'],
                   'scenario':row['scenario'],'selected':sorted(chosen),
                   'missed_observed':sorted(observed-chosen),
                   'observed_count':len(observed),
                   'recipe_support':model.get(recipe_key(row),{}).get('trials',0)})
        folds.append({'heldout_dataset':heldout,'training_scenarios':len(train),
                      'test_scenarios':len(test), 'unique_training_recipes':len(model)})
    total = len(rows)*len(TESTS)
    summary = {}
    for t in thresholds:
        decisions = all_choices[str(t)]
        executions = sum(len(x['selected']) for x in decisions)
        missed = sum(len(x['missed_observed']) for x in decisions)
        summary[str(t)] = {'selected_tests':executions,'total_tests':total,
            'test_reduction_fraction':round(1-executions/total,4),
            'missed_observable_events':missed,
            'scenarios_with_misses':sum(bool(x['missed_observed']) for x in decisions),
            'decisions':decisions}
    return {'protocol':'Leave-one-dataset-out replay of Phase 6 results; scenario-recipe empirical sensitivity, Wilson upper bounds, fail closed on unseen recipes',
        'folds':folds,'min_support':min_support,'thresholds':list(thresholds),
        'policies':summary,'limitations':[
            'The same mutation recipes and multiple correlated seeds are present on both datasets; not novel-recipe generalization.',
            'The upper confidence bounds describe empirical binary metric-change rates, not calibrated probabilities of harm or formal guarantees.',
            'The two datasets and reused exploratory test designs are insufficient for reliable risk calibration.',
            'Tests omitted by a policy are scored against precomputed full evaluations in this offline replay only.',
            'Metric changes are finite-sample differences; no security or fairness assurance is established.',
            'Any threshold selected after examining held-out outcomes is exploratory and must be tuned on separate calibration data.'
        ]}


def run(input_path=Path('results/phase6_unseen.json'), output_path=Path('results/phase7_adaptive.json')):
    raw = input_path.read_bytes()
    source = json.loads(raw)
    result = evaluate(source['runs'])
    result['source_sha256'] = hashlib.sha256(raw).hexdigest()
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(result,indent=2)+'\n')
    for threshold, metrics in result['policies'].items():
        print(f'threshold={threshold:>4} selected={metrics["selected_tests"]:3}/{metrics["total_tests"]} '
              f'reduction={metrics["test_reduction_fraction"]:.1%} missed={metrics["missed_observable_events"]}')
    print('Saved',output_path)
    return result

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=Path('results/phase6_unseen.json'))
    parser.add_argument('--output',type=Path,default=Path('results/phase7_adaptive.json'))
    opts=parser.parse_args()
    run(opts.input,opts.output)
