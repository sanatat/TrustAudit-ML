"""TrustAudit-ML: small, auditable CPU-only change-impact experiment.

Not a security proof. 'Ground truth' for invalidated tests is defined by the
experiment's declared dependency oracle, not by empirical safety guarantees.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
TESTS = ('accuracy', 'robustness', 'subgroup_gap', 'confidence')
# Explicit dependency model; config/report changes intentionally do not invalidate
# model-behaviour tests. A hidden dependency is injected separately in evaluation.
DEPENDENCIES = {
    'accuracy': {'data', 'preprocess', 'weights'},
    'robustness': {'data', 'preprocess', 'weights', 'noise_config'},
    'subgroup_gap': {'data', 'preprocess', 'weights', 'group_config'},
    'confidence': {'data', 'preprocess', 'weights', 'threshold_config'},
}
SCENARIOS = {
    'documentation_only': {'documentation'},
    'noise_config_change': {'noise_config'},
    'group_definition_change': {'group_config'},
    'threshold_change': {'threshold_config'},
    'model_update': {'weights'},
    'preprocessing_update': {'preprocess'},
    'combined_config_change': {'noise_config', 'threshold_config'},
    'hidden_dependency': {'documentation'},  # unknown to selector; oracle needs accuracy
}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def make_dataset(dataset):
    loader = load_breast_cancer if dataset == 'breast_cancer' else load_wine
    d = loader()
    X = d.data.astype(np.float32)
    y = (d.target if dataset == 'breast_cancer' else (d.target == 0).astype(int)).astype(np.int64)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, stratify=y, test_size=0.25, random_state=SEED)
    scaler = StandardScaler().fit(Xtrain)
    return (scaler.transform(Xtrain).astype(np.float32), ytrain,
            scaler.transform(Xtest).astype(np.float32), ytest)


def train_model(X, y, epochs=45, seed=SEED):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 16), torch.nn.ReLU(),
                              torch.nn.Linear(16, 2))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.015)
    xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    net.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(xt), yt)
        loss.backward()
        optimizer.step()
    net.eval()
    return net


def predictions(net, X):
    with torch.no_grad():
        return torch.softmax(net(torch.from_numpy(X)), 1).numpy()


def run_tests(net, X, y):
    t0 = time.perf_counter()
    p = predictions(net, X)
    score = float(np.mean(p.argmax(1) == y))
    # Fixed synthetic noise, explicitly not an adversarial robustness certification.
    noise = np.random.default_rng(SEED).normal(0, 0.12, X.shape).astype(np.float32)
    rob = float(np.mean(predictions(net, X + noise).argmax(1) == y))
    # Group is a feature-based proxy, not a sensitive demographic attribute.
    group = X[:, 0] >= np.median(X[:, 0])
    gap = abs(float(np.mean(p.argmax(1)[group] == y[group])) -
              float(np.mean(p.argmax(1)[~group] == y[~group])))
    confidence = float(np.mean(np.max(p, axis=1) >= 0.8))
    return dict(zip(TESTS, (score, rob, gap, confidence))), time.perf_counter() - t0


def select_tests(changes, dependencies=DEPENDENCIES):
    return set(t for t, deps in dependencies.items() if deps.intersection(changes))


def evaluate_scenarios(test_cost_seconds=1.0):
    """Test cost in equivalent units; NOT measured runtime saving."""
    rows = []
    for scenario, changes in SCENARIOS.items():
        selected = select_tests(changes)
        required = select_tests(changes)
        if scenario == 'hidden_dependency':
            required = required | {'accuracy'}
        missed = required - selected
        rows.append({
            'scenario': scenario, 'changed_components': sorted(changes),
            'required_tests_oracle': sorted(required), 'selected_tests': sorted(selected),
            'missed_required_tests': sorted(missed),
            'full_test_units': len(TESTS), 'selected_test_units': len(selected),
            'naive_hash_test_units': len(TESTS),
            'work_reduction_vs_full': round(1 - len(selected)/len(TESTS), 4),
            'undetected_invalidations': len(missed),
            'simulated_full_cost_seconds': len(TESTS)*test_cost_seconds,
            'simulated_selected_cost_seconds': len(selected)*test_cost_seconds,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('results/pilot.json'))
    p.add_argument('--epochs', type=int, default=45)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds_results = {}
    artifacts = {}
    start = time.perf_counter()
    for dataset in ('breast_cancer', 'wine'):
        Xtr, ytr, Xte, yte = make_dataset(dataset)
        t0 = time.perf_counter()
        model = train_model(Xtr, ytr, epochs=args.epochs)
        training_s = time.perf_counter() - t0
        scores, eval_s = run_tests(model, Xte, yte)
        weights = b''.join(v.detach().numpy().tobytes() for v in model.parameters())
        artifacts[dataset] = {'train_data_sha256': sha256_bytes(Xtr.tobytes()+ytr.tobytes()),
                              'test_data_sha256': sha256_bytes(Xte.tobytes()+yte.tobytes()),
                              'weights_sha256': sha256_bytes(weights)}
        ds_results[dataset] = {'train_samples': len(Xtr), 'test_samples': len(Xte),
                               'model_training_seconds': round(training_s, 5),
                               'all_tests_wall_seconds': round(eval_s, 5),
                               'metrics': {k: round(v, 5) for k,v in scores.items()}}
    scenarios = evaluate_scenarios()
    output = {
        'project': 'TrustAudit-ML', 'seed': SEED,
        'software': {'python': platform.python_version(), 'torch': torch.__version__,
                     'numpy': np.__version__, 'scikit_learn': sklearn.__version__},
        'datasets': ds_results, 'artifact_hashes': artifacts,
        'declared_dependencies': {k: sorted(v) for k,v in DEPENDENCIES.items()},
        'scenarios': scenarios,
        'scenario_summary': {
            'n_scenarios': len(scenarios),
            'test_unit_reduction_vs_full': round(1-sum(r['selected_test_units'] for r in scenarios)/(len(scenarios)*len(TESTS)), 4),
            'total_missed_oracle_tests': sum(r['undetected_invalidations'] for r in scenarios),
            'note': 'Savings are COUNTED test units, not measured time; oracle is synthetic.'},
        'total_wall_seconds': round(time.perf_counter() - start, 5),
        'limitations': [
            'Dependency matrix is manually specified, not learned or formally verified.',
            'Scenario oracle is a synthetic assumption; not proof of real-world risk.',
            'Hidden dependency intentionally demonstrates unsafe evidence reuse.',
            'No cryptographic signatures, access control, poisoning, or formal assurance.',
            'Model metrics are measured; scenario savings are counted test units.'
        ]
    }
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'saved': str(args.output), 'datasets': ds_results,
                      'scenario_summary': output['scenario_summary']}, indent=2))

if __name__ == '__main__':
    main()
