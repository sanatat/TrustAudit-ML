import numpy as np
from real_mutations import evaluate, run, measure_one
from trustaudit import make_dataset, train_model


def test_reproduce_scores():
    a, b, x, y = make_dataset('wine')
    model = train_model(a, b, epochs=5)
    v1, _ = measure_one('accuracy', model, x, y)
    v2, _ = measure_one('accuracy', model, x, y)
    assert v1 == v2


def test_select_only_selected():
    a, b, x, y = make_dataset('wine')
    model = train_model(a, b, epochs=5)
    scores, cost = evaluate(model, x, y, names=['confidence'])
    assert list(scores) == ['confidence']
    assert cost >= 0


def test_real_mutations_have_observable_effect(tmp_path):
    result = run(tmp_path / 'out.json', epochs=5)
    assert len(result['runs']) == 12
    assert result['summary']['changed_metric_events'] > 0
    assert result['summary']['missed_metric_events_declared'] > 0
    assert result['summary']['missed_metric_events_conservative'] == 0
