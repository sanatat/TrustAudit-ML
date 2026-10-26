from phase5_discovery import learned_edges, select_policy, cross_dataset_evaluation
from trustaudit import TESTS


def row(dataset, component, observed):
    return {'dataset': dataset, 'scenario': component, 'changed_components': [component],
            'observably_changed_metrics': list(observed)}


def test_learned_edges_only_training():
    learned = learned_edges([row('train', 'mystery', ['accuracy'])])
    assert learned['mystery']['suggested_tests'] == ['accuracy']
    assert select_policy(['mystery'], 'learned', learned) == {'accuracy'}


def test_unseen_component_guarded_fails_closed():
    assert select_policy(['never_seen'], 'learned_guarded', {}) == set(TESTS)
    assert select_policy(['never_seen'], 'learned', {}) == set()


def test_hidden_dependency_learned_generalizes_between_groups():
    runs = [row('a', 'mystery', ['accuracy']), row('b', 'mystery', ['accuracy'])]
    result = cross_dataset_evaluation(runs)
    assert result['aggregate']['learned']['missed_observable_metric_events'] == 0
    assert result['aggregate']['declared']['missed_observable_metric_events'] == 2


def test_holdout_miss_not_silently_erased():
    runs = [row('a', 'mystery', ['accuracy']), row('b', 'mystery', ['confidence'])]
    result = cross_dataset_evaluation(runs)
    assert result['aggregate']['learned']['missed_observable_metric_events'] == 2


def test_guarded_unknown_keeps_full():
    runs = [row('a', 'never_a', ['accuracy']), row('b', 'never_b', ['confidence'])]
    result = cross_dataset_evaluation(runs)
    assert result['aggregate']['learned_guarded']['executed_tests'] == 2 * len(TESTS)


def test_multi_component_is_not_attributed():
    learned = learned_edges([{'changed_components': ['a', 'b'], 'observably_changed_metrics': ['accuracy']}])
    assert learned == {}


def test_at_least_two_datasets():
    import pytest
    with pytest.raises(ValueError):
        cross_dataset_evaluation([row('a', 'mystery', ['accuracy'])])
