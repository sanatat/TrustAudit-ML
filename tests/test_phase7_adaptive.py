from phase7_adaptive import wilson_upper, fit, select, evaluate
from trustaudit import TESTS

def row(dataset, comps, changed, seed=1):
    return {'dataset':dataset,'changed_components':comps,'changed_observed_metrics':changed,'seed':seed,'scenario':'test'}

def test_unknown_recipe_fails_closed():
    training=fit([row('a',['weights'],['accuracy'])]*5)
    assert select(row('b',['weights','new'],[]),training,0.5)==set(TESTS)

def test_sparse_recipe_fails_closed():
    training=fit([row('a',['weights'],['accuracy'])])
    assert select(row('b',['weights'],[]),training,0.5)==set(TESTS)

def test_bound_valid_and_conservative():
    assert wilson_upper(0,0)==1
    assert 0 < wilson_upper(0,10)<1
    assert wilson_upper(10,10)==1

def test_selection_no_heldout_label_leak():
    training=fit([row('a',['weights'],['accuracy'])]*8)
    a=select(row('b',['weights'],[]),training,0.5)
    b=select(row('b',['weights'],list(TESTS)),training,0.5)
    assert a==b

def test_cross_dataset_replay():
    rows=[row('a',['weights'],['accuracy'],i) for i in range(4)]+[row('b',['weights'],['robustness'],i) for i in range(4)]
    result=evaluate(rows,thresholds=(0.5,1.01))
    assert result['policies']['1.01']['selected_tests']==0
    assert result['policies']['1.01']['missed_observable_events']==8
    assert len(result['folds'])==2

def test_missing_datasets_error():
    import pytest
    with pytest.raises(ValueError):
        evaluate([row('a',['weights'],[])])
