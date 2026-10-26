import copy
from phase3_audit import audit


def test_audit_counts_and_does_not_claim_reuse():
    row = dict(dataset='wine', scenario='unknown', changed_components=['untracked'],
               observably_changed_metrics=['accuracy'], declared_selected_tests=[],
               conservative_selected_tests=['accuracy', 'robustness', 'subgroup_gap', 'confidence'])
    result = audit({'runs': [row]})
    assert result['summary']['declared_omitted_observable_changes'] == 1
    assert result['summary']['conservative_omitted_observable_changes'] == 0
    assert result['summary']['full_test_executions'] == 4
    assert result['cases'][0]['evidence_reuse_allowed'] is False


def test_audit_does_not_mutate_source():
    row = dict(dataset='wine', scenario='known', changed_components=['weights'],
               observably_changed_metrics=[], declared_selected_tests=['accuracy'],
               conservative_selected_tests=['accuracy'])
    source = {'runs': [row]}
    before = copy.deepcopy(source)
    audit(source)
    assert source == before
