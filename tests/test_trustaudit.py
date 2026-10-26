from trustaudit import DEPENDENCIES, TESTS, evaluate_scenarios, select_tests

def test_no_change_requires_no_retest():
    assert select_tests(set()) == set()

def test_weights_update_invalidates_all_model_tests():
    assert select_tests({'weights'}) == set(TESTS)

def test_noise_change_only_affects_robustness():
    assert select_tests({'noise_config'}) == {'robustness'}

def test_documentation_only_skips_assessments_under_declared_model():
    assert select_tests({'documentation'}) == set()

def test_hidden_dependency_exposes_unsound_reuse():
    row = next(r for r in evaluate_scenarios() if r['scenario'] == 'hidden_dependency')
    assert row['missed_required_tests'] == ['accuracy']

def test_scenario_counts():
    rows = evaluate_scenarios()
    assert len(rows) == 8
    assert all(0 <= r['selected_test_units'] <= len(DEPENDENCIES) for r in rows)
