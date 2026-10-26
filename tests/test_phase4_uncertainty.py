from phase4_uncertainty import choose, evaluate
from trustaudit import TESTS


def test_metadata_only_selection():
    assert choose({'noise_config'}, 'declared') == {'robustness'}
    assert choose({'untracked_pipeline_change'}, 'declared') == set()
    assert choose({'untracked_pipeline_change'}, 'fail_closed') == set(TESTS)


def test_fail_closed_known_uncertainty():
    assert choose({'preprocess'}, 'fail_closed', {'preprocess'}) == set(TESTS)
    assert choose({'noise_config'}, 'fail_closed', {'preprocess'}) == {'robustness'}


def test_empty_cases():
    assert evaluate([])['full']['execution_fraction'] is None


def test_unknown_policies():
    import pytest
    with pytest.raises(ValueError):
        choose({'weights'}, 'bad')
