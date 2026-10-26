import json
from phase6_unseen import run, SEEDS, POLICIES
from trustaudit import TESTS


def test_unseen_split_and_counts(tmp_path):
    result=run(output=tmp_path/'p6.json', epochs=3, seeds=(11,))
    assert len(result['runs'])==12
    assert len(set(r['scenario'] for r in result['runs']))==6
    assert len(set(r['dataset'] for r in result['runs']))==2
    assert all(r['scenario'] not in {'noise_scale_change','weight_perturbation'} for r in result['runs'])
    assert result['aggregation']['full']['executions']==len(result['runs'])*len(TESTS)
    assert json.loads((tmp_path/'p6.json').read_text())['seeds']==[11]


def test_guarded_unknown_fail_closed(tmp_path):
    result=run(output=tmp_path/'p6.json', epochs=3, seeds=(11,))
    unknown=[r for r in result['runs'] if 'input_shift_v2' in r['changed_components']]
    assert len(unknown)==4
    assert all(set(r['selection']['learned_guarded']['selected']) == set(TESTS) for r in unknown)
    assert all(not r['selection']['learned_guarded']['missed_observed'] for r in unknown)


def test_no_heldout_label_in_selection(tmp_path):
    result=run(output=tmp_path/'p6.json', epochs=3, seeds=(11,))
    for r in result['runs']:
        assert set(r['selection']['declared']['selected']) <= set(TESTS)
        assert set(r['selection']['learned']['selected']) <= set(TESTS)
        assert set(r['selection']['full']['selected']) == set(TESTS)
