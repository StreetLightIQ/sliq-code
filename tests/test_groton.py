import pytest

from sliq.physics import groton_eval as G


def test_held_out_errors_match_the_paper(tmp_path, monkeypatch):
    monkeypatch.setattr(G, 'OUT', str(tmp_path / 'groton.json'))
    G.main()
    import json
    s = json.load(open(tmp_path / 'groton.json'))['summary']
    assert s['all']['lines'] == 13
    assert (round(100 * s['all']['model']), round(100 * s['all']['interp'])) == (23, 37)
    assert (round(100 * s['edges']['model']), round(100 * s['edges']['interp'])) == (34, 70)
    assert (round(100 * s['interior']['model']), round(100 * s['interior']['interp'])) == (22, 31)
    assert round(100 * s['within_1.5']['model']) == 68


def test_rel_error_and_within():
    import numpy as np
    obs = np.array([1.0, 2.0, 3.0])
    assert G.rel_error(obs, obs) == 0
    assert G.rel_error(obs * 1.5, obs) == pytest.approx(0.5)
    assert G.within(np.array([1.5, 1.6]), np.array([1.0, 1.0])).tolist() == [True, False]
