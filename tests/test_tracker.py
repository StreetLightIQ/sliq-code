import json
import os

import pytest

from sliq.tracker.centre_scale import CentreScaleTracker, hungarian

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'tracker_parity.json')


def box(cx, cy, w=0.04, h=0.06):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_hungarian_finds_the_minimum():
    assert hungarian([[1, 2, 3], [2, 1, 3]]) == [0, 1]
    assert hungarian([[5, 1], [1, 5]]) == [1, 0]
    assert hungarian([[0.9, 0.2, 1.0, 1.0], [0.1, 0.3, 1.0, 1.0]]) == [1, 0]


def test_moving_lamp_keeps_its_id_and_a_far_jump_starts_a_new_track():
    t = CentreScaleTracker()
    for k in range(6):
        active, matched, _ = t.update([box(0.4 + 0.03 * k, 0.3 - 0.01 * k)], [0.9], 0.2 * k)
    assert [m.id for m in matched] == [1] and active[0].matches == 6 and active[0].vx > 0
    _, matched, _ = t.update([box(0.9, 0.9)], [0.9], 1.4)
    assert [m.id for m in matched] == [2]


def test_velocity_is_clamped():
    t = CentreScaleTracker(centre_gate=10.0)
    t.update([box(0.1, 0.5)], [0.9], 0.0)
    t.update([box(0.6, 0.5)], [0.9], 0.01)
    assert t.tracks[1].vx == 1.0


def test_tracks_end_after_the_missing_time():
    t = CentreScaleTracker()
    t.update([box(0.5, 0.5)], [0.9], 0.0)
    _, _, completed = t.update([], [], 1.4)
    assert completed == []
    _, _, completed = t.update([], [], 1.6)
    assert [c.id for c in completed] == [1] and t.tracks == {}


def test_timestamps_must_increase():
    t = CentreScaleTracker()
    t.update([], [], 1.0)
    with pytest.raises(ValueError):
        t.update([], [], 1.0)


@pytest.mark.skipif(not os.path.exists(FIXTURE), reason='parity fixture not generated')
def test_matches_the_app_tracker():
    """Fixture written by the app unit test CentreScaleTrackerParityTest."""
    fx = json.load(open(FIXTURE))
    t = CentreScaleTracker()
    for frame in fx['frames']:
        _, matched, _ = t.update([tuple(b) for b in frame['boxes']], frame['scores'], frame['t'])
        assert sorted(m.id for m in matched) == sorted(frame['matched_ids'])
