import numpy as np
import pytest

from sliq.physics import fieldmodel as FM
from sliq.physics import scene as S
from sliq.physics.geometry import Camera


def test_height_is_recovered_from_sight_lines():
    cam = Camera(1280, 720, 70.0, 5.0, 0.0, 1.3)
    H, x, z0 = 9.0, 3.0, 60.0
    dist = np.linspace(0, 40, 12)
    obs, cams = [], []
    for i, d in enumerate(dist):
        u, v, front = cam.project(np.array([x, -(H - 1.3), z0 - d]))
        assert front
        obs.append((i, (u - 5, v - 5, u + 5, v + 5)))
        cams.append(cam)
    h, rms = S.fit_height(obs, cams, dist, 1.3)
    assert h == pytest.approx(H, abs=0.05) and rms < 0.05


def test_fit_height_needs_enough_motion():
    cam = Camera(1280, 720)
    obs = [(i, (600, 200, 610, 210)) for i in range(6)]
    assert S.fit_height(obs, [cam] * 6, np.zeros(6), 1.3) == (None, None)


def test_tracks_seen_side_by_side_are_never_merged():
    boxes = {1: {0: [100, 100, 120, 120], 1: [110, 100, 130, 120]}, 2: {0: [200, 100, 220, 120], 1: [210, 100, 230, 120]},
             3: {5: [300, 100, 320, 120]}}
    assert FM._seen_apart([1], [2], boxes)
    assert not FM._seen_apart([1], [3], boxes)


def test_field_falls_off_with_distance_and_scales_with_intensity():
    S_grid, X_grid = np.meshgrid(np.arange(0, 50.0), np.zeros(1), indexing='ij')
    lamp = dict(s=0.0, x=0.0, height=9.0, intensity=1.0, n=2)
    E = FM.field([lamp, dict(lamp, intensity=0.5)], S_grid, X_grid)
    assert np.all(np.diff(E[0, :, 0]) < 0)
    assert np.allclose(E[1], 0.5 * E[0])
    assert E[0, 0, 0] == pytest.approx(1 / 81)


def test_longest_run():
    assert FM.longest_run([0, 1, 1, 0, 1, 1, 1, 0], 0.5) == 1.5
