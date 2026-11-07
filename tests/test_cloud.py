import gzip
import json

import numpy as np

from sliq.cloud import process_session as P


def _session(tmp_path):
    frame = lambda fid, ts: dict(frame_id=fid, utc_ms=1000 + 200 * fid, sensor_timestamp_ns=ts)
    lamp = dict(track_id=7, score=0.8, bbox=dict(x1=0.4, y1=0.2, x2=0.45, y2=0.27))
    files = tmp_path / 'files' / 'u' / 's1'
    files.mkdir(parents=True)
    obs = []
    for fid, ts in ((2, 300), (1, 100), (2, 300)):            # frame 2 uploaded twice
        (files / f'{fid}.jpg').write_bytes(b'')
        obs.append(dict(kind='OBSERVATION', payload=dict(session_id='s1', track_id=7, physics_frames=dict(frames=[
            dict(frame=frame(fid, ts), storage_path=f'u/s1/u/s1/{fid}.jpg', context=dict(frame_lamps=[lamp]))]))))
    (tmp_path / 'records.jsonl').write_text('\n'.join(json.dumps(o) for o in obs))
    tele = tmp_path / 'files' / 'survey'
    tele.mkdir(parents=True)
    with gzip.open(tele / 'telemetry-00000.jsonl.gz', 'wt') as f:
        for k in range(5):
            f.write(json.dumps(dict(kind='location', data=dict(utc_ms=1000 * k, speed_mps=10.0, latitude=0, longitude=0, accuracy_m=5))) + '\n')
    return tmp_path


def test_frames_are_deduplicated_and_sorted(tmp_path):
    sdir = _session(tmp_path)
    obs, locs = P.load_session(str(sdir))
    frames = P.frames_of(obs, str(sdir))
    assert [f['frame']['frame_id'] for f in frames] == [1, 2] and len(locs) == 5
    assert frames[0]['lamps'][0]['track_id'] == 7


def test_distance_integrates_gps_speed():
    locs = [dict(utc_ms=1000 * k, speed_mps=10.0 if k else None) for k in range(5)]
    d = P.travelled(locs, np.array([0.0, 2000.0, 4000.0]))
    assert np.allclose(d, [0, 20, 40])
