"""On-device numbers from recorded survey sessions, grouped by deployed model.

Each session folder holds records.jsonl (with the SESSION record) and files/survey/telemetry-*.jsonl.gz.
Latency and frame counts come from the session diagnostics, battery and temperature from the telemetry.
--drives gives speed and lamps per km of the recorded drives (physics outputs, train and val).
--memory measures the LiteRT runtime memory of a model file on this machine.
    python -m sliq.deploy.device_report <session_dir> [<session_dir> ...]
    python -m sliq.deploy.device_report --drives
    python -m sliq.deploy.device_report --memory weights/student_n_int8.tflite
"""
import argparse
import collections
import glob
import gzip
import json
import os
import resource

import numpy as np


def read_session(sdir):
    session = None
    for line in open(os.path.join(sdir, 'records.jsonl')):
        r = json.loads(line)
        if r['kind'] == 'SESSION':
            session = r['payload']
    status, bad = [], 0
    for f in sorted(glob.glob(os.path.join(sdir, 'files/survey/telemetry-*.jsonl.gz'))):
        for line in gzip.open(f, 'rt'):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            d = e['data']
            if e['kind'] == 'device_status':
                status.append((e['utc_ms'], 100.0 * d['battery_level'] / d['battery_scale'], d['battery_temperature_tenths_c'] / 10))
    return dict(session=session, status=sorted(status), bad_lines=bad)


def battery_per_hour(status):
    """Drain from the first time each battery level is reached (levels are whole percent)."""
    first = {}
    for t, level, _ in status:
        first.setdefault(round(level), t)
    levels = sorted(first, reverse=True)
    if len(levels) < 3:
        return None
    a, b = levels[1], levels[-1]           # the starting level may be partial
    hours = (first[b] - first[a]) / 3.6e6
    return (a - b) / hours if hours > 0 else None


def end_ms(s):
    """Session end, or the last telemetry reading for a session that was not closed."""
    return s['session']['end_utc_ms'] or (s['status'][-1][0] if s['status'] else s['session']['start_utc_ms'])


def report(dirs):
    groups = collections.defaultdict(list)
    for d in dirs:
        s = read_session(d)
        if s['session'] is None:
            print('no SESSION record in', d)
            continue
        m = s['session']['model_deployment']
        groups[(m.get('model_version'), m.get('quantization'))].append(s)
    for (model, quant), ss in groups.items():
        diag = [s['session']['diagnostics'] for s in ss]
        frames = np.array([g['processed_frames'] for g in diag], float)
        ms = np.array([g['mean_inference_ms'] or np.nan for g in diag], float)
        dur = np.array([(end_ms(s) - s['session']['start_utc_ms']) / 1000 for s in ss], float)
        fps = frames / np.maximum(dur, 1)
        status = sorted(x for s in ss for x in s['status'])
        temps = [x[2] for x in status]
        drain = battery_per_hour(status)
        print(f'{model} ({quant}): {len(ss)} sessions, {int(frames.sum())} frames')
        print(f'  latency {np.nansum(ms * frames) / frames[np.isfinite(ms)].sum():.1f} ms, pipeline {fps.min():.1f} to {fps.max():.1f} fps, '
              f"failed frames {sum(g['failed_frames'] for g in diag)}")
        if drain:
            print(f'  battery {drain:.1f} %/h ({100 / drain:.1f} h per charge)')
        if temps:
            print(f'  battery temperature {min(temps):.1f} to {max(temps):.1f} C')


def drives(name='field', splits=('train', 'val')):
    """Mean speed and localised lamps per km over the drive spans (lead-in excluded)."""
    from sliq.physics import fieldmodel
    lamps = metres = seconds = 0.0
    for split in splits:
        for f in glob.glob(f'{fieldmodel.ROOT}field/{name}/{split}/*.json'):
            r = json.load(open(f))
            frames = [x for x in json.load(open(f'{fieldmodel.ROOT}cache/{split}/{r["span"]}/frames.json'))['frames'] if not x['lead_in']]
            lamps += len(r['lamps'])
            metres += r['travelled_m']
            seconds += frames[-1]['t'] - frames[0]['t']
    print(f'drives: {metres / 1000:.1f} km, {3.6 * metres / seconds:.0f} km/h, {1000 * lamps / metres:.0f} lamps/km')


def memory(path, runs=20):
    """Resident memory added by loading and running one LiteRT model (MB)."""
    from ai_edge_litert.interpreter import Interpreter

    def rss():
        for line in open('/proc/self/status'):
            if line.startswith('VmRSS'):
                return int(line.split()[1]) / 1024
    base = rss()
    it = Interpreter(model_path=path, num_threads=4)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    x = np.random.rand(*inp['shape']).astype(inp['dtype'])
    for _ in range(runs):
        it.set_tensor(inp['index'], x)
        it.invoke()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f'{os.path.basename(path)}: file {os.path.getsize(path) / 1e6:.1f} MB, peak runtime memory +{peak - base:.0f} MB')


def main():
    a = argparse.ArgumentParser(description='On-device numbers from survey sessions')
    a.add_argument('sessions', nargs='*')
    a.add_argument('--drives', nargs='?', const='field', help='physics output name (default field)')
    a.add_argument('--memory', help='LiteRT model file to measure (one per process)')
    args = a.parse_args()
    if args.sessions:
        report(args.sessions)
    if args.drives:
        drives(args.drives)
    if args.memory:
        memory(args.memory)


if __name__ == '__main__':
    main()
