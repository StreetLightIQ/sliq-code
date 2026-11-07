"""Collect the distillation and quantisation table rows from finished runs into runs/tables/.

  students.json  every student run at its val tau, plus mean and SD over seeds per configuration
  quant.json     every exported .tflite scored on validation (and on the test footage if scored)

    python -m sliq.deploy.collect"""
import glob
import json
import os
import statistics

from sliq.common import paths

KEYS = ('mAP50', 'mAP50_95', 'recall', 'precision', 'fp_per_frame')


def run_summary(d):
    sel = json.load(open(d + '/selection.json'))
    rec = json.load(open(d + '/run_record.json'))
    at = sel['val_at_tau'] or sel['val_at_025']
    return dict(run=os.path.basename(d), model=rec['args']['model'], imgsz=rec['args']['imgsz'], seed=rec['args']['seed'], kd=rec['kd']['mode'],
                kd_weights=rec['kd']['weights'] if rec['kd']['mode'] != 'off' else None, selected=sel['best_checkpoint'], tau=sel['tau_star'],
                val_at_tau={k: at[k] for k in KEYS}, train_minutes=round(sel['train_seconds'] / 60))


def config_key(r):
    w = r['kd_weights']
    return f"{r['model']}@{r['imgsz']} kd={r['kd']}" + (f" cls{w['cls']:g}/box{w['box']:g}/feat{w['feat']:g}" if w else '')


def main():
    out_dir = paths.RUNS + '/tables/'
    os.makedirs(out_dir, exist_ok=True)
    runs = []
    for d in sorted(glob.glob(paths.RUNS + '/detector/*')):
        if os.path.exists(d + '/selection.json') and 'kd' in json.load(open(d + '/run_record.json')):
            runs.append(run_summary(d))
    configs = {}
    for r in runs:
        configs.setdefault(config_key(r), []).append(r)
    seeds = {}
    for k, rs in configs.items():
        vals = [r['val_at_tau']['mAP50'] for r in rs]
        seeds[k] = dict(runs=[r['run'] for r in rs], seeds=[r['seed'] for r in rs], mAP50=vals, mean=statistics.mean(vals),
                        sd=statistics.stdev(vals) if len(vals) > 1 else None)
    json.dump(dict(runs=runs, configs=seeds), open(out_dir + 'students.json', 'w'), indent=1)

    test = {}
    for f in glob.glob(paths.RUNS + '/deploy/test_footage/*.json'):
        test.update({k: v['at_val_tau'] for k, v in json.load(open(f))['files'].items()})
    exports = []
    for f in sorted(glob.glob(paths.RUNS + '/deploy/*/*_val.json')):
        v = json.load(open(f))
        s = v['at_own_tau']
        name = os.path.basename(v['file'])
        exports.append(dict(file=name, input_hw=v['input_hw'], bytes=os.path.getsize(v['file']) if os.path.exists(v['file']) else None,
                            tau=s['tau'], **{k: s[k] for k in KEYS}, cpu_ms_median=v['cpu_ms']['median'], test_footage=test.get(name)))
    json.dump(dict(exports=exports), open(out_dir + 'quant.json', 'w'), indent=1)
    print('wrote', len(runs), 'student runs and', len(exports), 'exports to', out_dir)


if __name__ == '__main__':
    main()
