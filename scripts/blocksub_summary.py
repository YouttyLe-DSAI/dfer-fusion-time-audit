#!/usr/bin/env python
"""Block-subset summary (prereg.md): pooled endpoints as P2-OT (paired clip bootstrap 2000, 90% CI), P1 class,
per-fold dUAR, the trace, and the P2-OT reproduction check.

Usage: blocksub_summary.py <prefix> [<p2_ot fold prefix for the check>]   (reads <prefix>{1..5}.npz/.json)
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2_ot_summary import B, FOLDS, ci, p1_class, uar_w  # noqa: E402


def main(prefix, check_prefix=None):
    rng = np.random.default_rng(0)
    data, meta = {}, {}
    for f in FOLDS:
        z = np.load(f'{prefix}{f}.npz')
        data[f] = {'y': z['labels'], 'pred': {k[5:]: z[k].argmax(1) for k in z.files if k.startswith('prob_')}}
        meta[f] = json.load(open(f'{prefix}{f}.json'))
    arms = list(data[FOLDS[0]]['pred'])
    y = np.concatenate([data[f]['y'] for f in FOLDS])
    ncls = int(y.max()) + 1
    pred = {k: np.concatenate([data[f]['pred'][k] for f in FOLDS]) for k in arms}
    w = np.concatenate([rng.multinomial(len(data[f]['y']), np.full(len(data[f]['y']), 1 / len(data[f]['y'])),
                                        size=B) for f in FOLDS], axis=1)
    ones = np.ones((1, len(y)))
    base_u = float(uar_w(pred['base'], y, ones, ncls)[0])
    base_b = uar_w(pred['base'], y, w, ncls)
    out = {'run': meta[1]['run'], 'n_clips': int(len(y)), 'base_uar': base_u, 'arms': {}, 'per_fold_d_uar': {},
           'trace': {}}
    print(f'{meta[1]["run"]}\n{len(y)} test clips, base UAR {base_u:.2f}')
    print('\narm            UAR    dUAR [90% CI]          changed % [90% CI]     class')
    for k in arms:
        u = float(uar_w(pred[k], y, ones, ncls)[0])
        db = uar_w(pred[k], y, w, ncls) - base_b
        ch = pred[k] != pred['base']
        chb = 100.0 * (w * ch).sum(1) / w.sum(1)
        r = {'uar': u, 'd_uar': u - base_u, 'd_uar_ci': ci(db), 'changed_pct': 100.0 * float(ch.mean()),
             'changed_ci': ci(chb)}
        r['class'] = p1_class(r) if k != 'base' else '-'
        out['arms'][k] = r
        out['per_fold_d_uar'][k] = [meta[f]['arms'][k]['uar'] - meta[f]['arms']['base']['uar'] for f in FOLDS]
        print(f'{k:<12} {u:6.2f}  {r["d_uar"]:+6.2f} [{r["d_uar_ci"][0]:+6.2f}, {r["d_uar_ci"][1]:+6.2f}]'
              f'   {r["changed_pct"]:5.2f} [{r["changed_ci"][0]:5.2f}, {r["changed_ci"][1]:5.2f}]   {r["class"]}')
    print('\nper-fold dUAR')
    for k in arms[1:]:
        print(f'  {k:<12} ' + '  '.join(f'{v:+6.2f}' for v in out['per_fold_d_uar'][k]))
    keys = list(meta[1]['trace'][arms[1]])
    print('\ntrace (relative change against base, mean over clips and folds): ' + ', '.join(keys))
    for k in arms[1:]:
        out['trace'][k] = {t: float(np.mean([meta[f]['trace'][k][t] for f in FOLDS])) for t in keys}
        print(f'  {k:<12} ' + '  '.join(f'{t} {out["trace"][k][t]:.3e}' for t in keys))
    if 'tanh_all_gate' in meta[1]:
        out['tanh_all_gate_mean'] = np.mean([meta[f]['tanh_all_gate'] for f in FOLDS], 0).tolist()
        out['tanh_gate_a2v_mean'] = np.mean([meta[f]['gate_tanh_a2v'] for f in FOLDS], 0).tolist()
        print('tanh(all_gate) per block, fold mean: ' + ' '.join(f'{v:+.3f}' for v in out['tanh_all_gate_mean']))
        print('tanh(g_a2v) per block, fold mean:    ' + ' '.join(f'{v:+.3f}' for v in out['tanh_gate_a2v_mean']))
    if check_prefix:
        diffs = []
        for f in FOLDS:
            ref = json.load(open(f'{check_prefix}{f}.json'))['arms']
            diffs.append((meta[f]['arms']['base']['uar'] - ref['base']['uar'],
                          meta[f]['arms']['indep_all']['uar'] - ref['indep']['uar']))
        ok = all(abs(x) < 1e-9 and abs(z) < 1e-9 for x, z in diffs)
        out['check_p2_ot'] = {'diffs_base_indep': diffs, 'pass': ok}
        print(f'\nsanity: base / indep_all vs P2-OT per fold {diffs} -> {"PASS" if ok else "FAIL"}')
    path = prefix.rsplit('/', 1)[0] + '/summary.json'
    json.dump(out, open(path, 'w'), indent=1)
    print(f'-> {path}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
