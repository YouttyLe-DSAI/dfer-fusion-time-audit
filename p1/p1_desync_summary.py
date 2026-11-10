#!/usr/bin/env python
"""P1-desync-control summary: pooled endpoints, D_d = dUAR(both_d) - dUAR(aud_d), readings C1-C3 (prereg.md)."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_summary import uar_w  # noqa: E402

B, ALPHA = 2000, 0.10
FOLDS = (1, 2, 3, 4, 5)


def ci(x):
    return [float(v) for v in np.quantile(x, [ALPHA / 2, 1 - ALPHA / 2])]


def main(prefix):
    rng = np.random.default_rng(0)
    data, meta = {}, {}
    for f in FOLDS:
        z = np.load(f'{prefix}{f}.npz')
        data[f] = {'y': z['labels'], 'pred': {k[5:]: z[k].argmax(1) for k in z.files if k.startswith('prob_')}}
        meta[f] = json.load(open(f'{prefix}{f}.json'))
    arms = list(data[FOLDS[0]]['pred'])
    D = meta[FOLDS[0]]['d_ms']
    y = np.concatenate([data[f]['y'] for f in FOLDS])
    pred = {k: np.concatenate([data[f]['pred'][k] for f in FOLDS]) for k in arms}
    w = np.concatenate([rng.multinomial(len(data[f]['y']), np.full(len(data[f]['y']), 1 / len(data[f]['y'])),
                                        size=B) for f in FOLDS], axis=1)
    ones = np.ones((1, len(y)))
    base_u = float(uar_w(pred['base'], y, ones)[0])
    base_b = uar_w(pred['base'], y, w)
    out = {'n_clips': int(len(y)), 'base_uar': base_u, 'arms': {}, 'D': {}, 'per_fold_d_uar': {}}
    dub = {}
    print(f'pooled over folds {list(FOLDS)}: {len(y)} clips; base UAR {base_u:.2f}')
    print('\narm          UAR    dUAR [90% CI]          changed %')
    for k in arms:
        u = float(uar_w(pred[k], y, ones)[0])
        dub[k] = uar_w(pred[k], y, w) - base_b
        r = {'uar': u, 'd_uar': u - base_u, 'd_uar_ci': ci(dub[k]),
             'changed_pct': 100.0 * float((pred[k] != pred['base']).mean())}
        out['arms'][k] = r
        out['per_fold_d_uar'][k] = [float(uar_w(data[f]['pred'][k], data[f]['y'], np.ones((1, len(data[f]['y']))))[0]
                                          - uar_w(data[f]['pred']['base'], data[f]['y'],
                                                  np.ones((1, len(data[f]['y']))))[0]) for f in FOLDS]
        print(f'{k:<10} {u:6.2f}  {r["d_uar"]:+6.2f} [{r["d_uar_ci"][0]:+6.2f}, {r["d_uar_ci"][1]:+6.2f}]'
              f'   {r["changed_pct"]:5.2f}')
    A = out['arms']
    readings = {}
    print('\nC1: D_d = dUAR(both_d) - dUAR(aud_d)')
    for d in D:
        a_, b_ = f'aud{d:+d}', f'both{d:+d}'
        val = A[b_]['d_uar'] - A[a_]['d_uar']
        c = ci(dub[b_] - dub[a_])
        aud_hurts = A[a_]['d_uar'] <= -1.0 and A[a_]['d_uar_ci'][1] < 0
        if not aud_hurts:
            rd = 'not applicable'
        elif val >= 1.0 and c[0] > 0:
            rd = 'alignment'
        elif -1.0 <= c[0] and c[1] <= 1.0:
            rd = 'content'
        else:
            rd = 'mixed'
        out['D'][str(d)] = {'value': val, 'ci': c, 'reading': rd}
        c2 = A[f'vid{d:+d}']['d_uar'] - A[a_]['d_uar']
        out['D'][str(d)]['vid_minus_aud'] = {'value': c2, 'ci': ci(dub[f'vid{d:+d}'] - dub[a_])}
        print(f'  d={d:+5d}: D = {val:+5.2f} [{c[0]:+5.2f}, {c[1]:+5.2f}]  -> {rd};   vid - aud = {c2:+5.2f}')
    app = [v['reading'] for v in out['D'].values() if v['reading'] != 'not applicable']
    if app.count('alignment') >= 2 and 'content' not in app:
        overall = 'alignment'
    elif app.count('content') >= 2 and 'alignment' not in app:
        overall = 'content'
    else:
        overall = 'mixed'
    readings['C1'] = overall
    readings['C3_vid_early_minus_late_160'] = {
        'value': A['vid-160']['d_uar'] - A['vid+160']['d_uar'], 'ci': ci(dub['vid-160'] - dub['vid+160'])}
    readings['C3_aud_early_minus_late_160'] = {
        'value': A['aud-160']['d_uar'] - A['aud+160']['d_uar'], 'ci': ci(dub['aud-160'] - dub['aud+160'])}
    out['readings'] = readings
    rm = {m: [float(np.mean([meta[f]['realised_move_ms'][m][0] for f in FOLDS])),
              min(meta[f]['realised_move_ms'][m][1] for f in FOLDS),
              max(meta[f]['realised_move_ms'][m][2] for f in FOLDS)] for m in meta[FOLDS[0]]['realised_move_ms']}
    out['realised_move_ms'] = rm
    out['sanity'] = {f: meta[f]['check_max_diff'] for f in FOLDS}
    print(f'\nC1 overall: {overall}')
    print(f'C3 (video moved, audio intact): vid-160 - vid+160 = {readings["C3_vid_early_minus_late_160"]}')
    print(f'   (audio moved):               aud-160 - aud+160 = {readings["C3_aud_early_minus_late_160"]}')
    print(f'realised moves (mean, min, max ms): {rm}')
    print(f'sanity: max |model() - base| per fold {out["sanity"]}')
    print('\nper-fold dUAR')
    for k in arms:
        print(f'  {k:<10} ' + '  '.join(f'{v:+6.2f}' for v in out['per_fold_d_uar'][k]))
    path = prefix.rsplit('/', 1)[0] + '/summary.json'
    json.dump(out, open(path, 'w'), indent=1)
    print(f'-> {path}')


if __name__ == '__main__':
    main(sys.argv[1])
