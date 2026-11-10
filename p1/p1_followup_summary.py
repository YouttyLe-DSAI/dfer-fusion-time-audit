#!/usr/bin/env python
"""P1-followup summary: pooled endpoints and the pre-registered readings T1 and D1 (prereg.md).

Paired bootstrap over clips within folds (2000 resamples, the same resample for every arm), 90% CIs.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_summary import uar_w  # noqa: E402

B, ALPHA = 2000, 0.10
FOLDS = (1, 2, 3, 4, 5)
ITU = {'early': 45, 'late': 125}           # ITU-R BT.1359 detectability thresholds, ms


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
    delays = meta[FOLDS[0]]['delays_ms']
    y = np.concatenate([data[f]['y'] for f in FOLDS])
    pred = {k: np.concatenate([data[f]['pred'][k] for f in FOLDS]) for k in arms}
    w = np.concatenate([rng.multinomial(len(data[f]['y']), np.full(len(data[f]['y']), 1 / len(data[f]['y'])),
                                        size=B) for f in FOLDS], axis=1)
    ones = np.ones((1, len(y)))
    base_u = float(uar_w(pred['base'], y, ones)[0])
    base_b = uar_w(pred['base'], y, w)
    out = {'n_clips': int(len(y)), 'base_uar': base_u, 'arms': {}, 'per_fold_d_uar': {}}
    print(f'pooled over folds {list(FOLDS)}: {len(y)} clips; base UAR {base_u:.2f} (seeded gain)')
    print('\narm               UAR    dUAR [90% CI]          changed %')
    for k in arms:
        u = float(uar_w(pred[k], y, ones)[0])
        db = uar_w(pred[k], y, w) - base_b
        r = {'uar': u, 'd_uar': u - base_u, 'd_uar_ci': ci(db),
             'changed_pct': 100.0 * float((pred[k] != pred['base']).mean())}
        out['arms'][k] = r
        out['per_fold_d_uar'][k] = [float(uar_w(data[f]['pred'][k], data[f]['y'], np.ones((1, len(data[f]['y']))))[0]
                                          - uar_w(data[f]['pred']['base'], data[f]['y'],
                                                  np.ones((1, len(data[f]['y']))))[0]) for f in FOLDS]
        print(f'{k:<15} {u:6.2f}  {r["d_uar"]:+6.2f} [{r["d_uar_ci"][0]:+6.2f}, {r["d_uar_ci"][1]:+6.2f}]'
              f'   {r["changed_pct"]:5.2f}')
    A = out['arms']

    def inert(r):
        return -1.0 <= r['d_uar_ci'][0] and r['d_uar_ci'][1] <= 1.0

    def hurts(r):
        return r['d_uar'] <= -1.0 and r['d_uar_ci'][1] < 0

    if inert(A['rowperm_s']) and hurts(A['rowperm_t']):
        t1 = 'time carries the alignment'
    elif hurts(A['rowperm_s']):
        t1 = 'space also carries alignment'
    else:
        t1 = 'mixed'
    tol = {}
    for side, sgn in (('late', 1), ('early', -1)):
        best = 0
        for d in sorted(abs(x) for x in delays if np.sign(x) == sgn):
            if A[f'delay{sgn * d:+d}']['d_uar_ci'][0] >= -1.0:
                best = d
            else:
                break
        tol[side] = best
    out['readings'] = {
        'T1': t1, 'D1_tolerance_ms': tol, 'ITU_R_BT1359_detectability_ms': ITU,
        'delay_curve': {d: A[f'delay{d:+d}']['d_uar'] for d in sorted(delays)},
        'delay_curve_fus': {d: A[f'delay{d:+d}_fus']['d_uar'] for d in sorted(delays)},
        'silence': A['silence']['d_uar'], 'silence_fus': A['silence_fus']['d_uar'],
    }
    print('\npre-registered readings')
    for k, v in out['readings'].items():
        print(f'  {k}: {v}')
    print('\nper-fold dUAR')
    for k in arms:
        print(f'  {k:<15} ' + '  '.join(f'{v:+6.2f}' for v in out['per_fold_d_uar'][k]))
    out['sanity'] = {f: meta[f]['check_max_diff'] for f in FOLDS}
    print(f'\nsanity: max |model() - base| per fold {out["sanity"]}')
    path = prefix.rsplit('/', 1)[0] + '/summary.json'
    json.dump(out, open(path, 'w'), indent=1)
    print(f'-> {path}')


if __name__ == '__main__':
    main(sys.argv[1])
