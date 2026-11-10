#!/usr/bin/env python
"""P2-MMA summary: pooled endpoints, P1 classes and the readings M1-M3 (prereg.md).

Paired bootstrap over clips within folds (2000 resamples). The same resample is used for every arm and for
both models: MMA-DFER (none) and the OT router (scr) are scored on the identical test clips, so the
difference of their offset curves, Delta(d), is bootstrapped paired.
Usage: p2_mma_summary.py <results/p2_mma/DS> <results/p2_ot/DS>
"""

import json
import os
import sys

import numpy as np

B, ALPHA = 2000, 0.10
FOLDS = (1, 2, 3, 4, 5)
TAGS = ('none', 'scr')


def uar_w(pred, y, w, n_cls):
    rec = []
    for c in range(n_cls):
        m = y == c
        rec.append(w[:, m & (pred == c)].sum(1) / np.maximum(w[:, m].sum(1), 1))
    return 100.0 * np.mean(np.stack(rec, 1), 1)


def ci(x):
    return [float(v) for v in np.quantile(x, [ALPHA / 2, 1 - ALPHA / 2])]


def p1_class(r):
    if r['changed_pct'] < 1.0 and -1.0 <= r['d_uar_ci'][0] and r['d_uar_ci'][1] <= 1.0:
        return 'inert'
    if r['changed_pct'] >= 5.0 or (r['d_uar'] <= -1.0 and r['d_uar_ci'][1] < 0):
        return 'load-bearing'
    return 'weak'


def tolerance(A, delays):
    tol = {}
    for side, sgn in (('late', 1), ('early', -1)):
        best = 0
        for d in sorted(abs(x) for x in delays if np.sign(x) == sgn):
            if A[f'delay{sgn * d:+d}']['d_uar_ci'][0] >= -1.0:
                best = d
            else:
                break
        tol[side] = best
    return tol


def main(root, ot_root):
    rng = np.random.default_rng(0)
    data, meta = {}, {}
    for tag in TAGS:
        data[tag], meta[tag] = {}, {}
        for f in FOLDS:
            z = np.load(os.path.join(root, tag, f'fold{f}.npz'))
            data[tag][f] = {'y': z['labels'], 'pred': {k[5:]: z[k].argmax(1) for k in z.files if k.startswith('prob_')}}
            meta[tag][f] = json.load(open(os.path.join(root, tag, f'fold{f}.json')))
    for f in FOLDS:
        assert np.array_equal(data['none'][f]['y'], data['scr'][f]['y']), f'labels differ in fold {f}'
    y = np.concatenate([data['none'][f]['y'] for f in FOLDS])
    ncls = int(y.max()) + 1
    w = np.concatenate([rng.multinomial(len(data['none'][f]['y']),
                                        np.full(len(data['none'][f]['y']), 1 / len(data['none'][f]['y'])), size=B)
                        for f in FOLDS], axis=1)
    ones = np.ones((1, len(y)))
    delays = meta['none'][FOLDS[0]]['delays_ms']
    out = {'n_clips': int(len(y)), 'n_classes': ncls, 'models': {}}
    dub = {}
    for tag in TAGS:
        arms = list(data[tag][FOLDS[0]]['pred'])
        pred = {k: np.concatenate([data[tag][f]['pred'][k] for f in FOLDS]) for k in arms}
        base_u = float(uar_w(pred['base'], y, ones, ncls)[0])
        base_b = uar_w(pred['base'], y, w, ncls)
        M = {'base_uar': base_u, 'arms': {}, 'per_fold_d_uar': {},
             'check_copy_vs_base': [meta[tag][f]['check_copy_vs_base'] for f in FOLDS],
             'check_delay0_fbank': [meta[tag][f]['check_delay0_fbank'] for f in FOLDS]}
        dub[tag] = {}
        print(f'\n=== {tag}: {meta[tag][FOLDS[0]]["run"]}\npooled over folds {list(FOLDS)}: {len(y)} test clips; '
              f'base UAR {base_u:.2f}')
        print('arm            UAR    dUAR [90% CI]          changed %     class')
        for k in arms:
            u = float(uar_w(pred[k], y, ones, ncls)[0])
            dub[tag][k] = uar_w(pred[k], y, w, ncls) - base_b
            ch = pred[k] != pred['base']
            r = {'uar': u, 'd_uar': u - base_u, 'd_uar_ci': ci(dub[tag][k]), 'changed_pct': 100.0 * float(ch.mean())}
            r['class'] = p1_class(r)
            M['arms'][k] = r
            pf = []
            for f in FOLDS:
                yf = data[tag][f]['y']
                one = np.ones((1, len(yf)))
                pf.append(float(uar_w(data[tag][f]['pred'][k], yf, one, ncls)[0]
                                - uar_w(data[tag][f]['pred']['base'], yf, one, ncls)[0]))
            M['per_fold_d_uar'][k] = pf
            print(f'{k:<12} {u:6.2f}  {r["d_uar"]:+6.2f} [{r["d_uar_ci"][0]:+6.2f}, {r["d_uar_ci"][1]:+6.2f}]'
                  f'   {r["changed_pct"]:6.2f}   {r["class"]}')
        M['D1_tolerance_ms'] = tolerance(M['arms'], delays)
        M['delay_curve'] = {d: M['arms'][f'delay{d:+d}']['d_uar'] for d in sorted(delays)}
        out['models'][tag] = M

    # cross-check: zero_route on the OT router must reproduce P2-OT zero
    agree = []
    for f in FOLDS:
        p = os.path.join(ot_root, f'fold{f}.npz')
        if os.path.exists(p):
            z = np.load(p)
            if np.array_equal(z['labels'], data['scr'][f]['y']):
                agree.append(100.0 * float((z['prob_zero'].argmax(1) == data['scr'][f]['pred']['zero_route']).mean()))
    N, S = out['models']['none']['arms'], out['models']['scr']['arms']
    delta = {}
    for d in sorted(delays):
        k = f'delay{d:+d}'
        db = dub['scr'][k] - dub['none'][k]
        delta[d] = {'value': S[k]['d_uar'] - N[k]['d_uar'], 'ci': ci(db)}
    if all(-1.0 <= v['ci'][0] and v['ci'][1] <= 1.0 for v in delta.values()):
        m3 = 'OT routing adds no alignment sensitivity'
    elif any(v['value'] <= -1.0 and v['ci'][1] < 0 for v in delta.values()):
        m3 = 'adds alignment sensitivity'
    else:
        m3 = 'weak'
    out['readings'] = {
        'M1_zero_nat_class': N['zero_nat']['class'], 'M1_zero_all_class': N['zero_all']['class'],
        'M2_tolerance_none_ms': out['models']['none']['D1_tolerance_ms'],
        'tolerance_scr_ms': out['models']['scr']['D1_tolerance_ms'],
        'M3': m3, 'M3_delta': delta,
        'zero_route_agree_with_p2ot_zero_pct': agree,
    }
    print('\nreadings')
    for k, v in out['readings'].items():
        print(f'  {k}: {v}')
    print('\nper-fold dUAR')
    for tag in TAGS:
        for k, v in out['models'][tag]['per_fold_d_uar'].items():
            print(f'  {tag:<4} {k:<12} ' + '  '.join(f'{x:+6.2f}' for x in v))
    path = os.path.join(root, 'summary.json')
    json.dump(out, open(path, 'w'), indent=1)
    print(f'-> {path}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
