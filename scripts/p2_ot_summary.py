#!/usr/bin/env python
"""P2-OT summary: pooled endpoints, P1 classes, readings O1 (coupling) and O2 (removal) (prereg.md).

Paired bootstrap over clips within folds (2000 resamples, same resample for every arm), 90% CIs.
Usage: p2_ot_summary.py <prefix> <run_dir>   (prefix = .../fold, reads <prefix>{1..5}.npz/.json)
"""

import json
import os
import sys

import numpy as np

B, ALPHA = 2000, 0.10
FOLDS = (1, 2, 3, 4, 5)


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


def main(prefix, run_dir):
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
    val10 = []
    for f in FOLDS:
        mp = os.path.join(run_dir, f'fold{f}', 'metrics.json')
        val10.append(json.load(open(mp))['uar'] if os.path.exists(mp) else float('nan'))
    out = {'n_clips': int(len(y)), 'n_classes': ncls, 'base_uar_test': base_u, 'val10_uar_per_fold': val10,
           'arms': {}, 'per_fold_d_uar': {}}
    print(f'{run_dir}\npooled over folds {list(FOLDS)}: {len(y)} test clips; base UAR {base_u:.2f} '
          f'(val10 UAR per fold: {", ".join(f"{v:.2f}" for v in val10)})')
    print('\narm          UAR    dUAR [90% CI]          changed % [90% CI]     class')
    for k in arms:
        u = float(uar_w(pred[k], y, ones, ncls)[0])
        db = uar_w(pred[k], y, w, ncls) - base_b
        ch = pred[k] != pred['base']
        chb = 100.0 * (w * ch).sum(1) / w.sum(1)
        r = {'uar': u, 'd_uar': u - base_u, 'd_uar_ci': ci(db), 'changed_pct': 100.0 * float(ch.mean()),
             'changed_ci': ci(chb)}
        r['class'] = p1_class(r)
        out['arms'][k] = r
        pf = []
        for f in FOLDS:
            yf = data[f]['y']
            one = np.ones((1, len(yf)))
            pf.append(float(uar_w(data[f]['pred'][k], yf, one, ncls)[0] - uar_w(data[f]['pred']['base'], yf, one, ncls)[0]))
        out['per_fold_d_uar'][k] = pf
        print(f'{k:<10} {u:6.2f}  {r["d_uar"]:+6.2f} [{r["d_uar_ci"][0]:+6.2f}, {r["d_uar_ci"][1]:+6.2f}]'
              f'   {r["changed_pct"]:5.2f} [{r["changed_ci"][0]:5.2f}, {r["changed_ci"][1]:5.2f}]   {r["class"]}')
    A = out['arms']
    cls = {A['indep']['class'], A['rowperm']['class']}
    o1 = ('OT routing predictively inert' if cls == {'inert'} else
          'load-bearing' if 'load-bearing' in cls else 'weak')
    out['readings'] = {'O1': o1, 'O2_zero_class': A['zero']['class'],
                       'check_copy_vs_base': [meta[f]['check_copy_vs_base'] for f in FOLDS],
                       'gate_tanh_a2v_mean_abs': float(np.mean([np.mean(np.abs(meta[f]['gate_tanh_a2v'])) for f in FOLDS]))}
    print('\nper-fold dUAR')
    for k in arms:
        print(f'  {k:<10} ' + '  '.join(f'{v:+6.2f}' for v in out['per_fold_d_uar'][k]))
    print('\nreadings')
    for k, v in out['readings'].items():
        print(f'  {k}: {v}')
    path = prefix.rsplit('/', 1)[0] + '/summary.json'
    json.dump(out, open(path, 'w'), indent=1)
    print(f'-> {path}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
