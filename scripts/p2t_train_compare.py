#!/usr/bin/env python
"""P2T follow-up (b), prereg.md: the order-agnostic training control against its reference run.

Both dirs hold p2t_mma.py outputs (fold{1..5}.{json,npz}) for the same test folds and clips.
Endpoint: dUAR_train = UAR(control, base arm) - UAR(reference, base arm), pooled, paired bootstrap over clips within
folds (2000, 90% CI, seed 0). Also each stage-0 arm's dUAR for both models and their frame budgets K.
Usage: p2t_train_compare.py <reference dir> <control dir> <label> [<out json>]
"""

import json
import os
import sys

import numpy as np

B, ALPHA, FOLDS, MARGIN = 2000, 0.10, (1, 2, 3, 4, 5), 1.0
HOLDS = (8, 4, 2, 1)


def uar_w(pred, y, w):
    rec = [w[:, (y == c) & (pred == c)].sum(1) / np.maximum(w[:, y == c].sum(1), 1e-12) for c in np.unique(y)]
    return 100.0 * np.mean(np.stack(rec, 1), 1)


def ci(x):
    return [float(v) for v in np.quantile(x, [ALPHA / 2, 1 - ALPHA / 2])]


def load(d):
    Z = [np.load(os.path.join(d, f'fold{f}.npz'), allow_pickle=False) for f in FOLDS]
    arms = [k[5:] for k in Z[0].files if k.startswith('prob_')]
    return Z, arms


def main(ref_dir, ctl_dir, label, out=None):
    R, arms = load(ref_dir)
    C, arms_c = load(ctl_dir)
    assert arms == arms_c, (arms, arms_c)
    for f, (r, c) in enumerate(zip(R, C), 1):
        assert np.array_equal(r['clips'], c['clips']) and np.array_equal(r['labels'], c['labels']), f'fold {f} clips differ'
    y = np.concatenate([z['labels'] for z in R])
    sizes = [len(z['labels']) for z in R]
    rng = np.random.default_rng(0)
    w = np.concatenate([rng.multinomial(n, np.full(n, 1 / n), size=B) for n in sizes], axis=1).astype(np.float64)
    one = np.ones((1, len(y)))
    pr = {k: np.concatenate([z[f'prob_{k}'].argmax(1) for z in R]) for k in arms}
    pc = {k: np.concatenate([z[f'prob_{k}'].argmax(1) for z in C]) for k in arms}
    u = {(m, k): float(uar_w(p[k], y, one)[0]) for m, p in (('ref', pr), ('ctl', pc)) for k in arms}
    ub = {(m, k): uar_w(p[k], y, w) for m, p in (('ref', pr), ('ctl', pc)) for k in arms}

    d = u[('ctl', 'base')] - u[('ref', 'base')]
    lo, hi = ci(ub[('ctl', 'base')] - ub[('ref', 'base')])
    if lo > -MARGIN:
        reading = 'order not needed'
    elif d <= -MARGIN and hi < 0:
        reading = 'order used for accuracy'
    else:
        reading = 'inconclusive'
    folds = []
    off = 0
    for n in sizes:
        s = slice(off, off + n)
        o1 = np.ones((1, n))
        folds.append(float(uar_w(pc['base'][s], y[s], o1)[0] - uar_w(pr['base'][s], y[s], o1)[0]))
        off += n
    res = {'label': label, 'ref_dir': ref_dir, 'ctl_dir': ctl_dir, 'n_clips': int(len(y)),
           'ref_uar': u[('ref', 'base')], 'ctl_uar': u[('ctl', 'base')], 'd_uar_train': d, 'ci': [lo, hi],
           'per_fold': folds, 'reading': reading,
           'changed_pct': 100.0 * float((pc['base'] != pr['base']).mean()), 'arms': {}}
    K = {}
    for m in ('ref', 'ctl'):
        K[m] = 'none'
        for k in HOLDS:
            if ci(ub[(m, f'hold{k}')] - ub[(m, 'base')])[0] > -MARGIN:
                K[m] = k
            else:
                break
    res['K'] = K
    print(f'=== {label}: {len(y)} test clips; reference {res["ref_uar"]:.2f}, shuffle-trained {res["ctl_uar"]:.2f}')
    print(f'dUAR_train {d:+.2f} [{lo:+.2f}, {hi:+.2f}]  per fold {[round(x, 2) for x in folds]}  -> {reading}'
          f'  ({res["changed_pct"]:.1f}% predictions differ)')
    print('arm              ref dUAR [90% CI]         ctl dUAR [90% CI]')
    for k in arms:
        if k in ('base', 'identity'):
            continue
        e = {}
        for m in ('ref', 'ctl'):
            e[m] = {'d_uar': u[(m, k)] - u[(m, 'base')], 'ci': ci(ub[(m, k)] - ub[(m, 'base')])}
        res['arms'][k] = e
        print(f'{k:<16} {e["ref"]["d_uar"]:+6.2f} [{e["ref"]["ci"][0]:+6.2f},{e["ref"]["ci"][1]:+6.2f}]   '
              f'{e["ctl"]["d_uar"]:+6.2f} [{e["ctl"]["ci"][0]:+6.2f},{e["ctl"]["ci"][1]:+6.2f}]')
    print(f'K: reference {K["ref"]}, shuffle-trained {K["ctl"]}')
    if out:
        json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:4], *(sys.argv[4:5] or [None]))
