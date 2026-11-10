#!/usr/bin/env python
"""P2T stage 0 summary (prereg.md): pooled endpoints and the readings O, R, K (all models), M, S (HiCMAE).

Paired bootstrap over clips within folds (2000 resamples, 90% CI), the same resample for every arm of a pair.
Usage: p2t_summary.py <dir with fold1..5.{json,npz}> [<label>]
"""

import json
import os
import sys

import numpy as np

B, ALPHA, FOLDS, MARGIN = 2000, 0.10, (1, 2, 3, 4, 5), 1.0
HOLDS = (8, 4, 2, 1)


def uar_w(pred, y, w, n_cls):
    rec = [w[:, (y == c) & (pred == c)].sum(1) / np.maximum(w[:, y == c].sum(1), 1) for c in range(n_cls)
           if (y == c).any()]
    return 100.0 * np.mean(np.stack(rec, 1), 1)


def ci(x):
    return [float(v) for v in np.quantile(x, [ALPHA / 2, 1 - ALPHA / 2])]


def cls3(r, pos='uses order', neg='order-invariant'):
    lo, hi = r['d_uar_ci']
    if r['d_uar'] <= -MARGIN and hi < 0:
        return pos
    if -MARGIN <= lo and hi <= MARGIN:
        return neg
    return 'inconclusive'


def main(d, label):
    rng = np.random.default_rng(0)
    Z = {f: np.load(os.path.join(d, f'fold{f}.npz'), allow_pickle=False) for f in FOLDS}
    J = {f: json.load(open(os.path.join(d, f'fold{f}.json'))) for f in FOLDS}
    arms = [k[5:] for k in Z[1].files if k.startswith('prob_')]
    y = np.concatenate([Z[f]['labels'] for f in FOLDS])
    ncls = int(max(Z[f]['prob_base'].shape[1] for f in FOLDS))
    w = np.concatenate([rng.multinomial(len(Z[f]['labels']), np.full(len(Z[f]['labels']), 1 / len(Z[f]['labels'])),
                                        size=B) for f in FOLDS], axis=1)
    ones = np.ones((1, len(y)))
    pred = {k: np.concatenate([Z[f][f'prob_{k}'].argmax(1) for f in FOLDS]) for k in arms}
    base_u, base_b = float(uar_w(pred['base'], y, ones, ncls)[0]), uar_w(pred['base'], y, w, ncls)
    Db, A = {}, {}
    for k in arms:
        u = float(uar_w(pred[k], y, ones, ncls)[0])
        Db[k] = uar_w(pred[k], y, w, ncls) - base_b
        A[k] = {'uar': u, 'd_uar': u - base_u, 'd_uar_ci': ci(Db[k]),
                'changed_pct': 100.0 * float((pred[k] != pred['base']).mean()),
                'per_fold_d_uar': [J[f]['arms'][k]['uar'] - J[f]['arms']['base']['uar'] for f in FOLDS]}

    def diff(a, b):
        return {'value': A[a]['d_uar'] - A[b]['d_uar'], 'ci': ci(Db[a] - Db[b])}

    rd = {'O_order': cls3(A['shuffle_s0']),
          'O_seed_classes': [cls3(A[k]) for k in ('shuffle_s0', 'shuffle_s1', 'shuffle_s2')],
          'R_direction': cls3(A['reverse'], 'uses direction', 'direction-invariant')}
    K = 'none'
    for k in HOLDS:
        if A[f'hold{k}']['d_uar_ci'][0] > -MARGIN:
            K = k
        else:
            break
    rd['K_frame_budget'] = K
    if 'shuffle_pair_av' in A:
        m, s = diff('shuffle_pair', 'shuffle_s0'), diff('shuffle_pair_av', 'shuffle_pair')
        rd['M_local_motion'] = {**m, 'reading': 'local motion used' if m['value'] >= MARGIN and m['ci'][0] > 0 else 'not shown'}
        rd['S_av_sync'] = {**s, 'reading': 'sync part of the cost' if s['value'] >= MARGIN and s['ci'][0] > 0 else 'not shown'}
    sanity = {'identity_vs_base_max': max(J[f]['check_identity_vs_base'] for f in FOLDS),
              'identity_changed_pct': A['identity']['changed_pct']}
    if 'run_metrics_uar' in J[1]:
        sanity['base_minus_run_metrics_uar'] = [None if J[f].get('run_metrics_uar') is None else
                                                J[f]['arms']['base']['uar'] - J[f]['run_metrics_uar'] for f in FOLDS]
    out = {'label': label, 'dir': d, 'n_clips': int(len(y)), 'n_classes': ncls, 'base_uar': base_u,
           'base_uar_per_fold': [J[f]['arms']['base']['uar'] for f in FOLDS], 'sanity': sanity, 'arms': A,
           'readings': rd}
    print(f'=== {label}: {len(y)} test clips, 5 folds, base UAR {base_u:.2f}')
    print(f'sanity: {sanity}')
    print('arm              UAR    dUAR [90% CI]          changed %')
    for k in arms:
        r = A[k]
        print(f'{k:<16} {r["uar"]:6.2f}  {r["d_uar"]:+6.2f} [{r["d_uar_ci"][0]:+6.2f}, {r["d_uar_ci"][1]:+6.2f}]   '
              f'{r["changed_pct"]:6.2f}')
    print('readings:', json.dumps(rd))
    json.dump(out, open(os.path.join(d, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.basename(sys.argv[1].rstrip('/')))
