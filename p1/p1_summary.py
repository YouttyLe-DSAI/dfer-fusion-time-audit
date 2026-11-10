#!/usr/bin/env python
"""P1 summary: stage-0 reproduction verdict and stage-1 intervention endpoints, pooled over folds.

Paired bootstrap over clips within each fold (same resample for both arms), 2000 resamples,
90% percentile CI for pooled dUAR and pooled % predictions changed. Rules: prereg.md, P1.
"""

import json
import sys

import numpy as np

from p1_common import DATASETS

B, ALPHA = 2000, 0.10


def uar_w(pred, y, w, n_cls=None):
    """UAR [B] under resample weights w [B, n]; n_cls defaults to the classes present in y."""
    n_cls = n_cls or int(y.max()) + 1
    rec = []
    for c in range(n_cls):
        m = y == c
        tot = w[:, m].sum(1)
        rec.append(w[:, m & (pred == c)].sum(1) / np.maximum(tot, 1))
    return 100.0 * np.mean(np.stack(rec, 1), 1)


def main(prefix, dataset='cremad', folds=(1, 2, 3, 4, 5)):
    cfg = DATASETS[dataset]
    README_UAR = cfg['readme_uar']
    README_TOTAL_UAR, README_TOTAL_WAR = cfg['readme_total']
    rng = np.random.default_rng(0)
    data = {}
    for f in folds:
        z = np.load(f'{prefix}{f}.npz')
        arms = [k[5:] for k in z.files if k.startswith('prob_')]
        data[f] = {'y': z['labels'], 'pred': {a: z[f'prob_{a}'].argmax(1) for a in arms}}
    arms = list(data[folds[0]]['pred'])
    y_all = np.concatenate([data[f]['y'] for f in folds])
    pred_all = {a: np.concatenate([data[f]['pred'][a] for f in folds]) for a in arms}
    # resample clips within folds -> pooled weights
    w = np.concatenate([rng.multinomial(len(data[f]['y']), np.full(len(data[f]['y']), 1 / len(data[f]['y'])),
                                        size=B) for f in folds], axis=1)
    ones = np.ones((1, len(y_all)))

    print(f'pooled over folds {list(folds)}: {len(y_all)} clips')
    print('\nstage 0: reproduction (README per-fold UAR in brackets)')
    fold_ok = True
    for f in folds:
        u = float(uar_w(data[f]['pred']['base'], data[f]['y'], np.ones((1, len(data[f]['y']))))[0])
        war = 100.0 * float((data[f]['pred']['base'] == data[f]['y']).mean())
        gap = u - README_UAR[f]
        fold_ok &= gap >= -2.0
        print(f'  fold {f}: UAR {u:6.2f} [{README_UAR[f]:.2f}]  diff {gap:+.2f}   WAR {war:6.2f}')
    tot_uar = float(uar_w(pred_all['base'], y_all, ones)[0])
    tot_war = 100.0 * float((pred_all['base'] == y_all).mean())
    stage0 = tot_uar >= README_TOTAL_UAR - 1.0 and fold_ok
    print(f'  pooled: UAR {tot_uar:.2f} [{README_TOTAL_UAR}]  WAR {tot_war:.2f} [{README_TOTAL_WAR}]')
    print(f'  stage 0 verdict: {"PASS" if stage0 else "FAIL"} (pooled UAR >= {README_TOTAL_UAR - 1.0:.2f} and every fold >= README - 2.0)')

    print('\nstage 1: interventions vs base (pooled; 90% bootstrap CI)')
    base_b = uar_w(pred_all['base'], y_all, w)
    out = {'stage0': {'pooled_uar': tot_uar, 'pooled_war': tot_war, 'pass': bool(stage0)}, 'arms': {}}
    for a in arms:
        u = float(uar_w(pred_all[a], y_all, ones)[0])
        d = u - tot_uar
        db = uar_w(pred_all[a], y_all, w) - base_b
        ch = pred_all[a] != pred_all['base']
        chb = 100.0 * (w * ch).sum(1) / w.sum(1)
        r = {'uar': u, 'war': 100.0 * float((pred_all[a] == y_all).mean()), 'd_uar': d,
             'd_uar_ci': [float(x) for x in np.quantile(db, [ALPHA / 2, 1 - ALPHA / 2])],
             'changed_pct': 100.0 * float(ch.mean()),
             'changed_ci': [float(x) for x in np.quantile(chb, [ALPHA / 2, 1 - ALPHA / 2])]}
        out['arms'][a] = r
        print(f'  {a:>10}: UAR {u:6.2f}  dUAR {d:+6.2f} [{r["d_uar_ci"][0]:+.2f}, {r["d_uar_ci"][1]:+.2f}]'
              f'   changed {r["changed_pct"]:5.2f}% [{r["changed_ci"][0]:.2f}, {r["changed_ci"][1]:.2f}]')
    r = out['arms'].get('indep')
    if r:
        if r['changed_pct'] < 1.0 and -1.0 <= r['d_uar_ci'][0] and r['d_uar_ci'][1] <= 1.0:
            cls = 'predictively inert'
        elif r['changed_pct'] >= 5.0 or (r['d_uar'] <= -1.0 and r['d_uar_ci'][1] < 0):
            cls = 'load-bearing'
        else:
            cls = 'weak'
        out['indep_class'] = cls
        print(f'\npre-registered classification of indep: {cls}'
              + ('' if stage0 else '   (stage 0 FAILED: not a primary result)'))
    path = prefix.rsplit('/', 1)[0] + '/summary.json'
    json.dump(out, open(path, 'w'), indent=1)
    print(f'-> {path}')


if __name__ == '__main__':
    main(sys.argv[1], *(sys.argv[2:3]))
