#!/usr/bin/env python
"""P1-nulls summary: pooled endpoints, P1 classes and the pre-registered readings N1-N3 (prereg.md).

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


def ci(x):
    return [float(v) for v in np.quantile(x, [ALPHA / 2, 1 - ALPHA / 2])]


def p1_class(r):
    if r['changed_pct'] < 1.0 and -1.0 <= r['d_uar_ci'][0] and r['d_uar_ci'][1] <= 1.0:
        return 'inert'
    if r['changed_pct'] >= 5.0 or (r['d_uar'] <= -1.0 and r['d_uar_ci'][1] < 0):
        return 'load-bearing'
    return 'weak'


def n1(rp, ind):
    if rp['class'] == 'inert' and ind['class'] == 'load-bearing':
        return 'distribution reading'
    if rp['class'] == 'load-bearing' and rp['d_uar'] <= ind['d_uar'] + 1.0:
        return 'alignment reading'
    return 'mixed'


def main(prefix):
    rng = np.random.default_rng(0)
    data, meta = {}, {}
    for f in FOLDS:
        z = np.load(f'{prefix}{f}.npz')
        arms = [k[5:] for k in z.files if k.startswith('prob_')]
        data[f] = {'y': z['labels'], 'pred': {a: z[f'prob_{a}'].argmax(1) for a in arms}}
        meta[f] = json.load(open(f'{prefix}{f}.json'))
    arms = list(data[FOLDS[0]]['pred'])
    y = np.concatenate([data[f]['y'] for f in FOLDS])
    pred = {a: np.concatenate([data[f]['pred'][a] for f in FOLDS]) for a in arms}
    w = np.concatenate([rng.multinomial(len(data[f]['y']), np.full(len(data[f]['y']), 1 / len(data[f]['y'])),
                                        size=B) for f in FOLDS], axis=1)
    ones = np.ones((1, len(y)))
    base_u = float(uar_w(pred['base'], y, ones)[0])
    base_b = uar_w(pred['base'], y, w)
    out = {'n_clips': int(len(y)), 'base_uar': base_u, 'arms': {}, 'per_fold_d_uar': {}}
    dub = {}
    print(f'pooled over folds {list(FOLDS)}: {len(y)} clips; base UAR {base_u:.2f}')
    print('\narm            UAR    dUAR [90% CI]          changed % [90% CI]     class')
    for a in arms:
        u = float(uar_w(pred[a], y, ones)[0])
        dub[a] = uar_w(pred[a], y, w) - base_b
        ch = pred[a] != pred['base']
        chb = 100.0 * (w * ch).sum(1) / w.sum(1)
        r = {'uar': u, 'd_uar': u - base_u, 'd_uar_ci': ci(dub[a]), 'changed_pct': 100.0 * float(ch.mean()),
             'changed_ci': ci(chb)}
        r['class'] = p1_class(r)
        out['arms'][a] = r
        pf = []
        for f in FOLDS:
            yf, pf_a, pf_b = data[f]['y'], data[f]['pred'][a], data[f]['pred']['base']
            one = np.ones((1, len(yf)))
            pf.append(float(uar_w(pf_a, yf, one)[0] - uar_w(pf_b, yf, one)[0]))
        out['per_fold_d_uar'][a] = pf
        print(f'{a:<12} {u:6.2f}  {r["d_uar"]:+6.2f} [{r["d_uar_ci"][0]:+6.2f}, {r["d_uar_ci"][1]:+6.2f}]'
              f'   {r["changed_pct"]:5.2f} [{r["changed_ci"][0]:5.2f}, {r["changed_ci"][1]:5.2f}]   {r["class"]}')

    A = out['arms']
    print('\nper-fold dUAR')
    for a in arms:
        print(f'  {a:<12} ' + '  '.join(f'{v:+6.2f}' for v in out['per_fold_d_uar'][a]))

    readings = {'N1': n1(A['rowperm'], A['indep']),
                'N1_seeds': {s: n1(A[s], A['indep']) for s in ('rowperm', 'rowperm_s1', 'rowperm_s2')},
                'N1_time_only': n1(A['rowperm_t'], A['indep'])}
    readings['N1_stable'] = len(set(readings['N1_seeds'].values())) == 1
    if A['emap']['class'] == 'inert' and A['indep']['class'] == 'load-bearing':
        readings['N2'] = 'mechanism-function dissociation'
    elif A['emap']['class'] == 'load-bearing':
        readings['N2'] = 'input-level AV interactions used'
    else:
        readings['N2'] = 'weak'
    readings['N3_papa_class'] = A['papa']['class']
    inter = dub['zero'] - dub['indep'] - dub['devonly']
    readings['factorial_interaction'] = {'value': A['zero']['d_uar'] - A['indep']['d_uar'] - A['devonly']['d_uar'],
                                         'ci': ci(inter)}
    readings['shrink_dose'] = {al: A[k]['d_uar'] for al, k in ((0.0, 'indep'), (0.25, 'shrink25'),
                                                               (0.5, 'shrink50'), (0.75, 'shrink75'), (1.0, 'base'))}
    readings['ordering_by_d_uar'] = sorted(('indep', 'uniform', 'rowperm', 'papa', 'zero', 'devonly'),
                                           key=lambda k: A[k]['d_uar'])
    readings['rowperm_minus_indep'] = {'value': A['rowperm']['d_uar'] - A['indep']['d_uar'],
                                       'ci': ci(dub['rowperm'] - dub['indep'])}
    readings['emap_minus_base'] = A['emap']['d_uar']
    out['readings'] = readings
    print('\npre-registered readings')
    for k, v in readings.items():
        print(f'  {k}: {v}')

    # sanity + trace, from the per-fold JSONs
    out['sanity'] = {f: {'check_max_diff': meta[f]['check_max_diff'], 'agree_with_p1': meta[f].get('agree_with_p1'),
                         'n_views': meta[f]['n_views'], 'emap_valid_partners': meta[f]['emap_mean_valid_partners'],
                         'papa_train_views': meta[f]['papa_train_views']} for f in FOLDS}
    out['f3_max_rel_mean_diff'] = {arm: max(max(meta[f]['f3_max_rel_mean_diff'][arm].values()) for f in FOLDS)
                                   for arm in meta[FOLDS[0]]['f3_max_rel_mean_diff']}
    tr = {}
    for arm in meta[FOLDS[0]]['trace']:
        tr[arm] = {}
        for s in ('video', 'audio'):
            tr[arm][s] = {k: [float(np.mean([meta[f]['trace'][arm][s][k][j] for f in FOLDS])) for j in (0, 1)]
                          for k in meta[FOLDS[0]]['trace'][arm][s]}
    out['trace'] = tr
    print('\nsanity')
    for f, s in out['sanity'].items():
        print(f'  fold {f}: {s}')
    print(f'  F3 max relative difference of the token mean (same input): {out["f3_max_rel_mean_diff"]}')
    print('\nF3 trace: relative divergence (token mean, tokens) vs base, mean over folds')
    for arm, t in tr.items():
        for s, d in t.items():
            print(f'  {arm:<9} {s:<5} ' + '  '.join(f'{k}=({v[0]:.3f},{v[1]:.3f})' for k, v in d.items()))
    path = prefix.rsplit('/', 1)[0] + '/summary.json'
    json.dump(out, open(path, 'w'), indent=1)
    print(f'-> {path}')


if __name__ == '__main__':
    main(sys.argv[1])
