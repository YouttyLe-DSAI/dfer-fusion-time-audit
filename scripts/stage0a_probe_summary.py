#!/usr/bin/env python
"""Summarise stage 0a-bis probes: mean UAR over folds and the pre-registered paired comparisons.

Paired bootstrap over val10 clips within each fold (same resample for both arms), 2000 resamples,
90% percentile CI of the mean over folds. A difference counts if >= 2.0 points and its CI excludes 0.
Caveat: the DFEW val10 splits overlap across folds, so the folds are not independent.

    python stage0a_probe_summary.py results/stage0a/probe_MMA_DFEW_scr.json
"""

import json
import sys

import numpy as np

MARGIN, B, ALPHA, PRIMARY = 2.0, 2000, 0.10, 9


def uar_boot(pred, y, n_cls, w):
    """UAR [B] of predictions under resample weights w [B, n]."""
    rec = []
    for c in range(n_cls):
        m = y == c
        if not m.any():
            continue
        tot = w[:, m].sum(1)
        hit = w[:, m & (pred == c)].sum(1)
        rec.append(np.where(tot > 0, hit / np.maximum(tot, 1), np.nan))
    return 100.0 * np.nanmean(np.stack(rec, 1), 1)


def uar(pred, y, n_cls):
    return float(uar_boot(pred, y, n_cls, np.ones((1, len(y))))[0])


def main(path):
    d = json.load(open(path))
    folds = sorted(d['per_fold'], key=int)
    rng = np.random.default_rng(0)
    data = {}
    for f in folds:
        pf = d['per_fold'][f]
        y = np.asarray(pf['val_labels'])
        n = len(y)
        data[f] = (y, len(pf['classes']), rng.multinomial(n, np.full(n, 1.0 / n), size=B),
                   {k: np.asarray(v['pred']) for k, v in pf['probes'].items()})

    def diff(a, b):
        """Observed and bootstrap mean-over-folds UAR(a) - UAR(b); b may be 'chance'."""
        obs, boots = [], []
        for f in folds:
            y, n_cls, w, preds = data[f]
            ua = uar(preds[a], y, n_cls)
            ba = uar_boot(preds[a], y, n_cls, w)
            if b == 'chance':
                ub, bb = 100.0 / n_cls, np.full(B, 100.0 / n_cls)
            else:
                ub, bb = uar(preds[b], y, n_cls), uar_boot(preds[b], y, n_cls, w)
            obs.append(ua - ub)
            boots.append(ba - bb)
        boots = np.mean(boots, 0)
        lo, hi = np.quantile(boots, [ALPHA / 2, 1 - ALPHA / 2])
        o = float(np.mean(obs))
        return {'diff': o, 'ci': [float(lo), float(hi)], 'per_fold': obs,
                'counts': bool(o >= MARGIN and (lo > 0 or hi < 0))}

    blocks, levels, feats = d['blocks'], d['levels'], d['feats']
    print(f'{path}\nfolds {folds}, PCA {d["pca"]}, limit {d["limit"]}')
    print('\nmean val10 UAR over folds (chance %.2f)' % (100.0 / data[folds[0]][1]))
    print('block level ' + ''.join(f'{f:>8}' for f in feats))
    means = {}
    for b in blocks:
        for lv in levels:
            row = []
            for f in feats:
                k = f'{b}/{lv}/{f}'
                means[k] = float(np.mean([d['per_fold'][x]['probes'][k]['uar'] for x in folds]))
                row.append(means[k])
            print(f'{b:>5} {lv:>5} ' + ''.join(f'{v:8.2f}' for v in row))

    comps = []
    for b in blocks:
        comps += [(b, 'H3a', f'{b}/r4/D', 'chance')]
        comps += [(b, 'H3b', f'{b}/{lv}/SD', f'{b}/{lv}/S') for lv in levels]
        comps += [(b, 'H2', f'{b}/r4/D', f'{b}/cls/D'), (b, 'H2', f'{b}/r4/D', f'{b}/mean/D'),
                  (b, 'H2', f'{b}/patch/D', f'{b}/cls/D'), (b, 'H2', f'{b}/patch/D', f'{b}/r4/D')]
    res = []
    print(f'\npaired comparisons (counts = diff >= {MARGIN} and 90% CI excludes 0)')
    for b, h, a, c in comps:
        r = diff(a, c)
        res.append({'block': b, 'hyp': h, 'a': a, 'b': c, **r})
        print(f'{b:>5} {h:>4}  {a:>14} - {c:<14} {r["diff"]:+6.2f}  [{r["ci"][0]:+6.2f}, {r["ci"][1]:+6.2f}]'
              f'  {"COUNTS" if r["counts"] else ""}')

    def get(h, a, c):
        return next(r for r in res if r['block'] == PRIMARY and r['hyp'] == h and r['a'] == a and r['b'] == c)

    verdict = {
        'H3a': get('H3a', f'{PRIMARY}/r4/D', 'chance')['counts'],
        'H3b': get('H3b', f'{PRIMARY}/r4/SD', f'{PRIMARY}/r4/S')['counts'],
        'H2': (get('H2', f'{PRIMARY}/r4/D', f'{PRIMARY}/cls/D')['counts']
               and get('H2', f'{PRIMARY}/r4/D', f'{PRIMARY}/mean/D')['counts']),
    }
    print(f'\npre-registered verdict at block {PRIMARY}: ' +
          ', '.join(f'{k} {"SUPPORTED" if v else "not supported"}' for k, v in verdict.items()))
    out = path.replace('.json', '_summary.json')
    json.dump({'source': path, 'means': means, 'comparisons': res, 'verdict': verdict}, open(out, 'w'), indent=1)
    print(f'-> {out}')


if __name__ == '__main__':
    main(sys.argv[1])
