#!/usr/bin/env python
"""Stage 0b summary (prereg.md): val10 UAR per configuration, paired bootstrap differences, the decision rule.

Usage: s0b_summary.py <results dir holding {base,framewise,motionpool,mixer}_b12.{json,npz} [+ *_b9]>
"""

import json
import os
import sys

import numpy as np

B, ALPHA, NCLS = 2000, 0.10, 7
NAMES = {'base': '(i) baseline', 'framewise': '(ii) framewise', 'motionpool': '(iii) motion pool (A)',
         'mixer': '(iv) mixer (C)'}


def uar_w(pred, y, w):
    rec = [w[:, (y == c) & (pred == c)].sum(1) / np.maximum(w[:, y == c].sum(1), 1) for c in range(NCLS)]
    return 100.0 * np.mean(np.stack(rec, 1), 1)


def ci(x):
    return [float(v) for v in np.quantile(x, [ALPHA / 2, 1 - ALPHA / 2])]


def main(d):
    rng = np.random.default_rng(0)
    runs = {}
    for f in sorted(os.listdir(d)):
        if f.endswith('.json') and f != 'summary.json':
            k = f[:-5]
            runs[k] = (json.load(open(os.path.join(d, f))), np.load(os.path.join(d, k + '.npz')))
    y = runs['base_b12'][1]['labels']
    for k, (_, z) in runs.items():
        assert np.array_equal(z['labels'], y), k
    w = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=B)
    ones = np.ones((1, len(y)))
    pred = {k: z['prob'].argmax(1) for k, (_, z) in runs.items()}
    U = {k: float(uar_w(p, y, ones)[0]) for k, p in pred.items()}
    Ub = {k: uar_w(p, y, w) for k, p in pred.items()}
    out = {'n_eval': int(len(y)), 'runs': {}, 'diffs': {}}
    print(f'val10: {len(y)} clips\n\nrun                      last UAR   best UAR (ep)   WAR    spread f / CLS   adapter params')
    for k, (j, _) in runs.items():
        out['runs'][k] = {'uar_last': U[k], 'war_last': j['last']['war'], 'best_uar': j['best_epoch']['uar'],
                          'best_epoch': j['best_epoch']['epoch'], 'spread_f': j['spread_f'],
                          'spread_cls': j['spread_cls'], 'adapter_params': j['adapter_params'],
                          'gate_abs_mean': j['last']['gate_abs_mean']}
        print(f"{k:<22} {U[k]:8.2f}   {j['best_epoch']['uar']:8.2f} ({j['best_epoch']['epoch']:2d})  "
              f"{j['last']['war']:6.2f}   {j['spread_f']:.4f} / {j['spread_cls']:.4f}   {j['adapter_params']}")

    def diff(a, b):
        if a in U and b in U:
            v = {'value': U[a] - U[b], 'ci': ci(Ub[a] - Ub[b]),
                 'changed_pct': 100.0 * float((pred[a] != pred[b]).mean())}
            out['diffs'][f'{a} - {b}'] = v
            print(f"  {a:<16} - {b:<16} {v['value']:+6.2f} [{v['ci'][0]:+6.2f}, {v['ci'][1]:+6.2f}]   "
                  f"changed {v['changed_pct']:5.2f}%")
            return v
    print('\npaired differences (90% bootstrap CI)')
    main_d = diff('motionpool_b12', 'framewise_b12')
    diff('framewise_b12', 'base_b12')
    diff('motionpool_b12', 'base_b12')
    diff('mixer_b12', 'framewise_b12')
    diff('mixer_b12', 'base_b12')
    diff('motionpool_b9', 'framewise_b9')
    diff('framewise_b9', 'base_b12')
    go = main_d['value'] >= 1.0 and main_d['ci'][0] > 0
    out['decision'] = {'rule': 'UAR(iii) - UAR(ii) >= +1.0 and 90% CI lower bound > 0',
                       'iii_minus_ii': main_d, 'verdict': 'CONTINUE to stage 1' if go else 'STOP'}
    j = runs['motionpool_b12'][0]
    out['mechanism_check'] = {'suppress_uar': j.get('suppress_uar'), 'uar': U['motionpool_b12'],
                              'suppress_d_uar': j.get('suppress_uar', float('nan')) - U['motionpool_b12'],
                              'suppress_changed_pct': j.get('suppress_changed_pct'),
                              'weight_entropy': j.get('motion_weight_entropy'),
                              'weight_entropy_uniform': j.get('motion_weight_entropy_uniform')}
    print(f"\ndecision: {out['decision']['verdict']}  ({out['decision']['rule']})")
    print(f"mechanism check (iii, temporal contrasts suppressed at inference): {out['mechanism_check']}")
    json.dump(out, open(os.path.join(d, 'summary.json'), 'w'), indent=1)
    print(f"-> {os.path.join(d, 'summary.json')}")


if __name__ == '__main__':
    main(sys.argv[1])
