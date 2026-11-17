#!/usr/bin/env python
"""Head attenuation control (prereg.md, 2026-11-13): sanity checks, endpoints per arm and the registered reading.

Per setting (results/headctl/<name>/fold{f}.json, arms base, indep_10_12 and the ctl_* arms): the trace means over clips
(in_mean, in_frame, out, logit; weighted by clips per fold), changed % pooled, dUAR mean over folds.
Reading: R = out(ctl_rand, mean of 3 seeds) / out(ctl_frameonly) on CREMA-D; R <= 2 generic attenuation, R >= 3 the routing's
change lies in directions the head ignores, otherwise inconclusive.

    python headctl_summary.py results/headctl      -> results/headctl/summary.{txt,json}
"""

import json
import os
import statistics as st
import sys

D = sys.argv[1] if len(sys.argv) > 1 else 'results/headctl'
R0 = os.path.dirname(os.path.abspath(D))
EARLIER = {'CREMAD_router': 'blocksub/CREMAD_mma/fold', 'DFEW_A0_s1': 'blocksub/DFEW_mma/fold'}
ARMS = ('indep_10_12', 'ctl_meanonly', 'ctl_frameonly', 'ctl_rand0', 'ctl_rand1', 'ctl_rand2', 'ctl_perm')
lines = []


def out(s=''):
    lines.append(s)
    print(s)


def load(name):
    p = os.path.join(D, name)
    if not os.path.exists(os.path.join(p, 'fold5.json')):
        return None
    return [json.load(open(os.path.join(p, f'fold{f}.json'))) for f in range(1, 6)]


def main():
    res = {}
    for name in ('CREMAD_router', 'DFEW_A0_s1'):
        js = load(name)
        if js is None:
            out(f'{name}: not complete')
            continue
        n = [j['n_clips'] for j in js]
        N = sum(n)
        r = {}
        for arm in ARMS:
            t = {k: sum(j['trace'][arm][k] * m for j, m in zip(js, n)) / N for k in ('in_mean', 'in_frame', 'out', 'logit')}
            t['changed'] = sum(j['arms'][arm]['changed_pct'] * m for j, m in zip(js, n)) / N
            t['dUAR'] = st.mean(j['arms'][arm]['uar'] - j['arms']['base']['uar'] for j in js)
            r[arm] = t
        # sanity
        old = os.path.join(R0, EARLIER[name])
        dev = max(max(abs(json.load(open(f'{old}{f}.json'))['arms'][a]['uar'] - js[f - 1]['arms'][a]['uar'])
                      for a in ('base', 'indep_10_12')) for f in range(1, 6))
        fr = r['indep_10_12']['in_frame']
        s1 = all(abs(r[a]['in_frame'] - fr) <= 0.03 * fr for a in ('ctl_frameonly', 'ctl_rand0', 'ctl_rand1', 'ctl_rand2',
                                                                   'ctl_perm'))
        s1m = all(r[a]['in_mean'] <= 0.25 * r['indep_10_12']['in_mean'] for a in ('ctl_frameonly', 'ctl_rand0', 'ctl_rand1', 'ctl_rand2', 'ctl_perm'))
        out(f'{name} ({N} clips)')
        out(f'  sanity 1: in_frame within 3% {s1}, in_mean <= 0.25 x indep {s1m} (bf16 amendment); sanity 2: max |dUAR| vs earlier block-subset {dev:.4f}')
        out('  arm             in_mean   in_frame  out       logit     changed%  dUAR')
        for arm in ARMS:
            t = r[arm]
            out(f"  {arm:<15} {t['in_mean']:.5f}   {t['in_frame']:.4f}    {t['out']:.5f}   {t['logit']:.5f}   "
                f"{t['changed']:.2f}     {t['dUAR']:+.2f}")
        rand = st.mean(r[f'ctl_rand{s}']['out'] for s in range(3))
        R = rand / r['ctl_frameonly']['out']
        Rl = st.mean(r[f'ctl_rand{s}']['logit'] for s in range(3)) / r['ctl_frameonly']['logit']
        Rp = r['ctl_perm']['out'] / r['ctl_frameonly']['out']
        att = {a: r[a]['out'] / r[a]['in_frame'] for a in ('ctl_frameonly', 'ctl_perm', 'ctl_rand0')}
        reading = ('generic attenuation: the head attenuates frame-specific changes of this size whatever their direction'
                   if R <= 2 else ('the routing change lies in directions the head ignores' if R >= 3 else 'inconclusive'))
        out(f'  R = out(rand) / out(frameonly) = {R:.2f} (logits {Rl:.2f}); perm / frameonly = {Rp:.2f}')
        out('  out / in_frame: ' + ', '.join(f'{k} {v:.3f}' for k, v in att.items()))
        if name == 'CREMAD_router':
            out(f'  READING (registered, CREMA-D): {reading}')
        res[name] = {'arms': r, 'R': R, 'R_logit': Rl, 'R_perm': Rp, 'attenuation': att, 'reading': reading,
                     'sanity': {'in_frame_matched': s1, 'in_mean_zero': s1m, 'max_dev_earlier': dev}}
    json.dump(res, open(os.path.join(D, 'summary.json'), 'w'), indent=1)
    open(os.path.join(D, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
