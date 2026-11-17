#!/usr/bin/env python
"""Exact product reference, WAR and a continuous endpoint (prereg.md, 2026-11-13): sanity checks, endpoints per
checkpoint, and the registered readings reapplied with the exact product.

Per checkpoint and fold (results/exactprod/<name>/fold{f}.{json,npz}, arms base / indep_all / exact_all):
  Delta_route = U_real - U_indep (solver marginals, the registered primary), Delta_exact = U_real - U_exact,
  the same in WAR, and Delta_logp = mean over clips of log p_y(real) - log p_y(product), in nats.
Between-checkpoint endpoints are paired by fold: mean, sd, 90% paired-fold t interval (n = 5, t_0.95,4 = 2.132).

    python exactprod_summary.py results/exactprod      -> results/exactprod/summary.{txt,json}
"""

import json
import os
import statistics as st
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else 'results/exactprod'
R = os.path.dirname(os.path.abspath(D))
T95_4 = 2.1318
FOLDS = range(1, 6)
OLD = {   # earlier evaluation of the same checkpoint (sanity 1)
    'A4_s1_pooled': 'e345/A4/s1_native', 'DFEW_A0_s1': 'e345/eps/A0_eps0.05', 'DFEW_A0_s2': 'e345/seeds/A0_s2',
    'DFEW_A0_s3': 'e345/seeds/A0_s3', 'DFEW_A1': 'e345/A1', 'DFEW_A2': 'e6/A2', 'DFEW_A3': 'e6/A3',
    'DFEW_trained_eps0.02': 'e345/eps/trained_eps0.02', 'CREMAD_router': 'blocksub/CREMAD_mma/fold',
    'DFEW_abv': 'blocksub/DFEW_abv/fold'}
EPS = {'eps0.2': 'e345/eps/A0_eps0.2', 'eps0.1': 'e345/eps/A0_eps0.1', 'eps0.02': 'e345/eps/A0_eps0.02',
       'eps0.01_it100': 'e345/eps/A0_eps0.01_it100'}            # converged settings only, as in E4
lines = []


def out(s=''):
    lines.append(s)
    print(s)


def ci(v):
    m, s = st.mean(v), st.stdev(v)
    h = T95_4 * s / len(v) ** 0.5
    return {'mean': m, 'sd': s, 'ci90': [m - h, m + h], 'folds': v}


def fmt(c, p=2):
    return f"{c['mean']:+.{p}f} ± {c['sd']:.{p}f} [{c['ci90'][0]:+.{p}f}, {c['ci90'][1]:+.{p}f}]"


def resolved_pos(c, ref):
    return c['ci90'][0] > 0 and c['mean'] > ref


def logp_true(P, y):
    return np.log(np.clip(P[np.arange(len(y)), y], 1e-300, None))


def fold_path(base, f):
    """Old results live either as <dir>/fold{f} or as <prefix>{f} (blocksub summaries use a prefix)."""
    return base + str(f) if base.endswith('fold') else os.path.join(base, f'fold{f}')


def load(name):
    p = os.path.join(D, name)
    if not os.path.exists(os.path.join(p, 'fold5.json')):
        return None
    r = {k: [] for k in ('U_real', 'U_indep', 'U_exact', 'W_real', 'W_indep', 'W_exact', 'ch_indep', 'ch_exact',
                         'lp_indep', 'lp_exact', 'agree', 'n')}
    probs = []
    for f in FOLDS:
        j = json.load(open(os.path.join(p, f'fold{f}.json')))
        z = np.load(os.path.join(p, f'fold{f}.npz'))
        A = j['arms']
        r['U_real'].append(A['base']['uar'])
        r['U_indep'].append(A['indep_all']['uar'])
        r['U_exact'].append(A['exact_all']['uar'])
        r['W_real'].append(A['base']['war'])
        r['W_indep'].append(A['indep_all']['war'])
        r['W_exact'].append(A['exact_all']['war'])
        r['ch_indep'].append(A['indep_all']['changed_pct'])
        r['ch_exact'].append(A['exact_all']['changed_pct'])
        y = z['labels']
        l0 = logp_true(z['prob_base'], y)
        r['lp_indep'].append(float((l0 - logp_true(z['prob_indep_all'], y)).mean()))
        r['lp_exact'].append(float((l0 - logp_true(z['prob_exact_all'], y)).mean()))
        r['agree'].append(100.0 * float((z['prob_exact_all'].argmax(1) == z['prob_indep_all'].argmax(1)).mean()))
        r['n'].append(j['n_clips'])
        probs.append((y, z['prob_exact_all']))
    r['D_route'] = [a - b for a, b in zip(r['U_real'], r['U_indep'])]
    r['D_exact'] = [a - b for a, b in zip(r['U_real'], r['U_exact'])]
    r['DW_route'] = [a - b for a, b in zip(r['W_real'], r['W_indep'])]
    r['DW_exact'] = [a - b for a, b in zip(r['W_real'], r['W_exact'])]
    n = r['n']
    for k in ('ch_indep', 'ch_exact', 'agree'):
        r[k + '_pooled'] = sum(c * m for c, m in zip(r[k], n)) / sum(n)
    r['_probs_exact'] = probs
    return r


def sanity1(name, r):
    """base and indep_all reproduce the earlier evaluation within 0.01 UAR."""
    old = os.path.join(R, OLD[name])
    dev = []
    for f in FOLDS:
        p = fold_path(old, f) + '.json'
        if not os.path.exists(p):
            return None
        A = json.load(open(p))['arms']
        dev.append(max(abs(A['base']['uar'] - r['U_real'][f - 1]), abs(A['indep_all']['uar'] - r['U_indep'][f - 1])))
    return max(dev)


def eps_arm(path, a0):
    """eps setting of A0 s1: U_real and log p from its stored evaluation; the exact product is A0 s1's (eps-free)."""
    U, lp_exact, lp_indep, Ui = [], [], [], []
    for f in FOLDS:
        j = json.load(open(os.path.join(R, path, f'fold{f}.json')))
        z = np.load(os.path.join(R, path, f'fold{f}.npz'))
        y0, P0 = a0['_probs_exact'][f - 1]
        y = z['labels']
        assert len(y) == len(y0) and (y == y0).all(), (path, f)
        U.append(j['arms']['base']['uar'])
        Ui.append(j['arms']['indep_all']['uar'])
        l = logp_true(z['prob_base'], y)
        lp_exact.append(float((l - logp_true(P0, y)).mean()))
        lp_indep.append(float((l - logp_true(z['prob_indep_all'], y)).mean()))
    return {'U_real': U, 'D_exact': [u - e for u, e in zip(U, a0['U_exact'])],
            'D_route': [u - i for u, i in zip(U, Ui)], 'lp_exact': lp_exact, 'lp_indep': lp_indep}


def diff(x, y, k):
    return ci([p - q for p, q in zip(x[k], y[k])])


def main():
    names = list(OLD)
    S = {k: load(k) for k in names}
    res = {'sanity': {}, 'checkpoints': {}, 'readings': {}}
    out('SANITY')
    for k, r in S.items():
        if r is None:
            out(f'  {k}: not complete')
            continue
        d = sanity1(k, r)
        ok = d is not None and d <= 0.01
        res['sanity'][k] = {'max_dev_vs_earlier': d, 'ok': ok}
        out(f"  1. {k:<22} base/indep vs earlier evaluation: max |dUAR| = {d if d is None else round(d, 4)}  "
            f"{'OK' if ok else 'CHECK'}")
    a4 = S.get('A4_s1_pooled')
    if a4:
        ok2 = all(c == 0.0 for c in a4['ch_exact'])
        res['sanity']['pooled_exact_equals_base'] = ok2
        out(f"  2. A4 s1 pooled: exact_all changed % per fold {a4['ch_exact']}  {'OK' if ok2 else 'FAIL'}")
    out('  3. agreement exact vs indep (pooled %): ' + ', '.join(
        f"{k} {r['agree_pooled']:.2f}" for k, r in S.items() if r and k != 'A4_s1_pooled'))

    out('\nCHECKPOINTS (mean over folds; changed % pooled; Delta_logp in nats)')
    out('name                   U_real  U_indep U_exact  D_route        D_exact        ch_ind ch_ex  '
        'DW_route DW_exact  dlogp_ind dlogp_ex')
    for k, r in S.items():
        if not r or k == 'A4_s1_pooled':
            continue
        m = st.mean
        out(f"{k:<22} {m(r['U_real']):6.2f}  {m(r['U_indep']):6.2f}  {m(r['U_exact']):6.2f}  "
            f"{m(r['D_route']):+.2f}±{st.stdev(r['D_route']):.2f}     {m(r['D_exact']):+.2f}±{st.stdev(r['D_exact']):.2f}"
            f"     {r['ch_indep_pooled']:.2f}  {r['ch_exact_pooled']:.2f}   {m(r['DW_route']):+.2f}    {m(r['DW_exact']):+.2f}"
            f"     {m(r['lp_indep']):+.4f}  {m(r['lp_exact']):+.4f}")
        res['checkpoints'][k] = {kk: v for kk, v in r.items() if not kk.startswith('_')}

    seeds = [S[k] for k in ('DFEW_A0_s1', 'DFEW_A0_s2', 'DFEW_A0_s3') if S.get(k)]
    if len(seeds) < 3 or not all(S.get(k) for k in ('DFEW_A1', 'DFEW_A2', 'DFEW_A3')):
        out('\nREADINGS: DFEW checkpoints incomplete')
    else:
        a0, a1, a2, a3 = S['DFEW_A0_s1'], S['DFEW_A1'], S['DFEW_A2'], S['DFEW_A3']
        rng = {key: max(st.mean(s[key]) for s in seeds) - min(st.mean(s[key]) for s in seeds)
               for key in ('D_route', 'D_exact')}
        out(f"\nREADINGS (registered rules; seed range of Delta over A0 s1-s3: route {rng['D_route']:.2f}, "
            f"exact {rng['D_exact']:.2f})")
        E = {k: eps_arm(p, a0) for k, p in EPS.items()}
        rd = {}
        for key, lab in (('D_route', 'solver-marginal product (registered primary)'), ('D_exact', 'exact product')):
            I1, I2, I3 = diff(a1, a0, key), diff(a2, a0, key), diff(a3, a0, key)
            C = diff(a2, a3, key)
            dd = {e: ci([x - y for x, y in zip(E[e][key], a0[key])]) for e in E}
            mx = max(v['mean'] for v in dd.values())
            r = {'I_A1': I1, 'I_A2': I2, 'I_A3': I3, 'C': C, 'DD': dd, 'max_DD': mx,
                 'A1_moves': resolved_pos(I1, rng[key]), 'A2_moves': resolved_pos(I2, rng[key]),
                 'A3_moves': resolved_pos(I3, rng[key]), 'A2_special': resolved_pos(C, rng[key]),
                 'eps_reproduces_A1': I1['mean'] > 0 and mx >= I1['mean'],
                 'eps_reproduces_A2': I2['mean'] > 0 and mx >= I2['mean']}
            rd[key] = r
            out(f'  [{lab}]')
            out(f"    I_A1 = {fmt(I1)}  -> {'MOVES' if r['A1_moves'] else 'does not move'}")
            out(f"    I_A2 = {fmt(I2)}  -> {'MOVES' if r['A2_moves'] else 'does not move'}")
            out(f"    I_A3 = {fmt(I3)}  -> {'MOVES' if r['A3_moves'] else 'does not move'}")
            out(f"    C = Delta_A2 - Delta_A3 = {fmt(C)}  -> {'A2 exceeds A3' if r['A2_special'] else 'A2 does not exceed A3'}")
            out('    DD(eps) vs eps 0.05: ' + ', '.join(f"{e} {fmt(v)}" for e, v in dd.items()))
            out(f"    max converged DD {mx:+.2f}: reproduces A1 {r['eps_reproduces_A1']}, A2 {r['eps_reproduces_A2']}")
        same = {k: rd['D_route'][k] == rd['D_exact'][k] for k in
                ('A1_moves', 'A2_moves', 'A3_moves', 'A2_special', 'eps_reproduces_A1', 'eps_reproduces_A2')}
        out('  ROBUSTNESS: ' + ', '.join(f"{k} {'unchanged' if v else 'CHANGED'}" for k, v in same.items()))
        res['readings'] = {'seed_range': rng, 'by_reference': rd, 'unchanged': same}
        # WAR and continuous endpoint (secondary, descriptive)
        out('\nSECONDARY (paired by fold against A0 s1): I in WAR and in Delta_logp (nats), G in WAR')
        sec = {}
        for k, a in (('A1', a1), ('A2', a2), ('A3', a3), ('trained_eps0.02', S.get('DFEW_trained_eps0.02'))):
            if not a:
                continue
            s = {'I_war_route': diff(a, a0, 'DW_route'), 'I_war_exact': diff(a, a0, 'DW_exact'),
                 'I_logp_route': diff(a, a0, 'lp_indep'), 'I_logp_exact': diff(a, a0, 'lp_exact'),
                 'G_war': diff(a, a0, 'W_real')}
            sec[k] = s
            out(f"  {k:<16} I_WAR route {fmt(s['I_war_route'])}  exact {fmt(s['I_war_exact'])}")
            out(f"  {'':<16} I_logp route {fmt(s['I_logp_route'], 4)}  exact {fmt(s['I_logp_exact'], 4)}")
            out(f"  {'':<16} G_WAR {fmt(s['G_war'])}")
        for e, v in E.items():
            out(f"  {e:<16} Delta_logp exact {st.mean(v['lp_exact']):+.4f}, route {st.mean(v['lp_indep']):+.4f} "
                f"(A0 eps 0.05: exact {st.mean(a0['lp_exact']):+.4f}, route {st.mean(a0['lp_indep']):+.4f})")
        res['secondary'] = sec
        res['eps'] = E
    # contrast class (CREMA-D router, and the DFEW hosts for description)
    out('\nCONTRAST CLASS (inert: changed < 1% and 90% interval of Delta within ±1; load-bearing: changed >= 5% or '
        'Delta >= 1 with interval excluding 0)')
    cls = {}
    for k in ('CREMAD_router', 'DFEW_A0_s1', 'DFEW_abv'):
        r = S.get(k)
        if not r:
            continue
        for key, ch in (('D_route', 'ch_indep_pooled'), ('D_exact', 'ch_exact_pooled')):
            c = ci(r[key])
            inert = r[ch] < 1 and -1 <= c['ci90'][0] and c['ci90'][1] <= 1
            lb = r[ch] >= 5 or (c['mean'] >= 1 and c['ci90'][0] > 0)
            lab = 'inert' if inert else ('load-bearing' if lb else 'intermediate')
            cls[f'{k}:{key}'] = lab
            out(f"  {k:<14} {key:<8} {fmt(c)}  changed {r[ch]:.2f}%  -> {lab}")
    res['contrast_class'] = cls
    json.dump(res, open(os.path.join(D, 'summary.json'), 'w'), indent=1, default=float)
    open(os.path.join(D, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
