#!/usr/bin/env python
"""E6 summary (prereg.md "E6"): endpoints of A2 / A3 against A0, the prespecified A2 - A3 contrast, the registered
readings, and the manipulation check (cost-space profile and branch gates per block).

Per arm and fold: U_real = base UAR, U_product = indep_all UAR (blocksub_ot.py fold JSONs), Delta_route = U_real - U_product.
Between-run endpoints are paired by fold: mean, sd, 90% paired-fold t interval (n = 5, t_0.95,4 = 2.132).

    python e6_summary.py results/e6        -> results/e6/summary.{json,txt}   (reads results/e345 and results/e2)
"""

import json
import os
import statistics as st
import sys

D = sys.argv[1] if len(sys.argv) > 1 else 'results/e6'
R = os.path.dirname(os.path.abspath(D))
T95_4 = 2.1318
MAX_DD_CONVERGED = None        # read from results/e345/summary.json
lines = []


def out(s=''):
    lines.append(s)
    print(s)


def arm(path):
    if not os.path.exists(os.path.join(path, 'fold5.json')):
        return None
    js = [json.load(open(os.path.join(path, f'fold{f}.json'))) for f in range(1, 6)]
    ur = [j['arms']['base']['uar'] for j in js]
    up = [j['arms']['indep_all']['uar'] for j in js]
    ch = [j['arms']['indep_all']['changed_pct'] for j in js]
    n = [j['n_clips'] for j in js]
    res = {'U_real': ur, 'U_product': up, 'Delta': [a - b for a, b in zip(ur, up)], 'changed': ch, 'n': n,
           'changed_pooled': sum(c * k for c, k in zip(ch, n)) / sum(n)}
    for sub in ('indep_1_9', 'indep_10_12'):
        if sub in js[0]['arms']:
            c = [j['arms'][sub]['changed_pct'] for j in js]
            res[f'changed_pooled_{sub}'] = sum(x * k for x, k in zip(c, n)) / sum(n)
    if 'cost_gate_tanh' in js[0]:
        res['cost_gate_tanh'] = [[j['cost_gate_tanh'][b] for j in js] for b in range(len(js[0]['cost_gate_tanh']))]
    return res


def ci(v):
    m, s = st.mean(v), st.stdev(v)
    h = T95_4 * s / len(v) ** 0.5
    return {'mean': m, 'sd': s, 'ci90': [m - h, m + h], 'folds': v}


def fmt(c):
    return f"{c['mean']:+.2f} ± {c['sd']:.2f} [{c['ci90'][0]:+.2f}, {c['ci90'][1]:+.2f}]"


def diff(x, y, key):
    return ci([p - q for p, q in zip(x[key], y[key])])


def covers0(c):
    return c['ci90'][0] <= 0 <= c['ci90'][1]


def main():
    A = {'A0_s1': arm(os.path.join(R, 'e345/eps/A0_eps0.05')), 'A0_s2': arm(os.path.join(R, 'e345/seeds/A0_s2')),
         'A0_s3': arm(os.path.join(R, 'e345/seeds/A0_s3')), 'A1': arm(os.path.join(R, 'e345/A1')),
         'A2': arm(os.path.join(D, 'A2')), 'A3': arm(os.path.join(D, 'A3'))}
    e345 = json.load(open(os.path.join(R, 'e345/summary.json')))
    max_dd = e345['E4']['max_DD_converged']
    out('ARMS (DFEW test, 11697 clips; mean over folds)')
    out('arm      U_real   U_product  Delta_route (mean ± sd)   changed % (pooled: all / 1-9 / 10-12)')
    for k, a in A.items():
        if a:
            out(f"{k:<8} {st.mean(a['U_real']):6.2f}   {st.mean(a['U_product']):6.2f}     "
                f"{st.mean(a['Delta']):+.2f} ± {st.stdev(a['Delta']):.2f}             {a['changed_pooled']:.2f} / "
                f"{a.get('changed_pooled_indep_1_9', float('nan')):.2f} / {a.get('changed_pooled_indep_10_12', float('nan')):.2f}")
    seeds = [A[k] for k in ('A0_s1', 'A0_s2', 'A0_s3') if A[k]]
    seed_range = max(st.mean(s['Delta']) for s in seeds) - min(st.mean(s['Delta']) for s in seeds)
    out(f"run-to-run reference: range of Delta_route over A0 seeds ({len(seeds)}) = {seed_range:.2f}")
    res = {'arms': A, 'seed_range': seed_range, 'max_DD_converged': max_dd}
    a0 = A['A0_s1']
    for k in ('A2', 'A3'):
        if not A[k]:
            out(f'\n{k}: not complete')
            continue
        I, G = diff(A[k], a0, 'Delta'), diff(A[k], a0, 'U_real')
        moves = I['ci90'][0] > 0 and I['mean'] > seed_range
        out(f"\n{k}  I = Delta_{k} - Delta_A0 = {fmt(I)}  (folds {[round(x, 2) for x in I['folds']]})")
        out(f"    G = U_real,{k} - U_real,A0 = {fmt(G)}")
        out(f"    READING: {k} {'MOVES' if moves else 'does NOT move'} I_a (registered rule); "
            f"G {'covers 0: no resolved gain' if covers0(G) else 'resolved'}")
        if k == 'A2':
            rep = I['mean'] > 0 and max_dd >= I['mean']
            out(f"    eps: max DD (converged) {max_dd:+.2f} vs I_A2 {I['mean']:+.2f} -> "
                f"{'eps alone reproduces the dependence increase' if rep else 'not reproduced by eps'}")
            res['E6_eps_reproduces'] = rep
        if 'cost_gate_tanh' in A[k]:
            g = [st.mean(b) for b in A[k]['cost_gate_tanh']]
            out('    tanh(g) per block (mean over folds): ' + ' '.join(f'{x:+.3f}' for x in g))
        res[k] = {'I': I, 'G': G, 'moves': moves}
    if A['A2'] and A['A3']:
        C, GC = diff(A['A2'], A['A3'], 'Delta'), diff(A['A2'], A['A3'], 'U_real')
        special = C['ci90'][0] > 0 and C['mean'] > seed_range
        out(f"\nCONTRAST (prespecified)  C = Delta_A2 - Delta_A3 = {fmt(C)}  (folds {[round(x, 2) for x in C['folds']]})")
        out(f"    U_real,A2 - U_real,A3 = {fmt(GC)}")
        out(f"    READING: {'learned deviations are special (A2 exceeds A3)' if special else 'A2 does NOT exceed A3 -> drop the claim that learned deviations are special; no adapter search'}")
        res['contrast'] = {'C': C, 'G_A2_minus_A3': GC, 'special': special}
    # manipulation check: cost-space profile per block (test folds)
    prof = {'A0': os.path.join(R, 'e2/OT_DFEW_test.json'), 'A1': os.path.join(R, 'e345/E2_A1_DFEW_test.json'),
            'A2': os.path.join(D, 'E2_A2_DFEW_test.json'), 'A3': os.path.join(D, 'E2_A3_DFEW_test.json')}
    P = {k: json.load(open(p)) for k, p in prof.items() if os.path.exists(p)}
    if P:
        out('\nMANIPULATION CHECK (e2_ot.py, DFEW test, mean over folds): kappa | cs_spread_v | H per block')
        for stat in ('kappa', 'cs_spread_v', 'H'):
            out(f'  {stat}')
            out('  block ' + ''.join(f'{k:>9}' for k in P))
            for b in range(12):
                out(f'  {b + 1:>5} ' + ''.join(f"{P[k]['blocks'][b][stat]:>9.4f}" for k in P))
        res['profile'] = {k: {s: [r[s] for r in v['blocks']] for s in ('kappa', 'cs_spread_v', 'H')} for k, v in P.items()}
        res['profile_runs'] = {k: v['run'] for k, v in P.items()}
    json.dump(res, open(os.path.join(D, 'summary.json'), 'w'), indent=1)
    open(os.path.join(D, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
