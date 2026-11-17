#!/usr/bin/env python
"""E3-E5 summary (prereg.md "E3, E4, E5"): per-fold endpoints and the registered readings.

Per arm and fold: U_real = base UAR, U_product = indep_all UAR (blocksub_ot.py fold JSONs), Delta_route = U_real - U_product.
Between-run endpoints are paired by fold: mean, sd and a 90% paired-fold t interval (n = 5, t_0.95,4 = 2.132).

    python e345_summary.py results/e345        -> results/e345/summary.{json,txt}
"""

import json
import os
import statistics as st
import sys

D = sys.argv[1] if len(sys.argv) > 1 else 'results/e345'
T95_4 = 2.1318
lines = []


def out(s=''):
    lines.append(s)
    print(s)


def arm(path):
    p = os.path.join(D, path)
    if not os.path.exists(os.path.join(p, 'fold5.json')):
        return None
    js = [json.load(open(os.path.join(p, f'fold{f}.json'))) for f in range(1, 6)]
    ur = [j['arms']['base']['uar'] for j in js]
    up = [j['arms']['indep_all']['uar'] for j in js]
    ch = [j['arms']['indep_all']['changed_pct'] for j in js]
    n = [j['n_clips'] for j in js]
    return {'U_real': ur, 'U_product': up, 'Delta': [a - b for a, b in zip(ur, up)], 'changed': ch, 'n': n,
            'changed_pooled': sum(c * k for c, k in zip(ch, n)) / sum(n)}


def changed_between(dir_a, key_a, dir_b, key_b):
    """Pooled % of test clips whose predicted class differs between two evaluations (fold npz files, same clip
    order); None when the npz files are not present (they stay on the server)."""
    import numpy as np
    tot = n = 0
    for f in range(1, 6):
        pa, pb = (os.path.join(D, d, f'fold{f}.npz') for d in (dir_a, dir_b))
        if not (os.path.exists(pa) and os.path.exists(pb)):
            return None
        za, zb = np.load(pa), np.load(pb)
        assert (za['labels'] == zb['labels']).all(), (pa, pb)
        tot += int((za[key_a].argmax(1) != zb[key_b].argmax(1)).sum())
        n += len(za['labels'])
    return 100.0 * tot / n


def ci(v):
    m, s = st.mean(v), st.stdev(v)
    h = T95_4 * s / len(v) ** 0.5
    return {'mean': m, 'sd': s, 'ci90': [m - h, m + h], 'folds': v}


def fmt(c):
    return f"{c['mean']:+.2f} ± {c['sd']:.2f} [{c['ci90'][0]:+.2f}, {c['ci90'][1]:+.2f}]"


def main():
    res = {}
    A = {k: arm(p) for k, p in {
        'A0_s1': 'eps/A0_eps0.05', 'A0_s2': 'seeds/A0_s2', 'A0_s3': 'seeds/A0_s3', 'A1': 'A1',
        'eps0.01': 'eps/A0_eps0.01', 'eps0.01_it100': 'eps/A0_eps0.01_it100', 'eps0.02': 'eps/A0_eps0.02',
        'eps0.1': 'eps/A0_eps0.1', 'eps0.2': 'eps/A0_eps0.2', 'trained_eps0.02': 'eps/trained_eps0.02',
        'A4_s1_native': 'A4/s1_native', 'A4_s1_scr': 'A4/s1_scr', 'A4_s2_native': 'A4/s2_native', 'A4_s2_scr': 'A4/s2_scr',
        'A4_s3_native': 'A4/s3_native', 'A4_s3_scr': 'A4/s3_scr'}.items()}
    out('ARMS (DFEW test, 11697 clips; mean over folds)')
    out('arm              U_real   U_product  Delta_route (mean ± sd)   changed % (pooled)')
    for k, a in A.items():
        if a:
            out(f"{k:<16} {st.mean(a['U_real']):6.2f}   {st.mean(a['U_product']):6.2f}     {st.mean(a['Delta']):+.2f} ± {st.stdev(a['Delta']):.2f}"
                f"             {a['changed_pooled']:.2f}")
    res['arms'] = A
    a0, a1 = A['A0_s1'], A['A1']
    # E3
    I = ci([x - y for x, y in zip(a1['Delta'], a0['Delta'])])
    G = ci([x - y for x, y in zip(a1['U_real'], a0['U_real'])])
    seeds = [A[k] for k in ('A0_s1', 'A0_s2', 'A0_s3') if A[k]]
    seed_range = max(st.mean(s['Delta']) for s in seeds) - min(st.mean(s['Delta']) for s in seeds)
    moves = I['ci90'][0] > 0 and I['mean'] > seed_range
    out(f"\nE3  I_A1 = Delta_A1 - Delta_A0 = {fmt(I)}  (folds {[round(x, 2) for x in I['folds']]})")
    out(f"    run-to-run reference: range of Delta_route over A0 seeds ({len(seeds)}) = {seed_range:.2f}")
    out(f"    G_A1 = U_real,A1 - U_real,A0 = {fmt(G)}")
    out(f"    changed % under indep: A0 {a0['changed_pooled']:.2f} -> A1 {a1['changed_pooled']:.2f}")
    out(f"    READING: {'A1 moves I_a -> E6 not run' if moves else 'A1 does NOT move I_a (registered rule) -> E6 is the conditional next step'}; "
        f"G_A1 {'covers 0: no resolved gain' if G['ci90'][0] <= 0 <= G['ci90'][1] else 'resolved'}")
    res['E3'] = {'I_A1': I, 'G_A1': G, 'seed_range': seed_range, 'moves': moves}
    # E4
    dd = {}
    for k in ('eps0.2', 'eps0.1', 'eps0.02', 'eps0.01', 'eps0.01_it100'):
        if A[k]:
            dd[k] = ci([x - y for x, y in zip(A[k]['Delta'], a0['Delta'])])
            out(f"E4  DD({k}) = Delta(eps) - Delta(0.05) = {fmt(dd[k])}")
    te = A['trained_eps0.02']
    if te:
        out(f"    matched training eps 0.02: Delta {st.mean(te['Delta']):+.2f}, I vs A0 {fmt(ci([x - y for x, y in zip(te['Delta'], a0['Delta'])]))}, "
            f"G vs A0 {fmt(ci([x - y for x, y in zip(te['U_real'], a0['U_real'])]))}")
    mx = max((v['mean'] for k, v in dd.items() if k != 'eps0.01'), default=float('nan'))   # eps 0.01 at 10 iterations is unconverged
    rep = I['mean'] > 0 and mx >= I['mean']
    out(f"    READING: max DD (converged) {mx:+.2f} vs I_A1 {I['mean']:+.2f} -> "
        f"{'eps alone reproduces the dependence increase' if rep else 'not reproduced by eps'}")
    res['E4'] = {'DD': dd, 'max_DD_converged': mx, 'reproduces': rep}
    # E5
    e5 = {}
    for s in (1, 2, 3):
        nat, scr, ref = A[f'A4_s{s}_native'], A[f'A4_s{s}_scr'], A[f'A0_s{s}']
        if not (nat and scr and ref):
            continue
        g = ci([x - y for x, y in zip(ref['U_real'], nat['U_real'])])
        dr = ci([x - y for x, y in zip(scr['U_real'], nat['U_real'])])
        # Corrected 2026-11-12 (prereg "Corrections found in the paper review"): the registered sanity check compares the
        # native pooled model with the switched model's product arm; the field stored before under that name was the
        # switched model's own real-vs-product changed %. A4's changed % is against the native model (its U_product).
        dev = max(abs(x - y) for x, y in zip(scr['U_product'], nat['U_real']))
        e5[s] = {'G_train': g, 'Delta_route_A4': dr,
                 'changed_native_vs_switched_real': changed_between(f'A4/s{s}_native', 'prob_base', f'A4/s{s}_scr', 'prob_base'),
                 'sanity_native_vs_switched_product_changed': changed_between(f'A4/s{s}_native', 'prob_base', f'A4/s{s}_scr', 'prob_indep_all'),
                 'sanity_max_fold_abs_U_native_minus_switched_product': dev,
                 'switched_real_vs_switched_product_changed': scr['changed_pooled']}
        out(f"E5  seed {s}: G_train = U_real,A0 - U_A4 = {fmt(g)};  Delta_route,A4 = U_real(scr plan) - U_product = {fmt(dr)}")
        c1, c2 = e5[s]['changed_native_vs_switched_real'], e5[s]['sanity_native_vs_switched_product_changed']
        out(f"    changed % native vs switched (A4's Delta_route reference): {'n/a' if c1 is None else f'{c1:.2f}'};  "
            f"sanity, native vs switched product arm: {'n/a' if c2 is None else f'{c2:.2f}'}% changed, "
            f"max fold |U diff| {dev:.2f}")
    res['E5'] = e5
    json.dump(res, open(os.path.join(D, 'summary.json'), 'w'), indent=1)
    open(os.path.join(D, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
