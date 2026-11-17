#!/usr/bin/env python
"""E2 summary (prereg.md): sanity checks, the per-layer profiles, and the pre-registered rules R1 and R2.

    python e2_summary.py results/e2        -> results/e2/summary.{json,txt}
"""

import glob
import json
import os
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else 'results/e2'
SITE = 'b1_v2a'
lines = []


def out(s=''):
    lines.append(s)
    print(s)


def fstats(vals):
    v = np.array(vals, dtype=float)
    return {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            'min': float(v.min()), 'max': float(v.max()), 'folds': v.tolist()}


def load_ot(name):
    p = os.path.join(D, name + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


def ot_checks(r):
    c = r['checks']
    ok = all(v['cost_max_abs_diff'] < 1e-5 and v['kappa_gt_beta'] == 0 for v in c.values())
    return ok, {f: (v['cost_max_abs_diff'], v['kappa_gt_beta']) for f, v in c.items()}


def ot_table(r, title):
    out(f'\n{title}: {r["host"]} {r["dataset"]} {r["split"]}, clips {sum(r["n_clip"].values())}, eps {r["eps"]}')
    cols = [('raw_spread_v', 'sv_raw'), ('cs_spread_v', 'sv_cost'), ('sigma_v', 'sig_v'), ('cs_spread_a', 'sa_cost'),
            ('sigma_a', 'sig_a'), ('beta', 'beta'), ('kappa', 'kappa'), ('H', 'H'), ('frame_part', 'frame%'),
            ('gate_a2v_abs', '|tanh g|'), ('raw_spread_p', 'sp_raw'), ('cs_spread_p', 'sp_cost'), ('beta_p', 'beta_p')]
    out('block' + ''.join(f'{c[1]:>10}' for c in cols) + '   kappa fold range')
    for b in r['blocks']:
        out(f'{b["block"]:>5}' + ''.join(f'{b[c[0]]:>10.4g}' for c in cols)
            + f'   [{b["kappa_fmin"]:.4g}, {b["kappa_fmax"]:.4g}]')


def main():
    res = {'site': SITE}
    ot = load_ot('OT_CREMAD_test')
    hdir = 'HiCMAE_CREMAD_test_v2' if os.path.isdir(os.path.join(D, 'HiCMAE_CREMAD_test_v2')) else 'HiCMAE_CREMAD_test'
    hi_files = sorted(glob.glob(os.path.join(D, hdir, 'fold*.json')))
    hi = [json.load(open(p)) for p in hi_files]
    assert ot is not None and len(hi) == 5, (ot is not None, len(hi))
    res['hicmae_dir'] = hdir
    out(f'HiCMAE results from {hdir}')
    if hdir.endswith('_v2'):
        v1 = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(D, 'HiCMAE_CREMAD_test', 'fold*.json')))]
        if len(v1) == 5:
            rep = max(abs(a['sites'][s][k] - b['sites'][s][k]) / max(abs(b['sites'][s][k]), 1e-12)
                      for a, b in zip(hi, v1) for s in a['sites'] for k in ('kappa', 'beta', 'beta_a', 'q_spread', 'H_attn'))
            out(f'v2 against v1 statistics (reproducibility): max relative difference {rep:.2e}')
            res['v1_v2_max_rel'] = rep

    # ---- sanity checks
    out('SANITY CHECKS')
    san = os.path.join(D, 'sanity_OT_CREMAD_val10_f1.json')
    if os.path.exists(san):
        rc = json.load(open(san))['reference_check']['max_rel_diff']
        ok1 = all(v < 5e-4 for v in rc.values())
        out(f'1 stage-0a reproduction (val10 fold 1): max rel diff {rc} -> {"PASS" if ok1 else "FAIL"}')
    else:
        ok1 = False
        out('1 stage-0a reproduction: MISSING')
    ok2, c2 = ot_checks(ot)
    out(f'2+4 router cost diff / kappa>beta per fold: {c2} -> {"PASS" if ok2 else "FAIL"}')
    ok3 = ok4 = ok5 = True
    for h in hi:
        for s, c in h['checks'].items():
            if not isinstance(c, dict):
                continue
            ok3 &= c['beff_rel'] < 1e-4 and c['out_rel'] < 1e-2
            ok4 &= c['viol_beta'] == 0 and c['viol_beta_a'] == 0
        ok5 &= abs(h['checks'].get('base_uar_diff', 1.0)) < 1e-9
    beff = max(c['beff_rel'] for h in hi for c in h['checks'].values() if isinstance(c, dict))
    orel = max(c['out_rel'] for h in hi for c in h['checks'].values() if isinstance(c, dict))
    f32 = max(c.get('fp32_rel', float('nan')) for h in hi for c in h['checks'].values() if isinstance(c, dict))
    out(f'3 HiCMAE logits: max B_eff rel {beff:.2e} (fp64), fp32 stats logits vs fp64 {f32:.2e}, '
        f'max output rel {orel:.2e} -> {"PASS" if ok3 else "FAIL"}')
    out(f'4 HiCMAE kappa <= beta, beta_a: {"PASS" if ok4 else "FAIL"}')
    out(f'5 HiCMAE base UAR = P2 eval: {[round(h["base_uar"], 4) for h in hi]} diffs '
        f'{[h["checks"].get("base_uar_diff") for h in hi]} -> {"PASS" if ok5 else "FAIL"}')
    res['sanity'] = {'1': ok1, '2+4_router': ok2, '3': ok3, '4_hicmae': ok4, '5': ok5}
    all_ok = ok1 and ok2 and ok3 and ok4 and ok5

    # ---- profiles
    ot_table(ot, 'ROUTER (primary)')
    sites = list(hi[0]['sites'])
    keys = ['kappa', 'beta', 'beta_a', 'H_attn', 'q_spread', 'k_spread', 'q_sigma', 'k_sigma', 'xq_spread', 'xk_spread',
            'vis_c1_spread', 'vis_c1_in_spread', 'beta_c1']
    hs = {s: {k: fstats([h['sites'][s][k] for h in hi]) for k in keys} for s in sites}
    for s in sites:
        hs[s]['kappa_headmax'] = fstats([h['sites'][s]['kappa_headmax'] for h in hi])
    out(f'\nHiCMAE CREMA-D test (primary), clips {sum(h["n_clips"] for h in hi)}, head mean; d_h {hi[0]["sites"][SITE]["d_h"]}')
    out('site    ' + ''.join(f'{k[:10]:>11}' for k in keys) + '  kappa_headmax   kappa fold range')
    for s in sites:
        out(f'{s:<8}' + ''.join(f'{hs[s][k]["mean"]:>11.4g}' for k in keys)
            + f'  {hs[s]["kappa_headmax"]["mean"]:>12.4g}   [{hs[s]["kappa"]["min"]:.4g}, {hs[s]["kappa"]["max"]:.4g}]')
    raw = {b: {k: float(np.mean([h['raw'][b][k] for h in hi])) for k in hi[0]['raw'][b]} for b in hi[0]['raw']}
    out('raw stream tokens before cross_norm (fold mean): ' + json.dumps({b: {k: round(v, 4) for k, v in r.items()}
                                                                         for b, r in raw.items()}))
    res['hicmae'] = hs
    res['hicmae_raw'] = raw

    # ---- R1
    bl = ot['blocks']
    bstar = max(bl, key=lambda b: b['kappa'])
    ot_folds = [ot['per_fold'][str(f)][bl.index(bstar)]['kappa'] for f in ot['folds']]
    k_hi = hs[SITE]['kappa']
    r1_mean = k_hi['mean'] > bstar['kappa']
    r1_overlap = k_hi['min'] > max(ot_folds)
    r1 = r1_mean and r1_overlap
    out(f'\nR1 interaction: HiCMAE {SITE} kappa {k_hi["mean"]:.4g} (folds {[round(x, 4) for x in k_hi["folds"]]}) vs '
        f'router max block {bstar["block"]} kappa {bstar["kappa"]:.4g} (folds {[round(x, 4) for x in ot_folds]})')
    out(f'   mean exceeds: {r1_mean}; no fold overlap: {r1_overlap} -> R1 {"HOLDS" if r1 else "FAILS"}')
    others = sorted(bl, key=lambda b: -b['kappa'])
    out('   router blocks by kappa: ' + ', '.join(f'{b["block"]}:{b["kappa"]:.3g}' for b in others))

    # ---- R2
    qual = [b for b in bl if b['beta'] < 1]
    vac = [b['block'] for b in bl if b['beta'] >= 1]
    head = max((b['beta'] for b in qual), default=None)
    b_hi = hs[SITE]['beta']['mean']
    r2 = head is not None and b_hi > head
    out(f'\nR2 spread (D3 headline): router headline beta {head if head is None else round(head, 4)} '
        f'(max over blocks with beta < 1: {[b["block"] for b in qual]}); bound vacuous at router blocks {vac}')
    out(f'   HiCMAE {SITE} beta {b_hi:.4g} ({"vacuous, unconstrained" if b_hi >= 1 else "non-vacuous"}) '
        f'-> R2 {"HOLDS" if r2 else "FAILS"}')
    late = [b['block'] for b in bl if b['beta'] >= 1 and b['kappa'] < 0.1]
    out(f'   router blocks with vacuous bound but kappa < 0.1 (late-block limitation): {late}')

    verdict = ('CHECKS FAILED: fix and rerun' if not all_ok else ('PASS' if (r1 and r2) else 'STOP'))
    out(f'\nVERDICT (prereg E2): {verdict}')
    res.update({'R1': {'holds': r1, 'mean_exceeds': r1_mean, 'no_overlap': r1_overlap, 'hicmae': k_hi,
                       'router_block': bstar['block'], 'router_kappa': bstar['kappa'], 'router_folds': ot_folds},
                'R2': {'holds': r2, 'router_headline': head, 'router_qualifying': [b['block'] for b in qual],
                       'router_vacuous': vac, 'hicmae_beta': b_hi},
                'verdict': verdict})

    # ---- descriptive: within-HiCMAE site ordering, secondary settings
    out('\nDescriptive: HiCMAE kappa by site ' + ', '.join(f'{s} {hs[s]["kappa"]["mean"]:.3g}' for s in sites)
        + ' (damage: v2a -2.52 > a2v -0.19; b1 -1.95 > b2 -0.37)')
    for name, title in (('OT_DFEW_test', 'ROUTER DFEW (secondary)'), ('ABV_DFEW_test', 'ABVnet DFEW (secondary)')):
        r = load_ot(name)
        if r is not None:
            ok, c = ot_checks(r)
            ot_table(r, title)
            out(f'checks {c} -> {"PASS" if ok else "FAIL"}')
            res[name] = {'checks_ok': ok, 'blocks': [{k: b[k] for k in ('block', 'kappa', 'beta', 'cs_spread_v',
                                                                           'raw_spread_v', 'beta_p', 'H')}
                                                    for b in r['blocks']]}
    res['router_cremad'] = [{k: b[k] for k in ('block', 'kappa', 'kappa_fmin', 'kappa_fmax', 'beta', 'beta_fmin',
                                                'beta_fmax', 'cs_spread_v', 'raw_spread_v', 'ln_spread_v', 'sigma_v',
                                                'sigma_a', 'cs_spread_p', 'raw_spread_p', 'beta_p', 'H', 'frame_part',
                                                'gate_a2v_abs')} for b in bl]
    with open(os.path.join(D, 'summary.json'), 'w') as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(D, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
