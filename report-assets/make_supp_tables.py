#!/usr/bin/env python
"""Supplementary tables for paper-v2 (appendix.tex inputs supp_tables.tex). Every number is read from a result file:

  tab:repro   HiCMAE reproduction per fold, CREMA-D and MAFW     p1/results/{cremad_of/eval,mafw}/summary.txt (stage 0 lines)
  tab:sites   HiCMAE product replacement by site                  p1/results/{cremad_of/eval,mafw}/summary.json
  tab:mafw    the interventions of Table 1 on MAFW                p1/results/mafw/{nulls,followup}/summary.json
  tab:offsets audio offsets, HiCMAE / MMA-DFER / router (CREMA-D) p1/results/cremad_of/followup, results/p2_mma
  tab:inputlevel EMAP and perceptual-score permutations         p1/results/{cremad_of,mafw}/nulls, results/p2_mma
  tab:folds   per-fold U_real and Delta_route of every DFEW arm   results/e345/summary.json, results/e6/summary.json

    python report-assets/make_supp_tables.py   -> paper-v2/supp_tables.tex
"""

import json
import os
import re
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'paper-v2', 'supp_tables.tex')


def J(*p):
    return json.load(open(os.path.join(ROOT, *p)))


def stage0(path):
    rows = []
    for line in open(os.path.join(ROOT, path)):
        m = re.match(r'\s+fold (\d): UAR\s+([\d.]+) \[([\d.]+)\]\s+diff\s+([+-][\d.]+)', line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
        m = re.match(r'\s+pooled: UAR ([\d.]+) \[([\d.]+)\]', line)
        if m:
            pooled = (float(m.group(1)), float(m.group(2)))
    return rows, pooled


def pm(x):
    return f'${x:+.2f}$'


def ci(a):
    return f"${a['d_uar']:+.2f}$ & $[{a['d_uar_ci'][0]:+.2f}, {a['d_uar_ci'][1]:+.2f}]$ & {a['changed_pct']:.2f}"


def main():
    L = []
    # --- reproduction
    c_rows, c_pool = stage0('p1/results/cremad_of/eval/summary.txt')
    m_rows, m_pool = stage0('p1/results/mafw/summary.txt')
    L += [r'\begin{table}[t]', r'\centering',
          r'\caption{HiCMAE reproduction per fold: our evaluation of the released checkpoints against the per-fold UAR in the '
          r'released README. CREMA-D (OpenFace crops) passes the registered criterion (pooled UAR at most 1 point and every fold at most 2 points '
          r'below the README); MAFW does not, so its results are secondary.}',
          r'\label{tab:repro}', r'\footnotesize', r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
          r'& \multicolumn{3}{c}{CREMA-D} & \multicolumn{3}{c}{MAFW} \\', r'\cmidrule(lr){2-4}\cmidrule(l){5-7}',
          r'Fold & ours & README & diff. & ours & README & diff. \\', r'\midrule']
    for (f, u, r, d), (_, u2, r2, d2) in zip(c_rows, m_rows):
        L.append(f'{f} & {u:.2f} & {r:.2f} & {pm(d)} & {u2:.2f} & {r2:.2f} & {pm(d2)} \\\\')
    L += [r'\midrule', f'pooled & {c_pool[0]:.2f} & {c_pool[1]:.2f} & {pm(c_pool[0] - c_pool[1])} & {m_pool[0]:.2f} & '
          f'{m_pool[1]:.2f} & {pm(m_pool[0] - m_pool[1])} \\\\', r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    # --- sites
    sc, sm = J('p1/results/cremad_of/eval/summary.json')['arms'], J('p1/results/mafw/summary.json')['arms']
    names = [('indep', 'all four maps'), ('indep_v2a', 'video-query maps (blocks 1, 2)'),
             ('indep_a2v', 'audio-query maps (blocks 1, 2)'), ('indep_b1', 'fusion block 1 (both maps)'),
             ('indep_b2', 'fusion block 2 (both maps)'), ('zero', 'message removed')]
    L += [r'\begin{table}[t]', r'\centering',
          r'\caption{HiCMAE: product replacement by site. Change in UAR (points) against the unmodified model with a 90\% '
          r'bootstrap interval over clips, and share of predictions changed (\%), pooled over the five folds.}',
          r'\label{tab:sites}', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}lrrrrrr@{}}',
          r'\toprule', r'& \multicolumn{3}{c}{CREMA-D} & \multicolumn{3}{c}{MAFW (secondary)} \\',
          r'\cmidrule(lr){2-4}\cmidrule(l){5-7}',
          r'Replaced & $\Delta$UAR & 90\% interval & changed & $\Delta$UAR & 90\% interval & changed \\', r'\midrule']
    for k, lab in names:
        L.append(f'{lab} & {ci(sc[k])} & {ci(sm[k])} \\\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    # --- MAFW interventions
    nm, fu = J('p1/results/mafw/nulls/summary.json'), J('p1/results/mafw/followup/summary.json')
    rows = [('indep', 'Product replacement'), ('rowperm', 'Row permutation of the coupling'),
            ('uniform', 'Uniform coupling'), ('papa', 'Training-mean coupling (PAPA)'),
            ('devonly', 'Frame-specific part only'), ('zero', 'Message removed'), ('perm_a', 'Audio of another clip')]
    L += [r'\begin{table}[t]', r'\centering',
          f"\\caption{{HiCMAE on MAFW ({nm['n_clips']:,} test clips, base UAR {nm['base_uar']:.2f}; secondary, see "
          r"Table~\ref{tab:repro}): the interventions of Table~\ref{tab:contrast}, plus the time-only row permutation and the "
          r"audio offsets. Change in UAR with a 90\% bootstrap interval over clips, and share of predictions changed.}",
          r'\label{tab:mafw}', r'\footnotesize', r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
          r'Intervention & $\Delta$UAR & 90\% interval & changed \% \\', r'\midrule']
    for k, lab in rows:
        L.append(f"{lab} & {ci(nm['arms'][k])} \\\\")
    rt = fu['arms']['rowperm_t']
    L.append(f"Time-only row permutation & {ci(rt)} \\\\")
    dl = {k: v for k, v in fu['arms'].items() if k.startswith('delay') and not k.endswith('_fus')}
    worst = min(dl.values(), key=lambda a: a['d_uar'])
    best = max(dl.values(), key=lambda a: a['d_uar'])
    covers = all(a['d_uar_ci'][0] <= 0 <= a['d_uar_ci'][1] for a in dl.values())
    L.append(f"Audio offsets $\\pm$40 to $\\pm$640\\,ms & ${worst['d_uar']:+.2f}$ to ${best['d_uar']:+.2f}$ & "
             f"{'all cover 0' if covers else 'see text'} & {min(a['changed_pct'] for a in dl.values()):.1f}--"
             f"{max(a['changed_pct'] for a in dl.values()):.1f} \\\\")
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    # --- audio offsets, CREMA-D: HiCMAE against the content-only MMA-DFER and the router (P1-followup, P2-MMA)
    hf = J('p1/results/cremad_of/followup/summary.json')['arms']
    pm_ = J('results/p2_mma/CREMAD/summary.json')
    mn, ms = pm_['models']['none']['arms'], pm_['models']['scr']['arms']
    ds = ['-640', '-320', '-160', '-80', '-40', '+40', '+80', '+160', '+320', '+640']
    L += [r'\begin{table}[t]', r'\centering',
          r'\caption{Audio offsets on CREMA-D: change in UAR (points) when the audio is shifted by $d$\,ms against the video '
          r'($d > 0$: sound late); values whose 90\% bootstrap interval excludes zero are in bold. MMA-DFER, '
          r'whose exchange has no path from audio time to video time, is the content-only reference; the router adds its '
          r'routing step to the same host. The router’s change minus MMA-DFER’s lies between $-0.21$ and $+0.27$ at every '
          r'offset, every interval inside $[-0.60, +0.66]$: routing adds no resolved alignment sensitivity. '
          r'\citet{groupaccess} ran a similar probe ($\pm 80$ to $\pm 640$\,ms) on the router and its MMA-DFER host on DFEW development data.}',
          r'\label{tab:offsets}', r'\scriptsize', r'\setlength{\tabcolsep}{2.6pt}',
          r'\begin{tabular}{@{}l' + 'r' * len(ds) + r'@{}}', r'\toprule',
          r'$d$ (ms) & ' + ' & '.join(f'${d}$' for d in ds) + r' \\', r'\midrule']

    def cell(a):
        v = f"{a['d_uar']:+.2f}"
        return f'$\\mathbf{{{v}}}$' if not (a['d_uar_ci'][0] <= 0 <= a['d_uar_ci'][1]) else f'${v}$'
    for lab, arms in (('HiCMAE', hf), ('MMA-DFER', mn), ('Router (MMA-DFER host)', ms)):
        L.append(lab + ' & ' + ' & '.join(cell(arms[f'delay{d}']) for d in ds) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    # --- input-level references: EMAP and the perceptual score
    hc, hm = J('p1/results/cremad_of/nulls/summary.json')['arms'], J('p1/results/mafw/nulls/summary.json')['arms']
    pmm = J('results/p2_mma/MAFW/summary.json')['models']
    L += [r'\begin{table}[t]', r'\centering',
          r'\caption{Input-level references: change in UAR (points, 90\% bootstrap interval) under EMAP \citep{hessel2020emap} '
          r'and under the perceptual-score permutations \citep{gat2021perceptual}, which give each clip the audio or the video '
          r'of another test clip. MAFW: secondary for HiCMAE (Table~\ref{tab:repro}).}',
          r'\label{tab:inputlevel}', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}llrrr@{}}',
          r'\toprule', r'Model & Corpus & EMAP & audio of another clip & video of another clip \\', r'\midrule']

    def iv(a):
        return f"${a['d_uar']:+.2f}$ $[{a['d_uar_ci'][0]:+.2f}, {a['d_uar_ci'][1]:+.2f}]$"
    L.append(f"HiCMAE & CREMA-D & {iv(hc['emap'])} & {iv(hc['perm_a'])} & {iv(hc['perm_v'])} \\\\")
    L.append(f"HiCMAE & MAFW & {iv(hm['emap'])} & {iv(hm['perm_a'])} & {iv(hm['perm_v'])} \\\\")
    L.append(f"MMA-DFER & CREMA-D & -- & {iv(mn['perm_a'])} & -- \\\\")
    L.append(f"MMA-DFER & MAFW & -- & {iv(pmm['none']['arms']['perm_a'])} & -- \\\\")
    L.append(f"Router & CREMA-D & -- & {iv(ms['perm_a'])} & -- \\\\")
    L.append(f"Router & MAFW & -- & {iv(pmm['scr']['arms']['perm_a'])} & -- \\\\")
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    # --- exact product reference (prereg "Exact product reference, WAR and a continuous endpoint")
    ex = J('results', 'exactprod', 'summary.json')['checkpoints']
    names = [('DFEW_A0_s1', 'A0, seed 1'), ('DFEW_A0_s2', 'A0, seed 2'), ('DFEW_A0_s3', 'A0, seed 3'), ('DFEW_A1', 'A1'),
             ('DFEW_A2', 'A2'), ('DFEW_A3', 'A3'), ('DFEW_trained_eps0.02', r'trained, $\varepsilon = 0.02$'),
             ('CREMAD_router', 'router, CREMA-D'), ('DFEW_abv', 'ABVnet router')]
    L += [r'\begin{table}[t]', r'\centering',
          r'\caption{Exact product reference. $U_{\mathrm{product}}$ uses the solver\textquoteright s own marginals (the '
          r'registered primary), $U_{\mathrm{exact}}$ the uniform plan over the real slots, the exact product of the target '
          r'marginals; agreement: share of predictions on which the two references agree. $\Delta_{\log p}$: mean drop in '
          r'the log-probability of the true class ($10^{-3}$ nats); $\Delta$WAR: WAR-level dependence. DFEW unless stated; '
          r'means over the five folds.}',
          r'\label{tab:exact}', r'\scriptsize', r'\setlength{\tabcolsep}{2.6pt}', r'\begin{tabular}{@{}lrrrrrrrrrr@{}}',
          r'\toprule', r'& & & & \multicolumn{2}{c}{$\Delta_{\mathrm{route}}$} & & \multicolumn{2}{c}{$\Delta_{\log p}$} & '
          r'\multicolumn{2}{c}{$\Delta$WAR} \\', r'\cmidrule(lr){5-6}\cmidrule(lr){8-9}\cmidrule(l){10-11}',
          r'Checkpoint & $U_{\mathrm{real}}$ & $U_{\mathrm{product}}$ & $U_{\mathrm{exact}}$ & solver & exact & agree \% & '
          r'solver & exact & solver & exact \\', r'\midrule']
    m = st.mean
    for k, lab in names:
        c = ex[k]
        L.append(f"{lab} & {m(c['U_real']):.2f} & {m(c['U_indep']):.2f} & {m(c['U_exact']):.2f} & ${m(c['D_route']):+.2f}$ & "
                 f"${m(c['D_exact']):+.2f}$ & {c['agree_pooled']:.2f} & {1e3 * m(c['lp_indep']):.1f} & "
                 f"{1e3 * m(c['lp_exact']):.1f} & ${m(c['DW_route']):+.2f}$ & ${m(c['DW_exact']):+.2f}$ \\\\")
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    # --- per-fold DFEW arms
    e3, e6 = J('results', 'e345', 'summary.json')['arms'], J('results', 'e6', 'summary.json')['arms']
    arms = [('A0, seed 1', e3['A0_s1']), ('A0, seed 2', e3['A0_s2']), ('A0, seed 3', e3['A0_s3']), ('A1', e3['A1']),
            ('A2', e6['A2']), ('A3', e6['A3']), (r'A0, $\varepsilon = 0.2$', e3['eps0.2']),
            (r'A0, $\varepsilon = 0.1$', e3['eps0.1']), (r'A0, $\varepsilon = 0.02$', e3['eps0.02']),
            (r'A0, $\varepsilon = 0.01$', e3['eps0.01_it100']), (r'trained, $\varepsilon = 0.02$', e3['trained_eps0.02'])]
    L += [r'\begin{table}[t]', r'\centering',
          r'\caption{Per-fold values of every DFEW arm in Table~\ref{tab:arms}: UAR with the model’s own plans '
          r'($U_{\mathrm{real}}$) and the routing dependence $\Delta_{\mathrm{route}}$ (points). For A4, $U_{\mathrm{real}}$ '
          r'is the model with the transport plan switched on at inference and $\Delta_{\mathrm{route}}$ its difference from '
          r'the pooled model as trained.}',
          r'\label{tab:folds}', r'\scriptsize', r'\setlength{\tabcolsep}{3.2pt}', r'\begin{tabular}{@{}lrrrrrrrrrr@{}}',
          r'\toprule', r'& \multicolumn{5}{c}{$U_{\mathrm{real}}$, folds 1--5} & '
          r'\multicolumn{5}{c}{$\Delta_{\mathrm{route}}$, folds 1--5} \\', r'\cmidrule(lr){2-6}\cmidrule(l){7-11}',
          r'Arm & 1 & 2 & 3 & 4 & 5 & 1 & 2 & 3 & 4 & 5 \\', r'\midrule']
    for lab, a in arms:
        L.append(lab + ' & ' + ' & '.join(f'{x:.2f}' for x in a['U_real']) + ' & ' +
                 ' & '.join(f'${x:+.2f}$' for x in a['Delta']) + r' \\')
    for s in (1, 2, 3):
        nat, scr = e3[f'A4_s{s}_native'], e3[f'A4_s{s}_scr']
        L.append(f'A4, seed {s} & ' + ' & '.join(f'{x:.2f}' for x in scr['U_real']) + ' & ' +
                 ' & '.join(f'${x - y:+.2f}$' for x, y in zip(scr['U_real'], nat['U_real'])) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    open(OUT, 'w').write('% Generated by report-assets/make_supp_tables.py; do not edit by hand.\n' + '\n'.join(L))
    print('->', OUT)


if __name__ == '__main__':
    main()
