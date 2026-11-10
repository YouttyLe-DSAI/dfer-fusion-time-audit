#!/usr/bin/env python
"""P2 report assets: reproduced CREMA-D (OpenFace faces), MAFW (non-primary) and the group's OT router.

Reuses make_assets.py's figure, table and macro code with its data globals swapped for the P2 results,
and writes under report-assets/p2/ (figures/{vi,en}, tables, data, numbers.tex). New here: the
reproduction table, the HiCMAE-vs-OT contrast figure and table, and the readings table.

    python make_assets_p2.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

import make_assets as M   # loads the phase-A data at import; the globals are swapped below

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'p2')


def R(*p):
    return os.path.join(ROOT, *p)


def load(*p):
    with open(os.path.join(*p)) as fh:
        return json.load(fh)


COF, MAF, OT = R('p1', 'results', 'cremad_of'), R('p1', 'results', 'mafw'), R('results', 'p2_ot')
MAFW_README = {1: 36.02, 2: 41.57, 3: 46.46, 4: 47.50, 5: 41.88}
MAFW_TOTAL = (42.69, 56.21)

# phase-A (non-reproduced crops) results, kept for the robustness table
OLD = {'P1S': M.P1S, 'NS': M.NS, 'FS': M.FS, 'DCS': M.DCS}

# primary CREMA-D = the OpenFace runs
M.HERE = OUT
M.P1S = load(COF, 'eval', 'summary.json')
M.P1F = {f: load(COF, 'eval', f'fold{f}.json') for f in range(1, 6)}
M.NS = load(COF, 'nulls', 'summary.json')
M.FS = load(COF, 'followup', 'summary.json')
M.DCS = load(COF, 'desync', 'summary.json')
MAF_EVAL = load(MAF, 'eval', 'summary.json') if os.path.exists(os.path.join(MAF, 'eval', 'summary.json')) \
    else load(MAF, 'summary.json')
MAF_EVAL_F = {f: load(MAF, 'eval' if os.path.isdir(os.path.join(MAF, 'eval')) else '', f'fold{f}.json')
              for f in range(1, 6)}
MAF_NS = load(MAF, 'nulls', 'summary.json')
MAF_FS = load(MAF, 'followup', 'summary.json')
MAF_DCS = load(MAF, 'desync', 'summary.json')
OT_C, OT_M = load(OT, 'CREMAD', 'summary.json'), load(OT, 'MAFW', 'summary.json')

KEEP = {'tab5_p1_reproduction', 'tab6_p1_interventions', 'tab7_p1_per_fold', 'tab8_p1_attention_H',
        'tab9_p1_nulls', 'tab10_f3_trace', 'tab11_p1_followup', 'tab12_desync_control',
        'p1_reproduction', 'p1_interventions', 'p1_per_fold', 'p1_attention_H', 'p1_nulls', 'f3_trace',
        'p1_followup', 'desync_control'}
_wt, _wc, _save = M.write_tex, M.write_csv, M.save
M.write_tex = lambda name, lines: _wt(name, lines) if name in KEEP else None
M.write_csv = lambda name, header, rows: _wc(name, header, rows) if name in KEEP else None
MAFW_KEEP = {'tab9_p1_nulls', 'tab11_p1_followup', 'tab12_desync_control', 'p1_nulls', 'p1_followup',
             'desync_control'}
PREFIXES = ('Null', 'Fup', 'Dc')


def mafw_assets():
    """MAFW ladder tables/figures (suffix _mafw) and its Null*/Fup*/Dc* macros renamed Hm*.

    Must run before the CREMA-D M.numbers() call: M.numbers() rewrites numbers.tex.
    """
    saved = (M.NS, M.FS, M.DCS, M.P1S, M.P1F, M.write_tex, M.write_csv, M.save)
    M.NS, M.FS, M.DCS, M.P1S, M.P1F = MAF_NS, MAF_FS, MAF_DCS, MAF_EVAL, MAF_EVAL_F
    M.write_tex = lambda name, lines: _wt(name + '_mafw', lines) if name in MAFW_KEEP else None
    M.write_csv = lambda name, h, rows: _wc(name + '_mafw', h, rows) if name in MAFW_KEEP else None
    M.save = lambda fig, name, lang: _save(fig, name + '_mafw', lang)
    try:
        M.tables()
        N = M.numbers()
        for lang in ('vi', 'en'):
            M.fig_nulls(M.T[lang], lang)
            M.fig_followup(M.T[lang], lang)
            M.fig_desync_control(M.T[lang], lang)
    finally:
        M.NS, M.FS, M.DCS, M.P1S, M.P1F, M.write_tex, M.write_csv, M.save = saved
    return {'Hm' + k: v for k, v in N.items() if k.startswith(PREFIXES)}


def tab_robustness():
    """Phase A (the group's crops, not reproduced) vs P2 (OpenFace, reproduced), same scripts and readings."""
    sg, ci, f = M.sg, M.ci, M.f
    rows_d = [('P1: ghép nối độc lập', 'P1S', 'indep'), ('P1: bỏ cross-attention', 'P1S', 'zero'),
              ('Hoán vị token', 'NS', 'rowperm'), ('Hoán vị theo thời gian', 'NS', 'rowperm_t'),
              ('Hoán vị theo không gian', 'FS', 'rowperm_s'), ('Attention đều', 'NS', 'uniform'),
              ('Map TB tập train (PAPA)', 'NS', 'papa'), ('EMAP', 'NS', 'emap'),
              ('Thay audio của clip khác', 'NS', 'perm_a'), ('Lệch $-160$ ms (tiếng sớm)', 'FS', 'delay-160'),
              ('Lệch $+160$ ms (tiếng muộn)', 'FS', 'delay+160'), ('Mất tiếng', 'FS', 'silence')]
    new = {'P1S': M.P1S, 'NS': M.NS, 'FS': M.FS, 'DCS': M.DCS}
    L = ['\\begin{tabular}{l rr r}', '\\toprule',
         'Đại lượng (chênh UAR, KTC 90\\%) & Crop của nhóm & OpenFace & Chênh \\\\', '\\midrule',
         f"UAR gốc & {f(OLD['P1S']['stage0']['pooled_uar'])} & {f(M.P1S['stage0']['pooled_uar'])} & "
         f"{sg(M.P1S['stage0']['pooled_uar'] - OLD['P1S']['stage0']['pooled_uar'])} \\\\", '\\midrule']
    rows, diffs = [], {}
    for lab, src, arm in rows_d:
        a, b = OLD[src]['arms'][arm], new[src]['arms'][arm]
        diffs[arm] = b['d_uar'] - a['d_uar']
        L.append(f"{lab} & {sg(a['d_uar'])} {ci(a['d_uar_ci'])} & {sg(b['d_uar'])} {ci(b['d_uar_ci'])} & "
                 f"{sg(diffs[arm])} \\\\")
        rows.append([arm, a['d_uar'], *a['d_uar_ci'], b['d_uar'], *b['d_uar_ci']])
    ia, ib = OLD['NS']['readings']['factorial_interaction'], M.NS['readings']['factorial_interaction']
    L.append(f"Tương tác (zero $-$ indep $-$ devonly) & {sg(ia['value'])} {ci(ia['ci'])} & "
             f"{sg(ib['value'])} {ci(ib['ci'])} & {sg(ib['value'] - ia['value'])} \\\\")
    da, db = OLD['DCS']['D']['-160'], M.DCS['D']['-160']
    L.append(f"$D_d$ ở tiếng sớm 169 ms (C1) & {sg(da['value'])} {ci(da['ci'])} & {sg(db['value'])} {ci(db['ci'])} & "
             f"{sg(db['value'] - da['value'])} \\\\")
    ta, tb = OLD['FS']['readings']['D1_tolerance_ms'], M.FS['readings']['D1_tolerance_ms']
    L += ['\\midrule', f"Ngưỡng chịu lệch muộn / sớm (ms) & {ta['late']} / {ta['early']} & {tb['late']} / {tb['early']} & \\\\",
          '\\bottomrule', '\\end{tabular}']
    _wt('tab15_robustness', L)
    _wc('robustness', ['arm', 'crop_d_uar', 'crop_lo', 'crop_hi', 'openface_d_uar', 'openface_lo', 'openface_hi'], rows)
    coupling = ('indep', 'rowperm', 'rowperm_t', 'emap')
    return {'RobCouplingMaxShift': f(max(abs(diffs[a]) for a in coupling)),
            'RobPapaOld': sg(OLD['NS']['arms']['papa']['d_uar']), 'RobPapaOldCI': ci(OLD['NS']['arms']['papa']['d_uar_ci']),
            'RobSilenceOld': sg(OLD['FS']['arms']['silence']['d_uar']),
            'RobPermAudioOld': sg(OLD['NS']['arms']['perm_a']['d_uar']),
            'RobBaseOld': f(OLD['P1S']['stage0']['pooled_uar'])}

SERIES = [('hc', 'HiCMAE · CREMA-D', 'HiCMAE · CREMA-D', M.SLOT[0], True),
          ('hm', 'HiCMAE · MAFW (sơ bộ)', 'HiCMAE · MAFW (preliminary)', M.SLOT[1], False),
          ('oc', 'Router OT · CREMA-D', 'OT router · CREMA-D', M.SLOT[2], True),
          ('om', 'Router OT · MAFW', 'OT router · MAFW', M.SLOT[3], True)]
CONTRAST_ARMS = [('indep', 'Ghép nối độc lập', 'Independent coupling'),
                 ('rowperm', 'Hoán vị token / frame', 'Token / frame permutation'),
                 ('papa', 'Map TB tập train (PAPA)', 'Training-mean map (PAPA)'),
                 ('devonly', 'Bỏ phần chung của clip', 'Remove the clip-level part'),
                 ('zero', 'Bỏ thông điệp', 'Remove the message')]


def arms_of(key):
    return {'hc': M.NS, 'hm': MAF_NS, 'oc': OT_C, 'om': OT_M}[key]['arms']


def fig_contrast(lang):
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    n = len(CONTRAST_ARMS)
    for j, (key, vi, en, c, primary) in enumerate(SERIES):
        A = arms_of(key)
        for i, (arm, _, _) in enumerate(CONTRAST_ARMS):
            y = (n - 1 - i) + (1.5 - j) * 0.19
            r = A[arm]
            ax.plot(r['d_uar_ci'], [y, y], color=c, lw=1.5, zorder=3)
            ax.plot([r['d_uar']], [y], 'o', ms=4.8, zorder=4, color=c, mec=c if not primary else M.SURF,
                    mfc=c if primary else M.SURF, mew=1.1, label=(vi if lang == 'vi' else en) if i == 0 else None)
    ax.axvline(0, color=M.AXIS, lw=0.9, zorder=1)
    ax.axvline(-1, color=M.INK2, lw=0.6, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([(vi if lang == 'vi' else en) for _, vi, en in CONTRAST_ARMS][::-1], fontsize=7.2)
    ax.tick_params(axis='y', length=0)
    for k in range(n - 1):
        ax.axhline(k + 0.5, color=M.GRID, lw=0.6, zorder=0)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel(M.T[lang]['duar'], fontsize=7.6)
    ax.legend(loc='lower center', ncol=2, fontsize=6.8, bbox_to_anchor=(0.45, 1.0))
    fig.tight_layout()
    M.save(fig, 'fig11_hicmae_vs_ot', lang)


def tables_p2():
    sg, ci, f = M.sg, M.ci, M.f
    # reproduction: CREMA-D (OpenFace) and MAFW
    L = ['\\begin{tabular}{l rrr rrr}', '\\toprule',
         ' & \\multicolumn{3}{c}{CREMA-D (OpenFace)} & \\multicolumn{3}{c}{MAFW (crop của nhóm)} \\\\',
         '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}',
         'Fold & UAR & README & Chênh & UAR & README & Chênh \\\\', '\\midrule']
    rows = []
    for k in range(1, 6):
        c = M.P1F[k]['arms']['base']['uar']
        m = MAF_EVAL_F[k]['arms']['base']['uar']
        L.append(f'{k} & {f(c)} & {f(M.README_UAR[k])} & {sg(c - M.README_UAR[k])} & '
                 f'{f(m)} & {f(MAFW_README[k])} & {sg(m - MAFW_README[k])} \\\\')
        rows.append([k, c, M.README_UAR[k], m, MAFW_README[k]])
    c, m = M.P1S['stage0']['pooled_uar'], MAF_EVAL['stage0']['pooled_uar']
    L += ['\\midrule', f'Gộp & {f(c)} & {f(M.README_TOTAL[0])} & {sg(c - M.README_TOTAL[0])} & '
          f'{f(m)} & {f(MAFW_TOTAL[0])} & {sg(m - MAFW_TOTAL[0])} \\\\',
          f"Kết luận & \\multicolumn{{3}}{{c}}{{{'đạt' if M.P1S['stage0']['pass'] else 'không đạt'}}} & "
          f"\\multicolumn{{3}}{{c}}{{{'đạt' if MAF_EVAL['stage0']['pass'] else 'không đạt'}}} \\\\",
          '\\bottomrule', '\\end{tabular}']
    _wt('tab0_reproduction', L)
    _wc('reproduction', ['fold', 'cremad_of_uar', 'cremad_readme', 'mafw_uar', 'mafw_readme'], rows)

    # HiCMAE vs OT contrast
    L = ['\\begin{tabular}{l rrrr}', '\\toprule',
         'Nhánh & ' + ' & '.join(s[1] for s in SERIES) + ' \\\\', '\\midrule']
    rows = []
    for arm, vi, _ in CONTRAST_ARMS + [('perm_a', 'Audio của clip khác', '')]:
        cells = []
        for key, *_ in SERIES:
            r = arms_of(key)[arm]
            cells.append(f"{sg(r['d_uar'])} ({f(r['changed_pct'], 1)}\\%)")
            rows.append([arm, key, r['d_uar'], *r['d_uar_ci'], r['changed_pct']])
        L.append(f'{vi} & ' + ' & '.join(cells) + ' \\\\')
    L += ['\\midrule', 'UAR gốc & ' + ' & '.join(f(v) for v in (M.NS['base_uar'], MAF_NS['base_uar'],
                                                                 OT_C['base_uar_test'], OT_M['base_uar_test'])) + ' \\\\',
          '\\bottomrule', '\\end{tabular}']
    _wt('tab13_hicmae_vs_ot', L)
    _wc('hicmae_vs_ot', ['arm', 'series', 'd_uar', 'ci_lo', 'ci_hi', 'changed_pct'], rows)

    # pre-registered readings across models and datasets
    vi = {'alignment reading': 'khớp cặp', 'distribution reading': 'phân phối', 'mixed': 'lẫn lộn',
          'input-level AV interactions used': 'có dùng', 'mechanism-function dissociation': 'tách cơ chế/hàm',
          'weak': 'yếu', 'load-bearing': 'có tác dụng', 'inert': 'trơ',
          'time carries the alignment': 'thời gian', 'space also carries alignment': 'cả không gian',
          'alignment': 'khớp thời gian', 'content': 'nội dung', 'OT routing predictively inert': 'trơ'}

    def t(x):
        return vi.get(x, x)

    def star(ns):          # label produced only by the ">= 5% changed" arm of the class rule
        A = ns['arms']
        sub1 = lambda r: r['d_uar'] > -1.0  # noqa: E731
        n1 = t(ns['readings']['N1']) + ('$^{*}$' if sub1(A['rowperm']) and sub1(A['indep']) else '')
        e = A['emap']
        n2 = t(ns['readings']['N2']) + ('$^{*}$' if e['d_uar_ci'][0] <= 0 <= e['d_uar_ci'][1] else '')
        return n1, n2

    def tol(fs):
        mx = max(abs(int(d)) for d in fs['readings']['delay_curve'])
        v = fs['readings']['D1_tolerance_ms']
        return ' / '.join(('$\\geq$' if v[k] == mx else '') + str(v[k]) for k in ('late', 'early'))

    def c1(dcs):
        if all(r['reading'] == 'not applicable' for r in dcs['D'].values()):
            return 'không áp dụng (lệch không hại)'
        return t(dcs['readings']['C1'])
    (hn1, hn2), (mn1, mn2) = star(M.NS), star(MAF_NS)
    rd = [
        ('N1: trao nhầm token so với ghép nối độc lập', hn1, mn1),
        ('N2: EMAP (tương tác mức đầu vào)', hn2, mn2),
        ('N3: map cố định (PAPA)', t(M.NS['readings']['N3_papa_class']), t(MAF_NS['readings']['N3_papa_class'])),
        ('T1: không gian hay thời gian', t(M.FS['readings']['T1']), t(MAF_FS['readings']['T1'])),
        ('D1: ngưỡng chịu lệch (muộn / sớm, ms)', tol(M.FS), tol(MAF_FS)),
        ('C1: lệch thời gian hay nội dung', c1(M.DCS), c1(MAF_DCS)),
    ]
    L = ['\\begin{tabular}{l ll}', '\\toprule', 'Kết luận ghi trước & HiCMAE · CREMA-D & HiCMAE · MAFW (sơ bộ) \\\\',
         '\\midrule'] + [f'{a} & {b} & {c} \\\\' for a, b, c in rd] + [
        '\\midrule', '\\multicolumn{3}{l}{\\textit{Router OT}} \\\\',
        f"O1: ghép cặp (CREMA-D / MAFW) & {t(OT_C['readings']['O1'])} & {t(OT_M['readings']['O1'])} \\\\",
        f"O2: bỏ thông điệp (CREMA-D / MAFW) & {t(OT_C['readings']['O2_zero_class'])} & "
        f"{t(OT_M['readings']['O2_zero_class'])} \\\\", '\\bottomrule', '\\end{tabular}']
    _wt('tab14_readings', L)


def numbers_p2(extra):
    N = dict(extra)
    f, sg, ci = M.f, M.sg, M.ci
    N['MafwPooledUAR'] = f(MAF_EVAL['stage0']['pooled_uar'])
    N['MafwReadmeUAR'] = f(MAFW_TOTAL[0])
    for key, src in (('Hm', MAF_NS), ('Oc', OT_C), ('Om', OT_M)):
        for arm, an in (('indep', 'Indep'), ('rowperm', 'Rowperm'), ('zero', 'Zero'), ('devonly', 'Devonly'),
                        ('papa', 'Papa'), ('perm_a', 'PermAudio')):
            r = src['arms'][arm]
            N[f'{key}{an}DUAR'] = sg(r['d_uar'])
            N[f'{key}{an}DUARCI'] = ci(r['d_uar_ci'])
            N[f'{key}{an}Changed'] = f(r['changed_pct'])
    N['OcBaseUAR'], N['OmBaseUAR'] = f(OT_C['base_uar_test']), f(OT_M['base_uar_test'])
    N['OtGateMeanAbs'] = f(np.mean([OT_C['readings']['gate_tanh_a2v_mean_abs'],
                                    OT_M['readings']['gate_tanh_a2v_mean_abs']]), 3)
    N['HmBaseUAR'] = f(MAF_NS['base_uar'])
    N['HmReadingNOne'] = MAF_NS['readings']['N1']
    N['HmReadingTOne'] = MAF_FS['readings']['T1']
    N['HmReadingCOne'] = MAF_DCS['readings']['C1']
    N['HmTolLate'] = str(MAF_FS['readings']['D1_tolerance_ms']['late'])
    N['HmTolEarly'] = str(MAF_FS['readings']['D1_tolerance_ms']['early'])
    for arm, an in (('indep_v2a', 'IndepVtoA'), ('indep_a2v', 'IndepAtoV'), ('indep_b1', 'IndepBlockOne'),
                    ('indep_b2', 'IndepBlockTwo')):
        N[f'HmPone{an}DUAR'] = sg(MAF_EVAL['arms'][arm]['d_uar'])
    dc = [r['d_uar'] for k, r in MAF_FS['arms'].items() if k.startswith('delay') and not k.endswith('_fus')]
    N['HmDelayMin'], N['HmDelayMax'] = sg(min(dc)), sg(max(dc))
    N['HmInteraction'] = sg(MAF_NS['readings']['factorial_interaction']['value'])
    N['HmInteractionCI'] = ci(MAF_NS['readings']['factorial_interaction']['ci'])
    with open(os.path.join(OUT, 'numbers.tex'), 'a') as fh:
        fh.write('% P2 additions (make_assets_p2.py)\n')
        for k, v in N.items():
            fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    return N


# ---------------------------------------------------------------- P2-MMA (MMA-DFER published exchange)
MMA = R('results', 'p2_mma')
MMA_S = {ds: load(MMA, ds, 'summary.json') for ds in ('CREMAD', 'MAFW')
         if os.path.exists(os.path.join(MMA, ds, 'summary.json'))}
MMA_ROWS = [('zero_nat', 'Bỏ trao đổi trung bình (hai chiều)', 'Mean exchange removed (both directions)'),
            ('zero_nat_a2v', 'Bỏ: hình nhận trung bình audio', 'Removed: video receives the audio mean'),
            ('zero_nat_v2a', 'Bỏ: audio nhận trung bình hình', 'Removed: audio receives the video mean'),
            ('zero_late', 'Bỏ cộng CLS audio vào frame', 'Audio class token not added to frames'),
            ('zero_route', 'Bỏ bước định tuyến OT', 'OT routing step removed'),
            ('zero_all', 'Bỏ mọi đường audio', 'Every audio path removed'),
            ('perm_a', 'Audio của clip khác', 'Another clip\'s audio'),
            ('silence', 'Mất tiếng', 'Silence')]


def fig_offset_models(lang):
    """Audio-offset curves: HiCMAE vs MMA-DFER (content-only by construction) vs the OT router, per dataset."""
    N = M.NT[lang]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    lows = []
    for ax, ds, hic, title, prim in ((axes[0], 'CREMAD', M.FS, 'CREMA-D', True),
                                     (axes[1], 'MAFW', MAF_FS, 'MAFW', False)):
        S = MMA_S[ds]['models']
        delays = sorted(int(d) for d in hic['readings']['delay_curve'])
        grid = sorted(delays + [0])
        pos = list(range(len(grid)))
        xp = lambda ms: float(np.interp(ms, grid, np.arange(len(grid))))  # noqa: E731
        ax.axvspan(xp(-45), xp(125), color=M.BAND, zorder=0)
        series = [(hic['arms'], 'HiCMAE', M.SLOT[0], prim),
                  (S['none']['arms'], 'MMA-DFER', M.SLOT[2], True),
                  (S['scr']['arms'], 'Router OT' if lang == 'vi' else 'OT router', M.SLOT[1], True)]
        for A, lab, c, filled in series:
            mid = [0.0 if d == 0 else A[f'delay{d:+d}']['d_uar'] for d in grid]
            lo = [0.0 if d == 0 else A[f'delay{d:+d}']['d_uar_ci'][0] for d in grid]
            hi = [0.0 if d == 0 else A[f'delay{d:+d}']['d_uar_ci'][1] for d in grid]
            lows.append(min(lo))
            ax.fill_between(pos, lo, hi, color=c, alpha=0.12, lw=0, zorder=2)
            ax.plot(pos, mid, color=c, lw=1.5, marker='o', ms=4.0, mfc=c if filled else M.SURF,
                    mec=M.SURF if filled else c, mew=1.0, zorder=3, label=lab)
        ax.axhline(0, color=M.AXIS, lw=0.9, zorder=1)
        ax.axhline(-1, color=M.INK2, lw=0.6, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(pos)
        ax.set_xticklabels([f'{d:+d}'.replace('-', '−') if d else '0' for d in grid], fontsize=6.0, rotation=90)
        ax.set_xlim(-0.4, len(grid) - 0.6)
        ax.set_title(title + ('' if prim else (' (HiCMAE sơ bộ)' if lang == 'vi' else ' (HiCMAE preliminary)')),
                     fontsize=7.6, color=M.INK2, loc='left')
        ax.grid(axis='x', visible=False)
        ax.set_xlabel(N['delay'], fontsize=7.0)
    axes[0].set_ylim(min(lows) - 0.5, 1.6)
    axes[0].set_ylabel(M.T[lang]['duar'], fontsize=7.2)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper center', ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout(w_pad=0.8)
    M.save(fig, 'fig12_offset_models', lang)


def tables_mma():
    sg, f = M.sg, M.f
    cols = [('CREMAD', 'none'), ('CREMAD', 'scr'), ('MAFW', 'none'), ('MAFW', 'scr')]
    head = ['MMA-DFER · CREMA-D', 'Router OT · CREMA-D', 'MMA-DFER · MAFW', 'Router OT · MAFW']
    L = ['\\begin{tabular}{l rrrr}', '\\toprule', 'Nhánh & ' + ' & '.join(head) + ' \\\\', '\\midrule']
    rows = []
    for arm, vi, _ in MMA_ROWS:
        cells = []
        for ds, tag in cols:
            r = MMA_S[ds]['models'][tag]['arms'].get(arm)
            cells.append('---' if r is None else f"{sg(r['d_uar'])} ({f(r['changed_pct'], 1)}\\%)")
            if r is not None:
                rows.append([arm, ds, tag, r['d_uar'], *r['d_uar_ci'], r['changed_pct']])
        L.append(f'{vi} & ' + ' & '.join(cells) + ' \\\\')
    tol = []
    for ds, tag in cols:
        t = MMA_S[ds]['models'][tag]['D1_tolerance_ms']
        tol.append(' / '.join(('$\\geq$' if t[k] == 640 else '') + str(t[k]) for k in ('late', 'early')))
    L += ['\\midrule',
          'UAR gốc & ' + ' & '.join(f(MMA_S[ds]['models'][tag]['base_uar']) for ds, tag in cols) + ' \\\\',
          'Ngưỡng chịu lệch muộn / sớm (ms) & ' + ' & '.join(tol) + ' \\\\', '\\bottomrule', '\\end{tabular}']
    _wt('tab16_mma', L)
    _wc('mma', ['arm', 'dataset', 'model', 'd_uar', 'ci_lo', 'ci_hi', 'changed_pct'], rows)


def numbers_mma():
    N = {}
    f, sg, ci = M.f, M.sg, M.ci
    for ds, dk in (('CREMAD', 'C'), ('MAFW', 'M')):
        S = MMA_S[ds]
        for tag, tk in (('none', 'N'), ('scr', 'O')):
            Mo = S['models'][tag]
            N[f'Mma{dk}{tk}BaseUAR'] = f(Mo['base_uar'])
            N[f'Mma{dk}{tk}TolLate'] = str(Mo['D1_tolerance_ms']['late'])
            N[f'Mma{dk}{tk}TolEarly'] = str(Mo['D1_tolerance_ms']['early'])
            for arm, an in (('zero_nat', 'ZeroNat'), ('zero_nat_a2v', 'ZeroNatAtoV'), ('zero_nat_v2a', 'ZeroNatVtoA'),
                            ('zero_late', 'ZeroLate'), ('zero_route', 'ZeroRoute'), ('zero_all', 'ZeroAll'),
                            ('perm_a', 'PermAudio'), ('silence', 'Silence'), ('delay-640', 'DelayEarlySixForty'),
                            ('delay+640', 'DelayLateSixForty'), ('delay-160', 'DelayEarlyOneSixty'),
                            ('delay+160', 'DelayLateOneSixty')):
                r = Mo['arms'].get(arm)
                if r is None:
                    continue
                N[f'Mma{dk}{tk}{an}DUAR'] = sg(r['d_uar'])
                N[f'Mma{dk}{tk}{an}DUARCI'] = ci(r['d_uar_ci'])
                N[f'Mma{dk}{tk}{an}Changed'] = f(r['changed_pct'])
            dc = [v for k, v in Mo['delay_curve'].items()]
            N[f'Mma{dk}{tk}DelayMin'], N[f'Mma{dk}{tk}DelayMax'] = sg(min(dc)), sg(max(dc))
        R3 = S['readings']
        N[f'Mma{dk}MThree'] = R3['M3']
        dv = [v['value'] for v in R3['M3_delta'].values()]
        lo = [v['ci'][0] for v in R3['M3_delta'].values()]
        hi = [v['ci'][1] for v in R3['M3_delta'].values()]
        N[f'Mma{dk}DeltaMin'], N[f'Mma{dk}DeltaMax'] = sg(min(dv)), sg(max(dv))
        N[f'Mma{dk}DeltaCILow'], N[f'Mma{dk}DeltaCIHigh'] = sg(min(lo)), sg(max(hi))
        ag = R3['zero_route_agree_with_p2ot_zero_pct']
        N[f'Mma{dk}RouteAgreeMin'] = f(min(ag)) if ag else 'n/a'
    return N


# ---------------------------------------------------------------- stage 0b: the pre-registered visual repair screen
S0B_DIR = R('results', 's0b')
S0B = load(S0B_DIR, 'summary.json') if os.path.exists(os.path.join(S0B_DIR, 'summary.json')) else None
S0B_ROWS = [('base_b12', '(i) Class token only'), ('framewise_b12', '(ii) Framewise patch adapter (control)'),
            ('motionpool_b12', '(iii) Motion-weighted pooling (candidate)'),
            ('mixer_b12', '(iv) Temporal mixer (exploratory)')]
S0B_ROWS9 = [('framewise_b9', '(ii) Framewise patch adapter'), ('motionpool_b9', '(iii) Motion-weighted pooling')]


def s0b_diff(a, b):
    """UAR(a) - UAR(b) from the summary; the reverse of a stored pair is negated, with the interval mirrored."""
    D = S0B['diffs']
    if f'{a} - {b}' in D:
        d = D[f'{a} - {b}']
        return d['value'], d['ci'], d['changed_pct']
    d = D[f'{b} - {a}']
    return -d['value'], [-d['ci'][1], -d['ci'][0]], d['changed_pct']


def tables_s0b():
    sg, ci, f = M.sg, M.ci, M.f
    runs = S0B['runs']

    def rows(items, ref):
        out = []
        for k, lab in items:
            r = runs[k]
            if k == ref:
                dd = '---'
            else:
                v, c, _ = s0b_diff(k, ref)
                dd = f'{sg(v)} {ci(c)}'
            out.append(f"\\quad {lab} & {r['adapter_params']:,} & {f(r['uar_last'])} & {dd} & "
                       f"{f(r['spread_f'], 3)} \\\\")
        return out
    L = ['\\begin{tabular}{l rrrr}', '\\toprule',
         'Configuration & Adapter parameters & UAR & $\\Delta$UAR vs (ii) [90\\% CI] & Within-clip spread \\\\',
         '\\midrule', '\\multicolumn{5}{l}{\\textit{Block 12 (after the final norm)}} \\\\']
    L += rows(S0B_ROWS, 'framewise_b12')
    L += ['\\multicolumn{5}{l}{\\textit{Block 9 (exploratory)}} \\\\']
    L += rows(S0B_ROWS9, 'framewise_b9')
    L += ['\\bottomrule', '\\end{tabular}']
    _wt('en_repair', L)


def numbers_s0b():
    f, sg, ci = M.f, M.sg, M.ci
    runs, N = S0B['runs'], {}
    for k, key in (('base_b12', 'Base'), ('framewise_b12', 'Frame'), ('motionpool_b12', 'Motion'),
                   ('mixer_b12', 'Mixer'), ('framewise_b9', 'FrameNine'), ('motionpool_b9', 'MotionNine')):
        N[f'Sb{key}UAR'] = f(runs[k]['uar_last'])
        N[f'Sb{key}Spread'] = f(runs[k]['spread_f'], 3)
    N['SbSpreadCls'] = f(runs['base_b12']['spread_cls'], 3)
    N['SbParamsFrame'] = f"{runs['framewise_b12']['adapter_params']:,}"
    N['SbParamsMotion'] = f"{runs['motionpool_b12']['adapter_params']:,}"
    for a, b, key in (('motionpool_b12', 'framewise_b12', 'MotionMinusFrame'),
                      ('framewise_b12', 'base_b12', 'FrameMinusBase'),
                      ('motionpool_b12', 'base_b12', 'MotionMinusBase'),
                      ('mixer_b12', 'framewise_b12', 'MixerMinusFrame'),
                      ('mixer_b12', 'base_b12', 'MixerMinusBase'),
                      ('motionpool_b9', 'framewise_b9', 'NineMotionMinusFrame')):
        v, c, ch = s0b_diff(a, b)
        N[f'Sb{key}'], N[f'Sb{key}CI'], N[f'Sb{key}Changed'] = sg(v), ci(c), f(ch, 1)
    mc = S0B['mechanism_check']
    N['SbSuppressDUAR'] = sg(mc['suppress_d_uar'])
    N['SbSuppressChanged'] = f(mc['suppress_changed_pct'], 1)
    N['SbEntropy'], N['SbEntropyUniform'] = f(mc['weight_entropy']), f(mc['weight_entropy_uniform'])
    N['SbNEval'] = str(S0B['n_eval'])
    N['SbVerdict'] = S0B['decision']['verdict']
    return N


# ---------------------------------------------------------------- English tables for the paper draft
def tables_paper():
    sg, ci, f = M.sg, M.ci, M.f
    E = {k: v.replace('α', '$\\alpha$') for k, v in M.NT['en']['arm'].items()}
    # 0. reproduction per fold: CREMA-D (OpenFace faces) and MAFW (own crops)
    L = ['\\begin{tabular}{l rrr rrr}', '\\toprule',
         ' & \\multicolumn{3}{c}{CREMA-D (OpenFace faces)} & \\multicolumn{3}{c}{MAFW (own face crops)} \\\\',
         '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}',
         'Fold & Ours & Released & Gap & Ours & Released & Gap \\\\', '\\midrule']
    for k in range(1, 6):
        c, m = M.P1F[k]['arms']['base']['uar'], MAF_EVAL_F[k]['arms']['base']['uar']
        L.append(f'{k} & {f(c)} & {f(M.README_UAR[k])} & {sg(c - M.README_UAR[k])} & '
                 f'{f(m)} & {f(MAFW_README[k])} & {sg(m - MAFW_README[k])} \\\\')
    c, m = M.P1S['stage0']['pooled_uar'], MAF_EVAL['stage0']['pooled_uar']
    L += ['\\midrule', f'Pooled & {f(c)} & {f(M.README_TOTAL[0])} & {sg(c - M.README_TOTAL[0])} & '
          f'{f(m)} & {f(MAFW_TOTAL[0])} & {sg(m - MAFW_TOTAL[0])} \\\\',
          f"Stage 0 & \\multicolumn{{3}}{{c}}{{{'pass' if M.P1S['stage0']['pass'] else 'fail'}}} & "
          f"\\multicolumn{{3}}{{c}}{{{'pass' if MAF_EVAL['stage0']['pass'] else 'fail'}}} \\\\",
          '\\bottomrule', '\\end{tabular}']
    _wt('en_repro', L)
    # 1. intervention ladder: CREMA-D (reproduced) and MAFW (preliminary)
    L = ['\\begin{tabular}{l rr rr}', '\\toprule',
         ' & \\multicolumn{2}{c}{CREMA-D (reproduced)} & \\multicolumn{2}{c}{MAFW (preliminary)} \\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
         'Arm & $\\Delta$UAR [90\\% CI] & Changed \\% & $\\Delta$UAR [90\\% CI] & Changed \\% \\\\', '\\midrule']
    groups = M.NULL_GROUPS + [('video', ('perm_v_fus', 'perm_v'))]
    gname = {**M.NT['en']['group'], 'video': 'Video replaced'}
    for g, arms in groups:
        L.append(f'\\multicolumn{{5}}{{l}}{{\\textit{{{gname[g]}}}}} \\\\')
        for arm in arms:
            c, m = M.NS['arms'][arm], MAF_NS['arms'][arm]
            L.append(f"\\quad {E[arm]} & {sg(c['d_uar'])} {ci(c['d_uar_ci'])} & {f(c['changed_pct'], 1)} & "
                     f"{sg(m['d_uar'])} {ci(m['d_uar_ci'])} & {f(m['changed_pct'], 1)} \\\\")
    L += ['\\midrule', f"Base UAR & \\multicolumn{{2}}{{c}}{{{f(M.NS['base_uar'])}}} & "
          f"\\multicolumn{{2}}{{c}}{{{f(MAF_NS['base_uar'])}}} \\\\", '\\bottomrule', '\\end{tabular}']
    _wt('en_ladder', L)

    # 2. robustness to the face crops
    rows = [('Independent coupling (P1)', 'P1S', 'indep'), ('Knockout (P1)', 'P1S', 'zero'),
            ('Token permutation', 'NS', 'rowperm'), ('Time-only permutation', 'NS', 'rowperm_t'),
            ('Space-only permutation', 'FS', 'rowperm_s'), ('Uniform attention', 'NS', 'uniform'),
            ('Training-mean map (PAPA)', 'NS', 'papa'), ('EMAP', 'NS', 'emap'),
            ('Other clip audio', 'NS', 'perm_a'), ('Offset $-160$ ms (sound early)', 'FS', 'delay-160'),
            ('Offset $+160$ ms (sound late)', 'FS', 'delay+160'), ('Silence', 'FS', 'silence')]
    new = {'P1S': M.P1S, 'NS': M.NS, 'FS': M.FS}
    L = ['\\begin{tabular}{l rr}', '\\toprule', '$\\Delta$UAR [90\\% CI] & Own crops & OpenFace \\\\', '\\midrule',
         f"Base UAR & {f(OLD['P1S']['stage0']['pooled_uar'])} & {f(M.P1S['stage0']['pooled_uar'])} \\\\", '\\midrule']
    for lab, src, arm in rows:
        a, b = OLD[src]['arms'][arm], new[src]['arms'][arm]
        L.append(f"{lab} & {sg(a['d_uar'])} {ci(a['d_uar_ci'])} & {sg(b['d_uar'])} {ci(b['d_uar_ci'])} \\\\")
    ia, ib = OLD['NS']['readings']['factorial_interaction'], M.NS['readings']['factorial_interaction']
    da, db = OLD['DCS']['D']['-160'], M.DCS['D']['-160']
    ta, tb = OLD['FS']['readings']['D1_tolerance_ms'], M.FS['readings']['D1_tolerance_ms']
    L += [f"Interaction $I$ & {sg(ia['value'])} {ci(ia['ci'])} & {sg(ib['value'])} {ci(ib['ci'])} \\\\",
          f"$D_d$, 169 ms early & {sg(da['value'])} {ci(da['ci'])} & {sg(db['value'])} {ci(db['ci'])} \\\\",
          '\\midrule', f"Tolerance late / early (ms) & {ta['late']} / {ta['early']} & {tb['late']} / {tb['early']} \\\\",
          '\\bottomrule', '\\end{tabular}']
    _wt('en_robustness', L)

    # 3. MMA-DFER and the OT router: audio-path removal and offsets
    if len(MMA_S) == 2:
        cols = [('CREMAD', 'none'), ('CREMAD', 'scr'), ('MAFW', 'none'), ('MAFW', 'scr')]
        L = ['\\begin{tabular}{l rrrr}', '\\toprule',
             ' & \\multicolumn{2}{c}{CREMA-D} & \\multicolumn{2}{c}{MAFW} \\\\',
             '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
             '$\\Delta$UAR (changed \\%) & MMA-DFER & OT router & MMA-DFER & OT router \\\\', '\\midrule']
        for arm, _, en in MMA_ROWS:
            cells = []
            for ds, tag in cols:
                r = MMA_S[ds]['models'][tag]['arms'].get(arm)
                cells.append('---' if r is None else f"{sg(r['d_uar'])} ({f(r['changed_pct'], 1)})")
            L.append(f'{en} & ' + ' & '.join(cells) + ' \\\\')
        tol = []
        for ds, tag in cols:
            t = MMA_S[ds]['models'][tag]['D1_tolerance_ms']
            tol.append(' / '.join(('$\\geq$' if t[k] == 640 else '') + str(t[k]) for k in ('late', 'early')))
        L += ['\\midrule',
              'Base UAR & ' + ' & '.join(f(MMA_S[ds]['models'][tag]['base_uar']) for ds, tag in cols) + ' \\\\',
              'Tolerance late / early (ms) & ' + ' & '.join(tol) + ' \\\\', '\\bottomrule', '\\end{tabular}']
        _wt('en_mma', L)


def main():
    os.makedirs(OUT, exist_ok=True)
    hm = mafw_assets()            # before M.numbers(), which rewrites numbers.tex
    M.tables()
    n1 = M.numbers()
    tables_p2()
    extra = {**hm, **tab_robustness()}
    if len(MMA_S) == 2:
        tables_mma()
        extra.update(numbers_mma())
    if S0B is not None:
        tables_s0b()
        extra.update(numbers_s0b())
    n2 = numbers_p2(extra)
    tables_paper()
    for lang in ('vi', 'en'):
        L = M.T[lang]
        M.fig_p1_repro(L, lang)
        M.fig_p1_arms(L, lang)
        M.fig_nulls(L, lang)
        M.fig_trace(L, lang)
        M.fig_followup(L, lang)
        M.fig_desync_control(L, lang)
        fig_contrast(lang)
        if len(MMA_S) == 2:
            fig_offset_models(lang)
    print(f'{len(n1) + len(n2)} macros, tables and figures written under {OUT}')


if __name__ == '__main__':
    main()
