#!/usr/bin/env python
"""P2T report assets (does DFER use the time axis?): stage 0, follow-up (a) and the training control (b).

Reads results/p2t/<pair>/summary.json, results/p2t/breakdown.json and results/p2t_train/<ds>/compare.json and
writes report-assets/p2t/{figures/vi/*.{pdf,png}, tables/*.tex, numbers.tex}. Style and palette follow
make_assets.py (reference categorical slots, print chrome).

    python make_assets_p2t.py
"""

import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'p2t')
P2T, TRN = os.path.join(ROOT, 'results', 'p2t'), os.path.join(ROOT, 'results', 'p2t_train')

INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff'
SLOT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
MODEL_COLOR = {'mma': SLOT[0], 'ot': SLOT[1], 'hicmae': SLOT[2]}
CTL_COLOR = SLOT[3]          # MMA-DFER trained on shuffled frames (a new entity, the next free slot)
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 8.5, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': INK2, 'text.color': INK,
    'axes.titlecolor': INK, 'xtick.color': AXIS, 'ytick.color': AXIS,
    'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'grid.linestyle': '-',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False,
})

DATASETS = ('DFEW', 'MAFW', 'CREMAD')
DS_NAME = {'DFEW': 'DFEW', 'MAFW': 'MAFW', 'CREMAD': 'CREMA-D'}
MODEL_NAME = {'mma': 'MMA-DFER', 'ot': 'Router OT', 'hicmae': 'HiCMAE'}
PAIRS = [('hicmae', 'CREMAD'), ('hicmae', 'MAFW'), ('mma', 'DFEW'), ('mma', 'MAFW'), ('mma', 'CREMAD'),
         ('ot', 'DFEW'), ('ot', 'MAFW'), ('ot', 'CREMAD')]
ARM_VI = {'shuffle_s0': 'xáo', 'shuffle_pair': 'xáo cặp', 'reverse': 'đảo ngược', 'hold8': 'giữ 8',
          'hold4': 'giữ 4', 'hold2': 'giữ 2', 'hold1': 'giữ 1', 'static_rand': 'ảnh tĩnh'}
READ_VI = {'uses order': 'dùng thứ tự', 'order-invariant': 'không nhạy', 'inconclusive': 'chưa kết luận',
           'uses direction': 'dùng chiều', 'direction-invariant': 'không nhạy', 'none': 'không có',
           'order not needed': 'không cần thứ tự', 'order used for accuracy': 'thứ tự giúp độ chính xác'}


def load(*p):
    with open(os.path.join(*p)) as fh:
        return json.load(fh)


S = {(m, d): load(P2T, f'{m}_{d.lower()}', 'summary.json') for m, d in PAIRS}
BD = load(P2T, 'breakdown.json')
TR = {d: load(TRN, d.lower(), 'compare.json') for d in DATASETS}


def pair_dir(m, d):
    return os.path.join(P2T, f'{m}_{d.lower()}')


def calib(ds):
    """OT (scr) against MMA-DFER (none) on the same folds: the run-to-run scale the clip bootstrap lacks."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('cmp', os.path.join(ROOT, 'scripts', 'p2t_train_compare.py'))
    cmp = importlib.util.module_from_spec(spec)
    import sys
    argv, sys.argv = sys.argv, ['x']
    spec.loader.exec_module(cmp)
    sys.argv = argv
    tmp = os.path.join(OUT, 'data', f'calib_{ds.lower()}.json')
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        cmp.main(pair_dir('mma', ds), pair_dir('ot', ds), f'calib {ds}', tmp)
    return load(tmp)


CAL = {d: calib(d) for d in DATASETS}


def save(fig, name):
    d = os.path.join(OUT, 'figures', 'vi')
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name + '.pdf'), bbox_inches='tight', pad_inches=0.03)
    fig.savefig(os.path.join(d, name + '.png'), dpi=220, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


def arm(m, d, k):
    a = S[(m, d)]['arms'][k]
    return a['d_uar'], a['d_uar_ci']


def err(v, ci):
    return np.array([[v - ci[0]], [ci[1] - v]])


# ---------------------------------------------------------------- figures
def fig_budget():
    """dUAR against the number of distinct frames, MMA-DFER and the OT router (HiCMAE in the table: its scale)."""
    xs = ['16', '8', '4', '2', '1', '1 ngẫu\nnhiên']
    keys = [None, 'hold8', 'hold4', 'hold2', 'hold1', 'static_rand']
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
    for ax, d in zip(axes, DATASETS):
        ax.axhline(-1, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        for j, m in enumerate(('mma', 'ot')):
            off = (j - 0.5) * 0.16
            v = [0.0] + [arm(m, d, k)[0] for k in keys[1:]]
            ci = [(0.0, 0.0)] + [arm(m, d, k)[1] for k in keys[1:]]
            x = np.arange(len(xs)) + off
            ax.plot(x[:5], v[:5], color=MODEL_COLOR[m], lw=1.5, zorder=3)
            for i in range(len(xs)):
                ax.errorbar(x[i], v[i], yerr=err(v[i], ci[i]), color=MODEL_COLOR[m], lw=1.0, capsize=0, zorder=3)
                ax.plot(x[i], v[i], marker='o', ms=4.6, mec=SURF if i < 5 else MODEL_COLOR[m], mew=0.9,
                        mfc=MODEL_COLOR[m] if i < 5 else SURF, ls='none', zorder=4,
                        label=MODEL_NAME[m] if (i == 0 and d == 'DFEW') else None)
        k_m, k_o = S[('mma', d)]['readings']['K_frame_budget'], S[('ot', d)]['readings']['K_frame_budget']
        ax.set_title(f'{DS_NAME[d]}\nK: MMA-DFER {k_m} · router OT {k_o}', loc='left')
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs)
    fig.supxlabel('Số frame khác nhau trong 16 frame đầu vào', fontsize=8.5, color=INK2, y=-0.1)
    axes[0].set_ylabel('Chênh UAR so với gốc (điểm)')
    axes[0].text(5.3, -0.82, 'ngưỡng −1', ha='right', va='bottom', fontsize=7, color=INK2)
    axes[0].legend(loc='lower left')
    save(fig, 'fig1_budget')


def fig_order():
    """Order arms per pair: shuffle, pair shuffle, reverse, with 90% CIs."""
    keys = ['shuffle_s0', 'shuffle_pair', 'reverse']
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    for ax, d in zip(axes, DATASETS):
        models = [m for m in ('mma', 'ot', 'hicmae') if (m, d) in S]
        ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
        ax.axvline(-1, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
        for j, m in enumerate(models):
            off = (j - (len(models) - 1) / 2) * 0.22
            for i, k in enumerate(keys):
                v, ci = arm(m, d, k)
                y = len(keys) - 1 - i + off
                ax.errorbar(v, y, xerr=err(v, ci), color=MODEL_COLOR[m], lw=1.0, zorder=3)
                ax.plot(v, y, marker='o', ms=4.6, mfc=MODEL_COLOR[m], mec=SURF, mew=0.9, ls='none', zorder=4,
                        label=MODEL_NAME[m] if i == 0 else None)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([ARM_VI[k] for k in reversed(keys)])
        ax.set_title(DS_NAME[d], loc='left')
        ax.set_xlabel('Chênh UAR (điểm)')
        ax.grid(axis='y', visible=False)
        lo = min(arm(m, d, k)[1][0] for m in models for k in keys)
        ax.set_xlim(min(lo, -1.6) * 1.08, 0.6)
    h, lab = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, 'fig2_order')


def fig_class():
    """Per-class recall change: MMA-DFER reverse on DFEW and CREMA-D, HiCMAE hold4 on MAFW (collapse)."""
    panels = [('mma_dfew', 'reverse', 'MMA-DFER · DFEW · đảo ngược', 'mma'),
              ('mma_cremad', 'reverse', 'MMA-DFER · CREMA-D · đảo ngược', 'mma'),
              ('hicmae_mafw', 'hold4', 'HiCMAE · MAFW · giữ 4', 'hicmae')]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), gridspec_kw={'width_ratios': [1, 1, 1.25]})
    for ax, (p, k, title, m) in zip(axes, panels):
        pc = BD[p]['per_class']
        names = list(pc)
        v = np.array([pc[c]['arms'][k]['d_recall'] for c in names])
        ci = np.array([pc[c]['arms'][k]['ci'] for c in names])
        flag = np.array([pc[c]['arms'][k]['time_sensitive'] for c in names])
        y = np.arange(len(names))[::-1]
        ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
        ax.axvline(-3, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
        for yi, vi, cii, fl in zip(y, v, ci, flag):
            ax.barh(yi, vi, height=0.62, color=MODEL_COLOR[m], alpha=1.0 if fl else 0.35, zorder=2)
            ax.errorbar(vi, yi, xerr=err(vi, cii), color=INK2, lw=0.8, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f'{c} ({pc[c]["n"]})' for c in names], fontsize=6.8)
        ax.set_title(title, loc='left', fontsize=8)
        ax.set_xlabel('Chênh recall (điểm)')
        ax.grid(axis='y', visible=False)
    fig.tight_layout()
    save(fig, 'fig3_class')


def fig_train():
    """Training control: shuffle-trained minus reference, per fold and pooled, against the OT-vs-none calibration."""
    order = ['DFEW', 'CREMAD', 'MAFW']
    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    ax.axvspan(-3.0, -1.0, color='#f1f0eb', zorder=0)
    ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    ax.axvline(-1, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
    for i, d in enumerate(order):
        y = len(order) - 1 - i
        t, c = TR[d], CAL[d]
        ax.scatter(t['per_fold'], [y + 0.14] * 5, s=14, color=CTL_COLOR, alpha=0.55, lw=0, zorder=3,
                   label='từng fold (train xáo − gốc)' if i == 0 else None)
        ax.errorbar(t['d_uar_train'], y + 0.14, xerr=err(t['d_uar_train'], t['ci']), color=CTL_COLOR, lw=1.4,
                    zorder=4)
        ax.plot(t['d_uar_train'], y + 0.14, marker='D', ms=5.4, mfc=CTL_COLOR, mec=SURF, mew=0.9, ls='none',
                zorder=5, label='gộp 5 fold, KTC 90%' if i == 0 else None)
        ax.text(t['d_uar_train'], y + 0.4, f'{t["d_uar_train"]:+.2f}'.replace('-', '−'), ha='center',
                va='center', fontsize=7, color=INK)
        ax.errorbar(c['d_uar_train'], y - 0.2, xerr=err(c['d_uar_train'], c['ci']), color=MUTED, lw=1.2, zorder=4)
        ax.plot(c['d_uar_train'], y - 0.2, marker='o', ms=4.4, mfc=SURF, mec=MUTED, mew=1.2, ls='none', zorder=5,
                label='hiệu chuẩn: router OT − gốc' if i == 0 else None)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DS_NAME[d] for d in reversed(order)])
    ax.grid(axis='y', visible=False)
    ax.set_xlim(-2.75, 1.45)
    ax.set_ylim(-0.55, 2.75)
    ax.text(-2.7, 2.62, 'vùng "thứ tự giúp ≥ 1 điểm"', fontsize=6.8, color=INK2, va='center')
    ax.set_xlabel('Chênh UAR trên test (điểm): mô hình train với frame bị xáo − mô hình gốc')
    ax.legend(loc='lower center', fontsize=6.8, ncol=3, bbox_to_anchor=(0.5, 1.0))
    save(fig, 'fig4_train')


def fig_ctrl_arms():
    """Test-time arms on the reference and on the shuffle-trained model."""
    keys = ['shuffle_s0', 'shuffle_pair', 'reverse', 'hold8', 'hold4', 'hold2', 'hold1', 'static_rand']
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    for ax, d in zip(axes, DATASETS):
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        ax.axhline(-1, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
        A = TR[d]['arms']
        for j, (mk, col, lab, filled) in enumerate((('ref', MODEL_COLOR['mma'], 'MMA-DFER gốc', True),
                                                    ('ctl', CTL_COLOR, 'train với frame bị xáo', False))):
            x = np.arange(len(keys)) + (j - 0.5) * 0.3
            for i, k in enumerate(keys):
                v, ci = A[k][mk]['d_uar'], A[k][mk]['ci']
                ax.errorbar(x[i], v, yerr=err(v, ci), color=col, lw=1.0, zorder=3)
                ax.plot(x[i], v, marker='o' if filled else 's', ms=4.4, mfc=col if filled else SURF,
                        mec=SURF if filled else col, mew=1.0 if filled else 1.2, ls='none', zorder=4,
                        label=lab if (i == 0 and d == 'DFEW') else None)
        K = TR[d]['K']
        ax.set_title(f'{DS_NAME[d]}   (K: gốc {K["ref"]} → train xáo {K["ctl"]})', loc='left')
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([ARM_VI[k] for k in keys], rotation=45, ha='right')
    axes[0].set_ylabel('Chênh UAR so với gốc (điểm)')
    axes[0].legend(loc='lower left')
    fig.tight_layout()
    save(fig, 'fig5_ctrl_arms')


# ---------------------------------------------------------------- tables and macros
def num(v, d=2):
    s = f'{v:+.{d}f}' if v != 0 else f'{0:.{d}f}'
    return f'${s}$'


def ci_s(ci, d=2):
    return f'$[{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]$'


def write_tex(name, lines):
    os.makedirs(os.path.join(OUT, 'tables'), exist_ok=True)
    with open(os.path.join(OUT, 'tables', name + '.tex'), 'w') as fh:
        fh.write('% generated by make_assets_p2t.py -- do not edit by hand\n' + '\n'.join(lines) + '\n')


def tab_stage0():
    keys = ['shuffle_s0', 'shuffle_pair', 'reverse', 'hold8', 'hold4', 'hold2', 'hold1', 'static_rand']
    L = [r'\begin{tabular}{l r ' + 'r' * len(keys) + ' l l c}', r'\toprule',
         r'Mô hình · dữ liệu & Gốc & ' + ' & '.join(ARM_VI[k] for k in keys) + r' & O (thứ tự) & R (chiều) & K \\',
         r'\midrule']
    for m, d in PAIRS:
        s = S[(m, d)]
        r = s['readings']
        cells = []
        for k in keys:
            v, ci = arm(m, d, k)
            cells.append(f'${v:+.2f}$')
        L.append(f'{MODEL_NAME[m]} · {DS_NAME[d]} & {s["base_uar"]:.2f} & ' + ' & '.join(cells) +
                 f' & {READ_VI[r["O_order"]]} & {READ_VI[r["R_direction"]]} & '
                 f'{READ_VI.get(str(r["K_frame_budget"]), r["K_frame_budget"])} \\\\')
        if (m, d) == ('hicmae', 'MAFW') or (m, d) == ('mma', 'CREMAD'):
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}']
    write_tex('tab1_stage0', L)


def tab_train():
    L = [r'\begin{tabular}{l r r c l c c c}', r'\toprule',
         r'Dữ liệu & Gốc & Train xáo & Chênh [KTC 90\%] & Từng fold & Kết luận & K & Dự đoán đổi \\', r'\midrule']
    for d in ('DFEW', 'CREMAD', 'MAFW'):
        t = TR[d]
        folds = ' / '.join(f'{x:+.2f}' for x in t['per_fold']).replace('-', '−')
        L.append(f'{DS_NAME[d]} & {t["ref_uar"]:.2f} & {t["ctl_uar"]:.2f} & {num(t["d_uar_train"])} '
                 f'{ci_s(t["ci"])} & {{\\footnotesize {folds}}} & {READ_VI[t["reading"]]} & '
                 f'{t["K"]["ref"]} $\\to$ {t["K"]["ctl"]} & {t["changed_pct"]:.1f}\\% \\\\')
    L.append(r'\midrule')
    for d in ('DFEW', 'CREMAD', 'MAFW'):
        c = CAL[d]
        folds = ' / '.join(f'{x:+.2f}' for x in c['per_fold']).replace('-', '−')
        L.append(f'\\textit{{hiệu chuẩn {DS_NAME[d]}}} & {c["ref_uar"]:.2f} & {c["ctl_uar"]:.2f} & '
                 f'{num(c["d_uar_train"])} {ci_s(c["ci"])} & {{\\footnotesize {folds}}} & -- & -- & '
                 f'{c["changed_pct"]:.1f}\\% \\\\')
    L += [r'\bottomrule', r'\end{tabular}']
    write_tex('tab2_train', L)


def numbers():
    N = {}
    for d in DATASETS:
        t, c, tag = TR[d], CAL[d], {'DFEW': 'Dfew', 'MAFW': 'Mafw', 'CREMAD': 'Cremad'}[d]
        N[f'Tr{tag}Ref'] = f'{t["ref_uar"]:.2f}'
        N[f'Tr{tag}Ctl'] = f'{t["ctl_uar"]:.2f}'
        N[f'Tr{tag}D'] = num(t['d_uar_train'])
        N[f'Tr{tag}CI'] = ci_s(t['ci'])
        N[f'Tr{tag}SD'] = f'{np.std(t["per_fold"], ddof=1):.2f}'
        N[f'Cal{tag}SD'] = f'{np.std(c["per_fold"], ddof=1):.2f}'
        N[f'Cal{tag}D'] = num(c['d_uar_train'])
        for k in ('shuffle_s0', 'reverse', 'hold4', 'hold1', 'static_rand'):
            kk = {'shuffle_s0': 'Shuf', 'reverse': 'Rev', 'hold4': 'Hfour', 'hold1': 'Hone',
                  'static_rand': 'Stat'}[k]
            N[f'Tr{tag}Ref{kk}'] = num(t['arms'][k]['ref']['d_uar'])
            N[f'Tr{tag}Ctl{kk}'] = num(t['arms'][k]['ctl']['d_uar'])
    for m, d in PAIRS:
        tag = {'hicmae': 'Hi', 'mma': 'Mma', 'ot': 'Ot'}[m] + {'DFEW': 'Dfew', 'MAFW': 'Mafw', 'CREMAD': 'Cremad'}[d]
        s = S[(m, d)]
        N[f'{tag}Base'] = f'{s["base_uar"]:.2f}'
        for k in ('shuffle_s0', 'shuffle_pair', 'reverse', 'hold4', 'hold1', 'static_rand'):
            kk = {'shuffle_s0': 'Shuf', 'shuffle_pair': 'Pair', 'reverse': 'Rev', 'hold4': 'Hfour',
                  'hold1': 'Hone', 'static_rand': 'Stat'}[k]
            N[f'{tag}{kk}'] = num(s['arms'][k]['d_uar'])
            N[f'{tag}{kk}CI'] = ci_s(s['arms'][k]['d_uar_ci'])
        N[f'{tag}K'] = READ_VI.get(str(s['readings']['K_frame_budget']), str(s['readings']['K_frame_budget']))
    for d, tag in (('CREMAD', 'Cremad'), ('MAFW', 'Mafw')):
        r = S[('hicmae', d)]['readings']
        N[f'Hi{tag}M'] = num(r['M_local_motion']['value'])
        N[f'Hi{tag}MCI'] = ci_s(r['M_local_motion']['ci'])
        N[f'Hi{tag}S'] = num(r['S_av_sync']['value'])
        N[f'Hi{tag}SCI'] = ci_s(r['S_av_sync']['ci'])
    with open(os.path.join(OUT, 'numbers.tex'), 'w') as fh:
        fh.write('% generated by make_assets_p2t.py -- do not edit by hand\n')
        for k, v in N.items():
            assert k.isalpha(), k
            fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    return N


if __name__ == '__main__':
    fig_budget()
    fig_order()
    fig_class()
    fig_train()
    fig_ctrl_arms()
    tab_stage0()
    tab_train()
    N = numbers()
    print(f'{len(N)} macros; figures and tables in {OUT}')
