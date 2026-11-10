#!/usr/bin/env python
"""Build the LaTeX report assets from the stage-0a, probe and P1 result files.

Reads only:  ../results/stage0a/*.json   ../p1/results/{cremad,cremad_nulls,cremad_followup,cremad_desync}/*.json
Writes:      figures/{vi,en}/*.pdf|png, tables/*.tex, data/*.csv, numbers.tex

Every number in a figure, table or macro is computed here from those JSON files, except the
HiCMAE README values (constants below, from github.com/sunlicai/HiCMAE, commit cdd28d4) and the
two paperdiag values noted in PAPERDIAG.

    python make_assets.py
"""

import csv
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R0 = os.path.join(ROOT, 'results', 'stage0a')
P1R = os.path.join(ROOT, 'p1', 'results', 'cremad')

README_UAR = {1: 86.66, 2: 83.27, 3: 87.23, 4: 83.70, 5: 83.88}
README_WAR = {1: 86.67, 2: 83.20, 3: 87.19, 4: 83.79, 5: 83.79}
README_TOTAL = (84.94, 84.91)
# from ~/AVDFEROT/paperdiag (2026-10-18/20), not recomputed here
PAPERDIAG = {'F1_max_mafw_col_b9': 0.0219, 'ABV_col_cls_max_b1_9': 0.0786}

RUNS = [('MMA_DFEW_scr', 'MMA-DFER · DFEW'), ('MMA_MAFW_scr', 'MMA-DFER · MAFW'),
        ('MMA_CREMAD_scr', 'MMA-DFER · CREMA-D'), ('ABV_DFEW_scr', 'ABVnet · DFEW')]
LEVELS = ('cls', 'mean', 'r4', 'patch')
ARMS = ('indep', 'indep_v2a', 'indep_a2v', 'indep_b1', 'indep_b2', 'zero')

# palette: reference categorical slots 1-4 (adjacent-pair validated), ink and chrome for print
INK, INK2, MUTED, GRID, AXIS, SURF, BAND = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff', '#f1f0eb'
SLOT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
LV_COLOR = dict(zip(LEVELS, SLOT))

T = {
    'vi': {
        'lv': {'cls': 'CLS', 'mean': 'TB patch', 'r4': 'Vùng 4×4', 'patch': 'Từng patch'},
        'block': 'Block encoder', 'spread': 'Độ biến động trong clip\n(‖xⱼ − x̄‖ / ‖x̄‖, thang log)',
        'thr': 'ngưỡng 0.044 (2 × 2.2%)', 'f1': 'phạm vi của F1 (block 1–9)',
        'ratio': 'Tỉ số biến động\ntrong clip / giữa clip', 'uar': 'UAR trên val10 (%)',
        'feat': {'S': 'S: trung bình clip', 'D': 'D: biến động trong clip', 'SD': 'S + D'},
        'chance': 'ngẫu nhiên 14.29', 'blk': 'Block {}',
        'diff': 'Chênh UAR (điểm), KTC 90%', 'margin': 'ngưỡng ±2',
        'tests': ['H3a: D(4×4) − ngẫu nhiên', 'H3b: S+D(4×4) − S(4×4)', 'H2: D(4×4) − D(CLS)',
                  'H2: D(4×4) − D(TB patch)'],
        'pass': 'đạt', 'fail': 'không đạt',
        'fold': 'Fold {}', 'pooled': 'Gộp 5 fold', 'readme': 'README tác giả', 'ours': 'Chạy lại (crop của nhóm)',
        'uar_axis': 'UAR (%)',
        'arms': {'indep': 'Ghép nối độc lập (tất cả)', 'indep_v2a': '   chỉ video ← audio',
                 'indep_a2v': '   chỉ audio ← video', 'indep_b1': '   chỉ block fusion 1',
                 'indep_b2': '   chỉ block fusion 2', 'zero': 'Bỏ cross-attention (tham chiếu)'},
        'changed': 'Dự đoán thay đổi so với gốc (%)', 'duar': 'Chênh UAR so với gốc (điểm)',
        'thr15': 'ngưỡng 1% và 5%', 'thrm1': 'ngưỡng −1',
    },
    'en': {
        'lv': {'cls': 'CLS', 'mean': 'patch mean', 'r4': '4×4 regions', 'patch': 'single patch'},
        'block': 'Encoder block', 'spread': 'Within-clip spread\n(‖xⱼ − x̄‖ / ‖x̄‖, log scale)',
        'thr': 'threshold 0.044 (2 × 2.2%)', 'f1': 'scope of F1 (blocks 1–9)',
        'ratio': 'Within-clip /\nbetween-clip ratio', 'uar': 'val10 UAR (%)',
        'feat': {'S': 'S: clip mean', 'D': 'D: within-clip dynamics', 'SD': 'S + D'},
        'chance': 'chance 14.29', 'blk': 'Block {}',
        'diff': 'UAR difference (points), 90% CI', 'margin': 'margin ±2',
        'tests': ['H3a: D(4×4) − chance', 'H3b: S+D(4×4) − S(4×4)', 'H2: D(4×4) − D(CLS)',
                  'H2: D(4×4) − D(patch mean)'],
        'pass': 'supported', 'fail': 'not supported',
        'fold': 'Fold {}', 'pooled': 'Pooled', 'readme': 'Author README', 'ours': 'Re-run (our face crops)',
        'uar_axis': 'UAR (%)',
        'arms': {'indep': 'Independent coupling (all)', 'indep_v2a': '   video ← audio only',
                 'indep_a2v': '   audio ← video only', 'indep_b1': '   fusion block 1 only',
                 'indep_b2': '   fusion block 2 only', 'zero': 'Cross-attention removed (reference)'},
        'changed': 'Predictions changed vs base (%)', 'duar': 'UAR change vs base (points)',
        'thr15': 'thresholds 1% and 5%', 'thrm1': 'threshold −1',
    },
}

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
LINE = dict(lw=1.5, marker='o', ms=4.2, mec=SURF, mew=0.9, zorder=3)


def load(*p):
    with open(os.path.join(*p)) as f:
        return json.load(f)


def save(fig, name, lang):
    d = os.path.join(HERE, 'figures', lang)
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name + '.pdf'), bbox_inches='tight', pad_inches=0.03)
    fig.savefig(os.path.join(d, name + '.png'), dpi=220, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


# ---------------------------------------------------------------- data
S0 = {k: load(R0, k + '.json')['blocks'] for k, _ in RUNS}
PROBE = load(R0, 'probe_MMA_DFEW_scr_summary.json')
P1S = load(P1R, 'summary.json')
P1F = {f: load(P1R, f'fold{f}.json') for f in range(1, 6)}
NUL = os.path.join(ROOT, 'p1', 'results', 'cremad_nulls')
FUP = os.path.join(ROOT, 'p1', 'results', 'cremad_followup')
NS = load(NUL, 'summary.json')
FS = load(FUP, 'summary.json')
DCS = load(ROOT, 'p1', 'results', 'cremad_desync', 'summary.json')
DC_ARMS = (('aud', 'Chỉ dịch audio', 'Audio moved only'), ('vid', 'Chỉ dịch video', 'Video moved only'),
           ('both', 'Dịch cả hai (vẫn khớp)', 'Both moved (aligned)'))

# P1-nulls arms, grouped (figure 7, table 9)
NULL_GROUPS = [
    ('remove', ('zero', 'devonly')),
    ('collapse', ('indep', 'shrink25', 'shrink50', 'shrink75')),
    ('misassign', ('rowperm', 'rowperm_s1', 'rowperm_s2', 'rowperm_t')),
    ('static', ('uniform', 'papa')),
    ('input', ('emap', 'perm_a_fus', 'perm_a')),
]
NULL_COLOR = {'remove': INK2, 'collapse': SLOT[0], 'misassign': SLOT[1], 'static': SLOT[2], 'input': SLOT[3]}
TRACE_KEYS = ('b1_cross', 'b1_self', 'b1_mlp', 'b2_cross', 'b2_self', 'b2_mlp', 'norm')
TRACE_ARMS = ('indep', 'rowperm', 'shrink50')
NT = {
    'vi': {
        'group': {'remove': 'Loại bỏ', 'collapse': 'Gộp về thông điệp chung', 'misassign': 'Trao nhầm token',
                  'static': 'Attention cố định', 'input': 'Mức đầu vào'},
        'arm': {'zero': 'Bỏ cả thông điệp (zero)', 'devonly': 'Chỉ giữ phần riêng (devonly)',
                'indep': 'Ghép nối độc lập, α = 0', 'shrink25': 'Co phần riêng, α = 0.25',
                'shrink50': 'Co phần riêng, α = 0.5', 'shrink75': 'Co phần riêng, α = 0.75',
                'rowperm': 'Hoán vị token (seed 0)', 'rowperm_s1': 'Hoán vị token (seed 1)',
                'rowperm_s2': 'Hoán vị token (seed 2)', 'rowperm_t': 'Chỉ hoán vị thời gian',
                'rowperm_s': 'Chỉ hoán vị không gian',
                'uniform': 'Attention đều', 'papa': 'Map TB tập train (PAPA)',
                'emap': 'EMAP (bỏ tương tác)', 'perm_a_fus': 'Audio clip khác, chỉ vào fusion',
                'perm_a': 'Audio clip khác (perceptual)', 'perm_v': 'Video clip khác',
                'perm_v_fus': 'Video clip khác, chỉ vào fusion'},
        'trace_keys': ['B1\ncross', 'B1\nself', 'B1\nMLP', 'B2\ncross', 'B2\nself', 'B2\nMLP', 'norm\ncuối'],
        'trace_arm': {'indep': 'Ghép nối độc lập', 'rowperm': 'Hoán vị token', 'shrink50': 'Co α = 0.5'},
        'trace_mean': 'Lệch tương đối của\ntrung bình token', 'trace_tok': 'Lệch tương đối\ncủa từng token',
        'f3': 'F3: bằng 0', 'video': 'Luồng video', 'audio': 'Luồng audio',
        'delay': 'Độ lệch audio so với hình (ms)', 'early': '← tiếng sớm', 'late': 'tiếng muộn →',
        'full': 'Lệch cả audio', 'fus': 'Lệch chỉ ở đầu vào fusion', 'silence': 'mất tiếng: {}',
        'itu': 'ngưỡng nhận biết\nITU-R BT.1359', 'axis_perm': 'Chênh UAR (điểm), KTC 90%',
    },
    'en': {
        'group': {'remove': 'Removal', 'collapse': 'Collapse to the clip message', 'misassign': 'Misassignment',
                  'static': 'Static attention', 'input': 'Input level'},
        'arm': {'zero': 'Remove the message (zero)', 'devonly': 'Token part only (devonly)',
                'indep': 'Independent coupling, α = 0', 'shrink25': 'Shrink token part, α = 0.25',
                'shrink50': 'Shrink token part, α = 0.5', 'shrink75': 'Shrink token part, α = 0.75',
                'rowperm': 'Token permutation (seed 0)', 'rowperm_s1': 'Token permutation (seed 1)',
                'rowperm_s2': 'Token permutation (seed 2)', 'rowperm_t': 'Time-only permutation',
                'rowperm_s': 'Space-only permutation',
                'uniform': 'Uniform attention', 'papa': 'Training-mean map (PAPA)',
                'emap': 'EMAP (no interactions)', 'perm_a_fus': 'Other clip audio, fusion only',
                'perm_a': 'Other clip audio (perceptual)', 'perm_v': 'Other clip video',
                'perm_v_fus': 'Other clip video, fusion only'},
        'trace_keys': ['B1\ncross', 'B1\nself', 'B1\nMLP', 'B2\ncross', 'B2\nself', 'B2\nMLP', 'final\nnorm'],
        'trace_arm': {'indep': 'Independent coupling', 'rowperm': 'Token permutation', 'shrink50': 'Shrink α = 0.5'},
        'trace_mean': 'Relative divergence\nof the token mean', 'trace_tok': 'Relative divergence\nof the tokens',
        'f3': 'F3: zero', 'video': 'Video stream', 'audio': 'Audio stream',
        'delay': 'Audio offset (ms)', 'early': '← sound early', 'late': 'sound late →',
        'full': 'Whole audio offset', 'fus': 'Offset at the fusion input only', 'silence': 'audio dropout: {}',
        'itu': 'ITU-R BT.1359\ndetectability', 'axis_perm': 'UAR change (points), 90% CI',
    },
}


def col(run, key):
    return [r[key] for r in S0[run]]


def comp(block, hyp, a, b):
    return next(c for c in PROBE['comparisons'] if c['block'] == block and c['hyp'] == hyp and c['a'] == a
                and c['b'] == b)


B9 = [comp(9, 'H3a', '9/r4/D', 'chance'), comp(9, 'H3b', '9/r4/SD', '9/r4/S'),
      comp(9, 'H2', '9/r4/D', '9/cls/D'), comp(9, 'H2', '9/r4/D', '9/mean/D')]


# ---------------------------------------------------------------- figures
def fig_spread(L, lang):
    d = S0['MMA_DFEW_scr']
    x = [r['block'] for r in d]
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.axvspan(0.5, 9.5, color=BAND, lw=0, zorder=0)
    ax.text(5.0, 0.62, L['f1'], ha='center', va='top', color=INK2, fontsize=7)
    for lv in LEVELS:
        y = [r[f'{lv}_spread'] for r in d]
        ax.plot(x, y, color=LV_COLOR[lv], label=L['lv'][lv], **LINE)
        ax.text(0.72, y[0], L['lv'][lv], ha='right', va='center', color=INK2, fontsize=7)
    ax.axhline(0.044, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.text(7.7, 0.040, L['thr'], ha='center', va='top', color=INK2, fontsize=7)
    ax.set_yscale('log')
    ax.set_ylim(0.0018, 0.7)
    ax.set_xlim(-1.6, 12.5)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel(L['block'])
    ax.set_ylabel(L['spread'])
    ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.55, 1.0))
    save(fig, 'fig1_spread_mma_dfew', lang)


def fig_ratio(L, lang):
    fig, axes = plt.subplots(1, 4, figsize=(6.4, 2.2), sharey=True)
    for i, (ax, (run, title)) in enumerate(zip(axes, RUNS)):
        x = col(run, 'block')
        ax.axvspan(0.5, 9.5, color=BAND, lw=0, zorder=0)
        for lv in LEVELS:
            ax.plot(x, col(run, f'{lv}_ratio'), color=LV_COLOR[lv], label=L['lv'][lv],
                    **{**LINE, 'ms': 2.8, 'lw': 1.2})
        ax.set_title(title, fontsize=7.5)
        ax.set_xticks([1, 3, 6, 9, 12])
        ax.set_ylim(0, 0.95)
        ax.set_xlim(0.5, 12.5)
        if i == 0:
            ax.set_ylabel(L['ratio'])
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.text(0.5, 0.015, L['block'], ha='center', va='bottom', fontsize=8.5, color=INK2)
    save(fig, 'fig2_ratio_all_runs', lang)


def fig_probe(L, lang):
    m = PROBE['means']
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.5), sharey=True)
    w = 0.26
    for ax, b in zip(axes, (6, 9, 12)):
        xs = np.arange(len(LEVELS))
        for j, (f, c) in enumerate(zip(('S', 'D', 'SD'), SLOT)):
            ax.bar(xs + (j - 1) * w, [m[f'{b}/{lv}/{f}'] for lv in LEVELS], width=w, color=c,
                   edgecolor=SURF, linewidth=0.8, label=L['feat'][f], zorder=3)
        ax.axhline(100 / 7, color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=4, label=L['chance'])
        ax.set_xticks(xs)
        ax.set_xticklabels([L['lv'][lv].replace(' ', '\n', 1) for lv in LEVELS], fontsize=7)
        ax.set_title(L['blk'].format(b))
        ax.grid(axis='x', visible=False)
        ax.set_ylim(0, 75)
    axes[0].set_ylabel(L['uar'])
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=4, loc='lower center', bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, 'fig3_probe_uar', lang)


def fig_tests(L, lang):
    fig, ax = plt.subplots(figsize=(6.0, 1.9))
    ys = np.arange(len(B9))[::-1]
    for y, c in zip(ys, B9):
        ax.plot(c['ci'], [y, y], color=SLOT[0], lw=1.6, solid_capstyle='round', zorder=3)
        ax.plot([c['diff']], [y], 'o', color=SLOT[0], ms=5.5, mec=SURF, mew=1.0, zorder=4)
        ax.text(11.6, y, L['pass'] if c['counts'] else L['fail'], va='center', ha='right', color=INK2, fontsize=7.5)
    ax.axvline(0, color=AXIS, lw=0.9, zorder=2)
    for v in (-2, 2):
        ax.axvline(v, color=INK2, lw=0.7, ls=(0, (4, 3)), zorder=2)
    ax.text(2.15, ys[0] + 0.45, L['margin'], color=INK2, fontsize=6.8, va='center')
    ax.set_yticks(ys)
    ax.set_yticklabels(L['tests'], fontsize=7.5)
    ax.set_xlim(-4.5, 11.8)
    ax.set_ylim(-0.6, len(B9) - 0.3)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel(L['diff'])
    save(fig, 'fig4_probe_tests_block9', lang)


def fig_p1_repro(L, lang):
    rows = [(L['fold'].format(f), P1F[f]['arms']['base']['uar'], README_UAR[f]) for f in range(1, 6)]
    rows.append((L['pooled'], P1S['stage0']['pooled_uar'], README_TOTAL[0]))
    fig, ax = plt.subplots(figsize=(6.0, 2.2))
    ys = np.arange(len(rows))[::-1].astype(float)
    ys[-1] -= 0.4
    for y, (name, ours, ref) in zip(ys, rows):
        ax.plot([ours, ref], [y, y], color=AXIS, lw=1.2, zorder=2)
        ax.plot([ref], [y], 'o', ms=5.5, mfc=SURF, mec=INK2, mew=1.2, zorder=3,
                label=L['readme'] if y == ys[0] else None)
        ax.plot([ours], [y], 'o', ms=5.5, color=SLOT[0], mec=SURF, mew=0.9, zorder=4,
                label=L['ours'] if y == ys[0] else None)
        ax.text(88.35, y, f'{ours - ref:+.2f}'.replace('-', '−'), va='center', ha='right', color=INK2, fontsize=7.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(80, 88.4)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel(L['uar_axis'])
    ax.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.45, 1.0))
    save(fig, 'fig5_p1_reproduction', lang)


def fig_p1_arms(L, lang):
    A = P1S['arms']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.4), sharey=True, gridspec_kw={'width_ratios': [1, 1]})
    ys = np.arange(len(ARMS))[::-1].astype(float)
    ys[-1] -= 0.4
    for y, arm in zip(ys, ARMS):
        c = INK2 if arm == 'zero' else SLOT[0]
        r = A[arm]
        a1.plot(r['changed_ci'], [y, y], color=c, lw=1.6, zorder=3)
        a1.plot([r['changed_pct']], [y], 'o', color=c, ms=5, mec=SURF, mew=0.9, zorder=4)
        a2.plot(r['d_uar_ci'], [y, y], color=c, lw=1.6, zorder=3)
        a2.plot([r['d_uar']], [y], 'o', color=c, ms=5, mec=SURF, mew=0.9, zorder=4)
    for v in (1, 5):
        a1.axvline(v, color=INK2, lw=0.7, ls=(0, (4, 3)), zorder=2)
    a1.text(5.2, ys[0] + 0.55, L['thr15'], color=INK2, fontsize=6.8, va='center')
    a2.axvline(0, color=AXIS, lw=0.9, zorder=2)
    a2.axvline(-1, color=INK2, lw=0.7, ls=(0, (4, 3)), zorder=2)
    a2.text(-0.95, ys[0] + 0.55, L['thrm1'], color=INK2, fontsize=6.8, va='center')
    a1.set_yticks(ys)
    a1.set_yticklabels([L['arms'][a] for a in ARMS], fontsize=7.3)
    a1.set_xlim(0, 13)
    a2.set_xlim(-3.6, 0.6)
    a1.set_ylim(-0.9, len(ARMS) - 0.2)
    for ax, lab in ((a1, L['changed']), (a2, L['duar'])):
        ax.grid(axis='y', visible=False)
        ax.set_xlabel(lab, fontsize=7.8)
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fig6_p1_interventions', lang)


def fig_nulls(L, lang):
    """Forest plot of the P1-nulls ladder: % changed and dUAR with 90% CIs, grouped."""
    N = NT[lang]
    A = NS['arms']
    rows, y, ys, heads = [], 0.0, [], []
    for g, arms in NULL_GROUPS:
        heads.append((y, N['group'][g]))
        y -= 0.85
        for arm in arms:
            rows.append((arm, g))
            ys.append(y)
            y -= 1.0
        y -= 0.35
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 4.6), sharey=True, gridspec_kw={'width_ratios': [1, 1.25]})
    for yy, (arm, g) in zip(ys, rows):
        r, c = A[arm], NULL_COLOR[g]
        a1.plot(r['changed_ci'], [yy, yy], color=c, lw=1.6, zorder=3)
        a1.plot([r['changed_pct']], [yy], 'o', color=c, ms=4.8, mec=SURF, mew=0.9, zorder=4)
        lo, hi = r['d_uar_ci']
        a2.plot([max(lo, -11.0), hi], [yy, yy], color=c, lw=1.6, zorder=3)
        if r['d_uar'] >= -11.0:
            a2.plot([r['d_uar']], [yy], 'o', color=c, ms=4.8, mec=SURF, mew=0.9, zorder=4)
        else:                                    # off-scale: arrow + value
            a2.annotate('', xy=(-11.0, yy), xytext=(-10.2, yy),
                        arrowprops=dict(arrowstyle='-|>', color=c, lw=1.2, mutation_scale=7))
            a2.text(-10.1, yy + 0.05, f"{r['d_uar']:.1f}".replace('-', '−'), color=INK2, fontsize=6.8,
                    va='center', ha='left')
    a2.axvline(0, color=AXIS, lw=0.9, zorder=2)
    a2.axvline(-1, color=INK2, lw=0.7, ls=(0, (4, 3)), zorder=2)
    for v in (1, 5):
        a1.axvline(v, color=INK2, lw=0.7, ls=(0, (4, 3)), zorder=2)
    a1.set_yticks(ys + [h[0] for h in heads])
    a1.set_yticklabels([N['arm'][a] for a, _ in rows] + [h[1] for h in heads], fontsize=7.0)
    for t in a1.get_yticklabels()[len(rows):]:
        t.set_fontweight('bold')
        t.set_color(INK)
    a1.tick_params(axis='y', length=0)
    a1.set_xlim(0, 25)
    a2.set_xlim(-11.2, 1.6)
    a1.set_ylim(y + 0.6, 0.6)
    for ax, lab in ((a1, L['changed']), (a2, L['duar'])):
        ax.grid(axis='y', visible=False)
        ax.set_xlabel(lab, fontsize=7.8)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'fig7_p1_nulls', lang)


def fig_trace(L, lang):
    """F3 mechanism trace, video stream: divergence of the token mean and of the tokens per sublayer."""
    N = NT[lang]
    tr = NS['trace']
    x = np.arange(len(TRACE_KEYS))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
    for ax, j, lab in ((axes[0], 0, N['trace_mean']), (axes[1], 1, N['trace_tok'])):
        for c, arm in zip(SLOT, TRACE_ARMS):
            v = [tr[arm]['video'][k][j] for k in TRACE_KEYS]
            ax.plot(x, v, color=c, label=N['trace_arm'][arm], **LINE)
        ax.axvspan(-0.3, 0.3, color=BAND, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(N['trace_keys'], fontsize=6.8)
        ax.set_title(lab, fontsize=7.8, color=INK2, loc='left')
        ax.set_ylim(-0.02, 0.36)
        ax.grid(axis='x', visible=False)
    axes[0].text(0, 0.345, N['f3'], fontsize=6.6, color=INK2, ha='center', va='top')
    axes[0].legend(loc='lower right', fontsize=6.8, handlelength=1.6)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'fig8_f3_trace', lang)


def fig_followup(L, lang):
    """(a) space vs time message permutation, (b) audio-visual offset curve."""
    N = NT[lang]
    A = FS['arms']
    delays = sorted(int(d) for d in FS['readings']['delay_curve'])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={'width_ratios': [1, 2.1]})
    arms = ('rowperm', 'rowperm_t', 'rowperm_s')
    ys = np.arange(len(arms))[::-1].astype(float)
    for yy, arm in zip(ys, arms):
        r = A[arm]
        a1.plot(r['d_uar_ci'], [yy, yy], color=SLOT[1], lw=1.6, zorder=3)
        a1.plot([r['d_uar']], [yy], 'o', color=SLOT[1], ms=5, mec=SURF, mew=0.9, zorder=4)
    a1.axvline(0, color=AXIS, lw=0.9, zorder=2)
    a1.axvline(-1, color=INK2, lw=0.7, ls=(0, (4, 3)), zorder=2)
    a1.set_yticks(ys)
    a1.set_yticklabels([N['arm'][a] for a in arms], fontsize=7.0)
    a1.set_ylim(-0.6, len(arms) - 0.4)
    a1.set_xlim(min(-4.5, min(A[a]['d_uar_ci'][0] for a in arms) - 0.3), 1.0)
    a1.grid(axis='y', visible=False)
    a1.set_xlabel(N['axis_perm'], fontsize=7.4)
    a1.tick_params(axis='y', length=0)

    # evenly spaced delay positions (the axis is not linear in ms; ITU band placed by interpolation)
    grid = sorted(delays + [0])
    pos = {d: i for i, d in enumerate(grid)}
    xp = lambda ms: float(np.interp(ms, grid, np.arange(len(grid))))  # noqa: E731
    a2.axvspan(xp(-45), xp(125), color=BAND, zorder=0)
    sil = A['silence']['d_uar']
    a2.text((xp(-45) + xp(125)) / 2, sil + 2.2, N['itu'], fontsize=6.4, color=INK2, ha='center', va='center')
    for key, c, lab in (('', SLOT[0], N['full']), ('_fus', SLOT[1], N['fus'])):
        mid = [0.0 if d == 0 else A[f'delay{d:+d}{key}']['d_uar'] for d in grid]
        xs = [pos[d] for d in grid]
        if key == '':
            lo = [0.0 if d == 0 else A[f'delay{d:+d}']['d_uar_ci'][0] for d in grid]
            hi = [0.0 if d == 0 else A[f'delay{d:+d}']['d_uar_ci'][1] for d in grid]
            a2.fill_between(xs, lo, hi, color=c, alpha=0.15, lw=0, zorder=2)
            a2.plot(xs, mid, color=c, label=lab, **LINE)
        else:
            a2.plot(xs, mid, color=c, label=lab, lw=1.0, ls=(0, (3, 2)), marker='o', ms=4.6, mfc=SURF,
                    mec=c, mew=1.1, zorder=4)
    a2.axhline(sil, color=INK2, lw=0.8, ls=(0, (2, 2)), zorder=2)
    a2.axhline(0, color=AXIS, lw=0.9, zorder=1)
    a2.axhline(-1, color=INK2, lw=0.6, ls=(0, (4, 3)), zorder=1)
    a2.set_xticks(range(len(grid)))
    a2.set_xticklabels([f'{d:+d}'.replace('-', '−') if d else '0' for d in grid], fontsize=6.6)
    a2.set_xlim(-0.4, len(grid) - 0.6)
    lo_all = min(min(A[k]['d_uar_ci'][0] for k in A if k.startswith('delay')), sil)
    hi_all = max(A[k]['d_uar_ci'][1] for k in A if k.startswith('delay'))
    a2.set_ylim(lo_all - 0.8, max(0.9, hi_all + 0.25))
    a2.text(0, sil + 0.15, N['silence'].format(f'{sil:.1f}'.replace('-', '−')), fontsize=6.6,
            color=INK2, va='bottom', ha='left')
    a2.set_xlabel(N['delay'] + f"   ({N['early']}  {N['late']})", fontsize=7.4)
    a2.set_ylabel(L['duar'], fontsize=7.4)
    a2.grid(axis='x', visible=False)
    a2.legend(loc='lower center', fontsize=6.8, ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fig9_p1_followup', lang)


def fig_desync_control(L, lang):
    """Content vs alignment: audio moved, video moved, both moved, per offset."""
    A = DCS['arms']
    rm = DCS['realised_move_ms']
    ds = sorted(int(d) for d in DCS['D'])
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    width = 0.24
    for j, ((key, vi, en), c) in enumerate(zip(DC_ARMS, (SLOT[0], SLOT[1], SLOT[2]))):
        for i, d in enumerate(ds):
            r = A[f'{key}{d:+d}']
            x = i + (j - 1) * width
            ax.plot([x, x], r['d_uar_ci'], color=c, lw=1.6, zorder=3)
            ax.plot([x], [r['d_uar']], 'o', color=c, ms=5, mec=SURF, mew=0.9, zorder=4,
                    label=(vi if lang == 'vi' else en) if i == 0 else None)
    ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
    ax.axhline(-1, color=INK2, lw=0.6, ls=(0, (4, 3)), zorder=1)
    ax.axvline(1.5, color=GRID, lw=0.8, zorder=0)
    ax.set_xticks(range(len(ds)))
    lab = []
    for d in ds:
        real = abs(rm[str(d)][0])
        s = '−' if d < 0 else '+'
        lab.append(f'{s}{real:.0f} ms\n' + (('tiếng sớm' if d < 0 else 'tiếng muộn') if lang == 'vi'
                                           else ('sound early' if d < 0 else 'sound late')))
    ax.set_xticklabels(lab, fontsize=7.0)
    ax.set_ylabel(L['duar'], fontsize=7.6)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='lower center', ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, 'fig10_desync_control', lang)


# ---------------------------------------------------------------- tables
def f(x, d=2):
    return f'{x:.{d}f}'


def sg(x, d=2):
    return f'${x:+.{d}f}$'.replace('-', '-')


def ci(c, d=2):
    return f'$[{c[0]:+.{d}f},\\ {c[1]:+.{d}f}]$'


def write_tex(name, lines):
    os.makedirs(os.path.join(HERE, 'tables'), exist_ok=True)
    with open(os.path.join(HERE, 'tables', name + '.tex'), 'w') as fh:
        fh.write(f'% generated by make_assets.py -- do not edit by hand\n' + '\n'.join(lines) + '\n')


def write_csv(name, header, rows):
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    with open(os.path.join(HERE, 'data', name + '.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def tables():
    # T1: spread and ratio per block, MMA-DFER DFEW
    d = S0['MMA_DFEW_scr']
    L = ['\\begin{tabular}{r rrrr rrrr}', '\\toprule',
         ' & \\multicolumn{4}{c}{Độ biến động (định nghĩa F1)} & \\multicolumn{4}{c}{Tỉ số trong / giữa clip} \\\\',
         '\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}',
         'Block & CLS & TB patch & Vùng $4\\times4$ & Từng patch & CLS & TB patch & Vùng $4\\times4$ & Từng patch \\\\',
         '\\midrule']
    rows = []
    for r in d:
        r4 = f(r['r4_spread'], 4)
        if r['block'] <= 9 and r['r4_spread'] > 0.044:
            r4 = f'\\textbf{{{r4}}}'
        if r['block'] == 10:
            L.append('\\midrule')
        L.append(f"{r['block']} & {f(r['cls_spread'], 4)} & {f(r['mean_spread'], 4)} & {r4} & "
                 f"{f(r['patch_spread'], 4)} & {f(r['cls_ratio'])} & {f(r['mean_ratio'])} & {f(r['r4_ratio'])} & "
                 f"{f(r['patch_ratio'])} \\\\")
        rows.append([r['block']] + [r[f'{lv}_spread'] for lv in LEVELS] + [r[f'{lv}_ratio'] for lv in LEVELS])
    L += ['\\bottomrule', '\\end{tabular}']
    write_tex('tab1_spread_mma_dfew', L)
    write_csv('spread_mma_dfew', ['block'] + [f'{lv}_spread' for lv in LEVELS] + [f'{lv}_ratio' for lv in LEVELS], rows)

    # all runs, full per-block CSV
    for run, _ in RUNS:
        write_csv(f'spread_{run}', ['block', 'frame_spread', 'slot_spread'] + [f'{lv}_spread' for lv in LEVELS]
                  + [f'{lv}_ratio' for lv in LEVELS],
                  [[r['block'], r['frame_spread'], r['slot_spread']] + [r[f'{lv}_spread'] for lv in LEVELS]
                   + [r[f'{lv}_ratio'] for lv in LEVELS] for r in S0[run]])

    # T2: cross-run summary
    L = ['\\begin{tabular}{l rrrrrr}', '\\toprule',
         'Run & CLS max b1--9 & Vùng $4\\times4$ b3--9 & CLS b12 & $4\\times4$ b12 & Tỉ số CLS b1--9 & Tỉ số $4\\times4$ b1--9 \\\\',
         '\\midrule']
    rows = []
    for run, name in RUNS:
        b = S0[run]
        cls9 = max(r['cls_spread'] for r in b[:9])
        r4 = [r['r4_spread'] for r in b[2:9]]
        rc = [r['cls_ratio'] for r in b[:9]]
        rr = [r['r4_ratio'] for r in b[:9]]
        L.append(f"{name.replace('·', '/')} & {f(cls9, 3)} & {f(min(r4), 3)}--{f(max(r4), 3)} & "
                 f"{f(b[11]['cls_spread'], 3)} & {f(b[11]['r4_spread'], 3)} & {f(min(rc))}--{f(max(rc))} & "
                 f"{f(min(rr))}--{f(max(rr))} \\\\")
        rows.append([run, cls9, min(r4), max(r4), b[11]['cls_spread'], b[11]['r4_spread'], min(rc), max(rc),
                     min(rr), max(rr)])
    L += ['\\bottomrule', '\\end{tabular}']
    write_tex('tab2_runs_summary', L)
    write_csv('runs_summary', ['run', 'cls_max_b1_9', 'r4_min_b3_9', 'r4_max_b3_9', 'cls_b12', 'r4_b12',
                               'cls_ratio_min_b1_9', 'cls_ratio_max_b1_9', 'r4_ratio_min_b1_9',
                               'r4_ratio_max_b1_9'], rows)

    # T3: probe UAR
    m = PROBE['means']
    names = {'cls': 'CLS', 'mean': 'TB patch', 'r4': 'Vùng $4\\times4$', 'patch': 'Từng patch'}
    L = ['\\begin{tabular}{l l rrr r}', '\\toprule',
         'Block & Mức & S & D & S+D & S+D $-$ S \\\\', '\\midrule']
    rows = []
    for b in (6, 9, 12):
        if b != 6:
            L.append('\\midrule')
        for lv in LEVELS:
            s, dd, sd = (m[f'{b}/{lv}/{x}'] for x in ('S', 'D', 'SD'))
            L.append(f'{b} & {names[lv]} & {f(s)} & {f(dd)} & {f(sd)} & {sg(sd - s)} \\\\')
            rows.append([b, lv, s, dd, sd, sd - s])
    L += ['\\bottomrule', '\\end{tabular}']
    write_tex('tab3_probe_uar', L)
    write_csv('probe_uar', ['block', 'level', 'S', 'D', 'SD', 'SD_minus_S'], rows)

    # T4: pre-registered comparisons at block 9, and all blocks
    lab = ['H3a & D($4\\times4$) $-$ ngẫu nhiên', 'H3b & S+D($4\\times4$) $-$ S($4\\times4$)',
           'H2 & D($4\\times4$) $-$ D(CLS)', 'H2 & D($4\\times4$) $-$ D(TB patch)']
    L = ['\\begin{tabular}{l l r r l}', '\\toprule',
         'Giả thuyết & So sánh & Chênh UAR & KTC 90\\% & Kết quả \\\\', '\\midrule']
    for name, c in zip(lab, B9):
        L.append(f"{name} & {sg(c['diff'])} & {ci(c['ci'])} & {'đạt' if c['counts'] else 'không đạt'} \\\\")
    L += ['\\bottomrule', '\\end{tabular}']
    write_tex('tab4_probe_tests_block9', L)
    write_csv('probe_comparisons_all', ['block', 'hypothesis', 'a', 'b', 'diff', 'ci_lo', 'ci_hi', 'counts']
              + [f'fold{i}' for i in range(1, 6)],
              [[c['block'], c['hyp'], c['a'], c['b'], c['diff'], c['ci'][0], c['ci'][1], c['counts']]
               + c['per_fold'] for c in PROBE['comparisons']])

    # T5: P1 reproduction
    L = ['\\begin{tabular}{l rr rr r}', '\\toprule',
         ' & \\multicolumn{2}{c}{Chạy lại} & \\multicolumn{2}{c}{README tác giả} & \\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
         'Fold & UAR & WAR & UAR & WAR & Chênh UAR \\\\', '\\midrule']
    rows = []
    for k in range(1, 6):
        a = P1F[k]['arms']['base']
        L.append(f"{k} & {f(a['uar'])} & {f(a['war'])} & {f(README_UAR[k])} & {f(README_WAR[k])} & "
                 f"{sg(a['uar'] - README_UAR[k])} \\\\")
        rows.append([k, a['uar'], a['war'], README_UAR[k], README_WAR[k], a['uar'] - README_UAR[k]])
    s0 = P1S['stage0']
    L += ['\\midrule', f"Gộp & {f(s0['pooled_uar'])} & {f(s0['pooled_war'])} & {f(README_TOTAL[0])} & "
                       f"{f(README_TOTAL[1])} & {sg(s0['pooled_uar'] - README_TOTAL[0])} \\\\",
          '\\bottomrule', '\\end{tabular}']
    rows.append(['pooled', s0['pooled_uar'], s0['pooled_war'], README_TOTAL[0], README_TOTAL[1],
                 s0['pooled_uar'] - README_TOTAL[0]])
    write_tex('tab5_p1_reproduction', L)
    write_csv('p1_reproduction', ['fold', 'uar', 'war', 'readme_uar', 'readme_war', 'diff_uar'], rows)

    # T6: P1 interventions (pooled)
    an = {'indep': 'Ghép nối độc lập (tất cả)', 'indep_v2a': '\\quad chỉ video $\\leftarrow$ audio',
          'indep_a2v': '\\quad chỉ audio $\\leftarrow$ video', 'indep_b1': '\\quad chỉ block fusion 1',
          'indep_b2': '\\quad chỉ block fusion 2', 'zero': 'Bỏ cross-attention (tham chiếu)'}
    A = P1S['arms']
    L = ['\\begin{tabular}{l rr rr}', '\\toprule',
         'Nhánh & Dự đoán đổi (\\%) & KTC 90\\% & Chênh UAR & KTC 90\\% \\\\', '\\midrule']
    rows = []
    for arm in ARMS:
        r = A[arm]
        if arm == 'zero':
            L.append('\\midrule')
        L.append(f"{an[arm]} & {f(r['changed_pct'])} & $[{r['changed_ci'][0]:.2f},\\ {r['changed_ci'][1]:.2f}]$ & "
                 f"{sg(r['d_uar'])} & {ci(r['d_uar_ci'])} \\\\")
        rows.append([arm, r['changed_pct'], *r['changed_ci'], r['uar'], r['d_uar'], *r['d_uar_ci']])
    L += ['\\bottomrule', '\\end{tabular}']
    write_tex('tab6_p1_interventions', L)
    write_csv('p1_interventions', ['arm', 'changed_pct', 'changed_ci_lo', 'changed_ci_hi', 'uar', 'd_uar',
                                   'd_uar_ci_lo', 'd_uar_ci_hi'], rows)

    # T7: P1 per fold, indep and zero
    L = ['\\begin{tabular}{l r rr rr}', '\\toprule',
         ' & & \\multicolumn{2}{c}{Ghép nối độc lập} & \\multicolumn{2}{c}{Bỏ cross-attention} \\\\',
         '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}',
         'Fold & UAR gốc & Đổi (\\%) & Chênh UAR & Đổi (\\%) & Chênh UAR \\\\', '\\midrule']
    rows = []
    for k in range(1, 6):
        a = P1F[k]['arms']
        b = a['base']['uar']
        L.append(f"{k} & {f(b)} & {f(a['indep']['changed_pct'])} & {sg(a['indep']['uar'] - b)} & "
                 f"{f(a['zero']['changed_pct'])} & {sg(a['zero']['uar'] - b)} \\\\")
        rows.append([k, b, a['indep']['changed_pct'], a['indep']['uar'] - b, a['zero']['changed_pct'],
                     a['zero']['uar'] - b])
    L += ['\\bottomrule', '\\end{tabular}']
    write_tex('tab7_p1_per_fold', L)
    write_csv('p1_per_fold', ['fold', 'base_uar', 'indep_changed_pct', 'indep_d_uar', 'zero_changed_pct',
                              'zero_d_uar'], rows)

    # T8: HiCMAE cross-attention concentration
    keys = ['1_v2a', '2_v2a', '1_a2v', '2_a2v']
    kn = {'1_v2a': 'Block 1, video $\\leftarrow$ audio', '2_v2a': 'Block 2, video $\\leftarrow$ audio',
          '1_a2v': 'Block 1, audio $\\leftarrow$ video', '2_a2v': 'Block 2, audio $\\leftarrow$ video'}
    L = ['\\begin{tabular}{l rrr}', '\\toprule', 'Cross-attention & Trung bình 5 fold & Thấp nhất & Cao nhất \\\\',
         '\\midrule']
    rows = []
    for k in keys:
        v = [P1F[i]['H_base'][k] for i in range(1, 6)]
        L.append(f'{kn[k]} & {f(np.mean(v), 3)} & {f(min(v), 3)} & {f(max(v), 3)} \\\\')
        rows.append([k, float(np.mean(v)), min(v), max(v)] + v)
    L += ['\\bottomrule', '\\end{tabular}']
    write_tex('tab8_p1_attention_H', L)
    write_csv('p1_attention_H', ['map', 'mean', 'min', 'max'] + [f'fold{i}' for i in range(1, 6)], rows)

    # T9: P1-nulls ladder (pooled)
    N = NT['vi']
    A = NS['arms']
    L = ['\\begin{tabular}{l rr rr}', '\\toprule',
         'Nhánh & Dự đoán đổi (\\%) & KTC 90\\% & Chênh UAR & KTC 90\\% \\\\']
    rows = []
    for g, arms in NULL_GROUPS + [('input2', ('perm_v_fus', 'perm_v'))]:
        if g != 'input2':
            L += ['\\midrule', f"\\multicolumn{{5}}{{l}}{{\\textit{{{N['group'][g]}}}}} \\\\"]
        for arm in arms:
            r = A[arm]
            L.append(f"\\quad {N['arm'][arm]} & {f(r['changed_pct'])} & "
                     f"$[{r['changed_ci'][0]:.2f},\\ {r['changed_ci'][1]:.2f}]$ & {sg(r['d_uar'])} & {ci(r['d_uar_ci'])} \\\\")
            rows.append([arm, r['changed_pct'], *r['changed_ci'], r['uar'], r['d_uar'], *r['d_uar_ci']])
    L += ['\\bottomrule', '\\end{tabular}']
    write_tex('tab9_p1_nulls', L)
    write_csv('p1_nulls', ['arm', 'changed_pct', 'changed_ci_lo', 'changed_ci_hi', 'uar', 'd_uar',
                           'd_uar_ci_lo', 'd_uar_ci_hi'], rows)

    # T10: F3 mechanism trace (indep), both streams
    tr = NS['trace']
    names = ['B1 cross', 'B1 self', 'B1 MLP', 'B2 cross', 'B2 self', 'B2 MLP', 'norm cuối']
    L = ['\\begin{tabular}{l rr rr}', '\\toprule',
         ' & \\multicolumn{2}{c}{Luồng video} & \\multicolumn{2}{c}{Luồng audio} \\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
         'Sau lớp & TB token & Từng token & TB token & Từng token \\\\', '\\midrule']
    rows = []
    for k, nm in zip(TRACE_KEYS, names):
        v, au = tr['indep']['video'][k], tr['indep']['audio'][k]
        L.append(f'{nm} & {f(v[0], 3)} & {f(v[1], 3)} & {f(au[0], 3)} & {f(au[1], 3)} \\\\')
        for arm in TRACE_ARMS:
            rows.append([arm, k, *tr[arm]['video'][k], *tr[arm]['audio'][k]])
    L += ['\\bottomrule', '\\end{tabular}']
    write_tex('tab10_f3_trace', L)
    write_csv('f3_trace', ['arm', 'sublayer', 'video_mean', 'video_tokens', 'audio_mean', 'audio_tokens'], rows)

    # T11: follow-up, space vs time and audio-visual offset
    A = FS['arms']
    delays = sorted(int(d) for d in FS['readings']['delay_curve'])
    L = ['\\begin{tabular}{l rr rr}', '\\toprule',
         ' & \\multicolumn{2}{c}{Lệch cả audio} & \\multicolumn{2}{c}{Lệch chỉ ở fusion} \\\\',
         '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
         'Điều kiện & Chênh UAR & KTC 90\\% & Chênh UAR & KTC 90\\% \\\\', '\\midrule']
    rows = []
    for arm in ('rowperm', 'rowperm_t', 'rowperm_s'):
        r = A[arm]
        L.append(f"{N['arm'][arm]} & {sg(r['d_uar'])} & {ci(r['d_uar_ci'])} & & \\\\")
        rows.append([arm, r['d_uar'], *r['d_uar_ci'], '', '', ''])
    L.append('\\midrule')
    for d in delays + ['silence']:
        k = 'silence' if d == 'silence' else f'delay{d:+d}'
        r, rf = A[k], A[k + '_fus']
        nm = 'Mất tiếng (waveform $=0$)' if d == 'silence' else f'Lệch ${d:+d}$ ms'
        L.append(f"{nm} & {sg(r['d_uar'])} & {ci(r['d_uar_ci'])} & {sg(rf['d_uar'])} & {ci(rf['d_uar_ci'])} \\\\")
        rows.append([k, r['d_uar'], *r['d_uar_ci'], rf['d_uar'], *rf['d_uar_ci']])
    L += ['\\bottomrule', '\\end{tabular}']
    write_tex('tab11_p1_followup', L)
    write_csv('p1_followup', ['condition', 'd_uar', 'd_uar_ci_lo', 'd_uar_ci_hi', 'fus_d_uar', 'fus_ci_lo',
                              'fus_ci_hi'], rows)

    # T12: desync control, content vs alignment
    A, Dd, rm = DCS['arms'], DCS['D'], DCS['realised_move_ms']
    rd_vi = {'alignment': 'khớp thời gian', 'content': 'nội dung', 'mixed': 'lẫn lộn', 'not applicable': 'không áp dụng'}
    L = ['\\begin{tabular}{l rrr rl}', '\\toprule',
         ' & \\multicolumn{3}{c}{Chênh UAR so với gốc} & & \\\\', '\\cmidrule(lr){2-4}',
         'Độ lệch & Chỉ audio & Chỉ video & Cả hai & $D_d$ [KTC 90\\%] & Kết luận C1 \\\\', '\\midrule']
    rows = []
    for d in sorted(int(x) for x in Dd):
        r = {k: A[f'{k}{d:+d}'] for k in ('aud', 'vid', 'both')}
        dd = Dd[str(d)]
        real = rm[str(d)][0]
        L.append(f"${real:+.0f}$ ms & {sg(r['aud']['d_uar'])} & {sg(r['vid']['d_uar'])} & {sg(r['both']['d_uar'])} & "
                 f"{sg(dd['value'])} {ci(dd['ci'])} & {rd_vi[dd['reading']]} \\\\")
        rows.append([d, real, r['aud']['d_uar'], *r['aud']['d_uar_ci'], r['vid']['d_uar'], *r['vid']['d_uar_ci'],
                     r['both']['d_uar'], *r['both']['d_uar_ci'], dd['value'], *dd['ci'], dd['reading']])
    L += ['\\bottomrule', '\\end{tabular}']
    write_tex('tab12_desync_control', L)
    write_csv('desync_control', ['d_ms', 'realised_ms', 'aud', 'aud_lo', 'aud_hi', 'vid', 'vid_lo', 'vid_hi',
                                 'both', 'both_lo', 'both_hi', 'D', 'D_lo', 'D_hi', 'C1'], rows)


# ---------------------------------------------------------------- numbers.tex
def numbers():
    N = {}
    d = S0['MMA_DFEW_scr']
    r4 = [r['r4_spread'] for r in d[2:9]]
    N['SpreadDfewRfourMin'] = f(min(r4), 3)
    N['SpreadDfewRfourMax'] = f(max(r4), 3)
    folds = [r['r4_spread'] / r['cls_spread'] for r in d[2:9]]
    N['SpreadDfewRfourOverClsMin'] = f'{min(folds):.0f}'
    N['SpreadDfewRfourOverClsMax'] = f'{max(folds):.0f}'
    N['SpreadThreshold'] = '0.044'
    N['FOneMaxBlocksOneToNine'] = f(PAPERDIAG['F1_max_mafw_col_b9'], 4)
    for run, key in (('MMA_DFEW_scr', 'Dfew'), ('MMA_MAFW_scr', 'Mafw'), ('MMA_CREMAD_scr', 'Cremad'),
                     ('ABV_DFEW_scr', 'AbvDfew')):
        b = S0[run]
        N[f'Cls{key}BlockTwelve'] = f(b[11]['cls_spread'], 3)
        N[f'Cls{key}MaxBlocksOneToNine'] = f(max(r['cls_spread'] for r in b[:9]), 3)
        N[f'Ratio{key}ClsMin'] = f(min(r['cls_ratio'] for r in b[:9]))
        N[f'Ratio{key}ClsMax'] = f(max(r['cls_ratio'] for r in b[:9]))
        N[f'Ratio{key}RfourMin'] = f(min(r['r4_ratio'] for r in b[:9]))
        N[f'Ratio{key}RfourMax'] = f(max(r['r4_ratio'] for r in b[:9]))
    N['RatioDfewMeanMin'] = f(min(r['mean_ratio'] for r in d[:9]))
    N['RatioDfewMeanMax'] = f(max(r['mean_ratio'] for r in d[:9]))
    N['RatioDfewPatchMin'] = f(min(r['patch_ratio'] for r in d[:9]))
    N['RatioDfewPatchMax'] = f(max(r['patch_ratio'] for r in d[:9]))
    N['AbvColClsMaxBlocksOneToNine'] = f(PAPERDIAG['ABV_col_cls_max_b1_9'], 3)

    m = PROBE['means']
    N['ProbeChance'] = f(100 / 7)
    N['ProbeDRfourBlockNine'] = f(m['9/r4/D'])
    sdms = [m[f'{b}/{lv}/SD'] - m[f'{b}/{lv}/S'] for b in (6, 9, 12) for lv in LEVELS]
    N['ProbeSDminusSMin'] = sg(min(sdms))
    N['ProbeSDminusSMax'] = sg(max(sdms))
    for key, c in zip(('HThreeA', 'HThreeB', 'HTwoCls', 'HTwoMean'), B9):
        N[f'Probe{key}Diff'] = sg(c['diff'])
        N[f'Probe{key}CI'] = ci(c['ci'])
    for b, key in ((6, 'Six'), (9, 'Nine'), (12, 'Twelve')):
        vals = [m[f'{b}/{lv}/S'] for lv in LEVELS]
        N[f'ProbeStaticBlock{key}Min'] = f(min(vals))
        N[f'ProbeStaticBlock{key}Max'] = f(max(vals))

    s0 = P1S['stage0']
    N['PoneReadmeUAR'] = f(README_TOTAL[0])
    N['PoneReadmeWAR'] = f(README_TOTAL[1])
    N['PonePooledUAR'] = f(s0['pooled_uar'])
    N['PonePooledWAR'] = f(s0['pooled_war'])
    N['PonePooledGap'] = sg(s0['pooled_uar'] - README_TOTAL[0])
    gaps = [P1F[k]['arms']['base']['uar'] - README_UAR[k] for k in range(1, 6)]
    N['PoneFoldGapMin'] = sg(min(gaps))
    N['PoneFoldGapMax'] = sg(max(gaps))
    names = {'indep': 'Indep', 'zero': 'Zero', 'indep_v2a': 'IndepVtoA', 'indep_a2v': 'IndepAtoV',
             'indep_b1': 'IndepBlockOne', 'indep_b2': 'IndepBlockTwo'}
    for arm, key in names.items():
        r = P1S['arms'][arm]
        N[f'Pone{key}Changed'] = f(r['changed_pct'])
        N[f'Pone{key}ChangedCI'] = f"$[{r['changed_ci'][0]:.2f},\\ {r['changed_ci'][1]:.2f}]$"
        N[f'Pone{key}DUAR'] = sg(r['d_uar'])
        N[f'Pone{key}DUARCI'] = ci(r['d_uar_ci'])
    ch = [P1F[k]['arms']['indep']['changed_pct'] for k in range(1, 6)]
    du = [P1F[k]['arms']['indep']['uar'] - P1F[k]['arms']['base']['uar'] for k in range(1, 6)]
    N['PoneIndepFoldChangedMin'] = f(min(ch))
    N['PoneIndepFoldChangedMax'] = f(max(ch))
    N['PoneIndepFoldDUARMin'] = sg(min(du))
    N['PoneIndepFoldDUARMax'] = sg(max(du))
    hv = [P1F[k]['H_base'][x] for k in range(1, 6) for x in P1F[k]['H_base']]
    N['PoneAttnHMin'] = f(min(hv))
    N['PoneAttnHMax'] = f(max(hv))
    N['PoneClass'] = P1S.get('indep_class', '')

    # P1-nulls
    A, R = NS['arms'], NS['readings']
    nn = {'zero': 'Zero', 'indep': 'Indep', 'devonly': 'Devonly', 'shrink25': 'ShrinkQuarter',
          'shrink50': 'ShrinkHalf', 'shrink75': 'ShrinkThreeQ', 'rowperm': 'Rowperm', 'rowperm_s1': 'RowpermSeedOne',
          'rowperm_s2': 'RowpermSeedTwo', 'rowperm_t': 'RowpermTime', 'uniform': 'Uniform', 'papa': 'Papa',
          'emap': 'Emap', 'perm_a': 'PermAudio', 'perm_a_fus': 'PermAudioFus', 'perm_v': 'PermVideo',
          'perm_v_fus': 'PermVideoFus'}
    for arm, key in nn.items():
        r = A[arm]
        N[f'Null{key}DUAR'] = sg(r['d_uar'])
        N[f'Null{key}DUARCI'] = ci(r['d_uar_ci'])
        N[f'Null{key}Changed'] = f(r['changed_pct'])
    N['NullBaseUAR'] = f(NS['base_uar'])
    N['NullRowpermMinusIndep'] = sg(R['rowperm_minus_indep']['value'])
    N['NullRowpermMinusIndepCI'] = ci(R['rowperm_minus_indep']['ci'])
    N['NullInteraction'] = sg(R['factorial_interaction']['value'])
    N['NullInteractionCI'] = ci(R['factorial_interaction']['ci'])
    N['NullReadingNOne'] = R['N1']
    N['NullReadingNTwo'] = R['N2']
    ag = [v for s in NS['sanity'].values() for k, v in (s.get('agree_with_p1') or {}).items() if k in ('base', 'indep')]
    if ag:
        N['NullAgreeMin'], N['NullAgreeMax'] = f(min(ag)), f(max(ag))
    mant, expo = f'{max(NS["f3_max_rel_mean_diff"].values()):.1e}'.split('e')
    N['NullFThreeMaxRel'] = f'${mant}\\times 10^{{{int(expo)}}}$'
    tv = NS['trace']['indep']['video']
    N['NullTraceTokBOneCross'] = f(tv['b1_cross'][1], 2)
    N['NullTraceMeanBOneSelf'] = f(tv['b1_self'][0], 2)
    N['NullTraceMeanNorm'] = f(tv['norm'][0], 2)

    # P1-followup
    A, R = FS['arms'], FS['readings']
    N['FupBaseUAR'] = f(FS['base_uar'])
    for arm, key in (('rowperm', 'Rowperm'), ('rowperm_t', 'RowpermTime'), ('rowperm_s', 'RowpermSpace'),
                     ('silence', 'Silence'), ('silence_fus', 'SilenceFus')):
        N[f'Fup{key}DUAR'] = sg(A[arm]['d_uar'])
        N[f'Fup{key}DUARCI'] = ci(A[arm]['d_uar_ci'])
    for d in sorted(int(x) for x in R['delay_curve']):
        key = ('Late' if d > 0 else 'Early') + {40: 'Forty', 80: 'Eighty', 160: 'OneSixty', 320: 'ThreeTwenty',
                                                 640: 'SixForty'}[abs(d)]
        N[f'FupDelay{key}DUAR'] = sg(A[f'delay{d:+d}']['d_uar'])
        N[f'FupDelay{key}DUARCI'] = ci(A[f'delay{d:+d}']['d_uar_ci'])
        N[f'FupDelay{key}FusDUAR'] = sg(A[f'delay{d:+d}_fus']['d_uar'])
    N['FupTolLate'] = str(R['D1_tolerance_ms']['late'])
    N['FupTolEarly'] = str(R['D1_tolerance_ms']['early'])
    N['FupReadingTOne'] = R['T1']

    # P1-desync-control
    A, Dd, R = DCS['arms'], DCS['D'], DCS['readings']
    N['DcBaseUAR'] = f(DCS['base_uar'])
    N['DcReadingCOne'] = R['C1']
    words = {-320: 'EarlyLong', -160: 'EarlyShort', 160: 'LateShort', 320: 'LateLong'}
    for d in sorted(int(x) for x in Dd):
        w = words[d]
        N[f'DcRealised{w}'] = f"{abs(DCS['realised_move_ms'][str(d)][0]):.0f}"
        for k, kk in (('aud', 'Aud'), ('vid', 'Vid'), ('both', 'Both')):
            N[f'Dc{kk}{w}DUAR'] = sg(A[f'{k}{d:+d}']['d_uar'])
            N[f'Dc{kk}{w}DUARCI'] = ci(A[f'{k}{d:+d}']['d_uar_ci'])
        N[f'DcD{w}'] = sg(Dd[str(d)]['value'])
        N[f'DcD{w}CI'] = ci(Dd[str(d)]['ci'])
    N['DcCThreeVid'] = sg(R['C3_vid_early_minus_late_160']['value'])
    N['DcCThreeVidCI'] = ci(R['C3_vid_early_minus_late_160']['ci'])
    N['DcCThreeAud'] = sg(R['C3_aud_early_minus_late_160']['value'])
    N['DcCThreeAudCI'] = ci(R['C3_aud_early_minus_late_160']['ci'])

    with open(os.path.join(HERE, 'numbers.tex'), 'w') as fh:
        fh.write('% generated by make_assets.py -- every value computed from the result JSON files\n'
                 '% usage: \\input{numbers} in the preamble, then e.g. \\PonePooledUAR\n')
        for k, v in N.items():
            fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
    return N


def main():
    tables()
    N = numbers()
    for lang in ('vi', 'en'):
        L = T[lang]
        fig_spread(L, lang)
        fig_ratio(L, lang)
        fig_probe(L, lang)
        fig_tests(L, lang)
        fig_p1_repro(L, lang)
        fig_p1_arms(L, lang)
        fig_nulls(L, lang)
        fig_trace(L, lang)
        fig_followup(L, lang)
        fig_desync_control(L, lang)
    print(f'{len(N)} macros, tables and figures written under {HERE}')


if __name__ == '__main__':
    main()
