#!/usr/bin/env python
"""Paper figure 1 (plan v2 §5): per-block profiles of the router's cost input, with HiCMAE as reference, and the
block-wise replacement panel. Reads results/e2 and results/blocksub (JSON); writes paper-v2/figures/fig_e2_profiles.{pdf,png}.

    <mplenv>/bin/python report-assets/make_fig_e2.py
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'paper-v2', 'figures')
INK, INK2, MUTED, GRID, AXIS, SURF, BAND = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff', '#f1f0eb'
SLOT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 8.5, 'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.2,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': INK2, 'text.color': INK,
    'axes.titlecolor': INK, 'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.axisbelow': True, 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False,
})


def load(*p):
    return json.load(open(os.path.join(ROOT, *p)))


SETTINGS = [('OT_CREMAD_test', 'CREMAD_mma', 'Router · CREMA-D', SLOT[0]),
            ('OT_DFEW_test', 'DFEW_mma', 'Router · DFEW', SLOT[1]),
            ('ABV_DFEW_test', 'DFEW_abv', 'ABVnet router · DFEW', SLOT[2])]
MARK = {SLOT[0]: 'o', SLOT[1]: 's', SLOT[2]: '^'}      # secondary encoding: never colour alone (Elsevier artwork)
HATCH = {SLOT[0]: '', SLOT[1]: '////', SLOT[2]: '....'}
summ = load('results', 'e2', 'summary.json')
hic = summ['hicmae']


def main():
    from matplotlib.patches import Patch
    ref_fill = '#e3e2da'        # HiCMAE reference band: a neutral one step darker than BAND, so it survives print
    dodge = {SLOT[0]: -0.3, SLOT[1]: 0.0, SLOT[2]: 0.3}   # small x offsets so coincident markers stay separate
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw={'width_ratios': [1, 1, 1.0], 'wspace': 0.5})
    a, b, c = axes
    for e2name, _, label, col in SETTINGS:
        r = load('results', 'e2', e2name + '.json')['blocks']
        x = [q['block'] + dodge[col] for q in r]
        a.fill_between(x, [q['cs_spread_v_fmin'] for q in r], [q['cs_spread_v_fmax'] for q in r], color=col, alpha=0.14, lw=0)
        a.plot(x, [q['cs_spread_v'] for q in r], color=col, lw=1.3, marker=MARK[col], ms=3.4, mec=SURF, mew=0.7, zorder=3)
        b.fill_between(x, [q['kappa_fmin'] for q in r], [q['kappa_fmax'] for q in r], color=col, alpha=0.14, lw=0)
        b.plot(x, [q['kappa'] for q in r], color=col, lw=1.3, zorder=3)
        for xi, q in zip(x, r):   # open marker where the bound beta >= 1 (uninformative)
            vac = q['beta'] >= 1
            b.plot(xi, q['kappa'], marker=MARK[col], ms=3.8 if vac else 3.4, mec=col if vac else SURF,
                   mew=0.9 if vac else 0.7, mfc=SURF if vac else col, zorder=4)
    # HiCMAE reference in (b), labelled in place; HiCMAE was run on CREMA-D only, so the label says so
    ks = [hic[s]['kappa']['mean'] for s in hic]
    k1 = hic['b1_v2a']['kappa']['mean']
    b.axhspan(min(ks), max(ks), color=ref_fill, lw=0, zorder=0)
    b.axhline(k1, color=INK2, lw=0.9, ls=(0, (4, 2)), zorder=1)
    b.text(0.75, max(ks) * 1.22, 'HiCMAE (CREMA-D):\nband = its 4 maps,\ndashed = 1st video-query map', fontsize=6.4,
           color=INK2, va='bottom', ha='left', linespacing=1.2)
    a.text(0.03, 0.90, 'higher = frames of a clip\ndiffer more', transform=a.transAxes, fontsize=6.4, color=MUTED,
           va='top', ha='left', linespacing=1.15, bbox=dict(fc=SURF, ec='none', pad=0.6))
    for ax, yl, t in ((a, 'Frame spread in cost space (log scale)', '(a) Within-clip variation\nthe cost reads'),
                      (b, 'Cost interaction κ (log scale)', '(b) Cost interaction\nvs HiCMAE\'s maps')):
        ax.set_yscale('log')
        ax.set_xlim(0.4, 12.6)
        ax.set_xticks([1, 3, 5, 7, 9, 12])
        ax.set_xticks(range(1, 13), minor=True)
        ax.tick_params(axis='x', which='minor', length=1.8)
        ax.grid(axis='x', which='minor', visible=False)
        ax.set_xlabel('Transformer block')
        ax.set_ylabel(yl)
        ax.set_title(t, loc='left', linespacing=1.15)
    a.set_ylim(1e-3, 1.5)
    b.set_ylim(5e-4, 16)
    # figure legend: settings on row 1, the fold band and the marker fill of (b) on row 2
    h = [Line2D([], [], color=col, lw=1.3, marker=MARK[col], ms=3.6, mec=SURF) for _, _, _, col in SETTINGS]
    h += [Patch(facecolor=MUTED, alpha=0.3, lw=0),
          Line2D([], [], marker='o', ls='', mfc=MUTED, mec=SURF, ms=4.2),
          Line2D([], [], marker='o', ls='', mfc=SURF, mec=MUTED, mew=0.9, ms=4.2)]
    lab = [l for _, _, l, _ in SETTINGS] + ['range over 5 folds', '(b) filled: bound informative (β < 1)',
                                             '(b) open: bound uninformative (β ≥ 1)']
    order = [0, 3, 1, 4, 2, 5]      # the legend fills column-wise
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc='upper center', ncol=3, fontsize=6.8,
               handlelength=1.6, columnspacing=1.8, labelspacing=0.35, bbox_to_anchor=(0.5, 1.135))
    # (c) horizontal bars grouped by setting; whiskers: 90% paired clip-bootstrap interval (blocksub_summary.py)
    subsets = [('indep_1_9', '1–9'), ('indep_10_12', '10–12'), ('indep_all', 'all')]
    yt, ylab = [], []
    y = 0.0
    for _, bsname, label, col in SETTINGS:
        s = load('results', 'blocksub', bsname, 'summary.json')['arms']
        c.text(0.0, y - 0.05, label, fontsize=6.6, color=INK, va='bottom', ha='left')
        y += 0.55
        for arm, sl in subsets:
            v = s[arm]['changed_pct']
            lo, hi = s[arm]['changed_ci']
            c.barh(y, v, height=0.72, color=col, hatch=HATCH[col], edgecolor=SURF, linewidth=0, zorder=3)
            c.plot([lo, hi], [y, y], color=INK2, lw=0.7, zorder=4, solid_capstyle='butt')
            c.text(hi + 0.06, y, f'{v:.2f}', fontsize=6.3, color=INK2, va='center', ha='left')
            if arm == 'indep_10_12':      # the paper's claim: blocks 10-12 carry nearly all of the full replacement
                share = v / s['indep_all']['changed_pct']
                c.annotate(f'(= {share:.0%} of ‘all’)', (hi + 0.06, y), xytext=(15.5, 0), textcoords='offset points',
                           fontsize=6.3, color=MUTED, va='center', ha='left', annotation_clip=False)
            yt.append(y)
            ylab.append(sl)
            y += 0.85
        y += 0.25
    c.set_yticks(yt)
    c.set_yticklabels(ylab)
    c.tick_params(axis='y', labelsize=6.6, length=0)
    c.set_ylim(y - 0.2, -0.45)
    c.set_xlim(0, 3.55)
    c.set_xlabel('Test predictions changed (%)')
    c.set_ylabel('Product replacement in blocks')
    c.set_title('(c) Where the router\'s\ncoupling acts', loc='left', linespacing=1.15)
    c.grid(axis='y', visible=False)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_e2_profiles.{ext}'), bbox_inches='tight', dpi=220)
    print('->', os.path.join(OUT, 'fig_e2_profiles.pdf'))


if __name__ == '__main__':
    main()
