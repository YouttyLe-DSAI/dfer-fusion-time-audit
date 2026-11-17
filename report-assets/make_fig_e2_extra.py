#!/usr/bin/env python
"""Supplementary figures from E2 (plan v2 §5): per-clip distributions (fig_e2_perclip) and the measured cost
interaction against its bound (fig_e2_bound, sized for 0.55\\linewidth). Reads results/e2 (JSON, NPZ); writes
paper-v2/figures/fig_e2_perclip.* and fig_e2_bound.*.

    <mplenv>/bin/python report-assets/make_fig_e2_extra.py
"""

import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

import make_fig_e2 as F  # noqa: E402   style, palette, settings

ROOT, OUT = F.ROOT, F.OUT
INK, INK2, MUTED, BAND, SURF = F.INK, F.INK2, F.MUTED, F.BAND, F.SURF
BLUES = ['#a3c4ee', '#6e9fe0', '#3f80d4', '#2662b0', '#123d78']        # one hue, light -> dark = early -> late block
SHOW = [1, 6, 9, 11, 12]
REF = '#e9e8e1'          # neutral fill for the region where the bound is uninformative
INSET = [0.06, 0.535, 0.46]   # zoom in fig_e2_bound: left, bottom, right edge (axes fraction); above the diagonal
KAPPA1 = dict(color=MUTED, lw=0.6, ls=(0, (1, 1.5)))                  # the kappa = 1 reference, same style in both figures
HIC_LS = {'b1_v2a': (0, (4, 2)), 'b2_v2a': (0, (1, 1.5))}             # HiCMAE video-query maps: 1st / 2nd fusion block


def ecdf(ax, x, **kw):
    x = np.sort(x)
    ax.plot(x, np.arange(1, len(x) + 1) / len(x), **kw)


def perclip():
    z = np.load(os.path.join(ROOT, 'results', 'e2', 'OT_CREMAD_test.npz'))
    blocks = list(z['blocks'])
    hz = [np.load(p) for p in sorted(glob.glob(os.path.join(ROOT, 'results', 'e2', 'HiCMAE_CREMAD_test_v2', 'fold*.npz')))]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.85), sharey=True,
                               gridspec_kw={'wspace': 0.14, 'left': 0.085, 'right': 0.985})
    for key, ax in (('cs_spread_v', a), ('kappa', b)):
        for col, blk in zip(BLUES, SHOW):
            i = blocks.index(blk)
            vals = np.concatenate([z[f'fold{f}_{key}'][:, i] for f in range(1, 6)])
            ecdf(ax, vals, color=col, lw=1.4, zorder=3)
            # the router's five fold medians: a white halo keeps the ticks apart from the curves they sit on
            med = [np.median(z[f'fold{f}_{key}'][:, i]) for f in range(1, 6)]
            ax.plot(med, [0.5] * 5, ls='', marker='|', ms=7, color=SURF, mew=2.6, zorder=4)
            ax.plot(med, [0.5] * 5, ls='', marker='|', ms=7, color=col, mew=1.1, zorder=5)
            # direct label left of the curve where it crosses 80% of clips; in (b) block 12 is labelled on its right,
            # clear of the HiCMAE curves
            right = key == 'kappa' and blk == 12
            xq = np.quantile(vals, 0.80)
            ax.text(xq * 1.15 if right else xq / 1.12, 0.80, 'block 1' if blk == 1 else str(blk), color=INK2,
                    fontsize=6.6, ha='left' if right else 'right', va='center')
    hv = {}
    for site, ls in HIC_LS.items():
        hv[site] = np.concatenate([h[f'{site}_kappa'].mean(1) for h in hz])      # mean over the eight heads
        ecdf(b, hv[site], color=INK2, lw=1.3, ls=ls, zorder=2)
    b.axvline(1.0, zorder=0, **KAPPA1)
    b.text(1.06, 0.03, 'κ = 1', fontsize=6.4, color=MUTED, ha='left', va='bottom')
    # HiCMAE's two video-query maps labelled in place, both set vertically: the 1st fusion block right of its
    # (dashed) curve, the 2nd fusion block left of its (dotted) curve
    b.text(np.quantile(hv['b1_v2a'], 0.30) * 1.22, 0.31, 'HiCMAE 1st\nvideo-query map', fontsize=6.4, color=INK2,
           ha='left', va='center', rotation=90, linespacing=1.1)
    b.text(np.quantile(hv['b2_v2a'], 0.55) / 1.15, 0.55, 'HiCMAE 2nd\nvideo-query map', fontsize=6.4, color=INK2,
           ha='right', va='center', rotation=90, linespacing=1.1)
    # titles carry the message, a muted line under each says how to read the curves
    for ax, xl, t, cue in (
            (a, 'Frame spread in cost space per clip (log scale)', '(a) Blocks 1–11: low spread in nearly every clip',
             'steep curve = nearly all clips alike; further right = frames differ more'),
            (b, 'Cost interaction κ per clip (log scale)', "(b) Only block 12 reaches HiCMAE's interaction",
             'further right = more room for frame-specific coupling')):
        ax.set_xscale('log')
        ax.set_xlabel(xl)
        ax.set_title(t, loc='left', pad=13)
        ax.text(0, 1.02, cue, transform=ax.transAxes, fontsize=6.4, color=MUTED, ha='left', va='bottom')
        ax.set_ylim(0, 1.0)
    a.set_xlim(6e-4, 2.5)
    b.set_xlim(2e-4, 8)
    a.set_ylabel('Cumulative share of test clips')
    # HiCMAE's two curves are labelled in place, so the legend keeps only the router's encodings
    h = [Line2D([], [], color=BLUES[-2], lw=1.4),
         Line2D([], [], ls='', marker='|', ms=7, mew=1.1, color=BLUES[-2])]
    lab = ['Router · CREMA-D (light → dark: block 1 → 12)', "router's fold medians (5 folds)"]
    fig.legend(h, lab, loc='upper center', ncol=2, fontsize=6.8, bbox_to_anchor=(0.535, 1.06), handlelength=2.0,
               columnspacing=2.5)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_e2_perclip.{ext}'), bbox_inches='tight', dpi=220)


def _bound_points(ax, series, hic, ms, line=True):
    """Router settings as filled markers joined in block order; HiCMAE maps as open diamonds drawn on top (hollow,
    so a router marker beneath stays visible)."""
    for col, bk in series:
        if line:
            ax.plot([p[0] for p in bk], [p[1] for p in bk], color=col, lw=0.6, alpha=0.5, zorder=2.6)
        ax.plot([p[0] for p in bk], [p[1] for p in bk], ls='', marker=F.MARK[col], ms=ms, color=col,
                mec=SURF, mew=0.6, zorder=3)
    hb, hk = hic
    ax.plot(hb, hk, ls='', marker='D', ms=ms + 0.2, mfc='none', mec=INK2, mew=0.9, zorder=3.5)


def bound():
    summ = json.load(open(os.path.join(ROOT, 'results', 'e2', 'summary.json')))
    fig, ax = plt.subplots(figsize=(3.5, 3.8))
    lim = (5e-4, 30)
    ax.axvspan(1, lim[1], color=REF, lw=0, zorder=0)
    ax.plot(lim, lim, color=INK2, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.text(1.3e-3, 1.9e-3, 'κ = β (the bound)', fontsize=6.4, color=INK2, rotation=45, rotation_mode='anchor',
            ha='left', va='bottom')
    ax.text(0.03, 0.40, 'κ > β: excluded\nby the bound', transform=ax.transAxes, fontsize=6.4, color=MUTED,
            ha='left', va='bottom', linespacing=1.15, style='italic')
    # kappa = 1 drawn from a little right of the zoom on, clear of the zoom's (right-hand) tick labels
    ax.plot([lim[0] * (lim[1] / lim[0]) ** (INSET[2] + 0.07), lim[1]], [1, 1], zorder=1, **KAPPA1)
    ax.text(lim[1] / 1.15, 1.12, 'κ = 1', fontsize=6.4, color=MUTED, ha='right', va='bottom')
    # the two regions of the bound, explained in place
    ax.text(0.85, 6.5e-4, 'β < 1: bound\ninformative\n(forces κ < 1)', fontsize=6.4, color=INK2, ha='right',
            va='bottom', linespacing=1.15)
    ax.text(1.18, 6.5e-4, 'β ≥ 1: bound\nuninformative\n(allows κ ≥ 1)', fontsize=6.4, color=INK2, ha='left',
            va='bottom', linespacing=1.15)
    series, handles = [], []
    for e2name, _, label, col in F.SETTINGS:
        r = json.load(open(os.path.join(ROOT, 'results', 'e2', e2name + '.json')))['blocks']
        assert [q['block'] for q in r] == list(range(1, 13))
        bk = [(q['beta'], q['kappa']) for q in r]
        assert all(k <= bb for bb, k in bk)
        series.append((col, bk))
        handles.append((Line2D([], [], ls='', marker=F.MARK[col], ms=4.2, color=col, mec=SURF, mew=0.6), label))
    hs = summ['hicmae']
    hb, hk = [hs[s]['beta']['mean'] for s in hs], [hs[s]['kappa']['mean'] for s in hs]
    assert all(k <= bb for bb, k in zip(hb, hk))
    handles.append((Line2D([], [], ls='', marker='D', ms=4.4, mfc='none', mec=INK2, mew=0.9),
                    'HiCMAE · CREMA-D (4 maps)'))
    # how far below the bound: beta / kappa of the plotted fold means, router (all three settings) and HiCMAE
    gap_r = [bb / k for _, bk in series for bb, k in bk]
    gap_h = [bb / k for bb, k in zip(hb, hk)]
    _bound_points(ax, series, (hb, hk), 4.2)
    # block order: the trajectories start at block 1 (lower left) and end at block 12 (in the zoom)
    b1 = min((bk[0] for _, bk in series), key=lambda p: p[0])
    ax.text(b1[0] / 1.3, b1[1], 'block 1', fontsize=6.4, color=INK2, ha='right', va='center')
    # zoom on the corner where router block 12 and HiCMAE's maps sit; it lies in the excluded triangle (kappa > beta)
    zx, zy = (1.9, 6.4), (0.34, 1.55)
    ins = ax.inset_axes(INSET[:2] + [INSET[2] - INSET[0]] * 2)
    ins.set_facecolor(REF)
    ins.axhline(1, zorder=1, **KAPPA1)
    _bound_points(ins, series, (hb, hk), 4.6)
    # block numbers: block 12 of each router setting; ABVnet's blocks 9-11 form one tight cluster, labelled once to
    # the right of block 10, which leaves the space right of block 11 for the HiCMAE map that sits on it
    for col, bk in series:
        ins.text(bk[11][0] * 1.07, bk[11][1], '12', fontsize=6.2, color=INK2, ha='left', va='center')
    abv = series[2][1]
    assert max(abv[8][1], abv[9][1], abv[10][1]) < zy[0] * 1.6 and min(b for b, _ in abv[8:11]) > zx[0]
    ins.text(abv[9][0] * 1.07, abv[9][1], '9–11', fontsize=6.2, color=INK2, ha='left', va='center')
    ins.text(3.2, 1.33, 'HiCMAE', fontsize=6.2, color=INK2, ha='center', va='bottom')
    # the fourth HiCMAE map (2nd block, video query) sits on ABVnet block 11: its own label just right of it,
    # slightly low so it stays under the green line that leaves block 11
    lone = min(zip(hb, hk), key=lambda p: p[1])
    ins.text(abv[10][0] * 1.055, lone[1] / 1.04, 'HiCMAE', fontsize=6.2, color=INK2, ha='left', va='center')
    ins.set_xscale('log')
    ins.set_yscale('log')
    ins.set_xlim(*zx)
    ins.set_ylim(*zy)
    ins.xaxis.set_major_locator(FixedLocator([2, 3, 5]))
    ins.yaxis.set_major_locator(FixedLocator([0.5, 1.0]))
    ins.set_xticklabels(['2', '3', '5'])
    ins.set_yticklabels(['0.5', '1'])
    ins.yaxis.tick_right()          # the zoom's y labels on its right, apart from the main axis's labels
    for axis in (ins.xaxis, ins.yaxis):
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    ins.tick_params(labelsize=6.2, length=2, pad=1.5)
    ins.grid(False)
    for sp in ('top', 'right'):
        ins.spines[sp].set_visible(True)
    for sp in ins.spines.values():
        sp.set_edgecolor(MUTED)
        sp.set_linewidth(0.6)
    ins.set_title('zoom; numbers = transformer block', fontsize=6.2, color=INK2, pad=2, loc='left')
    ind = ax.indicate_inset_zoom(ins, edgecolor=MUTED, lw=0.6, alpha=1)
    # keep only the connector between the upper-right corners (index 3); the automatic lower one ran from the
    # zoom's lower-left corner to the box's and passed just above ABVnet blocks 6-8
    for i, cline in enumerate(ind.connectors):
        cline.set_linewidth(0.6)
        cline.set_color(MUTED)
        cline.set_visible(i == 3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_box_aspect(1)
    ax.set_xlabel('Bound β = ‖B‖ σ$_v$ σ$_a$ / ε (log scale)')
    ax.set_ylabel('Measured cost interaction κ (log scale)')
    ax.set_title('κ stays below its bound β at every block and map', loc='left', fontsize=8, pad=35)
    ax.annotate(f'router {min(gap_r):.1f}–{max(gap_r):.1f}× below the bound, '
                f'HiCMAE {min(gap_h):.1f}–{max(gap_h):.1f}× below', (0, 1), xycoords='axes fraction',
                xytext=(0, 25.5), textcoords='offset points', fontsize=6.4, color=MUTED, ha='left', va='bottom')
    # legend rows: the two routers on the MMA-DFER host, then ABVnet's router and HiCMAE (fills column-wise)
    order = [0, 2, 1, 3]
    ax.legend([handles[i][0] for i in order], [handles[i][1] for i in order], loc='lower left',
              bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=6.4, handlelength=1.0, handletextpad=0.4,
              columnspacing=1.0, labelspacing=0.3, borderaxespad=0.2)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_e2_bound.{ext}'), bbox_inches='tight', dpi=220)


if __name__ == '__main__':
    perclip()
    bound()
    print('->', OUT)
