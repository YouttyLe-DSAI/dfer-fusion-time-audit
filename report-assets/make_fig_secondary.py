#!/usr/bin/env python
"""Paper figure (Section 4.5, fig:secondary): the change in routing dependence of every DFEW arm against A0 seed 1, measured on
three endpoints side by side, so that the reader sees on which endpoint an arm moves (prereg "Exact product reference, WAR and a
continuous endpoint", results/exactprod/summary.json). Same row layout, labels, colours, markers and encoding as fig_ablation
(group header "Run-to-run reference", hollow reference circle at 0, centred band label, end caps, "resolved" tags). Panel titles
carry the message in the paper's vocabulary: (a) no resolved rise for any arm; (b), (c) resolved rises for A1, A2 (and low eps in
(b)), not for A3.

(a) UAR: I_a = Delta_route,a - Delta_route,A0s1 (UAR points), with the registered solver-marginal product (the exact product of
    the target marginals changes only the A2-A3 contrast; supplementary Table tab:exact). Repeats fig_ablation (b) without the
    training-rule (A4) rows.
(b) True-class log-probability: the same change in Delta_logp, the mean drop in log p(true class) under the replacement (10^-3 nats).
(c) WAR: the same change in Delta_route measured in WAR points (not computed for the inference-temperature settings).
Line with end caps: 90% paired-fold t interval; marker: mean over the five folds; dots: single folds; "resolved": the interval
excludes 0, tagged just right of max(upper end, largest fold) with a white halo where the tag crosses the band (fig_ablation's
rule). Narrow intervals: smaller marker without a white edge (stays visibly filled; hollow = the reference), haloed end caps on top.
Legend labels as fig_ablation's, with "±" on the band entry. Band: +- the range over A0's three seeds of the endpoint's Delta (0.17 UAR points, 0.43 x 10^-3 nats, 0.09 WAR points,
from the unrounded fold means; the prereg results entry quotes 0.0005 nats and 0.10 from the rounded summary.txt values). Registered
rule (cue in (a), caption): an arm changes the dependence only if its interval excludes 0 AND its mean lies outside the band.
Temperature rows evaluate A0 seed 1 at another eps (eps 0.01 with 100 solver iterations); "trained at eps = 0.02" is a separate
model. Every interval drawn is checked against the registered values stored in the summary.

    <mplenv>/bin/python report-assets/make_fig_secondary.py   -> paper-v2/figures/fig_secondary.{pdf,png}
"""

import json
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.transforms as mtrans  # noqa: E402
from matplotlib.legend_handler import HandlerLine2D  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import make_fig_e2 as F  # noqa: E402  (style: rcParams, ink tokens, palette)

ROOT, OUT = F.ROOT, F.OUT
INK, INK2, MUTED, GRID, AXIS, SURF, BAND = F.INK, F.INK2, F.MUTED, F.GRID, F.AXIS, F.SURF, F.BAND
GREY = '#9a998f'
T90 = 2.1318                                   # t_{0.95, 4}: two-sided 90% interval over five paired folds
# paper-wide arm roles (as make_fig_ablation.py): A0 seeds grey, A1 blue, A2 green, A3 orange, temperature/training amber
COL = {'seed': GREY, 'A1': F.SLOT[0], 'A2': F.SLOT[2], 'A3': F.SLOT[1], 'temp': F.SLOT[3]}
MRK = {'seed': 'o', 'A1': 's', 'A2': 'D', 'A3': 'v', 'temp': '^'}
MS, MS_SMALL = 5.0, 3.6                        # mean marker; smaller where the interval is shorter than the marker
CAP = 0.21                                     # end-cap half height (rows)
NOTE = 6.3                                     # size of muted notes (pt)

# group header, then rows: (key, style, label, source) with source ('C', checkpoint) or ('E', inference-eps setting)
GROUPS = [
    ('Run-to-run reference', [
        ('A0s1', 'seed', 'A0  seed 1 (ref., ε = 0.05)', ('C', 'DFEW_A0_s1')),
        ('A0s2', 'seed', 'A0  seed 2', ('C', 'DFEW_A0_s2')),
        ('A0s3', 'seed', 'A0  seed 3', ('C', 'DFEW_A0_s3'))]),
    ('Cost descriptor', [
        ('A1', 'A1', 'A1  patch mean', ('C', 'DFEW_A1')),
        ('A2', 'A2', 'A2  learned deviation', ('C', 'DFEW_A2')),
        ('A3', 'A3', 'A3  random deviation', ('C', 'DFEW_A3'))]),
    ('Temperature ε', [
        ('e0.2', 'temp', 'A0 at ε = 0.2', ('E', 'eps0.2')),
        ('e0.1', 'temp', 'A0 at ε = 0.1', ('E', 'eps0.1')),
        ('e0.02', 'temp', 'A0 at ε = 0.02', ('E', 'eps0.02')),
        ('e0.01', 'temp', 'A0 at ε = 0.01', ('E', 'eps0.01_it100')),
        ('t0.02', 'temp', 'trained at ε = 0.02', ('C', 'DFEW_trained_eps0.02'))]),
]
# endpoint: (field of the per-fold Delta, scale for drawing)
ENDPOINTS = {'uar': ('D_route', 1.0), 'uar_exact': ('D_exact', 1.0), 'logp': ('lp_indep', 1e3), 'war': ('DW_route', 1.0)}


def ci(v):
    m = st.mean(v)
    h = T90 * st.stdev(v) / len(v) ** 0.5
    return m, m - h, m + h


def load():
    S = json.load(open(os.path.join(ROOT, 'results', 'exactprod', 'summary.json')))
    C, E = S['checkpoints'], S['eps']
    ref = C['DFEW_A0_s1']
    rows, rng = {}, {}
    for _, members in GROUPS:
        for key, sty, lab, (kind, name) in members:
            src = C[name] if kind == 'C' else E[name]
            r = {'style': sty, 'label': lab}
            for ep, (field, scale) in ENDPOINTS.items():
                if key == 'A0s1' or field not in src:
                    continue
                r[ep] = [scale * (a - b) for a, b in zip(src[field], ref[field])]
            rows[key] = r
    # seed range: max - min over A0's three seeds of the endpoint's fold-mean Delta (drawn as +- this value)
    for ep, (field, scale) in ENDPOINTS.items():
        means = [st.mean(C[k][field]) for k in ('DFEW_A0_s1', 'DFEW_A0_s2', 'DFEW_A0_s3')]
        rng[ep] = scale * (max(means) - min(means))

    # every interval drawn must equal the registered one in the summary
    R, X, D = S['readings']['by_reference']['D_route'], S['readings']['by_reference']['D_exact'], S['secondary']
    reg = [('A1', 'uar', R['I_A1'], 1), ('A2', 'uar', R['I_A2'], 1), ('A3', 'uar', R['I_A3'], 1),
           ('A1', 'uar_exact', X['I_A1'], 1), ('A2', 'uar_exact', X['I_A2'], 1), ('A3', 'uar_exact', X['I_A3'], 1)]
    reg += [(k, 'uar', R['DD'][e], 1) for k, e in (('e0.2', 'eps0.2'), ('e0.1', 'eps0.1'), ('e0.02', 'eps0.02'),
                                                    ('e0.01', 'eps0.01_it100'))]
    for k, a in (('A1', 'A1'), ('A2', 'A2'), ('A3', 'A3'), ('t0.02', 'trained_eps0.02')):
        reg += [(k, 'logp', D[a]['I_logp_route'], 1e3), (k, 'war', D[a]['I_war_route'], 1)]
    for key, ep, ent, scale in reg:
        m, lo, hi = ci(rows[key][ep])
        assert abs(m - scale * ent['mean']) < 1e-9 * scale, (key, ep)
        assert abs(lo - scale * ent['ci90'][0]) < 1e-6 * scale and abs(hi - scale * ent['ci90'][1]) < 1e-6 * scale, (key, ep)
    assert abs(rng['uar'] - S['readings']['seed_range']['D_route']) < 1e-9
    assert abs(rng['uar_exact'] - S['readings']['seed_range']['D_exact']) < 1e-9
    # seed ranges from the unrounded fold means: 0.17 UAR points (as quoted), 0.43 x 10^-3 nats and 0.094 WAR points (the prereg
    # results entry quotes 0.0005 and 0.10, computed from the rounded summary.txt values 1.3 - 0.8 and +0.08 - (-0.02))
    assert f"{rng['uar']:.2f}" == '0.17' and f"{rng['logp']:.2f}" == '0.43' and f"{rng['war']:.3f}" == '0.094', rng
    return rows, rng


def pt_per_unit(fig, ax):
    """Points per x data unit of an axes, from its position on the figure."""
    x0, x1 = ax.get_xlim()
    return ax.get_position().width * fig.get_figwidth() * 72 / (x1 - x0)


def main():
    rows, rng = load()
    # y layout (top to bottom): a header slot per group, then its rows
    ypos, heads, y = {}, [], 0.0
    for gi, (head, members) in enumerate(GROUPS):
        if gi:
            y += 0.55
        heads.append((y, head))
        y += 1.0
        for key, *_ in members:
            ypos[key] = y
            y += 1.0
    ybot = y - 0.4
    ytop = ypos['A0s1'] - 0.42                      # reference lines start below the band labels

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 3.8), sharey=True,
                                  gridspec_kw={'width_ratios': [1.0, 1.08, 1.0], 'wspace': 0.12})
    fig.subplots_adjust(left=0.205, right=0.995, top=0.825, bottom=0.125)
    for ax in (a, b, c):
        ax.set_ylim(ybot, -0.75)
        ax.set_yticks([])
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        ax.grid(visible=False)                      # the reference lines (zero, band edges) carry the scale
        ax.tick_params(axis='x', length=2.5, width=0.6)
        for yh, _ in heads[1:]:                     # thin separators between the groups
            ax.axhline(yh - 0.55 / 2 - 0.25, color=GRID, lw=0.6, zorder=0)

    # row labels and group headers in the margin left of panel (a) (as fig_ablation)
    tr = mtrans.blended_transform_factory(fig.transFigure, a.transData)
    for yh, head in heads:
        fig.text(0.008, yh + 0.08, head, transform=tr, ha='left', va='center', fontsize=7.3, color=INK, fontweight='semibold')
    for key, r in rows.items():
        fig.text(0.022, ypos[key], r['label'], transform=tr, ha='left', va='center', fontsize=7.2, color=INK2)

    panels = (
        (a, 'uar', '(a) UAR: no resolved rise\nfor any arm', '$I_a$ (UAR points)', (-0.72, 1.08), [-0.5, 0, 0.5, 1.0]),
        (b, 'logp', '(b) Log-probability: resolved\nrise for A1, A2, low ε; not A3',
         '$I_a$ ($10^{-3}$ nats)', (-2.0, 8.6), [0, 2, 4, 6, 8]),
        (c, 'war', '(c) WAR: resolved rise for\nA1 and A2, not A3', '$I_a$ (WAR points)', (-0.26, 0.6),
         [-0.2, 0, 0.2, 0.4]),
    )
    resolved = []
    for ax, ep, title, xl, xlim, xt in panels:
        ax.set_xlim(*xlim)
        ax.set_xticks(xt)
        ppu = pt_per_unit(fig, ax)
        r_ = rng[ep]
        ax.axvspan(-r_, r_, color=BAND, zorder=0, lw=0)
        ax.vlines([-r_, r_], ytop, ybot, color=AXIS, lw=0.6, zorder=0.5)
        ax.vlines(0, ytop, ybot, color=INK2, lw=0.7, zorder=1)
        ax.text(0, heads[0][0] - 0.06, f'A0 seed range\n±{r_:.2f}', ha='center', va='center', fontsize=NOTE, color=INK2,
                linespacing=0.95)
        for key, r in rows.items():
            yk, col, mk = ypos[key], COL[r['style']], MRK[r['style']]
            if key == 'A0s1':                       # the reference itself: zero by definition (explained in the legend)
                ax.plot(0, yk, marker=mk, ms=MS, mfc=SURF, mec=col, mew=1.0, zorder=4)
                continue
            if ep not in r:
                continue
            vals = r[ep]
            m, lo, hi = ci(vals)
            narrow = (hi - lo) * ppu < MS + 3.0     # interval (pt) shorter than the marker plus a margin
            ax.plot(vals, [yk + 0.27] * 5, ls='', marker='o', ms=2.6, mfc=col, mec=col, mew=0.3, alpha=0.85, zorder=2)
            ax.plot([lo, hi], [yk, yk], color=col, lw=1.5, solid_capstyle='butt', zorder=3)
            caps = ax.vlines([lo, hi], yk - CAP, yk + CAP, color=col, lw=0.8, zorder=5 if narrow else 3)
            if narrow:
                caps.set_path_effects([pe.withStroke(linewidth=1.9, foreground=SURF)])
            # a narrow row keeps its small marker visibly filled (no white edge: a hollow marker is the reference's code);
            # only the end caps carry the halo
            ax.plot(m, yk, marker=mk, ms=MS_SMALL if narrow else MS, color=col, mec=col if narrow else SURF,
                    mew=0.3 if narrow else 0.6, zorder=4)
            if lo > 0 or hi < 0:                    # resolved: the 90% paired-fold interval excludes zero
                resolved.append((ep, key))
                # tag just right of the interval and of its folds (fig_ablation's rule), haloed where it crosses the band
                ax.text(max(hi, max(vals)) + 0.03 * (xlim[1] - xlim[0]), yk, 'resolved', ha='left', va='center',
                        fontsize=NOTE, color=INK2, style='italic', zorder=6,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground=SURF)])
        ax.set_xlabel(xl + '\n> 0: more dependence')
        ax.set_title(title, loc='left')
    # the registered decision rule, as a reading cue in (a) (as fig_ablation (b))
    a.text(rng['uar'] + 0.06, ypos['A0s2'] - 0.42, 'rise only if mean\nright of band and\ninterval right of 0',
           ha='left', va='top', fontsize=NOTE, color=MUTED, linespacing=1.15)
    # (c): WAR was not computed for the inference-temperature settings
    yc = (ypos['e0.2'] + ypos['e0.01']) / 2
    c.text(rng['war'] + 0.03, yc, 'not computed for\nthese settings', ha='left', va='center', fontsize=NOTE, color=MUTED,
           style='italic', linespacing=1.05)
    # the resolved intervals are exactly these: UAR only the two resolved falls at high eps; log-probability A1, A2, the
    # two high-eps falls, eps 0.01 and the model trained at 0.02; WAR A1, A2 and A0 seed 2 (a fall, mean on the band edge)
    assert sorted(resolved) == sorted([('uar', 'e0.2'), ('uar', 'e0.1'), ('logp', 'A1'), ('logp', 'A2'), ('logp', 'e0.2'),
                                       ('logp', 'e0.1'), ('logp', 'e0.01'), ('logp', 't0.02'), ('war', 'A0s2'),
                                       ('war', 'A1'), ('war', 'A2')]), resolved

    # encoding key, once, above the panels (as fig_ablation)
    cap_line = Line2D([], [], color=INK2, lw=1.5, marker='|', ms=4.6, mew=0.8)
    h = [Line2D([], [], ls='', marker='o', ms=5.0, color=INK2, mec=SURF, mew=0.6),
         cap_line,
         Line2D([], [], ls='', marker='o', ms=2.6, color=INK2, alpha=0.85),
         Patch(facecolor=BAND, edgecolor=AXIS, lw=0.6),
         Line2D([], [], ls='', marker='o', ms=4.6, mfc=SURF, mec=GREY, mew=1.0)]
    # labels as fig_ablation's legend; the band is drawn at +- the seed range, so its entry says "±"
    fig.legend(h, ['mean over 5 folds (shape = group)', '90% paired-fold interval', 'single folds', '± range over A0 seeds',
                   'reference (0 by definition)'],
               handler_map={cap_line: HandlerLine2D(numpoints=2)},
               loc='upper right', bbox_to_anchor=(0.995, 1.005), ncol=5, fontsize=6.6, handlelength=1.5, columnspacing=1.2,
               handletextpad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_secondary.{ext}'), bbox_inches='tight', dpi=220)
    print('->', os.path.join(OUT, 'fig_secondary.pdf'),
          '(seed ranges: ' + ', '.join(f'{k} {v:.4f}' for k, v in rng.items()) + ')')
    for key, r in rows.items():
        print(f'{key:6s}', '  '.join(f'{ep} {ci(r[ep])[0]:+.3f} [{ci(r[ep])[1]:+.3f}, {ci(r[ep])[2]:+.3f}]'
                                      for ep in ENDPOINTS if ep in r))


if __name__ == '__main__':
    main()
