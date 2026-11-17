#!/usr/bin/env python
"""Paper figure (Section 4.5, fig:ablation): the interventions on the cost input on DFEW, one row per arm, grouped by what the
arm changes (descriptor, temperature, training rule), with A0's three seeds on top as the run-to-run reference.

Panel titles carry the message (a) the coupling decides more predictions, (b) ... but no resolved rise in I_a, (c) ... and no
resolved gain in G_a; the quantity names sit in the x-labels.
(a) Share of test predictions that the product replacement changes (pooled over folds; the table's "changed %").
    Band: range over A0's three seeds. Notes: A3 did not reach the cost (kappa within 7% of A0 at every block, prereg E6
    manipulation check); A4 never trains the cost.
(b) I_a = Delta_route,a - Delta_route,ref: 90% paired-fold t interval (line with end caps), mean (marker), folds (dots).
    Band: +- the range of Delta_route over A0's seeds. Registered rule (prereg E3/E6, method Sec. endpoints): an arm raises the
    dependence only if I_a > seed range (mean right of the band) AND its 90% interval lies above 0. The cue in the panel (right of
    the band, Training-rule rows) says so.
(c) G_a = U_a - U_real,ref, same encoding. Band: +- the range of U_real over A0's seeds.
Every interval that excludes 0 ("resolved") carries a "resolved" tag in ink just right of its upper cap (white halo where it
crosses the zero line or a band edge); intervals shorter than the mean marker get a smaller marker and haloed end caps drawn on
top, so no interval is hidden. Legend: mean markers differ in shape by group (key says so); fold dots are smaller and lighter.
References: A0 seed 1 (trained at eps 0.05), except A4 seed k against A0 seed k. Temperature rows evaluate A0 seed 1 at another eps
(eps 0.01 with 100 solver iterations); "trained at eps = 0.02" is a separately trained model with A0's configuration otherwise.
A4: Delta_route with the transport plan switched on at inference against the pooled model as trained, so its I in (b) is
descriptive (A4's registered endpoint is G_train); G for the pooled model as trained (G = -G_train); changed share = pooled vs
switched model. A4 never trains the descriptor or the cost's normalisations (noted in (a)).
Reads results/e345/summary.json and results/e6/summary.json, and checks every interval it draws against the registered values
stored there (E3, E4, E5, E6).

    <mplenv>/bin/python report-assets/make_fig_ablation.py   -> paper-v2/figures/fig_ablation.{pdf,png}
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
# paper-wide arm roles (as make_fig_secondary.py): A0 seeds grey, A1 blue, A2 green, A3 orange, temperature/training amber
COL = {'seed': GREY, 'A1': F.SLOT[0], 'A2': F.SLOT[2], 'A3': F.SLOT[1], 'temp': F.SLOT[3], 'A4': F.SLOT[3]}
MRK = {'seed': 'o', 'A1': 's', 'A2': 'D', 'A3': 'v', 'temp': '^', 'A4': 'p'}
MS, MS_SMALL = 5.0, 3.6                        # mean marker; smaller where the interval is shorter than the marker
CAP = 0.21                                     # end-cap half height (rows)
NOTE = 6.3                                     # size of muted notes (pt)

# group header, then rows: (key, style, label, real arm, product arm, reference key)
GROUPS = [
    ('Run-to-run reference', [
        ('A0s1', 'seed', 'A0  seed 1 (ref., ε = 0.05)', 'A0_s1', 'A0_s1', None),
        ('A0s2', 'seed', 'A0  seed 2', 'A0_s2', 'A0_s2', 'A0s1'),
        ('A0s3', 'seed', 'A0  seed 3', 'A0_s3', 'A0_s3', 'A0s1')]),
    ('Cost descriptor', [
        ('A1', 'A1', 'A1  patch mean', 'A1', 'A1', 'A0s1'),
        ('A2', 'A2', 'A2  learned deviation', 'A2', 'A2', 'A0s1'),
        ('A3', 'A3', 'A3  random deviation', 'A3', 'A3', 'A0s1')]),
    ('Temperature ε', [
        ('e0.2', 'temp', 'A0 at ε = 0.2', 'eps0.2', 'eps0.2', 'A0s1'),
        ('e0.1', 'temp', 'A0 at ε = 0.1', 'eps0.1', 'eps0.1', 'A0s1'),
        ('e0.02', 'temp', 'A0 at ε = 0.02', 'eps0.02', 'eps0.02', 'A0s1'),
        ('e0.01', 'temp', 'A0 at ε = 0.01', 'eps0.01_it100', 'eps0.01_it100', 'A0s1'),
        ('t0.02', 'temp', 'trained at ε = 0.02', 'trained_eps0.02', 'trained_eps0.02', 'A0s1')]),
    ('Training rule', [
        ('A4s1', 'A4', 'A4  seed 1', 'A4_s1_scr', 'A4_s1_native', 'A0s1'),
        ('A4s2', 'A4', 'A4  seed 2', 'A4_s2_scr', 'A4_s2_native', 'A0s2'),
        ('A4s3', 'A4', 'A4  seed 3', 'A4_s3_scr', 'A4_s3_native', 'A0s3')]),
]


def ci(v):
    m = st.mean(v)
    h = T90 * st.stdev(v) / len(v) ** 0.5
    return m, m - h, m + h


def load():
    E = json.load(open(os.path.join(ROOT, 'results', 'e345', 'summary.json')))
    S6 = json.load(open(os.path.join(ROOT, 'results', 'e6', 'summary.json')))
    arms = {**E['arms'], **S6['arms']}
    rows = {}
    for _, members in GROUPS:
        for key, sty, lab, real, prod, ref in members:
            ur = arms[real]['U_real']
            if key.startswith('A4'):           # product arm = the pooled model as trained (its own U_real)
                up, share = arms[prod]['U_real'], E['E5'][key[-1]]['changed_native_vs_switched_real']
            else:
                up, share = arms[prod]['U_product'], arms[real]['changed_pooled']
            rows[key] = {'style': sty, 'label': lab, 'ref': ref, 'real': ur, 'prod': up, 'share': share,
                         'droute': [a - b for a, b in zip(ur, up)]}
    for key, r in rows.items():
        if r['ref'] is None:
            continue
        q = rows[r['ref']]
        mine = r['prod'] if key.startswith('A4') else r['real']   # A4: G for the pooled model as trained
        r['I'] = [a - b for a, b in zip(r['droute'], q['droute'])]
        r['G'] = [a - b for a, b in zip(mine, q['real'])]
    # every interval drawn must equal the registered one in the summaries
    reg = [('A1', 'I', E['E3']['I_A1']), ('A1', 'G', E['E3']['G_A1']), ('A2', 'I', S6['A2']['I']), ('A2', 'G', S6['A2']['G']),
           ('A3', 'I', S6['A3']['I']), ('A3', 'G', S6['A3']['G'])]
    reg += [(k, 'I', E['E4']['DD'][e]) for k, e in (('e0.2', 'eps0.2'), ('e0.1', 'eps0.1'), ('e0.02', 'eps0.02'),
                                                     ('e0.01', 'eps0.01_it100'))]
    for s in '123':
        g = E['E5'][s]['G_train']
        reg.append((f'A4s{s}', 'G', {'mean': -g['mean'], 'ci90': [-g['ci90'][1], -g['ci90'][0]]}))
    for key, what, ent in reg:
        m, lo, hi = ci(rows[key][what])
        assert abs(m - ent['mean']) < 1e-9 and abs(lo - ent['ci90'][0]) < 1e-3 and abs(hi - ent['ci90'][1]) < 1e-3, (key, what)
    for s in '123':
        assert abs(st.mean(rows[f'A4s{s}']['droute']) - E['E5'][s]['Delta_route_A4']['mean']) < 1e-9
    seeds = ('A0s1', 'A0s2', 'A0s3')
    rng_d = max(st.mean(rows[k]['droute']) for k in seeds) - min(st.mean(rows[k]['droute']) for k in seeds)
    rng_u = max(st.mean(rows[k]['real']) for k in seeds) - min(st.mean(rows[k]['real']) for k in seeds)
    assert abs(rng_d - S6['seed_range']) < 1e-9 and abs(rng_d - E['E3']['seed_range']) < 1e-9
    shares = [rows[k]['share'] for k in seeds]
    return rows, rng_d, rng_u, (min(shares), max(shares))


def pt_per_unit(fig, ax):
    """Points per x data unit of an axes, from its position on the figure."""
    x0, x1 = ax.get_xlim()
    return ax.get_position().width * fig.get_figwidth() * 72 / (x1 - x0)


def main():
    rows, rng_d, rng_u, share_rng = load()
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

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 4.35), sharey=True,
                                  gridspec_kw={'width_ratios': [1.0, 1.12, 1.12], 'wspace': 0.12})
    fig.subplots_adjust(left=0.235, right=0.995, top=0.855, bottom=0.11)
    for ax in (a, b, c):
        ax.set_ylim(ybot, -0.75)
        ax.set_yticks([])
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        ax.grid(visible=False)                      # the reference lines (zero, band edges) carry the scale
        ax.tick_params(axis='x', length=2.5, width=0.6)
        for yh, _ in heads[1:]:                     # thin separators between the groups
            ax.axhline(yh - 0.55 / 2 - 0.25, color=GRID, lw=0.6, zorder=0)

    # row labels and group headers in the margin left of panel (a)
    tr = mtrans.blended_transform_factory(fig.transFigure, a.transData)
    for yh, head in heads:
        fig.text(0.008, yh + 0.08, head, transform=tr, ha='left', va='center', fontsize=7.3, color=INK, fontweight='semibold')
    for key, r in rows.items():
        fig.text(0.022, ypos[key], r['label'], transform=tr, ha='left', va='center', fontsize=7.2, color=INK2)

    # (a) share of predictions decided by frame-specific coupling
    ytop = ypos['A0s1'] - 0.42                      # reference lines start below the band labels
    a.axvspan(*share_rng, color=BAND, zorder=0, lw=0)
    a.vlines(share_rng, ytop, ybot, color=AXIS, lw=0.6, zorder=0.5)
    a.text(sum(share_rng) / 2, heads[0][0] - 0.06, 'A0 seed\nrange', ha='center', va='center', fontsize=NOTE, color=INK2,
           linespacing=0.95)
    ref_share = rows['A0s1']['share']
    for key, r in rows.items():
        col = COL[r['style']]
        a.barh(ypos[key], r['share'], height=0.62, color=col, edgecolor='none', zorder=2)
        a.text(r['share'] + 0.03, ypos[key], f"{r['share']:.2f}", ha='left', va='center', fontsize=6.4, color=INK2, zorder=3)
    # the largest share against the reference, written inside its bar so the ratio names what it is relative to
    kmax = max(rows, key=lambda k: rows[k]['share'])
    ratio = rows[kmax]['share'] / ref_share
    assert kmax == 'e0.01' and abs(ratio - 2.67) < 0.01, (kmax, ratio)
    a.text(0.04, ypos[kmax] + 0.02, f'{ratio:.1f} × A0 seed 1', ha='left', va='center', fontsize=6.4, color=INK, zorder=3)
    # A4 never trains the descriptor or the cost's normalisations (its share: plan switched on vs the pooled model)
    a.text(0.62, ypos['A4s2'], 'cost never trained', ha='left', va='center', fontsize=NOTE, color=MUTED, style='italic')
    # A3's random branch left the cost interaction within 7% of A0's at every block (results/e6/summary.txt): it never reached
    # the cost, so its share inside the band fits message (a)
    a.text(0.76, ypos['A3'] - 0.12, 'did not reach the cost\n(κ within 7% of A0)', ha='left', va='top', fontsize=NOTE,
           color=MUTED, style='italic', linespacing=1.05)
    a.set_xlim(0, 1.95)
    a.set_xticks([0, 0.5, 1.0, 1.5])
    a.set_xlabel('% of test predictions changed\nby the product replacement')
    a.set_title('(a) Frame-specific coupling\ndecides more predictions', loc='left')

    # (b), (c) paired differences against the reference
    b.set_xlim(-0.72, 1.08)
    b.set_xticks([-0.5, 0, 0.5, 1.0])
    c.set_xlim(-2.2, 1.76)
    c.set_xticks([-2, -1, 0, 1])
    resolved = []
    for ax, what, rng, title, xl in (
            (b, 'I', rng_d, '(b) ... but no resolved rise in\nrouting dependence $I_a$', '$I_a$ (UAR points)\n> 0: more dependence'),
            (c, 'G', rng_u, '(c) ... and no resolved gain\nin recognition $G_a$', '$G_a$ (UAR points)\n> 0: better recognition')):
        ppu = pt_per_unit(fig, ax)
        ax.axvspan(-rng, rng, color=BAND, zorder=0, lw=0)
        ax.vlines([-rng, rng], ytop, ybot, color=AXIS, lw=0.6, zorder=0.5)
        ax.vlines(0, ytop, ybot, color=INK2, lw=0.7, zorder=1)
        ax.text(0, heads[0][0] - 0.06, f'A0 seed range\n±{rng:.2f}', ha='center', va='center', fontsize=NOTE, color=INK2,
                linespacing=0.95)
        x0, x1 = ax.get_xlim()
        for key, r in rows.items():
            yk, col, mk = ypos[key], COL[r['style']], MRK[r['style']]
            if what not in r:                       # the reference itself: zero by definition
                ax.plot(0, yk, marker=mk, ms=MS, mfc=SURF, mec=col, mew=1.0, zorder=4)   # explained in the legend
                continue
            vals = r[what]
            m, lo, hi = ci(vals)
            narrow = (hi - lo) * ppu < MS + 3.0     # interval (pt) shorter than the marker plus a margin
            ax.plot(vals, [yk + 0.27] * 5, ls='', marker='o', ms=2.6, mfc=col, mec=col, mew=0.3, alpha=0.85, zorder=2)
            ax.plot([lo, hi], [yk, yk], color=col, lw=1.5, solid_capstyle='butt', zorder=3)
            caps = ax.vlines([lo, hi], yk - CAP, yk + CAP, color=col, lw=0.8, zorder=5 if narrow else 3)
            if narrow:
                caps.set_path_effects([pe.withStroke(linewidth=1.9, foreground=SURF)])
            ax.plot(m, yk, marker=mk, ms=MS_SMALL if narrow else MS, color=col, mec=SURF, mew=0.6, zorder=4)
            if lo > 0 or hi < 0:                    # resolved: the 90% paired-fold interval excludes zero
                resolved.append((what, key, round(m, 3), round(lo, 3), round(hi, 3)))
                ax.text(hi + 0.03 * (x1 - x0), yk, 'resolved', ha='left', va='center', fontsize=NOTE, color=INK2,
                        style='italic', zorder=6, path_effects=[pe.withStroke(linewidth=1.5, foreground=SURF)])
        ax.set_xlabel(xl)
        ax.set_title(title, loc='left')
    # the registered decision rule for the dependence, as a reading cue in (b)
    b.text(rng_d + 0.045, heads[3][0] - 0.3,
           'an arm raises\ndependence only if\nits mean is right of\nthe band and its\ninterval right of 0',
           ha='left', va='top', fontsize=NOTE, color=MUTED, linespacing=1.15)
    # the resolved intervals are exactly these (and none lies above zero)
    assert sorted((w, k) for w, k, *_ in resolved) == sorted(
        [('I', 'e0.2'), ('I', 'e0.1'), ('G', 'A0s2'), ('G', 'e0.2'), ('G', 'e0.1')]), resolved
    assert all(hi < 0 for *_, hi in resolved)

    # encoding key, once, above the panels
    cap_line = Line2D([], [], color=INK2, lw=1.5, marker='|', ms=4.6, mew=0.8)
    h = [Line2D([], [], ls='', marker='o', ms=5.0, color=INK2, mec=SURF, mew=0.6),
         cap_line,
         Line2D([], [], ls='', marker='o', ms=2.2, color=INK2, alpha=0.6),
         Patch(facecolor=BAND, edgecolor=AXIS, lw=0.6),
         Line2D([], [], ls='', marker='o', ms=4.6, mfc=SURF, mec=GREY, mew=1.0)]
    fig.legend(h, ['mean over 5 folds (shape = group)', '90% paired-fold interval', 'single folds', 'range over A0 seeds',
                   'reference (0 by definition)'],
               handler_map={cap_line: HandlerLine2D(numpoints=2)},
               loc='upper right', bbox_to_anchor=(0.995, 1.005), ncol=5, fontsize=6.6, handlelength=1.5, columnspacing=1.2,
               handletextpad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_ablation.{ext}'), bbox_inches='tight', dpi=220)
    print('->', os.path.join(OUT, 'fig_ablation.pdf'),
          f'(A0 seed range: Delta_route {rng_d:.3f}, U_real {rng_u:.3f}, share {share_rng[0]:.3f}-{share_rng[1]:.3f};'
          f' max share {rows[kmax]["share"]:.3f} = {ratio:.2f} x A0 seed 1)')
    for w, k, m, lo, hi in resolved:
        print(f'   resolved {w} {k}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]')


if __name__ == '__main__':
    main()
