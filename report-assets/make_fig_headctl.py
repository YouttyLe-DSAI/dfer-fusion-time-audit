#!/usr/bin/env python
"""Paper figure (Section 4.4, fig:headctl): head attenuation control (prereg "Head attenuation control", results/headctl).
Writes paper-v2/figures/fig_headctl.{pdf,png}.

Rows: for each setting (router on CREMA-D, router on DFEW), three frame-specific changes added to the unmodified input of the
temporal transformer:
  - the routing's own frame-specific change (ctl_frameonly: the frame-specific part of the change that the product replacement at
    blocks 10-12 causes in the same clip), solid;
  - the same change with its frames permuted (ctl_perm), hatched;
  - a random frame-specific change of the same size per clip (ctl_rand, mean of three seeds), grey.
(a) Relative change of the frame-specific part of the head's input (pale bar, matched across rows by construction) and relative
    change of the head's output (coloured bar), pooled over clips; triangle above the routing's row, labelled directly: the full
    product replacement at blocks 10-12 (indep_10_12; 0.89% / 0.49%). The two use different reference norms (scripts/blocksub_ot.py,
    trace): in_frame = ||change of the frame-specific part|| / ||frame-specific variation of x0|| (RMS over frames),
    out = ||o' - o|| / ||o||, the full output norm. Both are named in the figure.
(b) out / in_frame, the ratio itself (0.075 / 0.044 / 0.017 on CREMA-D, 0.074 / 0.049 / 0.022 on DFEW, as analysis.tex and the
    summary's "attenuation" quote it; checked below): bar = ratio of the changes pooled over clips, dots = single folds in a lane
    under the bar, nudged up or down (deterministically) where two folds would touch, so all five show. Because the two changes are
    relative to different norms, (b) is not a gain; only the comparison between rows (which share the norms) is free of this
    choice. Header rows: the registered ratio R = out(random) / out(routing's frame-specific change); note: the registered rule
    R <= 2 = generic attenuation, and R < 1 (not anticipated by the rule) = the random change is attenuated more. For DFEW also R
    after bounding the effect of the clip-mean change left by bf16 rounding (prereg results entry: 0.28; explained in the caption).
The clip-mean arm (ctl_meanonly) changes no frame-specific part and so has no place on these axes; its output change (0.33% on
CREMA-D, 0.17% on DFEW; checked below) is stated in the caption.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtrans  # noqa: E402

import make_fig_e2 as F  # noqa: E402  (style: rcParams, ink tokens, palette)

ROOT, OUT = F.ROOT, F.OUT
INK, INK2, MUTED, GRID, AXIS, SURF, BAND = F.INK, F.INK2, F.MUTED, F.GRID, F.AXIS, F.SURF, F.BAND
GREY = '#9a998f'
NOTE = 6.4                                   # value labels and notes (pt)
CLEAR = dict(fc=F.SURF, ec='none', pad=0.4)  # keeps grid lines out of free-standing notes
# paper-wide router roles (fig_e2_profiles): router on CREMA-D blue, router on DFEW orange
SETTINGS = [('CREMAD_router', 'Router, CREMA-D', F.SLOT[0]), ('DFEW_A0_s1', 'Router, DFEW', F.SLOT[1])]
ARMS = [('ctl_frameonly', "routing's change"), ('ctl_perm', "routing's, frames permuted"),
        ('ctl_rand', 'random, same size')]


def pooled(A, arm, key):
    if arm == 'ctl_rand':
        return sum(A[f'ctl_rand{s}'][key] for s in range(3)) / 3
    return A[arm][key]


def fold_shares(name, arm):
    d = os.path.join(ROOT, 'results', 'headctl', name)
    out = []
    for f in range(1, 6):
        t = json.load(open(os.path.join(d, f'fold{f}.json')))['trace']
        out.append(pooled(t, arm, 'out') / pooled(t, arm, 'in_frame'))
    return out


def bounded_R(A):
    """R after removing the bound on the output change caused by the arms' residual clip-mean change (bf16 rounding),
    using ctl_meanonly's sensitivity out / in_mean (prereg results entry "Head attenuation control")."""
    sens = A['ctl_meanonly']['out'] / A['ctl_meanonly']['in_mean']
    rand = pooled(A, 'ctl_rand', 'out') - sens * pooled(A, 'ctl_rand', 'in_mean')
    frame = A['ctl_frameonly']['out'] - sens * A['ctl_frameonly']['in_mean']
    return rand / frame


def draw_bar(ax, y, w, arm, col, h):
    if arm == 'ctl_perm':
        ax.barh(y, w, height=h, color=SURF, edgecolor=col, hatch='//////', linewidth=0.8, zorder=3)
    else:
        ax.barh(y, w, height=h, color=col if arm == 'ctl_frameonly' else GREY, edgecolor='none', zorder=3)


def main():
    S = json.load(open(os.path.join(ROOT, 'results', 'headctl', 'summary.json')))
    c, d = S['CREMAD_router'], S['DFEW_A0_s1']
    # numbers the paper quotes (analysis.tex, prereg results entry) and the caption states
    assert f"{c['R']:.2f}" == '0.23' and f"{d['R']:.2f}" == '0.30'
    assert f"{bounded_R(c['arms']):.2f}" == '0.22' and f"{bounded_R(d['arms']):.2f}" == '0.28'
    assert f"{c['attenuation']['ctl_frameonly']:.3f}" == '0.075' and f"{c['attenuation']['ctl_rand0']:.3f}" == '0.017'
    assert f"{pooled(c['arms'], 'ctl_rand', 'out') / pooled(c['arms'], 'ctl_rand', 'in_frame'):.3f}" == '0.017'
    assert f"{100 * c['arms']['ctl_frameonly']['out']:.2f}" == '0.88' and f"{100 * c['arms']['indep_10_12']['out']:.2f}" == '0.89'
    assert f"{100 * d['arms']['indep_10_12']['out']:.2f}" == '0.49'
    assert f"{100 * pooled(c['arms'], 'ctl_rand', 'out'):.2f}" == '0.20' and f"{c['R_perm']:.2f}" == '0.59'
    assert f"{100 * c['arms']['ctl_frameonly']['in_frame']:.1f}" == '11.7'
    assert f"{100 * d['arms']['ctl_frameonly']['in_frame']:.1f}" == '6.4'
    # clip-mean arm, stated in the caption only (the prereg's "0.34%" rounds the table's 0.335; unrounded 0.3347)
    assert f"{100 * c['arms']['ctl_meanonly']['out']:.2f}" == '0.33' and f"{100 * d['arms']['ctl_meanonly']['out']:.2f}" == '0.17'

    # y layout: a header slot per setting, then its three rows
    ypos, heads, y = {}, [], 0.0
    for gi, (name, head, col) in enumerate(SETTINGS):
        if gi:
            y += 0.45
        heads.append((y, head, col, name))
        y += 0.95
        for arm, _ in ARMS:
            ypos[(name, arm)] = y
            y += 1.0
    ybot = y - 0.3

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.95), sharey=True,
                               gridspec_kw={'width_ratios': [1.1, 1.0], 'wspace': 0.1})
    fig.subplots_adjust(left=0.222, right=0.985, top=0.83, bottom=0.215)
    for ax in (a, b):
        ax.set_ylim(ybot, -0.7)
        ax.set_yticks([])
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='y', visible=False)
        ax.axhline(heads[1][0] - 0.45 / 2 - 0.25, color=GRID, lw=0.6, zorder=0)
    tr = mtrans.blended_transform_factory(fig.transFigure, a.transData)
    for yh, head, col, _ in heads:
        fig.text(0.008, yh + 0.05, head, transform=tr, ha='left', va='center', fontsize=7.3, color=INK, fontweight='semibold')
    for (name, arm), yk in ypos.items():
        fig.text(0.022, yk, dict(ARMS)[arm], transform=tr, ha='left', va='center', fontsize=7.1, color=INK2)

    H = 0.52
    # (a) input change (pale, % of the input's frame-specific variation) and output change (coloured, % of the output norm)
    for name, _, col in SETTINGS:
        A = S[name]['arms']
        crema = name.startswith('CREMAD')
        for arm, _ in ARMS:
            yk = ypos[(name, arm)]
            xin, xout = 100 * pooled(A, arm, 'in_frame'), 100 * pooled(A, arm, 'out')
            a.barh(yk, xin, height=H, color=BAND, edgecolor=AXIS, linewidth=0.6, zorder=2)
            draw_bar(a, yk, xout, arm, col, H)
            # output label: the denominator is named on each setting's first row
            a.text(xout + 0.2, yk, f'{xout:.2f}%' + (' of output norm' if arm == 'ctl_frameonly' else ''), ha='left',
                   va='center', fontsize=NOTE, color=INK, zorder=4)
            if arm == 'ctl_perm':               # input sizes are matched across a setting's rows: label them once, mid-group
                if crema:                       # long pale bar: inside, right-aligned
                    a.text(xin - 0.2, yk, f'input {xin:.1f}% of its frame-specific variation', ha='right', va='center',
                           fontsize=NOTE, color=INK2, zorder=4)
                else:                           # shorter pale bar: just after it, on two lines
                    a.text(xin + 0.2, yk, f'input {xin:.1f}% of its\nframe-specific variation', ha='left', va='center',
                           fontsize=NOTE, color=INK2, zorder=4, linespacing=1.0, bbox=CLEAR)
        # the full product replacement at blocks 10-12, a triangle above the routing's row, labelled directly
        xi, yv = 100 * A['indep_10_12']['out'], ypos[(name, 'ctl_frameonly')] - H / 2 - 0.24
        a.plot(xi, yv, marker='v', ms=4.2, color=INK2, ls='', zorder=5)
        a.text(xi + 0.32, yv - 0.05, ('full product replacement, blocks 10–12: ' if crema else 'full replacement: ')
               + f'{xi:.2f}%', ha='left', va='center', fontsize=NOTE, color=INK2, zorder=5, bbox=CLEAR)
    a.set_xlim(0, 12.5)
    a.set_xticks([0, 2, 4, 6, 8, 10, 12])
    a.set_xlabel('relative change (%): input vs its frame-specific\nvariation, output vs its norm')
    a.set_title('(a) A frame-specific input change of 6–12%\nmoves the head\'s output by under 1%', loc='left')

    # (b) relative output change per unit of relative frame-specific input change (the ratio itself, as the text quotes it);
    # bar = ratio of the changes pooled over clips, dots = folds in a lane under the bar, nudged apart where they would touch
    XB = 0.112
    b.set_xlim(0, XB)
    b.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.10])
    b.set_xticklabels(['0', '0.02', '0.04', '0.06', '0.08', '0.10'])
    ppu = b.get_position().width * fig.get_figwidth() * 72 / XB         # points per unit of the ratio
    DOT, DY, SEP = H / 2 + 0.22, 0.13, 2.9                                # lane centre, nudge (rows), min gap (pt)
    att = {'ctl_frameonly': 'ctl_frameonly', 'ctl_perm': 'ctl_perm', 'ctl_rand': 'ctl_rand0'}
    for name, _, col in SETTINGS:
        A = S[name]['arms']
        for arm, _ in ARMS:
            yk = ypos[(name, arm)]
            sh = pooled(A, arm, 'out') / pooled(A, arm, 'in_frame')
            # value labels as quoted in the text and the summary (rand: mean of three seeds; agrees with seed 0 to 3 dp)
            assert f'{sh:.3f}' == f"{S[name]['attenuation'][att[arm]]:.3f}", (name, arm)
            draw_bar(b, yk, sh, arm, col, H)
            b.text(sh + 0.0018, yk, f'{sh:.3f}', ha='left', va='center', fontsize=NOTE, color=INK, zorder=4)
            # deterministic nudge: a fold closer than SEP pt to a fold already placed in its lane moves up or down
            placed = []
            for v in sorted(fold_shares(name, arm)):
                for lane in (0, -1, 1):
                    if all(abs(v - u) * ppu >= SEP for u, l in placed if l == lane):
                        break
                placed.append((v, lane))
            b.plot([v for v, _ in placed], [yk + DOT + DY * l for _, l in placed], ls='', marker='o', ms=2.4,
                   color=INK2, mec=SURF, mew=0.3, alpha=0.9, zorder=4)
        # registered ratio R = out(random) / out(routing's frame-specific change), in the setting's header row
        yh = [h[0] for h in heads if h[3] == name][0]
        if name == 'CREMAD_router':
            txt = f"R = output change, random ÷ routing's = {S[name]['R']:.2f}"
        else:
            txt = f"R = {S[name]['R']:.2f}  ({bounded_R(A):.2f} with bf16 rounding bounded)"
        b.text(0.0, yh + 0.05, txt, ha='left', va='center', fontsize=NOTE, color=INK)
    # what R means under the registered rule, in the free space right of CREMA-D's lower rows
    b.text(0.0605, (ypos[('CREMAD_router', 'ctl_perm')] + ypos[('CREMAD_router', 'ctl_rand')]) / 2 + 0.1,
           'registered rule: R ≤ 2\n→ generic attenuation;\nR < 1 here: the random\nchange attenuated more',
           ha='left', va='center', fontsize=NOTE, color=MUTED, linespacing=1.12, bbox=CLEAR)
    # what bar and dots are, in the free space right of DFEW's lower rows (the bar is not the mean of the dots)
    b.text(0.0605, ypos[('DFEW_A0_s1', 'ctl_rand')] + 0.1,
           'bar: changes pooled\nover clips; dots: folds', ha='left', va='center', fontsize=NOTE, color=MUTED,
           linespacing=1.12, bbox=CLEAR)
    b.set_xlabel('relative output change ÷ relative\nframe-specific input change')
    b.set_title('(b) The routing\'s change passes better\nthan a random one', loc='left')
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_headctl.{ext}'), bbox_inches='tight', dpi=220)
    print('->', os.path.join(OUT, 'fig_headctl.pdf'))
    for name, *_ in SETTINGS:
        A = S[name]['arms']
        print(name, {arm: (round(100 * pooled(A, arm, 'in_frame'), 2), round(100 * pooled(A, arm, 'out'), 3),
                           round(pooled(A, arm, 'out') / pooled(A, arm, 'in_frame'), 4),
                           [round(v, 4) for v in fold_shares(name, arm)]) for arm, _ in ARMS},
              'R', round(S[name]['R'], 3), 'bounded', round(bounded_R(A), 3), 'indep out', round(100 * A['indep_10_12']['out'], 3),
              'meanonly out', round(100 * A['ctl_meanonly']['out'], 3))


if __name__ == '__main__':
    main()
