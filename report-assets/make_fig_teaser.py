#!/usr/bin/env python
"""Fig. 1 (Introduction): the paper's question and main answer at a glance.

(a) The product replacement on real data. The CREMA-D test clip 1006_IOM_ANG_XX (label anger; the clip with the median
    residual at block 12, the declared p50 rule of results/plandump, as in Figs. architecture and plan decomposition):
    the frames the test loader reads (four of the 16 are shown, beside plan rows 2, 6, 10 and 14) and the loader's Kaldi
    filterbank frame the router's real transport plan pi at block 12, the last of the 12 blocks (rows = the 16 frames,
    columns = the 21 real audio slots). On the right, what each of the four frames receives from the audio: its mixture
    pi_j. / r_j (bars), against the single mixture c / S that every frame receives under the product
    pi' = r c^T / S (dashed). The frame thumbnails and the filterbank come from make_fig_architecture, so Figs. 1 and 2
    show the same input.
(b) The CREMA-D contrast of Table tab:contrast: change in UAR under the product replacement and under removal of the
    message, HiCMAE vs the router, mean +/- SD over the five folds with the fold values as dots (placed above or
    below the whisker, never on it). The key of (a) names the plotted quantities exactly; the dashed line is labelled
    1/21 (in %), with a note that it is flat only because the router's plan is balanced.

Reads  results/plandump/MMA_CREMAD.npz, report-assets/examples_cremad/1006_IOM_ANG_XX{/,.wav},
       p1/results/cremad_of/nulls/summary.json (HiCMAE), results/p2_ot/CREMAD/summary.json (router).
Writes paper-v2/figures/fig_teaser.{pdf,png}.

    cd report-assets && <mplenv>/bin/python make_fig_teaser.py
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

import make_fig_e2 as F  # noqa: E402  (shared style: rcParams, ink colours, SLOT palette)
import make_fig_architecture as A  # noqa: E402  (the loader's frame sampling and filterbank, as drawn in Fig. 2)
from make_fig_plans import SEQ  # noqa: E402  (the plan's sequential map in Figs. 2 and 3)

ROOT, OUT = F.ROOT, F.OUT
INK, INK2, MUTED, SURF = F.INK, F.INK2, F.MUTED, F.SURF
CLIP, K_CLIP, BLOCK = A.CLIP, 1, 12
ROWS = (2, 6, 10, 14)                  # plan rows (1-based) whose frames are shown: one per quarter of the clip
ROUTER, HIC = F.SLOT[0], INK2          # paper-wide colour roles (router on CREMA-D blue, HiCMAE dark neutral)
TABLE = {('hicmae', 'indep'): (-2.58, 1.21), ('hicmae', 'zero'): (-0.32, 0.57),     # Table tab:contrast
         ('router', 'indep'): (-0.13, 0.08), ('router', 'zero'): (-3.98, 1.30)}
MINUS = '−'


# ----------------------------------------------------------------------------------------------------------- data
def plan():
    z = np.load(os.path.join(ROOT, 'results', 'plandump', 'MMA_CREMAD.npz'))
    assert str(z['names'][K_CLIP]) == CLIP, z['names']
    assert A.CLASSES[int(z['labels'][K_CLIP])] == 'anger'
    i = list(z['blocks']).index(BLOCK)
    assert BLOCK == int(max(z['blocks']))                             # block 12 is the last block
    u = z['usable'][K_CLIP, i]
    pi = z['pi'][K_CLIP, i].astype(float)[:, u]                     # padded audio slots carry no mass: drop them
    r, c, s = pi.sum(1), pi.sum(0), pi.sum()
    prod = np.outer(r, c) / s                                         # row sums x column sums / total
    assert np.abs(prod - (pi - z['D'][K_CLIP, i].astype(float)[:, u])).max() < 1e-8   # agrees with the dumped residual
    mix = pi / r[:, None]                                             # frame j's audio mixture pi_j. / r_j
    shared = c / s                                                    # the mixture every frame gets under the product
    assert np.allclose(prod / r[:, None], shared[None, :])            # every row of pi' / r is c / S
    return pi, mix, shared, float(np.ptp(shared) / shared.mean())


def frames():
    ims, idx, n = A.sampled_frames()                                  # the 16 frames the test loader reads
    return [ims[j - 1] for j in ROWS], [idx[j - 1] for j in ROWS], n


def contrast():
    """Per-fold change in UAR (points) for the product replacement ('indep') and removal ('zero'); checked against the
    rounded values printed in Table tab:contrast (mean +/- sample SD over the five folds)."""
    src = {'hicmae': ('p1', 'results', 'cremad_of', 'nulls', 'summary.json'),
           'router': ('results', 'p2_ot', 'CREMAD', 'summary.json')}
    out = {}
    for model, p in src.items():
        pf = json.load(open(os.path.join(ROOT, *p)))['per_fold_d_uar']
        for arm in ('indep', 'zero'):
            v = np.asarray(pf[arm], dtype=float)
            m, sd = v.mean(), v.std(ddof=1)
            want = TABLE[(model, arm)]
            if (round(m, 2), round(sd, 2)) != want:
                raise SystemExit(f'{model}/{arm}: files give {m:.3f} +/- {sd:.3f}, Table tab:contrast says {want}')
            out[(model, arm)] = (v, m, sd)
    return out


# ----------------------------------------------------------------------------------------------------------- helpers
def ax_in(fig, x, y, w, h, W, H):
    """Axes from a rectangle in inches (x, y = lower-left corner)."""
    return fig.add_axes([x / W, y / H, w / W, h / H])


def bare(ax):
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def line_in(fig, x0, y0, x1, y1, W, H, **kw):
    fig.add_artist(Line2D([x0 / W, x1 / W], [y0 / H, y1 / H], transform=fig.transFigure, **kw))


# ----------------------------------------------------------------------------------------------------------- panel (a)
def panel_a(fig, W, H):
    pi, mix, shared, flat = plan()
    nf, ns = pi.shape
    wave, sr, E = A.fbank()
    assert int(np.ceil(E.shape[0] / A.ROWS_PER_SLOT)) == ns          # 325 filterbank rows -> 21 real slots
    imgs, idx, nfr = frames()

    hh, hw = 1.54, 1.12           # plan height and width (in); one row = 0.096 in
    th = hh / 4                   # thumbnail: four plan rows high, square
    sh = 0.29                     # spectrogram height
    y0 = 0.60                     # bottom of the plan
    yt = y0 + hh                  # top of the plan
    x_th = 0.17
    x_pl = x_th + th + 0.06
    x_mx = x_pl + hw + 0.34       # left edge of the per-frame mixtures
    w_mx = 1.36
    rowh = hh / nf
    y_row = lambda j: yt - (j - 0.5) * rowh                         # noqa: E731  centre of plan row j (1-based)
    y_head = H - 0.31             # baseline row of the column headers

    # --- video: four of the frames the loader reads, each beside four plan rows; a tick marks its exact row
    a_th = ax_in(fig, x_th, y0, th, hh, W, H)
    for q, im in enumerate(imgs):
        a_th.imshow(im, extent=(0, 1, 4 * q + 3.94, 4 * q + 0.06), aspect='auto', interpolation='antialiased')
    a_th.set_xlim(0, 1)
    a_th.set_ylim(nf, 0)
    bare(a_th)
    for j in ROWS:
        line_in(fig, x_th + th + 0.005, y_row(j), x_pl - 0.005, y_row(j), W, H, color=INK, lw=0.8,
                solid_capstyle='butt')
    fig.text(0.075 / W, (y0 + hh / 2) / H, f'{nf} frames (rows)', rotation=90, ha='center', va='center',
             fontsize=6.6, color=INK2)

    # --- audio: the loader's filterbank over the real slots (rows beyond the real ones are empty)
    a_sp = ax_in(fig, x_pl, yt + 0.04, hw, sh, W, H)
    a_sp.imshow(E.T, origin='lower', aspect='auto', cmap=A.SPEC_CM, vmin=np.percentile(E, 2), vmax=E.max(),
                extent=(0, E.shape[0] / A.ROWS_PER_SLOT, 0, 1), interpolation='antialiased')
    a_sp.set_xlim(0, ns)
    a_sp.set_ylim(0, 1)
    bare(a_sp)
    a_sp.add_patch(Rectangle((0, 0), ns, 1, fill=False, ec=F.AXIS, lw=0.6))
    fig.text((x_th + th / 2) / W, (yt + 0.04 + sh / 2) / H, f'audio\n{len(wave) / sr:.1f} s', ha='center',
             va='center', fontsize=6.4, color=INK2, linespacing=1.1)

    # --- the router's plan, with the four rows of the shown frames outlined
    norm = plt.Normalize(0, pi.max())
    a_pl = ax_in(fig, x_pl, y0, hw, hh, W, H)
    a_pl.imshow(pi, cmap=SEQ, norm=norm, aspect='auto', interpolation='nearest', extent=(0, ns, nf, 0))
    bare(a_pl)
    a_pl.set_xlim(0, ns)
    a_pl.set_ylim(nf, 0)
    a_pl.add_patch(Rectangle((0, 0), ns, nf, fill=False, ec=F.AXIS, lw=0.6))
    for j in ROWS:
        a_pl.add_patch(Rectangle((0, j - 1), ns, 1, fill=False, ec=INK, lw=0.7, zorder=4))
    a_pl.set_xlabel(f'{ns} audio slots (time)', fontsize=6.6, labelpad=2)
    fig.text(x_th / W, y_head / H, f'Router\'s plan $\\pi$, block {BLOCK} of {BLOCK}', ha='left',
             va='baseline', fontsize=7.0, color=INK)

    # colour bar: transport mass per cell, horizontal, under the plan's x label
    y_cb = 0.19
    a_cb = ax_in(fig, x_pl, y_cb, 0.52, 0.055, W, H)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), cax=a_cb, orientation='horizontal')
    a_cb.grid(False)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(F.AXIS)
    cb.set_ticks([0, 0.005, 0.010, 0.015])
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 1e3:g}'))
    cb.ax.tick_params(labelsize=6.0, length=2, width=0.6, pad=1.2)
    fig.text((x_pl + 0.52 + 0.06) / W, (y_cb + 0.0275) / H, 'mass per cell\n(×10$^{-3}$)', ha='left',
             va='center', fontsize=6.2, color=INK2, linespacing=1.1)

    # --- what each shown frame receives: its own mixture (bars) vs the one mixture of the product (dashed)
    ymax = 100 * max(mix[j - 1].max() for j in ROWS) * 1.12
    axes = []
    for q, j in enumerate(ROWS):
        ytop = yt - q * th
        ax = ax_in(fig, x_mx, ytop - th + 0.045, w_mx, th - 0.09, W, H)
        axes.append(ax)
        ax.bar(np.arange(1, ns + 1), 100 * mix[j - 1], width=0.74, color=ROUTER, lw=0, zorder=2)
        ax.plot([0.4, ns + 0.6], [100 * shared.mean()] * 2, color=INK, lw=0.9, ls=(0, (3.2, 1.6)), zorder=3)
        ax.set_xlim(0.4, ns + 0.6)
        ax.set_ylim(0, ymax)
        ax.set_yticks([0, 10, 20])
        ax.yaxis.tick_right()
        ax.tick_params(axis='y', labelsize=6.0, length=2, width=0.6, pad=1.2)
        ax.set_xticks([1, 11, 21])
        ax.tick_params(axis='x', labelsize=6.0, length=2, width=0.6, pad=1.2)
        if q < len(ROWS) - 1:
            ax.set_xticklabels([])
        ax.grid(axis='x', visible=False)
        ax.grid(axis='y', linewidth=0.6)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(True)
        # connector from the outlined plan row to this frame's mixture, with the row number
        yc = ytop - th / 2
        line_in(fig, x_pl + hw + 0.17, y_row(j), x_mx - 0.03, yc, W, H, color=INK2, lw=0.6)
        fig.text((x_pl + hw + 0.035) / W, y_row(j) / H, f'{j}', ha='left', va='center', fontsize=6.2, color=INK)
    axes[-1].set_xlabel('audio slot', fontsize=6.6, labelpad=1)
    fig.text((x_mx + w_mx + 0.21) / W, (y0 + hh / 2) / H, "share of the frame's mass (%)", rotation=270,
             ha='center', va='center', fontsize=6.4, color=INK2)
    # direct label of the product's mixture c/S (flat at 1/ns, in %), on frame 10's axes: right-aligned at the end of
    # the line, over slots 13-21, whose bars all stay below the line (checked here and against the rendered box below)
    q_lab, lab_slots = ROWS.index(10), (13, ns)
    assert (mix[ROWS[q_lab] - 1, lab_slots[0] - 1:lab_slots[1]] < shared.mean()).all()
    assert abs(100 * shared.mean() - 100 / ns) < 1e-6
    lab = axes[q_lab].text(ns + 0.2, 100 * shared.mean() + 0.8, f'1/{ns} ≈ {100 / ns:.1f}%', ha='right',
                           va='bottom', fontsize=6.0, color=INK)
    lab.set_gid('line_label')
    bb = lab.get_window_extent(fig.canvas.get_renderer()).transformed(axes[q_lab].transData.inverted())
    under = [k for k in range(1, ns + 1) if k + 0.37 > bb.x0 and k - 0.37 < bb.x1]      # bars under the label
    assert bb.y0 > 100 * shared.mean() and all(100 * mix[ROWS[q_lab] - 1, k - 1] < bb.y0 for k in under), \
        (bb, under)
    assert bb.y1 < axes[q_lab].get_ylim()[1], bb
    fig.text(x_mx / W, y_head / H, 'What each frame receives', ha='left', va='baseline', fontsize=7.0, color=INK)

    # key above the mixtures, beside the spectrogram: the two plotted quantities, named exactly
    yk1 = y_head - 0.125
    yk2, yk3 = yk1 - 0.135, yk1 - 0.250
    xk = x_mx + 0.16
    fig.patches.append(Rectangle(((x_mx + 0.01) / W, (yk1 - 0.03) / H), 0.11 / W, 0.06 / H,
                                 transform=fig.transFigure, fc=ROUTER, ec='none'))
    fig.text(xk / W, yk1 / H, "real plan $\\pi$: frame $j$'s mixture $\\pi_{j\\cdot}/r_j$", ha='left',
             va='center', fontsize=6.4, color=INK2)
    line_in(fig, x_mx + 0.005, yk2, x_mx + 0.125, yk2, W, H, color=INK, lw=0.9, ls=(0, (3.2, 1.6)))
    fig.text(xk / W, yk2 / H, "product $\\pi'$: shared mixture $c/S$", ha='left', va='center', fontsize=6.4,
             color=INK2)
    fig.text(xk / W, yk3 / H, 'flat here: the plan is balanced', ha='left', va='center', fontsize=6.2,
             color=MUTED)
    # the plain-language cue, under the mixtures; it also defines the product
    cue = ["under $\\pi' = r\\,c^{\\top}\\!/S$, every frame gets",
           'the same audio mixture; the masses $r$, $c$',
           'and the clip-level message are kept']
    for n, s_ in enumerate(cue):                                      # one baseline per line: even spacing
        fig.text(x_mx / W, (0.298 - 0.122 * n) / H, s_, ha='left', va='baseline', fontsize=6.4, color=INK,
                 gid=f'cue{n}')

    fig.text(0.0, (H - 0.02) / H, '(a) Replacing the coupling by its product', ha='left', va='top',
             fontsize=8.5, color=INK)
    return idx, nfr, len(wave) / sr, flat, [a_pl, a_sp] + axes


# ----------------------------------------------------------------------------------------------------------- panel (b)
def panel_b(fig, W, H, x0):
    """Two stacked axes (one per model) on one x scale; the model headers sit between them, outside the plot area."""
    data = contrast()
    lw_lab, w_num = 0.98, 0.70    # label column and number column (in)
    aw = W - x0 - lw_lab - w_num
    ah, line = 0.66, 0.15         # axes height and header line height (in)
    lab = {'indep': 'product replacement', 'zero': 'message removed'}
    head = {'hicmae': ('HiCMAE', 'jointly trained cross-attention', ['frame-specific coupling load-bearing']),
            'router': ('OT router', 'MMA-DFER host, frozen encoders',
                       ['frame-specific coupling inert;', 'the clip-level message carries the effect'])}
    y_bot = 0.44
    gap = 0.11                                   # from a header's last line to the axes top, and axes to next header
    y_top = y_bot + ah + gap + (len(head['router'][2]) + 1) * line + 0.03   # bottom of the HiCMAE axes
    r = fig.canvas.get_renderer()
    axes = {}
    xpu = aw / (0.9 + 6.6)                       # inches per x unit (x limits below)
    ypu = ah / 2.0                               # inches per y unit (two rows over 2 units)
    rad = 0.5 * np.sqrt(7) / 72                  # dot radius (in) for s=7
    cap = 1.8                                    # whisker cap half-length (pt)
    cand = (-0.14, 0.14, -0.26, 0.26, -0.38, 0.38)   # fold-dot offsets (y units) in order of preference
    assert min(abs(c) for c in cand) * ypu - rad > cap / 72 + 0.002      # no dot reaches the whisker or its caps
    for model, yb in (('hicmae', y_top), ('router', y_bot)):
        ax = ax_in(fig, x0 + lw_lab, yb, aw, ah, W, H)
        axes[model] = ax
        col = HIC if model == 'hicmae' else ROUTER
        for y, arm in enumerate(('indep', 'zero')):
            v, m, sd = data[(model, arm)]
            ax.barh(y, m, height=0.6, color=col, hatch='////' if model == 'hicmae' else '', edgecolor=SURF, lw=0,
                    zorder=2)
            # whisker above the dots, so its caps stay visible
            ax.errorbar(m, y, xerr=sd, fmt='none', ecolor=INK, elinewidth=0.8, capsize=cap, capthick=0.8, zorder=5)
            # fold dots off the whisker row: each fold takes the first offset at which it touches no dot already
            # placed (the outer offsets are used only where folds coincide)
            off = []
            for k in range(v.size):
                def dmin(c):
                    return min([np.hypot((v[k] - v[i]) * xpu, (c - off[i]) * ypu) for i in range(k)] or [9.0])
                ok = [c for c in cand if dmin(c) >= 2 * rad + 0.002]
                off.append(ok[0] if ok else max(cand, key=dmin))
            dist = min([np.hypot((v[a] - v[b]) * xpu, (off[a] - off[b]) * ypu)
                        for a in range(v.size) for b in range(a)])
            if dist < 2 * rad:
                print(f'warning: fold dots touch in {model}/{arm} ({dist:.3f} in)')
            ax.scatter(v, y + np.asarray(off), s=7, marker='o', facecolor=SURF, edgecolor=INK, lw=0.6, zorder=4)
            ax.text(-0.02, y, lab[arm], transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=7.0,
                    color=INK2)
            ax.text(1.03, y, f'{m:.2f} ± {sd:.2f}'.replace('-', MINUS), transform=ax.get_yaxis_transform(),
                    ha='left', va='center', fontsize=6.6, color=INK)
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        ax.set_ylim(1.5, -0.5)
        ax.set_yticks([])
        ax.set_xlim(-6.6, 0.9)
        ax.set_xticks([-6, -4, -2, 0])
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', linewidth=0.6)
        ax.spines['left'].set_visible(False)
        # header: bold model name, a short description, and the verdict on the line(s) below
        name, desc, verdict = head[model]
        y_last = yb + ah + gap                      # centre of the header's last verdict line
        y_name = y_last + len(verdict) * line + 0.015
        t = fig.text(x0 / W, y_name / H, name, ha='left', va='center', fontsize=7.4, color=INK,
                     fontweight='bold')
        xe = t.get_window_extent(r).x1 / fig.dpi
        fig.text((xe + 0.07) / W, y_name / H, desc, ha='left', va='center', fontsize=6.4, color=INK2)
        for n, s in enumerate(verdict):
            fig.text(x0 / W, (y_last + (len(verdict) - 1 - n) * line) / H, s, ha='left', va='center', fontsize=6.8,
                     color=INK, style='italic')
    # HiCMAE's removal row: the mean sits inside the spread of its folds (one SD either side of it crosses zero)
    v, m, sd = data[('hicmae', 'zero')]
    assert m + sd > 0 > m - sd and v.min() < 0 < v.max()
    # the note sits in the free part of the row, left of the leftmost fold, masking only the grid lines
    axes['hicmae'].text(v.min() - 0.30, 1, 'within fold spread', ha='right', va='center', fontsize=6.2,
                        color=MUTED, zorder=3, bbox=dict(boxstyle='square,pad=0.15', fc=SURF, ec='none'))
    axes['hicmae'].set_xticklabels([])
    axes['hicmae'].tick_params(axis='x', length=0)
    axes['hicmae'].spines['bottom'].set_visible(False)
    a = axes['router']
    a.set_xticklabels([MINUS + '6', MINUS + '4', MINUS + '2', '0'])
    a.set_xlabel('Change in UAR (points)   ← larger loss', fontsize=7.0, labelpad=2)
    # key with glyphs, top right (the HiCMAE verdict line is short, so the key sits beside it)
    xr = W - 0.02
    yk2 = y_top + ah + gap
    yk1 = yk2 + line + 0.015
    t1 = fig.text(xr / W, yk1 / H, 'mean ± SD', ha='right', va='center', fontsize=6.4, color=INK2)
    t2 = fig.text(xr / W, yk2 / H, '5 folds', ha='right', va='center', fontsize=6.4, color=INK2)
    x1 = t1.get_window_extent(r).x0 / fig.dpi - 0.05
    line_in(fig, x1 - 0.17, yk1, x1, yk1, W, H, color=INK, lw=0.8)
    for xx in (x1 - 0.17, x1):
        line_in(fig, xx, yk1 - 0.03, xx, yk1 + 0.03, W, H, color=INK, lw=0.8)
    x2 = t2.get_window_extent(r).x0 / fig.dpi - 0.05
    k = fig.add_axes([0, 0, 1, 1], facecolor='none')
    k.set_axis_off()
    k.set_xlim(0, W)
    k.set_ylim(0, H)
    k.scatter([x2 - 0.13, x2 - 0.04], [yk2, yk2], s=7, marker='o', facecolor=SURF, edgecolor=INK, lw=0.6,
              clip_on=False)
    fig.text(x0 / W, (H - 0.02) / H, '(b) Which part each model uses, CREMA-D', ha='left', va='top', fontsize=8.5,
             color=INK)
    return data


# ----------------------------------------------------------------------------------------------------------- checks
def check_text(fig, data_axes):
    """Fail loudly if any text leaves the canvas, sits on a data area of (a), or overlaps another text box."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    boxes = [(t, t.get_text(), t.get_window_extent(r)) for t in texts]
    for t, s, b in boxes:
        if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > W + 0.5 or b.y1 > H + 0.5:
            raise SystemExit(f'text outside the canvas: {s!r} {b}')
        for ax in data_axes:
            if t.axes is ax:
                continue
            if b.overlaps(ax.get_window_extent(r)) and t not in ax.get_xticklabels() + ax.get_yticklabels():
                print(f'warning: text {s!r} overlaps a data area')
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][2], boxes[j][2]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 0 and oy > 1.0:          # boxes include the descent: ignore overlaps under 1 px (0.01 in)
                print(f'warning: text overlap {boxes[i][1]!r} / {boxes[j][1]!r} ({oy:.1f} px)')


def main():
    W, H = 7.2, 2.90
    fig = plt.figure(figsize=(W, H))
    idx, nfr, dur, flat, data_axes = panel_a(fig, W, H)
    data = panel_b(fig, W, H, x0=4.10)
    check_text(fig, data_axes)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_teaser.{ext}'), dpi=220)
    for k, (v, m, sd) in data.items():
        print(k, np.round(v, 2), f'{m:.3f} +/- {sd:.3f}')
    print(f'frames shown: plan rows {ROWS} = loader files {idx} of {nfr}; audio {dur:.2f} s; '
          f'product mixture flat to {flat:.1e} (relative range)')
    print('->', os.path.join(OUT, 'fig_teaser.pdf'))


if __name__ == '__main__':
    main()
