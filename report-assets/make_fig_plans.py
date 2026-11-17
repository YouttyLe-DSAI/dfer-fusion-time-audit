#!/usr/bin/env python
"""Paper figures 2-3 (plan v2 §5) from results/plandump (prereg: example clips = p10/p50/p90 of rho at block 12).

Fig. 2 (fig_plan_decomposition): the decomposition pi = r c^T / S + D, drawn as an equation for the median CREMA-D clip at
        block 12 (plan = product + frame-specific part D; along the frame axis the 16 frames the model receives, i.e. the
        test loader's middle frame of each of 16 equal segments, as in make_fig_architecture.sampled_frames), the per-frame
        departure TV(pi_j/r_j, c/S) whose mean is H, and where D lives: rho = ||D||_F/||pi||_F per block over all test clips
        of the fold, D of the low / high clips at block 12 and of the median clip at block 9. One sequential scale for plan
        and product; one zero-centred diverging scale (grey at zero) for every D panel, spanning the plan's range so that a
        step in the plan and the same step in D look alike, with the floor D >= -r_j c_k/S of each D panel marked (it
        depends on the clip: about -1/(16 n_real)).
        Padded audio slots hatched.
Fig. 3 (fig_cost_interaction): double-centred cost C_int / eps for the same clip at blocks 9 (low kappa) and 12 (high kappa)
        and for the median ABVnet DFEW clip at block 9, on one zero-centred scale coloured by its effect on the plan (blue =
        cheaper than the marginals predict = more mass, as D > 0 in Fig. 2), and plan / product against C_int / eps per cell:
        for a balanced entropic plan log(pi / (r c^T/S)) = -C_int/eps + f_j + g_k exactly, so the largest |C_int|/eps (not its RMS kappa) bounds how far it can move.

    <mplenv>/bin/python report-assets/make_fig_plans.py  -> paper-v2/figures/fig_plan_decomposition, fig_cost_interaction
SEQ is imported by make_fig_architecture.py: keep its name and colours.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

import make_fig_e2 as F  # noqa: E402

ROOT, OUT, INK, INK2, MUTED, SURF = F.ROOT, F.OUT, F.INK, F.INK2, F.MUTED, F.SURF
SEQ = LinearSegmentedColormap.from_list('seq', ['#f7f9fd', '#c6dbf5', '#5a95dd', '#164a8c'])
# zero-centred diverging map: orange (negative) - neutral grey (zero) - blue (positive)
DIV = LinearSegmentedColormap.from_list('div', ['#a8441c', '#eb6834', '#f2b597', '#e3e2dd', '#a9c8ef', '#2a78d6', '#164a8c'])
DIV_CINT = DIV.reversed()          # C_int < 0 (cheaper -> more mass) in blue, as D > 0
GREY = '#9a998f'
PAD_EC = '#cfcec6'                 # hatch colour of the padded audio slots
LOCAL_RC = {'hatch.linewidth': 0.6, 'grid.linewidth': 0.6}  # per figure: importers (make_fig_architecture) keep theirs
EPS = 0.05                         # temperature of all three A0 checkpoints (results/e2/*_test.json "eps")
FR = os.path.join(ROOT, 'report-assets', 'examples_cremad')
FS_T, FS_L, FS_K, FS_A = 7.0, 6.6, 6.0, 6.0   # title, label, tick, annotation font sizes (pt at 7.2 in width)


def load(name):
    return np.load(os.path.join(ROOT, 'results', 'plandump', name + '.npz'))


def axes_in(fig, x, y_top, w, h, **kw):
    """Axes placed in inches from the left and top edges of the figure."""
    W, H = fig.get_size_inches()
    return fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H], **kw)


def real(x, usable):
    return x.astype(float)[:, usable]


def title(ax, head, sub=None, dx=0.0):
    """Two-level panel title: the message in ink, the specifics on a muted line below it (dx: left offset in points)."""
    off = 2.5
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords='axes fraction', xytext=(dx, off), textcoords='offset points', ha='left',
                    va='bottom', fontsize=FS_A, color=INK2)
        off += FS_A * 1.35
    ax.annotate(head, xy=(0, 1), xycoords='axes fraction', xytext=(dx, off), textcoords='offset points', ha='left',
                va='bottom', fontsize=FS_T, color=INK)


def colour_key(ax, items):
    """A one-line colour key in the subtitle slot: [(colour, text), ...], swatch then label, ink text."""
    parts = []
    for i, (col, txt) in enumerate(items):
        da = DrawingArea(5.0, 5.0, 0, 0)
        da.add_artist(Rectangle((0, 0), 5.0, 5.0, facecolor=col, edgecolor='none'))
        parts += [da, TextArea(txt + ('   ' if i < len(items) - 1 else ''), textprops=dict(fontsize=FS_A, color=INK2))]
    box = HPacker(children=parts, align='center', pad=0, sep=2.0)
    ax.add_artist(AnnotationBbox(box, (0, 1), xycoords='axes fraction', xybox=(0, 2.5), boxcoords='offset points',
                                 box_alignment=(0, 0), frameon=False, pad=0, annotation_clip=False))


def heat(ax, x, usable, cmap, norm, xlabel=True, yticks=True, pad_label=False):
    """16 frames x M audio slots on 1-based axes; padded slots (no mass, no cost) hatched."""
    T, M = x.shape
    n = int(usable.sum())
    m = np.ma.masked_array(x.astype(float), mask=np.broadcast_to(~usable, x.shape))
    cm = cmap.copy()
    cm.set_bad(SURF)
    im = ax.imshow(m, aspect='auto', cmap=cm, norm=norm, interpolation='nearest', extent=(0.5, M + 0.5, T + 0.5, 0.5))
    if n < M:
        ax.add_patch(Rectangle((n + 0.5, 0.5), M - n, T, facecolor=SURF, edgecolor=PAD_EC, hatch='////', lw=0, zorder=2))
        ax.axvline(n + 0.5, color=PAD_EC, lw=0.6, zorder=3)
        if pad_label:
            ax.text((n + M + 1) / 2, (T + 1) / 2, 'padding (no audio)', rotation=90, ha='center', va='center',
                    fontsize=FS_A, color=INK2, zorder=4,
                    bbox=dict(boxstyle='round,pad=0.18', fc=SURF, ec='none'))
    ax.set_xlim(0.5, M + 0.5)
    ax.set_ylim(T + 0.5, 0.5)
    ax.set_xticks([1, M // 2, M])
    ax.set_yticks([1, 8, 16])
    if not yticks:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=FS_K, length=2, pad=1.5)
    if xlabel:
        ax.set_xlabel('audio time slot', fontsize=FS_L, labelpad=1.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(F.AXIS)
        s.set_linewidth(0.6)
    return im


def loader_indices(n, T=16):
    """The test loader's frame indices (dataset._test_indices; prereg 'middle frame of each of 16 segments'): the middle
    frame of each of T equal segments of the n extracted frames. Same rule as make_fig_architecture.sampled_frames()."""
    tick = n / float(T)
    return [int(tick / 2.0 + tick * x) for x in range(T)]


def thumbs(ax, clip, T=16, px=72):
    """The 16 frames the model receives (loader_indices over the clip's extracted frames, lexicographic order), centre-
    cropped on the face and stacked along the frame axis."""
    d = os.path.join(FR, clip)
    fs = sorted(f for f in os.listdir(d) if f.endswith('.jpg'))
    idx = loader_indices(len(fs), T)
    from PIL import Image
    tiles = []
    for i in idx:
        im = Image.open(os.path.join(d, fs[i])).convert('RGB')
        w, h = im.size
        c = int(0.72 * min(w, h))                       # centre crop on the face
        x0, y0 = (w - c) // 2, int((h - c) * 0.45)
        tiles.append(np.asarray(im.crop((x0, y0, x0 + c, y0 + c)).resize((px, px), Image.LANCZOS)))
    ax.imshow(np.concatenate(tiles, 0), extent=(0, 1, T + 0.5, 0.5), aspect='auto', interpolation='antialiased')
    for j in range(1, T):
        ax.axhline(j + 0.5, color=SURF, lw=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(T + 0.5, 0.5)
    ax.set_xticks([])
    ax.set_yticks([1, 8, 16])
    ax.tick_params(labelsize=FS_K, length=2, pad=1.5)
    ax.set_ylabel('video frame', fontsize=FS_L, labelpad=2)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    return idx, len(fs)


def milli(cb, label):
    cb.formatter = FuncFormatter(lambda v, _: f'{v * 1e3:g}')
    cb.update_ticks()
    cb.ax.tick_params(labelsize=FS_K, length=2, pad=1.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(F.AXIS)
    cb.set_label(label, fontsize=FS_A, color=INK2, labelpad=2)


@plt.rc_context(LOCAL_RC)
def fig2():
    z = load('MMA_CREMAD')
    blocks = list(z['blocks'])
    i12, i9 = blocks.index(12), blocks.index(9)
    k = 1                                                    # the median clip (p50 of rho at block 12)
    clip = str(z['names'][k])
    u = z['usable'][k, i12]
    pi, D = z['pi'][k, i12].astype(float), z['D'][k, i12].astype(float)
    prod = pi - D                                            # = r c^T / S on the real slots, 0 on padding
    pr = real(pi, u)
    assert np.allclose(real(prod, u), np.outer(pr.sum(1), pr.sum(0)) / pr.sum(), atol=1e-8)
    T, M = pi.shape
    n_real = int(u.sum())
    assert np.allclose(real(prod, u), 1.0 / (T * n_real), rtol=1e-4)   # balanced, uniform marginals: product is flat
    q = pr / pr.sum(1, keepdims=True)                        # each frame's audio mixture pi_jk / r_j
    cs = pr.sum(0) / pr.sum()                                # c / S, the mixture every frame gets under the product
    tv = 0.5 * np.abs(q - cs).sum(1)                         # TV(pi_j/r_j, c/S); H = mean over frames
    H = tv.mean()
    rho = {kk_ii: float(z['rho_sel'][kk_ii]) for kk_ii in [(k, i12), (0, i12), (2, i12), (k, i9)]}
    rows2 = [(0, i12, '(f) Low-$\\rho$ clip, block 12', 'p10 of $\\rho$'),
             (2, i12, '(g) High-$\\rho$ clip, block 12', 'p90 of $\\rho$'),
             (1, i9, '(h) Median clip, block 9', 'clip of (a)–(d)')]
    # floor of D in each D panel: D_jk >= -r_j c_k/S (pi >= 0), so the lowest value D can take is -max_jk r_j c_k/S;
    # = -1/(16 n_real) for uniform marginals. It depends on the clip (n_real = 21, 13, 25 for (c, h), (f), (g)).
    def d_floor(kk, ii):
        uu = z['usable'][kk, ii]
        return float(-real(z['pi'][kk, ii] - z['D'][kk, ii].astype(float), uu).max()), int(uu.sum())
    floors = {'c': d_floor(k, i12), 'f': d_floor(0, i12), 'g': d_floor(2, i12), 'h': d_floor(k, i9)}
    assert abs(floors['c'][0] - floors['h'][0]) < 1e-6      # same clip at blocks 12 and 9: one mark for (c) and (h)
    for kk, ii in [(k, i12), (0, i12), (2, i12), (k, i9)]:
        assert real(z['D'][kk, ii], z['usable'][kk, ii]).min() >= d_floor(kk, ii)[0] - 1e-7
    dabs = {r[:2]: float(np.abs(real(z['D'][r[0], r[1]], z['usable'][r[0], r[1]])).max()) for r in [(k, i12)] + rows2}
    smax = max(pr.max(), real(prod, u).max())
    snorm = Normalize(0, smax)
    dlim = smax                                              # D on the plan's span: same step, similar intensity
    dnorm = TwoSlopeNorm(0, -dlim, dlim)
    n_over = int((np.abs(real(z['D'][2, i12], z['usable'][2, i12])) > dlim).sum())

    fig = plt.figure(figsize=(7.2, 4.50))
    # ---- row 1: plan = product + frame-specific part ------------------------------------------------------------------
    y1, h1, w1 = 0.42, 1.6, 1.30
    a_th = axes_in(fig, 0.30, y1, h1 / T, h1)
    a_pi = axes_in(fig, 0.43, y1, w1, h1)
    a_pr = axes_in(fig, 2.00, y1, w1, h1)
    c_seq = axes_in(fig, 3.36, y1 + 0.12, 0.06, h1 - 0.24)
    a_d = axes_in(fig, 4.02, y1, w1, h1)
    a_tv = axes_in(fig, 5.43, y1, 0.80, h1)
    th_idx, n_files = thumbs(a_th, clip)
    heat(a_pi, pi, u, SEQ, snorm, yticks=False, pad_label=True)
    im_s = heat(a_pr, prod, u, SEQ, snorm, yticks=False)
    im_d = heat(a_d, D, u, DIV, dnorm, yticks=False)
    W, Hf = fig.get_size_inches()
    yc = 1 - (y1 + h1 / 2) / Hf
    fig.text((0.43 + w1 + 2.00) / 2 / W, yc, '=', ha='center', va='center', fontsize=14, color=INK)
    fig.text(3.875 / W, yc, '+', ha='center', va='center', fontsize=14, color=INK)
    title(a_pi, '(a) Plan $\\pi$', f'block 12, median clip, $\\rho$ = {rho[(k, i12)]:.2f}')
    title(a_pr, '(b) Product $r\\,c^{\\top}\\!/S$', 'every frame, same audio mix')
    title(a_d, '(c) Frame-specific part $D$', ' ')            # subtitle slot holds the colour key
    colour_key(a_d, [(DIV(0.86), 'more than product'), (DIV(0.14), 'less')])
    a_pr.text((n_real + 1) / 2, (T + 1) / 2, f'1/(16 × {n_real})\nin every cell', ha='center', va='center',
              fontsize=FS_A, color=INK2)
    cb = fig.colorbar(im_s, cax=c_seq)
    milli(cb, '$\\pi$ and $r\\,c^{\\top}\\!/S$  ($\\times 10^{-3}$)')
    cb.set_ticks([0, 0.005, 0.010, 0.015])
    # per-frame departure, aligned with the rows of D
    yy = np.arange(1, T + 1)
    a_tv.barh(yy, tv, height=0.72, color=GREY, edgecolor='none', zorder=2)
    a_tv.axvline(H, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=3)
    a_tv.text(H + 0.025, T + 0.4, f'mean\n$H$ = {H:.2f}', fontsize=FS_A, color=INK2, ha='left', va='bottom',
              linespacing=1.0)
    a_tv.set_ylim(T + 0.5, 0.5)
    a_tv.set_yticks([1, 8, 16])
    a_tv.set_yticklabels([])
    a_tv.set_xlim(0, 0.65)
    a_tv.set_xticks([0, 0.5])
    a_tv.set_xticklabels(['0', '0.5'])
    a_tv.tick_params(labelsize=FS_K, length=2, pad=1.5)
    a_tv.grid(axis='y', visible=False)
    a_tv.grid(axis='x', lw=0.6)
    a_tv.set_xlabel('TV distance', fontsize=FS_L, labelpad=1.5)
    a_tv.annotate('0 = same mix as product', xy=(0.5, 0), xycoords='axes fraction', xytext=(0, -19.5),
                  textcoords='offset points', ha='center', va='top', fontsize=FS_A, color=MUTED)
    title(a_tv, '(d) Departure per frame', 'from shared mix $c/S$')

    # ---- row 2: where the deviation lives ------------------------------------------------------------------------------
    y2, h2, w2 = 2.82, 1.3, 1.25
    a_r = axes_in(fig, 0.465, y2, 1.52, h2)
    xs2 = [2.42, 3.79, 5.16]
    a_low = [axes_in(fig, x, y2, w2, h2) for x in xs2]
    c_div = axes_in(fig, 6.50, y1 + 0.40, 0.06, y2 + h2 - y1 - 0.62)
    ra = z['rho_all'].astype(float)                          # rho per clip and block, all test clips of fold 1
    bx = np.array(blocks)
    p10, p50, p90 = (np.percentile(ra, p, axis=0) for p in (10, 50, 90))
    a_r.fill_between(bx, p10, p90, color=F.SLOT[0], alpha=0.16, lw=0, zorder=1)
    a_r.plot(bx, p50, color=F.SLOT[0], lw=1.3, marker='o', ms=2.6, mec=SURF, mew=0.5, zorder=3)
    a_r.set_yscale('log')
    a_r.set_ylim(1e-3, 3)
    a_r.set_xlim(0.4, 13.7)
    a_r.set_xticks([1, 3, 5, 7, 9, 12])
    a_r.tick_params(labelsize=FS_K, length=2, pad=1.5)
    a_r.set_xlabel('routing block', fontsize=FS_L, labelpad=1.5)
    a_r.set_ylabel('$\\rho = \\Vert D\\Vert_F/\\Vert\\pi\\Vert_F$ (log scale)', fontsize=FS_L, labelpad=1.5)
    shown = [(2, i12, 'g', 1.65), (1, i12, 'a–d', 1.0), (0, i12, 'f', 0.6), (1, i9, 'h', 1.0)]
    for kk, ii, lab, f in shown:
        v = float(z['rho_sel'][kk, ii])
        a_r.plot(blocks[ii], v, marker='o', ms=3.4, mfc=SURF, mec=INK, mew=0.7, zorder=4)
        if ii == i12:
            a_r.annotate(lab, xy=(blocks[ii] + 0.25, v), xytext=(blocks[ii] + 0.55, v * f), fontsize=FS_A,
                         color=INK, ha='left', va='center',
                         arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6, shrinkA=0, shrinkB=0))
        else:
            a_r.text(blocks[ii] + 0.45, v * 0.82, lab, fontsize=FS_A, color=INK, ha='left', va='top')
    hdl = [Line2D([], [], color=F.SLOT[0], lw=1.3, marker='o', ms=2.6, mec=SURF, mew=0.5),
           Patch(color=F.SLOT[0], alpha=0.16, lw=0),
           Line2D([], [], ls='', marker='o', ms=3.4, mfc=SURF, mec=INK, mew=0.7)]
    a_r.legend(hdl, ['median', '10th–90th pct.', 'clips shown'], loc='upper left', fontsize=FS_A, handlelength=1.3,
               handletextpad=0.4, borderaxespad=0.25, labelspacing=0.2, frameon=False)
    title(a_r, '(e) D is large only in the last blocks', f'$\\rho$ per block, {ra.shape[0]} test clips')
    for c, (kk, ii, head, sub) in enumerate(rows2):
        uu = z['usable'][kk, ii]
        heat(a_low[c], z['D'][kk, ii], uu, DIV, dnorm, yticks=(c == 0))
        title(a_low[c], head, f'$\\rho$ = {rho[(kk, ii)]:.2f}, {sub}')
    a_low[0].set_ylabel('video frame', fontsize=FS_L, labelpad=2)
    d9 = dabs[(1, i9)]
    a_low[2].text((M + 1) / 2, 8.5, 'too small to see\non this scale:\n'
                  f'max $|D|$ = {d9 * 1e3:.2f} $\\times\\,10^{{-3}}$\n({100 * d9 / dlim:.0f}% of the scale)',
                  ha='center', va='center', fontsize=FS_A, color=INK2, zorder=5, linespacing=1.15,
                  bbox=dict(boxstyle='round,pad=0.3', fc=SURF, ec='none', alpha=0.9))
    cb = fig.colorbar(im_d, cax=c_div, extend='max', extendfrac=0.03)
    milli(cb, '$D$  ($\\times 10^{-3}$), panels (c) and (f)–(h)')
    cb.set_ticks([-0.015, -0.010, 0, 0.005, 0.010, 0.015], labels=['−15', '−10', '0', '5', '10', '15'])
    # the clip-dependent floors: a dark mark across the bar at each, labelled with its panels (leaders fan the labels out)
    tr = cb.ax.get_yaxis_transform()
    cb.ax.annotate('floor\n($\\pi$ = 0) of', xy=(1.0, -0.0016), xycoords=tr, xytext=(3.5, 0), textcoords='offset points',
                   ha='left', va='center', fontsize=FS_A, color=INK2, linespacing=1.0)
    for key, lab, ylab in [('g', '(g)', -0.0031), ('c', '(c, h)', -0.0043), ('f', '(f)', -0.0055)]:
        v = floors[key][0]
        cb.ax.plot([0, 1], [v, v], color=INK, lw=0.7, transform=tr, clip_on=False, zorder=5)
        cb.ax.annotate(lab, xy=(1.0, v), xycoords=tr, xytext=(3.2, ylab), textcoords=tr, ha='left', va='center',
                       fontsize=FS_A, color=INK2,
                       arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6, shrinkA=0.8, shrinkB=0, relpos=(0, 0.5)))
    cb.ax.yaxis.set_label_coords(9.3, 0.5)                  # clear of the floor labels
    cb.ax.text(0.5, 1.045, 'more than\nproduct', transform=cb.ax.transAxes, ha='center', va='bottom',
               fontsize=FS_A, color=INK2, linespacing=1.0)
    cb.ax.text(0.5, -0.008, 'less than\nproduct', transform=cb.ax.transAxes, ha='center', va='top', fontsize=FS_A,
               color=INK2, linespacing=1.0)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_plan_decomposition.{ext}'), bbox_inches='tight', pad_inches=0.03, dpi=220)
    plt.close(fig)
    return dict(clip=clip, frames_shown_1based=[i + 1 for i in th_idx], n_files=n_files, n_real=n_real,
                H=round(float(H), 4), frame_max_tv=int(tv.argmax()) + 1, d_scale=float(dlim),
                floors_milli_nreal={kk: (round(v * 1e3, 3), n) for kk, (v, n) in floors.items()},
                d_absmax={f'{kk},{blocks[ii]}': v for (kk, ii), v in dabs.items()}, n_cells_over_scale_g=n_over,
                rho=[round(float(z['rho_sel'][kk, ii]), 4) for kk, ii, _, _ in shown],
                rho_median_by_block=np.round(p50, 4).tolist())


@plt.rc_context(LOCAL_RC)
def fig3():
    zm, za = load('MMA_CREMAD'), load('ABV_DFEW')
    k = 1                                                    # median clip by rho at block 12, in each setting
    panels = [(zm, 9, '(a) Low', 'router, CREMA-D, block 9'),
              (zm, 12, '(b) High', 'same clip, block 12'),
              (za, 9, '(c) Intermediate', 'ABVnet router, DFEW, block 9')]
    mats = []
    for z, blk, head, sub in panels:
        i = list(z['blocks']).index(blk)
        u = z['usable'][k, i]
        ci = z['Cint'][k, i].astype(float) / EPS
        pi = real(z['pi'][k, i], u)
        ratio = pi / (np.outer(pi.sum(1), pi.sum(0)) / pi.sum())
        kap = float(np.sqrt(np.mean(real(ci, u) ** 2)))
        mats.append(dict(ci=ci, u=u, head=head, sub=sub, kappa=kap, x=real(ci, u).ravel(), y=ratio.ravel(),
                         clip=str(z['names'][k]), blk=blk))
    vmax = max(np.abs(m['x']).max() for m in mats)
    norm = TwoSlopeNorm(0, -vmax, vmax)
    fig = plt.figure(figsize=(7.2, 2.70))
    yt, hh, ww = 0.42, 1.48, 1.16
    a_th = axes_in(fig, 0.30, yt, hh / 16, hh)
    x0 = 0.30 + hh / 16 + 0.03
    xs = [x0, x0 + ww + 0.17, x0 + 2 * ww + 0.51]          # (a)-(b) gap wide enough to keep the '32' and '1' ticks apart
    axs = [axes_in(fig, x, yt, ww, hh) for x in xs]
    c_ax = axes_in(fig, x0, yt + hh + 0.45, xs[-1] + ww - x0, 0.07)
    a_sc = axes_in(fig, 5.06, yt, 2.06, hh + 0.52)
    thumbs(a_th, mats[0]['clip'])
    for c, (ax, m) in enumerate(zip(axs, mats)):
        im = heat(ax, m['ci'], m['u'], DIV_CINT, norm, yticks=(c == 2))
        title(ax, f'{m["head"]}: $\\kappa$ = {m["kappa"]:.2f}', m['sub'])
    n0 = int(mats[0]['u'].sum())
    axs[0].text((n0 + 1) / 2, 8.5, '≈ 0 everywhere\non this scale\n'
                f'($|C_{{\\mathrm{{int}}}}|/\\varepsilon \\leq$ {np.abs(mats[0]["x"]).max():.2f})',
                ha='center', va='center', fontsize=FS_A, color=INK2)
    cb = fig.colorbar(im, cax=c_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=FS_K, length=2, pad=1.5)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor(F.AXIS)
    cb.set_ticks([-4, -2, 0, 2, 4])
    cb.ax.set_xlabel('$C_{\\mathrm{int}}/\\varepsilon$ per cell: double-centred cost in units of the temperature '
                     '($\\kappa$ = its RMS over cells)', fontsize=FS_A, color=INK2, labelpad=1.5)
    cb.ax.text(0.0, 1.4, '← cheaper than the marginals predict: more mass', transform=cb.ax.transAxes, ha='left',
               va='bottom', fontsize=FS_A, color=INK2)
    cb.ax.text(1.0, 1.4, 'costlier: less mass →', transform=cb.ax.transAxes, ha='right', va='bottom', fontsize=FS_A,
               color=INK2)
    # plan / product against the interaction, one dot per (frame, real slot) cell
    a_sc.axhline(1.0, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
    a_sc.scatter(mats[1]['x'], mats[1]['y'], marker='o', s=5, color=F.SLOT[0], alpha=0.45, lw=0, zorder=3)
    a_sc.scatter(mats[2]['x'], mats[2]['y'], marker='^', s=6, color=F.SLOT[2], alpha=0.45, lw=0, zorder=3)
    a_sc.scatter(mats[0]['x'], mats[0]['y'], marker='o', s=11, facecolor=SURF, edgecolor=F.SLOT[0], lw=0.6, zorder=4)
    a_sc.set_yscale('log')
    a_sc.set_ylim(1e-4, 40)
    a_sc.set_xlim(-vmax * 1.05, vmax * 1.05)
    a_sc.set_xticks([-4, -2, 0, 2, 4])
    a_sc.tick_params(labelsize=FS_K, length=2, pad=1.5)
    a_sc.set_xlabel('$C_{\\mathrm{int}}/\\varepsilon$ of the cell', fontsize=FS_L, labelpad=1.5)
    a_sc.set_ylabel('plan ÷ product (log scale)', fontsize=FS_L, labelpad=1.5)
    a_sc.text(vmax * 1.0, 1.2, 'plan = product', fontsize=FS_A, color=INK2, ha='right', va='bottom')
    a_sc.text(-vmax * 1.0, 1.4e-4, '← cheaper', fontsize=FS_A, color=MUTED, ha='left', va='bottom')
    a_sc.text(vmax * 1.0, 1.4e-4, 'costlier →', fontsize=FS_A, color=MUTED, ha='right', va='bottom')
    y0 = mats[0]['y']
    a_sc.annotate(f'(a): every cell within\n{100 * max(1 - y0.min(), y0.max() - 1):.0f}% of the product',
                  xy=(-0.15, 0.93), xytext=(-2.4, 0.03), fontsize=FS_A, color=INK2, ha='center', va='center',
                  arrowprops=dict(arrowstyle='-', color=INK2, lw=0.6, shrinkB=1.5))
    h = [Line2D([], [], marker='o', ls='', mfc=SURF, mec=F.SLOT[0], mew=0.6, ms=3.6),
         Line2D([], [], marker='o', ls='', color=F.SLOT[0], ms=3.0),
         Line2D([], [], marker='^', ls='', color=F.SLOT[2], ms=3.2)]
    a_sc.legend(h, ['(a) block 9', '(b) block 12', '(c) ABVnet'], loc='upper right', fontsize=FS_A,
                handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25, frameon=False)
    fig.canvas.draw()                                        # title starts over the y-label: it is wider than the axes
    r = fig.canvas.get_renderer()
    dx = (a_sc.yaxis.label.get_window_extent(r).x0 - a_sc.get_window_extent(r).x0) / fig.dpi * 72
    title(a_sc, '(d) Cheaper cells get more mass', 'one dot per cell of (a)–(c)', dx=dx)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUT, f'fig_cost_interaction.{ext}'), bbox_inches='tight', pad_inches=0.03, dpi=220)
    plt.close(fig)
    return [(m['clip'], m['blk'], round(m['kappa'], 3), round(float(np.abs(m['x']).max()), 3),
             float(m['y'].min()), float(m['y'].max())) for m in mats]


if __name__ == '__main__':
    print(fig2())
    print(fig3())
    print('->', OUT)
