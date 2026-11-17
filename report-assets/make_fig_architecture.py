#!/usr/bin/env python
"""Paper figure (method, Fig. architecture): the router in the MMA-DFER host on one real CREMA-D clip, and one routing step.

Facts drawn (AV-DFER-OT at bf34671: avdfer/hosts/mma_dfer/{model.py, models_vit.py, audiomae/audio_models_vit.py},
avdfer/fusion/routing.py, avdfer/data/dataset.py; nothing else is implied):
- Video input: 16 frames, the test loader's sampling (middle frame of each of 16 equal segments of the clip), 160 px.
  Audio input: Kaldi filterbank, 128 mel bins, 25 ms window, 10 ms hop, zero-padded to 512 rows (5.12 s); AudioMAE's
  16 x 16 patches give 32 time slots x 8 frequency patches; a slot is real when it holds at least one real row.
- MAE-Face ViT-B and AudioMAE ViT-B, 12 blocks each, frozen. Trained (MMADFER.TRAINABLE via set_trainable): the
  learnable prompts and the position embeddings of both encoders, the per-block projections into the 128-d exchange
  space and back (with a tanh gate that starts at 0), the routing modules and the head.
- In each block both streams are projected to 128-d; each receives the other's mean (MMA-DFER's own, clip-level
  exchange) and the routing step adds a2v (to frame j) and v2a (to the audio tokens). The update goes back through a
  trained linear + GELU projection and a tanh gate and is added to the block's OUTPUT, which then enters the next block
  (forward_block_post); panel (a) therefore ends each amber return arrow on the connector between block b and b+1
  (after block 12: on the line to the head).
- Routing step: v_j = frame j's class token (A0), a_k = mean of the 8 frequency patches of slot k; cosine cost after LN
  and l2 normalisation; balanced entropic plan (eps 0.05, 10 Sinkhorn iterations, padding masked); a2v from
  sum_k pi_jk a_k / r_j (times r_j T = 1 for a balanced plan), v2a from the plan-weighted frame descriptors pooled over
  the real slots, which equals sum_j r_j v_j; each through a linear projection and a tanh gate that starts at 0.
  Descriptor arms (method.tex, Sec. descriptors): A1 replaces v_j in the cost and in the message to the audio stream;
  A2/A3 change only the descriptor the cost reads (v^cost_j), the message keeps the class token. The matrices drawn
  in (b) are the default class-token router's (A0 descriptor) on CREMA-D fold 1.
- Head: the audio class token is added to each of the 16 frame class tokens, linear 768 -> 512, one-layer temporal
  transformer, linear classifier (6 classes on CREMA-D, 7 on DFEW).
- Real data (CREMA-D, Cao et al. 2014): clip 1006_IOM_ANG_XX (anger, fold-1 test part), index 1 of
  results/plandump/MMA_CREMAD.npz (the median clip of the residual size at block 12, the clip of Fig. plan
  decomposition). The 16 thumbnails are the frames the test loader samples from the 98 face crops; the spectrogram is a
  numpy re-implementation of the loader's Kaldi filterbank; cost, plan and product (all 12 blocks) are from the dump.

Simplified 2026-11-16: only names, symbols and scale values are drawn; the explanation is in the caption (method.tex).

    <mplenv>/bin/python make_fig_architecture.py   (from report-assets) -> paper-v2/figures/fig_architecture.{pdf,png}
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io.wavfile  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

import make_fig_e2 as F  # noqa: E402  (style, palette)
from make_fig_plans import SEQ  # noqa: E402  (the plan's sequential map in Fig. plan decomposition)

OUT, INK, INK2, MUTED, SURF = F.OUT, F.INK, F.INK2, F.MUTED, F.SURF
VID, AUD, ROUT, TRN = F.SLOT[0], F.SLOT[1], F.SLOT[2], F.SLOT[3]
VID_D, AUD_D, ROUT_D, TRN_D = '#164a8c', '#a8441c', '#0f6e4c', '#7a5200'
VID_BG, AUD_BG = '#eef4fc', '#fff4ee'               # tower bands
VID_M, AUD_M = '#cfe0f6', '#fbd9c8'                 # frozen blocks (pale)
TRN_L, TRN_M = '#fdf5e1', '#f8dc94'                 # trained parts
MEAN = '#9a998f'                                    # MMA-DFER's clip-level exchange (neutral grey)
EDGE = '#8a8981'
# paper-wide arm roles (make_fig_ablation.py, make_fig_secondary.py): A0 grey, A1 blue, A2 green, A3 orange, A4 amber
ARM_COL = {'A0': '#9a998f', 'A1': F.SLOT[0], 'A2': F.SLOT[2], 'A3': F.SLOT[1], 'A4': F.SLOT[3]}
ARM_MRK = {'A0': 'o', 'A1': 's', 'A2': 'D', 'A3': 'v', 'A4': 'p'}
CLIP = '1006_IOM_ANG_XX'
EX = os.path.join(F.ROOT, 'report-assets', 'examples_cremad')
CLASSES = ['happiness', 'sadness', 'neutral', 'anger', 'disgust', 'fear']   # CREMA-D order of the loader
N_FRAMES, N_SLOTS, ROWS_PER_SLOT, FBANK_ROWS = 16, 32, 16, 512
FW, FH = 7.2, 4.42                                  # inches; drawing units are inches
FS = 6.4                                            # base font size (pt)
SPEC_CM = LinearSegmentedColormap.from_list('spec', ['#fffaf6', '#f6c3ad', '#eb6834', '#a8441c', '#4a1c08'])
COST_CM = LinearSegmentedColormap.from_list('cost', ['#ffffff', '#cfcec6', '#6d6c66', '#2b2a27'])


# ----------------------------------------------------------------------------------------------- data
def sampled_frames():
    """The 16 frames the test loader takes (dataset._test_indices, lexicographic order), resized to 160 px."""
    d = os.path.join(EX, CLIP)
    files = sorted(f for f in os.listdir(d) if f.endswith('.jpg'))
    n = len(files)                                  # 98, the count in annotation/CREMAD_set_1_test.txt
    tick = n / float(N_FRAMES)
    idx = [int(tick / 2.0 + tick * x) for x in range(N_FRAMES)]
    ims = [np.asarray(Image.open(os.path.join(d, files[i])).convert('RGB').resize((160, 160), Image.BILINEAR))
           for i in idx]
    return ims, idx, n


def fbank():
    """Kaldi fbank as the loader computes it (torchaudio.compliance.kaldi.fbank: 128 mel bins, hanning, 25 ms / 10 ms,
    pre-emphasis 0.97, DC removal, no dither), normalised with the AudioSet statistics. Returns (wave, sr, E [rows, 128])."""
    sr, x = scipy.io.wavfile.read(os.path.join(EX, CLIP + '.wav'))
    x = x.astype(np.float64) / 32768.0
    x = x - x.mean()
    n_win, hop, n_fft = int(0.025 * sr), int(0.010 * sr), 512
    m = 1 + (len(x) - n_win) // hop
    fr = np.stack([x[i * hop:i * hop + n_win] for i in range(m)])
    fr = fr - fr.mean(1, keepdims=True)
    fr = np.concatenate([fr[:, :1] - 0.97 * fr[:, :1], fr[:, 1:] - 0.97 * fr[:, :-1]], 1)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_win) / (n_win - 1))
    P = np.abs(np.fft.rfft(fr * win, n_fft)) ** 2
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)  # noqa: E731
    lo, hi = mel(20.0), mel(sr / 2.0)
    d = (hi - lo) / 129.0
    fm = mel(np.arange(n_fft // 2) * sr / n_fft)
    B = np.zeros((128, n_fft // 2 + 1))
    for b in range(128):
        l, c, r = lo + b * d, lo + (b + 1) * d, lo + (b + 2) * d
        B[b, :n_fft // 2] = np.maximum(0, np.minimum((fm - l) / (c - l), (r - fm) / (r - c)))
    E = np.log(np.maximum(P @ B.T, np.finfo(np.float32).eps))
    E = (E + 4.2677393) / (4.5689974 * 2)          # FBANK_MEAN, FBANK_STD of avdfer/data/dataset.py
    return x, sr, E


# ----------------------------------------------------------------------------------------------- drawing helpers
def rbox(ax, x0, y0, w, h, fc=SURF, ec=EDGE, lw=0.7, ls='-', r=0.05, z=2, hatch=None):
    p = FancyBboxPatch((x0, y0), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, ls=ls,
                       zorder=z, hatch=hatch)
    ax.add_patch(p)
    return p


def text(ax, x, y, s, size=FS, color=INK, ha='center', va='center', weight='normal', z=8, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, zorder=z, **kw)


def arrow(ax, p0, p1, color=MUTED, lw=0.8, style='-|>', ls='-', ms=5.5, conn='arc3', z=4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=conn, zorder=z, shrinkA=0, shrinkB=0))


def img(ax, a, x0, y0, w, h, z=5, **kw):
    ax.imshow(a, extent=(x0, x0 + w, y0, y0 + h), aspect='auto', zorder=z, **kw)


def heat(ax, M, x0, y0, w, h, cmap, vmin, vmax, ec=EDGE, lw=0.6, z=5):
    ax.imshow(M, extent=(x0, x0 + w, y0, y0 + h), cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest',
              aspect='auto', zorder=z)
    ax.add_patch(Rectangle((x0, y0), w, h, fc='none', ec=ec, lw=lw, zorder=z + 0.5))


def framed(ax, a, x0, y0, s, z=5):
    img(ax, a, x0, y0, s, s, z=z, interpolation='antialiased')
    ax.add_patch(Rectangle((x0, y0), s, s, fc='none', ec=INK2, lw=0.6, zorder=z + 0.2))


def spectrogram(ax, E, x0, y0, w, h, rows, z=5, edge=AUD):
    """E [real rows, 128] on [x0, x0 + w] covering `rows` filterbank rows; rows beyond the real ones are padding."""
    S = np.full((E.shape[1], rows), np.nan)
    k = min(rows, E.shape[0])
    S[:, :k] = E[:k].T
    cm = SPEC_CM.copy()
    cm.set_bad(SURF)
    ax.imshow(S, extent=(x0, x0 + w, y0, y0 + h), origin='lower', cmap=cm, vmin=np.percentile(E, 2),
              vmax=E.max(), aspect='auto', interpolation='antialiased', zorder=z)
    ax.add_patch(Rectangle((x0, y0), w, h, fc='none', ec=edge, lw=0.6, zorder=z + 0.5))
    return k


# ----------------------------------------------------------------------------------------------- layout (inches)
# Simplified 2026-11-16 (student): no explanatory text inside the figure, only names, symbols and scale values; the
# explanation moved to the caption in method.tex. Previous version: dfer-visual/paper-v2-backup-2026-11-16/
# fig_architecture_before_simplify/.
TITLE_A = FH - 0.09
A_TOP = FH - 0.22
VB = (A_TOP - 0.56, A_TOP)                   # video tower band
EB = (VB[0] - 0.14 - 0.68, VB[0] - 0.14)     # 128-d exchange band
AB = (EB[0] - 0.14 - 0.56, EB[0] - 0.14)     # audio tower band
TX0, TX1, LAB_W = 1.64, 5.86, 0.62
TOP_B = AB[0] - 0.27                         # label of panel (b)
HY0, HH, HW = 0.36, 1.18, 1.50               # panel (b): bottom, height and width of the three matrices


# ----------------------------------------------------------------------------------------------- panel (a)
def panel_a(ax, z, ims, wave, sr, E):
    k = 1
    n_real = int(z['usable'][k, -1].sum())
    vmax = float(max(z['pi'][k, i][:, z['usable'][k, i]].max() for i in range(len(z['blocks']))))
    text(ax, 0.02, TITLE_A, '(a)', size=7.6, ha='left', weight='bold')
    v_mid, e_mid, a_mid = (VB[0] + VB[1]) / 2, (EB[0] + EB[1]) / 2, (AB[0] + AB[1]) / 2

    # --- input: the 16 frames the test loader samples, as a two-row filmstrip next to the video tower
    x0, s, g = 0.04, 0.172, 0.011
    film_w = 8 * s + 7 * g
    ftop = v_mid + s + g / 2
    for j, a in enumerate(ims):
        r, c = divmod(j, 8)
        framed(ax, a, x0 + c * (s + g), ftop - (r + 1) * s - r * g, s)
    arrow(ax, (x0 + film_w + 0.02, v_mid), (TX0 - 0.005, v_mid), color=VID, lw=0.9)

    # --- input: waveform (level with the exchange band) and filterbank (next to the audio tower) on one time axis
    W = film_w
    xr = lambda rows: x0 + W * rows / FBANK_ROWS  # noqa: E731   filterbank row -> x
    wy0, wh = e_mid - 0.15, 0.30
    nb = 360
    seg = np.array_split(wave, nb)
    hi_, lo_ = np.array([q.max() for q in seg]), np.array([q.min() for q in seg])
    amp = max(abs(hi_).max(), abs(lo_).max())
    t = (np.arange(nb) + 0.5) * len(wave) / nb / (sr * 0.010)            # sample position in filterbank rows
    ax.fill_between(xr(t), wy0 + wh / 2 * (1 + lo_ / amp), wy0 + wh / 2 * (1 + hi_ / amp), color=AUD, lw=0,
                    zorder=5)
    ax.plot([x0, x0 + W], [wy0 + wh / 2] * 2, color=F.AXIS, lw=0.6, zorder=4)
    sh = 0.40
    sy0 = a_mid - sh / 2 + 0.035
    rows = spectrogram(ax, E, x0, sy0, W, sh, FBANK_ROWS)
    ax.add_patch(Rectangle((xr(rows), sy0), x0 + W - xr(rows), sh, fc='none', ec=F.AXIS, lw=0, hatch='//////',
                           zorder=5.2))
    ty = sy0 - 0.075
    for q in range(N_SLOTS):                         # one box per slot, white gaps so that all 32 can be counted
        xa, xb = xr(q * ROWS_PER_SLOT), xr((q + 1) * ROWS_PER_SLOT)
        real = q < n_real
        ax.add_patch(Rectangle((xa + 0.006, ty), xb - xa - 0.012, 0.055, fc=AUD if real else SURF,
                               ec=AUD_D if real else AUD, lw=0.0 if real else 0.6, hatch=None if real else '//////',
                               zorder=5))
    arrow(ax, (x0 + W + 0.02, sy0 + sh / 2), (TX0 - 0.005, sy0 + sh / 2), color=AUD, lw=0.9)

    # --- the two frozen towers and the trained 128-d exchange between them, block by block
    bx0 = TX0 + LAB_W
    p = (TX1 - 0.06 - bx0) / 12.0
    gap = 0.054                                      # room between blocks for the connector and the return arrow
    bw = p - gap
    xr_off = 0.016                                   # return arrow: this far right of a block's right edge
    rbox(ax, TX0, VB[0], TX1 - TX0, VB[1] - VB[0], fc=VID_BG, ec=VID, lw=0.8, r=0.06, z=1)
    rbox(ax, TX0, AB[0], TX1 - TX0, AB[1] - AB[0], fc=AUD_BG, ec=AUD, lw=0.8, r=0.06, z=1)
    rbox(ax, TX0, EB[0], TX1 - TX0, EB[1] - EB[0], fc=TRN_L, ec=TRN, lw=0.8, r=0.06, z=1)
    for (ym, dark, name) in ((v_mid, VID_D, 'MAE-Face\nViT-B ❄'), (a_mid, AUD_D, 'AudioMAE\nViT-B ❄')):
        text(ax, TX0 + 0.06, ym, name, size=6.4, color=dark, ha='left', weight='bold', linespacing=1.25)
    text(ax, TX0 + 0.06, e_mid, '128-d', size=6.4, color=TRN_D, ha='left', weight='bold')
    bh = (VB[1] - VB[0]) - 0.16
    chip_y = EB[1] - 0.075
    py0, ph = EB[0] + 0.075, (EB[1] - 0.13) - (EB[0] + 0.075)
    a_tow = AB[0] + 0.08 + bh / 2                    # mid-height of the audio blocks
    for b in range(12):
        x = bx0 + b * p
        xc = x + bw / 2
        for (y0, fc, ec, dark) in ((VB[0], VID_M, VID, VID_D), (AB[0], AUD_M, AUD, AUD_D)):
            rbox(ax, x, y0 + 0.08, bw, bh, fc=fc, ec=ec, lw=0.6, r=0.03, z=3)
            text(ax, xc, y0 + 0.08 + bh / 2, str(b + 1), size=6.4, color=dark, weight='bold')
            if b < 11:                               # block b -> block b + 1 (after block 12 the line goes to the head)
                arrow(ax, (x + bw + 0.002, y0 + 0.08 + bh / 2), (x + p - 0.003, y0 + 0.08 + bh / 2), color=dark,
                      lw=0.6, ms=3.2)
        ax.add_patch(Rectangle((x, chip_y - 0.018), bw, 0.036, fc=MEAN, ec='none', zorder=4))
        uu = z['usable'][k, b]
        heat(ax, z['pi'][k, b][:, uu], x, py0, bw, ph, SEQ, 0.0, vmax, ec=ROUT, lw=1.3 if b == 11 else 0.7)
        # block output -> 128-d space (stream colour); trained, gated projection back, added to the block's output on
        # its way to the next block (amber, ends on the connector; forward_block_post)
        xg = x + bw + xr_off
        for (ya, yb, ymid, col) in ((VB[0] + 0.08, EB[1], v_mid, VID), (AB[1] - 0.08, EB[0], a_tow, AUD)):
            up = ya > yb                             # video tower above the exchange band
            arrow(ax, (xc, ya), (xc, yb + (0.004 if up else -0.004)), color=col, lw=0.6, ms=3.2)
            arrow(ax, (xg, yb + (0.004 if up else -0.004)), (xg, ymid + (-0.006 if up else 0.006)), color=TRN_D,
                  lw=0.6, ms=3.2)
    # blocks whose plan lies within 10% of the uniform value everywhere (1-9 for this clip; stated in the caption)
    flat = [b for b in range(12) if np.abs(z['pi'][k, b][:, z['usable'][k, b]] * N_FRAMES * n_real - 1).max() < 0.10]
    assert flat == list(range(9)), flat
    # the block shown in (b)
    x12 = bx0 + 11 * p
    rbox(ax, x12 - 0.010, AB[0] + 0.05, bw + xr_off + 0.038, VB[1] - 0.05 - AB[0] - 0.05, fc='none', ec=ROUT_D,
         lw=0.8, ls=(0, (2.5, 1.5)), r=0.04, z=7)
    text(ax, x12 + bw / 2, AB[0] - 0.02, '(b)', size=6.4, color=ROUT_D, weight='bold', va='top')

    # --- head (trained): audio CLS added to each frame CLS, temporal transformer, classifier
    hx0, hx1 = TX1 + 0.17, FW - 0.04
    hc = (hx0 + hx1) / 2
    ts, tg = 0.034, 0.006
    tw = 16 * ts + 15 * tg
    tx = hc - tw / 2
    fy = v_mid - ts / 2                              # frame tokens on the line out of video block 12
    ay = fy - ts - 0.075                             # the single audio class token, below the 16 frame tokens
    for j in range(16):
        ax.add_patch(Rectangle((tx + j * (ts + tg), fy), ts, ts, fc=VID, ec='none', zorder=5))
    axc = tx + tw / 2                                # one audio CLS, broadcast to each frame token by a bracket
    ax.add_patch(Rectangle((axc - ts / 2, ay), ts, ts, fc=AUD, ec='none', zorder=5))
    yk_ = fy - 0.024
    ax.plot([tx + ts / 2, tx + ts / 2, tx + tw - ts / 2, tx + tw - ts / 2], [fy - 0.006, yk_, yk_, fy - 0.006],
            color=AUD_D, lw=0.6, zorder=5, solid_joinstyle='miter')
    ax.plot([axc, axc], [ay + ts, yk_], color=AUD_D, lw=0.6, zorder=5)
    x12r = x12 + bw                                  # right edge of block 12: the output lines start here
    arrow(ax, (x12r + 0.002, v_mid), (tx - 0.01, v_mid), color=VID_D, lw=0.8, ms=4.5)
    gx = TX1 + 0.07
    ax.plot([x12r + 0.002, gx, gx], [a_tow, a_tow, ay + ts / 2], color=AUD_D, lw=0.8, zorder=4,
            solid_joinstyle='miter')
    arrow(ax, (gx - 0.0004, ay + ts / 2), (axc - ts / 2 - 0.01, ay + ts / 2), color=AUD_D, lw=0.8, ms=4.5)
    bxl, bwh = hx0 + 0.02, hx1 - hx0 - 0.04
    y1 = ay - 0.12
    b1h = 0.30
    rbox(ax, bxl, y1 - b1h, bwh, b1h, fc=TRN_M, ec=TRN, lw=0.8, r=0.04, z=3)
    text(ax, hc, y1 - b1h / 2, 'temporal\ntransformer', size=6.4, color=TRN_D, linespacing=1.15)
    arrow(ax, (hc, ay - 0.008), (hc, y1 + 0.004), color=TRN_D, lw=0.8)
    y2 = y1 - b1h - 0.09
    b2h = 0.20
    rbox(ax, bxl, y2 - b2h, bwh, b2h, fc='#f3c65a', ec=TRN, lw=0.8, r=0.04, z=3)
    text(ax, hc, y2 - b2h / 2, 'classifier', size=6.4, color=TRN_D, weight='bold')
    arrow(ax, (hc, y1 - b1h - 0.004), (hc, y2 + 0.004), color=TRN_D, lw=0.8)
    y3 = y2 - b2h - 0.09
    arrow(ax, (hc, y2 - b2h - 0.004), (hc, y3 + 0.004), color=TRN_D, lw=0.8)
    cg = 0.03
    cw, chh = (bwh - cg) / 2, 0.15
    for q, name in enumerate(CLASSES):
        r, c = divmod(q, 2)
        xq, yq = bxl + c * (cw + cg), y3 - (r + 1) * chh - r * cg
        lab = name == 'anger'
        rbox(ax, xq, yq, cw, chh, fc=INK2 if lab else SURF, ec=INK2 if lab else EDGE, lw=0.8 if lab else 0.6,
             r=0.03, z=3)
        text(ax, xq + cw / 2, yq + chh / 2, name, size=6.0, color=SURF if lab else INK2,
             weight='bold' if lab else 'normal')
    assert yq > AB[0] - 0.05, yq                     # the head column ends above the bottom of the audio band
    return vmax


# ----------------------------------------------------------------------------------------------- panel (b)
def panel_b(ax, z, ims, E, vmax):
    k, blk = 1, 12
    i = list(z['blocks']).index(blk)
    u = z['usable'][k, i]
    n_real = int(u.sum())
    C = z['C'][k, i][:, u].astype(float)
    pi = z['pi'][k, i][:, u].astype(float)
    prod = pi - z['D'][k, i][:, u].astype(float)
    text(ax, 0.02, TOP_B, '(b)', size=7.6, ha='left', weight='bold')
    hy0, hh, hw = HY0, HH, HW
    th = hh / N_FRAMES
    yt_ = hy0 + hh + 0.10                            # matrix titles

    # cost: rows = frames (descriptor v_j), columns = real audio slots (descriptor a_k)
    cx0 = 0.36
    for j, a in enumerate(ims):
        framed(ax, a, cx0 - 0.02 - th, hy0 + hh - (j + 1) * th, th, z=6)
    lx = cx0 - 0.07 - th
    text(ax, lx, hy0 + hh / 2, '$v_j$', size=7.4, color=VID_D, ha='right')
    heat(ax, C, cx0, hy0, hw, hh, COST_CM, float(C.min()), float(C.max()))
    spectrogram(ax, E, cx0, hy0 - 0.14, hw, 0.11, n_real * ROWS_PER_SLOT, z=5)
    text(ax, lx, hy0 - 0.085, '$a_k$', size=7.4, color=AUD_D, ha='right')
    text(ax, cx0 + hw / 2, yt_, 'cost $C$', size=7.0, weight='bold')
    cby0, cbx0, cbx1 = hy0 - 0.29, cx0 + 0.30, cx0 + hw - 0.30
    ax.imshow(np.linspace(C.min(), C.max(), 256)[None, :], extent=(cbx0, cbx1, cby0, cby0 + 0.05), cmap=COST_CM,
              vmin=float(C.min()), vmax=float(C.max()), aspect='auto', zorder=5)
    ax.add_patch(Rectangle((cbx0, cby0), cbx1 - cbx0, 0.05, fc='none', ec=EDGE, lw=0.6, zorder=5.5))
    text(ax, cbx0 - 0.03, cby0 + 0.025, f'{C.min():.2f}', size=6.0, color=INK2, ha='right')
    text(ax, cbx1 + 0.03, cby0 + 0.025, f'{C.max():.2f}', size=6.0, color=INK2, ha='left')

    # Sinkhorn
    px0 = cx0 + hw + 0.66
    ym = hy0 + hh / 2
    arrow(ax, (cx0 + hw + 0.06, ym), (px0 - 0.06, ym), color=INK2, lw=1.0)
    text(ax, (cx0 + hw + px0) / 2, ym + 0.09, 'Sinkhorn', size=6.4, color=INK2, weight='bold')

    # plan, with the colour scale shared by every plan in the figure (tick: the uniform value)
    heat(ax, pi, px0, hy0, hw, hh, SEQ, 0.0, vmax)
    text(ax, px0 + hw / 2, yt_, 'plan $\\pi$', size=7.0, weight='bold')
    cb = np.linspace(0, vmax, 256)[None, :]
    ax.imshow(cb, extent=(px0, px0 + hw, hy0 - 0.11, hy0 - 0.06), cmap=SEQ, vmin=0, vmax=vmax, aspect='auto',
              zorder=5)
    ax.add_patch(Rectangle((px0, hy0 - 0.11), hw, 0.05, fc='none', ec=EDGE, lw=0.6, zorder=5.5))
    uni = 1.0 / (N_FRAMES * n_real)
    xu = px0 + hw * uni / vmax
    ax.plot([xu, xu], [hy0 - 0.13, hy0 - 0.045], color=INK, lw=0.8, zorder=6)
    text(ax, px0, hy0 - 0.15, '0', size=6.0, color=INK2, va='top')
    text(ax, px0 + hw, hy0 - 0.15, f'{vmax:.3f}', size=6.0, color=INK2, va='top', ha='right')

    # messages: to frame j (blue) and to the audio tokens (orange)
    mx0, mw = px0 + hw + 0.15, 0.92
    gap = 0.12
    bh = (hh - gap) / 2
    yv = hy0 + bh + gap
    rbox(ax, mx0, yv, mw, bh, fc='#eaf2fc', ec=VID, lw=0.8, r=0.05, z=3)
    text(ax, mx0 + mw / 2, yv + bh / 2, '$\\Sigma_k\\, \\pi_{jk}\\, a_k / r_j$', size=7.2)
    rbox(ax, mx0, hy0, mw, bh, fc='#fdeee6', ec=AUD, lw=0.8, r=0.05, z=3)
    text(ax, mx0 + mw / 2, hy0 + bh / 2, '$\\Sigma_j\\, r_j\\, v_j$', size=7.2)
    arrow(ax, (px0 + hw + 0.01, yv + bh / 2), (mx0 - 0.01, yv + bh / 2), color=VID, lw=0.9)
    arrow(ax, (px0 + hw + 0.01, hy0 + bh / 2), (mx0 - 0.01, hy0 + bh / 2), color=AUD, lw=0.9)

    # intervention: the product replacement
    ix0, ix1 = mx0 + mw + 0.15, FW - 0.04
    rbox(ax, ix0, hy0 - 0.12, ix1 - ix0, hh + 0.12 + 0.25, fc='#f3f2ee', ec=INK2, lw=0.8, ls=(0, (3, 2)), r=0.06,
         z=1)
    qx0 = (ix0 + ix1) / 2 - hw / 2
    heat(ax, prod, qx0, hy0, hw, hh, SEQ, 0.0, vmax)
    text(ax, (ix0 + ix1) / 2, yt_, 'product $\\pi^{\\prime} = r\\,c^{\\!\\!\\top}\\!/S$', size=7.0, weight='bold')
    arrow(ax, (qx0 - 0.01, yv + bh / 2), (mx0 + mw + 0.01, yv + bh / 2), color=INK2, lw=0.9, ls=(0, (3, 2)))
    assert qx0 > ix0 + 0.08, (qx0, ix0)

def main():
    z = np.load(os.path.join(F.ROOT, 'results', 'plandump', 'MMA_CREMAD.npz'))
    assert str(z['names'][1]) == CLIP and CLASSES[int(z['labels'][1])] == 'anger'
    ims, idx, n_files = sampled_frames()
    wave, sr, E = fbank()
    n_real = int(z['usable'][1, -1].sum())
    assert int(np.ceil(E.shape[0] / ROWS_PER_SLOT)) == n_real          # 325 filterbank rows -> 21 real slots
    print(f'clip {CLIP}: {n_files} frames, sampled {[i + 1 for i in idx]} (1-based); {len(wave) / sr:.3f} s, '
          f'{E.shape[0]} filterbank rows, {n_real} real slots')
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FW)
    ax.set_ylim(0, FH)
    ax.axis('off')
    vmax = panel_a(ax, z, ims, wave, sr, E)
    panel_b(ax, z, ims, E, vmax)
    i12 = list(z['blocks']).index(12)
    C = z['C'][1, i12][:, z['usable'][1, i12]]
    print(f'block 12: C in [{C.min():.3f}, {C.max():.3f}], plan max (all blocks) {vmax:.4f}')
    ax.set_aspect('equal')
    fig.savefig(os.path.join(OUT, 'fig_architecture.pdf'), dpi=450)
    fig.savefig(os.path.join(OUT, 'fig_architecture.png'), dpi=220)
    print('->', os.path.join(OUT, 'fig_architecture.pdf'))


if __name__ == '__main__':
    main()
