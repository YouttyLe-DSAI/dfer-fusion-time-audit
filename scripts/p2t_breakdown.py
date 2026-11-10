#!/usr/bin/env python
"""P2T stage 0 follow-up (a), prereg.md: per-class and clip-length breakdowns of the saved stage-0 predictions.

Descriptive only. Same paired bootstrap as p2t_summary.py (within folds, 2000 resamples, 90% CI, seed 0).
Usage: p2t_breakdown.py <results/p2t dir>   (expects <pair>/fold{1..5}.npz and _ann/<DATASET>_set_<k>_test*.txt)
"""

import json
import os
import sys

import numpy as np

B, ALPHA, FOLDS = 2000, 0.10, (1, 2, 3, 4, 5)
ARMS = ('shuffle_s0', 'reverse', 'hold4', 'hold1')
FLAG_CLASS, FLAG_LEN = -3.0, 1.0
AVDFER = {'DFEW': ['happiness', 'sadness', 'neutral', 'anger', 'surprise', 'disgust', 'fear'],
          'MAFW': ['happiness', 'sadness', 'neutral', 'anger', 'surprise', 'disgust', 'fear',
                   'contempt', 'anxiety', 'helplessness', 'disappointment'],
          'CREMAD': ['happiness', 'sadness', 'neutral', 'anger', 'disgust', 'fear']}
HICMAE = {'CREMAD': ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness'],
          'MAFW': ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise',
                   'contempt', 'anxiety', 'helplessness', 'disappointment']}
PAIRS = {'hicmae_cremad': ('CREMAD', 'hicmae'), 'hicmae_mafw': ('MAFW', 'hicmae'),
         'mma_dfew': ('DFEW', 'mma'), 'mma_mafw': ('MAFW', 'mma'), 'mma_cremad': ('CREMAD', 'mma'),
         'ot_dfew': ('DFEW', 'mma'), 'ot_mafw': ('MAFW', 'mma'), 'ot_cremad': ('CREMAD', 'mma')}


def ann_file(root, ds, k):
    return os.path.join(root, '_ann', f'{ds}_set_{k}_test{"_faces" if ds == "MAFW" else ""}.txt')


def read_ann(path):
    rows = []
    for line in open(path):
        d, n, lab = line.strip().rsplit(' ', 2)
        rows.append((os.path.basename(d.rstrip('/')), int(n), int(lab)))
    return rows


def ci(x):
    return [float(v) for v in np.quantile(x, [ALPHA / 2, 1 - ALPHA / 2])]


def recall_w(pred, y, w, c):
    m = y == c
    return 100.0 * w[:, m & (pred == c)].sum(1) / np.maximum(w[:, m].sum(1), 1e-12)


def uar_w(pred, y, w):
    return np.mean(np.stack([recall_w(pred, y, w, c) for c in np.unique(y)], 1), 1)


def load_pair(root, pair):
    ds, kind = PAIRS[pair]
    ann = {k: read_ann(ann_file(root, ds, k)) for k in FOLDS}
    length_by_id = {cid: n for k in FOLDS for cid, n, _ in ann[k]}
    y, lengths, pred = [], [], {}
    fold_sizes = []
    for k in FOLDS:
        z = np.load(os.path.join(root, pair, f'fold{k}.npz'), allow_pickle=False)
        lab = z['labels']
        if kind == 'mma':
            idx = z['clips'].astype(int)
            assert all(ann[k][i][2] == lab[j] for j, i in enumerate(idx)), (pair, k, 'label mismatch')
            lengths.append(np.array([ann[k][i][1] for i in idx]))
        else:
            ids = [os.path.basename(str(c).rstrip('/')) for c in z['clips']]
            miss = [c for c in ids if c not in length_by_id]
            assert not miss, (pair, k, len(miss), miss[:3])
            lengths.append(np.array([length_by_id[c] for c in ids]))
        y.append(lab)
        fold_sizes.append(len(lab))
        for a in ('base',) + ARMS:
            pred.setdefault(a, []).append(z[f'prob_{a}'].argmax(1))
    names = HICMAE[ds] if kind == 'hicmae' else AVDFER[ds]
    return ds, names, np.concatenate(y), np.concatenate(lengths), {a: np.concatenate(v) for a, v in pred.items()}, \
        fold_sizes


def boot_weights(fold_sizes, rng):
    return np.concatenate([rng.multinomial(n, np.full(n, 1 / n), size=B) for n in fold_sizes], axis=1).astype(np.float64)


def analyse(root, pair):
    ds, names, y, L, pred, fs = load_pair(root, pair)
    w = boot_weights(fs, np.random.default_rng(0))
    one = np.ones((1, len(y)))
    out = {'pair': pair, 'dataset': ds, 'n': int(len(y)), 'per_class': {}, 'length': {}}
    for c in np.unique(y):
        nm = names[c]
        rb, rb1 = recall_w(pred['base'], y, w, c), float(recall_w(pred['base'], y, one, c)[0])
        rec = {'n': int((y == c).sum()), 'base_recall': rb1, 'arms': {}}
        for a in ARMS:
            d1 = float(recall_w(pred[a], y, one, c)[0]) - rb1
            lo, hi = ci(recall_w(pred[a], y, w, c) - rb)
            rec['arms'][a] = {'d_recall': d1, 'ci': [lo, hi], 'time_sensitive': bool(d1 <= FLAG_CLASS and hi < 0)}
        out['per_class'][nm] = rec
    q1, q2 = np.quantile(L, [1 / 3, 2 / 3])
    bins = {'short': L <= q1, 'mid': (L > q1) & (L <= q2), 'long': L > q2}
    out['length_cuts'] = [float(q1), float(q2)]
    dB = {}
    for b, m in bins.items():
        yb, wb = y[m], w[:, m]
        base_b, base1 = uar_w(pred['base'][m], yb, wb), float(uar_w(pred['base'][m], yb, np.ones((1, m.sum())))[0])
        ent = {'n': int(m.sum()), 'frames_median': float(np.median(L[m])), 'base_uar': base1, 'arms': {}}
        for a in ARMS:
            db = uar_w(pred[a][m], yb, wb) - base_b
            d1 = float(uar_w(pred[a][m], yb, np.ones((1, m.sum())))[0]) - base1
            dB[(b, a)] = db
            ent['arms'][a] = {'d_uar': d1, 'ci': ci(db)}
        out['length'][b] = ent
    out['long_minus_short'] = {}
    for a in ARMS:
        v = out['length']['long']['arms'][a]['d_uar'] - out['length']['short']['arms'][a]['d_uar']
        lo, hi = ci(dB[('long', a)] - dB[('short', a)])
        out['long_minus_short'][a] = {'value': v, 'ci': [lo, hi],
                                      'length_effect': bool(abs(v) >= FLAG_LEN and (lo > 0 or hi < 0))}
    return out


def report(o):
    s = [f"=== {o['pair']} ({o['dataset']}, {o['n']} clips)", 'per class: n, base recall, d_recall [90% CI] per arm (* = time-sensitive)']
    for nm, r in o['per_class'].items():
        cells = '  '.join(f"{a}:{v['d_recall']:+6.1f} [{v['ci'][0]:+5.1f},{v['ci'][1]:+5.1f}]{'*' if v['time_sensitive'] else ' '}"
                          for a, v in r['arms'].items())
        s.append(f"  {nm:<15} {r['n']:5d} {r['base_recall']:5.1f}  {cells}")
    s.append(f"length tertiles (num_frames cuts {o['length_cuts'][0]:.0f} / {o['length_cuts'][1]:.0f}):")
    for b, e in o['length'].items():
        cells = '  '.join(f"{a}:{v['d_uar']:+6.2f} [{v['ci'][0]:+5.2f},{v['ci'][1]:+5.2f}]" for a, v in e['arms'].items())
        s.append(f"  {b:<5} n={e['n']:5d} med={e['frames_median']:5.0f} base={e['base_uar']:5.2f}  {cells}")
    cells = '  '.join(f"{a}:{v['value']:+6.2f} [{v['ci'][0]:+5.2f},{v['ci'][1]:+5.2f}]{'!' if v['length_effect'] else ' '}"
                      for a, v in o['long_minus_short'].items())
    s.append(f"  long-short  {cells}   (! = length effect)")
    return '\n'.join(s)


def main(root):
    res, txt = {}, []
    for pair in PAIRS:
        o = analyse(root, pair)
        res[pair] = o
        txt.append(report(o))
        print(txt[-1], flush=True)
    json.dump(res, open(os.path.join(root, 'breakdown.json'), 'w'), indent=1)
    open(os.path.join(root, 'breakdown.txt'), 'w').write('\n\n'.join(txt) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
