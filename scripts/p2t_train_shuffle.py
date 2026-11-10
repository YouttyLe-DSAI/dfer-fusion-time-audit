#!/usr/bin/env python
"""P2T follow-up (b), prereg.md: order-agnostic training control for MMA-DFER (none).

Retrains a reference run with its own config (same config ID, asserted), except that the 16 frames of every
training clip are put in a fresh random order after frame selection and augmentation. The permutation comes from
its own generator seeded by (loader worker seed, clip index), so the global RNG streams that pick frames, crops
and flips are untouched and the run sees the reference run's frames, crops and batches in another order only.
Test clips keep their true order. Imports the original code read-only (PYTHONDONTWRITEBYTECODE=1); writes only
under --out-root.

    p2t_train_shuffle.py --src AV-DFER-OT --like <reference run dir> --fold 1 --out-root runs_p2t/train_shuffle
    p2t_train_shuffle.py ... --self-test        # CPU check of the patch on a few training clips, no training
"""

import argparse
import json
import os
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--src', required=True)
ap.add_argument('--like', required=True, help='reference run dir holding fold<k>/config.json')
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--out-root', required=True)
ap.add_argument('--resume', action='store_true')
ap.add_argument('--self-test', action='store_true')
a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.src))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from avdfer.data import dataset as D  # noqa: E402
from avdfer.engine import trainer as T  # noqa: E402
from avdfer.run.config import LATER_KEYS, config_id  # noqa: E402

RECORD_KEYS = ('config_id', 'fold', 'data_fingerprint', 'train_annotation_file', 'eval_annotation_file',
               'train_annotation_sha256', 'eval_annotation_sha256')
_orig_getitem = D.VideoDataset.__getitem__


def shuffled_getitem(self, index):
    images, label, fbank, idx, timing = _orig_getitem(self, index)
    if self.mode == 'train':
        g = np.random.default_rng([torch.initial_seed() % (2 ** 63), int(index), 7])
        images = images[torch.as_tensor(g.permutation(images.shape[0]))]
    return images, label, fbank, idx, timing


D.VideoDataset.__getitem__ = shuffled_getitem


def load_reference(like, fold, out_root):
    rec = json.load(open(os.path.join(like, f'fold{fold}', 'config.json')))
    ref_cid = rec['config_id']
    cfg = {k: v for k, v in rec.items() if k not in RECORD_KEYS}
    # A run older than a setting in LATER_KEYS has no entry for it; the legacy value computes what the code did
    # then and is left out of the config ID (avdfer/run/config.py), so the ID check below still holds.
    for path, legacy in LATER_KEYS.items():
        node = cfg
        for k in path[:-1]:
            node = node[k]
        if path[-1] not in node:
            node[path[-1]] = legacy
            print(f'reference config has no {".".join(path)}: legacy value {legacy!r}', flush=True)
    assert cfg['fusion']['rule'] == 'none', cfg['fusion']
    cfg['out_dir'] = os.path.abspath(out_root)
    cid = config_id(cfg)
    assert cid == ref_cid, (cid, ref_cid)
    return cfg, ref_cid


def self_test(cfg):
    """Same seeds -> the patched item is a permutation of the reference item's frames, and not the identity."""
    import random
    from avdfer.hosts import audio_input
    from avdfer.run.config import annotation_paths
    d = cfg['data']
    train_ann, _ = annotation_paths(cfg, a.fold)
    ds = D.build_dataset(train_ann, d['dataset'], 'train', cfg['host']['img_size'], frames_root=d['frames_root'],
                         audio_root=d['audio_root'], missing_audio=d['missing_audio'], frame_order=d['frame_order'],
                         audio_channels=d['audio_channels'], frame_times=d['frame_times'], **audio_input(cfg))
    moved = 0
    for i in (0, 1, 2, len(ds) - 1):
        for fn, store in ((_orig_getitem, 'ref'), (shuffled_getitem, 'new')):
            random.seed(123 + i); np.random.seed(123 + i); torch.manual_seed(123 + i)
            out = fn(ds, i)
            if store == 'ref':
                ref = out
        x, y = ref[0], out[0]
        assert x.shape == y.shape and ref[1] == out[1] and torch.equal(ref[2], out[2]), i
        assert torch.equal(ref[4]['frame_t'], out[4]['frame_t']), i
        match = [int(torch.nonzero((x == y[j]).flatten(1).all(1)).flatten()[0]) for j in range(y.shape[0])]
        assert sorted(match) == list(range(x.shape[0])) or len(set(match)) < x.shape[0], (i, match)
        moved += int(match != list(range(x.shape[0])))
        print(f'clip {i}: frame order {match}')
    assert moved >= 3, moved
    print('self-test ok')


def main():
    cfg, cid = load_reference(a.like, a.fold, a.out_root)
    print(f'reference {a.like} fold {a.fold}: config {cid} (identical config, frames shuffled in training)', flush=True)
    if a.self_test:
        self_test(cfg)
        return
    out = os.path.join(cfg['out_dir'], cfg['data']['dataset'], cfg['data']['protocol'], f'none-{cid}', f'fold{a.fold}')
    os.makedirs(out, exist_ok=True)
    marker = os.path.join(out, 'p2t_control.json')
    if not os.path.exists(marker):
        json.dump({'control': 'train_shuffle', 'reference_run': os.path.abspath(a.like), 'fold': a.fold,
                   'config_id': cid, 'note': 'training frames in a random order per clip and epoch; test order true'},
                  open(marker, 'w'), indent=1)
    m = T.train_fold(cfg, a.fold, resume=a.resume)
    print(f'fold {a.fold}: UAR {m["uar"]:.2f}  WAR {m["war"]:.2f}  -> {out}')


if __name__ == '__main__':
    main()
