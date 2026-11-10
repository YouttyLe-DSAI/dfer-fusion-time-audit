#!/usr/bin/env python
"""Stage 0b (prereg.md): cache frozen MAE-Face features for every labelled DFEW clip.

Encoder: the MMA-DFER host's visual encoder weights (MAE-Face ViT-B/16, 160 px), run as a plain ViT:
no prompts, no MMA-DFER adapters, eval mode. 16 frames per clip at the loader's test indices, with the
loader's test transform (the group's code, imported read-only).

Writes, under --out:
  clips.txt                 clip id per row (the row order of every array)
  labels.npy                [N] int64, the 0-based label of the group's annotation
  b12.npy                   [N, 16, 17, 768] fp16: final-norm block-12 CLS + 4x4 region means
  b9.npy                    [N, 16, 17, 768] fp16: raw block-9 CLS + 4x4 region means
"""

import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--frames_root', required=True)
ap.add_argument('--anns', nargs='+', required=True, help='annotation files whose union is extracted')
ap.add_argument('--video_ckpt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--workers', type=int, default=12)
ap.add_argument('--batch', type=int, default=16, help='clips per batch')
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()

sys.path.insert(0, os.path.abspath(a.src))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from avdfer.data import build_dataset  # noqa: E402
from avdfer.hosts.mma_dfer import model as MM  # noqa: E402
from avdfer.hosts.mma_dfer import models_vit  # noqa: E402

DEV = torch.device('cuda')
IMG, T = 160, 16


def build_encoder():
    enc = models_vit.vit_base_patch16(global_pool=False, num_classes=12, drop_path_rate=0.0, img_size=IMG,
                                      n_seq=(IMG // 16) ** 2, n_progr=3, n_frames=T)
    ck = torch.load(a.video_ckpt, map_location='cpu', weights_only=False)['model']
    ck['pos_embed'] = MM.resize_pos_embed(ck['pos_embed'], enc.pos_embed)
    res = enc.load_state_dict(ck, strict=False)
    needed = [k for k in enc.state_dict() if k.startswith(('patch_embed', 'blocks', 'cls_token', 'pos_embed', 'norm.'))]
    missing = [k for k in needed if k in res.missing_keys]
    assert not missing, missing
    assert enc.pos_embed.shape[1] == 1 + (IMG // 16) ** 2, enc.pos_embed.shape
    return enc.to(DEV).eval()


@torch.no_grad()
def features(enc, img):
    """img [B, 3, 160, 160] -> (b9, b12): [B, 17, 768] each (CLS + 4x4 region means)."""
    x = enc.patch_embed(img)
    x = torch.cat([enc.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + enc.pos_embed
    x = enc.pos_drop(x)
    b9 = None
    for i, blk in enumerate(enc.blocks):
        x = blk(x)
        if i == 8:
            b9 = x
    b12 = enc.norm(x)
    g = IMG // 16

    def pack(t):
        grid = t[:, 1:].transpose(1, 2).reshape(t.shape[0], t.shape[2], g, g)
        r4 = F.adaptive_avg_pool2d(grid, 4).flatten(2).transpose(1, 2)          # [B, 16, 768]
        return torch.cat([t[:, :1], r4], dim=1)

    return pack(b9), pack(b12)


def main():
    os.makedirs(a.out, exist_ok=True)
    # union of the annotation files, first occurrence wins, sorted by clip id
    seen = {}
    for ann in a.anns:
        for line in open(ann):
            p = line.split()
            if p:
                seen.setdefault(os.path.basename(p[0]), line)
    ids = sorted(seen)
    if a.limit:
        ids = ids[:a.limit]
    union = os.path.join(a.out, 'union_annotation.txt')
    with open(union, 'w') as fh:
        fh.writelines(seen[i] if seen[i].endswith('\n') else seen[i] + '\n' for i in ids)
    ds = build_dataset(union, 'DFEW', 'test', IMG, frames_root=a.frames_root, audio_root='/nonexistent/{dataset}',
                       missing_audio='zeros')
    assert [os.path.basename(c.frames_dir) for c in ds.clips] == ids
    n = len(ds)
    print(f'{n} clips from {a.anns}', flush=True)
    enc = build_encoder()
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers)
    m9 = np.lib.format.open_memmap(os.path.join(a.out, 'b9.npy'), mode='w+', dtype=np.float16, shape=(n, T, 17, 768))
    m12 = np.lib.format.open_memmap(os.path.join(a.out, 'b12.npy'), mode='w+', dtype=np.float16, shape=(n, T, 17, 768))
    labels = np.zeros(n, dtype=np.int64)
    t0, k = time.time(), 0
    for it, (images, label, _, index, _) in enumerate(loader):
        b = images.shape[0]
        img = images.to(DEV, non_blocking=True).reshape(b * T, 3, IMG, IMG)
        with torch.autocast('cuda', dtype=torch.float16):
            f9, f12 = features(enc, img)
        idx = index.numpy()
        assert (idx == np.arange(k, k + b)).all()
        m9[k:k + b] = f9.float().reshape(b, T, 17, 768).cpu().numpy().astype(np.float16)
        m12[k:k + b] = f12.float().reshape(b, T, 17, 768).cpu().numpy().astype(np.float16)
        labels[k:k + b] = label.numpy()
        k += b
        if it % 50 == 0:
            print(f'  {k}/{n}  {time.time() - t0:.0f}s', flush=True)
    assert k == n
    m9.flush()
    m12.flush()
    np.save(os.path.join(a.out, 'labels.npy'), labels)
    with open(os.path.join(a.out, 'clips.txt'), 'w') as fh:
        fh.write('\n'.join(ids) + '\n')
    print(f'-> {a.out}  {n} clips  {time.time() - t0:.0f}s; label counts {np.bincount(labels).tolist()}', flush=True)


if __name__ == '__main__':
    main()
