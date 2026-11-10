#!/usr/bin/env python
"""P2-OT (prereg.md): the P1 null ladder applied to the group's OT router (AV-DFER-OT, MMA-DFER host, SCR).

Imports the original code read-only (PYTHONDONTWRITEBYTECODE=1). Loads a dev run (trained on train90 of
the fold) and evaluates it on the official test fold, which the run never saw. The routing module sits
in every encoder block; each arm changes every block and reruns the whole model on the same batch.

At a routing block, frame j receives a2v_j = gate * proj(sum_k pi_jk a_k / r_j) (SCR rescale = 1 under
balanced marginals); v2a is the pooled mean frame under SCR and does not depend on the plan.
  base       released dev model
  copy       the re-implemented routing step with no change (must equal base exactly)
  zero       a2v and v2a removed in every block (the routing step off; native fusion stays)
  zero_a2v   a2v removed
  indep      pi' = r c^T / S (product of the plan's marginals: the Access paper's product control)
  uniform    pi' = a b^T (the pooled rule: every frame takes the slot-measure mean of the audio)
  rowperm    frame j takes the plan row of frame sigma(j), sigma random per clip and block (seed 0)
  devonly    a2v'_j = a2v_j - mean_j a2v_j
  shrink50 / shrink25   a2v'_j = mean + alpha (a2v_j - mean)
  papa       pi' = the mean plan over the fold's training clips, restricted to the clip's real slots and
             rescaled to the clip's row masses
  perm_a     the clip's audio (and its length) replaced by the next clip's in the batch (perceptual score)
"""

import argparse
import json
import os
import sys
import time

ARMS = ('base', 'copy', 'zero', 'zero_a2v', 'indep', 'uniform', 'rowperm', 'devonly', 'shrink50', 'shrink25',
        'papa', 'perm_a')

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--src', required=True, help='AV-DFER-OT repo (imported read-only)')
ap.add_argument('--run', required=True, help='dev run dir holding fold<k>/{config.json,model.pth}')
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--arms', nargs='+', default=list(ARMS))
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--limit', type=int, default=0, help='batches, 0 = all (smoke tests only)')
ap.add_argument('--out', required=True)
a = ap.parse_args()
assert a.arms[0] == 'base'

sys.path.insert(0, os.path.abspath(a.src))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from avdfer.data import build_dataset, timing_to  # noqa: E402
from avdfer.engine.evaluate import autocast  # noqa: E402
from avdfer.fusion import routing as RT  # noqa: E402
from avdfer.fusion.plans import uniform_plan  # noqa: E402
from avdfer.hosts import audio_input, build_host, load_checkpoint_state  # noqa: E402

DEV = torch.device('cuda')
ORIG = RT.OTRouting._forward


def routed(self, image_lowdim, audio_lowdim, n, t, timing=None):
    """OTRouting._forward (avdfer/fusion/routing.py @ bf34671) with intervention switches."""
    mode = getattr(self, '_p2_mode', None)
    if mode is None:
        return ORIG(self, image_lowdim, audio_lowdim, n, t, timing)
    v = self.video_tokens(image_lowdim, n, t)
    au = self.audio_tokens(audio_lowdim)
    pi, log_b = self.transport_plan(v, au, timing)
    if getattr(self, '_p2_accum', False):
        s = pi.detach().double().sum(0)
        self._p2_sum = s if self._p2_sum is None else self._p2_sum + s
        self._p2_n += pi.shape[0]
    if mode == 'indep':
        r = pi.sum(dim=2, keepdim=True)
        c = pi.sum(dim=1, keepdim=True)
        pi = r * c / pi.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    elif mode == 'uniform':
        pi = uniform_plan(pi, log_b)
    elif mode == 'papa':
        valid = torch.ones_like(pi[:, :1, :]) if log_b is None else (log_b.exp() > 0).to(pi.dtype).unsqueeze(1)
        q = self._p2_papa.to(pi.dtype).unsqueeze(0) * valid
        pi = q / q.sum(dim=2, keepdim=True).clamp_min(1e-12) * pi.sum(dim=2, keepdim=True)
    elif mode == 'rowperm':
        pi = torch.gather(pi, 1, self._p2_perm.unsqueeze(-1).expand_as(pi))

    row_mass = pi.sum(dim=2, keepdim=True)
    col_mass = pi.sum(dim=1, keepdim=True).transpose(1, 2)
    a2v = torch.bmm(pi / row_mass.clamp_min(1e-8), au)
    v2a = torch.bmm(pi.transpose(1, 2) / col_mass.clamp_min(1e-8), v)
    b = None if log_b is None else log_b.exp().unsqueeze(2)
    if self.rule in RT._SINKHORN_RULES:
        a2v = a2v * (row_mass * self.n_frames)
        if b is None:
            v2a = v2a * (col_mass * self.n_audio_t)
        else:
            v2a = v2a * (col_mass / b.clamp_min(1e-12))
    a2v = torch.tanh(self.gate_a2v) * self.proj_a2v(a2v)
    v2a = torch.tanh(self.gate_v2a) * self.proj_v2a(v2a)

    if mode == 'affine':
        m = a2v.mean(dim=1, keepdim=True)
        a2v = self._p2_beta * m + self._p2_alpha * (a2v - m)
    if mode in ('zero', 'zero_a2v'):
        a2v = torch.zeros_like(a2v)
    if mode == 'zero':
        v2a = torch.zeros_like(v2a)

    a2v = a2v.reshape(n * t, 1, self.dim)
    if self.v2a == 'slot':
        if b is not None:
            v2a = torch.where(b > 0, v2a, torch.zeros_like(v2a))
        return a2v, v2a
    if b is None:
        v2a = v2a.mean(dim=1, keepdim=True)
    else:
        v2a = (b * v2a).sum(dim=1, keepdim=True)
    return a2v, v2a


RT.OTRouting._forward = routed
SPEC = {'base': None, 'copy': 'copy', 'zero': 'zero', 'zero_a2v': 'zero_a2v', 'indep': 'indep',
        'uniform': 'uniform', 'rowperm': 'rowperm', 'papa': 'papa', 'devonly': ('affine', 0.0, 1.0),
        'shrink50': ('affine', 1.0, 0.5), 'shrink25': ('affine', 1.0, 0.25), 'perm_a': None}


def set_arm(mods, arm, n=0, t=16, it=0):
    s = SPEC[arm]
    for blk, m in enumerate(mods):
        m._p2_mode = s[0] if isinstance(s, tuple) else s
        if isinstance(s, tuple):
            m._p2_beta, m._p2_alpha = s[1], s[2]
        if m._p2_mode == 'rowperm':
            g = torch.Generator().manual_seed(a.seed * 10_000_000 + a.fold * 1_000_000 + it * 100 + blk)
            m._p2_perm = torch.argsort(torch.rand(n, t, generator=g), dim=1).to(DEV)


def make_loader(cfg, ann, shuffle=False):
    d = cfg['data']
    ds = build_dataset(ann, d['dataset'], 'test', cfg['host']['img_size'], frames_root=d['frames_root'],
                       audio_root=d['audio_root'], missing_audio=d['missing_audio'],
                       frame_order=d.get('frame_order', 'lexicographic'),
                       audio_channels=d.get('audio_channels', 'first'),
                       frame_times=d.get('frame_times', 'uniform'), **audio_input(cfg))
    return ds, torch.utils.data.DataLoader(ds, batch_size=cfg['eval']['batch_size'], shuffle=shuffle,
                                           num_workers=a.workers)


def ann_path(cfg, part):
    d = cfg['data']
    suffix = '_faces' if d['dataset'] == 'MAFW' else ''
    p = os.path.join(a.src, d.get('annotation_dir', 'annotation'), f"{d['dataset']}_set_{a.fold}_{part}{suffix}.txt")
    assert os.path.exists(p), p
    return p


def main():
    fold_dir = os.path.join(a.run, f'fold{a.fold}')
    cfg = json.load(open(os.path.join(fold_dir, 'config.json')))
    model = build_host(cfg)
    state = torch.load(os.path.join(fold_dir, 'model.pth'), map_location='cpu', weights_only=False)['state_dict']
    load_checkpoint_state(model, state)
    model.to(DEV).eval()
    mods = model.routing_modules
    prec = cfg['train'].get('precision', 'fp32')
    print(f'{a.run} fold {a.fold}: {len(mods)} routing modules, rule {mods[0].rule}, precision {prec}', flush=True)

    # PAPA: mean plan over the fold's training clips (the run's train90 list)
    papa_n = 0
    if 'papa' in a.arms:
        train_ann = os.path.join(a.src, cfg['train_annotation_file'])
        _, tl = make_loader(cfg, train_ann)
        set_arm(mods, 'copy')
        for m in mods:
            m._p2_accum, m._p2_sum, m._p2_n = True, None, 0
        t0 = time.time()
        with torch.no_grad(), autocast(DEV, prec):
            for it, (images, _, audio, _, timing) in enumerate(tl):
                if a.limit and it >= a.limit:
                    break
                model(images.to(DEV), audio.to(DEV), timing_to(timing, DEV))
        for m in mods:
            m._p2_accum = False
            m._p2_papa = (m._p2_sum / m._p2_n).float()
            papa_n = m._p2_n
        print(f'papa: {papa_n} training clips ({time.time() - t0:.0f}s)', flush=True)

    test_ann = ann_path(cfg, 'test')
    ds, loader = make_loader(cfg, test_ann)
    print(f'test: {len(ds)} clips ({test_ann})', flush=True)
    logits = {arm: [] for arm in a.arms}
    labels, check, prev = [], None, None
    t0 = time.time()
    with torch.no_grad(), autocast(DEV, prec):
        for it, (images, label, audio, _, timing) in enumerate(loader):
            if a.limit and it >= a.limit:
                break
            images, audio = images.to(DEV), audio.to(DEV)
            tm = timing_to(timing, DEV)
            n, t = images.shape[:2]
            for arm in a.arms:
                set_arm(mods, arm, n, t, it)
                if arm == 'perm_a':
                    if n > 1:
                        idx = (torch.arange(n, device=DEV) + 1) % n
                        aud, tmp = audio[idx], {k: (v[idx] if torch.is_tensor(v) and v.shape[:1] == (n,) else v)
                                                for k, v in tm.items()}
                    else:                        # a batch of one clip: partner from the previous batch
                        aud, tmp = prev
                    out = model(images, aud, tmp)
                else:
                    out = model(images, audio, tm)
                logits[arm].append(out.float().cpu())
            prev = (audio[-1:], {k: (v[-1:] if torch.is_tensor(v) and v.shape[:1] == (n,) else v)
                                 for k, v in tm.items()})
            if check is None and 'copy' in a.arms:
                check = (logits['copy'][-1] - logits['base'][-1]).abs().max().item()
                print(f'check: max |copy - base| = {check:.2e}', flush=True)
                assert check < 1e-4, check
            labels.append(label)
            if it % 50 == 0:
                print(f'  batch {it}/{len(loader)}  {time.time() - t0:.0f}s', flush=True)
    set_arm(mods, 'base')
    y = torch.cat(labels).numpy()
    P = {arm: torch.softmax(torch.cat(v).double(), dim=-1).numpy() for arm, v in logits.items()}
    pred = {arm: p.argmax(1) for arm, p in P.items()}
    ncls = cfg['data']['num_classes']
    res = {'run': a.run, 'fold': a.fold, 'n_clips': int(len(y)), 'test_annotation': test_ann, 'limit': a.limit,
           'check_copy_vs_base': check, 'papa_train_clips': papa_n, 'arms': {},
           'gate_tanh_a2v': [float(torch.tanh(m.gate_a2v).item()) for m in mods],
           'gate_tanh_v2a': [float(torch.tanh(m.gate_v2a).item()) for m in mods]}
    for arm in a.arms:
        rec = [(pred[arm][y == c] == c).mean() for c in range(ncls) if (y == c).any()]
        res['arms'][arm] = {'uar': 100.0 * float(np.mean(rec)), 'war': 100.0 * float((pred[arm] == y).mean()),
                            'changed_pct': 100.0 * float((pred[arm] != pred['base']).mean())}
    with open(a.out + '.json', 'w') as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(a.out + '.npz', clips=np.arange(len(y)), labels=y,
                        **{f'prob_{arm}': P[arm] for arm in a.arms})
    print(json.dumps({k: (round(v['uar'], 2), round(v['changed_pct'], 2)) for k, v in res['arms'].items()}))
    print(f'-> {a.out}.json  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
