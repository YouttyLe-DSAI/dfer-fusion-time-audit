#!/usr/bin/env python
"""Block-subset product replacement on the group's OT router (prereg.md, 2026-11-11).

indep = the product of the plan's marginals, pi' = r c^T / S (the Access paper's product control, as in P2-OT),
applied only at the listed blocks; the plan's three fixed subsets:
  base, indep_all (1-12), indep_1_9, indep_10_12
Every arm reruns the whole model on the same batch. Descriptive trace against base, per clip (MMA-DFER: the
temporal-transformer input [n, T, 512] split into clip-mean and frame-specific parts, its output; all hosts: the
logits).

Imports the AV-DFER-OT code read-only (PYTHONDONTWRITEBYTECODE=1); writes only --out(.json, .npz).
"""

import argparse
import json
import os
import sys
import time

CTL = ('ctl_meanonly', 'ctl_frameonly', 'ctl_rand0', 'ctl_rand1', 'ctl_rand2', 'ctl_perm')
ARMS = ('base', 'indep_all', 'indep_1_9', 'indep_10_12', 'exact_all') + CTL
DEFAULT_ARMS = ARMS[:4]
SUBSETS = {'base': set(), 'indep_all': set(range(1, 13)), 'indep_1_9': set(range(1, 10)),
           'indep_10_12': {10, 11, 12}, 'exact_all': set(range(1, 13)), **{c: set() for c in CTL}}
EXACT = {'exact_all'}   # opt-in (prereg "Exact product reference"): the uniform plan a x b over real slots, cost-free
# CTL (prereg "Head attenuation control"), opt-in: real plans everywhere, and the temporal-transformer input replaced by
# x0 + a perturbation built from delta = x(indep_10_12) - x0 of the same batch; they need indep_10_12 earlier in --arms.

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--src', required=True)
ap.add_argument('--run', required=True)
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--limit', type=int, default=0, help='batches, 0 = all (smoke tests only)')
ap.add_argument('--set', nargs='+', action='extend', default=[], metavar='KEY=VALUE')
ap.add_argument('--arms', nargs='+', default=list(DEFAULT_ARMS), choices=ARMS)
ap.add_argument('--out', required=True)
a = ap.parse_args()
assert a.arms[0] == 'base'
assert not any(c in a.arms for c in CTL) or ('indep_10_12' in a.arms and
                                             max(a.arms.index(c) for c in CTL if c in a.arms) > a.arms.index('indep_10_12')
                                             and min(a.arms.index(c) for c in CTL if c in a.arms) > a.arms.index('indep_10_12'))
ARMS = tuple(a.arms)

sys.path.insert(0, os.path.abspath(os.path.expanduser(a.src)))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from avdfer.data import build_dataset, timing_to  # noqa: E402
from avdfer.engine.evaluate import autocast  # noqa: E402
from avdfer.fusion import routing as RT  # noqa: E402
from avdfer.fusion.plans import uniform_plan  # noqa: E402
from avdfer.hosts import audio_input, build_host, load_checkpoint_state  # noqa: E402
from avdfer.run.config import apply_overrides  # noqa: E402
from e6_costdev import attach_cost_dev, cost_gates  # noqa: E402  (E6 branch; no-op for runs without it)

DEV = torch.device('cuda')
TINY = 1e-12
ORIG_TP = RT.OTRouting.transport_plan


def transport_plan(self, v, a_, timing=None):
    pi, log_b = ORIG_TP(self, v, a_, timing)
    if getattr(self, '_bs_track', None) is not None and self.rule != 'pooled':    # plan residuals (E4)
        q = pi.float() / pi.float().sum(dim=(1, 2), keepdim=True)
        row = (q.sum(2) * q.shape[1] - 1).abs().amax(1)
        b = torch.full_like(q[:, 0, :], 1.0 / q.shape[2]) if log_b is None else log_b.exp().float()
        col = torch.where(b > 0, (q.sum(1) / b.clamp_min(1e-12) - 1).abs(), torch.zeros_like(b)).amax(1)
        self._bs_track.append(torch.stack([row, col], 1).cpu())
    if getattr(self, '_bs_exact', False):       # exact product of the target marginals (the pooled rule's plan)
        return uniform_plan(pi, log_b), log_b
    if getattr(self, '_bs_indep', False):
        r = pi.sum(dim=2, keepdim=True)
        c = pi.sum(dim=1, keepdim=True)
        pi = r * c / pi.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    return pi, log_b


RT.OTRouting.transport_plan = transport_plan


def set_arm(model, arm):
    for b, m in zip(model.fusion_blocks, model.routing_modules):
        m._bs_indep = b in SUBSETS[arm] and arm not in EXACT
        m._bs_exact = b in SUBSETS[arm] and arm in EXACT
        if arm == 'base':                       # record the real plan's residuals only
            if getattr(m, '_bs_track', None) is None:
                m._bs_track = getattr(m, '_bs_tracked', None) or []
        elif isinstance(getattr(m, '_bs_track', None), list):
            m._bs_tracked, m._bs_track = m._bs_track, None


def control_input(arm, x0, x10, it):
    """Temporal-transformer input of a control arm (prereg "Head attenuation control"); x0, x10: [n, T, D] float."""
    d = x10 - x0
    dm = d.mean(1, keepdim=True)
    df = d - dm
    if arm == 'ctl_meanonly':
        return x0 + dm
    if arm == 'ctl_frameonly':
        return x0 + df
    if arm.startswith('ctl_rand'):
        g = torch.Generator(device=x0.device).manual_seed(100003 * int(arm[-1]) + it)
        xi = torch.randn(x0.shape, generator=g, device=x0.device, dtype=x0.dtype)
        xi = xi - xi.mean(1, keepdim=True)
        scale = df.flatten(1).norm(dim=1) / xi.flatten(1).norm(dim=1).clamp_min(TINY)
        return x0 + xi * scale[:, None, None]
    if arm == 'ctl_perm':
        g = torch.Generator(device='cpu').manual_seed(7 * 100003 + it)
        idx = torch.stack([torch.randperm(x0.shape[1], generator=g) for _ in range(x0.shape[0])]).to(x0.device)
        return x0 + torch.gather(df, 1, idx[:, :, None].expand_as(df))
    raise ValueError(arm)


def main():
    fold_dir = os.path.join(a.run, f'fold{a.fold}')
    cfg = apply_overrides(json.load(open(os.path.join(fold_dir, 'config.json'))), a.set)
    model = attach_cost_dev(build_host(cfg), cfg)
    state = torch.load(os.path.join(fold_dir, 'model.pth'), map_location='cpu', weights_only=False)['state_dict']
    load_checkpoint_state(model, state)
    model.to(DEV).eval()
    assert list(model.fusion_blocks) == list(range(1, 13)), model.fusion_blocks
    prec = cfg['train'].get('precision', 'fp32')
    tnet = getattr(model, 'temporal_net', None)
    cap, ctl = {}, {}
    if tnet is not None:
        tnet.register_forward_hook(lambda m, inp, out: cap.update(x=inp[0].float(), o=out.float()))
        tnet.register_forward_pre_hook(lambda m, inp: (ctl['x'].to(inp[0].dtype),) if ctl.get('x') is not None else None)

    d = cfg['data']
    suffix = '_faces' if d['dataset'] == 'MAFW' else ''
    ann = os.path.join(os.path.expanduser(a.src), d.get('annotation_dir', 'annotation'),
                       f"{d['dataset']}_set_{a.fold}_test{suffix}.txt")
    assert os.path.exists(ann), ann
    ds = build_dataset(ann, d['dataset'], 'test', cfg['host']['img_size'], frames_root=d['frames_root'],
                       audio_root=d['audio_root'], missing_audio=d['missing_audio'],
                       frame_order=d.get('frame_order', 'lexicographic'),
                       audio_channels=d.get('audio_channels', 'first'),
                       frame_times=d.get('frame_times', 'uniform'), **audio_input(cfg))
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg['eval']['batch_size'], shuffle=False,
                                         num_workers=a.workers)
    print(f'{a.run} fold {a.fold}: test {len(ds)} clips ({ann}), precision {prec}, trace {tnet is not None}', flush=True)
    logits = {arm: [] for arm in ARMS}
    trace = {arm: {k: [] for k in ('in_mean', 'in_frame', 'out', 'logit')} for arm in ARMS[1:]}
    labels = []
    t0 = time.time()
    with torch.no_grad(), autocast(DEV, prec):
        for it, (images, label, audio, _, timing) in enumerate(loader):
            if a.limit and it >= a.limit:
                break
            images, audio = images.to(DEV), audio.to(DEV)
            tm = timing_to(timing, DEV)
            ref = None
            for arm in ARMS:
                set_arm(model, arm)
                if arm in CTL:
                    ctl['x'] = control_input(arm, ref[0], x10, it)
                out = model(images, audio, tm).float()
                ctl['x'] = None
                if arm == 'indep_10_12' and tnet is not None:
                    x10 = cap['x']
                logits[arm].append(out.cpu())
                if arm == 'base':
                    ref = (cap.get('x'), cap.get('o'), out)
                    continue
                tr = trace[arm]
                tr['logit'].append(((out - ref[2]).norm(dim=-1) / ref[2].norm(dim=-1).clamp_min(TINY)).cpu())
                if tnet is not None:
                    x0, x1 = ref[0], cap['x']
                    m0, m1 = x0.mean(1), x1.mean(1)
                    dv0, dv1 = x0 - m0.unsqueeze(1), x1 - m1.unsqueeze(1)
                    tr['in_mean'].append(((m1 - m0).norm(dim=-1) / m0.norm(dim=-1).clamp_min(TINY)).cpu())
                    tr['in_frame'].append(((dv1 - dv0).pow(2).sum(-1).mean(1).sqrt()
                                           / dv0.pow(2).sum(-1).mean(1).sqrt().clamp_min(TINY)).cpu())
                    tr['out'].append(((cap['o'] - ref[1]).norm(dim=-1) / ref[1].norm(dim=-1).clamp_min(TINY)).cpu())
            labels.append(label)
            if it % 50 == 0:
                print(f'  batch {it}/{len(loader)}  {time.time() - t0:.0f}s', flush=True)
    set_arm(model, 'base')
    y = torch.cat(labels).numpy()
    P = {arm: torch.softmax(torch.cat(v).double(), dim=-1).numpy() for arm, v in logits.items()}
    pred = {arm: p.argmax(1) for arm, p in P.items()}
    ncls = cfg['data']['num_classes']
    res = {'run': a.run, 'fold': a.fold, 'n_clips': int(len(y)), 'test_annotation': ann, 'limit': a.limit,
           'arms': {}, 'trace': {},
           'gate_tanh_a2v': [float(torch.tanh(m.gate_a2v).item()) for m in model.routing_modules]}
    if cost_gates(model):
        res['cost_gate_tanh'] = cost_gates(model)
    if hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'all_gate'):
        res['tanh_all_gate'] = [float(torch.tanh(g).item()) for g in model.image_encoder.all_gate]
    for arm in ARMS:
        rec = [(pred[arm][y == c] == c).mean() for c in range(ncls) if (y == c).any()]
        res['arms'][arm] = {'uar': 100.0 * float(np.mean(rec)), 'war': 100.0 * float((pred[arm] == y).mean()),
                            'changed_pct': 100.0 * float((pred[arm] != pred['base']).mean())}
    for arm, tr in trace.items():
        res['trace'][arm] = {k: float(torch.cat(v).double().mean()) for k, v in tr.items() if v}
    resid = [getattr(m, '_bs_tracked', None) or getattr(m, '_bs_track', None) for m in model.routing_modules]
    if all(r for r in resid):
        R = [torch.cat(r) for r in resid]                       # per block [n_clip, 2] (row, col), real plan
        res['plan_residual'] = {'eps': float(model.routing_modules[0].eps), 'n_iters': model.routing_modules[0].n_iters,
                                'row_mean': float(torch.stack([x[:, 0].mean() for x in R]).mean()),
                                'col_mean': float(torch.stack([x[:, 1].mean() for x in R]).mean()),
                                'row_max': float(max(x[:, 0].max() for x in R)),
                                'col_max': float(max(x[:, 1].max() for x in R)),
                                'col_mean_per_block': [float(x[:, 1].mean()) for x in R]}
        print('plan residuals (real plan):', json.dumps({k: v for k, v in res['plan_residual'].items()
                                                         if k != 'col_mean_per_block'}))
    with open(a.out + '.json', 'w') as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(a.out + '.npz', clips=np.arange(len(y)), labels=y, **{f'prob_{arm}': P[arm] for arm in ARMS})
    print(json.dumps({k: (round(v['uar'], 3), round(v['changed_pct'], 2)) for k, v in res['arms'].items()}))
    print(json.dumps(res['trace']))
    print(f'-> {a.out}.json  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
