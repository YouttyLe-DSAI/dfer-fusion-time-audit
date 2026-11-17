#!/usr/bin/env python
"""E2 (prereg.md): cost-space spread and interaction at every routing block of the group's OT router.

At each routing block, on the tensors the module itself builds (fp32, as the module runs):
  v      per-frame CLS descriptor [n, T, D]                       (D1: frame_spread = spread(v))
  v_ln   norm_v(v);  v_cs = normalize(v_ln)                       (D2: what the cosine cost reads)
  p      mean of the frame's patch tokens [n, T, D]; p_cs = normalize(norm_v(p))
         (A1 proxy: uses A0's LayerNorm, which was trained on the CLS)
  a      audio slots [n, M, D]; a_cs = normalize(norm_a(a)), statistics over the usable slots (mask_padding)
Per clip:
  spread(x)   mean_j ||x_j - xbar|| / ||xbar||                    (D1)
  sigma(x)    sqrt(mean_j ||x_j - xbar||^2)                       (absolute RMS deviation, the bound's input, D6)
  kappa       RMS of the double-centred cost over frames x usable slots / eps   (the Access paper's kappa_int)
  beta        sigma(v_cs) sigma(a_cs) / eps   (>= kappa: I_jk = -(v_j - vbar).(a_k - abar), Cauchy-Schwarz, B = I)
  beta_p      sigma(p_cs) sigma(a_cs) / eps
  H           mean_j TV(pi_j / r_j, c / S)                        (plan departure from the product of its marginals)
  frame_part  mean_j ||a2v_j - mean_j a2v|| / mean image-token norm
Checks: the recomputed cost equals mod.cost (max |diff|); kappa <= beta per clip (violations counted).

Imports the AV-DFER-OT code read-only (PYTHONDONTWRITEBYTECODE=1 expected); writes only --out(.json, .npz).

    python e2_ot.py --src ~/AVDFEROT/AV-DFER-OT --run <src>/runs/CREMAD/dev/scr-f1dc1fc2dff2 \
        --split test --folds 1 2 3 4 5 --out results/e2/OT_CREMAD_test
"""

import argparse
import json
import os
import sys
from collections import defaultdict

TINY = 1e-12
CLIP_KEYS = ('raw_spread_v', 'ln_spread_v', 'cs_spread_v', 'sigma_v', 'vbar_norm_cs',
             'raw_spread_p', 'cs_spread_p', 'sigma_p',
             'raw_spread_a_all', 'cs_spread_a', 'sigma_a',
             'kappa', 'beta', 'beta_p', 'H', 'frame_part')
NPZ_KEYS = ('raw_spread_v', 'cs_spread_v', 'sigma_v', 'cs_spread_p', 'sigma_p', 'cs_spread_a', 'sigma_a',
            'kappa', 'beta', 'beta_p', 'H')


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='AV-DFER-OT source tree (read-only)')
    ap.add_argument('--run', required=True)
    ap.add_argument('--folds', nargs='+', type=int, required=True)
    ap.add_argument('--split', choices=('test', 'val10'), required=True)
    ap.add_argument('--limit', type=int, default=0, help='clips per fold, 0 = all (smoke tests only)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--set', nargs='+', action='extend', default=[], metavar='KEY=VALUE')
    ap.add_argument('--reference', default='', help='stage 0a JSON for sanity check 1 (val10)')
    ap.add_argument('--out', required=True, help='output prefix')
    return ap.parse_args()


args = parse()
sys.path.insert(0, os.path.abspath(os.path.expanduser(args.src)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from avdfer.data import build_dataset, timing_to  # noqa: E402
from avdfer.engine.evaluate import autocast  # noqa: E402
from avdfer.hosts import audio_input, build_host, load_checkpoint_state  # noqa: E402
from avdfer.run.config import annotation_paths, apply_overrides  # noqa: E402
from e6_costdev import attach_cost_dev  # noqa: E402  (E6 branch; no-op for runs without it)


def dev_stats(x, w=None):
    """x [n, K, D], w [n, K] weights (None = all) -> spread [n], sigma [n], ||xbar|| [n]."""
    if w is None:
        w = torch.ones(x.shape[:2], dtype=x.dtype, device=x.device)
    ws = w.sum(1).clamp_min(TINY)
    mu = (x * w.unsqueeze(-1)).sum(1) / ws.unsqueeze(-1)
    d = (x - mu.unsqueeze(1)).norm(dim=-1)
    mean_d = (d * w).sum(1) / ws
    sigma = ((d.pow(2) * w).sum(1) / ws).sqrt()
    mun = mu.norm(dim=-1)
    return mean_d / mun.clamp_min(TINY), sigma, mun


def usable_slots(mod, timing, n, device):
    if timing is None or not mod.mask_padding:
        return torch.ones(n, mod.n_audio_t, dtype=torch.bool, device=device)
    valid = timing['slot_valid']
    return valid | ~valid.any(1, keepdim=True)


def cost_int_std(C, usable):
    """As upstream tools/routing_diagnostics.py cost_stats: RMS of the double-centred cost over usable slots."""
    w = usable.float().unsqueeze(1)
    n_slots = w.sum(2, keepdim=True)
    row = (C * w).sum(2, keepdim=True) / n_slots
    col = (C * w).mean(1, keepdim=True)
    m = row.mean(1, keepdim=True)
    inter = (C - row - col + m) * w
    return (inter.pow(2).sum((1, 2)) / (n_slots.flatten() * C.shape[1])).sqrt()


class Block:
    def __init__(self):
        self.vals = defaultdict(list)
        self.cost_diff = 0.0
        self.viol = 0
        self.viol_rel = 0.0
        self.gate = None
        self.cost_gate = None

    def add(self, d):
        for k, v in d.items():
            self.vals[k].append(v.detach().float().cpu())


def block_stats(mod, img, aud, n, t, timing, out, acc):
    assert mod.video_token in ('cls', 'mean'), mod.video_token      # v = what the cost reads
    assert getattr(mod, 'time_lambda', 0.0) == 0.0
    with torch.autocast(device_type=img.device.type, enabled=False):
        img, aud = img.float(), aud.float()
        if timing is not None:
            timing = {k: (x.float() if x.is_floating_point() else x) for k, x in timing.items()}
        eps = float(mod.eps)
        v = getattr(mod, 'video_tokens_cost', mod.video_tokens)(img, n, t)  # [n, T, D] CLS (E6: + branch)
        a = mod.audio_tokens(aud)                                           # [n, M, D]
        x = img.view(n, t, -1, mod.dim)
        p = x[:, :, 1:1 + mod.n_image, :].mean(dim=2)                       # patch mean
        usable = usable_slots(mod, timing, n, img.device)
        wu = usable.float()
        v_ln = mod.norm_v(v)
        v_cs = F.normalize(v_ln, dim=-1)
        p_cs = F.normalize(mod.norm_v(p), dim=-1)
        a_cs = F.normalize(mod.norm_a(a), dim=-1)
        C = mod.cost(v, a)
        acc.cost_diff = max(acc.cost_diff, (1.0 - torch.bmm(v_cs, a_cs.transpose(1, 2)) - C).abs().max().item())
        pi, log_b = mod.transport_plan(v, a, timing)
        pi = pi.float()

        raw_v, _, _ = dev_stats(v)
        ln_v, _, _ = dev_stats(v_ln)
        cs_v, sig_v, vbar = dev_stats(v_cs)
        raw_p, _, _ = dev_stats(p)
        cs_p, sig_p, _ = dev_stats(p_cs)
        raw_a, _, _ = dev_stats(a)
        cs_a, sig_a, _ = dev_stats(a_cs, wu)
        kappa = cost_int_std(C, usable) / eps
        beta = sig_v * sig_a / eps
        viol = kappa > beta * (1 + 1e-5) + 1e-7
        acc.viol += int(viol.sum())
        if viol.any():
            acc.viol_rel = max(acc.viol_rel, ((kappa - beta) / beta.clamp_min(TINY))[viol].max().item())

        r = pi.sum(2, keepdim=True)
        c = pi.sum(1, keepdim=True)
        S = pi.sum((1, 2), keepdim=True)
        q = pi / r.clamp_min(TINY)
        H = 0.5 * (q - c / S.clamp_min(TINY)).abs().sum(2).mean(1)

        a2v = out[0].float().reshape(n, t, -1)
        part = (a2v - a2v.mean(1, keepdim=True)).norm(dim=-1).mean(1)
        token = img.norm(dim=-1).mean(1).view(n, t).mean(1)
        acc.gate = float(torch.tanh(mod.gate_a2v).abs().item())
        if hasattr(mod, 'cost_dev'):
            acc.cost_gate = float(torch.tanh(mod.cost_dev.gate).item())
        acc.add({'raw_spread_v': raw_v, 'ln_spread_v': ln_v, 'cs_spread_v': cs_v, 'sigma_v': sig_v,
                 'vbar_norm_cs': vbar, 'raw_spread_p': raw_p, 'cs_spread_p': cs_p, 'sigma_p': sig_p,
                 'raw_spread_a_all': raw_a, 'cs_spread_a': cs_a, 'sigma_a': sig_a,
                 'kappa': kappa, 'beta': beta, 'beta_p': sig_p * sig_a / eps, 'H': H,
                 'frame_part': part / token.clamp_min(TINY)})


def ann_for(cfg, fold, split):
    d = cfg['data']
    if split == 'val10':
        return annotation_paths(cfg, fold)[1].format(fold=fold)
    suffix = '_faces' if d['dataset'] == 'MAFW' else ''
    p = os.path.join(os.path.expanduser(args.src), d.get('annotation_dir', 'annotation'),
                     f"{d['dataset']}_set_{fold}_test{suffix}.txt")
    assert os.path.exists(p), p
    return p


def measure_fold(cfg, fold_dir, ann, device):
    model = attach_cost_dev(build_host(cfg), cfg)
    state = torch.load(os.path.join(fold_dir, 'model.pth'), map_location='cpu', weights_only=False)['state_dict']
    load_checkpoint_state(model, state)
    model.to(device).eval()
    mods = model.routing_modules
    accs = [Block() for _ in mods]

    def hook_for(i):
        def hook(mod, inp, out):
            img, aud, n, t, *rest = inp
            assert rest or not mod.needs_timing
            block_stats(mod, img, aud, n, t, rest[0] if rest else None, out, accs[i])
        return hook

    for i, mod in enumerate(mods):
        mod.register_forward_hook(hook_for(i))

    d = cfg['data']
    ds = build_dataset(ann, d['dataset'], 'test', cfg['host']['img_size'], frames_root=d['frames_root'],
                       audio_root=d['audio_root'], missing_audio=d['missing_audio'],
                       frame_order=d.get('frame_order', 'lexicographic'),
                       audio_channels=d.get('audio_channels', 'first'),
                       frame_times=d.get('frame_times', 'uniform'), **audio_input(cfg))
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg['eval']['batch_size'], shuffle=False,
                                         num_workers=args.workers)
    seen = 0
    with torch.no_grad(), autocast(device, cfg['train'].get('precision', 'fp32')):
        for images, _, audio, _, timing in loader:
            if args.limit and seen >= args.limit:
                break
            model(images.to(device), audio.to(device), timing_to(timing, device))
            seen += images.size(0)

    blocks, arrays = [], {}
    for i, acc in enumerate(accs):
        row = {'block': model.fusion_blocks[i], 'gate_a2v_abs': acc.gate}
        if acc.cost_gate is not None:
            row['cost_gate_tanh'] = acc.cost_gate
        for k in CLIP_KEYS:
            v = torch.cat(acc.vals[k]).double()
            row[k] = float(v.mean())
            row[k + '_med'] = float(v.median())
            row[k + '_p90'] = float(torch.quantile(v, 0.9))
        blocks.append(row)
        for k in NPZ_KEYS:
            arrays.setdefault(k, []).append(torch.cat(acc.vals[k]).numpy())
    checks = {'cost_max_abs_diff': max(a.cost_diff for a in accs), 'kappa_gt_beta': sum(a.viol for a in accs),
              'kappa_gt_beta_max_rel': max(a.viol_rel for a in accs)}
    n_clip = int(torch.cat(accs[0].vals['kappa']).numel())
    return {'n_clip': n_clip, 'eps': float(mods[0].eps), 'blocks': blocks, 'checks': checks,
            'arrays': {k: np.stack(v, 1).astype(np.float32) for k, v in arrays.items()}}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    per_fold, n_clip, checks, npz = {}, {}, {}, {}
    for fold in args.folds:
        fold_dir = os.path.join(args.run, f'fold{fold}')
        with open(os.path.join(fold_dir, 'config.json')) as f:
            cfg = apply_overrides(json.load(f), args.set)
        if cfg['fusion']['rule'] == 'none':
            raise SystemExit('the backbone has no routing module')
        ann = ann_for(cfg, fold, args.split)
        res = measure_fold(cfg, fold_dir, ann, device)
        per_fold[fold], n_clip[fold], checks[fold] = res['blocks'], res['n_clip'], res['checks']
        for k, v in res['arrays'].items():
            npz[f'fold{fold}_{k}'] = v
        print(f'fold {fold}: {res["n_clip"]} clips ({ann}); checks {res["checks"]}', flush=True)

    first = per_fold[args.folds[0]]
    keys = [k for k in first[0] if k != 'block']
    blocks = []
    for i, r in enumerate(first):
        row = {'block': r['block']}
        for k in keys:
            vals = np.array([per_fold[f][i][k] for f in args.folds], dtype=float)
            row[k] = float(vals.mean())
            if not (k.endswith('_med') or k.endswith('_p90')):
                row[k + '_fstd'] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                row[k + '_fmin'], row[k + '_fmax'] = float(vals.min()), float(vals.max())
        blocks.append(row)
    res = {'run': args.run, 'split': args.split, 'folds': args.folds, 'host': cfg['host']['name'],
           'dataset': cfg['data']['dataset'], 'rule': cfg['fusion']['rule'], 'eps': cfg['fusion'].get('eps'),
           'video_token': cfg['fusion'].get('video_token'),
           'limit': args.limit, 'n_clip': n_clip, 'checks': checks, 'blocks': blocks, 'per_fold': per_fold}

    if args.reference:
        ref = json.load(open(args.reference))
        worst = {}
        for f in args.folds:
            for mine, theirs in zip(per_fold[f], ref['per_fold'][str(f)]):
                assert mine['block'] == theirs['block']
                for a_k, b_k in (('raw_spread_v', 'cls_spread'), ('raw_spread_a_all', 'slot_spread')):
                    rel = abs(mine[a_k] - theirs[b_k]) / max(abs(theirs[b_k]), TINY)
                    worst[a_k] = max(worst.get(a_k, 0.0), rel)
        res['reference_check'] = {'file': args.reference, 'max_rel_diff': worst}
        print(f'sanity 1 (stage 0a reproduction): max relative difference {worst}', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + '.json', 'w') as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(args.out + '.npz', blocks=np.array([r['block'] for r in blocks]), **npz)

    print(f'\n{cfg["host"]["name"]} {cfg["data"]["dataset"]} {args.split}, mean over folds {args.folds}')
    cols = ('raw_spread_v', 'cs_spread_v', 'sigma_v', 'cs_spread_a', 'sigma_a', 'beta', 'kappa', 'H',
            'raw_spread_p', 'cs_spread_p', 'beta_p')
    print('block' + ''.join(f'{c[:11]:>12}' for c in cols))
    for r in blocks:
        print(f'{r["block"]:>5}' + ''.join(f'{r[c]:>12.4g}' for c in cols))
    print(f'-> {args.out}.json')


if __name__ == '__main__':
    main()
