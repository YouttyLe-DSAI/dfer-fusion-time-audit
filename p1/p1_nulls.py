#!/usr/bin/env python
"""P1-nulls: a ladder of null interventions on HiCMAE's fusion cross-attention (prereg.md, P1-nulls).

Same model, data, views and checkpoint loading as p1_eval.py. The encoders run once per batch.
Message-level arms change the cross-attention inside the fusion encoder. Input-level arms
(perm_*, emap) pair a clip's video with another clip's audio from the same shuffled batch.

Message at a site (query tokens i): delta_i = proj((A V)_i) = dbar + d_i, dbar = mean_i delta_i.
  base / zero / indep              as in P1 (indep: mean row -> delta'_i = dbar)
  devonly                          delta'_i = d_i
  shrink25/50/75                   delta'_i = dbar + alpha d_i
  rowperm, rowperm_s1, rowperm_s2  delta'_i = delta_sigma(i), sigma random over query tokens (per clip, site)
  rowperm_t                        sigma permutes only the time index of the query tokens
  uniform                          A' = 1 / T2
  papa                             A' = mean attention map over the fold's training clips (per site, head)
Input level (partner = another clip in the batch, never a view of the same clip):
  perm_a / perm_v                  audio / video input of the partner (perceptual score)
  perm_a_fus / perm_v_fus          only the tokens entering the fusion come from the partner
  emap                             additive projection of the logits (Hessel & Lee 2020), sampled partners
"""

import argparse
import json
import os
import sys
import time
import types

MSG_ARMS = ('base', 'zero', 'indep', 'devonly', 'shrink25', 'shrink50', 'shrink75',
            'rowperm', 'rowperm_s1', 'rowperm_s2', 'rowperm_t', 'uniform', 'papa')
PAIR_ARMS = ('perm_a', 'perm_a_fus', 'perm_v', 'perm_v_fus', 'emap')
TRACE_ARMS = ('indep', 'rowperm', 'shrink50')

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--hicmae', required=True)
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--ckpt', required=True)
ap.add_argument('--data', required=True, help='dir holding split0<k>/test.csv')
ap.add_argument('--train-data', required=True, help='dir holding split0<k>/test.csv = the training list')
ap.add_argument('--p1-npz', default='', help='P1 fold npz, for the agreement check')
ap.add_argument('--batch-size', type=int, default=32)
ap.add_argument('--partners', type=int, default=8, help='shifts +-1..+-M for emap')
ap.add_argument('--trace-batches', type=int, default=8)
ap.add_argument('--workers', type=int, default=10)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--limit', type=int, default=0, help='batches, 0 = all (smoke tests only)')
ap.add_argument('--dataset', default='cremad', choices=('cremad', 'mafw'))
ap.add_argument('--out', required=True)
a = ap.parse_args()

sys.path.insert(0, os.path.abspath(a.hicmae))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from timm.models import create_model  # noqa: E402

import modeling_finetune_av  # noqa: E402,F401
import run_class_finetuning_av as R  # noqa: E402
from datasets_av import build_dataset  # noqa: E402
from modeling_finetune_av import GeneralAttention  # noqa: E402

import kinetics_av  # noqa: E402
from p1_common import DATASETS, numeric_frame_order, prepare_dataset, seed_views  # noqa: E402

CFG = DATASETS[a.dataset]
numeric_frame_order(kinetics_av)

DEV = torch.device('cuda')
QUERY_GRID = {'v2a': (8, 100), 'a2v': (16, 8)}      # query tokens = (time, space/freq), time-major


def hicmae_args(data_root):
    argv = ['--model', 'avit_dim512_patch16_160_a256', '--data_set', CFG['data_set'], '--nb_classes', str(CFG['nb_classes']),
            '--data_path', os.path.join(data_root, f'split0{a.fold}'), '--batch_size', str(a.batch_size),
            '--num_sample', '1', '--input_size', '160', '--input_size_audio', '256', '--short_side_size', '160',
            '--depth', '10', '--depth_audio', '10', '--fusion_depth', '2', '--num_frames', '16',
            '--sampling_rate', '4', '--test_num_segment', '2', '--test_num_crop', '2', '--eval',
            '--num_workers', str(a.workers)]
    old, sys.argv = sys.argv, ['run_class_finetuning_av.py'] + argv
    args, _ = R.get_args()
    sys.argv = old
    norm = [-4.2677393, 4.5689974]
    conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.input_size_audio, 'freqm': 0, 'timem': 0,
            'mean': norm[0], 'std': norm[1], 'noise': False}
    args.audio_conf = {'train': conf, 'validation': conf, 'test': conf}
    return args


def ga_forward(self, x, context=None):
    """GeneralAttention.forward with intervention switches; the original path when _p1_mode is None."""
    mode = getattr(self, '_p1_mode', None)
    if mode == 'zero':
        return torch.zeros_like(x)
    B, T1, C = x.shape
    q_bias, kv_bias = self.q_bias, None
    if self.q_bias is not None:
        kv_bias = torch.cat((torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
    q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
    q = q.reshape(B, T1, self.num_heads, -1).transpose(1, 2)
    kv = F.linear(input=x if context is None else context, weight=self.kv.weight, bias=kv_bias)
    _, T2, _ = kv.shape
    kv = kv.reshape(B, T2, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    k, v = kv[0], kv[1]
    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))
    attn = attn.softmax(dim=-1)
    if getattr(self, '_p1_accum', False):                                   # PAPA training pass
        s = attn.detach().double().sum(0)
        self._p1_sum = s if self._p1_sum is None else self._p1_sum + s
        self._p1_n += B
    attn0 = attn
    if mode == 'indep':
        attn = attn.mean(dim=2, keepdim=True).expand_as(attn)
    elif mode == 'uniform':
        attn = torch.full_like(attn, 1.0 / T2)
    elif mode == 'papa':
        attn = self._p1_papa.to(attn.dtype).unsqueeze(0).expand_as(attn)
    out = self.proj((attn @ v).transpose(1, 2).reshape(B, T1, -1))
    if mode == 'affine':
        m = out.mean(dim=1, keepdim=True)
        out = self._p1_beta * m + self._p1_alpha * (out - m)
    elif mode == 'rowperm':
        out = torch.gather(out, 1, self._p1_perm.unsqueeze(-1).expand_as(out))
    if getattr(self, '_p1_f3', False) and mode in ('indep', 'affine', 'rowperm') and \
            getattr(self, '_p1_beta', 1.0) == 1.0:                         # F3: token mean preserved
        ref = self.proj((attn0 @ v).transpose(1, 2).reshape(B, T1, -1)).mean(1)
        rel = (out.mean(1) - ref).norm(dim=-1) / ref.norm(dim=-1)
        self._p1_f3_rel = max(getattr(self, '_p1_f3_rel', 0.0), rel.max().item())
    return self.proj_drop(out)


def cross_modules(model):
    fu = model.encoder_fusion
    mods = {}
    for i, blk in enumerate(fu.blocks):
        mods[(i + 1, 'v2a')] = blk.cross_attn
    for i, blk in enumerate(fu.blocks_audio):
        mods[(i + 1, 'a2v')] = blk.cross_attn
    return mods


def make_perm(B, direction, kind, seed, it, site):
    g = torch.Generator().manual_seed(seed * 10_000_000 + a.fold * 1_000_000 + it * 100 + site * 2 + (kind == 'time'))
    nt, ns = QUERY_GRID[direction]
    if kind == 'all':
        return torch.argsort(torch.rand(B, nt * ns, generator=g), dim=1).to(DEV)
    pt = torch.argsort(torch.rand(B, nt, generator=g), dim=1)               # [B, nt]
    idx = pt.unsqueeze(-1) * ns + torch.arange(ns).view(1, 1, ns)            # token (t, s) <- (pt[t], s)
    return idx.reshape(B, nt * ns).to(DEV)


ARM_SPEC = {
    'base': {}, 'zero': {'mode': 'zero'}, 'indep': {'mode': 'indep'},
    'devonly': {'mode': 'affine', 'beta': 0.0, 'alpha': 1.0},
    'shrink25': {'mode': 'affine', 'beta': 1.0, 'alpha': 0.25},
    'shrink50': {'mode': 'affine', 'beta': 1.0, 'alpha': 0.5},
    'shrink75': {'mode': 'affine', 'beta': 1.0, 'alpha': 0.75},
    'rowperm': {'mode': 'rowperm', 'kind': 'all', 'seed': 0},
    'rowperm_s1': {'mode': 'rowperm', 'kind': 'all', 'seed': 1},
    'rowperm_s2': {'mode': 'rowperm', 'kind': 'all', 'seed': 2},
    'rowperm_t': {'mode': 'rowperm', 'kind': 'time', 'seed': 0},
    'uniform': {'mode': 'uniform'}, 'papa': {'mode': 'papa'},
}


def set_arm(mods, arm, B=0, it=0):
    spec = ARM_SPEC[arm]
    for site, ((b, d), m) in enumerate(sorted(mods.items())):
        m._p1_mode = spec.get('mode')
        m._p1_beta, m._p1_alpha = spec.get('beta', 1.0), spec.get('alpha', 1.0)
        if m._p1_mode == 'rowperm':
            m._p1_perm = make_perm(B, d, spec['kind'], a.seed + spec['seed'], it, site)


def head(model, vf, af, f, fa):
    feat = torch.cat([vf, af, f.mean(dim=1), fa.mean(dim=1)], dim=-1)
    return model.head_activation_func(model.head(model.fc_norm(feat))).float()


def encode(model, videos, audios):
    x = model.encoder(videos)
    xa = model.encoder_audio(audios)
    vf = (x.mean(dim=1) * torch.softmax(model.video_layer_weights, dim=0).view(1, -1, 1)).sum(dim=1)
    af = (xa.mean(dim=1) * torch.softmax(model.audio_layer_weights, dim=0).view(1, -1, 1)).sum(dim=1)
    return vf, af, x[:, :, -1], xa[:, :, -1]


# --- F3 mechanism trace (fp32): the fusion encoder unrolled into sublayers ---
def _sub(blk, x, ctx, rec, tag):
    g = [getattr(blk, n) for n in ('gamma_0', 'gamma_1', 'gamma_2')]
    sc = (lambda i, y: y) if g[0] is None else (lambda i, y: g[i] * y)
    x = x + sc(0, blk.cross_attn(blk.cross_norm1(x), context=blk.cross_norm2(ctx)))
    rec[tag + 'cross'] = x
    x = x + sc(1, blk.attn(blk.norm1(x)))
    rec[tag + 'self'] = x
    x = x + sc(2, blk.mlp(blk.norm2(x)))
    rec[tag + 'mlp'] = x
    return x


def fusion_trace(fu, x, xa):
    rv, ra = {}, {}
    for i, (blk, blka) in enumerate(zip(fu.blocks, fu.blocks_audio)):
        x, xa = _sub(blk, x, xa, rv, f'b{i + 1}_'), _sub(blka, xa, x, ra, f'b{i + 1}_')
    x, xa = fu.norm(x), fu.norm_audio(xa)
    rv['norm'], ra['norm'] = x, xa
    return x, xa, rv, ra


def divergence(r0, r1):
    out = {}
    for k in r0:
        m0, m1 = r0[k].mean(1), r1[k].mean(1)
        rel_mean = ((m1 - m0).norm(dim=-1) / m0.norm(dim=-1)).mean().item()
        rel_tok = ((r1[k] - r0[k]).pow(2).sum(-1).mean(1).sqrt() / r0[k].pow(2).sum(-1).mean(1).sqrt()).mean().item()
        out[k] = (rel_mean, rel_tok)
    return out


def build_model(args):
    model = create_model(args.model, pretrained=False, num_classes=args.nb_classes,
                         all_frames=args.num_frames * args.num_segments, tubelet_size=args.tubelet_size,
                         drop_rate=args.drop, drop_path_rate=args.drop_path, attn_drop_rate=args.attn_drop_rate,
                         drop_block_rate=None, init_scale=args.init_scale, attn_type=args.attn_type,
                         depth=args.depth, depth_audio=args.depth_audio, fusion_depth=args.fusion_depth,
                         use_mean_pooling=args.use_mean_pooling)
    ck = torch.load(a.ckpt, map_location='cpu')
    print(f'loaded {a.ckpt} (epoch {ck.get("epoch")}): {model.load_state_dict(ck["model"], strict=True)}', flush=True)
    model.to(DEV).eval()
    for m in model.modules():
        if isinstance(m, GeneralAttention):
            m.forward = types.MethodType(ga_forward, m)
    return model


def papa_pass(model, mods, args_tr):
    ds, _ = build_dataset(is_train=False, test_mode=True, args=args_tr)
    prepare_dataset(ds, CFG)
    seed_views(ds)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                                         pin_memory=True, drop_last=False)
    print(f'papa: {len(ds)} training views', flush=True)
    set_arm(mods, 'base')
    for m in mods.values():
        m._p1_accum, m._p1_sum, m._p1_n = True, None, 0
    t0 = time.time()
    with torch.no_grad():
        for it, batch in enumerate(loader):
            if a.limit and it >= a.limit:
                break
            (videos, audios) = batch[0]
            with torch.cuda.amp.autocast():
                vf, af, xl, xal = encode(model, videos.to(DEV, non_blocking=True), audios.to(DEV, non_blocking=True))
                model.encoder_fusion(xl, xal)
            if it % 100 == 0:
                print(f'  papa batch {it}/{len(loader)}  {time.time() - t0:.0f}s', flush=True)
    maps = {}
    for (b, d), m in mods.items():
        m._p1_accum = False
        m._p1_papa = (m._p1_sum / m._p1_n).float()
        maps[f'{b}_{d}'] = m._p1_papa.half().cpu().numpy()
        n = m._p1_n
        m._p1_sum = None
    return maps, n


def main():
    torch.manual_seed(a.seed)
    args = hicmae_args(a.data)
    model = build_model(args)
    mods = cross_modules(model)
    papa_maps, papa_n = papa_pass(model, mods, hicmae_args(a.train_data))

    ds, _ = build_dataset(is_train=False, test_mode=True, args=args)

    prepare_dataset(ds, CFG)
    seed_views(ds)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                                         pin_memory=True, drop_last=False,
                                         generator=torch.Generator().manual_seed(a.seed))
    print(f'fold {a.fold}: {len(ds)} test views', flush=True)
    M = a.partners
    shifts = list(range(1, M + 1)) + [-s for s in range(1, M + 1)]
    arms = list(MSG_ARMS) + list(PAIR_ARMS)
    probs = {arm: {} for arm in arms}
    labels = {}
    emap_terms, emap_const, emap_cnt, n_valid = {}, None, 0, []
    trace = {arm: [] for arm in TRACE_ARMS}
    f3_rel, check_diff = {}, None
    t0 = time.time()
    with torch.no_grad():
        for it, batch in enumerate(loader):
            if a.limit and it >= a.limit:
                break
            (videos, audios), target, ids, chunk, split = batch[0], batch[1], batch[2], batch[3], batch[4]
            videos, audios = videos.to(DEV, non_blocking=True), audios.to(DEV, non_blocking=True)
            B = videos.shape[0]
            outs = {}
            with torch.cuda.amp.autocast():
                vf, af, xl, xal = encode(model, videos, audios)
                for arm in MSG_ARMS:
                    set_arm(mods, arm, B, it)
                    f, fa = model.encoder_fusion(xl, xal)
                    outs[arm] = head(model, vf, af, f, fa)
                set_arm(mods, 'base')
                if check_diff is None:
                    check_diff = (model(videos, audios).float() - outs['base']).abs().max().item()
                    print(f'check: max |model() - base| = {check_diff:.2e}', flush=True)
                    assert check_diff < 1e-2, check_diff
                # input-level partners within the batch
                code = {c: j for j, c in enumerate(dict.fromkeys(ids))}
                cid = torch.tensor([code[c] for c in ids])
                ar = torch.arange(B)
                ap_full, ap_fus, vp_full, vp_fus, ok_a, ok_v = [], [], [], [], [], []
                for s in shifts:
                    p = (ar + s) % B                     # audio partner of view b
                    q = (ar - s) % B                     # video partner of view b
                    f, fa = model.encoder_fusion(xl, xal[p.to(DEV)])     # pair (v_j, a_{j+s}) at row j
                    qd = q.to(DEV)
                    ap_full.append(head(model, vf, af[p.to(DEV)], f, fa))
                    ap_fus.append(head(model, vf, af, f, fa))
                    vp_full.append(head(model, vf[qd], af, f[qd], fa[qd]))
                    vp_fus.append(head(model, vf, af, f[qd], fa[qd]))
                    ok_a.append(cid[p] != cid)
                    ok_v.append(cid[q] != cid)
            ok_a, ok_v = torch.stack(ok_a).to(DEV), torch.stack(ok_v).to(DEV)              # [S, B]
            assert ok_a.any(0).all() and ok_v.any(0).all(), 'a view without a valid partner'
            first_a, first_v = ok_a.float().argmax(0), ok_v.float().argmax(0)             # first valid shift
            bi = torch.arange(B, device=DEV)
            for name, stack, first in (('perm_a', ap_full, first_a), ('perm_a_fus', ap_fus, first_a),
                                       ('perm_v', vp_full, first_v), ('perm_v_fus', vp_fus, first_v)):
                outs[name] = torch.stack(stack)[first, bi]
            A, V = torch.stack(ap_full), torch.stack(vp_full)                               # [S, B, C]
            wa, wv = ok_a.float().unsqueeze(-1), ok_v.float().unsqueeze(-1)
            term = (A * wa).sum(0) / wa.sum(0) + (V * wv).sum(0) / wv.sum(0)
            s_all = (A * wa).sum((0, 1)).double()
            emap_const = s_all if emap_const is None else emap_const + s_all
            emap_cnt += int(ok_a.sum())
            n_valid.append(ok_a.sum(0).float().mean().item())
            for i, c in enumerate(ids):
                emap_terms.setdefault(c, {})[(int(chunk[i]), int(split[i]))] = term[i].double().cpu()

            if it < a.trace_batches:                     # F3 mechanism trace, true fp32 (TF32 off)
                tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
                with torch.cuda.amp.autocast(enabled=False):
                    x32, xa32 = xl.float(), xal.float()
                    set_arm(mods, 'base')
                    ref = model.encoder_fusion(x32, xa32)
                    f0, fa0, rv0, ra0 = fusion_trace(model.encoder_fusion, x32, xa32)
                    tdiff = max((ref[0] - f0).abs().max().item(), (ref[1] - fa0).abs().max().item())
                    assert tdiff < 1e-4, tdiff
                    for arm in TRACE_ARMS:
                        set_arm(mods, arm, B, it)
                        for m in mods.values():
                            m._p1_f3 = True
                        f1, fa1, rv1, ra1 = fusion_trace(model.encoder_fusion, x32, xa32)
                        for (b, d), m in mods.items():
                            f3_rel.setdefault(arm, {})[f'{b}_{d}'] = max(f3_rel.get(arm, {}).get(f'{b}_{d}', 0.0),
                                                                         m._p1_f3_rel)
                            m._p1_f3, m._p1_f3_rel = False, 0.0
                        dv, da = divergence(rv0, rv1), divergence(ra0, ra1)
                        trace[arm].append({'video': dv, 'audio': da})
                    set_arm(mods, 'base')
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32

            for arm in arms:
                if arm == 'emap':
                    continue
                p = torch.softmax(outs[arm].double(), dim=-1).cpu().numpy()
                for i, c in enumerate(ids):
                    probs[arm].setdefault(c, {})[(int(chunk[i]), int(split[i]))] = p[i]
            for i, c in enumerate(ids):
                labels[c] = int(target[i])
            if it % 25 == 0:
                print(f'  batch {it}/{len(loader)}  {time.time() - t0:.0f}s', flush=True)

    const = (emap_const / emap_cnt).cpu()
    for c, views in emap_terms.items():
        for k, t in views.items():
            probs['emap'].setdefault(c, {})[k] = torch.softmax(t - const, dim=-1).numpy()

    clips = sorted(labels)
    y = np.array([labels[c] for c in clips])
    P = {arm: np.stack([np.mean(list(probs[arm][c].values()), axis=0) for c in clips]) for arm in arms}
    pred = {arm: P[arm].argmax(1) for arm in arms}
    res = {'fold': a.fold, 'ckpt': a.ckpt, 'n_clips': len(clips), 'limit': a.limit, 'batch_size': a.batch_size,
           'seed': a.seed, 'partners_per_side': 2 * M, 'check_max_diff': check_diff,
           'emap_mean_valid_partners': float(np.mean(n_valid)), 'papa_train_views': papa_n,
           'n_views': float(np.mean([len(probs['base'][c]) for c in clips])), 'arms': {}}
    for arm in arms:
        rec = [(pred[arm][y == c] == c).mean() for c in range(CFG['nb_classes']) if (y == c).any()]
        res['arms'][arm] = {'uar': 100.0 * float(np.mean(rec)), 'war': 100.0 * float((pred[arm] == y).mean()),
                            'changed_pct': 100.0 * float((pred[arm] != pred['base']).mean())}
    if a.p1_npz and os.path.exists(a.p1_npz) and not a.limit:
        z = np.load(a.p1_npz)
        assert list(z['clips']) == clips
        res['agree_with_p1'] = {arm: 100.0 * float((z[f'prob_{arm}'].argmax(1) == pred[arm]).mean())
                                for arm in ('base', 'indep', 'zero')}
    res['f3_max_rel_mean_diff'] = f3_rel
    res['trace'] = {arm: {s: {k: [float(np.mean([t[s][k][j] for t in trace[arm]])) for j in (0, 1)]
                              for k in trace[arm][0][s]} for s in ('video', 'audio')}
                    for arm in TRACE_ARMS if trace[arm]}
    with open(a.out + '.json', 'w') as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(a.out + '.npz', clips=np.array(clips), labels=y, **{f'prob_{arm}': P[arm] for arm in arms})
    np.savez_compressed(a.out + '_papa.npz', **papa_maps)
    print(json.dumps(res['arms'], indent=1))
    print('agree with P1:', res.get('agree_with_p1'))
    print('F3 max rel diff:', f3_rel)
    print(f'-> {a.out}.json  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
