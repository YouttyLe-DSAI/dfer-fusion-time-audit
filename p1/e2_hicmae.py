#!/usr/bin/env python
"""E2 (prereg.md): cost-space spread and interaction at HiCMAE's four fusion cross-attention maps.

Same model, data, views, seeding and precision as p1_eval.py. The four sites are the maps indep replaces:
  (b, 'v2a') = encoder_fusion.blocks[b-1].cross_attn        queries = 800 video tokens, keys = 128 audio tokens
  (b, 'a2v') = encoder_fusion.blocks_audio[b-1].cross_attn  queries = 128 audio tokens, keys = 800 video tokens
Video tokens are time-major (8 temporal x 100 spatial); audio tokens 16 time x 8 frequency.

The module output is untouched (the original forward runs; statistics are computed beside it, in fp32 with TF32 off,
from the inputs the module received). Per view and head:
  q = W_q,h x~ + b_q,h, k = W_k,h c~ (x~ = cross_norm1(x), c~ = cross_norm2(context)); logits L = q.k / sqrt(d_h)
  spread / sigma        of q over its tokens, of k over its tokens; also of x~, c~ (inputs) and of the pre-norm
                        stream tokens (raw); for the video side also C1 = pooled over space to 8 temporal positions
  kappa                 RMS of the double-centred logits (cost -q.k, eps = sqrt(d_h): D5, B = I)
  beta                  sigma_q sigma_k / sqrt(d_h)                      (>= kappa, Cauchy-Schwarz)
  beta_a                ||B_eff,h||_2 sigma_x~ sigma_c~, B_eff,h = W_q,h^T W_k,h / sqrt(d_h), eps = 1   (>= kappa)
  H_attn                mean_j TV(A_j, mean_j A_j)                       (distance from the row indep substitutes)
Checks: B_eff logits = projected logits (relative); module output recomputed from our logits = module output
(relative, fp16 path); kappa <= beta and kappa <= beta_a for every view and head; base UAR = P2 eval base UAR.

Writes only --out(.json, .npz).
"""

import argparse
import json
import math
import os
import sys
import time
import types

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--hicmae', required=True)
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--ckpt', required=True)
ap.add_argument('--data', required=True, help='dir holding split0<k>/test.csv')
ap.add_argument('--batch-size', type=int, default=16)
ap.add_argument('--workers', type=int, default=10)
ap.add_argument('--limit', type=int, default=0, help='batches, 0 = all (smoke tests only)')
ap.add_argument('--dataset', default='cremad', choices=('cremad', 'mafw'))
ap.add_argument('--eval-json', default='', help='P2 eval fold JSON, for sanity check 5')
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

import kinetics_av  # noqa: E402
from p1_common import DATASETS, numeric_frame_order, prepare_dataset, seed_views  # noqa: E402

CFG = DATASETS[a.dataset]
numeric_frame_order(kinetics_av)
DEV = torch.device('cuda')
TINY = 1e-12
VID = (8, 100)                      # video tokens: (time, space), time-major
SITE_KEYS = ('q_spread', 'q_sigma', 'k_spread', 'k_sigma', 'xq_spread', 'xq_sigma', 'xk_spread', 'xk_sigma',
             'vis_c1_spread', 'vis_c1_sigma', 'vis_c1_in_spread', 'kappa', 'beta', 'beta_a', 'beta_c1', 'H_attn')
RAW_KEYS = ('raw_v_spread', 'raw_v_sigma', 'raw_v_c1_spread', 'raw_a_spread', 'raw_a_sigma')


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


def build_model(args):
    model = create_model(args.model, pretrained=False, num_classes=args.nb_classes,
                         all_frames=args.num_frames * args.num_segments, tubelet_size=args.tubelet_size,
                         drop_rate=args.drop, drop_path_rate=args.drop_path, attn_drop_rate=args.attn_drop_rate,
                         drop_block_rate=None, init_scale=args.init_scale, attn_type=args.attn_type,
                         depth=args.depth, depth_audio=args.depth_audio, fusion_depth=args.fusion_depth,
                         use_mean_pooling=args.use_mean_pooling)
    ck = torch.load(a.ckpt, map_location='cpu')
    print(f'loaded {a.ckpt} (epoch {ck.get("epoch")}): {model.load_state_dict(ck["model"], strict=True)}', flush=True)
    return model.to(DEV).eval()


def dev_stats(x):
    """x [..., K, D] -> spread [...], sigma [...] over the K axis."""
    mu = x.mean(-2, keepdim=True)
    d = (x - mu).norm(dim=-1)
    return d.mean(-1) / mu.squeeze(-2).norm(dim=-1).clamp_min(TINY), d.pow(2).mean(-1).sqrt()


class Site:
    def __init__(self, name, mod):
        self.name, self.mod = name, mod
        self.direction = name.split('_')[1]
        self.vals = {k: [] for k in SITE_KEYS}
        self.chk = {'beff_rel': 0.0, 'out_rel': 0.0, 'viol_beta': 0, 'viol_beta_a': 0, 'n': 0}
        H = mod.num_heads
        hd = mod.q.weight.shape[0]
        self.H, self.dh = H, hd // H
        assert abs(mod.scale - self.dh ** -0.5) < 1e-12, (mod.scale, self.dh)
        with torch.no_grad():
            Wq = mod.q.weight.float().view(H, self.dh, -1)                       # [H, dh, C]
            Wk = mod.kv.weight.float()[:hd].view(H, self.dh, -1)
            self.Beff = torch.einsum('hdc,hde->hce', Wq, Wk) / math.sqrt(self.dh)   # [H, C, Ck]
            bq = mod.q_bias.float().view(H, self.dh) if mod.q_bias is not None else torch.zeros(H, self.dh, device=DEV)
            self.bcol = torch.einsum('hd,hde->he', bq, Wk) / math.sqrt(self.dh)    # [H, Ck], key-only term
            # the D5 check runs in fp64 (2026-11-11 incident: in fp32 the two conventions differed by 1.2e-4 relative,
            # accumulation rounding over two 512-d contractions)
            self.Beff64 = torch.einsum('hdc,hde->hce', Wq.double(), Wk.double()) / math.sqrt(self.dh)
            self.bcol64 = torch.einsum('hd,hde->he', bq.double(), Wk.double()) / math.sqrt(self.dh)
            self.Beff_norm = torch.stack([torch.linalg.matrix_norm(self.Beff[h].double(), ord=2) for h in range(H)]).float()

    def stats(self, xq, xk, out):
        m, H, dh = self.mod, self.H, self.dh
        B, T1, _ = xq.shape
        T2 = xk.shape[1]
        hd = H * dh
        q = F.linear(xq, m.q.weight.float(), None if m.q_bias is None else m.q_bias.float())
        q = q.view(B, T1, H, dh).transpose(1, 2)                                  # [B, H, T1, dh]
        kv_w = m.kv.weight.float()
        k = F.linear(xk, kv_w[:hd]).view(B, T2, H, dh).transpose(1, 2)
        L = q @ k.transpose(-2, -1) / math.sqrt(dh)                               # [B, H, T1, T2]

        # D5 check (fp64): the B_eff convention on the LayerNorm-ed inputs gives the same logits as projected Q/K
        xq64, xk64 = xq.double(), xk.double()
        q64 = F.linear(xq64, m.q.weight.double(), None if m.q_bias is None else m.q_bias.double())
        q64 = q64.view(B, T1, H, dh).transpose(1, 2)
        k64 = F.linear(xk64, kv_w[:hd].double()).view(B, T2, H, dh).transpose(1, 2)
        L64 = q64 @ k64.transpose(-2, -1) / math.sqrt(dh)
        Lb = torch.einsum('btc,hce,bse->bhts', xq64, self.Beff64, xk64) + \
            torch.einsum('he,bse->bhs', self.bcol64, xk64).unsqueeze(2)
        self.chk['beff_rel'] = max(self.chk['beff_rel'], ((Lb - L64).abs().max() / L64.abs().max()).item())
        self.chk['fp32_rel'] = max(self.chk.get('fp32_rel', 0.0), ((L.double() - L64).abs().max() / L64.abs().max()).item())
        del xq64, xk64, q64, k64, L64, Lb
        # the module output recomputed from our logits
        vb = None if m.v_bias is None else m.v_bias.float()
        vv = F.linear(xk, kv_w[hd:], vb).view(B, T2, H, dh).transpose(1, 2)
        A = L.softmax(-1)
        o = m.proj((A @ vv).transpose(1, 2).reshape(B, T1, hd).float())
        self.chk['out_rel'] = max(self.chk['out_rel'], ((o - out.float()).abs().max() / out.float().abs().max()).item())

        q_sp, q_sg = dev_stats(q)                                                 # [B, H]
        k_sp, k_sg = dev_stats(k)
        xq_sp, xq_sg = dev_stats(xq)                                              # [B]
        xk_sp, xk_sg = dev_stats(xk)
        Lc = L - L.mean(-1, keepdim=True) - L.mean(-2, keepdim=True) + L.mean((-2, -1), keepdim=True)
        kappa = Lc.pow(2).mean((-2, -1)).sqrt()
        beta = q_sg * k_sg / math.sqrt(dh)
        beta_a = self.Beff_norm.view(1, H) * (xq_sg * xk_sg).view(B, 1)
        if self.direction == 'v2a':                                               # video = queries
            vis, vis_in, other_sg = q, xq, k_sg
        else:                                                                     # video = keys
            vis, vis_in, other_sg = k, xk, q_sg
        c1 = vis.reshape(B, H, VID[0], VID[1], dh).mean(3)
        c1_sp, c1_sg = dev_stats(c1)
        c1_in_sp, _ = dev_stats(vis_in.reshape(B, VID[0], VID[1], -1).mean(2))
        Hattn = 0.5 * (A - A.mean(-2, keepdim=True)).abs().sum(-1).mean(-1)
        tol = 1e-5
        self.chk['viol_beta'] += int((kappa > beta * (1 + tol) + 1e-7).sum())
        self.chk['viol_beta_a'] += int((kappa > beta_a * (1 + tol) + 1e-7).sum())
        self.chk['n'] += kappa.numel()
        hb = lambda x: x.view(B, 1).expand(B, H)                                  # noqa: E731
        rec = {'q_spread': q_sp, 'q_sigma': q_sg, 'k_spread': k_sp, 'k_sigma': k_sg,
               'xq_spread': hb(xq_sp), 'xq_sigma': hb(xq_sg), 'xk_spread': hb(xk_sp), 'xk_sigma': hb(xk_sg),
               'vis_c1_spread': c1_sp, 'vis_c1_sigma': c1_sg, 'vis_c1_in_spread': hb(c1_in_sp),
               'kappa': kappa, 'beta': beta, 'beta_a': beta_a, 'beta_c1': c1_sg * other_sg / math.sqrt(dh),
               'H_attn': Hattn}
        for kk, v in rec.items():
            self.vals[kk].append(v.detach().float().cpu())


def instrument(model):
    fu = model.encoder_fusion
    sites, raw = {}, {}
    for i, (blk, blka) in enumerate(zip(fu.blocks, fu.blocks_audio)):
        for name, b in ((f'b{i + 1}_v2a', blk), (f'b{i + 1}_a2v', blka)):
            s = Site(name, b.cross_attn)
            sites[name] = s
            orig = b.cross_attn.forward

            def fwd(self, x, context=None, _s=s, _orig=orig):
                out = _orig(x, context)
                tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                    _s.stats(x.float(), context.float(), out)
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
                return out
            b.cross_attn.forward = types.MethodType(fwd, b.cross_attn)
        raw[i + 1] = {k: [] for k in RAW_KEYS}

        def pre_v(mod, inp, _r=raw[i + 1]):
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                x = inp[0].float()
                sp, sg = dev_stats(x)
                c1, _ = dev_stats(x.reshape(x.shape[0], VID[0], VID[1], -1).mean(2))
                _r['raw_v_spread'].append(sp.cpu()); _r['raw_v_sigma'].append(sg.cpu()); _r['raw_v_c1_spread'].append(c1.cpu())

        def pre_a(mod, inp, _r=raw[i + 1]):
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                sp, sg = dev_stats(inp[0].float())
                _r['raw_a_spread'].append(sp.cpu()); _r['raw_a_sigma'].append(sg.cpu())
        blk.register_forward_pre_hook(pre_v)
        blka.register_forward_pre_hook(pre_a)
    return sites, raw


def per_clip(vals, inv, n_clip):
    """vals [n_view, ...] -> [n_clip, ...], mean over each clip's views."""
    out = np.zeros((n_clip,) + vals.shape[1:], dtype=np.float64)
    cnt = np.zeros(n_clip)
    np.add.at(out, inv, vals)
    np.add.at(cnt, inv, 1)
    return out / cnt.reshape((-1,) + (1,) * (vals.ndim - 1))


def main():
    torch.manual_seed(0)
    args = hicmae_args(a.data)
    model = build_model(args)
    sites, raw = instrument(model)
    ds, _ = build_dataset(is_train=False, test_mode=True, args=args)
    prepare_dataset(ds, CFG)
    seed_views(ds)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                                         pin_memory=True, drop_last=False)
    print(f'fold {a.fold}: {len(ds)} test views', flush=True)
    ids_all, probs, labels = [], {}, {}
    t0 = time.time()
    with torch.no_grad():
        for it, batch in enumerate(loader):
            if a.limit and it >= a.limit:
                break
            (videos, audios), target, ids, chunk, split = batch[0], batch[1], batch[2], batch[3], batch[4]
            with torch.cuda.amp.autocast():                     # p1_eval.fused_heads, base arm
                x = model.encoder(videos.to(DEV, non_blocking=True))
                xa = model.encoder_audio(audios.to(DEV, non_blocking=True))
                vf = (x.mean(dim=1) * torch.softmax(model.video_layer_weights, dim=0).view(1, -1, 1)).sum(dim=1)
                af = (xa.mean(dim=1) * torch.softmax(model.audio_layer_weights, dim=0).view(1, -1, 1)).sum(dim=1)
                f, fa = model.encoder_fusion(x[:, :, -1], xa[:, :, -1])
                feat = torch.cat([vf, af, f.mean(dim=1), fa.mean(dim=1)], dim=-1)
                o = model.head_activation_func(model.head(model.fc_norm(feat))).float()
            p = torch.softmax(o.double(), dim=-1).cpu().numpy()
            for i, c in enumerate(ids):
                probs.setdefault(c, {})[(int(chunk[i]), int(split[i]))] = p[i]
                labels[c] = int(target[i])
            ids_all.extend(ids)
            if it % 50 == 0:
                print(f'  batch {it}/{len(loader)}  {time.time() - t0:.0f}s', flush=True)

    clips = sorted(labels)
    y = np.array([labels[c] for c in clips])
    P = np.stack([np.mean(list(probs[c].values()), axis=0) for c in clips])
    pred = P.argmax(1)
    rec = [(pred[y == c] == c).mean() for c in range(CFG['nb_classes']) if (y == c).any()]
    uar = 100.0 * float(np.mean(rec))
    idx = {c: j for j, c in enumerate(clips)}
    inv = np.array([idx[c] for c in ids_all])
    n_clip = len(clips)

    res = {'fold': a.fold, 'ckpt': a.ckpt, 'n_clips': n_clip, 'n_views': len(ids_all), 'limit': a.limit,
           'base_uar': uar, 'sites': {}, 'raw': {}, 'checks': {}}
    if a.eval_json and os.path.exists(a.eval_json) and not a.limit:
        ref = json.load(open(a.eval_json))['arms']['base']['uar']
        res['checks']['base_uar_ref'] = ref
        res['checks']['base_uar_diff'] = uar - ref
    npz = {'clips': np.array(clips)}
    for name, s in sites.items():
        row = {'heads': s.H, 'd_h': s.dh, 'Beff_spec_norm': s.Beff_norm.cpu().tolist()}
        for k in SITE_KEYS:
            v = per_clip(torch.cat(s.vals[k]).numpy().astype(np.float64), inv, n_clip)     # [n_clip, H]
            hm = v.mean(1)
            row[k] = float(hm.mean())                     # head mean, clip mean
            row[k + '_headmax'] = float(v.mean(0).max())  # max over heads of the clip mean
            row[k + '_perhead'] = v.mean(0).tolist()
            row[k + '_med'] = float(np.median(hm))
            row[k + '_p90'] = float(np.quantile(hm, 0.9))
            if k in ('kappa', 'beta', 'beta_a', 'q_spread', 'k_spread', 'q_sigma', 'k_sigma', 'vis_c1_spread', 'H_attn'):
                npz[f'{name}_{k}'] = v.astype(np.float32)
        res['sites'][name] = row
        res['checks'][name] = s.chk
    for b, r in raw.items():
        res['raw'][f'b{b}'] = {k: float(per_clip(torch.cat(r[k]).numpy().astype(np.float64), inv, n_clip).mean())
                               for k in RAW_KEYS}
    with open(a.out + '.json', 'w') as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(a.out + '.npz', **npz)
    print(f'base UAR {uar:.4f}  checks {json.dumps(res["checks"])}')
    for name, r in res['sites'].items():
        print(f'{name}: kappa {r["kappa"]:.4g} beta {r["beta"]:.4g} beta_a {r["beta_a"]:.4g} H {r["H_attn"]:.4g} '
              f'q_spread {r["q_spread"]:.4g} k_spread {r["k_spread"]:.4g} c1 {r["vis_c1_spread"]:.4g}')
    print(f'-> {a.out}.json  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
