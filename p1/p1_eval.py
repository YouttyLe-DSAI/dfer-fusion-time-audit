#!/usr/bin/env python
"""P1: HiCMAE on CREMA-D: reproduction (stage 0) and cross-attention interventions (stage 1).

Reuses HiCMAE's own argument parser, dataset (VideoClsDatasetFrame, test mode: 2 temporal
segments x 2 spatial crops), model (avit_dim512_patch16_160_a256) and checkpoint format.
The encoders run once per batch; the fusion encoder and head then run once per arm, so every
arm sees identical inputs. Per-view logits -> softmax -> mean over the clip's views -> argmax,
as in HiCMAE's merge().

Arms (cross-attention maps A [B, H, T1, T2], softmax over T2, in the fusion encoder):
  base        released model
  indep       A'[i,:] = mean_i A[i,:] in both blocks and directions: row sums (=1) and
              column sums kept, token-specific mixture removed (independent coupling)
  zero        cross-attention residual removed in both blocks and directions
  indep_v2a   indep only where video tokens attend to audio (blocks[*].cross_attn)
  indep_a2v   indep only where audio tokens attend to video (blocks_audio[*].cross_attn)
  indep_b1 / indep_b2   indep in fusion block 1 / 2 only, both directions

Also records, for base, a cross-attention concentration diagnostic per block and direction:
H = mean over heads and query tokens of TV(A[i,:], mean_i A[i,:]).
"""

import argparse
import json
import os
import sys
import time
import types

ARMS = ('base', 'indep', 'zero', 'indep_v2a', 'indep_a2v', 'indep_b1', 'indep_b2')

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--hicmae', required=True, help='HiCMAE repo (imported read-only)')
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--ckpt', required=True)
ap.add_argument('--data', required=True, help='dir holding split0<k>/{train,test}.csv')
ap.add_argument('--arms', nargs='+', default=list(ARMS))
ap.add_argument('--batch-size', type=int, default=16)
ap.add_argument('--workers', type=int, default=10)
ap.add_argument('--limit', type=int, default=0, help='batches, 0 = all (smoke tests only)')
ap.add_argument('--dataset', default='cremad', choices=('cremad', 'mafw'))
ap.add_argument('--out', required=True, help='output prefix; writes <out>.json and <out>.npz')
a = ap.parse_args()
assert a.arms[0] == 'base', 'base must be the first arm'

sys.path.insert(0, os.path.abspath(a.hicmae))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from timm.models import create_model  # noqa: E402

import modeling_finetune_av  # noqa: E402,F401  (registers the models)
import run_class_finetuning_av as R  # noqa: E402
from datasets_av import build_dataset  # noqa: E402
from modeling_finetune_av import GeneralAttention  # noqa: E402

import kinetics_av  # noqa: E402
from p1_common import DATASETS, numeric_frame_order, prepare_dataset, seed_views  # noqa: E402

CFG = DATASETS[a.dataset]
numeric_frame_order(kinetics_av)

DEV = torch.device('cuda')


def hicmae_args():
    argv = ['--model', 'avit_dim512_patch16_160_a256', '--data_set', CFG['data_set'], '--nb_classes', str(CFG['nb_classes']),
            '--data_path', os.path.join(a.data, f'split0{a.fold}'), '--batch_size', str(a.batch_size),
            '--num_sample', '1', '--input_size', '160', '--input_size_audio', '256', '--short_side_size', '160',
            '--depth', '10', '--depth_audio', '10', '--fusion_depth', '2', '--num_frames', '16',
            '--sampling_rate', '4', '--test_num_segment', '2', '--test_num_crop', '2', '--eval',
            '--num_workers', str(a.workers)]
    old, sys.argv = sys.argv, ['run_class_finetuning_av.py'] + argv
    args, _ = R.get_args()
    sys.argv = old
    norm = [-4.2677393, 4.5689974]                       # HiCMAE main(): AudioSet stats
    conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.input_size_audio, 'freqm': 0, 'timem': 0,
            'mean': norm[0], 'std': norm[1], 'noise': False}
    args.audio_conf = {'train': conf, 'validation': conf, 'test': conf}
    return args


# --- cross-attention with an intervention switch (original code path when _p1_mode is None) ---
def ga_forward(self, x, context=None):
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
    if getattr(self, '_p1_record', False):
        af = attn.float()
        tv = 0.5 * (af - af.mean(dim=2, keepdim=True)).abs().sum(-1)        # [B, H, T1]
        self._p1_H = tv.mean(dim=(1, 2)).detach().cpu()                      # [B]
    if mode == 'indep':
        attn = attn.mean(dim=2, keepdim=True).expand_as(attn)
    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, T1, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def cross_modules(model):
    fu = model.encoder_fusion
    mods = {}
    for i, blk in enumerate(fu.blocks):
        mods[(i + 1, 'v2a')] = blk.cross_attn            # video queries, audio context
    for i, blk in enumerate(fu.blocks_audio):
        mods[(i + 1, 'a2v')] = blk.cross_attn            # audio queries, video context
    return mods


def set_arm(mods, arm):
    for (b, d), m in mods.items():
        on = {'base': False, 'zero': True, 'indep': True, 'indep_v2a': d == 'v2a', 'indep_a2v': d == 'a2v',
              'indep_b1': b == 1, 'indep_b2': b == 2}[arm]
        m._p1_mode = None if not on else ('zero' if arm == 'zero' else 'indep')
        m._p1_record = arm == 'base'


def fused_heads(model, videos, audios, arms, mods):
    """HiCMAE forward_features + head, encoders once, fusion once per arm."""
    x = model.encoder(videos)
    xa = model.encoder_audio(audios)
    vf = (x.mean(dim=1) * torch.softmax(model.video_layer_weights, dim=0).view(1, -1, 1)).sum(dim=1)
    af = (xa.mean(dim=1) * torch.softmax(model.audio_layer_weights, dim=0).view(1, -1, 1)).sum(dim=1)
    xl, xal = x[:, :, -1], xa[:, :, -1]
    outs, H = {}, {}
    for arm in arms:
        set_arm(mods, arm)
        f, fa = model.encoder_fusion(xl, xal)
        feat = torch.cat([vf, af, f.mean(dim=1), fa.mean(dim=1)], dim=-1)
        outs[arm] = model.head_activation_func(model.head(model.fc_norm(feat))).float()
        if arm == 'base':
            H = {f'{b}_{d}': m._p1_H for (b, d), m in mods.items()}
    set_arm(mods, 'base')
    return outs, H


def scores(pred, y, n_cls=None):
    n_cls = n_cls or CFG['nb_classes']
    rec = [(pred[y == c] == c).mean() for c in range(n_cls) if (y == c).any()]
    return {'uar': 100.0 * float(np.mean(rec)), 'war': 100.0 * float((pred == y).mean())}


def main():
    args = hicmae_args()
    model = create_model(args.model, pretrained=False, num_classes=args.nb_classes,
                         all_frames=args.num_frames * args.num_segments, tubelet_size=args.tubelet_size,
                         drop_rate=args.drop, drop_path_rate=args.drop_path, attn_drop_rate=args.attn_drop_rate,
                         drop_block_rate=None, init_scale=args.init_scale, attn_type=args.attn_type,
                         depth=args.depth, depth_audio=args.depth_audio, fusion_depth=args.fusion_depth,
                         use_mean_pooling=args.use_mean_pooling)
    ps = model.encoder.patch_embed.patch_size
    args.window_size = (args.num_frames // 2, args.input_size // ps[0], args.input_size // ps[1])
    args.patch_size = ps
    psa = model.encoder_audio.patch_embed.patch_size
    args.window_size_audio = (args.input_size_audio // psa[0], args.num_mel_bins // psa[1])
    args.patch_size_audio = psa
    ck = torch.load(a.ckpt, map_location='cpu')
    missing = model.load_state_dict(ck['model'], strict=True)
    print(f'loaded {a.ckpt} (epoch {ck.get("epoch")}): {missing}', flush=True)
    model.to(DEV).eval()
    for m in model.modules():
        if isinstance(m, GeneralAttention):
            m.forward = types.MethodType(ga_forward, m)
    mods = cross_modules(model)

    ds, _ = build_dataset(is_train=False, test_mode=True, args=args)

    prepare_dataset(ds, CFG)
    seed_views(ds)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                                         pin_memory=True, drop_last=False)
    print(f'fold {a.fold}: {len(ds)} test views', flush=True)

    probs = {arm: {} for arm in a.arms}
    labels, Hs = {}, {}
    checked = False
    t0 = time.time()
    with torch.no_grad():
        for it, batch in enumerate(loader):
            if a.limit and it >= a.limit:
                break
            (videos, audios), target, ids, chunk, split = batch[0], batch[1], batch[2], batch[3], batch[4]
            videos, audios = videos.to(DEV, non_blocking=True), audios.to(DEV, non_blocking=True)
            with torch.cuda.amp.autocast():
                outs, H = fused_heads(model, videos, audios, a.arms, mods)
                if not checked:                         # the decomposition must equal HiCMAE's own forward
                    ref = model(videos, audios).float()
                    diff = (ref - outs['base']).abs().max().item()
                    print(f'check: max |model() - base| = {diff:.2e}', flush=True)
                    assert diff < 1e-2, diff
                    checked = True
            for arm, o in outs.items():
                p = torch.softmax(o.double(), dim=-1).cpu().numpy()
                for i, cid in enumerate(ids):
                    key = (int(chunk[i]), int(split[i]))
                    probs[arm].setdefault(cid, {})[key] = p[i]
            for i, cid in enumerate(ids):
                labels[cid] = int(target[i])
                for k, v in H.items():
                    Hs.setdefault(k, {}).setdefault(cid, []).append(float(v[i]))
            if it % 50 == 0:
                print(f'  batch {it}/{len(loader)}  {time.time() - t0:.0f}s', flush=True)

    clips = sorted(labels)
    y = np.array([labels[c] for c in clips])
    res = {'fold': a.fold, 'ckpt': a.ckpt, 'n_clips': len(clips), 'n_views': {}, 'arms': {}, 'limit': a.limit}
    preds, mean_probs = {}, {}
    for arm in a.arms:
        P = np.stack([np.mean(list(probs[arm][c].values()), axis=0) for c in clips])
        mean_probs[arm] = P
        preds[arm] = P.argmax(1)
        res['n_views'][arm] = int(np.mean([len(probs[arm][c]) for c in clips]) * 1000) / 1000
    for arm in a.arms:
        s = scores(preds[arm], y)
        s['changed_pct'] = 100.0 * float((preds[arm] != preds['base']).mean())
        res['arms'][arm] = s
    res['H_base'] = {k: float(np.mean([np.mean(v[c]) for c in clips])) for k, v in Hs.items()}
    with open(a.out + '.json', 'w') as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(a.out + '.npz', clips=np.array(clips), labels=y,
                        **{f'prob_{arm}': mean_probs[arm] for arm in a.arms})
    print(json.dumps(res['arms'], indent=1))
    print('H (base):', res['H_base'])
    print(f'-> {a.out}.json  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
