#!/usr/bin/env python
"""P1-followup (prereg.md): space vs time permutation of cross-attention messages, and audio-visual desync.

Same model, data, views and checkpoints as p1_eval.py / p1_nulls.py. Batch 32, unshuffled.
The dataset RNG is seeded from the view index before every audio load, so the author's test-time
gain augmentation (roll_mag_aug, on at test) gives each view the same gain in every condition and run.

Message arms (aligned audio): base, rowperm, rowperm_t, rowperm_s (seed 0).
Audio conditions: delay_d (audio window shifted by d ms, d > 0 = sound late, zero-padded outside the
file), each also as *_fus (delayed tokens enter only the fusion), and silence (zero waveform).
"""

import argparse
import json
import os
import sys
import time
import types

DELAYS = (-640, -320, -160, -80, -40, 40, 80, 160, 320, 640)
CONDS = ['delay0'] + [f'delay{d:+d}' for d in DELAYS] + ['silence']
MSG_ARMS = ('base', 'rowperm', 'rowperm_t', 'rowperm_s')
QUERY_GRID = {'v2a': (8, 100), 'a2v': (16, 8)}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--hicmae', required=True)
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--ckpt', required=True)
ap.add_argument('--data', required=True)
ap.add_argument('--batch-size', type=int, default=32)
ap.add_argument('--workers', type=int, default=10)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--dataset', default='cremad', choices=('cremad', 'mafw'))
ap.add_argument('--out', required=True)
a = ap.parse_args()

sys.path.insert(0, os.path.abspath(a.hicmae))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchaudio  # noqa: E402
from timm.models import create_model  # noqa: E402

import modeling_finetune_av  # noqa: E402,F401
import run_class_finetuning_av as R  # noqa: E402
from datasets_av import build_dataset  # noqa: E402
from modeling_finetune_av import GeneralAttention  # noqa: E402

import kinetics_av  # noqa: E402
from p1_common import DATASETS, numeric_frame_order, prepare_dataset  # noqa: E402

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
    norm = [-4.2677393, 4.5689974]
    conf = {'num_mel_bins': args.num_mel_bins, 'target_length': args.input_size_audio, 'freqm': 0, 'timem': 0,
            'mean': norm[0], 'std': norm[1], 'noise': False}
    args.audio_conf = {'train': conf, 'validation': conf, 'test': conf}
    return args


# --- dataset: one fbank per audio condition, gain seeded per view ---
def _shift(seg_src, s, e, d):
    """Samples [s - d, e - d) of seg_src (1, n), zero-padded outside [0, n)."""
    n = seg_src.shape[1]
    out = np.zeros((1, e - s), dtype=np.float32)
    lo, hi = max(s - d, 0), min(e - d, n)
    if hi > lo:
        out[:, lo - (s - d):hi - (s - d)] = seg_src[:, lo:hi]
    return out


def load_audio_conds(self, audio_name, video_reader, start_frame_idx, end_frame_idx, min_audio_length=1024):
    """HiCMAE's VideoClsDatasetFrame.load_audio, once per condition (delay0 reproduces the original)."""
    audio_start = video_reader.get_frame_timestamp_relative(start_frame_idx)
    audio_end = video_reader.get_frame_timestamp_relative(end_frame_idx)
    audio, sr = torchaudio.load(audio_name)
    assert sr == self.audio_sample_rate
    assert audio.shape[1] > min_audio_length
    n = audio.shape[1]
    s, e = int(audio_start * n), int(audio_end * n)
    whole = (e - s) <= min_audio_length
    if whole:
        assert n < self.audio_conf.get('target_length') / 100.0 * self.audio_sample_rate
        s, e = 0, n
    src = audio.numpy()
    out = []
    for cond in self._p1_conds:
        np.random.seed(self._p1_seed + self._p1_index)          # same gain in every condition
        if cond == 'delay0':
            seg = audio if whole else src[:, s:e]
        elif cond == 'silence':
            seg = np.zeros((1, e - s), dtype=np.float32)
        else:
            seg = _shift(src, s, e, int(round(int(cond[5:]) * sr / 1000.0)))
        fbank, _ = self._wav2fbank(seg, sr=self.audio_sample_rate)
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)   # freqm = timem = 0, noise False at test
        out.append(fbank.unsqueeze(0))
    return torch.stack(out)                                      # [n_cond, 1, T, F]


def patch_dataset(ds):
    cls = type(ds)
    orig = cls.__getitem__

    def getitem(self, index):
        self._p1_index = index
        return orig(self, index)
    cls.__getitem__ = getitem
    cls.load_audio = load_audio_conds
    ds._p1_conds, ds._p1_seed = CONDS, a.seed * 1_000_000
    assert not ds.noise and ds.freqm == 0 and ds.timem == 0


# --- cross-attention with message permutation ---
def ga_forward(self, x, context=None):
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
    attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    out = self.proj((self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, T1, -1))
    if getattr(self, '_p1_perm', None) is not None:
        out = torch.gather(out, 1, self._p1_perm.unsqueeze(-1).expand_as(out))
    return self.proj_drop(out)


def make_perm(B, direction, kind, it, site):
    g = torch.Generator().manual_seed(a.seed * 10_000_000 + a.fold * 1_000_000 + it * 100 + site * 3
                                      + {'all': 0, 'time': 1, 'space': 2}[kind])
    nt, ns = QUERY_GRID[direction]
    if kind == 'all':
        return torch.argsort(torch.rand(B, nt * ns, generator=g), dim=1).to(DEV)
    if kind == 'time':
        pt = torch.argsort(torch.rand(B, nt, generator=g), dim=1)
        idx = pt.unsqueeze(-1) * ns + torch.arange(ns).view(1, 1, ns)          # (t, s) <- (pt[t], s)
    else:
        ps = torch.argsort(torch.rand(B, ns, generator=g), dim=1)
        idx = torch.arange(nt).view(1, nt, 1) * ns + ps.unsqueeze(1)             # (t, s) <- (t, ps[s])
    return idx.reshape(B, nt * ns).to(DEV)


def set_arm(mods, arm, B=0, it=0):
    kind = {'rowperm': 'all', 'rowperm_t': 'time', 'rowperm_s': 'space'}.get(arm)
    for site, ((b, d), m) in enumerate(sorted(mods.items())):
        m._p1_perm = None if kind is None else make_perm(B, d, kind, it, site)


def cross_modules(model):
    fu = model.encoder_fusion
    mods = {(i + 1, 'v2a'): blk.cross_attn for i, blk in enumerate(fu.blocks)}
    mods.update({(i + 1, 'a2v'): blk.cross_attn for i, blk in enumerate(fu.blocks_audio)})
    return mods


def head(model, vf, af, f, fa):
    feat = torch.cat([vf, af, f.mean(dim=1), fa.mean(dim=1)], dim=-1)
    return model.head_activation_func(model.head(model.fc_norm(feat))).float()


def enc_video(model, videos):
    x = model.encoder(videos)
    return (x.mean(dim=1) * torch.softmax(model.video_layer_weights, dim=0).view(1, -1, 1)).sum(dim=1), x[:, :, -1]


def enc_audio(model, audios):
    xa = model.encoder_audio(audios)
    return (xa.mean(dim=1) * torch.softmax(model.audio_layer_weights, dim=0).view(1, -1, 1)).sum(dim=1), xa[:, :, -1]


def main():
    torch.manual_seed(a.seed)
    args = hicmae_args()
    model = create_model(args.model, pretrained=False, num_classes=args.nb_classes,
                         all_frames=args.num_frames * args.num_segments, tubelet_size=args.tubelet_size,
                         drop_rate=args.drop, drop_path_rate=args.drop_path, attn_drop_rate=args.attn_drop_rate,
                         drop_block_rate=None, init_scale=args.init_scale, attn_type=args.attn_type,
                         depth=args.depth, depth_audio=args.depth_audio, fusion_depth=args.fusion_depth,
                         use_mean_pooling=args.use_mean_pooling)
    ck = torch.load(a.ckpt, map_location='cpu')
    print(f'loaded {a.ckpt}: {model.load_state_dict(ck["model"], strict=True)}', flush=True)
    model.to(DEV).eval()
    for m in model.modules():
        if isinstance(m, GeneralAttention):
            m.forward = types.MethodType(ga_forward, m)
    mods = cross_modules(model)

    ds, _ = build_dataset(is_train=False, test_mode=True, args=args)

    prepare_dataset(ds, CFG)
    patch_dataset(ds)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                                         pin_memory=True, drop_last=False)
    print(f'fold {a.fold}: {len(ds)} test views, {len(CONDS)} audio conditions', flush=True)
    arms = list(MSG_ARMS) + [c for c in CONDS[1:]] + [c + '_fus' for c in CONDS[1:]]
    probs = {arm: {} for arm in arms}
    labels, check = {}, None
    t0 = time.time()
    with torch.no_grad():
        for it, batch in enumerate(loader):
            if a.limit and it >= a.limit:
                break
            (videos, audios), target, ids, chunk, split = batch[0], batch[1], batch[2], batch[3], batch[4]
            videos, audios = videos.to(DEV, non_blocking=True), audios.to(DEV, non_blocking=True)  # [B, K, 1, T, F]
            B = videos.shape[0]
            outs = {}
            with torch.cuda.amp.autocast():
                vf, xl = enc_video(model, videos)
                af0, xal0 = enc_audio(model, audios[:, 0])
                for arm in MSG_ARMS:
                    set_arm(mods, arm, B, it)
                    outs[arm] = head(model, vf, af0, *model.encoder_fusion(xl, xal0))
                set_arm(mods, 'base')
                if check is None:
                    check = (model(videos, audios[:, 0]).float() - outs['base']).abs().max().item()
                    print(f'check: max |model() - base| = {check:.2e}', flush=True)
                    assert check < 1e-2, check
                for k, cond in enumerate(CONDS[1:], start=1):
                    afk, xalk = enc_audio(model, audios[:, k])
                    f, fa = model.encoder_fusion(xl, xalk)
                    outs[cond] = head(model, vf, afk, f, fa)
                    outs[cond + '_fus'] = head(model, vf, af0, f, fa)
            for arm in arms:
                p = torch.softmax(outs[arm].double(), dim=-1).cpu().numpy()
                for i, c in enumerate(ids):
                    probs[arm].setdefault(c, {})[(int(chunk[i]), int(split[i]))] = p[i]
            for i, c in enumerate(ids):
                labels[c] = int(target[i])
            if it % 25 == 0:
                print(f'  batch {it}/{len(loader)}  {time.time() - t0:.0f}s', flush=True)

    clips = sorted(labels)
    y = np.array([labels[c] for c in clips])
    P = {arm: np.stack([np.mean(list(probs[arm][c].values()), axis=0) for c in clips]) for arm in arms}
    pred = {arm: P[arm].argmax(1) for arm in arms}
    res = {'fold': a.fold, 'ckpt': a.ckpt, 'n_clips': len(clips), 'limit': a.limit, 'seed': a.seed,
           'check_max_diff': check, 'delays_ms': list(DELAYS), 'arms': {}}
    for arm in arms:
        rec = [(pred[arm][y == c] == c).mean() for c in range(CFG['nb_classes']) if (y == c).any()]
        res['arms'][arm] = {'uar': 100.0 * float(np.mean(rec)), 'war': 100.0 * float((pred[arm] == y).mean()),
                            'changed_pct': 100.0 * float((pred[arm] != pred['base']).mean())}
    with open(a.out + '.json', 'w') as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(a.out + '.npz', clips=np.array(clips), labels=y, **{f'prob_{arm}': P[arm] for arm in arms})
    print(json.dumps({k: round(v['uar'], 2) for k, v in res['arms'].items()}))
    print(f'-> {a.out}.json  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
