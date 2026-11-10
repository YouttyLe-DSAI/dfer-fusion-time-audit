#!/usr/bin/env python
"""P1-desync-control (prereg.md): is the delay effect due to AV misalignment or to changed content?

Same model, data, views and checkpoints as p1_followup.py; gain seeded per view; batch 16, unshuffled.
For d in D_MS (d > 0 = sound late), with whole-frame shifts so both modalities move by the same time:
  aud_d   video unchanged, audio window moved by -d          offset d, audio content changed
  vid_d   video moved by +d (edge frames repeated), audio as is   offset d, video content changed
  both_d  video and audio moved by -d (audio = aud_d's)      offset 0
"""

import argparse
import json
import os
import sys
import time

D_MS = (-320, -160, 160, 320)
MOVES = (0, 160, -160, 320, -320)          # window moves in ms (positive = later); index = tensor slot

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--hicmae', required=True)
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--ckpt', required=True)
ap.add_argument('--data', required=True)
ap.add_argument('--batch-size', type=int, default=16)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--dataset', default='cremad', choices=('cremad', 'mafw'))
ap.add_argument('--out', required=True)
a = ap.parse_args()

sys.path.insert(0, os.path.abspath(a.hicmae))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from timm.models import create_model  # noqa: E402

import modeling_finetune_av  # noqa: E402,F401
import run_class_finetuning_av as R  # noqa: E402
from datasets_av import build_dataset  # noqa: E402
from kinetics_av import VideoReaderFrame  # noqa: E402

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


def _shift(src, s, e, d):
    """Samples [s + d, e + d) of src (1, n), zero-padded outside [0, n)."""
    n = src.shape[1]
    out = np.zeros((1, e - s), dtype=np.float32)
    lo, hi = max(s + d, 0), min(e + d, n)
    if hi > lo:
        out[:, lo - (s + d):hi - (s + d)] = src[:, lo:hi]
    return out


def getitem_moves(self, index):
    """HiCMAE's VideoClsDatasetFrame test path, once per window move (MOVES[0] = 0 reproduces it)."""
    sample, sample_audio = self.test_dataset[index], self.test_dataset_audio[index]
    chunk_nb, split_nb = self.test_seg[index]
    vr = VideoReaderFrame(sample, file_ext=self.file_ext)
    N = len(vr)
    all_index = [x for x in range(0, N, self.frame_sample_rate)]
    while len(all_index) < self.clip_len:
        all_index.append(all_index[-1])
    L = len(all_index)
    temporal_step = max(1.0 * (L - self.clip_len) / (self.test_num_segment - 1), 0)
    temporal_start = int(chunk_nb * temporal_step)
    sel = all_index[temporal_start:temporal_start + self.clip_len]
    start_f = all_index[temporal_start]
    end_f = min(all_index[min(temporal_start + self.clip_len - 1, L - 1)] + self.frame_sample_rate, N - 1)

    audio, sr = torchaudio.load(sample_audio)
    assert sr == self.audio_sample_rate and audio.shape[1] > 1024
    n = audio.shape[1]
    s, e = int(start_f / (N - 1.0) * n), int(end_f / (N - 1.0) * n)
    whole = (e - s) <= 1024
    if whole:
        assert n < self.audio_conf.get('target_length') / 100.0 * self.audio_sample_rate
        s, e = 0, n
    src = audio.numpy()
    frame_ms = (n / sr) / (N - 1.0) * 1000.0
    samples_per_frame = n / (N - 1.0)

    vids, auds, real = [], [], []
    for m in MOVES:
        df = int(np.sign(m)) * int(round(abs(m) / frame_ms))                  # whole frames
        real.append(df * frame_ms)
        idx = [min(max(i + df, 0), N - 1) for i in sel]
        buf = self.data_resize(vr.load(idx))
        if isinstance(buf, list):
            buf = np.stack(buf, 0)
        spatial_step = 1.0 * (max(buf.shape[1], buf.shape[2]) - self.short_side_size) / (self.test_num_crop - 1)
        ss = int(split_nb * spatial_step)
        if buf.shape[1] >= buf.shape[2]:
            buf = buf[:, ss:ss + self.short_side_size, :, :]
        else:
            buf = buf[:, :, ss:ss + self.short_side_size, :]
        vids.append(self.data_transform(buf))
        np.random.seed(self._p1_seed + index)                                  # same gain in every move
        if m == 0:
            seg = audio if whole else src[:, s:e]
        else:
            seg = _shift(src, s, e, int(round(df * samples_per_frame)))
        fbank, _ = self._wav2fbank(seg, sr=self.audio_sample_rate)
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        auds.append(fbank.unsqueeze(0))
    return ((torch.stack(vids), torch.stack(auds)), self.test_label_array[index], sample, chunk_nb, split_nb,
            torch.tensor(real))


def patch_dataset(ds):
    type(ds).__getitem__ = getitem_moves
    ds._p1_seed = a.seed * 1_000_000
    assert ds.mode == 'test' and not ds.noise and ds.freqm == 0 and ds.timem == 0


def head(model, vf, af, f, fa):
    feat = torch.cat([vf, af, f.mean(dim=1), fa.mean(dim=1)], dim=-1)
    return model.head_activation_func(model.head(model.fc_norm(feat))).float()


def enc_video(model, videos):
    x = model.encoder(videos)
    vf = (x.mean(dim=1) * torch.softmax(model.video_layer_weights, dim=0).view(1, -1, 1)).sum(dim=1)
    return vf, x[:, :, -1].contiguous()


def enc_audio(model, audios):
    xa = model.encoder_audio(audios)
    af = (xa.mean(dim=1) * torch.softmax(model.audio_layer_weights, dim=0).view(1, -1, 1)).sum(dim=1)
    return af, xa[:, :, -1].contiguous()


def arm_list():
    return ['base'] + [f'{k}{d:+d}' for d in D_MS for k in ('aud', 'vid', 'both')]


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

    ds, _ = build_dataset(is_train=False, test_mode=True, args=args)

    prepare_dataset(ds, CFG)
    patch_dataset(ds)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                                         pin_memory=True, drop_last=False)
    print(f'fold {a.fold}: {len(ds)} test views, moves {MOVES} ms', flush=True)
    slot = {m: k for k, m in enumerate(MOVES)}
    arms = arm_list()
    probs = {arm: {} for arm in arms}
    labels, check, real = {}, None, []
    t0 = time.time()
    with torch.no_grad():
        for it, batch in enumerate(loader):
            if a.limit and it >= a.limit:
                break
            (videos, audios), target, ids, chunk, split, rl = batch
            videos, audios = videos.to(DEV, non_blocking=True), audios.to(DEV, non_blocking=True)
            real.append(rl.numpy())
            outs = {}
            with torch.cuda.amp.autocast():
                V = {m: enc_video(model, videos[:, slot[m]]) for m in MOVES}
                A = {m: enc_audio(model, audios[:, slot[m]]) for m in MOVES}

                def run(vm, am):
                    vf, xl = V[vm]
                    af, xal = A[am]
                    return head(model, vf, af, *model.encoder_fusion(xl, xal))
                outs['base'] = run(0, 0)
                if check is None:
                    check = (model(videos[:, 0], audios[:, 0]).float() - outs['base']).abs().max().item()
                    print(f'check: max |model() - base| = {check:.2e}', flush=True)
                    assert check < 1e-2, check
                for d in D_MS:
                    outs[f'aud{d:+d}'] = run(0, -d)
                    outs[f'vid{d:+d}'] = run(d, 0)
                    outs[f'both{d:+d}'] = run(-d, -d)
            for arm in arms:
                p = torch.softmax(outs[arm].double(), dim=-1).cpu().numpy()
                for i, c in enumerate(ids):
                    probs[arm].setdefault(c, {})[(int(chunk[i]), int(split[i]))] = p[i]
            for i, c in enumerate(ids):
                labels[c] = int(target[i])
            if it % 50 == 0:
                print(f'  batch {it}/{len(loader)}  {time.time() - t0:.0f}s', flush=True)

    clips = sorted(labels)
    y = np.array([labels[c] for c in clips])
    P = {arm: np.stack([np.mean(list(probs[arm][c].values()), axis=0) for c in clips]) for arm in arms}
    pred = {arm: P[arm].argmax(1) for arm in arms}
    real = np.concatenate(real)
    res = {'fold': a.fold, 'ckpt': a.ckpt, 'n_clips': len(clips), 'limit': a.limit, 'seed': a.seed,
           'check_max_diff': check, 'd_ms': list(D_MS), 'moves_ms': list(MOVES),
           'realised_move_ms': {str(m): [float(real[:, k].mean()), float(real[:, k].min()), float(real[:, k].max())]
                                for k, m in enumerate(MOVES)}, 'arms': {}}
    for arm in arms:
        rec = [(pred[arm][y == c] == c).mean() for c in range(CFG['nb_classes']) if (y == c).any()]
        res['arms'][arm] = {'uar': 100.0 * float(np.mean(rec)), 'war': 100.0 * float((pred[arm] == y).mean()),
                            'changed_pct': 100.0 * float((pred[arm] != pred['base']).mean())}
    with open(a.out + '.json', 'w') as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(a.out + '.npz', clips=np.array(clips), labels=y, **{f'prob_{arm}': P[arm] for arm in arms})
    print(json.dumps({k: round(v['uar'], 2) for k, v in res['arms'].items()}))
    print('realised moves (mean, min, max ms):', res['realised_move_ms'])
    print(f'-> {a.out}.json  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
