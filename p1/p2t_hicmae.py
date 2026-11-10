#!/usr/bin/env python
"""P2T stage 0 (prereg.md): temporal interventions on HiCMAE's released checkpoints.

Same model, data, views and protocol as p1_followup.py / p1_desync_control.py (gain seeded per view, batch 16,
unshuffled). The video tensor of each view (C, T=16, H, W) is reordered or repeated along T after HiCMAE's own test
transform; the audio is unchanged, except in shuffle_pair_av, where the same pair permutation is applied to 8 equal
chunks of the filterbank over its real (unpadded) length.
"""

import argparse
import json
import os
import sys
import time

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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'scripts'))
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
import p2t_common as C  # noqa: E402
from p1_common import DATASETS, numeric_frame_order, prepare_dataset  # noqa: E402

CFG = DATASETS[a.dataset]
numeric_frame_order(kinetics_av)
DEV = torch.device('cuda')
ARMS = list(C.ARMS) + ['shuffle_pair_av']
WIN, HOP = 400, 160          # kaldi fbank at 16 kHz: 25 ms window, 10 ms shift, snip_edges


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


def getitem_p2t(self, index):
    """HiCMAE's VideoClsDatasetFrame test path (as in p1_desync_control.py with move 0), plus the real fbank length."""
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
    if (e - s) <= 1024:
        assert n < self.audio_conf.get('target_length') / 100.0 * self.audio_sample_rate
        seg = audio
    else:
        seg = audio.numpy()[:, s:e]
    seg_len = seg.shape[1]

    buf = self.data_resize(vr.load(sel))
    if isinstance(buf, list):
        buf = np.stack(buf, 0)
    spatial_step = 1.0 * (max(buf.shape[1], buf.shape[2]) - self.short_side_size) / (self.test_num_crop - 1)
    ss = int(split_nb * spatial_step)
    if buf.shape[1] >= buf.shape[2]:
        buf = buf[:, ss:ss + self.short_side_size, :, :]
    else:
        buf = buf[:, :, ss:ss + self.short_side_size, :]
    video = self.data_transform(buf)
    np.random.seed(self._p1_seed + index)                       # the same gain as in every P1 run
    fbank, _ = self._wav2fbank(seg, sr=self.audio_sample_rate)
    fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
    n_real = min(1 + (seg_len - WIN) // HOP, fbank.shape[0])
    return ((video, fbank.unsqueeze(0)), self.test_label_array[index], sample, chunk_nb, split_nb, n_real)


def patch_dataset(ds):
    type(ds).__getitem__ = getitem_p2t
    ds._p1_seed = a.seed * 1_000_000
    assert ds.mode == 'test' and not ds.noise and ds.freqm == 0 and ds.timem == 0


def permute_audio_chunks(audios, n_real, perms):
    """audios (B, 1, F, M): for each view, reorder 8 near-equal chunks of the first n_real frames by perms[b]."""
    out = audios.clone()
    for b in range(audios.shape[0]):
        n = int(n_real[b])
        k = len(perms[b])
        cuts = [int(round(j * n / k)) for j in range(k + 1)]
        chunks = [audios[b, :, cuts[j]:cuts[j + 1]] for j in range(k)]
        out[b, :, :n] = torch.cat([chunks[p] for p in perms[b]], dim=1)
    return out


def main():
    C.self_test()
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
    print(f'fold {a.fold}: {len(ds)} test views; arms {ARMS}', flush=True)
    probs = {arm: {} for arm in ARMS}
    labels, check, nreal = {}, None, []
    t0 = time.time()
    with torch.no_grad():
        for it, batch in enumerate(loader):
            if a.limit and it >= a.limit:
                break
            (videos, audios), target, ids, chunk, split, n_real = batch
            videos, audios = videos.to(DEV, non_blocking=True), audios.to(DEV, non_blocking=True)
            assert videos.shape[2] == C.T, videos.shape
            nreal.append(n_real.numpy())
            vids = [C.view_id(os.path.basename(str(c).rstrip('/')), int(chunk[i]), int(split[i])) for i, c in enumerate(ids)]
            rows = torch.arange(videos.shape[0], device=DEV).unsqueeze(1)
            outs = {}
            with torch.cuda.amp.autocast():
                for arm in ARMS:
                    if arm == 'base':
                        outs[arm] = model(videos, audios).float()
                        continue
                    idx = torch.as_tensor(np.stack([C.index_for(arm, v) for v in vids]), device=DEV)
                    x = videos.permute(0, 2, 1, 3, 4)[rows, idx].permute(0, 2, 1, 3, 4).contiguous()
                    au = audios
                    if arm == 'shuffle_pair_av':
                        au = permute_audio_chunks(audios, n_real, [C.pair_perm(v) for v in vids])
                    outs[arm] = model(x, au).float()
            if check is None:
                check = (outs['identity'] - outs['base']).abs().max().item()
                print(f'check: max |identity - base| = {check:.2e}', flush=True)
                assert check < 1e-3, check
            for arm in ARMS:
                p = torch.softmax(outs[arm].double(), dim=-1).cpu().numpy()
                for i, c in enumerate(ids):
                    probs[arm].setdefault(c, {})[(int(chunk[i]), int(split[i]))] = p[i]
            for i, c in enumerate(ids):
                labels[c] = int(target[i])
            if it % 50 == 0:
                print(f'  batch {it}/{len(loader)}  {time.time() - t0:.0f}s', flush=True)

    clips = sorted(labels)
    y = np.array([labels[c] for c in clips])
    P = {arm: np.stack([np.mean(list(probs[arm][c].values()), axis=0) for c in clips]) for arm in ARMS}
    pred = {arm: P[arm].argmax(1) for arm in ARMS}
    nreal = np.concatenate(nreal)
    res = {'fold': a.fold, 'ckpt': a.ckpt, 'dataset': a.dataset, 'n_clips': len(clips), 'limit': a.limit,
           'seed': a.seed, 'check_identity_vs_base': check,
           'fbank_real_frames': [int(nreal.min()), float(nreal.mean()), int(nreal.max())], 'arms': {}}
    for arm in ARMS:
        rec = [(pred[arm][y == c] == c).mean() for c in range(CFG['nb_classes']) if (y == c).any()]
        res['arms'][arm] = {'uar': 100.0 * float(np.mean(rec)), 'war': 100.0 * float((pred[arm] == y).mean()),
                            'changed_pct': 100.0 * float((pred[arm] != pred['base']).mean())}
    with open(a.out + '.json', 'w') as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(a.out + '.npz', clips=np.array(clips), labels=y, **{f'prob_{arm}': P[arm] for arm in ARMS})
    print(json.dumps({k: (round(v['uar'], 2), round(v['changed_pct'], 2)) for k, v in res['arms'].items()}))
    print(f'-> {a.out}.json  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
