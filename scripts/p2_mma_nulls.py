#!/usr/bin/env python
"""P2-MMA (prereg.md): the audit on MMA-DFER's published fusion, and audio offsets on the OT router.

Imports the original code read-only (PYTHONDONTWRITEBYTECODE=1). Loads a dev run (trained on train90)
and evaluates it on the official test fold. MMA-DFER's published exchange: in every block each stream
receives the other's token mean as a broadcast bias (native_fusion), and the audio class token is added
to every frame before the temporal transformer. The OT router (SCR) adds its routing step on top.

  base          released dev model (the original forward)
  copy          the re-implemented forward with every path on (must equal base)
  zero_nat      native exchange off in every block (both directions)
  zero_nat_a2v  video no longer receives the audio mean
  zero_nat_v2a  audio no longer receives the video mean
  zero_late     the audio class token is not added to the frames
  zero_route    routing step off (OT router only; = P2-OT zero)
  zero_all      every audio path off: native, late, routing
  perm_a        the next clip's audio and timing (perceptual score)
  delay<d>      waveform shifted by d ms with the loader's audio_offset_ms (positive = sound late)
  silence       all-zero waveform
"""

import argparse
import json
import os
import sys
import time

MODEL_ARMS = ('base', 'copy', 'zero_nat', 'zero_nat_a2v', 'zero_nat_v2a', 'zero_late', 'zero_route', 'zero_all',
              'perm_a')
DELAYS = (-640, -320, -160, -80, -40, 40, 80, 160, 320, 640)
SILENCE_MS = 1e7          # beyond any clip: shift_waveform returns an all-zero waveform

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--src', required=True, help='AV-DFER-OT repo (imported read-only)')
ap.add_argument('--run', required=True, help='dev run dir holding fold<k>/{config.json,model.pth}')
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--limit', type=int, default=0, help='batches, 0 = all (smoke tests only)')
ap.add_argument('--out', required=True)
a = ap.parse_args()

sys.path.insert(0, os.path.abspath(a.src))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from avdfer.data import build_dataset, timing_to  # noqa: E402
from avdfer.data import dataset as DS  # noqa: E402
from avdfer.engine.evaluate import autocast  # noqa: E402
from avdfer.hosts import audio_input, build_host, load_checkpoint_state  # noqa: E402
from avdfer.hosts.mma_dfer import model as MM  # noqa: E402

DEV = torch.device('cuda')
ORIG_FWD = MM.MMADFER.forward
ON = dict(nat_a2v=True, nat_v2a=True, late=True, route=True)
SW = {'copy': ON,
      'zero_nat': {**ON, 'nat_a2v': False, 'nat_v2a': False},
      'zero_nat_a2v': {**ON, 'nat_a2v': False},
      'zero_nat_v2a': {**ON, 'nat_v2a': False},
      'zero_late': {**ON, 'late': False},
      'zero_route': {**ON, 'route': False},
      'zero_all': dict(nat_a2v=False, nat_v2a=False, late=False, route=False)}


def forward_sw(self, image, audio, timing=None):
    """MMADFER.forward (avdfer/hosts/mma_dfer/model.py @ bf34671) with switches on each audio path."""
    sw = getattr(self, '_p2', None)
    if sw is None:
        return ORIG_FWD(self, image, audio, timing)
    n, t, c, h, w = image.shape
    assert t == MM.N_FRAMES
    route_timing = self.routing_timing(timing, n, image.device) if self.needs_timing else None
    image = image.contiguous().view(-1, c, h, w)
    B = image.shape[0]
    last = len(self.audio_model.blocks) - 1
    read = {b - 1 for b in self.readout_blocks}
    video_cls, audio_cls = [], []

    for ii in range(len(self.audio_model.blocks)):
        audio = self.audio_model.forward_block_pre(ii, audio)
        image = self.image_encoder.forward_block_pre(ii, image, B)

        image_lowdim = self.image_encoder.temporal_pre_norm[ii](self.image_encoder.temporal_pre[ii](image))
        audio_lowdim = self.image_encoder.audio_proj_pre[ii](audio)

        image_update, audio_update = image_lowdim, audio_lowdim
        if self.native_fusion and sw['nat_a2v']:
            image_update = image_lowdim + audio_lowdim.mean(1).unsqueeze(1).repeat_interleave(t, 0)
        if self.native_fusion and sw['nat_v2a']:
            audio_update = audio_lowdim + image_lowdim.view(B // t, t, self.n_image + 6 + 1, MM.LOWDIM).mean(1).mean(1).unsqueeze(1)

        if ii in self._fusion_at and sw['route']:
            a2v, v2a = self.fusion[self._fusion_at[ii]](image_lowdim, audio_lowdim, n, t, route_timing)
            if v2a.shape[1] > 1:
                v2a = MM.common.spread_slots(v2a, audio_update.shape[1], MM.N_AUDIO_F)
            image_update = image_update + a2v
            audio_update = audio_update + v2a

        image = self.image_encoder.forward_block_post(ii, image, image_update, B)
        audio = self.audio_model.forward_block_post(ii, audio, audio_update)
        if ii in read:
            video_cls.append(image if ii == last else self.image_encoder.norm(image[:, 0]))
            audio_cls.append(audio if ii == last else self.audio_model.norm(audio[:, 0]))

    image = self._mix(video_cls, self.readout_video)
    audio = self._mix(audio_cls, self.readout_audio)
    image = image.contiguous().view(n, t, -1)
    image = self.vision_proj((image + audio.unsqueeze(1)) if sw['late'] else image)
    return self.our_classifier(self.temporal_net(image))


MM.MMADFER.forward = forward_sw


class MultiAudio(torch.utils.data.Dataset):
    """The loader's item plus one normalised filterbank per audio offset (the loader's own _fbank)."""

    def __init__(self, ds, offsets):
        self.ds, self.offsets = ds, list(offsets)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        images, label, fbank, index, timing = self.ds[i]
        clip = self.ds.clips[i]
        extra = []
        for off in self.offsets:
            if os.path.exists(clip.audio_path):
                self.ds.audio_offset_ms = float(off)
                try:
                    f, _, _ = self.ds._fbank(clip.audio_path)
                finally:
                    self.ds.audio_offset_ms = 0.0
            else:
                f = torch.zeros(self.ds.fbank_frames, self.ds.mel_bins)
            extra.append(((f - DS.FBANK_MEAN) / (DS.FBANK_STD * 2)).unsqueeze(0))
        return images, label, fbank, index, timing, torch.stack(extra)


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
    routed = model.fusion is not None
    arms = [k for k in MODEL_ARMS if routed or k != 'zero_route']
    audio_arms = [f'delay{d:+d}' for d in DELAYS] + ['silence']
    offsets = [0.0] + [float(d) for d in DELAYS] + [SILENCE_MS]
    prec = cfg['train'].get('precision', 'fp32')
    print(f'{a.run} fold {a.fold}: routing {"on" if routed else "none"}, native_fusion {model.native_fusion}, '
          f'precision {prec}', flush=True)

    d = cfg['data']
    test_ann = ann_path(cfg, 'test')
    base_ds = build_dataset(test_ann, d['dataset'], 'test', cfg['host']['img_size'], frames_root=d['frames_root'],
                            audio_root=d['audio_root'], missing_audio=d['missing_audio'],
                            frame_order=d.get('frame_order', 'lexicographic'),
                            audio_channels=d.get('audio_channels', 'first'),
                            frame_times=d.get('frame_times', 'uniform'), **audio_input(cfg))
    loader = torch.utils.data.DataLoader(MultiAudio(base_ds, offsets), batch_size=cfg['eval']['batch_size'],
                                         shuffle=False, num_workers=a.workers)
    print(f'test: {len(base_ds)} clips ({test_ann}); arms {arms + audio_arms}', flush=True)
    logits = {k: [] for k in arms + audio_arms}
    labels, check, check0, prev = [], None, None, None
    t0 = time.time()
    with torch.no_grad(), autocast(DEV, prec):
        for it, (images, label, audio, _, timing, extra) in enumerate(loader):
            if a.limit and it >= a.limit:
                break
            if check0 is None:
                check0 = (extra[:, 0] - audio).abs().max().item()
                print(f'check: max |delay0 fbank - loader fbank| = {check0:.2e}', flush=True)
                assert check0 == 0.0, check0
            images, audio = images.to(DEV), audio.to(DEV)
            tm = timing_to(timing, DEV)
            n = images.shape[0]
            for arm in arms:
                model._p2 = SW.get(arm)
                if arm == 'perm_a':
                    model._p2 = None
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
            model._p2 = None
            for j, arm in enumerate(audio_arms, 1):
                logits[arm].append(model(images, extra[:, j].to(DEV), tm).float().cpu())
            prev = (audio[-1:], {k: (v[-1:] if torch.is_tensor(v) and v.shape[:1] == (n,) else v)
                                 for k, v in tm.items()})
            if check is None:
                check = (logits['copy'][-1] - logits['base'][-1]).abs().max().item()
                print(f'check: max |copy - base| = {check:.2e}', flush=True)
                assert check < 1e-4, check
            labels.append(label)
            if it % 50 == 0:
                print(f'  batch {it}/{len(loader)}  {time.time() - t0:.0f}s', flush=True)
    y = torch.cat(labels).numpy()
    P = {k: torch.softmax(torch.cat(v).double(), dim=-1).numpy() for k, v in logits.items()}
    pred = {k: p.argmax(1) for k, p in P.items()}
    ncls = cfg['data']['num_classes']
    res = {'run': a.run, 'fold': a.fold, 'routed': routed, 'n_clips': int(len(y)), 'test_annotation': test_ann,
           'limit': a.limit, 'check_copy_vs_base': check, 'check_delay0_fbank': check0, 'delays_ms': list(DELAYS),
           'arms': {}}
    for k in logits:
        rec = [(pred[k][y == c] == c).mean() for c in range(ncls) if (y == c).any()]
        res['arms'][k] = {'uar': 100.0 * float(np.mean(rec)), 'war': 100.0 * float((pred[k] == y).mean()),
                          'changed_pct': 100.0 * float((pred[k] != pred['base']).mean())}
    with open(a.out + '.json', 'w') as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(a.out + '.npz', clips=np.arange(len(y)), labels=y, **{f'prob_{k}': P[k] for k in logits})
    print(json.dumps({k: (round(v['uar'], 2), round(v['changed_pct'], 2)) for k, v in res['arms'].items()}))
    print(f'-> {a.out}.json  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
