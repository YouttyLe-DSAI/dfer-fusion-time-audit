#!/usr/bin/env python
"""P2T stage 0 (prereg.md): temporal interventions on MMA-DFER and the OT router.

Imports the original code read-only (PYTHONDONTWRITEBYTECODE=1). Loads a run (dev or confirm) and evaluates it on
the official test fold. The arms (p2t_common.ARMS) reorder or repeat the 16 input frames of each clip after the
loader's frame selection and transform; audio and timing are unchanged. One forward per arm on the same batch.
"""

import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--src', required=True, help='AV-DFER-OT repo (imported read-only)')
ap.add_argument('--run', required=True, help='run dir holding fold<k>/{config.json,model.pth}')
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--limit', type=int, default=0, help='batches, 0 = all (smoke tests only)')
ap.add_argument('--out', required=True)
a = ap.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(a.src))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import p2t_common as C  # noqa: E402
from avdfer.data import build_dataset, timing_to  # noqa: E402
from avdfer.engine.evaluate import autocast  # noqa: E402
from avdfer.hosts import audio_input, build_host, load_checkpoint_state  # noqa: E402

DEV = torch.device('cuda')


def ann_path(cfg, part):
    d = cfg['data']
    suffix = '_faces' if d['dataset'] == 'MAFW' else ''
    p = os.path.join(a.src, d.get('annotation_dir', 'annotation'), f"{d['dataset']}_set_{a.fold}_{part}{suffix}.txt")
    assert os.path.exists(p), p
    return p


def main():
    C.self_test()
    fold_dir = os.path.join(a.run, f'fold{a.fold}')
    cfg = json.load(open(os.path.join(fold_dir, 'config.json')))
    model = build_host(cfg)
    state = torch.load(os.path.join(fold_dir, 'model.pth'), map_location='cpu', weights_only=False)['state_dict']
    load_checkpoint_state(model, state)
    model.to(DEV).eval()
    routed = model.fusion is not None
    prec = cfg['train'].get('precision', 'fp32')
    mpath = os.path.join(fold_dir, 'metrics.json')
    run_metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}
    print(f'{a.run} fold {a.fold}: routing {"on" if routed else "none"}, precision {prec}', flush=True)

    d = cfg['data']
    test_ann = ann_path(cfg, 'test')
    ds = build_dataset(test_ann, d['dataset'], 'test', cfg['host']['img_size'], frames_root=d['frames_root'],
                       audio_root=d['audio_root'], missing_audio=d['missing_audio'],
                       frame_order=d.get('frame_order', 'lexicographic'),
                       audio_channels=d.get('audio_channels', 'first'),
                       frame_times=d.get('frame_times', 'uniform'), **audio_input(cfg))
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg['eval']['batch_size'], shuffle=False,
                                         num_workers=a.workers)
    arms = list(C.ARMS)
    print(f'test: {len(ds)} clips ({test_ann}); arms {arms}', flush=True)
    logits = {k: [] for k in arms}
    labels, ids, check = [], [], None
    t0 = time.time()
    with torch.no_grad(), autocast(DEV, prec):
        for it, (images, label, audio, index, timing) in enumerate(loader):
            if a.limit and it >= a.limit:
                break
            images, audio = images.to(DEV), audio.to(DEV)
            tm = timing_to(timing, DEV)
            n, t = images.shape[:2]
            assert t == C.T, t
            vids = [C.view_id('mma', int(i)) for i in index]
            rows = torch.arange(n, device=DEV).unsqueeze(1)
            for arm in arms:
                if arm == 'base':
                    x = images
                else:
                    idx = torch.as_tensor(np.stack([C.index_for(arm, v) for v in vids]), device=DEV)
                    x = images[rows, idx]
                logits[arm].append(model(x, audio, tm).float().cpu())
            if check is None:
                check = (logits['identity'][-1] - logits['base'][-1]).abs().max().item()
                print(f'check: max |identity - base| = {check:.2e}', flush=True)
                assert check < 1e-3, check
            labels.append(label)
            ids.append(index)
            if it % 50 == 0:
                print(f'  batch {it}/{len(loader)}  {time.time() - t0:.0f}s', flush=True)
    y = torch.cat(labels).numpy()
    P = {k: torch.softmax(torch.cat(v).double(), dim=-1).numpy() for k, v in logits.items()}
    pred = {k: p.argmax(1) for k, p in P.items()}
    ncls = cfg['data']['num_classes']
    res = {'run': a.run, 'fold': a.fold, 'routed': routed, 'precision': prec, 'n_clips': int(len(y)),
           'test_annotation': test_ann, 'limit': a.limit, 'check_identity_vs_base': check,
           'run_metrics_uar': run_metrics.get('uar'), 'arms': {}}
    for k in arms:
        rec = [(pred[k][y == c] == c).mean() for c in range(ncls) if (y == c).any()]
        res['arms'][k] = {'uar': 100.0 * float(np.mean(rec)), 'war': 100.0 * float((pred[k] == y).mean()),
                          'changed_pct': 100.0 * float((pred[k] != pred['base']).mean())}
    with open(a.out + '.json', 'w') as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(a.out + '.npz', clips=torch.cat(ids).numpy(), labels=y, **{f'prob_{k}': P[k] for k in arms})
    print(json.dumps({k: (round(v['uar'], 2), round(v['changed_pct'], 2)) for k, v in res['arms'].items()}))
    print(f'base UAR {res["arms"]["base"]["uar"]:.2f} vs run metrics {res["run_metrics_uar"]}')
    print(f'-> {a.out}.json  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
