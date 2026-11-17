#!/usr/bin/env python
"""E6 (prereg.md): A2 / A3 = the reference SCR run retrained with the D7 cost-descriptor branch (e6_costdev.py).

Loads the reference run's own config and fills settings added after it with their legacy values, as e3_train.py
does. It adds fusion.cost_dev (A2: learned, A3: random) and fusion.cost_dev_rank, and changes nothing else (asserted).
The fold is trained with the original trainer; only two of its functions are wrapped:
  build_host  -> build_host + attach_cost_dev (no-op without the keys)
  gate_stats  -> the original statistics over the a2v/v2a gates only, plus the branch gates as cost_gate_*
                 (the original pairs gates as a2v, v2a, which a third gate per module would scramble).
The original code is imported read-only (PYTHONDONTWRITEBYTECODE=1). Writes go only under --out-root.

    e6_train.py --src AV-DFER-OT --like <src>/runs/DFEW/dev/scr-8a30b61e0beb --arm A2 --fold 1 --out-root runs_e6 --resume
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--src', required=True)
ap.add_argument('--like', required=True, help='reference run dir holding fold<k>/config.json')
ap.add_argument('--arm', required=True, choices=('A2', 'A3'))
ap.add_argument('--rank', type=int, default=16)
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--out-root', required=True)
ap.add_argument('--resume', action='store_true')
ap.add_argument('--dry-run', action='store_true', help='print the config and its ID, no training')
a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.src))

import numpy as np  # noqa: E402

from avdfer.engine import trainer as T  # noqa: E402
from avdfer.run.config import LATER_KEYS, config_id  # noqa: E402
from e6_costdev import attach_cost_dev  # noqa: E402

KIND = {'A2': 'learned', 'A3': 'random'}
RECORD_KEYS = ('config_id', 'fold', 'data_fingerprint', 'train_annotation_file', 'eval_annotation_file',
               'train_annotation_sha256', 'eval_annotation_sha256')
HERE = os.path.dirname(os.path.abspath(__file__))


def flat(d, p=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + '.'))
        else:
            out[p + k] = v
    return out


def build_config():
    rec = json.load(open(os.path.join(a.like, f'fold{a.fold}', 'config.json')))
    cfg = {k: v for k, v in rec.items() if k not in RECORD_KEYS}
    for path, legacy in LATER_KEYS.items():
        node = cfg
        for k in path[:-1]:
            node = node[k]
        if path[-1] not in node:
            node[path[-1]] = legacy
    ref = copy.deepcopy(cfg)
    ref['out_dir'] = os.path.abspath(a.out_root)
    assert cfg['fusion']['rule'] == 'scr' and cfg['fusion']['video_token'] == 'cls', cfg['fusion']
    assert 'cost_dev' not in cfg['fusion']
    cfg['fusion']['cost_dev'] = KIND[a.arm]
    cfg['fusion']['cost_dev_rank'] = a.rank
    cfg['out_dir'] = os.path.abspath(a.out_root)
    diff = {k for k in set(flat(cfg)) | set(flat(ref)) if flat(cfg).get(k) != flat(ref).get(k)}
    assert diff == {'fusion.cost_dev', 'fusion.cost_dev_rank'}, diff
    return cfg, rec['config_id']


ORIG_BUILD = T.build_host


def build_host(cfg):
    return attach_cost_dev(ORIG_BUILD(cfg), cfg)


def gate_stats(model):
    """The trainer's gate_stats over the a2v/v2a gates only, plus the branch gates."""
    vals, grads, cvals, cgrads = [], [], [], []
    for n, p in model.named_parameters():
        if not (n.startswith('fusion.') and 'gate' in n):
            continue
        v, g = (cvals, cgrads) if '.cost_dev.' in n else (vals, grads)
        v.append(p.item())
        if p.grad is not None:
            g.append(abs(p.grad.item()))
    if not vals:
        return {}
    out = {'gate_abs_mean': sum(abs(v) for v in vals) / len(vals),
           'gate_abs_max': max(abs(v) for v in vals),
           'gate_tanh_a2v_mean': float(np.mean([math.tanh(v) for v in vals[0::2]])),
           'gate_tanh_v2a_mean': float(np.mean([math.tanh(v) for v in vals[1::2]]))}
    if grads:
        out['gate_grad_abs_mean'] = sum(grads) / len(grads)
    if cvals:
        out['cost_gate_tanh_mean'] = float(np.mean([math.tanh(v) for v in cvals]))
        out['cost_gate_tanh_absmax'] = float(max(abs(math.tanh(v)) for v in cvals))
        if cgrads:
            out['cost_gate_grad_abs_mean'] = sum(cgrads) / len(cgrads)
    return out


T.build_host = build_host
T.gate_stats = gate_stats


def main():
    cfg, ref_cid = build_config()
    cid = config_id(cfg)
    print(f'reference {a.like} fold {a.fold} (config {ref_cid}); {a.arm} config {cid}: '
          f'fusion.cost_dev={KIND[a.arm]}, cost_dev_rank={a.rank}, everything else equal', flush=True)
    if a.dry_run:
        print(json.dumps(cfg, indent=1))
        return
    out = os.path.join(cfg['out_dir'], cfg['data']['dataset'], cfg['data']['protocol'], f'scr-{cid}', f'fold{a.fold}')
    os.makedirs(out, exist_ok=True)
    marker = os.path.join(out, 'e6_arm.json')
    if not os.path.exists(marker):
        src = {f: hashlib.sha256(open(os.path.join(HERE, f), 'rb').read()).hexdigest()
               for f in ('e6_costdev.py', 'e6_train.py')}
        json.dump({'arm': a.arm, 'change': f'fusion.cost_dev={KIND[a.arm]}, fusion.cost_dev_rank={a.rank}',
                   'reference_run': os.path.abspath(a.like), 'reference_config_id': ref_cid, 'fold': a.fold,
                   'config_id': cid, 'wrapper_sha256': src}, open(marker, 'w'), indent=1)
    m = T.train_fold(cfg, a.fold, resume=a.resume)
    print(f'fold {a.fold}: UAR {m["uar"]:.2f}  WAR {m["war"]:.2f}  -> {out}')


if __name__ == '__main__':
    main()
