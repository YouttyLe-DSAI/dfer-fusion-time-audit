#!/usr/bin/env python
"""E3 (prereg.md): A1 = the reference SCR run retrained with the patch-mean cost descriptor (fusion.video_token: mean).

Loads the reference run's own config, fills settings added after it with their legacy values (as the code does,
avdfer/run/config.py LATER_KEYS), changes fusion.video_token from cls to mean and nothing else (asserted), and trains
the fold with the original trainer. Imports the original code read-only (PYTHONDONTWRITEBYTECODE=1); writes only
under --out-root.

    e3_train.py --src AV-DFER-OT --like <src>/runs/DFEW/dev/scr-8a30b61e0beb --fold 1 --out-root runs_e3 --resume
"""

import argparse
import copy
import json
import os
import sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--src', required=True)
ap.add_argument('--like', required=True, help='reference run dir holding fold<k>/config.json')
ap.add_argument('--fold', type=int, required=True)
ap.add_argument('--out-root', required=True)
ap.add_argument('--resume', action='store_true')
ap.add_argument('--dry-run', action='store_true', help='print the config and its ID, no training')
a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.src))

from avdfer.engine import trainer as T  # noqa: E402
from avdfer.run.config import LATER_KEYS, config_id  # noqa: E402

RECORD_KEYS = ('config_id', 'fold', 'data_fingerprint', 'train_annotation_file', 'eval_annotation_file',
               'train_annotation_sha256', 'eval_annotation_sha256')


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
    cfg['fusion']['video_token'] = 'mean'
    cfg['out_dir'] = os.path.abspath(a.out_root)
    diff = {k for k in set(flat(cfg)) | set(flat(ref)) if flat(cfg).get(k) != flat(ref).get(k)}
    assert diff == {'fusion.video_token'}, diff
    return cfg, rec['config_id']


def main():
    cfg, ref_cid = build_config()
    cid = config_id(cfg)
    print(f'reference {a.like} fold {a.fold} (config {ref_cid}); A1 config {cid}: fusion.video_token cls -> mean, '
          f'everything else equal', flush=True)
    if a.dry_run:
        print(json.dumps(cfg, indent=1))
        return
    out = os.path.join(cfg['out_dir'], cfg['data']['dataset'], cfg['data']['protocol'], f'scr-{cid}', f'fold{a.fold}')
    os.makedirs(out, exist_ok=True)
    marker = os.path.join(out, 'e3_arm.json')
    if not os.path.exists(marker):
        json.dump({'arm': 'A1', 'change': 'fusion.video_token cls -> mean', 'reference_run': os.path.abspath(a.like),
                   'reference_config_id': ref_cid, 'fold': a.fold, 'config_id': cid}, open(marker, 'w'), indent=1)
    m = T.train_fold(cfg, a.fold, resume=a.resume)
    print(f'fold {a.fold}: UAR {m["uar"]:.2f}  WAR {m["war"]:.2f}  -> {out}')


if __name__ == '__main__':
    main()
