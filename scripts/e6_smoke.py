#!/usr/bin/env python
"""E6 pre-run checks of the D7 branch (e6_costdev.py) on A0 fold 1. Writes only under --out.

1. Same seed -> every weight shared with A0 is identical at initialisation (the branch draws no global RNG).
2. With A0's trained weights and g = 0: logits equal A0's bit for bit (A2 and A3).
3. g != 0 changes the logits (the plan reads v_cost).
4. Gradients: at g = 0 only g gets a gradient; at g != 0 down/up (A2) do too. A3 trains 12 more scalars than A0.
5. The branch gates fall in the optimiser's no-decay group.
6. A checkpoint with g != 0 goes through blocksub_ot.py and e2_ot.py (--limit), which reload it.
"""

import argparse
import copy
import json
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--ref', required=True, help='A0 run dir')
ap.add_argument('--out', required=True)
a = ap.parse_args()
sys.path.insert(0, os.path.abspath(a.src))

import torch  # noqa: E402

from avdfer.data import build_dataset, timing_to  # noqa: E402
from avdfer.engine import trainer as T  # noqa: E402
from avdfer.engine.evaluate import autocast  # noqa: E402
from avdfer.hosts import audio_input, build_host, checkpoint_state, load_checkpoint_state  # noqa: E402
from avdfer.run.config import LATER_KEYS  # noqa: E402
from avdfer.run.seed import fold_seed, set_seed  # noqa: E402
from e6_costdev import attach_cost_dev, cost_gates  # noqa: E402

DEV = torch.device('cuda')
HERE = os.path.dirname(os.path.abspath(__file__))
fold_dir = os.path.join(a.ref, 'fold1')
rec = json.load(open(os.path.join(fold_dir, 'config.json')))
cfg0 = {k: v for k, v in rec.items() if k not in ('config_id', 'fold', 'data_fingerprint', 'train_annotation_file',
                                                    'eval_annotation_file', 'train_annotation_sha256',
                                                    'eval_annotation_sha256')}
for path, legacy in LATER_KEYS.items():
    node = cfg0
    for k in path[:-1]:
        node = node[k]
    node.setdefault(path[-1], legacy)
state0 = torch.load(os.path.join(fold_dir, 'model.pth'), map_location='cpu', weights_only=False)['state_dict']


def cfg_for(kind):
    c = copy.deepcopy(cfg0)
    if kind:
        c['fusion']['cost_dev'], c['fusion']['cost_dev_rank'] = kind, 16
    return c


def fresh(kind):
    set_seed(fold_seed(cfg0['seed'], 1))
    m = attach_cost_dev(build_host(cfg_for(kind)), cfg_for(kind))
    m.set_trainable()
    return m


def trained(kind):
    m = attach_cost_dev(build_host(cfg_for(kind)), cfg_for(kind))
    if kind is None:
        load_checkpoint_state(m, state0)
    else:
        res = m.load_state_dict(state0, strict=False)
        missing = set(res.missing_keys) - m.pretrained_keys
        assert all('.cost_dev.' in k for k in missing) and len(missing) == 12 * 3, sorted(missing)[:6]
        assert not res.unexpected_keys
    return m.to(DEV).eval()


d = cfg0['data']
ann = os.path.join(a.src, 'annotation', 'DFEW_set_1_test.txt')
ds = build_dataset(ann, 'DFEW', 'test', cfg0['host']['img_size'], frames_root=d['frames_root'],
                   audio_root=d['audio_root'], missing_audio=d['missing_audio'], frame_order=d['frame_order'],
                   audio_channels=d['audio_channels'], frame_times=d.get('frame_times', 'uniform'), **audio_input(cfg0))
batches = []
for i, b in enumerate(torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)):
    batches.append(b)
    if i == 1:
        break
prec = cfg0['train']['precision']


def logits(m):
    with torch.no_grad(), autocast(DEV, prec):
        return torch.cat([m(x.to(DEV), au.to(DEV), timing_to(tm, DEV)).float() for x, _, au, _, tm in batches])


# 1. initialisation
f0, f2 = fresh(None).state_dict(), fresh('learned').state_dict()
shared = [k for k in f0]
assert set(f2) - set(f0) == {k for k in f2 if '.cost_dev.' in k}
assert all(torch.equal(f0[k], f2[k]) for k in shared), 'initial weights differ from A0'
n0 = sum(p.numel() for p in fresh(None).parameters() if p.requires_grad)
n2 = sum(p.numel() for p in fresh('learned').parameters() if p.requires_grad)
n3 = sum(p.numel() for p in fresh('random').parameters() if p.requires_grad)
print(f'1. init identical to A0 on all {len(shared)} shared tensors; trainable A0 {n0:,}, A2 +{n2 - n0:,}, A3 +{n3 - n0}')
assert n2 - n0 == 12 * (2 * 128 * 16 + 1) and n3 - n0 == 12
r2, r3 = fresh('learned'), fresh('random')
assert all(torch.equal(m2.cost_dev.down, m3.cost_dev.down) for m2, m3 in zip(r2.routing_modules, r3.routing_modules))
print('   A2 and A3 start from the same W')
del r2, r3, f0, f2

# 2. g = 0 reproduces A0
m0 = trained(None)
L0 = logits(m0)
del m0
for kind in ('learned', 'random'):
    m = trained(kind)
    L = logits(m)
    print(f'2. {kind}: max |logit - A0| at g = 0: {(L - L0).abs().max().item():.3g}')
    assert torch.equal(L, L0)
    # 3. g != 0 changes the plan
    for mod in m.routing_modules:
        mod.cost_dev.gate.data.fill_(0.5)
    print(f'3. {kind}: max |logit - A0| at tanh(g) = {cost_gates(m)[0]:.3f}: {(logits(m) - L0).abs().max().item():.3g}')
    del m

# 4. gradients, 5. optimiser groups
for kind in ('learned', 'random'):
    m = trained(kind)
    m.set_trainable()
    m.train()
    x, y, au, _, tm = batches[0]
    for gval in (0.0, 0.5):
        for mod in m.routing_modules:
            mod.cost_dev.gate.data.fill_(gval)
        m.zero_grad()
        with autocast(DEV, prec):
            loss = torch.nn.functional.cross_entropy(m(x.to(DEV), au.to(DEV), timing_to(tm, DEV)).float(), y.to(DEV))
        loss.backward()
        mods = m.routing_modules
        gg = [abs(mod.cost_dev.gate.grad.item()) for mod in mods]
        wg = [mod.cost_dev.down.grad.abs().max().item() if mod.cost_dev.down.grad is not None else None for mod in mods]
        print(f'4. {kind} g={gval}: |grad g| blocks 1/6/12 {gg[0]:.2e} {gg[5]:.2e} {gg[11]:.2e}; '
              f'max |grad down| block 12 {wg[11]}')
        assert all(v > 0 for v in gg)
        if kind == 'learned':
            assert (wg[11] == 0.0) if gval == 0.0 else (wg[11] > 0.0)
        else:
            assert all(v is None for v in wg) and not mods[0].cost_dev.down.requires_grad
    opt = T.build_optimizer(m, cfg_for(kind))
    nodecay = {id(p) for p in opt.param_groups[1]['params']}
    assert all(id(mod.cost_dev.gate) in nodecay for mod in m.routing_modules)
    assert opt.param_groups[1]['weight_decay'] == cfg0['train']['gate_weight_decay']
    print(f'5. {kind}: branch gates in the no-decay group ({len(nodecay)} gates there)')

    # 6. eval path
    for mod in m.routing_modules:
        mod.cost_dev.gate.data.fill_(0.5)
    run = os.path.join(a.out, f'run_{kind}', 'fold1')
    os.makedirs(run, exist_ok=True)
    torch.save({'state_dict': checkpoint_state(m.cpu())}, os.path.join(run, 'model.pth'))
    json.dump({**cfg_for(kind), 'fold': 1}, open(os.path.join(run, 'config.json'), 'w'), indent=1)
    del m, opt
    torch.cuda.empty_cache()
    py = sys.executable
    for cmd in ([py, os.path.join(HERE, 'blocksub_ot.py'), '--src', a.src, '--run', os.path.dirname(run), '--fold', '1',
                 '--limit', '3', '--arms', 'base', 'indep_all', '--out', os.path.join(a.out, f'bs_{kind}')],
                [py, os.path.join(HERE, 'e2_ot.py'), '--src', a.src, '--run', os.path.dirname(run), '--split', 'test',
                 '--folds', '1', '--limit', '16', '--out', os.path.join(a.out, f'e2_{kind}')]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
    bs = json.load(open(os.path.join(a.out, f'bs_{kind}.json')))
    e2 = json.load(open(os.path.join(a.out, f'e2_{kind}.json')))
    print(f"6. {kind}: blocksub ran (changed % under indep {bs['arms']['indep_all']['changed_pct']:.1f}, "
          f"cost gates {bs['cost_gate_tanh'][0]:.3f}); e2 ran (cost check {e2['checks']['1']['cost_max_abs_diff']:.2g}, "
          f"block 12 kappa {e2['blocks'][11]['kappa']:.3f}, cost_gate {e2['blocks'][11]['cost_gate_tanh']:.3f})")
print('ALL CHECKS PASSED')
