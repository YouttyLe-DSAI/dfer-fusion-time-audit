"""E6 (prereg.md): the A2/A3 cost-descriptor branch of plan D7, attached to the AV-DFER-OT router without editing it.

    v_cost_j = CLS_j + tanh(g) * W (p_j - mean_k p_k),    W = up @ down, rank r;  g = 0 at initialisation

p_j is the mean of frame j's patch tokens. Only the cost, and so the plan, reads v_cost. The payload stays the CLS:
`video_tokens()` is untouched, a2v never reads v, and v2a averages the CLS over frames with the plan's row masses,
as in A0. At g = 0 the branch adds exactly 0, so the module computes A0's numbers bit for bit.

Arms (config keys fusion.cost_dev and fusion.cost_dev_rank):
  learned  A2: down, up and g trained.
  random   A3: down and up frozen at their random initialisation; only g is trained. They are buffers, so the host's
           set_trainable() cannot unfreeze them, and the checkpoint holds them.

down and up come from a private generator seeded with torch.initial_seed() (the trainer's fold seed) and the block
index. The global RNG stream is not touched, so every other initial weight, and the data order, are A0's.
A2 and A3 of the same fold therefore start from the same W.
"""

import math

import torch
import torch.nn as nn

from avdfer.fusion.routing import OTRouting

KINDS = ('learned', 'random')


class CostDev(nn.Module):
    """tanh(g) * W (p_j - mean_k p_k) for p [n, T, D]. The output has zero clip mean by construction."""

    def __init__(self, dim, rank, kind, seed):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f'cost_dev must be one of {KINDS}, got {kind!r}')
        self.kind, self.rank = kind, rank
        gen = torch.Generator().manual_seed(int(seed))
        # nn.Linear's default range, U(-1/sqrt(fan_in), 1/sqrt(fan_in)), drawn from the private generator.
        down = (torch.rand(rank, dim, generator=gen) * 2 - 1) / math.sqrt(dim)
        up = (torch.rand(dim, rank, generator=gen) * 2 - 1) / math.sqrt(rank)
        if kind == 'learned':
            self.down, self.up = nn.Parameter(down), nn.Parameter(up)
        else:
            self.register_buffer('down', down)
            self.register_buffer('up', up)
        self.gate = nn.Parameter(torch.zeros(1))

    def extra_repr(self):
        return f'kind={self.kind}, rank={self.rank}'

    def forward(self, p):
        d = p - p.mean(dim=1, keepdim=True)
        return torch.tanh(self.gate) * (d @ self.down.t() @ self.up.t())


class OTRoutingCostDev(OTRouting):
    """OTRouting whose cost reads video_tokens_cost(). Everything else is the parent's code."""

    def video_tokens_cost(self, image_lowdim, n, t):
        """[n*t, L_img, D] -> [n, t, D]: the CLS plus the gated, centred, projected patch-mean deviation."""
        x = image_lowdim.view(n, t, -1, self.dim)
        p = x[:, :, 1:1 + self.n_image, :].mean(dim=2)
        return x[:, :, 0, :] + self.cost_dev(p)

    def transport_plan(self, v, a, timing=None):
        # Inside forward, the parent passes the CLS; the plan is built on v_cost instead. Called from outside
        # (diagnostics), the caller passes video_tokens_cost() itself. OTRouting.transport_plan is looked up at call
        # time, so a diagnostic's patch of it (blocksub_ot.py) applies here too.
        if self._v_cost is not None:
            v = self._v_cost
        return OTRouting.transport_plan(self, v, a, timing)

    def _forward(self, image_lowdim, audio_lowdim, n, t, timing=None):
        self._v_cost = self.video_tokens_cost(image_lowdim, n, t)
        try:
            return OTRouting._forward(self, image_lowdim, audio_lowdim, n, t, timing)
        finally:
            self._v_cost = None


def attach_cost_dev(model, cfg):
    """Add the branch to every routing module when cfg['fusion'] asks for it; otherwise leave the model as it is.

    Call after build_host and before set_trainable / load_checkpoint_state / .to(device).
    """
    fu = cfg.get('fusion', {})
    kind = fu.get('cost_dev')
    if kind is None:
        return model
    rank = int(fu['cost_dev_rank'])
    base = torch.initial_seed()
    rng = torch.get_rng_state()
    for i, m in enumerate(model.routing_modules):
        if type(m) is not OTRouting:
            raise TypeError(f'routing module {i} is {type(m).__name__}, expected a plain OTRouting')
        if m.video_token != 'cls' or m.rule == 'pooled' or m.time_lambda > 0:
            raise ValueError('the branch is defined on the CLS descriptor of a cost-reading rule without time prior')
        m.cost_dev = CostDev(m.dim, rank, kind, seed=(base * 1009 + i + 1) % (2 ** 63))
        m._v_cost = None
        m.__class__ = OTRoutingCostDev
    assert torch.equal(rng, torch.get_rng_state()), 'the global RNG stream was touched'
    return model


def cost_gates(model):
    """tanh(g) of each block's branch, [] when there is none."""
    return [float(torch.tanh(m.cost_dev.gate).item()) for m in model.routing_modules if hasattr(m, 'cost_dev')]
