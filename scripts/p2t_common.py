"""P2T stage 0 (prereg.md): the temporal interventions shared by every model.

Each arm maps a view to an index array over the T = 16 input frames: new_frames[j] = frames[idx[j]].
Random arms are seeded from (arm seed, view id), so a view gets the same permutation in every run and model.
Python 3.8 compatible (HiCMAE's environment).
"""

import zlib

import numpy as np

T = 16
SEEDS = {'shuffle_s0': 0, 'shuffle_s1': 1, 'shuffle_s2': 2, 'shuffle_pair': 10, 'static_rand': 20}
HOLDS = (8, 4, 2, 1)
ARMS = ['base', 'identity', 'shuffle_s0', 'shuffle_s1', 'shuffle_s2', 'shuffle_pair', 'reverse'] + \
       [f'hold{k}' for k in HOLDS] + ['static_rand']


def view_id(*parts):
    """A stable integer id for a view, from strings or ints."""
    return zlib.crc32('|'.join(str(p) for p in parts).encode())


def hold_index(k, t=T):
    picks = [int((j + 0.5) * t / k) for j in range(k)]
    return np.repeat(np.array(picks), t // k)


def pair_perm(vid, t=T):
    """Permutation of the t // 2 consecutive frame pairs for this view (shared by shuffle_pair and shuffle_pair_av)."""
    return np.random.default_rng([SEEDS['shuffle_pair'], vid]).permutation(t // 2)


def index_for(arm, vid, t=T):
    if arm in ('base', 'identity'):
        return np.arange(t)
    if arm.startswith('shuffle_s'):
        return np.random.default_rng([SEEDS[arm], vid]).permutation(t)
    if arm in ('shuffle_pair', 'shuffle_pair_av'):
        p = pair_perm(vid, t)
        return np.stack([2 * p, 2 * p + 1], 1).reshape(-1)
    if arm == 'reverse':
        return np.arange(t)[::-1].copy()
    if arm.startswith('hold'):
        return hold_index(int(arm[4:]), t)
    if arm == 'static_rand':
        r = int(np.random.default_rng([SEEDS['static_rand'], vid]).integers(t))
        return np.full(t, r)
    raise ValueError(arm)


def self_test():
    assert list(hold_index(8)) == [1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15]
    assert list(hold_index(1)) == [8] * 16
    assert list(hold_index(4)) == [2] * 4 + [6] * 4 + [10] * 4 + [14] * 4
    for arm in ARMS + ['shuffle_pair_av']:
        idx = index_for(arm, 12345)
        assert idx.shape == (T,) and idx.min() >= 0 and idx.max() < T, arm
        if arm.startswith('shuffle') or arm in ('reverse', 'identity'):
            assert sorted(idx) == list(range(T)), arm
    sp = index_for('shuffle_pair', 7)
    assert all(sp[2 * i + 1] == sp[2 * i] + 1 and sp[2 * i] % 2 == 0 for i in range(T // 2))
    assert (index_for('shuffle_s0', 5) == index_for('shuffle_s0', 5)).all()
    assert not (index_for('shuffle_s0', 5) == index_for('shuffle_s1', 5)).all()


if __name__ == '__main__':
    self_test()
    print('ok')
