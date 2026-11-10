"""Dataset settings shared by the P1 scripts, and the frame-order patch.

README values are from github.com/sunlicai/HiCMAE (commit cdd28d4).
MAFW: the server's frames are named 0.jpg, 1.jpg, ..., so HiCMAE's lexicographic sort (VideoReaderFrame)
would put 10.jpg before 2.jpg. numeric_frame_order() sorts by the integer stem when every stem is an
integer and falls back to HiCMAE's order otherwise; on zero-padded names (CREMA-D) both orders agree.
HiCMAE's MAFW branch builds the dataset with file_ext='png'; our MAFW frames are jpg.
"""

import glob
import os

DATASETS = {
    'cremad': {'data_set': 'CREMA-D', 'nb_classes': 6, 'file_ext': None,
               'readme_uar': {1: 86.66, 2: 83.27, 3: 87.23, 4: 83.70, 5: 83.88}, 'readme_total': (84.94, 84.91)},
    'mafw': {'data_set': 'MAFW', 'nb_classes': 11, 'file_ext': 'jpg',
             'readme_uar': {1: 36.02, 2: 41.57, 3: 46.46, 4: 47.50, 5: 41.88}, 'readme_total': (42.69, 56.21)},
}


def numeric_frame_order(kinetics_av):
    cls = kinetics_av.VideoReaderFrame

    def __init__(self, video_dir, file_ext='jpg'):
        self.video_dir = video_dir
        frames = glob.glob(os.path.join(video_dir, f'*.{file_ext}'))
        stems = [os.path.splitext(os.path.basename(p))[0] for p in frames]
        if stems and all(s.isdigit() for s in stems):
            self.frames = [p for _, p in sorted(zip((int(s) for s in stems), frames))]
        else:
            self.frames = sorted(frames)
        assert len(self.frames) >= 1, f"Error: no frame image detected for video '{video_dir}'!"
    cls.__init__ = __init__


def prepare_dataset(ds, cfg):
    if cfg['file_ext']:
        ds.file_ext = cfg['file_ext']
    return ds


def seed_views(ds, seed=0):
    """Seed numpy from the view index before each load, so HiCMAE's test-time audio gain (roll_mag_aug is
    on at test) is the same for a view in every run and condition. Must match p1_followup.py and
    p1_desync_control.py, which seed seed * 1_000_000 + index right before the audio load; nothing in
    the test path draws from numpy before that, so both give the same gain."""
    import numpy as np
    cls = type(ds)
    ds._p1_view_seed = seed * 1_000_000
    if getattr(cls, '_p1_seeded', False):
        return ds
    orig = cls.__getitem__

    def __getitem__(self, index):
        np.random.seed(self._p1_view_seed + index)
        return orig(self, index)
    cls.__getitem__ = __getitem__
    cls._p1_seeded = True
    ds._p1_view_seed = seed * 1_000_000
    return ds
