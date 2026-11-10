#!/usr/bin/env python3
"""Build HiCMAE-format MAFW split files from the group's annotation (AV-DFER-OT/annotation, read-only).

Line format (HiCMAE): <frames_dir> <wav_path> <label>, label in HiCMAE's MAFW order.
The group's MAFW labels (0-based) follow avdfer CLASS_NAMES['MAFW']; they are remapped by name.
Clips without a wav are dropped, as HiCMAE's preprocess/mafw_av.py does.
Writes data/mafw/split0k/{train,test}.csv and data/mafw-trainview/split0k/test.csv (= train list, for PAPA).
"""
import os

ANN = os.path.expanduser('~/AVDFEROT/AV-DFER-OT/annotation')
DATA = os.path.expanduser("~/AVDFEROT/data/mafw")
WAV = "/mnt/debian135/datadeb135/dfer-visual/data/mafw_audio_16k"
OURS = ['happiness', 'sadness', 'neutral', 'anger', 'surprise', 'disgust', 'fear',
        'contempt', 'anxiety', 'helplessness', 'disappointment']
HICMAE = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise',
          'contempt', 'anxiety', 'helplessness', 'disappointment']
MAP = {i: HICMAE.index(n) for i, n in enumerate(OURS)}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

for k in range(1, 6):
    for part in ('train', 'test'):
        rows, missing = [], 0
        with open(os.path.join(ANN, f'MAFW_set_{k}_{part}_faces.txt')) as f:
            for line in f:
                path, _, lab = line.strip().rsplit(' ', 2)
                cid = os.path.basename(path)
                fr, wav = os.path.join(DATA, 'frames', cid), os.path.join(WAV, f"{cid}.wav")
                if not os.path.exists(wav):
                    missing += 1
                    continue
                assert os.path.isdir(fr), fr
                rows.append(f'{fr} {wav} {MAP[int(lab)]}\n')
        d = os.path.join(OUT, 'mafw', f'split0{k}')
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, f'{part}.csv'), 'w').writelines(rows)
        if part == 'train':
            d2 = os.path.join(OUT, 'mafw-trainview', f'split0{k}')
            os.makedirs(d2, exist_ok=True)
            open(os.path.join(d2, 'test.csv'), 'w').writelines(rows)
        print(f'set {k} {part}: {len(rows)} clips ({missing} without wav dropped)')
