# dfer-fusion-time-audit

Do audio-visual dynamic facial expression recognition (DFER) models use what they are built to use?
This repository holds the code and result summaries for two inference-only audits:

- **Part 1, fusion.** Is a cross-modal fusion module used, and how: through a clip-level summary of the other
  stream, or through which token receives which message? The arms are a marginal-preserving independent coupling,
  row permutations, removal, a ladder of nulls (uniform, PAPA, EMAP, perceptual score), and audio-visual offsets
  with a content-matched control. The models are HiCMAE, MMA-DFER and an optimal-transport (OT) router.
- **Part 2, time.** Do the models use the temporal axis of the video? The test-time arms are frame shuffles,
  reversal and frame budgets. For MMA-DFER there is also an order-agnostic training control.

## Layout

| Path | Part | Contents |
|---|---|---|
| `p1/p1_eval.py` | 1 | HiCMAE reproduction and the coupling interventions (base, indep, zero, per-site variants) |
| `p1/p1_nulls.py` | 1 | Null ladder on HiCMAE: devonly, shrink, rowperm, uniform, PAPA, perm_a/perm_v, sampled EMAP; F3 mechanism trace |
| `p1/p1_followup.py` | 1 | Space vs time permutations, silence, audio-visual offset curve |
| `p1/p1_desync_control.py` | 1 | Content-matched desynchronisation control |
| `p1/p1_common.py` | 1, 2 | Dataset configs, per-view seeding, shared HiCMAE helpers |
| `p1/*_summary.py` | 1 | Pooled endpoints with the paired bootstrap |
| `scripts/p2_ot_*` | 1 | Null ladder on the OT router |
| `scripts/p2_mma_*` | 1 | MMA-DFER's published exchange as a known-answer case; offsets on the OT router |
| `scripts/stage0a_*`, `scripts/s0b_*` | 1 | Repair screen: within-clip spread, linear probes, three-configuration adapter screen |
| `scripts/p2t_mma.py`, `p1/p2t_hicmae.py` | 2 | Test-time temporal arms for MMA-DFER / OT router and for HiCMAE |
| `scripts/p2t_summary.py`, `scripts/p2t_breakdown.py` | 2 | Pooled readings; per-class and duration breakdowns |
| `scripts/p2t_train_shuffle.py`, `scripts/p2t_train_compare.py` | 2 | Order-agnostic training control (shuffles frames in the training loader only) and its comparison |
| `p1/run_*.sh`, `scripts/run_*.sh` | both | Queue scripts, one per stage |
| `openface/` | 1 | Build dlib and OpenFace 2.2.0 without root; CREMA-D face alignment |
| `p1/download_hicmae*.sh`, `p1/make_mafw_splits.py` | 1 | Released HiCMAE checkpoints; MAFW splits |
| `report-assets/make_assets*.py` | both | Build every figure, table and number macro from the result summaries |
| `results/`, `p1/results/` | both | Result summaries (JSON, TXT) per stage, fold and arm |

**Naming.** In file names, `p2_*` is the final protocol stage of Part 1 (OpenFace reproduction, MAFW, OT router).
Part 2 files carry `p2t_*`.

**Result folders.**
- Part 1: `p1/results/cremad_of/` (HiCMAE on OpenFace-aligned faces, the primary HiCMAE result), `p1/results/mafw/`,
  `results/p2_ot/`, `results/p2_mma/`, `results/stage0a/`, `results/s0b/`.
- Part 2: `results/p2t/`, `results/p2t_train/`.

## Not included

- Datasets (CREMA-D, DFEW, MAFW), aligned faces and feature caches.
- Checkpoints: the released HiCMAE checkpoints and the MMA-DFER / OT-router runs.
- Per-clip predictions (`*.npz`). The bootstrap intervals in the summaries were computed from these.
- Third-party code, used unchanged: HiCMAE (github.com/sunlicai/HiCMAE, commit `cdd28d4`), OpenFace 2.2.0, dlib 19.24.6,
  and the MMA-DFER / OT-router training code, which is imported read-only.

## Environment

- HiCMAE: Python 3.8, torch 1.10.1+cu113, timm 0.4.12 (`p1/setup_env.sh`, `p1/requirements-lock.txt`).
  On glibc 2.41 it also needs `GLIBC_TUNABLES=glibc.rtld.execstack=2`.
- MMA-DFER and the OT router: the environment of their training code.
- Hardware: one RTX 3090. The queue scripts and `p1/make_mafw_splits.py` use absolute data and checkpoint paths of
  the original server, mostly set at the top of each file; edit them for your machine.
- Statistics: endpoints are pooled over the 5 official folds, with a paired bootstrap over clips within folds
  (2000 resamples, 90% CI).
