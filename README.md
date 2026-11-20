# dfer-fusion-time-audit

Code and result summaries for the manuscript *Measuring dependence on frame-specific coupling in audio-visual dynamic
facial expression recognition*, and for a second audit of temporal use that the manuscript does not report.

Do audio-visual dynamic facial expression recognition (DFER) models use what they are built to use?

- **Part 1, fusion (the manuscript).** Does a fusion module pass each frame its own part of the other stream
  (frame-specific coupling), or does the classifier use only a clip-level message? The main tool is product
  replacement: at inference, every cross-attention map or transport plan is replaced by the product of its marginals,
  which keeps each frame's and slot's mass but gives every frame the same mixture. The controls are row permutation
  (reassignment), message removal, the frame-specific message alone, uniform and training-mean (PAPA) maps, audio-visual
  offsets with a content-matched control, and checks at the temporal head. The models are HiCMAE (released checkpoints),
  MMA-DFER and an optimal-transport (OT) router in the MMA-DFER host (development checkpoints). On DFEW, the router's
  cost descriptor is retrained in three variants and its temperature is varied.
- **Part 2, time.** Do the models use the temporal axis of the video? The test-time arms are frame shuffles,
  reversal and frame budgets. For MMA-DFER there is also an order-agnostic training control.

Every experiment runs on fixed trained weights, except the DFEW descriptor variants and the Part 2 training control.

## Manuscript results and their sources

Table and figure numbers follow the current manuscript draft.

| Manuscript item | Computed by | Result folder | Figure or table builder |
|---|---|---|---|
| Test protocols and HiCMAE reproduction (Table 1) | `p1/p1_eval.py` | `p1/results/cremad_of/eval/`, `p1/results/mafw/eval/` | `report-assets/make_supp_tables.py` |
| Coupling interventions on CREMA-D and MAFW (Table 2) | `p1/p1_eval.py`, `p1/p1_nulls.py` (HiCMAE); `scripts/p2_ot_nulls.py` (router) | `p1/results/cremad_of/{eval,nulls}/`, `p1/results/mafw/{eval,nulls}/`, `results/p2_ot/CREMAD/` | `p1/*_summary.py`, `scripts/p2_ot_summary.py` |
| Product replacement on one CREMA-D clip (Fig. 1b-d) | `scripts/dump_plans.py` | `results/plandump/` | `report-assets/make_fig_plans.py` |
| HiCMAE time and space permutation, audio offsets, content-matched shift (Fig. 2) | `p1/p1_followup.py`, `p1/p1_desync_control.py` | `p1/results/cremad_of/{followup,desync}/` | `report-assets/make_assets_p2.py` |
| Descriptor spread, cost interaction and its bound per block (Fig. 3) | `scripts/e2_ot.py`, `p1/e2_hicmae.py`, `scripts/e2_summary.py` | `results/e2/` | `report-assets/make_fig_e2.py`, `report-assets/make_fig_e2_extra.py` |
| Token-mean trace through HiCMAE's fusion stages (Fig. 4a) | `p1/p1_nulls.py` | `p1/results/cremad_of/nulls/` | `report-assets/make_assets_p2.py` |
| Temporal-head controls (Fig. 4b-d) | `scripts/blocksub_ot.py` (head arms), `scripts/headctl_summary.py` | `results/headctl/` | `report-assets/make_fig_headctl.py` |
| Block-subset localization (Table 3, lower panel) | `scripts/blocksub_ot.py`, `scripts/blocksub_summary.py` | `results/blocksub/` | |
| DFEW descriptor variants and temperature (Tables 3 and 4) | `scripts/e3_train.py`, `scripts/e6_costdev.py`, `scripts/e6_train.py`, `scripts/e345_summary.py`, `scripts/e6_summary.py` | `results/e345/{seeds,A1,eps}/`, `results/e6/{A2,A3}/` | `report-assets/make_supp_tables.py` (Table 4) |
| Product-reference sensitivity (Table 3 note) | `scripts/exactprod_summary.py` | `results/exactprod/` | |

Supporting runs that the manuscript does not report: training under product coupling (`results/e345/A4/`), a positive
control with composite clips (`results/p6/`), MMA-DFER's published exchange as a known-answer case (`results/p2_mma/`),
input-level references (EMAP, perceptual score) in `p1/results/*/nulls/`, the repair screen (`results/stage0a/`,
`results/s0b/`) and all of Part 2.

**Stage codes in file names.** `e2` is the cost-interaction analysis; `e3` the patch-mean descriptor (arm A1); `e4` the
temperature sweep (`eps`); `e5` training under product coupling (arm A4); `e6` the learned and random deviations (arms A2
and A3). A0 is the class-token baseline, whose three seeds are in `results/e345/seeds/`. `p2_*` is the final protocol
stage of Part 1 (OpenFace reproduction, MAFW, OT router); Part 2 files carry `p2t_*`. In the intervention code,
`indep` is product replacement, `rowperm` reassignment, `zero` message removal and `devonly` the frame-specific message
alone.

## Layout

| Path | Part | Contents |
|---|---|---|
| `p1/p1_eval.py` | 1 | HiCMAE reproduction and the coupling interventions (base, product replacement `indep`, removal `zero`, per-site variants) |
| `p1/p1_nulls.py` | 1 | Control interventions on HiCMAE: devonly, shrink, rowperm, uniform, PAPA, perm_a/perm_v, sampled EMAP; token-mean trace (F3) |
| `p1/p1_followup.py` | 1 | Space vs time permutations, silence, audio-visual offset curve |
| `p1/p1_desync_control.py` | 1 | Content-matched desynchronisation control |
| `p1/p1_common.py` | 1, 2 | Dataset configs, per-view seeding, shared HiCMAE helpers |
| `p1/*_summary.py` | 1 | Pooled endpoints with the paired bootstrap |
| `scripts/p2_ot_*` | 1 | Coupling interventions on the OT router |
| `scripts/p2_mma_*` | 1 | MMA-DFER's published exchange as a known-answer case; offsets on the OT router |
| `scripts/stage0a_*`, `scripts/s0b_*` | 1 | Repair screen: within-clip spread, linear probes, three-configuration adapter screen |
| `scripts/e2_ot.py`, `p1/e2_hicmae.py`, `scripts/e2_summary.py` | 1 | Cost-space spread, cost interaction and its bound at every routing block and every HiCMAE fusion map (E2) |
| `scripts/blocksub_ot.py`, `scripts/blocksub_summary.py` | 1 | Product replacement in block subsets (1-9, 10-12, all); also runs the arms of the head attenuation control |
| `scripts/e3_train.py`, `scripts/e345_summary.py` | 1 | Patch-mean cost descriptor (A1), temperature sweep and training under product coupling (A4) (E3-E5) |
| `scripts/e6_costdev.py`, `scripts/e6_train.py`, `scripts/e6_smoke.py`, `scripts/e6_summary.py` | 1 | Learned and random centred deviation of the cost descriptor (A2, A3) (E6) |
| `scripts/exactprod_summary.py`, `scripts/headctl_summary.py` | 1 | Exact product reference with WAR and log-probability endpoints; head attenuation control |
| `scripts/dump_plans.py`, `scripts/p6_stage0.py` | 1 | Example transport plans for the figures; positive control with composite clips |
| `scripts/p2t_mma.py`, `p1/p2t_hicmae.py` | 2 | Test-time temporal arms for MMA-DFER / OT router and for HiCMAE |
| `scripts/p2t_summary.py`, `scripts/p2t_breakdown.py` | 2 | Pooled readings; per-class and duration breakdowns |
| `scripts/p2t_train_shuffle.py`, `scripts/p2t_train_compare.py` | 2 | Order-agnostic training control (shuffles frames in the training loader only) and its comparison |
| `p1/run_*.sh`, `scripts/run_*.sh` | both | Queue scripts, one per stage |
| `openface/` | 1 | Build dlib and OpenFace 2.2.0 without root; CREMA-D face alignment |
| `p1/download_hicmae*.sh`, `p1/make_mafw_splits.py` | 1 | Released HiCMAE checkpoints; MAFW splits |
| `report-assets/make_assets*.py`, `report-assets/make_fig_*.py`, `report-assets/make_supp_tables.py` | both | Build every figure, table and number macro from the result summaries |
| `results/`, `p1/results/` | both | Result summaries (JSON, TXT) per stage, fold and arm |

**Result folders.**
- Part 1: `p1/results/cremad_of/` (HiCMAE on OpenFace-aligned faces, the primary HiCMAE result; the older
  `p1/results/cremad*/` folders use the model's own face crops and are superseded), `p1/results/mafw/`,
  `results/p2_ot/`, `results/p2_mma/`, `results/stage0a/`, `results/s0b/`, and for the cost-input analysis
  `results/e2/`, `results/blocksub/`, `results/e345/`, `results/e6/`, `results/exactprod/`, `results/headctl/`,
  `results/p6/`, `results/plandump/`.
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
  Comparisons between separately trained router models use a paired-fold t interval over the 5 folds (90%).
