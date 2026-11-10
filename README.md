# dfer-fusion-time-audit

Do audio-visual dynamic facial expression recognition (DFER) models use what they are built to use?
This repository holds the code, pre-registration log, result summaries, reports and paper drafts for two parts:

- **Part 1, fusion.** Does a cross-modal fusion module get used, and how: through a clip-level summary of
  the other stream, or through which token receives which message? This is an inference-only audit
  (marginal-preserving independent coupling, row permutation, removal, offsets with a content-matched control)
  on HiCMAE, MMA-DFER and an optimal-transport (OT) router. It is the ESWA draft in `paper/`.
- **Part 2, time.** Do the models use the temporal axis of the video? This part applies frame shuffles,
  reversal and frame budgets at test time, and adds an order-agnostic training control for MMA-DFER.

Every stage was written into [`prereg.md`](prereg.md) before it was run: the question, the arms, the
endpoints, the decision rule, and then the result.

## Layout

| Path | Part | Contents |
|---|---|---|
| `prereg.md` | both | Pre-registration log: design, rules and results of every stage, with corrections |
| `p1/` | 1 (and 2) | HiCMAE scripts: reproduction (`p1_eval.py`), null ladder (`p1_nulls.py`), space/time and offsets (`p1_followup.py`), content-matched desync control (`p1_desync_control.py`), shared configs (`p1_common.py`). `p2t_hicmae.py` is the Part 2 runner for HiCMAE. |
| `p1/results/` | 1 | HiCMAE result summaries: `cremad_of/` = reproduced checkpoints on OpenFace faces (primary); `mafw/` = MAFW (not reproduced); `cremad*` without `_of` = the first run on our own face crops |
| `scripts/p2_ot_*`, `scripts/p2_mma_*` | 1 | Null ladder on the OT router; MMA-DFER's published exchange as a known-answer case |
| `scripts/stage0a_*`, `scripts/s0b_*` | 1 | Repair screen: within-clip spread (0a), linear probes (0a-bis), adapter screen (0b) |
| `scripts/p2t_*`, `scripts/run_p2t*.sh` | 2 | Test-time temporal arms, breakdowns, and the shuffle-trained control with its comparison |
| `openface/` | 1 | Build scripts for dlib and OpenFace 2.2.0 without root, and CREMA-D face alignment |
| `results/` | both | Summaries (JSON, CSV, logs): `stage0a`, `s0b`, `p2_ot`, `p2_mma` (Part 1); `p2t`, `p2t_train` (Part 2) |
| `report-assets/` | both | Scripts that build every figure, table and number macro from the result files (`make_assets*.py`), and the Vietnamese reports to the advisor (`p2/`, `p2t/`) |
| `paper/` | 1 | English paper draft (elsarticle), supplementary, cover letter, highlights, preview PDFs |
| `notes/` | 2 | Prior-art check for Part 2 |

**Naming.** In file names, `p2_*` is the final protocol stage of Part 1 (OpenFace reproduction, MAFW, OT router).
Part 2 files carry `p2t_*`.

## Not in this repository

- Datasets (CREMA-D, DFEW, MAFW), face crops and the feature cache.
- Checkpoints: the released HiCMAE checkpoints and the group's MMA-DFER / OT-router runs.
- Per-clip predictions (`*.npz`, about 300 MB). The bootstrap statistics are recomputed from these; they stay on the server.
- Third-party code, used unchanged: HiCMAE (github.com/sunlicai/HiCMAE, commit `cdd28d4`), OpenFace 2.2.0, dlib 19.24.6.
  The MMA-DFER and OT-router code is the group's AV-DFER-OT repository, imported read-only.

## Running

The scripts run on a single RTX 3090 server and use absolute server paths.
- HiCMAE needs Python 3.8 and torch 1.10.1+cu113 (`p1/setup_env.sh`, `p1/requirements-lock.txt`). On Debian 13 it also needs `GLIBC_TUNABLES=glibc.rtld.execstack=2`.
- MMA-DFER and the OT router use the AV-DFER-OT environment.
- Each stage has a `run_*.sh` queue script. Its stage section in `prereg.md` gives the exact arms and rules.
