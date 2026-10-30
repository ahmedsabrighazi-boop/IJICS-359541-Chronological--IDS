# IJICS-359541 — Original and reviewer experiment source code

Revised manuscript: **Chronological Reliability of Lightweight Network Intrusion Detection: Eligibility, Feature-Selection Stability, and Threshold Transport**.

Two folders only:

- `original_study/` contains the historical baseline study source embedded in the submitted unified archive.
- `reviewer_experiments/` contains the historical reviewer experiment scripts as embedded in that same archive, plus the `frozen_distributed_source.py` source variant required by some audits.

The 11 Python files were extracted **verbatim from the strings inside** `IJICS_359541_ALL_STUDY_AND_REVIEWER_EXPERIMENTS_ONE_FILE(1).py`; no scientific script was rewritten. Existing reviewer scripts may import the original study by its original basename. Dependencies, historical V5 records and the `NF-UNSW-NB15-v3.csv` dataset are separate; not all reviewer runs can execute without them.

## Requirements

Python 3.10 was used in recorded runs. Install the packages listed in `requirements.txt`. Version-specific results may vary with the exact environment, especially scikit-learn; this archive does not independently prove every historical installed version.

The expected dataset SHA-256 is:

`4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5`

The dataset is **not** included in this source-code repository. Obtain the original dataset externally and verify its hash before executing any scientific run.

## Example: original study

The study script contains `CFG.DATA_PATHS` and `CFG.OUTPUT_DIR` configuration values. Place the dataset at the configured location or update its path locally; do not change the archived reference copy if preserving code provenance.

```bash
python -m pip install -r requirements.txt
python original_study/nf_unsw_nb15_v3_temporal_reliability_18W_DUAL_FINAL_READY_BOOTSTRAP_FIX.py
```

## Reviewer experiments

The reviewer experiment source files are separated and keep their original filenames and code. Some assume they share a directory with the original source file or require command-line parameters; in a working environment, supply the paths and imports documented at the top of each original script. If a script has a historical SHA-256 guard, a guard failure must be investigated—not bypassed by editing that original script.

Examples of source entry points: `NF_UNSW_NB15_v3_Revision_R1_Runs.py` (R1), `IJICS_359541_R2_Additional_Experiments.py` (R2), and `IJICS_Remaining4_20261024.py` (bounded follow-up tasks). Other source files implement read-only and saved-score audits. Some reviewer comments require interpretation rather than a separate training run.

## Code integrity

This ZIP establishes **exact extraction from the uploaded combined source**, not exact historical provenance for all 11 modules. Declared historical fingerprints differ from extracted UTF-8 fingerprints for six modules. The digest record below is included to avoid claiming exact historical-source equality without the old execution archives.

| Source file | Folder | SHA-256 of extracted UTF-8 code | Matches declared historical SHA-256 |
|---|---|---|---|
| `nf_unsw_nb15_v3_temporal_reliability_18W_DUAL_FINAL_READY_BOOTSTRAP_FIX.py` | `original_study` | `7f7cc64f63a31959e2d9ab478fe4a24e608d269b5e48113ef95c22a3c44b6493` | No |
| `frozen_distributed_source.py` | `reviewer_experiments` | `4c63e9ebc4b371a861c41a8bc4c12b2ef120934afa626f4652abb05e4d71359e` | No |
| `NF_UNSW_NB15_v3_Revision_R1_Runs.py` | `reviewer_experiments` | `a00a77ce7de52e4324345e3d80789002b5eb64984818fe12ef83168a19c49770` | Yes |
| `NF_UNSW_NB15_v3_Computational_Cost_Addendum_RUN_ONLY.py` | `reviewer_experiments` | `64ab3877e1c1ffc8293ed809a7415ff4729aaf02ab7c81919f770469a9ee77a2` | No |
| `IJICS_359541_R1_11_R2_9_ReadOnly_Audits.py` | `reviewer_experiments` | `58c495813610b3c024b75c574d5b1b23320fb659be68a49151e1f948052593f6` | No |
| `IJICS_Remaining4_20261024.py` | `reviewer_experiments` | `46a965668020822260cc1ad12020f1a87ade8e640751c20dd983ab4c0a38ef1f` | Yes |
| `IJICS_359541_R1_5_Purged_Score_Replay_V2.py` | `reviewer_experiments` | `77e7f201024ee273b1e1220ba2fe85066198f0a84b3f60fbac1280d448f3c816` | No |
| `IJICS_359541_R1_5_FlowEnd_ReadOnly_Audit.py` | `reviewer_experiments` | `747c09c36c1ade0a7c33e8377c0a8ea52cd12d2a6556281b4d675bcbff999764` | No |
| `IJICS_359541_E6_Calibration_Knot_Forensics_V1.py` | `reviewer_experiments` | `5bf2fb07b700367471980f15bc29bdd0a557ca352154bb4b8f6fc1063cea4631` | Yes |
| `IJICS_359541_Legacy_Threshold_Diagnostics.py` | `reviewer_experiments` | `77554d249cdb02744a947ed2440c3e1afac2eb5ff6c452fd76a9638ab1188b44` | Yes |
| `IJICS_359541_R2_Additional_Experiments.py` | `reviewer_experiments` | `23033e9dab50e6f4e5411e25dff2b9dd7ab08893c5c8b5930ef2d62b54732733` | Yes |

No original experiment was retrained or fully replayed while creating this ZIP. For rigorous R1-12 verification, compare the saved historical execution logs, full metric CSV files, and per-fold test outputs with the manuscript; this source archive alone is not an end-to-end reproduction claim.
