#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IJICS-359541: single-entry-point temporal IDS and reviewer experiments.

Run:
  python IJICS_359541_ALL_IN_ONE.py --data NF-UNSW-NB15-v3.csv --out IJICS_359541_results

The original study, original reviewer-1 extension, synchronized time addendum,
follow-up experiments, temporal split verification and reviewer analyses are
packaged into this one distributable Python file. All source is unpacked into
an automatically managed private runtime directory for the duration of execution.
The raw dataset is external to the program and is checked by SHA-256.
"""
from __future__ import annotations
import argparse
import base64
import contextlib
import csv
import dataclasses
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import types
import zlib
from pathlib import Path

RAW_DATA_SHA256 = "4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5"
BASE_SOURCE_SHA256 = "5f51d87016d66d99a41f17eb65e1919635502502b7826e2ddaadaedb766e4b33"
STAGE_ORDER = (
    'splits', 'baseline', 'r1', 'cost', 'r2-2-matched', 'r2-1', 'r2-5', 'r2-7', 'r2-8',
    'r1-11-r2-9-audit', 'remaining-plan', 'r1-6', 'r1-11', 'r2-3-15',
    'r2-3-46', 'r2-9',
)

SOURCE_TEXT = {
    'baseline': r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chronological, leakage-resistant temporal IDS evaluation on NF-UNSW-NB15-v3.

Final design implemented here:
- 18 equal-count chronological windows for both audit and full study.
- Two future-evaluation protocols: limited-memory sliding and expanding-history.
- Shortcut-Exposed vs Strict-Clean sensitivity arms, with TTL ablations.
- Train-only preprocessing and RFE with fixed estimator randomness across folds.
- Capped-positive RFE sampling (15% cap) for the full study, preventing extreme
  class-prevalence distortion in late expanding folds.
- Preflight RFE feature-selection stability analysis: 4 base variants x stress/typical folds x
  repeated sample seeds, with the RFE forest seed held fixed.
- Sampling-prevalence sensitivity on the typical fold: natural-uniform,
  capped-positive, and positive-complete.
- Stable-core statistics within seed replicates and across temporal folds.
- Separate calibration and historical threshold-support reserves from past data.
- Raw threshold-window failure is preserved; an additional
  FPR-budget-with-historical-pool rule supplies a past-only operational remedy.
- Eligibility tiers are exported explicitly so averages never silently mix
  incompatible fold sets.
- Sliding and expanding stability are reported as descriptive references; no
  standard significance tests are applied to dependent pairwise Jaccard values.
- Wilcoxon outputs report the effective sample size after zero differences are
  removed.
- 2,000-round bootstrap confidence intervals use an efficient multinomial
  confusion-cell bootstrap and do not retrain any model.
- Older 12-window/uniform outputs are archived away from final results.

Workflow:
1) Run with EXECUTION_MODE="preflight" and review the RFE audit/sensitivity files.
2) Set EXECUTION_MODE="full" and run the complete 18-window dual-protocol study.
"""
from __future__ import annotations

import gc
import hashlib
import glob
import json
import os
import re
import shutil
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    matthews_corrcoef, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# USER CONFIGURATION
# =============================================================================
@dataclass
class Config:
    DATA_PATHS: Tuple[str, ...] = (
        r"NF-UNSW-NB15-v3.csv",
    )
    OUTPUT_DIR: str = r"NF_UNSW_NB15_v3_18W_DualProtocol_Final_Results"

    # Run preflight once, inspect its outputs, then change to "full".
    EXECUTION_MODE: str = "full"  # preflight | full | preflight_then_full

    BINARY_LABEL_COLUMN: Optional[str] = None
    ATTACK_CATEGORY_COLUMN: Optional[str] = None
    TIME_COLUMN: Optional[str] = None

    MAX_ROWS_PER_FILE: Optional[int] = None
    CSV_LOW_MEMORY: bool = False
    RANDOM_STATE: int = 42

    STRICT_DROP_FEATURES: Tuple[str, ...] = (
        "FLOW_END_MILLISECONDS",
        "L4_SRC_PORT",
        "L4_DST_PORT",
        "DNS_QUERY_ID",
    )
    BASE_VARIANT_ORDER: Tuple[str, ...] = (
        "Clean-All",
        "No-TTL",
        "Strict-Clean",
        "Strict-Clean-No-TTL",
    )
    DROP_PORTS_AS_IDENTIFIERS: bool = False
    DROP_GLOBAL_EXACT_DUPLICATES: bool = True
    ENFORCE_TEMPORAL_DUPLICATE_ISOLATION: bool = False

    # Separate past-only support reserves. They are both excluded from model fit.
    CALIBRATION_RESERVE_FRACTION_PER_TRAIN_WINDOW: float = 0.05
    THRESHOLD_RESERVE_FRACTION_PER_TRAIN_WINDOW: float = 0.025
    CALIBRATION_FALLBACK_PER_CLASS: int = 2000
    THRESHOLD_FALLBACK_PER_CLASS: int = 2000
    MIN_TRAIN_SAMPLES_PER_CLASS_AFTER_RESERVE: int = 1

    TOP_K_FEATURES: int = 15
    RFE_MAX_SAMPLES: int = 100_000
    RFE_N_ESTIMATORS: int = 120
    RFE_STEP: float = 0.20

    # Full-study RFE sample composition.
    FULL_RFE_SAMPLING_MODE: str = "capped_positive"
    AUDIT_RFE_SAMPLING_MODE: str = "capped_positive"
    RFE_POSITIVE_CAP_FRACTION: float = 0.15
    RFE_MIN_OTHER_CLASS_SAMPLES: int = 2_000
    RFE_FIXED_SAMPLE_SEED: int = 14_042
    RFE_FIXED_MODEL_SEED: int = 24_042
    AUDIT_SAMPLE_SEEDS: Tuple[int, ...] = (1103, 2207, 3301, 4409, 5519)

    # Audit and full study use the same temporal granularity.
    AUDIT_N_TEMPORAL_WINDOWS: int = 18
    N_TEMPORAL_WINDOWS: int = 18
    ROLLING_TRAIN_WINDOWS: int = 3
    EVALUATION_PROTOCOLS: Tuple[str, ...] = ("sliding", "expanding")

    # Audit all four base variants on an automatically selected stress fold and
    # typical fold. The sampling-prevalence sensitivity is run on Strict-Clean.
    RFE_NOISE_AUDIT_BASE_VARIANTS: Tuple[str, ...] = (
        "Clean-All",
        "No-TTL",
        "Strict-Clean",
        "Strict-Clean-No-TTL",
    )
    RFE_NOISE_REPEATS: int = 5
    RFE_SENSITIVITY_BASE_VARIANT: str = "Strict-Clean"
    RFE_SENSITIVITY_MODES: Tuple[str, ...] = (
        "natural_uniform",
        "capped_positive",
        "positive_complete",
    )

    SINGLE_TEMPORAL_TRAIN_FRACTION: float = 0.70
    SINGLE_TEMPORAL_CALIBRATION_FRACTION: float = 0.10
    SINGLE_TEMPORAL_THRESHOLD_FRACTION: float = 0.10

    RUN_MODELS: Tuple[str, ...] = ("RF", "LR", "LinearSVM")
    RF_N_ESTIMATORS: int = 300
    RF_MAX_DEPTH: Optional[int] = None
    RF_MIN_SAMPLES_LEAF: int = 1
    LR_MAX_ITER: int = 2500

    MU_SIGMA_K_VALUES: Tuple[float, ...] = (0.05, 0.10, 0.20)
    FPR_BUDGETS: Tuple[float, ...] = (0.005, 0.010, 0.020)

    BOOTSTRAP_ROUNDS: int = 2000
    BOOTSTRAP_CONFIDENCE: float = 0.95

    RUN_RANDOM_BASELINE: bool = True
    RUN_SINGLE_TEMPORAL: bool = True
    RUN_TEMPORAL_PROTOCOLS: bool = True
    RUN_RFE_VARIANTS: bool = True
    SAVE_DETAILED_PLOTS: bool = True
    SAVE_MAIN_PREDICTIONS: bool = False

    MAIN_VARIANT: str = "Strict-Clean-RFE-TopK"
    MAIN_MODEL: str = "RF"
    MAIN_PROBABILITY_SOURCE: str = "isotonic"
    MAIN_THRESHOLD_NAME: str = "fixed_0.5"

    # Keep superseded 12-window/uniform outputs out of the final directory.
    ARCHIVE_SUPERSEDED_OUTPUTS: bool = True
    SUPERSEDED_ARCHIVE_NAME: str = "superseded_12window_uniform"
    LEGACY_OUTPUT_DIRS: Tuple[str, ...] = (
        "NF_UNSW_NB15_v3_RFE_Noise_Aware_Final_Results",
    )

CFG = Config()

PAPER_VARIANT_LABELS: Dict[str, str] = {
    "Clean-All": "Shortcut-Exposed",
    "RFE-TopK": "Shortcut-Exposed RFE-TopK",
    "No-TTL": "Shortcut-Exposed No-TTL",
    "No-TTL-RFE-TopK": "Shortcut-Exposed No-TTL RFE-TopK",
    "Strict-Clean": "Strict-Clean Baseline",
    "Strict-Clean-RFE-TopK": "Proposed Lightweight Strict-Clean",
    "Strict-Clean-No-TTL": "Strict-Clean No-TTL Ablation",
    "Strict-Clean-No-TTL-RFE-TopK": "Strict-Clean No-TTL RFE-TopK",
}

NORMAL_TOKENS = {
    "0", "normal", "benign", "non-attack", "nonattack", "legitimate",
    "background", "safe", "no attack", "no_attack",
}
LABEL_CANDIDATES = (
    "label", "binary_label", "binarylabel", "attack_label", "attacklabel",
    "is_attack", "isattack", "class", "target",
)
ATTACK_CANDIDATES = (
    "attack", "attack_cat", "attack_category", "attackcategory", "category",
    "attack_type", "attacktype", "traffic_type", "traffictype",
)
TIME_CANDIDATES = (
    "timestamp", "time", "stime", "ltime", "starttime", "endtime",
    "flow_start_milliseconds", "flow_end_milliseconds",
    "flow_start_seconds", "flow_end_seconds", "first_seen", "last_seen",
    "date", "datetime",
)
DIRECT_ID_PATTERNS = (
    r"(^|_)flow(_|$).*id", r"(^|_)record(_|$).*id",
    r"(^|_)session(_|$).*id", r"(^|_)src(_|$).*ip",
    r"(^|_)dst(_|$).*ip", r"ipv4_src_addr", r"ipv4_dst_addr",
    r"ipv6_src_addr", r"ipv6_dst_addr", r"src_addr", r"dst_addr",
    r"source_ip", r"destination_ip", r"mac(_|$)",
)
PORT_PATTERNS = (
    r"(^|_)src(_|$).*port", r"(^|_)dst(_|$).*port",
    r"l4_src_port", r"l4_dst_port", r"source_port", r"destination_port",
)
TTL_PATTERNS = (
    r"(^|_)ttl($|_)", r"min_ttl", r"max_ttl", r"src_ttl", r"dst_ttl",
    r"sttl", r"dttl",
)

# =============================================================================
# UTILITIES
# =============================================================================
def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")[:180]


def ensure_dirs(root: Path) -> Dict[str, Path]:
    dirs = {k: root / k for k in ("csv", "plots", "predictions", "logs")}
    dirs["root"] = root
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def archive_superseded_outputs(cfg: Config, new_root: Path) -> List[str]:
    """Move known legacy output directories under a clearly superseded archive.

    This prevents 12-window/uniform-sampling stability numbers from being mixed
    with the final 18-window capped-positive results. The move is conservative:
    only explicitly named legacy directories are touched.
    """
    moved: List[str] = []
    if not cfg.ARCHIVE_SUPERSEDED_OUTPUTS:
        return moved
    archive_root = new_root / cfg.SUPERSEDED_ARCHIVE_NAME
    archive_root.mkdir(parents=True, exist_ok=True)
    parent = new_root.parent
    for legacy_name in cfg.LEGACY_OUTPUT_DIRS:
        legacy = (parent / legacy_name).resolve()
        if legacy == new_root or not legacy.exists():
            continue
        target = archive_root / legacy.name
        if target.exists():
            stamp = time.strftime("%Y%m%d_%H%M%S")
            target = archive_root / f"{legacy.name}_{stamp}"
        shutil.move(str(legacy), str(target))
        moved.append(str(target))
    return moved


def write_json(data: Any, path: Path) -> None:
    def conv(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, Path): return str(obj)
        return obj
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=conv)


def log(msg: str, path: Path) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def resolve_paths(patterns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches: out.extend(matches)
        elif Path(pattern).is_file(): out.append(pattern)
    out = list(dict.fromkeys(out))
    if not out:
        raise FileNotFoundError("No CSV files found. Edit CFG.DATA_PATHS.")
    return out


def load_dataset(cfg: Config, log_path: Path) -> pd.DataFrame:
    frames = []
    raw_columns: List[str] = []
    for file_order, path in enumerate(resolve_paths(cfg.DATA_PATHS)):
        log(f"Loading {path}", log_path)
        kwargs: Dict[str, Any] = {"low_memory": cfg.CSV_LOW_MEMORY}
        if cfg.MAX_ROWS_PER_FILE is not None:
            kwargs["nrows"] = cfg.MAX_ROWS_PER_FILE
        try:
            part = pd.read_csv(path, **kwargs)
        except UnicodeDecodeError:
            part = pd.read_csv(path, encoding="latin-1", **kwargs)

        # Preserve the true raw schema before adding helper order columns.
        for column in part.columns:
            if column not in raw_columns:
                raw_columns.append(column)

        part["__source_file_order__"] = file_order
        part["__source_row_order__"] = np.arange(len(part), dtype=np.int64)
        frames.append(part)
        log(f"Loaded {len(part):,} rows x {part.shape[1]:,} columns", log_path)

    df = pd.concat(frames, ignore_index=True, sort=False)
    del frames
    gc.collect()

    if cfg.DROP_GLOBAL_EXACT_DUPLICATES:
        before = len(df)
        # Remove only rows that are exactly identical in the original CSV fields.
        # Helper source-order columns must not participate in duplicate detection.
        duplicate_subset = [column for column in raw_columns if column in df.columns]
        df = df.drop_duplicates(subset=duplicate_subset, keep="first").reset_index(drop=True)
        log(
            f"Removed {before-len(df):,} globally exact duplicate rows "
            f"using {len(duplicate_subset)} original columns",
            log_path,
        )

    log(f"Combined shape: {df.shape}", log_path)
    return df

# =============================================================================
# COLUMN DETECTION, LABELS, ORDERING
# =============================================================================
def find_column(columns: Sequence[str], explicit: Optional[str], candidates: Sequence[str]) -> Optional[str]:
    if explicit is not None:
        if explicit not in columns:
            raise KeyError(f"Configured column '{explicit}' not found")
        return explicit
    normalized = {normalize_name(c): c for c in columns}
    for candidate in candidates:
        if normalize_name(candidate) in normalized:
            return normalized[normalize_name(candidate)]
    for col in columns:
        n = normalize_name(col)
        for candidate in candidates:
            c = normalize_name(candidate)
            if n.startswith(c) or n.endswith(c):
                return col
    return None


def detect_columns(df: pd.DataFrame, cfg: Config) -> Dict[str, Optional[str]]:
    binary = find_column(df.columns, cfg.BINARY_LABEL_COLUMN, LABEL_CANDIDATES)
    attack = find_column(df.columns, cfg.ATTACK_CATEGORY_COLUMN, ATTACK_CANDIDATES)
    time_col = find_column(df.columns, cfg.TIME_COLUMN, TIME_CANDIDATES)
    if binary is not None and df[binary].nunique(dropna=True) > 2 and attack is None:
        attack, binary = binary, None
    if binary is None and attack is None:
        raise ValueError("No label column detected. Set label columns in Config.")
    return {"binary_label": binary, "attack_category": attack, "time": time_col}


def map_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        unique = set(pd.Series(numeric).dropna().unique().tolist())
        if unique.issubset({0, 1}):
            return numeric.fillna(0).astype(np.int8)
        minimum = min(unique)
        return (numeric != minimum).astype(np.int8)
    text = series.astype(str).str.strip().str.lower()
    return (~text.isin(NORMAL_TOKENS)).astype(np.int8)


def derive_labels(df: pd.DataFrame, detected: Mapping[str, Optional[str]]) -> pd.DataFrame:
    out = df.copy()
    bcol, acol = detected["binary_label"], detected["attack_category"]
    source = out[bcol] if bcol is not None else out[acol]
    out["__binary_label__"] = map_binary(source)
    if acol is not None:
        category = out[acol].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"})
        category.loc[out["__binary_label__"].eq(0)] = "Normal"
        out["__attack_category__"] = category
    else:
        out["__attack_category__"] = np.where(out["__binary_label__"].eq(0), "Normal", "Attack")
    return out


def prepare_order(df: pd.DataFrame, time_col: Optional[str], log_path: Path) -> Tuple[pd.DataFrame, str]:
    out = df.copy()
    if time_col is None:
        out = out.sort_values(["__source_file_order__", "__source_row_order__"], kind="mergesort").reset_index(drop=True)
        log("No time column detected; using source file/row order", log_path)
        return out, "file_and_row_order"
    raw = out[time_col]
    if pd.api.types.is_numeric_dtype(raw):
        numeric = pd.to_numeric(raw, errors="coerce")
        median_abs = float(np.nanmedian(np.abs(numeric.to_numpy(dtype=float))))
        if median_abs > 1e14: parsed = pd.to_datetime(numeric, unit="us", errors="coerce")
        elif median_abs > 1e11: parsed = pd.to_datetime(numeric, unit="ms", errors="coerce")
        elif median_abs > 1e9: parsed = pd.to_datetime(numeric, unit="s", errors="coerce")
        else: parsed = numeric
    else:
        parsed = pd.to_datetime(raw, errors="coerce", utc=True)
    out["__time_order__"] = parsed
    valid = out["__time_order__"].notna().mean()
    if valid < 0.50:
        out = out.sort_values(["__source_file_order__", "__source_row_order__"], kind="mergesort").reset_index(drop=True)
        log(f"Time parsing only {valid:.1%}; using source order", log_path)
        return out, "file_and_row_order"
    out = out.sort_values(["__time_order__", "__source_file_order__", "__source_row_order__"], kind="mergesort", na_position="last").reset_index(drop=True)
    log(f"Chronological order uses {time_col} ({valid:.1%} parsed)", log_path)
    return out, f"time_column:{time_col}"

# =============================================================================
# LEAKAGE AUDIT AND SPLITS
# =============================================================================
def matches(name: str, patterns: Sequence[str]) -> bool:
    n = normalize_name(name)
    return any(re.search(p, n, flags=re.I) for p in patterns)


def leakage_audit(
    df: pd.DataFrame,
    detected: Mapping[str, Optional[str]],
    cfg: Config,
) -> Tuple[List[str], List[str], List[str], pd.DataFrame]:
    """Create the current clean set and flag stricter shortcut removals.

    Clean-All keeps the requested strict-audit fields so their effect can be
    measured directly. Strict-Clean removes them later when the four feature
    variants are constructed.
    """
    excluded = {
        "__binary_label__", "__attack_category__", "__source_file_order__",
        "__source_row_order__", "__time_order__",
    }
    excluded.update(c for c in detected.values() if c is not None)
    strict_norm = {normalize_name(c) for c in cfg.STRICT_DROP_FEATURES}

    clean: List[str] = []
    ttl: List[str] = []
    strict_present: List[str] = []
    rows: List[Dict[str, Any]] = []

    for col in df.columns:
        if col in excluded:
            rows.append({
                "feature": col, "action": "exclude",
                "reason": "label/order/metadata", "ttl": False,
                "strict_shortcut_candidate": False,
            })
            continue
        if matches(col, DIRECT_ID_PATTERNS):
            rows.append({
                "feature": col, "action": "drop",
                "reason": "direct identifier", "ttl": False,
                "strict_shortcut_candidate": False,
            })
            continue
        if cfg.DROP_PORTS_AS_IDENTIFIERS and matches(col, PORT_PATTERNS):
            rows.append({
                "feature": col, "action": "drop",
                "reason": "endpoint port", "ttl": False,
                "strict_shortcut_candidate": True,
            })
            continue
        if df[col].nunique(dropna=False) <= 1:
            rows.append({
                "feature": col, "action": "drop",
                "reason": "constant", "ttl": False,
                "strict_shortcut_candidate": False,
            })
            continue
        if df[col].isna().all():
            rows.append({
                "feature": col, "action": "drop",
                "reason": "all missing", "ttl": False,
                "strict_shortcut_candidate": False,
            })
            continue

        clean.append(col)
        is_ttl = matches(col, TTL_PATTERNS)
        is_strict = normalize_name(col) in strict_norm
        if is_ttl:
            ttl.append(col)
        if is_strict:
            strict_present.append(col)

        if is_strict:
            action = "keep_in_current_drop_in_strict"
            reason = "absolute-time_or_endpoint_shortcut_candidate"
        elif is_ttl:
            action = "keep_for_clean_all"
            reason = "TTL feature"
        else:
            action = "keep"
            reason = "eligible predictor"

        rows.append({
            "feature": col,
            "action": action,
            "reason": reason,
            "ttl": is_ttl,
            "strict_shortcut_candidate": is_strict,
        })

    return clean, ttl, strict_present, pd.DataFrame(rows)


def feature_hash(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    return pd.util.hash_pandas_object(df.loc[:, cols], index=False).astype("uint64")


def remove_temporal_duplicates(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    threshold: pd.DataFrame,
    test: pd.DataFrame,
    cols: Sequence[str],
    enabled: bool,
):
    """Remove exact predictor duplicates sequentially without looking forward.

    A record is retained only at its first occurrence among training,
    calibration, threshold-selection, and test partitions. The label is not part
    of the hash, preventing identical predictor vectors from crossing partitions.
    """
    zero_stats = {
        "train_duplicates_removed": 0,
        "calibration_duplicates_removed": 0,
        "threshold_duplicates_removed": 0,
        "test_duplicates_removed": 0,
    }
    if not enabled:
        return train, calibration, threshold, test, zero_stats

    ht = feature_hash(train, cols)
    keep_t = ~ht.duplicated(keep="first")
    train2 = train.loc[keep_t].copy()
    seen = set(ht.loc[keep_t].tolist())

    hc = feature_hash(calibration, cols)
    keep_c = (~hc.duplicated(keep="first")) & (~hc.isin(seen))
    calibration2 = calibration.loc[keep_c].copy()
    seen.update(hc.loc[keep_c].tolist())

    hv = feature_hash(threshold, cols)
    keep_v = (~hv.duplicated(keep="first")) & (~hv.isin(seen))
    threshold2 = threshold.loc[keep_v].copy()
    seen.update(hv.loc[keep_v].tolist())

    hs = feature_hash(test, cols)
    keep_s = (~hs.duplicated(keep="first")) & (~hs.isin(seen))
    test2 = test.loc[keep_s].copy()

    return train2, calibration2, threshold2, test2, {
        "train_duplicates_removed": int((~keep_t).sum()),
        "calibration_duplicates_removed": int((~keep_c).sum()),
        "threshold_duplicates_removed": int((~keep_v).sum()),
        "test_duplicates_removed": int((~keep_s).sum()),
    }

@dataclass
class DataSplit:
    protocol: str
    fold_id: str
    train_idx: np.ndarray
    calibration_idx: np.ndarray
    threshold_idx: np.ndarray
    threshold_pool_idx: np.ndarray
    test_idx: np.ndarray
    calibration_strategy: str = "dedicated_calibration"
    threshold_pool_strategy: str = "raw_threshold_only"
    calibration_reserve_rows: int = 0
    threshold_reserve_rows: int = 0
    train_window_ids: Tuple[int, ...] = ()
    calibration_window_id: Optional[int] = None
    threshold_window_id: Optional[int] = None
    test_window_id: Optional[int] = None
    candidate_number: Optional[int] = None
    training_mode: str = "non_temporal"


def _concat_unique_sorted(parts: Sequence[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(part, dtype=np.int64) for part in parts if len(part)]
    if not valid:
        return np.asarray([], dtype=np.int64)
    return np.sort(np.unique(np.concatenate(valid)))


def _class_count(df: pd.DataFrame, indices: np.ndarray) -> int:
    if len(indices) == 0:
        return 0
    return int(df.iloc[indices]["__binary_label__"].nunique())


def _move_missing_classes_from_train(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    per_class_limit: int,
    min_train_per_class: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move recent past samples of missing classes from training into a reserve."""
    if _class_count(df, target_idx) >= 2 or _class_count(df, train_idx) < 2:
        return train_idx, target_idx, np.asarray([], dtype=np.int64)

    target_labels = set(
        df.iloc[target_idx]["__binary_label__"].astype(int).unique().tolist()
    )
    missing = [label for label in (0, 1) if label not in target_labels]
    train_labels = df.iloc[train_idx]["__binary_label__"].to_numpy(dtype=np.int8)
    moved_parts: List[np.ndarray] = []
    for label in missing:
        candidates = train_idx[train_labels == label]
        available = max(0, len(candidates) - int(min_train_per_class))
        take_n = min(int(per_class_limit), available)
        if take_n > 0:
            moved_parts.append(candidates[-take_n:])
    moved = _concat_unique_sorted(moved_parts)
    if len(moved):
        train_idx = train_idx[~np.isin(train_idx, moved)]
        target_idx = _concat_unique_sorted([target_idx, moved])
    return train_idx, target_idx, moved


def _build_past_only_support_pools(
    df: pd.DataFrame,
    train_windows: Sequence[np.ndarray],
    dedicated_calibration_idx: np.ndarray,
    dedicated_threshold_idx: np.ndarray,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, int, int]:
    """Build disjoint train, calibration, and threshold-support pools.

    The model is fit only on ``train_idx``. A chronological calibration reserve
    and a smaller, separate historical threshold reserve are withheld from every
    training window. The threshold support pool is used only by the explicitly
    named historical-pool FPR rule; the raw threshold window remains untouched so
    its single-class failure mode is still visible.
    """
    cal_fraction = float(cfg.CALIBRATION_RESERVE_FRACTION_PER_TRAIN_WINDOW)
    thr_fraction = float(cfg.THRESHOLD_RESERVE_FRACTION_PER_TRAIN_WINDOW)
    if cal_fraction < 0 or thr_fraction < 0 or cal_fraction + thr_fraction >= 0.5:
        raise ValueError(
            "Calibration + threshold reserve fractions must be non-negative and sum to < 0.5."
        )

    train_parts: List[np.ndarray] = []
    cal_reserve_parts: List[np.ndarray] = []
    thr_reserve_parts: List[np.ndarray] = []

    for window in train_windows:
        window = np.asarray(window, dtype=np.int64)
        if len(window) < 8:
            train_parts.append(window)
            continue
        cal_n = int(round(len(window) * cal_fraction)) if cal_fraction > 0 else 0
        thr_n = int(round(len(window) * thr_fraction)) if thr_fraction > 0 else 0
        cal_n = max(1, cal_n) if cal_fraction > 0 else 0
        thr_n = max(1, thr_n) if thr_fraction > 0 else 0
        max_reserve = max(0, len(window) - 2)
        if cal_n + thr_n > max_reserve:
            scale = max_reserve / max(1, cal_n + thr_n)
            cal_n = int(np.floor(cal_n * scale))
            thr_n = max_reserve - cal_n

        model_end = len(window) - cal_n - thr_n
        threshold_end = len(window) - cal_n
        train_parts.append(window[:model_end])
        if thr_n:
            thr_reserve_parts.append(window[model_end:threshold_end])
        if cal_n:
            cal_reserve_parts.append(window[threshold_end:])

    train_idx = _concat_unique_sorted(train_parts)
    cal_reserve = _concat_unique_sorted(cal_reserve_parts)
    thr_reserve = _concat_unique_sorted(thr_reserve_parts)
    calibration_idx = _concat_unique_sorted([cal_reserve, dedicated_calibration_idx])
    threshold_pool_idx = _concat_unique_sorted([thr_reserve, dedicated_threshold_idx])

    cal_strategy = "dedicated_plus_disjoint_past_calibration_reserve"
    thr_strategy = "raw_threshold_plus_disjoint_past_threshold_reserve"

    train_idx, calibration_idx, moved_cal = _move_missing_classes_from_train(
        df,
        train_idx,
        calibration_idx,
        cfg.CALIBRATION_FALLBACK_PER_CLASS,
        cfg.MIN_TRAIN_SAMPLES_PER_CLASS_AFTER_RESERVE,
    )
    if len(moved_cal):
        cal_reserve = _concat_unique_sorted([cal_reserve, moved_cal])
        cal_strategy += "_plus_class_aware_past_fallback"

    train_idx, threshold_pool_idx, moved_thr = _move_missing_classes_from_train(
        df,
        train_idx,
        threshold_pool_idx,
        cfg.THRESHOLD_FALLBACK_PER_CLASS,
        cfg.MIN_TRAIN_SAMPLES_PER_CLASS_AFTER_RESERVE,
    )
    if len(moved_thr):
        thr_reserve = _concat_unique_sorted([thr_reserve, moved_thr])
        thr_strategy += "_plus_class_aware_past_fallback"

    return (
        train_idx,
        calibration_idx,
        threshold_pool_idx,
        cal_strategy,
        thr_strategy,
        int(len(cal_reserve)),
        int(len(thr_reserve)),
    )

def random_split(df: pd.DataFrame, cfg: Config) -> List[DataSplit]:
    """Stratified random 70/10/10/10 baseline.

    Global exact duplicates are already removed during loading. No feature-vector
    grouping is used because repeated traffic patterns are valid observations.
    """
    train_fraction = cfg.SINGLE_TEMPORAL_TRAIN_FRACTION
    cal_fraction = cfg.SINGLE_TEMPORAL_CALIBRATION_FRACTION
    threshold_fraction = cfg.SINGLE_TEMPORAL_THRESHOLD_FRACTION
    test_fraction = 1.0 - train_fraction - cal_fraction - threshold_fraction
    if min(train_fraction, cal_fraction, threshold_fraction, test_fraction) <= 0:
        raise ValueError("Random split fractions must all be positive and sum to 1.")

    all_idx = np.arange(len(df), dtype=np.int64)
    labels = df["__binary_label__"].to_numpy(dtype=np.int8)

    train_idx, rest_idx = train_test_split(
        all_idx,
        train_size=train_fraction,
        random_state=cfg.RANDOM_STATE,
        stratify=labels,
        shuffle=True,
    )

    rest_labels = labels[rest_idx]
    cal_share = cal_fraction / (cal_fraction + threshold_fraction + test_fraction)
    calibration_idx, remain_idx = train_test_split(
        rest_idx,
        train_size=cal_share,
        random_state=cfg.RANDOM_STATE + 1,
        stratify=rest_labels,
        shuffle=True,
    )

    remain_labels = labels[remain_idx]
    threshold_share = threshold_fraction / (threshold_fraction + test_fraction)
    threshold_idx, test_idx = train_test_split(
        remain_idx,
        train_size=threshold_share,
        random_state=cfg.RANDOM_STATE + 2,
        stratify=remain_labels,
        shuffle=True,
    )

    return [DataSplit(
        protocol="random_stratified",
        fold_id="random_01",
        train_idx=np.sort(train_idx),
        calibration_idx=np.sort(calibration_idx),
        threshold_idx=np.sort(threshold_idx),
        threshold_pool_idx=np.sort(threshold_idx),
        test_idx=np.sort(test_idx),
        calibration_strategy="dedicated_stratified_random_calibration",
        threshold_pool_strategy="dedicated_stratified_random_threshold",
    )]


def single_temporal_split(df: pd.DataFrame, cfg: Config) -> List[DataSplit]:
    """Chronological holdout with disjoint past-only support reserves."""
    n = len(df)
    a = int(n * cfg.SINGLE_TEMPORAL_TRAIN_FRACTION)
    b = int(n * (
        cfg.SINGLE_TEMPORAL_TRAIN_FRACTION
        + cfg.SINGLE_TEMPORAL_CALIBRATION_FRACTION
    ))
    c = int(n * (
        cfg.SINGLE_TEMPORAL_TRAIN_FRACTION
        + cfg.SINGLE_TEMPORAL_CALIBRATION_FRACTION
        + cfg.SINGLE_TEMPORAL_THRESHOLD_FRACTION
    ))
    if not (0 < a < b < c < n):
        raise ValueError("Invalid single-temporal split fractions.")

    raw_train = np.arange(0, a, dtype=np.int64)
    dedicated_calibration = np.arange(a, b, dtype=np.int64)
    dedicated_threshold = np.arange(b, c, dtype=np.int64)
    (
        train_idx,
        calibration_idx,
        threshold_pool_idx,
        cal_strategy,
        thr_strategy,
        cal_reserve_rows,
        thr_reserve_rows,
    ) = _build_past_only_support_pools(
        df,
        [raw_train],
        dedicated_calibration,
        dedicated_threshold,
        cfg,
    )

    return [DataSplit(
        protocol="single_temporal",
        fold_id="single_01",
        train_idx=train_idx,
        calibration_idx=calibration_idx,
        threshold_idx=dedicated_threshold,
        threshold_pool_idx=threshold_pool_idx,
        test_idx=np.arange(c, n, dtype=np.int64),
        calibration_strategy=cal_strategy,
        threshold_pool_strategy=thr_strategy,
        calibration_reserve_rows=cal_reserve_rows,
        threshold_reserve_rows=thr_reserve_rows,
        training_mode="single_temporal",
    )]

def rolling_splits(
    df: pd.DataFrame,
    cfg: Config,
    training_mode: str = "sliding",
    log_path: Optional[Path] = None,
) -> List[DataSplit]:
    """Build sliding or expanding chronological folds.

    sliding: the previous ``ROLLING_TRAIN_WINDOWS`` blocks are model history.
    expanding: every block from W0 through W(cal_w-1) is model history.
    """
    training_mode = str(training_mode).strip().lower()
    if training_mode not in {"sliding", "expanding"}:
        raise ValueError("training_mode must be 'sliding' or 'expanding'")

    minimum_windows = cfg.ROLLING_TRAIN_WINDOWS + 3
    if cfg.N_TEMPORAL_WINDOWS < minimum_windows:
        raise ValueError(
            f"N_TEMPORAL_WINDOWS must be >= {minimum_windows} for separate "
            "calibration, threshold, and test windows."
        )

    windows = [
        np.asarray(chunk, dtype=np.int64)
        for chunk in np.array_split(np.arange(len(df)), cfg.N_TEMPORAL_WINDOWS)
        if len(chunk)
    ]
    out: List[DataSplit] = []
    protocol = f"{training_mode}_temporal"

    for candidate_number, cal_w in enumerate(
        range(cfg.ROLLING_TRAIN_WINDOWS, len(windows) - 2), start=1
    ):
        threshold_w = cal_w + 1
        test_w = cal_w + 2
        if training_mode == "expanding":
            train_ids = list(range(0, cal_w))
        else:
            train_ids = list(range(cal_w - cfg.ROLLING_TRAIN_WINDOWS, cal_w))

        (
            train_idx,
            calibration_idx,
            threshold_pool_idx,
            cal_strategy,
            thr_strategy,
            cal_reserve_rows,
            thr_reserve_rows,
        ) = _build_past_only_support_pools(
            df,
            [windows[i] for i in train_ids],
            windows[cal_w],
            windows[threshold_w],
            cfg,
        )

        if _class_count(df, train_idx) < 2:
            if log_path is not None:
                log(
                    f"Skipping {protocol} candidate {candidate_number:02d}: "
                    "model training history has one class after support reserves",
                    log_path,
                )
            continue

        out.append(DataSplit(
            protocol=protocol,
            fold_id=f"{training_mode}_{candidate_number:02d}",
            train_idx=train_idx,
            calibration_idx=calibration_idx,
            threshold_idx=windows[threshold_w],
            threshold_pool_idx=threshold_pool_idx,
            test_idx=windows[test_w],
            calibration_strategy=cal_strategy,
            threshold_pool_strategy=thr_strategy,
            calibration_reserve_rows=cal_reserve_rows,
            threshold_reserve_rows=thr_reserve_rows,
            train_window_ids=tuple(train_ids),
            calibration_window_id=cal_w,
            threshold_window_id=threshold_w,
            test_window_id=test_w,
            candidate_number=candidate_number,
            training_mode=training_mode,
        ))
    return out

def temporal_window_attack_profile(
    df: pd.DataFrame,
    n_windows: int,
    tag: str,
    dirs: Mapping[str, Path],
) -> pd.DataFrame:
    """Export attack counts/rates and actual elapsed time for equal-count blocks."""
    if n_windows < 2:
        raise ValueError("n_windows must be at least 2")
    windows = [
        np.asarray(chunk, dtype=np.int64)
        for chunk in np.array_split(np.arange(len(df)), n_windows)
        if len(chunk)
    ]
    rows: List[Dict[str, Any]] = []
    previous_end = None
    for window_id, idx in enumerate(windows):
        labels = df.iloc[idx]["__binary_label__"].to_numpy(dtype=np.int8)
        attack_rows = int(labels.sum())
        normal_rows = int(len(labels) - attack_rows)
        start_raw = df.iloc[idx[0]].get("__time_order__", idx[0])
        end_raw = df.iloc[idx[-1]].get("__time_order__", idx[-1])
        duration_hours = np.nan
        gap_prev_hours = np.nan
        try:
            start_ts = pd.Timestamp(start_raw)
            end_ts = pd.Timestamp(end_raw)
            duration_hours = float((end_ts - start_ts).total_seconds() / 3600.0)
            if previous_end is not None:
                gap_prev_hours = float((start_ts - previous_end).total_seconds() / 3600.0)
            previous_end = end_ts
        except Exception:
            pass
        rows.append({
            "profile_tag": tag,
            "n_temporal_windows": n_windows,
            "window_id": window_id,
            "window_label": f"W{window_id}",
            "row_start": int(idx[0]),
            "row_end": int(idx[-1]),
            "time_start": str(start_raw),
            "time_end": str(end_raw),
            "duration_hours": duration_hours,
            "gap_from_previous_window_hours": gap_prev_hours,
            "rows": int(len(idx)),
            "normal_rows": normal_rows,
            "attack_rows": attack_rows,
            "attack_rate": float(attack_rows / len(idx)),
            "class_count": int(np.unique(labels).size),
        })
    profile = pd.DataFrame(rows)
    stem = safe_filename(f"Temporal_Attack_Profile_{tag}_{n_windows}_Windows")
    profile.to_csv(dirs["csv"] / f"{stem}.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(profile["window_label"], profile["attack_rate"] * 100.0)
    ax.set(xlabel="Equal-count chronological window", ylabel="Attack rate (%)",
           title=f"Attack Rate per Equal-Count Chronological Window ({n_windows})")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(dirs["plots"] / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return profile

def rolling_candidate_audit(
    df: pd.DataFrame,
    cfg: Config,
    training_mode: str,
    tag: str,
    dirs: Mapping[str, Path],
) -> pd.DataFrame:
    """Audit candidate eligibility at train/calibration/threshold/test levels."""
    windows = [
        np.asarray(chunk, dtype=np.int64)
        for chunk in np.array_split(np.arange(len(df)), cfg.N_TEMPORAL_WINDOWS)
        if len(chunk)
    ]
    rows: List[Dict[str, Any]] = []
    training_mode = training_mode.lower()

    def part_stats(indices: np.ndarray, prefix: str) -> Dict[str, Any]:
        labels = df.iloc[indices]["__binary_label__"].to_numpy(dtype=np.int8)
        attacks = int(labels.sum())
        return {
            f"{prefix}_rows": int(len(indices)),
            f"{prefix}_attack_rows": attacks,
            f"{prefix}_attack_rate": float(attacks / len(indices)) if len(indices) else np.nan,
            f"{prefix}_class_count": int(np.unique(labels).size) if len(indices) else 0,
        }

    for candidate_number, cal_w in enumerate(
        range(cfg.ROLLING_TRAIN_WINDOWS, len(windows) - 2), start=1
    ):
        threshold_w = cal_w + 1
        test_w = cal_w + 2
        if training_mode == "expanding":
            train_ids = list(range(0, cal_w))
        else:
            train_ids = list(range(cal_w - cfg.ROLLING_TRAIN_WINDOWS, cal_w))
        raw_train_idx = _concat_unique_sorted([windows[i] for i in train_ids])
        (
            train_idx, calibration_idx, threshold_pool_idx,
            cal_strategy, thr_strategy, cal_reserve_rows, thr_reserve_rows,
        ) = _build_past_only_support_pools(
            df, [windows[i] for i in train_ids], windows[cal_w], windows[threshold_w], cfg
        )
        threshold_idx = windows[threshold_w]
        test_idx = windows[test_w]

        train_classes = _class_count(df, train_idx)
        cal_classes = _class_count(df, calibration_idx)
        threshold_classes = _class_count(df, threshold_idx)
        threshold_pool_classes = _class_count(df, threshold_pool_idx)
        test_classes = _class_count(df, test_idx)
        tier_a = train_classes >= 2 and test_classes >= 2
        tier_b = tier_a and cal_classes >= 2
        tier_c_raw_two_class = tier_b and threshold_classes >= 2
        tier_c_hist_pool = tier_b and threshold_pool_classes >= 2

        row: Dict[str, Any] = {
            "audit_tag": tag,
            "training_mode": training_mode,
            "protocol": f"{training_mode}_temporal",
            "n_temporal_windows": cfg.N_TEMPORAL_WINDOWS,
            "candidate_number": candidate_number,
            "fold_id": f"{training_mode}_{candidate_number:02d}",
            "train_window_ids": ",".join(map(str, train_ids)),
            "calibration_window_id": cal_w,
            "threshold_window_id": threshold_w,
            "test_window_id": test_w,
            "calibration_strategy": cal_strategy,
            "threshold_pool_strategy": thr_strategy,
            "calibration_reserve_rows": cal_reserve_rows,
            "threshold_reserve_rows": thr_reserve_rows,
            "tier_A_discrimination_eligible": bool(tier_a),
            "tier_B_calibration_eligible": bool(tier_b),
            "tier_C_raw_two_class_threshold_eligible": bool(tier_c_raw_two_class),
            "tier_C_historical_pool_threshold_eligible": bool(tier_c_hist_pool),
            "valid_training_after_reserve": bool(train_classes >= 2),
            "skip_reason": "" if train_classes >= 2 else "single_class_training_after_support_reserves",
        }
        row.update(part_stats(raw_train_idx, "raw_train"))
        row.update(part_stats(train_idx, "model_train"))
        row.update(part_stats(calibration_idx, "effective_calibration_pool"))
        row.update(part_stats(threshold_idx, "raw_threshold"))
        row.update(part_stats(threshold_pool_idx, "historical_threshold_pool"))
        row.update(part_stats(test_idx, "test"))
        rows.append(row)

    audit = pd.DataFrame(rows)
    stem = safe_filename(
        f"Rolling_Candidate_Audit_{training_mode}_{tag}_{cfg.N_TEMPORAL_WINDOWS}_Windows"
    )
    audit.to_csv(dirs["csv"] / f"{stem}.csv", index=False)
    return audit

def split_validation(
    val: pd.DataFrame,
    fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Legacy helper retained for compatibility; unused by the final protocol."""
    if len(val) < 4:
        return val.copy(), val.copy(), "legacy_shared_small_validation"
    cut = max(1, min(len(val) - 1, int(len(val) * fraction)))
    return val.iloc[:cut].copy(), val.iloc[cut:].copy(), "legacy_split"

# =============================================================================
# PREPROCESSING AND RFE
# =============================================================================
def sanitize_numeric_block(values):
    """Convert numeric columns safely and replace corrupted non-finite values.

    The transformation is deterministic and does not learn from validation/test data.
    Missing and non-finite values are imputed later using medians learned from the
    training partition only.
    """
    if isinstance(values, pd.DataFrame):
        array = values.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    else:
        array = np.asarray(values, dtype=np.float64)

    # NF flow files may contain +inf/-inf after rate or duration calculations.
    array[~np.isfinite(array)] = np.nan

    # Extremely large corrupted values can overflow StandardScaler variance.
    # 1e30 is far above meaningful NetFlow measurements and is treated as missing.
    array[np.abs(array) > 1e30] = np.nan
    return array


def finite_float32(values):
    """Return a finite float32 matrix after preprocessing."""
    array = np.asarray(values, dtype=np.float64)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    cap = np.finfo(np.float32).max / 100.0
    array = np.clip(array, -cap, cap)
    return array.astype(np.float32, copy=False)


def fit_transform(
    train: pd.DataFrame,
    cal: pd.DataFrame,
    thr: pd.DataFrame,
    thr_pool: pd.DataFrame,
    test: pd.DataFrame,
    cols: Sequence[str],
):
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in cols if c not in numeric]
    transformers = []

    if numeric:
        numeric_pipeline = Pipeline([
            ("sanitize", FunctionTransformer(
                sanitize_numeric_block,
                validate=False,
                feature_names_out="one-to-one",
            )),
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric))

    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("cat", categorical_pipeline, categorical))

    if not transformers:
        raise ValueError("No usable features")

    pre = ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    xt = finite_float32(pre.fit_transform(train.loc[:, cols]))
    xc = finite_float32(pre.transform(cal.loc[:, cols]))
    xv = finite_float32(pre.transform(thr.loc[:, cols]))
    xvp = finite_float32(pre.transform(thr_pool.loc[:, cols]))
    xs = finite_float32(pre.transform(test.loc[:, cols]))
    return xt, xc, xv, xvp, xs, list(pre.get_feature_names_out())


@dataclass
class RFEResult:
    selected: np.ndarray
    selected_names: List[str]
    ranking: np.ndarray
    importances: np.ndarray
    sampling_mode: str
    sample_seed: int
    model_seed: int
    n_available: int
    n_selected_for_rfe: int
    positives_available: int
    negatives_available: int
    positives_selected: int
    negatives_selected: int
    sample_attack_rate: float
    positive_cap_applied: bool
    sample_index_sha1: str


def rfe_sample_indices(
    y: np.ndarray,
    max_samples: int,
    seed: int,
    mode: str,
    positive_cap_fraction: float = 0.15,
    min_other_class_samples: int = 2_000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Construct an auditable two-class RFE sample.

    ``capped_positive`` is the final-study default. When subsampling is needed,
    positives are capped at ``positive_cap_fraction`` of the RFE budget, while
    rare-attack folds retain every available positive. ``positive_complete`` is
    retained only as a sensitivity arm and is rejected if positives alone fill
    the entire RFE budget.
    """
    y = np.asarray(y, dtype=np.int8)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("RFE labels must be a non-empty one-dimensional array")
    if max_samples < 2:
        raise ValueError("RFE_MAX_SAMPLES must be at least 2")
    if not (0.0 < positive_cap_fraction < 1.0):
        raise ValueError("RFE_POSITIVE_CAP_FRACTION must be in (0,1)")

    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("RFE requires both Normal and Attack classes")

    target = min(int(max_samples), len(y))
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "uniform":
        normalized_mode = "natural_uniform"

    cap_applied = False
    if target == len(y):
        # No subsampling: preserve the true training prevalence.
        idx = np.arange(len(y), dtype=np.int64)
    elif normalized_mode == "natural_uniform":
        idx = rng.choice(len(y), size=target, replace=False)
    elif normalized_mode == "proportional_stratified":
        positive_fraction = len(pos) / len(y)
        n_pos = int(round(target * positive_fraction))
        n_pos = max(1, min(n_pos, len(pos), target - 1))
        n_neg = target - n_pos
        if n_neg > len(neg):
            n_neg = len(neg)
            n_pos = target - n_neg
        idx = np.concatenate([
            rng.choice(pos, size=n_pos, replace=False),
            rng.choice(neg, size=n_neg, replace=False),
        ])
    elif normalized_mode == "capped_positive":
        positive_cap = max(1, int(np.floor(target * positive_cap_fraction)))
        n_pos = min(len(pos), positive_cap)
        n_neg = min(len(neg), target - n_pos)
        cap_applied = len(pos) > n_pos
        # Fill unused slots from negatives only so the positive cap is respected.
        if n_pos + n_neg < target:
            add_neg = min(target - n_pos - n_neg, len(neg) - n_neg)
            n_neg += add_neg
        if n_pos < 1 or n_neg < 1:
            raise RuntimeError("Could not construct capped-positive two-class RFE sample")
        idx = np.concatenate([
            rng.choice(pos, size=n_pos, replace=False),
            rng.choice(neg, size=n_neg, replace=False),
        ])
    elif normalized_mode == "positive_complete":
        if len(pos) >= target:
            raise ValueError(
                "positive_complete is invalid because positives fill or exceed "
                "the complete RFE sample budget. Use capped_positive instead."
            )
        n_pos = len(pos)
        n_neg = min(len(neg), target - n_pos)
        if n_neg < 1:
            raise RuntimeError("positive_complete produced no normal observations")
        idx = np.concatenate([
            pos,
            rng.choice(neg, size=n_neg, replace=False),
        ])
    else:
        raise ValueError(
            f"Unknown RFE sampling mode: {mode}. Expected natural_uniform, "
            "capped_positive, positive_complete, or proportional_stratified."
        )

    idx = np.asarray(idx, dtype=np.int64)
    rng.shuffle(idx)
    selected_y = y[idx]
    # Fingerprint the actual sampled row indices (order-independent) so that
    # sensitivity modes can be audited for accidental sample reuse/caching.
    sample_index_sha1 = hashlib.sha1(
        np.sort(idx).astype(np.int64, copy=False).tobytes()
    ).hexdigest()
    stats = {
        "sampling_mode": normalized_mode,
        "sample_seed": int(seed),
        "n_available": int(len(y)),
        "n_selected_for_rfe": int(len(idx)),
        "positives_available": int(len(pos)),
        "negatives_available": int(len(neg)),
        "positives_selected": int((selected_y == 1).sum()),
        "negatives_selected": int((selected_y == 0).sum()),
        "sample_attack_rate": float(np.mean(selected_y)),
        "positive_cap_applied": bool(cap_applied),
        "sample_index_sha1": sample_index_sha1,
    }
    return idx, stats


def run_rfe(
    x: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    cfg: Config,
    sampling_mode: Optional[str] = None,
    sample_seed: Optional[int] = None,
    model_seed: Optional[int] = None,
) -> RFEResult:
    """Run train-only RFE with separately controlled sample/model randomness."""
    k = min(cfg.TOP_K_FEATURES, x.shape[1])
    mode = sampling_mode or cfg.FULL_RFE_SAMPLING_MODE
    resolved_sample_seed = int(
        cfg.RFE_FIXED_SAMPLE_SEED if sample_seed is None else sample_seed
    )
    resolved_model_seed = int(
        cfg.RFE_FIXED_MODEL_SEED if model_seed is None else model_seed
    )
    idx, sample_stats = rfe_sample_indices(
        y,
        cfg.RFE_MAX_SAMPLES,
        resolved_sample_seed,
        mode,
        cfg.RFE_POSITIVE_CAP_FRACTION,
        cfg.RFE_MIN_OTHER_CLASS_SAMPLES,
    )
    xf, yf = x[idx], np.asarray(y, dtype=np.int8)[idx]

    if k >= x.shape[1]:
        ranking = np.ones(x.shape[1], dtype=int)
        importances = np.ones(x.shape[1], dtype=float) / max(1, x.shape[1])
        selected = np.arange(x.shape[1])
    else:
        est = RandomForestClassifier(
            n_estimators=cfg.RFE_N_ESTIMATORS,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=resolved_model_seed,
            max_depth=18,
        )
        selector = RFE(
            estimator=est,
            n_features_to_select=k,
            step=cfg.RFE_STEP,
        )
        selector.fit(xf, yf)
        imp_model = clone(est)
        imp_model.fit(xf, yf)
        selected = np.flatnonzero(selector.support_)
        ranking = np.asarray(selector.ranking_)
        importances = np.asarray(imp_model.feature_importances_)

    return RFEResult(
        selected=selected,
        selected_names=[names[i] for i in selected],
        ranking=ranking,
        importances=importances,
        sampling_mode=str(sample_stats["sampling_mode"]),
        sample_seed=resolved_sample_seed,
        model_seed=resolved_model_seed,
        n_available=int(sample_stats["n_available"]),
        n_selected_for_rfe=int(sample_stats["n_selected_for_rfe"]),
        positives_available=int(sample_stats["positives_available"]),
        negatives_available=int(sample_stats["negatives_available"]),
        positives_selected=int(sample_stats["positives_selected"]),
        negatives_selected=int(sample_stats["negatives_selected"]),
        sample_attack_rate=float(sample_stats["sample_attack_rate"]),
        positive_cap_applied=bool(sample_stats["positive_cap_applied"]),
        sample_index_sha1=str(sample_stats["sample_index_sha1"]),
    )


def _select_audit_folds(
    df: pd.DataFrame,
    splits: Sequence[DataSplit],
) -> Dict[str, DataSplit]:
    """Choose lowest-positive-rate and median-like valid sliding folds."""
    candidates = []
    for split in splits:
        y = df.iloc[split.train_idx]["__binary_label__"].to_numpy(dtype=np.int8)
        rate = float(np.mean(y)) if len(y) else np.nan
        if len(np.unique(y)) >= 2 and np.isfinite(rate) and rate > 0:
            candidates.append((split, rate))
    if not candidates:
        raise RuntimeError("No two-class sliding fold is available for RFE audit")
    stress = min(candidates, key=lambda item: item[1])
    rates = np.asarray([r for _, r in candidates], dtype=float)
    median_rate = float(np.median(rates))
    typical = min(candidates, key=lambda item: abs(item[1] - median_rate))
    return {"stress": stress[0], "typical": typical[0]}


def _core_features(results: Sequence[RFEResult]) -> List[str]:
    if not results:
        return []
    core = set(results[0].selected_names)
    for result in results[1:]:
        core &= set(result.selected_names)
    return sorted(core)


def _audit_one_configuration(
    df: pd.DataFrame,
    split: DataSplit,
    base: str,
    cols: Sequence[str],
    cfg: Config,
    sampling_mode: str,
    sample_seeds: Sequence[int],
    model_seed: int,
    dirs: Mapping[str, Path],
    log_path: Path,
    audit_role: str,
    file_prefix: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[RFEResult]]:
    train = df.iloc[split.train_idx].copy()
    dummy = train.iloc[:1].copy()
    x_train, _, _, _, _, names = fit_transform(train, dummy, dummy, dummy, dummy, cols)
    y_train = train["__binary_label__"].to_numpy(dtype=np.int8)
    runs: List[Dict[str, Any]] = []
    results: List[RFEResult] = []

    for repeat_id, sample_seed in enumerate(sample_seeds, start=1):
        log(
            f"RFE audit {audit_role} | {split.fold_id} | {base} | {sampling_mode} | "
            f"repeat {repeat_id}/{len(sample_seeds)}",
            log_path,
        )
        result = run_rfe(
            x_train, y_train, names, cfg,
            sampling_mode=sampling_mode,
            sample_seed=int(sample_seed),
            model_seed=int(model_seed),
        )
        results.append(result)
        runs.append({
            "audit_role": audit_role,
            "protocol": split.protocol,
            "fold_id": split.fold_id,
            "candidate_number": split.candidate_number,
            "base_variant": base,
            "paper_variant_label": PAPER_VARIANT_LABELS.get(base, base),
            "sampling_mode": result.sampling_mode,
            "repeat_id": repeat_id,
            "sample_seed": result.sample_seed,
            "model_seed": result.model_seed,
            "n_available": result.n_available,
            "n_selected_for_rfe": result.n_selected_for_rfe,
            "positives_available": result.positives_available,
            "negatives_available": result.negatives_available,
            "positives_selected": result.positives_selected,
            "negatives_selected": result.negatives_selected,
            "sample_attack_rate": result.sample_attack_rate,
            "positive_cap_applied": result.positive_cap_applied,
            "sample_index_sha1": result.sample_index_sha1,
            "selected_features": "|".join(result.selected_names),
        })
        ranking_df = pd.DataFrame({
            "feature": names,
            "rfe_rank": result.ranking,
            "rf_importance": result.importances,
            "selected": np.isin(np.arange(len(names)), result.selected),
        })
        for key, value in reversed(list(runs[-1].items())[:-1]):
            ranking_df.insert(0, key, value)
        ranking_df.to_csv(
            dirs["csv"] / safe_filename(
                f"{file_prefix}_{audit_role}_{split.fold_id}_{base}_{sampling_mode}_Repeat_{repeat_id}.csv"
            ),
            index=False,
        )

    pair_rows: List[Dict[str, Any]] = []
    for (i, a), (j, b) in combinations(list(enumerate(results, start=1)), 2):
        pair_rows.append({
            "audit_role": audit_role,
            "protocol": split.protocol,
            "fold_id": split.fold_id,
            "base_variant": base,
            "paper_variant_label": PAPER_VARIANT_LABELS.get(base, base),
            "sampling_mode": sampling_mode,
            "repeat_a": i,
            "repeat_b": j,
            "jaccard": jaccard(a.selected_names, b.selected_names),
            "kuncheva": kuncheva(a.selected_names, b.selected_names, len(names)),
            "spearman_importance": rank_spearman(
                names, a.importances, names, b.importances
            ),
        })
    del x_train, train, dummy
    gc.collect()
    return runs, pair_rows, results


def rfe_noise_floor_audit(
    df: pd.DataFrame,
    base_cols: Mapping[str, Sequence[str]],
    cfg: Config,
    dirs: Mapping[str, Path],
    log_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Noise-floor audit matched to the final 18-window capped-positive design.

    The RFE forest seed is fixed for a base variant. Only the RFE sample seed
    changes across repeats, so within-fold variability estimates subsampling
    variability rather than a mixture of sample and estimator randomness.
    """
    audit_cfg = replace(cfg, N_TEMPORAL_WINDOWS=cfg.AUDIT_N_TEMPORAL_WINDOWS)
    sliding = rolling_splits(df, audit_cfg, "sliding", log_path)
    chosen = _select_audit_folds(df, sliding)
    run_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    frequency_rows: List[Dict[str, Any]] = []

    for base_index, base in enumerate(cfg.RFE_NOISE_AUDIT_BASE_VARIANTS):
        cols = list(base_cols.get(base, []))
        if not cols:
            continue
        model_seed = cfg.RFE_FIXED_MODEL_SEED + base_index
        for audit_role, split in chosen.items():
            runs, pairs, results = _audit_one_configuration(
                df, split, base, cols, audit_cfg,
                cfg.AUDIT_RFE_SAMPLING_MODE,
                cfg.AUDIT_SAMPLE_SEEDS,
                model_seed,
                dirs, log_path, audit_role, "RFE_Noise",
            )
            run_rows.extend(runs)
            pair_rows.extend(pairs)
            core = _core_features(results)
            counts: Counter[str] = Counter()
            for result in results:
                counts.update(result.selected_names)
            for feature, count in counts.items():
                frequency_rows.append({
                    "audit_role": audit_role,
                    "fold_id": split.fold_id,
                    "base_variant": base,
                    "paper_variant_label": PAPER_VARIANT_LABELS.get(base, base),
                    "sampling_mode": cfg.AUDIT_RFE_SAMPLING_MODE,
                    "feature": feature,
                    "selected_count": count,
                    "n_repeats": len(results),
                    "selection_frequency": count / len(results),
                    "is_within_core_100pct": feature in core,
                })
            group = pd.DataFrame(pairs)
            if not group.empty:
                summary_rows.append({
                    "audit_role": audit_role,
                    "fold_id": split.fold_id,
                    "candidate_number": split.candidate_number,
                    "base_variant": base,
                    "paper_variant_label": PAPER_VARIANT_LABELS.get(base, base),
                    "sampling_mode": cfg.AUDIT_RFE_SAMPLING_MODE,
                    "n_independent_seed_replicates": len(results),
                    "n_dependent_pairwise_comparisons": len(group),
                    "mean_within_jaccard": float(group.jaccard.mean()),
                    "median_within_jaccard": float(group.jaccard.median()),
                    "std_within_jaccard_descriptive": float(group.jaccard.std(ddof=1)),
                    "min_within_jaccard": float(group.jaccard.min()),
                    "max_within_jaccard": float(group.jaccard.max()),
                    "mean_within_kuncheva": float(group.kuncheva.mean()),
                    "mean_within_spearman": float(group.spearman_importance.mean()),
                    "within_core_size": len(core),
                    "within_core_ratio": len(core) / max(1, cfg.TOP_K_FEATURES),
                    "within_core_features": "|".join(core),
                    "training_attack_rate": float(
                        df.iloc[split.train_idx]["__binary_label__"].mean()
                    ),
                    "mean_rfe_sample_attack_rate": float(
                        np.mean([r.sample_attack_rate for r in results])
                    ),
                })

    runs_df = pd.DataFrame(run_rows)
    pairs_df = pd.DataFrame(pair_rows)
    summary_df = pd.DataFrame(summary_rows)
    frequency_df = pd.DataFrame(frequency_rows)
    runs_df.to_csv(dirs["csv"] / "RFE_Noise_Audit_Runs.csv", index=False)
    pairs_df.to_csv(dirs["csv"] / "RFE_Noise_Audit_Pairwise.csv", index=False)
    summary_df.to_csv(dirs["csv"] / "RFE_Noise_Audit_Summary.csv", index=False)
    frequency_df.to_csv(dirs["csv"] / "RFE_Noise_Audit_Feature_Frequency.csv", index=False)

    # Sampling-prevalence sensitivity on the typical fold only.
    sensitivity_runs: List[Dict[str, Any]] = []
    sensitivity_pairs: List[Dict[str, Any]] = []
    sensitivity_summary: List[Dict[str, Any]] = []
    sensitivity_cores: Dict[str, set] = {}
    sensitivity_results_by_mode: Dict[str, List[RFEResult]] = {}
    typical = chosen["typical"]
    base = cfg.RFE_SENSITIVITY_BASE_VARIANT
    cols = list(base_cols.get(base, []))
    if cols:
        base_index = list(cfg.BASE_VARIANT_ORDER).index(base)
        model_seed = cfg.RFE_FIXED_MODEL_SEED + base_index
        for mode in cfg.RFE_SENSITIVITY_MODES:
            try:
                runs, pairs, results = _audit_one_configuration(
                    df, typical, base, cols, audit_cfg, mode,
                    cfg.AUDIT_SAMPLE_SEEDS, model_seed,
                    dirs, log_path, "typical_sampling_sensitivity", "RFE_Sensitivity",
                )
            except ValueError as exc:
                sensitivity_summary.append({
                    "sampling_mode": mode,
                    "status": "not_applicable",
                    "reason": str(exc),
                    "fold_id": typical.fold_id,
                    "base_variant": base,
                })
                continue
            sensitivity_runs.extend(runs)
            sensitivity_pairs.extend(pairs)
            sensitivity_results_by_mode[mode] = list(results)
            core = set(_core_features(results))
            sensitivity_cores[mode] = core
            g = pd.DataFrame(pairs)
            sensitivity_summary.append({
                "sampling_mode": mode,
                "status": "completed",
                "fold_id": typical.fold_id,
                "base_variant": base,
                "n_repeats": len(results),
                "mean_within_jaccard": float(g.jaccard.mean()) if not g.empty else np.nan,
                "min_within_jaccard": float(g.jaccard.min()) if not g.empty else np.nan,
                "max_within_jaccard": float(g.jaccard.max()) if not g.empty else np.nan,
                "core_size": len(core),
                "core_features": "|".join(sorted(core)),
                "mean_sample_attack_rate": float(np.mean([r.sample_attack_rate for r in results])),
            })
        common_core = set.intersection(*sensitivity_cores.values()) if sensitivity_cores else set()
        for row in sensitivity_summary:
            row["core_shared_across_completed_sampling_modes_size"] = len(common_core)
            row["core_shared_across_completed_sampling_modes"] = "|".join(sorted(common_core))

    # Direct seed-matched cross-mode set comparison. This specifically checks
    # whether identical mean Jaccard values hide different selected sets and
    # whether any modes accidentally reused the exact same sampled rows.
    sensitivity_mode_seedwise: List[Dict[str, Any]] = []
    completed_modes = sorted(sensitivity_results_by_mode)
    for mode_a, mode_b in combinations(completed_modes, 2):
        ra = sensitivity_results_by_mode[mode_a]
        rb = sensitivity_results_by_mode[mode_b]
        for repeat_index, (a, b) in enumerate(zip(ra, rb), start=1):
            sa, sb = set(a.selected_names), set(b.selected_names)
            sensitivity_mode_seedwise.append({
                "mode_a": mode_a,
                "mode_b": mode_b,
                "repeat_id": repeat_index,
                "sample_seed_a": a.sample_seed,
                "sample_seed_b": b.sample_seed,
                "sample_index_sha1_a": a.sample_index_sha1,
                "sample_index_sha1_b": b.sample_index_sha1,
                "same_sample_indices": bool(a.sample_index_sha1 == b.sample_index_sha1),
                "exact_selected_set_match": bool(sa == sb),
                "intersection_size": len(sa & sb),
                "union_size": len(sa | sb),
                "jaccard": jaccard(sa, sb),
                "features_only_a": "|".join(sorted(sa - sb)),
                "features_only_b": "|".join(sorted(sb - sa)),
            })

    seedwise_df = pd.DataFrame(sensitivity_mode_seedwise)
    seedwise_df.to_csv(
        dirs["csv"] / "RFE_Sampling_Mode_Seedwise_Set_Comparison.csv", index=False
    )

    # Add a compact cache/identity audit to the sensitivity summary.
    if not seedwise_df.empty:
        for row in sensitivity_summary:
            mode = row.get("sampling_mode")
            if row.get("status") != "completed":
                continue
            natural_cap = seedwise_df[
                ((seedwise_df.mode_a == "natural_uniform") & (seedwise_df.mode_b == "capped_positive"))
                | ((seedwise_df.mode_a == "capped_positive") & (seedwise_df.mode_b == "natural_uniform"))
            ]
            row["natural_vs_capped_all_selected_sets_identical"] = (
                bool(natural_cap["exact_selected_set_match"].all()) if not natural_cap.empty else np.nan
            )
            row["natural_vs_capped_any_same_sample_indices"] = (
                bool(natural_cap["same_sample_indices"].any()) if not natural_cap.empty else np.nan
            )
            row["natural_vs_capped_mean_seedmatched_jaccard"] = (
                float(natural_cap["jaccard"].mean()) if not natural_cap.empty else np.nan
            )

    pd.DataFrame(sensitivity_runs).to_csv(
        dirs["csv"] / "RFE_Sampling_Prevalence_Sensitivity_Runs.csv", index=False
    )
    pd.DataFrame(sensitivity_pairs).to_csv(
        dirs["csv"] / "RFE_Sampling_Prevalence_Sensitivity_Pairwise.csv", index=False
    )
    pd.DataFrame(sensitivity_summary).to_csv(
        dirs["csv"] / "RFE_Sampling_Prevalence_Sensitivity_Summary.csv", index=False
    )

    # Selection of the two audit folds is itself an auditable output.
    pd.DataFrame([
        {
            "audit_role": role,
            "fold_id": split.fold_id,
            "candidate_number": split.candidate_number,
            "train_window_ids": ",".join(map(str, split.train_window_ids)),
            "training_attack_rate": float(df.iloc[split.train_idx]["__binary_label__"].mean()),
            "train_rows": int(len(split.train_idx)),
        }
        for role, split in chosen.items()
    ]).to_csv(dirs["csv"] / "RFE_Noise_Audit_Selected_Folds.csv", index=False)

    return summary_df, pairs_df, runs_df



def rebuild_sampling_sensitivity_integrity_from_existing(
    df: pd.DataFrame,
    cfg: Config,
    dirs: Mapping[str, Path],
    log_path: Path,
) -> pd.DataFrame:
    """Rebuild the seed-matched sampling-mode comparison without rerunning RFE.

    The previous preflight already saved selected feature sets for every
    sensitivity repeat. This function compares those saved sets directly and
    independently recomputes the sample-index fingerprints from the same labels,
    seeds, and sampling rules. Therefore an identical feature-set result cannot
    be mistaken for accidental cache/sample reuse.
    """
    runs_path = dirs["csv"] / "RFE_Sampling_Prevalence_Sensitivity_Runs.csv"
    if not runs_path.exists():
        return pd.DataFrame()

    runs = pd.read_csv(runs_path)
    required = {"sampling_mode", "repeat_id", "sample_seed", "selected_features"}
    if not required.issubset(runs.columns):
        log("Sensitivity runs exist but lack fields needed for integrity audit", log_path)
        return pd.DataFrame()

    audit_cfg = replace(cfg, N_TEMPORAL_WINDOWS=cfg.AUDIT_N_TEMPORAL_WINDOWS)
    sliding = rolling_splits(df, audit_cfg, "sliding", log_path)
    chosen = _select_audit_folds(df, sliding)
    typical = chosen["typical"]
    y_train = df.iloc[typical.train_idx]["__binary_label__"].to_numpy(dtype=np.int8)

    # Recompute sample fingerprints only; this is cheap and does not refit RFE.
    fingerprint: Dict[Tuple[str, int], str] = {}
    for _, row in runs[["sampling_mode", "repeat_id", "sample_seed"]].drop_duplicates().iterrows():
        mode = str(row["sampling_mode"])
        repeat_id = int(row["repeat_id"])
        seed = int(row["sample_seed"])
        try:
            _, stats = rfe_sample_indices(
                y_train,
                cfg.RFE_MAX_SAMPLES,
                seed,
                mode,
                cfg.RFE_POSITIVE_CAP_FRACTION,
                cfg.RFE_MIN_OTHER_CLASS_SAMPLES,
            )
            fingerprint[(mode, repeat_id)] = str(stats["sample_index_sha1"])
        except Exception as exc:
            fingerprint[(mode, repeat_id)] = f"ERROR:{type(exc).__name__}:{exc}"

    mode_rows: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for _, row in runs.iterrows():
        mode_rows[(str(row["sampling_mode"]), int(row["repeat_id"]))] = row

    out_rows: List[Dict[str, Any]] = []
    modes = sorted(runs["sampling_mode"].astype(str).unique())
    repeats = sorted(runs["repeat_id"].astype(int).unique())
    for mode_a, mode_b in combinations(modes, 2):
        for repeat_id in repeats:
            ra = mode_rows.get((mode_a, repeat_id))
            rb = mode_rows.get((mode_b, repeat_id))
            if ra is None or rb is None:
                continue
            sa = {x for x in str(ra["selected_features"]).split("|") if x}
            sb = {x for x in str(rb["selected_features"]).split("|") if x}
            sha_a = fingerprint.get((mode_a, repeat_id), "")
            sha_b = fingerprint.get((mode_b, repeat_id), "")
            out_rows.append({
                "fold_id": typical.fold_id,
                "candidate_number": typical.candidate_number,
                "base_variant": cfg.RFE_SENSITIVITY_BASE_VARIANT,
                "mode_a": mode_a,
                "mode_b": mode_b,
                "repeat_id": repeat_id,
                "sample_seed_a": int(ra["sample_seed"]),
                "sample_seed_b": int(rb["sample_seed"]),
                "sample_index_sha1_a": sha_a,
                "sample_index_sha1_b": sha_b,
                "same_sample_indices": bool(sha_a == sha_b and not sha_a.startswith("ERROR:")),
                "exact_selected_set_match": bool(sa == sb),
                "intersection_size": len(sa & sb),
                "union_size": len(sa | sb),
                "jaccard": jaccard(sa, sb),
                "features_only_a": "|".join(sorted(sa - sb)),
                "features_only_b": "|".join(sorted(sb - sa)),
            })

    out = pd.DataFrame(out_rows)
    out.to_csv(
        dirs["csv"] / "RFE_Sampling_Mode_Seedwise_Set_Comparison.csv", index=False
    )

    if not out.empty:
        summary_rows = []
        for (mode_a, mode_b), g in out.groupby(["mode_a", "mode_b"], sort=False):
            summary_rows.append({
                "mode_a": mode_a,
                "mode_b": mode_b,
                "n_seed_matched_repeats": len(g),
                "exact_selected_set_match_count": int(g["exact_selected_set_match"].sum()),
                "all_selected_sets_identical": bool(g["exact_selected_set_match"].all()),
                "same_sample_indices_count": int(g["same_sample_indices"].sum()),
                "any_same_sample_indices": bool(g["same_sample_indices"].any()),
                "mean_seedmatched_jaccard": float(g["jaccard"].mean()),
                "min_seedmatched_jaccard": float(g["jaccard"].min()),
                "max_seedmatched_jaccard": float(g["jaccard"].max()),
            })
        pd.DataFrame(summary_rows).to_csv(
            dirs["csv"] / "RFE_Sampling_Mode_Integrity_Summary.csv", index=False
        )
    return out



def compare_noise_to_temporal_stability(
    noise_summary: pd.DataFrame,
    temporal_stability_summary: pd.DataFrame,
    cfg: Config,
    dirs: Mapping[str, Path],
) -> pd.DataFrame:
    """Create a descriptive three-reference stability table.

    Pairwise Jaccard values are dependent because the same feature sets appear in
    multiple pairs. No t/Mann-Whitney/Wilcoxon test is applied to those pairs.
    """
    rows: List[Dict[str, Any]] = []
    for base in cfg.RFE_NOISE_AUDIT_BASE_VARIANTS:
        base_noise = noise_summary[noise_summary["base_variant"] == base] if not noise_summary.empty else pd.DataFrame()
        sliding = temporal_stability_summary[
            (temporal_stability_summary["base_variant"] == base)
            & (temporal_stability_summary["protocol"] == "sliding_temporal")
        ] if not temporal_stability_summary.empty else pd.DataFrame()
        expanding = temporal_stability_summary[
            (temporal_stability_summary["base_variant"] == base)
            & (temporal_stability_summary["protocol"] == "expanding_temporal")
        ] if not temporal_stability_summary.empty else pd.DataFrame()

        row: Dict[str, Any] = {
            "base_variant": base,
            "paper_variant_label": PAPER_VARIANT_LABELS.get(base, base),
            "comparison_note": (
                "descriptive_only_pairwise_values_are_dependent; expanding is an "
                "overlap-favoured reference, not an independent temporal estimate"
            ),
        }
        for role in ("stress", "typical"):
            g = base_noise[base_noise["audit_role"] == role]
            if not g.empty:
                r = g.iloc[0]
                row[f"within_{role}_mean_jaccard"] = float(r["mean_within_jaccard"])
                row[f"within_{role}_min_jaccard"] = float(r["min_within_jaccard"])
                row[f"within_{role}_core_size"] = int(r["within_core_size"])
                row[f"within_{role}_fold_id"] = r["fold_id"]
        if not sliding.empty:
            r = sliding.iloc[0]
            row["sliding_mean_jaccard"] = float(r["mean_jaccard"])
            row["sliding_min_jaccard"] = float(r["min_jaccard"])
            row["sliding_between_core_size"] = int(r["between_core_size"])
            row["sliding_n_folds"] = int(r["n_folds"])
        if not expanding.empty:
            r = expanding.iloc[0]
            row["expanding_mean_jaccard"] = float(r["mean_jaccard"])
            row["expanding_min_jaccard"] = float(r["min_jaccard"])
            row["expanding_between_core_size"] = int(r["between_core_size"])
            row["expanding_n_folds"] = int(r["n_folds"])
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(dirs["csv"] / "RFE_Stability_Three_Reference_Summary.csv", index=False)
    return result


# =============================================================================
# MODELS, CALIBRATION, THRESHOLDS
# =============================================================================
def make_model(name: str, cfg: Config, offset: int) -> BaseEstimator:
    seed = cfg.RANDOM_STATE + offset
    if name == "RF":
        return RandomForestClassifier(n_estimators=cfg.RF_N_ESTIMATORS, max_depth=cfg.RF_MAX_DEPTH, min_samples_leaf=cfg.RF_MIN_SAMPLES_LEAF, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "LR":
        return LogisticRegression(max_iter=cfg.LR_MAX_ITER, class_weight="balanced", solver="liblinear", random_state=seed)
    if name == "LinearSVM":
        return LinearSVC(class_weight="balanced", random_state=seed)
    raise ValueError(name)


def fit_calibrator(model: BaseEstimator, x: np.ndarray, y: np.ndarray, method: str) -> Optional[CalibratedClassifierCV]:
    if len(np.unique(y)) < 2: return None
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(estimator=FrozenEstimator(model), method=method)
    except (ImportError, TypeError):
        cal = CalibratedClassifierCV(estimator=model, method=method, cv="prefit")
    cal.fit(x, y)
    return cal


def positive_probability(model: BaseEstimator, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)
        if p.ndim == 2 and p.shape[1] >= 2: return np.asarray(p[:,1], float)
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(x), float)
        return 1/(1+np.exp(-np.clip(s, -50, 50)))
    raise TypeError("Model has no probability or decision score")

@dataclass
class ProbabilityBundle:
    name: str
    val_prob: np.ndarray
    threshold_pool_prob: np.ndarray
    test_prob: np.ndarray


def probability_bundles(
    name: str,
    model: BaseEstimator,
    xcal: np.ndarray,
    ycal: np.ndarray,
    xval: np.ndarray,
    xthreshold_pool: np.ndarray,
    xtest: np.ndarray,
) -> List[ProbabilityBundle]:
    out: List[ProbabilityBundle] = []
    if name == "RF":
        out.append(ProbabilityBundle(
            "uncalibrated",
            positive_probability(model, xval),
            positive_probability(model, xthreshold_pool),
            positive_probability(model, xtest),
        ))
        for method in ("sigmoid", "isotonic"):
            cal = fit_calibrator(model, xcal, ycal, method)
            if cal is not None:
                out.append(ProbabilityBundle(
                    method,
                    positive_probability(cal, xval),
                    positive_probability(cal, xthreshold_pool),
                    positive_probability(cal, xtest),
                ))
    elif name == "LR":
        out.append(ProbabilityBundle(
            "native",
            positive_probability(model, xval),
            positive_probability(model, xthreshold_pool),
            positive_probability(model, xtest),
        ))
    elif name == "LinearSVM":
        cal = fit_calibrator(model, xcal, ycal, "sigmoid")
        if cal is not None:
            out.append(ProbabilityBundle(
                "sigmoid",
                positive_probability(cal, xval),
                positive_probability(cal, xthreshold_pool),
                positive_probability(cal, xtest),
            ))
    return out


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    valid = np.isfinite(thr)
    if not valid.any():
        return 0.5
    return float(np.clip(thr[valid][np.argmax(tpr[valid] - fpr[valid])], 0, 1))


def fpr_budget_threshold(y: np.ndarray, p: np.ndarray, alpha: float) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    valid = np.isfinite(thr)
    fpr, tpr, thr = fpr[valid], tpr[valid], thr[valid]
    idx = np.where(fpr <= alpha)[0]
    if len(idx) == 0:
        return 1.0
    values = thr[idx]
    minimum = np.min(values)
    candidates = idx[np.isclose(values, minimum)]
    best = candidates[np.argmax(tpr[candidates])]
    return float(np.clip(thr[best], 0, 1))


def threshold_rules(
    y_raw: np.ndarray,
    p_raw: np.ndarray,
    y_hist_pool: np.ndarray,
    p_hist_pool: np.ndarray,
    cfg: Config,
) -> List[Dict[str, Any]]:
    """Return threshold rules with explicit applicability/fallback metadata.

    Raw single-class threshold windows remain visible. The historical-pool FPR
    rule is the only remedy that borrows a disjoint, past-only labelled reserve.
    """
    y_raw = np.asarray(y_raw, dtype=np.int8)
    p_raw = np.asarray(p_raw, dtype=float)
    y_hist_pool = np.asarray(y_hist_pool, dtype=np.int8)
    p_hist_pool = np.asarray(p_hist_pool, dtype=float)
    raw_two_class = len(np.unique(y_raw)) >= 2
    pool_two_class = len(np.unique(y_hist_pool)) >= 2
    mu = float(np.mean(p_raw)) if len(p_raw) else 0.5
    sd = float(np.std(p_raw)) if len(p_raw) else 0.0

    rows: List[Dict[str, Any]] = [{
        "threshold_name": "fixed_0.5",
        "threshold_value": 0.5,
        "threshold_method_eligible": True,
        "threshold_selection_status": "fixed_no_selection_required",
        "fallback_used": False,
        "historical_threshold_pool_used": False,
        "eligibility_family": "tierC_fixed_05_all",
    }]

    for k in cfg.MU_SIGMA_K_VALUES:
        raw_value = float(np.clip(mu + k * sd, 0.0, 1.0))
        rows.append({
            "threshold_name": f"mu_sigma_raw_k_{k:.2f}",
            "threshold_value": raw_value,
            "threshold_method_eligible": True,
            "threshold_selection_status": (
                "raw_rule_single_class_failure_exposed" if not raw_two_class
                else "raw_rule_two_class_window"
            ),
            "fallback_used": False,
            "historical_threshold_pool_used": False,
            "eligibility_family": "tierC_mu_sigma_raw_all",
        })
        rows.append({
            "threshold_name": f"mu_sigma_guarded_k_{k:.2f}",
            "threshold_value": 0.5 if not raw_two_class else raw_value,
            "threshold_method_eligible": True,
            "threshold_selection_status": (
                "guarded_fallback_0.5" if not raw_two_class else "guard_not_needed"
            ),
            "fallback_used": bool(not raw_two_class),
            "historical_threshold_pool_used": False,
            "eligibility_family": "tierC_mu_sigma_guarded_all",
        })

    rows.append({
        "threshold_name": "youden",
        "threshold_value": youden_threshold(y_raw, p_raw),
        "threshold_method_eligible": bool(raw_two_class),
        "threshold_selection_status": (
            "selected_on_raw_threshold_window" if raw_two_class
            else "fallback_0.5_single_class_raw_window"
        ),
        "fallback_used": bool(not raw_two_class),
        "historical_threshold_pool_used": False,
        "eligibility_family": "tierC_youden_raw_two_class",
    })

    for alpha in cfg.FPR_BUDGETS:
        pct = alpha * 100
        rows.append({
            "threshold_name": f"fpr_{pct:.1f}pct",
            "threshold_value": fpr_budget_threshold(y_raw, p_raw, alpha),
            "threshold_method_eligible": bool(raw_two_class),
            "threshold_selection_status": (
                "selected_on_raw_threshold_window" if raw_two_class
                else "fallback_0.5_single_class_raw_window"
            ),
            "fallback_used": bool(not raw_two_class),
            "historical_threshold_pool_used": False,
            "eligibility_family": "tierC_fpr_raw_two_class",
        })

        if raw_two_class:
            hist_threshold = fpr_budget_threshold(y_raw, p_raw, alpha)
            hist_status = "raw_window_already_two_class"
            hist_pool_used = False
            hist_eligible = True
        elif pool_two_class:
            hist_threshold = fpr_budget_threshold(y_hist_pool, p_hist_pool, alpha)
            hist_status = "restored_with_past_only_historical_pool"
            hist_pool_used = True
            hist_eligible = True
        else:
            hist_threshold = 0.5
            hist_status = "historical_pool_still_single_class_fallback_0.5"
            hist_pool_used = True
            hist_eligible = False
        rows.append({
            "threshold_name": f"fpr_histpool_{pct:.1f}pct",
            "threshold_value": hist_threshold,
            "threshold_method_eligible": bool(hist_eligible),
            "threshold_selection_status": hist_status,
            "fallback_used": bool(not hist_eligible),
            "historical_threshold_pool_used": hist_pool_used,
            "eligibility_family": "tierC_fpr_historical_pool",
        })
    return rows

# =============================================================================
# METRICS
# =============================================================================
def ece_mce(y: np.ndarray, p: np.ndarray, bins: int=15) -> Tuple[float,float]:
    p = np.clip(np.asarray(p,float),0,1); y=np.asarray(y,int)
    edges=np.linspace(0,1,bins+1); ids=np.digitize(p,edges[1:-1],right=True)
    ece=0.; mce=0.; n=len(y)
    for b in range(bins):
        m=ids==b
        if not np.any(m): continue
        gap=abs(float(np.mean(y[m]))-float(np.mean(p[m])))
        ece += gap*np.sum(m)/n; mce=max(mce,gap)
    return float(ece),float(mce)


def class_metrics(y: np.ndarray, pred: np.ndarray, p: np.ndarray) -> Dict[str,float]:
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    fpr=fp/(fp+tn) if fp+tn else np.nan; fnr=fn/(fn+tp) if fn+tp else np.nan
    spec=tn/(tn+fp) if tn+fp else np.nan
    auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan
    prauc=float(average_precision_score(y,p)) if len(np.unique(y))==2 else np.nan
    return {
        "accuracy":float(accuracy_score(y,pred)), "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "precision":float(precision_score(y,pred,zero_division=0)), "recall":float(recall_score(y,pred,zero_division=0)),
        "f1":float(f1_score(y,pred,zero_division=0)), "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
        "mcc":float(matthews_corrcoef(y,pred)), "specificity":float(spec), "fpr":float(fpr), "fnr":float(fnr),
        "roc_auc":auc, "pr_auc":prauc, "tn":int(tn), "fp":int(fp), "fn":int(fn), "tp":int(tp),
    }


def prob_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str,float]:
    p=np.clip(np.asarray(p,float),1e-7,1-1e-7); ece,mce=ece_mce(y,p)
    return {"brier_score":float(brier_score_loss(y,p)), "log_loss":float(log_loss(y,p,labels=[0,1])), "ece":ece, "mce":mce}


def per_attack(categories: Sequence[str], y: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    f=pd.DataFrame({"attack_category":pd.Series(categories,dtype="string").fillna("Unknown"),"y":y,"pred":pred})
    f=f[f.y==1]; rows=[]
    for cat,g in f.groupby("attack_category",dropna=False):
        total=len(g); detected=int((g.pred==1).sum())
        rows.append({"attack_category":str(cat),"attack_samples":total,"detected":detected,"missed":total-detected,"detection_rate_recall":detected/total if total else np.nan})
    return pd.DataFrame(rows)

# =============================================================================
# PLOTS
# =============================================================================
def save_cm(y: np.ndarray, pred: np.ndarray, title: str, path: Path) -> None:
    cm=confusion_matrix(y,pred,labels=[0,1]); fig,ax=plt.subplots(figsize=(5.5,4.5)); im=ax.imshow(cm)
    ax.set(title=title,xlabel="Predicted label",ylabel="True label"); ax.set_xticks([0,1],labels=["Normal","Attack"]); ax.set_yticks([0,1],labels=["Normal","Attack"])
    for i in range(2):
        for j in range(2): ax.text(j,i,f"{cm[i,j]:,}",ha="center",va="center")
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(path,dpi=300,bbox_inches="tight"); plt.close(fig)


def save_reliability(y: np.ndarray, p: np.ndarray, title: str, path: Path) -> None:
    if len(np.unique(y))<2:return
    true,pred=calibration_curve(y,p,n_bins=12,strategy="quantile"); fig,ax=plt.subplots(figsize=(5.5,4.5))
    ax.plot([0,1],[0,1],"--",label="Perfect calibration"); ax.plot(pred,true,marker="o",label="Model")
    ax.set(title=title,xlabel="Mean predicted probability",ylabel="Observed attack frequency"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path,dpi=300,bbox_inches="tight"); plt.close(fig)


def save_roc_pr(y: np.ndarray, p: np.ndarray, title: str, roc_path: Path, pr_path: Path) -> None:
    if len(np.unique(y))<2:return
    fpr,tpr,_=roc_curve(y,p); prec,rec,_=precision_recall_curve(y,p)
    fig,ax=plt.subplots(figsize=(5.5,4.5)); ax.plot(fpr,tpr); ax.plot([0,1],[0,1],"--"); ax.set(title=title+" - ROC",xlabel="FPR",ylabel="TPR"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(roc_path,dpi=300,bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.5,4.5)); ax.plot(rec,prec); ax.set(title=title+" - PR",xlabel="Recall",ylabel="Precision"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(pr_path,dpi=300,bbox_inches="tight"); plt.close(fig)


def save_threshold_curves(y: np.ndarray, p: np.ndarray, title: str, fpr_path: Path, recall_path: Path) -> None:
    ts=np.linspace(0,1,201); fprs=[]; recalls=[]
    for t in ts:
        pred=(p>=t).astype(np.int8); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
        fprs.append(fp/(fp+tn) if fp+tn else np.nan); recalls.append(tp/(tp+fn) if tp+fn else np.nan)
    fig,ax=plt.subplots(figsize=(6,4.5)); ax.plot(ts,fprs); ax.set(title=title+" - Threshold vs FPR",xlabel="Threshold",ylabel="FPR"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(fpr_path,dpi=300,bbox_inches="tight"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4.5)); ax.plot(ts,recalls); ax.set(title=title+" - Threshold vs Recall",xlabel="Threshold",ylabel="Recall"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(recall_path,dpi=300,bbox_inches="tight"); plt.close(fig)

# =============================================================================
# STATISTICS AND STABILITY
# =============================================================================
def mcnemar_exact(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> Dict[str,float]:
    ca=a==y; cb=b==y; n01=int(np.sum(ca & ~cb)); n10=int(np.sum(~ca & cb)); n=n01+n10
    p=1.0 if n==0 else float(binomtest(min(n01,n10),n=n,p=.5,alternative="two-sided").pvalue)
    return {"a_correct_b_wrong":n01,"a_wrong_b_correct":n10,"discordant_pairs":n,"p_value_exact":p}


def _metrics_from_confusion_counts(tn: int, fp: int, fn: int, tp: int) -> Dict[str, float]:
    n = tn + fp + fn + tp
    accuracy = (tn + tp) / n if n else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else 0.0
    specificity = tn / (tn + fp) if tn + fp else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f1_neg_precision = tn / (tn + fn) if tn + fn else 0.0
    f1_neg_recall = specificity if np.isfinite(specificity) else 0.0
    f1_neg = (
        2 * f1_neg_precision * f1_neg_recall / (f1_neg_precision + f1_neg_recall)
        if f1_neg_precision + f1_neg_recall else 0.0
    )
    macro_f1 = (f1 + f1_neg) / 2.0
    # Convert each factor to float *before* multiplication.  During the
    # multinomial bootstrap the counts are Python integers; their four-way
    # product can exceed int64 even though the resulting MCC is perfectly
    # representable as float64.  Passing that huge Python integer directly to
    # np.sqrt makes NumPy fall back to an object ufunc and raises:
    #   TypeError: 'int' object has no attribute 'sqrt'
    denom_sq = (
        float(tp + fp)
        * float(tp + fn)
        * float(tn + fp)
        * float(tn + fn)
    )
    denom = float(np.sqrt(denom_sq)) if denom_sq > 0.0 else 0.0
    mcc = ((float(tp) * float(tn) - float(fp) * float(fn)) / denom) if denom else 0.0
    return {
        "accuracy": float(accuracy),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "fpr": float(fpr),
        "mcc": float(mcc),
    }


def bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    p: np.ndarray,
    rounds: int,
    confidence: float,
    seed: int,
) -> Dict[str, Tuple[float, float, float]]:
    """Efficient nonparametric bootstrap for confusion-matrix metrics.

    Each observation belongs to one of four (truth,prediction) cells. Resampling
    observations with replacement is therefore equivalent to drawing the four
    cell counts from a multinomial distribution with empirical cell probabilities.
    This makes 2,000 rounds inexpensive even on million-row test sets.
    """
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    counts = np.asarray([tn, fp, fn, tp], dtype=np.int64)
    n = int(counts.sum())
    if n == 0:
        return {}
    probs = counts / n
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, probs, size=int(rounds))
    values: Dict[str, List[float]] = defaultdict(list)
    for draw in draws:
        m = _metrics_from_confusion_counts(*map(int, draw))
        for key, value in m.items():
            values[key].append(value)
    point = _metrics_from_confusion_counts(int(tn), int(fp), int(fn), int(tp))
    alpha = (1.0 - confidence) / 2.0
    out: Dict[str, Tuple[float, float, float]] = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        out[key] = (
            float(point[key]),
            float(np.nanquantile(arr, alpha)),
            float(np.nanquantile(arr, 1.0 - alpha)),
        )
    return out


def holm(pvalues: Mapping[str,float]) -> Dict[str,float]:
    items=sorted(pvalues.items(),key=lambda x:x[1]); m=len(items); out={}; running=0.
    for rank,(name,p) in enumerate(items,1):
        running=max(running,min(1.,(m-rank+1)*p)); out[name]=running
    return out


def jaccard(a: Sequence[str],b: Sequence[str]) -> float:
    sa,sb=set(a),set(b); return len(sa&sb)/len(sa|sb) if sa|sb else np.nan


def kuncheva(a: Sequence[str],b: Sequence[str],n: int) -> float:
    sa,sb=set(a),set(b); k=min(len(sa),len(sb))
    if k==0 or n<=k:return np.nan
    r=len(sa&sb); return float((r-k*k/n)/(k-k*k/n))


def rank_spearman(names_a: Sequence[str],imp_a: Sequence[float],names_b: Sequence[str],imp_b: Sequence[float]) -> float:
    ma=dict(zip(names_a,imp_a)); mb=dict(zip(names_b,imp_b)); common=sorted(set(ma)&set(mb))
    if len(common)<3:return np.nan
    va=np.asarray([ma[x] for x in common],dtype=float); vb=np.asarray([mb[x] for x in common],dtype=float)
    if np.nanstd(va)==0 or np.nanstd(vb)==0:return np.nan
    c=spearmanr(va,vb,nan_policy="omit").correlation
    return float(c) if c is not None else np.nan

# =============================================================================
# EVALUATION
# =============================================================================
def _infer_session_boundary(df: pd.DataFrame) -> Dict[str, Any]:
    """Identify the dominant collection-session gap from chronological timestamps."""
    if "__time_order__" not in df.columns or len(df) < 2:
        return {"boundary_index": None, "max_gap_hours": np.nan}
    ts = pd.to_datetime(df["__time_order__"], errors="coerce", utc=True)
    valid = ts.notna().to_numpy()
    if valid.sum() < 2:
        return {"boundary_index": None, "max_gap_hours": np.nan}
    seconds = ts.astype("int64").to_numpy(dtype=np.int64) / 1e9
    diffs = np.diff(seconds)
    diffs[~(valid[:-1] & valid[1:])] = np.nan
    if not np.isfinite(diffs).any():
        return {"boundary_index": None, "max_gap_hours": np.nan}
    pos = int(np.nanargmax(diffs))
    max_gap_hours = float(diffs[pos] / 3600.0)
    # A one-day gap is vastly above within-session gaps in this dataset and keeps
    # the logic generic enough to avoid hard-coding the original dates.
    boundary_index = pos + 1 if max_gap_hours >= 24.0 else None
    return {"boundary_index": boundary_index, "max_gap_hours": max_gap_hours}


def _split_time_metadata(df: pd.DataFrame, split: DataSplit) -> Dict[str, Any]:
    boundary = _infer_session_boundary(df)
    boundary_idx = boundary["boundary_index"]

    def times(indices: np.ndarray) -> Tuple[str, str]:
        if len(indices) == 0 or "__time_order__" not in df.columns:
            return "", ""
        return str(df.iloc[indices[0]]["__time_order__"]), str(df.iloc[indices[-1]]["__time_order__"])

    train_start, train_end = times(split.train_idx)
    test_start, test_end = times(split.test_idx)
    gap_hours = np.nan
    try:
        gap_hours = float((pd.Timestamp(test_start) - pd.Timestamp(train_end)).total_seconds() / 3600.0)
    except Exception:
        pass

    transition = "unknown"
    train_session_composition = "unknown"
    test_session = "unknown"
    clean_cross = False
    if boundary_idx is not None:
        train_before = int(np.sum(split.train_idx < boundary_idx))
        train_after = int(np.sum(split.train_idx >= boundary_idx))
        test_before = int(np.sum(split.test_idx < boundary_idx))
        test_after = int(np.sum(split.test_idx >= boundary_idx))
        if train_before and not train_after:
            train_session_composition = "session_1_only"
        elif train_after and not train_before:
            train_session_composition = "session_2_only"
        elif train_before and train_after:
            train_session_composition = "mixed_sessions"
        if test_before and not test_after:
            test_session = "session_1"
        elif test_after and not test_before:
            test_session = "session_2"
        else:
            test_session = "mixed_sessions"

        if train_session_composition == "session_1_only" and test_session == "session_1":
            transition = "within_session_1"
        elif train_session_composition == "session_1_only" and test_session == "session_2":
            transition = "clean_cross_session"
            clean_cross = True
        elif train_session_composition == "mixed_sessions" and test_session == "session_2":
            transition = "mixed_training_to_session_2"
        elif train_session_composition == "session_2_only" and test_session == "session_2":
            transition = "within_session_2"
        else:
            transition = "mixed_other"

    return {
        "train_start_time": train_start,
        "train_end_time": train_end,
        "test_start_time": test_start,
        "test_end_time": test_end,
        "train_to_test_gap_hours": gap_hours,
        "largest_dataset_session_gap_hours": boundary["max_gap_hours"],
        "session_boundary_row_index": boundary_idx,
        "train_session_composition": train_session_composition,
        "test_session": test_session,
        "transition_type": transition,
        "cross_session_boundary": bool(clean_cross),
    }


def evaluate_variant(
    protocol: str,
    fold_id: str,
    candidate_number: Optional[int],
    variant: str,
    xtr: np.ndarray,
    xcal: np.ndarray,
    xval: np.ndarray,
    xthreshold_pool: np.ndarray,
    xtest: np.ndarray,
    ytr: np.ndarray,
    ycal: np.ndarray,
    yval: np.ndarray,
    ythreshold_pool: np.ndarray,
    ytest: np.ndarray,
    attack_test: Sequence[str],
    cfg: Config,
    dirs: Mapping[str, Path],
    offset: int,
):
    metrics_rows: List[Dict[str, Any]] = []
    prob_rows: List[Dict[str, Any]] = []
    attack_tables: List[pd.DataFrame] = []
    main: Dict[str, Any] = {}

    train_class_count = int(np.unique(ytr).size)
    calibration_class_count = int(np.unique(ycal).size)
    threshold_class_count = int(np.unique(yval).size)
    threshold_pool_class_count = int(np.unique(ythreshold_pool).size)
    test_class_count = int(np.unique(ytest).size)
    tier_a = train_class_count >= 2 and test_class_count >= 2
    tier_b = tier_a and calibration_class_count >= 2

    for mi, model_name in enumerate(cfg.RUN_MODELS):
        model = make_model(model_name, cfg, offset + mi)
        t0 = time.perf_counter()
        model.fit(xtr, ytr)
        train_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        bundles = probability_bundles(
            model_name, model, xcal, ycal, xval, xthreshold_pool, xtest
        )
        infer_sec = time.perf_counter() - t0

        for bundle in bundles:
            paper_label = PAPER_VARIANT_LABELS.get(variant, variant)
            probability_requires_calibration = (
                (model_name == "RF" and bundle.name in {"sigmoid", "isotonic"})
                or model_name == "LinearSVM"
            )
            probability_eligible = bool(
                tier_a and (not probability_requires_calibration or tier_b)
            )
            pm = prob_metrics(ytest, bundle.test_prob)
            prob_rows.append({
                "protocol": protocol,
                "fold_id": fold_id,
                "candidate_number": candidate_number,
                "variant": variant,
                "paper_variant_label": paper_label,
                "model": model_name,
                "probability_source": bundle.name,
                "train_class_count": train_class_count,
                "calibration_class_count": calibration_class_count,
                "test_class_count": test_class_count,
                "tier_A_discrimination_eligible": tier_a,
                "tier_B_calibration_eligible": tier_b,
                "probability_evaluation_eligible": probability_eligible,
                "probability_requires_calibration": probability_requires_calibration,
                "roc_auc": float(roc_auc_score(ytest, bundle.test_prob)) if test_class_count >= 2 else np.nan,
                "pr_auc": float(average_precision_score(ytest, bundle.test_prob)) if test_class_count >= 2 else np.nan,
                **pm,
            })

            rules = threshold_rules(
                yval, bundle.val_prob,
                ythreshold_pool, bundle.threshold_pool_prob,
                cfg,
            )
            for rule in rules:
                th_name = str(rule["threshold_name"])
                th = float(rule["threshold_value"])
                pred = (bundle.test_prob >= th).astype(np.int8)
                m = class_metrics(ytest, pred, bundle.test_prob)
                threshold_method_eligible = bool(rule["threshold_method_eligible"])
                aggregation_eligible = bool(probability_eligible and threshold_method_eligible)
                metrics_rows.append({
                    "protocol": protocol,
                    "fold_id": fold_id,
                    "candidate_number": candidate_number,
                    "variant": variant,
                    "paper_variant_label": paper_label,
                    "model": model_name,
                    "probability_source": bundle.name,
                    "threshold_name": th_name,
                    "threshold_value": th,
                    "threshold_selection_status": rule["threshold_selection_status"],
                    "fallback_used": bool(rule["fallback_used"]),
                    "historical_threshold_pool_used": bool(rule["historical_threshold_pool_used"]),
                    "eligibility_family": rule["eligibility_family"],
                    "threshold_method_eligible": threshold_method_eligible,
                    "aggregation_eligible": aggregation_eligible,
                    "train_class_count": train_class_count,
                    "calibration_class_count": calibration_class_count,
                    "threshold_selection_class_count": threshold_class_count,
                    "historical_threshold_pool_class_count": threshold_pool_class_count,
                    "test_class_count": test_class_count,
                    "tier_A_discrimination_eligible": tier_a,
                    "tier_B_calibration_eligible": tier_b,
                    "tier_C_fixed_05_eligible": bool(probability_eligible),
                    "tier_C_youden_eligible": bool(probability_eligible and threshold_class_count >= 2),
                    "tier_C_fpr_budget_eligible": bool(probability_eligible and threshold_class_count >= 2),
                    "tier_C_fpr_historical_pool_eligible": bool(probability_eligible and threshold_pool_class_count >= 2),
                    "threshold_selection_attack_rate": float(np.mean(yval)) if len(yval) else np.nan,
                    "historical_threshold_pool_attack_rate": float(np.mean(ythreshold_pool)) if len(ythreshold_pool) else np.nan,
                    "test_window_attack_rate": float(np.mean(ytest)) if len(ytest) else np.nan,
                    "mu_sigma_guard_applied": bool(
                        th_name.startswith("mu_sigma_guarded_") and threshold_class_count < 2
                    ),
                    "n_train": len(ytr),
                    "n_calibration": len(ycal),
                    "n_threshold_validation": len(yval),
                    "n_historical_threshold_pool": len(ythreshold_pool),
                    "n_test": len(ytest),
                    "n_features": xtr.shape[1],
                    "train_seconds": train_sec,
                    "probability_inference_seconds": infer_sec,
                    **m,
                })

                at = per_attack(attack_test, ytest, pred)
                if not at.empty:
                    for key, value in reversed([
                        ("protocol", protocol),
                        ("fold_id", fold_id),
                        ("candidate_number", candidate_number),
                        ("variant", variant),
                        ("paper_variant_label", paper_label),
                        ("model", model_name),
                        ("probability_source", bundle.name),
                        ("threshold_name", th_name),
                    ]):
                        at.insert(0, key, value)
                    attack_tables.append(at)

                is_main = (
                    model_name == cfg.MAIN_MODEL
                    and bundle.name == cfg.MAIN_PROBABILITY_SOURCE
                    and th_name == cfg.MAIN_THRESHOLD_NAME
                )
                if is_main:
                    main = {
                        "y_true": ytest.copy(),
                        "y_pred": pred.copy(),
                        "probabilities": bundle.test_prob.copy(),
                        "threshold": th,
                        "metrics": m,
                        "bootstrap": bootstrap_ci(
                            ytest, pred, bundle.test_prob,
                            cfg.BOOTSTRAP_ROUNDS,
                            cfg.BOOTSTRAP_CONFIDENCE,
                            cfg.RANDOM_STATE + offset,
                        ),
                    }
                    if cfg.SAVE_MAIN_PREDICTIONS:
                        pd.DataFrame({
                            "y_true": ytest,
                            "y_pred": pred,
                            "attack_probability": bundle.test_prob,
                            "attack_category": attack_test,
                        }).to_csv(
                            dirs["predictions"] / safe_filename(
                                f"{protocol}_{fold_id}_{variant}_main.csv"
                            ), index=False
                        )
                    if cfg.SAVE_DETAILED_PLOTS:
                        stem = safe_filename(
                            f"{protocol}_{fold_id}_{variant}_{model_name}_{bundle.name}_{th_name}"
                        )
                        title = (
                            f"{protocol} | {fold_id} | {variant} | {model_name} | "
                            f"{bundle.name} | {th_name}"
                        )
                        save_cm(ytest, pred, title, dirs["plots"] / f"{stem}_CM.png")
                        save_reliability(ytest, bundle.test_prob, title, dirs["plots"] / f"{stem}_Reliability.png")
                        save_roc_pr(
                            ytest, bundle.test_prob, title,
                            dirs["plots"] / f"{stem}_ROC.png",
                            dirs["plots"] / f"{stem}_PR.png",
                        )
                        save_threshold_curves(
                            yval, bundle.val_prob, title,
                            dirs["plots"] / f"{stem}_Threshold_FPR.png",
                            dirs["plots"] / f"{stem}_Threshold_Recall.png",
                        )
    return metrics_rows, prob_rows, attack_tables, main


def evaluate_split(
    df: pd.DataFrame,
    split: DataSplit,
    base_cols: Mapping[str, Sequence[str]],
    cfg: Config,
    dirs: Mapping[str, Path],
    log_path: Path,
    rfe_history: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    log(f"Evaluating {split.protocol}/{split.fold_id}", log_path)

    train = df.iloc[split.train_idx].copy()
    calibration = df.iloc[split.calibration_idx].copy()
    threshold = df.iloc[split.threshold_idx].copy()
    threshold_pool = df.iloc[split.threshold_pool_idx].copy()
    test = df.iloc[split.test_idx].copy()

    duplicate_stats = {
        "train_duplicates_removed": 0,
        "calibration_duplicates_removed": 0,
        "threshold_duplicates_removed": 0,
        "test_duplicates_removed": 0,
    }
    if cfg.ENFORCE_TEMPORAL_DUPLICATE_ISOLATION:
        log(
            "Temporal duplicate isolation is enabled, but final methodology recommends False; "
            "historical threshold-pool duplicates are not independently removed.",
            log_path,
        )
        hash_cols = list(base_cols.get("Strict-Clean", base_cols["Clean-All"]))
        train, calibration, threshold, test, duplicate_stats = remove_temporal_duplicates(
            train, calibration, threshold, test, hash_cols,
            split.protocol not in {"random_stratified"},
        )

    if train["__binary_label__"].nunique() < 2:
        log(f"Skipping {split.fold_id}: training partition has one class", log_path)
        return {"skipped": True, "reason": "single_class_training"}
    if len(test) == 0:
        return {"skipped": True, "reason": "empty_test"}

    y_train = train["__binary_label__"].to_numpy(np.int8)
    y_cal = calibration["__binary_label__"].to_numpy(np.int8)
    y_threshold = threshold["__binary_label__"].to_numpy(np.int8)
    y_threshold_pool = threshold_pool["__binary_label__"].to_numpy(np.int8)
    y_test = test["__binary_label__"].to_numpy(np.int8)
    attack_test = test["__attack_category__"].astype(str).to_numpy()

    all_metrics: List[Dict[str, Any]] = []
    all_probability: List[Dict[str, Any]] = []
    all_attack: List[pd.DataFrame] = []
    all_mcnemar: List[Dict[str, Any]] = []
    all_bootstrap: List[Dict[str, Any]] = []
    main_by_variant: Dict[str, Dict[str, Any]] = {}

    for base_index, base in enumerate(cfg.BASE_VARIANT_ORDER):
        cols = list(base_cols.get(base, []))
        if not cols:
            continue
        x_train, x_cal, x_threshold, x_threshold_pool, x_test, names = fit_transform(
            train, calibration, threshold, threshold_pool, test, cols
        )
        variants = [(base, np.arange(x_train.shape[1]))]

        if cfg.RUN_RFE_VARIANTS:
            # Fold number is deliberately absent from both seeds.
            sample_seed = cfg.RFE_FIXED_SAMPLE_SEED + base_index
            model_seed = cfg.RFE_FIXED_MODEL_SEED + base_index
            rfe_result = run_rfe(
                x_train, y_train, names, cfg,
                sampling_mode=cfg.FULL_RFE_SAMPLING_MODE,
                sample_seed=sample_seed,
                model_seed=model_seed,
            )
            rfe_name = "RFE-TopK" if base == "Clean-All" else f"{base}-RFE-TopK"
            variants.append((rfe_name, rfe_result.selected))

            if split.protocol in {"sliding_temporal", "expanding_temporal"}:
                history_key = f"{split.protocol}::{base}"
                rfe_history[history_key].append({
                    "protocol": split.protocol,
                    "training_mode": split.training_mode,
                    "fold_id": split.fold_id,
                    "candidate_number": split.candidate_number,
                    "feature_names": names,
                    "selected_names": rfe_result.selected_names,
                    "ranking": rfe_result.ranking,
                    "importances": rfe_result.importances,
                    "sampling_mode": rfe_result.sampling_mode,
                    "sample_seed": rfe_result.sample_seed,
                    "model_seed": rfe_result.model_seed,
                    "positives_available": rfe_result.positives_available,
                    "negatives_available": rfe_result.negatives_available,
                    "positives_selected": rfe_result.positives_selected,
                    "negatives_selected": rfe_result.negatives_selected,
                    "sample_attack_rate": rfe_result.sample_attack_rate,
                    "positive_cap_applied": rfe_result.positive_cap_applied,
                })

            ranking_df = pd.DataFrame({
                "feature": names,
                "rfe_rank": rfe_result.ranking,
                "rf_importance": rfe_result.importances,
                "selected": np.isin(np.arange(len(names)), rfe_result.selected),
            })
            metadata_columns = [
                ("protocol", split.protocol),
                ("fold_id", split.fold_id),
                ("candidate_number", split.candidate_number),
                ("base_variant", base),
                ("paper_variant_label", PAPER_VARIANT_LABELS.get(base, base)),
                ("rfe_sampling_mode", rfe_result.sampling_mode),
                ("rfe_sample_seed", rfe_result.sample_seed),
                ("rfe_model_seed", rfe_result.model_seed),
                ("positives_available", rfe_result.positives_available),
                ("negatives_available", rfe_result.negatives_available),
                ("positives_selected", rfe_result.positives_selected),
                ("negatives_selected", rfe_result.negatives_selected),
                ("sample_attack_rate", rfe_result.sample_attack_rate),
                ("positive_cap_applied", rfe_result.positive_cap_applied),
            ]
            for key, value in reversed(metadata_columns):
                ranking_df.insert(0, key, value)
            ranking_df.to_csv(
                dirs["csv"] / safe_filename(
                    f"{split.protocol}_{split.fold_id}_{base}_RFE_Ranking.csv"
                ), index=False
            )

        for vi, (variant_name, selected_idx) in enumerate(variants):
            xv_tr = x_train[:, selected_idx]
            xv_cal = x_cal[:, selected_idx]
            xv_thr = x_threshold[:, selected_idx]
            xv_thr_pool = x_threshold_pool[:, selected_idx]
            xv_test = x_test[:, selected_idx]
            result = evaluate_variant(
                split.protocol, split.fold_id, split.candidate_number,
                variant_name,
                xv_tr, xv_cal, xv_thr, xv_thr_pool, xv_test,
                y_train, y_cal, y_threshold, y_threshold_pool, y_test,
                attack_test, cfg, dirs,
                offset=10_000 + base_index * 100 + vi * 10,
            )
            metrics_rows, prob_rows, attack_tables, main = result
            all_metrics.extend(metrics_rows)
            all_probability.extend(prob_rows)
            all_attack.extend(attack_tables)
            if main:
                main_by_variant[variant_name] = main
                for metric, (point, lower, upper) in main["bootstrap"].items():
                    all_bootstrap.append({
                        "protocol": split.protocol,
                        "fold_id": split.fold_id,
                        "candidate_number": split.candidate_number,
                        "variant": variant_name,
                        "model": cfg.MAIN_MODEL,
                        "probability_source": cfg.MAIN_PROBABILITY_SOURCE,
                        "threshold_name": cfg.MAIN_THRESHOLD_NAME,
                        "metric": metric,
                        "point_estimate": point,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "confidence": cfg.BOOTSTRAP_CONFIDENCE,
                        "bootstrap_rounds": cfg.BOOTSTRAP_ROUNDS,
                    })
            gc.collect()

    comparisons = (
        ("Clean-All", "No-TTL"),
        ("Clean-All", "Strict-Clean"),
        ("Strict-Clean", "Strict-Clean-No-TTL"),
        ("No-TTL", "Strict-Clean-No-TTL"),
        ("RFE-TopK", "Strict-Clean-RFE-TopK"),
        ("Strict-Clean", "Strict-Clean-RFE-TopK"),
        ("Strict-Clean-RFE-TopK", "Strict-Clean-No-TTL-RFE-TopK"),
    )
    for variant_a, variant_b in comparisons:
        if variant_a not in main_by_variant or variant_b not in main_by_variant:
            continue
        aa, bb = main_by_variant[variant_a], main_by_variant[variant_b]
        all_mcnemar.append({
            "protocol": split.protocol,
            "fold_id": split.fold_id,
            "candidate_number": split.candidate_number,
            "comparison": f"{variant_a}_vs_{variant_b}",
            "variant_a": variant_a,
            "variant_b": variant_b,
            "model": cfg.MAIN_MODEL,
            "probability_source": cfg.MAIN_PROBABILITY_SOURCE,
            "threshold_name": cfg.MAIN_THRESHOLD_NAME,
            **mcnemar_exact(aa["y_true"], aa["y_pred"], bb["y_pred"]),
        })

    time_meta = _split_time_metadata(df, split)
    summary = {
        "protocol": split.protocol,
        "training_mode": split.training_mode,
        "fold_id": split.fold_id,
        "candidate_number": split.candidate_number,
        "partition_mode": "past_only_disjoint_calibration_and_threshold_support_then_future_test",
        "calibration_strategy": split.calibration_strategy,
        "threshold_pool_strategy": split.threshold_pool_strategy,
        "calibration_reserve_rows": split.calibration_reserve_rows,
        "threshold_reserve_rows": split.threshold_reserve_rows,
        "train_window_ids": ",".join(map(str, split.train_window_ids)),
        "calibration_window_id": split.calibration_window_id,
        "threshold_window_id": split.threshold_window_id,
        "test_window_id": split.test_window_id,
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "threshold_validation_rows": len(threshold),
        "historical_threshold_pool_rows": len(threshold_pool),
        "test_rows": len(test),
        "train_attack_rate": float(train["__binary_label__"].mean()),
        "calibration_attack_rate": float(calibration["__binary_label__"].mean()),
        "threshold_attack_rate": float(threshold["__binary_label__"].mean()),
        "historical_threshold_pool_attack_rate": float(threshold_pool["__binary_label__"].mean()),
        "test_attack_rate": float(test["__binary_label__"].mean()),
        "train_class_count": int(train["__binary_label__"].nunique()),
        "calibration_class_count": int(calibration["__binary_label__"].nunique()),
        "threshold_class_count": int(threshold["__binary_label__"].nunique()),
        "historical_threshold_pool_class_count": int(threshold_pool["__binary_label__"].nunique()),
        "test_class_count": int(test["__binary_label__"].nunique()),
        "tier_A_discrimination_eligible": bool(train["__binary_label__"].nunique() >= 2 and test["__binary_label__"].nunique() >= 2),
        "tier_B_calibration_eligible": bool(train["__binary_label__"].nunique() >= 2 and calibration["__binary_label__"].nunique() >= 2 and test["__binary_label__"].nunique() >= 2),
        "tier_C_raw_two_class_threshold_eligible": bool(threshold["__binary_label__"].nunique() >= 2),
        "tier_C_historical_pool_threshold_eligible": bool(threshold_pool["__binary_label__"].nunique() >= 2),
        **time_meta,
        **duplicate_stats,
    }
    return {
        "skipped": False,
        "metrics": all_metrics,
        "probability": all_probability,
        "attack_tables": all_attack,
        "mcnemar": all_mcnemar,
        "bootstrap": all_bootstrap,
        "summary": summary,
    }

# =============================================================================
# FINAL AGGREGATION
# =============================================================================
def feature_stability(
    history: Mapping[str, List[Dict[str, Any]]],
    dirs: Mapping[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute descriptive within-protocol temporal stability and stable cores."""
    stability_rows: List[Dict[str, Any]] = []
    frequency_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    def fold_number(item: Mapping[str, Any]) -> int:
        try:
            return int(item.get("candidate_number") or str(item["fold_id"]).split("_")[-1])
        except Exception:
            return 0

    for key, raw_items in history.items():
        if not raw_items:
            continue
        protocol, base = key.split("::", 1)
        items = sorted(raw_items, key=fold_number)
        counts: Counter[str] = Counter()
        selected_sets: List[set] = []
        for item in items:
            counts.update(item["selected_names"])
            selected_sets.append(set(item["selected_names"]))
        between_core = set.intersection(*selected_sets) if selected_sets else set()
        n_folds = len(items)
        for feature, count in counts.items():
            freq = count / n_folds
            frequency_rows.append({
                "protocol": protocol,
                "training_mode": protocol.replace("_temporal", ""),
                "base_variant": base,
                "paper_variant_label": PAPER_VARIANT_LABELS.get(base, base),
                "feature": feature,
                "selected_count": count,
                "n_folds": n_folds,
                "selection_frequency": freq,
                "selected_in_100pct_folds": bool(count == n_folds),
                "selected_in_at_least_80pct_folds": bool(freq >= 0.80),
                "selected_in_at_least_60pct_folds": bool(freq >= 0.60),
                "rfe_sampling_mode": items[0].get("sampling_mode", "unknown"),
                "rfe_sample_seed": items[0].get("sample_seed", np.nan),
                "rfe_model_seed": items[0].get("model_seed", np.nan),
            })

        local_pairs: List[Dict[str, Any]] = []
        for (index_a, item_a), (index_b, item_b) in combinations(
            list(enumerate(items)), 2
        ):
            universe = len(set(item_a["feature_names"]) | set(item_b["feature_names"]))
            row = {
                "protocol": protocol,
                "training_mode": protocol.replace("_temporal", ""),
                "base_variant": base,
                "paper_variant_label": PAPER_VARIANT_LABELS.get(base, base),
                "fold_a": item_a["fold_id"],
                "fold_b": item_b["fold_id"],
                "candidate_a": item_a.get("candidate_number"),
                "candidate_b": item_b.get("candidate_number"),
                "candidate_distance": abs(fold_number(item_b) - fold_number(item_a)),
                "is_consecutive_valid_fold": bool(index_b - index_a == 1),
                "jaccard": jaccard(item_a["selected_names"], item_b["selected_names"]),
                "kuncheva": kuncheva(item_a["selected_names"], item_b["selected_names"], universe),
                "spearman_importance": rank_spearman(
                    item_a["feature_names"], item_a["importances"],
                    item_b["feature_names"], item_b["importances"],
                ),
                "rfe_sampling_mode": item_a.get("sampling_mode", "unknown"),
                "rfe_sample_seed": item_a.get("sample_seed", np.nan),
                "rfe_model_seed": item_a.get("model_seed", np.nan),
            }
            stability_rows.append(row)
            local_pairs.append(row)

        local = pd.DataFrame(local_pairs)
        summary_rows.append({
            "protocol": protocol,
            "training_mode": protocol.replace("_temporal", ""),
            "base_variant": base,
            "paper_variant_label": PAPER_VARIANT_LABELS.get(base, base),
            "n_folds": n_folds,
            "n_dependent_pairwise_comparisons": len(local),
            "mean_jaccard": float(local.jaccard.mean()) if not local.empty else np.nan,
            "median_jaccard": float(local.jaccard.median()) if not local.empty else np.nan,
            "min_jaccard": float(local.jaccard.min()) if not local.empty else np.nan,
            "max_jaccard": float(local.jaccard.max()) if not local.empty else np.nan,
            "mean_kuncheva": float(local.kuncheva.mean()) if not local.empty else np.nan,
            "mean_spearman": float(local.spearman_importance.mean()) if not local.empty else np.nan,
            "between_core_size": len(between_core),
            "between_core_ratio": len(between_core) / max(1, len(items[0]["selected_names"])),
            "between_core_features": "|".join(sorted(between_core)),
            "descriptive_only": True,
            "dependence_note": "pairwise feature-set similarities are not independent observations",
        })

    stability_df = pd.DataFrame(stability_rows)
    frequency_df = pd.DataFrame(frequency_rows)
    summary_df = pd.DataFrame(summary_rows)
    stability_df.to_csv(dirs["csv"] / "RFE_Feature_Stability_All_Protocols.csv", index=False)
    frequency_df.to_csv(dirs["csv"] / "RFE_Feature_Selection_Frequency_All_Protocols.csv", index=False)
    summary_df.to_csv(dirs["csv"] / "RFE_Feature_Stability_Summary_All_Protocols.csv", index=False)
    return stability_df, frequency_df, summary_df


def _paired_wilcoxon_row(
    a: np.ndarray,
    b: np.ndarray,
    metadata: Dict[str, Any],
    zero_tolerance: float = 1e-15,
) -> Dict[str, Any]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    diff = a - b
    nonzero = np.abs(diff) > zero_tolerance
    effective = diff[nonzero]
    if len(effective) >= 2:
        try:
            test = wilcoxon(effective, zero_method="wilcox", alternative="two-sided")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        except ValueError:
            statistic, p_value = 0.0, 1.0
    else:
        statistic, p_value = 0.0, 1.0
    return {
        **metadata,
        "n_total_paired_folds": int(len(diff)),
        "n_zero_differences": int((~nonzero).sum()),
        "n_effective_after_dropping_zeros": int(len(effective)),
        "positive_difference_count": int((effective > 0).sum()),
        "negative_difference_count": int((effective < 0).sum()),
        "mean_difference_a_minus_b": float(np.mean(diff)) if len(diff) else np.nan,
        "median_difference_a_minus_b": float(np.median(diff)) if len(diff) else np.nan,
        "wilcoxon_statistic": statistic,
        "p_value_raw": p_value,
    }



def preregistered_holm_plan(cfg: Config) -> pd.DataFrame:
    """Primary inferential families fixed before the full run.

    Each Holm family contains at most three tests. Sliding-only effects remain
    descriptive (effect sizes/bootstrap CIs); inferential variant/calibration/
    threshold tests are centered on the expanding protocol because it has the
    larger operational fold set. A separate family compares protocols only on
    identical future test windows.
    """
    rows: List[Dict[str, Any]] = []

    def add(family_id: str, test_id: str, table: str, protocol: str,
            comparison_type: str, metric: str, a: str, b: str,
            family_role: str, note: str = "") -> None:
        rows.append({
            "family_id": family_id,
            "test_id": test_id,
            "table": table,
            "protocol": protocol,
            "comparison_type": comparison_type,
            "metric": metric,
            "level_a": a,
            "level_b": b,
            "family_role": family_role,
            "planned_max_family_size": 3,
            "note": note,
        })

    # Family 1: core variant hypotheses, primary metric chosen to avoid the
    # strongest ceiling saturation seen in ordinary binary F1.
    add("F1_expanding_variants_macroF1", "F1a_shortcut_exposure",
        "metrics", "expanding_temporal", "variant", "macro_f1",
        "Clean-All", "Strict-Clean", "primary",
        "Shortcut-exposed arm vs strict-clean baseline.")
    add("F1_expanding_variants_macroF1", "F1b_lightweight_cost",
        "metrics", "expanding_temporal", "variant", "macro_f1",
        "Strict-Clean", "Strict-Clean-RFE-TopK", "primary",
        "Strict-clean baseline vs proposed lightweight RFE variant.")
    add("F1_expanding_variants_macroF1", "F1c_ttl_ablation",
        "metrics", "expanding_temporal", "variant", "macro_f1",
        "Strict-Clean", "Strict-Clean-No-TTL", "primary",
        "TTL ablation within strict-clean predictors.")

    # Family 2: calibration reliability; ECE is the preregistered primary
    # calibration endpoint. Other calibration metrics remain descriptive.
    add("F2_expanding_calibration_ECE", "F2a_uncal_vs_sigmoid",
        "probability", "expanding_temporal", "probability_source", "ece",
        "uncalibrated", "sigmoid", "primary")
    add("F2_expanding_calibration_ECE", "F2b_uncal_vs_isotonic",
        "probability", "expanding_temporal", "probability_source", "ece",
        "uncalibrated", "isotonic", "primary")
    add("F2_expanding_calibration_ECE", "F2c_sigmoid_vs_isotonic",
        "probability", "expanding_temporal", "probability_source", "ece",
        "sigmoid", "isotonic", "primary")

    # Family 3: operational threshold behavior; FPR is the primary endpoint.
    add("F3_expanding_threshold_FPR", "F3a_fixed_vs_youden",
        "metrics", "expanding_temporal", "threshold_name", "fpr",
        "fixed_0.5", "youden", "primary")
    add("F3_expanding_threshold_FPR", "F3b_fixed_vs_histpool_1pct",
        "metrics", "expanding_temporal", "threshold_name", "fpr",
        "fixed_0.5", "fpr_histpool_1.0pct", "primary",
        "Tests whether the historical reserve changes future false alarms.")
    add("F3_expanding_threshold_FPR", "F3c_raw_vs_histpool_1pct",
        "metrics", "expanding_temporal", "threshold_name", "fpr",
        "fpr_1.0pct", "fpr_histpool_1.0pct", "primary",
        "Paired only where both methods are operationally eligible.")

    # Family 4: protocol comparison on exactly common future test windows.
    add("F4_common_protocols", "F4a_protocol_macroF1",
        "metrics", "sliding_vs_expanding", "protocol", "macro_f1",
        "sliding_temporal", "expanding_temporal", "primary")
    add("F4_common_protocols", "F4b_protocol_FPR",
        "metrics", "sliding_vs_expanding", "protocol", "fpr",
        "sliding_temporal", "expanding_temporal", "primary")
    add("F4_common_protocols", "F4c_protocol_ECE",
        "probability", "sliding_vs_expanding", "protocol", "ece",
        "sliding_temporal", "expanding_temporal", "primary")

    plan = pd.DataFrame(rows)
    counts = plan.groupby("family_id").size()
    if (counts > 3).any():
        raise RuntimeError("A preregistered Holm family exceeds the 3-test limit")
    return plan


def _paired_from_plan(
    plan_row: Mapping[str, Any],
    metrics: pd.DataFrame,
    probability: pd.DataFrame,
    cfg: Config,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    table = str(plan_row["table"])
    comparison_type = str(plan_row["comparison_type"])
    protocol = str(plan_row["protocol"])
    metric = str(plan_row["metric"])
    level_a = str(plan_row["level_a"])
    level_b = str(plan_row["level_b"])

    if table == "metrics":
        d = metrics.copy()
        d = d[
            (d.model == cfg.MAIN_MODEL)
            & (d.aggregation_eligible == True)
        ]
        # Variant comparisons use the main probability and decision rule.
        if comparison_type == "variant":
            d = d[
                (d.protocol == protocol)
                & (d.probability_source == cfg.MAIN_PROBABILITY_SOURCE)
                & (d.threshold_name == cfg.MAIN_THRESHOLD_NAME)
                & (d.variant.isin([level_a, level_b]))
            ]
            pivot = d.pivot_table(
                index="candidate_number", columns="variant",
                values=metric, aggfunc="first",
            )
        elif comparison_type == "threshold_name":
            d = d[
                (d.protocol == protocol)
                & (d.variant == cfg.MAIN_VARIANT)
                & (d.probability_source == cfg.MAIN_PROBABILITY_SOURCE)
                & (d.threshold_name.isin([level_a, level_b]))
            ]
            pivot = d.pivot_table(
                index="candidate_number", columns="threshold_name",
                values=metric, aggfunc="first",
            )
        elif comparison_type == "protocol":
            d = d[
                (d.variant == cfg.MAIN_VARIANT)
                & (d.probability_source == cfg.MAIN_PROBABILITY_SOURCE)
                & (d.threshold_name == cfg.MAIN_THRESHOLD_NAME)
                & (d.protocol.isin([level_a, level_b]))
            ]
            pivot = d.pivot_table(
                index="candidate_number", columns="protocol",
                values=metric, aggfunc="first",
            )
        else:
            return None

    elif table == "probability":
        d = probability.copy()
        d = d[
            (d.model == cfg.MAIN_MODEL)
            & (d.probability_evaluation_eligible == True)
        ]
        if comparison_type == "probability_source":
            d = d[
                (d.protocol == protocol)
                & (d.variant == cfg.MAIN_VARIANT)
                & (d.probability_source.isin([level_a, level_b]))
            ]
            pivot = d.pivot_table(
                index="candidate_number", columns="probability_source",
                values=metric, aggfunc="first",
            )
        elif comparison_type == "protocol":
            d = d[
                (d.variant == cfg.MAIN_VARIANT)
                & (d.probability_source == cfg.MAIN_PROBABILITY_SOURCE)
                & (d.protocol.isin([level_a, level_b]))
            ]
            pivot = d.pivot_table(
                index="candidate_number", columns="protocol",
                values=metric, aggfunc="first",
            )
        else:
            return None
    else:
        return None

    if not {level_a, level_b}.issubset(pivot.columns):
        return None
    paired = pivot[[level_a, level_b]].dropna()
    if len(paired) < 2:
        return None

    metadata = {
        "family_id": str(plan_row["family_id"]),
        "test_id": str(plan_row["test_id"]),
        "family_role": str(plan_row["family_role"]),
        "comparison_type": comparison_type,
        "protocol": protocol,
        "comparison": f"{level_a}_vs_{level_b}",
        "variant_a": level_a if comparison_type == "variant" else cfg.MAIN_VARIANT,
        "variant_b": level_b if comparison_type == "variant" else cfg.MAIN_VARIANT,
        "level_a": level_a,
        "level_b": level_b,
        "metric": metric,
        "planned_max_family_size": int(plan_row["planned_max_family_size"]),
        "preregistered_note": str(plan_row.get("note", "")),
    }
    return paired[level_a].to_numpy(), paired[level_b].to_numpy(), metadata


def across_window_tests(
    metrics: pd.DataFrame,
    probability: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Run only the preregistered primary paired tests.

    Holm correction is applied *within* each predeclared family (<=3 tests).
    Sliding-only model/variant comparisons are intentionally descriptive; the
    expanding protocol carries the primary within-protocol inferential tests.
    """
    plan = preregistered_holm_plan(cfg)
    rows: List[Dict[str, Any]] = []

    for _, plan_row in plan.iterrows():
        paired = _paired_from_plan(plan_row, metrics, probability, cfg)
        if paired is None:
            rows.append({
                "family_id": plan_row["family_id"],
                "test_id": plan_row["test_id"],
                "family_role": plan_row["family_role"],
                "comparison_type": plan_row["comparison_type"],
                "protocol": plan_row["protocol"],
                "comparison": f'{plan_row["level_a"]}_vs_{plan_row["level_b"]}',
                "variant_a": plan_row["level_a"],
                "variant_b": plan_row["level_b"],
                "level_a": plan_row["level_a"],
                "level_b": plan_row["level_b"],
                "metric": plan_row["metric"],
                "planned_max_family_size": int(plan_row["planned_max_family_size"]),
                "preregistered_note": plan_row["note"],
                "n_total_paired_folds": 0,
                "n_zero_differences": 0,
                "n_effective_after_dropping_zeros": 0,
                "positive_difference_count": 0,
                "negative_difference_count": 0,
                "mean_difference_a_minus_b": np.nan,
                "median_difference_a_minus_b": np.nan,
                "wilcoxon_statistic": np.nan,
                "p_value_raw": np.nan,
                "test_status": "not_estimable_from_available_common_folds",
            })
            continue
        a, b, metadata = paired
        row = _paired_wilcoxon_row(a, b, metadata)
        row["test_status"] = "estimated"
        n_eff = int(row["n_effective_after_dropping_zeros"])
        row["two_sided_sign_test_floor_from_n_effective"] = (
            min(1.0, 2.0 / (2 ** n_eff)) if n_eff > 0 else np.nan
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["family_size_planned"] = result.groupby("family_id")["test_id"].transform("size")
    result["family_size_estimable"] = result.groupby("family_id")["p_value_raw"].transform(
        lambda s: int(s.notna().sum())
    )
    result["p_value_holm"] = np.nan

    for family_id, group in result.groupby("family_id", sort=False):
        estimable = group[group["p_value_raw"].notna()]
        if len(group) > 3:
            raise RuntimeError(f"Holm family {family_id} exceeds three planned tests")
        if estimable.empty:
            continue
        pmap = {str(i): float(result.loc[i, "p_value_raw"]) for i in estimable.index}
        adjusted = holm(pmap)
        for i in estimable.index:
            result.loc[i, "p_value_holm"] = adjusted.get(str(i), np.nan)

    return result

def eligibility_aware_metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Create summaries whose metric column names encode the exact fold set."""
    rows: List[Dict[str, Any]] = []
    group_cols = [
        "protocol", "variant", "paper_variant_label", "model",
        "probability_source", "threshold_name", "eligibility_family",
    ]
    for keys, group in metrics.groupby(group_cols, dropna=False):
        eligible = group[group["aggregation_eligible"] == True].copy()
        n = len(eligible)
        family = safe_filename(str(keys[-1])).lower()
        set_name = f"{family}_eligible_n{n}"
        row = dict(zip(group_cols, keys))
        row.update({
            "aggregation_set": set_name,
            "n_candidate_rows": int(len(group)),
            "n_eligible_rows": int(n),
            "n_ineligible_rows": int(len(group) - n),
            "fallback_rows_all_candidates": int(group["fallback_used"].sum()),
        })
        if n:
            for metric in (
                "accuracy", "f1", "macro_f1", "recall", "fpr", "mcc", "pr_auc",
                "train_seconds",
            ):
                row[f"{metric}_mean__{set_name}"] = float(eligible[metric].mean())
            row[f"f1_worst__{set_name}"] = float(eligible.f1.min())
            row[f"macro_f1_worst__{set_name}"] = float(eligible.macro_f1.min())
        rows.append(row)
    return pd.DataFrame(rows)


def eligibility_aware_probability_summary(probability: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = ["protocol", "variant", "paper_variant_label", "model", "probability_source"]
    for keys, group in probability.groupby(group_cols, dropna=False):
        eligible = group[group["probability_evaluation_eligible"] == True].copy()
        n = len(eligible)
        requires_cal = bool(group["probability_requires_calibration"].iloc[0])
        tier = "tierB" if requires_cal else "tierA"
        set_name = f"{tier}_probability_eligible_n{n}"
        row = dict(zip(group_cols, keys))
        row.update({
            "aggregation_set": set_name,
            "n_candidate_rows": int(len(group)),
            "n_eligible_rows": int(n),
        })
        if n:
            for metric in ("ece", "mce", "brier_score", "log_loss", "roc_auc", "pr_auc"):
                row[f"{metric}_mean__{set_name}"] = float(eligible[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def protocol_common_fold_table(metrics: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    main = metrics[
        (metrics.variant == cfg.MAIN_VARIANT)
        & (metrics.model == cfg.MAIN_MODEL)
        & (metrics.probability_source == cfg.MAIN_PROBABILITY_SOURCE)
        & (metrics.threshold_name == cfg.MAIN_THRESHOLD_NAME)
        & (metrics.protocol.isin(["sliding_temporal", "expanding_temporal"]))
        & (metrics.aggregation_eligible == True)
    ].copy()
    if main.empty:
        return pd.DataFrame()
    keep = [
        "candidate_number", "protocol", "fold_id", "f1", "macro_f1", "fpr", "mcc", "pr_auc",
    ]
    wide = main[keep].pivot_table(
        index="candidate_number",
        columns="protocol",
        values=["f1", "macro_f1", "fpr", "mcc", "pr_auc"],
        aggfunc="first",
    )
    wide = wide.dropna().reset_index()
    wide.columns = [
        "candidate_number" if c[0] == "candidate_number" else f"{c[0]}__{c[1]}"
        if isinstance(c, tuple) else str(c)
        for c in wide.columns
    ]
    return wide


def rq6_stress_case_table(
    metrics: pd.DataFrame,
    probability: pd.DataFrame,
    windows: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    main = metrics[
        (metrics.variant == cfg.MAIN_VARIANT)
        & (metrics.model == cfg.MAIN_MODEL)
        & (metrics.probability_source == cfg.MAIN_PROBABILITY_SOURCE)
        & (metrics.threshold_name == cfg.MAIN_THRESHOLD_NAME)
        & (metrics.protocol.isin(["sliding_temporal", "expanding_temporal"]))
    ].copy()
    if main.empty:
        return pd.DataFrame()
    pmain = probability[
        (probability.variant == cfg.MAIN_VARIANT)
        & (probability.model == cfg.MAIN_MODEL)
        & (probability.probability_source == cfg.MAIN_PROBABILITY_SOURCE)
    ][["protocol", "candidate_number", "ece", "brier_score"]].drop_duplicates()
    merged = main.merge(pmain, on=["protocol", "candidate_number"], how="left")
    meta_cols = [
        "protocol", "candidate_number", "transition_type", "cross_session_boundary",
        "train_attack_rate", "train_to_test_gap_hours", "train_session_composition",
        "test_session", "train_rows", "test_rows",
    ]
    available = [c for c in meta_cols if c in windows.columns]
    merged = merged.merge(
        windows[available].drop_duplicates(["protocol", "candidate_number"]),
        on=["protocol", "candidate_number"], how="left",
    )
    return merged.sort_values(["candidate_number", "protocol"])


def aggregate_plots(metrics: pd.DataFrame, cfg: Config, dirs: Mapping[str, Path]) -> None:
    if metrics.empty:
        return
    main = metrics[
        (metrics.model == cfg.MAIN_MODEL)
        & (metrics.probability_source == cfg.MAIN_PROBABILITY_SOURCE)
        & (metrics.threshold_name == cfg.MAIN_THRESHOLD_NAME)
        & (metrics.variant == cfg.MAIN_VARIANT)
        & (metrics.protocol.isin(["sliding_temporal", "expanding_temporal"]))
        & (metrics.aggregation_eligible == True)
    ].copy()
    for metric in ("macro_f1", "fpr", "f1"):
        if main.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for protocol, group in main.groupby("protocol"):
            group = group.sort_values("candidate_number")
            ax.plot(group.candidate_number, group[metric], marker="o", label=protocol)
        ax.set(
            xlabel="Rolling candidate number (same future test window when paired)",
            ylabel=metric.upper(),
            title=f"Sliding vs Expanding | {metric.upper()} | {cfg.MAIN_VARIANT}",
        )
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(dirs["plots"] / f"Sliding_vs_Expanding_{metric.upper()}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    seed_everything(CFG.RANDOM_STATE)
    mode = str(CFG.EXECUTION_MODE).strip().lower()
    valid_modes = {"preflight", "full", "preflight_then_full"}
    if mode not in valid_modes:
        raise ValueError(
            f"EXECUTION_MODE must be one of {sorted(valid_modes)}; received {CFG.EXECUTION_MODE!r}"
        )

    root = Path(CFG.OUTPUT_DIR).resolve()
    root.mkdir(parents=True, exist_ok=True)
    archived = archive_superseded_outputs(CFG, root)
    dirs = ensure_dirs(root)
    log_path = dirs["logs"] / "run.log"
    write_json(asdict(CFG), root / "configuration.json")
    log(f"Starting final 18-window dual-protocol study | mode={mode}", log_path)
    if archived:
        log("Archived superseded output directories: " + "; ".join(archived), log_path)

    df = load_dataset(CFG, log_path)
    detected = detect_columns(df, CFG)
    log(f"Detected columns: {detected}", log_path)
    df = derive_labels(df, detected)
    df, order_type = prepare_order(df, detected["time"], log_path)

    clean, ttl, strict_present, audit = leakage_audit(df, detected, CFG)
    audit.to_csv(dirs["csv"] / "Leakage_Audit.csv", index=False)
    ttl_set = set(ttl)
    strict_norm = {normalize_name(c) for c in CFG.STRICT_DROP_FEATURES}
    strict_clean = [c for c in clean if normalize_name(c) not in strict_norm]
    base_cols = {
        "Clean-All": list(clean),
        "No-TTL": [c for c in clean if c not in ttl_set],
        "Strict-Clean": strict_clean,
        "Strict-Clean-No-TTL": [c for c in strict_clean if c not in ttl_set],
    }

    missing_strict = [
        requested for requested in CFG.STRICT_DROP_FEATURES
        if normalize_name(requested) not in {normalize_name(c) for c in strict_present}
    ]
    write_json({
        "ordering": order_type,
        "detected_columns": detected,
        "ttl_features": ttl,
        "strict_drop_requested": list(CFG.STRICT_DROP_FEATURES),
        "strict_drop_present": strict_present,
        "strict_drop_missing": missing_strict,
        "feature_counts": {k: len(v) for k, v in base_cols.items()},
        "paper_variant_labels": PAPER_VARIANT_LABELS,
        "superseded_outputs_archived": archived,
    }, root / "dataset_and_feature_metadata.json")

    dataset_row = {
        "rows": len(df),
        "columns_total": df.shape[1],
        "attack_rate": float(df["__binary_label__"].mean()),
        "normal_rows": int((df["__binary_label__"] == 0).sum()),
        "attack_rows": int((df["__binary_label__"] == 1).sum()),
        "attack_categories": int(df["__attack_category__"].nunique()),
        "ttl_feature_count": len(ttl),
        "strict_drop_present_count": len(strict_present),
        "ordering": order_type,
        "audit_temporal_windows": CFG.AUDIT_N_TEMPORAL_WINDOWS,
        "full_temporal_windows": CFG.N_TEMPORAL_WINDOWS,
        "full_rfe_sampling_mode": CFG.FULL_RFE_SAMPLING_MODE,
        "rfe_positive_cap_fraction": CFG.RFE_POSITIVE_CAP_FRACTION,
        "bootstrap_rounds": CFG.BOOTSTRAP_ROUNDS,
    }
    dataset_row.update({
        f"features_{safe_filename(name)}": len(cols) for name, cols in base_cols.items()
    })
    pd.DataFrame([dataset_row]).to_csv(dirs["csv"] / "Dataset_Summary.csv", index=False)

    attack_dist = (
        df["__attack_category__"].value_counts(dropna=False)
        .rename_axis("attack_category").reset_index(name="count")
    )
    attack_dist["percentage"] = attack_dist["count"] / len(df)
    attack_dist.to_csv(dirs["csv"] / "Attack_Category_Distribution.csv", index=False)

    # One 18-window profile is used for both audit and full study.
    profile = temporal_window_attack_profile(
        df, CFG.N_TEMPORAL_WINDOWS, "Final_18W", dirs
    )
    audit_cfg = replace(CFG, N_TEMPORAL_WINDOWS=CFG.AUDIT_N_TEMPORAL_WINDOWS)
    sliding_audit = rolling_candidate_audit(
        df, audit_cfg, "sliding", "Final_18W_Audit", dirs
    )
    expanding_audit = rolling_candidate_audit(
        df, audit_cfg, "expanding", "Final_18W_Audit", dirs
    )
    eligibility_candidates = pd.concat(
        [sliding_audit, expanding_audit], ignore_index=True
    )
    eligibility_candidates.to_csv(
        dirs["csv"] / "Temporal_Candidate_Eligibility_All_Protocols.csv", index=False
    )

    # Freeze the primary inferential families before any full-study model
    # results are produced. Every Holm family has at most three tests.
    prereg_plan = preregistered_holm_plan(CFG)
    prereg_plan.to_csv(
        dirs["csv"] / "Preregistered_Holm_Families.csv", index=False
    )

    noise_summary_path = dirs["csv"] / "RFE_Noise_Audit_Summary.csv"
    sensitivity_runs_path = dirs["csv"] / "RFE_Sampling_Prevalence_Sensitivity_Runs.csv"
    noise_summary = pd.DataFrame()
    noise_pairs = pd.DataFrame()
    noise_runs = pd.DataFrame()
    should_run_preflight = mode in {"preflight", "preflight_then_full"}
    if mode == "full" and (
        not noise_summary_path.exists() or not sensitivity_runs_path.exists()
    ):
        log(
            "Matched RFE preflight artifacts are incomplete; running preflight automatically",
            log_path,
        )
        should_run_preflight = True
    if should_run_preflight:
        noise_summary, noise_pairs, noise_runs = rfe_noise_floor_audit(
            df, base_cols, CFG, dirs, log_path
        )
    elif noise_summary_path.exists():
        noise_summary = pd.read_csv(noise_summary_path)
        pair_path = dirs["csv"] / "RFE_Noise_Audit_Pairwise.csv"
        runs_path = dirs["csv"] / "RFE_Noise_Audit_Runs.csv"
        if pair_path.exists():
            noise_pairs = pd.read_csv(pair_path)
        if runs_path.exists():
            noise_runs = pd.read_csv(runs_path)
        log("Loaded existing matched 18-window RFE preflight", log_path)

    # Cheap integrity audit for the identical natural-uniform/capped-positive
    # mean Jaccard observation. It reuses saved selected-feature sets and
    # independently reconstructs sampled-row fingerprints; no RFE refit.
    sensitivity_integrity = rebuild_sampling_sensitivity_integrity_from_existing(
        df, CFG, dirs, log_path
    )
    if not sensitivity_integrity.empty:
        log(
            f"Sampling-mode integrity comparison written ({len(sensitivity_integrity)} seed-matched rows)",
            log_path,
        )

    if mode == "preflight":
        write_json({
            "status": "preflight_completed",
            "next_step": "Review RFE noise/core and sampling-sensitivity outputs, then set EXECUTION_MODE='full'.",
            "output_directory": str(root),
            "rows_analyzed": len(df),
            "ordering": order_type,
            "n_temporal_windows": CFG.N_TEMPORAL_WINDOWS,
            "sliding_candidate_count": int(len(sliding_audit)),
            "sliding_train_valid_count": int(sliding_audit["valid_training_after_reserve"].sum()),
            "sliding_raw_threshold_two_class_count": int(sliding_audit["tier_C_raw_two_class_threshold_eligible"].sum()),
            "sliding_histpool_threshold_two_class_count": int(sliding_audit["tier_C_historical_pool_threshold_eligible"].sum()),
            "expanding_candidate_count": int(len(expanding_audit)),
            "expanding_train_valid_count": int(expanding_audit["valid_training_after_reserve"].sum()),
            "expanding_raw_threshold_two_class_count": int(expanding_audit["tier_C_raw_two_class_threshold_eligible"].sum()),
            "expanding_histpool_threshold_two_class_count": int(expanding_audit["tier_C_historical_pool_threshold_eligible"].sum()),
            "rfe_noise_summary_rows": int(len(noise_summary)),
            "rfe_noise_pairwise_rows": int(len(noise_pairs)),
        }, root / "completion_summary.json")
        log("Preflight completed. Do not use superseded 12-window stability numbers.", log_path)
        return

    # ------------------------------------------------------------------
    # Full evaluation.
    # ------------------------------------------------------------------
    protocols: List[List[DataSplit]] = []
    if CFG.RUN_RANDOM_BASELINE:
        protocols.append(random_split(df, CFG))
    if CFG.RUN_SINGLE_TEMPORAL:
        protocols.append(single_temporal_split(df, CFG))

    temporal_split_lists: Dict[str, List[DataSplit]] = {}
    if CFG.RUN_TEMPORAL_PROTOCOLS:
        for training_mode in CFG.EVALUATION_PROTOCOLS:
            split_list = rolling_splits(df, CFG, training_mode, log_path)
            temporal_split_lists[training_mode] = split_list
            protocols.append(split_list)
            log(
                f"{training_mode} protocol produced {len(split_list)} train-valid folds "
                f"from {CFG.N_TEMPORAL_WINDOWS} equal-count windows",
                log_path,
            )

    all_metrics: List[Dict[str, Any]] = []
    all_probability: List[Dict[str, Any]] = []
    all_attack: List[pd.DataFrame] = []
    all_mcnemar: List[Dict[str, Any]] = []
    all_bootstrap: List[Dict[str, Any]] = []
    all_summaries: List[Dict[str, Any]] = []
    history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for splits in protocols:
        for split in splits:
            result = evaluate_split(df, split, base_cols, CFG, dirs, log_path, history)
            if result.get("skipped"):
                continue
            all_metrics.extend(result["metrics"])
            all_probability.extend(result["probability"])
            all_attack.extend(result["attack_tables"])
            all_mcnemar.extend(result["mcnemar"])
            all_bootstrap.extend(result["bootstrap"])
            all_summaries.append(result["summary"])
            pd.DataFrame(all_metrics).to_csv(
                dirs["csv"] / "All_Evaluation_Metrics_Checkpoint.csv", index=False
            )
            pd.DataFrame(all_probability).to_csv(
                dirs["csv"] / "Probability_Quality_Metrics_Checkpoint.csv", index=False
            )
            log(f"Completed {split.protocol}/{split.fold_id}", log_path)
            gc.collect()

    metrics_df = pd.DataFrame(all_metrics)
    probability_df = pd.DataFrame(all_probability)
    window_df = pd.DataFrame(all_summaries)
    mcnemar_df = pd.DataFrame(all_mcnemar)
    bootstrap_df = pd.DataFrame(all_bootstrap)
    attack_df = pd.concat(all_attack, ignore_index=True) if all_attack else pd.DataFrame()

    if not mcnemar_df.empty:
        adjusted_parts = []
        for (_, _), group in mcnemar_df.groupby(["protocol", "fold_id"], dropna=False):
            pmap = {str(i): float(p) for i, p in zip(group.index, group["p_value_exact"])}
            adjusted = holm(pmap)
            group = group.copy()
            group["p_value_holm_within_fold"] = [adjusted.get(str(i), np.nan) for i in group.index]
            adjusted_parts.append(group)
        mcnemar_df = pd.concat(adjusted_parts, ignore_index=True)

    metrics_df.to_csv(dirs["csv"] / "Final_All_Evaluation_Metrics.csv", index=False)
    probability_df.to_csv(dirs["csv"] / "Final_Probability_Quality_Metrics.csv", index=False)
    window_df.to_csv(dirs["csv"] / "Final_Per_Window_Data_Summary.csv", index=False)
    mcnemar_df.to_csv(dirs["csv"] / "Final_McNemar_Tests.csv", index=False)
    bootstrap_df.to_csv(dirs["csv"] / "Final_Bootstrap_Confidence_Intervals.csv", index=False)
    if not attack_df.empty:
        attack_df.to_csv(dirs["csv"] / "Final_Per_Attack_Class_Detection.csv", index=False)

    # Explicit threshold-method eligibility/failure table.
    eligibility_cols = [
        "protocol", "fold_id", "candidate_number", "variant", "model", "probability_source",
        "threshold_name", "eligibility_family", "threshold_method_eligible",
        "aggregation_eligible", "threshold_selection_status", "fallback_used",
        "historical_threshold_pool_used", "train_class_count", "calibration_class_count",
        "threshold_selection_class_count", "historical_threshold_pool_class_count", "test_class_count",
        "threshold_selection_attack_rate", "historical_threshold_pool_attack_rate",
        "test_window_attack_rate",
    ]
    metrics_df[[c for c in eligibility_cols if c in metrics_df.columns]].to_csv(
        dirs["csv"] / "Final_Threshold_Method_Eligibility.csv", index=False
    )

    stability, frequency, stability_summary = feature_stability(history, dirs)
    noise_vs_temporal = compare_noise_to_temporal_stability(
        noise_summary, stability_summary, CFG, dirs
    )

    wilcoxon_df = across_window_tests(metrics_df, probability_df, CFG)
    wilcoxon_df.to_csv(
        dirs["csv"] / "Final_Preregistered_Inference_Holm.csv", index=False
    )
    # Compatibility filename; contains the same preregistered families only.
    wilcoxon_df.to_csv(
        dirs["csv"] / "Final_Across_Window_Wilcoxon_Holm.csv", index=False
    )

    model_summary = eligibility_aware_metric_summary(metrics_df)
    model_summary.to_csv(dirs["csv"] / "Final_Model_Summary.csv", index=False)
    probability_summary = eligibility_aware_probability_summary(probability_df)
    probability_summary.to_csv(dirs["csv"] / "Final_Probability_Summary.csv", index=False)

    common_folds = protocol_common_fold_table(metrics_df, CFG)
    common_folds.to_csv(dirs["csv"] / "Final_Sliding_vs_Expanding_Common_Folds.csv", index=False)
    rq6_table = rq6_stress_case_table(metrics_df, probability_df, window_df, CFG)
    rq6_table.to_csv(dirs["csv"] / "Final_RQ6_Cross_Session_Stress_Cases.csv", index=False)

    # Historical-pool treatment vs raw FPR-budget treatment.
    fpr_compare = metrics_df[
        metrics_df["threshold_name"].astype(str).str.startswith(("fpr_", "fpr_histpool_"))
    ].copy()
    fpr_compare.to_csv(
        dirs["csv"] / "Final_FPR_Budget_Raw_vs_Historical_Pool.csv", index=False
    )
    metrics_df[
        metrics_df["threshold_name"].astype(str).str.startswith(
            ("mu_sigma_raw_", "mu_sigma_guarded_")
        )
    ].to_csv(dirs["csv"] / "Final_Mu_Sigma_Raw_vs_Guarded.csv", index=False)

    metrics_df[metrics_df.variant.isin(CFG.BASE_VARIANT_ORDER)].to_csv(
        dirs["csv"] / "Final_Base_Variant_Comparison.csv", index=False
    )
    metrics_df[
        (metrics_df.variant == CFG.MAIN_VARIANT)
        & (metrics_df.model == CFG.MAIN_MODEL)
        & (metrics_df.probability_source == CFG.MAIN_PROBABILITY_SOURCE)
    ].to_csv(dirs["csv"] / "Final_Main_Strict_Clean_RFE_Results.csv", index=False)
    metrics_df[metrics_df.variant.isin(["Clean-All", "Strict-Clean"])].to_csv(
        dirs["csv"] / "Final_Shortcut_Exposure_Comparison.csv", index=False
    )
    metrics_df[metrics_df.variant.isin(["Clean-All", "No-TTL"])].to_csv(
        dirs["csv"] / "Final_Clean_All_vs_No_TTL_Comparison.csv", index=False
    )

    aggregate_plots(metrics_df, CFG, dirs)

    write_json({
        "status": "completed",
        "execution_mode": mode,
        "output_directory": str(root),
        "rows_analyzed": len(df),
        "ordering": order_type,
        "base_variants": list(CFG.BASE_VARIANT_ORDER),
        "main_variant": CFG.MAIN_VARIANT,
        "main_model": CFG.MAIN_MODEL,
        "main_probability_source": CFG.MAIN_PROBABILITY_SOURCE,
        "main_threshold_name": CFG.MAIN_THRESHOLD_NAME,
        "full_rfe_sampling_mode": CFG.FULL_RFE_SAMPLING_MODE,
        "rfe_positive_cap_fraction": CFG.RFE_POSITIVE_CAP_FRACTION,
        "rfe_fixed_sample_seed": CFG.RFE_FIXED_SAMPLE_SEED,
        "rfe_fixed_model_seed": CFG.RFE_FIXED_MODEL_SEED,
        "audit_temporal_windows": CFG.AUDIT_N_TEMPORAL_WINDOWS,
        "full_temporal_windows": CFG.N_TEMPORAL_WINDOWS,
        "bootstrap_rounds": CFG.BOOTSTRAP_ROUNDS,
        "sliding_train_valid_folds": len(temporal_split_lists.get("sliding", [])),
        "expanding_train_valid_folds": len(temporal_split_lists.get("expanding", [])),
        "metrics_rows": len(metrics_df),
        "probability_rows": len(probability_df),
        "per_attack_rows": len(attack_df),
        "feature_stability_rows": len(stability),
        "feature_frequency_rows": len(frequency),
        "rfe_noise_summary_rows": len(noise_summary),
        "rfe_noise_vs_temporal_rows": len(noise_vs_temporal),
        "common_protocol_fold_rows": len(common_folds),
    }, root / "completion_summary.json")
    log(f"Study completed. Final results: {root}", log_path)


if __name__ == "__main__":
    main()
''',
    'r1': r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reviewer-1 revision experiments for IJICS-359541 (NF-UNSW-NB15-v3).

RUN-ONLY VERSION. Put this file in the SAME folder as:
  1) NF-UNSW-NB15-v3.csv
  2) nf_unsw_nb15_v3_temporal_reliability_18W_DUAL_FINAL_READY_BOOTSTRAP_FIX.py   (original pipeline)
  3) Final_All_Evaluation_Metrics.csv   (optional; enables the reproduction check)
then press "Run Python File" in VS Code.

All loading, deduplication, ordering, schema construction, preprocessing, support
reserves, RFE sampling, and model construction are imported from the original
pipeline, so every part below is a controlled extension of the published run.

Parts (toggle in RUN_PARTS):
  P1  Boundary-overlap audit (Comment R1.5)                         ~ minutes
  P2  Main RF rerun with explicit threshold status, benign-quantile
      rule, per-flow scores, calibration diagnostics (R1.4, R1.3, R1.9, R1.1)   ~1-1.5 h
  P3  Purged-boundary rerun of P2 (R1.5)                              ~1-1.5 h
  P4  Linear SVM failure diagnostic (R1.10)                           ~20-40 min
  P5  Extended RFE sampling/estimator-seed reference (R1.8)           ~1.5 h
  P6  Fixed-duration and session-aligned window sensitivity (R1.6)   ~2-3 h
  P7  Leave-one-attack-category-out "unseen family" analysis (R1.11) ~1-1.5 h
  P8  Controlled-difficulty (reduced separability) experiment (R1.11) ~1 h

Every part writes its outputs when it finishes and is SKIPPED on the next run if
its "done" marker exists, so an interrupted run can simply be restarted.
Send back the folder  <OUTPUT_DIR>/summary  (small CSV/JSON files) and
<OUTPUT_DIR>/figures. The <OUTPUT_DIR>/perflow folder is large and is meant
for the public repository, not for e-mail.
"""
from __future__ import annotations

import copy
import gc
import importlib.util
import json
import math
import os
import platform
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (f1_score, log_loss, matthews_corrcoef, roc_auc_score,
                             average_precision_score, roc_curve, brier_score_loss)
from sklearn.svm import LinearSVC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# USER SETTINGS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "NF-UNSW-NB15-v3.csv"
ORIGINAL_CODE = SCRIPT_DIR / "nf_unsw_nb15_v3_temporal_reliability_18W_DUAL_FINAL_READY_BOOTSTRAP_FIX.py"
ORIGINAL_METRICS = SCRIPT_DIR / "Final_All_Evaluation_Metrics.csv"     # optional
OUTPUT_DIR = SCRIPT_DIR / "NF_UNSW_NB15_v3_Revision_R1_Runs"

RUN_PARTS = {
    "P1_boundary_audit": True,
    "P2_threshold_calibration_rerun": True,
    "P3_purged_rerun": "auto",        # True | False | "auto" (runs if P1 finds any boundary-crossing flow)
    "P4_svm_diagnostic": True,
    "P5_rfe_reference": True,
    "P6_window_sensitivity": True,
    "P7_unseen_category": True,
    "P8_controlled_difficulty": True,
}
SAVE_PERFLOW = True                 # per-flow scores for the repository (P2)

# Extended RFE reference (P5)
P5_SAMPLE_SIZES = (25_000, 50_000, 100_000)
P5_N_MODEL_SEEDS = 3

# Equivalence settings (same as the paper)
TOST_MARGIN_PP = 0.05

# Smoke-test switch used only to check the script on a tiny synthetic file.
SMOKE = os.environ.get("R1_SMOKE_TEST", "0") == "1"

# =============================================================================
# INFRASTRUCTURE
# =============================================================================
def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("final_temporal_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import original pipeline from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def say(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUTPUT_DIR / "run_log.txt", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def peak_rss_mb() -> float:
    try:
        import psutil  # optional
        return float(psutil.Process().memory_info().rss / 2**20)
    except Exception:
        try:
            import resource
            return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
        except Exception:
            return float("nan")


def done_marker(part: str) -> Path:
    return OUTPUT_DIR / "summary" / f".{part}.done"


def is_done(part: str) -> bool:
    return done_marker(part).exists()


def mark_done(part: str):
    done_marker(part).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def save(df: pd.DataFrame, name: str, sub: str = "summary"):
    p = OUTPUT_DIR / sub / name
    df.to_csv(p, index=False)
    return p


class Ctx:
    """Shared data loaded once."""
    pass


def build_context() -> Ctx:
    c = Ctx()
    c.mod = mod = load_module(ORIGINAL_CODE)
    c.cfg = cfg = mod.Config()
    cfg.DATA_PATHS = (str(DATA_FILE),)
    if SMOKE:
        cfg.RF_N_ESTIMATORS = 20
        cfg.RFE_N_ESTIMATORS = 10
        cfg.RFE_MAX_SAMPLES = 4_000
        cfg.RFE_MIN_OTHER_CLASS_SAMPLES = 50
        cfg.CALIBRATION_FALLBACK_PER_CLASS = 50
        cfg.THRESHOLD_FALLBACK_PER_CLASS = 50
    log_path = OUTPUT_DIR / "dataset_load.log"
    df = mod.load_dataset(cfg, log_path)
    detected = mod.detect_columns(df, cfg)
    df = mod.derive_labels(df, detected)
    df, order_type = mod.prepare_order(df, detected["time"], log_path)
    clean, ttl, strict_present, _ = mod.leakage_audit(df, detected, cfg)
    strict_norm = {mod.normalize_name(x) for x in cfg.STRICT_DROP_FEATURES}
    c.strict_clean = [x for x in clean if mod.normalize_name(x) not in strict_norm]
    c.df, c.detected, c.order_type, c.log_path = df, detected, order_type, log_path
    say(f"Loaded {len(df):,} flows; Strict-Clean predictors = {len(c.strict_clean)}; order = {order_type}")
    if not SMOKE:
        if len(df) != 2_350_609:
            say(f"WARNING: analytic cohort {len(df):,} differs from the paper (2,350,609)")
        if len(c.strict_clean) != 46:
            say(f"WARNING: Strict-Clean count {len(c.strict_clean)} differs from the paper (46)")
    # raw numeric chronology (milliseconds)
    tcol = detected["time"]
    c.start_ms = pd.to_numeric(df[tcol], errors="coerce").to_numpy(dtype=np.float64)
    endcol = next((x for x in df.columns if mod.normalize_name(x) == mod.normalize_name("FLOW_END_MILLISECONDS")), None)
    c.end_ms = pd.to_numeric(df[endcol], errors="coerce").to_numpy(dtype=np.float64) if endcol else None
    c.y = df["__binary_label__"].to_numpy(np.int8)
    c.cat = df["__attack_category__"].astype(str).to_numpy()
    c.splits = {m: mod.rolling_splits(df, cfg, training_mode=m, log_path=log_path) for m in ("sliding", "expanding")}
    c.orig_metrics = pd.read_csv(ORIGINAL_METRICS) if ORIGINAL_METRICS.exists() else None
    return c


# =============================================================================
# METRICS AND THRESHOLD SELECTION (revised: explicit status, no sentinel)
# =============================================================================
def confusion(y, pred):
    y = np.asarray(y, np.int8); pred = np.asarray(pred, np.int8)
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    return tn, fp, fn, tp


def metric_block(y, pred, p=None, prefix=""):
    """Metric-specific availability: each metric is reported only where defined."""
    tn, fp, fn, tp = confusion(y, pred)
    nb, na = tn + fp, tp + fn
    out = {f"{prefix}tn": tn, f"{prefix}fp": fp, f"{prefix}fn": fn, f"{prefix}tp": tp,
           f"{prefix}n_benign": nb, f"{prefix}n_attack": na,
           f"{prefix}M_benign_available": nb > 0, f"{prefix}M_attack_available": na > 0,
           f"{prefix}M_two_class_available": (nb > 0 and na > 0)}
    out[f"{prefix}fpr"] = fp / nb if nb else np.nan
    out[f"{prefix}alerts"] = fp + tp
    out[f"{prefix}recall"] = tp / na if na else np.nan
    if nb and na:
        out[f"{prefix}precision"] = tp / (tp + fp) if (tp + fp) else 0.0
        out[f"{prefix}f1"] = f1_score(y, pred, zero_division=0)
        out[f"{prefix}macro_f1"] = f1_score(y, pred, average="macro", zero_division=0)
        out[f"{prefix}mcc"] = matthews_corrcoef(y, pred)
        if p is not None:
            out[f"{prefix}roc_auc"] = roc_auc_score(y, p)
            out[f"{prefix}pr_auc"] = average_precision_score(y, p)
    else:
        for k in ("precision", "f1", "macro_f1", "mcc", "roc_auc", "pr_auc"):
            out[f"{prefix}{k}"] = np.nan
    return out


def roc_points(y, p):
    fpr, tpr, thr = roc_curve(y, p, drop_intermediate=False)
    ok = np.isfinite(thr)
    return fpr[ok], tpr[ok], thr[ok]


def rule_record(name, tau, status, ready, support, **extra):
    r = {"threshold_name": name, "tau": tau, "status": status, "ready": bool(ready), "support": support}
    r.update(extra)
    return r


def youden_v2(y, p, name="youden", support="raw_window"):
    if len(np.unique(y)) < 2:
        return rule_record(name, np.nan, "not_ready_single_class_support", False, support)
    fpr, tpr, thr = roc_points(y, p)
    j = tpr - fpr
    i = int(np.argmax(j))
    return rule_record(name, float(thr[i]), "feasible", True, support,
                       past_fpr=float(fpr[i]), past_tpr=float(tpr[i]), n_finite_roc_points=int(len(thr)))


def fpr_budget_v2(y, p, alpha, name, support):
    """Most permissive finite ROC threshold with past FPR <= alpha (same objective as the
    original selector) but infeasibility is a separate status, never a numeric 1.0."""
    if len(np.unique(y)) < 2:
        return rule_record(name, np.nan, "not_ready_single_class_support", False, support, alpha=alpha)
    fpr, tpr, thr = roc_points(y, p)
    ok = np.flatnonzero(fpr <= alpha + 1e-15)
    if len(ok) == 0:
        return rule_record(name, np.nan, "infeasible_no_roc_point_within_budget", True, support,
                           alpha=alpha, n_finite_roc_points=int(len(thr)))
    m = thr[ok].min()
    cand = ok[np.isclose(thr[ok], m)]
    i = int(cand[np.argmax(tpr[cand])])
    return rule_record(name, float(thr[i]), "feasible", True, support, alpha=alpha,
                       past_fpr=float(fpr[i]), past_tpr=float(tpr[i]), n_finite_roc_points=int(len(thr)))


def benign_quantile_v2(p_benign, alpha, name, support):
    """FPR budget from benign scores only: smallest attainable tau with
    mean(p_benign >= tau) <= alpha. Ties are handled exactly on the unique scores.
    If even the largest benign score exceeds the budget, tau is set just above it."""
    p_benign = np.sort(np.asarray(p_benign, float))
    n = len(p_benign)
    if n == 0:
        return rule_record(name, np.nan, "not_ready_no_benign_support", False, support, alpha=alpha)
    u = np.unique(p_benign)
    fpr_u = (n - np.searchsorted(p_benign, u, side="left")) / n
    ok = np.flatnonzero(fpr_u <= alpha + 1e-15)
    if len(ok):
        tau = float(u[ok[0]]); pf = float(fpr_u[ok[0]]); st = "feasible"
    else:
        tau = float(np.nextafter(u[-1], np.inf)); pf = 0.0
        st = "feasible_above_max_benign_score" if tau <= 1.0 else "feasible_no_alert_attainable"
    return rule_record(name, tau, st, True, support, alpha=alpha, past_fpr=pf,
                       n_unique_benign_scores=int(len(u)), n_benign_support=int(n))


def all_rules(cfg, mod, yq, pq, yb, pb):
    """Revised threshold family. yq/pq: raw threshold window; yb/pb: historical pool."""
    rules = [rule_record("fixed_0.5", 0.5, "fixed", True, "none")]
    raw_two = len(np.unique(yq)) >= 2
    mu, sd = (float(np.mean(pq)), float(np.std(pq))) if len(pq) else (0.5, 0.0)
    for k in cfg.MU_SIGMA_K_VALUES:
        raw = float(np.clip(mu + k * sd, 0, 1))
        rules.append(rule_record(f"mu_sigma_raw_k_{k:.2f}", raw,
                                 "raw_rule_two_class_window" if raw_two else "raw_rule_single_class_window", True, "raw_window"))
        rules.append(rule_record(f"mu_sigma_guarded_k_{k:.2f}", raw if raw_two else 0.5,
                                 "guard_not_needed" if raw_two else "guarded_fallback_0.5", True, "raw_window"))
    rules.append(youden_v2(yq, pq))
    for a in cfg.FPR_BUDGETS:
        pct = f"{a*100:.1f}pct"
        rec = fpr_budget_v2(yq, pq, a, f"fpr_{pct}", "raw_window")
        rec["original_implementation_tau"] = float(mod.fpr_budget_threshold(yq, pq, a)) if raw_two else np.nan
        rules.append(rec)
        if raw_two:
            h = fpr_budget_v2(yq, pq, a, f"fpr_histpool_{pct}", "raw_window_two_class")
        else:
            h = fpr_budget_v2(yb, pb, a, f"fpr_histpool_{pct}", "historical_pool")
        src_y, src_p = (yq, pq) if raw_two else (yb, pb)
        h["original_implementation_tau"] = float(mod.fpr_budget_threshold(src_y, src_p, a)) if len(np.unique(src_y)) >= 2 else np.nan
        rules.append(h)
        rules.append(benign_quantile_v2(pq[yq == 0], a, f"benign_quantile_raw_{pct}", "raw_window_benign_only"))
        rules.append(benign_quantile_v2(pb[yb == 0], a, f"benign_quantile_histpool_{pct}", "historical_pool_benign_only"))
    return rules


# ---------------- calibration diagnostics (R1.9)
def bin_table(y, p, edges):
    p = np.clip(np.asarray(p, float), 0, 1); y = np.asarray(y, int)
    ids = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = ids == b
        n = int(m.sum())
        rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n,
                     "n_attack": int(y[m].sum()) if n else 0,
                     "mean_p": float(p[m].mean()) if n else np.nan,
                     "frac_attack": float(y[m].mean()) if n else np.nan})
    t = pd.DataFrame(rows)
    t["gap"] = (t["frac_attack"] - t["mean_p"]).abs()
    return t


def calib_summary(y, p):
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), 0, 1)
    out = {}
    schemes = {f"eqwidth_{b}": np.linspace(0, 1, b + 1) for b in (10, 15, 20, 50)}
    q = np.unique(np.quantile(p, np.linspace(0, 1, 16)))
    if len(q) >= 2:
        q[0], q[-1] = 0.0, 1.0
        schemes["eqmass_15"] = q
    for name, edges in schemes.items():
        t = bin_table(y, p, edges)
        occ = t[t.n > 0]
        out[f"ece_{name}"] = float((occ.gap * occ.n).sum() / len(y))
        out[f"mce_{name}"] = float(occ.gap.max()) if len(occ) else np.nan
        out[f"occupied_bins_{name}"] = int(len(occ))
        if len(occ):
            w = occ.loc[occ.gap.idxmax()]
            out[f"mce_bin_{name}"] = f"[{w.lo:.3f},{w.hi:.3f}]"
            out[f"mce_bin_n_{name}"] = int(w.n)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    ll = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    out["log_loss"] = float(ll.mean())
    out["brier"] = float(np.mean((p - y) ** 2))
    srt = np.sort(ll)[::-1]
    out["log_loss_share_top10_flows"] = float(srt[:10].sum() / max(ll.sum(), 1e-300))
    out["log_loss_share_top100_flows"] = float(srt[:100].sum() / max(ll.sum(), 1e-300))
    out["n_confident_errors_p_ge_0.99_or_le_0.01"] = int(((y == 0) & (p >= 0.99)).sum() + ((y == 1) & (p <= 0.01)).sum())
    out["mean_p_benign"] = float(p[y == 0].mean()) if (y == 0).any() else np.nan
    out["mean_p_attack"] = float(p[y == 1].mean()) if (y == 1).any() else np.nan
    out["n_unique_scores"] = int(len(np.unique(p)))
    return out


# =============================================================================
# CORE PIPELINE STEP (same estimator settings and seeds as the paper)
# =============================================================================
def frames_for(c, split, keep_mask=None, drop_categories=None):
    """Return role dataframes; optional purge mask (boolean over df rows) and
    optional removal of one attack category from every past role."""
    df = c.df

    def take(idx, past=True):
        idx = np.asarray(idx, np.int64)
        if past and keep_mask is not None:
            idx = idx[keep_mask[idx]]
        if past and drop_categories is not None:
            idx = idx[~np.isin(c.cat[idx], list(drop_categories))]
        return idx
    roles = {"train": take(split.train_idx), "cal": take(split.calibration_idx), "thr": take(split.threshold_idx),
             "pool": take(split.threshold_pool_idx), "test": np.asarray(split.test_idx, np.int64)}
    return roles


def preprocess(c, roles, cols):
    df = c.df
    xs = c.mod.fit_transform(df.iloc[roles["train"]], df.iloc[roles["cal"]], df.iloc[roles["thr"]],
                             df.iloc[roles["pool"]], df.iloc[roles["test"]], cols)
    return xs  # xt, xc, xv, xvp, xs, names


def rfe_select(c, x, y, names, sample_seed=None, model_seed=None, max_samples=None, return_info=False):
    cfg, mod = c.cfg, c.mod
    ss = cfg.RFE_FIXED_SAMPLE_SEED + 2 if sample_seed is None else sample_seed
    ms = cfg.RFE_FIXED_MODEL_SEED + 2 if model_seed is None else model_seed
    idx, st = mod.rfe_sample_indices(y, max_samples or cfg.RFE_MAX_SAMPLES, ss, cfg.FULL_RFE_SAMPLING_MODE,
                                     cfg.RFE_POSITIVE_CAP_FRACTION, cfg.RFE_MIN_OTHER_CLASS_SAMPLES)
    k = min(cfg.TOP_K_FEATURES, x.shape[1])
    est = RandomForestClassifier(n_estimators=cfg.RFE_N_ESTIMATORS, class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=ms, max_depth=18)
    sel = RFE(estimator=est, n_features_to_select=k, step=cfg.RFE_STEP).fit(x[idx], y[idx])
    s = np.flatnonzero(sel.support_)
    if return_info:
        return s, [names[i] for i in s], st
    return s, [names[i] for i in s]


def run_main_fold(c, split, cols, tag, keep_mask=None, drop_categories=None, perturb=None,
                  save_perflow=False, want_calib=True):
    """Strict-Clean -> per-fold RFE-15 -> RF-300 (seed as in the paper) -> raw/Sigmoid/Isotonic
    -> revised threshold family -> metric-specific evaluation on the future window."""
    mod, cfg, df = c.mod, c.cfg, c.df
    roles = frames_for(c, split, keep_mask, drop_categories)
    y = {k: c.y[v] for k, v in roles.items()}
    base = {"tag": tag, "protocol": split.protocol, "fold_id": split.fold_id,
            "candidate_number": split.candidate_number, "test_window_id": split.test_window_id,
            **{f"rows_{k}": int(len(v)) for k, v in roles.items()},
            "calibration_reserve_rows": split.calibration_reserve_rows,
            "threshold_reserve_rows": split.threshold_reserve_rows,
            "train_attack_rate": float(y["train"].mean()) if len(y["train"]) else np.nan,
            "test_attack_rate": float(y["test"].mean())}
    ready = {"R0_model_fitting": len(np.unique(y["train"])) >= 2,
             "R1_calibration": len(np.unique(y["cal"])) >= 2,
             "R2_raw_window_two_class": len(np.unique(y["thr"])) >= 2,
             "R2_historical_pool_two_class": len(np.unique(y["pool"])) >= 2}
    base.update({f"ready_{k}": v for k, v in ready.items()})
    if not ready["R0_model_fitting"]:
        return {"fold": base, "rules": [], "calib": [], "perflow": None}
    xt, xc, xv, xvp, xs, names = preprocess(c, roles, cols)
    if perturb is not None:
        drop = [i for i, n in enumerate(names) if n in perturb]
        keepc = [i for i in range(len(names)) if i not in drop]
        xt, xc, xv, xvp, xs = (a[:, keepc] for a in (xt, xc, xv, xvp, xs))
        names = [names[i] for i in keepc]
    t0 = time.perf_counter()
    sel, sel_names = rfe_select(c, xt, y["train"], names)
    rfe_s = time.perf_counter() - t0
    xt, xc, xv, xvp, xs = (a[:, sel] for a in (xt, xc, xv, xvp, xs))
    model = mod.make_model("RF", cfg, 10_210)        # = seed used for Strict-Clean-RFE-TopK RF in the paper
    t0 = time.perf_counter(); model.fit(xt, y["train"]); fit_s = time.perf_counter() - t0
    base.update({"rfe_seconds": rfe_s, "rf_fit_seconds": fit_s, "selected_features": "|".join(sel_names),
                 "rf_total_nodes": int(sum(e.tree_.node_count for e in model.estimators_)),
                 "peak_rss_mb": peak_rss_mb()})
    probs = {"uncalibrated": {k: mod.positive_probability(model, x) for k, x in
                              (("thr", xv), ("pool", xvp), ("test", xs))}}
    if ready["R1_calibration"]:
        for meth in ("sigmoid", "isotonic"):
            cal = mod.fit_calibrator(model, xc, y["cal"], meth)
            probs[meth] = {k: mod.positive_probability(cal, x) for k, x in (("thr", xv), ("pool", xvp), ("test", xs))}
    rule_rows, calib_rows = [], []
    for src, P in probs.items():
        for r in all_rules(cfg, mod, y["thr"], P["thr"], y["pool"], P["pool"]):
            row = {**{k: base[k] for k in ("tag", "protocol", "fold_id", "candidate_number")},
                   "probability_source": src, **r,
                   "n_unique_scores_threshold_window": int(len(np.unique(P["thr"]))),
                   "n_unique_scores_test": int(len(np.unique(P["test"])))}
            if np.isfinite(r["tau"]):
                row.update(metric_block(y["test"], (P["test"] >= r["tau"]).astype(np.int8), P["test"]))
            ot = r.get("original_implementation_tau", np.nan)
            if ot is not None and np.isfinite(ot):
                row.update(metric_block(y["test"], (P["test"] >= ot).astype(np.int8), None, prefix="origimpl_"))
            rule_rows.append(row)
        if want_calib:
            cs = calib_summary(y["test"], P["test"])
            calib_rows.append({**{k: base[k] for k in ("tag", "protocol", "fold_id", "candidate_number")},
                               "probability_source": src, "test_attack_rate": base["test_attack_rate"], **cs})
    pf = None
    if save_perflow:
        tidx = roles["test"]
        pf = pd.DataFrame({"protocol": split.protocol, "fold_id": split.fold_id, "row_id": tidx,
                           "flow_start_ms": c.start_ms[tidx],
                           "flow_end_ms": c.end_ms[tidx] if c.end_ms is not None else np.nan,
                           "label": y["test"], "attack": c.cat[tidx]})
        for col in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL"):
            if col in df.columns:
                pf[col] = df[col].to_numpy()[tidx]
        for src, P in probs.items():
            pf[f"p_{src}"] = P["test"].astype(np.float32)
    del xt, xc, xv, xvp, xs, model
    gc.collect()
    return {"fold": base, "rules": rule_rows, "calib": calib_rows, "perflow": pf, "probs_test": probs, "y_test": y["test"]}


def reproduction_check(c, rules_df):
    if c.orig_metrics is None or rules_df.empty:
        return pd.DataFrame()
    o = c.orig_metrics
    o = o[(o.variant == "Strict-Clean-RFE-TopK") & (o.model == "RF") & (o.threshold_name == "fixed_0.5")]
    n = rules_df[rules_df.threshold_name == "fixed_0.5"][["protocol", "fold_id", "probability_source", "fp", "fn"]]
    m = n.merge(o[["protocol", "fold_id", "probability_source", "fp", "fn"]], on=["protocol", "fold_id", "probability_source"],
                suffixes=("_rerun", "_original"), how="left")
    m["match"] = (m.fp_rerun == m.fp_original) & (m.fn_rerun == m.fn_original)
    return m


def tost_paired(d_pp, margin):
    d = np.asarray(d_pp, float); n = len(d)
    if n < 2:
        return {"n": n}
    mean, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return {"n": n, "mean_pp": mean, "ci90_lo": mean, "ci90_hi": mean, "tost_p": 0.0 if abs(mean) < margin else 1.0}
    tcrit = stats.t.ppf(0.95, n - 1)
    p1 = 1 - stats.t.cdf((mean + margin) / se, n - 1)
    p2 = stats.t.cdf((mean - margin) / se, n - 1)
    return {"n": n, "mean_pp": mean, "ci90_lo": mean - tcrit * se, "ci90_hi": mean + tcrit * se, "tost_p": float(max(p1, p2))}


def protocol_comparison(rules_df, folds_df, label):
    """Sliding vs Expanding on candidates where both have a two-class future window."""
    r = rules_df[(rules_df.threshold_name == "fixed_0.5") & (rules_df.probability_source == "isotonic")]
    r = r[r.M_two_class_available == True]
    s = r[r.protocol.str.startswith("sliding")].set_index("candidate_number")
    e = r[r.protocol.str.startswith("expanding")].set_index("candidate_number")
    common = sorted(set(s.index) & set(e.index))
    if not common:
        return pd.DataFrame([{"partition": label, "n_common": 0}])
    d = (s.loc[common, "macro_f1"] - e.loc[common, "macro_f1"]) * 100
    t = tost_paired(d.values, TOST_MARGIN_PP)
    fs = folds_df[folds_df.protocol.str.startswith("sliding")].set_index("candidate_number")
    fe = folds_df[folds_df.protocol.str.startswith("expanding")].set_index("candidate_number")
    return pd.DataFrame([{
        "partition": label, "common_candidates": ",".join(map(str, common)), **t,
        "sliding_fp": int(s.loc[common, "fp"].sum()), "expanding_fp": int(e.loc[common, "fp"].sum()),
        "sliding_fn": int(s.loc[common, "fn"].sum()), "expanding_fn": int(e.loc[common, "fn"].sum()),
        "alert_burden_met": bool(s.loc[common, "fp"].sum() - e.loc[common, "fp"].sum() <= len(common)),
        "sliding_rf_fit_s_mean": float(fs.loc[common, "rf_fit_seconds"].mean()),
        "expanding_rf_fit_s_mean": float(fe.loc[common, "rf_fit_seconds"].mean()),
    }])


# =============================================================================
# P1  BOUNDARY-OVERLAP AUDIT (R1.5)
# =============================================================================
def p1_boundary_audit(c):
    if c.end_ms is None:
        say("P1: FLOW_END_MILLISECONDS not found; audit skipped")
        return pd.DataFrame()
    s, e, y = c.start_ms, c.end_ms, c.y
    n = len(s)
    q = pd.DataFrame({
        "flows": [n], "end_missing": [int(np.isnan(e).sum())], "end_before_start": [int((e < s).sum())],
        "duration_ms_median": [float(np.nanmedian(e - s))], "duration_ms_p99": [float(np.nanpercentile(e - s, 99))],
        "duration_ms_max": [float(np.nanmax(e - s))]})
    save(q, "P1_flow_timing_quality.csv")
    windows = [np.asarray(w, np.int64) for w in np.array_split(np.arange(n), c.cfg.N_TEMPORAL_WINDOWS)]
    rows = []
    for k in range(len(windows) - 1):
        b = windows[k + 1][0]
        nxt = s[b]
        past = np.arange(b)
        cross = e[:b] > nxt
        last_w = cross[windows[k][0]:b]
        rows.append({"boundary": f"W{k}|W{k+1}", "next_window_first_start_ms": nxt,
                     "past_flows_crossing": int(cross.sum()),
                     "past_attack_flows_crossing": int((cross & (y[:b] == 1)).sum()),
                     "crossing_from_last_window_only": int(last_w.sum()),
                     "share_of_last_window_pct": 100 * float(last_w.mean()),
                     "max_overrun_s": float(np.nanmax(np.where(cross, e[:b] - nxt, np.nan)) / 1000) if cross.any() else 0.0,
                     "median_overrun_s": float(np.nanmedian(np.where(cross, e[:b] - nxt, np.nan)) / 1000) if cross.any() else 0.0})
    bd = pd.DataFrame(rows)
    save(bd, "P1_window_boundary_overlap.csv")
    frows = []
    for m, splits in c.splits.items():
        for sp in splits:
            ts = s[sp.test_idx[0]]
            for role, idx in (("model_train", sp.train_idx), ("calibration_pool", sp.calibration_idx),
                              ("raw_threshold", sp.threshold_idx), ("historical_pool", sp.threshold_pool_idx)):
                cr = e[idx] > ts
                frows.append({"protocol": sp.protocol, "fold_id": sp.fold_id, "role": role, "rows": int(len(idx)),
                              "rows_ending_after_test_start": int(cr.sum()),
                              "attack_rows_ending_after_test_start": int((cr & (y[idx] == 1)).sum()),
                              "share_pct": 100 * float(cr.mean())})
    fd = pd.DataFrame(frows)
    save(fd, "P1_fold_role_overlap_with_test_start.csv")
    total = int(fd.rows_ending_after_test_start.sum())
    say(f"P1: {int(bd.past_flows_crossing.sum()):,} window-boundary crossings; "
        f"{total:,} past-role rows end after their test window starts")
    return fd


def purge_mask_for(c, split):
    """Keep a past-role flow only if it ended before the first flow of the test window started."""
    keep = np.ones(len(c.df), dtype=bool)
    ts = c.start_ms[split.test_idx[0]]
    keep[c.end_ms > ts] = False
    return keep


# =============================================================================
# P2 / P3  MAIN RERUN
# =============================================================================
def p2_main(c, purged=False):
    part = "P3" if purged else "P2"
    tag = "purged" if purged else "main"
    fold_rows, rule_rows, calib_rows = [], [], []
    pooled = {}
    for m in ("sliding", "expanding"):
        for sp in c.splits[m]:
            say(f"{part}: {sp.fold_id}")
            km = purge_mask_for(c, sp) if purged else None
            res = run_main_fold(c, sp, c.strict_clean, tag, keep_mask=km, save_perflow=(SAVE_PERFLOW and not purged))
            fold_rows.append(res["fold"]); rule_rows += res["rules"]; calib_rows += res["calib"]
            if res["perflow"] is not None:
                res["perflow"].to_csv(OUTPUT_DIR / "perflow" / f"P2_perflow_{sp.fold_id}.csv.gz", index=False, compression="gzip")
            if not purged and res.get("probs_test") and res["fold"].get("test_attack_rate", 0) > 0:
                key = "expanding" if m == "expanding" else "sliding"
                for src, P in res["probs_test"].items():
                    pooled.setdefault((key, src), []).append((res["y_test"], P["test"]))
            pd.DataFrame(rule_rows).to_csv(OUTPUT_DIR / "summary" / f"{part}_threshold_records_checkpoint.csv", index=False)
    F, R, C = pd.DataFrame(fold_rows), pd.DataFrame(rule_rows), pd.DataFrame(calib_rows)
    save(F, f"{part}_fold_records.csv"); save(R, f"{part}_threshold_records.csv"); save(C, f"{part}_calibration_diagnostics.csv")
    rc = reproduction_check(c, R)
    if len(rc):
        save(rc, f"{part}_reproduction_check_vs_original.csv")
        say(f"{part}: reproduction of original fixed-0.5 FP/FN: {int(rc.match.sum())}/{len(rc)} rows match")
    save(protocol_comparison(R, F, tag), f"{part}_sliding_vs_expanding.csv")
    # Table-9-style summary on two-class Sliding folds, Isotonic RF
    t = R[(R.protocol == "sliding_temporal") & (R.probability_source == "isotonic") & (R.M_two_class_available == True)]
    if len(t):
        g = t.groupby("threshold_name").agg(folds=("fold_id", "nunique"), feasible=("status", lambda s: int((s.str.startswith("feasible") | (s == "fixed")).sum())),
                                            infeasible=("status", lambda s: int((s == "infeasible_no_roc_point_within_budget").sum())),
                                            pooled_fp=("fp", "sum"), pooled_fn=("fn", "sum"), mean_fpr=("fpr", "mean"),
                                            mean_f1=("f1", "mean")).reset_index()
        save(g, f"{part}_table9_sliding_isotonic_summary.csv")
    # status counts across everything
    sc = R.groupby(["protocol", "probability_source", "threshold_name", "status"]).size().reset_index(name="n_folds")
    save(sc, f"{part}_threshold_status_counts.csv")
    if not purged and pooled:
        reliability_figures(pooled)
    return F, R, C


def reliability_figures(pooled):
    for (proto, src), parts in pooled.items():
        y = np.concatenate([a for a, _ in parts]); p = np.concatenate([b for _, b in parts])
        t = bin_table(y, p, np.linspace(0, 1, 16))
        save(t.assign(protocol=proto, probability_source=src), f"P2_reliability_bins_{proto}_{src}.csv")
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={"height_ratios": [3, 1.4]}, sharex=True)
        occ = t[t.n > 0]
        a1.plot([0, 1], [0, 1], ls="--", lw=0.8, color="grey")
        a1.plot(occ.mean_p, occ.frac_attack, marker="o", lw=1.2)
        a1.set_ylabel("Observed attack fraction"); a1.set_title(f"{proto.capitalize()} pooled, RF {src}")
        a2.bar((t.lo + t.hi) / 2, t.n.clip(lower=0.8), width=1 / 15 * 0.9)
        a2.set_yscale("log"); a2.set_ylabel("Flows (log)"); a2.set_xlabel("Predicted attack probability")
        fig.tight_layout(); fig.savefig(OUTPUT_DIR / "figures" / f"Reliability_{proto}_{src}.png", dpi=300); plt.close(fig)


# =============================================================================
# P4  LINEAR SVM DIAGNOSTIC (R1.10)
# =============================================================================
def platt_params(cal):
    try:
        cc = cal.calibrated_classifiers_[0]
        sc = cc.calibrators[0]
        return float(sc.a_), float(sc.b_)
    except Exception:
        return np.nan, np.nan


def margin_stats(prefix, y, m):
    out = {}
    for lab, name in ((0, "benign"), (1, "attack")):
        v = m[y == lab]
        out[f"{prefix}_{name}_n"] = int(len(v))
        if len(v):
            out[f"{prefix}_{name}_median"] = float(np.median(v))
            out[f"{prefix}_{name}_q25"] = float(np.quantile(v, 0.25))
            out[f"{prefix}_{name}_q75"] = float(np.quantile(v, 0.75))
    if (y == 1).any():
        out[f"{prefix}_attack_share_margin_below_0"] = float((m[y == 1] < 0).mean())
    if (y == 0).any():
        out[f"{prefix}_benign_share_margin_above_0"] = float((m[y == 0] >= 0).mean())
    return out


def p4_svm(c):
    mod, cfg = c.mod, c.cfg
    rows, diag = [], []
    for sp in c.splits["sliding"]:
        say(f"P4: {sp.fold_id}")
        roles = frames_for(c, sp)
        y = {k: c.y[v] for k, v in roles.items()}
        if len(np.unique(y["train"])) < 2:
            continue
        xt, xc, xv, xvp, xs, names = preprocess(c, roles, c.strict_clean)
        variants = {"balanced": mod.make_model("LinearSVM", cfg, 10_202),   # seed used for Strict-Clean SVM in the paper
                    "unweighted": LinearSVC(class_weight=None, random_state=cfg.RANDOM_STATE + 10_202)}
        for vname, svm in variants.items():
            svm.fit(xt, y["train"])
            mg = {k: svm.decision_function(x) for k, x in (("cal", xc), ("thr", xv), ("test", xs))}
            pipes = {"raw_margin": (mg["thr"], mg["test"], 0.0)}
            two_cal = len(np.unique(y["cal"])) >= 2
            if two_cal:
                for meth in ("sigmoid", "isotonic"):
                    cal = mod.fit_calibrator(svm, xc, y["cal"], meth)
                    pipes[meth] = (mod.positive_probability(cal, xv), mod.positive_probability(cal, xs), 0.5)
                    if meth == "sigmoid":
                        a, b = platt_params(cal)
            else:
                a, b = np.nan, np.nan
            d = {"fold_id": sp.fold_id, "svm_variant": vname, "platt_a": a, "platt_b": b,
                 "margin_at_sigmoid_0.5": (-b / a) if (a == a and a != 0) else np.nan,
                 "cal_attack_rate": float(y["cal"].mean()), "thr_attack_rate": float(y["thr"].mean()),
                 "test_attack_rate": float(y["test"].mean()), "train_attack_rate": float(y["train"].mean())}
            for role in ("cal", "thr", "test"):
                d.update(margin_stats(role, y[role], mg[role]))
            diag.append(d)
            for pname, (pthr, ptest, default_tau) in pipes.items():
                rules = [rule_record("default", default_tau, "fixed", True, "none"), youden_v2(y["thr"], pthr)]
                rules.append(benign_quantile_v2(pthr[y["thr"] == 0], 0.01, "benign_quantile_raw_1.0pct", "raw_window_benign_only"))
                for r in rules:
                    row = {"fold_id": sp.fold_id, "svm_variant": vname, "score_source": pname, **r}
                    if np.isfinite(r["tau"]):
                        row.update(metric_block(y["test"], (ptest >= r["tau"]).astype(np.int8), ptest))
                    rows.append(row)
        del xt, xc, xv, xvp, xs
        gc.collect()
    R, D = pd.DataFrame(rows), pd.DataFrame(diag)
    save(R, "P4_svm_rules.csv"); save(D, "P4_svm_margin_diagnostics.csv")
    s7 = R[(R.fold_id == "sliding_07") & (R.svm_variant == "balanced") & (R.score_source == "sigmoid") & (R.threshold_name == "default")]
    if len(s7):
        say(f"P4: reproduction check S7 SVM sigmoid 0.5 F1 = {float(s7.f1.iloc[0]):.6f} (paper 0.754367)")


# =============================================================================
# P5  EXTENDED RFE REFERENCE (R1.8)
# =============================================================================
def p5_rfe_reference(c):
    cfg = c.cfg
    sample_seeds = list(cfg.AUDIT_SAMPLE_SEEDS)
    model_seeds = [cfg.RFE_FIXED_MODEL_SEED + 2 + 1000 * i for i in range(P5_N_MODEL_SEEDS)]
    runs = []
    runs_path = OUTPUT_DIR / "summary" / "P5_rfe_runs.csv"
    prev = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()
    done = set(zip(prev.fold_id, prev.sample_size, prev.sample_seed, prev.model_seed)) if len(prev) else set()
    runs = prev.to_dict("records") if len(prev) else []
    for sp in c.splits["sliding"]:
        tr = c.df.iloc[sp.train_idx]
        y = c.y[sp.train_idx]
        if len(np.unique(y)) < 2:
            continue
        dummy = tr.iloc[:1]
        x, _, _, _, _, names = c.mod.fit_transform(tr, dummy, dummy, dummy, dummy, c.strict_clean)
        for size in P5_SAMPLE_SIZES:
            sz = min(size, c.cfg.RFE_MAX_SAMPLES) if SMOKE else size
            for ms in model_seeds:
                for ss in sample_seeds:
                    if (sp.fold_id, size, ss, ms) in done:
                        continue
                    t0 = time.perf_counter()
                    _, sel, st = rfe_select(c, x, y, names, sample_seed=ss, model_seed=ms, max_samples=sz, return_info=True)
                    runs.append({"fold_id": sp.fold_id, "train_rows": int(len(y)), "train_attack_rate": float(y.mean()),
                                 "sample_size": size, "sample_seed": ss, "model_seed": ms,
                                 "rfe_sample_attack_rate": float(st["sample_attack_rate"]),
                                 "seconds": time.perf_counter() - t0, "selected": "|".join(sel)})
            pd.DataFrame(runs).to_csv(runs_path, index=False)
            say(f"P5: {sp.fold_id} size {size} done")
        del x
        gc.collect()
    R = pd.DataFrame(runs)
    J = lambda a, b: len(set(a) & set(b)) / len(set(a) | set(b))
    summ = []
    for (f, size), g in R.groupby(["fold_id", "sample_size"]):
        sets = {(r.sample_seed, r.model_seed): r.selected.split("|") for r in g.itertuples()}
        samp = [J(sets[(a, m)], sets[(b, m)]) for m in model_seeds for a, b in combinations(sample_seeds, 2) if (a, m) in sets and (b, m) in sets]
        est = [J(sets[(s, a)], sets[(s, b)]) for s in sample_seeds for a, b in combinations(model_seeds, 2) if (s, a) in sets and (s, b) in sets]
        allp = [J(sets[a], sets[b]) for a, b in combinations(list(sets), 2)]
        core = set.intersection(*[set(v) for v in sets.values()])
        orig = [J(sets[(a, model_seeds[0])], sets[(b, model_seeds[0])]) for a, b in combinations(sample_seeds, 2)
                if (a, model_seeds[0]) in sets and (b, model_seeds[0]) in sets]
        summ.append({"fold_id": f, "sample_size": size, "train_attack_rate": float(g.train_attack_rate.iloc[0]),
                     "rfe_sample_attack_rate": float(g.rfe_sample_attack_rate.mean()), "train_rows": int(g.train_rows.iloc[0]),
                     "J_sampling_only_mean": np.mean(samp) if samp else np.nan,
                     "J_estimator_only_mean": np.mean(est) if est else np.nan,
                     "J_joint_mean": np.mean(allp) if allp else np.nan, "J_joint_min": np.min(allp) if allp else np.nan,
                     "joint_core_size": len(core), "n_runs": len(sets),
                     "J_paper_protocol_mean (sampling, original estimator seed)": np.mean(orig) if orig else np.nan})
    S = pd.DataFrame(summ)
    save(S, "P5_rfe_reference_summary.csv")
    for f, v in (("sliding_09", 0.8434), ("sliding_02", 0.8140)):
        r = S[(S.fold_id == f) & (S.sample_size == 100_000)]
        if len(r):
            say(f"P5: reproduction {f} 100k sampling-only (original estimator seed) J = "
                f"{float(r.iloc[0]['J_paper_protocol_mean (sampling, original estimator seed)']):.4f} (paper {v})")


# =============================================================================
# P6  WINDOW SENSITIVITY (R1.6)
# =============================================================================
def custom_splits(c, windows, mode, label):
    mod, cfg, df = c.mod, c.cfg, c.df
    out = []
    H = cfg.ROLLING_TRAIN_WINDOWS
    for cand, cal_w in enumerate(range(H, len(windows) - 2), start=1):
        thr_w, test_w = cal_w + 1, cal_w + 2
        ids = list(range(0, cal_w)) if mode == "expanding" else list(range(cal_w - H, cal_w))
        tr, cal, pool, cs, ts, cr, trr = mod._build_past_only_support_pools(df, [windows[i] for i in ids], windows[cal_w], windows[thr_w], cfg)
        out.append(mod.DataSplit(protocol=f"{mode}_temporal", fold_id=f"{label}_{mode}_{cand:02d}", train_idx=tr,
                                 calibration_idx=cal, threshold_idx=windows[thr_w], threshold_pool_idx=pool,
                                 test_idx=windows[test_w], calibration_strategy=cs, threshold_pool_strategy=ts,
                                 calibration_reserve_rows=cr, threshold_reserve_rows=trr, train_window_ids=tuple(ids),
                                 calibration_window_id=cal_w, threshold_window_id=thr_w, test_window_id=test_w,
                                 candidate_number=cand, training_mode=mode))
    return out


def session_boundary(c):
    """Index of the first flow after the largest start-time gap if that gap is >= 24 h
    (computed directly in milliseconds, independent of pandas datetime resolution)."""
    gaps = np.diff(c.start_ms)
    if not len(gaps) or not np.isfinite(gaps).any():
        return None
    k = int(np.nanargmax(gaps))
    return k + 1 if gaps[k] >= 24 * 3.6e6 else None


def alternative_windows(c):
    n, s = len(c.df), c.start_ms
    b = session_boundary(c)
    W = c.cfg.N_TEMPORAL_WINDOWS
    parts = {}
    # (a) fixed duration in active capture time (the inter-session gap is removed from the clock)
    if b is not None:
        active = np.where(np.arange(n) < b, s - s[0], (s[b - 1] - s[0]) + (s - s[b]))
    else:
        active = s - s[0]
    width = active[-1] / W if active[-1] > 0 else 1.0
    wid = np.clip((active // width).astype(int), 0, W - 1)
    fd = []
    for k in range(W):
        w = np.flatnonzero(wid == k)
        if not len(w):
            continue
        if b is not None and w[0] < b <= w[-1]:          # never let a window straddle the session gap
            fd += [w[w < b], w[w >= b]]
        else:
            fd.append(w)
    parts["fixed_duration"] = fd
    # (b) session-aligned equal-count windows, number of windows proportional to session size
    if b is not None:
        w1 = int(round(W * b / n)); w1 = min(max(w1, 1), W - 1)
        parts["session_aligned"] = ([np.asarray(x, np.int64) for x in np.array_split(np.arange(0, b), w1)] +
                                    [np.asarray(x, np.int64) for x in np.array_split(np.arange(b, n), W - w1)])
    return parts


def p6_windows(c):
    for label, windows in alternative_windows(c).items():
        if is_done(f"P6_{label}"):
            say(f"P6 {label}: already done"); continue
        prof = pd.DataFrame([{"window": i, "rows": len(w), "attacks": int(c.y[w].sum()),
                              "elapsed_hours": float((c.start_ms[w[-1]] - c.start_ms[w[0]]) / 3.6e6)} for i, w in enumerate(windows)])
        save(prof, f"P6_{label}_window_profile.csv")
        F, R = [], []
        for mode in ("sliding", "expanding"):
            for sp in custom_splits(c, windows, mode, label):
                say(f"P6 {label}: {sp.fold_id}")
                res = run_main_fold(c, sp, c.strict_clean, label, want_calib=False)
                F.append(res["fold"]); R += [r for r in res["rules"] if r["threshold_name"] in ("fixed_0.5", "youden", "fpr_1.0pct", "benign_quantile_histpool_1.0pct")]
        F, R = pd.DataFrame(F), pd.DataFrame(R)
        save(F, f"P6_{label}_fold_records.csv"); save(R, f"P6_{label}_threshold_records.csv")
        save(protocol_comparison(R, F, label), f"P6_{label}_sliding_vs_expanding.csv")
        mark_done(f"P6_{label}")


# =============================================================================
# P7  LEAVE-ONE-ATTACK-CATEGORY-OUT (R1.11, unseen families)
# =============================================================================
def p7_unseen(c):
    rows = []
    path = OUTPUT_DIR / "summary" / "P7_unseen_category_records.csv"
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
    done = set(zip(prev.fold_id, prev.held_out)) if len(prev) else set()
    rows = prev.to_dict("records") if len(prev) else []
    cats = sorted(set(c.cat[c.y == 1]))
    for sp in c.splits["sliding"]:
        yt = c.y[sp.test_idx]
        if yt.sum() == 0:
            continue
        test_cats = c.cat[sp.test_idx]
        for held in ["NONE"] + cats:
            if (sp.fold_id, held) in done:
                continue
            if held != "NONE" and not (test_cats == held).any():
                continue
            say(f"P7: {sp.fold_id} held-out {held}")
            res = run_main_fold(c, sp, c.strict_clean, f"loco_{held}", drop_categories=None if held == "NONE" else [held], want_calib=False)
            f = res["fold"]
            if "probs_test" not in res:
                rows.append({"fold_id": sp.fold_id, "held_out": held, "status": "not_ready"}); continue
            for src in ("isotonic", "uncalibrated"):
                if src not in res["probs_test"]:
                    continue
                p = res["probs_test"][src]["test"]
                pred = (p >= 0.5).astype(np.int8)
                for cat in (cats if held == "NONE" else [held]):
                    m = test_cats == cat
                    if not m.any():
                        continue
                    rows.append({"fold_id": sp.fold_id, "held_out": held, "category": cat, "probability_source": src,
                                 "train_flows_of_category": int((c.cat[sp.train_idx] == cat).sum()) if held == "NONE" else 0,
                                 "test_flows": int(m.sum()), "detected": int(pred[m].sum()),
                                 "recall_fixed_0.5": float(pred[m].mean()),
                                 "test_fp_fixed_0.5": int(((yt == 0) & (pred == 1)).sum()),
                                 "roc_auc_category_vs_benign": float(roc_auc_score(np.r_[np.zeros((yt == 0).sum()), np.ones(m.sum())],
                                                                                   np.r_[p[yt == 0], p[m]]))})
            pd.DataFrame(rows).to_csv(path, index=False)
    R = pd.DataFrame(rows)
    if len(R) and "category" in R:
        seen = R[(R.held_out == "NONE")].rename(columns={"recall_fixed_0.5": "recall_seen", "detected": "detected_seen"})
        unseen = R[(R.held_out != "NONE")].rename(columns={"recall_fixed_0.5": "recall_unseen", "detected": "detected_unseen",
                                                           "roc_auc_category_vs_benign": "auc_unseen"})
        mg = unseen.merge(seen[["fold_id", "category", "probability_source", "recall_seen", "detected_seen", "roc_auc_category_vs_benign"]],
                          on=["fold_id", "category", "probability_source"], how="left")
        save(mg, "P7_unseen_vs_seen_by_fold.csv")
        agg = mg.groupby(["category", "probability_source"]).agg(test_flows=("test_flows", "sum"), detected_seen=("detected_seen", "sum"),
                                                                 detected_unseen=("detected_unseen", "sum"), folds=("fold_id", "nunique")).reset_index()
        agg["pooled_recall_seen"] = agg.detected_seen / agg.test_flows
        agg["pooled_recall_unseen"] = agg.detected_unseen / agg.test_flows
        save(agg, "P7_unseen_vs_seen_pooled.csv")


# =============================================================================
# P8  CONTROLLED DIFFICULTY (R1.11): remove the most discriminative predictors
# =============================================================================
P8_LEVELS = {
    "D1_minus_7_core": ["DST_TO_SRC_AVG_THROUGHPUT", "MAX_TTL", "MIN_IP_PKT_LEN", "MIN_TTL", "SERVER_TCP_FLAGS",
                        "SHORTEST_FLOW_PKT", "SRC_TO_DST_SECOND_BYTES"],
    "D2_minus_15_E9_selected": ["MIN_TTL", "MAX_TTL", "MIN_IP_PKT_LEN", "DST_TO_SRC_AVG_THROUGHPUT", "SHORTEST_FLOW_PKT",
                                "SERVER_TCP_FLAGS", "SRC_TO_DST_SECOND_BYTES", "TCP_WIN_MAX_OUT", "OUT_BYTES", "TCP_FLAGS",
                                "DST_TO_SRC_SECOND_BYTES", "L7_PROTO", "NUM_PKTS_UP_TO_128_BYTES",
                                "FLOW_DURATION_MILLISECONDS", "OUT_PKTS"],
}


def p8_difficulty(c):
    for level, removed in P8_LEVELS.items():
        if is_done(f"P8_{level}"):
            say(f"P8 {level}: already done"); continue
        F, R, C = [], [], []
        for sp in c.splits["sliding"]:
            say(f"P8 {level}: {sp.fold_id}")
            res = run_main_fold(c, sp, c.strict_clean, level, perturb=set(removed))
            F.append(res["fold"]); R += res["rules"]; C += res["calib"]
        F, R, C = pd.DataFrame(F), pd.DataFrame(R), pd.DataFrame(C)
        save(F, f"P8_{level}_fold_records.csv"); save(R, f"P8_{level}_threshold_records.csv"); save(C, f"P8_{level}_calibration.csv")
        t = R[(R.probability_source == "isotonic") & (R.M_two_class_available == True)]
        if len(t):
            g = t.groupby("threshold_name").agg(folds=("fold_id", "nunique"), pooled_fp=("fp", "sum"), pooled_fn=("fn", "sum"),
                                                mean_fpr=("fpr", "mean"), mean_f1=("f1", "mean"), mean_auc=("roc_auc", "mean"),
                                                min_auc=("roc_auc", "min")).reset_index()
            save(g, f"P8_{level}_threshold_summary_isotonic.csv")
        mark_done(f"P8_{level}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    for sub in ("summary", "perflow", "figures"):
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    for label, path in (("dataset", DATA_FILE), ("original pipeline", ORIGINAL_CODE)):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}\nPut it in the same folder as this script.")
    env = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version, "platform": platform.platform(),
           "processor": platform.processor(), "cpus": os.cpu_count(), "numpy": np.__version__, "pandas": pd.__version__,
           "scikit_learn": sklearn.__version__, "run_parts": RUN_PARTS, "smoke_test": SMOKE}
    try:
        import hashlib
        h = hashlib.sha256()
        with open(DATA_FILE, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        env["dataset_sha256"] = h.hexdigest()
    except Exception as exc:
        env["dataset_sha256"] = f"failed: {exc}"
    (OUTPUT_DIR / "summary" / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    c = build_context()
    plan = [
        ("P1_boundary_audit", lambda: p1_boundary_audit(c)),
        ("P2_threshold_calibration_rerun", lambda: p2_main(c, purged=False)),
        ("P3_purged_rerun", lambda: p2_main(c, purged=True)),
        ("P4_svm_diagnostic", lambda: p4_svm(c)),
        ("P5_rfe_reference", lambda: p5_rfe_reference(c)),
        ("P6_window_sensitivity", lambda: p6_windows(c)),
        ("P7_unseen_category", lambda: p7_unseen(c)),
        ("P8_controlled_difficulty", lambda: p8_difficulty(c)),
    ]
    for name, fn in plan:
        flag = RUN_PARTS.get(name, False)
        if name == "P3_purged_rerun" and flag == "auto":
            f = OUTPUT_DIR / "summary" / "P1_fold_role_overlap_with_test_start.csv"
            flag = f.exists() and pd.read_csv(f).rows_ending_after_test_start.sum() > 0
            say(f"P3 auto decision: {'run' if flag else 'skip'}")
        if not flag:
            continue
        if is_done(name):
            say(f"{name}: already done (delete summary/.{name}.done to rerun)")
            continue
        say(f"===== {name} =====")
        t0 = time.time()
        try:
            fn()
            mark_done(name)
            say(f"{name}: finished in {(time.time()-t0)/60:.1f} min")
        except Exception:
            say(f"{name}: FAILED\n{traceback.format_exc()}")
    say("All requested parts processed. Send back the 'summary' and 'figures' folders.")


if __name__ == "__main__":
    main()
''',
    'cost': r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synchronized computational-cost addendum for the final NF-UNSW-NB15-v3 study.

Measures, in one execution environment and on the same folds:
  1) Strict-Clean RFE selector time (100k cap, 120-tree RF, step=0.20)
  2) RF-15 fit time using the ORIGINAL saved 15-feature set for that fold
  3) RF-46 Strict-Clean baseline fit time
  4) uncompressed (joblib compress=0) serialized model size and total tree nodes

Three repetitions per fold are recommended. RFE reselection is re-run and checked
against the saved original rankings, but RF-15 timing always uses the original
saved selected set so a reproducibility mismatch cannot silently change the arm
being timed.

Preprocessing is performed exactly as in the final pipeline but is deliberately
outside both timing arms because both configurations require the same full
Strict-Clean preprocessing before feature reduction.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("final_temporal_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import final pipeline from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def q1(x: Sequence[float]) -> float:
    return float(np.quantile(np.asarray(x, dtype=float), 0.25))


def q3(x: Sequence[float]) -> float:
    return float(np.quantile(np.asarray(x, dtype=float), 0.75))


def rfe_path(n_features: int, target: int, step_value: float) -> list[int]:
    # Matches sklearn RFE: fractional step is converted ONCE from initial n_features.
    step = int(max(1, step_value * n_features)) if 0.0 < step_value < 1.0 else int(step_value)
    path = [int(n_features)]
    current = int(n_features)
    while current > target:
        current -= min(step, current - target)
        path.append(int(current))
    return path


def training_transform_exact(mod, train: pd.DataFrame, cols: list[str]):
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in cols if c not in numeric]
    transformers = []
    if numeric:
        numeric_pipeline = Pipeline([
            ("sanitize", FunctionTransformer(
                mod.sanitize_numeric_block,
                validate=False,
                feature_names_out="one-to-one",
            )),
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric))
    if categorical:
        # The final Strict-Clean NF-UNSW-NB15-v3 schema is numeric, but fail rather
        # than silently change the final preprocessing if that ever stops being true.
        raise RuntimeError(f"Unexpected categorical Strict-Clean features: {categorical}")
    pre = ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    x = mod.finite_float32(pre.fit_transform(train.loc[:, cols]))
    names = list(pre.get_feature_names_out())
    return x, names


def timed_rfe(mod, cfg, x: np.ndarray, y: np.ndarray, names: list[str], base_index: int = 2):
    sample_seed = int(cfg.RFE_FIXED_SAMPLE_SEED + base_index)
    model_seed = int(cfg.RFE_FIXED_MODEL_SEED + base_index)
    idx, stats = mod.rfe_sample_indices(
        y,
        cfg.RFE_MAX_SAMPLES,
        sample_seed,
        cfg.FULL_RFE_SAMPLING_MODE,
        cfg.RFE_POSITIVE_CAP_FRACTION,
        cfg.RFE_MIN_OTHER_CLASS_SAMPLES,
    )
    xf = np.ascontiguousarray(x[idx])
    yf = np.asarray(y, dtype=np.int8)[idx]
    est = RandomForestClassifier(
        n_estimators=cfg.RFE_N_ESTIMATORS,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=model_seed,
        max_depth=18,
    )
    selector = RFE(
        estimator=est,
        n_features_to_select=cfg.TOP_K_FEATURES,
        step=cfg.RFE_STEP,
    )
    t0 = time.perf_counter()
    selector.fit(xf, yf)
    selector_seconds = time.perf_counter() - t0

    # This additional fit exists in the final study to export feature importances.
    # It is reported separately and excluded from the deployment reselection threshold.
    importance_model = clone(est)
    t0 = time.perf_counter()
    importance_model.fit(xf, yf)
    diagnostic_importance_seconds = time.perf_counter() - t0

    selected_idx = np.flatnonzero(selector.support_)
    selected_names = [names[i] for i in selected_idx]
    ranking = np.asarray(selector.ranking_, dtype=int)
    del xf, yf, selector, importance_model
    gc.collect()
    return selected_idx, selected_names, ranking, stats, selector_seconds, diagnostic_importance_seconds


def serialize_stats(model, path: Path):
    total_nodes = int(sum(tree.tree_.node_count for tree in model.estimators_))
    paths = joblib.dump(model, path, compress=0)
    size_bytes = int(sum(Path(p).stat().st_size for p in paths))
    for p in paths:
        try:
            Path(p).unlink()
        except FileNotFoundError:
            pass
    return total_nodes, size_bytes


def timed_rf(cfg, x: np.ndarray, y: np.ndarray, seed: int, temp_path: Path):
    model = RandomForestClassifier(
        n_estimators=cfg.RF_N_ESTIMATORS,
        max_depth=cfg.RF_MAX_DEPTH,
        min_samples_leaf=cfg.RF_MIN_SAMPLES_LEAF,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    t0 = time.perf_counter()
    model.fit(x, y)
    fit_seconds = time.perf_counter() - t0
    nodes, size_bytes = serialize_stats(model, temp_path)
    del model
    gc.collect()
    return fit_seconds, nodes, size_bytes


def main():
    # RUN-ONLY VERSION
    # Put this script and the three input files below in the SAME folder,
    # then press "Run Python File" in VS Code. No command-line arguments are required.
    script_dir = Path(__file__).resolve().parent

    default_data = script_dir / "NF-UNSW-NB15-v3.csv"
    default_final_code = script_dir / "nf_unsw_nb15_v3_temporal_reliability_18W_DUAL_FINAL_READY_BOOTSTRAP_FIX.py"
    default_expected_rfe = script_dir / "NF_UNSW_NB15_v3_Expected_Strict_Clean_RFE_Rankings.csv"
    default_output_dir = script_dir / "NF_UNSW_NB15_v3_Computational_Cost_Addendum"

    # Arguments remain optional for advanced use, but pressing Run works with defaults.
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(default_data), help="Path to NF-UNSW-NB15-v3.csv")
    ap.add_argument("--final-code", default=str(default_final_code), help="Path to final 18-window Python pipeline")
    ap.add_argument("--expected-rfe", default=str(default_expected_rfe), help="Combined saved Strict-Clean RFE ranking CSV")
    ap.add_argument("--output-dir", default=str(default_output_dir))
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    data_path = Path(args.data).resolve()
    code_path = Path(args.final_code).resolve()
    expected_path = Path(args.expected_rfe).resolve()
    out = Path(args.output_dir).resolve()

    missing = []
    for label, path in [
        ("NF-UNSW-NB15-v3 dataset", data_path),
        ("original final code", code_path),
        ("expected RFE rankings", expected_path),
    ]:
        if not path.exists():
            missing.append(f"  - {label}: {path}")
    if missing:
        raise FileNotFoundError(
            "\nRequired file(s) not found. Put all four files in the same folder as this script:\n"
            + "\n".join(missing)
        )

    out.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("NF-UNSW-NB15-v3 Computational-Cost Addendum — RUN-ONLY VERSION")
    print(f"Data          : {data_path.name}")
    print(f"Original code : {code_path.name}")
    print(f"Expected RFE  : {expected_path.name}")
    print(f"Output folder : {out}")
    print(f"Repetitions   : {args.reps}")
    print("=" * 78, flush=True)

    mod = load_module(code_path)
    cfg = mod.Config()
    cfg.DATA_PATHS = (str(data_path),)
    cfg.EXECUTION_MODE = "full"

    expected = pd.read_csv(expected_path)
    expected = expected[
        (expected["protocol"] == "expanding_temporal")
        & (expected["base_variant"] == "Strict-Clean")
        & expected["candidate_number"].between(4, 13)
    ].copy()
    if expected["fold_id"].nunique() != 10:
        raise RuntimeError("Expected RFE file must contain expanding_04 through expanding_13")

    env = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "n_jobs": -1,
        "joblib_compress": 0,
        "repetitions_per_fold": int(args.reps),
    }
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")

    log_path = out / "dataset_load.log"
    df = mod.load_dataset(cfg, log_path)
    detected = mod.detect_columns(df, cfg)
    df = mod.derive_labels(df, detected)
    df, order_type = mod.prepare_order(df, detected["time"], log_path)
    clean, ttl, strict_present, _ = mod.leakage_audit(df, detected, cfg)
    strict_norm = {mod.normalize_name(c) for c in cfg.STRICT_DROP_FEATURES}
    strict_clean = [c for c in clean if mod.normalize_name(c) not in strict_norm]

    if len(df) != 2_350_609:
        raise RuntimeError(f"Analytic cohort mismatch: {len(df):,} != 2,350,609")
    if len(strict_clean) != 46:
        raise RuntimeError(f"Strict-Clean feature count mismatch: {len(strict_clean)} != 46")

    expected_feature_order = expected[expected["fold_id"] == "expanding_04"]["feature"].astype(str).tolist()
    if strict_clean != expected_feature_order:
        raise RuntimeError("Strict-Clean feature order differs from the saved final ranking files")

    splits = {s.fold_id: s for s in mod.rolling_splits(df, cfg, training_mode="expanding", log_path=log_path)}
    final_rf_seed = int(cfg.RANDOM_STATE + 10_200)  # common controlled seed for both RF timing arms
    path = rfe_path(len(strict_clean), int(cfg.TOP_K_FEATURES), float(cfg.RFE_STEP))

    raw_rows = []
    raw_csv = out / "Computational_Cost_Addendum_Raw_Repeats.csv"

    # Small untimed warm-up to initialize thread pools/import paths before official timing.
    first_split = splits["expanding_04"]
    warm_train = df.iloc[first_split.train_idx[: min(10_000, len(first_split.train_idx))]]
    xw, _ = training_transform_exact(mod, warm_train, strict_clean)
    yw = warm_train["__binary_label__"].to_numpy(np.int8)
    if len(np.unique(yw)) >= 2:
        warm = RandomForestClassifier(n_estimators=5, max_depth=3, n_jobs=-1, random_state=12345)
        warm.fit(xw, yw)
        del warm
    del xw, yw, warm_train
    gc.collect()

    for candidate in range(4, 14):
        fold = f"expanding_{candidate:02d}"
        split = splits[fold]
        train = df.iloc[split.train_idx].copy()
        y = train["__binary_label__"].to_numpy(np.int8)
        prep_t0 = time.perf_counter()
        x, names = training_transform_exact(mod, train, strict_clean)
        preprocessing_seconds = time.perf_counter() - prep_t0
        if names != strict_clean:
            raise RuntimeError(f"Preprocessed feature order changed in {fold}")

        e = expected[expected["fold_id"] == fold].copy()
        original_selected = e.loc[e["selected"].astype(bool), "feature"].astype(str).tolist()
        original_indices = np.array([names.index(f) for f in original_selected], dtype=int)
        x15_original = np.ascontiguousarray(x[:, original_indices])
        expected_ranking = e.set_index("feature").loc[names, "rfe_rank"].to_numpy(dtype=int)

        for rep in range(1, args.reps + 1):
            _, rerun_selected, rerun_ranking, sample_stats, rfe_sec, diag_sec = timed_rfe(
                mod, cfg, x, y, names, base_index=2
            )
            selected_list_match = bool(rerun_selected == original_selected)
            selected_set_match = bool(set(rerun_selected) == set(original_selected))
            ranking_match = bool(np.array_equal(rerun_ranking, expected_ranking))

            p15 = out / f".tmp_{fold}_rep{rep}_rf15.joblib"
            p46 = out / f".tmp_{fold}_rep{rep}_rf46.joblib"
            if rep % 2 == 1:
                fit15, nodes15, size15 = timed_rf(cfg, x15_original, y, final_rf_seed, p15)
                fit46, nodes46, size46 = timed_rf(cfg, x, y, final_rf_seed, p46)
                fit_order = "RF15_then_RF46"
            else:
                fit46, nodes46, size46 = timed_rf(cfg, x, y, final_rf_seed, p46)
                fit15, nodes15, size15 = timed_rf(cfg, x15_original, y, final_rf_seed, p15)
                fit_order = "RF46_then_RF15"

            row = {
                "fold_id": fold,
                "candidate_number": candidate,
                "repetition": rep,
                "fit_order": fit_order,
                "train_rows": int(len(train)),
                "train_attack_rate": float(y.mean()),
                "rfe_sample_rows": int(sample_stats["n_selected_for_rfe"]),
                "rfe_sample_attack_rate": float(sample_stats["sample_attack_rate"]),
                "rfe_elimination_path": "->".join(map(str, path)),
                "n_rfe_elimination_steps": int(len(path) - 1),
                "n_rfe_selector_fits": int(len(path)),
                "n_rfe_total_rf_fits_including_diagnostic_importance": int(len(path) + 1),
                "rfe_selector_seconds": float(rfe_sec),
                "rfe_diagnostic_importance_seconds": float(diag_sec),
                "rfe_full_study_seconds": float(rfe_sec + diag_sec),
                "rf15_fit_seconds": float(fit15),
                "rf46_fit_seconds": float(fit46),
                "selection_plus_rf15_seconds": float(rfe_sec + fit15),
                "full_study_rfe_plus_rf15_seconds": float(rfe_sec + diag_sec + fit15),
                "rf15_total_nodes": int(nodes15),
                "rf46_total_nodes": int(nodes46),
                "rf15_size_bytes_compress0": int(size15),
                "rf46_size_bytes_compress0": int(size46),
                "rerun_selected_list_matches_original": selected_list_match,
                "rerun_selected_set_matches_original": selected_set_match,
                "rerun_ranking_matches_original": ranking_match,
                "rerun_selected_features": "|".join(rerun_selected),
                "original_selected_features_used_for_rf15": "|".join(original_selected),
                "preprocessing_seconds_excluded_from_timing": float(preprocessing_seconds),
                "common_final_rf_seed": final_rf_seed,
                "rfe_sample_seed": int(cfg.RFE_FIXED_SAMPLE_SEED + 2),
                "rfe_model_seed": int(cfg.RFE_FIXED_MODEL_SEED + 2),
                "n_jobs": -1,
                "joblib_compress": 0,
            }
            raw_rows.append(row)
            pd.DataFrame(raw_rows).to_csv(raw_csv, index=False)
            print(
                f"{fold} rep {rep}: RFE={rfe_sec:.2f}s, RF15={fit15:.2f}s, "
                f"RF46={fit46:.2f}s, original-set match={selected_list_match}",
                flush=True,
            )

        del x15_original, x, y, train
        gc.collect()

    raw = pd.DataFrame(raw_rows)
    summary_rows = []
    for fold, g in raw.groupby("fold_id", sort=True):
        def sm(col):
            vals = g[col].to_numpy(dtype=float)
            return float(np.median(vals)), q1(vals), q3(vals), float(q3(vals) - q1(vals))

        r_med, r_q1, r_q3, r_iqr = sm("rfe_selector_seconds")
        d_med, _, _, _ = sm("rfe_diagnostic_importance_seconds")
        f15_med, f15_q1, f15_q3, f15_iqr = sm("rf15_fit_seconds")
        f46_med, f46_q1, f46_q3, f46_iqr = sm("rf46_fit_seconds")
        sp_med, sp_q1, sp_q3, sp_iqr = sm("selection_plus_rf15_seconds")
        full_med, _, _, _ = sm("full_study_rfe_plus_rf15_seconds")
        fit_saving = f46_med - f15_med
        if fit_saving > 0:
            # Strict inequality k > R/(F46-F15): floor(ratio)+1 is exact,
            # whereas ceil(ratio) is wrong in the exact-integer edge case.
            ratio = r_med / fit_saving
            k_min = int(math.floor(ratio) + 1)
        else:
            ratio = np.nan
            k_min = np.nan

        row0 = g.iloc[0]
        delta_sec = sp_med - f46_med
        delta_pct = 100.0 * delta_sec / f46_med
        size15 = int(np.median(g["rf15_size_bytes_compress0"]))
        size46 = int(np.median(g["rf46_size_bytes_compress0"]))
        nodes15 = int(np.median(g["rf15_total_nodes"]))
        nodes46 = int(np.median(g["rf46_total_nodes"]))
        summary_rows.append({
            "fold_id": fold,
            "train_rows": int(row0["train_rows"]),
            "rfe_sample_rows": int(row0["rfe_sample_rows"]),
            "rfe_sample_attack_rate": float(row0["rfe_sample_attack_rate"]),
            "rfe_selector_median_s": r_med,
            "rfe_selector_q1_s": r_q1,
            "rfe_selector_q3_s": r_q3,
            "rfe_selector_iqr_s": r_iqr,
            "rf15_fit_median_s": f15_med,
            "rf15_fit_iqr_s": f15_iqr,
            "rf46_fit_median_s": f46_med,
            "rf46_fit_iqr_s": f46_iqr,
            "selection_plus_rf15_median_s": sp_med,
            "selection_plus_rf15_iqr_s": sp_iqr,
            "full_study_rfe_plus_rf15_median_s": full_med,
            "delta_selection_plus_fit_vs_rf46_s": delta_sec,
            "delta_selection_plus_fit_vs_rf46_percent": delta_pct,
            "per_cycle_rf_fit_saving_s": fit_saving,
            "reselection_ratio_R_over_fit_saving": ratio,
            "minimum_integer_reselection_interval_k_for_net_saving": k_min,
            "rf15_size_bytes_compress0": size15,
            "rf46_size_bytes_compress0": size46,
            "model_size_reduction_percent": 100.0 * (size46 - size15) / size46,
            "rf15_total_nodes": nodes15,
            "rf46_total_nodes": nodes46,
            "total_node_reduction_percent": 100.0 * (nodes46 - nodes15) / nodes46,
            "all_three_selected_lists_match_original": bool(g["rerun_selected_list_matches_original"].all()),
            "all_three_rankings_match_original": bool(g["rerun_ranking_matches_original"].all()),
            "preprocessing_common_and_excluded": True,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "Computational_Cost_Addendum_Per_Fold_Summary.csv", index=False)

    deltas = summary["delta_selection_plus_fit_vs_rf46_percent"].to_numpy(float)
    rfe_shares = summary["rfe_selector_median_s"].to_numpy(float) / summary["selection_plus_rf15_median_s"].to_numpy(float)
    overall = {
        "folds": int(len(summary)),
        "repetitions_per_fold": int(args.reps),
        "feature_count_baseline": 46,
        "feature_count_proposed": 15,
        "feature_reduction_percent": 100.0 * (46 - 15) / 46,
        "folds_where_per_cycle_selection_plus_fit_is_faster": int((deltas < 0).sum()),
        "folds_where_per_cycle_selection_plus_fit_is_slower": int((deltas > 0).sum()),
        "median_delta_percent_across_folds": float(np.median(deltas)),
        "rfe_relative_share_strictly_decreases_across_folds": bool(np.all(np.diff(rfe_shares) < 0)),
        "all_rfe_reruns_match_saved_original_selected_lists": bool(summary["all_three_selected_lists_match_original"].all()),
        "all_rfe_reruns_match_saved_original_rankings": bool(summary["all_three_rankings_match_original"].all()),
        "ordering": order_type,
        "preprocessing_note": "Full Strict-Clean preprocessing is common to both arms and excluded from timed differences.",
        "rfe_cost_note": "Deployment reselection threshold uses RFE selector time only; the extra diagnostic importance fit from the study code is reported separately.",
        "rfe_path_note": f"With sklearn fractional step semantics, 46 features follow {'->'.join(map(str, path))}: {len(path)-1} eliminations, {len(path)} selector RF fits, plus one diagnostic importance RF fit.",
    }
    (out / "Computational_Cost_Addendum_Overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    pd.DataFrame([overall]).to_csv(out / "Computational_Cost_Addendum_Overall.csv", index=False)
    print("\nCompleted. Outputs written to:", out, flush=True)


if __name__ == "__main__":
    main()''',
    'audit': r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IJICS-359541 | Read-only R1-11 and R2-9 audits; never fit any model.

Requires the frozen August source and the matching original NF-UNSW-NB15-v3 CSV.
Dataset and distributed source SHA-256 fingerprints are pinned to the enclosed package.

python IJICS_359541_R1_11_R2_9_ReadOnly_Audits.py --self-test
python IJICS_359541_R1_11_R2_9_ReadOnly_Audits.py --dataset NF-UNSW-NB15-v3.csv --source nf_unsw_nb15_v3_temporal_reliability_18W_DUAL_FINAL_READY_BOOTSTRAP_FIX.py --out R1_11_R2_9_ReadOnly
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_SHA = "4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5"
SOURCE_SHA = "5f51d87016d66d99a41f17eb65e1919635502502b7826e2ddaadaedb766e4b33"
ORIGINAL_SOURCE_SHA_FOR_PROVENANCE = "a3a164daa118e12c859d5bba31c1b02ff730b3bb753a9eed5b18b8bfd3c26848"
EXPECTED_ROWS = 2350609
EXPECTED_ATTACKS = 127679
EXPECTED_WINDOWS = 18
EXPECTED_WINDOW_SIZES = (130590,) * 7 + (130589,) * 11
EXPECTED_ZERO_ATTACK_WINDOWS = (2, 3, 4, 5, 6, 7)
EXPECTED_STRICT_CLEAN_FEATURES = 46


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def verify_guard(path: Path) -> None:
    """Fail if the pinned module ceases to guard its top-level experiment call."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    guarded = False
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                and test.left.id == "__name__" and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq) and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"):
            guarded = any(isinstance(item, ast.Expr) and isinstance(item.value, ast.Call)
                          and isinstance(item.value.func, ast.Name) and item.value.func.id == "main"
                          for item in node.body)
    if not guarded:
        raise AssertionError("Original main() is not protected by __name__ == '__main__'; STOP")
    # Pinned SHA is the primary guarantee. The source's known module-level
    # warning filters/config initialization execute during import, not main().


def import_source(path: Path):
    verify_guard(path)
    spec = importlib.util.spec_from_file_location("ijics_18w_frozen_source_readonly", path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot construct import specification")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("CFG", "load_dataset", "detect_columns", "derive_labels", "prepare_order",
                 "leakage_audit", "rolling_splits", "sanitize_numeric_block"):
        if not hasattr(module, name):
            raise AssertionError(f"Frozen source missing expected API {name!r}")
    return module


def parse_original_times(values: pd.Series) -> pd.Series:
    """Preserve datetime64; numeric fallback must interpret milliseconds, not ns."""
    if pd.api.types.is_datetime64_any_dtype(values):
        out = pd.to_datetime(values, errors="coerce")
    elif pd.api.types.is_numeric_dtype(values):
        out = pd.to_datetime(values, unit="ms", errors="coerce")
    else:
        out = pd.to_datetime(values, errors="coerce")
    if out.isna().any():
        raise AssertionError("Source chronological column contains unparsed/null timestamps")
    if not out.is_monotonic_increasing:
        raise AssertionError("Chronological order not monotonic")
    if out.iloc[0].year != 2015 or out.iloc[-1].year != 2015:
        raise AssertionError("Dates unexpectedly outside capture year 2015")
    return out


def windows_and_sessions(df: pd.DataFrame):
    n = len(df)
    sizes = [len(idx) for idx in np.array_split(np.arange(n, dtype=np.int64), EXPECTED_WINDOWS)]
    if tuple(sizes) != EXPECTED_WINDOW_SIZES:
        raise AssertionError(f"Unexpected window sizes: {sizes}")
    win = np.empty(n, dtype=np.int8)
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    for w in range(EXPECTED_WINDOWS):
        win[offsets[w]:offsets[w + 1]] = w
    time = parse_original_times(df["__time_order__"])
    gaps = time.diff().dt.total_seconds().to_numpy(dtype=np.float64)
    long_gap_positions = np.flatnonzero(gaps > 24 * 3600)
    if len(long_gap_positions) != 1:
        raise AssertionError(f"Expected exactly one >24h session gap; found {len(long_gap_positions)}")
    cut = int(long_gap_positions[0])
    if win[cut - 1] != 8 or win[cut] != 8:
        raise AssertionError("Session cut does not fall inside W8")
    if not (time.iloc[cut - 1].date().isoformat() in ("2015-01-22", "2015-01-23")
            and time.iloc[cut].date().isoformat() == "2015-02-18"):
        raise AssertionError("Unexpected capture dates on either side of W8 cut")
    session_code = (np.arange(n, dtype=np.int64) >= cut).astype(np.int8)
    segment_code = np.where(win <= 8, win, win + 1).astype(np.int8)
    segment_code[(win == 8) & (session_code == 1)] = 9
    segment_names = [f"W{i}" for i in range(8)] + ["W8a", "W8b"] + [f"W{i}" for i in range(9, 18)]
    assert len(segment_names) == 19
    if not (np.count_nonzero(segment_code == 8) > 0 and np.count_nonzero(segment_code == 9) > 0):
        raise AssertionError("W8 halves must each be nonempty")
    if not (np.all(session_code[win < 8] == 0) and np.all(session_code[win > 8] == 1)):
        raise AssertionError("Unexpected session boundary outside W8")
    gap_hours = float(gaps[cut] / 3600.0)
    span_hours = float((time.iloc[offsets[9] - 1] - time.iloc[offsets[8]]).total_seconds() / 3600)
    if abs(span_hours - 625.4921411111111) > 0.01:
        raise AssertionError("W8 span differs from frozen audit")
    return win, segment_code, segment_names, session_code, time, cut, gap_hours, span_hours


def check_split_consistency(df, mod, cfg, win, out):
    """Read-only fold reconstruction, including exact test-window ID consistency."""
    result = {}
    for mode in ("sliding", "expanding"):
        folds = mod.rolling_splits(df, cfg, training_mode=mode, log_path=out / "audit_log.txt")
        if not folds:
            raise AssertionError(f"No eligible {mode} folds reconstructed")
        for split in folds:
            test = np.asarray(split.test_idx, dtype=np.int64)
            if not len(test) or np.any(win[test] != split.test_window_id):
                raise AssertionError(f"{mode} {split.fold_id}: actual test observations do not match test_window_id")
            if int(split.test_window_id) != int(split.calibration_window_id) + 2:
                raise AssertionError(f"{mode} {split.fold_id}: nonstandard role ordering")
            # Check ALL past-only support roles.
            for attr in ("train_idx", "calibration_idx", "threshold_idx", "threshold_pool_idx"):
                indices = np.asarray(getattr(split, attr), dtype=np.int64)
                if len(indices) and (np.min(indices) < 0 or np.max(indices) >= np.min(test)):
                    raise AssertionError(f"{mode} {split.fold_id}: {attr} reaches future test data")
        result[mode] = folds
    return result


def r111_audit(df, folds_by_mode, win, segment_code, segment_names, out):
    y = df["__binary_label__"].to_numpy(dtype=np.int8)
    cat = df["__attack_category__"].astype("string").fillna("Unknown")
    catcodes, names = pd.factorize(cat, sort=False)
    names = list(map(str, names))
    first_seen = {}
    window_rows = []
    for s, segment in enumerate(segment_names):
        ix = np.flatnonzero(segment_code == s)
        attacks = ix[y[ix] == 1]
        keys, counts = np.unique(catcodes[attacks], return_counts=True)
        original_window = "W8" if segment in ("W8a", "W8b") else segment
        for k, count in zip(keys.tolist(), counts.tolist()):
            category = names[k]
            novel = category not in first_seen
            if novel:
                first_seen[category] = segment
            first_segment = first_seen[category]
            window_rows.append({"window": original_window, "window_segment": segment,
                                "capture_session": "Jan22_23" if s <= 8 else "Feb18",
                                "attack_category": category, "attack_rows": int(count),
                                "first_seen_segment": first_segment,
                                "first_seen_window": "W8" if first_segment in ("W8a", "W8b") else first_segment,
                                "first_seen_here": novel})
    win_df = pd.DataFrame(window_rows, columns=["window", "window_segment", "capture_session",
                                                "attack_category", "attack_rows", "first_seen_segment",
                                                "first_seen_window", "first_seen_here"])
    win_df.to_csv(out / "R1_11_Window_Category_FirstSeen.csv", index=False)

    fold_rows, detail_rows = [], []
    for mode, folds in folds_by_mode.items():
        for split in folds:
            model_idx = np.asarray(split.train_idx, dtype=np.int64)
            test_idx = np.asarray(split.test_idx, dtype=np.int64)
            past_parts = [np.asarray(getattr(split, attr), dtype=np.int64) for attr in
                          ("train_idx", "calibration_idx", "threshold_idx", "threshold_pool_idx")]
            # Every record used before this fold's evaluation, including support reserves.
            past_used = np.unique(np.concatenate(past_parts))
            past_chrono = np.flatnonzero(win < int(split.test_window_id))
            train_c = set(catcodes[model_idx[y[model_idx] == 1]].tolist())
            used_c = set(catcodes[past_used[y[past_used] == 1]].tolist())
            earlier_c = set(catcodes[past_chrono[y[past_chrono] == 1]].tolist())
            attacks = test_idx[y[test_idx] == 1]
            values, counts = np.unique(catcodes[attacks], return_counts=True)
            model_new = used_new = chrono_new = 0
            for category_code, count in zip(values.tolist(), counts.tolist()):
                nt = category_code not in train_c
                nu = category_code not in used_c
                nc = category_code not in earlier_c
                model_new += count * int(nt)
                used_new += count * int(nu)
                chrono_new += count * int(nc)
                detail_rows.append({"protocol": mode, "fold_id": split.fold_id,
                                    "candidate_number": int(split.candidate_number),
                                    "test_window": f"W{split.test_window_id}",
                                    "attack_category": names[category_code], "test_attack_rows": count,
                                    "unseen_in_model_training": nt,
                                    "unseen_in_permitted_support": nu,
                                    "first_seen_chronologically": nc})
            fold_rows.append({"protocol": mode, "fold_id": split.fold_id,
                              "candidate_number": int(split.candidate_number),
                              "train_window_ids": ",".join(map(str, split.train_window_ids)),
                              "test_window": f"W{split.test_window_id}",
                              "train_rows": len(model_idx), "test_rows": len(test_idx),
                              "test_attack_rows": len(attacks),
                              "test_attack_rows_unseen_model": model_new,
                              "test_attack_rows_unseen_permitted_support": used_new,
                              "test_attack_rows_first_seen_chronologically": chrono_new,
                              "test_class_count": int(np.unique(y[test_idx]).size)})
    pd.DataFrame(fold_rows).to_csv(out / "R1_11_Fold_Known_vs_FirstSeen.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(out / "R1_11_Fold_Category_Detail.csv", index=False)
    return {"attack_categories_total": len(first_seen),
            "first_seen_attack_flows_in_first_observed_segment": int(win_df.loc[win_df.first_seen_here, "attack_rows"].sum()),
            "n_eligible_fold_comparisons": len(fold_rows)}


def binc(flags, codes, n):
    return np.bincount(codes, weights=np.asarray(flags, dtype=np.uint8), minlength=n).astype(np.int64)


def r29_audit(df, mod, cfg, win, segment_code, segment_names, session_code, out):
    detected = mod.detect_columns(df, cfg)
    candidate, _, strict_excluded, eligible_table = mod.leakage_audit(df, detected, cfg)
    selected = [col for col in candidate if col not in set(strict_excluded)]
    eligible_table.to_csv(out / "R2_9_Original_Feature_Eligibility.csv", index=False)
    if len(selected) != EXPECTED_STRICT_CLEAN_FEATURES:
        raise AssertionError(f"Strict-Clean expected 46 features; found {len(selected)}")
    y = df["__binary_label__"].to_numpy(dtype=np.int8)
    categories = df["__attack_category__"].astype("string").fillna("Unknown")
    category_codes, category_names = pd.factorize(categories, sort=False)
    category_names = list(map(str, category_names))
    n_cat = len(category_names)
    # Original train[col] dtypes remain unchanged by row slicing; follow those
    # same numeric/categorical branches as original fit_transform.
    # Compact 19 x 2 x attack-category bin code; avoid constructing millions of
    # string tuples / giant DataFrames in limited-RAM environments.
    n_bins = 19 * 2 * n_cat
    code = (segment_code.astype(np.int64) * 2 + y.astype(np.int64)) * n_cat + category_codes
    nonzero = np.unique(code)
    base_counts = np.bincount(code, minlength=n_bins).astype(np.int64)
    fields = ("raw_na_cells", "original_numeric_inf_cells", "original_numeric_extreme_cells",
              "original_numeric_coercion_failure_cells", "original_categorical_na_cells",
              "original_imputation_candidate_cells", "diagnostic_object_numeric_like_cells",
              "diagnostic_object_inf_cells", "diagnostic_object_extreme_cells",
              "diagnostic_object_failed_coercion_cells")
    aggregates = {k: np.zeros(n_bins, dtype=np.int64) for k in fields}
    affected_any = np.zeros(len(df), dtype=bool)
    feature_rows = []
    for i, col in enumerate(selected, 1):
        raw = df[col]
        raw_na = raw.isna().to_numpy(dtype=bool)
        original_numeric = bool(pd.api.types.is_numeric_dtype(raw))
        flags = {"raw_na_cells": raw_na}
        if original_numeric:
            # Original fit_transform selects 'num' by train[col].dtype, which
            # matches the dtype in every row subset of this frozen DataFrame.
            vals = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=np.float64)
            inf = np.isinf(vals)
            extreme = np.isfinite(vals) & (np.abs(vals) > 1e30)
            failed = np.isnan(vals) & ~raw_na
            candidates = raw_na | inf | extreme | failed
            sanitized = mod.sanitize_numeric_block(df[[col]])
            if sanitized.shape != (len(df), 1):
                raise AssertionError(f"Original sanitizer returned unexpected shape for {col}")
            if not np.array_equal(np.isnan(sanitized[:, 0]), candidates):
                raise AssertionError(f"Original sanitizer does not match audited mask in {col}")
            flags.update(original_numeric_inf_cells=inf,
                         original_numeric_extreme_cells=extreme,
                         original_numeric_coercion_failure_cells=failed,
                         original_imputation_candidate_cells=candidates)
            branch = "numeric"
        else:
            # Original preprocessing treats object/string dtype as categorical,
            # including object columns that visually resemble numeric strings.
            candidates = raw_na
            flags.update(original_categorical_na_cells=raw_na,
                         original_imputation_candidate_cells=candidates)
            branch = "categorical"
            # Separate diagnostic probe; DO NOT classify these as observed
            # numeric replacements or claim original preprocessing used them.
            parsed = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=np.float64)
            meaningful = ~raw_na
            numeric_like = meaningful & ~np.isnan(parsed)
            flags.update(diagnostic_object_numeric_like_cells=numeric_like,
                         diagnostic_object_inf_cells=meaningful & np.isinf(parsed),
                         diagnostic_object_extreme_cells=meaningful & np.isfinite(parsed) & (np.abs(parsed) > 1e30),
                         diagnostic_object_failed_coercion_cells=meaningful & np.isnan(parsed))
        affected_any |= candidates
        item = {"feature": col, "original_dtype": str(raw.dtype),
                "original_preprocessing_branch": branch}
        for field, mask in flags.items():
            counts = binc(mask, code, n_bins)
            aggregates[field] += counts
            item[field] = int(counts.sum())
        feature_rows.append(item)
        if i % 10 == 0 or i == len(selected):
            print(f"[R2-9] inspected {i}/{len(selected)} features", flush=True)

    records = []
    for k in nonzero:
        seg = int(k // (2 * n_cat))
        binary = int((k // n_cat) % 2)
        cat = int(k % n_cat)
        name = segment_names[seg]
        original_window = "W8" if name in ("W8a", "W8b") else name
        session = "Jan22" if seg <= 8 else "Feb18"
        record = {"window": original_window, "window_segment": name,
                  "capture_session": session, "binary_class": "Attack" if binary else "Normal",
                  "attack_category": category_names[cat], "n_rows": int(base_counts[k])}
        record.update({field: int(values[k]) for field, values in aggregates.items()})
        records.append(record)
    affected_per_bin = binc(affected_any, code, n_bins)
    for row, k in zip(records, nonzero):
        row["affected_records_any_feature"] = int(affected_per_bin[k])
        row["pct_affected_records"] = 100 * row["affected_records_any_feature"] / row["n_rows"]
    detailed = pd.DataFrame(records)
    detailed.to_csv(out / "R2_9_Quality_By_Window_Session_Class.csv", index=False)
    measures = list(fields) + ["n_rows", "affected_records_any_feature"]
    window = detailed.groupby(["window", "window_segment", "capture_session"], sort=False, as_index=False)[measures].sum()
    window["pct_affected_records"] = 100 * window.affected_records_any_feature / window.n_rows
    window.to_csv(out / "R2_9_Quality_By_Window_Session.csv", index=False)
    pd.DataFrame(feature_rows).fillna(0).to_csv(out / "R2_9_Quality_By_Feature.csv", index=False)
    return {"n_strict_clean_features": len(selected),
            "original_imputation_candidate_cells": int(aggregates["original_imputation_candidate_cells"].sum()),
            "affected_records_any_feature": int(affected_any.sum()),
            "n_object_columns_with_numeric_like_values": sum(x["original_preprocessing_branch"] == "categorical" and x.get("diagnostic_object_numeric_like_cells", 0) > 0 for x in feature_rows),
            "interpretation": "Original numeric sanitizer and categorical imputers are distinguished. Replacement counts are pre-fit candidate cells, not empirically fitted per-fold imputations. Object-to-numeric probes are diagnostics only."}


def self_test():
    from types import SimpleNamespace
    # Numeric milliseconds MUST not be misread as nanoseconds.
    millis = pd.Series([pd.Timestamp("2015-01-22 20:00:00").value // 10**6,
                        pd.Timestamp("2015-02-18 01:00:00").value // 10**6], dtype=np.int64)
    assert parse_original_times(millis).iloc[0].year == 2015
    assert parse_original_times(pd.to_datetime(millis, unit="ms")).iloc[1].year == 2015
    # Independent sanitizer edge conditions are checked against the pinned
    # source when real data run begins; this is only a logic smoke test.
    vals = np.array([np.nan, np.inf, -np.inf, 1e31, 1e30, 0.0])
    bad = ~np.isfinite(vals) | (np.abs(vals) > 1e30)
    assert bad.tolist() == [True, True, True, True, False, False]
    assert (np.array([130590] * 7 + [130589] * 11).sum() == EXPECTED_ROWS)
    print("SELF-TEST PASS: millisecond timestamps, datetime timestamps, threshold strictness, 18-window sizes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, default=Path("R1_11_R2_9_ReadOnly"))
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.dataset is None or args.source is None:
        parser.error("--dataset and --source are required except for --self-test")
    # Validate original inputs before importing code or loading any observations.
    for kind, path, expected in (("dataset", args.dataset, DATA_SHA), ("source", args.source, SOURCE_SHA)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {kind} file: {path}")
        actual = digest(path)
        if actual != expected:
            raise AssertionError(f"{kind} SHA-256 mismatch: {actual}; expected {expected}; STOP")
        print(f"{kind} SHA-256 verified", flush=True)
    mod = import_source(args.source)
    cfg = replace(mod.CFG, DATA_PATHS=(str(args.dataset.resolve()),),
                  N_TEMPORAL_WINDOWS=EXPECTED_WINDOWS, MAX_ROWS_PER_FILE=None,
                  DROP_GLOBAL_EXACT_DUPLICATES=True)
    args.out.mkdir(parents=True, exist_ok=True)
    df = mod.load_dataset(cfg, args.out / "audit_log.txt")
    if len(df) != EXPECTED_ROWS:
        raise AssertionError(f"Deduplicated row mismatch: {len(df)}")
    detected = mod.detect_columns(df, cfg)
    df = mod.derive_labels(df, detected)
    df, reason = mod.prepare_order(df, detected["time"], args.out / "audit_log.txt")
    if reason != "time_column:FLOW_START_MILLISECONDS":
        raise AssertionError(f"Unexpected time ordering: {reason}")
    if int(df["__binary_label__"].sum()) != EXPECTED_ATTACKS:
        raise AssertionError("Post-sort attack total changed")
    win, segments, segment_names, sessions, time, cut, gap_hours, span_hours = windows_and_sessions(df)
    y = df["__binary_label__"].to_numpy(dtype=np.int8)
    if any(int(np.count_nonzero(y[win == i])) != 0 for i in EXPECTED_ZERO_ATTACK_WINDOWS):
        raise AssertionError("Original attack-free window control mismatch")
    folds = check_split_consistency(df, mod, cfg, win, args.out)
    r1 = r111_audit(df, folds, win, segments, segment_names, args.out)
    r2 = r29_audit(df, mod, cfg, win, segments, segment_names, sessions, args.out)
    manifest = {
        "status": "read_only_completed", "source_sha256": SOURCE_SHA, "dataset_sha256": DATA_SHA,
        "deduplicated_rows": int(len(df)), "attack_rows": int(y.sum()),
        "number_original_windows": 18, "expected_window_sizes": list(EXPECTED_WINDOW_SIZES),
        "true_capture_gap_split_row_index_zero_based": cut,
        "true_gap_hours": gap_hours, "W8_full_span_hours": span_hours,
        "W8a_rows": int(np.count_nonzero(segments == 8)),
        "W8b_rows": int(np.count_nonzero(segments == 9)),
        "R1_11": r1, "R2_9": r2,
        "limits": ["No model training or scoring performed",
                   "A read-only descriptive W8 session split does not alter the original fold partition",
                   "Original missing-value imputation is fitted per fold on permitted training rows; counts are candidate cells rather than actually fitted imputer output"]}
    (args.out / "ReadOnly_Audit_Manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
''',
    'remaining': r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IJICS-359541: strictly bounded reviewer follow-up experiments.

Run the explicit --task one at a time. FROZEN source and data hashes are mandatory.
No changed dataset, model search, alternative partition sweep, or test-set k tuning.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import gc, hashlib, json, os, sys, time, traceback
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, confusion_matrix, average_precision_score,
                             roc_auc_score, brier_score_loss)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# This is the exact auditor distributed in the previously verified read-only kit.
import IJICS_359541_R1_11_R2_9_ReadOnly_Audits as audit

ORIGINAL_SOURCE = audit.SOURCE_SHA
ORIGINAL_DATA = audit.DATA_SHA
NROWS=2350609
NATTACK=127679
W8A=87190
W8B=43399
MAX_FEATURES=6  # identical candidate count for original RF-46 / RF-15 arms
ALLOWED_TASKS=("plan", "r1-6", "r1-11", "r2-3", "r2-9", "self-test", "pack")

def jwrite(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str,allow_nan=False),encoding="utf-8")

def sha_array(ix):
    ix=np.asarray(ix,dtype=np.int64)
    return hashlib.sha256(ix.tobytes()).hexdigest()

def roles(split):
    return {name:sha_array(getattr(split,name)) for name in (
        "train_idx","calibration_idx","threshold_idx","threshold_pool_idx","test_idx")}

def same_roles(a,b):
    return roles(a)==roles(b)

def get_windows(n,cut):
    old=[np.asarray(x,dtype=np.int64) for x in np.array_split(np.arange(n,dtype=np.int64),18)]
    assert len(old)==18 and len(old[8])==130589 and old[8][0]<cut<=old[8][-1]
    new=old[:8]+[old[8][old[8]<cut],old[8][old[8]>=cut]]+old[9:]
    names=[f"W{x}" for x in range(8)]+["W8a","W8b"]+[f"W{x}" for x in range(9,18)]
    assert len(new)==len(names)==19
    assert len(new[8])==W8A and len(new[9])==W8B
    assert sum(map(len,new))==n
    assert np.array_equal(np.concatenate(new),np.arange(n,dtype=np.int64))
    return old,new,names

def make_new_splits(df,mod,cfg,new,names,mode):
    """Use frozen past-only support function; Expanding retains W8 block when whole in history.
    A W8a/W8b split must not change support-reserve slices of complete W8 history.
    """
    out=[]
    for cal_w in range(cfg.ROLLING_TRAIN_WINDOWS,len(new)-2):
        thr_w=cal_w+1; test_w=cal_w+2
        train_ids=list(range(cal_w)) if mode=="expanding" else list(range(cal_w-3,cal_w))
        history=[];j=0
        while j<len(train_ids):
            idx=train_ids[j]
            if mode=="expanding" and idx==8 and j+1<len(train_ids) and train_ids[j+1]==9:
                history.append(np.concatenate((new[8],new[9])))
                j+=2
            else:
                history.append(new[idx]);j+=1
        (train,cal,pool,cal_strategy,thr_strategy,cal_reserve,thr_reserve) = (
            mod._build_past_only_support_pools(df,history,new[cal_w],new[thr_w],cfg))
        if mod._class_count(df,train)<2:
            continue
        cand=cal_w-2
        out.append(mod.DataSplit(
            protocol=f"{mode}_session_aligned",fold_id=f"{mode}_alt_{cand:02d}",
            candidate_number=cand,train_idx=train,calibration_idx=cal,
            threshold_idx=new[thr_w],threshold_pool_idx=pool,test_idx=new[test_w],
            calibration_strategy=cal_strategy,threshold_pool_strategy=thr_strategy,
            calibration_reserve_rows=cal_reserve,threshold_reserve_rows=thr_reserve,
            train_window_ids=tuple(train_ids),calibration_window_id=cal_w,
            threshold_window_id=thr_w,test_window_id=test_w,training_mode=mode))
    return out

def original_window_name(w):return f"W{w}"

def plan_partition(df,mod,cfg,old,new,names):
    output=[]; map_old={};map_new={}
    for mode in ("sliding","expanding"):
        o=mod.rolling_splits(df,cfg,training_mode=mode)
        n=make_new_splits(df,mod,cfg,new,names,mode)
        # Testing row IDs, not fold sequence number, define matching.
        oldby={sha_array(z.test_idx):z for z in o}
        newby={sha_array(z.test_idx):z for z in n}
        map_old[mode]=oldby;map_new[mode]=newby
        for sp in n:
            tid=names[sp.test_window_id]
            oldmatch=oldby.get(sha_array(sp.test_idx))
            eq=bool(oldmatch is not None and same_roles(sp,oldmatch))
            if mode=="expanding" and eq is False and oldmatch is not None:
                # Complete W8 in historical model support should remain exactly unchanged
                # UNLESS calibration/threshold/test uses a separate W8 half.
                original_hist=set(range(oldmatch.calibration_window_id))
                new_hist=set(range(sp.calibration_window_id))
                if 8 in original_hist and {8,9}.issubset(new_hist):
                    assert sha_array(sp.train_idx)==sha_array(oldmatch.train_idx), (
                        f"Expanding complete-W8 train history unexpectedly changed for {tid}")
            yy=df.iloc[sp.test_idx]["__binary_label__"].to_numpy(dtype=np.int8)
            caly=df.iloc[sp.calibration_idx]["__binary_label__"].to_numpy(dtype=np.int8)
            output.append(dict(mode=mode,partition="session_aligned",test_window=tid,
                fold_id=sp.fold_id,original_fold_id=oldmatch.fold_id if oldmatch else "",
                same_test_as_original=bool(oldmatch is not None),
                all_roles_identical=eq,affected=not eq,
                old_test_sha256=sha_array(oldmatch.test_idx) if oldmatch else "",
                new_test_sha256=sha_array(sp.test_idx),
                n_train=int(len(sp.train_idx)),n_calibration=int(len(sp.calibration_idx)),
                n_threshold=int(len(sp.threshold_idx)),n_threshold_pool=int(len(sp.threshold_pool_idx)),
                n_test=int(len(sp.test_idx)),n_test_attack=int(yy.sum()),
                eligible_discrimination=bool(np.unique(yy).size==2),
                eligible_isotonic=bool(np.unique(caly).size==2 and np.unique(yy).size==2),
                original_published_seven=bool(oldmatch is not None and
                        oldmatch.fold_id in ([f"sliding_{i:02d}" for i in range(7,14)]
                                              if mode=="sliding" else
                                              [f"expanding_{i:02d}" for i in range(7,14)]))))
    return pd.DataFrame(output),map_old,map_new

def get_base_features(mod,df,cfg):
    columns=mod.detect_columns(df,cfg)
    clean,ttl,strict_present,leak=mod.leakage_audit(df,columns,cfg)
    bad={mod.normalize_name(c) for c in cfg.STRICT_DROP_FEATURES}
    cols=[c for c in clean if mod.normalize_name(c) not in bad]
    if len(cols)!=46:
        raise AssertionError(f"Strict-clean feature count {len(cols)} !=46; STOP")
    return cols

def load_all(dataset,source,out):
    for label,path,expected in (("dataset",dataset,ORIGINAL_DATA),("source",source,ORIGINAL_SOURCE)):
        if not path.is_file():raise FileNotFoundError(f"Missing {label} {path}")
        got=audit.digest(path)
        if got!=expected:raise AssertionError(f"{label} SHA256 mismatch: {got}")
        print(f"{label} SHA-256 verified",flush=True)
    mod=audit.import_source(source)
    cfg=replace(mod.CFG, DATA_PATHS=(str(dataset.resolve()),),EXECUTION_MODE="preflight",
                N_TEMPORAL_WINDOWS=18,MAX_ROWS_PER_FILE=None, DROP_GLOBAL_EXACT_DUPLICATES=True,
                ENFORCE_TEMPORAL_DUPLICATE_ISOLATION=False, SAVE_DETAILED_PLOTS=False)
    out.mkdir(parents=True,exist_ok=True)
    df=mod.load_dataset(cfg,out/"load.log")
    assert len(df)==NROWS
    detected=mod.detect_columns(df,cfg)
    df=mod.derive_labels(df,detected)
    df,reason=mod.prepare_order(df,detected["time"],out/"load.log")
    assert reason=="time_column:FLOW_START_MILLISECONDS"
    assert int(df["__binary_label__"].sum())==NATTACK
    win,seg,names,sessions,timev,cut,gap,span = audit.windows_and_sessions(df)
    old,new,names2=get_windows(len(df),cut)
    assert names==names2
    cols=get_base_features(mod,df,cfg)
    return mod,cfg,df,old,new,names,cols,cut

def train_arrays(mod,df,sp,cols):
    t0=time.perf_counter()
    idxs=(sp.train_idx,sp.calibration_idx,sp.threshold_idx,sp.threshold_pool_idx,sp.test_idx)
    raw=[df.iloc[idx] for idx in idxs]
    labels=[d["__binary_label__"].to_numpy(dtype=np.int8) for d in raw]
    xx=mod.fit_transform(*raw,cols)
    mats=tuple(xx[:5]);names=list(xx[5]);prep_sec=time.perf_counter()-t0
    return raw,labels,mats,names,prep_sec

def choose_features(mod,xtr,ytr,names,cfg):
    t0=time.perf_counter()
    rfe=mod.run_rfe(xtr,ytr,names,cfg,
        sampling_mode=cfg.FULL_RFE_SAMPLING_MODE,
        sample_seed=cfg.RFE_FIXED_SAMPLE_SEED+2,
        model_seed=cfg.RFE_FIXED_MODEL_SEED+2)
    return rfe,time.perf_counter()-t0

def model_offset(which):return 10200 if which==46 else 10210

def new_rf(mod,cfg,nfeatures,which,max_candidates=MAX_FEATURES):
    # For candidate-matched 46-vs-15 controls, hold the RF estimator seed
    # identical. Use the original Strict-Clean-RFE-TopK RF seed in every arm.
    model=mod.make_model("RF",cfg,offset=10210)
    model.set_params(max_features=max_candidates,random_state=cfg.RANDOM_STATE+10210)
    if nfeatures<max_candidates:raise AssertionError("Fewer features than matched candidates")
    return model

def calc_metrics(mod,y,prob,tau):
    pred=(prob>=tau).astype(np.int8)
    cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=map(int,cm.ravel())
    ans=dict(macro_f1=float(f1_score(y,pred,average="macro",zero_division=0)),
        fp=fp,fn=fn,tp=tp,tn=tn,fpr=float(fp/(fp+tn)) if (fp+tn) else None,
        pr_auc=float(average_precision_score(y,prob)) if np.unique(y).size==2 else None,
        roc_auc=float(roc_auc_score(y,prob)) if np.unique(y).size==2 else None,
        brier=float(brier_score_loss(y,prob)))
    ans["ece_15"]=float(mod.ece_mce(y,prob,bins=15)[0])
    return ans,pred

def calibrated(mod,model,xcal,ycal,xval,xpool,xtest):
    cal=mod.fit_calibrator(model,xcal,ycal,"isotonic")
    if cal is None:raise AssertionError("No two-class calibration support")
    out=[]
    for x in (xval,xpool,xtest):
        a=mod.positive_probability(cal,x)
        if not np.isfinite(a).all():raise AssertionError("Nonfinite probabilities")
        out.append(a)
    return cal,out

def evaluate_prepared(mod,cfg,mats,labels,select,which,protocol,fold_id,window,arm="base",
                      extra=None,max_candidates=MAX_FEATURES):
    # Shared exact rows/labels and frozen RFE feature list; a 3 vs 3 RF split count not silently assumed.
    if extra is None: extra={}
    if select is None:select=np.arange(mats[0].shape[1])
    x=[np.ascontiguousarray(a[:,select]) for a in mats]
    if len(select)<max_candidates:raise AssertionError("RF candidate count exceeds feature count")
    ytr,ycal,yval,ypool,ytest=labels
    model=new_rf(mod,cfg,len(select),which,max_candidates=max_candidates)
    t0=time.perf_counter();model.fit(x[0],ytr);fit_sec=time.perf_counter()-t0
    cal,pp=calibrated(mod,model,x[1],ycal,x[2],x[3],x[4])
    pv,pool,test=pp
    rules=mod.threshold_rules(yval,pv,ypool,pool,cfg)
    required={"fixed_0.5","youden","mu_sigma_raw_k_0.10","fpr_0.5pct",
              "fpr_1.0pct","fpr_2.0pct","fpr_histpool_0.5pct"}
    out=[];fixed_pred=None
    for rule in rules:
        if rule["threshold_name"] not in required:continue
        tau=float(rule["threshold_value"])
        metrics,pred=calc_metrics(mod,ytest,test,tau)
        out.append(dict(partition=protocol,fold_id=fold_id,test_window=window,
            arm=arm,feature_set=which,max_features=max_candidates,
            rf_random_state_common=int(cfg.RANDOM_STATE+10210),n_train=len(ytr),
            n_test=len(ytest),n_attack_test=int(ytest.sum()),n_selected=len(select),
            rf_fitting_seconds=fit_sec,threshold_name=rule["threshold_name"],
            threshold_value=tau,threshold_selection_status=rule["threshold_selection_status"],
            threshold_method_eligible=bool(rule["threshold_method_eligible"]),
            **metrics,**extra))
        if rule["threshold_name"]=="fixed_0.5":fixed_pred=pred
    return out,model,cal,fixed_pred,test

def one_fold(df,mod,cfg,sp,cols,arm_prefix,window,out_dir,plan_signature):
    """Matched six-candidate control: Strict-Clean 46 vs frozen-source RFE 15.
    A separate structured file is written only when both have finished.
    """
    raw,labels,xx,names,prep_sec=train_arrays(mod,df,sp,cols)
    if xx[0].shape[1]!=46:raise AssertionError("Expected 46 transformed features")
    rfe,selection_sec=choose_features(mod,xx[0],labels[0],names,cfg)
    if len(rfe.selected)!=15:raise AssertionError("RFE did not select 15 features")
    results=[]
    for size,select in ((46,None),(15,rfe.selected)):
        rr,model,cal,pred,p = evaluate_prepared(
            mod,cfg,xx,labels,select,size,arm_prefix,sp.fold_id,window,
            extra=dict(feature_selection_plus_rf_fitting_seconds=None,
                       preprocessing_seconds=prep_sec,rfe_seconds=selection_sec,
                       role_signature=plan_signature))
        for z in rr:
            z["feature_selection_plus_rf_fitting_seconds"] = z["rf_fitting_seconds"]+(selection_sec if size==15 else 0.0)
        results+=rr
        del model,cal,pred,p
        gc.collect()
    del raw,xx,labels
    gc.collect()
    return pd.DataFrame(results),list(rfe.selected_names)


def do_r16(df,mod,cfg,old,new,names,cols,out):
    plan,oldmap,newmap=plan_partition(df,mod,cfg,old,new,names)
    plan.to_csv(out/"R1_6_Exact_Role_and_Eligibility_Plan.csv",index=False)
    rows=[]
    for mode in ("sliding","expanding"):
        o=oldmap[mode];nn=newmap[mode]
        new_folds={sp.fold_id:sp for sp in nn.values()}
        sub=plan[plan["mode"]==mode]
        for idx,pr in sub.iterrows():
            if not bool(pr.eligible_isotonic):
                print("[R1-6] recorded in eligibility manifest, skipped non-eligible",mode,str(pr.test_window),flush=True)
                continue
            s=new_folds[str(pr.fold_id)]
            w=str(pr.test_window)
            original=o.get(sha_array(s.test_idx))
            cases=[("session_aligned",s)]
            if original is not None and not pr.all_roles_identical:
                # Run the original matched-control only if the exact roles changed.
                cases=[("original_equalcount",original),("session_aligned",s)]
            flds={}
            for partition,sp in cases:
                sig="|".join(roles(sp).values())
                run_id=hashlib.sha256(sig.encode()).hexdigest()[:16]
                stem=f"R1_6_{mode}_{w}_{partition}_{run_id}"
                target=out/f"{stem}.csv"
                if target.is_file():
                    current=pd.read_csv(target)
                    if not len(current) or not (current.role_signature==sig).all():
                        raise AssertionError("Cached role signature mismatch; STOP")
                    print("[R1-6] verified existing",target.name,flush=True)
                else:
                    print(f"[R1-6] fitting {mode} {w} {partition} 46 and 15",flush=True)
                    current,chosen=one_fold(df,mod,cfg,sp,cols,partition,w,out,sig)
                    current["rfe_names"]="|".join(chosen)
                    tmp=target.with_suffix(".partial")
                    current.to_csv(tmp,index=False)
                    tmp.replace(target)
                flds[partition]=current
            if original is not None and pr.all_roles_identical:
                # Exact role reuse: change the display label, not data, prediction or RF fitting result.
                flds["original_equalcount"]=flds["session_aligned"].copy()
                flds["original_equalcount"]["partition"]="original_equalcount"
                flds["original_equalcount"]["reused_exact_roles"]=True
            for partition,x in flds.items():
                x=x.copy()
                x["mode"]=mode;x["matched_test_window"]=bool(original is not None)
                x["roles_identical_across_partitions"]=bool(pr.all_roles_identical)
                x["original_published_seven"]=bool(pr.original_published_seven)
                x["eligible_isotonic"] = bool(pr.eligible_isotonic)
                x["eligible_discrimination"] = bool(pr.eligible_discrimination)
                rows.append(x)
            if len(rows):
                pd.concat(rows,ignore_index=True).to_csv(out/"R1_6_All_Fold_Results.csv",index=False)
    if not rows:raise AssertionError("No eligible folds after session split")
    res=pd.concat(rows,ignore_index=True)
    # Explicit three-set eligibility and same-test support on BOTH protocols.
    fixed=res[(res.threshold_name=="fixed_0.5") & res.eligible_isotonic]
    csets={}
    for partition in ("original_equalcount","session_aligned"):
        q=fixed[fixed.partition==partition]
        s=set(q[q["mode"]=="sliding"].test_window.astype(str))
        e=set(q[q["mode"]=="expanding"].test_window.astype(str))
        csets[partition]=s&e
    c_matched=csets["original_equalcount"] & csets["session_aligned"]
    # Record the full sets, not only their sizes; W8a/b cannot be treated as W8.
    all_w=set(csets["original_equalcount"])|set(csets["session_aligned"])
    comparability=pd.DataFrame([dict(
        test_window=w,in_original_common=w in csets["original_equalcount"],
        in_new_common=w in csets["session_aligned"],in_cross_partition_common=w in c_matched,
        eligible_for_published_seven_comparison=(w in c_matched and w in
            {"W11","W12","W13","W14","W15","W16","W17"}))
        for w in sorted(all_w,key=lambda x:(int(x[1:].rstrip("ab")),x))])
    comparability.to_csv(out/"R1_6_Original_New_and_Matched_Eligible_Test_Sets.csv",index=False)
    paired=fixed[fixed.test_window.isin(c_matched)].copy()
    paired.to_csv(out/"R1_6_Only_CrossPartition_Common_Eligible_PerFold.csv",index=False)
    # Aggregate descriptive means only over an identical window set.
    # Inferential independence of chronologically overlapping folds is NOT assumed.
    summary=paired.groupby(["partition","mode","feature_set"],as_index=False).agg(
        n_unique_test_windows=("test_window","nunique"),
        mean_macro_f1=("macro_f1","mean"),
        sum_fp=("fp","sum"),sum_fn=("fn","sum"),
        total_benign=("tn","sum"),
        sum_rf_fitting_seconds=("rf_fitting_seconds","sum"),
        sum_rfe_plus_rf_fitting_seconds=("feature_selection_plus_rf_fitting_seconds","sum"),
        mean_model_train_rows=("n_train","mean"))
    summary.to_csv(out/"R1_6_Descriptive_Comparison_Common_Test_Only.csv",index=False)
    old7=paired[paired.test_window.isin({"W11","W12","W13","W14","W15","W16","W17"})]
    old7.to_csv(out/"R1_6_Published_Seven_Intersection_Only.csv",index=False)
    jwrite(out/"R1_6_Eligibility_Sets.json",dict(
        original_common=sorted(csets["original_equalcount"]),
        session_aligned_common=sorted(csets["session_aligned"]),
        cross_partition_common=sorted(c_matched),
        published_seven_intersection=sorted(c_matched&{"W11","W12","W13","W14","W15","W16","W17"}),
        note="Original RF-46 vs RF-15 defaults were UNEQUAL sqrt candidates; these sensitivity controls instead use 6 vs 6 and may only be compared against other matched-candidate controls."))
    jwrite(out/"R1_6_Completion.json",dict(status="completed" if len(c_matched) else "no_cross_partition_common_eligible",n_fit_result_rows=int(len(res)),
        tested_alt_partitions=1,original_windows=18,new_windows=19,
        note="Comparisons require identical test row IDs. Model baseline uses matched 6/6 split candidates, not default sqrt. RF fitting and RFE fitting times are separate. Within-history availability reserves preserved when W8 halves both fall in Expanding training."))
    return res


def do_r111(df,mod,cfg,old,cols,out,k):
    if k!=8:
        raise AssertionError("Prespecified difficulty drop-k=8; do not optimize on final test")
    folds=mod.rolling_splits(df,cfg,"sliding")
    fold=next(z for z in folds if z.fold_id=="sliding_09")
    assert fold.test_window_id==13
    raw,labels,xx,names,prep=train_arrays(mod,df,fold,cols)
    rfe,rfe_sec=choose_features(mod,xx[0],labels[0],names,cfg)
    base_indices=np.asarray(rfe.selected,dtype=int)
    # RFE selection is training-only; importance ranking is also training-only.
    order=sorted(base_indices,key=lambda i:(-float(rfe.importances[i]),str(names[i])))
    removed=order[:k]
    retained=[i for i in base_indices if i not in set(removed)]
    assert len(retained)==7
    freeze=dict(k=k,fold="S9",train_test_roles=roles(fold),
         selected_features=[names[i] for i in base_indices],
         removed_features=[names[i] for i in removed],
         retained_features=[names[i] for i in retained],
         selection_source="Frozen-source RFE on historical model training rows and RF importances",
         criterion="Fixed k=8, predeclared; actual test difficulty reported without k search",
         baseline_max_features=3,ablation_max_features=3)
    jwrite(out/"R1_11_Frozen_Difficulty_Design_BEFORE_MODEL_TEST.json",freeze)
    rows=[];preds={}
    for arm,indices in (("original_15",base_indices),("remove_top8",retained)):
        z,model,cal,pr,piso=evaluate_prepared(mod,cfg,xx,labels,np.asarray(indices),15,
                    "original_18window",fold.fold_id,"W13",arm=arm,
                    extra=dict(k_dropped=(0 if arm=="original_15" else k)),max_candidates=3)
        p_raw=mod.positive_probability(model,xx[4][:,indices])
        pm_raw,_=calc_metrics(mod,labels[4],p_raw,0.5)
        pm_iso,_=calc_metrics(mod,labels[4],piso,0.5)
        for row in z:
            row["pr_auc_raw"]=pm_raw["pr_auc"]
            row["ece_raw"]=pm_raw["ece_15"]
            row["brier_raw"]=pm_raw["brier"]
            row["n_calibration_changed_fixed05"]=int(np.count_nonzero((p_raw>=0.5)!=(piso>=0.5)))
            row["pr_auc_isotonic"]=pm_iso["pr_auc"]
            row["observed_difficulty_not_target_tuned"]=True
        rows+=z;preds[arm]=np.asarray(pr,dtype=np.int8)
        del model,cal
        gc.collect()
    # pr from evaluate_prepared fixed0.5, not last method.
    pd.DataFrame(rows).to_csv(out/"R1_11_Difficulty_Threshold_and_Calibration.csv",index=False)
    jwrite(out/"R1_11_Completion.json",dict(status="completed",k_frozen_before_test=8,
        actual_removed_features=freeze["removed_features"],
        n_original_fixed_05_vs_ablated_flips=int(np.count_nonzero(preds["original_15"]!=preds["remove_top8"])),
        note="Single prespecified ablation; target PR-AUC~0.95 is not guaranteed and no test-set tuning used. Feature removal does not by itself prove attack-class overlap."))


def value_masks_on_raw(rows):
    """Fixed one indicator for recorded-zero duration / undefined rates, no model labels used."""
    v=rows["FLOW_DURATION_MILLISECONDS"]
    return v.eq(0).to_numpy(dtype=np.float32)

def impute_sentinel(dfpart):
    # Uses the same physical negative sentinel for rate columns whose finite values
    # on training support must all be >=0. No fitting on future test data.
    result=dfpart.copy()
    for col in ("SRC_TO_DST_SECOND_BYTES","DST_TO_SRC_SECOND_BYTES"):
        raw=pd.to_numeric(result[col],errors="coerce").to_numpy(dtype=np.float64)
        mask=~np.isfinite(raw) | (np.abs(raw)>1e30)
        raw[mask]=-1.0
        result[col]=raw
    return result

def do_r29(df,mod,cfg,old,cols,out):
    fold=next(x for x in mod.rolling_splits(df,cfg,"sliding") if x.fold_id=="sliding_09")
    assert fold.test_window_id==13
    raw,labels,xx,names,prep=train_arrays(mod,df,fold,cols)
    rfe,rfe_sec=choose_features(mod,xx[0],labels[0],names,cfg)
    chosen=np.asarray(rfe.selected,dtype=int)
    if "SRC_TO_DST_SECOND_BYTES" not in [names[i] for i in chosen]:
        raise AssertionError("Critical affected rate not selected in frozen S9 RFE; STOP")
    for c in ("SRC_TO_DST_SECOND_BYTES","DST_TO_SRC_SECOND_BYTES"):
        tr=raw[0][c].to_numpy(dtype=np.float64)
        valid=tr[np.isfinite(tr)&(np.abs(tr)<=1e30)]
        if np.any(valid<0):
            raise AssertionError("Sentinel -1 not guaranteed outside training valid rate range")
    orig=tuple(np.ascontiguousarray(x[:,chosen]) for x in xx)
    indicator=tuple(value_masks_on_raw(r) for r in raw)
    # A single shared bit, because masks were proved exactly coincident for both rates.
    with_indicator=tuple(np.column_stack((x,indicator[i])).astype(np.float32)
                         for i,x in enumerate(orig))
    alter_raw=[impute_sentinel(r) for r in raw]
    x_alt=mod.fit_transform(*alter_raw,cols)
    if list(x_alt[5])!=names:
        raise AssertionError("C preprocessing altered frozen feature names/order; STOP")
    alt=tuple(np.ascontiguousarray(x[:,chosen]) for x in x_alt[:5])
    jwrite(out/"R2_9_PreRegistered_ThreeArm_Config.json",dict(
        fold="S9",test_window="W13",role_signatures=roles(fold),
        selected_features=[names[i] for i in chosen],
        arms=["A_original_median","B_original_plus_one_indicator","C_sentinel_minus1"],
        nonfinite_handling="Frozen original sanitizer for A/B; replace undefined rate with -1 before preprocessing for C",
        matched_max_features=3,indicator_definition="FLOW_DURATION_MILLISECONDS==0 recorded strictly before fitting, true on both originally nonfinite rate masks",
        sentinel_value=-1.0,indicator_append_after_frozen_RFE=True,
        note="No reuse of test labels for imputation or feature choice"))
    rows=[];preds={}
    for arm,matrices in (("A_original_median",orig),
                         ("B_original_plus_one_indicator",with_indicator),
                         ("C_sentinel_minus1",alt)):
        print("[R2-9] fitting",arm,flush=True)
        z,model,cal,pred,p=evaluate_prepared(mod,cfg,matrices,labels,None,15,
                            "original_18window",fold.fold_id,"W13",arm=arm,
                            extra=dict(n_original_selected_features=len(chosen),
                                       n_model_features=matrices[0].shape[1]),max_candidates=3)
        rows+=z;preds[arm]=pred
        del model,cal,pred,p
        gc.collect()
        pd.DataFrame(rows).to_csv(out/"R2_9_Three_Arm_Results.csv",index=False)
    flip={}
    for arm in ("B_original_plus_one_indicator","C_sentinel_minus1"):
        flip[arm]=int(np.count_nonzero(preds[arm]!=preds["A_original_median"]))
    jwrite(out/"R2_9_Completion.json",dict(status="completed",test_flips_vs_original_fixed_05=flip,
        processed_fold="S9",selected_15_frozen_once=True,
        caveat="Feature additions change input dimension; matched RF candidate count is fixed at six. Inference not proof of cause."))
    return pd.DataFrame(rows)

def get_peak_rss_bytes():
    """Process peak working set (Windows) or ru_maxrss (Linux); scoped to this process."""
    if os.name=="nt":
        import ctypes
        from ctypes import wintypes
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD),
                ("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage",ctypes.c_size_t),
                ("QuotaPagedPoolUsage",ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage",ctypes.c_size_t),
                ("PagefileUsage",ctypes.c_size_t),("PeakPagefileUsage",ctypes.c_size_t)]
        proc=ctypes.WinDLL("kernel32",use_last_error=True).GetCurrentProcess
        proc.restype=wintypes.HANDLE
        psapi=ctypes.WinDLL("psapi",use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
        pm=PROCESS_MEMORY_COUNTERS();pm.cb=ctypes.sizeof(pm)
        if not psapi.GetProcessMemoryInfo(proc(),ctypes.byref(pm),pm.cb):
            raise OSError("GetProcessMemoryInfo failed; do not infer peak RSS")
        return int(pm.PeakWorkingSetSize)
    if sys.platform.startswith("linux"):
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    raise RuntimeError("Peak RSS measurement not supported on this OS")

def preprocessor_with_retained_state(mod,raw,cols):
    # Exact structure used by the original `fit_transform`, retaining the
    # training-fitted ColumnTransformer so its footprint can be measured.
    tr=raw[0]
    numeric=[c for c in cols if pd.api.types.is_numeric_dtype(tr[c])]
    categorical=[c for c in cols if c not in numeric]
    transformers=[]
    if numeric:
        numeric_pipe=Pipeline([
            ("sanitize",FunctionTransformer(mod.sanitize_numeric_block,validate=False,
                                              feature_names_out="one-to-one")),
            ("imputer",SimpleImputer(strategy="median",keep_empty_features=True)),
            ("scaler",StandardScaler()),
        ])
        transformers.append(("num",numeric_pipe,numeric))
    if categorical:
        cat_pipe=Pipeline([
            ("imputer",SimpleImputer(strategy="most_frequent",keep_empty_features=True)),
            ("encoder",OrdinalEncoder(handle_unknown="use_encoded_value",
               unknown_value=-1,encoded_missing_value=-1)),
            ("scaler",StandardScaler())
        ])
        transformers.append(("cat",cat_pipe,categorical))
    pre=ColumnTransformer(transformers,remainder="drop",sparse_threshold=0.0,
                          verbose_feature_names_out=False)
    t0=time.perf_counter()
    mats=[mod.finite_float32(pre.fit_transform(raw[0].loc[:,cols]))]
    for part in raw[1:]:
        mats.append(mod.finite_float32(pre.transform(part.loc[:,cols])))
    return pre,mats,list(pre.get_feature_names_out()),time.perf_counter()-t0

def do_r23_resource(df,mod,cfg,old,cols,out,arm):
    import joblib
    if arm not in (15,46):raise AssertionError("Only 15 or 46 arms permitted")
    fold=next(x for x in mod.rolling_splits(df,cfg,"sliding") if x.fold_id=="sliding_09")
    assert fold.test_window_id==13
    peak_after_load=get_peak_rss_bytes()
    raw=[df.iloc[getattr(fold,k)] for k in
         ("train_idx","calibration_idx","threshold_idx","threshold_pool_idx","test_idx")]
    labels=[r["__binary_label__"].to_numpy(dtype=np.int8) for r in raw]
    pre,mats,names,prepsec=preprocessor_with_retained_state(mod,raw,cols)
    assert len(names)==46
    peak_after_pre=get_peak_rss_bytes()
    rfe_seconds=0.0;rfe=None
    if arm==15:
        rfe,rfe_seconds=choose_features(mod,mats[0],labels[0],names,cfg)
        select=np.asarray(rfe.selected)
    else:select=np.arange(46)
    peak_after_rfe=get_peak_rss_bytes()
    x=[np.ascontiguousarray(m[:,select]) for m in mats]
    model=new_rf(mod,cfg,len(select),arm)
    t0=time.perf_counter()
    model.fit(x[0],labels[0]); fit_seconds=time.perf_counter()-t0
    peak_after_fit=get_peak_rss_bytes()
    cal=mod.fit_calibrator(model,x[1],labels[1],"isotonic")
    if cal is None:raise AssertionError("No valid two-class calibration support")
    ptest=mod.positive_probability(cal,x[4])
    met,_=calc_metrics(mod,labels[4],ptest,0.5)
    peak_after_cal=get_peak_rss_bytes()
    artifacts=out/"serialized_artifacts"
    artifacts.mkdir(parents=True,exist_ok=True)
    # Preserve fitted preprocessing, RF and calibrator file footprints separately.
    prepath=artifacts/f"S9_RF{arm}_preprocessor.joblib"
    modelpath=artifacts/f"S9_RF{arm}_model.joblib"
    calpath=artifacts/f"S9_RF{arm}_calibrator.joblib"
    joblib.dump(pre,prepath,compress=0)
    joblib.dump(model,modelpath,compress=0)
    joblib.dump(cal,calpath,compress=0)
    stored={p.name:int(p.stat().st_size) for p in (prepath,modelpath,calpath)}
    if rfe is not None:
        rankpath=artifacts/"S9_RF15_RFE_selected_and_rankings.json"
        jwrite(rankpath,dict(names=names,ranking=rfe.ranking.tolist(),
                selected_feature_names=list(rfe.selected_names),
                importances=rfe.importances.tolist()))
        stored[rankpath.name]=int(rankpath.stat().st_size)
    peak_after_artifacts=get_peak_rss_bytes()
    result={"status":"completed_resource_arm","arm":arm,"fold":"S9",
        "dataset_sha256":ORIGINAL_DATA,"source_sha256":ORIGINAL_SOURCE,
        "n_original_dataset_rows":len(df),"n_model_training_rows":len(fold.train_idx),
        "n_calibration_rows":len(fold.calibration_idx),"n_threshold_rows":len(fold.threshold_idx),
        "n_historical_threshold_pool_rows":len(fold.threshold_pool_idx),
        "n_test_rows":len(fold.test_idx),"n_features_model":len(select),
        "matched_split_candidate_count":MAX_FEATURES,
        "rf_random_state_common":int(cfg.RANDOM_STATE+10210),
        "peak_rss_bytes":int(peak_after_artifacts),
        "peak_rss_by_stage":dict(after_load=peak_after_load,after_preprocessing=peak_after_pre,
            after_rfe=peak_after_rfe,after_rf_fitting=peak_after_fit,
            after_calibration=peak_after_cal,after_serialized_artifacts=peak_after_artifacts),
        "original_dataframe_retained_bytes":int(df.memory_usage(deep=True).sum()),
        "five_role_dataframe_retained_bytes":sum(int(z.memory_usage(deep=True).sum()) for z in raw),
        "role_dataframe_retained_bytes":dict(zip(("train","calibration","raw_threshold",
           "historical_threshold_pool","test"),
           (int(z.memory_usage(deep=True).sum()) for z in raw))),
        "original_transformed_matrix_bytes":sum(int(z.nbytes) for z in mats),
        "role_original_transformed_bytes":dict(zip(("train","calibration","raw_threshold",
           "historical_threshold_pool","test"), (int(z.nbytes) for z in mats))),
        "selected_transformed_matrix_bytes":sum(int(z.nbytes) for z in x),
        "role_selected_transformed_bytes":dict(zip(("train","calibration","raw_threshold",
           "historical_threshold_pool","test"), (int(z.nbytes) for z in x))),
        "support_index_bytes":sum(int(getattr(fold,k).nbytes) for k in
             ("train_idx","calibration_idx","threshold_idx","threshold_pool_idx","test_idx")),
        "stored_artifact_bytes_total":sum(stored.values()),
        "artifact_bytes_by_name":stored,"preprocessing_seconds":prepsec,
        "rfe_seconds":rfe_seconds,"rf_fitting_seconds":fit_seconds,
        "feature_selection_plus_rf_fitting_seconds":rfe_seconds+fit_seconds,
        "model_metrics_fixed05":met,
        "limits":"Peak RSS is process-wide incl. CSV dataframe and OS allocation, and not a sum of retained component sizes. Each arm must run in a FRESH independent process on same machine/environment. Source RFE and RF hyperparameters retained except explicit 6-vs-6 matched max_features. Full end-to-end execution wall time is NOT inferred from partial timings."}
    jwrite(out/f"R2_3_RF{arm}_Resource_Manifest.json",result)
    print(json.dumps(result,indent=2),flush=True)


def selftest():
    # Test immutable role hashing and exact reuse without loading the ~577 MB source CSV.
    from types import SimpleNamespace
    a=np.array([0,1,2,3,4]);b=np.array([0,1,2,3,4]);c=np.array([0,1,2,4,3])
    assert sha_array(a)==sha_array(b) and sha_array(a)!=sha_array(c)
    arrs=[np.array([3,4],dtype=np.int64),np.array([5]),np.array([6]),np.array([7]),np.array([8])]
    xx=SimpleNamespace(**dict(zip(("train_idx","calibration_idx","threshold_idx",
               "threshold_pool_idx","test_idx"),arrs)))
    yy=SimpleNamespace(**dict(zip(("train_idx","calibration_idx","threshold_idx",
               "threshold_pool_idx","test_idx"),[np.array(z) for z in arrs])))
    assert same_roles(xx,yy)
    yy.test_idx=np.array([9]);assert not same_roles(xx,yy)
    rate=np.array([np.nan,np.inf,0.5]);mask=~np.isfinite(rate)|(np.abs(rate)>1e30)
    assert mask.tolist()==[True,True,False]
    print("SYNTHETIC CONTRACT SELF-TEST PASS (no scientific evaluation)")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",choices=ALLOWED_TASKS,required=True)
    p.add_argument("--dataset",type=Path)
    p.add_argument("--source",type=Path)
    p.add_argument("--out",type=Path,default=Path("IJICS_Remaining4_Actual_Results"))
    p.add_argument("--resource-arm",type=int,choices=(15,46))
    p.add_argument("--difficulty-drop-k",type=int,default=8)
    a=p.parse_args()
    if a.task=="self-test":selftest();return
    if a.task=="pack":
        root=a.out.resolve()
        files=[f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in
               (".csv",".json",".txt",".log")]
        if not files:raise FileNotFoundError("No results found to pack")
        zip_path=root.parent/"IJICS_359541_Remaining4_Results_FOR_REVIEW.zip"
        with ZipFile(zip_path,"w",ZIP_DEFLATED) as z:
            for f in files:z.write(f,arcname=str(f.relative_to(root)))
        with ZipFile(zip_path) as z:
            assert z.testzip() is None
        print(f"PACKED {len(files)} small result files (excluded large .joblib): {zip_path}")
        return
    if a.dataset is None or a.source is None:
        p.error("--dataset and --source mandatory for real tasks")
    out=a.out.resolve()/a.task
    mod,cfg,df,old,new,names,cols,cut=load_all(a.dataset,a.source,out)
    if a.task=="plan":
        plan,_,_=plan_partition(df,mod,cfg,old,new,names)
        plan.to_csv(out/"R1_6_Exact_Role_and_Eligibility_Plan.csv",index=False)
        print(plan[["mode","test_window","original_fold_id","affected","all_roles_identical",
                    "n_train","n_test","eligible_isotonic"]].to_string(index=False),flush=True)
        jwrite(out/"Plan_Completion.json",dict(status="completed",n_rows=len(plan),n_affected=int(plan.affected.sum())))
    elif a.task=="r1-6":
        do_r16(df,mod,cfg,old,new,names,cols,out)
    elif a.task=="r1-11":
        do_r111(df,mod,cfg,old,cols,out,a.difficulty_drop_k)
    elif a.task=="r2-9":
        do_r29(df,mod,cfg,old,cols,out)
    elif a.task=="r2-3":
        if a.resource_arm is None:
            p.error("--resource-arm 46 or --resource-arm 15 required; run separately")
        do_r23_resource(df,mod,cfg,old,cols,out,a.resource_arm)

if __name__=="__main__":
    main()
''',
}
PAYLOAD = {'rfe_reference': 'c-qZ<+m2+%k>vXV`w#m8#YBXAgkK&x5~U%QVspusM)TC@k$MKiawLM{4EFQa>{M3e?PTT=C}cUi)U0k*Cat3U_$A!j{KudE`tx7^^w%Gr{{62%{Q1N8e}4MY$G`mf`#*pD`O}BL{MUc}+ozwN{_W%6K7IJ_k3W6?@h?9={rK_6Pd~l<>BA2n|LxNcPe1+prw@Po_@6)i@cm!@<HLXc_0OLUe}jL0`t;}J&-_1rc>d#$fBoC{KY#z<pZ@mY<9~nr9&hr)r>DPs`j3z2e|Px0zkT}Q)1Q9EOAlZ6`d9n^(8r&D{`jZ=`{5@be|_PHKYjf1!^a<i3}5%}pFaNlUq3yA`|0Cf{x3fH@cn=O_}8C){`i+aeR_I%zyJ9^KK;KRc?S4qe*VP&|Kkr&hreq5sXu-9^H1OZ>F3XW^~0x+e>wTX|N18!7S7=lj`7DIKmGZ`3zDZk+89r={N7C@o^)8B{?un{?J1Vu+pf>2_dosDr>AeeefR#|uikz2G@o)er$qA*?UAbb>9aPUe)I8%zkRBK^(+6dz{P*8?OJccM@t{0waBzszNiM$FTeiqPyhP<i|?KWkv3_T?&dAqkfte>P8C?c`QPuqd(s8!Q0Zn<tv4MRUJ2>D-@o4<>$3qwMhp?p)-<L{Ap;=P<y2oxHC&b#kDRk>A*Anr_05OheD(Qn_fYXVvzTr<;0UISLQsG8)t6s<{r<xl;x;B6mb*C2Acyush`;;d+dq8q?K$ERIM*R`UYdPYa$LXp>fOJ5`1S9<{rvrx@4o)<yDz``>dWuG_|?0wf4%n)T~o!v91k5#q(vqF@QVA(ulJ6(tLdBxM*&lwRpG)}yaWnIoKS%6rg)c9aDRtD@&5f+iwXA(d?~&>(i;x8l9%}1=YM>~DT1&-*W1$^UZPTn`0Cx)zy0F7_aBx=4zc1%787VGI950`!?~K&iooCf^Sf`~pMYK*TD0k7wr)8M7HpNQ0ap9vH-!4(t1rHOHePsz5gpzOXC2d|nDHLLea^|cBgO@?T~|eXvr<X<ci;Z%!~1t1ehs5|jugi!8k`s$Fp>$MHZw~lK12-Q#D59A2aeqF`$S=(Ei1+7Z@+l|?dM;A_q#9OzyIRbAI?gRj#>yMQCFlEAwX2fR4#xo8S*pkHb)2^$3XRwDP9=(p$yrVOwlz^Fby?n(xo=?<;<T$O%5?rB%uSoC7Tp=E1&=2w;$gB^SAGQ|J#552APuX?FEjzAqVjYoo1!N<dJJR1HU+z96bexPdoQMXCdG6_3wYTSjdOpf5Y#fOPqK`REEe%CjFv}9##=KF$20xJ8_H+W%uyTWEI6?rd2YSSMWp)@UzPv{Ct`_Q4yc%2t_hOs$I*yf+uPkO_TA8zV9ps@Z)_DmRxd{g}>k?{@D-s7t<MCnzHisNDZ@6w!_PXe?c%^QpcCVcg~SWzBJ)x9Qkfl!0%psq8L(JRh)BkILJyq@ylQR4$kmjzxiT$7~MMZ*a(k_T+6Ci^8)tEZ~icz5$}^iQ{+|1XNNBLPNfk4`s?pL{GZ={@$J8!AQoCf*Ez$RghVyO@85s*0g2VWeDUo*nF{!~uDp>RDA!8mQonis&4*vT``zz8M~#Do5TF10ix=01910JfxwVmzBdll=_~qyCKcJR>f!)Ogx!-`BLtIB-uMGS1KkjV=QLv$YM}T28Y+4vJ^45K(?T(a5tUCe<ymMs(`R@I%fBnTDUT#e?+~yGV<^e;gS|J=PyYFz=&cwQTcSDl>>9am7!-i$Qlw;Wt86`FGX1$odK1EH_esLp#)D9&u>PJ9FxG4<##JKS~ybw{+hoYF=7;Xf9=pTZ6nB|mvl!DhADMFmh9DW+^uM*Z*O2Pg{L?<#kWTCUqoIWZ+eZA=0ySJD~>IOW9>hyY8kM5ZU$|fZC)>~Th*()JE=M)GIb0qHw6pxo|yWw~o)B~rWWkY;JE*}}lhTui&AU)?4z`@2~TNA~dS*vuY7o38<$<7|p```h)rB*W8E2IJxc9WhK$a9Wf>5y-*3Z89w(v2Z^KsnMfH40%rCKWt;UKKTzdb4+dfoLz~c~3|MFEt*!^@&6nPvL0dmGB<33Q*CZa)uF`%$eRQnZXIGpgXd!hWf&hEU3&vI4@WQ<E8SP_=P77HS@GmA^0k(K&Qej$h1VMFO`g8g;aq0L17FwyoXs-v)v=Ody5EDm;w%WqBe$aC}k42Nd+XJX%6Y^0%|48xI-$?Xx0)zgE^)a!hMBPun!=bj$>vE%0f0*$!~=Gh*Z#dKWJq6^ysrz;Jd~tct&nWFPdrud|!CjtPTB`R4_^5keT?zHnSlf?p9Vw1-uJV#K>T~&$CpNyvHeksnBFlu^`*4R1Dss6lhEDyzWGqIDAA!<ku(#!G7>iLo|hBEahHqF$$iG0wgSmN2nSdsdlkzxto*%MG62zAvLehz&md5!9Ln<E*S-T%Rx{|8B2e2AQOv9@XrVZ`wM3jy_4*8fx%e#tcv)+D8Ls@MadhEL~-;{$g><%3Z5Ggd=?RO;KB`$q}IC4UE>rqW`yv1+Q`{xt&|C$aSHZ5DsL<$;qOwk_mR~QA8`tvuf>q~Ida#Yi;w7)ZuHhE1+n9VTL!NnQ7^QFYxbfU;GyptJBjHs%#k`FUe0O(Wra}iQkjLI#u?W6Jo<>js1J++GLh!g5#h)|dKLzK(sUK&O^HqGC=*7pI(U^)5Ie<DBc$M%5y0cqXFV!|KJW>Aj%JReI)-*B=BqFH1pE3CVQ`8HBF#WdOe>6fZ`semTyfNM^l-SqiOP3l`l0x~{~^F<xDVgPC*b`VM=jv*!X&r$<{pFfluyu`!xEa~L?-}`Qc%wZhcwM`q!}b1ZBC9J2KA`%Jx7>k@0N-JaAWu7F+k7R1k=sBPn3U=iR!5s)FXug=9n#ovAN;ZS_tVmn_x1r6gaw;{K`=X>NPe2va}}F6^?c`Ddk<)*#tOTh6L24LTL@PLWg`$qo4(%0-m<))8Hwrl-FI+DClVs0w;l^w#<@fkJtoi&25OG@Brji%9u~t1SEBO=g72*Y9*I;!X{{eVgd(>^MnD+N;ogr1TQwgc>zr(H%J@$uu>lF3ZI}`3!ueu(<xL68>=D(y`fA-LLVrY$?&aZy_b9fS!7Fax-+y%DK~bTPSAykuach3ahOV>?hc)xCypDgZITV#DmjiTe1ex@fwsRk)>t&WD%r|9pFm%<S%S5_chBxkr7rMuLcw6c+>qttEj*glMt<NE^kwGUlw3<mol?}Vtn&$|!oiF1oG$oNtxWU|p#U#M>JKN=q$4ja<XrAB3fSOTu9i8fE#?)GUt<&`ih;qaEmHckl(AoD6zs8+`PeLSAf06<>m|{zixeOZ;lN?;IQ*P{hw|y8?Pg7*z<1*0xw*HA!q<B>_^T2HiVWB1lH;~)wIb3wp@0rW(_kaW60KFbcsvja_N^)dP7dDTeKJa&?HQZkg#a;P5ja7-jF48!^PRB?UMJ^ADv}c$cpslzM_wNX`I1iHHW3nY%t?i<Ug=J6olhXUfhTP!V!GJ$@M@mzhC)F{x@|6V%+Z@ps|@?lcO@$gDG0^ycugXOQ6KuQyr3Qi%m^yUIk>{8Px`Ks@Ou%6C4DR+azzV-gKf*sL{pJ2ai*8ej8C6+uMGQDePfWpp>r9Qrm9R~&_|SlAx;&i#h>_WQ5f|JS%%bBo4hK{4iDHj*1iKz#y<qOx0pk=QVM*K+-kxaMz-pEJqGI;r2sKSg>p{LX;~YUpq}0HsdddBc;F&KxANm32K8uYW#+_B-6Xo;c(+Eq#~?l96g2o>MZRe&@M2r%A|D6!m{Txkr^Y4Xnlq;;g!E$T0e}u|LdI?MN=Pp#1$z#q%#6mdyw(#2F)Nwu6;1)2*P!kxiX|#T7dqt|oPxc)Tgp^653RY5Eal5qI0a+_^A@xxmrTqk3HROWMY#+L6gWvIR2iG4j`x&QFqT~|v-5hTkO3bv3VOyAK4Zue63KcccXq)jcu8|nN^}(Y4ZlK5+3OWP0ZTXc)@35V&0fhC*60Kh@LH7orWUVhG1I-+d7yoAa$9Yp{=uiE^4vRo0_v&rln3<*vQn74!zb{G$B-)7&3UI;2>8vN2W@K1q_!Nh_ZX#Y<>t}@Ua_?ZWbr}O$69H{?X3q!5%?u&cf~)I5B*#?4TaP55CL~s%}NE-bwa_O#j#}l_&}{nDfm)p-Az70Z}cf#9TtuV-z%N^9X^3`;-F7$K`A1MipZ}nJy01oBL=U=-Ag^BcO?qCAs~?n!P#i`P)ed-CluryoH>V^NxVz^9gKoUTh1DzKzGVoOhVMCD5zEq|Efp<z8tR6hohKa!Cn>c5v9Po!ncg%&4@{*O6@|U0I{d3kdTMSIcKTUU0-{kfNZGM#VF@n>jHY6RG^YLS`Lc|YhtN}_+ssW8#P_Mcb~@Hwv_;1Ar<tUnoZ-bV_32cA*;ZO)|ZFQ>)cUmG)$Hry<}8#$=6j1B=GPFYRl-sc;ba&v-P@fv{2#%`8L-f3$oXmb;2d5z_r<Mk^wyznp({w;n=bp35=*AlP$O|bGuH#+rWP1%LvjYw~4A?*>73-tPh+5p@2mII#)Hc&GV>lg8EdZz?It}Q3cQ0T($;uSlB-Xcq}5X_c#SFPLKK~;w?>=TjEg(>=~&*9d$}{>jUYXW>kWDcG)+W)scyty^mrhaH>-<9MvBz6%H*0ADiOGV}PD<3WUo<NWL{VVLVyIkRDMAkVsIc0w+4gw2d>pk3o7tDM;9-jzuLrW)wntMWtZEDNCbJkdm&`D?M3P7zL>Jfp}7Rn`xyLI^?@51ytX5L1H|F>%Fzoaj!56xW0p|NlnCxHcFy>_j*y@km7_0KqGaBX-g%%r-Xuze4RV&g6my-B@;g36vV<Nu^<~i`bIL>gOZsmoB||XFoB6Lr(DHW4ebXxv~^N}=WLwOAbmDlpOvg(ZRr8`WlbMGs5(8pnDJh)3SbVD4R(}2Epzi2^n*;|_R@nnJWxkua`3;BVcZibKt*r8*)l(ug>c{I6ri3r64=M^i6o<x^SHV5fLrW1EHK9CjMjC5pHm73C1u#|lSW2FuZ{e8=K-5%PM(qoPQ9p8S(hk4el+01NhqgIS*<kmjzmEs5tT+ZCeo;0sl>Xs^dMdOM77<K{cDs&evMLq8o*F-rJ-mM^-|V;i&8LlOsv@PX)#=eZ10~u82lQefRnJAV-5~>lhhk!;jc0Zo(UbOa~%1=((;bbtb~7+QJ{l)mvI5o=m4x$5g#Z8<{dercFUG+R7w;t83la`nZAbd!Np6N^SVrdN^*4<(%IzDP_1i_Q%1pyx+EDJoPFi$u@+Vh@r6nOwLp^oS%p)im9F%z@d-$*pGH!FQ2}TB(u0TRfaf9w9Jc5)C{T+UUMr*i{Esgt4dVzl00S;V+$*2;@zR4q)i{aZU}UWp+d7><_7PAQfu4yYpL0wrS|FUP^T<FY#^+IyE6H{?!P}rd+IrA?2A)HPaIwh3sA<_RHx474!)FiPlZX@seWFl+_h=lWo=S;6-B`%jll^0G=OjUKt3ZL0Mdi+*h*Gx{)-x`Fj0~~SjI%_!wNvyMsAmENW0|1$j*z0VmAvyXq({P|lhU0{_;_XZts8S5gY-h5AdMRY5aLMN-`c}?9M%JyfcYg!`h8ipiVUt0(u<`BB$#mwJsfdC3n9I>^I#%#kU-9Z=W8H8u4J?8bOIzd$j>I%cvEjy3LWx2eS+R$6vhithv)5Ll}`JTO`w@G6?HU<Q#NW#qCLnHbl$n)t0#i5&8^evkAZtkCLjl9!mVX*@Drnu{hrDbP;IBj=+lwvZrqCVpj_q(l^{CXBPku{Yg^|oALPrftUEwKkMd_adw{Z0pWtyOd%+~=6Qu*Wr^p{rvlX-53nsw~A^()4p{R|$R&={bBoJ=($%Sr&^t_T?+$IuaQ*D$E3Zxa<D<x|;nFNy5zY68La3#62lBwKe5(s(0!gpb`mQfe@Ig@~b5R}3uZ*gu`YkDi>M};0DjG)$|bn{a37j-MER024aY>xWhxEgv&sY{A`>IBNZwx<lmlUgeEze6QJVW<OTRyUhXy&~|d;sn!#ThBcjyiAr#t8Xz0bjP_4Ug+C&y2i1+7yn@N>*53`5Zrn3F~m8}T^9Z-l^}OPoucFq0HJl1gMY>(*x!d(hn;g6OgbuGtqS-+BnXr}Tt+gVcS{uVEXPEG*r9G%#)?BnYi8A~;mW=P6NgRkN+E~W3iPW)f+mf-6ua06AEQ`wUtM=VnI{&Nc;Fn~qSAHVI+I}ElXnTvW%VChm|ipoyrdHJoY8{9h@*F@4ErEW&?A<WKL)d&DOnixkurgDwW;W&BrV_<vg~y#!3)=T3EZ9Y7kG6aMGJ)EZ3o;Xytt1@SoPmQ?Rgv2uOykKq^c?E0yk%sP#E=rN+2+8l3OQi8)ZS|qdpNP7^oY$vkjRD-eN0>-k<yz<4&Z=Po#0RsrQ8mY&Uqi-hv6V{5^}#9t8G$+kp!`oH9w8V|Xd7XBT~aB3GVx<jkD%6tf!GqkAU&{JA2LY7e%Z)b${!XN&@rODLL9I+^NN#G$u@y9Z)D(tKF<98BJ=Pg|Cr6oPtADOlQ?YKUt(b5=rng;Ib><{7-DYrONyQbv1)Qov5d<lKW<Z)%lJ`R>jGOT3<=>0H0dR#Bz%UXduEJ;V2Ml+Af5P!sQwNWr4~j1m>+EN!!*;)4L6s1!8kvJsTiF=)!A{Ms?MK*;ILj!KU5exsFgYZv5#7c;O%1qez~4{o5U<=3v$3zYO(gk`QOE@cpF`~tV$0UDX4_l-NDA85aq`wx(w^@(_gN3OT6EZRre#7%;MQsxmUyCYj;K2+(fZ&M7o&_mtF9L;92MDw;<f%l<8wQu!B`>~X{+$0$&M`9Ey5)wQ0t@{n%1o?$t0XMMqkuda$cPozkNJXt7k|We0`EV}9EaeUF3Kob=WFc(TcS~)x?pp3C77QhciQ^G-)bfn?6bs-LJCZ7LqsJ@?{i<Za<Qi$ptqnyMP%F>Ay9NPIYBrod(pe-0UKITr)4+E)^?=QDK$f)n?LylRH=Q-6!KB6ryd>fhuSou$-tq(QbS^~;G<xrNW;rsJYNIUVSF(W$*Y0v%R>T;!>dOn+g3(9=AIP?n{7A25((8PKsi=oJ!=SO;P}Yi?YlH(TLo^#U2XA!Jyd30<WeDW=baKN{5YJ-GiYt@@-_7o7uAOsFNT|xWxK+wQ?pnBT7uIMpThxdBI^^9EVz>dze0rn$Q)So(&VjRR5T(u;FmIeEc^lBDqyysyT({1#2N_beTsYXa?nH(+V>QpT+qe#4md5=`I~bj7(h}KH^XRiQ>Vq~%bJEp^bgmd^Ix3?+l`x=EZ0DfENUjdr20WhPmjb6`G2oL%thP<VfTDK_7s8>bX}UE$J_zh9?_gi<iXmA-(j-&K+1=#r9t8DF#DKD7p35GwP8xlwV?Dj>o52OoN+np*Wokh^XC26}9K4fEq!(?YM&kpq9&a>dzY`v}g|S&+m7rd%Md*EyxTq1pDy5F~8t;HRG2p&ZkZn?}vy$0f*^B@u%gkr8=tC&3bjtTOBXr&`P7h84Q`fatpt$562y%G|Dj&`1Iks+ccr@O3M}vC!a_0;XkJ!AHL7xg5_~d}nOj0vPl#0VAyaN%Y2@DpEyY5Q4;A^}C-c{hoKPFNL_pK@%kMe6*79<3Ydf6yAXuw*^AXaz>og<EeX?`f9NHqgK!Yli9GF$}NPZ%O~F`Kx_JK*9#>Yp@0f+-u7Lm%aZZ|_LZ9`tZA6oyi(+PK0yc)bdDkEtk~IY<**TWfEH{fGfQMmN)mP#EH^MPcBF%4A<g;iaQZ<kDlZn_L|FvAn@Rlw?|pyp*L?Rqt}vyWcCJauNCvJgz>YT2^|8cYqWXwvr9Wo=2@(_73r&_Y~5nTg=H-^d+HR5D#A5W1sjt;kwlb?%611G`E-sozfD9gk=iz@2S4M!}4hOYt)1Nz5(v}=8E3gIa@Yr;=i#lf!FA2oO&7fIm}w6<tq6=cp23NsUxgPP3l5EC>>DK%`%XKAr(MuDs|iY%076h+J*#jbJDK|1;uOK_nwjuULbSh^l&p0Rkc{Ibw)mTSxte0zD*`7Ik?|R%VEA;oR9_|H-++OGkdAh_zL^LcC%%Rp2?wbl$_;)%qsgJc3qK1rHw?0Pc~?a!~Xn_`y{q;jcd>0j*@f4)}gMqfqm%EdiR`M$w%%l=Z^}bKA8-sybDrTic9oC5$CF9!@;`yj(R`AUhQ1=kZu<}cq7<{1qvpVrkRT8)8P=aGVFuwL7M@@O&8*3v3mX%{Q&7{3_OqAi4<^#+kod&{$k)<G4RC6k?eV2_JBOzKo!G|*x=RRc7OdKu;=uHfn#-}KA!8)ZPRrRf_g?j;0j=l=TH0_ne^6t|8ZcCuAfXez)FVkqC!*41kU&e;?c&WsQc)Y{f|mEaK=9fW}f_wF#|rK64Z14fpIR;;c}6<MQogXdJxuYs}xdT3(b;hD7H0adl2R;x(8-3-OdS6CE;UKI_4V$g#CP+AY-$RRGHiwt%UuOe$aA~+dAFJRzo#X$~;zN4+dqDD6$)P{?ck?4JWDx9ksL}ZRa8*B*di*;Z*eiwko`K<1L}KQcv3j`(WQ^MxdZn)-Y9yVAcx8*VqS~1@lQcI2gv1QU<ZYK3LW}1}Y)$CD)B2KMyqE3)zF7+}VfRY)~z?RxWv)eSlIbc=_fj$$F|((73%=A-LDJKQH5<QeYwEH&qY%;Kn8nIC9mT)CzpJL=R9c1>JNGHl?yvMt-qcVFZ$WMjFY~-it#&*1QXfJPxXejp3eEoy%DRe{V8E+%ee?heMBA;qo5+z~CM!8Du>8T6xqR`T@#X!MikXTqaV>Io{Jf=*`)^M^!#&awAG9o4Kod(5dL=Fq#Z%t8QfrJsSQR`#^VLTx&6$tBO$NjB+=Q_mmHi#Zbf&-$x8}({kUyD*J%CFr){Y;)UW?7V&|6kTOyxDK=a_>7|_Gh4cYTVi>Y>?_Azj%OI{ST0pXe++0)*>FtzSVSiQmpp$dpNUo@7*|cjp$g9c+(L7=#ymiz<e1&|_cdFfn4l1T4r@Xc|Iv<}Ep08Wb;CLbhft=;fD2@BIK`VgU%te%oL|nBp>;wIPs`O;eQl`_rR|dUr(C!s{8#1Rr@xUe5UhJy%3i&|rEcQv!D28*7PoH%!jr&#npu@V=mQAKGl+6M#Zv*>i>4K_6PLw*Vb=s_q`c(aZyD6u4<)s@AeiQKCp78Mtk<adYpQS6%Hp(9?3y--lmZPm3(~yUOJ?9^=>Ac*-=FmPi-E%(->$(0xlIla2M7Cw}vedzzUi?uyM_ozy$z?>X6xPF4-}CYfWEC>GKNXfhxwCp`&o~G+P^xlgT}m%yvgafOvb|i{wbeGF*5h@Bgn*!i)E>W)^tCY}KMeB~5`y#&bC`&4hz>bQ9rHaB0<{~tweESdtISg1k68%&#GY*d7>M-|oTjK{A8R5Aa}hsgjlGPL7ejo)LttxPVGBYT``9Kf@HnJLL<BD36vPXKhsbSLS&vI`)>ROMBR|<mSjt6hqL^p9vVMW9klbdp?z{!4Wf5yS2uUi9P8>5b;GyMA_>zh+U=htHg}Q@TW--IK$wm;P7SsfKOUq)x;}#o1xVq3r!dwOtI;mwSx0f(X3vLrdQ93euu?p`NAECjMh^6yZ4gsdm%Fr*iFev;V{5YkfSpihXeq6Hz-y&yn&+lxtYL<J87=XT0P9S+Z3hOCm@AoJP2>RqqawK)?$Wljtm6IU*#eJ?07s<6zbM#v0ewUOmf}Ku_hPcX3S5f%aSP5M5>ZpM6j_JLt7RA3tOBlP(mRHhK4kVAVc^|_Yn9o&F1W66$7!NCZtyZ|5(Gp(zFwawRgLLw(T<14|e8fwbT!#hZ$Rvl?x&c0-CFm~P7N$+-Rod*Ei7JnUdqGSP{sS*AO*m>&?#g6U7NLg=*9Lnd7qM-Wj(jNSBW}W;_nGR+b;lY&UMyT*;U?$~!9|kI+s5&!G3uH1Eph^v&`@)K3JReawSm7jaGM5~3v<0N-mOKJ#(mh$U{3XKJd^H?t^TTY#6<^J*aI0Os0K}q*RmzV!N6_D%miR%(VA4mn`(E4SLG3Sr8SLHZNM%=dTHE8`Un{_%z84B(pK9{PUJsE3~r;2XgK%H!KXjr;};@_3sWItd;_|TqR>znP{kOffwON`4DBmNLHGLnqann!97cgDsRj1+Zg8(1=|cTTp$H5I(93~6nv<Exr`Zrw^C6mA3+wrs2Dk;3x*qA;IpMP@_B|5p(cV=fRYinD4a<J8N>I-!3X?zj(H9Qh*4gKWfxX};yv(M-X|Vx0kUMuPX1P~3HOvOT(4up*t@6Dp9rO*B!n0#b3j6h7{%Ja@dEra71k|i5X>i8^BXQB9fDdvBuxC~2N~Su}h^-~}qY$48CQK1aB=uwus@HNFCqxCO;ugHcLdqOc3+V+_;U#^a>?^X^3wADwWf505HZ*ZvzIdU#)v}2ds)9_?UQzwKQ(oQ68S%x=hRJZZaK+T{`_}&L!wlm#RRM7i-eNY&(A7HZ+qwzvM*RUxim!C51^u>gg3h^Iy+|7f+|?Y(EvkZc#BA+jm`PX<uMGWysxWwG-nl=zA>|v@u^);;l-Ve_c^NXD+Y!C2hgsz+NOA=aJR`fdIf|jbM^ylLL2}*aq`YaA@-w&P6IuqzmEswKBB3bsD@=vvY|bHR8TiF6T`jA*#Z=H8TVB~aFP%~(w|9>o4u6fP(069Y!w{cRQ%0y2{swNWZ;;-#iz1aw>Z2B`j@K0wMut*x(#$PX>q0)>-+;JEVYU$-!)lcn$7BWBkzkn(m1iv0UC*qqs3@3)b9~M5uWr@Su644)=-e#8&@)H5BzJX%L%`Rukqa9}sb4JAUzJhtG1XukESlB-j%^5UksqE@6&kQJuj-CNmr)z{VU2?YiCGKJT#;a9Y1{{<!ZOGt&Uw<Nwfyw0Ee=CEhoaNT?w{o@jn{YzNCA!JEr^I51+mRK3U7w{z*9h!L68wKm<m3((zuU!3WMu{jHgX;$C0IRpIUfN6q=mR>eGB{u*82+ekpQS<?;#Z?m0GUDNs7vO;Hecl$INJU_A`&nV5nI+y){HSuEK%dc8+sJrh&l@)S#D8m*+L);id;Z4Q%zvBsr3M!A(*SkHM1l8AY1Pb<@Bl~Pvldf6xY%f6&+^q58h$n;t#d(KoCJ_h$I8O(O}S_gZDs6f#%*zXXRzLg#OD9Bf63NQ-1Iz3Qi=eFhA4}*Mvp#y?qB;}&H0cK-M?@{PiND5)h+H!J38#P#JS;z@VL0)R@R4Q7FBZnBhn9DdJDNxRBH16azwlhlE!pT+#K`|{&&5@MhrL~^9s~m*{Sb>+|DW?^S%Gcx+I4e&zh1O@vs1-w27dr&EXY^$wVni`_e8y4OH*T9XvAOI861cZn%rI_q6!0NtHn?@S*E;RnBn9pn;6@7M${Nv2A-~O0m^m#*??XaHk7A|BI!EEi2XY$-ayz&bkIK+5_B+^Mwj_gVr8f6}-VXcGxpPRvDY6HD-lbYWcyGf)_lc-U9_?rnc&B1X-yM>Iu%{(9rhA9GEahnKkQ7Xp#pI}QQ1*HoHL+h~DX8Ivh^%Zedu%4RJRJWDO~H0EpK~uN&)_@Q-Z^?W{56__>`=yMDs|eqp<+nkZ{UWY!dM!+oujW3u2%29=R#3IdxxJ#r6Iy=z^n`TKvSS*g;VyE6F{R@7J0GhfzyiQfN;qcQb;Xk)>mi>V|b%B$OPaqvKZ)fngX?xIKs_jSF*X*1AR_Z=*SZ}Nz5Cn7F;4;>N;?ps37<PndxqVswk0bZI{1!gCz0L%FP`jyKn@Wo%60yFUnk(R2ZBKLt2}p@@-a8d?Va!<?iR&Y?{tAWG7q~ouy%)baIi1XBu~{vClAP*)rl_;JzEGJ`l>64SnLDKI>5%_hF+64!b)KTD#4rwa@#o(nOl&7@R5W<UVyTjr)wNFlP4WtMRs#il|Tc_=Uh>nxj1-@#Jc>jZ}r9o}`wkjnLTYw;zS}m8$UE`0eVGJ3tWAja5M{uxDHah83hB-HcZcW9vB01JNGMEsehHPHWs&i>Ixa8Jv?9V5iUAG44``ZQ0!;u^wqEAf;=IS};D~PNWppi@gs{+V^A&dXimJ3G0=;4^-FVQa4WVZr-T)Fwj@X3XLSwIq^x3tz$=zf_#Il@Va5SO)eapuommZPW+Osz)=xq5iPQF{Z&!C2StU>_1nU+0B-$5Jyi?wDOn-8;2d5fI`Xwv%N3un6&ipdvuoTojNHsiA-%#@pt+*taJ&I(k7mWf@-?<XUsm!iLIr}{vX)J(Eq_pi=$T5HztOCm4PWej7}UKrj+swSjbbkCHdkR#VoJ!2)<D>vrB3_K?gu8Djc+r>Hmb)w3i%zff{1B|&W_Y>RI5pEkrnV{CQrSOguzHuMt-5EFr$+O?h_eE)AHDlRzD=FR#g!=MP7Q<dP?t76%cw<JNIX~>nL{ax0wp4YWuT{HGAdUiCgL9?=Tf&xr`>SQc1$Ln%J)~6-dxWwT{Fa53XpjsCtX2Fm|ZghjNaXL<OCH$BgU4@vk!#=;@j0PK2jvQ5gSqqQdAh%w2L(64PQ)c9o|vDB#z*X5NEY!mNw=Kvd}L_a}~@n@);lkjF%YxeGW$8NH{3A`h^7W_^XIupDntS3w(}$YLRMov4spC^tqV-p6Mf^*~<`6}mDfx8MMfMb@(AtIHo|AT=4>|G>8B9QDk4m8jsn1se1lv%`UmULW_XqC$s3FDp$HPA^-vZqf=-;boMtanB>taP6DBxNn5|boWCx&t?58uVYi{RuKmSN5D}eATA7%yzPd8du`kYqC#(>$@^J+3)yFF+y_wwO7&_sqg?OA(bu+<Q=)>Iq``lWM&joS@W1`={VD91zi9pG7Yjk~K0BUt7=OZdC<+U;Y)Uj|ITpO2J@(jdjq^-Lp$#96%vo;cNqgz)y*0|St3dK<+>J40aOZ<mInC2+zk#a6Y^rr$Cmb#;-x}ow6Ct@$R}d+Xz|B|LgbFZ^NC<5NVgg@0lDSV2lov_~&6Yve9C+ZTj3O{Em<N4-y9=iThmQn76WL7>wCj`ulyEW_F48rr=}-ggJ+eVp?qv_;G)X>Siz<F#MHE3P_>*u5Fx^=Z>VaVZf933^%x+xiQ3mQUwV-F<OY!BA-k75lG1*fI1h{;<*3Ok3+N-$#3p&AG1tSH=38%%J&!kRNy+$Nx)5qjqBCb4et6~AG6asRL;GuBFB&QZK)(iFkXGnR2GhA7fTbYFVHgN#uiZVH%rWT9aB1e2j-vCF=EvX#U(3Vx=?rpXJcZ-a{1vv<2tyeLRTYDGCN)$YLfvO>?;8_#&1w(+FsZit$HEGhNF6x8)TFW73ibQh2hh&q2?qhY!!ZYy?IfYY_#;i>EyQf_+dI}DmcI|!6B0l6c7oek(6AIqJ_fa3ks-eEd0zi2@obnul^FL`-jO14Tf9^ynHkpnv!@UHRuF39gQi!44Pb3IAE;&m=zuEI|$|}_(HO$J`%UQes*}CT(i9|{h?!>#61bgW78`4vyxK(pF!76_5w8QTl@zMxqb1tls^`5o%_sN}6!Us9&(B(d;4BE4v{)>__Er9#L!kZk?KP#X;YUb}}k%Z$Kyb&HKvC8B{SK9c`oC(SxQo^x~j2vMFlezPi2vnt2r<OV8iicN){Mx%mbZe-y5j+^>nid6oXxejC4U+us2o_XEEDHFfU*9vFlEv!HW6V)8yFXZR-!=B>kr+4cZb*_peb#4H$gfTM#*sAZvaG$L&V5nDH02j}5lGQcuA<%p?5<T0`NWv<I=rw`J6uuxZ3rdu^ow;5GW(NG*L?mo6Q%xh!Jqr<l=Id+Up6&zsZ6(}K99lq+M0jy4a;=9a7Q&Y^vd8oUvFThoLs<*WJp*goX27TK^+D<u!Tra1>{+8o~p9h0>|0N9MSt1#*c$}=*?@JBs=g65woHf0eRk<HzZtPVPhmR@(s5+kAZomH9t{|qa>e;5ZHK~p?Pbx>#cce{%7DfL@G0xMZn%$T0ohIIcDcAUmoVE-x~IDWBzrAnoQ(aj-sqf+tT%XFjU@nEQ{)t+T)2b3~6_0rGR=m^zX9lQP4h|@|Hz@^rxM9>g0FB!_Vo_GUmBH=#Q+FBNvgK-T8<zA@xdQUXV=>fF-HOV+&L3aaMZXnHRNi@7>g>rqBx+>qTdt;#MqRoAlY!M;SA?)tE;W2>+iE&O|1vq**un@;vEbD9EDYO+2e)2siuk15USzz@n$1maS7A+-%HSw`get<*$qPtO@#1JM0;9Igl|V+Ya3EIO?K4kfu9>a_c=<mUYCFg6?CzF+bRkXWx><lc}m@QncHhdDt{%;dyy(Zlg*Vyw{oMCV@6}DqITUO*Pb4HwiR2>ppD?Y=t|>s#wXb)_m-ogIL^`&57FR$0J!~fUmXY_stfLi#UmxaNjY?Az$sx=PuzzT6RlqEc_~!QtNa0&JlbuQo-N_tt8sl&U{B$Y(W{Pp~+#LA}-|6n&031a$zhJ>cRc*bxh!5=stG{eu+esYdT2dsbsKco%#K^rlD4yPFgn7j+9DZFB<a`rC%&}EBu(F%+ap4=ErNRjjGH7N5%n9DVWV2`kAp4q(T9oFb2YfSS16#s{Y^Eze*%=RQ^Y}sEYWY`%msegAI?)De1X|ukj{`Pr8|t;}OVzP*2V2O$uiBCmfTi_+AMmqdUwtXLsl2yA=g|7?LMhhn|-!czagGdpaENllz6{!hq2&_ZI%-zZA2CkEcc3s6T(sgutG+Gre3wTuAs&>NX{wtf?qKrq@Aw)`kzu1~Ukio|#$*=-Evha$D4EC>^9zUwq3f=L3-*->z|p6X8obcg$0EtPaxiQF)|^Y|1ECN%7X?Zq4vWphv^<oLUfyn<H18y=fVohraQirPsM9JR9dZC8E&dbkUaI?^vclVP`8QxxcNAI*6}K&AS6-3uIh6t!?OYJqqt#m3~8#KrO|_+2q_Z5BDhOYYP2B4w;&f@7=c0vmOrirAN=X$bovyxro@t#zOs3a8C#5yG}*goz7(wI?EXEaibp2i<9V-GDr2Ob-e8rN&e3DPGKF#3g~T1?)@+yw%)8yI9;Ox<uJn^q>dS^bnB6G8KZ>3;~wcw*Ra_murK2&K0M$OT10P~=BXZL47XbLD8h_<c_+z0x|KQQYfbz8HC}p~&+2e|1`CyBx991*B~?tgXfkw+Ss3@R@;-NW3`#L|6;djz!ag6WZ``=txJq@%KtWs@__5|0FV&OND-O@+qogxgZQ#>J5DO^?5W1-``vwkWwSjNux?HGc_Esk}-s#}C8B`oY_1`-=gY|G<^zdJoN!ubX61p&Jq>iOrZ{ojf3Jy}Vd&EFSH7XS;_a^M=$EQ<3nDZr3t7O?QV=p7QGtW4V2zxK&7*~7wicsI3_gr{Vt!quVug&}^VS~Gk;JNXlDC10y2knzGC<9e2Rhf|h_T$~?MWCLK+9QF|!P^ZKU3kZ=^wh2Q^k>f*T!}6=XUke}kGlG=gB?^aZbR{`h2pU50KPJFAG=06SZ0tBi+_Lmtb55k@3^(U^c!uo?kHePYQ>N1Q}-dGBQHg@*t}Kh;jDD`_xqEUNve+Ij@o=t#}%)4_xG#9W}iBoV>YOBB`RJf9IQEaIw^XACvY}HQiLbaO4chc*8D=+m5#~~>D$souX@O@w%lXUT0>&tQ%0c@WxeMwd(E=3Y+n<^lexji#Z&xZp#aW#Kk0~;x&K2@orjm9%b!+z{=zYKe0N;+MRu`Q!g{8Q4?mWRet+XWp$#%lk3)JkelN+f8ef)T&D37$P+#um_5`3Iq^&6;+nkw|fS$Ge$-5mHYKe>|r)!}f|7qi2rsJbePK=7=C<OGZEv<^5JQYLR<f6q&NH047d*)AP1;k!MN5~DWWUyCS|EW3#PM6HJR_TcE4d$aBqA&++c+Ma>unzPU)%*^>Kw2m_r}&m6{SD9_g!4PxnUbfU$raGOlp&wY=y$F%r6767>+n^Dtl(r$zoUTYF$Fo|()2Ba^MV2JDm5)ADydgW9h^}rPrSkb@MTdA7p{{nn3arSZDe0s=VOK#hl0|ynC)KB08jvTDmFFt2eOn8yGa8OBofFT1;!lOG%Ka%w`KKx_QCo6;M9gy3URlY00slGWr!8eqEs@Jbtb^e-jGBtWF$gI@Vd1w@N+J}#91dJHY(+eK5HXCn&n3YYKSv)(X5+U)TgYE^K*gsME<EAX|#+|c79K5e<E#yOQAS2D(gz2=Qbw*B^Iic=3HMttt9SioPfdg$B01PMmW2_mF(r#Z2xl;%D7pr4G$E#hCh@<zs?L;go(qAdw?`I&TPJg<y@EF=N_wpq_*)!q*y_=D#4Ev_dv3RSBt`Cssd)wyM%w~oZ1=gsMBn!l)7s^a0Fi0n>IrtPD&RoVw3}WJoaz9M&U>8+@#~-QSAvlqY1q5agbC61=y)HG_xGyBc{N<McGjSQ&y4N8W3+P-R9j8=SQ)?F+{U5r$;T{c7-aS@X*?Y`{m%0M>TuCt^}a85)|)^n+97H27M3%K<(Z*P&tvSCM>-2QJ-q{kM5QvXv68XT8*|Va{b6ro$_QJ)V^u4jkxo6un&^`xy*qnA|lFzjLL_7*g(NGT)lHDaQ1Lc3)e+`rra;4A+c;7uq}^dKIJbaKJaqmNs-FOwQU1O%KgtVd{UTkz~kXTMEYL_=oxzeP9lQ(*j-bQ-Q5k{k3o6H93b0;K!-^03ZGo%G*3r*j6(Rrnu%>88C6i8a|Uc6!4yTt);Y+dMSkd~d;wG%$j=nnu=GGZx8eHyF)+`y`-L>2L=nfXaBfhkd<@JhRDofW(`GiQbTqfzd_D&56`DXyBw)M+N*A#z?&$_c;5l_P(qg&B6&8`3ZXq56{Fou|(tL1f%la&)Ft%k`{ez(%=mDsB%#pcpph6L~tP0i>Y5>Li#kf&cW^Yy8!HFn9a^rhL)(VdW=hcf~yrv1j)-QhUG7OnOXqg}T6*fR1yB4R@iaw&s;jS_PP-&%2WCDxSO(k2s6ayGY0Xd=;<UD&FBe+EYpkkPlM#zbS-lfO^-|qg0E0P0qlr3&j#VBrf{%6PAkz$;NLmy?V<E8?D8Qz8(x}Nxai*VyxfIjc}qf(zKM)=)Fj#(G>fn<R|41*^39Ecl|hl1|otYQ8#ioj(9$b&st29>EPZ#Vio<<r7qSrr;mrYO2I_^;xQ3Ld50M2+078tN<3{XojyFv9`~vSyi@?tXKh8wX4~DSn(ajdIAZjQs0Fn!hYuS%eI?^<f@uELXewJsU-@d6^ohs-w(x+F3{cMYpA8be*y6ax|+5_I28ymy@ZYsQ+LeTIR;$ypi9zSdN*VGPYzL^Sx-|?;S-Ly+upcIi=3E)maDsrBp#V6_jH(rb%N|LVKa!pIqJM%Nj>>>t%fKmDc^(p-@dcWHW~)0?4*j;ls1JLwD16gX%Dpwv3jomkJkGTlO-9^5@F+Vu9?!iXuLY_aoD!7Ej|w2W0qFa?V?wdafTIF?iGyk6^*%{$R#)C)6~Q2nu%tG9J9Biuk}h;S@IlUBbN$j%pP}Of!Dz9lJbd@<GONSXIcUmW+c&GSSn5*Yo#1a&J%g_=TY1mKs3wX7Fxj{&gowPM_R|Cnjw}EBR4CPdoF{bV(j~H<aG*F4%n(%Ja@Vl{cluM0`V?zrlq)5a_Y^qb($^#1DR=)U5MUKkv<Re+QiFY>NtZn^k`9XU%yZgU`=c#-R1Ss@bDpUNq;~Gn{zj#m9zG$49}uqTr8;AuQ(NMNy&Ed6=#Z@gqqjPoqub1y#=WZgU=aVhrl?A&b%Gbx!%XJHL0Y3Xs$zbsb1#B<o>)cNk1V1~7+bH<V`Kg>ryX-Tui%Ie5~X6a4p^o(}_gGPjTVg|#A5ZAwR46_6J#`WF@nN8@w|m7thj)$v?cCiW@OPSr^6_li*_%&yPtBWF=^A7~Y3l??U>>z?^EWWZxNe1SFBsACAX`}Ayh!x-i=kELbo;ZCPsR48Vy+@v*U8Pr#L^{;duSE{jR*cKJkHv@gN2n?@aU>l-bS~W9@qQ02V?_E*#q@<zf$zB-t@o;``-2T#0VrYtMMIW--tnW_oON10<Qq27-Wzw}f?Ro=RN$!?KVOJ)?+?mODbyYGtc%RcpJ=j<J^~gHdgym#NQiryVf#2%b>n?SmLY<_v;pD09+GaiIYb|@(QQCZrR9d+;s|o!^%U+$+B}R^iZ)t)*^1|gpqJL^9>)lYWQfXzDMElya7ZX@DH>4>(&nR<~ajeikcOwS^%d=V#>VQ%fe5Gr@T)rWgiy?dR4N5f+3x{j^{47`ClZU_<sCb{H;GR$2cT_2cLnW6VK}5^kV_fOm_gyT93Lgd|MwO+dEcn{Mz4MNMDlV!stTg1;nPlO77tXn7o8rx@B0elIpB=BFZW}4ZTbi_Qg!p8Ix$swJn)q0FbQQDxV`C<hj#SIZ`4kzOh%;{m`L%(M{CIZbK%I}->LNaB;LqeDF7v^9YKB#Xd}7U-#`~n`iX<a83?uaZgpXegnk$h28ppM*fiJBYzCH#Er_>FyEsp|vK3d<Ynl%6vPVIhe`XG>JmvJ~m&a3yyD>;i%<}gp&_i$SwT(AZgX1;;p{wSE|?R&&7pZFTq$6VM_<;Q;3zDLmNIglU>7sP@pD9^k1h5};7vXr)wCszgMmD&19Vm()zaAVKdFrxS<yjR-yo{C^$oK~M3s@?~Iz0tql`z&FTNh49K``93F^eE&j`up$(R0c=5k}1tcJ*-FL_1Tx(1?wo>QCQYV0Z!)Y6>*9dJBc(ks@UzB?!F*P;hH};?Ow*iUi9z#zGBRg%vXB%@KMGWT<hL*>5-}OMmOu^R%^|BPOMN2V)8Cjt&*``^zP@d279&1k@dcT-2E^!xY@hsjBFoC@;{~esvPmn*1cFv1l{TQpP{}jliFNq-M`%E3jg7{N7X4&$VAq=_j}j?JK)`rOTxMX(XA-#Lp3q>d&poK(VklqAsd!rza8$Qg$AhBs0S;8y0@XysOU#lrS&^^UQC&WOeiSFTqj+-t*akN4L|B<Dn83HInkYYd&<vowF@#NxbSA>U|%%wUnxwS@1-(q@W!l;8@%4Y-{<($p-JNng>*_P)<eG5zu))3`H~TMI?X9XUJv_?2L5z&_Yn5&#Oh#Qow-L9fV|oTR?Z8JK1#wp>fkqOu5~fS_pEc-dep#w9_`~5JJ%{H`O=ln^`e6xJ7-?nIW{u`u($O-9~BSR=I&ccDo5tjsKZ*wEw2yWb81rXtQmK+l)-wXf3G`~hn>5Jmcg{PwHf~S)b7yol$~>xpiZZbC`=<tV?GQmOW>f9%vo11R~S_xAH?!UTXuFTcRZ_PRluh$PaWJ!%*W3(3gtqp{d?IBa%VcHmTlxhpy_on9~ANtJH}MyBE;Fe(jE9Ud3<D=6vYffte%T}Sy9L*=FFA3sno`7RC3vnV#*Ufeko|3y#aKD&y6Gau-$BfV@)oz_cA!oC-8-1jCh5zl^^%-@E-*8tc5Rwoe*AI1PY_v!jO3!(4(F+T-O{;@l<hjgOoYW^A<iLnQ+aY(LU#lN;nV4$<5xK&$%d<DvqoI^P-23?2A8owRFI(0`r<Y{uJfvxW;yE)NxJg9sC)_E+b4eShHJ|!`+~duQKOg3j_({hNX^=LVh`dFMZL-gHfa)40t`LM;&|{BavZ+QfAC%ofP1tgU@M}-jT2vDzaI{Xixh0)bI-=Q8-TP_cx?GEJImW$cK}qyg2tbAbM?3WPX_2y3)T7i%71eMXu|0Qi0X!drlUT&FNHgv`W@`p^;C%u?Fw;WvaA7;p_JBeWz$x@M1|x@T_viw}<bSy>I3kR8DL6C9WTZ`gR8&=?&K~!low+@(s)8-wgNh{5^8<NMml!%GQd4KJVZ2h<%yIk&)q*VIN85%R*e^6~bUUTq}B!_4a*cAbX!&WH+K0$|~=P<>SPiJ>}6h$#^^n_Py>s<td$Fj~ZIYZQmciN3@w7Wp^a^u5}*ETjTeAHy5(wGFw2o|Bd7KkA}XknBSJINU)QMBZmX8iT##jzIo7WxY%<>9qg;^dm(R3keqZIuo#b$a9Ov?^G$17jAZVIwAHiD19{oNcj`#;V$fjEt&kC4H1OxHI~g-EoVsl4rJlL<27d5L6R8JZiUXEXa4#nC$&z-YNR30l$o0xxzpZ!heFvC6MLV?*M+gkUDkgV_o~Q0?swY)OCtaD8vf-=!du=EPb>N)o&P2+d@}cEP#c|<I$S~zBNL9qAi|!j&0wN)jWCMSC#f<-W5rq2Eo^6f-Jjq6pRu(d>S@zqv9lugh3Zh8yEPKjF4g5(OinqZ#M%P05+RX;OHeEOX=`(W62);hy;}^3I@ZTlt0Hx>F20qfY=G}vM=N5x_5YY1xe3a8Hm8g=V&F%;22f;k=;9F4JxebmS<QCPy$KgEI(9a%0zJ3lApuG;v^A0{j0!c@UmZ3zIANpDUo`hv2G@ZS$kPR}_4}y8#zjx&|rWs{!=ICW`UTNUNI&;j1oQqy~i&Dgc@LrS9cWRixMa|&C@G7T!w}UUiG12JUxFi!tUFM+IrtlZI!d6WfQkZ&Jc^9(3Vi@MWY}uu5Ubiw>Pg?k7^vpg{HE=`0h7RtdKpwBTM|v`l`?;eGYck6`To+yZy_tp}wbI$hIoQTWew6RJvgm$v3B>~iJ3u6>V+1R0{LTRYL1LX;ARn`m!CvmVH($sci>~7+6~*qT=;M^5;~!*SR^^Cq58_ky%Gow_YG9SIi<|52;U|K(j<YclZ?jH7TW{j;?-3|mYz;Au4bNdw6!oE;*vn$zfJw9DVoBr@y)f(}5&fPpYw8bCBV{(P=t9;-^qWUpmbEGTpce{*_uBY<AcrTPm6tbO#+BSr(KjX?6ad4qM|e5Z7hU{)o@~!Wh?&|*JfCIE<W?7d?7VZVWFp6l>@Bx7G#-t7T|?iNjp5`5QQW<0P3)^peBYV&l<0WBfK!>Ju&;{f!v%Kwl;w`KOG&VY3H)d6=H7yhFOxFK-}&NuxDCw?OKw~*Go;SBu50LPEOmedZMSEg<YTpgPeueU;5yNATD271iv~W6tl2$1I2|5k?l!M9@MRZx#)RuOhc_BWDKEC#!Jj*%7S273n#(vvX~>76W%4={b*!al-?Yo}Z6F_Y@JUI>>)qK<s(ELy*1;!vbWFqyE@`)l$^OBb0l6%|SKIB#d=TRj`9m94$PiZA_4}~Vxfi4d_uV?evR8$Cy7IocHie^@Iwt=20{Z_0loRzj', 'split_manifest': 'c-rlo+iqPsa)$2*m^WDnSY+|^G)D0>t;B5`wkJKw)8{MMlGWRZj7&Ck71(e~`>+<t`s+WCt3Q4I`1Rxb`2IQ{->-KM>zejqJuLgW-+lP|?_+=Fh4&BRvweN|_~G4`ul%t8_8-^R-#+gzzxh`_ef;?T#p|E#;lsm+zrFkY<J#BP{pIVsKOR1;kAL{3AAhy4Uv2)|mv6s({`kk2Z{Oklf8751<u{AD{AmEspSpeV`)7ZQ>&Nx}^wLkCdmR6N^Wo>$pT6Pa{qeb{-@d>5<<t8I=Jt91_VDlf&%b^D$Y1~RnU($c`?s}vI1hVW-(LxGU2Ct0$r`Q~@4x&0{)@T3`nl)9UmsYNulx5;A3xjs>ylVYyJE_r#7l~1jpZ7%EICCr8rLk+VzARvSaX@3ZmG4@lFep0$8eU#Im)hcr?GNl@Q>G%tH_))#db!KqoEidIAg0xl2%Gfam85roVjq{nE0$ow-R%xrKb4pI^h~UZkg6(MCnmW8ob;3YOBT-4rAD%>)b_qY+BPv7E+9TkHT<8?|Yqi&9u^9JLi5z9uoNZ%%*YV&aBpHmS)(c?lPEX%(m>D)oRr#T0*>jD)+nRa)Y7w`w>Y0^5N5eUcH#=VGUQzOGfE=>q-`P_O(xWSgPyfu}+EHGY(mKoSnAGihG{F{^biT#}%JT{{8nq1MAcKzkhkR9=`nR$A=GJ-+eL={IZL{&i)K|KiePgUbMl_U;6Rp(^qb9f5iaIo)98A@_WC`+E`;7mFO+X8b?)gn`JPi)63aMR#<JWG<Tkbnyh$w;OneHjaQPI0L-Z(5@uLKPMmRvBPr;t+xm#gMPqFvi!RftNpqbo>7s&TbWPTt#x>`vGt1E4SngR>Y?o7kN$<@j@Yt!#>C?sz3@i`Ezh#%Shm9o{Z6&>4Cx@Owol~Xi$o!a;UZT79;1U3BF{yD{o*=*1nM#T^?mC9dtnxITvDyw}?~-enDYG6l+ZxSkoV{q32`(4#=zzNeyCqxF*0X4&oa=2F-aVt>25&o6(Ym9YHl#?T8{4*cbnM!)m^R>*TK2W1P>ej%C?kz7LjXH?d;hSU<$nO~`wJjW-Uf2tZg}M@T$DfeaHSF%^hG74H!oY3s7x-iNZ_bKa79L5miL&pTI|_cZPWnLYdx&fTa^l4gS2q9v|_LOB#J*H>k$AdWhm!>s2u5YcrLE8az4e^GKHvXx0QgL_NQU>y`h=~>3GeJzHMm^%c*v=qODcFKJz{qjvP?OSkr=q;2p5mx<rG|tBhJk=iRD}v4*%8Wh`KQ+nGz^T&363mi4w~1!E9bm25C?*Bqb5Gs;}l*0{&0=LFy!BLtcAg6>Z35kfDykE&&t1?U<$OtS^rp<29di~sIN@o#`88k_Yp(@}@4mRKO!L=BE<&)RIyZlc^ngoR|8GCZeSOY6zjYd}Ld{?|bJ_ObcizI=ISf7s{!?UTpezaN;Ae*%ZUy5bW|%L|z1=H;vdfI*YCbB~6=qzH>BNRYV=EM<$7c){;wpOw(r&1xc5=YUCdT6LKA0C}b!`YQaJVu7xDEulhPDX`dcgE79dL42b%B&XE-C`u`;hr7XoV*9)VQiM->%>zZV$gm4HfW0#JJI#4I!L4u^JahPYJ&u1t-65y)JRZcL<VV$p4(Y@WM2wzT@GL#8scvbF#ww>kzTlq$u?#eCKXbeS(|7~Z<}De{s2z!icm-u&Fh}eKi48f7Ze7+H&DU(oj5USMP2-qn4SNly^{Q{mhXMgpwr|VW`V%RF0A@GRiFsBiqR?k>OP)P<0e+FH$beCXwUUCb>{uyasLDd9)*~s_dLbAt$8PJ;v(EYlNtzBilE!EyG#CiEoO=!xNqzEm;bT%}LFS%^Wtc{6=sCDGmJrBxoEV)DYVA{3v6iUt%u}5`861)-yoDb{p1zTyUS0T*MTO8)Qs(4{Ijt)8wG6Yp0#*=cYaTRO!`GsFyhP36Z+RMzw!(C}ZLRoHY!`N<1-xTn1tJQ(XDKsuG3G_#8#tD|d&84}nb%0M-m34wLe!CW*2yYs<7z=>X|>qkHS64~pY1ptL$OCpC?hIsvP5CLhPHOKw?!Lm9}A}W&uBCFyz3!i1U%A`d}OvUdnaYg({r|tUeVDwYi&4a0tIA&k(+hsir>&yE()uX5L9iT3X7aOLV`(##|lK!*4|J9X0L4A8Elky&D0(YEVSir)%}h(OvFK2F<}F*PM*nbD=4Gweq<PSY_b3W0$f%bw5gM#jd&^+c0HrNLN_B$guiq3V&tkr+&~0QS)t-3U{sM3L)t=Qs`wIHgtU4JyqsP^Tf0SD17s{&&tU_(;R*{Adt##sjwmC$!L3fB_=i20`$-8EiiO5sLtDT4+oCN&Z=v0yZQ;{0IvHwEO_*!R72^)CC7sbQV?^DCagAo+u%S74M2vhrWA&7u&?eJ6i~?X02cB?uoYUT}LT8uBow4DBGKdp%1(NaH$Q$}*NbuZutL}HSZQ{VTGLXumgn_njpZp=V8orN=gcvE)f)<04L`v-mfbRu!(o9d|ft*Q?m<FN<%;>kGc@Caq4Y&r6iI$Fb;u1(Em4l+snuzE%Fof68)^E^eF-3yXo@v<X;|$DmMEVUMt%(lq?r&rfy*64)M<Uuzb9G^pyS{=pp?kvH!VRFYw6{{4@fV?ZZbe}R?N4*V=~MV?dmJb6oT3n<8qOQX79=w}3CNE)34DWFZ^OjWX+^_sgv7ClsaSk8Yb8O<DoZjEf`M693{^lL>x9Lc;PmcR-tTZL4S%D7VN1s-kYb^T32*mD7Z6E5s8N7p1vYhmS_soI1i?4bXV2q-UFWD6)>u;O*a#vu&VUG_Ov~yb;=t!BSlQ;|sYMg@M+M?g`=8+^;RbG*<zZQ2*cLMWFAGMwh5*?Q*1E5mCdm~~tH5&lvhSb<7t>FtoY&wcZl&H7ZV(aB_{q7|synKOgihgF@%_8ZlHs*tKvm0*P@N7EBTBjOM;IGY7e<*<K5J<lgNPXu4H#+7RF|(v7g!{cDE9(kvvW-&h7=WFQp0fL?=bW-?^fO?Eyd9_*P{U#&qSBV9bAiQ9XXNug`9m3q~W*|T8dP|Br+o15S~TAFp);spdEe7h2fbm2!=9OpLhqeTO87KQHmpm2*G22O|b&~m7E~Aan|T5OpH!MBl%|1NC=KvYSucfSKA!ow$-;E6Uvil);(yxMW@-TaC0N|wr~SzgqU~(Hy|*=z?|WII++Zl_3cbXl~5SgI3(PCw}o-{?YuRL?^mosPL;Ty*)8h<AvvG6T5TttV&Nc`l~n?3U^)#7bl|QOWF&0v#sOnHYy}S-cPj69xQ)p@+7aK4n@90+G!mg;ZOgt}Sz+B?%2v7z6)k-}rd)NuaYA|y3JYSzJ&Y5PEfU6rZwy@{dqO;!DUpsa1Ky7Kk;)BKpxmms39QyDaJ!AA^!g+R;dGVcOjKe3G#FY*Xe#5JYPGJ))W!Ij#YiXTv^PD7=ZUYu&DG;=;Z^}Sq5TGKz70`n@pd%N!e9IVZ~0bSqKHYU$q4}F*z$aUFN0?qx~6sc9_)9xDLsZx(tpc$7AT{lPk1(L9!z4pkQx$hq;x|5ZiEu9;wK?PceQZ0@_vUKse~|#)yCZAVio}}xg_-El6n-h;eSF*5Q=sD<Fr-U#x@f1Y8*et(`f+ZhJc|qL=KNQV4Tx?M$d`c=_G_9WL+5jA<#Fi3C1B~t*^i>-N0=@Z)2%gV3Aoc--6sEdDdu=$;K$yhBL7SEin3?SVvqAp)AT?S^oN${f|$6c=!@<KVEvXo#E$O{{P<a+wTqk41sqqA#nHo;nyMe&+HNZ-yrv5pI1>(paPu@00xN0Da5lKTg8PBW=trrKoKjtER6+*8hMjq!{D@-SjC+IIT(_wqFY7;(Biwdu{F|d%N1dS!;oyaFYi*<f=H(Ek1A3A5M$}u9`Cu&n@SFsw+m4Apn_yFLc%b^o;1=Z`7GZbAtYtq#nlZmPAHfK<XPRC*1PTVz7DxtQN=YRL;W$DEZ-b0LFf~b%?Xl5FtDQPFVe6$X#~Yu&`nR{accP`<kFSi7;^P`R<9B$QARW}(i7{RBapPsgCFJ0YfDEI&Oo-dxqZvd1<CxV+j%BT_0_=}rl_z}AyUjW2m{MEP7p(Ky^0Ff*E)c!3pgW@G-{eDMS_acA^{PLb-xlrWu17Gx2qB!Fxz9eFl#e70S9yc!2JjI6?@u(7m)D{gxUP8pMW0nv#2WbB2+8`iV?^7=w%Zzh>%MLV@4-=_t+WURKVoyP!~#Q9_)Vsx%88d>rCZz=l?S1<9QEq<Qr9UDt8vGI?p7;15RXoptUo_iSaBY^j|?vFMVUkwF{g)Te%vkHB}Z?h((*>pn|?V&>T_G%}T*61*i2?hg%3+SPgHxM4Zb)U^46h62*0T_wvIc8QY`DJ>16P>0+UXw>b8m+E6?SrS&6$EL_1KSRvJ;=y$8~qqQXPQYM_<TDg;%1-XSzsV1~gVT@5T8U+HY(>ozo8sb`g7ZK5KKaM;s<I5q-K~6kNeY41qHzKK9(Ln;ZqQ>-Yz@qp@45A*YCK?S+O8N`b>Fq-$i^#PnH^%_%X~=vDtphlcHpWFz8eJxCK<4TQ?slJG({xR~^efcSj&F>*?hi7>=fK)D=aMf?ftX$JGBJ;XVd?>m^$3vtC<R+IOId#4(6p`2+cPo2lbg=OTKIuw?7$^wkpWfp0oCD>8X@7hK^X&!cLro*qZ2|<+<}h^I%f3;B0#COeYY|TsYjJIFeE7U9e3WsFywRsyJW6F_#yBxo!Ihdb^G%vgA@|+btiJ$G@fC1x}S~JtRpmn$UdMRR%Td%phN>$I)gqU($F9QC1|NU-lmO1die$H@(t{YpX|~NIYb#dV<EuZti-uCJ-d|Zn~#0g3@yMMw%x%H@-itV`dRAhuq#-sw}su~#xt(#X+cS(sSp7M6HGlskFC>BoqXqXR}V_nw@-n{!0MMr^G{CPrIFXeJll*!cfk|<xhO?D*!3+-`&l{qp*)Iw(!}87LHw9jeRD+MzE-(gLd7wRBOt-)ZgqB_83^Vx!nD*(k>s!dl+ZHLj5!D`M>4_^Mht^MoX)_pfMOCd0leC}+X?UMxGT3J+gX`#V$K1n$Hu|?;`{M7qVHY1qAY8>?8isVd>?@3J}ZpMgh%=n?%wJg{{`-DPVyy?Kr?*RT3*D#%Iy5mXB6MjIKImd!6YfMCS>)n#0NsFm^*A#s@Bj~NnD8Ea-~yQAN<JNwyHne0NWTCAc&y9uwfYRiP5_4WE@}!C?uRzx^Ml)dcULm+<Vn4VUx6TaU*>bqaigPHkj(*63dsIW^t?=K}%yJ${2Dz=~yjzy}KRuz7D+G<NR{`K%bVKmCImq@hcsHh1*cyffFZU<G5LXY3^IPB+NIuP&4`Pm%w|gBmEb^yE)XC7%*Aks^q>#j_ZQU>N7+v+|IG=y2@Zq$q>efycUFbv^bjhv$1Oz{*!!@p=U((Qs}+~xNu{9S+4<ck}Lcffvu#kfL)G+6}mX}$R`LJ&=EWpW^HorcdAd!p&t|rq{#N+JzM}3E94;U%O4AcJ}7QRU%8U`rptt4bGJfpX$bSKcRTle9eB4V`@v$ZF@-=XBMxo6=xi)AI*adNkcYOy&kR30fzWmFA&{xqBmDU<W5rvY?!N%u&H28h*odt-dB`jS5qEExo6t+@{0ag1@zIPFVh(_V+xK0N7X0Ykd_RS+mwsXX6+<6%;_bE_KaF*dyC>!<P{@M+J*36Z^(->1mF(FB>pu|SwnAFy==~1(3tFWYPBMUJ5m3GTB;-g2x4SmArk-_DP%DSv2m$9q^dS(f!y=r#g52!{_;ujj9`Z{~0DndId3o|L8Llo_vxW#62TV1Kw~Ik@z>xuCmrw3#Dyh7dUjXmz4*Fjp@8+mq)VCkQwXxNGHzPQZ4tt(>pj2S1Sx0*OpB{Jxk{VRn4>Q|JF;j;<kEPiJ<TY&nwNMFyXFnmM$Z%*3q3=-8Ud8e0K4i5j;MsRL_J9Cq@<$CrEa}Bz=l}hz9}iGG)2h~9BLX$}7T~ibJ3-h7?+Q}U?b^T~!M%F0@7Z0&r0XIf>wbs9uOjcK^Zv3_wm;mn@d?2f8fxqK;6%i)^dk$4(OyKs9w-w^B*ncFl(^@Lze3)do%z2&-p#4M_y%PnTthk}#VMZsdL}!})s7#Pw*}D~pddCSpjJIlC{hTXSW^6NC_pYs)|xIPFyhVd@9S|J#Y(NDUzcc2vDbM#+fJF)z*C`ln=fg#>5y)6zhnPOP$$+7B86ie*2SX(VOS7um(I(`cXNB_Wb9E0&(!b})s>xEo|(tp&V^q`-cLvWzQZk9BE(E90jorO7>&gx$6`am)`ed#9mb|DlO)z3V49yNw_l0v+a3SEK;F#*08oSQ<7La7)?=xj;MKQjp%!W%0nHxp9=h6xNu~6C#_Id!R27XRlLWhbv(1lQ@F$}vg(`xD#QBkg;~^e{u#9f|^I<x4CV$?prHF;wgp?CjBiXk5Jpu4-gNFdY*M*tzjwy4`*jnER$Gmo~2=_(WEE2@w62b@}I!N>wtay&Q9TC5dyxRi{S)$OLZv)^*M%}&*DcQHt&mncpRyHz%TZRZ21o>Q*C78oqZ2zU&zS)xiFOYZhEPyo17yZdP7K)@nP#k_h4;h@$0^O9<9~JTQvPTYz;t!gmnk>cR(@K*Cj*h&RlU!u0!QYw5kpU}4L_TGNoWU-Pry0|we!@ja&Rc?ewv_$BF_gt@-|t}n2N>#a>ejneOJyBK+jmN~ccle_ZVYMT4!aSrDc?ck(mwDwpclW^-5v&b9eKCs7WCu9WnxdF?`mzX8`{7d`$2|0Zo3{_-Xd3=UYG`h&H@33?M;N*{uhJgf6o', 'group_mapping': 'c-oDW&yJfg5XSE-?VE6DqNMGWf;T}mHnmN%IgMBX5h@FaIE~a->!FX-SLm4ZPXbxP4fOp!d&V>Re*O5FXNf+wnUkNYR2OxXlyGS4Q}g)X(SR@o%8THMx-04Ew1K=#4o66;J)DbbU!NZzd?w*bOvDb<E+S6T*1AsXvSX5_DT_2Q=UzaiD7Z{+iLikZZ{;9`2kvGCYAtxYT1T;RberUPk;1pSJiTU}vrTk%V+mYkY))R*XVArKQ5^<)=sw%^fLV2^e4lLIL?rPz^WoD@5oxJ87Oo07gTZBg1&%G>C{L7c;h!7^lm?f(WRFu3csvYUtU?fl-@a`!ffq&)M4QFf2s?0$VPlOW5q(w`fPR2TaK$aRP|j8aU%Yz}Jk4p6^lPT!SbWc5UmSH?yqqp54jY>@5KzKKKwMC#Xh7={|FO_%Mg*^oEDpO4VfJpmFiSC8&K7@O<?U1cyXGH_X})|v-D6<%ENAoS9s`@=Op%jKanyOR#4!tQ@7N|7#J&pM9(bHuRsk1WV5WqDQpAVx^&9Z^3U2WUO$`n93nUR6B1UsouDMVXSKfpWPRT^l)C9e<+)dD)K1k5M-Ve2&elJ0<kZB1z_GCui;()sZU=tt{xc8~we(-xu'}

def bundle_bytes(name):
    if name in SOURCE_TEXT:
        return SOURCE_TEXT[name].encode('utf-8')
    return zlib.decompress(base64.b85decode(PAYLOAD[name].encode('ascii')))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_runtime(runtime):
    runtime.mkdir(parents=True, exist_ok=True)
    original = runtime / 'nf_unsw_nb15_v3_temporal_reliability_18W_DUAL_FINAL_READY_BOOTSTRAP_FIX.py'
    original.write_bytes(bundle_bytes('baseline'))
    assert sha256_file(original) == BASE_SOURCE_SHA256
    (runtime/'NF_UNSW_NB15_v3_Expected_Strict_Clean_RFE_Rankings.csv').write_bytes(bundle_bytes('rfe_reference'))
    return original


def module_from_bundle(which, name, file_path):
    module = types.ModuleType(name)
    module.__file__ = str(file_path)
    module.__package__ = ''
    sys.modules[name] = module
    source = bundle_bytes(which).decode('utf-8-sig')
    if which=='r1':
        source=source.replace('P["test"].astype(np.float32)', 'P["test"].astype(np.float64)')
    if which=='baseline':
        # Add only lossless per-flow score recording for the reviewer boundary audit.
        needle='        for bundle in bundles:\n            paper_label ='
        injected="""        for bundle in bundles:
            if globals().get('EXPORT_REVIEWER_SCORES', False) and protocol in ('sliding_temporal','expanding_temporal') and variant in ('Strict-Clean','Strict-Clean-RFE-TopK'):
                score_dir = Path(cfg.OUTPUT_DIR) / 'score_vectors'
                score_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(score_dir / f'{protocol}__{fold_id}__{variant}__{model_name}__{bundle.name}.npz', y=ytest.astype(np.int8), p=bundle.test_prob.astype(np.float64))
            paper_label ="""
        if source.count(needle)!=1:raise AssertionError('Baseline score-recording insertion point changed')
        source=source.replace(needle,injected)
    exec(compile(source, str(file_path), 'exec'), module.__dict__)
    return module


def load_base(original):
    return module_from_bundle('baseline', 'ijics_unified_study', original)


def load_audit(runtime):
    return module_from_bundle('audit', 'IJICS_359541_R1_11_R2_9_ReadOnly_Audits', runtime/'embedded_audit.py')


def patched_argv(args):
    class Swap:
        def __enter__(self):
            self.before = sys.argv
            sys.argv = [sys.argv[0]] + list(args)
        def __exit__(self, *exc):
            sys.argv = self.before
    return Swap()


def dataset_frame(base, data, out):
    cfg = dataclasses.replace(base.CFG, DATA_PATHS=(str(data),),
                             N_TEMPORAL_WINDOWS=18, MAX_ROWS_PER_FILE=None,
                             DROP_GLOBAL_EXACT_DUPLICATES=True)
    out.mkdir(parents=True,exist_ok=True)
    log = out / 'dataset_load.log'
    df = base.load_dataset(cfg, log)
    detected = base.detect_columns(df, cfg)
    df = base.derive_labels(df, detected)
    df, _ = base.prepare_order(df, detected['time'], log)
    return cfg, df


def all_splits(base, cfg, df, out):
    return (base.random_split(df, cfg) + base.single_temporal_split(df, cfg)
            + base.rolling_splits(df, cfg, training_mode='sliding', log_path=out/'dataset_load.log')
            + base.rolling_splits(df, cfg, training_mode='expanding', log_path=out/'dataset_load.log'))


def verify_splits(base, cfg, df, out):
    import pandas as pd
    rows = pd.read_csv(io.BytesIO(bundle_bytes('split_manifest'))).fillna('')
    expected = {(str(r.protocol), str(r.fold_id)):r for r in rows.itertuples(index=False)}
    splits = all_splits(base, cfg, df, out)
    if len(splits) != 24 or len(expected) != 24:
        raise AssertionError('Expected exactly 24 generated split records')
    audit=[]
    for sp in splits:
        key = (sp.protocol, sp.fold_id)
        if key not in expected:
            raise AssertionError(f'Unexpected split {key}')
        ref=expected[key]
        rec={'protocol':sp.protocol,'fold_id':sp.fold_id}
        for role in ('train','calibration','threshold','threshold_pool','test'):
            array = getattr(sp,role+'_idx')
            digest = hashlib.sha256(__import__('numpy').asarray(array,dtype='int64').tobytes()).hexdigest()
            if len(array) != int(getattr(ref,role+'_idx_rows')) or digest != getattr(ref,role+'_idx_sha256'):
                raise AssertionError(f'Generated split differs: {key}, role={role}')
            rec[role+'_rows'] = len(array)
            rec[role+'_sha256']=digest
        audit.append(rec)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(audit).to_csv(out/'Generated_24_Temporal_and_Holdout_Split_Roles.csv',index=False)
    print('PASS: 24 split roles regenerated and matched (120 role-array fingerprints)', flush=True)
    return splits


def execute_baseline(base, data, out):
    base.EXPORT_REVIEWER_SCORES=True
    base.CFG = dataclasses.replace(
        base.CFG, DATA_PATHS=(str(data),), OUTPUT_DIR=str(out),
        EXECUTION_MODE='full', SAVE_MAIN_PREDICTIONS=True,
        ARCHIVE_SUPERSEDED_OUTPUTS=False)
    base.main()


def execute_r1(data, original, out, base_result):
    out.mkdir(parents=True,exist_ok=True)
    r1=module_from_bundle('r1','ijics_reviewer_one',out/'embedded_reviewer_1.py')
    r1.DATA_FILE=data
    r1.ORIGINAL_CODE=original
    r1.ORIGINAL_METRICS=base_result/'csv'/'Final_All_Evaluation_Metrics.csv'
    r1.OUTPUT_DIR=out
    r1.RUN_PARTS={key:True for key in r1.RUN_PARTS}
    r1.SAVE_PERFLOW=True
    r1.main()
    missing=[key for key in r1.RUN_PARTS if not (out/'summary'/('.'+key+'.done')).is_file()]
    if missing:raise RuntimeError('R1 experiments did not complete: '+', '.join(missing))


def execute_cost(data, original, runtime, out, reps):
    out.mkdir(parents=True,exist_ok=True)
    cost=module_from_bundle('cost','ijics_synchronized_cost',out/'embedded_cost.py')
    with patched_argv(['--data',str(data),'--final-code',str(original),
                       '--expected-rfe',str(runtime/'NF_UNSW_NB15_v3_Expected_Strict_Clean_RFE_Rankings.csv'),
                       '--output-dir',str(out),'--reps',str(reps)]):
        cost.main()


def execute_audit(data, original, runtime, out):
    audit=load_audit(runtime)
    with patched_argv(['--dataset',str(data),'--source',str(original),'--out',str(out)]):
        audit.main()


def execute_remaining(task, data, original, runtime, out, arm=None):
    load_audit(runtime)
    rem=module_from_bundle('remaining','ijics_remaining_reviewers',runtime/'embedded_remaining.py')
    args=['--task',task,'--dataset',str(data),'--source',str(original),'--out',str(out)]
    if arm is not None:
        args+=['--resource-arm',str(arm)]
    with patched_argv(args):
        rem.main()


def execute_r2_1(base, data, original, out):
    import numpy as np
    import pandas as pd
    r1=module_from_bundle('r1','ijics_reviewer_one_label_delay',out/'embedded_r1_label_delay.py')
    r1.DATA_FILE = data
    r1.ORIGINAL_CODE=original
    r1.OUTPUT_DIR=out
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary').mkdir(exist_ok=True)
    context=r1.build_context()
    cfg=context.cfg
    df=context.df
    spans=[np.asarray(x,dtype=np.int64) for x in np.array_split(np.arange(len(df)),18)]
    allrows=[]
    for sp in context.splits['sliding']:
        if sp.candidate_number < 7:
            continue
        for lag in (0,1,2):
            history=[int(x)-lag for x in sp.train_window_ids]
            if min(history)<0: continue
            (tri,cali,pooli,cs,ts,cr,tr)=base._build_past_only_support_pools(
                df,[spans[h] for h in history],spans[sp.calibration_window_id],
                spans[sp.threshold_window_id],cfg)
            delayed=dataclasses.replace(sp,train_idx=tri,calibration_idx=cali,
                threshold_pool_idx=pooli,calibration_strategy=cs,
                threshold_pool_strategy=ts,calibration_reserve_rows=cr,
                threshold_reserve_rows=tr,train_window_ids=tuple(history))
            record=r1.run_main_fold(context,delayed,context.strict_clean,
                                    f'label_delay_{lag}',save_perflow=False)
            choices=[r for r in record['rules'] if r.get('probability_source')=='isotonic'
                     and r.get('threshold_name') in ('fixed_0.5','fpr_1.0pct')]
            for policy in ('fixed_0.5','fpr_1.0pct'):
                present=[r for r in choices if r['threshold_name']==policy]
                if len(present)>1:raise AssertionError('Duplicated threshold policy')
                entry=dict(present[0]) if present else {'threshold_name':policy,
                      'status':'training_or_calibration_not_eligible', 'tau':float('nan')}
                entry.update({'protocol':sp.protocol,'fold_id':sp.fold_id,
                              'lag_windows':lag,
                              'history_train_windows':';'.join('W'+str(i) for i in history),
                              'train_rows':len(tri),
                              'train_attacks':int(context.y[tri].sum()),
                              'calibration_rows':len(cali),
                              'test_rows':len(sp.test_idx),
                              'test_window':sp.test_window_id})
                allrows.append(entry)
    if len(allrows)!=42:raise AssertionError(f'R2-1 expected 42 policy records; got {len(allrows)}')
    pd.DataFrame(allrows).to_csv(out/'R2_1_Sliding_0_1_2Window_LabelDelay.csv',index=False)
    print('R2-1 label-delay records:',len(allrows),flush=True)


def get_predictions(base_result, protocol, fold, variant):
    folder=base_result/'predictions'
    name = f'{protocol}_{fold}_{variant}_main.csv'
    p=folder/base_safe_name(name)
    if p.is_file():return p
    found=[x for x in folder.glob('*.csv') if x.name.endswith(base_safe_name(name))]
    if len(found)!=1:raise FileNotFoundError(f'Main prediction file absent or ambiguous: {name}')
    return found[0]


def base_safe_name(s):
    return re.sub(r'[^0-9A-Za-z_.-]+','_',s)


def execute_r2_5(base_result, out):
    import pandas as pd
    out.mkdir(parents=True,exist_ok=True)
    features=[]; comparisons=[]
    groups=pd.read_csv(io.BytesIO(bundle_bytes('group_mapping'))).set_index('feature')['functional_group'].to_dict()
    for protocol in ('sliding_temporal','expanding_temporal'):
        previous=None
        prefix='sliding' if protocol=='sliding_temporal' else 'expanding'
        for num in range(1,14):
            fold=f'{prefix}_{num:02d}'
            rank=base_result/'csv'/f'{protocol}_{fold}_Strict-Clean_RFE_Ranking.csv'
            if not rank.is_file():continue
            a=pd.read_csv(rank)
            selected=set(a.loc[a['selected'].astype(str).str.lower().eq('true'),'feature'].astype(str))
            rec={'protocol':protocol,'fold':fold,'selected_n':len(selected),
                 'jaccard_previous':len(selected&previous)/len(selected|previous) if previous is not None and (selected|previous) else None,
                 'newly_selected_features':len(selected-previous) if previous is not None else None}
            for group in sorted(set(groups.values())):
                rec['group_'+group]=sum(groups.get(f)==group for f in selected)
            features.append(rec)
            previous=selected
            p_full=get_predictions(base_result,protocol,fold,'Strict-Clean')
            p_rfe=get_predictions(base_result,protocol,fold,'Strict-Clean-RFE-TopK')
            full=pd.read_csv(p_full);rfe=pd.read_csv(p_rfe)
            if len(full)!=len(rfe) or not (full.y_true.values==rfe.y_true.values).all():
                raise AssertionError(f'R2-5 row alignment mismatch: {protocol} {fold}')
            y=full.y_true.to_numpy(dtype=int);fa=full.y_pred.to_numpy(dtype=int);rb=rfe.y_pred.to_numpy(dtype=int)
            diff=fa!=rb
            comparisons.append(dict(protocol=protocol,fold=fold,n_test=len(y),n_attack=int(y.sum()),
                same_predictions=int((~diff).sum()),agreement_fraction=float((~diff).mean()),
                different_predictions=int(diff.sum()),changed_benign=int((diff&(y==0)).sum()),
                changed_attack=int((diff&(y==1)).sum()),
                full_fp=int(((fa==1)&(y==0)).sum()),full_fn=int(((fa==0)&(y==1)).sum()),
                rfe_fp=int(((rb==1)&(y==0)).sum()),rfe_fn=int(((rb==0)&(y==1)).sum())))
    if len(comparisons)!=22:raise AssertionError('R2-5 needs 22 original eligible temporal fold predictions')
    pd.DataFrame(features).to_csv(out/'R2_5_Selected_Feature_Group_Stability_22_Folds.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'R2_5_Fixed_Threshold_Paired_Prediction_Agreement_22_Folds.csv',index=False)


def alert_groups(df, times, host_a, host_b, window=60):
    """Count chronological directed-host-pair alert runs separated by more than 60 seconds."""
    import pandas as pd
    if len(df)==0:return 0
    part=pd.DataFrame({'a':host_a,'b':host_b,'t':times}).sort_values(['a','b','t'],kind='stable')
    if part[['a','b','t']].isna().any().any():raise AssertionError('Missing host/timestamp in alert grouping')
    grp=part.groupby(['a','b'],sort=False)['t']
    return int((grp.diff().isna() | (grp.diff()>window*1000)).sum())


def execute_r2_7(r1_out, out):
    import pandas as pd
    import numpy as np
    out.mkdir(parents=True,exist_ok=True)
    rule=pd.read_csv(r1_out/'summary/P2_threshold_records.csv')
    rows=[];marginal=[];recovered=[]
    for pf in sorted((r1_out/'perflow').glob('P2_perflow_sliding_*.csv.gz')):
        df=pd.read_csv(pf)
        fold=str(df.fold_id.iloc[0]);proto=str(df.protocol.iloc[0])
        if int(fold.split('_')[-1])<7:continue
        thresholds={}
        for rule_name in ('fixed_0.5','fpr_1.0pct'):
            x=rule[(rule.protocol==proto)&(rule.fold_id==fold)&(rule.probability_source=='isotonic')]
            rr=x[x.threshold_name==rule_name]
            if len(rr)!=1:raise AssertionError(f'R2-7 threshold missing: {fold}, {rule_name}')
            threshold=float(rr.iloc[0].tau)
            if not np.isfinite(threshold):raise AssertionError(f'R2-7 threshold infeasible: {fold}, {rule_name}')
            thresholds[rule_name]=threshold
            alert=df[df['p_isotonic']>=threshold]
            for scope,area in (('all_alert_flows',alert),('false_positive_flows',alert[alert.label==0])):
                group=alert_groups(area, area.flow_start_ms, area.IPV4_SRC_ADDR,area.IPV4_DST_ADDR)
                rows.append(dict(fold=fold,rule=rule_name,scope=scope,n_alert_flows=len(area),
                    n_proxy_groups=group,n_directed_host_pairs=area[['IPV4_SRC_ADDR','IPV4_DST_ADDR']].drop_duplicates().shape[0],
                    tn=int(rr.iloc[0].tn),fp=int(rr.iloc[0].fp),fn=int(rr.iloc[0].fn),tp=int(rr.iloc[0].tp)))
        fixed=(df.p_isotonic>=thresholds['fixed_0.5'])
        budget=(df.p_isotonic>=thresholds['fpr_1.0pct'])
        new=df[budget&~fixed]
        new_fp=new[new.label==0]
        new_tp=new[new.label==1]
        marginal.append(dict(fold=fold,marginal_FP_flows=len(new_fp),
            marginal_FP_groups=alert_groups(new_fp,new_fp.flow_start_ms,new_fp.IPV4_SRC_ADDR,new_fp.IPV4_DST_ADDR),
            recovered_attack_flows=len(new_tp),
            marginal_FP_pairs=new_fp[['IPV4_SRC_ADDR','IPV4_DST_ADDR']].drop_duplicates().shape[0]))
        fixed_hits=df[fixed]
        for r in new_tp.itertuples(index=False):
            prior=fixed_hits[(fixed_hits.IPV4_SRC_ADDR==r.IPV4_SRC_ADDR)&
                             (fixed_hits.IPV4_DST_ADDR==r.IPV4_DST_ADDR)]
            distances=np.abs(prior.flow_start_ms.to_numpy(dtype=float)-float(r.flow_start_ms))
            nearest=float(distances.min()/1000) if len(distances) else float('nan')
            recovered.append(dict(fold=fold,row_id=int(r.row_id),
                src_ip=str(r.IPV4_SRC_ADDR),dst_ip=str(r.IPV4_DST_ADDR),
                timestamp_ms=float(r.flow_start_ms),previously_detected_directed_pair=bool(len(prior)),
                previously_detected_within_60s=bool(len(distances) and (distances<=60000).any()),
                nearest_fixed_detection_delta_seconds=nearest))
    if len(rows)!=28:raise AssertionError(f'R2-7 needs 28 fold/rule/scope comparisons, got {len(rows)}')
    pd.DataFrame(rows).to_csv(out/'R2_7_PerFold_AlertGroup_Proxy.csv',index=False)
    pd.DataFrame(marginal).to_csv(out/'R2_7_Marginal_Alert_Group_Proxy.csv',index=False)
    pd.DataFrame(recovered).to_csv(out/'R2_7_Recovered_Attack_Flows.csv',index=False)


def execute_r2_8(base_result,out):
    import pandas as pd
    import numpy as np
    out.mkdir(parents=True,exist_ok=True)
    records=pd.read_csv(base_result/'csv'/'Final_All_Evaluation_Metrics.csv')
    records=records[(records.protocol.isin(['sliding_temporal','expanding_temporal']))
                    &(records.variant.isin(['Strict-Clean','Strict-Clean-RFE-TopK']))]
    rows=[]
    keys=['protocol','fold_id','variant','model','probability_source']
    for key,sub in records.groupby(keys,sort=True):
        proto,fold,variant,model,source=key
        scores=base_result/'score_vectors'/f'{proto}__{fold}__{variant}__{model}__{source}.npz'
        if not scores.is_file():raise FileNotFoundError(f'R2-8 score vector not found: {scores.name}')
        with np.load(scores,allow_pickle=False) as packet:
            y=packet['y'];prob=packet['p']
        if len(y)!=len(prob):raise AssertionError('Score/label alignment failed')
        b=y==0;a=y==1
        for rec in sub.itertuples(index=False):
            tau=float(rec.threshold_value)
            pred_ge=prob>=tau
            pred_gt=prob>tau
            tie=prob==tau
            old_fp=int((pred_ge&b).sum());old_fn=int((~pred_ge&a).sum())
            if old_fp!=int(rec.fp) or old_fn!=int(rec.fn):
                raise AssertionError(f'Original decision replay mismatch: {key} {rec.threshold_name}')
            record=dict(zip(keys,key));record.update(dict(threshold_name=rec.threshold_name,
                threshold_value=tau,threshold_hex=tau.hex(),n_test=len(y),
                n_benign=int(b.sum()),n_attack=int(a.sum()),boundary_n=int(tie.sum()),
                boundary_benign=int((tie&b).sum()),boundary_attack=int((tie&a).sum()),
                fp_ge=old_fp,fn_ge=old_fn,tn_ge=int((~pred_ge&b).sum()),tp_ge=int((pred_ge&a).sum()),
                fp_gt=int((pred_gt&b).sum()),fn_gt=int((~pred_gt&a).sum()),
                tn_gt=int((~pred_gt&b).sum()),tp_gt=int((pred_gt&a).sum()),
                threshold_method_eligible=bool(rec.threshold_method_eligible),
                aggregation_eligible=bool(rec.aggregation_eligible),
                threshold_selection_status=str(rec.threshold_selection_status)))
            record['fp_due_to_exact_tie']=record['fp_ge']-record['fp_gt']
            record['tp_due_to_exact_tie']=record['tp_ge']-record['tp_gt']
            rows.append(record)
    if not rows:raise AssertionError('R2-8 generated no records')
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'R2_8_All_Generated_Score_Policy_Threshold_Boundaries.csv',index=False)
    frame[frame.boundary_n>0].to_csv(out/'R2_8_NonZero_Exact_Ties.csv',index=False)
    print(f'R2-8 verified original inclusive-threshold confusion counts for {len(frame)} records',flush=True)


def execute_r2_2_matched(base, data, out, cost_out):
    """Use the original expanding folds and fixed sets to compare RF-46 and RF-15 with 6 candidate splits."""
    import pandas as pd
    import numpy as np
    import time
    import gc
    out.mkdir(parents=True,exist_ok=True)
    cfg,df=dataset_frame(base,data,out)
    clean,ttl,strict_present,_=base.leakage_audit(df,base.detect_columns(df,cfg),cfg)
    drops={base.normalize_name(x) for x in cfg.STRICT_DROP_FEATURES}
    cols=[x for x in clean if base.normalize_name(x) not in drops]
    if len(cols)!=46:raise AssertionError('Strict-Clean predictor count mismatch')
    refs=pd.read_csv(io.BytesIO(bundle_bytes('rfe_reference')))
    folds=[s for s in base.rolling_splits(df,cfg,training_mode='expanding',log_path=out/'load.log') if 4<=s.candidate_number<=13]
    if len(folds)!=10:raise AssertionError('Expected 10 expanding folds E4-E13')
    rows=[]
    for sp in folds:
        frames=[df.iloc[idx] for idx in (sp.train_idx,sp.calibration_idx,sp.threshold_idx,sp.threshold_pool_idx,sp.test_idx)]
        ys=[f['__binary_label__'].to_numpy(dtype=np.int8) for f in frames]
        t0=time.perf_counter()
        arrays=base.fit_transform(*frames,cols)
        prep_s=time.perf_counter()-t0
        xtr,xcal,xval,xpool,xtest,names=arrays
        if list(names)!=cols:raise AssertionError('Feature order mismatch during preprocessing')
        subset=refs[(refs.protocol=='expanding_temporal')&(refs.fold_id==sp.fold_id)&
                    (refs.base_variant=='Strict-Clean')]
        wanted=subset.loc[subset['selected'].astype(bool),'feature'].astype(str).tolist()
        if len(wanted)!=15:raise AssertionError('Expected 15 frozen selected features')
        indices=np.asarray([names.index(feature) for feature in wanted],dtype=int)
        for count in (46,15):
            select=np.arange(46,dtype=int) if count==46 else indices
            xt=np.ascontiguousarray(xtr[:,select])
            xc=np.ascontiguousarray(xcal[:,select]);xe=np.ascontiguousarray(xtest[:,select])
            model=base.make_model('RF',cfg,offset=10200)
            model.set_params(max_features=6,random_state=cfg.RANDOM_STATE+10200)
            begin=time.perf_counter();model.fit(xt,ys[0]);fit_seconds=time.perf_counter()-begin
            cal=base.fit_calibrator(model,xc,ys[1],'isotonic')
            if cal is None:raise AssertionError(f'R2-2 isotonic eligibility failed in {sp.fold_id}')
            prob=base.positive_probability(cal,xe)
            pred=prob>=0.5
            y=ys[4];a=y==1;b=y==0
            rows.append(dict(fold_id=sp.fold_id,arm=('RF46_sqrt_max6' if count==46 else 'RF15_max6'),
                common_seed=int(cfg.RANDOM_STATE+10200),rf_max_features=6,n_features=count,
                train_rows=len(xt),test_rows=len(y),test_attacks=int(a.sum()),
                rf_fitting_seconds=fit_seconds,preprocessing_seconds_excluded=prep_s,
                tn=int((~pred&b).sum()),fp=int((pred&b).sum()),
                fn=int((~pred&a).sum()),tp=int((pred&a).sum())))
            pd.DataFrame(rows).to_csv(out/'R2_2_E4_E13_Candidate_Matched_RF_Fitting.csv',index=False)
            del model,cal,xt,xc,xe,prob,pred
            gc.collect()
        print(f'R2-2 candidate-matched comparison: {sp.fold_id}',flush=True)
        del arrays,frames,ys
        gc.collect()
    table=pd.DataFrame(rows)
    paired=table.pivot(index='fold_id',columns='n_features',values='rf_fitting_seconds').rename(columns={15:'rf15_fit_s',46:'rf46_fit_s'}).reset_index()
    cost_file=cost_out/'Computational_Cost_Addendum_Per_Fold_Summary.csv'
    if not cost_file.is_file():raise FileNotFoundError('Cost addendum must complete first')
    cost=pd.read_csv(cost_file)
    cost=cost[['fold_id','rfe_selector_median_s']]
    paired=paired.merge(cost,on='fold_id',validate='one_to_one')
    paired['feature_selection_plus_rf15_fitting_s']=paired.rfe_selector_median_s+paired.rf15_fit_s
    paired['net_saving_percent']=100*(paired.rf46_fit_s-paired.feature_selection_plus_rf15_fitting_s)/paired.rf46_fit_s
    paired.to_csv(out/'R2_2_E4_E13_Candidate_Matched_Net_Time.csv',index=False)
    print('R2-2 candidate-matched net median (%)',float(paired.net_saving_percent.median()),flush=True)


def stage_task(stage, data, out, runtime, original, reps):
    base_result=out/'baseline'
    r1_out=out/'reviewer_R1'
    if stage == 'r2-5':return execute_r2_5(base_result,out/'R2_5')
    if stage == 'r2-7':return execute_r2_7(r1_out,out/'R2_7')
    if stage == 'r2-8':return execute_r2_8(base_result,out/'R2_8')
    if stage == 'cost':return execute_cost(data,original,runtime,out/'R2_2_synchronized_cost',reps)
    if stage == 'r2-2-matched':return execute_r2_2_matched(load_base(original),data,out/'R2_2_matched',out/'R2_2_synchronized_cost')
    if stage == 'r1':return execute_r1(data,original,r1_out,base_result)
    if stage == 'r1-11-r2-9-audit':return execute_audit(data,original,runtime,out/'R1_11_R2_9_audit')
    if stage.startswith('remaining-'):
        return execute_remaining(stage.split('-',1)[1],data,original,runtime,out/'remaining_tasks')
    if stage == 'r2-3-15':return execute_remaining('r2-3',data,original,runtime,out/'remaining_tasks/resource_15',15)
    if stage == 'r2-3-46':return execute_remaining('r2-3',data,original,runtime,out/'remaining_tasks/resource_46',46)
    if stage in ('r1-6','r1-11','r2-9'):
        return execute_remaining(stage,data,original,runtime,out/'remaining_tasks')
    base=load_base(original)
    if stage=='splits':
        cfg,df=dataset_frame(base,data,out/'split_integrity')
        return verify_splits(base,cfg,df,out/'split_integrity')
    if stage=='baseline':return execute_baseline(base,data,base_result)
    if stage=='r2-1':return execute_r2_1(base,data,original,out/'R2_1')
    raise ValueError(f'Unknown stage {stage}')


def self_test():
    import ast
    manifest=list(csv.DictReader(io.StringIO(bundle_bytes('split_manifest').decode('utf-8-sig'))))
    assert len(manifest)==24
    assert len({(r['protocol'],r['fold_id']) for r in manifest})==24
    for name in ('baseline','r1','cost','audit','remaining'):
        code=bundle_bytes(name)
        ast.parse(code.decode('utf-8-sig'),filename=name+'.py')
    assert hashlib.sha256(bundle_bytes('baseline')).hexdigest()==BASE_SOURCE_SHA256
    assert len(list(csv.DictReader(io.StringIO(bundle_bytes('rfe_reference').decode('utf-8-sig')))))==1012
    assert set(STAGE_ORDER) == set(('splits','baseline','r1','cost','r2-2-matched','r2-1','r2-5','r2-7','r2-8',
                    'r1-11-r2-9-audit','remaining-plan','r1-6','r1-11','r2-3-15','r2-3-46','r2-9'))
    assert alert_groups(range(4),[0,20000,80000,0],['a','a','a','b'],['z','z','z','y'])==2
    assert base_safe_name('hello world.csv')=='hello_world.csv'
    print(f'PASS: source AST, embedded 24-fold manifest, 1012 RFE ranks, all {len(STAGE_ORDER)} stages, alert groups')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('NF-UNSW-NB15-v3.csv'))
    p.add_argument('--out',type=Path,default=Path('IJICS_359541_ALL_IN_ONE_Results'))
    p.add_argument('--stage',choices=['all',*STAGE_ORDER],default='all')
    p.add_argument('--reps',type=int,default=3)
    p.add_argument('--force',action='store_true')
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--list-stages',action='store_true')
    p.add_argument('--_worker',choices=STAGE_ORDER,help=argparse.SUPPRESS)
    p.add_argument('--_runtime',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.self_test:self_test();return
    if args.list_stages:
        for s in STAGE_ORDER:print(s)
        return
    data=args.data.resolve();out=args.out.resolve()
    if not data.is_file():raise FileNotFoundError(f'NF-UNSW-NB15-v3 CSV required at {data}')
    if args.reps<1:raise ValueError('reps must be >=1')
    if not args._worker:
        got=sha256_file(data)
        if got!=RAW_DATA_SHA256:raise ValueError(f'Original dataset fingerprint mismatch {got}')
    out.mkdir(parents=True,exist_ok=True)
    if args._worker:
        if args._runtime is None:raise ValueError('Worker runtime path required')
        runtime=args._runtime.resolve()
        original=runtime/'nf_unsw_nb15_v3_temporal_reliability_18W_DUAL_FINAL_READY_BOOTSTRAP_FIX.py'
        if sha256_file(original)!=BASE_SOURCE_SHA256:raise AssertionError('Base source fingerprint differs')
        return stage_task(args._worker,data,out,runtime,original,args.reps)
    stages=list(STAGE_ORDER) if args.stage=='all' else [args.stage]
    with tempfile.TemporaryDirectory(prefix='ijics_run_') as transient:
        runtime=Path(transient)
        write_runtime(runtime)
        for idx,stage in enumerate(stages,1):
            marker=out/'.stage_done'/f'{stage}.json'
            current_script_hash=sha256_file(Path(__file__).resolve())
            if marker.is_file() and not args.force:
                recorded=json.loads(marker.read_text(encoding='utf-8'))
                if recorded.get('script_sha256')==current_script_hash and recorded.get('dataset_sha256')==RAW_DATA_SHA256:
                    print(f'[{idx}/{len(stages)}] already completed: {stage}',flush=True)
                    continue
            if args.force and stage=='r1':
                for f in (out/'reviewer_R1'/'summary').glob('.*.done'):
                    f.unlink()
            print(f'[{idx}/{len(stages)}] executing: {stage}',flush=True)
            cmd=[sys.executable,str(Path(__file__).resolve()),'--data',str(data),'--out',str(out),
                 '--reps',str(args.reps),'--_worker',stage,'--_runtime',str(runtime)]
            subprocess.run(cmd,check=True)
            marker.parent.mkdir(parents=True,exist_ok=True)
            marker.write_text(json.dumps({'stage':stage,'dataset_sha256':RAW_DATA_SHA256,
                      'original_pipeline_sha256':BASE_SOURCE_SHA256,
                      'script_sha256':current_script_hash},indent=2),encoding='utf-8')
    print('COMPLETE: unified execution finished:',out,flush=True)


if __name__=='__main__':
    main()
