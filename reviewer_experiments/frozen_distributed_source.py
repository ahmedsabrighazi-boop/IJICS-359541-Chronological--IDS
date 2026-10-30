#!/usr/bin/env python3
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
