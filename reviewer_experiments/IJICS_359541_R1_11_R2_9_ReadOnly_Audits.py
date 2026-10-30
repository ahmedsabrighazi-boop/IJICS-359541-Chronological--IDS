#!/usr/bin/env python3
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
ORIGINAL_SOURCE_SHA_FOR_PROVENANCE = "48883b3ec91a63e09e98a74705d160e5ac1819b6f72b19bd78a67a280877d058"
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
