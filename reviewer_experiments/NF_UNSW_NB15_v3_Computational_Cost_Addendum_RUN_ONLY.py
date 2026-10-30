#!/usr/bin/env python3
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
    main()