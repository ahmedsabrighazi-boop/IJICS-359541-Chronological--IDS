#!/usr/bin/env python3
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
