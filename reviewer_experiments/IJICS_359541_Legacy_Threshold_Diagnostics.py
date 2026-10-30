#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only diagnostics for IJICS-359541 LEGACY R1-4, R2-8, and S12.

This is a diagnostic *module*, not a full model-training runner.
Call export_threshold_support(...) only after original legacy scores have
been computed. It does not fit models, alter arrays, set random seeds, or call RNG.

Data requirements:
- y_support: 1D labels for a past threshold-selection support set.
- scores: 1D predicted probabilities from that same set.
- source: uncalibrated, sigmoid, or isotonic.
- support_role: raw_threshold_window or historical_pool.
- fold_id and protocol: original fold identifiers.

It writes one summary JSON, legacy and full ROC CSVs, and maximum-Youden
candidates. The caller should invoke it after persisting original legacy
metrics, not in the middle of their computation. No result changes are
permitted under the LEGACY reproduction contract.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve

EPS_GRID = 1.0 / 300.0
ALPHAS = (0.0001, 0.001, 0.005, 0.01, 0.02)


def _csv_write(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _roc_rows(y: np.ndarray, p: np.ndarray, drop: bool):
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=drop)
    rows = []
    for idx, (fp, tp, t) in enumerate(zip(fpr, tpr, thresholds)):
        rows.append({
            "roc_index": idx,
            "threshold": float(t),
            "fpr": float(fp),
            "tpr": float(tp),
            "youden_j": float(tp - fp) if np.isfinite(tp - fp) else None,
            "threshold_is_finite": bool(np.isfinite(t)),
        })
    return rows


def _past_counts(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    neg = y == 0
    pos = y == 1
    fp = int(np.count_nonzero(pred & neg))
    tp = int(np.count_nonzero(pred & pos))
    nb = int(np.count_nonzero(neg))
    na = int(np.count_nonzero(pos))
    return {"threshold": float(threshold), "fp": fp, "tp": tp,
            "fpr": fp / nb if nb else None,
            "tpr": tp / na if na else None}


def _budget_minimum(roc_rows: list[dict], alpha: float):
    """Smallest finite ROC threshold with FPR <= alpha, ignoring sentinel infinity."""
    feasible = [r for r in roc_rows
                if r["threshold_is_finite"] and r["fpr"] <= alpha]
    if not feasible:
        return None
    return min(feasible, key=lambda r: r["threshold"])


def export_threshold_support(*, fold_id: str, protocol: str,
                             support_role: str, source: str,
                             y_support, scores, output_dir,
                             platt_a: float | None = None,
                             platt_b: float | None = None) -> dict:
    """Save deterministic read-only support diagnostics for a single fold/source.

    Neither input array is ever modified. score values must be the full-precision
    arrays used to select thresholds, not rounded saved CSV displays.
    """
    y = np.asarray(y_support)
    p = np.asarray(scores)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape:
        raise ValueError("y_support and scores must be 1D arrays of equal length")
    if y.size == 0:
        raise ValueError("empty threshold support")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("support labels must be 0 or 1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probability scores must be finite and in [0,1]")
    if source not in {"uncalibrated", "sigmoid", "isotonic"}:
        raise ValueError("unrecognized RF probability source")
    if support_role not in {"raw_threshold_window", "historical_pool"}:
        raise ValueError("unrecognized support role")

    root = Path(output_dir) / str(protocol) / str(fold_id) / support_role / source
    root.mkdir(parents=True, exist_ok=True)
    benign = p[y == 0]
    attacks = p[y == 1]
    n = int(len(p))
    has_two = bool(len(benign) > 0 and len(attacks) > 0)
    non_grid = np.abs(p - np.rint(p * 300.0) / 300.0) > 1e-12
    min_pos = p[p > 0]
    zero_benign = int(np.count_nonzero(benign == 0.0))
    summary = {
        "protocol": protocol, "fold_id": fold_id, "support_role": support_role,
        "source": source, "n_support": n,
        "n_benign": int(len(benign)), "n_attack": int(len(attacks)),
        "n_benign_score_zero": zero_benign,
        "fraction_benign_score_zero": (zero_benign / len(benign)) if len(benign) else None,
        "fraction_benign_score_positive": (float(np.count_nonzero(benign > 0) / len(benign))
                                          if len(benign) else None),
        "smallest_positive_score": float(np.min(min_pos)) if len(min_pos) else None,
        "n_scores_open_0_to_1_div_300": int(np.count_nonzero((p > 0) & (p < EPS_GRID))),
        "n_benign_scores_open_0_to_1_div_300": int(np.count_nonzero((benign > 0) & (benign < EPS_GRID))),
        "n_off_grid_scores": int(np.count_nonzero(non_grid)),
        "fraction_off_grid_scores": float(np.mean(non_grid)),
        "n_unique_scores": int(len(np.unique(p))),
        "one_over_300": EPS_GRID,
        "at_one_over_300": _past_counts(y, p, EPS_GRID),
        "platt_a": float(platt_a) if platt_a is not None else None,
        "platt_b": float(platt_b) if platt_b is not None else None,
        "legacy_roc_drop_intermediate": True,
        "full_roc_drop_intermediate": False,
        "note": "Off-grid RF scores do not by themselves prove conflicting labels or identical feature vectors.",
    }
    if has_two:
        legacy = _roc_rows(y, p, True)
        full = _roc_rows(y, p, False)
        summary["n_roc_points_legacy"] = len(legacy)
        summary["n_roc_points_full"] = len(full)
        summary["budget_comparisons"] = {}
        for alpha in ALPHAS:
            x = _budget_minimum(legacy, alpha)
            z = _budget_minimum(full, alpha)
            summary["budget_comparisons"][f"{alpha:.4g}"] = {
                "legacy_minimum_roc_threshold": x["threshold"] if x else None,
                "full_minimum_roc_threshold": z["threshold"] if z else None,
                "legacy_fpr_at_threshold": _past_counts(y, p, x["threshold"])["fpr"] if x else None,
                "full_fpr_at_threshold": _past_counts(y, p, z["threshold"])["fpr"] if z else None,
                "thresholds_equal": bool(x and z and x["threshold"] == z["threshold"]),
            }
        # np.argmax behavior exactly matches the legacy youden_threshold
        # selector, except for any final clipping (thresholds here are [0,1]).
        jl = np.array([r["youden_j"] if r["youden_j"] is not None else -np.inf for r in legacy])
        j_max = float(np.max(jl))
        j_idx = int(np.argmax(jl))
        maxima = [r for r in legacy if r["youden_j"] is not None
                  and abs(r["youden_j"] - j_max) <= 1e-14]
        summary["legacy_youden_threshold"] = float(legacy[j_idx]["threshold"])
        summary["legacy_youden_max_j"] = j_max
        summary["n_max_j_legacy_points_within_1e_minus_14"] = len(maxima)
        summary["legacy_youden_selected_past_counts"] = _past_counts(y, p, float(legacy[j_idx]["threshold"]))
        _csv_write(root / "ROC_Legacy_DropIntermediate_True.csv", legacy,
                   ["roc_index", "threshold", "fpr", "tpr", "youden_j", "threshold_is_finite"])
        _csv_write(root / "ROC_Full_DropIntermediate_False.csv", full,
                   ["roc_index", "threshold", "fpr", "tpr", "youden_j", "threshold_is_finite"])
        _csv_write(root / "Youden_Maximum_J_Candidates.csv", maxima,
                   ["roc_index", "threshold", "fpr", "tpr", "youden_j", "threshold_is_finite"])
    else:
        summary["n_roc_points_legacy"] = None
        summary["n_roc_points_full"] = None
        summary["budget_comparisons"] = None
        summary["legacy_youden_threshold"] = None
        summary["legacy_youden_max_j"] = None
        summary["n_max_j_legacy_points_within_1e_minus_14"] = None

    (root / "Threshold_Support_Diagnostics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary



def compare_s12_raw_sigmoid(*, y_support, raw_support, sigmoid_support,
                            y_test, raw_test, sigmoid_test,
                            fold_id="sliding_12", output_dir=".") -> dict:
    """Diagnose numerical tie/ordering differences and S12 Youden decisions.

    Call with full precision scores from the SAME RF fitted on the SAME fold.
    Calibrator Platt coefficients should be recorded by export_threshold_support.
    Read-only; does not fit models, calibrate, or consume RNG.
    """
    yv, yr, ys = np.asarray(y_support), np.asarray(raw_support), np.asarray(sigmoid_support)
    yt, pr, ps = np.asarray(y_test), np.asarray(raw_test), np.asarray(sigmoid_test)
    if yv.ndim != 1 or yr.shape != yv.shape or ys.shape != yv.shape:
        raise ValueError("support labels and both score vectors must have equal 1D shapes")
    if yt.ndim != 1 or pr.shape != yt.shape or ps.shape != yt.shape:
        raise ValueError("test labels and score vectors must have equal 1D shapes")
    if not all(np.isfinite(x).all() for x in (yr, ys, pr, ps)):
        raise ValueError("nonfinite score")
    if len(np.unique(yv)) < 2:
        raise ValueError("cannot select Youden on single-class threshold support")
    # Exact legacy code semantics: roc_curve default drop_intermediate=True;
    # choose the first maximum J, after excluding only nonfinite thresholds.
    def legacy_youden(y, p):
        fpr, tpr, tau = roc_curve(y, p)
        ok = np.isfinite(tau)
        return float(np.clip(tau[ok][np.argmax(tpr[ok] - fpr[ok])], 0, 1))
    tau_raw = legacy_youden(yv, yr)
    tau_sig = legacy_youden(yv, ys)
    idx = np.argsort(yr, kind="stable")
    d_raw = np.diff(yr[idx]); d_sig = np.diff(ys[idx])
    n_merged_adjacent_distinct_raw = int(np.count_nonzero((d_raw > 0) & (d_sig == 0)))
    n_strict_order_reversals_adjacent = int(np.count_nonzero((d_raw > 0) & (d_sig < 0)))
    n_same_raw_different_sigmoid_adjacent = int(np.count_nonzero((d_raw == 0) & (d_sig != 0)))
    pred_r, pred_s = pr >= tau_raw, ps >= tau_sig
    pb = (yt == 0)
    attack = (yt == 1)
    result = {
        "fold_id": fold_id,
        "n_support": int(len(yv)), "n_test": int(len(yt)),
        "n_unique_raw_support": int(np.unique(yr).size),
        "n_unique_sigmoid_support": int(np.unique(ys).size),
        "n_adjacent_distinct_raw_scores_merged_by_sigmoid": n_merged_adjacent_distinct_raw,
        "n_adjacent_strict_order_reversals": n_strict_order_reversals_adjacent,
        "n_adjacent_identical_raw_scores_with_distinct_sigmoid_scores": n_same_raw_different_sigmoid_adjacent,
        "selected_raw_youden_threshold": tau_raw,
        "selected_sigmoid_youden_threshold": tau_sig,
        "raw_test_fp": int(np.count_nonzero(pb & pred_r)),
        "sigmoid_test_fp": int(np.count_nonzero(pb & pred_s)),
        "raw_test_fn": int(np.count_nonzero(attack & ~pred_r)),
        "sigmoid_test_fn": int(np.count_nonzero(attack & ~pred_s)),
        "n_discordant_test_predictions": int(np.count_nonzero(pred_r != pred_s)),
        "n_benign_only_sigmoid_alerts": int(np.count_nonzero(pb & (~pred_r) & pred_s)),
        "n_benign_only_raw_alerts": int(np.count_nonzero(pb & pred_r & (~pred_s))),
        "n_attack_only_sigmoid_alerts": int(np.count_nonzero(attack & (~pred_r) & pred_s)),
        "n_attack_only_raw_alerts": int(np.count_nonzero(attack & pred_r & (~pred_s))),
        "interpretation_limit": "Summary supports detecting merged scores but cannot prove their causal source without calibration parameters and saved per-flow scores."
    }
    p = Path(output_dir) / str(fold_id) / "S12_Raw_Sigmoid_YOUDEN_Comparison.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def self_test() -> None:
    import tempfile
    # This case demonstrates that ROC point pruning can change the smallest
    # eligible finite FPR-budget threshold even when score ordering is unchanged.
    y = np.array([1, 1, 0, 0, 0, 1, 0, 1, 0, 0], dtype=int)
    p = np.array([1, 26, 20, 22, 14, 13, 7, 10, 6, 12], dtype=float) / 29
    with tempfile.TemporaryDirectory() as tmp:
        result = export_threshold_support(
            fold_id="toy", protocol="toy", support_role="raw_threshold_window",
            source="uncalibrated", y_support=y, scores=p, output_dir=tmp)
        # 0.2 is deliberately outside the production alpha list; check it directly.
        before = _budget_minimum(_roc_rows(y, p, True), 0.2)
        after = _budget_minimum(_roc_rows(y, p, False), 0.2)
        assert before is not None and after is not None
        assert before["threshold"] > after["threshold"]
        assert result["n_unique_scores"] == 10
    print("Self-test PASSED: read-only diagnostics saved; ROC-pruning discrepancy detected.")


if __name__ == "__main__":
    self_test()