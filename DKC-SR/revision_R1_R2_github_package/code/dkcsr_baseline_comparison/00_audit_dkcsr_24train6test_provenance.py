from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _dkcsr_common import (
    EXISTING_PRED_TEST_SIX,
    Q_SCALE,
    RESULTS_DIR,
    REPORT_DIR,
    SELECTED_ARTIFACT_DIR,
    build_dkc_callable,
    canonical_30_train_or_none,
    ensure_dirs,
    load_24_canonical,
    map_prediction_file_to_canonical,
    md_table,
    rel,
    replay_dkc_for_curve,
)


CANDIDATES = [
    Path("evaluation/pred_train.csv"),
    Path("evaluation/pred_test-six.csv"),
    Path("evaluation/pred_test.csv"),
    Path("artifacts/archive/ivrt-pair-251007/pred_train.csv"),
    Path("artifacts/archive/ivrt-pair-251007/pred_test.csv"),
]


def audit_one(path: Path, curve_train: pd.DataFrame, curve_test: pd.DataFrame, curve_train_30: pd.DataFrame | None) -> dict[str, Any]:
    full_path = path if path.is_absolute() else Path.cwd() / path
    row: dict[str, Any] = {
        "file_path": path.as_posix(),
        "exists": full_path.exists(),
        "number_of_rows": np.nan,
        "unique_record_indices": np.nan,
        "unique_time_points": np.nan,
        "record_idx_values": "",
        "time_h_values": "",
        "matches_canonical_24_train_curve_set": False,
        "matches_canonical_6_test_curve_set": False,
        "matches_canonical_30_train_curve_set": False,
        "matches_canonical_order_24_train": False,
        "matches_canonical_order_6_test": False,
        "max_abs_diff_Q_obs_vs_24_train": np.nan,
        "max_abs_diff_Q_obs_vs_6_test": np.nan,
        "appears_to_correspond_to": "missing",
        "can_be_safely_used_for_24plus6": False,
        "short_notes": "file not found",
    }
    if not full_path.exists():
        return row

    df = pd.read_csv(full_path)
    row["number_of_rows"] = int(len(df))
    row["unique_record_indices"] = int(df["record_idx"].nunique()) if "record_idx" in df.columns else np.nan
    row["unique_time_points"] = int(df["time_h"].nunique()) if "time_h" in df.columns else np.nan
    row["record_idx_values"] = json.dumps(sorted(df["record_idx"].dropna().astype(int).unique().tolist())) if "record_idx" in df.columns else ""
    row["time_h_values"] = json.dumps(sorted(df["time_h"].dropna().astype(float).unique().tolist())) if "time_h" in df.columns else ""

    _, train_info = map_prediction_file_to_canonical(full_path, curve_train, "audit")
    _, test_info = map_prediction_file_to_canonical(full_path, curve_test, "audit")
    row["matches_canonical_24_train_curve_set"] = bool(train_info.get("curve_set_matches_canonical"))
    row["matches_canonical_6_test_curve_set"] = bool(test_info.get("curve_set_matches_canonical"))
    row["matches_canonical_order_24_train"] = bool(train_info.get("sequence_matches_canonical_order"))
    row["matches_canonical_order_6_test"] = bool(test_info.get("sequence_matches_canonical_order"))
    row["max_abs_diff_Q_obs_vs_24_train"] = train_info.get("curve_set_max_abs_diff_Q_obs", np.nan)
    row["max_abs_diff_Q_obs_vs_6_test"] = test_info.get("curve_set_max_abs_diff_Q_obs", np.nan)

    if curve_train_30 is not None:
        _, train30_info = map_prediction_file_to_canonical(full_path, curve_train_30, "audit")
        row["matches_canonical_30_train_curve_set"] = bool(train30_info.get("curve_set_matches_canonical"))

    n_records = int(row["unique_record_indices"]) if pd.notna(row["unique_record_indices"]) else 0
    if row["matches_canonical_6_test_curve_set"] and n_records == 6:
        row["appears_to_correspond_to"] = "6 test formulations"
        row["can_be_safely_used_for_24plus6"] = True
        note = "all six canonical test curves match after Q_obs curve remapping"
        if not row["matches_canonical_order_6_test"]:
            note += "; row order differs from the 24+6 canonical test file"
        row["short_notes"] = note
    elif row["matches_canonical_24_train_curve_set"] and n_records == 24:
        row["appears_to_correspond_to"] = "24 training formulations"
        row["can_be_safely_used_for_24plus6"] = True
        note = "all 24 canonical training curves match after Q_obs curve remapping"
        if not row["matches_canonical_order_24_train"]:
            note += "; row order differs from first-24 Excel order"
        row["short_notes"] = note
    elif row["matches_canonical_30_train_curve_set"] and n_records == 30:
        row["appears_to_correspond_to"] = "30 training formulations"
        row["short_notes"] = "matches the older 30-training canonical split, not the 24-training split as a whole"
    elif n_records == 12:
        row["appears_to_correspond_to"] = "12 test/ranking formulations"
        row["short_notes"] = "contains 12 curves; first six may match the held-out test set but the file is not a clean 6-test file"
    elif n_records == 24:
        row["appears_to_correspond_to"] = "24 training formulations (unverified/mismatched)"
        row["short_notes"] = "row count suggests 24 training curves, but canonical Q_obs matching was incomplete"
    else:
        row["appears_to_correspond_to"] = "unknown"
        row["short_notes"] = "does not fully match canonical 24-train or 6-test curve sets"
    return row


def replay_vs_existing(curve_test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    replay = replay_dkc_for_curve(curve_test)
    existing, existing_info = map_prediction_file_to_canonical(
        EXISTING_PRED_TEST_SIX,
        curve_test,
        "DKC-SR existing pred_test-six.csv",
    )
    meta: dict[str, Any] = {"existing_pred_test_six": existing_info}
    if existing is None:
        return pd.DataFrame(), meta
    replay_s = replay.sort_values(["record_index", "time_h"]).reset_index(drop=True)
    existing_s = existing.sort_values(["record_index", "time_h"]).reset_index(drop=True)
    comp = replay_s[["record_index", "run_no", "time_h", "Q_obs", "Q_pred"]].rename(columns={"Q_pred": "Q_pred_replay"})
    comp["Q_pred_existing"] = existing_s["Q_pred"].to_numpy(float)
    comp["abs_diff"] = np.abs(comp["Q_pred_replay"] - comp["Q_pred_existing"])
    q6 = comp[np.isclose(comp["time_h"], 6.0)]
    meta.update(
        {
            "max_abs_diff_full_curve": float(comp["abs_diff"].max()),
            "mean_abs_diff_full_curve": float(comp["abs_diff"].mean()),
            "max_abs_diff_Q6": float(q6["abs_diff"].max()),
            "mean_abs_diff_Q6": float(q6["abs_diff"].mean()),
            "Q6_agrees_within_5_Q_units": bool(np.allclose(q6["Q_pred_replay"], q6["Q_pred_existing"], atol=5.0, rtol=0.0)),
            "full_curve_agrees_small_tolerance": bool(np.allclose(comp["Q_pred_replay"], comp["Q_pred_existing"], atol=1e-6, rtol=1e-9)),
        }
    )
    return comp, meta


def write_report(audit: pd.DataFrame, comp_meta: dict[str, Any]) -> None:
    _, normalized_expr, expr_source = build_dkc_callable()
    text: list[str] = []
    text.append("# DKC-SR Prediction Provenance Audit, 24 Train + 6 Test")
    text.append("")
    text.append("## Scope")
    text.append(f"- q_scale used for replay: `{Q_SCALE}`.")
    text.append("- The selected artifact config q_scale was not read or used.")
    text.append(f"- Replayed selected expression from `{expr_source}`.")
    text.append(f"- Normalized expression used for replay: `{normalized_expr}`.")
    text.append("- No symbolic-regression search or DKC-SR retraining was run.")
    text.append("")
    text.append("## Prediction File Audit")
    cols = [
        "file_path",
        "number_of_rows",
        "unique_record_indices",
        "unique_time_points",
        "matches_canonical_24_train_curve_set",
        "matches_canonical_6_test_curve_set",
        "matches_canonical_30_train_curve_set",
        "matches_canonical_order_24_train",
        "matches_canonical_order_6_test",
        "appears_to_correspond_to",
        "can_be_safely_used_for_24plus6",
        "short_notes",
    ]
    text.extend(md_table(audit, cols))
    text.append("")
    text.append("## Replay Versus Existing pred_test-six.csv")
    if comp_meta.get("existing_pred_test_six", {}).get("loaded"):
        text.append("- `evaluation/pred_test-six.csv` can be mapped to the same six canonical test formulations by matching `Q_obs` curves.")
        text.append(f"- Row order matches canonical 24+6 test order: `{comp_meta['existing_pred_test_six'].get('sequence_matches_canonical_order')}`.")
        text.append(f"- Matched Run No order in file: `{comp_meta['existing_pred_test_six'].get('matched_run_no')}`.")
        text.append(f"- Max absolute prediction difference over full curve: `{comp_meta.get('max_abs_diff_full_curve')}`.")
        text.append(f"- Mean absolute prediction difference over full curve: `{comp_meta.get('mean_abs_diff_full_curve')}`.")
        text.append(f"- Max absolute prediction difference at Q6: `{comp_meta.get('max_abs_diff_Q6')}`.")
        text.append(f"- Q6 agrees within 5 Q units: `{comp_meta.get('Q6_agrees_within_5_Q_units')}`.")
        text.append(f"- Full curve agrees within small tolerance: `{comp_meta.get('full_curve_agrees_small_tolerance')}`.")
    else:
        text.append(f"- Existing `pred_test-six.csv` could not be safely mapped: `{comp_meta.get('existing_pred_test_six')}`.")
    text.append("")
    text.append("## Outputs")
    text.append(f"- `{rel(RESULTS_DIR / 'dkcsr_prediction_file_audit.csv')}`")
    text.append(f"- `{rel(RESULTS_DIR / 'dkcsr_qscale3008_replay_vs_existing.csv')}`")
    (REPORT_DIR / "00_dkcsr_24train6test_provenance_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    canon = load_24_canonical()
    curve_train_30 = canonical_30_train_or_none()
    audit_rows = [audit_one(path, canon["curve_train"], canon["curve_test"], curve_train_30) for path in CANDIDATES]
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(RESULTS_DIR / "dkcsr_prediction_file_audit.csv", index=False)
    comp, comp_meta = replay_vs_existing(canon["curve_test"])
    comp.to_csv(RESULTS_DIR / "dkcsr_qscale3008_replay_vs_existing.csv", index=False)
    write_report(audit, comp_meta)
    print("[OK] DKC-SR provenance audit complete.")
    print(f"[OK] q_scale used: {Q_SCALE}")
    print(f"[OK] Results: {rel(RESULTS_DIR / 'dkcsr_prediction_file_audit.csv')}")


if __name__ == "__main__":
    main()
