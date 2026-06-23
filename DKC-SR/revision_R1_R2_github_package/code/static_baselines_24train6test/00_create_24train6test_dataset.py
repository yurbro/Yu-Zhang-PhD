from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_validation_24train6test"
DATA_DIR = OUT / "data"
CONFIG_DIR = OUT / "config"
REPORT_DIR = OUT / "reports"

DATA_XLSX = ROOT / "data" / "IVRT-Pure.xlsx"
Q_SCALE = 3008.198194823261

C_COLS = ["Poloxamer 407", "Ethanol", "Propylene glycol"]
C_ALIASES = {"Poloxamer 407": "C1", "Ethanol": "C2", "Propylene glycol": "C3"}
C_BOUNDS = {"C1": (20.0, 30.0), "C2": (10.0, 20.0), "C3": (10.0, 20.0)}
TIME_POINTS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
R_COLS = ["R_t1", "R_t2", "R_t3", "R_t4", "R_t5", "R_t6", "R_t7"]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def ensure_dirs() -> None:
    for directory in [DATA_DIR, CONFIG_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def normalize_c(name: str, value: float) -> float:
    lo, hi = C_BOUNDS[name]
    return (float(value) - lo) / (hi - lo)


def as_run_list(df: pd.DataFrame) -> list[str]:
    return df["Run No"].astype(str).tolist()


def read_required_sheet(xls: pd.ExcelFile, sheet_name: str, columns: list[str]) -> pd.DataFrame:
    if sheet_name not in xls.sheet_names:
        raise ValueError(f"Missing sheet: {sheet_name}")
    df = pd.read_excel(xls, sheet_name=sheet_name)
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{sheet_name} missing required columns: {missing}")
    return df[columns].copy()


def select_split_rows(xls: pd.ExcelFile) -> dict[str, pd.DataFrame]:
    formulas_train = read_required_sheet(xls, "Formulas-train", ["Run No"] + C_COLS)
    release_train = read_required_sheet(xls, "Release-train", ["Run No"] + R_COLS)
    formulas_test = read_required_sheet(xls, "Formulas-test", ["Run No"] + C_COLS)
    release_test = read_required_sheet(xls, "Release-test", ["Run No"] + R_COLS)

    train_formulas_24 = formulas_train.iloc[:24].copy()
    train_release_24 = release_train.iloc[:24].copy()
    excluded_formulas_6 = formulas_train.iloc[24:].copy()
    excluded_release_6 = release_train.iloc[24:].copy()

    if as_run_list(train_formulas_24) != as_run_list(train_release_24):
        raise ValueError("First 24 Run No values differ between Formulas-train and Release-train.")
    if as_run_list(excluded_formulas_6) != as_run_list(excluded_release_6):
        raise ValueError("Excluded Run No values differ between Formulas-train and Release-train.")
    if as_run_list(formulas_test) != as_run_list(release_test):
        raise ValueError("Run No values differ between Formulas-test and Release-test.")

    return {
        "train_formulas_24": train_formulas_24,
        "train_release_24": train_release_24,
        "excluded_formulas_6": excluded_formulas_6,
        "excluded_release_6": excluded_release_6,
        "test_formulas_6": formulas_test.copy(),
        "test_release_6": release_test.copy(),
        "all_formulas_train": formulas_train,
        "all_release_train": release_train,
    }


def merge_formula_release(formulas: pd.DataFrame, release: pd.DataFrame) -> pd.DataFrame:
    formulas = formulas.copy()
    release = release.copy()
    formulas["_source_order"] = np.arange(len(formulas))
    merged = pd.merge(formulas, release, on="Run No", how="inner", sort=False)
    merged = merged.sort_values("_source_order").drop(columns=["_source_order"]).reset_index(drop=True)
    if len(merged) != len(formulas):
        missing = sorted(set(formulas["Run No"].astype(str)) - set(merged["Run No"].astype(str)))
        raise ValueError(f"Merge lost formulation rows; missing Run No values: {missing}")
    return merged


def build_canonical(merged: pd.DataFrame, dataset: str, record_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []

    for idx, row in merged.iterrows():
        run_no = str(row["Run No"])
        record_id = f"{record_prefix}_{idx:03d}"
        c_values = {alias: float(row[col]) for col, alias in C_ALIASES.items()}
        c_norm = {f"{name}n": normalize_c(name, value) for name, value in c_values.items()}
        q_nonzero = [float(row[col]) if pd.notna(row[col]) else np.nan for col in R_COLS]
        q_values = [0.0] + q_nonzero

        for time_h, q_obs in zip(TIME_POINTS, q_values):
            curve_rows.append(
                {
                    "record_id": record_id,
                    "record_index": int(idx),
                    "run_no": run_no,
                    "dataset": dataset,
                    "time_h": float(time_h),
                    "Q_obs": float(q_obs) if np.isfinite(q_obs) else np.nan,
                    "C1": c_values["C1"],
                    "C2": c_values["C2"],
                    "C3": c_values["C3"],
                    "C1n": c_norm["C1n"],
                    "C2n": c_norm["C2n"],
                    "C3n": c_norm["C3n"],
                }
            )

        endpoint_rows.append(
            {
                "record_id": record_id,
                "record_index": int(idx),
                "run_no": run_no,
                "dataset": dataset,
                "C1": c_values["C1"],
                "C2": c_values["C2"],
                "C3": c_values["C3"],
                "C1n": c_norm["C1n"],
                "C2n": c_norm["C2n"],
                "C3n": c_norm["C3n"],
                "Q6_obs": float(q_values[-1]),
                "AUC_obs": float(np.trapz(np.asarray(q_values, dtype=float), np.asarray(TIME_POINTS, dtype=float))),
            }
        )

    curve_cols = ["record_id", "record_index", "run_no", "dataset", "time_h", "Q_obs", "C1", "C2", "C3", "C1n", "C2n", "C3n"]
    endpoint_cols = ["record_id", "record_index", "run_no", "dataset", "C1", "C2", "C3", "C1n", "C2n", "C3n", "Q6_obs", "AUC_obs"]
    return pd.DataFrame(curve_rows, columns=curve_cols), pd.DataFrame(endpoint_rows, columns=endpoint_cols)


def condition_set(endpoint: pd.DataFrame) -> set[tuple[float, float, float]]:
    return set(map(tuple, endpoint[["C1", "C2", "C3"]].round(8).to_numpy()))


def bool_text(value: bool) -> str:
    return "yes" if value else "no"


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, list):
                val = ", ".join(map(str, val))
            vals.append(str(val).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_report(
    train_endpoint: pd.DataFrame,
    test_endpoint: pd.DataFrame,
    excluded_formulas: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    train_runs = train_endpoint["run_no"].astype(str).tolist()
    test_runs = test_endpoint["run_no"].astype(str).tolist()
    excluded_runs = excluded_formulas["Run No"].astype(str).tolist()
    leak_checks = config["leak_checks"]

    text: list[str] = []
    text.append("# 24 Train + 6 Test Dataset Report")
    text.append("")
    text.append("## Split Definition")
    text.append(f"- q_scale used for revision-validation context: `{Q_SCALE}`.")
    text.append("- Training set: first 24 rows from `Formulas-train` and first 24 rows from `Release-train`.")
    text.append("- Test set: all 6 rows from `Formulas-test` and all 6 rows from `Release-test`.")
    text.append("- The remaining 6 rows from the original training sheets were excluded from training.")
    text.append("- Split level: formulation-level by `Run No`; each formulation contributes an entire curve to one split.")
    text.append("")
    text.append("## Run No Values")
    text.append(f"- First 24 training formulations used: `{train_runs}`.")
    text.append(f"- Excluded 6 original training rows: `{excluded_runs}`.")
    text.append(f"- Held-out 6 test formulations: `{test_runs}`.")
    text.append("")
    text.append("## Leakage Checks")
    text.append(f"- Train/test `Run No` overlap: `{leak_checks['train_test_run_no_overlap']}`.")
    text.append(f"- Train/test formulation-condition overlap: `{leak_checks['train_test_formulation_condition_overlap']}`.")
    text.append(f"- Any train/test leakage detected: `{bool_text(leak_checks['any_train_test_leakage'])}`.")
    text.append(f"- Excluded rows accidentally present in train: `{leak_checks['excluded_run_no_in_train']}`.")
    text.append(f"- Excluded rows present in held-out test: `{leak_checks['excluded_run_no_in_test']}`.")
    text.append("")
    text.append("## Canonical Data")
    text.append(f"- Time points: `{TIME_POINTS}`.")
    text.append("- `Q(0)=0` was added internally to every curve.")
    text.append("- `R_t1..R_t7` were mapped to `0.5, 1, 2, 3, 4, 5, 6` hours.")
    text.append("- AUC was calculated by trapezoidal integration over the full curve including `Q(0)=0`.")
    text.append(f"- Train curve rows: `{config['train']['curve_rows']}`; train endpoint rows: `{config['train']['endpoint_rows']}`.")
    text.append(f"- Test curve rows: `{config['test']['curve_rows']}`; test endpoint rows: `{config['test']['endpoint_rows']}`.")
    text.append("")
    text.append("## Output Files")
    for path in config["outputs"].values():
        text.append(f"- `{path}`")
    text.append("")
    text.append("## Endpoint Preview")
    preview_rows = []
    for label, df in [("train", train_endpoint.head(5)), ("test", test_endpoint)]:
        for _, row in df.iterrows():
            preview_rows.append(
                {
                    "dataset": label,
                    "run_no": row["run_no"],
                    "C1": f"{row['C1']:.4g}",
                    "C2": f"{row['C2']:.4g}",
                    "C3": f"{row['C3']:.4g}",
                    "Q6_obs": f"{row['Q6_obs']:.6g}",
                    "AUC_obs": f"{row['AUC_obs']:.6g}",
                }
            )
    text.extend(md_table(preview_rows, ["dataset", "run_no", "C1", "C2", "C3", "Q6_obs", "AUC_obs"]))
    (REPORT_DIR / "00_dataset_24train6test_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not DATA_XLSX.exists():
        raise FileNotFoundError(DATA_XLSX)

    xls = pd.ExcelFile(DATA_XLSX)
    split = select_split_rows(xls)

    train_merged = merge_formula_release(split["train_formulas_24"], split["train_release_24"])
    test_merged = merge_formula_release(split["test_formulas_6"], split["test_release_6"])

    curve_train, endpoint_train = build_canonical(train_merged, dataset="train", record_prefix="train24")
    curve_test, endpoint_test = build_canonical(test_merged, dataset="test", record_prefix="test6")

    train_runs = set(endpoint_train["run_no"].astype(str))
    test_runs = set(endpoint_test["run_no"].astype(str))
    excluded_runs = set(split["excluded_formulas_6"]["Run No"].astype(str))
    train_conditions = condition_set(endpoint_train)
    test_conditions = condition_set(endpoint_test)

    leak_checks = {
        "train_test_run_no_overlap": sorted(train_runs & test_runs),
        "train_test_formulation_condition_overlap": sorted(train_conditions & test_conditions),
        "excluded_run_no_in_train": sorted(excluded_runs & train_runs),
        "excluded_run_no_in_test": sorted(excluded_runs & test_runs),
        "any_train_test_leakage": bool((train_runs & test_runs) or (train_conditions & test_conditions)),
        "split_level": "formulation-level by Run No",
    }

    outputs = {
        "curve_train": rel(DATA_DIR / "canonical_curve_train_24.csv"),
        "curve_test": rel(DATA_DIR / "canonical_curve_test_6.csv"),
        "endpoint_train": rel(DATA_DIR / "canonical_endpoint_train_24.csv"),
        "endpoint_test": rel(DATA_DIR / "canonical_endpoint_test_6.csv"),
        "config": rel(CONFIG_DIR / "revision_24train6test_config.json"),
        "report": rel(REPORT_DIR / "00_dataset_24train6test_report.md"),
    }
    config: dict[str, Any] = {
        "q_scale": Q_SCALE,
        "q_scale_source": "fixed by codex_task_24train6test_baseline.md; artifact cfg q_scale was not read",
        "data_xlsx": rel(DATA_XLSX),
        "split_definition": {
            "train": "first 24 rows from Formulas-train and Release-train",
            "excluded": "rows 25-30 from Formulas-train and Release-train",
            "test": "all 6 rows from Formulas-test and Release-test",
        },
        "time_points": TIME_POINTS,
        "release_columns": R_COLS,
        "c_bounds": C_BOUNDS,
        "run_no": {
            "train_first_24": endpoint_train["run_no"].astype(str).tolist(),
            "excluded_original_train_6": split["excluded_formulas_6"]["Run No"].astype(str).tolist(),
            "test_6": endpoint_test["run_no"].astype(str).tolist(),
        },
        "train": {
            "formulation_rows": int(len(train_merged)),
            "curve_rows": int(len(curve_train)),
            "endpoint_rows": int(len(endpoint_train)),
            "missing_q_values": int(curve_train["Q_obs"].isna().sum()),
        },
        "test": {
            "formulation_rows": int(len(test_merged)),
            "curve_rows": int(len(curve_test)),
            "endpoint_rows": int(len(endpoint_test)),
            "missing_q_values": int(curve_test["Q_obs"].isna().sum()),
        },
        "leak_checks": leak_checks,
        "outputs": outputs,
    }

    curve_train.to_csv(DATA_DIR / "canonical_curve_train_24.csv", index=False)
    curve_test.to_csv(DATA_DIR / "canonical_curve_test_6.csv", index=False)
    endpoint_train.to_csv(DATA_DIR / "canonical_endpoint_train_24.csv", index=False)
    endpoint_test.to_csv(DATA_DIR / "canonical_endpoint_test_6.csv", index=False)
    (CONFIG_DIR / "revision_24train6test_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    write_report(endpoint_train, endpoint_test, split["excluded_formulas_6"], config)

    print("[OK] Created canonical 24-train/6-test datasets.")
    print(f"[OK] Train Run No values: {config['run_no']['train_first_24']}")
    print(f"[OK] Excluded Run No values: {config['run_no']['excluded_original_train_6']}")
    print(f"[OK] Test Run No values: {config['run_no']['test_6']}")


if __name__ == "__main__":
    main()
