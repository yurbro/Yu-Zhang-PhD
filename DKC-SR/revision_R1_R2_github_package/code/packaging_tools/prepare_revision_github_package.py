from __future__ import annotations

import csv
import fnmatch
import os
import shutil
import stat
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "revision_R1_R2_github_package"

SKIP_DIRS = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".git",
    "revision_R1_R2_github_package",
}

COPY_PATH_FORBIDDEN_PARTS = [
    "SKIPPED",
    "all30",
    "30train",
    "24train_vs_30train",
]

BOOTSTRAP_INCLUDE_PATTERNS = [
    "*fixed_parameter_uncertainty*.csv",
    "*fixed_parameter_prediction_uncertainty*.csv",
    "*bootstrap*constant*.csv",
    "*bootstrap*metric*.csv",
    "*bootstrap*prediction*.csv",
]

BOOTSTRAP_EXCLUDE_PATTERNS = [
    "*SKIPPED*",
    "*all30*",
    "*30train*",
    "*24train_vs_30train*",
    "*old*",
    "*obsolete*",
]

MANUAL_REVIEW_PATTERNS = [
    "*table_bootstrap_constant_stability.csv",
    "*table_dfdq_consistency_summary.csv",
    "*table_repeated_split_baseline_summary.csv",
    "*dkcsr_fixed_structure_bootstrap_constants.csv",
    "*dkcsr_fixed_structure_bootstrap_metrics.csv",
    "*dkcsr_fixed_structure_bootstrap_test_predictions.csv",
    "*all30*b2*bootstrap*.csv",
    "*30train*b2*bootstrap*.csv",
    "*30train*comparison*.csv",
]

CODE_GROUPS = [
    {
        "code_category": "static_baseline",
        "dest_dir": "code/static_baselines_24train6test",
        "notes": "Code used to build the canonical 24-train/6-test split and static baseline evidence.",
        "files": [
            "revision_validation_24train6test/00_create_24train6test_dataset.py",
            "revision_validation_24train6test/01_static_endpoint_baselines_24train6test.py",
            "revision_validation_24train6test/02_curve_proxy_baselines_24train6test.py",
            "revision_validation_24train6test/config/revision_24train6test_config.json",
        ],
    },
    {
        "code_category": "dkcsr_comparison",
        "dest_dir": "code/dkcsr_baseline_comparison",
        "notes": "Code used to add DKC-SR to unified same-split baseline comparison tables and plots.",
        "files": [
            "revision_validation_24train6test_dkcsr/00_audit_dkcsr_24train6test_provenance.py",
            "revision_validation_24train6test_dkcsr/01_add_dkcsr_to_endpoint_table.py",
            "revision_validation_24train6test_dkcsr/02_add_dkcsr_to_curve_table.py",
            "revision_validation_24train6test_dkcsr/_dkcsr_common.py",
        ],
    },
    {
        "code_category": "vanilla_odesr",
        "dest_dir": "code/vanilla_odesr_baseline",
        "notes": "Code used to configure, run, evaluate, and summarize vanilla ODE-SR baselines.",
        "files": [
            "revision_validation_vanilla_odesr_24train6test/00_prepare_vanilla_odesr_config.py",
            "revision_validation_vanilla_odesr_24train6test/01_run_vanilla_odesr_multiseed.py",
            "revision_validation_vanilla_odesr_24train6test/02_evaluate_vanilla_odesr_candidates.py",
            "revision_validation_vanilla_odesr_24train6test/03_structural_validity_audit_vanilla_odesr.py",
            "revision_validation_vanilla_odesr_24train6test/04_physical_optimisation_audit_vanilla_vs_dkcsr.py",
            "revision_validation_vanilla_odesr_24train6test/05_make_vanilla_odesr_summary.py",
            "revision_validation_vanilla_odesr_24train6test/_vanilla_common.py",
            "revision_validation_vanilla_odesr_24train6test/config/vanilla_odesr_config.json",
        ],
    },
    {
        "code_category": "bootstrap_b2_only",
        "dest_dir": "code/bootstrap_b2_only",
        "notes": "Corrected fixed-structure bootstrap and fixed-parameter uncertainty code with numerator coefficient a=2 fixed.",
        "files": [
            "revision_validation_bootstrap_correction_24train6test/00_reconstruct_final_equation.py",
            "revision_validation_bootstrap_correction_24train6test/01_full_train_b2_refit_sanity_check.py",
            "revision_validation_bootstrap_correction_24train6test/02_corrected_fixed_structure_bootstrap.py",
            "revision_validation_bootstrap_correction_24train6test/03_fixed_parameter_uncertainty_if_needed.py",
            "revision_validation_bootstrap_correction_24train6test/04_make_bootstrap_correction_summary.py",
            "revision_validation_bootstrap_correction_24train6test/_corrected_bootstrap_common.py",
            "revision_validation_bootstrap_correction_24train6test/config/corrected_bootstrap_config.json",
        ],
    },
    {
        "code_category": "plotting_audit",
        "dest_dir": "code/plotting_and_audits",
        "notes": "Audit/plotting code for physical plausibility and DKC-SR comparison figures.",
        "files": [
            "revision_validation_24train6test_dkcsr/03_physical_plausibility_audit_static_vs_dkcsr.py",
        ],
    },
    {
        "code_category": "packaging_tool",
        "dest_dir": "code/packaging_tools",
        "notes": "Script used to assemble this GitHub-ready revision evidence package.",
        "files": [
            "tools/prepare_revision_github_package.py",
        ],
    },
]

CODE_MANUAL_REVIEW_PATTERNS = [
    "revision_validation_bootstrap_b2_all30_diagnostic/*.py",
    "revision_validation_bootstrap_b2_30train6test_diagnostic/*.py",
    "revision_validation_qscale975/*.py",
    "revision_validation_robustness_24train6test/*.py",
    "revision_validation_unconstrained_sr_24train6test/*.py",
    "revision_validation_unconstrained_sr_structural_audit/*.py",
]


EXPECTED_GROUPS = [
    {
        "category": "manuscript",
        "dest_dir": "manuscript",
        "notes": "Manuscript revision file requested by package task.",
        "files": [
            "Manuscript.docx",
            "Response to Reviewers.docx",
            "Supplementary Material.docx",
        ],
    },
    {
        "category": "static_baseline",
        "dest_dir": "reports/static_baselines_24train6test",
        "preferred": "revision_validation_24train6test/reports",
        "notes": "Static baseline report for canonical 24-train/6-test split.",
        "files": [
            "00_dataset_24train6test_report.md",
            "01_static_endpoint_baselines_24train6test_report.md",
            "summary_24train6test_baseline_report.md",
        ],
    },
    {
        "category": "dkcsr_comparison",
        "dest_dir": "reports/dkcsr_baseline_comparison",
        "preferred": "revision_validation_24train6test_dkcsr/reports",
        "notes": "DKC-SR versus same-split static baseline comparison report.",
        "files": [
            "00_dkcsr_24train6test_provenance_report.md",
            "01_unified_endpoint_with_dkcsr_report.md",
            "02_unified_curve_with_dkcsr_report.md",
            "summary_dkcsr_static_baseline_24train6test_report.md",
        ],
    },
    {
        "category": "vanilla_odesr",
        "dest_dir": "reports/vanilla_odesr_baseline",
        "preferred": "revision_validation_vanilla_odesr_24train6test/reports",
        "notes": "Vanilla ODE-SR same-split baseline evidence.",
        "files": [
            "00_vanilla_odesr_config_report.md",
            "01_vanilla_odesr_multiseed_run_report.md",
            "02_vanilla_odesr_evaluation_report.md",
            "03_vanilla_odesr_structural_validity_report.md",
            "04_vanilla_vs_dkcsr_physical_optimisation_audit_report.md",
            "summary_vanilla_odesr_baseline_24train6test.md",
        ],
    },
    {
        "category": "bootstrap_b2_only",
        "dest_dir": "reports/bootstrap_b2_only",
        "preferred": "revision_validation_bootstrap_correction_24train6test/reports",
        "notes": "Corrected final-equation and fixed-parameter uncertainty evidence; skipped bootstrap reports are excluded.",
        "files": [
            "00_final_equation_reconstruction_report.md",
            "03_fixed_parameter_uncertainty_report.md",
            "summary_bootstrap_correction_for_R1_2.md",
        ],
    },
    {
        "category": "table",
        "dest_dir": "tables",
        "notes": "Requested table/CSV evidence.",
        "files": [
            "static_endpoint_baseline_metrics_24train6test.csv",
            "static_endpoint_baseline_predictions_24train6test.csv",
            "table_vanilla_odesr_predictive_structural_comparison.csv",
            "table_vanilla_odesr_physical_optimisation_comparison.csv",
        ],
    },
    {
        "category": "figure",
        "dest_dir": "figures",
        "notes": "Requested manuscript-revision figures.",
        "files": [
            "unified_auc_parity_with_dkcsr_24train6test.png",
            "unified_curve_parity_with_dkcsr_24train6test.png",
            "unified_q6_parity_with_dkcsr_24train6test.png",
            "physical_plausibility_summary_barplot.png",
        ],
    },
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def iter_files() -> Iterable[Path]:
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.relative_to(ROOT).parts)
        if parts & SKIP_DIRS:
            continue
        yield path


def is_forbidden_copy_source(path: Path) -> bool:
    text = rel(path).replace("\\", "/")
    return any(part in text for part in COPY_PATH_FORBIDDEN_PARTS)


def score_candidate(path: Path, preferred: str | None = None) -> tuple[int, int, str]:
    path_text = rel(path).replace("\\", "/")
    preferred_hit = 0 if preferred and preferred in path_text else 1
    return (preferred_hit, len(path.relative_to(ROOT).parts), path_text)


def find_candidates(filename: str) -> list[Path]:
    return [path for path in iter_files() if path.name == filename]


def choose_file(filename: str, preferred: str | None = None, allow_forbidden_source: bool = False) -> Path | None:
    candidates = find_candidates(filename)
    if not allow_forbidden_source:
        candidates = [path for path in candidates if not is_forbidden_copy_source(path)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: score_candidate(p, preferred))[0]


def ensure_clean_package() -> None:
    if PACKAGE.exists():
        if PACKAGE.resolve() == ROOT.resolve() or ROOT.resolve() not in PACKAGE.resolve().parents:
            raise RuntimeError(f"Refusing to remove suspicious package path: {PACKAGE}")
        def _handle_remove_readonly(func, path, exc_info):
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                raise exc_info[1]

        shutil.rmtree(PACKAGE, onerror=_handle_remove_readonly)
    for subdir in [
        "manuscript",
        "reports/static_baselines_24train6test",
        "reports/dkcsr_baseline_comparison",
        "reports/vanilla_odesr_baseline",
        "reports/bootstrap_b2_only",
        "tables",
        "figures",
        "code/static_baselines_24train6test",
        "code/dkcsr_baseline_comparison",
        "code/vanilla_odesr_baseline",
        "code/bootstrap_b2_only",
        "code/plotting_and_audits",
        "code/packaging_tools",
    ]:
        (PACKAGE / subdir).mkdir(parents=True, exist_ok=True)


def copy_file(source: Path, relative_dest: str) -> None:
    dest = PACKAGE / relative_dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def manifest_row(relative_path: str, source_path: str, file_type: str, category: str, status: str, notes: str) -> dict[str, str]:
    return {
        "relative_path": relative_path,
        "source_path": source_path,
        "file_type": file_type,
        "category": category,
        "status": status,
        "notes": notes,
    }


def code_manifest_row(relative_path: str, source_path: str, file_type: str, code_category: str, status: str, notes: str) -> dict[str, str]:
    return {
        "relative_path": relative_path,
        "source_path": source_path,
        "code_category": code_category,
        "file_type": file_type,
        "status": status,
        "notes": notes,
    }


def copy_expected_files(rows: list[dict[str, str]]) -> None:
    for group in EXPECTED_GROUPS:
        for filename in group["files"]:
            source = choose_file(filename, preferred=group.get("preferred"))
            file_type = Path(filename).suffix.lstrip(".").lower() or "file"
            dest_rel = f"{group['dest_dir']}/{filename}"
            if source is None:
                rows.append(
                    manifest_row(
                        dest_rel,
                        "",
                        file_type,
                        group["category"],
                        "missing",
                        f"Expected file not found: {filename}",
                    )
                )
                continue
            copy_file(source, dest_rel)
            rows.append(
                manifest_row(
                    dest_rel,
                    rel(source),
                    file_type,
                    group["category"],
                    "copied",
                    group["notes"],
                )
            )


def path_matches_any(path: Path, patterns: list[str]) -> bool:
    text = rel(path).replace("\\", "/")
    name = path.name
    return any(fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(text, pattern) for pattern in patterns)


def copy_bootstrap_csvs(rows: list[dict[str, str]]) -> None:
    preferred_root = "revision_validation_bootstrap_correction_24train6test/results"
    for path in sorted(iter_files(), key=lambda p: rel(p)):
        if path.suffix.lower() != ".csv":
            continue
        path_text = rel(path).replace("\\", "/")
        if not path_matches_any(path, BOOTSTRAP_INCLUDE_PATTERNS):
            continue
        if path_matches_any(path, BOOTSTRAP_EXCLUDE_PATTERNS) or is_forbidden_copy_source(path):
            continue
        if preferred_root not in path_text:
            rows.append(
                manifest_row(
                    "",
                    path_text,
                    "csv",
                    "manual_review",
                    "manual_review",
                    "Bootstrap-like CSV outside corrected b2-only package; not copied automatically.",
                )
            )
            continue
        dest_rel = f"reports/bootstrap_b2_only/{path.name}"
        copy_file(path, dest_rel)
        rows.append(
            manifest_row(
                dest_rel,
                path_text,
                "csv",
                "bootstrap_b2_only",
                "copied",
                "Corrected fixed-parameter uncertainty CSV related to b2-only robustness evidence.",
            )
        )


def add_manual_review_candidates(rows: list[dict[str, str]]) -> None:
    already = {row["source_path"] for row in rows if row["source_path"]}
    for path in sorted(iter_files(), key=lambda p: rel(p)):
        path_text = rel(path).replace("\\", "/")
        if path_text in already:
            continue
        if path_matches_any(path, MANUAL_REVIEW_PATTERNS):
            rows.append(
                manifest_row(
                    "",
                    path_text,
                    path.suffix.lstrip(".").lower() or "file",
                    "manual_review",
                    "manual_review",
                    "Potentially relevant but excluded from automatic package because it may be obsolete, optional, all30/30train diagnostic, or superseded.",
                )
            )


def copy_code_files(code_rows: list[dict[str, str]]) -> None:
    for group in CODE_GROUPS:
        for source_rel in group["files"]:
            source = ROOT / source_rel
            filename = Path(source_rel).name
            file_type = Path(filename).suffix.lstrip(".").lower() or "file"
            dest_rel = f"{group['dest_dir']}/{filename}"
            if not source.exists():
                code_rows.append(
                    code_manifest_row(
                        dest_rel,
                        source_rel.replace("\\", "/"),
                        file_type,
                        group["code_category"],
                        "missing",
                        f"Expected revision code file not found: {source_rel}",
                    )
                )
                continue
            if is_forbidden_copy_source(source):
                code_rows.append(
                    code_manifest_row(
                        "",
                        rel(source),
                        file_type,
                        "manual_review",
                        "manual_review",
                        "Code path contains an excluded keyword; not copied automatically.",
                    )
                )
                continue
            copy_file(source, dest_rel)
            code_rows.append(
                code_manifest_row(
                    dest_rel,
                    rel(source),
                    file_type,
                    group["code_category"],
                    "copied",
                    group["notes"],
                )
            )


def add_code_manual_review_candidates(code_rows: list[dict[str, str]]) -> None:
    already = {row["source_path"] for row in code_rows if row["source_path"]}
    for path in sorted(iter_files(), key=lambda p: rel(p)):
        if path.suffix.lower() not in {".py", ".ipynb", ".md", ".yaml", ".yml", ".toml", ".json"}:
            continue
        path_text = rel(path).replace("\\", "/")
        if path_text in already:
            continue
        if any(fnmatch.fnmatch(path_text, pattern) for pattern in CODE_MANUAL_REVIEW_PATTERNS):
            code_rows.append(
                code_manifest_row(
                    "",
                    path_text,
                    path.suffix.lstrip(".").lower() or "file",
                    "manual_review",
                    "manual_review",
                    "Potentially related revision code but excluded from automatic package because it is diagnostic, exploratory, qscale-specific, or not part of final R1-2/R1-5 evidence.",
                )
            )


def copied_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["status"] == "copied"]


def missing_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["status"] == "missing"]


def manual_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["status"] == "manual_review"]


def write_manifest(rows: list[dict[str, str]]) -> None:
    manifest_path = PACKAGE / "MANIFEST.csv"
    fieldnames = ["relative_path", "source_path", "file_type", "category", "status", "notes"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_code_manifest(code_rows: list[dict[str, str]]) -> None:
    manifest_path = PACKAGE / "CODE_MANIFEST.csv"
    fieldnames = ["relative_path", "source_path", "code_category", "file_type", "status", "notes"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(code_rows)


def bullet_list(items: list[str], fallback: str = "None.") -> list[str]:
    if not items:
        return [fallback]
    return [f"- {item}" for item in items]


def package_tree() -> list[str]:
    lines: list[str] = []
    for path in sorted(PACKAGE.rglob("*"), key=lambda p: rel(p)):
        if path == PACKAGE:
            continue
        rel_path = path.relative_to(PACKAGE)
        depth = len(rel_path.parts) - 1
        prefix = "  " * depth
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{prefix}- {rel_path.name}{suffix}")
    return lines


def write_readme(rows: list[dict[str, str]], code_rows: list[dict[str, str]] | None = None) -> None:
    code_rows = code_rows or []
    missing = [row["relative_path"] for row in missing_rows(rows)]
    manual = [f"{row['source_path']} ({row['notes']})" for row in manual_rows(rows)]
    manual.extend(
        f"{row['source_path']} ({row['notes']})"
        for row in code_rows
        if row["status"] == "manual_review"
    )
    category_counts = Counter(row["category"] for row in copied_rows(rows))
    code_counts = Counter(row["code_category"] for row in code_rows if row["status"] == "copied")

    lines = [
        "# Revision evidence package for DKC-SR dermal formulation manuscript",
        "",
        "## Purpose",
        "",
        "This folder contains the final evidence files supporting the revised manuscript and response to reviewers, especially R1-2 (small dataset robustness and fixed-structure bootstrap refitting) and R1-5 (same-split baseline comparisons against static regressors and vanilla ODE-SR).",
        "",
        "## Folder Structure",
        "",
        "- `manuscript/`: manuscript and response documents when available.",
        "- `reports/static_baselines_24train6test/`: same-split static baseline reports.",
        "- `reports/dkcsr_baseline_comparison/`: DKC-SR versus static baseline comparison reports.",
        "- `reports/vanilla_odesr_baseline/`: vanilla ODE-SR baseline reports.",
        "- `reports/bootstrap_b2_only/`: corrected final-equation and b2-only/fixed-parameter robustness evidence.",
        "- `tables/`: selected CSV tables used as revision evidence.",
        "- `figures/`: selected manuscript/revision figures.",
        "- `code/`: scripts and helper/config files used to generate the current revision evidence.",
        "",
        "## Key Evidence Summary",
        "",
        "### R1-2 Robustness",
        "",
        "- Corrected fixed-structure bootstrap refitting was performed with the selected symbolic structure fixed.",
        "- The numerator term `softplus(2 Qtilde)` was treated as structural and was not refitted.",
        "- Only the denominator offset `b2` was refitted.",
        "- Bootstrap success rate: 500/500.",
        "- Bootstrap b2: 2.564 +/- 0.199.",
        "- 95% interval: 2.212 to 2.942.",
        "- Selected final-equation b2 = 2.355029 lies within this interval.",
        "- Bootstrap-refit test curve RMSE: 499.990 +/- 37.032.",
        "",
        "### R1-5 Baseline Comparison",
        "",
        "- Same 24-training/6-test formulation-level split was used.",
        "- Static baselines included PLS, ridge, polynomial RSM degree 2, RF, and GPR.",
        "- DKC-SR achieved the lowest test curve RMSE among the compared models.",
        "- DKC-SR also achieved the lowest AUC RMSE among the compared models.",
        "- PLS achieved the lowest Q6 endpoint RMSE.",
        "- Vanilla ODE-SR candidates partially fitted the curves but collapsed to formulation-independent expressions.",
        "",
        "## Copied File Counts",
        "",
    ]
    for category in sorted(category_counts):
        lines.append(f"- {category}: {category_counts[category]}")
    if not category_counts:
        lines.append("None.")

    lines.extend(
        [
            "",
            "## Code included in this package",
            "",
            "The `code/` folder contains scripts used to generate the current revision evidence.",
            "",
            "The code files are included for revision traceability and reproducibility of the reported baseline comparisons, vanilla ODE-SR baseline, and corrected b2-only fixed-structure bootstrap analysis. Raw confidential experimental data are not included unless explicitly intended for repository sharing.",
            "",
            "| Revision evidence | Code folder |",
            "|---|---|",
            "| Static baseline comparison | `code/static_baselines_24train6test/` |",
            "| DKC-SR vs baseline comparison | `code/dkcsr_baseline_comparison/` |",
            "| Vanilla ODE-SR baseline | `code/vanilla_odesr_baseline/` |",
            "| Corrected b2-only bootstrap | `code/bootstrap_b2_only/` |",
            "| Diagnostic plots and audits | `code/plotting_and_audits/` |",
            "",
            "### Code File Counts",
            "",
        ]
    )
    for category in sorted(code_counts):
        lines.append(f"- {category}: {code_counts[category]}")
    if not code_counts:
        lines.append("None.")

    lines.extend(["", "## Missing Files", ""])
    lines.extend(bullet_list(missing))
    lines.extend(["", "## Files Requiring Manual Review", ""])
    lines.extend(bullet_list(manual))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "This package is intended for manuscript revision traceability. It should not include raw confidential experimental data unless the repository is private and sharing is intentional.",
            "",
            "Obsolete and exploratory diagnostics, including all30 bootstrap diagnostics, 30train6test bootstrap diagnostics, old qscale3008 exploratory files, skipped bootstrap reports, and bootstrap versions where the numerator coefficient was refitted, were not copied automatically.",
        ]
    )
    (PACKAGE / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validation_summary(rows: list[dict[str, str]]) -> None:
    copied = copied_rows(rows)
    missing = missing_rows(rows)
    manual = manual_rows(rows)
    counts = Counter(row["category"] for row in copied)

    print(f"Output folder exists: {PACKAGE.exists()} ({PACKAGE})")
    print("Copied files by category:")
    for category in sorted(counts):
        print(f"  {category}: {counts[category]}")
    if not counts:
        print("  none")

    print("Missing expected files:")
    if missing:
        for row in missing:
            print(f"  {row['relative_path']}")
    else:
        print("  none")

    print("Manual-review candidate files:")
    if manual:
        for row in manual:
            print(f"  {row['source_path']}")
    else:
        print("  none")

    forbidden_hits = [
        row["relative_path"]
        for row in copied
        if any(part in row["relative_path"] for part in COPY_PATH_FORBIDDEN_PARTS)
    ]
    print("Forbidden copied path substring check:")
    if forbidden_hits:
        for hit in forbidden_hits:
            print(f"  FAIL {hit}")
    else:
        print("  PASS: no copied file path contains SKIPPED, all30, 30train, or 24train_vs_30train")

    forbidden_dirs = [
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*")
        if path.is_dir() and path.name in {"__pycache__", ".ipynb_checkpoints"}
    ]
    print("Cache/checkpoint folder check:")
    if forbidden_dirs:
        for hit in forbidden_dirs:
            print(f"  FAIL {hit}")
    else:
        print("  PASS: no __pycache__ or .ipynb_checkpoints folders copied")

    print("Final package tree:")
    for line in package_tree():
        print(line)


def code_validation_summary(code_rows: list[dict[str, str]]) -> None:
    copied = [row for row in code_rows if row["status"] == "copied"]
    manual = [row for row in code_rows if row["status"] == "manual_review"]
    counts = Counter(row["code_category"] for row in copied)

    print("Code files copied by category:")
    for category in sorted(counts):
        print(f"  {category}: {counts[category]}")
    if not counts:
        print("  none")

    print("Code files requiring manual review:")
    if manual:
        for row in manual:
            print(f"  {row['source_path']}")
    else:
        print("  none")

    forbidden_hits = [
        row["relative_path"]
        for row in copied
        if any(part in row["relative_path"] for part in [*COPY_PATH_FORBIDDEN_PARTS, "obsolete"])
    ]
    print("Copied code path excluded-keyword check:")
    if forbidden_hits:
        for hit in forbidden_hits:
            print(f"  FAIL {hit}")
    else:
        print("  PASS: no copied code path contains SKIPPED, all30, 30train, 24train_vs_30train, or obsolete")

    print("Final code tree:")
    code_root = PACKAGE / "code"
    for path in sorted(code_root.rglob("*"), key=lambda p: p.relative_to(code_root).as_posix()):
        rel_path = path.relative_to(code_root)
        depth = len(rel_path.parts) - 1
        prefix = "  " * depth
        suffix = "/" if path.is_dir() else ""
        print(f"{prefix}- {rel_path.name}{suffix}")


def main() -> None:
    rows: list[dict[str, str]] = []
    code_rows: list[dict[str, str]] = []
    ensure_clean_package()
    copy_expected_files(rows)
    copy_bootstrap_csvs(rows)
    add_manual_review_candidates(rows)
    copy_code_files(code_rows)
    add_code_manual_review_candidates(code_rows)
    write_manifest(rows)
    write_code_manifest(code_rows)
    write_readme(rows, code_rows)
    validation_summary(rows)
    code_validation_summary(code_rows)


if __name__ == "__main__":
    main()
