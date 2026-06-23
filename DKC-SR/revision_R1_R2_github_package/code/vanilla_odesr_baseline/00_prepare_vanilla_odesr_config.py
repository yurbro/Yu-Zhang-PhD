from __future__ import annotations

from _vanilla_common import CONFIG_DIR, Q_SCALE, REPORT_DIR, default_config_dict, ensure_dirs, rel, write_json


def main() -> None:
    ensure_dirs()
    cfg = default_config_dict()
    config_path = CONFIG_DIR / "vanilla_odesr_config.json"
    write_json(config_path, cfg)
    lines = [
        "# Vanilla ODE-SR Configuration",
        "",
        f"- q_scale: `{Q_SCALE}`.",
        "- q_scale is fixed by this revision-validation task; artifact cfg q_scale is not read or used.",
        "- Search budget: 5 seeds `[0, 1, 2, 3, 4]`, population 300, 30 generations, Hall of Fame size 20.",
        "- Dataset: same canonical 24 train + 6 test split.",
        "- Excluded rows are not used because the canonical split files are used directly: `S10`, `Opt-2`, `Opt-4`, `Opt-6`, `Opt-7`, `Opt-10`.",
        "",
        "## Retained Components",
        "",
        *[f"- {item}" for item in cfg["retained_components"]],
        "",
        "## Removed Domain-Knowledge Components",
        "",
        *[f"- {item}" for item in cfg["removed_domain_knowledge_components"]],
        "",
        "## Primitive Set",
        "",
        "- `add`, `sub`, `mul`, protected `div`, `pow2`.",
        "- Optional nonlinear primitives were not added in this first vanilla run.",
        "- `softplus` is absent: `True`.",
        "",
        "## Notes",
        "",
        "- No variable-inclusion constraint is imposed during fitting; dependence on Q/C1/C2/C3 is audited after fitting.",
        "- Nonnegative RHS, nonpositive df/dQ, and nonnegative prediction clamping are not used.",
        "",
        "## Outputs",
        "",
        f"- `{rel(config_path)}`",
    ]
    (REPORT_DIR / "00_vanilla_odesr_config_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {rel(config_path)}")


if __name__ == "__main__":
    main()
