import re
import pandas as pd
from pathlib import Path

"""
我们做一个非常宽松的 stage-1 filter:
只要标题+摘要里出现 ibuprofen 且出现一些 “in vitro/ permeation / release / diffusion / topical / transdermal / Franz / membrane / skin” 相关词，就放行。
TODO: 需要加上去除重复文献的筛查功能
"""

IN_PATH = "GenAI/data/corpus_ibuprofen.csv"
OUT_DIR = Path("GenAI/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 宽松关键词：不要求 Franz
PASS_PATTERNS = [
    r"\bin vitro\b",
    r"\bpermeat",          # permeate/permeation/permeability
    r"\bdiffus",           # diffusion/diffusing
    r"\brelease\b",
    r"\btransdermal\b",
    r"\btopical\b",
    r"\bskin\b",
    r"\bmembrane\b",
    r"\bfranz\b",
    r"\bdiffusion cell\b",
]

# 可选：明显“非我们方向”的排除词（先别太 aggressive）
EXCLUDE_PATTERNS = [
    r"\boral\b",
    r"\btablet\b",
    r"\bcapsule\b",
    r"\bplasma\b",
    r"\bpharmacokinet",    # pharmacokinetics
    r"\bin vivo\b",
]

def has_any(text: str, patterns) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    text = (df["title"].fillna("") + " " + df["abstract"].fillna("")).astype(str)

    # 必须提到 ibuprofen（通常 corpus 已经满足，但再保险）
    has_ibu = text.str.contains(r"\bibuprofen\b", case=False, regex=True)

    pass_kw = text.apply(lambda t: has_any(t, PASS_PATTERNS))
    exclude_kw = text.apply(lambda t: has_any(t, EXCLUDE_PATTERNS))

    # 规则：包含关键词 + 不命中排除词（排除词你也可以暂时不启用）
    stage1_pass = has_ibu & pass_kw & (~exclude_kw)

    df_pass = df[stage1_pass].copy()
    df_fail = df[~stage1_pass].copy()

    df_pass.to_csv(OUT_DIR / "screen_stage1_pass.csv", index=False, encoding="utf-8-sig")
    df_fail.to_csv(OUT_DIR / "screen_stage1_fail.csv", index=False, encoding="utf-8-sig")

    print("Stage-1 pass:", len(df_pass))
    print("Stage-1 fail:", len(df_fail))
