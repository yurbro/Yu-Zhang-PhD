import os
import re
import time
import json
import hashlib
from pathlib import Path

import pandas as pd
import requests

IN_PATH = "GenAI/outputs/fulltext_inventory.csv"   
OUT_PATH = "GenAI/outputs/fulltext_inventory.csv"  # 原地更新（断点续跑）
PDF_DIR = Path("GenAI/papers/pdf")
PDF_DIR.mkdir(parents=True, exist_ok=True)

# ====== 配置 ======
TIMEOUT = 40
SLEEP_BETWEEN = 0.4
CHECKPOINT_EVERY = 10

# Unpaywall 需要一个 email（官方要求用于联系/限流识别）
UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "").strip()  # 建议设成环境变量
# 你也可以直接写死（不推荐）
# UNPAYWALL_EMAIL = "your_email@surrey.ac.uk"

UA = {
    "User-Agent": "PermeationNet/0.1 (OA downloader; academic use)"
}

def safe_filename(s: str, max_len: int = 120) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"https?://", "", s)
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = s.strip("_")
    if not s:
        s = "unknown"
    return s[:max_len]

def make_id(doi: str, title: str, fallback: str = "") -> str:
    base = (doi or "").strip().lower()
    if not base:
        base = (title or "").strip().lower()
    if not base:
        base = fallback
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    return h

def download_file(url: str, out_path: Path) -> bool:
    r = requests.get(url, headers=UA, timeout=TIMEOUT, stream=True, allow_redirects=True)
    if r.status_code != 200:
        return False
    ctype = (r.headers.get("Content-Type") or "").lower()
    # 某些站点 Content-Type 不标准，但如果内容以 %PDF 开头也行
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        first_chunk = True
        for chunk in r.iter_content(chunk_size=1024 * 64):
            if not chunk:
                continue
            if first_chunk:
                first_chunk = False
            f.write(chunk)

    # 简单检查是否像 PDF
    with open(out_path, "rb") as f:
        head = f.read(5)
    if head != b"%PDF-":
        # 不是 PDF，删掉
        try:
            out_path.unlink()
        except Exception:
            pass
        return False
    return True

def is_pdf_url(url: str) -> bool:
    if not url:
        return False
    u = url.lower()
    return u.endswith(".pdf") or "pdf" in u

def get_pdf_from_unpaywall(doi: str) -> str | None:
    if not doi or not UNPAYWALL_EMAIL:
        return None
    api = f"https://api.unpaywall.org/v2/{doi}"
    params = {"email": UNPAYWALL_EMAIL}
    r = requests.get(api, params=params, headers=UA, timeout=TIMEOUT)
    if r.status_code != 200:
        return None
    data = r.json()
    loc = data.get("best_oa_location") or {}
    pdf = loc.get("url_for_pdf")
    return pdf

def get_pdf_from_epmc(doi: str = "", pmid: str = "") -> str | None:
    # Europe PMC search
    base = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    if doi:
        q = f'DOI:"{doi}"'
    elif pmid:
        q = f'EXT_ID:"{pmid}"'
    else:
        return None

    params = {"query": q, "format": "json", "pageSize": 1, "resultType": "core"}
    r = requests.get(base, params=params, headers=UA, timeout=TIMEOUT)
    if r.status_code != 200:
        return None
    data = r.json()
    res = data.get("resultList", {}).get("result", [])
    if not res:
        return None
    item = res[0]
    ft = item.get("fullTextUrlList", {}).get("fullTextUrl", []) or []
    # 优先 PDF
    for x in ft:
        url = x.get("url")
        style = (x.get("documentStyle") or "").lower()
        if url and ("pdf" in style or url.lower().endswith(".pdf")):
            return url
    # 次选：有些给了 html，只能先返回 None（我们只下 pdf）
    return None

def main():
    df = pd.read_csv(IN_PATH)
    # 把NaN统一成空字符串，并把“status==ok”以外都强制再跑一遍
    df["pdf_status"] = df["pdf_status"].fillna("").astype(str)
    df["pdf_path"] = df["pdf_path"].fillna("").astype(str)

    # 确保字段存在
    for c in ["pdf_status","pdf_path","supp_status","supp_path","notes"]:
        if c not in df.columns:
            df[c] = ""

    total = len(df)
    changed = 0

    for idx, row in df.iterrows():
        status = str(row.get("pdf_status") or "").strip().lower()
        if status == "ok" and str(row.get("pdf_path") or "").strip():
            continue  # 已下载

        doi = str(row.get("doi") or "").strip().lower()
        pmid = str(row.get("pmid") or "").strip()
        title = str(row.get("title") or "").strip()
        url0 = str(row.get("url") or "").strip()

        rec_id = make_id(doi, title, fallback=str(row.get("id") or idx))
        fname_base = safe_filename(doi if doi else title if title else f"rec_{rec_id}")
        pdf_path = PDF_DIR / f"{fname_base}__{rec_id}.pdf"

        pdf_url = None
        tried = []

        # 1) 现成链接
        if url0 and is_pdf_url(url0):
            pdf_url = url0
            tried.append("csv_url_pdf")

        # 2) Unpaywall
        if not pdf_url:
            u = get_pdf_from_unpaywall(doi)
            if u:
                pdf_url = u
                tried.append("unpaywall")
            else:
                tried.append("unpaywall_none")

        # 3) Europe PMC
        if not pdf_url:
            u = get_pdf_from_epmc(doi=doi, pmid=pmid)
            if u:
                pdf_url = u
                tried.append("epmc")
            else:
                tried.append("epmc_none")

        # 下载
        ok = False
        if pdf_url:
            try:
                ok = download_file(pdf_url, pdf_path)
            except Exception as e:
                ok = False
                df.at[idx, "pdf_status"] = f"error:{type(e).__name__}"
                df.at[idx, "notes"] = f"download exception; tried={tried}; url={pdf_url}"
        else:
            df.at[idx, "pdf_status"] = "paywalled_or_no_oa_link"
            df.at[idx, "notes"] = f"no pdf url found; tried={tried}"

        if ok:
            df.at[idx, "pdf_status"] = "ok"
            df.at[idx, "pdf_path"] = str(pdf_path)
            df.at[idx, "notes"] = f"downloaded; source={tried[-1] if tried else ''}; url={pdf_url}"
        else:
            # 如果是 403/404 或非 PDF，基本就是 paywalled 或链接失效
            if str(df.at[idx, "pdf_status"] or "").strip() == "":
                df.at[idx, "pdf_status"] = "paywalled_or_link_invalid"
                df.at[idx, "notes"] = f"pdf download failed or not pdf; tried={tried}; url={pdf_url}"

        changed += 1

        # checkpoint
        if changed % CHECKPOINT_EVERY == 0:
            df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
            print(f"Checkpoint: processed {idx+1}/{total}", flush=True)

        time.sleep(SLEEP_BETWEEN)

    df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print("Done. Updated:", OUT_PATH)
    print("pdf_status counts:")
    print(df["pdf_status"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
