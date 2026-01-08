**Config LLM 设置环境变量**

```
setx GROQ_API_KEY "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"
```

OpenAI Platform

```My
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

setx OPENAI_API_KEY "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"
```

- Install the OpenAI Python SDK and execute the code below to generate a haiku for free using the gpt-5-nano model via the Responses API.

```
pip install openai
```

```
from openai import OpenAI

client = OpenAI(
  api_key="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"
)

response = client.responses.create(
  model="gpt-5-nano",
  input="write a haiku about ai",
  store=True,
)

print(response.output_text);

```



# 建候选论文池 `corpus_ibuprofen.csv`

### 目标

先拉一个“尽量不漏”的候选集（500–2000 篇都正常），字段只要 title/abstract/DOI/年份等即可。

### 怎么做（建议）

* 用 Google Scholar / PubMed/Europe PMC 搜关键词，把结果导出成 CSV。

* 检索式不要过窄，推荐至少两条并集：

### 产出

`data/corpus_ibuprofen.csv`  

建议字段：`source, title, authors, year, journal, doi, url, abstract`
Step 2：Stage-1 规则预筛（便宜）

-----------------------

### 目标

把明显无关的（体内、临床、纯药代等）先踢掉，减少后续 LLM 成本。

### 推荐规则（宽松即可）

通过条件：

* title/abstract 包含 `ibuprofen` **或** 明确为 topical/transdermal/gel/cream/ointment

* 且包含 `in vitro` 或 `permeation` 或 `release` 或 `diffusion` 或 `Franz`

> 这一步宁可“放进来”，不要漏。

### 产出

* `outputs/screen_stage1_pass.csv`

* `outputs/screen_stage1_fail.csv`
  
  

Step 3：Stage-2 LLM 精筛（title+abstract）
-------------------------------------

### 目标

让 LLM 判断“是否值得下载全文继续处理”，而不是直接判定最终纳入（最终纳入要靠全文）。

### 你让 LLM 输出什么（建议固定成 JSON）

* `priority`: high / medium / low（是否值得进入全文阶段）

* `likely_franz`: yes / no / uncertain（摘要看不出来就 uncertain）

* `likely_ivpt_or_ivrt`: IVPT / IVRT / both / uncertain

* `likely_has_numeric_endpoint`: yes / no / uncertain

* `likely_has_formulation_info`: yes / no / uncertain

* `notes`: 一句话说明为什么

* `what_to_check_in_fulltext`: 让它列 2–3 个要在全文确认的点（比如 Franz、5% w/w、Q_final 是否表格给出）

### 产出

* `outputs/screen_stage2_ranked.csv`（带 priority）

* `outputs/screened_for_fulltext.csv`（priority=high/medium）
  
  

Step 4：下载全文 + 建“证据索引” `evidence_index.csv`
------------------------------------------

### 目标

对 high/medium 的论文拿到 PDF/HTML/补充材料，并定位信息在哪里。

### 怎么做

* 下载 PDF（能 OA 就直接下载；不能就先记录 DOI/链接，后续手动获取）

* 对每篇全文做一个 evidence index：
  
  * Franz 是否明确写了（methods 哪段）
  
  * ibuprofen 是否 5% w/w（配方表/方法）
  
  * Q_final/flux/Jss 在表格/段落哪儿
  
  * 如果只有图（figure-only），标记出来给 Paper 3 用

### 产出

* `papers/` 文件夹（PDF等）

* `outputs/evidence_index.csv`
   
  
  ## 5.1 Evidence indexing
  
      No  108
      Yes 10

Step 5：结构化抽取（按 schema 输出 JSONL）
-------------------------------

### 目标

从全文证据片段抽出结构化记录，然后做硬过滤：

* Franz confirmed

* ibuprofen 5% w/w confirmed（或等价表达）

* Q_final@t_last 数值可得

### 产出

* `outputs/raw_extractions.jsonl`
  
  

## 

Step 6：Verifier + QC + A/B 分档 + v1 数据集导出
----------------------------------------

### 产出

* `outputs/PermeationNet_IBU_Franz_5pct_v1.parquet`

* `outputs/qc_report.md`
