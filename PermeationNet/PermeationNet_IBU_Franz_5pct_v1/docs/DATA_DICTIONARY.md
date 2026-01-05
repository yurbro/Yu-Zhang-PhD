# Data dictionary — PermeationNet-IBU-Franz-5pct (v1)

This dictionary describes the **core columns** expected in the v1 release.  
Some releases may contain additional *pipeline/provenance* columns; keep them if useful, but treat them as optional.

Conventions:
- “Required” means required for a valid record in `schema_v1.json`.
- Units:
  - Standard per-area endpoint (when convertible): **µg/cm²** in `endpoint_value_ug_cm2`.
  - Time normalization: **hours** in `endpoint_time_h`.

## Core study metadata

| Column | Type | Required | Description |
|---|---:|:---:|---|
| doi | string | ✓ | DOI of the source publication. |
| title | string | ✓ | Paper title (as indexed). |
| record_idx | integer | ✓ | Record index within a paper (0-based or 1-based depending on extraction script). |
| source | string | ✓ | Evidence source: `text` or `figure`. |
| study_type | string | ✓ | `IVPT` or `IVRT` (in vitro permeation or release). |
| cell_type | string | ✓ | `Franz` (or a Franz-equivalent label if used). |
| dosage_form | string | ◻︎ | Dosage form when reported: gel/cream/solution/microemulsion/hydrogel/etc. |
| barrier_category | string | ✓ | High-level barrier type: `skin`, `synthetic_membrane`, or `both` (if the paper compares). |
| barrier_name | string | ◻︎ | Raw/normalized barrier name (e.g., porcine skin, human skin, nylon membrane). |

## API (active) information

| Column | Type | Required | Description |
|---|---:|:---:|---|
| api_name | string | ✓ | Active ingredient name. v1 is limited to `ibuprofen`. |
| api_conc_raw | string | ◻︎ | Raw concentration snippet from the paper (as extracted). |
| api_conc_value | number | ✓ | Parsed concentration numeric value. |
| api_conc_unit | string | ✓ | Unit for `api_conc_value` (e.g., `%`). |
| api_conc_basis | string | ✓ | Basis (e.g., `w/w`). |
| is_ibuprofen_5pct_w_w | boolean | ✓ | Whether the record passes the “IBU 5% w/w” criterion. |

## Endpoint (Q_final@t_last)

| Column | Type | Required | Description |
|---|---:|:---:|---|
| endpoint_type | string | ✓ | Endpoint kind, typically `amount_per_area` or `amount_total`. |
| endpoint_value | number | ✓ | Raw endpoint numeric value (as reported or digitized). |
| endpoint_unit | string | ✓ | Raw unit for `endpoint_value` (e.g., µg/cm², mg/cm², µg, mg). |
| endpoint_time_value | number | ✓ | Raw time value at which the endpoint is taken (e.g., 24). |
| endpoint_time_unit | string | ✓ | Raw time unit (e.g., `h`, `min`). |
| endpoint_time_h | number | ◻︎ | Time normalized to hours (float) when convertible. |
| endpoint_value_ug_cm2 | number | ◻︎ | Endpoint converted to µg/cm² when `endpoint_type=amount_per_area` and conversion is possible. |

## Provenance / QC (recommended)

| Column | Type | Required | Description |
|---|---:|:---:|---|
| needs_supp | boolean | ◻︎ | Flag indicating that key evidence likely sits in supplementary material. |
| confidence | number | ◻︎ | LLM-reported confidence (0–1). Use as a heuristic only. |
| notes | string | ◻︎ | Free-text notes, including extraction caveats or evidence hints. |

## Optional figure-specific fields (may appear)

If `source=figure`, you may also see columns like:
- `figure_id`, `page_number`, `subplot`
- `curve_id`, `curve_color`, `curve_label`
- `image_path` (local path during curation)
- `mapping_status` (e.g., vision-based alignment outcome)

These are useful for auditing but are not required for model training.

## Recommended minimal training view

For a first predictive model on v1, a minimal feature set is:

- Formulation inputs: (not fully standardized in v1; consider v2 to include full excipient vectors)
- Barrier: `barrier_category`, `barrier_name`
- Endpoint target: `endpoint_value_ug_cm2` when available, else `endpoint_value` + `endpoint_unit` + `endpoint_type`
- Time: `endpoint_time_h`
