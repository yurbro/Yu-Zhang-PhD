# PermeationNet-IBU-Franz-5pct (v1)

A curated, evidence-linked dataset for **in vitro skin permeation/release** studies of **ibuprofen (IBU) 5% w/w** measured using **Franz diffusion cells**. Records are extracted from full-text PDFs using a multi-stage LLM-assisted pipeline (retrieval → triage → evidence indexing → extraction → verification), with an additional figure-digitization branch.

## What’s inside (v1 snapshot)

- **Total records:** 26
  - **Text-extracted:** 22
  - **Figure-digitized:** 4
- **Endpoints (Q_final@t_last):**
  - `amount_per_area`: 21 records (standardized where possible to µg/cm²)
  - `amount_total`: 5 records (reported as total amount; unit varies)
- **Study types:** IVPT and IVRT are allowed (see `study_type`)
- **Barrier types:** skin and synthetic membranes are both allowed (see `barrier_category`, `barrier_name`)
- **Scope restriction:** this release only includes **IBU 5% w/w** formulations (dataset is intentionally small/clean to validate the pipeline).

> Note: the figure branch often yields non-5% or non-comparable “drug loading” units in the source literature; v1 keeps only records that pass the 5% w/w confirmation rule.

## Inclusion criteria (v1)

1. Uses a **Franz diffusion cell** (static vertical or equivalent)
2. Reports **ibuprofen** as the active ingredient
3. Ibuprofen concentration is **5% w/w**
4. Contains a numeric **final-time endpoint**: cumulative permeation/release amount at the last sampling time, with time explicitly reported  
   - Flux/Jss may appear in the paper but are not required for this release.

## Files to publish together

Place these files in the same folder (or keep paths consistent):

- `PermeationNet_IBU_Franz_5pct_v1_merged.csv`  
  The dataset (26 rows; text+figure merged).
- `schema_v1.json`  
  Machine-readable schema (required/optional fields and types).
- `DATA_DICTIONARY.md`  
  Human-readable field descriptions and units.
- `normalisation_rules.md`  
  Unit normalization targets and preservation of raw fields.
- `inclusion_criteria.md`  
  Scope + inclusion rules used during curation.
- `barrier_vocab.yml`, `key_ingredients.yml`  
  Controlled vocabularies (barrier/material naming + ingredient normalization).
- `CITATION.cff`  
  Citation metadata for GitHub/Zenodo style releases.
- `CHANGELOG.md`  
  Release notes.

Optional but recommended for transparency/reproducibility:
- `step6_raw_records.jsonl`, `figure_digitized_curves.jsonl` (provenance-rich intermediates)

## Column conventions (high level)

- **Raw vs standardized units**
  - Raw endpoint is stored in `endpoint_value` + `endpoint_unit`.
  - If the endpoint is per-area and conversion is possible, a standardized value is stored in `endpoint_value_ug_cm2` (µg/cm²).
  - Time is represented as the original `endpoint_time_value` + `endpoint_time_unit`, and a normalized numeric hour value in `endpoint_time_h` when possible.
- **Evidence linkage**
  - `notes` may include extraction hints.
  - `confidence` is a model-provided extraction confidence (0–1), to be treated as a heuristic—not ground truth.
  - Records coming from figures are tagged by `source=figure` (and may have extra fields like curve identifiers).

## Known limitations (v1)

- **Small scale by design:** fixed 5% w/w makes the dataset clean but narrow; many ibuprofen formulations in the literature use other concentrations or “drug loading” units.
- **Open access bias:** full-text download is limited by OA availability; paywalled papers are underrepresented.
- **Heterogeneous reporting:** some studies report total amount, others amount per area; some omit area. v1 keeps both kinds but distinguishes them.
- **Figure digitization:** only figures classified as digitizable (clear axes/legend) were processed. Some endpoints are in supplementary material and are not included in v1.

## Reproducing the pipeline (code)

This release was produced by a scripted pipeline (names may match your repo):

- Step 1–3: corpus build + abstract triage
- Step 4: full-text download inventory
- Step 5: evidence indexing (route text/figure/supp)
- Step 6: text extraction + verification
- Step 7: figure triage + calibration + digitization + curve↔formulation alignment
- Step 7.5: merge text + figure into v1

## How to cite

See `CITATION.cff`. If you publish a paper describing the method/dataset, cite that paper and the dataset release.
