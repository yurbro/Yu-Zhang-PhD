# PermeationNet-IBU-Franz v1 — Inclusion Criteria

## Scope (v1 pilot)

- API: ibuprofen only
- Device: Franz diffusion cell only
- Study type: IVPT and IVRT
- Endpoint: Q_final at the last reported time (required)
- Optional endpoints: flux, J_ss (if reported)
- API concentration constraint (v1): 5% w/w only (or equivalent expressions)

## Hard inclusion requirements (record-level: paper × formulation × condition)

A record is included if ALL conditions below are satisfied:

1) API
- api_name must be "ibuprofen" (case-insensitive)
2) Device
- Must explicitly mention Franz diffusion cell in the paper/experiment description.
  Acceptable phrases include:
  - "Franz diffusion cell", "Franz cell"
  - "vertical Franz diffusion cell", "static Franz diffusion cell"
  - "Franz-type diffusion cell" (mark as `franz_confirmed=uncertain` unless methods confirm)
3) Study type
- Must be either:
  - IVPT (skin as barrier), or
  - IVRT (synthetic membrane as barrier)
- Must be labelled with:
  - study_type ∈ {IVPT, IVRT}
  - barrier_category ∈ {skin, synthetic_membrane}
4) Endpoint (required)
- Must provide Q_final at the last reported time:
  - endpoint_Q_final.value (numeric)
  - endpoint_Q_final.unit (must be parsable)
  - endpoint_Q_final.time_h (numeric)
- Q_final can be in a table, text, or supplementary material.
- If only a figure is provided with no extractable numbers, do NOT include in v1
  (put into "quarantine_figure_only" list for Paper 3).
5) API concentration constraint (required for v1)
- Must be 5% w/w, or an equivalent expression that can be unambiguously converted to 5% w/w.
  Accepted equivalents:
  - "5% w/w", "5 wt%", "5% wt/wt", "5% (w/w)"
  - "50 mg/g"  (= 5% w/w)
  - "0.05 g/g" (= 5% w/w)
  - "5 g per 100 g" (= 5% w/w)
- Tolerance rule (to handle rounding):
  - treat as target 5% w/w if parsed w/w% is within [4.8, 5.2]
- Exclude (or send to holdout) if:
  - only "5% w/v" is reported
  - only mg/mL is reported without density or a clear mass basis
  - "5%" is reported with no basis (w/w vs w/v unclear)
6) Minimum formulation information (required)
- Must list formulation ingredients (vehicle/excipients) as a list (names).
- Must include ibuprofen concentration (already required above).
- Ingredient concentrations:
  - allowed to be missing for some ingredients (B class),
  - but must be present for key ingredients to enter A class (see below).
7) Evidence traceability (required)
- Each record must include at least ONE evidence pointer:
  - evidence.source_type ∈ {Table, Text, Supplement}
  - evidence.locator (e.g., "Table 2", "Results paragraph", "Supp. Table S3")
  - evidence.snippet (1–3 sentences or a short table row transcription)

## A/B completeness classes (stored as `formulation_completeness_class`)

We use a key-ingredient whitelist defined in `key_ingredients.yml`.

A-class (Gold / modelling core):

- If a key ingredient appears in the formulation, its concentration MUST be parseable.
- Water can be represented as "qs to 100%" / "balance" and is treated as parseable.
- Ibuprofen concentration must be parseable (already required).

B-class (Coverage / noisy):

- Meets all hard inclusion requirements, but at least one appearing key ingredient
  has missing/unclear concentration.
- Still included, but mark missing flags and expect lower confidence.

## Holdout & quarantine policy

- holdout_not_5pct: ibuprofen Franz IVPT/IVRT records with non-5% w/w API concentration
- holdout_uncertain_basis: API concentration basis unclear (e.g., "5%")
- quarantine_figure_only: only figures available (no extractable endpoint numbers)
  These are stored separately for future expansion.

## What we export in v1

- PermeationNet_IBU_Franz_5pct_v1: only records satisfying all hard inclusion requirements above.
- Flux/Jss: included when available, never required.
