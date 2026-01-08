# Normalisation Rules — PermeationNet IBU-Franz v1

## General principle

- Always store both:
  - raw text (e.g., "50 mg/g", "qs to 100%")
  - parsed/normalised fields when conversion is reliable
- If conversion is ambiguous, keep raw and set parsed fields to null + `basis="unknown"`.

## API concentration normalisation (target: 5% w/w)

Accept as 5% w/w if any of the following can be reliably parsed:

- 5% w/w, 5 wt%, 5% wt/wt, 5% (w/w)
- 50 mg/g  -> 5% w/w
- 0.05 g/g -> 5% w/w
- 5 g per 100 g -> 5% w/w

Tolerance:

- if parsed `w_w_percent` in [4.8, 5.2], set `is_target_5pct="yes"`.

Do not convert (v1):

- 5% w/v (keep as basis w/v; set is_target_5pct="no")
- mg/mL unless density or an explicit mass basis is provided
- "5%" with no basis -> `is_target_5pct="uncertain"` (holdout)

## Ingredient concentration normalisation

Store:

- conc_raw: original text
- conc_basis: one of {w/w, w/v, v/v, mg/mL, balance, unknown}

Water "qs/balance":

- If text includes: "qs", "q.s.", "qs to 100%", "balance", "make up to"
  then set:
  - conc_basis = "balance"
  - conc_raw = original phrase
  - conc_value/unit = null

## Endpoint normalisation

### Q_final

Target unit for modelling: ug/cm^2 (micrograms per square centimeter)

Common conversions:

- if unit is mg/cm^2: multiply by 1000 -> ug/cm^2
- if unit is ug/cm^2: keep
- if unit is amount without area (e.g., ug): only convert if area_cm2 is available:
  ug/cm^2 = ug / area_cm2
  If area is missing, keep raw and leave normalized_value null.

Time:

- store endpoint_time_h in hours
- if time in minutes: hours = min / 60

### flux / J_ss

Target unit: ug/cm^2/h

Common conversions:

- mg/cm^2/h -> *1000 = ug/cm^2/h

## Missingness policy

- Never guess missing concentrations.
- Use `missing_conc=true` when ingredient is listed but concentration is not reported.
- A/B class is determined AFTER normalisation using key-ingredient whitelist.

## Canonical naming (synonym mapping)

- Use `key_ingredients.yml` to map `name_raw` -> `name_canonical` (case-insensitive).
- Keep name_raw always for audit.
