# Vanilla ODE-SR Configuration

- q_scale: `3008.198194823261`.
- q_scale is fixed by this revision-validation task; artifact cfg q_scale is not read or used.
- Search budget: 5 seeds `[0, 1, 2, 3, 4]`, population 300, 30 generations, Hall of Fame size 20.
- Dataset: same canonical 24 train + 6 test split.
- Excluded rows are not used because the canonical split files are used directly: `S10`, `Opt-2`, `Opt-4`, `Opt-6`, `Opt-7`, `Opt-10`.

## Retained Components

- Q, C1, C2, C3 as available variables
- ODE replay framework
- same 24 train + 6 held-out test split
- fixed q_scale = 3008.198194823261
- protected division
- finite-value failure handling
- RK4 integration with substeps
- complexity penalty
- tree-length limit

## Removed Domain-Knowledge Components

- softplus primitive
- hard nonnegative RHS
- hard nonpositive df/dQ
- formulation-gradient sign prior
- gradient magnitude/effect filters
- Q-sensitivity penalty
- denominator-Q physical penalty
- nonnegative release prediction clamp

## Primitive Set

- `add`, `sub`, `mul`, protected `div`, `pow2`.
- Optional nonlinear primitives were not added in this first vanilla run.
- `softplus` is absent: `True`.

## Notes

- No variable-inclusion constraint is imposed during fitting; dependence on Q/C1/C2/C3 is audited after fitting.
- Nonnegative RHS, nonpositive df/dQ, and nonnegative prediction clamping are not used.

## Outputs

- `revision_validation_vanilla_odesr_24train6test/config/vanilla_odesr_config.json`
