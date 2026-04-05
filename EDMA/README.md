#  <p align="center">An early decision-making algorithm for accelerating topical drug formulation optimisation

This repository includes all of codes about EDMA for accelerating topical drug formulation optimisation.

❗❗❗ - Some places such as path may need you change and match your own local path, then the code might be run. Many thanks.

### Citation
    Zhang, Yu, et al. "An early decision-making algorithm for accelerating topical drug formulation optimisation." Computers & Chemical Engineering (2025): 109224.

    # 🧪 EDMA: Early Decision-Making Algorithm for Accelerating Topical Drug Formulation Optimisation

[![DOI](https://img.shields.io/badge/DOI-10.1016/j.compchemeng.2025.109224-blue)](https://doi.org/10.1016/j.compchemeng.2025.109224)
[![Journal](https://img.shields.io/badge/Journal-Computers%20%26%20Chemical%20Engineering-green)](https://www.elsevier.com/locate/compchemeng)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> **📄 Paper:** *An early decision-making algorithm for accelerating topical drug formulation optimisation*
>
> **✍️ Authors:** Yu Zhang, Yongrui Xiao, Dimitrios Tsaoulidis, Tao Chen\*
>
> **🏫 Affiliation:** School of Chemistry and Chemical Engineering, University of Surrey, Guildford GU2 7XH, UK
>
> **📅 Published in:** *Computers and Chemical Engineering*, Volume 201, 2025

---

## 🌟 Overview

**EDMA** is a data-driven algorithm designed to **terminate unpromising experiments early** during formulation optimisation, saving significant time and resources. It was developed for *in vitro permeation tests (IVPTs)* — experiments that typically run for **24 hours** per formulation — but the methodology is broadly applicable to any time-consuming batch experimental process.

The core idea: instead of waiting until the end of a lengthy experiment, EDMA predicts whether a formulation has the potential to outperform the current best, and makes a **go/no-go decision** at each sampling time point.

### 💡 Key Highlights

- 🔮 **Gaussian Process Regression (GPR)** for prediction with built-in uncertainty quantification
- 📊 **Probability of Exceedance (PoE)** as the decision metric — accounts for prediction uncertainty
- ⏱️ **Up to 96.4% Lead Rate** — experiments terminated early with 100% decision accuracy
- 💊 Validated on **ibuprofen-loaded poloxamer 407-based formulations**

---

## 🏗️ How It Works

```
┌─────────────────────────────────────────────────────┐
│                    EDMA Workflow                     │
│                                                     │
│  Historical Data (H)                                │
│        │                                            │
│        ▼                                            │
│  ┌──────────┐    ┌───────────┐    ┌──────────────┐  │
│  │ Build GPR│───▶│ Predict ŷ │───▶│ Calculate PoE│  │
│  └──────────┘    └───────────┘    └──────┬───────┘  │
│                                          │          │
│                                    PoE < T ?        │
│                                    /       \        │
│                                 Yes         No      │
│                                  │           │      │
│                              🛑 STOP    ✅ CONTINUE │
│                           (Save time!)  (Sample &   │
│                                          Update)    │
└─────────────────────────────────────────────────────┘
```

### Algorithm Steps

1. **Build a GPR model** using historical IVPT data (excipient concentrations → cumulative permeation)
2. **Predict** the final cumulative permeation for a new formulation at each sampling time point
3. **Calculate the PoE** — the probability that the new formulation exceeds the current best (`y_best`)
4. **Make a decision:**
   - If `PoE < T` (threshold) → **Stop** the experiment (formulation unlikely to improve)
   - Otherwise → **Continue** sampling and update the model
5. **Repeat** at each subsequent sampling time until "Stop" or the experiment ends

---

## 📐 Mathematical Framework

### Gaussian Process Regression

GPR provides both a **prediction mean** (ŷ\*) and **prediction variance** (σ²), enabling uncertainty-aware decision-making:

$$\hat{y}^* = \mathbf{k}^T(\mathbf{x}^*)\mathbf{C}^{-1}\mathbf{y}$$

$$\sigma^2_{\hat{y}^*} = C(\mathbf{x}^*, \mathbf{x}^*) - \mathbf{k}^T(\mathbf{x}^*)\mathbf{C}^{-1}\mathbf{k}(\mathbf{x}^*)$$

### Probability of Exceedance

$$PoE(\hat{y}(\mathbf{x}^*) \geq y_{best}) = 1 - \Phi(z)$$

where $z = (y_{best} - \hat{y}^*) / \sigma^2_{\hat{y}^*}$, and $\Phi(\cdot)$ is the standard normal CDF.

### Decision Rule

$$PoE < T \implies \text{Stop (terminate experiment)}$$

---

## 🧫 Case Study: Ibuprofen Formulation

The algorithm was applied to optimise **ibuprofen-loaded poloxamer 407-based gel formulations** for topical drug delivery.

### Formulation Components
| Component | Role |
|---|---|
| Ibuprofen (5% w/w) | Active Pharmaceutical Ingredient (API) |
| Poloxamer 407 | Gel base |
| Ethanol | Permeation enhancer |
| Propylene glycol | Co-solvent |
| Medium chain triglycerides (MCT) | Emollient |
| Ultra-pure water | Solvent |

### Experimental Setup
- **Membrane:** Strat-M® synthetic membrane
- **Apparatus:** Franz diffusion cells
- **Sampling times:** 1, 2, 3, 4, 6, 8, 22, 24, 26, 28 hours
- **Analysis:** HPLC for ibuprofen quantification
- **Replications:** 5 per formulation

### 📈 Results Summary

| Formulation | PoE (1st Iteration) | Decision | Actual Outcome |
|---|---|---|---|
| E-1 | 21.72% | Continue → Stop (3rd iter) | Below y_best ✅ |
| E-2 | 13.97% | **Stop** (1st iter) | Below y_best ✅ |
| E-3 | 54.48% | Continue | Above y_best ✅ |
| E-4 | 53.99% | Continue | Above y_best ✅ |
| E-5 | 48.50% | Continue | Above y_best ✅ |

> **🎯 Decision Accuracy: 100% | Type I Error: 0 | Type II Error: 0 | Overall Lead Rate: 96.4%**

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy scipy scikit-learn matplotlib bayesian-optimization
```

### Usage

```python
# 1. Load historical IVPT data
# 2. Train GPR model
# 3. Run EDMA on new formulations

# Example: Setting the decision threshold
from edma import EDMA

edma = EDMA(threshold=0.2)
edma.fit(X_train, y_train)

# Predict and decide for a new formulation
decision, poe = edma.predict_and_decide(x_new, y_best)
print(f"PoE: {poe:.2%} → Decision: {decision}")
```

### 📁 Repository Structure

```
EDMA/
├── README.md                # This file
├── data/                    # Historical IVPT dataset
├── models/                  # GPR model implementation
├── edma.py                  # Core EDMA algorithm
├── threshold_setting.py     # Threshold determination
├── utils.py                 # Utility functions
└── results/                 # Figures and result tables
```

> ⚠️ *Please refer to the actual repository contents for the precise file structure.*

---

## ⚙️ Threshold Selection

The decision threshold **T = 0.2** was determined through a systematic process:

1. Split 20 historical datasets into training (12) and test (8) groups
2. Evaluate PoE at thresholds from 0.1 to 1.0
3. Select based on **priority**: Accuracy → Type I error → Type II error → Lead Rate
4. Repeat across 10 random splits and average

| Threshold | Avg. Accuracy | Avg. Type I | Avg. Type II | Avg. Lead Rate |
|---|---|---|---|---|
| 0.1 | 81.3% | 3.1% | 18.8% | 84% |
| **0.2** | **91.7%** | **0%** | **8.3%** | **76%** |
| 0.3 | 83.3% | 8.3% | 8.3% | 81% |

---

## 🔗 Dependencies

- **GPR & Bayesian Optimisation:** [BayesianOptimization](https://github.com/bayesian-optimization/BayesianOptimization)
- **Scientific Computing:** NumPy, SciPy
- **Visualisation:** Matplotlib

---

## 📚 Citation

If you use this code or find the work helpful, please cite our paper:

```bibtex
@article{ZHANG2025109224,
  title     = {An early decision-making algorithm for accelerating topical drug formulation optimisation},
  author    = {Zhang, Yu and Xiao, Yongrui and Tsaoulidis, Dimitrios and Chen, Tao},
  journal   = {Computers and Chemical Engineering},
  volume    = {201},
  pages     = {109224},
  year      = {2025},
  publisher = {Elsevier},
  doi       = {10.1016/j.compchemeng.2025.109224}
}
```

---

## 🤝 Related Work

- Xiao, Y. et al. (2025). *Topical drug formulation for enhanced permeation: a comparison of Bayesian optimisation and response surface methodology.* International Journal of Pharmaceutics, 672, 125306. [DOI](https://doi.org/10.1016/j.ijpharm.2025.125306)

---

## 📬 Contact

For questions, data requests, or collaboration:

- **Yu Zhang** — [yu.zhang@surrey.ac.uk](mailto:yu.zhang@surrey.ac.uk) | [LinkedIn](https://www.linkedin.com/in/yurbro/)
- **Prof. Tao Chen** (Corresponding Author) — University of Surrey

---

## 🏛️ Acknowledgements

Yu Zhang's PhD is supported by the **China Scholarship Council** in partnership with the **University of Surrey** (Grant No. 202306440060).

---

<p align="center">
  <i>Made with ❤️ at the University of Surrey</i>
</p>
