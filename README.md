# PODEX: Population Diversity and Extinction Analysis

PODEX is a collection of Python utilities for computing diversity measures, Shapley values, and extinction‑aware prioritization rankings for conservation or population‑genomic applications.

This repository includes:
- Deterministic and Monte‑Carlo pooled heterozygosity (HET) calculators.
- Deterministic and Monte‑Carlo SSD (Split System Diversity) calculators.
- Exact and Monte‑Carlo Shapley value implementations.
- Extinction‑aware Shapley values.
- Expected Diversity Gain ranking (baseline + iterative iHED‑style ranking).
- Plotting helper for cumulative expected diversity gain.

---
## 📦 Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/nabhari/podex.git
cd podex
pip install -r requirements.txt
```

---
## 📂 Repository Structure
```
podex/
│
├── diversity.py        # Main module containing all functionality
├── test_example.csv    # Example toy dataset
├── README.md           # This file
└── …
```

---
## ✨ Features
### 1. Diversity Measures
- **Het_Pooling_with_pop_size**: deterministic pooled heterozygosity.
- **Het_Pooling_mc_with_extinction**: heterozygosity under extinction via Monte‑Carlo.
- **SSD_Pooling_with_pop_size**: weighted SSD score.
- **SSD_Pooling_mc_with_extinction**: Monte‑Carlo SSD under extinction.

### 2. Shapley Values
- **Exact Shapley** (for ≤6 populations).
- **Monte‑Carlo Shapley**.
- **Extinction‑aware Shapley**, returning unconditional or conditional values. Optionally, if return_conditional=True, you also get the contribution of a population given that it survives.
### 3. Expected Diversity Gain Ranking
Implements:
- **Baseline W(i)** ranking (protect one population at a time).
- **Iterative iHED‑style ranking** (stepwise protection).
- *Option*: `only_baseline=True` to return only the baseline ranking.

### 4. Plotting
- **plot_iterative_cumulative_gain**: cumulative iHED curve.

---
## 🧪 Example Usage
```python
from podex.diversity import (
    load_frequencies,
    expected_diversity_gain_ranking,
    plot_iterative_cumulative_gain
)

d = load_frequencies(pops_col_name="pops")
pop_size = [17, 27, 43]
ext_probs = [0.2, 0.8, 0.1]

baseline, iterative = expected_diversity_gain_ranking(
    freqs=d,
    pop_size=pop_size,
    ext_probs=ext_probs,
    mode="het_mc",
    n_samples=5000
)

plot_iterative_cumulative_gain(iterative)
```

---
## 📘 Citation Requirements
If you use this software in your work, please cite **both**:
1. The GitHub repository, and
2. Measuring genetic diversity across populations — Abhari N, Colijn C, Mooers A, Tupper P (2024) PLoS Comput Biol 20(12): e1012651. https://doi.org/10.1371/journal.pcbi.1012651

---
## 👩‍💻 Author
Maintained by **Niloufar Abhari**.

For questions or issues, open a GitHub issue.

