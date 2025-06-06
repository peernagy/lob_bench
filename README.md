# LOB‑Bench

*Benchmarking generative models for **Limit Order Book** data (LOBSTER format)*

---

## Table of Contents
1. [Why LOB‑Bench?](#why)
2. [Installation](#installation)
3. [Quick Start](#quickstart)
4. [Benchmarking Workflow](#workflow)
    * [Data Loader](#dataloader)
    * [Score Functions](#scores)
    * [Distance Metrics](#metrics)
    * [Impact Response](#impact)
5. [Extending the Library](#extend)
6. [Citing](#citing)
7. [Resources](#resources)

---

## <a id="why"></a> Why LOB‑Bench?
*   Quantitatively compares **real** and **generated** LOB sequences.
*   Provides ready‑made 1‑D score functions and distance metrics (L1, Wasserstein‑1).
*   Measures *conditional* distributions and *impact response* curves to reveal model drift.
*   Fully open‑source and easy to extend with custom scores or metrics.

---

## <a id="installation"></a> Installation
```bash
pip install lob_bench
```
Requires Python ≥ 3.9 and the usual scientific‑Python stack (`numpy`, `pandas`, `scipy`, `matplotlib`).

---

## <a id="quickstart"></a> Quick Start
```python
from lob_bench import data_loading, scoring, impact

# 1  Load data ──────────────
loader = data_loading.Simple_Loader(
    cond_path    = "./data/seed_sequences",   # optional
    generated_path = "./data/generated_sequences",
    real_path      = "./data/real_sequences"
)

# 2  Benchmark distributions ─
scores = {
    "Spread"          : {"fn": scoring.spread},
    "Interarrival"    : {"fn": scoring.interarrival, "Discrete": False},
    "Imbalance"       : {"fn": scoring.imbalance},
    # … add or replace as you like
}

metrics = {
    "L1"         : scoring.l1_distance,
    "Wasserstein": scoring.wasserstein_distance,
}

results = scoring.run_benchmark(loader, scores=scores, metrics=metrics)

# results is a tuple:
# A  distances + bootstrap CIs
# B  raw score values + bin indices
# C  plotting helpers

# 3  Impact response ─────────
impact_curves, impact_score = impact.impact_compare(loader)
```
See `tutorial.ipynb` for a complete walk‑through.

---

## <a id="workflow"></a> Benchmarking Workflow
### <a id="dataloader"></a> 1. Data Loader
`Simple_Loader` expects three folders, each containing the **same number** of CSV files in LOBSTER format:
* **conditioning sequences** (optional)
* **generated sequences** (model output)
* **real sequences** (ground truth)

### <a id="scores"></a> 2. Score Functions
Built‑in (all 1‑D):
| Category | Name | Description |
|----------|------|-------------|
| Price | `Spread` | Best ask − best bid |
| Time & Flow | `Interarrival` | Time between successive orders (ms) |
| Volume | `AskVolume` / `BidVolume` | Volume on first *n* levels (default 10) |
| Imbalance | `OrderbookImbalance` | Volume difference between best bid/ask |
| Depth | `LimitDepth`, `CancelDepth` | Distance of orders from mid‑price |
| Level | `LimitLevel`, `CancelLevel` | Price level index (1…n) |
| Durations | `TimeToCancel` | Submit → first cancel / modify |

Conditional variants are easy—provide nested dicts with `eval` and `cond` keys.

### <a id="metrics"></a> 3. Distance Metrics
Two metrics ship by default:
* **L1 (Total Variation)** – bounded in [0, 1].
* **Wasserstein‑1 (Earth Mover)** – mean‑variance normalised.

Add your own by passing a callable with signature `(real_values, gen_values) -> float`.

### <a id="impact"></a> 4. Impact Response
`impact_compare(loader)` computes six response curves (MO₀/₁, LO₀/₁, CA₀/₁) following [Eisler *et al.* 2012]. The mean absolute difference across lags gives a single impact score.

---

## <a id="extend"></a> Extending the Library
1. **Custom score** – create a function `my_score(messages, books) -> ndarray` and add it to the `scores` dict.
2. **Custom metric** – add a function to the `metrics` dict.
3. **New loader** – subclass `BaseLoader` for bespoke formats.
Pull requests are welcome!

---

## <a id="citing"></a> Citing
If you use LOB‑Bench in academic work, please cite:
```bibtex
@misc{nagy2025lobbenchbenchmarkinggenerativeai,
  title   = {LOB-Bench: Benchmarking Generative AI for Finance -- an Application to Limit Order Book Data},
  author  = {Peer Nagy and Sascha Frey and Kang Li and Bidipta Sarkar and Svitlana Vyetrenko and Stefan Zohren and Ani Calinescu and Jakob Foerster},
  year    = {2025},
  eprint  = {2502.09172},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url     = {https://arxiv.org/abs/2502.09172}
}
```

---

## <a id="resources"></a> Resources
* 📘 **Paper:** <https://arxiv.org/abs/2502.09172>
* 🔗 **Project site:** <https://lobbench.github.io/>
* 🐙 **Source code:** <https://github.com/peernagy/lob_bench>

---

Licensed under **MIT**.
