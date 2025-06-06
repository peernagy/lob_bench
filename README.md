# LOB‑Bench

*Benchmarking generative models for **Limit Order Book** data*

---

## Table of Contents
1. [Why LOB‑Bench?](#why)
2. [Library Architecture](#arch)
3. [Installation](#installation)
4. [Quick Start](#quickstart)
5. [Benchmarking Workflow](#workflow)
6. [Extending](#extend)
7. [Citing](#citing)
8. [Resources](#resources)

---

## <a id="why"></a> Why LOB‑Bench?
*   Quantitatively compares **real** and **generated** LOB sequences.
*   Ready‑made score functions, distance metrics, and impact‑response analysis.
*   Works out‑of‑the‑box with LOBSTER CSVs, yet fully extensible.

---

## <a id="arch"></a> Library Architecture

### I. Distributional Evaluation
For every sequence the library computes **1‑D scores** and measures the distance between
real and generated distributions.

*Implemented distance metrics*
* **L1 / Total Variation**
* **Wasserstein‑1**

*Implemented score functions*
| # | Score | Description |
|---|-------|-------------|
| i. | **Spread** | Best ask − best bid price |
| ii. | **Interarrival time** | Time between two successive orders (ms) |
| iii. | **Orderbook imbalance** | Volume difference at best bid/ask |
| iv. | **Time to cancel** | Order submission → first cancel/modify |
| v./vi. | **Ask/Bid volume (n levels)** | Volume on each side, first *n* levels (default 10) |
| vii./viii. | **Limit‑order depths** | Distance of new limit orders from mid‑price |
| ix./x. | **Limit‑order levels** | Price level index (1…*n*) of new limits |
| xi./xii. | **Cancel‑order depths** | Distance of cancellations from mid‑price |
| xiii./xiv. | **Cancel‑order levels** | Level index (1…*n*) of cancellations |

*Conditional score functions*
| # | Conditional metric |
|---|-------------------|
| i. | Ask‑volume (level 1) **conditional on spread** |
| ii. | Spread **conditional on hour‑of‑day** |
| iii. | Spread **conditional on volatility** (std. dev. of returns) |

### II. Impact‑Response Evaluation
The price‑response function is calculated for six event types:
| Code | Event | Mid‑price change? |
|------|-------|-------------------|
| `MO_0` | Market order | ✗ |
| `MO_1` | Market order | ✓ |
| `LO_0` | Limit order  | ✗ |
| `LO_1` | Limit order  | ✓ |
| `CA_0` | Cancel       | ✗ |
| `CA_1` | Cancel       | ✓ |

For each event type the library plots the response curve and reports the mean
absolute deviation between real and generated curves.

---

## <a id="installation"></a> Installation
```bash
pip install lob_bench
```
Requires Python ≥ 3.9 plus `numpy`, `pandas`, `scipy`, `matplotlib`.

---

## <a id="quickstart"></a> Quick Start
```python
from lob_bench import data_loading, scoring, impact

# 1  Load data ─────────────────────────
loader = data_loading.Simple_Loader(
    cond_path     = "./data/seed_sequences",   # optional
    generated_path = "./data/generated_sequences",
    real_path      = "./data/real_sequences"
)
# Each folder must contain the **same number** of LOBSTER‑format CSVs.

# 2  Define scores & metrics ───────────
score_cfg = {
    "Spread"       : {"fn": scoring.spread},
    "Interarrival" : {"fn": scoring.interarrival, "Discrete": False},
    "Imbalance"    : {"fn": scoring.imbalance},
    # Conditional example
    "AskVol|Spread": {
        "eval": {"fn": scoring.ask_volume_lvl1},
        "cond": {"fn": scoring.spread}
    },
}

metric_cfg = {
    "L1"         : scoring.l1_distance,
    "Wasserstein": scoring.wasserstein_distance,
}

# 3  Run benchmark ─────────────────────
results = scoring.run_benchmark(loader, score_cfg, metric_cfg)
# returns (distances, raw_scores, plotting_helpers)

# 4  Impact response ───────────────────
impact_curves, impact_score = impact.impact_compare(loader)
```
See `tutorial.ipynb` for a full walk‑through.

---

## <a id="workflow"></a> Benchmarking Workflow
1. **Prepare data** – three equal‑length CSV folders (conditioning, generated, real).
2. **Define score & metric dicts** – or import defaults from `lob_bench.defaults`.
3. **Call `run_benchmark`** – get distances, bootstrap CIs, and plotting helpers.
4. **Call `impact_compare`** – analyse six event‑type response curves.

---

## <a id="extend"></a> Extending
* **Custom score** – implement `my_score(messages, books) -> ndarray` and add to the score dict.
* **Custom metric** – function `(real_vals, gen_vals) -> float`, add to metric dict.
* **Custom loader** – subclass `BaseLoader` for alternative storage formats.

Pull requests are welcome 🎉.

---

## <a id="citing"></a> Citing
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

## <a id="resources"></a> Resources
* 📄 **Paper:** <https://arxiv.org/abs/2502.09172>
* 🌐 **Project site:** <https://lobbench.github.io/>
* 🐙 **Source code:** <https://github.com/peernagy/lob_bench>

---

Licensed under **MIT**.
