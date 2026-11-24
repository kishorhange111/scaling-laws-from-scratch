# scaling-laws-from-scratch

Chinchilla-style scaling laws from scratch: byte-level GPTs trained on the Python standard library. The experiment runs IsoFLOP sweeps, fits the compute-optimal N\*(C) and D\*(C) (approach 2), and fits a parametric L(N, D) = E + A/N^α + B/D^β (approach 3).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kishorhange111/scaling-laws-from-scratch/blob/main/notebooks/colab_sweep.ipynb)

## Setup

- **Data:** `<python>/Lib/**/*.py`, byte-level (vocab 256), split **by file** with 5% held out, so there's no leakage between near-duplicate lines. It is already present wherever Python is installed, so nothing needs downloading. The corpus depends on the Python version, so every run records a fingerprint (version, byte counts, crc32), and results are only compared within one fingerprint.
- **Model:** pre-LN GPT with causal SDPA, GELU MLP and tied embeddings. N counts non-embedding parameters (12·L·d²), and C = 6·N·D.
- **Grid:** 11 shapes from 3k to 2.6M parameters, with 6 shapes per budget centred on the running estimate of N\*. Each run trains with AdamW, warmup plus cosine to 10%, and a width-scaled learning rate.

## What the pilots showed (CPU)

**Batch size decides whether the IsoFLOP curve has a minimum at all.** At C = 1e12 FLOPs:

| model N | 4,096-token batches (`results/ablation_b32_t128`) | 1,024-token batches (`results/cpu_pilot_b16_t64`) |
|---|---|---|
| 3k | — | 1.964 (53k steps) |
| 7k | — | 1.789 |
| 12k | **1.783** (3.3k steps) | 1.692 |
| 25k | 1.918 | **1.627** (6.6k steps) |
| 55k | 2.063 | 1.656 |
| 147k | 2.313 (275 steps) | 1.906 (1.1k steps) |

Validation loss is in nats per byte.

- **With 4,096-token batches the curve falls all the way to the smallest model.** The larger models got only 275–735 optimizer steps, so they were step-starved rather than data-starved, and the "optimum" sat on the grid edge.
- **With 1,024-token batches there is a clear interior minimum** at N ≈ 25k, and every model does better at equal compute.
- **At small compute the batch must sit below the critical batch size.** Otherwise the fit measures optimisation steps, not the N/D trade-off, and an IsoFLOP fit taken from the first grid would have been meaningless.

## Full sweep (Colab GPU)

The full sweep covers 5 budgets (1e12–1e14) × 6 sizes, and the notebook (badge above) runs and fits it on a Colab T4. Results and the fitted exponents will be added here from that run. Nothing is extrapolated from the pilots.

## Run

```
pip install -e ".[dev]"
pytest -q
python -m scaling.cli plan --budgets 1e12 3e12 1e13 3e13 1e14 --max-tokens 3e8
python -m scaling.cli run  --budgets 1e12 3e12 --device auto --workers 2 --out results
python -m scaling.cli fit  --out results       # fits.json, fits.md, scaling.png
```
