"""IsoFLOP sweep runner (resumable, process-parallel) and analysis.

scaling run --budgets 1e12 3e12 1e13 3e13 --workers 2 --threads 3 --out results
scaling fit --out results
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import torch

from .core import Shape, corpus_fingerprint, load_corpus, train
from .fit import analyse

SHAPES = [
    Shape(1, 16, 2),
    Shape(1, 24, 2),
    Shape(1, 32),
    Shape(2, 32),
    Shape(2, 48),
    Shape(3, 64),
    Shape(4, 80),
    Shape(4, 112),
    Shape(5, 128),
    Shape(6, 160),
    Shape(6, 192),
]
# 1,024-token batches: with 4,096-token batches the smallest budgets gave too few
# steps to larger models and the IsoFLOP minimum sat at the grid edge
# (results/ablation_b32_t128).
B, T = 16, 64
MIN_STEPS = 400
MAX_TOKENS = 100_000_000  # wall-clock cap; tiny models at large C are off-optimum anyway


def plan(
    budgets: list[float], per_budget: int = 5, max_tokens: int = MAX_TOKENS
) -> list[tuple[float, Shape, int]]:
    """For each budget pick ``per_budget`` consecutive sizes around a centre guess
    N ≈ 35k * sqrt(C / 1e12) (from a pilot at C = 1e12), keeping at least
    ``MIN_STEPS`` optimizer steps."""
    out = []
    for c in budgets:
        ok = [s for s in SHAPES if MIN_STEPS * B * T <= c / (6 * s.n_params) <= max_tokens]
        guess = 35_000 * (c / 1e12) ** 0.5
        centre = min(range(len(ok)), key=lambda i: abs(ok[i].n_params - guess))
        lo = max(0, min(centre - per_budget // 2, len(ok) - per_budget))
        for s in ok[lo : lo + per_budget]:
            out.append((c, s, int(c / (6 * s.n_params))))
    return out


_DATA = None


def _work(args):
    c, n_layer, d_model, tokens, threads, device = args
    global _DATA
    torch.set_num_threads(threads)
    if _DATA is None:
        _DATA = load_corpus()
    shape = next(s for s in SHAPES if (s.n_layer, s.d_model) == (n_layer, d_model))
    r = train(shape, tokens, *_DATA, B=B, T=T, device=device)
    return {"budget": c, "device": device, "corpus": corpus_fingerprint(*_DATA), **asdict(r)}


def run(a) -> None:
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "runs.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["budget"], r["n_layer"], r["d_model"]))
    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    jobs = [
        (c, s.n_layer, s.d_model, D, a.threads, device)
        for c, s, D in plan(a.budgets, a.per_budget, a.max_tokens)
        if (c, s.n_layer, s.d_model) not in done
    ]
    print(f"{len(jobs)} runs to do ({len(done)} already done)", flush=True)
    # Biggest runs first so the long ones don't end up alone at the tail.
    jobs.sort(key=lambda j: -j[0])
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_work, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            with path.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(
                f"C={r['budget']:.0e} N={r['N']:>8} D={r['D']:>10} "
                f"val={r['val_loss']:.4f} ({r['wall_s']:.0f}s)",
                flush=True,
            )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="scaling")
    ap.add_argument("cmd", choices=["plan", "run", "fit"])
    ap.add_argument("--budgets", type=float, nargs="+", default=[1e12, 3e12, 1e13, 3e13])
    ap.add_argument("--per-budget", type=int, default=6)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    ap.add_argument(
        "--max-tokens", type=float, default=MAX_TOKENS, help="per-run token cap (wall clock)"
    )
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args(argv)
    a.max_tokens = int(a.max_tokens)
    if a.cmd == "plan":
        for c, s, D in plan(a.budgets, a.per_budget, a.max_tokens):
            print(
                f"C={c:.0e} L={s.n_layer} d={s.d_model} N={s.n_params} D={D} steps={D // (B * T)}"
            )
    elif a.cmd == "run":
        run(a)
    else:
        rows = [json.loads(x) for x in (a.out / "runs.jsonl").read_text().splitlines()]
        print(analyse(rows, a.out))


if __name__ == "__main__":
    main()
