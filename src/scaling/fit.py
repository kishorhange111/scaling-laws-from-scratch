"""Scaling-law fits (Hoffmann et al., 2022).

* **IsoFLOP (approach 2):** for each compute budget, fit a parabola to loss vs
  log N; its minimum gives N_opt(C). Then fit power laws N_opt ∝ C^a and
  D_opt ∝ C^b (b = 1 − a since C = 6ND).
* **Parametric (approach 3):** fit L(N, D) = E + A / N^α + B / D^β to all runs with
  a Huber loss on log-loss, from a grid of initialisations.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def isoflop_minima(rows: list[dict]) -> list[dict]:
    out = []
    for c in sorted({r["budget"] for r in rows}):
        rs = sorted((r for r in rows if r["budget"] == c), key=lambda r: r["N"])
        if len(rs) < 3:
            continue
        x = np.log10([r["N"] for r in rs])
        y = np.array([r["val_loss"] for r in rs])
        a2, a1, a0 = np.polyfit(x, y, 2)
        interior = a2 > 0 and x.min() <= -a1 / (2 * a2) <= x.max()
        xo = -a1 / (2 * a2) if a2 > 0 else x[np.argmin(y)]
        n_opt = 10**xo
        out.append(
            {
                "budget": c,
                "N_opt": float(n_opt),
                "D_opt": float(c / (6 * n_opt)),
                "L_opt": float(np.polyval([a2, a1, a0], xo)),
                "interior_minimum": bool(interior),
                "points": [(r["N"], r["val_loss"]) for r in rs],
                "parabola": [float(a2), float(a1), float(a0)],
            }
        )
    return out


def power_law(xs, ys) -> tuple[float, float]:
    """y = k x^a in log space; returns (a, k)."""
    a, logk = np.polyfit(np.log10(xs), np.log10(ys), 1)
    return float(a), float(10**logk)


def fit_parametric(rows: list[dict], delta: float = 1e-3) -> dict:
    N = np.array([r["N"] for r in rows], float)
    D = np.array([r["D"] for r in rows], float)
    L = np.array([r["val_loss"] for r in rows], float)

    def huber(r):
        a = np.abs(r)
        return np.where(a <= delta, 0.5 * r**2, delta * (a - 0.5 * delta)).sum()

    def obj(p):
        e, a, b, al, be = p  # log E, log A, log B, alpha, beta
        pred = np.logaddexp.reduce([np.full_like(N, e), a - al * np.log(N), b - be * np.log(D)])
        return huber(pred - np.log(L))

    best = None
    for e, a, b, al, be in itertools.product(
        [-1, 0, 0.5], [0, 5, 10], [0, 5, 10], [0, 0.5, 1], [0, 0.5, 1]
    ):
        r = minimize(obj, [e, a, b, al, be], method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    e, a, b, al, be = best.x
    E, A, Bc = np.exp([e, a, b])
    pred = E + A / N**al + Bc / D**be
    # compute-optimal allocation implied by the fit: N_opt ∝ C^(β/(α+β))
    return {
        "E": float(E),
        "A": float(A),
        "B": float(Bc),
        "alpha": float(al),
        "beta": float(be),
        "a_implied": float(be / (al + be)),
        "rmse": float(np.sqrt(np.mean((pred - L) ** 2))),
    }


def analyse(rows: list[dict], out: Path) -> str:
    mins = isoflop_minima(rows)
    res: dict = {"isoflop": mins}
    lines = [
        "| budget C | N_opt | D_opt | D/N | L_opt | interior min |",
        "|---|---|---|---|---|---|",
    ]
    for m in mins:
        lines.append(
            f"| {m['budget']:.0e} | {m['N_opt']:.3g} | {m['D_opt']:.3g} | "
            f"{m['D_opt'] / m['N_opt']:.1f} | {m['L_opt']:.4f} | {m['interior_minimum']} |"
        )
    good = [m for m in mins if m["interior_minimum"]]
    if len(good) >= 2:
        a, k = power_law([m["budget"] for m in good], [m["N_opt"] for m in good])
        b, _ = power_law([m["budget"] for m in good], [m["D_opt"] for m in good])
        res["N_opt_exponent"], res["D_opt_exponent"] = a, b
        lines.append(f"\nIsoFLOP fit (interior minima only): N_opt ∝ C^{a:.3f}, D_opt ∝ C^{b:.3f}")
    p = fit_parametric(rows)
    res["parametric"] = p
    lines.append(
        f"\nParametric fit: L = {p['E']:.3f} + {p['A']:.3g}/N^{p['alpha']:.3f} + "
        f"{p['B']:.3g}/D^{p['beta']:.3f} (RMSE {p['rmse']:.4f}); "
        f"implies N_opt ∝ C^{p['a_implied']:.3f}"
    )
    (out / "fits.json").write_text(json.dumps(res, indent=1))
    try:
        _plot(rows, mins, res, out)
    except ImportError:
        pass
    md = "\n".join(lines)
    (out / "fits.md").write_text(md + "\n")
    return md


def _plot(rows, mins, res, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, m in enumerate(mins):
        ns, ls = zip(*m["points"], strict=True)
        ax[0].scatter(ns, ls, color=f"C{i}", label=f"C={m['budget']:.0e}")
        xs = np.linspace(np.log10(min(ns)), np.log10(max(ns)), 50)
        ax[0].plot(10**xs, np.polyval(m["parabola"], xs), color=f"C{i}", lw=1)
        ax[0].scatter([m["N_opt"]], [m["L_opt"]], color=f"C{i}", marker="*", s=150, edgecolor="k")
    ax[0].set(
        xscale="log",
        xlabel="non-embedding params N",
        ylabel="val loss (nats/byte)",
        title="IsoFLOP curves",
    )
    ax[0].legend(fontsize=8)
    cs = np.array([m["budget"] for m in mins])
    ax[1].scatter(cs, [m["N_opt"] for m in mins], label="N_opt")
    ax[1].scatter(cs, [m["D_opt"] for m in mins], label="D_opt")
    if "N_opt_exponent" in res:
        for key, col in (("N_opt", "C0"), ("D_opt", "C1")):
            good = [m for m in mins if m["interior_minimum"]]
            a, k = power_law([m["budget"] for m in good], [m[key] for m in good])
            ax[1].plot(cs, k * cs**a, color=col, lw=1, label=f"{key} ∝ C^{a:.2f}")
    ax[1].set(
        xscale="log", yscale="log", xlabel="compute C (FLOPs)", title="compute-optimal allocation"
    )
    ax[1].legend(fontsize=8)
    p = res["parametric"]
    L = np.array([r["val_loss"] for r in rows])
    pred = np.array(
        [p["E"] + p["A"] / r["N"] ** p["alpha"] + p["B"] / r["D"] ** p["beta"] for r in rows]
    )
    ax[2].scatter(pred, L)
    lo, hi = min(L.min(), pred.min()), max(L.max(), pred.max())
    ax[2].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[2].set(
        xlabel="parametric fit L(N, D)",
        ylabel="measured val loss",
        title=f"parametric fit (RMSE {p['rmse']:.3f})",
    )
    fig.tight_layout()
    fig.savefig(out / "scaling.png", dpi=120)
    plt.close(fig)
