import numpy as np

from scaling.cli import plan
from scaling.core import GPT, Shape, batch, train
from scaling.fit import fit_parametric, isoflop_minima, power_law


def test_param_count_matches_model():
    s = Shape(2, 32)
    m = GPT(s, ctx=16)
    non_emb = sum(
        p.numel()
        for n, p in m.named_parameters()
        if p.dim() == 2 and "tok" not in n and "pos" not in n
    )
    assert non_emb == s.n_params


def test_tiny_training_reduces_loss():
    rng = np.random.default_rng(0)
    data = np.tile(np.frombuffer(b"def f(x):\n    return x + 1\n", np.uint8), 400)
    r = train(Shape(1, 32), 32 * 16 * 120, data, data, B=32, T=16, lr=3e-3)
    assert r.val_loss < 1.0 < r.curve[0][1]
    assert batch(data, 4, 16, rng).shape == (4, 17)


def test_plan_respects_budget_and_min_steps():
    for c, s, D in plan([1e12, 1e13]):
        assert abs(6 * s.n_params * D - c) / c < 1e-3


def test_isoflop_recovers_known_minimum():
    rows = []
    for c, n_opt in ((1e12, 1e5), (1e13, 3e5), (1e14, 9e5)):
        for n in np.geomspace(n_opt / 8, n_opt * 8, 5):
            rows.append(
                {"budget": c, "N": n, "D": c / (6 * n), "val_loss": 2 + (np.log10(n / n_opt)) ** 2}
            )
    mins = isoflop_minima(rows)
    for m, n_opt in zip(mins, (1e5, 3e5, 9e5), strict=True):
        assert abs(np.log10(m["N_opt"] / n_opt)) < 1e-6 and m["interior_minimum"]
    a, _ = power_law([m["budget"] for m in mins], [m["N_opt"] for m in mins])
    assert abs(a - np.log10(9) / 2) < 1e-6


def test_parametric_fit_recovers_synthetic_law():
    E, A, B, al, be = 1.5, 30.0, 60.0, 0.35, 0.3
    rows = [
        {"N": n, "D": d, "val_loss": E + A / n**al + B / d**be}
        for n in np.geomspace(1e4, 1e7, 6)
        for d in np.geomspace(1e6, 1e9, 6)
    ]
    p = fit_parametric(rows)
    assert p["rmse"] < 1e-3 and abs(p["alpha"] - al) < 0.05 and abs(p["beta"] - be) < 0.05
