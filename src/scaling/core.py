"""Data, model and training for small-scale scaling-law experiments.

Corpus: the Python standard library's own source (``<python>/Lib/**/*.py``), byte-level
(vocab 256). It is already on every machine with Python, so the experiments need
no download; it is split **by file** (5% of files held out) to avoid leakage.

Model: a GPT (pre-LN, causal SDPA attention, GELU MLP, tied embeddings).
``N`` = non-embedding parameters (Kaplan et al. 2020); compute ``C = 6 N D``.
"""

from __future__ import annotations

import glob
import math
import os
import sys
import time
import zlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def load_corpus(root: str | None = None, val_frac: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    root = root or os.path.join(sys.base_prefix, "Lib")
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    tr, va = [], []
    for f in files:
        rel = os.path.relpath(f, root).replace("\\", "/")
        with open(f, "rb") as fh:
            b = fh.read()
        (va if zlib.crc32(rel.encode()) % 1000 < val_frac * 1000 else tr).append(b)
    join = lambda bs: np.frombuffer(b"\n".join(bs), dtype=np.uint8).copy()  # noqa: E731
    return join(tr), join(va)


def batch(data: np.ndarray, B: int, T: int, gen: np.random.Generator) -> torch.Tensor:
    ix = gen.integers(0, len(data) - T - 1, B)
    return torch.from_numpy(np.stack([data[i : i + T + 1] for i in ix]).astype(np.int64))


@dataclass(frozen=True)
class Shape:
    n_layer: int
    d_model: int
    n_head: int = 4

    @property
    def n_params(self) -> int:  # non-embedding: attn (4d^2) + MLP (8d^2) per layer
        return 12 * self.n_layer * self.d_model**2


class Block(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.out = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, 2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, s: Shape, ctx: int = 128, vocab: int = 256):
        super().__init__()
        self.tok = nn.Embedding(vocab, s.d_model)
        self.pos = nn.Embedding(ctx, s.d_model)
        self.blocks = nn.ModuleList(Block(s.d_model, s.n_head) for _ in range(s.n_layer))
        self.ln = nn.LayerNorm(s.d_model)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = (
                    0.02 / math.sqrt(2 * s.n_layer)
                    if name.endswith(("proj.weight", "out.weight"))
                    else 0.02
                )
                nn.init.normal_(p, 0.0, std)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x)
        return self.ln(x) @ self.tok.weight.T  # tied output head


@dataclass
class RunResult:
    n_layer: int
    d_model: int
    N: int
    D: int
    C: float
    lr: float
    val_loss: float
    train_loss: float
    steps: int
    wall_s: float
    curve: list[tuple[int, float]]


def default_lr(d_model: int) -> float:
    """Wider models get smaller LRs (μP-flavoured 1/sqrt(width) heuristic)."""
    return min(3e-3, 3e-3 * math.sqrt(64 / d_model))


@torch.no_grad()
def evaluate(model: GPT, data: np.ndarray, T: int, n_batches: int = 40, B: int = 32) -> float:
    model.eval()
    gen = np.random.default_rng(1234)  # fixed eval windows for every model
    tot = 0.0
    for _ in range(n_batches):
        xb = batch(data, B, T, gen)
        tot += F.cross_entropy(model(xb[:, :-1]).flatten(0, 1), xb[:, 1:].flatten()).item()
    model.train()
    return tot / n_batches


def train(
    shape: Shape,
    tokens: int,
    train_data: np.ndarray,
    val_data: np.ndarray,
    B: int = 32,
    T: int = 128,
    lr: float | None = None,
    seed: int = 0,
    log_every: int = 50,
) -> RunResult:
    torch.manual_seed(seed)
    gen = np.random.default_rng(seed)
    model = GPT(shape, ctx=T)
    lr = lr or default_lr(shape.d_model)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
    )
    steps = max(1, tokens // (B * T))
    warm = max(1, steps // 50)

    def lr_at(s: int) -> float:  # linear warmup, cosine to 10%
        if s < warm:
            return lr * (s + 1) / warm
        p = (s - warm) / max(1, steps - warm)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    t0, curve, ema = time.time(), [], None
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        xb = batch(train_data, B, T, gen)
        loss = F.cross_entropy(model(xb[:, :-1]).flatten(0, 1), xb[:, 1:].flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        if s % log_every == 0 or s == steps - 1:
            curve.append((s, ema))
    D = steps * B * T
    return RunResult(
        shape.n_layer,
        shape.d_model,
        shape.n_params,
        D,
        6.0 * shape.n_params * D,
        lr,
        evaluate(model, val_data, T),
        float(ema),
        steps,
        time.time() - t0,
        curve,
    )
