#!/usr/bin/env python3
"""Precompute fast browser cache for Gram-eigen SIREN editor.

Outputs:
- <outdir>/meta.json
- <outdir>/cache.bin   float32, mode-major layout: [base, mode1..modeK], each length n_vox
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, w0=30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.w0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


class MultiHeadSiren(nn.Module):
    def __init__(
        self,
        in_dim=3,
        hidden_dim=128,
        hidden_layers=8,
        w0=30.0,
        w0_hidden=30.0,
        out_dim=2,
        skip_in=(4,),
    ):
        super().__init__()
        self.skip_in = tuple(skip_in)
        layers = [SineLayer(in_dim, hidden_dim, is_first=True, w0=w0)]
        for i in range(1, hidden_layers):
            in_features = hidden_dim + (in_dim if i in self.skip_in else 0)
            layers.append(SineLayer(in_features, hidden_dim, w0=w0_hidden))
        self.layers = nn.ModuleList(layers)
        self.final = nn.Linear(hidden_dim, out_dim)

    def features(self, x):
        h = x
        for idx, layer in enumerate(self.layers):
            if idx in self.skip_in:
                h = torch.cat([h, x], dim=-1) / math.sqrt(2)
            h = layer(h)
        return h


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute browser cache for editor")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--source_head", type=int, default=0)
    p.add_argument("--target_head", type=int, default=1)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--grid_res", type=int, default=64)
    p.add_argument("--box_min", type=float, default=-1.0)
    p.add_argument("--box_max", type=float, default=1.0)
    p.add_argument("--n_obs", type=int, default=120000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--batch_obs", type=int, default=65536)
    p.add_argument("--batch_grid", type=int, default=131072)
    return p.parse_args()


def build_model(state: dict[str, torch.Tensor], device: torch.device) -> MultiHeadSiren:
    out_dim = int(state["final.weight"].shape[0])
    model = MultiHeadSiren(
        in_dim=3,
        hidden_dim=128,
        hidden_layers=8,
        w0=30.0,
        w0_hidden=30.0,
        out_dim=out_dim,
        skip_in=(4,),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def features_in_batches(model: MultiHeadSiren, x_np: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    out = []
    for i in range(0, x_np.shape[0], batch):
        xb = torch.from_numpy(np.ascontiguousarray(x_np[i : i + batch])).to(device)
        hb = model.features(xb).detach().cpu().numpy().astype(np.float64)
        out.append(hb)
    return np.concatenate(out, axis=0)


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    state = torch.load(args.ckpt, map_location=device)
    model = build_model(state, device)

    W = state["final.weight"].detach().cpu().numpy().astype(np.float64)
    B = state["final.bias"].detach().cpu().numpy().astype(np.float64)
    out_dim = W.shape[0]

    hs = int(args.source_head)
    ht = int(args.target_head)
    if hs < 0 or hs >= out_dim or ht < 0 or ht >= out_dim:
        raise ValueError(f"head index out of range: source={hs}, target={ht}, out_dim={out_dim}")

    rng = np.random.default_rng(args.seed)
    Xobs = rng.uniform(args.box_min, args.box_max, size=(args.n_obs, 3)).astype(np.float32)

    print("computing Gram on observation features...")
    Hobs = features_in_batches(model, Xobs, device, args.batch_obs)
    G = Hobs.T @ Hobs
    evals, evecs = np.linalg.eigh(G)
    idx = np.argsort(evals)[::-1]
    evals = np.clip(evals[idx], 0.0, None)
    V = evecs[:, idx]

    k = min(int(args.k), V.shape[1])
    Vk = V[:, :k]

    ws = W[hs]
    wt = W[ht]
    bs = float(B[hs])
    bt = float(B[ht])

    cs = Vk.T @ ws
    ct = Vk.T @ wt
    w_perp = ws - Vk @ cs

    ws_k = Vk @ cs
    wt_k = Vk @ ct
    expl_s = float(np.dot(ws_k, ws_k) / max(1e-20, np.dot(ws, ws)))
    expl_t = float(np.dot(wt_k, wt_k) / max(1e-20, np.dot(wt, wt)))

    print(f"top-{k} explained: source={expl_s:.6f}, target={expl_t:.6f}")

    print("precomputing voxel mode volumes...")
    xs = np.linspace(args.box_min, args.box_max, args.grid_res, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
    n_vox = pts.shape[0]

    base = np.zeros((n_vox,), dtype=np.float32)
    modes = np.zeros((k, n_vox), dtype=np.float32)

    Vk_t = torch.from_numpy(Vk.astype(np.float32)).to(device)
    w_perp_t = torch.from_numpy(w_perp.astype(np.float32)).to(device).view(-1, 1)

    for i in range(0, n_vox, args.batch_grid):
        pb = pts[i : i + args.batch_grid]
        xb = torch.from_numpy(np.ascontiguousarray(pb)).to(device)
        hb = model.features(xb)  # [B, hidden]

        base_b = (hb @ w_perp_t).squeeze(-1)  # [B]
        mode_b = hb @ Vk_t  # [B, k]

        base[i : i + pb.shape[0]] = base_b.detach().cpu().numpy().astype(np.float32)
        modes[:, i : i + pb.shape[0]] = mode_b.detach().cpu().numpy().T.astype(np.float32)

    cache = np.concatenate([base[None, :], modes], axis=0).astype(np.float32, copy=False)
    bin_path = args.outdir / "cache.bin"
    cache.tofile(bin_path)

    meta = {
        "ckpt": str(args.ckpt),
        "device_precompute": str(device),
        "source_head": hs,
        "target_head": ht,
        "out_dim": int(out_dim),
        "k": int(k),
        "grid_res": int(args.grid_res),
        "box_min": float(args.box_min),
        "box_max": float(args.box_max),
        "n_vox": int(n_vox),
        "binary": "cache.bin",
        "cs": cs.tolist(),
        "ct": ct.tolist(),
        "bs": bs,
        "bt": bt,
        "expl_source_topk": expl_s,
        "expl_target_topk": expl_t,
        "top_eigenvalues": evals[: max(20, k)].tolist(),
    }
    meta_path = args.outdir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f)

    print("saved:", meta_path)
    print("saved:", bin_path, f"({bin_path.stat().st_size / (1024**2):.2f} MB)")


if __name__ == "__main__":
    main()
