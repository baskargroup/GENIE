#!/usr/bin/env python3
"""Export a two-head SIREN checkpoint (.pth state_dict) to browser JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export SIREN .pth checkpoint to JSON")
    p.add_argument("--input", type=Path, required=True, help="Path to .pth state_dict")
    p.add_argument("--output", type=Path, required=True, help="Path to output .json")
    p.add_argument("--w0", type=float, default=30.0)
    p.add_argument("--w0_hidden", type=float, default=30.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    state = torch.load(args.input, map_location="cpu")

    layer_ids = sorted(
        {
            int(k.split(".")[1])
            for k in state.keys()
            if k.startswith("layers.") and k.endswith(".linear.weight")
        }
    )
    if not layer_ids:
        raise ValueError("No layers.*.linear.weight found in checkpoint")

    in_dim = int(state[f"layers.{layer_ids[0]}.linear.weight"].shape[1])
    hidden_dim = int(state[f"layers.{layer_ids[0]}.linear.weight"].shape[0])
    hidden_layers = len(layer_ids)

    skip_in = []
    layers = []
    for i in layer_ids:
        w = state[f"layers.{i}.linear.weight"].detach().cpu()
        b = state[f"layers.{i}.linear.bias"].detach().cpu()
        if i > 0 and int(w.shape[1]) == hidden_dim + in_dim:
            skip_in.append(i)
        layers.append(
            {
                "weight": w.tolist(),
                "bias": b.tolist(),
                "w0": float(args.w0 if i == 0 else args.w0_hidden),
            }
        )

    w_final = state["final.weight"].detach().cpu()
    b_final = state["final.bias"].detach().cpu()
    out_dim = int(w_final.shape[0])

    payload = {
        "meta": {
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "out_dim": out_dim,
            "skip_in": skip_in,
            "w0": float(args.w0),
            "w0_hidden": float(args.w0_hidden),
        },
        "layers": layers,
        "final": {
            "weight": w_final.tolist(),
            "bias": b_final.tolist(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"saved: {args.output}")
    print("meta:", payload["meta"])


if __name__ == "__main__":
    main()
