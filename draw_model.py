"""Generate a PDF architecture diagram of Dinomaly using torchview.

Run:
    python draw_model.py [--encoder NAME] [--decoder-depth N] [--out PATH]

Output:
    dinomaly_architecture.pdf  (vector PDF, good for papers)

Dependencies (install in your conda env, NOT inside the Codex sandbox):
    pip install torchview graphviz
    sudo apt install -y graphviz          # Linux/WSL only
    # On Windows: choco install graphviz   OR download from graphviz.org and add bin/ to PATH

This script is self-contained: it builds the same mock-encoder Dinomaly replica
as inspect_dinomaly.py, so it needs no ``import anomalib``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# Reuse the inline Dinomaly replica from inspect_dinomaly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_dinomaly import build_dry_model  # noqa: E402


def _make_parser():
    import argparse
    p = argparse.ArgumentParser(description="Draw Dinomaly architecture as PDF.")
    p.add_argument("--encoder", default="vit_base_patch14_reg4_dinov2")
    p.add_argument("--decoder-depth", type=int, default=8)
    p.add_argument("--image-size", type=int, default=392)
    p.add_argument("--out", default=None, help="Output PDF path. Default: dinomaly_<encoder>_d<N>.pdf")
    return p


def main() -> None:
    args = _make_parser().parse_args()
    args.no_forward = True  # we only need the graph, not the dummy pass
    args.no_tree = True

    model = build_dry_model(args)
    model.eval()

    try:
        from torchview import draw_graph
    except ImportError as exc:
        msg = (
            "torchview is required to draw the architecture graph.\n"
            "Install it with:\n"
            "    pip install torchview graphviz\n"
            "    sudo apt install -y graphviz   (Linux/WSL)\n"
            "On Windows, install graphviz from https://graphviz.org/download/ and "
            "make sure its bin/ is on PATH.\n\n"
            f"Original error: {exc}"
        )
        raise SystemExit(msg)

    out = args.out or f"dinomaly_{args.encoder.replace('/', '_')}_d{args.decoder_depth}.pdf"

    print(f"Drawing Dinomaly ({args.encoder}, decoder_depth={args.decoder_depth}) ...")
    graph = draw_graph(
        model,
        input_size=(1, 3, args.image_size, args.image_size),
        device="cpu",
        expand_nested=True,
        depth=4,
        save_graph=True,
        graph_name=out.rsplit(".", 1)[0],
        roll=False,
    )
    # Clean up: drop the dummy InferenceBatchShim end node, keep the underlying tensors
    # (torchview sometimes wraps dataclass outputs; the PDF still shows everything useful.)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
