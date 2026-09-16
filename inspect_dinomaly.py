"""Inspect the Dinomaly network architecture (self-contained).

This script prints, for the Dinomaly anomaly detection model:

  - The full module tree (``print(model)``).
  - Parameter counts broken down by submodule, plus trainable vs frozen totals.
  - A dummy forward pass with input/output shapes attached to every leaf module.
  - The final ``pred_score`` and ``anomaly_map`` shapes.
  - Wall-clock forward time on the selected device.

The script is *self-contained*: it ships inline copies of the small set of
Dinomaly submodules it needs (LinearAttention, DinomalyMLP, DecoderViTBlock,
GaussianBlur2d). This means you do NOT need a working ``import anomalib`` to
run it -- only ``torch``. The inline classes mirror the production code in
``src/anomalib/models/image/dinomaly/`` exactly enough for shape and parameter
inspection.

If you would rather use the real anomalib components (so the model is bit-for-bit
identical to the repo code), set ``USE_REAL_COMPONENTS = True`` near the top of
the file; the script will then try to import them and fall back to inline ones
on failure.

Usage (PowerShell or bash):

    # Default: vit_base_patch14_reg4_dinov2, decoder_depth=8, 392x392, CPU
    python inspect_dinomaly.py --mode dry

    # Skip the (large) DINOv2 module tree in the printout, keep everything else
    python inspect_dinomaly.py --mode dry --no-tree

    # Use a different encoder / decoder depth
    python inspect_dinomaly.py --mode dry --encoder vit_small_patch14_reg4_dinov2 --decoder-depth 4

    # A bigger image on GPU
    python inspect_dinomaly.py --mode dry --image-size 448 --device cuda

    # Just dump the model, no dummy forward
    python inspect_dinomaly.py --mode dry --no-forward
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Toggle: try the real anomalib components first if available.
# If you want a strict bit-for-bit replica of the repo, set this to True.
# ---------------------------------------------------------------------------
USE_REAL_COMPONENTS = False


# ===========================================================================
# Section 1. Inline Dinomaly submodules.
#
# These are near-verbatim copies from
#   src/anomalib/models/image/dinomaly/components/layers.py
#   src/anomalib/models/image/dinomaly/torch_model.py (DecoderViTBlock)
#   src/anomalib/models/components/filters/gaussian_blur.py
# Kept inline so this script has no import dependencies beyond torch.
# ===========================================================================

class LinearAttention(nn.Module):
    """Linear (ELU + 1) attention used inside DinomalyDecoderViTBlock.

    From ``anomalib.models.image.dinomaly.components.layers.LinearAttention``.
    """

    def __init__(self, input_dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = input_dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(input_dim, input_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(input_dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        kv = torch.matmul(k.transpose(-2, -1), v)
        k_sum = k.sum(dim=-2, keepdim=True)
        z = 1.0 / torch.sum(q * k_sum, dim=-1, keepdim=True)
        x = torch.matmul(q, kv) * z
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, kv


class DinomalyMLP(nn.Module):
    """Bottleneck / decoder MLP used in Dinomaly.

    ``apply_input_dropout=True`` turns on input dropout, which is the
    "noisy bottleneck" trick from the Dinomaly paper. Set to False in decoder
    blocks. From ``anomalib.models.image.dinomaly.components.layers.DinomalyMLP``.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=False,
        apply_input_dropout=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)
        self.apply_input_dropout = apply_input_dropout

    def forward(self, x):
        if self.apply_input_dropout:
            x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class DecoderViTBlock(nn.Module):
    """Transformer decoder block with LinearAttention + MLP.

    From ``anomalib.models.image.dinomaly.torch_model.DecoderViTBlock``.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn=LinearAttention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                         attn_drop=attn_drop, proj_drop=drop)
        # Use a simple DropPath stand-in if timm is not available.
        try:
            from timm.layers.drop import DropPath  # type: ignore
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        except Exception:
            self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = DinomalyMLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
            apply_input_dropout=False,
            bias=False,
        )

    def forward(self, x, return_attention=False, attn_mask=None):
        if attn_mask is not None:
            y, _ = self.attn(self.norm1(x), attn_mask=attn_mask)
        else:
            y, _ = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, None
        return x


class GaussianBlur2d(nn.Module):
    """2D separable Gaussian blur (single- or multi-channel).

    Inline copy of ``anomalib.models.components.filters.gaussian_blur.GaussianBlur2d``.
    Uses ``torch.nn.functional.conv2d`` with a separable (outer product) kernel.
    """

    def __init__(self, sigma, kernel_size, channels=1):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.channels = channels
        k = kernel_size // 2
        x = torch.arange(-k, k + 1, dtype=torch.float32)
        g = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g.unsqueeze(0) * g.unsqueeze(1)  # outer product -> (k, k)
        kernel = kernel.expand(channels, 1, kernel_size, kernel_size).contiguous()
        self.register_buffer("kernel", kernel)
        self.pad = kernel_size // 2

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=self.pad, groups=self.channels)


# Lightweight shim that mimics ``anomalib.data.InferenceBatch``.
@dataclass
class InferenceBatchShim:
    pred_score: torch.Tensor
    anomaly_map: torch.Tensor


# ===========================================================================
# Section 2. Mock DINOv2 encoder + dry-run model assembly.
# ===========================================================================

DINO_ARCHITECTURES = {
    "small": {"embed_dim": 384, "num_heads": 6, "target_layers": [2, 3, 4, 5, 6, 7, 8, 9]},
    "base": {"embed_dim": 768, "num_heads": 12, "target_layers": [2, 3, 4, 5, 6, 7, 8, 9]},
    "large": {"embed_dim": 1024, "num_heads": 16, "target_layers": [4, 6, 8, 10, 12, 14, 16, 18]},
    "huge": {"embed_dim": 1280, "num_heads": 20, "target_layers": [3, 9, 12, 15, 18, 21, 24, 27]},
    "giant": {"embed_dim": 1536, "num_heads": 24, "target_layers": [6, 10, 14, 18, 22, 26, 30, 34]},
}
DEFAULT_FUSE_LAYERS = [[0, 1, 2, 3], [4, 5, 6, 7]]
DEFAULT_RESIZE_SIZE = 256
DEFAULT_GAUSSIAN_KERNEL_SIZE = 5
DEFAULT_GAUSSIAN_SIGMA = 4
DEFAULT_MAX_RATIO = 0.01


def _infer_patch_size(name: str) -> int:
    m = re.search(r"patch(\d+)", name)
    return int(m.group(1)) if m else 14


def _infer_reg_count(name: str) -> int:
    m = re.search(r"reg(\d+)", name)
    return int(m.group(1)) if m else 0


class MockDinoV2Encoder(nn.Module):
    """Random-feature stand-in for the TimmFeatureExtractor-based DINOv2 encoder.

    Returns ``{layer_name: (B, 1+reg+H_p*W_p, D)}`` of random tensors.
    """

    def __init__(self, backbone_name, layers, embed_dim, num_register_tokens, patch_size):
        super().__init__()
        self.backbone = backbone_name
        self.layers = layers
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens
        self.patch_size = patch_size
        self.norm = False
        self.return_class_token = True

    def forward(self, x):
        B, _, H, W = x.shape
        H_p = H // self.patch_size
        W_p = W // self.patch_size
        N = 1 + self.num_register_tokens + H_p * W_p
        return {layer: torch.randn(B, N, self.embed_dim, device=x.device, dtype=x.dtype)
                for layer in self.layers}


class DryRunDinomaly(nn.Module):
    """Self-contained replica of DinomalyModel with a mock encoder.

    Reimplements the forward pass of
    ``anomalib.models.image.dinomaly.torch_model.DinomalyModel`` so we can
    run a dummy pass and capture shapes without needing pretrained weights
    or any of the deep anomalib import chain.
    """

    def __init__(self, encoder, target_layers, fuse_layer_encoder, fuse_layer_decoder,
                 bottleneck, decoder, gaussian_blur, resize_size, max_ratio, image_size):
        super().__init__()
        self.encoder = encoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.gaussian_blur = gaussian_blur
        self._resize_size = resize_size
        self._max_ratio = max_ratio
        self._image_size = image_size
        self.remove_class_token = False
        self.use_context_recentering = False

    def forward(self, x):
        feats = self.encoder(x)
        encoder_features = [feats[f"blocks.{i}"] for i in self.target_layers]
        num_reg = self.encoder.num_register_tokens
        encoder_features = [e[:, 1 + num_reg:, :] for e in encoder_features]

        # Fuse encoder features, run bottleneck
        z = torch.stack(encoder_features, dim=1).mean(dim=1)
        for block in self.bottleneck:
            z = block(z)

        # Decoder (8 blocks), collected in forward order then reversed
        decoder_features = []
        for block in self.decoder:
            z = block(z, attn_mask=None)
            decoder_features.append(z)
        decoder_features = decoder_features[::-1]

        # Group fusion (low / high)
        en = [torch.stack([encoder_features[idx] for idx in idxs], dim=1).mean(dim=1)
              for idxs in self.fuse_layer_encoder]
        de = [torch.stack([decoder_features[idx] for idx in idxs], dim=1).mean(dim=1)
              for idxs in self.fuse_layer_decoder]

        # Reshape to spatial (B, D, H_p, W_p)
        B, D = en[0].shape[0], en[0].shape[-1]
        H_p = self._image_size // self.encoder.patch_size
        W_p = self._image_size // self.encoder.patch_size
        en_spatial = [e.transpose(1, 2).reshape(B, D, H_p, W_p).contiguous() for e in en]
        de_spatial = [d.transpose(1, 2).reshape(B, D, H_p, W_p).contiguous() for d in de]

        # Per-group anomaly map, then average
        anomaly_maps = []
        for es, ds in zip(en_spatial, de_spatial):
            a_map = 1 - F.cosine_similarity(es, ds)
            a_map = a_map.unsqueeze(1)
            a_map = F.interpolate(
                a_map,
                size=(self._image_size, self._image_size),
                mode="bilinear", align_corners=True,
            )
            anomaly_maps.append(a_map)
        anomaly_map = torch.cat(anomaly_maps, dim=1).mean(dim=1, keepdim=True)

        # Image-level score via top-1% of the blur-smoothed map
        smoothed = self.gaussian_blur(
            F.interpolate(anomaly_map, size=self._resize_size, mode="bilinear", align_corners=False)
        )
        flat = smoothed.flatten(1)
        if self._max_ratio == 0:
            score = torch.max(flat, dim=1)[0]
        else:
            topk = max(1, int(flat.shape[1] * self._max_ratio))
            score = torch.sort(flat, dim=1, descending=True)[0][:, :topk].mean(dim=1)

        return InferenceBatchShim(pred_score=score, anomaly_map=anomaly_map)


def build_dry_model(args) -> nn.Module:
    encoder_name = args.encoder
    arch = None
    for key, cfg in DINO_ARCHITECTURES.items():
        if key in encoder_name:
            arch = cfg
            break
    if arch is None:
        raise ValueError(
            f"Encoder '{encoder_name}' not in DINO_ARCHITECTURES: {list(DINO_ARCHITECTURES)}"
        )
    embed_dim = arch["embed_dim"]
    num_heads = arch["num_heads"]
    target_layers = arch["target_layers"]

    mock_encoder = MockDinoV2Encoder(
        backbone_name=encoder_name,
        layers=[f"blocks.{i}" for i in target_layers],
        embed_dim=embed_dim,
        num_register_tokens=_infer_reg_count(encoder_name),
        patch_size=_infer_patch_size(encoder_name),
    )

    # Try to use the real anomalib components if available, else fall back to inline.
    MLP = DinomalyMLP
    Attn = LinearAttention
    DecBlock = DecoderViTBlock
    if USE_REAL_COMPONENTS:
        try:
            from anomalib.models.image.dinomaly.components.layers import (
                DinomalyMLP as RealMLP,
                LinearAttention as RealAttn,
            )
            from anomalib.models.image.dinomaly.torch_model import DecoderViTBlock as RealBlock
            MLP, Attn, DecBlock = RealMLP, RealAttn, RealBlock
            logging.getLogger("dinomaly_inspect").info("Using real anomalib components.")
        except Exception as exc:
            logging.getLogger("dinomaly_inspect").warning(
                f"Falling back to inline components ({type(exc).__name__}: {str(exc).splitlines()[0][:120]})"
            )

    bottleneck = nn.ModuleList([MLP(
        in_features=embed_dim,
        hidden_features=embed_dim * 4,
        out_features=embed_dim,
        act_layer=nn.GELU,
        drop=0.2,
        bias=False,
        apply_input_dropout=True,
    )])

    decoder = nn.ModuleList([
        DecBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=nn.LayerNorm,
            attn_drop=0.0,
            attn=Attn,
        )
        for _ in range(args.decoder_depth)
    ])

    gaussian_blur = GaussianBlur2d(
        sigma=DEFAULT_GAUSSIAN_SIGMA,
        kernel_size=DEFAULT_GAUSSIAN_KERNEL_SIZE,
        channels=1,
    )

    return DryRunDinomaly(
        encoder=mock_encoder,
        target_layers=target_layers,
        fuse_layer_encoder=DEFAULT_FUSE_LAYERS,
        fuse_layer_decoder=DEFAULT_FUSE_LAYERS,
        bottleneck=bottleneck,
        decoder=decoder,
        gaussian_blur=gaussian_blur,
        resize_size=DEFAULT_RESIZE_SIZE,
        max_ratio=DEFAULT_MAX_RATIO,
        image_size=args.image_size,
    )


# ===========================================================================
# Section 3. Real DinomalyModel wrapper (only used if --mode full / auto).
# ===========================================================================

def build_real_model(args):
    """Build the actual DinomalyModel from anomalib. Will try to download DINOv2 weights.

    Will only succeed in environments where:
      1. ``anomalib`` imports cleanly (no missing transitive deps).
      2. ``timm`` can reach HuggingFace / the internet to fetch pretrained weights.
    """
    sys_path = sys.path if "sys" in dir() else __import__("sys").path
    src_path = str(Path(__file__).resolve().parent / "src")
    if src_path not in sys_path:
        sys_path.insert(0, src_path)
    from anomalib.models.image.dinomaly.torch_model import DinomalyModel
    return DinomalyModel(
        encoder_name=args.encoder,
        bottleneck_dropout=0.2,
        decoder_depth=args.decoder_depth,
    )


# ===========================================================================
# Section 4. Forward hooks + pretty printing.
# ===========================================================================

def _fmt_output(obj):
    if isinstance(obj, torch.Tensor):
        return f"Tensor{tuple(obj.shape)}"
    if isinstance(obj, dict):
        return {k: _fmt_output(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_fmt_output(x) for x in obj]
    if hasattr(obj, "pred_score") and hasattr(obj, "anomaly_map"):
        return f"InferenceBatch(score={tuple(obj.pred_score.shape)}, map={tuple(obj.anomaly_map.shape)})"
    return type(obj).__name__


def install_shape_hooks(model):
    shapes = []
    handles = []
    for name, module in model.named_modules():
        if list(module.children()):
            continue
        n_params = sum(p.numel() for p in module.parameters())

        def _hook(_mod, _inp, out, _n=n_params, _name=name):
            shapes.append((_name, _fmt_output(out), _n))

        handles.append(module.register_forward_hook(_hook))
    return shapes, handles


def print_model_tree(model, title):
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)
    print(model)


def print_param_table(model):
    """Each row shows params *owned* by that module only (no double counting).

    TOTAL is the model unique parameter count so percentages add up to 100%.
    """
    rows = []
    for name, module in model.named_modules():
        n = sum(p.numel() for p in module.parameters(recurse=False))
        if n == 0:
            continue
        rows.append((name, n))
    unique_total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = unique_total - trainable

    print()
    print("=" * 96)
    print("Parameter breakdown (each row = params owned by that module only)")
    print("=" * 96)
    print("{:70s} {:>16s} {:>8s}".format("submodule", "#params", "%"))
    print("-" * 96)
    for name, n in rows:
        print("{:70s} {:>16,d} {:>7.2f}%".format(name, n, 100 * n / unique_total))
    print("-" * 96)
    print("{:70s} {:>16,d} {:>7.2f}%".format("TOTAL (unique)", unique_total, 100.0))
    print("{:70s} {:>16,d} {:>7.2f}%".format("Trainable", trainable, 100 * trainable / unique_total))
    print("{:70s} {:>16,d} {:>7.2f}%".format("Frozen", frozen, 100 * frozen / unique_total))
    print('=' * 96)
    if frozen == 0 and isinstance(model, DryRunDinomaly):
        print('NOTE (dry mode): real DINOv2 encoder is FROZEN (~86M for vit_base).')
        print('                  Only the 61M shown above (bottleneck + decoder) are trainable.')
        print('                  In full mode the encoder shows up here with its real weights.')
        print('=' * 96)

    print("=" * 96)




def print_shape_table(shapes):
    print()
    print("=" * 96)
    print("Forward pass shapes (per leaf module)")
    print("=" * 96)
    print(f"{'module':<60} {'output':<28} {'#params':>8}")
    print("-" * 96)
    for name, out, params in shapes:
        out_str = str(out)
        if len(out_str) > 28:
            out_str = out_str[:25] + "..."
        print(f"{name:<60} {out_str:<28} {params:>8,}")
    print("=" * 96)


# ===========================================================================
# Section 5. CLI + main.
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["auto", "full", "dry"], default="dry",
                   help="'auto' tries the real DinomalyModel and falls back to dry on any failure. "
                        "Default is 'dry' because it needs no weights and no internet.")
    p.add_argument("--encoder", default="vit_base_patch14_reg4_dinov2")
    p.add_argument("--decoder-depth", type=int, default=8)
    p.add_argument("--image-size", type=int, default=392,
                   help="Must be a multiple of 14.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--no-forward", action="store_true")
    p.add_argument("--no-tree", action="store_true")
    return p.parse_args()


def main():
    import sys
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("dinomaly_inspect")

    model = None
    mode_used = None
    if args.mode in ("full", "auto"):
        try:
            model = build_real_model(args)
            mode_used = "full"
        except Exception as exc:
            first_line = str(exc).splitlines()[0][:200] if str(exc) else "(empty)"
            logger.warning(f"Could not build the real DinomalyModel: {first_line}")
            if args.mode == "full":
                raise
            logger.info("Falling back to dry-run mode (mock DINOv2 encoder, real bottleneck/decoder).")
            model = build_dry_model(args)
            mode_used = "dry"
    else:
        model = build_dry_model(args)
        mode_used = "dry"

    logger.info(f"Mode: {mode_used}")

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    if not args.no_tree:
        print_model_tree(model, f"Model tree  [{mode_used}]")
    print_param_table(model)

    if args.no_forward:
        return

    logger.info(
        f"\nRunning dummy forward: batch={args.batch_size}, "
        f"image={args.image_size}x{args.image_size}, device={args.device}"
    )
    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    shapes, handles = install_shape_hooks(model)
    try:
        with torch.no_grad():
            t0 = time.time()
            out = model(x)
            dt = time.time() - t0
    finally:
        for h in handles:
            h.remove()

    print(f"\n  input shape       : {tuple(x.shape)}")
    if hasattr(out, "pred_score") and hasattr(out, "anomaly_map"):
        print(f"  pred_score shape  : {tuple(out.pred_score.shape)}")
        print(f"  anomaly_map shape : {tuple(out.anomaly_map.shape)}")
    print(f"  forward time      : {dt * 1000:.1f} ms ({args.device})")
    print_shape_table(shapes)


if __name__ == "__main__":
    main()
