#!/usr/bin/env python3
"""Convert Wan 2.1 TAEHV decoder (first-frame, NHWC) to CoreML."""
import argparse
import copy
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import coremltools as ct
except Exception as exc:
    raise SystemExit("coremltools is required. Install into your venv, e.g. `pip install coremltools`.") from exc

from taehv import Clamp, MemBlock, TGrow, conv


class DecoderOnlyTAEHV(nn.Module):
    """Decoder-only fork of TAEHV used for first-frame conversion."""

    def __init__(
        self,
        latent_channels: int = 16,
        patch_size: int = 1,
        decoder_time_upscale: tuple[bool, bool, bool] = (False, False, False),
        decoder_space_upscale: tuple[bool, bool, bool] = (True, True, True),
    ):
        super().__init__()
        self.patch_size = patch_size
        n_f = [256, 128, 64, 64]
        self.decoder = nn.Sequential(
            Clamp(), conv(latent_channels, n_f[0]), nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1), TGrow(n_f[0], 2 if decoder_time_upscale[0] else 1), conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[1] else 1), conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[2] else 1), conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True), conv(n_f[3], 3 * patch_size**2),
        )

    def patch_tgrow_layers(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Patch TGrow layers to match configured decoder stride if needed."""
        sd = dict(state_dict)
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if key in sd and sd[key].shape[0] > new_sd[key].shape[0]:
                    sd[key] = sd[key][-new_sd[key].shape[0]:]
        return sd

    def load_decoder_weights(self, checkpoint_path: str) -> None:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        raw = self.patch_tgrow_layers(raw)
        target = self.decoder.state_dict()
        dec_sd: dict[str, torch.Tensor] = {}
        for key, value in raw.items():
            if key.startswith("decoder."):
                dec_sd[key[len("decoder."):]] = value
            elif key in target:
                dec_sd[key] = value
        missing = sorted(set(target.keys()) - set(dec_sd.keys()))
        if missing:
            raise ValueError(f"Missing decoder keys in checkpoint: {missing[:5]} (total {len(missing)})")
        self.decoder.load_state_dict(dec_sd, strict=True)


class BaselineFirstFrameDecoder(nn.Module):
    """Original first-frame path: MemBlock(x, 0) and TGrow.stride==1 path."""

    def __init__(self, decoder: nn.Sequential, patch_size: int):
        super().__init__()
        self.decoder = decoder
        self.patch_size = patch_size

    def forward(self, x_nhwc: torch.Tensor) -> torch.Tensor:
        x = x_nhwc.permute(0, 3, 1, 2)
        for layer in self.decoder:
            if isinstance(layer, MemBlock):
                x = layer(x, x * 0)
            elif isinstance(layer, TGrow):
                x = layer.conv(x)
            else:
                x = layer(x)
        x = x.clamp(0, 1)
        if self.patch_size > 1:
            x = F.pixel_shuffle(x, self.patch_size)
        return x.permute(0, 2, 3, 1)


class MemBlockNoPast(nn.Module):
    """Fork of MemBlock for first-frame decode where past is always zero.

    It removes the concat([x, past]) and uses only the x-part of first conv weights.
    """

    def __init__(self, src: MemBlock):
        super().__init__()
        conv1_src = src.conv[0]
        conv2_src = src.conv[2]
        conv3_src = src.conv[4]

        if conv1_src.in_channels % 2 != 0:
            raise ValueError(f"Unexpected MemBlock first conv channels: {conv1_src.in_channels}")
        n_in = conv1_src.in_channels // 2
        n_out = conv1_src.out_channels

        self.conv1 = nn.Conv2d(n_in, n_out, 3, padding=1, bias=conv1_src.bias is not None)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = copy.deepcopy(conv2_src)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = copy.deepcopy(conv3_src)
        self.skip = copy.deepcopy(src.skip)
        self.act = nn.ReLU(inplace=True)

        with torch.no_grad():
            self.conv1.weight.copy_(conv1_src.weight[:, :n_in, :, :])
            if self.conv1.bias is not None and conv1_src.bias is not None:
                self.conv1.bias.copy_(conv1_src.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.conv3(y)
        return self.act(y + self.skip(x))


def build_optimized_decoder(decoder: nn.Sequential) -> nn.Sequential:
    layers: list[nn.Module] = []
    for layer in decoder:
        if isinstance(layer, MemBlock):
            layers.append(MemBlockNoPast(layer))
        elif isinstance(layer, TGrow):
            if layer.stride != 1:
                raise ValueError("First-frame optimized decoder expects TGrow stride=1")
            layers.append(copy.deepcopy(layer.conv))
        else:
            layers.append(copy.deepcopy(layer))
    return nn.Sequential(*layers)


class OptimizedFirstFrameDecoder(nn.Module):
    """NHWC first-frame decoder with MemBlock zero-past branch folded away."""

    def __init__(self, decoder: nn.Sequential, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.decoder = build_optimized_decoder(decoder)

    def forward(self, x_nhwc: torch.Tensor) -> torch.Tensor:
        x = x_nhwc.permute(0, 3, 1, 2)
        x = self.decoder(x)
        x = x.clamp(0, 1)
        if self.patch_size > 1:
            x = F.pixel_shuffle(x, self.patch_size)
        return x.permute(0, 2, 3, 1)


def build_decoder_only(checkpoint_path: str, disable_temporal_upscale: bool) -> DecoderOnlyTAEHV:
    decoder_time_upscale = (False, False, False) if disable_temporal_upscale else (False, True, True)
    model = DecoderOnlyTAEHV(
        latent_channels=16,
        patch_size=1,
        decoder_time_upscale=decoder_time_upscale,
    )
    model.load_decoder_weights(checkpoint_path)
    model.eval()
    return model


def check_exact_equivalence(
    baseline: nn.Module,
    optimized: nn.Module,
    input_hw: int,
    latent_channels: int,
) -> None:
    baseline.eval()
    optimized.eval()
    seeds = [0, 1, 2]
    with torch.no_grad():
        for seed in seeds:
            torch.manual_seed(seed)
            x = torch.randn(1, input_hw, input_hw, latent_channels, dtype=torch.float32)
            y_base = baseline(x)
            y_opt = optimized(x)
            if not torch.equal(y_base, y_opt):
                max_abs = float((y_base - y_opt).abs().max())
                raise SystemExit(
                    f"Optimized decoder mismatch vs baseline (seed={seed}, max_abs={max_abs:.6g})"
                )
    print("Equivalence check passed: optimized decoder is exactly equal to baseline on test inputs.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="taew2_1.pth", help="Path to taew2_1.pth")
    p.add_argument("--out-dir", default="coreml_out/wan21_decoder_firstframe_nhwc_96", help="Output directory")
    p.add_argument("--input-hw", type=int, default=96, help="Input latent H/W (default 96)")
    p.add_argument("--latent-channels", type=int, default=16, help="Latent channels (default 16)")
    p.add_argument("--disable-temporal-upscale", action="store_true", default=True, help="Disable temporal upscaling")
    p.add_argument("--enable-temporal-upscale", action="store_true", help="Enable temporal upscaling")
    p.add_argument("--io-float16", action="store_true", default=True, help="Use float16 I/O (default)")
    p.add_argument("--io-float32", action="store_true", help="Use float32 I/O")
    p.add_argument("--convert-to", default="mlprogram", choices=["mlprogram", "neuralnetwork"], help="CoreML format")
    p.add_argument("--min-deployment", default="macos13", choices=["macos13", "macos12", "macos11"], help="Minimum deployment target")
    p.add_argument("--skip-equivalence-check", action="store_true", help="Skip exact baseline-vs-optimized check")
    p.add_argument(
        "--memblock-mode",
        choices=["optimized", "fullmem"],
        default="optimized",
        help="`optimized` folds MemBlock(x,0) first conv channels; `fullmem` keeps original full MemBlock weights.",
    )
    p.add_argument("--compile", action="store_true", help="Also compile to .mlmodelc via xcrun")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.enable_temporal_upscale:
        args.disable_temporal_upscale = False
    if args.io_float32:
        args.io_float16 = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h = w = args.input_hw
    c = args.latent_channels
    io_dtype = np.float16 if args.io_float16 else np.float32
    dec_model = build_decoder_only(args.checkpoint, args.disable_temporal_upscale)
    baseline = BaselineFirstFrameDecoder(dec_model.decoder, dec_model.patch_size)
    if args.memblock_mode == "optimized":
        model = OptimizedFirstFrameDecoder(dec_model.decoder, dec_model.patch_size)
        if not args.skip_equivalence_check:
            check_exact_equivalence(baseline, model, h, c)
        model_stem = "wan21_decoder_firstframe_nhwc"
    else:
        # Keep full decoder weights (including the past branch of MemBlock conv1).
        model = baseline
        model_stem = "wan21_decoder_firstframe_nhwc_fullmem"
    model.eval()
    # Keep trace in fp32, then request fp16 lowering in ct.convert.
    example = torch.randn(1, h, w, c, dtype=torch.float32)

    with torch.no_grad():
        traced = torch.jit.trace(model, example, strict=False)

    inputs = [ct.TensorType(name="latent", shape=example.shape, dtype=io_dtype)]
    outputs = [ct.TensorType(name="image", dtype=io_dtype)]

    target_map = {
        "macos13": ct.target.macOS13,
        "macos12": ct.target.macOS12,
        "macos11": ct.target.macOS11,
    }
    minimum_deployment_target = target_map[args.min_deployment]

    convert_kwargs = dict(
        inputs=inputs,
        outputs=outputs,
        source="pytorch",
        convert_to=args.convert_to,
        minimum_deployment_target=minimum_deployment_target,
    )
    if args.convert_to == "mlprogram":
        convert_kwargs["compute_precision"] = (
            ct.precision.FLOAT16 if args.io_float16 else ct.precision.FLOAT32
        )
    mlmodel = ct.convert(traced, **convert_kwargs)

    if args.convert_to == "mlprogram":
        mlmodel_path = out_dir / f"{model_stem}.mlpackage"
    else:
        mlmodel_path = out_dir / f"{model_stem}.mlmodel"
    mlmodel.save(str(mlmodel_path))
    print(f"Saved {mlmodel_path}")

    if args.compile:
        mlmodelc_dir = out_dir / f"{model_stem}.mlmodelc"
        cmd = [
            "xcrun",
            "coremlcompiler",
            "compile",
            str(mlmodel_path),
            str(out_dir),
        ]
        print("Compiling:", " ".join(cmd))
        subprocess.run(cmd, check=False)
        if mlmodelc_dir.exists():
            print(f"Saved {mlmodelc_dir}")
        else:
            print("Compilation did not produce .mlmodelc; run the xcrun command manually.")


if __name__ == "__main__":
    main()
