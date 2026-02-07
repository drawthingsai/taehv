#!/usr/bin/env python3
"""Validate Wan 2.1 first-frame decoder: PyTorch vs CoreML."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import coremltools as ct
except Exception as exc:
    raise SystemExit("coremltools is required. Install into your venv, e.g. `pip install coremltools`.") from exc

from taehv import TAEHV, MemBlock, TGrow


class Wan21FirstFrameDecoder(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        disable_temporal_upscale: bool = True,
    ):
        super().__init__()
        if disable_temporal_upscale:
            decoder_time_upscale = (False, False, False)
        else:
            decoder_time_upscale = (False, True, True)
        self.taehv = TAEHV(
            checkpoint_path=checkpoint_path,
            decoder_time_upscale=decoder_time_upscale,
        )
        self.taehv.eval()

    def forward(self, x_nhwc: torch.Tensor) -> torch.Tensor:
        x = x_nhwc.permute(0, 3, 1, 2)
        for layer in self.taehv.decoder:
            if isinstance(layer, MemBlock):
                x = layer(x, x * 0)
            elif isinstance(layer, TGrow):
                x = layer.conv(x)
            else:
                x = layer(x)
        x = x.clamp(0, 1)
        if self.taehv.patch_size > 1:
            x = F.pixel_shuffle(x, self.taehv.patch_size)
        x = x.permute(0, 2, 3, 1)
        return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="taew2_1.pth")
    p.add_argument("--mlmodel", required=True, help="Path to .mlmodel or .mlmodelc")
    p.add_argument("--input-hw", type=int, default=96)
    p.add_argument("--latent-channels", type=int, default=16)
    p.add_argument("--disable-temporal-upscale", action="store_true", default=True)
    p.add_argument("--enable-temporal-upscale", action="store_true")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--save-dir", default="coreml_io/wan21_firstframe")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.enable_temporal_upscale:
        args.disable_temporal_upscale = False

    np_dtype = np.float16 if args.dtype == "float16" else np.float32

    h = w = args.input_hw
    c = args.latent_channels

    torch.manual_seed(0)
    # Keep PyTorch reference in fp32; cast only CoreML I/O.
    x = torch.randn(1, h, w, c, dtype=torch.float32)
    x_np = x.cpu().numpy().astype(np_dtype)

    model = Wan21FirstFrameDecoder(
        checkpoint_path=args.checkpoint,
        disable_temporal_upscale=args.disable_temporal_upscale,
    )
    model.eval()

    with torch.no_grad():
        y_pt = model(x).cpu().numpy()

    mlmodel = ct.models.MLModel(args.mlmodel)
    y_cm = mlmodel.predict({"latent": x_np})["image"]

    diff = y_cm.astype(np.float32) - y_pt.astype(np.float32)
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))

    print(f"MAE: {mae:.6g}")
    print(f"Max abs: {max_abs:.6g}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "input_shape.txt").write_text(f"1 {h} {w} {c}\n")
    (save_dir / "output_shape.txt").write_text(f"1 {y_pt.shape[1]} {y_pt.shape[2]} {y_pt.shape[3]}\n")
    x_np.tofile(save_dir / "input.raw")
    y_pt.astype(np_dtype).tofile(save_dir / "output_expected.raw")
    print(f"Saved input/output to {save_dir}")


if __name__ == "__main__":
    main()
