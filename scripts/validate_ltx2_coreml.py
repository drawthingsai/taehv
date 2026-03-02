#!/usr/bin/env python3
"""Validate LTX-2 decoder CoreML output against PyTorch."""
import argparse
from pathlib import Path

import numpy as np
import torch

try:
    import coremltools as ct
except Exception as exc:
    raise SystemExit("coremltools is required.") from exc

from convert_ltx2_tae_coreml import (
    DecoderOnlyTAEHV,
    LTX2StatefulChunkDecoderWrapper,
    LTX2TemporalDecoderWrapper,
    get_memblock_state_shapes,
)


def parse_int_list(value: str) -> list[int]:
    out = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("Expected at least one integer.")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="taeltx_2.pth")
    p.add_argument("--mlmodel", required=True, help="Path to .mlpackage or .mlmodelc")
    p.add_argument("--latent-hw", type=int, default=24)
    p.add_argument("--latent-t", type=int, default=11, help="One-shot validation latent T")
    p.add_argument("--latent-t-list", default="1,2,3,5,7,11,13", help="Stateful validation latent lengths")
    p.add_argument("--chunk-t", type=int, default=2, help="Stateful model chunk size")
    p.add_argument("--latent-channels", type=int, default=128)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--stateful", action="store_true", help="Validate stateful chunked model.")
    p.add_argument("--save-dir", default="coreml_io/ltx2_decoder")
    return p.parse_args()


def validate_one_shot(
    args: argparse.Namespace,
    np_dtype: np.dtype,
    wrapper: LTX2TemporalDecoderWrapper,
    mlmodel: ct.models.MLModel,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(1, args.latent_t, args.latent_hw, args.latent_hw, args.latent_channels, dtype=torch.float32)
    x_np = x.cpu().numpy().astype(np_dtype)

    with torch.no_grad():
        y_pt = wrapper(x).cpu().numpy()

    y_cm = mlmodel.predict({"latent": x_np})["image"]
    diff = y_cm.astype(np.float32) - y_pt.astype(np.float32)
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    print(f"MAE: {mae:.6g}")
    print(f"RMSE: {rmse:.6g}")
    print(f"Max abs: {max_abs:.6g}")
    print(f"Input shape: {tuple(x_np.shape)}")
    print(f"Output shape: {tuple(y_pt.shape)}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "input_shape.txt").write_text(" ".join(str(v) for v in x_np.shape) + "\n")
    (save_dir / "output_shape.txt").write_text(" ".join(str(v) for v in y_pt.shape) + "\n")
    x_np.tofile(save_dir / "input.raw")
    y_pt.astype(np_dtype).tofile(save_dir / "output_expected.raw")
    print(f"Saved input/output to {save_dir}")


def validate_stateful_variable_lengths(
    args: argparse.Namespace,
    np_dtype: np.dtype,
    full_wrapper: LTX2TemporalDecoderWrapper,
    stateful_wrapper: LTX2StatefulChunkDecoderWrapper,
    mlmodel: ct.models.MLModel,
) -> None:
    lengths = parse_int_list(args.latent_t_list)
    state_shapes_chw = get_memblock_state_shapes(stateful_wrapper.decoder, latent_hw=args.latent_hw)

    worst_mae = 0.0
    worst_rmse = 0.0
    worst_max_abs = 0.0
    worst_t = -1

    for t_total in lengths:
        torch.manual_seed(1000 + t_total)
        x = torch.randn(1, t_total, args.latent_hw, args.latent_hw, args.latent_channels, dtype=torch.float32)
        with torch.no_grad():
            y_pt = full_wrapper(x).cpu().numpy().astype(np.float32)

        states_np = [np.zeros((1, h, w, c), dtype=np_dtype) for (c, h, w) in state_shapes_chw]
        raw_parts: list[np.ndarray] = []
        for start in range(0, t_total, args.chunk_t):
            x_chunk = x[:, start:start + args.chunk_t]
            real_t = int(x_chunk.shape[1])
            if real_t < args.chunk_t:
                x_pad = torch.zeros(
                    1,
                    args.chunk_t - real_t,
                    args.latent_hw,
                    args.latent_hw,
                    args.latent_channels,
                    dtype=x_chunk.dtype,
                )
                x_chunk = torch.cat([x_chunk, x_pad], dim=1)

            x_chunk_np = x_chunk.cpu().numpy().astype(np_dtype)
            inputs = {"latent": x_chunk_np}
            for idx, state in enumerate(states_np):
                inputs[f"act_{idx}"] = state

            out = mlmodel.predict(inputs)
            raw = out["image"][:, : real_t * stateful_wrapper.t_upscale]
            raw_parts.append(raw.astype(np.float32))
            states_np = [out[f"act_{idx}_out"].astype(np_dtype) for idx in range(len(states_np))]

        y_cm_raw = np.concatenate(raw_parts, axis=1)
        y_cm = y_cm_raw[:, stateful_wrapper.frames_to_trim:]
        y_cm = y_cm[:, : y_pt.shape[1]]

        diff = y_cm - y_pt
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        max_abs = float(np.max(np.abs(diff)))
        print(f"T={t_total}: MAE={mae:.6g} RMSE={rmse:.6g} MaxAbs={max_abs:.6g}")

        if max_abs > worst_max_abs:
            worst_t = t_total
        worst_mae = max(worst_mae, mae)
        worst_rmse = max(worst_rmse, rmse)
        worst_max_abs = max(worst_max_abs, max_abs)

    print(
        f"Worst across lengths: T={worst_t}, MAE={worst_mae:.6g}, RMSE={worst_rmse:.6g}, MaxAbs={worst_max_abs:.6g}"
    )


def main() -> None:
    args = parse_args()
    np_dtype = np.float16 if args.dtype == "float16" else np.float32

    dec_model = DecoderOnlyTAEHV(
        latent_channels=args.latent_channels,
        patch_size=4,
        decoder_time_upscale=(True, True, True),
    )
    dec_model.load_decoder_weights(args.checkpoint)
    full_wrapper = LTX2TemporalDecoderWrapper(dec_model).eval()
    stateful_wrapper = LTX2StatefulChunkDecoderWrapper(dec_model).eval()
    mlmodel = ct.models.MLModel(args.mlmodel)

    if args.stateful:
        validate_stateful_variable_lengths(
            args=args,
            np_dtype=np_dtype,
            full_wrapper=full_wrapper,
            stateful_wrapper=stateful_wrapper,
            mlmodel=mlmodel,
        )
    else:
        validate_one_shot(
            args=args,
            np_dtype=np_dtype,
            wrapper=full_wrapper,
            mlmodel=mlmodel,
        )


if __name__ == "__main__":
    main()
