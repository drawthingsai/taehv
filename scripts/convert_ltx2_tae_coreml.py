#!/usr/bin/env python3
"""Convert LTX-2 TAEHV decoder to CoreML (one-shot or stateful chunked)."""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import coremltools as ct
except Exception as exc:
    raise SystemExit("coremltools is required. Install it into _env.") from exc

from taehv import Clamp, MemBlock, TGrow, TAEHV, conv


class DecoderOnlyTAEHV(nn.Module):
    """Decoder-only fork matching TAEHV decoder architecture."""

    def __init__(
        self,
        latent_channels: int,
        patch_size: int,
        decoder_time_upscale: tuple[bool, bool, bool],
        decoder_space_upscale: tuple[bool, bool, bool] = (True, True, True),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.frames_to_trim = (2 ** sum(1 for v in decoder_time_upscale if v)) - 1
        n_f = [256, 128, 64, 64]
        self.decoder = nn.Sequential(
            Clamp(), conv(latent_channels, n_f[0]), nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1), TGrow(n_f[0], 2 if decoder_time_upscale[0] else 1), conv(n_f[0], n_f[1], bias=False),
            MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[1] else 1), conv(n_f[1], n_f[2], bias=False),
            MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[2] else 1), conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True), conv(n_f[3], 3 * patch_size**2),
        )

    def patch_tgrow_layers(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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


def parse_decoder_time_upscale(value: str) -> tuple[bool, bool, bool]:
    s = value.strip().replace(",", "").replace("_", "")
    if len(s) != 3 or any(ch not in "01" for ch in s):
        raise argparse.ArgumentTypeError(
            f"Invalid decoder-time-upscale '{value}'. Use a 3-bit string like 111 or 000."
        )
    return tuple(ch == "1" for ch in s)  # type: ignore[return-value]


def decoder_time_upscale_tag(dtu: tuple[bool, bool, bool]) -> str:
    return "".join("1" if v else "0" for v in dtu)


def get_memblock_state_shapes(decoder: nn.Sequential, latent_hw: int) -> list[tuple[int, int, int]]:
    """Return [(C,H,W), ...] for each MemBlock input state, in decoder order."""
    h = latent_hw
    w = latent_hw
    shapes: list[tuple[int, int, int]] = []
    for layer in decoder:
        if isinstance(layer, MemBlock):
            c = layer.conv[0].in_channels // 2
            shapes.append((c, h, w))
        elif isinstance(layer, nn.Upsample):
            scale = layer.scale_factor
            if isinstance(scale, (tuple, list)):
                scale = scale[0]
            scale = int(scale)
            h *= scale
            w *= scale
    return shapes


def make_zero_states_nhwc(
    state_shapes_chw: list[tuple[int, int, int]],
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    return [torch.zeros(1, h, w, c, dtype=dtype, device=device) for (c, h, w) in state_shapes_chw]


class LTX2TemporalDecoderWrapper(nn.Module):
    """NTHWC latent -> NTHWC RGB for full temporal decode (trimmed)."""

    def __init__(self, dec_model: DecoderOnlyTAEHV):
        super().__init__()
        self.decoder = dec_model.decoder
        self.patch_size = dec_model.patch_size
        self.frames_to_trim = dec_model.frames_to_trim
        self.t_upscale = 2 ** sum(
            int(isinstance(layer, TGrow) and layer.stride == 2)
            for layer in self.decoder
        )

    def forward(self, x_nthwc: torch.Tensor) -> torch.Tensor:
        if x_nthwc.shape[0] != 1:
            raise RuntimeError("Only batch size 1 is supported for this fixed-shape wrapper.")

        # NTHWC -> TCHW (batch fixed to 1).
        x = x_nthwc.permute(0, 1, 4, 2, 3).squeeze(0)
        for layer in self.decoder:
            if isinstance(layer, MemBlock):
                past = torch.cat([x[:1] * 0, x[:-1]], dim=0)
                x = layer(x, past)
            elif isinstance(layer, TGrow):
                x = layer.conv(x)
                if layer.stride == 2:
                    # Equivalent to TGrow reshape, but avoids traced int/shape ops.
                    x1, x2 = torch.chunk(x, 2, dim=1)
                    x = torch.stack((x1, x2), dim=1).flatten(0, 1)
                elif layer.stride != 1:
                    raise RuntimeError(f"Unsupported TGrow stride {layer.stride}")
            else:
                x = layer(x)

        x = x.clamp(0, 1)
        if self.patch_size > 1:
            x = F.pixel_shuffle(x, self.patch_size)
        x = x.unsqueeze(0)
        x = x[:, self.frames_to_trim:]
        return x.permute(0, 1, 3, 4, 2)


class LTX2StatefulChunkDecoderWrapper(nn.Module):
    """Chunked NTHWC latent + per-MemBlock states -> raw NTHWC RGB + next states."""

    def __init__(self, dec_model: DecoderOnlyTAEHV):
        super().__init__()
        self.decoder = dec_model.decoder
        self.patch_size = dec_model.patch_size
        self.frames_to_trim = dec_model.frames_to_trim
        self.t_upscale = 2 ** sum(
            int(isinstance(layer, TGrow) and layer.stride == 2)
            for layer in self.decoder
        )
        self.memblock_count = sum(int(isinstance(layer, MemBlock)) for layer in self.decoder)

    def forward(self, x_nthwc: torch.Tensor, *state_nhwc: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if x_nthwc.shape[0] != 1:
            raise RuntimeError("Only batch size 1 is supported for this fixed-shape wrapper.")
        if len(state_nhwc) != self.memblock_count:
            raise RuntimeError(f"Expected {self.memblock_count} states but got {len(state_nhwc)}")

        x = x_nthwc.permute(0, 1, 4, 2, 3).squeeze(0)
        new_states_nchw: list[torch.Tensor] = []
        state_index = 0
        for layer in self.decoder:
            if isinstance(layer, MemBlock):
                prev_state = state_nhwc[state_index].permute(0, 3, 1, 2)
                past = torch.cat([prev_state, x[:-1]], dim=0)
                new_states_nchw.append(x[-1:].contiguous())
                x = layer(x, past)
                state_index += 1
            elif isinstance(layer, TGrow):
                x = layer.conv(x)
                if layer.stride == 2:
                    x1, x2 = torch.chunk(x, 2, dim=1)
                    x = torch.stack((x1, x2), dim=1).flatten(0, 1)
                elif layer.stride != 1:
                    raise RuntimeError(f"Unsupported TGrow stride {layer.stride}")
            else:
                x = layer(x)

        x = x.clamp(0, 1)
        if self.patch_size > 1:
            x = F.pixel_shuffle(x, self.patch_size)
        image_nthwc = x.unsqueeze(0).permute(0, 1, 3, 4, 2)
        state_outputs = [s.permute(0, 2, 3, 1) for s in new_states_nchw]
        return (image_nthwc, *state_outputs)


def run_stateful_chunked_decode_torch(
    wrapper: LTX2StatefulChunkDecoderWrapper,
    x_nthwc: torch.Tensor,
    chunk_t: int,
    state_shapes_chw: list[tuple[int, int, int]],
) -> torch.Tensor:
    """Run chunked decode with fixed chunk_t and carry state between calls."""
    total_t = int(x_nthwc.shape[1])
    states = make_zero_states_nhwc(state_shapes_chw, dtype=x_nthwc.dtype, device=x_nthwc.device)
    raw_chunks: list[torch.Tensor] = []
    for start in range(0, total_t, chunk_t):
        chunk = x_nthwc[:, start:start + chunk_t]
        real_t = int(chunk.shape[1])
        if real_t < chunk_t:
            pad = torch.zeros(
                1,
                chunk_t - real_t,
                x_nthwc.shape[2],
                x_nthwc.shape[3],
                x_nthwc.shape[4],
                dtype=x_nthwc.dtype,
                device=x_nthwc.device,
            )
            chunk = torch.cat([chunk, pad], dim=1)
        out = wrapper(chunk, *states)
        raw = out[0][:, :real_t * wrapper.t_upscale]
        states = list(out[1:])
        raw_chunks.append(raw)

    if not raw_chunks:
        return x_nthwc[:, :0, :, :, :3]
    raw_all = torch.cat(raw_chunks, dim=1)
    return raw_all[:, wrapper.frames_to_trim:]


def check_equivalence(
    checkpoint_path: str,
    wrapper: LTX2TemporalDecoderWrapper,
    latent_t: int,
    latent_hw: int,
    latent_channels: int,
    decoder_time_upscale: tuple[bool, bool, bool],
) -> None:
    if decoder_time_upscale != (True, True, True):
        print(
            "Skipping TAEHV.decode_video equivalence check because taehv.py forces LTX-2 decoder_time_upscale=(True,True,True)."
        )
        return
    ref = TAEHV(
        checkpoint_path=checkpoint_path,
        decoder_time_upscale=(True, True, True),
    ).eval()
    wrapper.eval()

    with torch.no_grad():
        for seed in (0, 1):
            torch.manual_seed(seed)
            x_nthwc = torch.randn(1, latent_t, latent_hw, latent_hw, latent_channels, dtype=torch.float32)
            x_ntchw = x_nthwc.permute(0, 1, 4, 2, 3)
            y_ref = ref.decode_video(x_ntchw, parallel=True, show_progress_bar=False).permute(0, 1, 3, 4, 2)
            y_new = wrapper(x_nthwc)
            max_abs = float((y_ref - y_new).abs().max())
            if max_abs > 3e-5:
                raise SystemExit(f"Equivalence failed (seed={seed}, max_abs={max_abs:.6g})")
    print("Equivalence check passed: wrapper matches TAEHV.decode_video within max_abs <= 3e-5.")


def check_stateful_equivalence(
    full_wrapper: LTX2TemporalDecoderWrapper,
    stateful_wrapper: LTX2StatefulChunkDecoderWrapper,
    latent_hw: int,
    latent_channels: int,
    chunk_t: int,
    state_shapes_chw: list[tuple[int, int, int]],
) -> None:
    lengths = sorted(set([
        1, 2, 3, 5, 7, 11, 13,
        max(chunk_t - 1, 1), chunk_t, chunk_t + 1, 2 * chunk_t + 1,
    ]))
    with torch.no_grad():
        for t_total in lengths:
            for seed in (0, 1):
                torch.manual_seed(1000 + 13 * t_total + seed)
                x_nthwc = torch.randn(1, t_total, latent_hw, latent_hw, latent_channels, dtype=torch.float32)
                y_full = full_wrapper(x_nthwc)
                y_chunk = run_stateful_chunked_decode_torch(stateful_wrapper, x_nthwc, chunk_t, state_shapes_chw)
                max_abs = float((y_full - y_chunk).abs().max())
                if max_abs > 5e-5:
                    raise SystemExit(
                        f"Stateful chunked equivalence failed (T={t_total}, seed={seed}, max_abs={max_abs:.6g})"
                    )
    print("Stateful chunked equivalence passed for variable lengths.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="taeltx_2.pth", help="Path to taeltx_2.pth")
    p.add_argument("--out-dir", default="coreml_out/ltx2_decoder_nthwc", help="Output directory")
    p.add_argument("--latent-hw", type=int, default=24, help="Latent H/W (e.g. 24 for 768 output)")
    p.add_argument("--latent-t", type=int, default=11, help="Latent frame count T (one-shot mode)")
    p.add_argument("--chunk-t", type=int, default=2, help="Fixed latent frame count per CoreML call (stateful mode)")
    p.add_argument("--latent-channels", type=int, default=128)
    p.add_argument(
        "--decoder-time-upscale",
        type=parse_decoder_time_upscale,
        default=(True, True, True),
        help="3-bit temporal upscale flags for decoder TGrow blocks (e.g. 111 or 000).",
    )
    p.add_argument("--stateful", action="store_true", help="Export chunked stateful model with activation I/O.")
    p.add_argument("--io-float16", action="store_true", default=True, help="Use float16 I/O")
    p.add_argument("--io-float32", action="store_true", help="Use float32 I/O")
    p.add_argument("--convert-to", default="mlprogram", choices=["mlprogram", "neuralnetwork"])
    p.add_argument("--min-deployment", default="macos13", choices=["macos13", "macos12", "macos11"])
    p.add_argument("--skip-equivalence-check", action="store_true")
    p.add_argument("--compile", action="store_true", help="Compile to .mlmodelc via coremlcompiler")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.io_float32:
        args.io_float16 = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    io_dtype = np.float16 if args.io_float16 else np.float32

    dec_model = DecoderOnlyTAEHV(
        latent_channels=args.latent_channels,
        patch_size=4,
        decoder_time_upscale=args.decoder_time_upscale,
    )
    dec_model.load_decoder_weights(args.checkpoint)
    dec_model.eval()

    target_map = {
        "macos13": ct.target.macOS13,
        "macos12": ct.target.macOS12,
        "macos11": ct.target.macOS11,
    }

    if args.stateful:
        state_shapes_chw = get_memblock_state_shapes(dec_model.decoder, latent_hw=args.latent_hw)
        full_wrapper = LTX2TemporalDecoderWrapper(dec_model).eval()
        wrapper = LTX2StatefulChunkDecoderWrapper(dec_model).eval()

        if not args.skip_equivalence_check:
            check_equivalence(
                checkpoint_path=args.checkpoint,
                wrapper=full_wrapper,
                latent_t=max(args.chunk_t, 11),
                latent_hw=args.latent_hw,
                latent_channels=args.latent_channels,
                decoder_time_upscale=args.decoder_time_upscale,
            )
            check_stateful_equivalence(
                full_wrapper=full_wrapper,
                stateful_wrapper=wrapper,
                latent_hw=args.latent_hw,
                latent_channels=args.latent_channels,
                chunk_t=args.chunk_t,
                state_shapes_chw=state_shapes_chw,
            )

        example_latent = torch.randn(
            1, args.chunk_t, args.latent_hw, args.latent_hw, args.latent_channels, dtype=torch.float32
        )
        example_states = make_zero_states_nhwc(
            state_shapes_chw=state_shapes_chw,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, (example_latent, *example_states), strict=False)

        inputs: list[ct.TensorType] = [ct.TensorType(name="latent", shape=example_latent.shape, dtype=io_dtype)]
        outputs: list[ct.TensorType] = [ct.TensorType(name="image", dtype=io_dtype)]
        for idx, state in enumerate(example_states):
            inputs.append(ct.TensorType(name=f"act_{idx}", shape=state.shape, dtype=io_dtype))
            outputs.append(ct.TensorType(name=f"act_{idx}_out", dtype=io_dtype))

        stem = f"ltx2_decoder_stateful_nthwc_chunk{args.chunk_t}_hw{args.latent_hw}"
    else:
        wrapper = LTX2TemporalDecoderWrapper(dec_model).eval()
        if not args.skip_equivalence_check:
            check_equivalence(
                checkpoint_path=args.checkpoint,
                wrapper=wrapper,
                latent_t=args.latent_t,
                latent_hw=args.latent_hw,
                latent_channels=args.latent_channels,
                decoder_time_upscale=args.decoder_time_upscale,
            )

        example = torch.randn(1, args.latent_t, args.latent_hw, args.latent_hw, args.latent_channels, dtype=torch.float32)
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, example, strict=False)

        inputs = [ct.TensorType(name="latent", shape=example.shape, dtype=io_dtype)]
        outputs = [ct.TensorType(name="image", dtype=io_dtype)]
        stem = f"ltx2_decoder_nthwc_t{args.latent_t}_hw{args.latent_hw}"

    dtu_tag = decoder_time_upscale_tag(args.decoder_time_upscale)
    if dtu_tag != "111":
        stem = f"{stem}_dtu{dtu_tag}"

    convert_kwargs = dict(
        inputs=inputs,
        outputs=outputs,
        source="pytorch",
        convert_to=args.convert_to,
        minimum_deployment_target=target_map[args.min_deployment],
    )
    if args.convert_to == "mlprogram":
        convert_kwargs["compute_precision"] = ct.precision.FLOAT16 if args.io_float16 else ct.precision.FLOAT32

    mlmodel = ct.convert(traced, **convert_kwargs)

    if args.convert_to == "mlprogram":
        mlmodel_path = out_dir / f"{stem}.mlpackage"
    else:
        mlmodel_path = out_dir / f"{stem}.mlmodel"
    mlmodel.save(str(mlmodel_path))
    print(f"Saved {mlmodel_path}")

    if args.compile:
        mlmodelc_path = out_dir / f"{stem}.mlmodelc"
        cmd = ["xcrun", "coremlcompiler", "compile", str(mlmodel_path), str(out_dir)]
        print("Compiling:", " ".join(cmd))
        subprocess.run(cmd, check=False)
        if mlmodelc_path.exists():
            print(f"Saved {mlmodelc_path}")
        else:
            print("Compilation did not produce .mlmodelc")


if __name__ == "__main__":
    main()
