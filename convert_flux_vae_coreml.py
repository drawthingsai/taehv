#!/usr/bin/env python3
"""
Convert TAESD FLUX.1/FLUX.2 decoder weights to Core ML (.mlmodel + .mlmodelc)
and validate outputs against PyTorch.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TMPDIR", "/tmp")

import numpy as np
import torch

try:
    import coremltools as ct
    from coremltools.optimize.torch.palettization import (
        PostTrainingPalettizer,
        PostTrainingPalettizerConfig,
    )
    from coremltools.optimize.torch.quantization import (
        PostTrainingQuantizer,
        PostTrainingQuantizerConfig,
    )
except Exception as exc:  # pragma: no cover - import-time error reporting
    raise SystemExit(
        "coremltools is required. Install it inside your venv, e.g.:\n"
        "  pip install coremltools"
    ) from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from taesd import TAESD
from taesd_nhwc import TAESD as TAESD_NHWC


FLUX_VARIANTS = {
    "flux1": {
        "latent_channels": 16,
        "arch_variant": None,
        "decoder_path": "taef1_decoder.pth",
    },
    "flux2": {
        "latent_channels": 32,
        "arch_variant": "flux_2",
        "decoder_path": "taef2_decoder.pth",
    },
}


class DecoderNHWCWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, x):
        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2)
        y = self.decoder(x)
        # NCHW -> NHWC
        return y.permute(0, 2, 3, 1)


def latent_hw(input_hw: int) -> int:
    if input_hw % 8 != 0:
        raise ValueError(f"input_hw must be divisible by 8 (got {input_hw})")
    return input_hw // 8


def build_model(variant: str, nhwc_model: bool = False) -> torch.nn.Module:
    cfg = FLUX_VARIANTS[variant]
    taesd_cls = TAESD_NHWC if nhwc_model else TAESD
    taesd = taesd_cls(
        encoder_path=None,
        decoder_path=cfg["decoder_path"],
        latent_channels=cfg["latent_channels"],
        arch_variant=cfg["arch_variant"],
    )
    model = taesd.decoder
    model.eval()
    return model


def make_input(layout: str, latent_channels: int, input_hw: int, dtype: np.dtype) -> np.ndarray:
    rng = np.random.default_rng(0)
    lat_hw = latent_hw(input_hw)
    shape = (1, lat_hw, lat_hw, latent_channels) if layout == "nhwc" else (1, latent_channels, lat_hw, lat_hw)
    return rng.uniform(-1.0, 1.0, size=shape).astype(dtype)


def make_input_type(
    layout: str,
    latent_channels: int,
    input_hw: int,
    dtype: np.dtype | None,
    input_hw_list: list[int] | None,
) -> ct.TensorType:
    if input_hw_list:
        shapes = []
        for hw in input_hw_list:
            lat_hw = latent_hw(hw)
            shape = (1, lat_hw, lat_hw, latent_channels) if layout == "nhwc" else (1, latent_channels, lat_hw, lat_hw)
            shapes.append(shape)
        shape_spec = ct.EnumeratedShapes(shapes=shapes)
    else:
        lat_hw = latent_hw(input_hw)
        shape_spec = (1, lat_hw, lat_hw, latent_channels) if layout == "nhwc" else (1, latent_channels, lat_hw, lat_hw)
    kwargs = {"name": "latent", "shape": shape_spec}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return ct.TensorType(**kwargs)


def wrap_for_layout(model: torch.nn.Module, layout: str, nhwc_model: bool) -> torch.nn.Module:
    if layout == "nchw":
        return model
    return model if nhwc_model else DecoderNHWCWrapper(model)


def save_and_compile(mlmodel: ct.models.MLModel, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = mlmodel.get_spec()
    is_mlprogram = spec.WhichOneof("Type") == "mlProgram"
    ext = "mlpackage" if is_mlprogram else "mlmodel"
    model_path = out_dir / f"{stem}.{ext}"
    mlmodel.save(model_path)
    mlmodelc_path = out_dir / f"{stem}.mlmodelc"
    if mlmodelc_path.exists():
        shutil.rmtree(mlmodelc_path)
    try:
        compiled_path = Path(
            ct.models.utils.compile_model(str(model_path), destination_path=str(mlmodelc_path))
        )
        return compiled_path
    except Exception as exc:
        # Fall back to coremlc when MLModelProxy compile fails in sandboxed environments.
        subprocess.run(
            ["xcrun", "coremlc", "compile", str(model_path), str(out_dir)],
            check=True,
        )
        return mlmodelc_path


def get_io_names(mlmodel: ct.models.MLModel) -> tuple[str, str]:
    spec = mlmodel.get_spec()
    input_name = spec.description.input[0].name
    output_name = spec.description.output[0].name
    return input_name, output_name


def validate_single(
    mlmodel: ct.models.MLModel,
    torch_model: torch.nn.Module,
    input_np: np.ndarray,
    layout: str,
    atol: float,
    input_name: str,
    output_name: str,
):
    coreml_out = mlmodel.predict({input_name: input_np})[output_name]
    if coreml_out.dtype != np.float32:
        coreml_out = coreml_out.astype(np.float32)

    with torch.no_grad():
        torch_in = torch.from_numpy(input_np.astype(np.float32))
        torch_out = torch_model(torch_in).cpu().numpy()

    diff = coreml_out - torch_out
    max_abs = float(np.max(np.abs(diff)))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ok = max_abs <= atol

    report = {
        "layout": layout,
        "max_abs_error": max_abs,
        "mae": mae,
        "rmse": rmse,
        "atol": atol,
        "pass": ok,
    }
    return ok, report


def convert_and_validate(args):
    cfg = FLUX_VARIANTS[args.variant]
    torch_model = build_model(args.variant, nhwc_model=args.nhwc_model)
    torch_model = wrap_for_layout(torch_model, args.layout, args.nhwc_model)
    print(f"torch_model {torch_model}")

    if args.quantize_weight_int8:
        qconfig = PostTrainingQuantizerConfig.from_dict(
            {
                "global_config": {
                    "weight_dtype": "int8",
                },
            }
        )
        quantizer = PostTrainingQuantizer(torch_model, qconfig)
        torch_model = quantizer.compress(inplace=False)
        torch_model.eval()

    if args.palettize_nbits is not None:
        config = PostTrainingPalettizerConfig.from_dict(
            {
                "global_config": {
                    "n_bits": args.palettize_nbits,
                },
            }
        )
        palettizer = PostTrainingPalettizer(torch_model, config)
        torch_model = palettizer.compress(inplace=False)
        torch_model.eval()

    io_dtype = None if args.io_precision == "auto" else (np.float16 if args.io_precision == "float16" else np.float32)
    input_type = make_input_type(
        args.layout,
        cfg["latent_channels"],
        args.input_hw,
        io_dtype,
        args.input_hw_list,
    )
    # Use first shape in list for validation input.
    input_hw = args.input_hw_list[0] if args.input_hw_list else args.input_hw
    input_np = make_input(args.layout, cfg["latent_channels"], input_hw, io_dtype or np.float32)

    precision_map = {
        "float32": ct.precision.FLOAT32,
        "float16": ct.precision.FLOAT16,
    }

    example_input = torch.from_numpy(input_np.astype(np.float32))
    traced = torch.jit.trace(torch_model, example_input, strict=False)
    traced.eval()

    min_target = ct.target.iOS18 if args.convert_to == "mlprogram" else ct.target.macOS11
    convert_kwargs = dict(
        inputs=[input_type],
        source="pytorch",
        convert_to=args.convert_to,
        minimum_deployment_target=min_target,
    )
    if args.convert_to == "mlprogram":
        convert_kwargs["compute_precision"] = precision_map[args.precision]
    if io_dtype is not None:
        convert_kwargs["outputs"] = [ct.TensorType(dtype=io_dtype)]
    mlmodel = ct.convert(traced, **convert_kwargs)

    stem = f"{args.variant}_decoder_{args.layout}_{args.precision}"
    if args.input_hw_list:
        stem += "_hw" + "_".join(str(x) for x in args.input_hw_list)
    if args.palettize_nbits is not None:
        stem += f"_pal{args.palettize_nbits}"
    if args.quantize_weight_int8:
        stem += "_int8w"
    if args.convert_to != "mlprogram":
        stem += f"_{args.convert_to}"
    if args.io_precision != "auto":
        stem += f"_io{args.io_precision}"
    input_name, output_name = get_io_names(mlmodel)
    mlmodelc_path = save_and_compile(mlmodel, args.out_dir, stem)

    report = None
    if args.validate:
        try:
            mlmodel = ct.models.MLModel(str(mlmodelc_path), compute_units=ct.ComputeUnit.CPU_ONLY)
        except Exception:
            mlmodel = ct.models.CompiledMLModel(str(mlmodelc_path), compute_units=ct.ComputeUnit.CPU_ONLY)

        reports = []
        if args.input_hw_list:
            for hw in args.input_hw_list:
                input_np = make_input(args.layout, cfg["latent_channels"], hw, io_dtype or np.float32)
                ok, rep = validate_single(
                    mlmodel,
                    torch_model,
                    input_np,
                    args.layout,
                    args.atol,
                    input_name,
                    output_name,
                )
                rep["input_hw"] = hw
                reports.append(rep)
                if not ok:
                    raise SystemExit(f"Validation failed (max_abs_error > {args.atol}). Report: {rep}")
            report = {"layout": args.layout, "reports": reports}
        else:
            ok, report = validate_single(
                mlmodel,
                torch_model,
                input_np,
                args.layout,
                args.atol,
                input_name,
                output_name,
            )
            if not ok:
                raise SystemExit(f"Validation failed (max_abs_error > {args.atol}). Report: {report}")
    return mlmodelc_path, report


def main():
    parser = argparse.ArgumentParser(description="Convert TAESD FLUX VAEs to Core ML (decoder only).")
    parser.add_argument("--variant", choices=FLUX_VARIANTS.keys(), required=True)
    parser.add_argument("--layout", choices=["nchw", "nhwc"], default="nchw")
    parser.add_argument("--nhwc-model", action="store_true", help="Use NHWC-wrapped model implementation.")
    parser.add_argument("--precision", choices=["float32", "float16"], default="float32")
    parser.add_argument("--convert-to", choices=["mlprogram", "neuralnetwork"], default="mlprogram")
    parser.add_argument("--io-precision", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--input-hw", type=int, default=768)
    parser.add_argument(
        "--input-hw-list",
        type=str,
        help="Comma-separated input image sizes (e.g. 768,1024) to create enumerated shapes.",
    )
    parser.add_argument(
        "--palettize-nbits",
        type=int,
        default=None,
        help="Apply post-training palettization with the given bit-width (e.g. 4).",
    )
    parser.add_argument(
        "--quantize-weight-int8",
        action="store_true",
        help="Apply post-training int8 weight quantization (activations remain float).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("coreml_out"))
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    if args.input_hw_list:
        args.input_hw_list = [int(x) for x in args.input_hw_list.split(",") if x.strip()]
        if args.convert_to != "mlprogram":
            raise SystemExit("--input-hw-list requires --convert-to mlprogram")
    if args.atol is None:
        args.atol = 1e-3 if args.precision == "float32" else 1e-2

    mlmodelc_path, report = convert_and_validate(args)
    print(f"Saved compiled model: {mlmodelc_path}")
    if report:
        print(json.dumps(report, indent=2))
        if args.report_json:
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
