# Learnings / Handoff Notes for TAESD / CoreML Work

This file captures the full context of what was discussed and implemented in the `taesd` repo, including decisions, scripts, artifacts, benchmarks, and current state. Use this as pre-context for future work (e.g., in `taehv`).

## High-Level Goals and Conclusions
- Goal: Optimize CoreML/ANE performance for FLUX.1/FLUX.2 TAEs, validate layouts and quantization, and create efficient artifacts for multiple resolutions.
- **NHWC vs NCHW**: We explored NHWC input/output and CoreML tracing behavior. Apple documentation indicates ANE prefers **channels-first** internally. NHWC-only pipelines did not produce net benefit; transposes may be optimized away by the compiler, but overall **channels-first (NCHW) remains preferred** at runtime. However, we still created **NHWC I/O wrappers** for experimentation and for easy use with NHWC inputs.
- **Flexible shapes**: Enumerated shapes (768,1024,...) were convenient, but **RAM usage was too high**. Fixed-shape models per resolution are preferred.
- **Quantization**:
  - Post-training palettization (4-bit/8-bit) and int8 weight-only were **slower** than FP16 baseline.
  - QAT with int8 activations+weights (coremltools LinearQuantizer) gave a **modest speedup** (~36ms vs ~38ms at 1024 for FLUX.1), but quality degradation was observed.
- **Benchmarking**: Swift harness is the reliable way to measure CoreML/ANE performance; sandboxed Python is too slow and misleading.

---

## Key Files & Scripts Added / Modified

### Conversion and CoreML Build
- `scripts/convert_flux_vae_coreml.py`
  - Options added: `--convert-to`, `--io-precision`, `--input-hw-list`, `--palettize-nbits`, `--quantize-weight-int8`.
  - Supports enumerated input shapes via `ct.EnumeratedShapes`.
  - For MLProgram conversion, uses macOS13 target; for neuralnetwork uses macOS11.
  - Validation now supports **multiple input sizes** in `--input-hw-list` (iterates through all sizes).
  - Added NHWC wrapper path using `DecoderNHWCWrapper` (NHWC input/output with internal transpose).

### Benchmarking
- `scripts/run_coreml_instruments.swift`
  - Swift benchmark harness for CoreML `.mlmodelc`.
  - Supports compute units, dtype, input-hw, latent channels, warmup, iters.
  - **Updated** to support `--layout nhwc|nchw` so input shapes are correct for NHWC I/O models.

- `scripts/benchmark_coreml_flux1.py` / `scripts/benchmark_coreml_flux1_python.py`
  - Python benchmarks for CoreML (used for quick sanity but slower in sandbox).

### Quantization
- `scripts/qat_flux_decoder_int8.py`
  - QAT pipeline using coremltools `LinearQuantizer`.
  - Uses a small dataset (downloaded separately).
  - Requires macOS14 target to keep quant ops.

### Datasets / Roundtrip
- `scripts/download_soa_subset.py`
  - Downloads ~20 images from `madebyollin/soa-aesthetic` (via datasets + requests).

- `scripts/roundtrip_compare.py`
  - Saves round-trip outputs and metrics.
  - Updated to support `--variant flux1|flux2` and to use current artifacts.
  - Produces `metrics.csv` with MAE/RMSE/PSNR.

### Notes on Dependencies
- Installed in `_env`: `requests`, `datasets`, `pillow`.

---

## Benchmarking Summary (User Swift Runs)

**Baseline FP16 NCHW (ANE, CPU+NE, local Swift)**
- FLUX.1 @768: ~20–21 ms
- FLUX.1 @1024: ~36–38 ms
- FLUX.2 @768: ~21–23 ms
- FLUX.2 @1024: ~38–41 ms

**Quantization**
- Palettization 4/8-bit and int8 weight-only: ~40 ms (slower)
- QAT int8 (weights+activations): ~36 ms (better than fp16 baseline but quality drop)

**NHWC I/O**
- Works, but no clear performance benefit. CPU/ANE still prefer channels-first kernels.

---

## Model Artifact Decisions & Organization

### Fixed-Shape NHWC FP16 (Final Preferred)
- Generated fixed-shape models per resolution because flexible shape had **excessive RAM**.
- Resolutions covered: **768, 1024, 1280, 1536, 1792, 2048**.

### New Directory Layout (for dedup strategy)
You asked to reorganize artifacts as:
```
flux_1_tae/
  768.mlmodelc/
  1024.mlmodelc/
  1280.mlmodelc/
  1536.mlmodelc/
  1792.mlmodelc/
  2048.mlmodelc/
  weight.bin

flux_2_tae/
  768.mlmodelc/
  1024.mlmodelc/
  1280.mlmodelc/
  1536.mlmodelc/
  1792.mlmodelc/
  2048.mlmodelc/
  weight.bin
```
- All `weights/weight.bin` **removed** from each `.mlmodelc` directory.
- `weight.bin` placed at top-level for each variant.
- This layout was created **without recompiling**, by copying existing `.mlmodelc` from `coreml_out`.

### Weight File Identity Check
- Verified SHA256 of `weight.bin` across all fixed-shape models:
  - FLUX.1: **all identical**
    - SHA256 = `3f0dc80e4e4f52cc36e10a9a6d0604ba80f87dc5f1d73034ea811f7690d49b9f`
  - FLUX.2: **all identical**
    - SHA256 = `d9d30f8c992b265b3a08c19e2bcadff8da06fb3f2f96bb2dd7b8ee01c1519058`

---

## Current CoreML Output Directory State
Current `coreml_out` (some older artifacts still present; main ones used):
- `coreml_out/ane_opt_fp16_nhwc_768/`
- `coreml_out/ane_opt_fp16_nhwc_1024/`
- `coreml_out/ane_opt_fp16_nhwc_1280/`
- `coreml_out/ane_opt_fp16_nhwc_1536/`
- `coreml_out/ane_opt_fp16_nhwc_1792/`
- `coreml_out/ane_opt_fp16_nhwc_2048/`

Flexible shape artifacts exist but are not preferred due to RAM:
- `coreml_out/ane_opt_flex_fp16_nhwc`
- `coreml_out/ane_opt_flex_fp16_nhwc_768_1024`

---

## Swift Benchmark Command (NHWC I/O)
Example (FLUX.1 @2048):
```
swiftc -O -o /tmp/run_coreml scripts/run_coreml_instruments.swift

/tmp/run_coreml coreml_out/ane_opt_fp16_nhwc_2048/flux1_decoder_nhwc_float16_iofloat16.mlmodelc \
  --compute-units cpu_ne --dtype float16 --input-hw 2048 --iters 200 --layout nhwc
```
For FLUX.2 add `--latent-channels 32`.

---

## Instrumentation / Core ML Tooling Notes
- Core ML Instruments report **latency and dispatch** but **no ANE utilization counters**.
- Performance report in Xcode gives per-op cost but not utilization.
- `run_coreml_instruments.swift` is the canonical benchmarking tool.

---

## Roundtrip and Quality Check
- `scripts/roundtrip_compare.py` can produce outputs and metrics.
- Outputs saved in directories like:
  - `coreml_io/roundtrip_compare_flux1_flex/`
  - `coreml_io/roundtrip_compare_flux2_flex/`
- Metrics include MAE, RMSE, PSNR.

---

## Decision Notes
- Flexible shapes are convenient but use too much RAM.
- Fixed shape models with shared weights are preferred.
- Quantization only marginally improved speed and degraded quality; FP16 is baseline.
- NHWC I/O wrappers exist, but channel-first inside is still preferred for ANE.

---

## Worklog Summary (Previously in WORKLOG.md)
- Added conversion options in `convert_flux_vae_coreml.py` for layout, IO precision, MLProgram vs NeuralNetwork, enumerated shapes, palettization, int8 weight-only.
- Implemented Swift benchmark harness (`run_coreml_instruments.swift`) and Python benchmark scripts.
- Confirmed sandboxed Python is too slow, Swift/Python local is correct.
- Built flex models and validated minimal perf impact (but higher RAM).
- Explored NHWC and later concluded ANE prefers channels-first.
- Ran palettization and int8 weight-only (no speedup).
- Implemented QAT int8 activations + weights; modest speedup but quality loss.
- Created roundtrip comparison pipeline and saved outputs.
- Later focused on fixed-shape NHWC FP16 for 768–2048 with validated accuracy.

---

## Where Things Stand Now
- Fixed-shape NHWC FP16 models exist for each size 768–2048.
- `flux_1_tae` and `flux_2_tae` directories contain per-size `.mlmodelc` and a top-level `weight.bin` (deduped).
- Next step is to update runtime code (in the next repo) to re-link or load `weight.bin` into the `.mlmodelc` bundles before running, or to copy it in at install time.

---

# TAEHV / Wan 2.1 / QwenImage CoreML Work (This Repo)

This section records the work done in `taehv` for Wan 2.1 first-frame decode and QwenImage-style fixed-shape artifacts.

## Scope and Goal
- Target model: Wan 2.1 TAE (`taew2_1.pth`) for first-frame decode only.
- CoreML I/O layout: NHWC.
- Fixed latent input shape per model (e.g. 96x96x16 for 768 image-space).
- Validate outputs in both Python and Swift.
- Build multiple fixed-shape models and dedupe `weight.bin`.

## Files Added / Updated

### New docs
- `TAEHV_ARCHITECTURE.md`
  - Short architecture note for `taehv.py` blocks and CoreML-relevant behavior.

### Conversion and validation scripts
- `scripts/convert_wan21_tae_coreml.py`
  - Converts first-frame decoder to CoreML (`mlprogram`) and optionally compiles `.mlmodelc`.
  - Uses NHWC input/output.
  - Uses fp32 trace + fp16 conversion pattern (`compute_precision=FLOAT16`) to avoid fp16 tracing issues.
  - Now uses a **decoder-only fork** (no encoder module in conversion graph).
  - Includes first-frame optimization:
    - `MemBlockNoPast`: folds `MemBlock(x, 0)` by removing `past` concat branch and slicing first conv input channels.
    - Replaces `TGrow(stride=1)` with underlying conv in first-frame path.
  - Includes exact equivalence gate:
    - baseline first-frame path vs optimized path must pass `torch.equal` on deterministic test inputs.

- `scripts/validate_wan21_coreml.py`
  - Python numerical comparison between PyTorch reference and CoreML model output.
  - Saves raw I/O tensors for Swift-side validation.

- `scripts/validate_wan21_coreml.swift`
  - Swift checker for `.mlmodelc` output vs saved expected tensor (MAE / max abs).

- `scripts/benchmark_wan21_firstframe.swift`
  - Swift runtime benchmark utility for first-frame model latency.
  - Reports mean/median/p90/p95/min/max latency.
  - Supports compute units (`cpu_ne`, `all`, `cpu_only`, `cpu_gpu`).

## Core Conversion Issue and Fix

Observed failure when trying fp16 internal tracing:
- Error:
  - `ValueError: ... bias has dtype fp16 whereas x has dtype fp32` at CoreML conv conversion.

Working fix (mirrors `convert_flux_vae_coreml.py` approach):
- Trace Torch model in **fp32**.
- Set CoreML I/O dtype via `ct.TensorType(dtype=np.float16)`.
- Set `compute_precision=ct.precision.FLOAT16` at `ct.convert(...)` for MLProgram.

This resolved conversion failures for the first-frame model.

## Accuracy / Validation Results

### CoreML vs PyTorch (first-frame model, float16 I/O)
- Python validation:
  - MAE ~ `3.72e-4`
  - Max abs ~ `6.44e-3`
- Swift validation:
  - close agreement within expected fp16 rounding behavior.

### `lighttaew2_1.pth` vs `taew2_1.pth` roundtrip check
- Ran roundtrip (`encode_video -> decode_video`) on `examples/decoded_taew2_1.mp4`.
- Result: `lighttaew2_1.pth` was **less accurate** than `taew2_1.pth` on tested clip.
  - `taew2_1.pth`: MAE `0.011538`, RMSE `0.021184`, PSNR `33.4799 dB`
  - `lighttaew2_1.pth`: MAE `0.015667`, RMSE `0.027019`, PSNR `31.3668 dB`
  - Delta (light - base): dMAE `+0.004129`, dRMSE `+0.005835`, dPSNR `-2.1131 dB`

## First-frame Model Generation (Fixed Shapes)

Generated fixed-shape models for image resolutions:
- 768, 1024, 1280, 1536, 1792, 2048

Corresponding latent H/W (`input-hw` in converter):
- 96, 128, 160, 192, 224, 256

Build outputs:
- `coreml_out/qwenimage_tae_build/768/wan21_decoder_firstframe_nhwc.mlmodelc`
- `coreml_out/qwenimage_tae_build/1024/wan21_decoder_firstframe_nhwc.mlmodelc`
- `coreml_out/qwenimage_tae_build/1280/wan21_decoder_firstframe_nhwc.mlmodelc`
- `coreml_out/qwenimage_tae_build/1536/wan21_decoder_firstframe_nhwc.mlmodelc`
- `coreml_out/qwenimage_tae_build/1792/wan21_decoder_firstframe_nhwc.mlmodelc`
- `coreml_out/qwenimage_tae_build/2048/wan21_decoder_firstframe_nhwc.mlmodelc`

## Weight Dedup Check and Final Layout

Verified that all six compiled models had identical weight file:
- SHA256 (`weights/weight.bin`) for all six:
  - `ef0e99b5e29ebc039b6859ed7e5b7e6dafd127d4d087b4c0e24e1c14445fe11b`

Created deduped final directory:
```
qwenimage_tae/
  768.mlmodelc/
  1024.mlmodelc/
  1280.mlmodelc/
  1536.mlmodelc/
  1792.mlmodelc/
  2048.mlmodelc/
  weight.bin
```

Dedup steps applied:
- Removed `weights/weight.bin` from each `qwenimage_tae/*.mlmodelc`.
- Kept a single shared copy at `qwenimage_tae/weight.bin`.

## Runtime Benchmark Command (Swift, ANE path)

Compile:
```
CLANG_MODULE_CACHE_PATH=/tmp/clang-module-cache \
swiftc -O scripts/benchmark_wan21_firstframe.swift -o /tmp/benchmark_wan21_firstframe
```

Run (CPU+NE):
```
/tmp/benchmark_wan21_firstframe \
  coreml_out/wan21_decoder_firstframe_nhwc_96/wan21_decoder_firstframe_nhwc.mlmodelc \
  --compute-units cpu_ne --dtype float16 --warmup 20 --iters 200
```

## Notes for Fork / Commit
- Conversion path is now decoder-only and first-frame optimized.
- Equivalence guard is built into conversion script (can be skipped with `--skip-equivalence-check` for batch builds).
- Final deduped artifacts for distribution are under `qwenimage_tae/`.
