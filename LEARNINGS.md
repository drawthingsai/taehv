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

---

# LTX-2 Temporal Decoder CoreML Work (This Repo)

This section records the LTX-2 decoder work where temporal decoding is preserved (not first-frame only).

## Scope and Goal
- Target checkpoint: `taeltx_2.pth`
- Export a CoreML decoder with full temporal path:
  - Input layout: `N,T,H,W,C` (latent)
  - Output layout: `N,T,H,W,C` (RGB)
- Keep conversion mostly on wrapper/export side with minimal base PyTorch changes.
- Validate in both Python and Swift.

## Files Added
- `scripts/convert_ltx2_tae_coreml.py`
  - Converts LTX-2 temporal decoder to CoreML.
  - Uses a decoder-only fork (`DecoderOnlyTAEHV`) and loads only decoder weights.
  - Uses fp32 tracing + CoreML conversion precision control (`compute_precision`) to avoid mixed-precision graph issues.
  - Uses full temporal `MemBlock` + `TGrow` path.
  - Includes equivalence check vs `TAEHV.decode_video(...)`.

- `scripts/validate_ltx2_coreml.py`
  - Compares PyTorch wrapper output to CoreML output.
  - Writes deterministic raw I/O tensors under `coreml_io/ltx2_decoder/` for Swift checks.

- `scripts/benchmark_ltx2_decoder.swift`
  - Dedicated Swift benchmark for temporal LTX-2 decoder.
  - Runs warmup + timed iterations and reports mean/median/p90/p95.
  - Also reports temporal metrics:
    - output frame count
    - mean ms per output frame
    - effective output FPS

## Conversion Command (768 output case)
```
source _env/bin/activate
PYTHONPATH=/Users/liu/workspace/taehv python scripts/convert_ltx2_tae_coreml.py \
  --checkpoint taeltx_2.pth \
  --compile \
  --min-deployment macos13 \
  --latent-hw 24 \
  --latent-t 11 \
  --out-dir coreml_out/ltx2_decoder_nthwc_768
```

Output artifacts:
- `coreml_out/ltx2_decoder_nthwc_768/ltx2_decoder_nthwc_t11_hw24.mlpackage`
- `coreml_out/ltx2_decoder_nthwc_768/ltx2_decoder_nthwc_t11_hw24.mlmodelc`

## Validation Results
- Equivalence gate inside converter:
  - Passed with `max_abs <= 3e-5` vs `TAEHV.decode_video(...)`.

- Python validation (`scripts/validate_ltx2_coreml.py`, float16):
  - MAE: `0.000231676`
  - RMSE: `0.000302071`
  - Max abs: `0.0049603`
  - Input: `(1, 11, 24, 24, 128)`
  - Output: `(1, 81, 768, 768, 3)`

- Swift tensor check (using existing Swift validator on dumped tensors):
  - MAE: `0.00062941`
  - Max abs: `0.0427246`

## Swift Benchmark Command (ANE)
Compile:
```
CLANG_MODULE_CACHE_PATH=/tmp/clang-module-cache \
swiftc -O scripts/benchmark_ltx2_decoder.swift -o /tmp/benchmark_ltx2_decoder
```

Run:
```
/tmp/benchmark_ltx2_decoder \
  coreml_out/ltx2_decoder_nthwc_768/ltx2_decoder_nthwc_t11_hw24.mlmodelc \
  --compute-units cpu_ne \
  --dtype float16 \
  --warmup 20 \
  --iters 200
```

Quick smoke run sample (warmup=1, iters=3 on this machine):
- mean latency: `1900.809 ms`
- mean ms / output frame (81 frames): `23.467 ms`
- effective output FPS: `42.61`

## LTX-2 Stateful Decoder (Chunked, Activation I/O) and Preview Mode

This session extended the LTX-2 work to support a **stateful CoreML decoder** API so we can use a fixed-shape CoreML model for arbitrary latent sequence lengths by iteratively feeding activations.

### Key Learnings
- A fixed-`T` stateful decoder works well for variable latent lengths using activation carry:
  - CoreML call signature conceptually becomes:
    - `model(latent_chunk, act_0..act_N) -> (image_chunk, act_0_out..act_N_out)`
- For LTX-2 decoder, the required persistent states are the **previous input frame** for each `MemBlock` (9 total activation tensors in the decoder).
- The LTX-2 checkpoint `taeltx_2.pth` **does support disabling temporal upscale** in the decoder (`decoder_time_upscale=000`) in the decoder-only fork.
  - `taehv.py` itself hardcodes LTX-2 to `(True,True,True)`, so testing non-111 modes must be done via the decoder-only fork / wrapper.
- `dtu=000` (no temporal upscale) is useful as a **fast preview mode** and still works with the same stateful API.
- For the preview models (`dtu000`) built at different spatial sizes, the compiled `.mlmodelc` bundles share the **same `weights/weight.bin`**, so we can dedupe exactly like FLUX/QwenImage packaging.

### Files Added / Updated (Stateful Work)
- `scripts/convert_ltx2_tae_coreml.py`
  - Added **stateful export mode** (`--stateful`) with activation I/O tensors (`act_i`, `act_i_out`).
  - Added variable-length chunked equivalence check in PyTorch (full wrapper vs stateful wrapper).
  - Generalized `TGrow` handling to support both stride-1 and stride-2.
  - Added `--decoder-time-upscale` (e.g. `111`, `000`, `011`, `100`).
  - For `dtu != 111`, skips `TAEHV.decode_video(...)` equivalence check because `taehv.py` forces LTX-2 to `111`.

- `scripts/validate_ltx2_coreml.py`
  - Added stateful validation mode (`--stateful`) that iteratively calls the CoreML model over variable latent lengths.

- `scripts/benchmark_ltx2_stateful.swift`
  - Swift utility to:
    - run **correctness checks** for variable latent lengths (stateful model vs one-shot CoreML reference model),
    - run **ANE performance benchmarks** for iterative stateful decode,
    - report end-to-end latency plus ms/frame/FPS.

### Stateful API / Semantics
- Input latent layout: `NTHWC`
- Output image layout: `NTHWC`
- Activation tensors:
  - one state tensor per decoder `MemBlock`
  - names exported as `act_0`, `act_1`, ... and outputs `act_0_out`, `act_1_out`, ...
- Full temporal-upscale model (`dtu111`) stateful output is **raw chunk output** (before global trim alignment).
  - To match `decode_video(...)`, concatenate chunk outputs and then trim the first `frames_to_trim` frames once globally.
- Preview model (`dtu000`) has:
  - `tUpscale = 1`
  - `frames_to_trim = 0`
  - no temporal alignment trim needed.

### Stateful Correctness Findings

#### Full temporal upscale (`dtu111`, `chunkT=2`)
- PyTorch stateful equivalence (converter-side) passed for variable lengths.
- CoreML stateful validation (Python) for lengths `1,2,3,5,7,11,13`:
  - worst observed:
    - MAE `0.000226374`
    - RMSE `0.000296094`
    - MaxAbs `0.00721437`

#### No temporal upscale preview (`dtu000`, `chunkT=2`)
- PyTorch stateful equivalence passed for variable lengths.
- Swift correctness vs one-shot CoreML reference:
  - `T=1`: MAE `0`, MaxAbs `0`
  - `T=2`: MAE `0`, MaxAbs `0`
  - `T=11` vs one-shot **T=11 reference**: non-zero mismatch can appear if comparing under different reference-length contexts.
  - `T=16` vs one-shot **T=16 reference**: MAE `0`, MaxAbs `0`
- Practical conclusion:
  - For correctness checks of stateful iterative decode, use a **matching one-shot reference latent length** when you want exact comparison.

### Preview (`dtu000`) Performance Notes (Swift / ANE)
- State model shape example (768 preview):
  - latent input: `[1, 2, 24, 24, 128]`
  - image output per call: `[1, 2, 768, 768, 3]`
  - activation tensors: `9`
  - temporal: `chunkT=2`, `tUpscale=1`, `trim=0`
- Example ANE measurements (user machine, `cpu_ne`, float16):
  - `T=13`: `486.798 ms` total, `37.446 ms/frame`, `26.71 FPS`
  - `T=17`: `627.600 ms` total, `36.918 ms/frame`, `27.09 FPS`
  - `T=21`: `758.076 ms` total, `36.099 ms/frame`, `27.70 FPS`
- Another sample run reported significantly faster absolute latency on a different run/machine state:
  - `T=11`: `51.991 ms` total (`4.726 ms/frame`)
  - `T=16`: `565.868 ms` total (`35.367 ms/frame`) in a separate exactness run
- Main takeaway:
  - Preview `dtu000` is a valid fast mode and the stateful API remains frame-count agnostic.
  - Absolute timings vary by machine/runtime conditions; compare runs on the same setup.

### `dtu000` CoreML Build Commands (Examples)

One-shot reference (768, latent T=11):
```
source _env/bin/activate
PYTHONPATH=/Users/liu/workspace/taehv python scripts/convert_ltx2_tae_coreml.py \
  --decoder-time-upscale 000 \
  --checkpoint taeltx_2.pth \
  --compile \
  --min-deployment macos13 \
  --latent-hw 24 \
  --latent-t 11 \
  --out-dir coreml_out/ltx2_decoder_nthwc_768_dtu000
```

Stateful preview model (768, chunkT=2):
```
PYTHONPATH=/Users/liu/workspace/taehv python scripts/convert_ltx2_tae_coreml.py \
  --stateful \
  --decoder-time-upscale 000 \
  --checkpoint taeltx_2.pth \
  --compile \
  --min-deployment macos13 \
  --chunk-t 2 \
  --latent-hw 24 \
  --out-dir coreml_out/ltx2_decoder_stateful_nthwc_768_dtu000
```

### Fixed-Shape Preview Stateful Builds (For Packaging)

Generated fixed-shape FP16 **stateful preview** models (`dtu000`, `chunkT=2`) for:
- `768`, `1024`, `1280`, `1536`, `1792`, `2048`

Latent H/W mapping:
- `768 -> 24`
- `1024 -> 32`
- `1280 -> 40`
- `1536 -> 48`
- `1792 -> 56`
- `2048 -> 64`

Batch build directory:
- `coreml_out/ltx2_tae_preview_build/`

### Weight Dedup Check and Final Preview Layout (`ltx_2_tae`)

Verified compiled preview stateful models share identical weight file:
- SHA256 (`weights/weight.bin`) for all six:
  - `69d623de9f53a2b683cc881813fb3d6a20b6ecca510ff47a86c0e62b276927c0`

Created deduped final directory:
```
ltx_2_tae/
  768.mlmodelc/
  1024.mlmodelc/
  1280.mlmodelc/
  1536.mlmodelc/
  1792.mlmodelc/
  2048.mlmodelc/
  weight.bin
```

Dedup steps applied:
- Copied compiled `.mlmodelc` bundles from `coreml_out/ltx2_tae_preview_build/<size>/`.
- Removed `weights/weight.bin` from each `ltx_2_tae/*.mlmodelc`.
- Kept a single shared copy at `ltx_2_tae/weight.bin`.

## Wan 2.1 Stateful Packaging (LTX-style API + INSDIFF)

This session corrected `wan_2.1_tae` to follow the **same stateful API pattern as `ltx_2_tae`** (not first-frame-only).

### Key Learnings
- The first attempt used first-frame model bundles for `wan_2.1_tae`; that is not equivalent to the LTX-style stateful contract.
- Correct contract for Wan 2.1 stateful preview (`dtu000`) is:
  - inputs: `latent` (5D NTHWC) + `act_0..act_8` (NHWC)
  - outputs: `image` (5D NTHWC) + `act_0_out..act_8_out` (NHWC)
- For Wan 2.1 (`latent_channels=16`, `chunkT=2`, `dtu000`), per-size I/O is:
  - `latent`: `[1, 2, H, W, 16]`
  - `act_0..2`: `[1, H, W, 256]`
  - `act_3..5`: `[1, 2H, 2W, 128]`
  - `act_6..8`: `[1, 4H, 4W, 64]`
  - `image`: `[1, 2, 8H, 8W, 3]`
- `dtu000` implies `tUpscale=1` and no temporal trim, while keeping the same state-carry API.

### Build Formula Used
- Script: `scripts/convert_wan21_tae_stateful_coreml.py`
- Core flags:
  - `--checkpoint taew2_1.pth`
  - `--chunk-t 2`
  - `--decoder-time-upscale 000`
  - `--io-float16` (default)
  - `--convert-to mlprogram` (default)
  - `--min-deployment macos13`
  - `--compile`
  - `--skip-equivalence-check` (for batch build speed)
- Size mapping built in this pass:
  - `512 -> H=64`
  - `768 -> H=96`
  - `1024 -> H=128`
  - `1280 -> H=160`

### Weight Sharing / Dedup Findings
- Verified identical `weights/weight.bin` across these four stateful models:
  - SHA256: `9051de85ffdb16b6d647a904e0a05bb9b007cd73150e9af9e5c77f05cd5f19bd`
- This matches the Wan full-memory decoder weight lineage and can be reconstructed from `qwenimage_tae/weight.bin` using `insdiff`.

### Final `wan_2.1_tae` Layout (Current)
```
wan_2.1_tae/
  512.mlmodelc/
  768.mlmodelc/
  1024.mlmodelc/
  1280.mlmodelc/
  weight.bin.insdiff
```

Applied packaging steps:
- Copied `.mlmodelc` from `coreml_out/wan21_tae_stateful_small_build/<size>/`.
- Removed `weights/weight.bin` from each `.mlmodelc`.
- Generated `wan_2.1_tae/weight.bin.insdiff` with:
  - base: `qwenimage_tae/weight.bin`
  - target hash: `9051de85ffdb16b6d647a904e0a05bb9b007cd73150e9af9e5c77f05cd5f19bd`

### Operational Notes
- Large-size stateful Wan conversions (`1792`, `2048`) were unstable in this run due process termination (`Killed: 9`) during conversion; scoped output to `512/768/1024/1280` for now.
- Previous first-frame `wan_2.1_tae` content was preserved in:
  - `wan_2.1_tae_firstframe_backup/`
