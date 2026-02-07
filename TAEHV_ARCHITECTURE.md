# TAEHV Architecture Notes

This document summarizes how `TAEHV` works in `taehv.py`, with emphasis on architecture and CoreML-relevant behavior.

## 1) What TAEHV is

`TAEHV` is a tiny video autoencoder with:
- an encoder: RGB video -> latent video
- a decoder: latent video -> RGB video

The model is causal across time (uses only current + past frames in temporal memory blocks).

Default tensor layout in code is `NTCHW`:
- `N`: batch
- `T`: time
- `C`: channels
- `H,W`: spatial

## 2) Core building blocks

## `conv(n_in, n_out)`
- `3x3` Conv2d with padding=1.

## `MemBlock(n_in, n_out)`
- Temporal causal block.
- Input: current feature map `x` and previous-timestep feature map `past`.
- Computes:
  - concat `[x, past]` on channels
  - `Conv -> ReLU -> Conv -> ReLU -> Conv`
  - residual skip (`1x1` if channel count changes, else identity)
  - final `ReLU`

So each block mixes spatial features and one-step temporal memory.

## `TPool(n_f, stride)`
- Temporal downscale operator.
- For `stride=2`: takes two timesteps, reshapes time into channels, applies `1x1` Conv2d.
- For `stride=1`: effectively no temporal downsampling.

## `TGrow(n_f, stride)`
- Temporal upscale operator (inverse idea of `TPool`).
- `1x1` Conv2d expands channels, then reshapes channels back into extra timesteps.
- For `stride=1`: no temporal expansion.

## `Clamp`
- `tanh(x/3)*3` before decode trunk, limiting latent magnitude.

## 3) Encoder topology

High-level encoder (3 stages):
1. input conv to 64 channels
2. stage A: `TPool` -> spatial downsample (`stride=2` conv) -> 3x `MemBlock(64,64)`
3. stage B: same pattern
4. stage C: same pattern
5. output conv to `latent_channels`

Spatially, each stage downsamples by 2, so total spatial factor is `8x`.
Temporally, downscale factor depends on `encoder_time_downscale` tuple.

## 4) Decoder topology

High-level decoder (3 stages):
1. `Clamp` -> conv from latent channels to 256
2. stage A: 3x `MemBlock(256,256)` -> optional spatial upsample -> `TGrow` -> conv to 128
3. stage B: 3x `MemBlock(128,128)` -> optional spatial upsample -> `TGrow` -> conv to 64
4. stage C: 3x `MemBlock(64,64)` -> optional spatial upsample -> `TGrow` -> conv to 64
5. `ReLU` -> final conv to output channels (`3 * patch_size^2`)
6. optional `pixel_shuffle` when `patch_size > 1`

Temporal upsample factor depends on `decoder_time_upscale` tuple.
`frames_to_trim = t_upscale - 1` is used after decode to keep causal alignment.

## 5) How temporal execution is implemented

`apply_model_with_memblocks(...)` has two modes:
- `parallel=True`: processes full sequence as `NT` merged batch; memory for each timestep is created by shifting features (`pad` + slice).
- `parallel=False`: graph traversal with a queue, needed because `TPool/TGrow` create nontrivial temporal fan-in/fan-out.

## 6) Variant-dependent channel/patch configs

`TAEHV` adjusts architecture based on checkpoint name:
- default (`taehv`, `taew2_1`): `patch_size=1`, `latent_channels=16`
- `taew2_2`: `patch_size=2`, `latent_channels=48`
- `taehv1_5`: `patch_size=2`, `latent_channels=32`
- `taeltx_2`: `patch_size=4`, `latent_channels=128`, and full temporal down/upscaling enabled

For Wan 2.1 (`taew2_1`), latent shape for first-frame decode is commonly `N,H,W,C = 1,96,96,16` (NHWC wrapper at API boundary, internal compute usually NCHW/NTCHW).

## 7) Why this matters for fp16 CoreML conversion

The decoder path contains many Conv + residual/memory operations. For CoreML conversion, Conv input dtype and Conv bias dtype must match. If traced graph introduces an fp32 tensor on the activation path while weights/bias are fp16, conversion can fail with dtype mismatch (the error we saw: `x` fp32 vs `bias` fp16).

Architecture-wise, likely sensitive points are:
- branch merges (`concat`, residual add)
- constant/zero memory path used in first-frame wrappers
- implicit casts around layout conversion (`NHWC <-> NCHW`) and tracing

That is why first-frame wrappers and explicit dtype control are important when exporting this model.
