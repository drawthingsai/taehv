# 🥮 Tiny AutoEncoder for Hunyuan Video & Wan 2.1

## What is TAEHV?

TAEHV is a Tiny AutoEncoder for Hunyuan Video (& Wan 2.1). TAEHV can decode latents into videos more cheaply (in time & memory) than the full-size VAEs, at the cost of slightly lower quality.

Here's a comparison of the output & memory usage of the Full Hunyuan VAE vs. TAEHV:

<table>
<tr><th><tt>pipe.vae</tt></th><th>Full Hunyuan VAE</th><th>TAEHV</th></tr>
<tr>
  <td>Decoded Video<br/><sup>(converted to GIF)</sup></td>
  <td><img src="https://github.com/user-attachments/assets/b9ee3405-c210-4410-95ac-639a4ed09c50"/></td>
  <td><img src="https://github.com/user-attachments/assets/3fe3cb6a-30e5-46fe-9458-f0a39e454b86"/></td>
</tr>
<tr>
  <td>Runtime<br/><sup>(in fp16, on GH200)</sup></td>
  <td><strong>~2-3s</strong> for decoding 61 frames of (512, 320) video</td>
  <td><strong>~0.5s</strong> for decoding 61 frames of (512, 320) video</td>
</tr>
<tr>
  <td>Memory<br/><sup>(in fp16, on GH200)</sup></td>
  <td><strong>~6-9GB Peak Memory Usage</strong><br/><img src="https://github.com/user-attachments/assets/d7837271-c748-4eef-ab37-eda6cc1e6a69"/></td>
  <td><strong><0.5GB Peak Memory Usage</strong><br/><img src="https://github.com/user-attachments/assets/c71e2ef5-12f1-431f-b193-29d9a5ee6343"/></td>
</tr>
</table>


See the [profiling notebook](./examples/TAEHV_Profiling.ipynb) for details on this comparison or the [example notebook](./examples/TAEHV_T2I_Demo.ipynb) for a simpler demo.

## How do I use TAEHV with Wan 2.1?

Since Wan 2.1 uses the same input / output shapes as Hunyuan VAE, you can also use TAEHV for Wan 2.1 decoding using the `taew2_1.pth` weights (see the [Wan 2.1 example notebook](./examples/TAEW2.1_T2I_Demo.ipynb)).

## How do I use TAEHV with CogVideoX?

Try the `taecvx.pth` weights (see the [example notebook](./examples/TAECVX_T2I_Demo.ipynb)).

## How can I reduce the TAEHV decoding cost further?

You can disable temporal or spatial upscaling to get even-cheaper decoding.

```python
TAEHV(decoder_time_upscale=(False, False), decoder_space_upscale=(True, True, True))
```

![Image](https://github.com/user-attachments/assets/c517e37b-e53b-4d7d-b282-fbbbce10ade7)

```python
TAEHV(decoder_time_upscale=(False, False), decoder_space_upscale=(False, False, False))
```

![Image](https://github.com/user-attachments/assets/62223493-8cad-427b-b13c-fa9919d3fd7b)

If you have a powerful GPU or are decoding at a reduced resolution, you can also set `parallel=True` in `TAEHV.decode_video` to decode all frames at once (which is faster but requires more memory).

## Limitations

TAEHV is still pretty experimental (specifically, it's a hacky finetune of [TAEM1](https://github.com/madebyollin/taem1) :) using a fairly limited dataset) and I haven't tested it much yet. Please report quality / performance issues as you discover them.
