# 🥮 Tiny AutoEncoder for Hunyuan Video

## What is TAEHV?

TAEHV is a Tiny AutoEncoder for Hunyuan Video. TAEHV can decode Hunyuan latents into videos more cheaply (in time & memory) than the full-size Hunyuan VAE, at the cost of slightly lower quality.

Here's a comparison of the output & memory usage of the Full VAE vs. TAEHV:

<table>
<tr><th><tt>pipe.vae</tt></th><th>Full VAE</th><th>TAEHV</th></tr>
<tr>
  <td>Decoded Video<br/><sup>(converted to GIF)</sup></td>
  <td><img src="https://github.com/user-attachments/assets/b9ee3405-c210-4410-95ac-639a4ed09c50"/></td>
  <td><img src="https://github.com/user-attachments/assets/3fe3cb6a-30e5-46fe-9458-f0a39e454b86"/></td>
</tr>
<tr>
  <td>Runtime<br/><sup>(in fp16, on GH200)</sup></td>
  <td></td>
  <td></td>
</tr>
<tr>
  <td>Memory<br/><sup>(in fp16, on GH200)</sup></td>
  <td></td>
  <td></td>
</tr>
</table>


See the [profiling notebook](./examples/TAEHV_Profiling.ipynb) for details on the comparison the [example notebook](./examples/TAEHV_T2I_Demo.ipynb) for a simpler demo.

Since Wan 2.1 uses the same settings as Hunyuan VAE, you can also use TAEHV for Wan 2.1 decoding using the `taew2_1.pth` weights (see the [Wan 2.1 example notebook](./examples/TAEW2.1_T2I_Demo.ipynb)).

## Limitations

TAEHV is still pretty experimental (specifically, it's a hacky finetune of [TAEM1](https://github.com/madebyollin/taem1) :) using a fairly limited dataset) and I haven't tested it much yet. Please report quality / performance issues as you discover them.
