# M1 offline KV quantization error: llama31-8b

- model `$KVQ_WORK/models/Llama-3.1-8B-Instruct` | GPU NVIDIA A100-PCIE-40GB | torch 2.14.1+cu130 (CUDA 13.0) | transformers 5.19.0
- corpus: 203 files, 1084385 tokens | 8 windows at prefix <= 4096, 4 at longer prefixes | continuation 128 tokens
- KL and dNLL are means over continuation positions and windows.

## Prefix 4096 tokens

| scheme | KL chunk_bf16 | KL chunk_q | top-1 chunk_q | dNLL chunk_q | K rel err | V rel err |
|---|---:|---:|---:|---:|---:|---:|
| fp8_e5m2 | 0.001903 | 0.004229 | 0.9873 | 0.008918 | 0.0525 | 0.05277 |
| fp8_e4m3 | 0.0005351 | 0.001229 | 0.9922 | 0.003706 | 0.02649 | 0.02668 |
| int8_pth | 0.0002303 | 0.0003439 | 0.9932 | 0.001202 | 0.01191 | 0.008154 |
| int4_pth_rht | 0.001272 | 0.003386 | 0.9844 | 0.008824 | 0.09821 | 0.1001 |
| int4_pth_norht | 0.004014 | 0.01286 | 0.9775 | 0.01294 | 0.1574 | 0.1162 |
| fp4_mx_block16 | 0.005641 | 0.01938 | 0.9746 | 0.0167 | 0.1266 | 0.117 |

## Channel-outlier ratio (max/median of per-channel absmax)

| prefix | K mean | K max (layer) | K after RHT mean | V mean |
|---:|---:|---:|---:|---:|
| 4096 | 3.29 | 4.49 (29) | 1.56 | 2.18 |
