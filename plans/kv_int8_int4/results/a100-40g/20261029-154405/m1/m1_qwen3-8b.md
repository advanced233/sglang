# M1 offline KV quantization error: qwen3-8b

- model `~/codex-work/myproject/CommitKV/models/Qwen3-8B` | GPU NVIDIA A100-PCIE-40GB | torch 2.14.1+cu130 (CUDA 13.0) | transformers 5.19.0
- corpus: 203 files, 1183139 tokens | 8 windows at prefix <= 4096, 4 at longer prefixes | continuation 128 tokens
- KL and dNLL are means over continuation positions and windows.

## Prefix 4096 tokens

| scheme | KL chunk_bf16 | KL chunk_q | top-1 chunk_q | dNLL chunk_q | K rel err | V rel err |
|---|---:|---:|---:|---:|---:|---:|
| fp8_e5m2 | 0.004234 | 0.006344 | 0.9756 | -0.0006119 | 0.05281 | 0.05276 |
| fp8_e4m3 | 0.001312 | 0.001776 | 0.9854 | 0.001188 | 0.02666 | 0.02676 |
| int8_pth | 0.0003786 | 0.0006255 | 0.9902 | -0.0004695 | 0.01376 | 0.008234 |
| int4_pth_rht | 0.002143 | 0.01113 | 0.9717 | 0.01282 | 0.09418 | 0.09996 |
| int4_pth_norht | 0.01352 | 0.2782 | 0.8828 | 0.2418 | 0.1664 | 0.1175 |
| fp4_mx_block16 | 0.01711 | 0.0272 | 0.9492 | 0.0257 | 0.1283 | 0.1175 |

## Channel-outlier ratio (max/median of per-channel absmax)

| prefix | K mean | K max (layer) | K after RHT mean | V mean |
|---:|---:|---:|---:|---:|
| 4096 | 6.62 | 45.51 (0) | 1.41 | 2.88 |

## One-layer scan (int4_pth_rht, prefix 4096, 2 windows, chunk_bf16)

Mean KL over layers 2.263e-05; top layers: L20 9.036e-05, L14 4.712e-05, L23 4.71e-05, L24 3.953e-05, L15 2.709e-05, L31 2.593e-05, L2 2.541e-05, L4 2.455e-05
