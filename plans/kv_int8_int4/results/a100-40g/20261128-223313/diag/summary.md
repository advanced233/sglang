# M3.0 decode-attention kernel diagnosis

- GPU NVIDIA A100-PCIE-40GB, driver 550.127.05, power limit 250.00 W; peak used for %: 1555 GB/s
- SGLang `052288b4c2c6`; ncu Version 2024.1.0.0 (build 33681293) (public-release)
- sglang env: torch 2.14.1 (CUDA 13.0), triton 3.8.0, vllm None, sglang 0.0.0.dev0
- Shape: 6 seqs x 16384 tokens, Q/KV heads 32/8, head_dim 128
- Timing: CUDA graph replay of the whole op (all kernels), triton do_bench with a cold L2; GB/s counts the KV bytes as stored (data plus scales).

## Timing

| config | KV MB | op us (median) | GB/s | % of peak | % of ref-sum | speed vs sglang-bf16 | cos vs fp32 | SM MHz | power cap | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sglang-bf16 | 403 | 315.4 | 1277 | 82% | - | 1.00x | 0.999998 | 1410 | 14% | kv_splits=[8] |
| sglang-int8 | 208 | 192.5 | 1078 | 69% | - | 1.64x | 0.999955 | 1395 | 86% | kv_splits=[8], convert=magic, block_n=128, num_warps=4, num_stages=3 |
| sglang-int4 | 107 | 141.3 | 757 | 49% | - | 2.23x | 0.989388 | 1410 | 14% | kv_splits=[8], convert=magic, block_n=256, num_warps=4, num_stages=2 |
| sglang-int4@magic-n128-w4-s3-k8 | 107 | 145.4 | 736 | 47% | - | 2.17x | 0.989387 | 1410 | 12% | kv_splits=[8], convert=magic, block_n=128, num_warps=4, num_stages=3 |
| sglang-int4@magic-n256-w4-s2-k8 | 107 | 141.3 | 757 | 49% | - | 2.23x | 0.989388 | 1410 | 0% | kv_splits=[8], convert=magic, block_n=256, num_warps=4, num_stages=2 |
| sglang-int4@magic-n256-w4-s3-k8 | 107 | 156.7 | 683 | 44% | - | 2.01x | 0.989388 | 1410 | 14% | kv_splits=[8], convert=magic, block_n=256, num_warps=4, num_stages=3 |
| sglang-int4@magic-n256-w8-s2-k8 | 107 | 196.6 | 544 | 35% | - | 1.60x | 0.989388 | 1410 | 0% | kv_splits=[8], convert=magic, block_n=256, num_warps=8, num_stages=2 |

