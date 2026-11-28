# M3.0 decode-attention kernel diagnosis

- GPU NVIDIA A100-PCIE-40GB, driver 550.127.05, power limit 250.00 W; peak used for %: 1555 GB/s
- SGLang `70430d8d5414`; ncu Version 2024.1.0.0 (build 33681293) (public-release)
- sglang env: torch 2.14.1 (CUDA 13.0), triton 3.8.0, vllm None, sglang 0.0.0.dev0
- vllm env: torch 2.13.0+cu129 (CUDA 12.9), triton 3.7.1, vllm 0.32.0+cu129, sglang None
- Shape: 6 seqs x 16384 tokens, Q/KV heads 32/8, head_dim 128
- Timing: CUDA graph replay of the whole op (all kernels), triton do_bench with a cold L2; GB/s counts the KV bytes as stored (data plus scales).

## Timing

| config | KV MB | op us (median) | GB/s | % of peak | % of ref-sum | speed vs sglang-bf16 | cos vs fp32 | SM MHz | power cap | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ref-sum | 403 | 308.2 | 1306 | 84% | 100% | 1.02x | - | 1410 | 0% |  |
| sglang-bf16 | 403 | 315.4 | 1277 | 82% | 98% | 1.00x | 0.999998 | 1410 | 0% | kv_splits=[8] |
| sglang-int8 | 208 | 192.5 | 1078 | 69% | 83% | 1.64x | 0.999955 | 1410 | 33% | kv_splits=[8], convert=magic, block_n=128, num_warps=4, num_stages=3 |
| sglang-int4 | 107 | 146.4 | 730 | 47% | 56% | 2.15x | 0.989387 | 1410 | 14% | kv_splits=[8], convert=magic, block_n=128, num_warps=4, num_stages=3 |
| vllm-int4 | 107 | 689.1 | 155 | 10% | 12% | 0.46x | 0.98895 | 1410 | 0% | decode_tile_used=16, vllm_path=3d, layout=inline |
| vllm-int4-t64 | 107 | 427.0 | 250 | 16% | 19% | 0.74x | 0.988947 | 1410 | 0% | decode_tile_used=64, vllm_path=3d, layout=inline |
| sglang-int4@magic-n64-w4-s3-k8 | 107 | 146.4 | 730 | 47% | 56% | 2.15x | 0.989388 | 1410 | 33% | kv_splits=[8], convert=magic, block_n=64, num_warps=4, num_stages=3 |
| sglang-int4@magic-n128-w4-s3-k8 | 107 | 146.4 | 730 | 47% | 56% | 2.15x | 0.989387 | 1410 | 14% | kv_splits=[8], convert=magic, block_n=128, num_warps=4, num_stages=3 |
| sglang-int4@magic-n128-w4-s3-k16 | 107 | 157.7 | 678 | 44% | 52% | 2.00x | 0.989387 | 1410 | 40% | kv_splits=[16], convert=magic, block_n=128, num_warps=4, num_stages=3 |
| sglang-int4@magic-n256-w4-s2-k8 | 107 | 141.3 | 757 | 49% | 58% | 2.23x | 0.989388 | 1410 | 14% | kv_splits=[8], convert=magic, block_n=256, num_warps=4, num_stages=2 |
| sglang-int4@cast-n128-w4-s3-k8 | 107 | 177.2 | 604 | 39% | 46% | 1.78x | 0.989387 | 1410 | 0% | kv_splits=[8], convert=cast, block_n=128, num_warps=4, num_stages=3 |

