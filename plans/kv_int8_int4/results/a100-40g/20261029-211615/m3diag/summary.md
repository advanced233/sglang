# M3.0 decode-attention kernel diagnosis

- GPU NVIDIA A100-PCIE-40GB, driver 550.127.05, power limit 250.00 W; peak used for %: 1555 GB/s
- SGLang `c1ca3b0c5c8c`; ncu Version 2024.1.0.0 (build 33681293) (public-release)
- sglang env: torch 2.14.1 (CUDA 13.0), triton 3.8.0, vllm None, sglang 0.0.0.dev0
- vllm env: torch 2.13.0+cu129 (CUDA 12.9), triton 3.7.1, vllm 0.32.0+cu129, sglang None
- Shape: 6 seqs x 16384 tokens, Q/KV heads 32/8, head_dim 128
- Timing: CUDA graph replay of the whole op (all kernels), triton do_bench with a cold L2; GB/s counts the KV bytes as stored (data plus scales).

## Timing

| config | KV MB | op us (median) | GB/s | % of peak | % of ref-sum | speed vs sglang-bf16 | cos vs fp32 | SM MHz | power cap | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ref-sum | 403 | 307.2 | 1311 | 84% | 100% | 1.03x | - | 1410 | 0% |  |
| sglang-bf16 | 403 | 316.4 | 1273 | 82% | 97% | 1.00x | 0.999998 | 1410 | 17% | kv_splits=[8] |
| sglang-fp8e5m2 | 201 | 230.4 | 874 | 56% | 67% | 1.37x | 0.99439 | 1410 | 14% | kv_splits=[8] |
| vllm-bf16 | 403 | 344.1 | 1170 | 75% | 89% | 0.92x | 0.999997 | 1410 | 33% | decode_tile_used=16, vllm_path=3d, layout=inline |
| vllm-bf16-t32 | 403 | 317.4 | 1268 | 82% | 97% | 1.00x | 0.999997 | 1410 | 14% | decode_tile_used=32, vllm_path=3d, layout=inline |
| vllm-int8 | 208 | 690.2 | 301 | 19% | 23% | 0.46x | 0.999954 | 1410 | 0% | decode_tile_used=16, vllm_path=3d, layout=inline |
| vllm-int8-t32 | 208 | 608.3 | 341 | 22% | 26% | 0.52x | 0.999954 | 1410 | 0% | decode_tile_used=32, vllm_path=3d, layout=inline |
| vllm-int8-t64 | 208 | 558.1 | 372 | 24% | 28% | 0.57x | 0.999954 | 1410 | 25% | decode_tile_used=64, vllm_path=3d, layout=inline |
| vllm-int8-sep | 208 | 353.3 | 588 | 38% | 45% | 0.90x | 0.999954 | 1395 | 67% | decode_tile_used=16, vllm_path=3d, layout=sep |
| vllm-int8-sep-t64 | 208 | 276.5 | 751 | 48% | 57% | 1.14x | 0.999954 | 1410 | 20% | decode_tile_used=64, vllm_path=3d, layout=sep |
| vllm-int4 | 107 | 688.1 | 155 | 10% | 12% | 0.46x | 0.98895 | 1410 | 0% | decode_tile_used=16, vllm_path=3d, layout=inline |
| vllm-int4-t64 | 107 | 429.1 | 249 | 16% | 19% | 0.74x | 0.988947 | 1410 | 0% | decode_tile_used=64, vllm_path=3d, layout=inline |

## Nsight Compute (one call, --set full, clocks locked to base)

| config | kernel | grid | us | DRAM % | DRAM read MB | L2 hit % | SM % | occupancy % | regs | sectors/ld req | top stalls (cycles per issue) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ref-sum | at::native::reduce_kernel | 1536 | 306.0 | 85 | 403 | 24 | 23 | 92 | 32 | 16.0 | long_scoreboard 52.6, barrier 2.0, wait 1.5, not_selected 1.3 |
| sglang-bf16 | _fwd_grouped_kernel_stage1 | 384 | 318.2 | 82 | 404 | 25 | 25 | 22 | 72 | 8.5 | long_scoreboard 6.5, short_scoreboard 2.5, wait 2.3, selected 1.0 |
| sglang-bf16 | _fwd_kernel_stage2 | 192 | 12.0 | 4 | 1 | 21 | 7 | 11 | 27 | 2.3 | long_scoreboard 10.9, imc_miss 2.1, wait 2.1, selected 1.0 |
| sglang-fp8e5m2 | _fwd_grouped_kernel_stage1 | 384 | 322.0 | 41 | 202 | 24 | 37 | 22 | 128 | 5.8 | long_scoreboard 3.3, wait 1.6, selected 1.0, short_scoreboard 0.8 |
| sglang-fp8e5m2 | _fwd_kernel_stage2 | 192 | 12.1 | 4 | 1 | 21 | 7 | 11 | 27 | 2.3 | long_scoreboard 10.9, imc_miss 2.1, wait 2.1, selected 1.0 |
| vllm-bf16 | kernel_unified_attention | 896 | 490.0 | 53 | 403 | 24 | 34 | 22 | 112 | 14.0 | long_scoreboard 2.5, short_scoreboard 2.5, wait 2.4, selected 1.0 |
| vllm-bf16 | reduce_segments | 192 | 10.2 | 10 | 2 | 12 | 7 | 11 | 32 | 5.5 | long_scoreboard 6.5, imc_miss 2.4, short_scoreboard 2.0, wait 1.8 |
| vllm-bf16-t32 | kernel_unified_attention | 896 | 335.1 | 78 | 403 | 24 | 36 | 22 | 128 | 14.8 | long_scoreboard 2.6, wait 2.3, short_scoreboard 1.6, selected 1.0 |
| vllm-bf16-t32 | reduce_segments | 192 | 10.4 | 10 | 2 | 15 | 7 | 11 | 32 | 5.5 | long_scoreboard 6.0, imc_miss 2.4, short_scoreboard 2.0, wait 1.8 |
| vllm-int8 | kernel_unified_attention | 896 | 1171.8 | 12 | 209 | 43 | 42 | 16 | 166 | 5.4 | wait 1.5, long_scoreboard 1.4, selected 1.0, short_scoreboard 0.6 |
| vllm-int8 | reduce_segments | 192 | 10.2 | 10 | 2 | 12 | 7 | 11 | 32 | 5.5 | long_scoreboard 6.2, imc_miss 2.4, short_scoreboard 2.0, wait 1.8 |
| vllm-int8-t32 | kernel_unified_attention | 896 | 989.7 | 14 | 209 | 43 | 42 | 11 | 254 | 2.0 | wait 1.2, selected 1.0, long_scoreboard 0.8, short_scoreboard 0.3 |
| vllm-int8-t32 | reduce_segments | 192 | 10.3 | 10 | 2 | 13 | 7 | 10 | 32 | 5.5 | long_scoreboard 6.0, imc_miss 2.4, short_scoreboard 2.0, wait 1.8 |
| vllm-int8-t64 | kernel_unified_attention | 896 | 903.5 | 15 | 210 | 44 | 45 | 11 | 255 | 2.0 | wait 1.2, selected 1.0, long_scoreboard 0.6, short_scoreboard 0.3 |
| vllm-int8-t64 | reduce_segments | 192 | 10.2 | 10 | 2 | 12 | 7 | 11 | 32 | 5.5 | long_scoreboard 6.1, imc_miss 2.4, short_scoreboard 2.0, wait 1.8 |
| vllm-int8-sep | kernel_unified_attention | 896 | 602.3 | 23 | 208 | 31 | 50 | 22 | 128 | 9.4 | wait 1.9, short_scoreboard 1.2, selected 1.0, not_selected 0.5 |
| vllm-int8-sep | reduce_segments | 192 | 10.1 | 10 | 2 | 12 | 7 | 11 | 32 | 5.5 | long_scoreboard 6.1, imc_miss 2.4, short_scoreboard 2.0, wait 1.8 |
| vllm-int8-sep-t64 | kernel_unified_attention | 896 | 457.1 | 30 | 209 | 31 | 37 | 11 | 197 | 10.9 | wait 1.6, selected 1.0, short_scoreboard 0.7, long_scoreboard 0.4 |
| vllm-int8-sep-t64 | reduce_segments | 192 | 10.2 | 10 | 2 | 12 | 7 | 11 | 32 | 5.5 | long_scoreboard 6.0, imc_miss 2.4, short_scoreboard 2.0, wait 1.8 |
| vllm-int4 | at::native::unrolled_elementwise_kernel | 48 | 10.4 | 1 | 0 | 79 | 1 | 5 | 32 | 2.0 | long_scoreboard 9.1, wait 4.0, imc_miss 3.6, no_instruction 1.7 |
| vllm-int4 | at::native::elementwise_kernel | 96 | 7.6 | 1 | 0 | 70 | 2 | 6 | 16 | 4.0 | imc_miss 13.0, long_scoreboard 9.7, wait 2.1, selected 1.0 |
| vllm-int4 | at::native::vectorized_elementwise_kernel | 24 | 5.2 | 1 | 0 | 55 | 0 | 6 | 28 | 16.0 | imc_miss 24.0, long_scoreboard 19.4, wait 2.4, drain 1.0 |
| vllm-int4 | hadacore::hadamard_transform_kernel | 48 | 5.2 | 1 | 0 | 71 | 1 | 1 | 34 | 16.0 | imc_miss 3.6, long_scoreboard 2.1, wait 1.2, selected 1.0 |
| vllm-int4 | at::native::unrolled_elementwise_kernel | 48 | 10.3 | 1 | 0 | 78 | 1 | 5 | 32 | 2.0 | long_scoreboard 9.2, wait 4.0, imc_miss 3.6, no_instruction 1.7 |
| vllm-int4 | at::native::vectorized_elementwise_kernel | 24 | 5.1 | 1 | 0 | 59 | 0 | 6 | 28 | 16.0 | imc_miss 21.6, long_scoreboard 17.4, wait 2.2, drain 1.3 |
| vllm-int4 | at::native::vectorized_elementwise_kernel | 24 | 5.2 | 2 | 0 | 56 | 0 | 6 | 28 | 16.0 | imc_miss 24.0, long_scoreboard 18.3, wait 2.4, drain 1.0 |
| vllm-int4 | _attn_packed | 896 | 1093.2 | 7 | 108 | 54 | 41 | 16 | 152 | 2.1 | wait 1.8, long_scoreboard 1.0, selected 1.0, short_scoreboard 0.8 |
| vllm-int4 | reduce_segments | 192 | 10.3 | 10 | 2 | 12 | 7 | 11 | 32 | 5.5 | long_scoreboard 6.0, imc_miss 2.4, short_scoreboard 2.0, wait 1.8 |
| vllm-int4 | at::native::unrolled_elementwise_kernel | 48 | 10.6 | 0 | 0 | 78 | 1 | 5 | 32 | 2.0 | long_scoreboard 9.2, wait 4.0, imc_miss 3.6, no_instruction 1.7 |
| vllm-int4 | at::native::vectorized_elementwise_kernel | 24 | 5.0 | 1 | 0 | 57 | 0 | 6 | 28 | 16.0 | imc_miss 23.9, long_scoreboard 19.3, wait 2.4, drain 1.0 |
| vllm-int4 | hadacore::hadamard_transform_kernel | 48 | 5.3 | 1 | 0 | 53 | 1 | 1 | 34 | 16.0 | imc_miss 3.6, long_scoreboard 2.1, wait 1.2, selected 1.0 |
| vllm-int4 | at::native::unrolled_elementwise_kernel | 48 | 10.4 | 1 | 0 | 79 | 1 | 5 | 32 | 2.0 | long_scoreboard 9.0, wait 4.0, imc_miss 3.6, no_instruction 1.7 |
| vllm-int4 | at::native::vectorized_elementwise_kernel | 24 | 5.1 | 1 | 0 | 66 | 0 | 5 | 28 | 16.0 | imc_miss 21.6, long_scoreboard 16.7, no_instruction 5.1, wait 2.2 |
| vllm-int4 | at::native::elementwise_kernel | 96 | 7.6 | 2 | 0 | 70 | 2 | 6 | 16 | 4.0 | imc_miss 13.0, long_scoreboard 9.8, wait 2.1, selected 1.0 |
| vllm-int4 | at::native::vectorized_elementwise_kernel | 24 | 5.2 | 1 | 0 | 66 | 0 | 6 | 28 | 16.0 | imc_miss 23.0, long_scoreboard 16.2, wait 2.2, drain 1.3 |
| vllm-int4 | at::native::vectorized_elementwise_kernel | 24 | 5.0 | 1 | 0 | 50 | 0 | 5 | 28 | 16.0 | imc_miss 23.9, long_scoreboard 18.8, wait 2.4, drain 1.0 |
| vllm-int4-t64 | at::native::unrolled_elementwise_kernel | 48 | 10.7 | 0 | 0 | 89 | 1 | 5 | 32 | 2.0 | long_scoreboard 9.1, wait 4.0, imc_miss 3.6, no_instruction 1.7 |
| vllm-int4-t64 | at::native::elementwise_kernel | 96 | 7.5 | 1 | 0 | 70 | 2 | 6 | 16 | 4.0 | imc_miss 12.9, long_scoreboard 9.7, wait 2.1, selected 1.0 |
| vllm-int4-t64 | at::native::vectorized_elementwise_kernel | 24 | 5.2 | 1 | 0 | 55 | 0 | 5 | 28 | 16.0 | imc_miss 25.0, long_scoreboard 18.6, wait 2.4, drain 1.0 |
| vllm-int4-t64 | hadacore::hadamard_transform_kernel | 48 | 5.4 | 1 | 0 | 71 | 1 | 1 | 34 | 16.0 | imc_miss 3.6, long_scoreboard 2.2, wait 1.2, selected 1.0 |
| vllm-int4-t64 | at::native::unrolled_elementwise_kernel | 48 | 10.4 | 1 | 0 | 78 | 1 | 5 | 32 | 2.0 | long_scoreboard 9.1, wait 4.0, imc_miss 3.6, no_instruction 1.7 |
| vllm-int4-t64 | at::native::vectorized_elementwise_kernel | 24 | 5.1 | 1 | 0 | 64 | 0 | 6 | 28 | 16.0 | imc_miss 21.6, long_scoreboard 17.1, wait 2.2, drain 1.3 |
| vllm-int4-t64 | at::native::vectorized_elementwise_kernel | 24 | 5.1 | 1 | 0 | 56 | 0 | 6 | 28 | 16.0 | imc_miss 23.9, long_scoreboard 18.9, wait 2.4, drain 1.0 |
| vllm-int4-t64 | _attn_packed | 896 | 676.3 | 11 | 108 | 55 | 44 | 11 | 248 | 2.2 | wait 1.4, selected 1.0, long_scoreboard 0.3, short_scoreboard 0.3 |
| vllm-int4-t64 | reduce_segments | 192 | 10.2 | 10 | 2 | 12 | 7 | 11 | 32 | 5.5 | long_scoreboard 6.0, imc_miss 2.4, short_scoreboard 2.0, wait 1.8 |
| vllm-int4-t64 | at::native::unrolled_elementwise_kernel | 48 | 10.4 | 1 | 0 | 78 | 1 | 5 | 32 | 2.0 | long_scoreboard 9.2, wait 4.0, imc_miss 3.6, no_instruction 1.7 |
| vllm-int4-t64 | at::native::vectorized_elementwise_kernel | 24 | 5.2 | 1 | 0 | 57 | 0 | 6 | 28 | 16.0 | imc_miss 27.5, long_scoreboard 19.2, wait 2.4, drain 1.0 |
| vllm-int4-t64 | hadacore::hadamard_transform_kernel | 48 | 5.5 | 1 | 0 | 71 | 1 | 2 | 34 | 16.0 | imc_miss 3.6, long_scoreboard 2.3, wait 1.2, selected 1.0 |
| vllm-int4-t64 | at::native::unrolled_elementwise_kernel | 48 | 10.7 | 0 | 0 | 79 | 1 | 5 | 32 | 2.0 | long_scoreboard 9.2, wait 4.0, imc_miss 3.7, no_instruction 1.7 |
| vllm-int4-t64 | at::native::vectorized_elementwise_kernel | 24 | 5.3 | 1 | 0 | 66 | 0 | 6 | 28 | 16.0 | imc_miss 22.0, long_scoreboard 17.4, wait 2.2, drain 1.3 |
| vllm-int4-t64 | at::native::elementwise_kernel | 96 | 7.7 | 2 | 0 | 70 | 2 | 6 | 16 | 4.0 | imc_miss 13.2, long_scoreboard 10.0, wait 2.1, selected 1.0 |
| vllm-int4-t64 | at::native::vectorized_elementwise_kernel | 24 | 5.4 | 1 | 0 | 66 | 0 | 6 | 28 | 16.0 | imc_miss 24.4, long_scoreboard 16.6, wait 2.2, drain 1.3 |
| vllm-int4-t64 | at::native::vectorized_elementwise_kernel | 24 | 5.2 | 1 | 0 | 57 | 0 | 5 | 28 | 16.0 | imc_miss 24.4, long_scoreboard 20.0, wait 2.4, drain 1.0 |

