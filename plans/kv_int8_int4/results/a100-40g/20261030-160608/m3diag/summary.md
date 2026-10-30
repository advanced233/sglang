# M3.0 decode-attention kernel diagnosis

- GPU NVIDIA A100-PCIE-40GB, driver 550.127.05, power limit 250.00 W; peak used for %: 1555 GB/s
- SGLang `ae581b09f82c`; ncu Version 2024.1.0.0 (build 33681293) (public-release)
- sglang env: torch 2.14.1 (CUDA 13.0), triton 3.8.0, vllm None, sglang 0.0.0.dev0
- vllm env: torch 2.13.0+cu129 (CUDA 12.9), triton 3.7.1, vllm 0.32.0+cu129, sglang None
- Shape: 6 seqs x 16384 tokens, Q/KV heads 32/8, head_dim 128
- Timing: CUDA graph replay of the whole op (all kernels), triton do_bench with a cold L2; GB/s counts the KV bytes as stored (data plus scales).

## Timing

| config | KV MB | op us (median) | GB/s | % of peak | % of ref-sum | speed vs sglang-bf16 | cos vs fp32 | SM MHz | power cap | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ref-sum | 403 | 323.6 | 1244 | 80% | 100% | 0.98x | - | 1095 | 0% |  |
| sglang-bf16 | 403 | 316.4 | 1273 | 82% | 102% | 1.00x | 0.999998 | 1410 | 0% | kv_splits=[8] |
| sglang-fp8e5m2 | 201 | 230.4 | 874 | 56% | 70% | 1.37x | 0.99439 | 1410 | 0% | kv_splits=[8] |
| sglang-int8 | 208 | 295.9 | 702 | 45% | 56% | 1.07x | 0.999955 | 1410 | 0% | kv_splits=[8], convert=magic, block_n=64, num_warps=4, num_stages=2 |
| vllm-int8-sep-t64 | 208 | 277.5 | 748 | 48% | 60% | 1.14x | 0.999954 | 1410 | 0% | decode_tile_used=64, vllm_path=3d, layout=sep |

## SGLang INT8 decode sweep

48 variants timed; the 16 fastest:

| convert | block_n | warps | stages | max splits | us | GB/s | % of ref-sum | speed vs sglang-bf16 | cos vs fp32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| magic | 128 | 4 | 3 | 8 | 193.5 | 1073 | 86% | 1.63x | 0.999955 |
| cast | 128 | 4 | 3 | 8 | 199.7 | 1040 | 84% | 1.58x | 0.999955 |
| magic | 128 | 4 | 3 | 16 | 201.7 | 1029 | 83% | 1.57x | 0.999955 |
| magic | 128 | 8 | 3 | 8 | 201.7 | 1029 | 83% | 1.57x | 0.999955 |
| magic | 128 | 4 | 2 | 8 | 203.8 | 1019 | 82% | 1.55x | 0.999955 |
| cast | 128 | 4 | 3 | 16 | 206.8 | 1004 | 81% | 1.53x | 0.999955 |
| magic | 32 | 4 | 3 | 8 | 212.0 | 979 | 79% | 1.49x | 0.999955 |
| magic | 128 | 4 | 2 | 16 | 212.0 | 979 | 79% | 1.49x | 0.999955 |
| magic | 128 | 8 | 3 | 16 | 213.0 | 975 | 78% | 1.49x | 0.999955 |
| cast | 128 | 4 | 2 | 8 | 213.0 | 975 | 78% | 1.49x | 0.999955 |
| cast | 64 | 4 | 3 | 8 | 214.0 | 970 | 78% | 1.48x | 0.999955 |
| magic | 64 | 4 | 3 | 16 | 217.1 | 956 | 77% | 1.46x | 0.999955 |
| magic | 32 | 4 | 3 | 16 | 219.1 | 947 | 76% | 1.44x | 0.999955 |
| magic | 128 | 8 | 2 | 8 | 222.2 | 934 | 75% | 1.42x | 0.999955 |
| cast | 32 | 4 | 3 | 8 | 223.2 | 930 | 75% | 1.42x | 0.999955 |
| cast | 128 | 4 | 2 | 16 | 223.2 | 930 | 75% | 1.42x | 0.999955 |

Best time per knob value (us):

- convert: magic 193.5, cast 199.7
- block_n: 32 212.0, 64 214.0, 128 193.5
- num_warps: 4 193.5, 8 201.7
- num_stages: 2 203.8, 3 193.5
- max_kv_splits: 8 193.5, 16 201.7

## Nsight Compute (one call, --set full, clocks locked to base)

| config | kernel | grid | us | DRAM % | DRAM read MB | L2 hit % | SM % | occupancy % | regs | sectors/ld req | top stalls (cycles per issue) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ref-sum | at::native::reduce_kernel | 1536 | 304.7 | 86 | 403 | 24 | 23 | 93 | 32 | 16.0 | long_scoreboard 52.8, barrier 2.0, wait 1.5, not_selected 1.3 |
| sglang-bf16 | _fwd_grouped_kernel_stage1 | 384 | 331.2 | 79 | 404 | 25 | 24 | 22 | 72 | 8.5 | long_scoreboard 6.5, short_scoreboard 2.5, wait 2.3, selected 1.0 |
| sglang-bf16 | _fwd_kernel_stage2 | 192 | 12.4 | 4 | 1 | 18 | 7 | 11 | 27 | 2.3 | long_scoreboard 11.2, imc_miss 2.1, wait 2.1, selected 1.0 |
| sglang-fp8e5m2 | _fwd_grouped_kernel_stage1 | 384 | 322.3 | 41 | 202 | 25 | 37 | 22 | 128 | 5.8 | long_scoreboard 3.4, wait 1.6, selected 1.0, short_scoreboard 0.8 |
| sglang-fp8e5m2 | _fwd_kernel_stage2 | 192 | 12.1 | 4 | 1 | 19 | 7 | 11 | 27 | 2.3 | long_scoreboard 11.1, imc_miss 2.1, wait 2.1, selected 1.0 |
| sglang-int8 | _fwd_grouped_kernel_stage1_int8 | 384 | 392.7 | 35 | 209 | 32 | 30 | 15 | 168 | 7.4 | long_scoreboard 2.7, wait 1.1, selected 1.0, not_selected 0.4 |
| sglang-int8 | _fwd_kernel_stage2 | 192 | 12.1 | 4 | 1 | 19 | 7 | 11 | 27 | 2.3 | long_scoreboard 10.8, imc_miss 2.1, wait 2.1, selected 1.0 |
| vllm-int8-sep-t64 | kernel_unified_attention | 896 | 456.6 | 30 | 209 | 31 | 37 | 11 | 197 | 10.9 | wait 1.6, selected 1.0, short_scoreboard 0.7, long_scoreboard 0.4 |
| vllm-int8-sep-t64 | reduce_segments | 192 | 10.4 | 10 | 2 | 12 | 7 | 11 | 32 | 5.5 | long_scoreboard 6.1, imc_miss 2.4, short_scoreboard 2.0, wait 1.8 |
| sglang-int8@cast-n128-w4-s3-k8 | _fwd_grouped_kernel_stage1_int8 | 384 | 279.7 | 49 | 209 | 32 | 57 | 11 | 254 | 12.1 | selected 1.0, wait 0.9, long_scoreboard 0.7, mio_throttle 0.5 |
| sglang-int8@cast-n128-w4-s3-k8 | _fwd_kernel_stage2 | 192 | 12.7 | 4 | 1 | 19 | 6 | 11 | 27 | 2.3 | long_scoreboard 11.3, imc_miss 2.1, wait 2.1, selected 1.0 |
| sglang-int8@magic-n128-w4-s3-k8 | _fwd_grouped_kernel_stage1_int8 | 384 | 256.4 | 53 | 209 | 32 | 41 | 11 | 255 | 12.1 | long_scoreboard 1.0, selected 1.0, wait 0.8, not_selected 0.3 |
| sglang-int8@magic-n128-w4-s3-k8 | _fwd_kernel_stage2 | 192 | 12.0 | 4 | 1 | 19 | 7 | 11 | 27 | 2.3 | long_scoreboard 11.1, imc_miss 2.1, wait 2.1, selected 1.0 |

## Main-kernel instruction mix (SASS)

| config | kernel | warp instructions | global loads by opcode | top opcodes (share of executed) |
|---|---|---:|---|---|
| ref-sum | reduce_kernel | 22,947,104 | LDG.E 48, LDG.E.128 786,432 | FADD 29%, PRMT 28%, BRA 6%, IADD3 4%, ISETP.GE.U32.AND 4%, LEA 4%, IMAD.WIDE.U32 4%, LDG.E.128 3%, ISETP.NE.AND 2%, MOV 2% |
| sglang-bf16 | _fwd_grouped_kernel_stage1 | 26,559,744 | LDG.E 4,608, LDG.E.128 1,536, LDG.E.64 798,720, LDGDEPBAR 199,680, LDGSTS.E.BYPASS.128 798,720 | FMUL 13%, IMAD 6%, HMMA.16816.F32.BF16 6%, FADD 6%, LDSM.16.M88.4 5%, SHFL.BFLY 5%, IADD3 4%, NOP 4%, FMNMX 3%, LEA 3% |
| sglang-fp8e5m2 | _fwd_grouped_kernel_stage1 | 39,660,288 | LDG.E 4,608, LDG.E.128 1,536, LDG.E.64 599,040, LDGDEPBAR 199,680, LDGSTS.E.64 399,360, LDGSTS.E.BYPASS.128 199,680 | PRMT 15%, IMAD.MOV.U32 14%, FMUL 8%, LDS.U8 8%, LDS.64 5%, HMMA.16816.F32 4%, FADD 4%, SHFL.BFLY 3%, IADD3 3%, IMAD 3% |
| sglang-int8 | _fwd_grouped_kernel_stage1_int8 | 39,098,112 | LDG.E 4,608, LDG.E.128 1,536, LDG.E.64 658,944, LDGDEPBAR 253,440, LDGSTS.E 152,064, LDGSTS.E.64 405,504, LDGSTS.E.BYPASS.128 202,752 | PRMT 14%, IMAD.MOV.U32 9%, FMUL 9%, LDS.U8 8%, HADD2 8%, LOP3.LUT 5%, HMMA.16816.F32 4%, LDS.64 4%, IMAD 4%, FADD 3% |
| vllm-int8-sep-t64 | kernel_unified_attention | 56,148,992 | LDG.E 20,480, LDG.E.128 6,144, LDGDEPBAR 374,784, LDGSTS.E 270,336, LDGSTS.E.64 417,792, LDGSTS.E.BYPASS.128 208,896 | PRMT 18%, I2F.S8 11%, IMAD.MOV.U32 8%, LDS.U8 6%, IMAD 5%, FMUL 5%, IMAD.IADD 4%, SHF.R.S32.HI 4%, HMMA.16816.F32.BF16 3%, IMAD.WIDE.U32 3% |
| sglang-int8@cast-n128-w4-s3-k8 | _fwd_grouped_kernel_stage1_int8 | 37,328,640 | LDG.E 4,608, LDG.E.128 1,536, LDGDEPBAR 132,096, LDGSTS.E 52,224, LDGSTS.E.64 445,440, LDGSTS.E.BYPASS.128 208,896 | PRMT 19%, I2F.F16.S16 17%, LDS.S8 8%, IMAD.MOV.U32 8%, FMUL 6%, LDS.64 5%, HMMA.16816.F32 4%, IMAD 4%, SEL 3%, ISETP.GE.AND 2% |
| sglang-int8@magic-n128-w4-s3-k8 | _fwd_grouped_kernel_stage1_int8 | 34,749,696 | LDG.E 4,608, LDG.E.128 1,536, LDGDEPBAR 132,096, LDGSTS.E 52,224, LDGSTS.E.64 445,440, LDGSTS.E.BYPASS.128 208,896 | PRMT 16%, LDS.U8 9%, HADD2 9%, IMAD.MOV.U32 8%, FMUL 6%, LOP3.LUT 6%, LDS.64 5%, HMMA.16816.F32 5%, IMAD 4%, SEL 4% |

