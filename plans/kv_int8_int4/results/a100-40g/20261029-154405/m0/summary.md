# M0 serving baselines

- GPU NVIDIA A100-PCIE-40GB (driver 550.127.05, power limit 250.00 W, max SM 1410 MHz); CUDA forward compat: True
- SGLang `fc55d790fe9b`, torch 2.14.1 (CUDA 13.0), triton 3.8.0, flashinfer 0.7.2, sglang-kernel 0.4.10
- vLLM env: 0.32.0 2.13.0+cu129 12.9

## Servers

| server | ready | load s | KV dtype | KV tokens | backend | mem fraction / util | decode graph max bs |
|---|---|---:|---|---:|---|---:|---:|
| sglang-bf16-flashinfer | yes | 75 | auto | 121,576 | flashinfer | 0.823 | 32 |
| sglang-bf16-triton | yes | 55 | auto | 121,576 | triton | 0.823 | 32 |
| sglang-fp8e5m2-triton | yes | 60 | fp8_e5m2 | 243,152 | triton | 0.823 | 32 |
| sglang-fp4mx-triton | yes | 105 | fp4_mx_block16 | 308,468 | triton | 0.7 | 32 |
| sglang-fp4mx-triton-pool40000 | yes | 70 | fp4_mx_block16 | 40,000 | triton | 0.7 | 32 |
| vllm-bf16-triton | yes | 80 | auto | 138,560 | TRITON_ATTN | 0.92 | - |
| vllm-int8-triton | yes | 80 | int8_per_token_head | 268,736 | TRITON_ATTN | 0.92 | - |
| vllm-int4-triton | yes | 95 | int4_per_token_head | 521,344 | TRITON_ATTN | 0.92 | - |

## SGLang sweep

| config | input | conc | prompts | out tok/s | median TPOT ms | median TTFT ms | max temp C | min SM MHz | thermal | power cap | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sglang-bf16-flashinfer | 16384 | 6 | 12 | 99.9 | 43.35 | 4303 | 56 | 765 | 0% | 58% |  |
| sglang-bf16-flashinfer | 16384 | 7 | 14 | 101.1 | 45.57 | 6069 | 63 | 1050 | 0% | 70% |  |
| sglang-bf16-triton | 16384 | 6 | 12 | 83.4 | 49.97 | 5654 | 65 | 1065 | 0% | 69% |  |
| sglang-bf16-triton | 16384 | 7 | 14 | 84.2 | 47.69 | 9091 | 69 | 765 | 0% | 75% |  |
| sglang-fp8e5m2-triton | 16384 | 6 | 12 | 87.6 | 38.31 | 7754 | 65 | 765 | 0% | 71% |  |
| sglang-fp8e5m2-triton | 16384 | 14 | 28 | 92.4 | 86.77 | 16668 | 74 | 765 | 0% | 84% |  |
| sglang-fp4mx-triton | 16384 | 6 | 12 | 13.9 | 389.43 | 11065 | 79 | 765 | 0% | 93% |  |
| sglang-fp4mx-triton | 16384 | 18 | 36 | 28.6 | 499.41 | 33469 | 81 | 1065 | 0% | 95% |  |

## vLLM sweep (text prompts, input lengths approximate)

| config | input | conc | prompts | out tok/s | median TPOT ms | median TTFT ms | max temp C | min SM MHz | thermal | power cap | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| vllm-bf16-triton | 16384 | 6 | 12 | 60.9 | 77.71 | 5917 | 68 | 1125 | 0% | 76% |  |
| vllm-bf16-triton | 16384 | 8 | 16 | 58.0 | 103.03 | 8537 | 75 | 765 | 0% | 84% |  |
| vllm-int8-triton | 16384 | 6 | 12 | 30.2 | 168.12 | 11538 | 74 | 1245 | 0% | 87% |  |
| vllm-int8-triton | 16384 | 16 | 32 | 30.0 | 463.43 | 18096 | 82 | 765 | 0% | 95% |  |
| vllm-int4-triton | 16384 | 6 | 12 | 35.7 | 140.09 | 8429 | 73 | 1200 | 0% | 85% |  |
| vllm-int4-triton | 16384 | 31 | 62 | 37.2 | 693.85 | 28431 | 83 | 1170 | 0% | 96% |  |

## Decode TPOT vs pool size (input 1024, output 256, concurrency 1)

| config | pool tokens | median TPOT ms | out tok/s | max temp C | min SM MHz | thermal | power cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-bf16-triton | 121,576 | 13.14 | 74.5 | 68 | 765 | 0% | 62% |
| sglang-fp4mx-triton | 308,468 | 343.74 | 2.9 | 81 | 765 | 0% | 97% |
| sglang-fp4mx-triton | 40,000 | 57.43 | 17.3 | 76 | 1305 | 0% | 79% |

## Decode profiles (16k context, 6 requests)

- sglang-bf16-triton-profile: ok, tables in `profiles/sglang-bf16-triton-profile_decode_tables.txt`
