# M0 serving baselines

- GPU NVIDIA A100-PCIE-40GB (driver 550.127.05, power limit 250.00 W, max SM 1410 MHz); CUDA forward compat: True
- SGLang `9ec83afdcc13`, torch 2.14.1 (CUDA 13.0), triton 3.8.0, flashinfer 0.7.2, sglang-kernel 0.4.10
- vLLM env: 0.32.0 2.13.0+cu129 12.9

## Servers

| server | ready | load s | KV dtype | KV tokens | backend | mem fraction / util | decode graph max bs |
|---|---|---:|---|---:|---|---:|---:|
| sglang-bf16-triton | yes | 50 | auto | 121,578 | triton | 0.823 | 32 |
| sglang-int8-triton | yes | 50 | int8_per_token_head | 235,787 | triton | 0.823 | 32 |

## M3b: BF16 vs fused INT8 decode (Triton)

| config | input | conc | prompts | out tok/s | median ITL ms | median TPOT ms | median TTFT ms | max temp C | min SM MHz | thermal | power cap | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sglang-bf16-triton | 16384 | 6 | 12 | 167.1 | 23.85 | 30.41 | 5620 | 67 | 1065 | 0% | 82% |  |
| sglang-int8-triton | 16384 | 6 | 12 | 187.2 | 19.67 | 26.33 | 5868 | 68 | 1050 | 0% | 80% |  |
| sglang-int8-triton | 16384 | 13 | 26 | 234.7 | 27.26 | 40.81 | 14933 | 75 | 1035 | 0% | 88% |  |

## Decode TPOT vs pool size (input 1024, output 256, concurrency 1)

| config | pool tokens | median TPOT ms | out tok/s | max temp C | min SM MHz | thermal | power cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-bf16-triton | 121,578 | 13.25 | 75.1 | 72 | 1065 | 0% | 87% |
| sglang-int8-triton | 235,787 | 13.07 | 76.1 | 74 | 1245 | 0% | 87% |

