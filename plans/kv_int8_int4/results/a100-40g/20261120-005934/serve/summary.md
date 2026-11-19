# M0 serving baselines

- GPU NVIDIA A100-PCIE-40GB (driver 550.127.05, power limit 250.00 W, max SM 1410 MHz); CUDA forward compat: True
- SGLang `c77cf41cda74`, torch 2.14.1 (CUDA 13.0), triton 3.8.0, flashinfer 0.7.2, sglang-kernel 0.4.10
- vLLM env: 0.32.0 2.13.0+cu129 12.9

## Servers

| server | ready | load s | KV dtype | KV tokens | backend | mem fraction / util | decode graph max bs |
|---|---|---:|---|---:|---|---:|---:|
| sglang-int8-triton | yes | 60 | int8_per_token_head | 237,905 | triton | 0.823 | 32 |

## M3b: BF16 vs fused INT8 decode (Triton)

| config | input | conc | prompts | out tok/s | median ITL ms | median TPOT ms | median TTFT ms | max temp C | min SM MHz | thermal | power cap | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sglang-int8-triton | 16384 | 6 | 12 | 187.0 | 19.53 | 26.37 | 5837 | 62 | 1050 | 0% | 81% |  |
| sglang-int8-triton | 16384 | 13 | 26 | 230.5 | 27.27 | 41.28 | 15598 | 72 | 1020 | 0% | 88% |  |

## Decode probe vs pool size (input 1024, concurrency 1)

| config | pool tokens | median TPOT ms | out tok/s | max temp C | min SM MHz | thermal | power cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-int8-triton | 237,905 | 13.09 | 76.0 | 73 | 1170 | 0% | 86% |

