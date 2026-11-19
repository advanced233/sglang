# M0 serving baselines

- GPU NVIDIA A100-PCIE-40GB (driver 550.127.05, power limit 250.00 W, max SM 1410 MHz); CUDA forward compat: True
- SGLang `f8499e359b17`, torch 2.14.1 (CUDA 13.0), triton 3.8.0, flashinfer 0.7.2, sglang-kernel 0.4.10
- vLLM env: 0.32.0 2.13.0+cu129 12.9

## Servers

| server | ready | load s | KV dtype | KV tokens | backend | mem fraction / util | decode graph max bs |
|---|---|---:|---|---:|---|---:|---:|
| sglang-int8-triton | yes | 50 | int8_per_token_head | 235,787 | triton | 0.823 | 32 |

## M3b: BF16 vs fused INT8 decode (Triton)

| config | input | conc | prompts | out tok/s | median ITL ms | median TPOT ms | median TTFT ms | max temp C | min SM MHz | thermal | power cap | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sglang-int8-triton | 16384 | 6 | 12 | 159.8 | 19.71 | 29.41 | 8335 | 67 | 1065 | 0% | 84% |  |
| sglang-int8-triton | 16384 | 13 | 26 | 189.8 | 27.34 | 47.11 | 21914 | 74 | 1035 | 0% | 90% |  |

## Decode probe vs pool size (input 1024, concurrency 1)

| config | pool tokens | median TPOT ms | out tok/s | max temp C | min SM MHz | thermal | power cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-int8-triton | 235,787 | 13.09 | 76.0 | 73 | 765 | 0% | 87% |

