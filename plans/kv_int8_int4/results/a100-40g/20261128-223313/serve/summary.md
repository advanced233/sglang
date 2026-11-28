# M0 serving baselines

- GPU NVIDIA A100-PCIE-40GB (driver 550.127.05, power limit 250.00 W, max SM 1410 MHz); CUDA forward compat: True
- SGLang `052288b4c2c6`, torch 2.14.1 (CUDA 13.0), triton 3.8.0, flashinfer 0.7.2, sglang-kernel 0.4.10
- vLLM env: 0.32.0 2.13.0+cu129 12.9

## Servers

| server | ready | load s | KV dtype | KV tokens | backend | mem fraction / util | decode graph max bs |
|---|---|---:|---|---:|---|---:|---:|
| sglang-int4-triton | yes | 150 | int4_per_token_head | 457,705 | triton | 0.823 | 32 |

## M4: BF16 vs fused INT4+RHT decode (Triton)

| config | input | conc | prompts | out tok/s | median ITL ms | median TPOT ms | median TTFT ms | max temp C | min SM MHz | thermal | power cap | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| sglang-int4-triton | 16384 | 6 | 12 | 188.8 | 18.00 | 24.25 | 7714 | 62 | 765 | 0% | 82% |  |
| sglang-int4-triton | 16384 | 26 | 52 | 258.0 | 35.10 | 63.42 | 38270 | 77 | 765 | 0% | 93% |  |

## Decode probe vs pool size (input 1024, concurrency 1)

| config | pool tokens | median TPOT ms | out tok/s | max temp C | min SM MHz | thermal | power cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-int4-triton | 457,705 | 13.37 | 74.4 | 74 | 1035 | 0% | 87% |

