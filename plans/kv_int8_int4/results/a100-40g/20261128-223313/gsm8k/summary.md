# Smoke: low-bit KV cache vs BF16 (Triton backend)

| config | ready | KV dtype | KV tokens | KV B/token/layer | answer | probe ITL ms | GSM8K % | examples | eval |
|---|---|---|---:|---:|---|---:|---:|---:|---|
| int4-triton | True | torch.uint8 | 457705 | 1088 | 391 | 13.26 | 92.87 | 1319 | ok |

