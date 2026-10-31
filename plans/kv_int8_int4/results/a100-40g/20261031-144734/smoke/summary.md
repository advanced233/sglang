# Smoke: int8_per_token_head vs BF16 (Triton backend)

| config | ready | KV dtype | KV tokens | KV B/token/layer | answer | probe ITL ms | GSM8K % | examples | eval |
|---|---|---|---:|---:|---|---:|---:|---:|---|
| bf16-triton | True | torch.bfloat16 | 121578 | 4097 | 391 | 13.01 | 91.96 | 1319 | ok |
| int8-triton | True | torch.int8 | 235787 | 2112 | 391 | 12.97 | 92.87 | 1319 | ok |

