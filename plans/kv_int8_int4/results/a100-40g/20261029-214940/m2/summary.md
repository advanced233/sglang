# M2 smoke: int8_per_token_head vs BF16 (Triton backend)

| config | ready | KV dtype | KV B/token/layer | answer | probe ITL ms | GSM8K % | examples | eval |
|---|---|---|---:|---|---:|---:|---:|---|
| bf16-triton | True | torch.bfloat16 | 4098 | 391 | 13.03309015929699 | 93.33 | 1319 | ok |
| int8-triton | True | torch.int8 | 2112 | 391 | 30.507558025419712 | 92.34 | 1319 | ok |

