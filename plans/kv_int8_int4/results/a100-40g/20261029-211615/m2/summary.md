# M2 smoke: int8_per_token_head vs BF16 (Triton backend)

| config | ready | KV dtype | KV B/token/layer | answer | probe ITL ms | GSM8K % | examples | eval |
|---|---|---|---:|---|---:|---:|---:|---|
| bf16-triton | True | torch.bfloat16 | 4098 | 391 | - | None | None | exit -15 |
| int8-triton | True | torch.int8 | 2112 | 391 | - | None | None | exit -15 |

