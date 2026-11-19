# M3c Extend Kernel & KL Consistency Summary

## Kernel Microbenchmark (cold L2, 32 Q / 8 KV / D=128)

| Shape (B × prefix + extend) | Impl | Median ms | vs BF16 | vs M3b | Temp MB | Cos vs BF16 | Max abs vs exact |
|---|---|---:|---:|---:|---:|---:|---:|
| prefix-hit (4×8192+512) | `bf16-regular` | 4.040 | 1.00× | 1.04× | 0.0 | 0.999997 | 0.00033 |
| prefix-hit (4×8192+512) | `bf16-unified` | 3.965 | 1.02× | 1.06× | 0.0 | 0.999997 | 0.00033 |
| prefix-hit (4×8192+512) | `int8-m3b-gather` | 4.190 | 0.96× | 1.00× | 134.5 | 0.999953 | 0.00062 |
| prefix-hit (4×8192+512) | `int8-fused-regular` | 4.086 | 0.99× | 1.03× | 0.0 | 0.999957 | 0.00035 |
| prefix-hit (4×8192+512) | `int8-fused-unified` | 4.618 | 0.87× | 0.91× | 0.0 | 0.999956 | 0.00037 |
| late-chunk (1×14336+2048) | `bf16-regular` | 7.252 | 1.00× | 1.00× | 0.0 | 0.999997 | 0.00028 |
| late-chunk (1×14336+2048) | `bf16-unified` | 7.611 | 0.95× | 0.96× | 0.0 | 0.999997 | 0.00028 |
| late-chunk (1×14336+2048) | `int8-m3b-gather` | 7.277 | 1.00× | 1.00× | 58.8 | 0.999955 | 0.00033 |
| late-chunk (1×14336+2048) | `int8-fused-regular` | 7.224 | 1.00× | 1.01× | 0.0 | 0.999959 | 0.00024 |
| late-chunk (1×14336+2048) | `int8-fused-unified` | 8.149 | 0.89× | 0.89× | 0.0 | 0.999956 | 0.00024 |
| fresh-prefill (4×0+4096) | `bf16-regular` | 6.216 | 1.00× | 1.00× | 0.0 | 0.999998 | 0.00957 |
| fresh-prefill (4×0+4096) | `bf16-unified` | 14.293 | 0.43× | 0.43× | 0.0 | 0.999998 | 0.00957 |
| fresh-prefill (4×0+4096) | `int8-m3b-gather` | 6.216 | 1.00× | 1.00× | 0.0 | 0.999998 | 0.00957 |
| fresh-prefill (4×0+4096) | `int8-fused-regular` | 7.437 | 0.84× | 0.84× | 0.0 | 0.999998 | 0.00957 |
| fresh-prefill (4×0+4096) | `int8-fused-unified` | 8.922 | 0.70× | 0.70× | 0.0 | 0.999961 | 0.00843 |

