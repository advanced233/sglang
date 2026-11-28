# M3c Extend Kernel & KL Consistency Summary

## Server Startup & KV Capacity

| Config | Ready | Ready s | KV Capacity (tokens) | mem_frac |
|---|---:|---:|---:|---:|
| `int4-chunk-bf16` | True | 55.0 | 457705 | 0.823 |
| `int4-chunk-q` | True | 55.0 | 457705 | 0.823 |
| `int4-det` | True | 60.0 | 457705 | 0.823 |

## Offline KL Consistency Checks

| Config | Check | Hits | Avg KL | p99 KL | Max KL | Max |Δlogp| | >1e-4 | >1e-2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `int4-chunk-bf16` | `match` | 32/32 | 0.009417 | 0.112344 | 17.374998 | 18.374998 | 16.24% | 4.97% |
| `int4-chunk-bf16` | `prefill_cache_hit` | 32/32 | 0.012770 | 0.085706 | 19.810672 | 20.810672 | 14.86% | 4.11% |
| `int4-chunk-bf16` | `decode_cache_hit` | 32/32 | 0.007197 | 0.071787 | 20.048143 | 21.048143 | 11.36% | 3.62% |
| `int4-chunk-bf16` | `prefill_cached_vs_fresh` | 32/32 | 0.066379 | 0.159757 | 426.780053 | 6.072705 | 28.17% | 9.20% |
| `int4-chunk-q` | `match` | 32/32 | 0.003806 | 0.038643 | 18.381103 | 19.381103 | 14.96% | 3.30% |
| `int4-chunk-q` | `prefill_cache_hit` | 32/32 | 0.003228 | 0.044730 | 13.101076 | 14.101075 | 15.43% | 3.42% |
| `int4-chunk-q` | `decode_cache_hit` | 32/32 | 0.002781 | 0.045596 | 7.107537 | 8.107235 | 13.56% | 3.28% |
| `int4-chunk-q` | `prefill_cached_vs_fresh` | 32/32 | 0 | 0 | 0 | 0 | 0.00% | 0.00% |
| `int4-det` | `match` | 32/32 | 0.003852 | 0.047198 | 13.196372 | 14.196371 | 14.69% | 3.43% |
| `int4-det` | `prefill_cache_hit` | 32/32 | 0.003852 | 0.047198 | 13.196372 | 14.196371 | 14.69% | 3.43% |
| `int4-det` | `decode_cache_hit` | 32/32 | 0.002248 | 0.031974 | 5.503003 | 6.501502 | 10.16% | 2.54% |
| `int4-det` | `prefill_cached_vs_fresh` | 32/32 | 0 | 0 | 0 | 0 | 0.00% | 0.00% |

