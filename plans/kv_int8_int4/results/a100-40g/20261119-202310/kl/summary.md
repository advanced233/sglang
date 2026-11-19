# M3c Extend Kernel & KL Consistency Summary

## Server Startup & KV Capacity

| Config | Ready | Ready s | KV Capacity (tokens) | mem_frac |
|---|---:|---:|---:|---:|
| `int8-chunk-bf16` | True | 140.0 | 235787 | 0.823 |
| `int8-chunk-q` | True | 70.0 | 235787 | 0.823 |
| `int8-det` | True | 70.0 | 235787 | 0.823 |
| `bf16-det` | True | 60.0 | 121578 | 0.823 |

## Offline KL Consistency Checks

| Config | Check | Hits | Avg KL | p99 KL | Max KL | Max |Δlogp| | >1e-4 | >1e-2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `int8-chunk-bf16` | `match` | 32/32 | 0.001446 | 0.006050 | 8.623399 | 9.623333 | 10.93% | 0.60% |
| `int8-chunk-bf16` | `prefill_cache_hit` | 32/32 | 0.001378 | 0.005496 | 7.931599 | 8.931466 | 11.71% | 0.57% |
| `int8-chunk-bf16` | `decode_cache_hit` | 32/32 | 7.524e-04 | 0.005748 | 3.194827 | 4.179521 | 8.54% | 0.54% |
| `int8-chunk-bf16` | `prefill_cached_vs_fresh` | 32/32 | 0.001624 | 0.020352 | 6.344316 | 2.262470 | 19.70% | 1.84% |
| `int8-chunk-q` | `match` | 32/32 | 2.109e-04 | 0.003249 | 0.905059 | 1.727295 | 9.38% | 0.18% |
| `int8-chunk-q` | `prefill_cache_hit` | 32/32 | 3.707e-04 | 0.003374 | 3.036139 | 4.018153 | 9.89% | 0.15% |
| `int8-chunk-q` | `decode_cache_hit` | 32/32 | 1.321e-04 | 0.002926 | 0.113838 | 0.442039 | 7.72% | 0.13% |
| `int8-chunk-q` | `prefill_cached_vs_fresh` | 32/32 | 8.003e-05 | 5.746e-04 | 0.209148 | 0.724657 | 1.38% | 0.14% |
| `int8-det` | `match` | 32/32 | 1.620e-04 | 0.003103 | 0.330203 | 0.939303 | 8.89% | 0.18% |
| `int8-det` | `prefill_cache_hit` | 32/32 | 1.630e-04 | 0.003116 | 0.330203 | 0.939303 | 8.89% | 0.19% |
| `int8-det` | `decode_cache_hit` | 32/32 | 7.200e-04 | 0.002278 | 9.372627 | 10.372596 | 6.00% | 0.13% |
| `int8-det` | `prefill_cached_vs_fresh` | 32/32 | 0.038428 | 6.407e-04 | 626.190112 | 6.451483 | 1.41% | 0.12% |
| `bf16-det` | `match` | 32/32 | 1.676e-04 | 0.002457 | 1.063049 | 1.915830 | 8.38% | 0.05% |
| `bf16-det` | `prefill_cache_hit` | 32/32 | 1.676e-04 | 0.002457 | 1.063049 | 1.915830 | 8.38% | 0.05% |
| `bf16-det` | `decode_cache_hit` | 32/32 | 5.650e-05 | 0.001416 | 0.024911 | 0.231829 | 4.84% | 0.04% |
| `bf16-det` | `prefill_cached_vs_fresh` | 32/32 | 0 | 0 | 0 | 0 | 0.00% | 0.00% |

