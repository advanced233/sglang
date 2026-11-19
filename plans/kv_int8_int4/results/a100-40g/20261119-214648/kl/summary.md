# M3c Extend Kernel & KL Consistency Summary

## Server Startup & KV Capacity

| Config | Ready | Ready s | KV Capacity (tokens) | mem_frac |
|---|---:|---:|---:|---:|
| `int8-det` | True | 60.0 | 235787 | 0.823 |

## Offline KL Consistency Checks

| Config | Check | Hits | Avg KL | p99 KL | Max KL | Max |Δlogp| | >1e-4 | >1e-2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `int8-det` | `match` | 32/32 | 1.644e-04 | 0.002829 | 0.405167 | 1.058027 | 9.59% | 0.14% |
| `int8-det` | `prefill_cache_hit` | 32/32 | 1.644e-04 | 0.002829 | 0.405167 | 1.058027 | 9.59% | 0.14% |
| `int8-det` | `decode_cache_hit` | 32/32 | 4.583e-04 | 0.002085 | 4.046781 | 5.040310 | 5.90% | 0.09% |
| `int8-det` | `prefill_cached_vs_fresh` | 32/32 | 2.186e-04 | 0.001842 | 1.980989 | 1.499799 | 2.07% | 0.18% |

