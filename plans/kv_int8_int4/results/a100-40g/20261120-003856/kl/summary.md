# M3c Extend Kernel & KL Consistency Summary

## Server Startup & KV Capacity

| Config | Ready | Ready s | KV Capacity (tokens) | mem_frac |
|---|---:|---:|---:|---:|
| `int8-det` | True | 65.0 | 235787 | 0.823 |

## Offline KL Consistency Checks

| Config | Check | Hits | Avg KL | p99 KL | Max KL | Max |Δlogp| | >1e-4 | >1e-2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `int8-det` | `match` | 32/32 | 1.525e-04 | 0.003082 | 0.249953 | 0.801132 | 9.80% | 0.10% |
| `int8-det` | `prefill_cache_hit` | 32/32 | 1.525e-04 | 0.003082 | 0.249953 | 0.801132 | 9.80% | 0.10% |
| `int8-det` | `decode_cache_hit` | 32/32 | 1.246e-04 | 0.002560 | 0.108289 | 0.431915 | 6.81% | 0.17% |
| `int8-det` | `prefill_cached_vs_fresh` | 32/32 | 0 | 0 | 0 | 0 | 0.00% | 0.00% |

