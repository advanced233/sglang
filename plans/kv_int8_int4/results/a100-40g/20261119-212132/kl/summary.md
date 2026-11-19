# M3c Extend Kernel & KL Consistency Summary

## Server Startup & KV Capacity

| Config | Ready | Ready s | KV Capacity (tokens) | mem_frac |
|---|---:|---:|---:|---:|
| `int8-det` | True | 65.0 | 235787 | 0.823 |

## Offline KL Consistency Checks

| Config | Check | Hits | Avg KL | p99 KL | Max KL | Max |Δlogp| | >1e-4 | >1e-2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `int8-det` | `match` | 32/32 | 1.435e-04 | 0.003138 | 0.059238 | 0.365134 | 9.66% | 0.10% |
| `int8-det` | `prefill_cache_hit` | 32/32 | 1.531e-04 | 0.003201 | 0.072789 | 0.407434 | 10.25% | 0.13% |
| `int8-det` | `decode_cache_hit` | 32/32 | 2.765e-04 | 0.002155 | 2.784181 | 3.760918 | 5.66% | 0.13% |
| `int8-det` | `prefill_cached_vs_fresh` | 32/32 | 2.807e-04 | 0.001548 | 3.003805 | 1.749832 | 2.02% | 0.19% |

