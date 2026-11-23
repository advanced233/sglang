Device: NVIDIA A100-PCIE-40GB | Shape: batch=6, seq_len=16384, Q=32, KV=8, D=128, max_kv_splits=8, pool_slots=200000

| Variant | Median (us) | Min (us) | HBM Traffic (MB) | Temp VRAM (MB) | Eff KV BW (GB/s) | vs BF16 | vs Opt-Magic | CosSim | RelErr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bf16_baseline` | 316.4 | 304.1 | 402.7 | 0.0 | 1273 | 1.00x | 1.64x slower | 0.999998 | 0.00223 |
| `unfused_whole_pool` | 1290.2 | 1281.0 | 1644.3 | 781.2 | 161 | 0.25x | 6.70x slower | 0.999952 | 0.00982 |
| `unfused_active_gather` | 799.7 | 792.6 | 1012.9 | 384.0 | 260 | 0.40x | 4.15x slower | 0.999952 | 0.00982 |
| `naive_fused_n32_s2` | 239.6 | 236.5 | 207.6 | 0.0 | 866 | 1.32x | 1.24x slower | 0.999952 | 0.00982 |
| `naive_fused_n64_s2` | 291.8 | 278.5 | 207.6 | 0.0 | 711 | 1.08x | 1.52x slower | 0.999952 | 0.00981 |
| `naive_fused_n128_s3` | 200.7 | 195.6 | 207.6 | 0.0 | 1034 | 1.58x | 1.04x slower | 0.999952 | 0.00981 |
| `opt_fused_cast_n128_s3` | 196.6 | 193.5 | 207.6 | 0.0 | 1056 | 1.61x | 1.02x slower | 0.999955 | 0.00946 |
| `opt_fused_magic_n128_s3` | 192.5 | 187.4 | 207.6 | 0.0 | 1078 | 1.64x | 1.00x | 0.999955 | 0.00946 |
