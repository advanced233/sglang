# M3c Extend Kernel & KL Consistency Summary

## Kernel Microbenchmark (cold L2, 32 Q / 8 KV / D=128)

| Shape (B × prefix + extend) | Impl | Median ms | vs BF16 | vs M3b | Temp MB | Cos vs BF16 | Max abs vs exact |
|---|---|---:|---:|---:|---:|---:|---:|
| prefix-hit (4×8192+512) | `bf16-regular` | 4.042 | 1.00× | 1.04× | 0.0 | 0.999997 | 0.00033 |
| prefix-hit (4×8192+512) | `bf16-unified` | 3.966 | 1.02× | 1.06× | 0.0 | 0.999997 | 0.00033 |
| prefix-hit (4×8192+512) | `int8-m3b-gather` | 4.190 | 0.96× | 1.00× | 134.5 | 0.999953 | 0.00062 |
| prefix-hit (4×8192+512) | `int8-fused-regular` | 5.763 | 0.70× | 0.73× | 0.0 | 0.999957 | 0.00035 |
| prefix-hit (4×8192+512) | `int8-fused-unified` | 6.299 | 0.64× | 0.67× | 0.0 | 0.999956 | 0.00037 |
| late-chunk (1×14336+2048) | `bf16-regular` | 7.259 | 1.00× | 1.00× | 0.0 | 0.999997 | 0.00028 |
| late-chunk (1×14336+2048) | `bf16-unified` | 7.613 | 0.95× | 0.96× | 0.0 | 0.999997 | 0.00028 |
| late-chunk (1×14336+2048) | `int8-m3b-gather` | 7.274 | 1.00× | 1.00× | 58.8 | 0.999955 | 0.00033 |
| late-chunk (1×14336+2048) | `int8-fused-regular` | 10.325 | 0.70× | 0.70× | 0.0 | 0.999959 | 0.00024 |
| late-chunk (1×14336+2048) | `int8-fused-unified` | 11.488 | 0.63× | 0.63× | 0.0 | 0.999956 | 0.00024 |
| fresh-prefill (4×0+4096) | `bf16-regular` | 6.255 | 1.00× | 1.01× | 0.0 | 0.999998 | 0.00957 |
| fresh-prefill (4×0+4096) | `bf16-unified` | 14.296 | 0.44× | 0.44× | 0.0 | 0.999998 | 0.00957 |
| fresh-prefill (4×0+4096) | `int8-m3b-gather` | 6.289 | 0.99× | 1.00× | 0.0 | 0.999998 | 0.00957 |
| fresh-prefill (4×0+4096) | `int8-fused-regular` | 6.968 | 0.90× | 0.90× | 0.0 | 0.999998 | 0.00957 |
| fresh-prefill (4×0+4096) | `int8-fused-unified` | 11.314 | 0.55× | 0.56× | 0.0 | 0.999961 | 0.00843 |

## Extend Tile Sweep

| Shape | Impl | Tile (M×N, w, s, conv) | Median ms | vs BF16 | vs M3b | Max abs vs exact |
|---|---|---|---:|---:|---:|---:|
| prefix-hit | `int8-fused-regular` | `64×128, w=4, s=1, magic` | 4.152 | 0.97× | 1.01× | 0.00035 |
| prefix-hit | `int8-fused-regular` | `64×64, w=4, s=2, magic` | 4.719 | 0.86× | 0.89× | 0.00035 |
| prefix-hit | `int8-fused-regular` | `64×64, w=4, s=1, magic` | 4.799 | 0.84× | 0.87× | 0.00035 |
| prefix-hit | `int8-fused-regular` | `64×64, w=4, s=1, cast` | 5.203 | 0.78× | 0.81× | 0.00035 |
| prefix-hit | `int8-fused-regular` | `128×64, w=8, s=1, magic` | 5.768 | 0.70× | 0.73× | 0.00035 |
| prefix-hit | `int8-fused-regular` | `64×128, w=8, s=1, magic` | 5.895 | 0.69× | 0.71× | 0.00035 |
| prefix-hit | `int8-fused-regular` | `128×64, w=4, s=1, magic` | 6.691 | 0.60× | 0.63× | 0.00035 |
| prefix-hit | `int8-fused-regular` | `128×64, w=8, s=1, cast` | 7.798 | 0.52× | 0.54× | 0.00035 |
| prefix-hit | `int8-fused-regular` | `128×64, w=8, s=2, magic` | 8.575 | 0.47× | 0.49× | 0.00035 |
| prefix-hit | `int8-fused-regular` | `128×128, w=8, s=1, magic` | 15.143 | 0.27× | 0.28× | 0.00035 |
| prefix-hit | `int8-fused-unified` | `64×64, w=4, s=2, magic` | 3.820 | 1.06× | 1.10× | 0.00037 |
| prefix-hit | `int8-fused-unified` | `64×128, w=4, s=1, magic` | 4.356 | 0.93× | 0.96× | 0.00037 |
| prefix-hit | `int8-fused-unified` | `64×64, w=4, s=1, magic` | 4.397 | 0.92× | 0.95× | 0.00037 |
| prefix-hit | `int8-fused-unified` | `64×64, w=4, s=1, cast` | 4.583 | 0.88× | 0.91× | 0.00037 |
| prefix-hit | `int8-fused-unified` | `64×128, w=8, s=1, magic` | 5.281 | 0.77× | 0.79× | 0.00037 |
| prefix-hit | `int8-fused-unified` | `128×64, w=8, s=1, magic` | 6.301 | 0.64× | 0.67× | 0.00037 |
| prefix-hit | `int8-fused-unified` | `128×64, w=4, s=1, magic` | 6.437 | 0.63× | 0.65× | 0.00037 |
| prefix-hit | `int8-fused-unified` | `128×128, w=8, s=1, magic` | 9.030 | 0.45× | 0.46× | 0.00037 |
| prefix-hit | `int8-fused-unified` | `128×64, w=8, s=1, cast` | 9.194 | 0.44× | 0.46× | 0.00037 |
| prefix-hit | `int8-fused-unified` | `128×64, w=8, s=2, magic` | 25.138 | 0.16× | 0.17× | 0.00037 |
| late-chunk | `int8-fused-regular` | `64×128, w=4, s=1, magic` | 7.426 | 0.98× | 0.98× | 0.00024 |
| late-chunk | `int8-fused-regular` | `64×64, w=4, s=2, magic` | 8.313 | 0.87× | 0.88× | 0.00024 |
| late-chunk | `int8-fused-regular` | `64×64, w=4, s=1, magic` | 8.698 | 0.83× | 0.84× | 0.00024 |
| late-chunk | `int8-fused-regular` | `64×64, w=4, s=1, cast` | 9.685 | 0.75× | 0.75× | 0.00024 |
| late-chunk | `int8-fused-regular` | `128×64, w=8, s=1, magic` | 10.324 | 0.70× | 0.70× | 0.00024 |
| late-chunk | `int8-fused-regular` | `64×128, w=8, s=1, magic` | 10.688 | 0.68× | 0.68× | 0.00024 |
| late-chunk | `int8-fused-regular` | `128×64, w=4, s=1, magic` | 12.044 | 0.60× | 0.60× | 0.00024 |
| late-chunk | `int8-fused-regular` | `128×64, w=8, s=1, cast` | 13.887 | 0.52× | 0.52× | 0.00024 |
| late-chunk | `int8-fused-regular` | `128×64, w=8, s=2, magic` | 15.138 | 0.48× | 0.48× | 0.00024 |
| late-chunk | `int8-fused-regular` | `128×128, w=8, s=1, magic` | 26.121 | 0.28× | 0.28× | 0.00024 |
| late-chunk | `int8-fused-unified` | `64×64, w=4, s=2, magic` | 6.864 | 1.06× | 1.06× | 0.00024 |
| late-chunk | `int8-fused-unified` | `64×128, w=4, s=1, magic` | 7.755 | 0.94× | 0.94× | 0.00025 |
| late-chunk | `int8-fused-unified` | `64×64, w=4, s=1, magic` | 8.002 | 0.91× | 0.91× | 0.00024 |
| late-chunk | `int8-fused-unified` | `64×64, w=4, s=1, cast` | 8.228 | 0.88× | 0.88× | 0.00024 |
| late-chunk | `int8-fused-unified` | `64×128, w=8, s=1, magic` | 9.616 | 0.75× | 0.76× | 0.00025 |
| late-chunk | `int8-fused-unified` | `128×64, w=8, s=1, magic` | 11.484 | 0.63× | 0.63× | 0.00024 |
| late-chunk | `int8-fused-unified` | `128×64, w=4, s=1, magic` | 12.109 | 0.60× | 0.60× | 0.00024 |
| late-chunk | `int8-fused-unified` | `128×128, w=8, s=1, magic` | 15.836 | 0.46× | 0.46× | 0.00025 |
| late-chunk | `int8-fused-unified` | `128×64, w=8, s=1, cast` | 16.817 | 0.43× | 0.43× | 0.00024 |
| late-chunk | `int8-fused-unified` | `128×64, w=8, s=2, magic` | 45.323 | 0.16× | 0.16× | 0.00024 |

