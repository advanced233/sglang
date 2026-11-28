# M3c Extend Kernel & KL Consistency Summary

## Kernel Microbenchmark (cold L2, 32 Q / 8 KV / D=128)

| Shape (B × prefix + extend) | Impl | Median ms | vs BF16 | vs M3b | Temp MB | Cos vs BF16 | Max abs vs exact |
|---|---|---:|---:|---:|---:|---:|---:|
| prefix-hit (4×8192+512) | `bf16-regular` | 4.040 | 1.00× | - | 0.0 | 0.999997 | 0.00033 |
| prefix-hit (4×8192+512) | `int8-fused-regular` | 4.087 | 0.99× | - | 0.0 | 0.999957 | 0.00035 |
| prefix-hit (4×8192+512) | `int4-fused-regular` | 5.224 | 0.77× | - | 0.0 | 0.989968 | 0.00039 |
| prefix-hit (4×8192+512) | `int4-fused-unified` | 5.191 | 0.78× | - | 0.0 | 0.989659 | 0.00039 |
| late-chunk (1×14336+2048) | `bf16-regular` | 7.249 | 1.00× | - | 0.0 | 0.999997 | 0.00028 |
| late-chunk (1×14336+2048) | `int8-fused-regular` | 7.228 | 1.00× | - | 0.0 | 0.999959 | 0.00024 |
| late-chunk (1×14336+2048) | `int4-fused-regular` | 9.260 | 0.78× | - | 0.0 | 0.990446 | 0.00026 |
| late-chunk (1×14336+2048) | `int4-fused-unified` | 9.417 | 0.77× | - | 0.0 | 0.989815 | 0.00024 |
| fresh-prefill (4×0+4096) | `bf16-regular` | 6.220 | 1.00× | - | 0.0 | 0.999998 | 0.00957 |
| fresh-prefill (4×0+4096) | `int8-fused-regular` | 7.442 | 0.84× | - | 0.0 | 0.999998 | 0.00957 |
| fresh-prefill (4×0+4096) | `int4-fused-regular` | 8.312 | 0.75× | - | 0.0 | 0.999998 | 0.00957 |
| fresh-prefill (4×0+4096) | `int4-fused-unified` | 10.405 | 0.60× | - | 0.0 | 0.990873 | 0.00799 |

## Extend Tile Sweep

| Shape | Impl | Tile (M×N, w, s, conv) | Median ms | vs BF16 | vs M3b | Max abs vs exact |
|---|---|---|---:|---:|---:|---:|
| prefix-hit | `int8-fused-regular` | `64×128, w=4, s=1, magic` | 4.099 | 0.99× | - | 0.00035 |
| prefix-hit | `int8-fused-regular` | `64×64, w=4, s=2, magic` | 4.460 | 0.91× | - | 0.00035 |
| prefix-hit | `int8-fused-regular` | `64×64, w=4, s=1, magic` | 4.801 | 0.84× | - | 0.00035 |
| prefix-hit | `int8-fused-regular` | `64×64, w=4, s=1, cast` | 5.167 | 0.78× | - | 0.00035 |
| prefix-hit | `int8-fused-regular` | `64×128, w=8, s=1, magic` | 5.705 | 0.71× | - | 0.00035 |
| prefix-hit | `int8-fused-regular` | `128×64, w=8, s=1, magic` | 6.328 | 0.64× | - | 0.00035 |
| prefix-hit | `int8-fused-regular` | `128×64, w=4, s=1, magic` | 6.631 | 0.61× | - | 0.00035 |
| prefix-hit | `int8-fused-regular` | `128×64, w=8, s=1, cast` | 7.715 | 0.52× | - | 0.00035 |
| prefix-hit | `int8-fused-regular` | `128×64, w=8, s=2, magic` | 8.581 | 0.47× | - | 0.00035 |
| prefix-hit | `int8-fused-regular` | `128×128, w=8, s=1, magic` | 15.173 | 0.27× | - | 0.00035 |
| prefix-hit | `int4-fused-regular` | `128×64(p64), w=8, s=1(ps=2), magic` | 5.069 | 0.80× | - | 0.00039 |
| prefix-hit | `int4-fused-regular` | `64×128(p64), w=4, s=1(ps=2), magic` | 5.234 | 0.77× | - | 0.00039 |
| prefix-hit | `int4-fused-regular` | `64×64(p64), w=4, s=1(ps=2), magic` | 5.291 | 0.76× | - | 0.00039 |
| prefix-hit | `int4-fused-regular` | `64×64(p64), w=4, s=2(ps=2), magic` | 5.404 | 0.75× | - | 0.00039 |
| prefix-hit | `int4-fused-regular` | `64×128(p64), w=4, s=1(ps=1), magic` | 5.506 | 0.73× | - | 0.00039 |
| prefix-hit | `int4-fused-regular` | `64×128(p128), w=4, s=1(ps=1), magic` | 5.553 | 0.73× | - | 0.00039 |
| prefix-hit | `int4-fused-regular` | `128×64(p64), w=4, s=1(ps=2), magic` | 7.428 | 0.54× | - | 0.00039 |
| prefix-hit | `int4-fused-regular` | `64×128(p64), w=4, s=1(ps=2), cast` | 7.532 | 0.54× | - | 0.00039 |
| prefix-hit | `int4-fused-regular` | `64×128(p128), w=4, s=1(ps=2), magic` | 7.835 | 0.52× | - | 0.00039 |
| prefix-hit | `int4-fused-unified` | `64×64, w=4, s=2, magic` | 5.245 | 0.77× | - | 0.00039 |
| prefix-hit | `int4-fused-unified` | `64×64, w=4, s=1, magic` | 6.864 | 0.59× | - | 0.00039 |
| prefix-hit | `int4-fused-unified` | `64×64, w=4, s=2, cast` | 6.986 | 0.58× | - | 0.00039 |
| prefix-hit | `int4-fused-unified` | `64×128, w=4, s=1, magic` | 10.653 | 0.38× | - | 0.00039 |
| prefix-hit | `int4-fused-unified` | `64×128, w=4, s=2, magic` | 10.969 | 0.37× | - | 0.00039 |
| late-chunk | `int8-fused-regular` | `64×128, w=4, s=1, magic` | 7.224 | 1.00× | - | 0.00024 |
| late-chunk | `int8-fused-regular` | `64×64, w=4, s=2, magic` | 7.775 | 0.93× | - | 0.00024 |
| late-chunk | `int8-fused-regular` | `64×64, w=4, s=1, magic` | 8.670 | 0.84× | - | 0.00024 |
| late-chunk | `int8-fused-regular` | `64×64, w=4, s=1, cast` | 9.414 | 0.77× | - | 0.00024 |
| late-chunk | `int8-fused-regular` | `64×128, w=8, s=1, magic` | 10.323 | 0.70× | - | 0.00024 |
| late-chunk | `int8-fused-regular` | `128×64, w=8, s=1, magic` | 11.349 | 0.64× | - | 0.00024 |
| late-chunk | `int8-fused-regular` | `128×64, w=4, s=1, magic` | 11.846 | 0.61× | - | 0.00024 |
| late-chunk | `int8-fused-regular` | `128×64, w=8, s=1, cast` | 13.819 | 0.52× | - | 0.00024 |
| late-chunk | `int8-fused-regular` | `128×64, w=8, s=2, magic` | 14.962 | 0.48× | - | 0.00024 |
| late-chunk | `int8-fused-regular` | `128×128, w=8, s=1, magic` | 26.121 | 0.28× | - | 0.00024 |
| late-chunk | `int4-fused-regular` | `128×64(p64), w=8, s=1(ps=2), magic` | 9.050 | 0.80× | - | 0.00026 |
| late-chunk | `int4-fused-regular` | `64×128(p64), w=4, s=1(ps=2), magic` | 9.352 | 0.78× | - | 0.00026 |
| late-chunk | `int4-fused-regular` | `64×64(p64), w=4, s=1(ps=2), magic` | 9.363 | 0.77× | - | 0.00026 |
| late-chunk | `int4-fused-regular` | `64×128(p64), w=4, s=1(ps=1), magic` | 9.538 | 0.76× | - | 0.00026 |
| late-chunk | `int4-fused-regular` | `64×64(p64), w=4, s=2(ps=2), magic` | 9.625 | 0.75× | - | 0.00026 |
| late-chunk | `int4-fused-regular` | `64×128(p128), w=4, s=1(ps=1), magic` | 9.748 | 0.74× | - | 0.00024 |
| late-chunk | `int4-fused-regular` | `128×64(p64), w=4, s=1(ps=2), magic` | 13.046 | 0.56× | - | 0.00026 |
| late-chunk | `int4-fused-regular` | `64×128(p64), w=4, s=1(ps=2), cast` | 13.058 | 0.56× | - | 0.00026 |
| late-chunk | `int4-fused-regular` | `64×128(p128), w=4, s=1(ps=2), magic` | 13.555 | 0.53× | - | 0.00024 |
| late-chunk | `int4-fused-unified` | `64×64, w=4, s=2, magic` | 9.422 | 0.77× | - | 0.00024 |
| late-chunk | `int4-fused-unified` | `64×64, w=4, s=2, cast` | 12.709 | 0.57× | - | 0.00024 |
| late-chunk | `int4-fused-unified` | `64×64, w=4, s=1, magic` | 12.892 | 0.56× | - | 0.00024 |
| late-chunk | `int4-fused-unified` | `64×128, w=4, s=1, magic` | 18.234 | 0.40× | - | 0.00024 |
| late-chunk | `int4-fused-unified` | `64×128, w=4, s=2, magic` | 19.417 | 0.37× | - | 0.00024 |

