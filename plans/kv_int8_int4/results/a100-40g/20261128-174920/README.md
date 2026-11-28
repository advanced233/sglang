# M4 运行记录

- 代码提交 `90fff88`。
- cpu_tests: 1 failed, 26 passed, 20 warnings in 91.78s (0:01:31)
- FAILED test/registered/unit/layers/quantization/test_fp4_kv_cache_quant_method.py::TestInt4PerTokenHeadKVCacheMethod::test_scale_zp_steganography_boundaries
- gpu_tests: 11 failed, 112 passed, 20 warnings in 242.57s (0:04:02)
- FAILED test/registered/kernels/ops/kvcache/test_int4_per_token_head.py::test_store_scatters_and_matches_reference[64-norht]
- FAILED test/registered/kernels/ops/kvcache/test_int4_per_token_head.py::test_store_scatters_and_matches_reference[128-norht]
- FAILED test/registered/kernels/ops/kvcache/test_int4_per_token_head.py::test_store_scatters_and_matches_reference[96-norht]
- FAILED test/registered/kernels/ops/attention/test_extend_attention_int4.py::test_regular_extend_matches_reference[gqa4_d96-causal-rht-magic]
- FAILED test/registered/kernels/ops/attention/test_extend_attention_int4.py::test_regular_extend_matches_reference[gqa4_d96-causal-rht-cast]
- FAILED test/registered/kernels/ops/attention/test_extend_attention_int4.py::test_regular_extend_matches_reference[gqa4_d96-noncausal-rht-magic]
- FAILED test/registered/kernels/ops/attention/test_extend_attention_int4.py::test_regular_extend_matches_reference[gqa4_d96-noncausal-rht-cast]
- FAILED test/registered/kernels/ops/attention/test_extend_attention_int4.py::test_unified_extend_matches_reference[gqa4_d96-causal-rht-magic]
- FAILED test/registered/kernels/ops/attention/test_extend_attention_int4.py::test_unified_extend_matches_reference[gqa4_d96-causal-rht-cast]
- FAILED test/registered/kernels/ops/attention/test_extend_attention_int4.py::test_unified_extend_matches_reference[gqa4_d96-noncausal-rht-magic]
- FAILED test/registered/kernels/ops/attention/test_extend_attention_int4.py::test_unified_extend_matches_reference[gqa4_d96-noncausal-rht-cast]
- triton_attn_test: 1 passed, 16 deselected, 20 warnings in 16.00s

```text
start=2026-11-28T17:49:26+08:00 code=90fff88230744c2b00465b6e310498c906c78226
parity_gate_policy=all no-RHT exact flags true; both RHT modes byte and scale match rates 1.0; attention cosine >= 0.999 (conservative operational threshold)
cpu_exit=1
gpu_exit=1
attention_exit=0
parity_exit=0
parity_gate=fail
diag_and_serve=skipped_test_or_parity_failure
experiments_end=2026-11-28T17:56:51+08:00

```

