"""Unit tests for KV cache quantization strategies - no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=28, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.test_utils import CustomTestCase


class TestKVCacheQuantRegistry(CustomTestCase):
    """Test the registry and factory function."""

    def test_factory_unknown_raises(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            get_kv_cache_quant_method,
        )

        with self.assertRaises(ValueError):
            get_kv_cache_quant_method("unknown_method")


class TestCPUFP8KVCacheMethod(CustomTestCase):
    def test_static_scale_quantize_and_store(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            CPUFP8KVCacheMethod,
        )

        method = CPUFP8KVCacheMethod()
        buffers = method.create_buffers(4, 2, 8, 1, "cpu")
        loc = torch.tensor([1, 3])
        cache_k = torch.randn(2, 2, 8, dtype=torch.bfloat16)
        cache_v = torch.randn(2, 2, 8, dtype=torch.bfloat16)

        method.quantize_and_store(
            buffers["k_buffer"][0],
            buffers["v_buffer"][0],
            buffers["k_scale_buffer"],
            buffers["v_scale_buffer"],
            loc,
            cache_k,
            cache_v,
            k_scale=0.5,
            v_scale=0.25,
            native_k_scale_buffer=None,
            native_v_scale_buffer=None,
        )

        torch.testing.assert_close(
            buffers["k_buffer"][0][loc].float(),
            (cache_k / 0.5).to(torch.float8_e4m3fn).float(),
        )
        torch.testing.assert_close(
            buffers["v_buffer"][0][loc].float(),
            (cache_v / 0.25).to(torch.float8_e4m3fn).float(),
        )
        self.assertIsNone(buffers["k_scale_buffer"])
        self.assertIsNone(buffers["v_scale_buffer"])
        self.assertEqual(method.compute_cell_size(2, 8, 1, 4), 128)

    def test_defaults_to_unit_scales(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            CPUFP8KVCacheMethod,
        )

        method = CPUFP8KVCacheMethod()
        buffers = method.create_buffers(1, 1, 8, 1, "cpu")
        cache = torch.ones(1, 1, 8, dtype=torch.bfloat16)
        method.quantize_and_store(
            buffers["k_buffer"][0],
            buffers["v_buffer"][0],
            buffers["k_scale_buffer"],
            buffers["v_scale_buffer"],
            torch.tensor([0]),
            cache,
            cache,
        )

        expected = cache.to(torch.float8_e4m3fn)
        torch.testing.assert_close(buffers["k_buffer"][0][0], expected[0])
        torch.testing.assert_close(buffers["v_buffer"][0][0], expected[0])


class TestNVFP4KVCacheMethod(CustomTestCase):
    """Test NVFP4KVCacheMethod buffer creation and properties."""

    def test_properties(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        m = NVFP4KVCacheMethod(num_layers=4, device="cpu", native_scale_layout=True)
        self.assertEqual(m.name, "nvfp4")
        self.assertEqual(m.SCALE_BLOCK_SIZE, 16)
        self.assertTrue(m.needs_dequant_workspace())
        self.assertTrue(m.needs_native_fp4_scales())
        self.assertTrue(m.needs_global_scale())

        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            KVCacheAttentionAccessKind,
        )

        self.assertEqual(
            m.resolve_attention_access("prefill", "trtllm_mha").kind,
            KVCacheAttentionAccessKind.NATIVE_FP4,
        )
        self.assertEqual(
            m.resolve_attention_access("prefill", "flashinfer").kind,
            KVCacheAttentionAccessKind.DEQUANT_WORKSPACE,
        )

    def test_create_buffers_shapes(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        m = NVFP4KVCacheMethod(num_layers=4, device="cpu", native_scale_layout=True)
        size, heads, dim, layers = 64, 8, 128, 4
        bufs = m.create_buffers(size, heads, dim, layers, "cpu")

        self.assertEqual(len(bufs["k_buffer"]), layers)
        self.assertEqual(len(bufs["v_buffer"]), layers)
        self.assertEqual(len(bufs["k_scale_buffer"]), layers)
        self.assertEqual(len(bufs["v_scale_buffer"]), layers)
        self.assertEqual(len(bufs["native_k_scale_buffer"]), layers)
        self.assertEqual(len(bufs["native_v_scale_buffer"]), layers)

        # FP4 packed: (size, heads, dim//2)
        self.assertEqual(bufs["k_buffer"][0].shape, (size, heads, dim // 2))
        # Block scales: (size, heads, dim//16)
        self.assertEqual(bufs["k_scale_buffer"][0].shape, (size, heads, dim // 16))
        self.assertEqual(
            bufs["native_k_scale_buffer"][0].shape,
            (size // 16, heads, 16, dim // 16),
        )
        # Dequant workspace: (size, heads, dim), FP8
        self.assertEqual(bufs["dq_k_buffer"].shape, (size, heads, dim))
        self.assertEqual(bufs["dq_k_buffer"].dtype, torch.float8_e4m3fn)
        self.assertEqual(bufs["store_dtype"], torch.uint8)

    def test_compute_cell_size(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        m = NVFP4KVCacheMethod(num_layers=4, device="cpu", native_scale_layout=True)
        cell = m.compute_cell_size(head_num=8, head_dim=128, num_layers=4, kv_size=1)
        # FP4: 4096, linear scales: 512, native scales: 512, shared DQ: 2048.
        self.assertEqual(cell, 4096 + 512 + 512 + 2048)

    def test_active_prefill_recipe_controls_auxiliary_memory(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        size, heads, dim, layers = 64, 8, 128, 4

        native = NVFP4KVCacheMethod(
            num_layers=layers,
            device="cpu",
            page_size=16,
            native_scale_layout=True,
        )
        native.configure_attention_backends("trtllm_mha", "trtllm_mha")
        native_bufs = native.create_buffers(size, heads, dim, layers, "cpu")
        self.assertIsNone(native_bufs["k_scale_buffer"])
        self.assertIsNone(native_bufs["v_scale_buffer"])
        self.assertIsNone(native_bufs["dq_k_buffer"])
        self.assertIsNotNone(native_bufs["native_k_scale_buffer"])
        self.assertEqual(native.compute_cell_size(heads, dim, layers, 1), 4096 + 512)

        mixed = NVFP4KVCacheMethod(
            num_layers=layers,
            device="cpu",
            page_size=16,
            native_scale_layout=True,
        )
        mixed.configure_attention_backends("flashinfer", "trtllm_mha")
        mixed_bufs = mixed.create_buffers(size, heads, dim, layers, "cpu")
        self.assertIsNotNone(mixed_bufs["k_scale_buffer"])
        self.assertIsNotNone(mixed_bufs["dq_k_buffer"])
        self.assertIsNotNone(mixed_bufs["native_k_scale_buffer"])
        self.assertEqual(
            mixed.compute_cell_size(heads, dim, layers, 1),
            4096 + 512 + 512 + 2048,
        )

    def test_server_args_backend_selection_uses_resolution_projection(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            KVCacheAttentionAccessKind,
            NVFP4KVCacheMethod,
        )

        # Model/backend hooks declare overrides without mutating the raw
        # ServerArgs fields. Pool sizing and allocation must observe the same
        # resolved pair, and must not depend on a private ServerArgs method.
        server_args = SimpleNamespace(
            attention_backend="triton",
            prefill_attention_backend=None,
            decode_attention_backend=None,
            _resolved_overrides=[
                (
                    "test_model_override",
                    {
                        "prefill_attention_backend": "flashinfer",
                        "decode_attention_backend": "trtllm_mha",
                    },
                )
            ],
        )
        method = NVFP4KVCacheMethod(
            num_layers=1,
            device="cpu",
            page_size=16,
            native_scale_layout=True,
        )

        method.configure_attention_backends_from_server_args(server_args)

        accesses = method.active_attention_accesses()
        self.assertEqual(
            [access.kind for access in accesses],
            [
                KVCacheAttentionAccessKind.DEQUANT_WORKSPACE,
                KVCacheAttentionAccessKind.NATIVE_FP4,
            ],
        )

    def test_xqa_recipe_retains_linear_scales(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        size, heads, dim, layers = 64, 8, 128, 4
        xqa = NVFP4KVCacheMethod(
            num_layers=layers,
            device="cpu",
            page_size=16,
            native_scale_layout=False,
        )
        xqa.configure_attention_backends("flashinfer", "trtllm_mha")
        bufs = xqa.create_buffers(size, heads, dim, layers, "cpu")

        self.assertIsNotNone(bufs["k_scale_buffer"])
        self.assertIsNotNone(bufs["dq_k_buffer"])
        self.assertIsNone(bufs["native_k_scale_buffer"])
        self.assertFalse(xqa.needs_native_fp4_scales())
        self.assertEqual(
            xqa.compute_cell_size(heads, dim, layers, 1),
            4096 + 512 + 2048,
        )

    def test_native_v_scale_swizzle_reference(self):
        from sglang.srt.layers.quantization.nvfp4_kv_cache import (
            nvfp4_v_scale_swizzle_indices,
        )

        token = torch.arange(16)[:, None]
        scale = torch.arange(8)[None, :]
        swizzled_token, swizzled_scale = nvfp4_v_scale_swizzle_indices(
            token, scale, scale_dim=8
        )

        # Every logical (token, scale) pair maps bijectively inside each
        # four-token group and agrees with FlashInfer/TRT-LLM's published map.
        flat = (swizzled_token * 8 + swizzled_scale).flatten()
        self.assertEqual(torch.unique(flat).numel(), 16 * 8)
        self.assertEqual(
            (swizzled_token[3, 7].item(), swizzled_scale[3, 7].item()), (3, 7)
        )
        self.assertEqual(
            (swizzled_token[1, 4].item(), swizzled_scale[1, 4].item()), (2, 1)
        )

    def test_scales_init(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        m = NVFP4KVCacheMethod(num_layers=4, device="cpu")
        # Default scales should be 1.0
        self.assertTrue(torch.all(m.k_scales_gpu == 1.0))
        self.assertTrue(torch.all(m.v_scales_gpu == 1.0))
        self.assertEqual(len(m.k_scales_gpu), 4)

    def test_sm100_scale_loading_preserves_uncalibrated_fallback(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        attention = SimpleNamespace(
            layer_id=0,
            k_scale=torch.tensor(1.0),
            v_scale=torch.tensor(1.0),
        )
        model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(attn=attention))]
        )
        method = NVFP4KVCacheMethod(num_layers=1, device="cpu")

        with patch(
            "sglang.srt.layers.quantization.fp4_kv_cache_quant_method.get_platform",
            return_value=SimpleNamespace(is_sm100=True),
        ):
            method.load_scales_from_model(model)

        self.assertEqual(method.get_bmm_scales(0), (1.0, 1.0))

    def test_sm100_scale_loading_converts_calibrated_checkpoint_scales(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            NVFP4KVCacheMethod,
        )

        attention = SimpleNamespace(
            layer_id=0,
            k_scale=torch.tensor(0.002),
            v_scale=torch.tensor(0.003),
        )
        model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(attn=attention))]
        )
        method = NVFP4KVCacheMethod(num_layers=1, device="cpu")

        with patch(
            "sglang.srt.layers.quantization.fp4_kv_cache_quant_method.get_platform",
            return_value=SimpleNamespace(is_sm100=True),
        ):
            method.load_scales_from_model(model)

        k_scale, v_scale = method.get_bmm_scales(0)
        self.assertAlmostEqual(k_scale, 0.012)
        self.assertAlmostEqual(v_scale, 0.018)


class TestFP4MXBlock16KVCacheMethod(CustomTestCase):
    """Test FP4MXBlock16KVCacheMethod buffer creation and roundtrip."""

    def test_properties(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            FP4MXBlock16KVCacheMethod,
            KVCacheAttentionAccessKind,
        )

        m = FP4MXBlock16KVCacheMethod()
        self.assertEqual(m.name, "fp4_mx_block16")
        self.assertFalse(m.needs_dequant_workspace())
        self.assertTrue(m.needs_plain_kv_dequant_read())
        self.assertFalse(m.needs_global_scale())
        self.assertEqual(m.plain_attention_kv_dtype(), torch.bfloat16)
        self.assertEqual(
            m.resolve_attention_access("prefill", "triton").kind,
            KVCacheAttentionAccessKind.PLAIN,
        )
        self.assertEqual(
            m.resolve_attention_access("decode", "trtllm_mha").kind,
            KVCacheAttentionAccessKind.PLAIN,
        )
        self.assertIsNone(m.resolve_attention_access("prefill", "flashinfer"))
        self.assertIsNone(m.resolve_attention_access("decode", "flashinfer"))

    def test_create_buffers_shapes(self):
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            FP4MXBlock16KVCacheMethod,
        )

        m = FP4MXBlock16KVCacheMethod()
        size, heads, dim, layers = 64, 8, 128, 4
        bufs = m.create_buffers(size, heads, dim, layers, "cpu")

        self.assertEqual(len(bufs["k_buffer"]), layers)
        self.assertEqual(bufs["k_buffer"][0].shape, (size, heads, dim // 2))
        # Block-16 FP4 flattens head dims for scales
        self.assertEqual(bufs["k_scale_buffer"][0].shape, (size, (heads * dim) // 16))
        self.assertIsNone(bufs["dq_k_buffer"])
        self.assertIsNone(bufs["dq_v_buffer"])

    def test_quantize_dequantize_roundtrip_cpu(self):
        """Test block-16 FP4 quantize->dequantize roundtrip on CPU."""
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            FP4MXBlock16KVCacheMethod,
        )

        m = FP4MXBlock16KVCacheMethod()
        size, heads, dim = 32, 8, 128
        bufs = m.create_buffers(size, heads, dim, 1, "cpu")

        k = torch.randn(4, heads, dim, dtype=torch.bfloat16)
        v = torch.randn(4, heads, dim, dtype=torch.bfloat16)
        loc = torch.arange(4)

        # Quantize
        m.quantize_and_store(
            bufs["k_buffer"][0],
            bufs["v_buffer"][0],
            bufs["k_scale_buffer"][0],
            bufs["v_scale_buffer"][0],
            loc,
            k,
            v,
        )

        # Dequantize
        k_fp4 = bufs["k_buffer"][0][loc]
        k_scales = bufs["k_scale_buffer"][0][loc]
        v_fp4 = bufs["v_buffer"][0][loc]
        v_scales = bufs["v_scale_buffer"][0][loc]
        k_out, v_out = m.dequantize_prev_kv(k_fp4, k_scales, v_fp4, v_scales, 0)

        self.assertEqual(k_out.shape, (4, heads, dim))
        self.assertEqual(v_out.shape, (4, heads, dim))
        self.assertEqual(k_out.dtype, torch.bfloat16)
        self.assertEqual(v_out.dtype, torch.bfloat16)


class TestFP4MXBlock16KVQuantizeUtil(CustomTestCase):
    """Test the existing block-16 FP4 FP4MXBlock16KVQuantizeUtil roundtrip."""

    def test_roundtrip_cpu(self):
        from sglang.srt.layers.quantization.kvfp4_tensor import (
            FP4MXBlock16KVQuantizeUtil,
        )

        x = torch.randn(4, 8, 128, dtype=torch.bfloat16)
        packed, scales = FP4MXBlock16KVQuantizeUtil.batched_quantize(x)
        reconstructed = FP4MXBlock16KVQuantizeUtil.batched_dequantize(packed, scales)

        self.assertEqual(reconstructed.shape, x.shape)
        rel_error = (
            x.float() - reconstructed.float()
        ).abs().mean() / x.float().abs().mean()
        self.assertLess(rel_error, 0.5)


class TestInt8PerTokenHeadKVCacheMethod(CustomTestCase):
    def test_storage_and_access_contract(self):
        """The pool, the store kernel and the hook all rely on these together."""
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            KVCacheAttentionAccessKind,
            get_kv_cache_quant_method,
            resolve_kv_cache_quant,
        )

        name = resolve_kv_cache_quant("int8_per_token_head")
        m = get_kv_cache_quant_method(name)
        self.assertEqual(m.kv_storage_dtype(), torch.int8)
        self.assertTrue(m.needs_plain_kv_dequant_read())
        self.assertFalse(m.needs_dequant_workspace())
        self.assertIsNone(m.plain_attention_kv_dtype())
        # Both Triton prefill/extend and decode read int8 + scales in-kernel.
        for phase in ("prefill", "decode"):
            self.assertEqual(
                m.resolve_attention_access(phase, "triton").kind,
                KVCacheAttentionAccessKind.FUSED_DEQUANT,
            )
            self.assertIsNone(m.resolve_attention_access(phase, "flashinfer"))

    def test_uses_fused_int8_kv_gating(self):
        """Triton fused read activates for both phases and rejects page_size != 1 / DCP."""
        from sglang.srt.layers.attention.triton_int8_kv import uses_fused_int8_kv
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            FP4MXBlock16KVCacheMethod,
            Int8PerTokenHeadKVCacheMethod,
        )

        m = Int8PerTokenHeadKVCacheMethod()
        for phase in ("prefill", "decode"):
            self.assertTrue(
                uses_fused_int8_kv(
                    quant_method=m, phase=phase, page_size=1, dcp_size=1
                )
            )
            with self.assertRaises(NotImplementedError):
                uses_fused_int8_kv(
                    quant_method=m, phase=phase, page_size=16, dcp_size=1
                )
            with self.assertRaises(NotImplementedError):
                uses_fused_int8_kv(
                    quant_method=m, phase=phase, page_size=1, dcp_size=2
                )
            self.assertFalse(
                uses_fused_int8_kv(
                    quant_method=None, phase=phase, page_size=1, dcp_size=1
                )
            )
            self.assertFalse(
                uses_fused_int8_kv(
                    quant_method=FP4MXBlock16KVCacheMethod(),
                    phase=phase,
                    page_size=1,
                    dcp_size=1,
                )
            )
            from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
                Int4PerTokenHeadKVCacheMethod,
            )

            self.assertFalse(
                uses_fused_int8_kv(
                    quant_method=Int4PerTokenHeadKVCacheMethod(),
                    phase=phase,
                    page_size=1,
                    dcp_size=1,
                )
            )

    def test_compatibility_allows_deterministic_inference(self):
        """Deterministic inference is allowed now that unified extend is fused."""
        from sglang.srt.arg_groups.kv_cache_hook import (
            handle_int8_kv_cache_compatibility,
        )

        server_args = SimpleNamespace(
            kv_cache_dtype="int8_per_token_head",
            attention_backend="triton",
            prefill_attention_backend=None,
            decode_attention_backend=None,
            enable_hierarchical_cache=False,
            enable_lmcache=False,
            enable_unified_memory=False,
            disaggregation_mode="null",
            speculative_algorithm=None,
            enable_deterministic_inference=True,
        )
        with (
            patch(
                "sglang.srt.arg_groups.kv_cache_hook.get_platform",
                return_value=SimpleNamespace(is_cuda=True),
            ),
            patch(
                "sglang.srt.arg_groups.kv_cache_hook.use_mla_backend",
                return_value=False,
            ),
            patch(
                "sglang.srt.arg_groups.kv_cache_hook.resolving_view",
                side_effect=lambda x: x,
            ),
            patch(
                "sglang.srt.arg_groups.kv_cache_hook.resolved_view",
                side_effect=lambda x: x,
            ),
            patch(
                "sglang.srt.arg_groups.kv_cache_hook.attention_backends_of",
                return_value=("triton", "triton"),
            ),
        ):
            handle_int8_kv_cache_compatibility(server_args)
            server_args.enable_hierarchical_cache = True
            with self.assertRaises(ValueError):
                handle_int8_kv_cache_compatibility(server_args)

    def test_cell_size_matches_allocation(self):
        """Capacity planning must count exactly the bytes create_buffers allocates."""
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            Int8PerTokenHeadKVCacheMethod,
        )

        m = Int8PerTokenHeadKVCacheMethod()
        size, heads, dim, layers = 16, 8, 128, 3
        bufs = m.create_buffers(size, heads, dim, layers, "cpu")
        allocated = sum(
            t.numel() * t.element_size()
            for key in ("k_buffer", "v_buffer", "k_scale_buffer", "v_scale_buffer")
            for t in bufs[key]
        )
        kv_size = torch.int8.itemsize
        self.assertEqual(
            allocated, size * m.compute_cell_size(heads, dim, layers, kv_size)
        )


class TestInt4PerTokenHeadKVCacheMethod(CustomTestCase):
    def test_storage_and_access_contract(self):
        """Recipe resolves to packed uint8 storage and FUSED_DEQUANT on Triton only."""
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            KVCacheAttentionAccessKind,
            get_kv_cache_quant_method,
            resolve_kv_cache_quant,
        )

        name = resolve_kv_cache_quant("int4_per_token_head")
        self.assertEqual(name, "int4_per_token_head")
        m = get_kv_cache_quant_method(name)
        self.assertEqual(m.kv_storage_dtype(), torch.uint8)
        self.assertTrue(m.needs_plain_kv_dequant_read())
        self.assertFalse(m.needs_dequant_workspace())
        self.assertIsNone(m.plain_attention_kv_dtype())
        for phase in ("prefill", "decode"):
            access = m.resolve_attention_access(phase, "triton")
            self.assertIsNotNone(access)
            self.assertEqual(access.kind, KVCacheAttentionAccessKind.FUSED_DEQUANT)
            self.assertEqual(access.scale_recipe, "int4_per_token_head")
            self.assertIsNone(m.resolve_attention_access(phase, "flashinfer"))

    def test_uses_fused_int4_kv_gating(self):
        """Triton fused INT4 activates only for int4_per_token_head and enforces page_size=1/no DCP."""
        from sglang.srt.layers.attention.triton_int4_kv import uses_fused_int4_kv
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            FP4MXBlock16KVCacheMethod,
            Int4PerTokenHeadKVCacheMethod,
            Int8PerTokenHeadKVCacheMethod,
        )

        m = Int4PerTokenHeadKVCacheMethod()
        for phase in ("prefill", "decode"):
            self.assertTrue(
                uses_fused_int4_kv(
                    quant_method=m, phase=phase, page_size=1, dcp_size=1
                )
            )
            with self.assertRaises(NotImplementedError):
                uses_fused_int4_kv(
                    quant_method=m, phase=phase, page_size=16, dcp_size=1
                )
            with self.assertRaises(NotImplementedError):
                uses_fused_int4_kv(
                    quant_method=m, phase=phase, page_size=1, dcp_size=2
                )
            for other in (None, FP4MXBlock16KVCacheMethod(), Int8PerTokenHeadKVCacheMethod()):
                self.assertFalse(
                    uses_fused_int4_kv(
                        quant_method=other, phase=phase, page_size=1, dcp_size=1
                    )
                )

    def test_compatibility_hook_boundaries(self):
        """Allows Triton + deterministic inference; rejects non-Triton, MLA, and HiCache."""
        from sglang.srt.arg_groups.kv_cache_hook import (
            handle_int4_kv_cache_compatibility,
        )

        server_args = SimpleNamespace(
            kv_cache_dtype="int4_per_token_head",
            attention_backend="triton",
            prefill_attention_backend=None,
            decode_attention_backend=None,
            enable_hierarchical_cache=False,
            enable_lmcache=False,
            enable_unified_memory=False,
            disaggregation_mode="null",
            speculative_algorithm=None,
            enable_deterministic_inference=True,
        )
        with (
            patch(
                "sglang.srt.arg_groups.kv_cache_hook.get_platform",
                return_value=SimpleNamespace(is_cuda=True),
            ),
            patch(
                "sglang.srt.arg_groups.kv_cache_hook.use_mla_backend",
                return_value=False,
            ),
            patch(
                "sglang.srt.arg_groups.kv_cache_hook.resolving_view",
                side_effect=lambda x: x,
            ),
            patch(
                "sglang.srt.arg_groups.kv_cache_hook.resolved_view",
                side_effect=lambda x: x,
            ),
            patch(
                "sglang.srt.arg_groups.kv_cache_hook.attention_backends_of",
                return_value=("triton", "triton"),
            ),
        ):
            handle_int4_kv_cache_compatibility(server_args)
            server_args.enable_hierarchical_cache = True
            with self.assertRaises(ValueError):
                handle_int4_kv_cache_compatibility(server_args)

        server_args.enable_hierarchical_cache = False
        with (
            patch(
                "sglang.srt.arg_groups.kv_cache_hook.get_platform",
                return_value=SimpleNamespace(is_cuda=True),
            ),
            patch(
                "sglang.srt.arg_groups.kv_cache_hook.use_mla_backend",
                return_value=False,
            ),
            patch(
                "sglang.srt.arg_groups.kv_cache_hook.resolving_view",
                side_effect=lambda x: x,
            ),
            patch(
                "sglang.srt.arg_groups.kv_cache_hook.resolved_view",
                side_effect=lambda x: x,
            ),
            patch(
                "sglang.srt.arg_groups.kv_cache_hook.attention_backends_of",
                return_value=("flashinfer", "flashinfer"),
            ),
        ):
            with self.assertRaises(ValueError):
                handle_int4_kv_cache_compatibility(server_args)

    def test_cell_size_matches_allocation_and_rejects_odd_head_dim(self):
        """Capacity planning must count exactly the bytes create_buffers allocates."""
        from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
            Int4PerTokenHeadKVCacheMethod,
        )

        m = Int4PerTokenHeadKVCacheMethod()
        size, heads, dim, layers = 16, 8, 128, 3
        bufs = m.create_buffers(size, heads, dim, layers, "cpu")
        self.assertEqual(bufs["k_buffer"][0].shape, (size, heads, dim // 2))
        self.assertEqual(bufs["k_scale_buffer"][0].shape, (size, heads))
        allocated = sum(
            t.numel() * t.element_size()
            for key in ("k_buffer", "v_buffer", "k_scale_buffer", "v_scale_buffer")
            for t in bufs[key]
        )
        kv_size = torch.uint8.itemsize
        cell_size = m.compute_cell_size(heads, dim, layers, kv_size)
        # 2 * (64 data + 4 scale) = 136 bytes per (layer, head).
        self.assertEqual(cell_size, heads * layers * 136)
        self.assertEqual(allocated, size * cell_size)

        with self.assertRaises(ValueError):
            m.create_buffers(size, heads, 127, layers, "cpu")
        with self.assertRaises(ValueError):
            m.compute_cell_size(heads, 127, layers, kv_size)

    def test_scale_zp_steganography_boundaries(self):
        """Low 4 mantissa bits round-trip all zp in [0, 15] within <= 15 / 2^23 scale error."""
        from sglang.kernels.ops.kvcache.int4_per_token_head import (
            pack_scale_and_zp,
            unpack_scale_and_zp,
        )

        scales = torch.tensor(
            [1e-6, 1e-5, 1e-3, 0.12345678, 1.0, 3.1415927, 1e4], dtype=torch.float32
        )
        zps = torch.arange(16, dtype=torch.int32)
        s_grid, z_grid = torch.meshgrid(scales, zps, indexing="ij")

        packed = pack_scale_and_zp(s_grid, z_grid)
        self.assertTrue(torch.isfinite(packed).all())
        self.assertTrue((packed > 0).all())

        out_s, out_zp = unpack_scale_and_zp(packed)
        self.assertTrue(torch.equal(out_zp.to(torch.int32), z_grid))
        rel_err = (out_s - s_grid).abs() / s_grid
        self.assertTrue((rel_err <= 15.0 / (1 << 23) + 1e-12).all())

        # Unwritten / zeroed cache slot must decode to scale=0.0, zp=0.0 without NaN.
        zero_s, zero_zp = unpack_scale_and_zp(torch.zeros(4, dtype=torch.float32))
        self.assertTrue(zero_s.eq(0).all())
        self.assertTrue(zero_zp.eq(0).all())

    def test_rht_orthogonality_seed_and_block_diagonal(self):
        """RHT uses vLLM seed 0x9E3779B9, is invertible, and scales inner products by b."""
        from sglang.kernels.ops.kvcache.int4_per_token_head import (
            RHT_SEED,
            apply_rht,
            get_rht_signs,
            get_rht_split_matrices,
            rht_block_size,
        )

        self.assertEqual(RHT_SEED, 0x9E3779B9)
        gen = torch.Generator(device="cpu").manual_seed(0x9E3779B9)
        expected_signs = 2.0 * torch.bernoulli(torch.full((128,), 0.5), generator=gen) - 1.0
        self.assertTrue(torch.equal(get_rht_signs(128, torch.device("cpu")), expected_signs))

        torch.manual_seed(0)
        for head_dim, expected_b in ((64, 64), (128, 128), (96, 32), (80, 16)):
            self.assertEqual(rht_block_size(head_dim), expected_b)
            x = torch.randn((5, 3, head_dim), dtype=torch.float32)
            y = torch.randn((5, 3, head_dim), dtype=torch.float32)
            xr = apply_rht(x)
            yr = apply_rht(y)
            # Inner product scales by b = largest power-of-2 divisor of head_dim.
            torch.testing.assert_close(
                (xr * yr).sum(dim=-1), (x * y).sum(dim=-1) * expected_b, rtol=1e-5, atol=1e-4
            )
            # Exact inverse roundtrip.
            torch.testing.assert_close(apply_rht(xr, inverse=True), x, rtol=1e-5, atol=1e-5)

        # Split matrices M_even / M_odd must match even/odd slices of apply_rht.
        x = torch.randn((4, 128), dtype=torch.float32)
        m_even, m_odd = get_rht_split_matrices(128, torch.float32, torch.device("cpu"))
        xr = apply_rht(x)
        torch.testing.assert_close(x @ m_even, xr[:, 0::2], rtol=1e-5, atol=1e-4)
        torch.testing.assert_close(x @ m_odd, xr[:, 1::2], rtol=1e-5, atol=1e-4)
        x_rec = ((xr[:, 0::2] @ m_even.T) + (xr[:, 1::2] @ m_odd.T)) / 128.0
        torch.testing.assert_close(x_rec, x, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
