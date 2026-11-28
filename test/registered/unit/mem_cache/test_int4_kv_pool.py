"""MHATokenToKVPool with the int4_per_token_head recipe.

Covers the wiring kernel tests cannot see: set_kv_buffer routes through the
recipe into packed uint8 data and fp32 scale+zp buffers of the right layer,
get_raw_kv_buffer exposes the packed buffers directly, and plain reads return
the inverse-RHT dequantized layer.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    Int4PerTokenHeadKVCacheMethod,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

POOL_SIZE = 63  # buffers get POOL_SIZE + page_size rows
HEAD_NUM = 2
HEAD_DIM = 128


@unittest.skipUnless(torch.cuda.is_available(), "the INT4 store is a Triton kernel")
class TestInt4PerTokenHeadPool(CustomTestCase):
    def test_store_raw_and_plain_read_roundtrip(self):
        pool = MHATokenToKVPool(
            size=POOL_SIZE,
            page_size=1,
            dtype=torch.uint8,
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
            enable_alt_stream=False,
            quant_method=Int4PerTokenHeadKVCacheMethod(),
        )
        torch.manual_seed(0)
        loc = torch.tensor([5, 1, 40, POOL_SIZE], device="cuda")
        shape = (len(loc), HEAD_NUM, HEAD_DIM)
        k = 2.0 * torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        v = 2.0 * torch.randn(shape, dtype=torch.bfloat16, device="cuda")

        pool.set_kv_buffer(SimpleNamespace(layer_id=1), loc, k, v)

        k_raw, v_raw, k_scale, v_scale = pool.get_raw_kv_buffer(1)
        self.assertEqual(k_raw.dtype, torch.uint8)
        self.assertEqual(k_raw.shape, (POOL_SIZE + 1, HEAD_NUM, HEAD_DIM // 2))
        self.assertEqual(v_raw.shape, (POOL_SIZE + 1, HEAD_NUM, HEAD_DIM // 2))
        self.assertEqual(k_scale.dtype, torch.float32)
        self.assertEqual(k_scale.shape, (POOL_SIZE + 1, HEAD_NUM))
        self.assertEqual(v_scale.shape, (POOL_SIZE + 1, HEAD_NUM))

        untouched = torch.ones(POOL_SIZE + 1, dtype=torch.bool, device="cuda")
        untouched[loc] = False
        for x, read in ((k, pool.get_key_buffer(1)), (v, pool.get_value_buffer(1))):
            self.assertEqual(read.dtype, torch.bfloat16)
            self.assertEqual(read.shape, (POOL_SIZE + 1, HEAD_NUM, HEAD_DIM))
            cos = torch.nn.functional.cosine_similarity(
                read[loc].float().flatten(), x.float().flatten(), dim=0
            )
            self.assertGreater(cos.item(), 0.985)
            self.assertTrue(read[untouched].eq(0).all())
        self.assertTrue(pool.get_key_buffer(0).eq(0).all())


if __name__ == "__main__":
    unittest.main()
