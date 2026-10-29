"""MHATokenToKVPool with the int8_per_token_head recipe.

Covers the wiring the kernel tests cannot see: set_kv_buffer routes through the
recipe into the int8 data and fp32 scale buffers of the right layer, and the
plain read returns the whole layer dequantized for the Triton backend.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    Int8PerTokenHeadKVCacheMethod,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

POOL_SIZE = 63  # buffers get POOL_SIZE + page_size rows
HEAD_NUM = 2
HEAD_DIM = 128


@unittest.skipUnless(torch.cuda.is_available(), "the INT8 store is a Triton kernel")
class TestInt8PerTokenHeadPool(unittest.TestCase):
    def test_plain_read_returns_the_dequantized_layer(self):
        pool = MHATokenToKVPool(
            size=POOL_SIZE,
            page_size=1,
            dtype=torch.int8,
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
            enable_alt_stream=False,
            quant_method=Int8PerTokenHeadKVCacheMethod(),
        )
        torch.manual_seed(0)
        loc = torch.tensor([5, 1, 40, POOL_SIZE], device="cuda")
        shape = (len(loc), HEAD_NUM, HEAD_DIM)
        k = 3 * torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        v = 3 * torch.randn(shape, dtype=torch.bfloat16, device="cuda")

        pool.set_kv_buffer(SimpleNamespace(layer_id=1), loc, k, v)

        untouched = torch.ones(POOL_SIZE + 1, dtype=torch.bool, device="cuda")
        untouched[loc] = False
        for x, read in ((k, pool.get_key_buffer(1)), (v, pool.get_value_buffer(1))):
            self.assertEqual(read.dtype, torch.bfloat16)
            self.assertEqual(read.shape, (POOL_SIZE + 1, HEAD_NUM, HEAD_DIM))
            # Half a quantization step, plus the bf16 rounding of the result.
            step = x.float().abs().amax(dim=-1, keepdim=True) / 127.0
            bound = step / 2 + x.float().abs() / 128
            self.assertTrue(((read[loc].float() - x.float()).abs() <= bound).all())
            self.assertTrue(read[untouched].eq(0).all())
        self.assertTrue(pool.get_key_buffer(0).eq(0).all())


if __name__ == "__main__":
    unittest.main()
