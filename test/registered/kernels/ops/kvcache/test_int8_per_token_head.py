"""INT8 per-(token, head) KV store and dequant kernels."""

import sys

import pytest
import torch

from sglang.kernels.ops.kvcache.int8_per_token_head import (
    dequant_int8_per_token_head,
    store_int8_per_token_head_kv,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"


def _empty_cache(slots, heads, dim):
    data = torch.zeros((slots, heads, dim), dtype=torch.int8, device=DEVICE)
    scale = torch.zeros((slots, heads), dtype=torch.float32, device=DEVICE)
    return data, scale


@pytest.mark.parametrize("head_dim", [64, 128, 96])
def test_store_scatters_within_half_a_step(head_dim):
    """Each written row reconstructs within half a quantization step; others stay zero.

    K and V come from one fused QKV tensor, so their token stride is not
    heads * head_dim, as in the attention layers.
    """
    torch.manual_seed(0)
    tokens, heads, slots = 37, 4, 256
    qkv = 3 * torch.randn(
        (tokens, 3 * heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )
    k, v = qkv[:, heads : 2 * heads], qkv[:, 2 * heads :]
    loc = torch.randperm(slots, device=DEVICE)[:tokens]
    k_cache, k_scale = _empty_cache(slots, heads, head_dim)
    v_cache, v_scale = _empty_cache(slots, heads, head_dim)

    store_int8_per_token_head_kv(
        k=k,
        v=v,
        loc=loc,
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    untouched = torch.ones(slots, dtype=torch.bool, device=DEVICE)
    untouched[loc] = False
    for x, data, scale in ((k, k_cache, k_scale), (v, v_cache, v_scale)):
        x = x.float()
        amax = x.abs().amax(dim=-1)
        torch.testing.assert_close(scale[loc], amax / 127.0, rtol=1e-6, atol=0)
        assert data[loc].abs().amax(dim=-1).eq(127).all()
        err = (data[loc].float() * scale[loc, :, None] - x).abs()
        assert (err <= scale[loc, :, None] * (0.5 + 1e-4)).all()
        assert data[untouched].eq(0).all() and scale[untouched].eq(0).all()


def test_store_rounds_to_nearest_and_floors_the_scale():
    """Like vLLM's int8_per_token_head: round to nearest (a bare int8 cast would
    truncate) and floor the scale at 1e-6 so an all-zero head stays finite.

    The first head's absmax is 127, so its scale is 1 and the inputs are the
    pre-rounding values; the second head is all zero.
    """
    head_dim = 64
    k = torch.zeros((1, 2, head_dim), dtype=torch.bfloat16, device=DEVICE)
    k[0, 0, :7] = torch.tensor([127.0, 2.6, 2.4, -1.6, -1.4, 0.6, -0.4])
    k_cache, k_scale = _empty_cache(4, 2, head_dim)
    v_cache, v_scale = _empty_cache(4, 2, head_dim)

    store_int8_per_token_head_kv(
        k=k,
        v=-k,
        loc=torch.tensor([2], device=DEVICE),
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    assert k_cache[2, 0, :7].tolist() == [127, 3, 2, -2, -1, 1, 0]
    assert v_cache[2, 0, :7].tolist() == [-127, -3, -2, 2, 1, -1, 0]
    assert k_scale[2, 0].item() == pytest.approx(1.0, rel=1e-6)
    assert k_cache[2, 1].eq(0).all()
    assert k_scale[2, 1].item() == pytest.approx(1e-6, rel=1e-6)


@pytest.mark.parametrize("gather", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dequant_matches_reference(dtype, gather):
    # 1000 slots x 3 heads is not a multiple of the 32 rows per program.
    torch.manual_seed(0)
    data = torch.randint(-128, 128, (1000, 3, 128), dtype=torch.int8, device=DEVICE)
    scale = torch.rand((1000, 3), dtype=torch.float32, device=DEVICE)
    # Repeated and out-of-order slots, as in a prefix gather.
    slots = torch.randint(0, 1000, (777,), device=DEVICE) if gather else None

    out = dequant_int8_per_token_head(data=data, scale=scale, dtype=dtype, slots=slots)

    ref = (data.float() * scale[..., None]).to(dtype)
    if gather:
        ref = ref[slots]
    assert out.dtype == dtype
    assert torch.equal(out, ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
