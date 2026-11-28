"""INT4 per-(token, head) asymmetric KV store (with RHT) and dequant kernels."""

import sys

import pytest
import torch

from sglang.kernels.ops.kvcache.int4_per_token_head import (
    apply_rht,
    dequant_int4_per_token_head,
    pack_scale_and_zp,
    store_int4_per_token_head_kv,
    unpack_scale_and_zp,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"


def _empty_cache(slots, heads, dim):
    data = torch.zeros((slots, heads, dim // 2), dtype=torch.uint8, device=DEVICE)
    scale = torch.zeros((slots, heads), dtype=torch.float32, device=DEVICE)
    return data, scale


def _round_half_away(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, x + 0.5, x - 0.5).trunc()


def _ref_quantize_int4(x: torch.Tensor, *, use_rht: bool):
    """Pure-PyTorch reference matching vLLM's int4_per_token_head and kvq_ref.py."""
    if use_rht:
        # Reference uses exact float32 butterfly/Hadamard then casts back to input dtype.
        xf = apply_rht(x.cpu().float()).to(x.dtype).float().to(x.device)
    else:
        xf = x.float()
    xmin = xf.amin(dim=-1, keepdim=True)
    xmax = xf.amax(dim=-1, keepdim=True)
    scale = ((xmax - xmin) / 15.0).clamp_min(1e-6)
    zp = _round_half_away(-xmin / scale).clamp(0.0, 15.0)
    q = _round_half_away(xf * (1.0 / scale) + zp).clamp(0.0, 15.0).to(torch.uint8)
    packed = (q[..., 0::2] & 0xF) | ((q[..., 1::2] & 0xF) << 4)
    scale_packed = pack_scale_and_zp(scale.squeeze(-1), zp.squeeze(-1))
    return packed, scale_packed


@pytest.mark.parametrize("use_rht", [True, False], ids=["rht", "norht"])
@pytest.mark.parametrize("head_dim", [64, 128, 96])
def test_store_scatters_and_matches_reference(head_dim, use_rht):
    """Written rows match vLLM/kvq_ref arithmetic byte-for-byte; untouched slots stay zero.

    K and V come from one fused QKV tensor, exercising non-contiguous token strides.
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

    store_int4_per_token_head_kv(
        k=k,
        v=v,
        loc=loc,
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        use_rht=use_rht,
    )

    untouched = torch.ones(slots, dtype=torch.bool, device=DEVICE)
    untouched[loc] = False
    for x, data, scale in ((k, k_cache, k_scale), (v, v_cache, v_scale)):
        ref_packed, ref_scale = _ref_quantize_int4(x, use_rht=use_rht)
        if not use_rht:
            assert torch.equal(data[loc], ref_packed)
            assert torch.equal(
                scale[loc].view(torch.int32), ref_scale.view(torch.int32)
            )
        else:
            # GPU Tensor Core MMA vs CPU fp32 butterfly can differ by 1 ULP in fp32
            # before bf16 rounding, flipping at most rare tie-boundary nibbles by 1.
            lo_got, hi_got = (data[loc] & 0xF).int(), ((data[loc] >> 4) & 0xF).int()
            lo_ref, hi_ref = (ref_packed & 0xF).int(), ((ref_packed >> 4) & 0xF).int()
            assert (lo_got - lo_ref).abs().max().item() <= 1
            assert (hi_got - hi_ref).abs().max().item() <= 1
            exact_rate = (data[loc] == ref_packed).float().mean().item()
            assert exact_rate > 0.99, f"exact byte match rate too low: {exact_rate}"
            got_s, got_zp = unpack_scale_and_zp(scale[loc])
            ref_s, ref_zp = unpack_scale_and_zp(ref_scale)
            torch.testing.assert_close(got_s, ref_s, rtol=1e-2, atol=1e-5)
            assert (got_zp - ref_zp).abs().max().item() <= 1.0
        assert data[untouched].eq(0).all() and scale[untouched].eq(0).all()

    # Fused single-launch RHT store must match two-step apply_rht + unfused store byte-for-byte.
    if use_rht and head_dim in (64, 128):
        k_rot = apply_rht(k).to(k.dtype)
        v_rot = apply_rht(v).to(v.dtype)
        k_c2, k_s2 = _empty_cache(slots, heads, head_dim)
        v_c2, v_s2 = _empty_cache(slots, heads, head_dim)
        store_int4_per_token_head_kv(
            k=k_rot,
            v=v_rot,
            loc=loc,
            k_cache=k_c2,
            v_cache=v_c2,
            k_scale=k_s2,
            v_scale=v_s2,
            use_rht=False,
        )
        assert torch.equal(k_cache[loc], k_c2[loc])
        assert torch.equal(v_cache[loc], v_c2[loc])
        assert torch.equal(k_scale[loc].view(torch.int32), k_s2[loc].view(torch.int32))
        assert torch.equal(v_scale[loc].view(torch.int32), v_s2[loc].view(torch.int32))


def test_store_asymmetric_rounding_zero_floor_and_negative_slot():
    """Verify exact asymmetric range [min, max], round-half-away, 1e-6 floor, and loc < 0 skip."""
    head_dim = 64
    k = torch.zeros((2, 2, head_dim), dtype=torch.bfloat16, device=DEVICE)
    # Row (0, 0): min=-3.0, max=12.0 -> scale=1.0, zp=3.0; exact integer offsets.
    k[0, 0, :8] = torch.tensor([-3.0, 12.0, -0.4, -0.6, 0.4, 0.6, 4.5, 11.4])
    # Row (0, 1): all zeros -> scale floor 1e-6, zp=0, all nibbles 0.
    # Token 1 has loc = -1 (padded) and must be ignored.
    k[1] = 5.0
    k_cache, k_scale = _empty_cache(4, 2, head_dim)
    v_cache, v_scale = _empty_cache(4, 2, head_dim)

    store_int4_per_token_head_kv(
        k=k,
        v=k,
        loc=torch.tensor([2, -1], device=DEVICE),
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        use_rht=False,
    )

    ks, kz = unpack_scale_and_zp(k_scale[2])
    assert ks[0].item() == pytest.approx(1.0, rel=1e-5)
    assert kz[0].item() == 3.0
    # q = round_half_away(x + 3): [0, 15, 3, 2, 3, 4, 8, 14]
    packed = k_cache[2, 0, :4]
    unpacked = []
    for byte in packed.tolist():
        unpacked.extend([byte & 0xF, (byte >> 4) & 0xF])
    assert unpacked == [0, 15, 3, 2, 3, 4, 8, 14]

    # All-zero head hits 1e-6 scale floor with zp=0.
    assert ks[1].item() == pytest.approx(1e-6, rel=1e-5)
    assert kz[1].item() == 0.0
    assert k_cache[2, 1].eq(0).all()
    # Other slots untouched.
    for s in (0, 1, 3):
        assert k_cache[s].eq(0).all() and k_scale[s].eq(0).all()


@pytest.mark.parametrize("use_rht", [True, False], ids=["rht", "norht"])
@pytest.mark.parametrize("gather", [False, True], ids=["full", "gather"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [128, 96])
def test_dequant_matches_reference(head_dim, dtype, gather, use_rht):
    torch.manual_seed(1)
    num_slots, num_heads = 197, 3
    x = 2.0 * torch.randn((num_slots, num_heads, head_dim), dtype=dtype, device=DEVICE)
    data, scale = _empty_cache(num_slots, num_heads, head_dim)
    v_data, v_scale = _empty_cache(num_slots, num_heads, head_dim)
    store_int4_per_token_head_kv(
        k=x,
        v=x,
        loc=torch.arange(num_slots, device=DEVICE),
        k_cache=data,
        v_cache=v_data,
        k_scale=scale,
        v_scale=v_scale,
        use_rht=use_rht,
    )
    slots = torch.randint(0, num_slots, (113,), device=DEVICE) if gather else None

    out = dequant_int4_per_token_head(
        data=data, scale=scale, dtype=dtype, slots=slots, use_rht=use_rht
    )

    s_val, zp = unpack_scale_and_zp(scale)
    lo = (data & 0xF).float()
    hi = ((data >> 4) & 0xF).float()
    q = torch.stack([lo, hi], dim=-1).reshape(num_slots, num_heads, head_dim)
    ref = (q - zp[..., None]) * s_val[..., None]
    if slots is not None:
        ref = ref[slots]
    if use_rht:
        ref = apply_rht(ref.cpu(), inverse=True).to(device=DEVICE, dtype=dtype)
    else:
        ref = ref.to(dtype)

    assert out.dtype == dtype
    if not use_rht:
        assert torch.equal(out, ref)
    else:
        torch.testing.assert_close(out.float(), ref.float(), atol=5e-3, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
