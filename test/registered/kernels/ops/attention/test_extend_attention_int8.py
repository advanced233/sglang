"""INT8 per-token-head fused extend (regular & unified) against fp32 reference."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.decode_attention_int8 import CONVERSIONS
from sglang.kernels.ops.attention.extend_attention import build_unified_kv_indices
from sglang.kernels.ops.attention.extend_attention_int8 import (
    extend_attention_fwd_int8,
    extend_attention_fwd_unified_int8,
)
from sglang.kernels.ops.kvcache.int8_per_token_head import (
    store_int8_per_token_head_kv,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
# Include zero prefix, partial-tile prefix/extend, and multi-tile lengths.
PREFIX_LENS = [0, 17, 64, 259]
EXTEND_LENS = [1, 33, 128, 195]


def _int8_cache(*, num_slots, num_kv_heads, head_dim):
    data = torch.zeros(
        (num_slots, num_kv_heads, head_dim), dtype=torch.int8, device=DEVICE
    )
    scale = torch.zeros((num_slots, num_kv_heads), dtype=torch.float32, device=DEVICE)
    return data, scale


def _dequant_rows(cache, scale, slots):
    return cache[slots].float() * scale[slots, :, None]


def _reference_extend(
    *,
    q_extend,
    k_prefix_list,
    v_prefix_list,
    k_chunk_list,
    v_chunk_list,
    sm_scale,
    is_causal,
    logit_cap,
    sliding_window_size=-1,
):
    group = q_extend.shape[1] // k_chunk_list[0].shape[1]
    outs = []
    q_offset = 0
    for kp, vp, kc, vc in zip(k_prefix_list, v_prefix_list, k_chunk_list, v_chunk_list):
        plen, elen = kp.shape[0], kc.shape[0]
        q_seq = q_extend[q_offset : q_offset + elen].float()
        q_offset += elen

        k_full = torch.cat([kp, kc], dim=0).repeat_interleave(group, dim=1)
        v_full = torch.cat([vp, vc], dim=0).repeat_interleave(group, dim=1)
        total = plen + elen

        s = torch.einsum("qhd,khd->qhk", q_seq, k_full) * sm_scale
        if logit_cap > 0:
            s = logit_cap * torch.tanh(s / logit_cap)

        q_pos = plen + torch.arange(elen, device=DEVICE)[:, None]
        k_pos = torch.arange(total, device=DEVICE)[None, :]
        mask = torch.ones((elen, total), dtype=torch.bool, device=DEVICE)
        if is_causal:
            mask &= q_pos >= k_pos
        if sliding_window_size > 0:
            mask &= q_pos <= k_pos + sliding_window_size
            if not is_causal:
                mask &= k_pos <= q_pos + sliding_window_size

        s = s.masked_fill(~mask[:, None, :], float("-inf"))
        p = s.softmax(dim=-1)
        outs.append(torch.einsum("qhk,khd->qhd", p, v_full))
    return torch.cat(outs, dim=0)


def _make_batch(
    *, prefix_lens, extend_lens, num_q_heads, num_kv_heads, head_dim, seed=0
):
    torch.manual_seed(seed)
    batch = len(prefix_lens)
    total_prefix = sum(prefix_lens)
    total_extend = sum(extend_lens)
    total = total_prefix + total_extend

    k_all = 2 * torch.randn(
        (total, num_kv_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )
    v_all = 2 * torch.randn(
        (total, num_kv_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )
    q_extend = torch.randn(
        (total_extend, num_q_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )

    # Scatter prefix and extend slots across the pool as in a reused KV pool.
    num_slots = total + 73
    all_slots = torch.randperm(num_slots - 1, device=DEVICE)[:total] + 1
    k_cache, k_scale = _int8_cache(
        num_slots=num_slots, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    v_cache, v_scale = _int8_cache(
        num_slots=num_slots, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    store_int8_per_token_head_kv(
        k=k_all,
        v=v_all,
        loc=all_slots,
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    prefix_slots, extend_slots = [], []
    k_ext_chunks, v_ext_chunks = [], []
    offset = 0
    for plen, elen in zip(prefix_lens, extend_lens):
        prefix_slots.append(all_slots[offset : offset + plen])
        extend_slots.append(all_slots[offset + plen : offset + plen + elen])
        k_ext_chunks.append(k_all[offset + plen : offset + plen + elen])
        v_ext_chunks.append(v_all[offset + plen : offset + plen + elen])
        offset += plen + elen

    qo_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
    qo_indptr[1:] = torch.tensor(extend_lens, dtype=torch.int32, device=DEVICE).cumsum(
        0
    )
    prefix_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
    prefix_indptr[1:] = torch.tensor(
        prefix_lens, dtype=torch.int32, device=DEVICE
    ).cumsum(0)
    prefix_kv_indices = (
        torch.cat(prefix_slots)
        if total_prefix > 0
        else torch.empty(0, dtype=torch.int32, device=DEVICE)
    ).to(torch.int32)
    extend_kv_indices = torch.cat(extend_slots).to(torch.int32)
    k_extend = torch.cat(k_ext_chunks, dim=0).contiguous()
    v_extend = torch.cat(v_ext_chunks, dim=0).contiguous()
    extend_seq_lens = torch.tensor(extend_lens, dtype=torch.int32, device=DEVICE)
    extend_start_loc = qo_indptr[:-1].contiguous()

    return dict(
        q_extend=q_extend,
        k_extend=k_extend,
        v_extend=v_extend,
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        qo_indptr=qo_indptr,
        prefix_indptr=prefix_indptr,
        prefix_kv_indices=prefix_kv_indices,
        extend_kv_indices=extend_kv_indices,
        extend_seq_lens=extend_seq_lens,
        extend_start_loc=extend_start_loc,
        prefix_slots=prefix_slots,
        extend_slots=extend_slots,
        k_ext_chunks=k_ext_chunks,
        v_ext_chunks=v_ext_chunks,
    )


@pytest.mark.parametrize("convert", CONVERSIONS)
@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim,logit_cap",
    [
        (32, 8, 128, 0.0),
        (16, 2, 64, 0.0),
        (8, 8, 128, 30.0),
        (8, 2, 80, 0.0),
    ],
    ids=["gqa4", "gqa8_d64", "mha_logit_cap", "gqa4_d80"],
)
def test_regular_extend_matches_reference(
    convert, is_causal, num_q_heads, num_kv_heads, head_dim, logit_cap
):
    b = _make_batch(
        prefix_lens=PREFIX_LENS,
        extend_lens=EXTEND_LENS,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    o = torch.empty_like(b["q_extend"])
    sm_scale = head_dim**-0.5

    extend_attention_fwd_int8(
        q_extend=b["q_extend"],
        k_extend=b["k_extend"],
        v_extend=b["v_extend"],
        o_extend=o,
        k_buffer=b["k_cache"],
        v_buffer=b["v_cache"],
        k_scale_buffer=b["k_scale"],
        v_scale_buffer=b["v_scale"],
        qo_indptr=b["qo_indptr"],
        kv_indptr=b["prefix_indptr"],
        kv_indices=b["prefix_kv_indices"],
        custom_mask=None,
        is_causal=is_causal,
        mask_indptr=None,
        max_len_extend=max(EXTEND_LENS),
        sm_scale=sm_scale,
        logit_cap=logit_cap,
        convert=convert,
    )

    ref = _reference_extend(
        q_extend=b["q_extend"],
        k_prefix_list=[
            _dequant_rows(b["k_cache"], b["k_scale"], s) for s in b["prefix_slots"]
        ],
        v_prefix_list=[
            _dequant_rows(b["v_cache"], b["v_scale"], s) for s in b["prefix_slots"]
        ],
        k_chunk_list=[x.float() for x in b["k_ext_chunks"]],
        v_chunk_list=[x.float() for x in b["v_ext_chunks"]],
        sm_scale=sm_scale,
        is_causal=is_causal,
        logit_cap=logit_cap,
    )
    torch.testing.assert_close(o.float(), ref, atol=5e-3, rtol=1e-2)


@pytest.mark.parametrize("convert", CONVERSIONS)
@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim,logit_cap",
    [
        (32, 8, 128, 0.0),
        (16, 2, 64, 0.0),
        (8, 8, 128, 30.0),
        (8, 2, 80, 0.0),
    ],
    ids=["gqa4", "gqa8_d64", "mha_logit_cap", "gqa4_d80"],
)
def test_unified_extend_matches_reference(
    convert, is_causal, num_q_heads, num_kv_heads, head_dim, logit_cap
):
    b = _make_batch(
        prefix_lens=PREFIX_LENS,
        extend_lens=EXTEND_LENS,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    unified_kv_indptr, unified_kv_indices, prefix_lens = build_unified_kv_indices(
        b["prefix_indptr"],
        b["prefix_kv_indices"],
        b["extend_start_loc"],
        b["extend_seq_lens"],
        b["extend_kv_indices"],
        len(PREFIX_LENS),
    )
    o = torch.empty_like(b["q_extend"])
    sm_scale = head_dim**-0.5

    extend_attention_fwd_unified_int8(
        q=b["q_extend"],
        o=o,
        k_buffer=b["k_cache"],
        v_buffer=b["v_cache"],
        k_scale_buffer=b["k_scale"],
        v_scale_buffer=b["v_scale"],
        qo_indptr=b["qo_indptr"],
        kv_indptr=unified_kv_indptr,
        kv_indices=unified_kv_indices,
        prefix_lens=prefix_lens.to(torch.int32),
        max_len_extend=max(EXTEND_LENS),
        sm_scale=sm_scale,
        logit_cap=logit_cap,
        is_causal=is_causal,
        convert=convert,
    )

    # Both prefix and current chunk are read from the quantized int8 cache.
    ref = _reference_extend(
        q_extend=b["q_extend"],
        k_prefix_list=[
            _dequant_rows(b["k_cache"], b["k_scale"], s) for s in b["prefix_slots"]
        ],
        v_prefix_list=[
            _dequant_rows(b["v_cache"], b["v_scale"], s) for s in b["prefix_slots"]
        ],
        k_chunk_list=[
            _dequant_rows(b["k_cache"], b["k_scale"], s) for s in b["extend_slots"]
        ],
        v_chunk_list=[
            _dequant_rows(b["v_cache"], b["v_scale"], s) for s in b["extend_slots"]
        ],
        sm_scale=sm_scale,
        is_causal=is_causal,
        logit_cap=logit_cap,
    )
    torch.testing.assert_close(o.float(), ref, atol=5e-3, rtol=1e-2)


def test_unified_extend_prefix_split_invariant():
    """Chunked prefill / prefix-cache hit must match fresh prefill bit-for-bit."""
    torch.manual_seed(7)
    seq_len, num_q_heads, num_kv_heads, head_dim = 300, 32, 8, 128
    q = torch.randn((seq_len, num_q_heads, head_dim), dtype=torch.bfloat16, device=DEVICE)
    k = 2 * torch.randn((seq_len, num_kv_heads, head_dim), dtype=torch.bfloat16, device=DEVICE)
    v = 2 * torch.randn((seq_len, num_kv_heads, head_dim), dtype=torch.bfloat16, device=DEVICE)
    slots = torch.randperm(seq_len + 32, device=DEVICE)[:seq_len] + 1
    k_cache, k_scale = _int8_cache(
        num_slots=seq_len + 33, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    v_cache, v_scale = _int8_cache(
        num_slots=seq_len + 33, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    store_int8_per_token_head_kv(
        k=k,
        v=v,
        loc=slots,
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    def run(prefix_len):
        elen = seq_len - prefix_len
        o = torch.empty((elen, num_q_heads, head_dim), dtype=torch.bfloat16, device=DEVICE)
        extend_attention_fwd_unified_int8(
            q=q[prefix_len:].contiguous(),
            o=o,
            k_buffer=k_cache,
            v_buffer=v_cache,
            k_scale_buffer=k_scale,
            v_scale_buffer=v_scale,
            qo_indptr=torch.tensor([0, elen], dtype=torch.int32, device=DEVICE),
            kv_indptr=torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE),
            kv_indices=slots.to(torch.int64),
            prefix_lens=torch.tensor([prefix_len], dtype=torch.int32, device=DEVICE),
            max_len_extend=elen,
            sm_scale=head_dim**-0.5,
            is_causal=True,
        )
        return o

    o_fresh = run(0)
    for split in (1, 63, 64, 137, 255):
        o_hit = run(split)
        assert torch.equal(o_hit, o_fresh[split:]), (
            f"prefix split {split} diverged from fresh prefill: "
            f"max diff={(o_hit.float() - o_fresh[split:].float()).abs().max().item()}"
        )


def test_regular_and_unified_match_when_chunk_is_dequantized():
    """Passing dequantized chunk K/V to regular extend must match unified extend."""
    b = _make_batch(
        prefix_lens=[0, 53, 140],
        extend_lens=[19, 97, 128],
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        seed=13,
    )
    unified_kv_indptr, unified_kv_indices, prefix_lens = build_unified_kv_indices(
        b["prefix_indptr"],
        b["prefix_kv_indices"],
        b["extend_start_loc"],
        b["extend_seq_lens"],
        b["extend_kv_indices"],
        3,
    )
    # Use fp16 (10-bit mantissa) rather than bf16 (7-bit mantissa) so the
    # dequantized int8 chunk does not lose precision relative to the in-kernel
    # int8 -> fp16 path.
    q_f16 = b["q_extend"].to(torch.float16)
    k_ext_dq = _dequant_rows(b["k_cache"], b["k_scale"], b["extend_kv_indices"]).to(
        torch.float16
    )
    v_ext_dq = _dequant_rows(b["v_cache"], b["v_scale"], b["extend_kv_indices"]).to(
        torch.float16
    )

    o_reg = torch.empty_like(q_f16)
    o_uni = torch.empty_like(q_f16)
    sm_scale = 128**-0.5

    extend_attention_fwd_int8(
        q_extend=q_f16,
        k_extend=k_ext_dq,
        v_extend=v_ext_dq,
        o_extend=o_reg,
        k_buffer=b["k_cache"],
        v_buffer=b["v_cache"],
        k_scale_buffer=b["k_scale"],
        v_scale_buffer=b["v_scale"],
        qo_indptr=b["qo_indptr"],
        kv_indptr=b["prefix_indptr"],
        kv_indices=b["prefix_kv_indices"],
        custom_mask=None,
        is_causal=True,
        mask_indptr=None,
        max_len_extend=128,
        sm_scale=sm_scale,
    )
    extend_attention_fwd_unified_int8(
        q=q_f16,
        o=o_uni,
        k_buffer=b["k_cache"],
        v_buffer=b["v_cache"],
        k_scale_buffer=b["k_scale"],
        v_scale_buffer=b["v_scale"],
        qo_indptr=b["qo_indptr"],
        kv_indptr=unified_kv_indptr,
        kv_indices=unified_kv_indices,
        prefix_lens=prefix_lens.to(torch.int32),
        max_len_extend=128,
        sm_scale=sm_scale,
        is_causal=True,
    )
    torch.testing.assert_close(o_reg.float(), o_uni.float(), atol=5e-3, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
