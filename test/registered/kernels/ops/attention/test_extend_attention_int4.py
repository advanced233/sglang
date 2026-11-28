"""INT4 per-token-head fused extend (regular & unified, with RHT) against fp32 reference."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.decode_attention_int4 import CONVERSIONS
from sglang.kernels.ops.attention.extend_attention import build_unified_kv_indices
from sglang.kernels.ops.attention.extend_attention_int4 import (
    extend_attention_fwd_int4,
    extend_attention_fwd_unified_int4,
)
from sglang.kernels.ops.kvcache.int4_per_token_head import (
    dequant_int4_per_token_head,
    pack_scale_and_zp,
    store_int4_per_token_head_kv,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
# Include zero prefix, partial-tile prefix/extend, and multi-tile lengths.
PREFIX_LENS = [0, 17, 64, 259]
EXTEND_LENS = [1, 33, 128, 195]


def _int4_cache(*, num_slots, num_kv_heads, head_dim):
    data = torch.zeros(
        (num_slots, num_kv_heads, head_dim // 2), dtype=torch.uint8, device=DEVICE
    )
    scale = torch.zeros((num_slots, num_kv_heads), dtype=torch.float32, device=DEVICE)
    return data, scale


def _dequant_rows(cache, scale, slots, *, use_rht):
    if slots.numel() == 0:
        return torch.empty(
            (0, cache.shape[1], cache.shape[2] * 2), dtype=torch.float32, device=DEVICE
        )
    return dequant_int4_per_token_head(
        data=cache, scale=scale, dtype=torch.float32, slots=slots, use_rht=use_rht
    )


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
    sinks=None,
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
        if sinks is not None:
            s_max = torch.maximum(
                s.max(dim=-1, keepdim=True).values, sinks[None, :, None]
            )
            exp_s = torch.exp(s - s_max)
            exp_sink = torch.exp(sinks[None, :] - s_max.squeeze(-1))
            p = exp_s / (exp_s.sum(dim=-1, keepdim=True) + exp_sink[:, :, None])
        else:
            p = s.softmax(dim=-1)
        outs.append(torch.einsum("qhk,khd->qhd", p, v_full))
    return torch.cat(outs, dim=0)


def _make_batch(
    *,
    prefix_lens,
    extend_lens,
    num_q_heads,
    num_kv_heads,
    head_dim,
    use_rht=True,
    seed=0,
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

    num_slots = total + 73
    all_slots = torch.randperm(num_slots - 1, device=DEVICE)[:total] + 1
    k_cache, k_scale = _int4_cache(
        num_slots=num_slots, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    v_cache, v_scale = _int4_cache(
        num_slots=num_slots, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    store_int4_per_token_head_kv(
        k=k_all,
        v=v_all,
        loc=all_slots,
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        use_rht=use_rht,
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
@pytest.mark.parametrize("use_rht", [True, False], ids=["rht", "norht"])
@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim,logit_cap",
    [
        (32, 8, 128, 0.0),
        (16, 2, 64, 0.0),
        (8, 8, 128, 30.0),
        (8, 2, 96, 0.0),
    ],
    ids=["gqa4_d128", "gqa8_d64", "mha_logit_cap", "gqa4_d96"],
)
def test_regular_extend_matches_reference(
    convert, use_rht, is_causal, num_q_heads, num_kv_heads, head_dim, logit_cap
):
    b = _make_batch(
        prefix_lens=PREFIX_LENS,
        extend_lens=EXTEND_LENS,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        use_rht=use_rht,
    )
    o = torch.empty_like(b["q_extend"])
    sm_scale = head_dim**-0.5

    extend_attention_fwd_int4(
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
        use_rht=use_rht,
        convert=convert,
    )

    ref = _reference_extend(
        q_extend=b["q_extend"],
        k_prefix_list=[
            _dequant_rows(b["k_cache"], b["k_scale"], s, use_rht=use_rht)
            for s in b["prefix_slots"]
        ],
        v_prefix_list=[
            _dequant_rows(b["v_cache"], b["v_scale"], s, use_rht=use_rht)
            for s in b["prefix_slots"]
        ],
        k_chunk_list=[x.float() for x in b["k_ext_chunks"]],
        v_chunk_list=[x.float() for x in b["v_ext_chunks"]],
        sm_scale=sm_scale,
        is_causal=is_causal,
        logit_cap=logit_cap,
    )
    torch.testing.assert_close(o.float(), ref, atol=8e-3, rtol=1.5e-2)


@pytest.mark.parametrize("convert", CONVERSIONS)
@pytest.mark.parametrize("use_rht", [True, False], ids=["rht", "norht"])
@pytest.mark.parametrize("is_causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim,logit_cap",
    [
        (32, 8, 128, 0.0),
        (16, 2, 64, 0.0),
        (8, 8, 128, 30.0),
        (8, 2, 96, 0.0),
    ],
    ids=["gqa4_d128", "gqa8_d64", "mha_logit_cap", "gqa4_d96"],
)
def test_unified_extend_matches_reference(
    convert, use_rht, is_causal, num_q_heads, num_kv_heads, head_dim, logit_cap
):
    b = _make_batch(
        prefix_lens=PREFIX_LENS,
        extend_lens=EXTEND_LENS,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        use_rht=use_rht,
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

    extend_attention_fwd_unified_int4(
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
        use_rht=use_rht,
        convert=convert,
    )

    ref = _reference_extend(
        q_extend=b["q_extend"],
        k_prefix_list=[
            _dequant_rows(b["k_cache"], b["k_scale"], s, use_rht=use_rht)
            for s in b["prefix_slots"]
        ],
        v_prefix_list=[
            _dequant_rows(b["v_cache"], b["v_scale"], s, use_rht=use_rht)
            for s in b["prefix_slots"]
        ],
        k_chunk_list=[
            _dequant_rows(b["k_cache"], b["k_scale"], s, use_rht=use_rht)
            for s in b["extend_slots"]
        ],
        v_chunk_list=[
            _dequant_rows(b["v_cache"], b["v_scale"], s, use_rht=use_rht)
            for s in b["extend_slots"]
        ],
        sm_scale=sm_scale,
        is_causal=is_causal,
        logit_cap=logit_cap,
    )
    torch.testing.assert_close(o.float(), ref, atol=8e-3, rtol=1.5e-2)


@pytest.mark.parametrize("sliding_window_size", [31, 128])
def test_sliding_window_and_sinks(sliding_window_size):
    b = _make_batch(
        prefix_lens=[0, 80, 200],
        extend_lens=[19, 75, 90],
        num_q_heads=16,
        num_kv_heads=4,
        head_dim=128,
        use_rht=True,
        seed=5,
    )
    sinks = torch.randn((16,), dtype=torch.float32, device=DEVICE)
    sm_scale = 128**-0.5

    o_reg = torch.empty_like(b["q_extend"])
    extend_attention_fwd_int4(
        q_extend=b["q_extend"],
        k_extend=b["k_extend"],
        v_extend=b["v_extend"],
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
        max_len_extend=90,
        sm_scale=sm_scale,
        sliding_window_size=sliding_window_size,
        sinks=sinks,
    )
    ref_reg = _reference_extend(
        q_extend=b["q_extend"],
        k_prefix_list=[
            _dequant_rows(b["k_cache"], b["k_scale"], s, use_rht=True)
            for s in b["prefix_slots"]
        ],
        v_prefix_list=[
            _dequant_rows(b["v_cache"], b["v_scale"], s, use_rht=True)
            for s in b["prefix_slots"]
        ],
        k_chunk_list=[x.float() for x in b["k_ext_chunks"]],
        v_chunk_list=[x.float() for x in b["v_ext_chunks"]],
        sm_scale=sm_scale,
        is_causal=True,
        logit_cap=0.0,
        sliding_window_size=sliding_window_size,
        sinks=sinks,
    )
    torch.testing.assert_close(o_reg.float(), ref_reg, atol=8e-3, rtol=1.5e-2)

    unified_kv_indptr, unified_kv_indices, prefix_lens = build_unified_kv_indices(
        b["prefix_indptr"],
        b["prefix_kv_indices"],
        b["extend_start_loc"],
        b["extend_seq_lens"],
        b["extend_kv_indices"],
        3,
    )
    o_uni = torch.empty_like(b["q_extend"])
    extend_attention_fwd_unified_int4(
        q=b["q_extend"],
        o=o_uni,
        k_buffer=b["k_cache"],
        v_buffer=b["v_cache"],
        k_scale_buffer=b["k_scale"],
        v_scale_buffer=b["v_scale"],
        qo_indptr=b["qo_indptr"],
        kv_indptr=unified_kv_indptr,
        kv_indices=unified_kv_indices,
        prefix_lens=prefix_lens.to(torch.int32),
        max_len_extend=90,
        sm_scale=sm_scale,
        is_causal=True,
        sliding_window_size=sliding_window_size,
        sinks=sinks,
    )
    ref_uni = _reference_extend(
        q_extend=b["q_extend"],
        k_prefix_list=[
            _dequant_rows(b["k_cache"], b["k_scale"], s, use_rht=True)
            for s in b["prefix_slots"]
        ],
        v_prefix_list=[
            _dequant_rows(b["v_cache"], b["v_scale"], s, use_rht=True)
            for s in b["prefix_slots"]
        ],
        k_chunk_list=[
            _dequant_rows(b["k_cache"], b["k_scale"], s, use_rht=True)
            for s in b["extend_slots"]
        ],
        v_chunk_list=[
            _dequant_rows(b["v_cache"], b["v_scale"], s, use_rht=True)
            for s in b["extend_slots"]
        ],
        sm_scale=sm_scale,
        is_causal=True,
        logit_cap=0.0,
        sliding_window_size=sliding_window_size,
        sinks=sinks,
    )
    torch.testing.assert_close(o_uni.float(), ref_uni, atol=8e-3, rtol=1.5e-2)


def test_unified_extend_chunked_kv_len_invariant():
    """2-pass chunked prefill at non-BLOCK_N-aligned splits must match fresh bit-for-bit."""
    torch.manual_seed(11)
    seq_len, num_q_heads, num_kv_heads, head_dim = 300, 32, 8, 128
    q = torch.randn(
        (seq_len, num_q_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )
    k = 2 * torch.randn(
        (seq_len, num_kv_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )
    v = 2 * torch.randn(
        (seq_len, num_kv_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )
    # Inject large per-token V spikes so tile-wide vs per-row-masked v_scale_max
    # strongly diverge if future tokens leak into earlier rows.
    v[7::19] *= 50.0

    num_slots = seq_len + 33
    slots = torch.randperm(num_slots - 1, device=DEVICE)[:seq_len] + 1

    k_cache_full, k_scale_full = _int4_cache(
        num_slots=num_slots, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    v_cache_full, v_scale_full = _int4_cache(
        num_slots=num_slots, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    store_int4_per_token_head_kv(
        k=k,
        v=v,
        loc=slots,
        k_cache=k_cache_full,
        v_cache=v_cache_full,
        k_scale=k_scale_full,
        v_scale=v_scale_full,
    )
    o_fresh = torch.empty(
        (seq_len, num_q_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )
    extend_attention_fwd_unified_int4(
        q=q,
        o=o_fresh,
        k_buffer=k_cache_full,
        v_buffer=v_cache_full,
        k_scale_buffer=k_scale_full,
        v_scale_buffer=v_scale_full,
        qo_indptr=torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE),
        kv_indptr=torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE),
        kv_indices=slots.to(torch.int64),
        prefix_lens=torch.tensor([0], dtype=torch.int32, device=DEVICE),
        max_len_extend=seq_len,
        sm_scale=head_dim**-0.5,
        is_causal=True,
    )

    sentinel_scale = pack_scale_and_zp(
        torch.full((num_slots, num_kv_heads), 1e4, dtype=torch.float32, device=DEVICE),
        torch.full((num_slots, num_kv_heads), 15, dtype=torch.int32, device=DEVICE),
    )
    for split in (5, 17, 63, 137):
        # Populate unused slots with huge sentinels (q=15, scale=1e4, zp=15) to verify
        # masked positions (including slot 0) never leak into v_scale_max or outputs.
        k_cache = torch.full(
            (num_slots, num_kv_heads, head_dim // 2),
            0xFF,
            dtype=torch.uint8,
            device=DEVICE,
        )
        v_cache = torch.full_like(k_cache, 0xFF)
        k_scale = sentinel_scale.clone()
        v_scale = sentinel_scale.clone()

        # Pass 1: chunk [0, split) with kv_len == split.
        store_int4_per_token_head_kv(
            k=k[:split],
            v=v[:split],
            loc=slots[:split],
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        o_pass1 = torch.empty(
            (split, num_q_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
        )
        extend_attention_fwd_unified_int4(
            q=q[:split].contiguous(),
            o=o_pass1,
            k_buffer=k_cache,
            v_buffer=v_cache,
            k_scale_buffer=k_scale,
            v_scale_buffer=v_scale,
            qo_indptr=torch.tensor([0, split], dtype=torch.int32, device=DEVICE),
            kv_indptr=torch.tensor([0, split], dtype=torch.int32, device=DEVICE),
            kv_indices=slots[:split].to(torch.int64),
            prefix_lens=torch.tensor([0], dtype=torch.int32, device=DEVICE),
            max_len_extend=split,
            sm_scale=head_dim**-0.5,
            is_causal=True,
        )
        assert torch.equal(o_pass1, o_fresh[:split]), (
            f"chunk pass 1 (split={split}) diverged from fresh prefill: "
            f"max diff={(o_pass1.float() - o_fresh[:split].float()).abs().max().item()}"
        )

        # Pass 2: chunk [split, seq_len) with prefix_len == split, kv_len == seq_len.
        store_int4_per_token_head_kv(
            k=k[split:],
            v=v[split:],
            loc=slots[split:],
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        elen = seq_len - split
        o_pass2 = torch.empty(
            (elen, num_q_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
        )
        extend_attention_fwd_unified_int4(
            q=q[split:].contiguous(),
            o=o_pass2,
            k_buffer=k_cache,
            v_buffer=v_cache,
            k_scale_buffer=k_scale,
            v_scale_buffer=v_scale,
            qo_indptr=torch.tensor([0, elen], dtype=torch.int32, device=DEVICE),
            kv_indptr=torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE),
            kv_indices=slots.to(torch.int64),
            prefix_lens=torch.tensor([split], dtype=torch.int32, device=DEVICE),
            max_len_extend=elen,
            sm_scale=head_dim**-0.5,
            is_causal=True,
        )
        assert torch.equal(o_pass2, o_fresh[split:]), (
            f"chunk pass 2 (split={split}) diverged from fresh prefill: "
            f"max diff={(o_pass2.float() - o_fresh[split:].float()).abs().max().item()}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
