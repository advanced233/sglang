"""Split-KV decode attention over the int4_per_token_head KV cache.

Reads packed uint8 K/V ``[slots, kv_heads, head_dim // 2]`` and fp32
per-(slot, head) scales ``[slots, kv_heads]`` (with 4-bit zero-points embedded
in the low 4 mantissa bits) straight from the pool.

Nibble unpacking uses PTX bit-manipulation into fp16 (``pack=4``), and both
split-dot QK and PV run in fp16 Tensor Cores with fp32 accumulation:
- K zero-point correction: ``(qk_even + qk_odd - q_sum * k_zp) * (k_scale * sm_scale_rot)``.
- V scale & zero-point: ``p_v = (p * (v_scale / v_scale_max)).to(fp16)``;
  ``acc_{even,odd} += dot(p_v, v_{even,odd}) * v_scale_max - sum(p * v_scale * v_zp)``.

For power-of-two head dims in ``(32, 64, 128)``, the forward RHT on Q is fused
into Stage 1's prologue and the inverse RHT on O is fused into Stage 2's
epilogue, keeping decode at 2 kernel launches with zero standalone RHT passes.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.decode_attention import (
    _GROUPED_BLOCK_H,
    _MIN_BLOCK_KV,
    _decode_softmax_reducev_fwd,
    _grouped_head_tiles,
    tanh,
)
from sglang.kernels.ops.kvcache.int4_per_token_head import (
    apply_rht,
    can_fuse_rht,
    get_rht_split_matrices,
    rht_block_size,
)

# "magic" unpacks 4 packed bytes (8 uint4 nibbles) into fp16 via PTX bit
# manipulation; "cast" uses Triton's elementwise shift/mask and cast.
CONVERSIONS = ("magic", "cast")

_STAGE2_BLOCK_H = 16


@triton.jit
def _unpack_lo_u4_to_f16(packed, CONVERT: tl.constexpr):
    """Unpack low nibbles (even indices, values in [0, 15]) from uint8 to fp16."""
    if CONVERT == "magic":
        # Mask low nibbles, insert into mantissa of fp16 1024.0 (0x6400),
        # yielding exact 1024 + lo, then subtract 1024.0 (0x64006400).
        return tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b32 u, magic, bias;
            mov.b32 magic, 0x64646464;
            mov.b32 bias, 0x64006400;
            and.b32 u, $2, 0x0f0f0f0f;
            prmt.b32 $0, u, magic, 0x5150;
            prmt.b32 $1, u, magic, 0x5352;
            sub.f16x2 $0, $0, bias;
            sub.f16x2 $1, $1, bias;
            }
            """,
            constraints="=r,=r,r",
            args=[packed],
            dtype=tl.float16,
            is_pure=True,
            pack=4,
        )
    else:
        return (packed & 0xF).to(tl.float16)


@triton.jit
def _unpack_hi_u4_to_f16(packed, CONVERT: tl.constexpr):
    """Unpack high nibbles (odd indices, values in [0, 15]) from uint8 to fp16."""
    if CONVERT == "magic":
        # Mask high nibbles (hi << 4), insert into mantissa of 1024.0 (0x6400)
        # yielding exact 1024 + 16*hi, then FMA by 1/16 (0x2c00) - 64 (0xd400).
        return tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b32 u, magic, scale, bias;
            mov.b32 magic, 0x64646464;
            mov.b32 scale, 0x2c002c00;
            mov.b32 bias, 0xd400d400;
            and.b32 u, $2, 0xf0f0f0f0;
            prmt.b32 $0, u, magic, 0x5150;
            prmt.b32 $1, u, magic, 0x5352;
            fma.rn.f16x2 $0, $0, scale, bias;
            fma.rn.f16x2 $1, $1, scale, bias;
            }
            """,
            constraints="=r,=r,r",
            args=[packed],
            dtype=tl.float16,
            is_pure=True,
            pack=4,
        )
    else:
        return ((packed >> 4) & 0xF).to(tl.float16)


@triton.jit
def _fwd_grouped_kernel_stage1_int4(
    Q,
    K_Buffer,
    V_Buffer,
    K_Scale,
    V_Scale,
    M_Even_K,
    M_Odd_K,
    sm_scale_rot,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs: tl.int64,
    stride_qh: tl.int64,
    stride_buf_kbs: tl.int64,
    stride_buf_kh: tl.int64,
    stride_buf_vbs: tl.int64,
    stride_buf_vh: tl.int64,
    stride_ks_bs: tl.int64,
    stride_vs_bs: tl.int64,
    stride_mid_ob: tl.int64,
    stride_mid_oh: tl.int64,
    stride_mid_os: tl.int64,
    stride_lse_b: tl.int64,
    stride_lse_h: tl.int64,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_PACKED_DK: tl.constexpr,
    BLOCK_PACKED_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    PACKED_LK: tl.constexpr,
    PACKED_LV: tl.constexpr,
    CONVERT: tl.constexpr,
    FUSED_RHT_K: tl.constexpr,
    INV_SQRT_RHT_K: tl.constexpr,
    SPLIT_OUT: tl.constexpr,
):
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_pk = tl.arange(0, BLOCK_PACKED_DK)
    offs_pv = tl.arange(0, BLOCK_PACKED_DV)
    mask_pk = offs_pk < PACKED_LK
    mask_pv = offs_pv < PACKED_LV

    kv_start = tl.load(kv_indptr + cur_batch)
    seq_len = tl.load(kv_indptr + cur_batch + 1) - kv_start
    kv_splits = tl.load(num_kv_splits + cur_batch)
    kv_len_per_split = tl.cdiv(tl.cdiv(seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    split_start = kv_len_per_split * split_kv_id
    split_end = tl.minimum(split_start + kv_len_per_split, seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_even = tl.zeros([BLOCK_H, BLOCK_PACKED_DV], dtype=tl.float32)
    acc_odd = tl.zeros([BLOCK_H, BLOCK_PACKED_DV], dtype=tl.float32)

    if split_end > split_start:
        q_base = Q + cur_batch * stride_qbs + cur_head[:, None] * stride_qh
        if FUSED_RHT_K:
            offs_d = tl.arange(0, BLOCK_DMODEL)
            q_raw = tl.load(q_base + offs_d[None, :], mask=mask_h[:, None], other=0.0)
            m_even = tl.load(
                M_Even_K + offs_d[:, None] * BLOCK_PACKED_DK + offs_pk[None, :]
            )
            q_even = (tl.dot(q_raw, m_even, out_dtype=tl.float32) * INV_SQRT_RHT_K).to(
                tl.float16
            )
            m_odd = tl.load(
                M_Odd_K + offs_d[:, None] * BLOCK_PACKED_DK + offs_pk[None, :]
            )
            q_odd = (tl.dot(q_raw, m_odd, out_dtype=tl.float32) * INV_SQRT_RHT_K).to(
                tl.float16
            )
        else:
            q_even = tl.load(
                q_base + (offs_pk * 2)[None, :],
                mask=mask_h[:, None] & mask_pk[None, :],
                other=0.0,
            ).to(tl.float16)
            q_odd = tl.load(
                q_base + (offs_pk * 2 + 1)[None, :],
                mask=mask_h[:, None] & mask_pk[None, :],
                other=0.0,
            ).to(tl.float16)

        # Compute sum(Q) from the fp16 operands used in tl.dot so constant/zero
        # keys cancel exactly against q_sum * k_zp.
        q_sum = tl.sum(q_even.to(tl.float32), axis=1) + tl.sum(
            q_odd.to(tl.float32), axis=1
        )

        base_k = cur_kv_head * stride_buf_kh + offs_pk[None, :]
        base_v = cur_kv_head * stride_buf_vh + offs_pv[None, :]

        for start_n in tl.range(split_start, split_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_end
            kv_loc = tl.load(kv_indices + kv_start + offs_n, mask=mask_n, other=0).to(
                tl.int64
            )

            # Row-major [BLOCK_N, BLOCK_PACKED_DK] enables 128-bit cp.async
            # loads and contiguous pack=4 PTX nibble unpacking.
            k_packed = tl.load(
                K_Buffer + kv_loc[:, None] * stride_buf_kbs + base_k,
                mask=mask_n[:, None] & mask_pk[None, :],
                other=0,
            )
            ks_raw = tl.load(
                K_Scale + kv_loc * stride_ks_bs + cur_kv_head,
                mask=mask_n,
                other=0.0,
            )
            ks_bits = ks_raw.to(tl.int32, bitcast=True)
            k_zp = (ks_bits & 0xF).to(tl.float32)
            k_scale = (ks_bits & -16).to(tl.float32, bitcast=True)

            k_even = _unpack_lo_u4_to_f16(k_packed, CONVERT)
            k_odd = _unpack_hi_u4_to_f16(k_packed, CONVERT)
            qk = tl.dot(q_even, tl.trans(k_even)) + tl.dot(q_odd, tl.trans(k_odd))
            qk = (qk - q_sum[:, None] * k_zp[None, :]) * (k_scale * sm_scale_rot)[
                None, :
            ]
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)
            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))

            v_packed = tl.load(
                V_Buffer + kv_loc[:, None] * stride_buf_vbs + base_v,
                mask=mask_n[:, None] & mask_pv[None, :],
                other=0,
            )
            vs_raw = tl.load(
                V_Scale + kv_loc * stride_vs_bs + cur_kv_head,
                mask=mask_n,
                other=0.0,
            )
            vs_bits = vs_raw.to(tl.int32, bitcast=True)
            v_zp = (vs_bits & 0xF).to(tl.float32)
            v_scale = (vs_bits & -16).to(tl.float32, bitcast=True)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])

            # Normalize by tile max so P * v_scale stays in fp16's normal range;
            # compute the zero-point sum from the same fp16 p_v so (v - zp) cancels cleanly.
            v_scale_max = tl.maximum(tl.max(v_scale, 0), 1e-30)
            p_v = (p * (v_scale / v_scale_max)[None, :]).to(tl.float16)
            pv_zp_sum = tl.sum(p_v.to(tl.float32) * v_zp[None, :], axis=1)

            v_even = _unpack_lo_u4_to_f16(v_packed, CONVERT)
            v_odd = _unpack_hi_u4_to_f16(v_packed, CONVERT)
            acc_even = (
                acc_even * re_scale[:, None]
                + (tl.dot(p_v, v_even) - pv_zp_sum[:, None]) * v_scale_max
            )
            acc_odd = (
                acc_odd * re_scale[:, None]
                + (tl.dot(p_v, v_odd) - pv_zp_sum[:, None]) * v_scale_max
            )

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        o_even = acc_even / e_sum[:, None]
        o_odd = acc_odd / e_sum[:, None]
        out_base = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
        )
        if SPLIT_OUT:
            tl.store(
                Att_Out + out_base + offs_pv[None, :],
                o_even,
                mask=mask_h[:, None] & mask_pv[None, :],
            )
            tl.store(
                Att_Out + out_base + PACKED_LV + offs_pv[None, :],
                o_odd,
                mask=mask_h[:, None] & mask_pv[None, :],
            )
        else:
            tl.store(
                Att_Out + out_base + (offs_pv * 2)[None, :],
                o_even,
                mask=mask_h[:, None] & mask_pv[None, :],
            )
            tl.store(
                Att_Out + out_base + (offs_pv * 2 + 1)[None, :],
                o_odd,
                mask=mask_h[:, None] & mask_pv[None, :],
            )
        offs_lse = cur_batch * stride_lse_b + cur_head * stride_lse_h + split_kv_id
        tl.store(Att_Lse + offs_lse, e_max + tl.log(e_sum), mask=mask_h)


@triton.jit
def _fwd_kernel_stage2_int4_rht(
    Mid_O,
    Mid_O_1,
    O,
    M_Even_V,
    M_Odd_V,
    kv_indptr,
    num_kv_splits,
    sink_ptr,
    stride_mid_ob: tl.int64,
    stride_mid_oh: tl.int64,
    stride_mid_os: tl.int64,
    stride_lse_b: tl.int64,
    stride_lse_h: tl.int64,
    stride_obs: tl.int64,
    stride_oh: tl.int64,
    q_head_num: tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    PACKED_LV: tl.constexpr,
    INV_RHT_NORM_V: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    """Split-KV log-sum-exp reduction with fused inverse RHT epilogue."""
    cur_batch = tl.program_id(0).to(tl.int64)
    offs_h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < q_head_num

    kv_start = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - kv_start
    kv_splits = tl.load(num_kv_splits + cur_batch)
    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )

    offs_pv = tl.arange(0, PACKED_LV)
    offs_dv = tl.arange(0, BLOCK_DV)

    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    acc_even = tl.zeros([BLOCK_H, PACKED_LV], dtype=tl.float32)
    acc_odd = tl.zeros([BLOCK_H, PACKED_LV], dtype=tl.float32)

    mid_base = cur_batch * stride_mid_ob + offs_h[:, None] * stride_mid_oh
    lse_base = cur_batch * stride_lse_b + offs_h * stride_lse_h

    for split_kv_id in tl.range(0, MAX_KV_SPLITS, num_stages=2):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            split_off = split_kv_id * stride_mid_os
            tv_even = tl.load(
                Mid_O + mid_base + split_off + offs_pv[None, :],
                mask=mask_h[:, None],
                other=0.0,
            )
            tv_odd = tl.load(
                Mid_O + mid_base + split_off + PACKED_LV + offs_pv[None, :],
                mask=mask_h[:, None],
                other=0.0,
            )
            # Load masked-out heads with finite 0.0 so e_max - n_e_max never NaNs.
            tlogic = tl.load(Mid_O_1 + lse_base + split_kv_id, mask=mask_h, other=0.0)
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            exp_logic = tl.exp(tlogic - n_e_max)
            acc_even = acc_even * old_scale[:, None] + exp_logic[:, None] * tv_even
            acc_odd = acc_odd * old_scale[:, None] + exp_logic[:, None] * tv_odd
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + offs_h, mask=mask_h, other=0.0)
        e_sum += tl.where(mask_h, tl.exp(cur_sink - e_max), 0.0)

    e_sum_safe = tl.where(e_sum == 0.0, 1.0, e_sum)
    o_even = tl.where(mask_h[:, None], acc_even / e_sum_safe[:, None], 0.0)
    o_odd = tl.where(mask_h[:, None], acc_odd / e_sum_safe[:, None], 0.0)

    # Row-normalize to [-1, 1] before fp16 Tensor Core inverse RHT so outputs of
    # arbitrary scale retain full 10-bit mantissa accuracy with zero overflow.
    row_max = tl.maximum(
        tl.maximum(tl.max(tl.abs(o_even), axis=1), tl.max(tl.abs(o_odd), axis=1)),
        1e-30,
    )
    oe_f16 = (o_even / row_max[:, None]).to(tl.float16)
    oo_f16 = (o_odd / row_max[:, None]).to(tl.float16)

    m_even = tl.load(M_Even_V + offs_dv[:, None] * PACKED_LV + offs_pv[None, :])
    out = tl.dot(oe_f16, tl.trans(m_even), out_dtype=tl.float32)
    m_odd = tl.load(M_Odd_V + offs_dv[:, None] * PACKED_LV + offs_pv[None, :])
    out += tl.dot(oo_f16, tl.trans(m_odd), out_dtype=tl.float32)
    out = out * (row_max[:, None] * INV_RHT_NORM_V)

    tl.store(
        O + cur_batch * stride_obs + offs_h[:, None] * stride_oh + offs_dv[None, :],
        out.to(O.dtype.element_ty),
        mask=mask_h[:, None],
    )


def _default_tile_int4(head_dim: int) -> tuple[int, int]:
    # Packed INT4 uses half the bytes of INT8 per token (64 B vs 128 B at D=128),
    # so 128 tokens x 3 stages takes only ~51 KB shared memory per CTA.
    if head_dim <= 128:
        return 128, 3
    return 64, 2


def decode_attention_fwd_int4(
    *,
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    k_scale_buffer: torch.Tensor,
    v_scale_buffer: torch.Tensor,
    o: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_kv_splits: int,
    sm_scale: float,
    logit_cap: float = 0.0,
    sinks: Optional[torch.Tensor] = None,
    use_rht: bool = True,
    block_n: Optional[int] = None,
    num_warps: int = 4,
    num_stages: Optional[int] = None,
    convert: str = "magic",
) -> None:
    """GQA/MHA decode over an int4_per_token_head KV cache with RHT.

    ``q`` / ``o``: ``[batch, q_heads, head_dim]``. ``k_buffer`` / ``v_buffer``:
    uint8 ``[slots, kv_heads, head_dim // 2]``. ``k_scale_buffer`` /
    ``v_scale_buffer``: fp32 ``[slots, kv_heads]`` with 4-bit zero-points in
    the low 4 mantissa bits.
    """
    assert k_buffer.dtype == torch.uint8 and v_buffer.dtype == torch.uint8
    assert k_scale_buffer.dtype == torch.float32
    assert v_scale_buffer.dtype == torch.float32
    assert k_scale_buffer.stride(1) == 1 and v_scale_buffer.stride(1) == 1
    assert k_buffer.stride(2) == 1 and v_buffer.stride(2) == 1
    assert q.stride(2) == 1 and o.stride(2) == 1
    assert attn_lse.stride(2) == 1
    assert max_kv_splits == attn_logits.shape[2]
    assert convert in CONVERSIONS, convert

    batch, head_num = q.shape[0], q.shape[1]
    if batch == 0:
        return
    kv_group_num = head_num // k_buffer.shape[1]
    packed_lk, packed_lv = k_buffer.shape[-1], v_buffer.shape[-1]
    Lk, Lv = packed_lk * 2, packed_lv * 2
    assert q.shape[-1] == Lk and o.shape[-1] == Lv

    block_packed_dk = max(16, triton.next_power_of_2(packed_lk))
    block_packed_dv = max(16, triton.next_power_of_2(packed_lv))
    effective_convert = (
        convert
        if (q.is_cuda and min(block_packed_dk, block_packed_dv) >= 16)
        else "cast"
    )

    fused_rht_k = (
        use_rht
        and can_fuse_rht(Lk)
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
    )
    fused_rht_v = use_rht and can_fuse_rht(Lv) and o.is_cuda

    rht_b_k = rht_block_size(Lk)
    inv_sqrt_rht_k = 1.0 / math.sqrt(rht_b_k)
    if use_rht:
        sm_scale_rot = sm_scale * inv_sqrt_rht_k
        if fused_rht_k:
            q_in = q
            m_even_k, m_odd_k = get_rht_split_matrices(Lk, q.dtype, q.device)
        else:
            # Normalize by 1/sqrt(b) so rotated Q matches original Q's dynamic range.
            q_in = (apply_rht(q.float()) * inv_sqrt_rht_k).to(q.dtype)
            m_even_k = m_odd_k = q_in
    else:
        sm_scale_rot = sm_scale
        q_in = q
        m_even_k = m_odd_k = q_in

    default_block_n, default_stages = _default_tile_int4(max(Lk, Lv))
    block_n = default_block_n if block_n is None else block_n
    num_stages = default_stages if num_stages is None else num_stages
    grid = (batch, _grouped_head_tiles(head_num, kv_group_num), max_kv_splits)
    _fwd_grouped_kernel_stage1_int4[grid](
        q_in,
        k_buffer,
        v_buffer,
        k_scale_buffer,
        v_scale_buffer,
        m_even_k,
        m_odd_k,
        sm_scale_rot,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        q_in.stride(0),
        q_in.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        k_scale_buffer.stride(0),
        v_scale_buffer.stride(0),
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        attn_lse.stride(0),
        attn_lse.stride(1),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_PACKED_DK=block_packed_dk,
        BLOCK_PACKED_DV=block_packed_dv,
        BLOCK_N=block_n,
        BLOCK_H=_GROUPED_BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        PACKED_LK=packed_lk,
        PACKED_LV=packed_lv,
        CONVERT=effective_convert,
        FUSED_RHT_K=fused_rht_k,
        INV_SQRT_RHT_K=inv_sqrt_rht_k,
        SPLIT_OUT=fused_rht_v,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if fused_rht_v:
        m_even_v, m_odd_v = get_rht_split_matrices(Lv, torch.float16, o.device)
        stage2_grid = (batch, triton.cdiv(head_num, _STAGE2_BLOCK_H))
        _fwd_kernel_stage2_int4_rht[stage2_grid](
            attn_logits,
            attn_lse,
            o,
            m_even_v,
            m_odd_v,
            kv_indptr,
            num_kv_splits,
            sinks,
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_logits.stride(2),
            attn_lse.stride(0),
            attn_lse.stride(1),
            o.stride(0),
            o.stride(1),
            q_head_num=head_num,
            MAX_KV_SPLITS=max_kv_splits,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            BLOCK_H=_STAGE2_BLOCK_H,
            BLOCK_DV=Lv,
            PACKED_LV=packed_lv,
            INV_RHT_NORM_V=1.0 / Lv,
            HAS_SINK=sinks is not None,
            num_warps=4,
            num_stages=1,
        )
        return

    # Fallback Stage 2: pass v_buffer=o so _decode_softmax_reducev_fwd sees the
    # unpacked head dim Lv rather than the packed dim Lv // 2.
    _decode_softmax_reducev_fwd(
        logits=attn_logits,
        lse=attn_lse,
        q=q_in,
        o=o,
        v_scale=1.0,
        v_buffer=o,
        kv_indptr=kv_indptr,
        num_kv_splits=num_kv_splits,
        max_kv_splits=max_kv_splits,
        sinks=sinks,
    )
    if use_rht:
        o.copy_(apply_rht(o.float(), inverse=True, out_dtype=o.dtype))
