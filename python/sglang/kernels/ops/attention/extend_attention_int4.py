"""Fused extend attention over the int4_per_token_head KV cache.

Provides two Triton kernels that read packed uint8 K/V
``[slots, kv_heads, head_dim // 2]`` and fp32 per-(slot, head) scales
``[slots, kv_heads]`` (with 4-bit zero-points embedded in the low 4 mantissa
bits) straight from the pool, without materializing any BF16 prefix copy:

- ``extend_attention_fwd_int4``: 2-stage extend matching SGLang's default
  ``chunk_bf16`` semantics (INT4 fused prefix + unquantized BF16 current
  chunk). For power-of-two head dims in ``(32, 64, 128)``, Stage 1 fuses Q's
  forward RHT and converts its prefix accumulator back via inverse RHT once
  before Stage 2, so Stage 2 runs unmodified BF16 attention with zero RHT on
  ``k_extend`` / ``v_extend`` and zero RHT overhead on fresh prefill.
- ``extend_attention_fwd_unified_int4``: 1-stage unified extend matching
  SGLang's deterministic mode and vLLM's ``chunk_q`` semantics (both prefix
  and current chunk are read back from the quantized INT4 cache with fixed
  ``BLOCK_N`` KV tiles starting at position 0 for bit-exact split invariance).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.decode_attention import tanh
from sglang.kernels.ops.attention.decode_attention_int4 import (
    CONVERSIONS,
    _unpack_hi_u4_to_f16,
    _unpack_lo_u4_to_f16,
)
from sglang.kernels.ops.kvcache.int4_per_token_head import (
    apply_rht,
    can_fuse_rht,
    get_rht_split_matrices,
    rht_block_size,
)


def _default_extend_tile_regular_int4(head_dim: int) -> tuple[int, int, int, int]:
    """Return ``(block_m, block_n, num_warps, num_stages)`` for 2-stage extend."""
    if head_dim <= 64:
        return 64, 64, 4, 1
    if head_dim <= 128:
        return 64, 128, 4, 1
    if head_dim <= 256:
        return 64, 64, 4, 1
    return 32, 32, 4, 1


def _default_extend_tile_unified_int4(head_dim: int) -> tuple[int, int, int, int]:
    """Return ``(block_m, block_n, num_warps, num_stages)`` for 1-stage unified extend."""
    if head_dim <= 64:
        return 64, 64, 4, 2
    if head_dim <= 128:
        return 64, 64, 4, 2
    if head_dim <= 256:
        return 64, 64, 4, 1
    return 32, 32, 4, 1


@triton.jit
def _fwd_extend_kernel_int4(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    K_Scale,
    V_Scale,
    M_Even_K,
    M_Odd_K,
    M_Even_V,
    M_Odd_V,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask_ptr,
    mask_indptr,
    sink_ptr,
    window_kv_offset_ptr,
    sm_scale,
    sm_scale_rot,
    kv_group_num,
    stride_qbs: tl.int64,
    stride_qh: tl.int64,
    stride_kbs: tl.int64,
    stride_kh: tl.int64,
    stride_vbs: tl.int64,
    stride_vh: tl.int64,
    stride_obs: tl.int64,
    stride_oh: tl.int64,
    stride_buf_kbs: tl.int64,
    stride_buf_kh: tl.int64,
    stride_buf_vbs: tl.int64,
    stride_buf_vh: tl.int64,
    stride_ks_bs: tl.int64,
    stride_vs_bs: tl.int64,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    PACKED_LK: tl.constexpr,
    PACKED_LV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_PACKED_DK: tl.constexpr,
    BLOCK_PACKED_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N_PREFIX: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,
    HAS_SINK: tl.constexpr,
    CONVERT: tl.constexpr,
    INV_SQRT_RHT_K: tl.constexpr,
    INV_RHT_NORM_V: tl.constexpr,
):
    """2-stage INT4 extend with in-kernel Stage-1 Q RHT and inverse-RHT handoff."""
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = (
        tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    ).to(tl.int32)
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = (tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx).to(
        tl.int32
    )
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    if cur_block_m * BLOCK_M >= cur_seq_len_extend:
        return

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    window_kv_offset = 0
    if USE_CUSTOM_MASK and SLIDING_WINDOW_SIZE > 0:
        window_kv_offset = tl.load(window_kv_offset_ptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_pk = tl.arange(0, BLOCK_PACKED_DK)
    offs_pv = tl.arange(0, BLOCK_PACKED_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend
    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    offs_q = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]).to(
            tl.int64
        )
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # Stage 1: attend to the cached INT4 prefix in rotated space, then apply
    # inverse RHT once at the end of Stage 1 so Stage 2 runs in original space.
    if cur_seq_len_prefix > 0:
        q_raw = tl.load(Q_Extend + offs_q, mask=mask_m[:, None], other=0.0)
        m_even_k = tl.load(
            M_Even_K + offs_d[:, None] * BLOCK_PACKED_DK + offs_pk[None, :]
        )
        q_even = (tl.dot(q_raw, m_even_k, out_dtype=tl.float32) * INV_SQRT_RHT_K).to(
            tl.float16
        )
        m_odd_k = tl.load(
            M_Odd_K + offs_d[:, None] * BLOCK_PACKED_DK + offs_pk[None, :]
        )
        q_odd = (tl.dot(q_raw, m_odd_k, out_dtype=tl.float32) * INV_SQRT_RHT_K).to(
            tl.float16
        )
        q_sum = tl.sum(q_even.to(tl.float32), axis=1) + tl.sum(
            q_odd.to(tl.float32), axis=1
        )

        acc_even = tl.zeros([BLOCK_M, BLOCK_PACKED_DV], dtype=tl.float32)
        acc_odd = tl.zeros([BLOCK_M, BLOCK_PACKED_DV], dtype=tl.float32)
        offs_n_prefix = tl.arange(0, BLOCK_N_PREFIX)
        base_k = cur_kv_head * stride_buf_kh + offs_pk[None, :]
        base_v = cur_kv_head * stride_buf_vh + offs_pv[None, :]

        for start_n in range(0, cur_seq_len_prefix, BLOCK_N_PREFIX):
            start_n = tl.multiple_of(start_n, BLOCK_N_PREFIX)
            mask_n = (start_n + offs_n_prefix) < cur_seq_len_prefix
            HAS_2D_PREFIX_MASK: tl.constexpr = (
                USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK
            ) or SLIDING_WINDOW_SIZE > 0

            SKIP_TILE = False
            if HAS_2D_PREFIX_MASK:
                final_mask = mask_m[:, None] & mask_n[None, :]
                if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:
                    custom_mask = tl.load(
                        mask_ptr
                        + cur_seq_mask_start_idx
                        + (cur_block_m * BLOCK_M + offs_m[:, None])
                        * (cur_seq_len + window_kv_offset)
                        + window_kv_offset
                        + start_n
                        + offs_n_prefix[None, :],
                        mask=mask_m[:, None] & mask_n[None, :],
                        other=0,
                    )
                    final_mask &= custom_mask != 0
                if SLIDING_WINDOW_SIZE > 0:
                    window_mask = (
                        cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m[:, None]
                    ) <= (start_n + offs_n_prefix[None, :] + SLIDING_WINDOW_SIZE)
                    final_mask &= window_mask
                SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

            if not SKIP_TILE:
                offs_kv_loc = tl.load(
                    kv_indices + cur_seq_kv_start_idx + start_n + offs_n_prefix,
                    mask=mask_n,
                    other=0,
                ).to(tl.int64)

                k_packed = tl.load(
                    K_Buffer + offs_kv_loc[:, None] * stride_buf_kbs + base_k,
                    mask=mask_n[:, None],
                    other=0,
                )
                ks_raw = tl.load(
                    K_Scale + offs_kv_loc * stride_ks_bs + cur_kv_head,
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
                if HAS_2D_PREFIX_MASK:
                    qk = tl.where(final_mask, qk, float("-inf"))
                else:
                    qk = tl.where(mask_n[None, :], qk, float("-inf"))

                row_max = tl.max(qk, 1)
                row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
                n_e_max = tl.maximum(row_max_fixed, e_max)

                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(qk - n_e_max[:, None])
                deno = deno * re_scale + tl.sum(p, 1)

                v_packed = tl.load(
                    V_Buffer + offs_kv_loc[:, None] * stride_buf_vbs + base_v,
                    mask=mask_n[:, None],
                    other=0,
                )
                vs_raw = tl.load(
                    V_Scale + offs_kv_loc * stride_vs_bs + cur_kv_head,
                    mask=mask_n,
                    other=0.0,
                )
                vs_bits = vs_raw.to(tl.int32, bitcast=True)
                v_zp = (vs_bits & 0xF).to(tl.float32)
                v_scale = (vs_bits & -16).to(tl.float32, bitcast=True)

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
                e_max = n_e_max

        # Rotate Stage 1's unnormalized prefix accumulator back to the original
        # basis once before entering Stage 2.
        row_max_a = tl.maximum(
            tl.maximum(
                tl.max(tl.abs(acc_even), axis=1), tl.max(tl.abs(acc_odd), axis=1)
            ),
            1e-30,
        )
        ae_f16 = (acc_even / row_max_a[:, None]).to(tl.float16)
        ao_f16 = (acc_odd / row_max_a[:, None]).to(tl.float16)
        m_even_v = tl.load(
            M_Even_V + offs_dv[:, None] * BLOCK_PACKED_DV + offs_pv[None, :]
        )
        acc = tl.dot(ae_f16, tl.trans(m_even_v), out_dtype=tl.float32)
        m_odd_v = tl.load(
            M_Odd_V + offs_dv[:, None] * BLOCK_PACKED_DV + offs_pv[None, :]
        )
        acc += tl.dot(ao_f16, tl.trans(m_odd_v), out_dtype=tl.float32)
        acc = acc * (row_max_a[:, None] * INV_RHT_NORM_V)

    # Stage 2: attend to the current chunk's unrotated BF16/FP16 K_Extend / V_Extend.
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )
    offs_n = tl.arange(0, BLOCK_N)
    cur_block_m_end = (
        cur_seq_len_extend
        if not IS_CAUSAL
        else tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    )
    extend_start = 0
    if SLIDING_WINDOW_SIZE > 0:
        extend_start = (
            tl.maximum(cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE, 0) // BLOCK_N
        ) * BLOCK_N

    for start_n in range(extend_start, cur_block_m_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end
        final_mask = mask_m[:, None] & mask_n[None, :]

        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None])
                * (cur_seq_len + window_kv_offset)
                + window_kv_offset
                + cur_seq_len_prefix
                + start_n
                + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0,
            )
            final_mask &= custom_mask != 0
        elif IS_CAUSAL:
            final_mask &= (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
                start_n + offs_n[None, :]
            )

        if SLIDING_WINDOW_SIZE > 0:
            if not IS_CAUSAL:
                window_mask = (
                    (cur_block_m * BLOCK_M + offs_m[:, None])
                    <= (start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE)
                ) & (
                    (start_n + offs_n[None, :])
                    <= (cur_block_m * BLOCK_M + offs_m[:, None] + SLIDING_WINDOW_SIZE)
                )
            else:
                window_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) <= (
                    start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE
                )
            final_mask &= window_mask

        SKIP_TILE = False
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            offs_k = (
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]).to(tl.int64)
                * stride_kbs
                + cur_kv_head * stride_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Extend + offs_k, mask=mask_n[None, :] & mask_d[:, None], other=0.0
            )
            qk = tl.dot(q, k, out_dtype=tl.float32) * sm_scale
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)
            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            offs_v = (
                (cur_seq_extend_start_idx + start_n + offs_n[:, None]).to(tl.int64)
                * stride_vbs
                + cur_kv_head * stride_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
            )
            dot = tl.dot(p.to(v.dtype), v)
            acc = acc * re_scale[:, None] + dot
            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        deno += tl.exp(cur_sink - e_max)

    no_kv = deno == 0.0
    deno_safe = tl.where(no_kv, 1.0, deno)
    offs_o = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]).to(
            tl.int64
        )
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(
        O_Extend + offs_o,
        acc / deno_safe[:, None],
        mask=mask_m[:, None] & mask_dv[None, :],
    )


@triton.jit
def _fwd_extend_kernel_int4_fallback(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    K_Scale,
    V_Scale,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask_ptr,
    mask_indptr,
    sink_ptr,
    window_kv_offset_ptr,
    sm_scale,
    sm_scale_rot,
    kv_group_num,
    stride_qbs: tl.int64,
    stride_qh: tl.int64,
    stride_kbs: tl.int64,
    stride_kh: tl.int64,
    stride_vbs: tl.int64,
    stride_vh: tl.int64,
    stride_obs: tl.int64,
    stride_oh: tl.int64,
    stride_buf_kbs: tl.int64,
    stride_buf_kh: tl.int64,
    stride_buf_vbs: tl.int64,
    stride_buf_vh: tl.int64,
    stride_ks_bs: tl.int64,
    stride_vs_bs: tl.int64,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    PACKED_LK: tl.constexpr,
    PACKED_LV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_PACKED_DK: tl.constexpr,
    BLOCK_PACKED_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N_PREFIX: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,
    HAS_SINK: tl.constexpr,
    CONVERT: tl.constexpr,
):
    """General 2-stage INT4 extend for non-power-of-two dims or use_rht=False."""
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = (
        tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    ).to(tl.int32)
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = (tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx).to(
        tl.int32
    )
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    if cur_block_m * BLOCK_M >= cur_seq_len_extend:
        return

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    window_kv_offset = 0
    if USE_CUSTOM_MASK and SLIDING_WINDOW_SIZE > 0:
        window_kv_offset = tl.load(window_kv_offset_ptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_pk = tl.arange(0, BLOCK_PACKED_DK)
    offs_pv = tl.arange(0, BLOCK_PACKED_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend
    mask_d = offs_d < Lq
    mask_pk = offs_pk < PACKED_LK
    mask_pv = offs_pv < PACKED_LV

    q_row_base = (
        cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]
    ).to(tl.int64) * stride_qbs + cur_head * stride_qh

    acc_even = tl.zeros([BLOCK_M, BLOCK_PACKED_DV], dtype=tl.float32)
    acc_odd = tl.zeros([BLOCK_M, BLOCK_PACKED_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    if cur_seq_len_prefix > 0:
        q_even = tl.load(
            Q_Extend + q_row_base + (offs_pk * 2)[None, :],
            mask=mask_m[:, None] & mask_pk[None, :],
            other=0.0,
        ).to(tl.float16)
        q_odd = tl.load(
            Q_Extend + q_row_base + (offs_pk * 2 + 1)[None, :],
            mask=mask_m[:, None] & mask_pk[None, :],
            other=0.0,
        ).to(tl.float16)
        q_sum = tl.sum(q_even.to(tl.float32), axis=1) + tl.sum(
            q_odd.to(tl.float32), axis=1
        )
        offs_n_prefix = tl.arange(0, BLOCK_N_PREFIX)
        base_k = cur_kv_head * stride_buf_kh + offs_pk[None, :]
        base_v = cur_kv_head * stride_buf_vh + offs_pv[None, :]

        for start_n in range(0, cur_seq_len_prefix, BLOCK_N_PREFIX):
            start_n = tl.multiple_of(start_n, BLOCK_N_PREFIX)
            mask_n = (start_n + offs_n_prefix) < cur_seq_len_prefix
            HAS_2D_PREFIX_MASK: tl.constexpr = (
                USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK
            ) or SLIDING_WINDOW_SIZE > 0

            SKIP_TILE = False
            if HAS_2D_PREFIX_MASK:
                final_mask = mask_m[:, None] & mask_n[None, :]
                if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:
                    custom_mask = tl.load(
                        mask_ptr
                        + cur_seq_mask_start_idx
                        + (cur_block_m * BLOCK_M + offs_m[:, None])
                        * (cur_seq_len + window_kv_offset)
                        + window_kv_offset
                        + start_n
                        + offs_n_prefix[None, :],
                        mask=mask_m[:, None] & mask_n[None, :],
                        other=0,
                    )
                    final_mask &= custom_mask != 0
                if SLIDING_WINDOW_SIZE > 0:
                    window_mask = (
                        cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m[:, None]
                    ) <= (start_n + offs_n_prefix[None, :] + SLIDING_WINDOW_SIZE)
                    final_mask &= window_mask
                SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

            if not SKIP_TILE:
                offs_kv_loc = tl.load(
                    kv_indices + cur_seq_kv_start_idx + start_n + offs_n_prefix,
                    mask=mask_n,
                    other=0,
                ).to(tl.int64)

                k_packed = tl.load(
                    K_Buffer + offs_kv_loc[:, None] * stride_buf_kbs + base_k,
                    mask=mask_n[:, None] & mask_pk[None, :],
                    other=0,
                )
                ks_raw = tl.load(
                    K_Scale + offs_kv_loc * stride_ks_bs + cur_kv_head,
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
                if HAS_2D_PREFIX_MASK:
                    qk = tl.where(final_mask, qk, float("-inf"))
                else:
                    qk = tl.where(mask_n[None, :], qk, float("-inf"))

                row_max = tl.max(qk, 1)
                row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
                n_e_max = tl.maximum(row_max_fixed, e_max)

                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(qk - n_e_max[:, None])
                deno = deno * re_scale + tl.sum(p, 1)

                v_packed = tl.load(
                    V_Buffer + offs_kv_loc[:, None] * stride_buf_vbs + base_v,
                    mask=mask_n[:, None] & mask_pv[None, :],
                    other=0,
                )
                vs_raw = tl.load(
                    V_Scale + offs_kv_loc * stride_vs_bs + cur_kv_head,
                    mask=mask_n,
                    other=0.0,
                )
                vs_bits = vs_raw.to(tl.int32, bitcast=True)
                v_zp = (vs_bits & 0xF).to(tl.float32)
                v_scale = (vs_bits & -16).to(tl.float32, bitcast=True)

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
                e_max = n_e_max

    q = tl.load(
        Q_Extend + q_row_base + offs_d[None, :],
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )
    offs_n = tl.arange(0, BLOCK_N)
    cur_block_m_end = (
        cur_seq_len_extend
        if not IS_CAUSAL
        else tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    )
    extend_start = 0
    if SLIDING_WINDOW_SIZE > 0:
        extend_start = (
            tl.maximum(cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE, 0) // BLOCK_N
        ) * BLOCK_N

    for start_n in range(extend_start, cur_block_m_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end
        final_mask = mask_m[:, None] & mask_n[None, :]

        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None])
                * (cur_seq_len + window_kv_offset)
                + window_kv_offset
                + cur_seq_len_prefix
                + start_n
                + offs_n[None, :],
                mask=mask_m[:, None] & mask_n[None, :],
                other=0,
            )
            final_mask &= custom_mask != 0
        elif IS_CAUSAL:
            final_mask &= (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
                start_n + offs_n[None, :]
            )

        if SLIDING_WINDOW_SIZE > 0:
            if not IS_CAUSAL:
                window_mask = (
                    (cur_block_m * BLOCK_M + offs_m[:, None])
                    <= (start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE)
                ) & (
                    (start_n + offs_n[None, :])
                    <= (cur_block_m * BLOCK_M + offs_m[:, None] + SLIDING_WINDOW_SIZE)
                )
            else:
                window_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) <= (
                    start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE
                )
            final_mask &= window_mask

        SKIP_TILE = False
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            offs_k = (
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]).to(tl.int64)
                * stride_kbs
                + cur_kv_head * stride_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Extend + offs_k, mask=mask_n[None, :] & mask_d[:, None], other=0.0
            )
            qk = tl.dot(q, k, out_dtype=tl.float32) * sm_scale
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)
            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            v_row_base = (cur_seq_extend_start_idx + start_n + offs_n[:, None]).to(
                tl.int64
            ) * stride_vbs + cur_kv_head * stride_vh
            v_even = tl.load(
                V_Extend + v_row_base + (offs_pv * 2)[None, :],
                mask=mask_n[:, None] & mask_pv[None, :],
                other=0.0,
            )
            v_odd = tl.load(
                V_Extend + v_row_base + (offs_pv * 2 + 1)[None, :],
                mask=mask_n[:, None] & mask_pv[None, :],
                other=0.0,
            )
            p_v = p.to(v_even.dtype)
            acc_even = acc_even * re_scale[:, None] + tl.dot(p_v, v_even)
            acc_odd = acc_odd * re_scale[:, None] + tl.dot(p_v, v_odd)
            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        deno += tl.exp(cur_sink - e_max)

    no_kv = deno == 0.0
    deno_safe = tl.where(no_kv, 1.0, deno)
    o_row_base = (
        cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]
    ).to(tl.int64) * stride_obs + cur_head * stride_oh
    tl.store(
        O_Extend + o_row_base + (offs_pv * 2)[None, :],
        (acc_even / deno_safe[:, None]).to(O_Extend.dtype.element_ty),
        mask=mask_m[:, None] & mask_pv[None, :],
    )
    tl.store(
        O_Extend + o_row_base + (offs_pv * 2 + 1)[None, :],
        (acc_odd / deno_safe[:, None]).to(O_Extend.dtype.element_ty),
        mask=mask_m[:, None] & mask_pv[None, :],
    )


def extend_attention_fwd_int4(
    *,
    q_extend: torch.Tensor,
    k_extend: torch.Tensor,
    v_extend: torch.Tensor,
    o_extend: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    k_scale_buffer: torch.Tensor,
    v_scale_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    custom_mask: Optional[torch.Tensor],
    is_causal: bool,
    mask_indptr: Optional[torch.Tensor],
    max_len_extend: int,
    sm_scale: Optional[float] = None,
    logit_cap: float = 0.0,
    skip_prefix_custom_mask: bool = True,
    sliding_window_size: int = -1,
    sinks: Optional[torch.Tensor] = None,
    window_kv_offsets: Optional[torch.Tensor] = None,
    use_rht: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_n_prefix: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    convert: str = "magic",
) -> None:
    """2-stage extend attention: fused INT4 prefix + unquantized BF16/FP16 current chunk."""
    assert k_buffer.dtype == torch.uint8 and v_buffer.dtype == torch.uint8
    assert k_scale_buffer.dtype == torch.float32
    assert v_scale_buffer.dtype == torch.float32
    assert k_buffer.stride(2) == 1 and v_buffer.stride(2) == 1
    assert k_scale_buffer.stride(1) == 1 and v_scale_buffer.stride(1) == 1
    assert k_extend.stride(2) == 1 and v_extend.stride(2) == 1
    assert k_extend.shape[1] == k_buffer.shape[1]
    assert convert in CONVERSIONS, convert

    batch_size = qo_indptr.shape[0] - 1
    if batch_size == 0 or max_len_extend == 0:
        return

    packed_lk, packed_lv = k_buffer.shape[-1], v_buffer.shape[-1]
    Lq, Lv = packed_lk * 2, packed_lv * 2
    assert q_extend.shape[-1] == Lq and v_extend.shape[-1] == Lv
    sm_scale = sm_scale if sm_scale is not None else 1.0 / (Lq**0.5)
    head_num = q_extend.shape[1]
    kv_group_num = head_num // k_buffer.shape[1]

    block_packed_dk = max(16, triton.next_power_of_2(packed_lk))
    block_packed_dv = max(16, triton.next_power_of_2(packed_lv))
    effective_convert = (
        convert
        if (q_extend.is_cuda and min(block_packed_dk, block_packed_dv) >= 16)
        else "cast"
    )

    def_m, def_n, def_warps, def_stages = _default_extend_tile_regular_int4(max(Lq, Lv))
    BLOCK_M = def_m if block_m is None else block_m
    BLOCK_N = def_n if block_n is None else block_n
    BLOCK_N_PREFIX = BLOCK_N if block_n_prefix is None else block_n_prefix
    warps = def_warps if num_warps is None else num_warps
    stages = def_stages if num_stages is None else num_stages

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    fused_rht = (
        use_rht
        and can_fuse_rht(Lq)
        and can_fuse_rht(Lv)
        and q_extend.is_cuda
        and q_extend.dtype in (torch.float16, torch.bfloat16)
    )

    rht_b_k = rht_block_size(Lq)
    inv_sqrt_rht_k = 1.0 / math.sqrt(rht_b_k)

    if fused_rht:
        m_even_k, m_odd_k = get_rht_split_matrices(Lq, q_extend.dtype, q_extend.device)
        m_even_v, m_odd_v = get_rht_split_matrices(Lv, torch.float16, o_extend.device)
        _fwd_extend_kernel_int4[grid](
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            k_scale_buffer,
            v_scale_buffer,
            m_even_k,
            m_odd_k,
            m_even_v,
            m_odd_v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            mask_indptr,
            sinks,
            window_kv_offsets,
            sm_scale,
            sm_scale * inv_sqrt_rht_k,
            kv_group_num,
            q_extend.stride(0),
            q_extend.stride(1),
            k_extend.stride(0),
            k_extend.stride(1),
            v_extend.stride(0),
            v_extend.stride(1),
            o_extend.stride(0),
            o_extend.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            k_scale_buffer.stride(0),
            v_scale_buffer.stride(0),
            SLIDING_WINDOW_SIZE=sliding_window_size,
            logit_cap=logit_cap,
            Lq=Lq,
            Lv=Lv,
            PACKED_LK=packed_lk,
            PACKED_LV=packed_lv,
            BLOCK_DMODEL=Lq,
            BLOCK_DV=Lv,
            BLOCK_PACKED_DK=packed_lk,
            BLOCK_PACKED_DV=packed_lv,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_N_PREFIX=BLOCK_N_PREFIX,
            USE_CUSTOM_MASK=custom_mask is not None,
            IS_CAUSAL=is_causal,
            SKIP_PREFIX_CUSTOM_MASK=skip_prefix_custom_mask,
            HAS_SINK=sinks is not None,
            CONVERT=effective_convert,
            INV_SQRT_RHT_K=inv_sqrt_rht_k,
            INV_RHT_NORM_V=1.0 / Lv,
            num_warps=warps,
            num_stages=stages,
        )
        return

    if use_rht:
        # Keep rotated intermediates out of 7-bit bf16: the unnormalized RHT
        # amplifies V/O magnitudes by sqrt(b), so truncating rotated Q/K/V/O to
        # bf16 before inverse RHT adds ~2e-2 noise on non-power-of-two dims.
        rot_dtype = (
            torch.float16 if q_extend.dtype == torch.bfloat16 else q_extend.dtype
        )
        q_in = (apply_rht(q_extend.float()) * inv_sqrt_rht_k).to(rot_dtype)
        k_in = (apply_rht(k_extend.float()) * inv_sqrt_rht_k).to(rot_dtype)
        v_in = apply_rht(v_extend.float()).to(rot_dtype)
        o_buf = torch.empty_like(o_extend, dtype=torch.float32)
        sm_scale_rot = sm_scale * inv_sqrt_rht_k
    else:
        q_in, k_in, v_in, o_buf = q_extend, k_extend, v_extend, o_extend
        sm_scale_rot = sm_scale

    _fwd_extend_kernel_int4_fallback[grid](
        q_in,
        k_in,
        v_in,
        o_buf,
        k_buffer,
        v_buffer,
        k_scale_buffer,
        v_scale_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        sinks,
        window_kv_offsets,
        sm_scale,
        sm_scale_rot,
        kv_group_num,
        q_in.stride(0),
        q_in.stride(1),
        k_in.stride(0),
        k_in.stride(1),
        v_in.stride(0),
        v_in.stride(1),
        o_buf.stride(0),
        o_buf.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        k_scale_buffer.stride(0),
        v_scale_buffer.stride(0),
        SLIDING_WINDOW_SIZE=sliding_window_size,
        logit_cap=logit_cap,
        Lq=Lq,
        Lv=Lv,
        PACKED_LK=packed_lk,
        PACKED_LV=packed_lv,
        BLOCK_DMODEL=triton.next_power_of_2(Lq),
        BLOCK_PACKED_DK=block_packed_dk,
        BLOCK_PACKED_DV=block_packed_dv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_N_PREFIX=BLOCK_N_PREFIX,
        USE_CUSTOM_MASK=custom_mask is not None,
        IS_CAUSAL=is_causal,
        SKIP_PREFIX_CUSTOM_MASK=skip_prefix_custom_mask,
        HAS_SINK=sinks is not None,
        CONVERT=effective_convert,
        num_warps=warps,
        num_stages=stages,
    )
    if use_rht:
        o_extend.copy_(apply_rht(o_buf, inverse=True, out_dtype=o_extend.dtype))


@triton.jit
def _fwd_extend_unified_kernel_int4(
    Q,
    O,
    K_Buffer,
    V_Buffer,
    K_Scale,
    V_Scale,
    M_Even_K,
    M_Odd_K,
    M_Even_V,
    M_Odd_V,
    qo_indptr,
    kv_indptr,
    kv_indices,
    prefix_lens,
    mask_ptr,
    mask_indptr,
    sink_ptr,
    window_start_pos,
    sm_scale_rot,
    kv_group_num,
    stride_qbs: tl.int64,
    stride_qh: tl.int64,
    stride_obs: tl.int64,
    stride_oh: tl.int64,
    stride_buf_kbs: tl.int64,
    stride_buf_kh: tl.int64,
    stride_buf_vbs: tl.int64,
    stride_buf_vh: tl.int64,
    stride_ks_bs: tl.int64,
    stride_vs_bs: tl.int64,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    PACKED_LK: tl.constexpr,
    PACKED_LV: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_PACKED_DK: tl.constexpr,
    BLOCK_PACKED_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    HAS_SINK: tl.constexpr,
    CONVERT: tl.constexpr,
    FUSED_RHT_K: tl.constexpr,
    FUSED_RHT_V: tl.constexpr,
    INV_SQRT_RHT_K: tl.constexpr,
    INV_RHT_NORM_V: tl.constexpr,
):
    """Unified 1-stage INT4 extend kernel: both prefix and chunk come from cache.

    Fixed ``BLOCK_N`` KV tiling starting at position 0 and single-loop causal
    execution with per-row masked ``v_scale_max`` on diagonal/tail tiles ensure
    bit-exact invariance across arbitrary prefix-cache / chunked-prefill splits.
    """
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_q_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_q_len = (tl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx).to(
        tl.int32
    )
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_kv_len = (tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx).to(
        tl.int32
    )
    cur_seq_prefix_len = tl.load(prefix_lens + cur_seq).to(tl.int32)

    if cur_block_m * BLOCK_M >= cur_seq_q_len:
        return

    cur_window_start = 0
    if SLIDING_WINDOW_SIZE > 0:
        cur_window_start = tl.load(window_start_pos + cur_seq)

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    offs_pk = tl.arange(0, BLOCK_PACKED_DK)
    offs_pv = tl.arange(0, BLOCK_PACKED_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_q_len
    mask_pk = offs_pk < PACKED_LK
    mask_pv = offs_pv < PACKED_LV

    q_row_base = (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]).to(
        tl.int64
    ) * stride_qbs + cur_head * stride_qh
    if FUSED_RHT_K:
        offs_d = tl.arange(0, BLOCK_DMODEL)
        q_raw = tl.load(
            Q + q_row_base + offs_d[None, :], mask=mask_m[:, None], other=0.0
        )
        m_even_k = tl.load(
            M_Even_K + offs_d[:, None] * BLOCK_PACKED_DK + offs_pk[None, :]
        )
        q_even = (tl.dot(q_raw, m_even_k, out_dtype=tl.float32) * INV_SQRT_RHT_K).to(
            tl.float16
        )
        m_odd_k = tl.load(
            M_Odd_K + offs_d[:, None] * BLOCK_PACKED_DK + offs_pk[None, :]
        )
        q_odd = (tl.dot(q_raw, m_odd_k, out_dtype=tl.float32) * INV_SQRT_RHT_K).to(
            tl.float16
        )
    else:
        q_even = tl.load(
            Q + q_row_base + (offs_pk * 2)[None, :],
            mask=mask_m[:, None] & mask_pk[None, :],
            other=0.0,
        ).to(tl.float16)
        q_odd = tl.load(
            Q + q_row_base + (offs_pk * 2 + 1)[None, :],
            mask=mask_m[:, None] & mask_pk[None, :],
            other=0.0,
        ).to(tl.float16)

    q_sum = tl.sum(q_even.to(tl.float32), axis=1) + tl.sum(q_odd.to(tl.float32), axis=1)

    offs_n = tl.arange(0, BLOCK_N)
    base_k = cur_kv_head * stride_buf_kh + offs_pk[None, :]
    base_v = cur_kv_head * stride_buf_vh + offs_pv[None, :]

    acc_even = tl.zeros([BLOCK_M, BLOCK_PACKED_DV], dtype=tl.float32)
    acc_odd = tl.zeros([BLOCK_M, BLOCK_PACKED_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    start_k = 0
    end_k = cur_seq_kv_len
    if IS_CAUSAL and not USE_CUSTOM_MASK:
        last_query = tl.minimum((cur_block_m + 1) * BLOCK_M, cur_seq_q_len)
        end_k = tl.minimum(cur_seq_kv_len, cur_seq_prefix_len + last_query)
        if SLIDING_WINDOW_SIZE > 0:
            first_query = cur_block_m * BLOCK_M
            start_k = tl.maximum(
                0, cur_seq_prefix_len + first_query - SLIDING_WINDOW_SIZE
            )
            start_k = (start_k // BLOCK_N) * BLOCK_N

    if IS_CAUSAL and not USE_CUSTOM_MASK and SLIDING_WINDOW_SIZE <= 0:
        first_query_abs = cur_seq_prefix_len + cur_block_m * BLOCK_M
        full_limit = tl.minimum(first_query_abs + 1, cur_seq_kv_len)
        q_abs = first_query_abs + offs_m[:, None]

        for start_n in range(start_k, end_k, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            mask_n = (start_n + offs_n) < cur_seq_kv_len
            is_full_tile = (start_n + BLOCK_N) <= full_limit

            offs_kv_loc = tl.load(
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
                mask=mask_n,
                other=0,
            ).to(tl.int64)

            k_packed = tl.load(
                K_Buffer + offs_kv_loc[:, None] * stride_buf_kbs + base_k,
                mask=mask_n[:, None] & mask_pk[None, :],
                other=0,
            )
            ks_raw = tl.load(
                K_Scale + offs_kv_loc * stride_ks_bs + cur_kv_head,
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
            if not is_full_tile:
                causal_mask = mask_n[None, :] & (q_abs >= (start_n + offs_n[None, :]))
                qk = tl.where(causal_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            v_packed = tl.load(
                V_Buffer + offs_kv_loc[:, None] * stride_buf_vbs + base_v,
                mask=mask_n[:, None] & mask_pv[None, :],
                other=0,
            )
            vs_raw = tl.load(
                V_Scale + offs_kv_loc * stride_vs_bs + cur_kv_head,
                mask=mask_n,
                other=0.0,
            )
            vs_bits = vs_raw.to(tl.int32, bitcast=True)
            v_zp = (vs_bits & 0xF).to(tl.float32)
            v_scale = (vs_bits & -16).to(tl.float32, bitcast=True)

            if is_full_tile:
                v_scale_max = tl.zeros([BLOCK_M], dtype=tl.float32) + tl.maximum(
                    tl.max(v_scale, 0), 1e-30
                )
            else:
                causal_mask_v = mask_n[None, :] & (q_abs >= (start_n + offs_n[None, :]))
                v_scale_masked = tl.where(causal_mask_v, v_scale[None, :], 0.0)
                v_scale_max = tl.maximum(tl.max(v_scale_masked, 1), 1e-30)

            p_v = ((p * v_scale[None, :]) / v_scale_max[:, None]).to(tl.float16)
            pv_zp_sum = tl.sum(p_v.to(tl.float32) * v_zp[None, :], axis=1)

            v_even = _unpack_lo_u4_to_f16(v_packed, CONVERT)
            v_odd = _unpack_hi_u4_to_f16(v_packed, CONVERT)
            acc_even = (
                acc_even * re_scale[:, None]
                + (tl.dot(p_v, v_even) - pv_zp_sum[:, None]) * v_scale_max[:, None]
            )
            acc_odd = (
                acc_odd * re_scale[:, None]
                + (tl.dot(p_v, v_odd) - pv_zp_sum[:, None]) * v_scale_max[:, None]
            )
            e_max = n_e_max
    else:
        for start_n in range(start_k, end_k, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            mask_n = (start_n + offs_n) < cur_seq_kv_len
            final_mask = mask_m[:, None] & mask_n[None, :]

            if USE_CUSTOM_MASK:
                custom_mask = tl.load(
                    mask_ptr
                    + cur_seq_mask_start_idx
                    + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_kv_len
                    + start_n
                    + offs_n[None, :],
                    mask=mask_m[:, None] & mask_n[None, :],
                    other=0,
                )
                final_mask &= custom_mask != 0
            elif IS_CAUSAL:
                q_abs = cur_seq_prefix_len + cur_block_m * BLOCK_M + offs_m[:, None]
                k_abs = start_n + offs_n[None, :]
                final_mask &= q_abs >= k_abs

            if SLIDING_WINDOW_SIZE > 0:
                q_abs_pos = (
                    cur_window_start
                    + cur_seq_prefix_len
                    + cur_block_m * BLOCK_M
                    + offs_m[:, None]
                )
                k_abs_pos = cur_window_start + start_n + offs_n[None, :]
                window_mask = q_abs_pos <= (k_abs_pos + SLIDING_WINDOW_SIZE)
                if not IS_CAUSAL:
                    window_mask &= k_abs_pos <= (q_abs_pos + SLIDING_WINDOW_SIZE)
                final_mask &= window_mask

            SKIP_TILE = False
            if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
                SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

            if not SKIP_TILE:
                offs_kv_loc = tl.load(
                    kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
                    mask=mask_n,
                    other=0,
                ).to(tl.int64)

                k_packed = tl.load(
                    K_Buffer + offs_kv_loc[:, None] * stride_buf_kbs + base_k,
                    mask=mask_n[:, None] & mask_pk[None, :],
                    other=0,
                )
                ks_raw = tl.load(
                    K_Scale + offs_kv_loc * stride_ks_bs + cur_kv_head,
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
                qk = tl.where(final_mask, qk, float("-inf"))

                row_max = tl.max(qk, 1)
                row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
                n_e_max = tl.maximum(row_max_fixed, e_max)

                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(qk - n_e_max[:, None])
                deno = deno * re_scale + tl.sum(p, 1)

                v_packed = tl.load(
                    V_Buffer + offs_kv_loc[:, None] * stride_buf_vbs + base_v,
                    mask=mask_n[:, None] & mask_pv[None, :],
                    other=0,
                )
                vs_raw = tl.load(
                    V_Scale + offs_kv_loc * stride_vs_bs + cur_kv_head,
                    mask=mask_n,
                    other=0.0,
                )
                vs_bits = vs_raw.to(tl.int32, bitcast=True)
                v_zp = (vs_bits & 0xF).to(tl.float32)
                v_scale = (vs_bits & -16).to(tl.float32, bitcast=True)

                v_scale_masked = tl.where(final_mask, v_scale[None, :], 0.0)
                v_scale_max = tl.maximum(tl.max(v_scale_masked, 1), 1e-30)
                p_v = ((p * v_scale[None, :]) / v_scale_max[:, None]).to(tl.float16)
                pv_zp_sum = tl.sum(p_v.to(tl.float32) * v_zp[None, :], axis=1)

                v_even = _unpack_lo_u4_to_f16(v_packed, CONVERT)
                v_odd = _unpack_hi_u4_to_f16(v_packed, CONVERT)
                acc_even = (
                    acc_even * re_scale[:, None]
                    + (tl.dot(p_v, v_even) - pv_zp_sum[:, None]) * v_scale_max[:, None]
                )
                acc_odd = (
                    acc_odd * re_scale[:, None]
                    + (tl.dot(p_v, v_odd) - pv_zp_sum[:, None]) * v_scale_max[:, None]
                )
                e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        deno += tl.exp(cur_sink - e_max)

    no_kv = deno == 0.0
    deno_safe = tl.where(no_kv, 1.0, deno)
    o_even = tl.where(mask_m[:, None], acc_even / deno_safe[:, None], 0.0)
    o_odd = tl.where(mask_m[:, None], acc_odd / deno_safe[:, None], 0.0)

    o_row_base = (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]).to(
        tl.int64
    ) * stride_obs + cur_head * stride_oh
    if FUSED_RHT_V:
        offs_dv = tl.arange(0, BLOCK_DV)
        row_max_o = tl.maximum(
            tl.maximum(tl.max(tl.abs(o_even), axis=1), tl.max(tl.abs(o_odd), axis=1)),
            1e-30,
        )
        oe_f16 = (o_even / row_max_o[:, None]).to(tl.float16)
        oo_f16 = (o_odd / row_max_o[:, None]).to(tl.float16)
        m_even_v = tl.load(
            M_Even_V + offs_dv[:, None] * BLOCK_PACKED_DV + offs_pv[None, :]
        )
        out = tl.dot(oe_f16, tl.trans(m_even_v), out_dtype=tl.float32)
        m_odd_v = tl.load(
            M_Odd_V + offs_dv[:, None] * BLOCK_PACKED_DV + offs_pv[None, :]
        )
        out += tl.dot(oo_f16, tl.trans(m_odd_v), out_dtype=tl.float32)
        out = out * (row_max_o[:, None] * INV_RHT_NORM_V)
        tl.store(
            O + o_row_base + offs_dv[None, :],
            out.to(O.dtype.element_ty),
            mask=mask_m[:, None],
        )
    else:
        tl.store(
            O + o_row_base + (offs_pv * 2)[None, :],
            o_even.to(O.dtype.element_ty),
            mask=mask_m[:, None] & mask_pv[None, :],
        )
        tl.store(
            O + o_row_base + (offs_pv * 2 + 1)[None, :],
            o_odd.to(O.dtype.element_ty),
            mask=mask_m[:, None] & mask_pv[None, :],
        )


def extend_attention_fwd_unified_int4(
    *,
    q: torch.Tensor,
    o: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    k_scale_buffer: torch.Tensor,
    v_scale_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_len_extend: int,
    custom_mask: Optional[torch.Tensor] = None,
    mask_indptr: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    logit_cap: float = 0.0,
    is_causal: bool = True,
    sliding_window_size: int = -1,
    sinks: Optional[torch.Tensor] = None,
    window_start_pos: Optional[torch.Tensor] = None,
    use_rht: bool = True,
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    convert: str = "magic",
) -> None:
    """1-stage unified extend attention reading both prefix and chunk from INT4 cache."""
    assert k_buffer.dtype == torch.uint8 and v_buffer.dtype == torch.uint8
    assert k_scale_buffer.dtype == torch.float32
    assert v_scale_buffer.dtype == torch.float32
    assert k_buffer.stride(2) == 1 and v_buffer.stride(2) == 1
    assert k_scale_buffer.stride(1) == 1 and v_scale_buffer.stride(1) == 1
    assert convert in CONVERSIONS, convert

    batch_size = qo_indptr.shape[0] - 1
    if batch_size == 0 or max_len_extend == 0:
        return

    packed_lk, packed_lv = k_buffer.shape[-1], v_buffer.shape[-1]
    Lq, Lv = packed_lk * 2, packed_lv * 2
    assert q.shape[-1] == Lq and o.shape[-1] == Lv
    sm_scale = sm_scale if sm_scale is not None else 1.0 / (Lq**0.5)
    head_num = q.shape[1]
    kv_group_num = head_num // k_buffer.shape[1]

    if sliding_window_size > 0 and window_start_pos is None:
        window_start_pos = torch.zeros(batch_size, dtype=torch.int32, device=q.device)

    block_packed_dk = max(16, triton.next_power_of_2(packed_lk))
    block_packed_dv = max(16, triton.next_power_of_2(packed_lv))
    effective_convert = (
        convert
        if (q.is_cuda and min(block_packed_dk, block_packed_dv) >= 16)
        else "cast"
    )

    fused_rht_k = (
        use_rht
        and can_fuse_rht(Lq)
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
    )
    fused_rht_v = use_rht and can_fuse_rht(Lv) and o.is_cuda

    rht_b_k = rht_block_size(Lq)
    inv_sqrt_rht_k = 1.0 / math.sqrt(rht_b_k)
    if use_rht:
        sm_scale_rot = sm_scale * inv_sqrt_rht_k
        if fused_rht_k:
            q_in = q
            m_even_k, m_odd_k = get_rht_split_matrices(Lq, q.dtype, q.device)
        else:
            rot_q_dtype = torch.float16 if q.dtype == torch.bfloat16 else q.dtype
            q_in = (apply_rht(q.float()) * inv_sqrt_rht_k).to(rot_q_dtype)
            m_even_k = m_odd_k = q_in
        if fused_rht_v:
            o_buf = o
            m_even_v, m_odd_v = get_rht_split_matrices(Lv, torch.float16, o.device)
        else:
            o_buf = torch.empty_like(o, dtype=torch.float32)
            m_even_v = m_odd_v = o_buf
    else:
        sm_scale_rot = sm_scale
        q_in = q
        o_buf = o
        m_even_k = m_odd_k = q_in
        m_even_v = m_odd_v = o_buf

    def_m, def_n, def_warps, def_stages = _default_extend_tile_unified_int4(max(Lq, Lv))
    BLOCK_M = def_m if block_m is None else block_m
    BLOCK_N = def_n if block_n is None else block_n
    warps = def_warps if num_warps is None else num_warps
    stages = def_stages if num_stages is None else num_stages

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    _fwd_extend_unified_kernel_int4[grid](
        q_in,
        o_buf,
        k_buffer,
        v_buffer,
        k_scale_buffer,
        v_scale_buffer,
        m_even_k,
        m_odd_k,
        m_even_v,
        m_odd_v,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        custom_mask,
        mask_indptr,
        sinks,
        window_start_pos,
        sm_scale_rot,
        kv_group_num,
        q_in.stride(0),
        q_in.stride(1),
        o_buf.stride(0),
        o_buf.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        k_scale_buffer.stride(0),
        v_scale_buffer.stride(0),
        SLIDING_WINDOW_SIZE=sliding_window_size,
        logit_cap=logit_cap,
        Lq=Lq,
        Lv=Lv,
        PACKED_LK=packed_lk,
        PACKED_LV=packed_lv,
        BLOCK_DMODEL=triton.next_power_of_2(Lq),
        BLOCK_DV=triton.next_power_of_2(Lv),
        BLOCK_PACKED_DK=block_packed_dk,
        BLOCK_PACKED_DV=block_packed_dv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=custom_mask is not None,
        HAS_SINK=sinks is not None,
        CONVERT=effective_convert,
        FUSED_RHT_K=fused_rht_k,
        FUSED_RHT_V=fused_rht_v,
        INV_SQRT_RHT_K=inv_sqrt_rht_k,
        INV_RHT_NORM_V=1.0 / Lv,
        num_warps=warps,
        num_stages=stages,
    )
    if use_rht and not fused_rht_v:
        o.copy_(apply_rht(o_buf, inverse=True, out_dtype=o.dtype))
