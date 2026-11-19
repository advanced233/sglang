"""Fused extend attention over the int8_per_token_head KV cache.

Provides two Triton kernels that read int8 K/V ``[slots, kv_heads, head_dim]``
and fp32 per-(slot, head) scales ``[slots, kv_heads]`` straight from the pool,
without materializing any BF16 copy of the cached prefix:

- ``extend_attention_fwd_int8``: 2-stage extend matching SGLang's default
  ``chunk_bf16`` semantics (int8 fused prefix + BF16 current chunk).
- ``extend_attention_fwd_unified_int8``: 1-stage unified extend matching
  SGLang's deterministic mode and vLLM's ``chunk_q`` semantics (both prefix
  and current chunk are read back from the quantized int8 cache).
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.decode_attention import tanh
from sglang.kernels.ops.attention.decode_attention_int8 import (
    CONVERSIONS,
    _int8_to_f16,
)


def _default_extend_tile_regular(head_dim: int) -> tuple[int, int, int, int]:
    """Return ``(block_m, block_n, num_warps, num_stages)`` for 2-stage extend.

    ``(64, 128, 4, 1)`` keeps the fp32 accumulator compact alongside Stage 2's
    BF16 loop while streaming 128 keys per prefix step (measured fastest on A100).
    """
    if head_dim <= 64:
        return 64, 64, 4, 1
    if head_dim <= 128:
        return 64, 128, 4, 1
    if head_dim <= 256:
        return 64, 64, 4, 1
    return 32, 32, 4, 1


def _default_extend_tile_unified(head_dim: int) -> tuple[int, int, int, int]:
    """Return ``(block_m, block_n, num_warps, num_stages)`` for 1-stage unified extend.

    ``(64, 64, 4, 2)`` fits 2 software-pipelined stages without spilling and
    outperforms BF16 extend on A100 at 0 MB temporary memory.
    """
    if head_dim <= 64:
        return 64, 64, 4, 2
    if head_dim <= 128:
        return 64, 64, 4, 2
    if head_dim <= 256:
        return 64, 64, 4, 1
    return 32, 32, 4, 1


@triton.jit
def _fwd_extend_kernel_int8(
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
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_ks_bs,
    stride_vs_bs,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_N_PREFIX: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,
    HAS_SINK: tl.constexpr,
    CONVERT: tl.constexpr,
):
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
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend
    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    offs_q = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # Stage 1: attend to the cached prefix in the int8 KV pool.
    # Load q_f16 only inside Stage 1 so q (bf16) and q_f16 (fp16) are never
    # simultaneously live in registers.
    if cur_seq_len_prefix > 0:
        q_f16 = tl.load(
            Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
        ).to(tl.float16)
        offs_n_prefix = tl.arange(0, BLOCK_N_PREFIX)
        base_k = cur_kv_head * stride_buf_kh + offs_d[None, :]
        base_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]

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
                SKIP_TILE = (
                    tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0
                )

            if not SKIP_TILE:
                offs_kv_loc = tl.load(
                    kv_indices + cur_seq_kv_start_idx + start_n + offs_n_prefix,
                    mask=mask_n,
                    other=0,
                ).to(tl.int64)

                # Load K in contiguous [BLOCK_N_PREFIX, BLOCK_DMODEL] layout for
                # vectorized 128-bit loads and contiguous pack=4 conversion.
                k_t = tl.load(
                    K_Buffer + offs_kv_loc[:, None] * stride_buf_kbs + base_k,
                    mask=mask_n[:, None] & mask_d[None, :],
                    other=0,
                )
                k_scale = tl.load(
                    K_Scale + offs_kv_loc * stride_ks_bs + cur_kv_head,
                    mask=mask_n,
                    other=0.0,
                )
                qk = tl.dot(q_f16, tl.trans(_int8_to_f16(k_t, CONVERT)))
                qk *= (k_scale * sm_scale)[None, :]
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

                v = tl.load(
                    V_Buffer + offs_kv_loc[:, None] * stride_buf_vbs + base_v,
                    mask=mask_n[:, None] & mask_dv[None, :],
                    other=0,
                )
                v_scale = tl.load(
                    V_Scale + offs_kv_loc * stride_vs_bs + cur_kv_head,
                    mask=mask_n,
                    other=0.0,
                )
                # Normalize by tile max so P * v_scale stays in fp16's normal
                # range, then restore the tile max after the dot.
                v_scale_max = tl.maximum(tl.max(v_scale, 0), 1e-30)
                p_v = (p * (v_scale / v_scale_max)[None, :]).to(tl.float16)
                pv = tl.dot(p_v, _int8_to_f16(v, CONVERT))
                acc = acc * re_scale[:, None] + pv * v_scale_max
                e_max = n_e_max

    # Stage 2: attend to the current chunk's unquantized K_Extend / V_Extend.
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
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
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
                (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
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
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(
        O_Extend + offs_o,
        acc / deno_safe[:, None],
        mask=mask_m[:, None] & mask_dv[None, :],
    )


def extend_attention_fwd_int8(
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
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    block_n_prefix: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    convert: str = "magic",
) -> None:
    """2-stage extend attention: fused int8 prefix + BF16/FP16 current chunk."""
    assert k_buffer.dtype == torch.int8 and v_buffer.dtype == torch.int8
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

    Lq, Lv = q_extend.shape[-1], v_extend.shape[-1]
    sm_scale = sm_scale if sm_scale is not None else 1.0 / (Lq**0.5)
    head_num = q_extend.shape[1]
    kv_group_num = head_num // k_buffer.shape[1]

    def_m, def_n, def_warps, def_stages = _default_extend_tile_regular(max(Lq, Lv))
    BLOCK_M = def_m if block_m is None else block_m
    BLOCK_N = def_n if block_n is None else block_n
    BLOCK_N_PREFIX = BLOCK_N if block_n_prefix is None else block_n_prefix
    warps = def_warps if num_warps is None else num_warps
    stages = def_stages if num_stages is None else num_stages

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    _fwd_extend_kernel_int8[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
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
        BLOCK_DMODEL=triton.next_power_of_2(Lq),
        BLOCK_DV=triton.next_power_of_2(Lv),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_N_PREFIX=BLOCK_N_PREFIX,
        USE_CUSTOM_MASK=custom_mask is not None,
        IS_CAUSAL=is_causal,
        SKIP_PREFIX_CUSTOM_MASK=skip_prefix_custom_mask,
        HAS_SINK=sinks is not None,
        CONVERT=convert,
        num_warps=warps,
        num_stages=stages,
    )


@triton.jit
def _fwd_extend_unified_kernel_int8(
    Q,
    O,
    K_Buffer,
    V_Buffer,
    K_Scale,
    V_Scale,
    qo_indptr,
    kv_indptr,
    kv_indices,
    prefix_lens,
    mask_ptr,
    mask_indptr,
    sink_ptr,
    window_start_pos,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_ks_bs,
    stride_vs_bs,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    HAS_SINK: tl.constexpr,
    CONVERT: tl.constexpr,
):
    """Unified 1-stage int8 extend kernel: both prefix and extend come from cache.

    Tile boundaries over KV start at position 0 in fixed ``BLOCK_N`` steps so
    the reduction order for any query position is independent of how the prompt
    was split between prefix cache and current extend chunk.

    For standard causal attention, strictly-valid prefix tiles run in a
    mask-free fast loop (Phase 1) and only the diagonal/tail tiles run with
    per-row causal masking and per-row masked ``v_scale_max`` (Phase 2). That
    both eliminates 2D mask register pressure on >90% of tiles and guarantees
    future keys beyond ``q_abs`` or ``cur_seq_kv_len`` never leak into
    ``v_scale_max``, preserving bit-exactness across arbitrary chunked-prefill
    splits.
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

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_q_len
    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    offs_q = (
        (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q + offs_q, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(
        tl.float16
    )

    offs_n = tl.arange(0, BLOCK_N)
    base_k = cur_kv_head * stride_buf_kh + offs_d[None, :]
    base_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
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

    # Phase 1: strictly full causal tiles (all BLOCK_N keys < first_query_abs
    # and < cur_seq_kv_len). Every query row sees all BLOCK_N keys, so 1D
    # v_scale_max matches per-row masked v_scale_max bit-for-bit.
    full_end_k = start_k
    if IS_CAUSAL and not USE_CUSTOM_MASK and SLIDING_WINDOW_SIZE <= 0:
        first_query_abs = cur_seq_prefix_len + cur_block_m * BLOCK_M
        full_limit = tl.minimum(first_query_abs, cur_seq_kv_len)
        full_end_k = tl.maximum((full_limit // BLOCK_N) * BLOCK_N, start_k)

        for start_n in range(start_k, full_end_k, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_kv_loc = tl.load(
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n
            ).to(tl.int64)

            k_t = tl.load(
                K_Buffer + offs_kv_loc[:, None] * stride_buf_kbs + base_k,
                mask=mask_d[None, :],
                other=0,
            )
            k_scale = tl.load(
                K_Scale + offs_kv_loc * stride_ks_bs + cur_kv_head,
            )
            qk = tl.dot(q, tl.trans(_int8_to_f16(k_t, CONVERT)))
            qk *= (k_scale * sm_scale)[None, :]
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            row_max = tl.max(qk, 1)
            n_e_max = tl.maximum(row_max, e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = tl.fma(deno, re_scale, tl.sum(p, 1))

            v = tl.load(
                V_Buffer + offs_kv_loc[:, None] * stride_buf_vbs + base_v,
                mask=mask_dv[None, :],
                other=0,
            )
            v_scale = tl.load(
                V_Scale + offs_kv_loc * stride_vs_bs + cur_kv_head,
            )
            # Use IEEE-754 div_rn and explicit fma in both Phase 1 and Phase 2
            # so a full tile produces bit-identical results regardless of which
            # phase executes it (Triton's `/` otherwise lowers 1D/scalar and
            # 2D/1D divisions with different reciprocal rounding).
            v_scale_max = tl.maximum(tl.max(v_scale, 0), 1e-30)
            v_scale_norm = tl.div_rn(v_scale, v_scale_max)
            p_v = (p * v_scale_norm[None, :]).to(tl.float16)
            pv = tl.dot(p_v, _int8_to_f16(v, CONVERT))
            acc = tl.fma(acc, re_scale[:, None], pv * v_scale_max)
            e_max = n_e_max

    # Phase 2: diagonal / tail / custom-mask / sliding-window tiles.
    for start_n in range(full_end_k, end_k, BLOCK_N):
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

            k_t = tl.load(
                K_Buffer + offs_kv_loc[:, None] * stride_buf_kbs + base_k,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0,
            )
            k_scale = tl.load(
                K_Scale + offs_kv_loc * stride_ks_bs + cur_kv_head,
                mask=mask_n,
                other=0.0,
            )
            qk = tl.dot(q, tl.trans(_int8_to_f16(k_t, CONVERT)))
            qk *= (k_scale * sm_scale)[None, :]
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)
            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = tl.fma(deno, re_scale, tl.sum(p, 1))

            v = tl.load(
                V_Buffer + offs_kv_loc[:, None] * stride_buf_vbs + base_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0,
            )
            v_scale = tl.load(
                V_Scale + offs_kv_loc * stride_vs_bs + cur_kv_head,
                mask=mask_n,
                other=0.0,
            )
            # Mask v_scale per row so keys outside final_mask (future tokens or
            # tokens beyond cur_seq_kv_len) never influence v_scale_max.
            v_scale_masked = tl.where(final_mask, v_scale[None, :], 0.0)
            v_scale_max = tl.maximum(tl.max(v_scale_masked, 1), 1e-30)
            v_scale_norm = tl.div_rn(v_scale_masked, v_scale_max[:, None])
            p_v = (p * v_scale_norm).to(tl.float16)
            pv = tl.dot(p_v, _int8_to_f16(v, CONVERT))
            acc = tl.fma(acc, re_scale[:, None], pv * v_scale_max[:, None])
            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        deno += tl.exp(cur_sink - e_max)

    no_kv = deno == 0.0
    deno_safe = tl.where(no_kv, 1.0, deno)
    offs_o = (
        (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(
        O + offs_o,
        acc / deno_safe[:, None],
        mask=mask_m[:, None] & mask_dv[None, :],
    )


def extend_attention_fwd_unified_int8(
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
    block_m: Optional[int] = None,
    block_n: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    convert: str = "magic",
) -> None:
    """1-stage unified extend attention reading both prefix and chunk from int8 cache."""
    assert k_buffer.dtype == torch.int8 and v_buffer.dtype == torch.int8
    assert k_scale_buffer.dtype == torch.float32
    assert v_scale_buffer.dtype == torch.float32
    assert k_buffer.stride(2) == 1 and v_buffer.stride(2) == 1
    assert k_scale_buffer.stride(1) == 1 and v_scale_buffer.stride(1) == 1
    assert convert in CONVERSIONS, convert

    batch_size = qo_indptr.shape[0] - 1
    if batch_size == 0 or max_len_extend == 0:
        return

    Lq, Lv = q.shape[-1], v_buffer.shape[-1]
    sm_scale = sm_scale if sm_scale is not None else 1.0 / (Lq**0.5)
    head_num = q.shape[1]
    kv_group_num = head_num // k_buffer.shape[1]

    if sliding_window_size > 0 and window_start_pos is None:
        window_start_pos = torch.zeros(batch_size, dtype=torch.int32, device=q.device)

    def_m, def_n, def_warps, def_stages = _default_extend_tile_unified(max(Lq, Lv))
    BLOCK_M = def_m if block_m is None else block_m
    BLOCK_N = def_n if block_n is None else block_n
    warps = def_warps if num_warps is None else num_warps
    stages = def_stages if num_stages is None else num_stages

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    _fwd_extend_unified_kernel_int8[grid](
        q,
        o,
        k_buffer,
        v_buffer,
        k_scale_buffer,
        v_scale_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        prefix_lens,
        custom_mask,
        mask_indptr,
        sinks,
        window_start_pos,
        sm_scale,
        kv_group_num,
        q.stride(0),
        q.stride(1),
        o.stride(0),
        o.stride(1),
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
        BLOCK_DMODEL=triton.next_power_of_2(Lq),
        BLOCK_DV=triton.next_power_of_2(Lv),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=is_causal,
        USE_CUSTOM_MASK=custom_mask is not None,
        HAS_SINK=sinks is not None,
        CONVERT=convert,
        num_warps=warps,
        num_stages=stages,
    )
