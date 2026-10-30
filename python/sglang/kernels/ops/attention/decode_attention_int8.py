"""Split-KV decode attention over the int8_per_token_head KV cache.

Reads int8 K/V ``[slots, kv_heads, head_dim]`` and fp32 per-(slot, head)
scales ``[slots, kv_heads]`` straight from the pool. The QK and PV dots run
in fp16 on the raw int8 values; the K scale multiplies the scores and the V
scale the probabilities, so no dequantized copy of the cache is made.
Stage 2 (the reduction over splits) is the BF16 path's.
"""

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

# "magic" builds fp16 from the int8 bits (two values per instruction); "cast"
# is Triton's per-element int-to-float conversion.
CONVERSIONS = ("magic", "cast")


@triton.jit
def _int8_to_f16(x, CONVERT: tl.constexpr):
    if CONVERT == "magic":
        # Bias each byte to x + 128, place it in the mantissa of 1024.0 so the
        # fp16 value is 1024 + x + 128 (exact), then subtract 1152.
        return tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b32 u, magic, bias;
            mov.b32 magic, 0x64646464;
            mov.b32 bias, 0x64806480;
            xor.b32 u, $2, 0x80808080;
            prmt.b32 $0, u, magic, 0x5150;
            prmt.b32 $1, u, magic, 0x5352;
            sub.f16x2 $0, $0, bias;
            sub.f16x2 $1, $1, bias;
            }
            """,
            constraints="=r,=r,r",
            args=[x],
            dtype=tl.float16,
            is_pure=True,
            pack=4,
        )
    else:
        return x.to(tl.float16)


@triton.jit
def _fwd_grouped_kernel_stage1_int8(
    Q,
    K_Buffer,
    V_Buffer,
    K_Scale,
    V_Scale,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_ks_bs,
    stride_vs_bs,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    CONVERT: tl.constexpr,
):
    # int64: flat offsets into Att_Out can exceed 2**31.
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

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    kv_start = tl.load(kv_indptr + cur_batch)
    seq_len = tl.load(kv_indptr + cur_batch + 1) - kv_start
    kv_splits = tl.load(num_kv_splits + cur_batch)
    # Must match stage 2's split boundaries.
    kv_len_per_split = tl.cdiv(tl.cdiv(seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    split_start = kv_len_per_split * split_kv_id
    split_end = tl.minimum(split_start + kv_len_per_split, seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_end > split_start:
        offs_q = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
        )
        q = tl.load(Q + offs_q, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
        q = q.to(tl.float16)
        base_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
        base_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]

        for start_n in tl.range(split_start, split_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_end
            kv_loc = tl.load(kv_indices + kv_start + offs_n, mask=mask_n, other=0)

            k = tl.load(
                K_Buffer + kv_loc[None, :] * stride_buf_kbs + base_k,
                mask=mask_n[None, :] & mask_d[:, None],
                other=0,
            )
            k_scale = tl.load(
                K_Scale + kv_loc * stride_ks_bs + cur_kv_head, mask=mask_n, other=0.0
            )
            qk = tl.dot(q, _int8_to_f16(k, CONVERT))
            qk *= (k_scale * sm_scale)[None, :]
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)
            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))

            v = tl.load(
                V_Buffer + kv_loc[:, None] * stride_buf_vbs + base_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0,
            )
            v_scale = tl.load(
                V_Scale + kv_loc * stride_vs_bs + cur_kv_head, mask=mask_n, other=0.0
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            # Scale by v_scale / tile max so P stays in fp16's normal range;
            # the max is multiplied back after the dot.
            v_scale_max = tl.maximum(tl.max(v_scale, 0), 1e-30)
            p_v = (p * (v_scale / v_scale_max)[None, :]).to(tl.float16)
            pv = tl.dot(p_v, _int8_to_f16(v, CONVERT))
            acc = acc * re_scale[:, None] + pv * v_scale_max

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )
        # Att_Lse is Att_Out without the head_dim axis (see the BF16 kernel).
        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv
        tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)


def _default_tile(head_dim):
    # Fastest of a 48-way sweep at head_dim 128 (results/a100-40g/20261030-160608):
    # 128 tokens and 3 stages keep 96 KB of int8 K/V in flight per CTA.
    # Wider heads halve both to stay within shared memory; not measured.
    if head_dim <= 128:
        return 128, 3
    return 64, 2


def decode_attention_fwd_int8(
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
    block_n: Optional[int] = None,
    num_warps: int = 4,
    num_stages: Optional[int] = None,
    convert: str = "magic",
) -> None:
    """GQA/MHA decode over an int8 KV cache with per-(slot, head) scales.

    ``q`` / ``o``: ``[batch, q_heads, head_dim]``. ``k_buffer`` / ``v_buffer``:
    int8 ``[slots, kv_heads, head_dim]``. ``k_scale_buffer`` /
    ``v_scale_buffer``: fp32 ``[slots, kv_heads]``. ``attn_logits`` /
    ``attn_lse`` and ``num_kv_splits`` are the BF16 decode's split workspace.
    ``block_n``, ``num_warps``, ``num_stages`` and ``convert`` are tuning knobs;
    ``None`` picks the tile from the head dims.
    """
    assert k_buffer.dtype == torch.int8 and v_buffer.dtype == torch.int8
    assert k_scale_buffer.dtype == torch.float32
    assert v_scale_buffer.dtype == torch.float32
    assert k_scale_buffer.stride(1) == 1 and v_scale_buffer.stride(1) == 1
    assert k_buffer.stride(2) == 1 and v_buffer.stride(2) == 1
    assert max_kv_splits == attn_logits.shape[2]
    assert convert in CONVERSIONS, convert

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = head_num // k_buffer.shape[1]
    Lk, Lv = k_buffer.shape[-1], v_buffer.shape[-1]
    default_block_n, default_stages = _default_tile(max(Lk, Lv))
    block_n = default_block_n if block_n is None else block_n
    num_stages = default_stages if num_stages is None else num_stages
    grid = (batch, _grouped_head_tiles(head_num, kv_group_num), max_kv_splits)
    _fwd_grouped_kernel_stage1_int8[grid](
        q,
        k_buffer,
        v_buffer,
        k_scale_buffer,
        v_scale_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        k_scale_buffer.stride(0),
        v_scale_buffer.stride(0),
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=triton.next_power_of_2(Lk),
        BLOCK_DV=triton.next_power_of_2(Lv),
        BLOCK_N=block_n,
        BLOCK_H=_GROUPED_BLOCK_H,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        Lk=Lk,
        Lv=Lv,
        CONVERT=convert,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _decode_softmax_reducev_fwd(
        logits=attn_logits,
        lse=attn_lse,
        q=q,
        o=o,
        v_scale=1.0,
        v_buffer=v_buffer,
        kv_indptr=kv_indptr,
        num_kv_splits=num_kv_splits,
        max_kv_splits=max_kv_splits,
        sinks=sinks,
    )
