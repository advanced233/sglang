"""Decode INT8 KV ablation: unfused vs naive fused vs optimized fused kernel.

Compares the full optimization ladder on identical inputs (default Qwen3-8B
M3.0 shape: 6 sequences x 16k context, 32 Q heads, 8 KV heads, head_dim 128),
using CUDA graph replay + cold L2 timing (triton.testing.do_bench):

  1. bf16_baseline          : SGLang BF16 decode_attention_fwd (403 MB HBM read)
  2. unfused_whole_pool     : M2-style whole-pool dequant to HBM + BF16 decode
  3. unfused_active_gather  : gather-dequant active slots to HBM + BF16 decode
                              (208 MB read + 403 MB write + 403 MB read = 1014 MB)
  4. naive_fused_n32_s2     : fused HBM read (208 MB, 0 MB temp), but naive
                              in-loop elementwise dequant:
                              (k.to(fp32) * k_scale).to(bf16) before tl.dot
  5. naive_fused_n64_s2     : naive fused with BLOCK_N=64, num_stages=2
  6. naive_fused_n128_s3    : naive fused with same optimal tile (128, 3)
  7. opt_fused_cast_n128_s3 : production decode_attention_fwd_int8 (convert="cast")
                              scale hoisted out of dot + v_scale_max + fp16 dot
  8. opt_fused_magic_n128_s3: production decode_attention_fwd_int8 (convert="magic")
                              + PTX magic bit-trick int8->fp16

Usage on A100-40G:
  python benchmark/kv_lowbit/bench_decode_int8_ablation.py

Optional single-launch mode for ncu profiling:
  ncu --set full -o naive_fused --profile-from-start off \\
      python benchmark/kv_lowbit/bench_decode_int8_ablation.py \\
      --ncu-target naive_fused_n128_s3
"""

import argparse
import statistics
import traceback
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.decode_attention import (
    _GROUPED_BLOCK_H,
    _MIN_BLOCK_KV,
    _decode_softmax_reducev_fwd,
    _grouped_head_tiles,
    decode_attention_fwd,
    tanh,
)
from sglang.kernels.ops.attention.decode_attention_int8 import decode_attention_fwd_int8
from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton
from sglang.kernels.ops.kvcache.int8_per_token_head import (
    _DEQUANT_ROWS,
    _dequant_kernel,
    store_int8_per_token_head_kv,
)


@dataclass(frozen=True)
class Shape:
    batch: int = 6
    seq_len: int = 16384
    num_q_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128

    @property
    def num_tokens(self) -> int:
        return self.batch * self.seq_len


# ----------------------------------------------------------------------------
# Naive fused INT8 decode kernel (for ablation against decode_attention_int8.py)
# ----------------------------------------------------------------------------


@triton.jit
def _fwd_grouped_kernel_stage1_int8_naive(
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
):
    """Naive fused INT8 decode Stage 1.

    Differences vs production ``_fwd_grouped_kernel_stage1_int8``:
      1. Keeps Q and dot products in BF16 (no FP16 switch, no PTX magic trick).
      2. Dequantizes every K and V element inside the inner loop BEFORE tl.dot:
            k_bf16 = (k.to(fp32) * k_scale[None, :]).to(bf16)
            v_bf16 = (v.to(fp32) * v_scale[:, None]).to(bf16)
         instead of hoisting k_scale out of tl.dot and folding v_scale into P
         with v_scale_max normalization.
    Still eliminates intermediate HBM round-trips (0 MB temp HBM, reads 208 MB).
    """
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
        # Keep Q in its original dtype (BF16).
        q = tl.load(Q + offs_q, mask=mask_h[:, None] & mask_d[None, :], other=0.0)
        base_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
        base_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]

        for start_n in tl.range(split_start, split_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_end
            kv_loc = tl.load(kv_indices + kv_start + offs_n, mask=mask_n, other=0)

            # --- Naive K dequant: elementwise int8 -> fp32 * k_scale -> bf16 ---
            k = tl.load(
                K_Buffer + kv_loc[None, :] * stride_buf_kbs + base_k,
                mask=mask_n[None, :] & mask_d[:, None],
                other=0,
            )
            k_scale = tl.load(
                K_Scale + kv_loc * stride_ks_bs + cur_kv_head, mask=mask_n, other=0.0
            )
            k_deq = (k.to(tl.float32) * k_scale[None, :]).to(q.dtype)
            qk = tl.dot(q, k_deq) * sm_scale
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)
            qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))

            # --- Naive V dequant: elementwise int8 -> fp32 * v_scale -> bf16 ---
            v = tl.load(
                V_Buffer + kv_loc[:, None] * stride_buf_vbs + base_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0,
            )
            v_scale = tl.load(
                V_Scale + kv_loc * stride_vs_bs + cur_kv_head, mask=mask_n, other=0.0
            )
            v_deq = (v.to(tl.float32) * v_scale[:, None]).to(q.dtype)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            pv = tl.dot(p.to(q.dtype), v_deq)
            acc = acc * re_scale[:, None] + pv

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
        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv
        tl.store(Att_Lse + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)


def decode_attention_fwd_int8_naive(
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
    block_n: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    """Launch the naive fused INT8 decode kernel (reuses BF16 Stage 2)."""
    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = head_num // k_buffer.shape[1]
    Lk, Lv = k_buffer.shape[-1], v_buffer.shape[-1]
    grid = (batch, _grouped_head_tiles(head_num, kv_group_num), max_kv_splits)
    _fwd_grouped_kernel_stage1_int8_naive[grid](
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


# ----------------------------------------------------------------------------
# Benchmark harness (same cold-L2 CUDA graph protocol as m3_kernel_diag.py)
# ----------------------------------------------------------------------------


def _dequant_into(
    *,
    data: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    slots: Optional[torch.Tensor] = None,
) -> None:
    """Zero-alloc wrapper around M2's _dequant_kernel for CUDA graph capture."""
    num_heads, head_dim = data.shape[-2], data.shape[-1]
    num_rows = out.numel() // head_dim
    if num_rows == 0:
        return
    _dequant_kernel[(triton.cdiv(num_rows, _DEQUANT_ROWS),)](
        data,
        scale,
        out,
        slots,
        num_rows,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        ROWS=_DEQUANT_ROWS,
        GATHER=slots is not None,
    )


def _make_inputs(shape: Shape, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*size):
        return torch.randn(*size, generator=gen, device="cuda", dtype=torch.bfloat16)

    q = randn(shape.batch, shape.num_q_heads, shape.head_dim)
    k = randn(shape.num_tokens, shape.num_kv_heads, shape.head_dim)
    v = randn(shape.num_tokens, shape.num_kv_heads, shape.head_dim)
    return q, k, v


def _reference_attention(*, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, shape: Shape):
    group = shape.num_q_heads // shape.num_kv_heads
    kv_shape = (shape.batch, shape.seq_len, shape.num_kv_heads, shape.head_dim)
    qf = q.float().view(shape.batch, shape.num_kv_heads, group, shape.head_dim)
    scores = torch.einsum("bhgd,bshd->bhgs", qf, k.float().view(kv_shape))
    probs = (scores * shape.head_dim**-0.5).softmax(dim=-1)
    out = torch.einsum("bhgs,bshd->bhgd", probs, v.float().view(kv_shape))
    return out.reshape(shape.batch, shape.num_q_heads, shape.head_dim)


def _make_meta(*, shape: Shape, max_kv_splits: int, device: torch.device):
    kv_indptr = torch.arange(shape.batch + 1, dtype=torch.int32, device=device)
    kv_indptr *= shape.seq_len
    num_kv_splits = torch.empty(shape.batch, dtype=torch.int32, device=device)
    get_num_kv_splits_triton[(1,)](
        num_kv_splits_ptr=num_kv_splits,
        seq_lens_ptr=torch.full(
            (shape.batch,), shape.seq_len, dtype=torch.int32, device=device
        ),
        num_seq=shape.batch,
        num_group=1,
        num_head=shape.num_q_heads,
        num_kv_head=shape.num_kv_heads,
        max_kv_splits=max_kv_splits,
        device_core_count=torch.cuda.get_device_properties(device).multi_processor_count,
        MAX_NUM_SEQ=256,
    )
    split_shape = (shape.batch, shape.num_q_heads, max_kv_splits)
    return {
        "kv_indptr": kv_indptr,
        "kv_indices": torch.arange(
            1, shape.num_tokens + 1, dtype=torch.int64, device=device
        ),
        "attn_logits": torch.empty(
            (*split_shape, shape.head_dim), dtype=torch.float32, device=device
        ),
        "attn_lse": torch.empty(split_shape, dtype=torch.float32, device=device),
        "num_kv_splits": num_kv_splits,
        "max_kv_splits": max_kv_splits,
    }


def _make_int8_pool(
    *, k: torch.Tensor, v: torch.Tensor, kv_indices: torch.Tensor, pool_slots: int
):
    device = k.device
    num_kv_heads, head_dim = k.shape[1], k.shape[2]
    k_buf = torch.zeros((pool_slots, num_kv_heads, head_dim), dtype=torch.int8, device=device)
    v_buf = torch.zeros_like(k_buf)
    k_scale = torch.zeros((pool_slots, num_kv_heads), dtype=torch.float32, device=device)
    v_scale = torch.zeros_like(k_scale)
    store_int8_per_token_head_kv(
        k=k,
        v=v,
        loc=kv_indices,
        k_cache=k_buf,
        v_cache=v_buf,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return k_buf, v_buf, k_scale, v_scale


def _capture_graph(op):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            op()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op()
    torch.cuda.synchronize()
    return graph


def _accuracy(*, out: torch.Tensor, ref: torch.Tensor):
    o, r = out.float().flatten(), ref.flatten()
    cos = torch.nn.functional.cosine_similarity(o, r, dim=0).item()
    rel = ((o - r).norm() / r.norm()).item()
    return cos, rel


ALL_VARIANTS = (
    "bf16_baseline",
    "unfused_whole_pool",
    "unfused_active_gather",
    "naive_fused_n32_s2",
    "naive_fused_n64_s2",
    "naive_fused_n128_s3",
    "opt_fused_cast_n128_s3",
    "opt_fused_magic_n128_s3",
)


def _build_variant(
    *,
    name: str,
    shape: Shape,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_kv_splits: int,
    pool_slots: int,
):
    """Return (op, out, total_hbm_bytes, temp_hbm_mb)."""
    device = q.device
    sm_scale = shape.head_dim**-0.5
    meta = _make_meta(shape=shape, max_kv_splits=max_kv_splits, device=device)
    out = torch.empty_like(q)

    active_bf16_bytes = shape.num_tokens * shape.num_kv_heads * 2 * shape.head_dim * 2
    active_int8_bytes = shape.num_tokens * shape.num_kv_heads * 2 * (shape.head_dim + 4)

    if name == "bf16_baseline":
        k_buf = torch.zeros(
            (shape.num_tokens + 1, shape.num_kv_heads, shape.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        v_buf = torch.zeros_like(k_buf)
        k_buf[1:] = k
        v_buf[1:] = v

        def op():
            decode_attention_fwd(
                q=q,
                k_buffer=k_buf,
                v_buffer=v_buf,
                o=out,
                sm_scale=sm_scale,
                k_scale=1.0,
                v_scale=1.0,
                **meta,
            )

        return op, out, active_bf16_bytes, 0.0

    if name == "unfused_whole_pool":
        slots = max(pool_slots, shape.num_tokens + 1)
        k_buf, v_buf, k_scale, v_scale = _make_int8_pool(
            k=k, v=v, kv_indices=meta["kv_indices"], pool_slots=slots
        )
        tmp_k = torch.empty((slots, shape.num_kv_heads, shape.head_dim), dtype=torch.bfloat16, device=device)
        tmp_v = torch.empty_like(tmp_k)

        def op():
            _dequant_into(data=k_buf, scale=k_scale, out=tmp_k, slots=None)
            _dequant_into(data=v_buf, scale=v_scale, out=tmp_v, slots=None)
            decode_attention_fwd(
                q=q,
                k_buffer=tmp_k,
                v_buffer=tmp_v,
                o=out,
                sm_scale=sm_scale,
                k_scale=1.0,
                v_scale=1.0,
                **meta,
            )

        pool_int8_bytes = slots * shape.num_kv_heads * 2 * (shape.head_dim + 4)
        pool_bf16_bytes = slots * shape.num_kv_heads * 2 * shape.head_dim * 2
        total_hbm_bytes = pool_int8_bytes + pool_bf16_bytes + active_bf16_bytes
        temp_mb = pool_bf16_bytes / (1024**2)
        return op, out, total_hbm_bytes, temp_mb

    if name == "unfused_active_gather":
        slots = shape.num_tokens + 1
        k_buf, v_buf, k_scale, v_scale = _make_int8_pool(
            k=k, v=v, kv_indices=meta["kv_indices"], pool_slots=slots
        )
        tmp_k = torch.empty(
            (shape.num_tokens, shape.num_kv_heads, shape.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        tmp_v = torch.empty_like(tmp_k)
        gather_slots = meta["kv_indices"]
        attn_meta = dict(meta)
        attn_meta["kv_indices"] = torch.arange(
            0, shape.num_tokens, dtype=torch.int64, device=device
        )

        def op():
            _dequant_into(data=k_buf, scale=k_scale, out=tmp_k, slots=gather_slots)
            _dequant_into(data=v_buf, scale=v_scale, out=tmp_v, slots=gather_slots)
            decode_attention_fwd(
                q=q,
                k_buffer=tmp_k,
                v_buffer=tmp_v,
                o=out,
                sm_scale=sm_scale,
                k_scale=1.0,
                v_scale=1.0,
                **attn_meta,
            )

        # Read INT8+scale (208 MB) + write tmp BF16 (403 MB) + read tmp BF16 (403 MB)
        total_hbm_bytes = active_int8_bytes + 2 * active_bf16_bytes
        temp_mb = active_bf16_bytes / (1024**2)
        return op, out, total_hbm_bytes, temp_mb

    # Fused variants (0 MB temp HBM, read active INT8+scale once)
    slots = shape.num_tokens + 1
    k_buf, v_buf, k_scale, v_scale = _make_int8_pool(
        k=k, v=v, kv_indices=meta["kv_indices"], pool_slots=slots
    )

    if name.startswith("naive_fused_"):
        cfg_map = {
            "naive_fused_n32_s2": (32, 4, 2),
            "naive_fused_n64_s2": (64, 4, 2),
            "naive_fused_n128_s3": (128, 4, 3),
        }
        block_n, num_warps, num_stages = cfg_map[name]

        def op():
            decode_attention_fwd_int8_naive(
                q=q,
                k_buffer=k_buf,
                v_buffer=v_buf,
                k_scale_buffer=k_scale,
                v_scale_buffer=v_scale,
                o=out,
                sm_scale=sm_scale,
                block_n=block_n,
                num_warps=num_warps,
                num_stages=num_stages,
                **meta,
            )

        return op, out, active_int8_bytes, 0.0

    if name in ("opt_fused_cast_n128_s3", "opt_fused_magic_n128_s3"):
        convert = "cast" if "cast" in name else "magic"

        def op():
            decode_attention_fwd_int8(
                q=q,
                k_buffer=k_buf,
                v_buffer=v_buf,
                k_scale_buffer=k_scale,
                v_scale_buffer=v_scale,
                o=out,
                sm_scale=sm_scale,
                block_n=128,
                num_warps=4,
                num_stages=3,
                convert=convert,
                **meta,
            )

        return op, out, active_int8_bytes, 0.0

    raise ValueError(f"Unknown variant: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-kv-splits", type=int, default=8)
    parser.add_argument(
        "--pool-slots",
        type=int,
        default=200000,
        help="KV pool capacity used by unfused_whole_pool (default: 200,000 tokens)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=ALL_VARIANTS,
        default=None,
        help="Run only selected variants",
    )
    parser.add_argument(
        "--ncu-target",
        choices=ALL_VARIANTS,
        default=None,
        help="Run one variant once inside cudaProfilerStart/Stop for ncu",
    )
    args = parser.parse_args()

    shape = Shape(
        batch=args.batch,
        seq_len=args.seq_len,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
    )
    q, k, v = _make_inputs(shape)

    if args.ncu_target is not None:
        op, _, _, _ = _build_variant(
            name=args.ncu_target,
            shape=shape,
            q=q,
            k=k,
            v=v,
            max_kv_splits=args.max_kv_splits,
            pool_slots=args.pool_slots,
        )
        op()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        op()
        torch.cuda.profiler.stop()
        torch.cuda.synchronize()
        return

    print(
        f"Device: {torch.cuda.get_device_name(0)} | "
        f"Shape: batch={shape.batch}, seq_len={shape.seq_len}, "
        f"Q={shape.num_q_heads}, KV={shape.num_kv_heads}, D={shape.head_dim}, "
        f"max_kv_splits={args.max_kv_splits}, pool_slots={args.pool_slots}"
    )
    ref = _reference_attention(q=q, k=k, v=v, shape=shape)
    torch.cuda.empty_cache()

    variants = args.only if args.only is not None else list(ALL_VARIANTS)
    results = []
    for name in variants:
        op = out = graph = None
        try:
            op, out, total_bytes, temp_mb = _build_variant(
                name=name,
                shape=shape,
                q=q,
                k=k,
                v=v,
                max_kv_splits=args.max_kv_splits,
                pool_slots=args.pool_slots,
            )
            op()
            torch.cuda.synchronize()
            cos, rel = _accuracy(out=out, ref=ref)
            graph = _capture_graph(op)
            times_ms = triton.testing.do_bench(
                graph.replay, warmup=100, rep=1000, return_mode="all"
            )
            med_us = 1e3 * statistics.median(times_ms)
            min_us = 1e3 * min(times_ms)
            # Effective bandwidth based on the active KV bytes read (for INT8: 208 MB, BF16: 403 MB)
            active_kv_bytes = (
                shape.num_tokens * shape.num_kv_heads * 2 * shape.head_dim * 2
                if name == "bf16_baseline"
                else shape.num_tokens * shape.num_kv_heads * 2 * (shape.head_dim + 4)
            )
            eff_bw_gbps = (active_kv_bytes / 1e9) / (med_us / 1e6)
            actual_bw_gbps = (total_bytes / 1e9) / (med_us / 1e6)
            results.append(
                {
                    "name": name,
                    "status": "ok",
                    "med_us": med_us,
                    "min_us": min_us,
                    "total_mb": total_bytes / 1e6,
                    "temp_mb": temp_mb,
                    "eff_bw": eff_bw_gbps,
                    "actual_bw": actual_bw_gbps,
                    "cos": cos,
                    "rel": rel,
                }
            )
        except Exception as exc:
            traceback.print_exc()
            results.append({"name": name, "status": f"FAILED: {type(exc).__name__}: {exc}"})
        finally:
            del op, out, graph
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    bf16_us = next((r["med_us"] for r in results if r["name"] == "bf16_baseline" and r["status"] == "ok"), None)
    magic_us = next((r["med_us"] for r in results if r["name"] == "opt_fused_magic_n128_s3" and r["status"] == "ok"), None)

    print("\n| Variant | Median (us) | Min (us) | HBM Traffic (MB) | Temp VRAM (MB) | Eff KV BW (GB/s) | vs BF16 | vs Opt-Magic | CosSim | RelErr |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        if r["status"] != "ok":
            print(f"| `{r['name']}` | {r['status']} | | | | | | | | |")
            continue
        vs_bf16 = f"{bf16_us / r['med_us']:.2f}x" if bf16_us else "-"
        vs_magic = f"{r['med_us'] / magic_us:.2f}x slower" if (magic_us and r["name"] != "opt_fused_magic_n128_s3") else ("1.00x" if magic_us else "-")
        print(
            f"| `{r['name']}` | {r['med_us']:.1f} | {r['min_us']:.1f} | "
            f"{r['total_mb']:.1f} | {r['temp_mb']:.1f} | {r['eff_bw']:.0f} | "
            f"{vs_bf16} | {vs_magic} | {r['cos']:.6f} | {r['rel']:.5f} |"
        )


if __name__ == "__main__":
    main()
