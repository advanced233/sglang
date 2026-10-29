"""Per-(token, head) symmetric INT8 KV cache: quantize-and-store and dequant.

Pool layout: data ``[slots, heads, head_dim]`` int8 and one fp32 scale per
(slot, head) in ``[slots, heads]``. Quantization uses the same arithmetic as
vLLM's ``int8_per_token_head`` writer.
"""

import torch
import triton
import triton.language as tl

_INT8_MAX = tl.constexpr(127.0)
_INT8_MIN = tl.constexpr(-128.0)
# vLLM's scale floor; keeps an all-zero row finite.
_MIN_SCALE = tl.constexpr(1e-6)
# Rows (slot, head pairs) per dequant program; arbitrary.
_DEQUANT_ROWS = 32


@triton.jit
def _quantize_row(src, dst, scale_dst, N: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(src + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x)) / _INT8_MAX, _MIN_SCALE)
    q = x * (1.0 / scale)
    # Round half away from zero; the int8 cast below truncates.
    q = tl.where(q >= 0, q + 0.5, q - 0.5)
    q = tl.clamp(q, _INT8_MIN, _INT8_MAX)
    tl.store(dst + offs, q.to(tl.int8), mask=mask)
    tl.store(scale_dst, scale)


@triton.jit
def _store_kv_kernel(
    k_ptr,
    v_ptr,
    loc_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    stride_k_tok,
    stride_k_head,
    stride_v_tok,
    stride_v_head,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    row = tl.load(loc_ptr + tok).to(tl.int64) * NUM_HEADS + head
    _quantize_row(
        k_ptr + tok * stride_k_tok + head * stride_k_head,
        k_cache_ptr + row * HEAD_DIM,
        k_scale_ptr + row,
        HEAD_DIM,
        BLOCK_D,
    )
    _quantize_row(
        v_ptr + tok * stride_v_tok + head * stride_v_head,
        v_cache_ptr + row * HEAD_DIM_V,
        v_scale_ptr + row,
        HEAD_DIM_V,
        BLOCK_D,
    )


def store_int8_per_token_head_kv(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Quantize ``k``/``v`` ``[tokens, heads, dim]`` and write them at slots ``loc``."""
    num_tokens, num_heads, head_dim = k.shape
    head_dim_v = v.shape[2]
    assert k.stride(2) == 1 and v.stride(2) == 1
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    assert k_scale.is_contiguous() and v_scale.is_contiguous()
    assert k_cache.dtype == torch.int8 and k_scale.dtype == torch.float32
    assert k_cache.shape[1:] == (num_heads, head_dim)
    assert v_cache.shape[1:] == (num_heads, head_dim_v)
    if num_tokens == 0:
        return
    _store_kv_kernel[(num_tokens, num_heads)](
        k,
        v,
        loc,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        HEAD_DIM_V=head_dim_v,
        BLOCK_D=triton.next_power_of_2(max(head_dim, head_dim_v)),
        num_warps=4,
    )


@triton.jit
def _dequant_kernel(
    src_ptr,
    scale_ptr,
    out_ptr,
    num_rows,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    offs = tl.arange(0, BLOCK_D)
    mask = (rows < num_rows)[:, None] & (offs < HEAD_DIM)[None, :]
    ptrs = rows[:, None] * HEAD_DIM + offs[None, :]
    x = tl.load(src_ptr + ptrs, mask=mask, other=0).to(tl.float32)
    scale = tl.load(scale_ptr + rows, mask=rows < num_rows, other=0.0)
    y = x * scale[:, None]
    tl.store(out_ptr + ptrs, y.to(out_ptr.dtype.element_ty), mask=mask)


def dequant_int8_per_token_head(
    *, data: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize ``data`` ``[..., heads, dim]`` with ``scale`` ``[..., heads]``."""
    assert data.is_contiguous() and scale.is_contiguous()
    assert data.shape[:-1] == scale.shape
    out = torch.empty(data.shape, dtype=dtype, device=data.device)
    head_dim = data.shape[-1]
    num_rows = scale.numel()
    if num_rows == 0:
        return out
    _dequant_kernel[(triton.cdiv(num_rows, _DEQUANT_ROWS),)](
        data,
        scale,
        out,
        num_rows,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        ROWS=_DEQUANT_ROWS,
    )
    return out
