"""Per-(token, head) asymmetric INT4 KV cache with Randomized Hadamard Transform.

Pool layout:
- Packed data: ``[slots, heads, head_dim // 2]`` ``uint8``; each byte holds two
  unsigned 4-bit integers ``[0, 15]`` with the even feature index in the low
  nibble and the odd index in the high nibble:
  ``packed[i] = (q[2*i] & 0xF) | ((q[2*i + 1] & 0xF) << 4)``.
- Scales: ``[slots, heads]`` ``float32`` stored in separate contiguous buffers
  (aligned for 128-bit vectorized loads). The 4-bit unsigned zero-point is
  embedded in the low 4 mantissa bits of the float32 scale:
  ``scale_packed = bitcast_f32((bitcast_i32(scale) & -16) | (zp & 0xF))``.

Quantization semantics and RHT sign seed (``0x9E3779B9``) match vLLM's
``int4_per_token_head`` mode so that writes produce byte-aligned cache values.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# vLLM's _get_rht_signs seed for round 0.
RHT_SEED = 0x9E3779B9

_INT4_MAX = tl.constexpr(15.0)
# Scale floor matching vLLM / INT8; keeps an all-zero or constant row finite
# and ensures scale_bits & -16 stays a positive normal float32.
_MIN_SCALE = tl.constexpr(1e-6)
# Rows (slot, head pairs) per dequant program.
_DEQUANT_ROWS = 32
# Rows per fused RHT+quantize store CTA; matches Triton tl.dot minimum M=16.
_STORE_RHT_BLOCK_R = 16
# Triton MMA RHT bounds (D x D bf16/fp16 matrix fits in shared memory).
_TRITON_RHT_MIN_D = 16
_TRITON_RHT_MAX_D = 128
# Power-of-two head dims where split-matrix fused RHT (D/2 >= 16) applies.
FUSED_RHT_HEAD_DIMS = (32, 64, 128)

_HADAMARD_CACHE: dict[tuple[int, str], torch.Tensor] = {}
_RHT_SIGNS_CACHE: dict[tuple[int, str], torch.Tensor] = {}
_RHT_MATRIX_CACHE: dict[tuple[int, torch.dtype, str], torch.Tensor] = {}
_RHT_SPLIT_MATRIX_CACHE: dict[
    tuple[int, torch.dtype, str], tuple[torch.Tensor, torch.Tensor]
] = {}


def rht_block_size(head_dim: int) -> int:
    """Largest power-of-two divisor of ``head_dim`` (vLLM block-diagonal RHT)."""
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    return head_dim & -head_dim


def can_fuse_rht(head_dim: int) -> bool:
    """Whether ``head_dim`` supports single-block split-matrix fused RHT."""
    return head_dim in FUSED_RHT_HEAD_DIMS


def _get_hadamard_matrix_f32(block_size: int, device: torch.device) -> torch.Tensor:
    """Unnormalized Sylvester Hadamard matrix ``H`` (``H @ H = block_size * I``)."""
    key = (block_size, str(device))
    cached = _HADAMARD_CACHE.get(key)
    if cached is None:
        h = torch.ones((1, 1), dtype=torch.float32, device="cpu")
        while h.shape[0] < block_size:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        cached = h.to(device=device).contiguous()
        _HADAMARD_CACHE[key] = cached
    return cached


def get_rht_signs(block_size: int, device: torch.device) -> torch.Tensor:
    """Cached deterministic ``+-1.0`` fp32 sign vector using vLLM's round-0 seed."""
    key = (block_size, str(device))
    cached = _RHT_SIGNS_CACHE.get(key)
    if cached is None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(RHT_SEED)
        signs = (
            2.0
            * torch.bernoulli(
                torch.full((block_size,), 0.5, device="cpu"), generator=gen
            )
            - 1.0
        )
        cached = signs.to(device=device).contiguous()
        _RHT_SIGNS_CACHE[key] = cached
    return cached


def get_rht_matrix(
    block_size: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Cached fused RHT matrix ``M = diag(signs) @ H`` (entries strictly ``+-1``).

    Forward RHT on row vectors is ``x @ M``; unnormalized inverse RHT is
    ``y @ M.T``, so ``(x @ M) @ M.T / block_size == x``.
    """
    key = (block_size, dtype, str(device))
    cached = _RHT_MATRIX_CACHE.get(key)
    if cached is None:
        h = _get_hadamard_matrix_f32(block_size, device)
        signs = get_rht_signs(block_size, device)
        m = (signs[:, None] * h).to(dtype=dtype).contiguous()
        _RHT_MATRIX_CACHE[key] = m
        cached = m
    return cached


def get_rht_split_matrices(
    block_size: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached ``(M_even, M_odd)`` with shapes ``[block_size, block_size // 2]``.

    For row vectors ``x``:
    - ``x @ M_even == (x @ M)[..., 0::2]``
    - ``x @ M_odd  == (x @ M)[..., 1::2]``
    - ``(y_even @ M_even.T + y_odd @ M_odd.T) / block_size == (y @ M.T) / block_size``
    """
    key = (block_size, dtype, str(device))
    cached = _RHT_SPLIT_MATRIX_CACHE.get(key)
    if cached is None:
        m = get_rht_matrix(block_size, dtype, device)
        m_even = m[:, 0::2].contiguous()
        m_odd = m[:, 1::2].contiguous()
        cached = (m_even, m_odd)
        _RHT_SPLIT_MATRIX_CACHE[key] = cached
    return cached


def pack_scale_and_zp(scale: torch.Tensor, zp: torch.Tensor) -> torch.Tensor:
    """Embed integer 4-bit ``zp`` ``[0, 15]`` into the low 4 mantissa bits of fp32 ``scale``."""
    scale_f32 = scale.to(torch.float32).contiguous()
    zp_i32 = zp.to(torch.int32)
    packed_bits = (scale_f32.view(torch.int32) & -16) | (zp_i32 & 0xF)
    return packed_bits.view(torch.float32)


def unpack_scale_and_zp(
    packed_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract ``(scale_f32, zp_f32)`` from steganographed fp32 ``packed_scale``."""
    bits = packed_scale.contiguous().view(torch.int32)
    zp = (bits & 0xF).to(torch.float32)
    scale = (bits & -16).view(torch.float32)
    return scale, zp


@triton.jit
def _rht_mma_kernel(
    x_ptr,
    m_ptr,
    out_ptr,
    n_rows,
    stride_x_row: tl.int64,
    stride_o_row: tl.int64,
    inv_norm,
    BLOCK_M: tl.constexpr,
    D: tl.constexpr,
    TRANSPOSE_M: tl.constexpr,
    USE_IEEE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    row_mask = rows < n_rows

    x = tl.load(
        x_ptr + rows[:, None] * stride_x_row + cols[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    m = tl.load(m_ptr + cols[:, None] * D + cols[None, :])
    if TRANSPOSE_M:
        m = tl.trans(m)
    if USE_IEEE:
        y = tl.dot(x, m, input_precision="ieee", out_dtype=tl.float32)
    else:
        y = tl.dot(x, m, out_dtype=tl.float32)
    if TRANSPOSE_M:
        y = y * inv_norm
    tl.store(
        out_ptr + rows[:, None] * stride_o_row + cols[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=row_mask[:, None],
    )


def apply_rht(
    x: torch.Tensor, *, inverse: bool = False, out_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Apply vLLM's single Randomized Hadamard Transform along the last dim.

    - Forward (``inverse=False``): ``(x * signs) @ H_b`` (norm ``sqrt(b)``).
    - Inverse (``inverse=True``): ``((x @ H_b) * signs) / b`` (exact inverse).

    For non-power-of-two ``head_dim``, applies block-diagonally with block size
    ``b = head_dim & -head_dim``.
    """
    orig_shape = x.shape
    target_dtype = out_dtype or x.dtype
    d = orig_shape[-1]
    b = rht_block_size(d)

    if x.numel() == 0:
        return torch.empty(orig_shape, dtype=target_dtype, device=x.device)

    if (
        x.is_cuda
        and _TRITON_RHT_MIN_D <= b <= _TRITON_RHT_MAX_D
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        x2d = x.contiguous().reshape(-1, b)
        n_rows = x2d.shape[0]
        out2d = torch.empty((n_rows, b), dtype=target_dtype, device=x.device)
        m = get_rht_matrix(b, x.dtype, x.device)
        inv_norm = (1.0 / b) if inverse else 1.0
        block_m = 16
        _rht_mma_kernel[(triton.cdiv(n_rows, block_m),)](
            x2d,
            m,
            out2d,
            n_rows,
            x2d.stride(0),
            out2d.stride(0),
            inv_norm,
            BLOCK_M=block_m,
            D=b,
            TRANSPOSE_M=inverse,
            USE_IEEE=(x.dtype == torch.float32),
            num_warps=4,
            num_stages=1,
        )
        return out2d.reshape(orig_shape)

    # Pure PyTorch fallback (CPU or block size outside [16, 128]).
    xf = x.float().reshape(*orig_shape[:-1], d // b, b)
    signs = get_rht_signs(b, x.device)
    if inverse:
        y = _hadamard_butterfly(xf) * signs * (1.0 / b)
    else:
        y = _hadamard_butterfly(xf * signs)
    return y.reshape(orig_shape).to(target_dtype)


def _hadamard_butterfly(x: torch.Tensor) -> torch.Tensor:
    """In-order unnormalized Walsh-Hadamard butterfly along the last dimension."""
    d = x.shape[-1]
    h = 1
    while h < d:
        xv = x.view(*x.shape[:-1], d // (2 * h), 2, h)
        a = xv[..., 0, :]
        b = xv[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).reshape(x.shape)
        h <<= 1
    return x


@triton.jit
def _round_half_away_u4(x):
    # Clamp before int32 cast to prevent overflow on extreme constant inputs
    # while preserving exact rounding across the full [0, 15] output domain.
    xc = tl.clamp(x, -1.0, 16.0)
    r = tl.where(xc >= 0, (xc + 0.5).to(tl.int32), (xc - 0.5).to(tl.int32))
    return tl.clamp(r.to(tl.float32), 0.0, _INT4_MAX)


@triton.jit
def _pack_int4_nibbles(lo, hi):
    return (lo & 0xF) | ((hi & 0xF) << 4)


@triton.jit
def _quantize_pack_row_int4(
    src, dst, scale_dst, HEAD_DIM: tl.constexpr, PACKED_BLOCK: tl.constexpr
):
    half_offs = tl.arange(0, PACKED_BLOCK)
    even_offs = half_offs * 2
    odd_offs = half_offs * 2 + 1
    half_d: tl.constexpr = HEAD_DIM // 2
    even_mask = even_offs < HEAD_DIM
    odd_mask = odd_offs < HEAD_DIM

    x_even = tl.load(src + even_offs, mask=even_mask, other=0.0).to(tl.float32)
    x_odd = tl.load(src + odd_offs, mask=odd_mask, other=0.0).to(tl.float32)

    x_min = tl.minimum(
        tl.min(tl.where(even_mask, x_even, float("inf"))),
        tl.min(tl.where(odd_mask, x_odd, float("inf"))),
    )
    x_max = tl.maximum(
        tl.max(tl.where(even_mask, x_even, float("-inf"))),
        tl.max(tl.where(odd_mask, x_odd, float("-inf"))),
    )
    scale = tl.maximum((x_max - x_min) / _INT4_MAX, _MIN_SCALE)
    zp_f = _round_half_away_u4(-x_min / scale)

    inv_scale = 1.0 / scale
    q_even = _round_half_away_u4(x_even * inv_scale + zp_f)
    q_odd = _round_half_away_u4(x_odd * inv_scale + zp_f)

    zp_int = zp_f.to(tl.int32)
    scale_bits = scale.to(tl.int32, bitcast=True)
    scale_packed = ((scale_bits & -16) | (zp_int & 0xF)).to(tl.float32, bitcast=True)
    tl.store(scale_dst, scale_packed)

    packed = _pack_int4_nibbles(q_even.to(tl.uint8), q_odd.to(tl.uint8))
    tl.store(dst + half_offs, packed, mask=half_offs < half_d)


@triton.jit
def _store_kv_int4_kernel(
    k_ptr,
    v_ptr,
    loc_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    stride_k_tok: tl.int64,
    stride_k_head: tl.int64,
    stride_v_tok: tl.int64,
    stride_v_head: tl.int64,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    PACKED_BLOCK: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    slot = tl.load(loc_ptr + tok).to(tl.int64)
    if slot < 0:
        return
    row = slot * NUM_HEADS + head
    _quantize_pack_row_int4(
        k_ptr + tok * stride_k_tok + head * stride_k_head,
        k_cache_ptr + row * (HEAD_DIM // 2),
        k_scale_ptr + row,
        HEAD_DIM,
        PACKED_BLOCK,
    )
    _quantize_pack_row_int4(
        v_ptr + tok * stride_v_tok + head * stride_v_head,
        v_cache_ptr + row * (HEAD_DIM_V // 2),
        v_scale_ptr + row,
        HEAD_DIM_V,
        PACKED_BLOCK,
    )


@triton.jit
def _quantize_pack_block_2d_int4(
    x_even,
    x_odd,
    dst_ptr,
    scale_dst_ptr,
    dst_rows,
    valid_mask,
    HALF_D: tl.constexpr,
):
    x_min = tl.minimum(tl.min(x_even, axis=1), tl.min(x_odd, axis=1))
    x_max = tl.maximum(tl.max(x_even, axis=1), tl.max(x_odd, axis=1))
    scale = tl.maximum((x_max - x_min) / _INT4_MAX, _MIN_SCALE)
    zp_f = _round_half_away_u4(-x_min / scale)

    inv_scale = 1.0 / scale
    q_even = _round_half_away_u4(x_even * inv_scale[:, None] + zp_f[:, None])
    q_odd = _round_half_away_u4(x_odd * inv_scale[:, None] + zp_f[:, None])

    zp_int = zp_f.to(tl.int32)
    scale_bits = scale.to(tl.int32, bitcast=True)
    scale_packed = ((scale_bits & -16) | (zp_int & 0xF)).to(tl.float32, bitcast=True)
    tl.store(scale_dst_ptr + dst_rows, scale_packed, mask=valid_mask)

    packed = _pack_int4_nibbles(q_even.to(tl.uint8), q_odd.to(tl.uint8))
    offs_h = tl.arange(0, HALF_D)
    tl.store(
        dst_ptr + dst_rows[:, None] * HALF_D + offs_h[None, :],
        packed,
        mask=valid_mask[:, None],
    )


@triton.jit
def _store_kv_int4_fused_rht_kernel(
    k_ptr,
    v_ptr,
    loc_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    m_even_ptr,
    m_odd_ptr,
    total_rows,
    stride_k_tok: tl.int64,
    stride_k_head: tl.int64,
    stride_v_tok: tl.int64,
    stride_v_head: tl.int64,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Fused RHT + asymmetric INT4 quantization + nibble packing for K and V.

    Processes ``BLOCK_R`` ``(token, head)`` rows per CTA, loading the split RHT
    matrices ``M_even`` / ``M_odd`` ``[D, D // 2]`` once and reusing them for
    both K and V so the write path takes a single kernel launch.
    """
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    row_mask = rows < total_rows
    tok = rows // NUM_HEADS
    head = rows % NUM_HEADS
    slot = tl.load(loc_ptr + tok, mask=row_mask, other=-1).to(tl.int64)
    valid = row_mask & (slot >= 0)
    dst_rows = tl.where(valid, slot * NUM_HEADS + head, 0)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_h = tl.arange(0, HALF_D)

    m_even = tl.load(m_even_ptr + offs_d[:, None] * HALF_D + offs_h[None, :])
    m_odd = tl.load(m_odd_ptr + offs_d[:, None] * HALF_D + offs_h[None, :])

    # K: rotate -> match vLLM's input-dtype rounding before quantization -> pack.
    k_ptrs = (
        k_ptr
        + tok[:, None] * stride_k_tok
        + head[:, None] * stride_k_head
        + offs_d[None, :]
    )
    k_blk = tl.load(k_ptrs, mask=valid[:, None], other=0.0)
    k_even = (
        tl.dot(k_blk, m_even, out_dtype=tl.float32)
        .to(k_ptr.dtype.element_ty)
        .to(tl.float32)
    )
    k_odd = (
        tl.dot(k_blk, m_odd, out_dtype=tl.float32)
        .to(k_ptr.dtype.element_ty)
        .to(tl.float32)
    )
    _quantize_pack_block_2d_int4(
        k_even, k_odd, k_cache_ptr, k_scale_ptr, dst_rows, valid, HALF_D
    )

    # V: reuse M_even / M_odd in registers.
    v_ptrs = (
        v_ptr
        + tok[:, None] * stride_v_tok
        + head[:, None] * stride_v_head
        + offs_d[None, :]
    )
    v_blk = tl.load(v_ptrs, mask=valid[:, None], other=0.0)
    v_even = (
        tl.dot(v_blk, m_even, out_dtype=tl.float32)
        .to(v_ptr.dtype.element_ty)
        .to(tl.float32)
    )
    v_odd = (
        tl.dot(v_blk, m_odd, out_dtype=tl.float32)
        .to(v_ptr.dtype.element_ty)
        .to(tl.float32)
    )
    _quantize_pack_block_2d_int4(
        v_even, v_odd, v_cache_ptr, v_scale_ptr, dst_rows, valid, HALF_D
    )


def store_int4_per_token_head_kv(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    loc: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    use_rht: bool = True,
) -> None:
    """RHT-rotate (if enabled), asymmetrically quantize to INT4, and store at ``loc``."""
    num_tokens, num_heads, head_dim = k.shape
    head_dim_v = v.shape[2]
    assert head_dim % 2 == 0 and head_dim_v % 2 == 0, (
        f"INT4 KV cache requires even head dims, got head_dim={head_dim}, "
        f"head_dim_v={head_dim_v}"
    )
    assert k.stride(2) == 1 and v.stride(2) == 1
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    assert k_scale.is_contiguous() and v_scale.is_contiguous()
    assert k_cache.dtype == torch.uint8 and v_cache.dtype == torch.uint8
    assert k_scale.dtype == torch.float32 and v_scale.dtype == torch.float32
    assert k_cache.shape[1:] == (num_heads, head_dim // 2)
    assert v_cache.shape[1:] == (num_heads, head_dim_v // 2)
    assert k_scale.shape == (k_cache.shape[0], num_heads)
    assert v_scale.shape == (v_cache.shape[0], num_heads)
    if num_tokens == 0:
        return

    if (
        use_rht
        and head_dim == head_dim_v
        and can_fuse_rht(head_dim)
        and k.is_cuda
        and k.dtype == v.dtype
        and k.dtype in (torch.float16, torch.bfloat16)
    ):
        m_even, m_odd = get_rht_split_matrices(head_dim, k.dtype, k.device)
        total_rows = num_tokens * num_heads
        grid = (triton.cdiv(total_rows, _STORE_RHT_BLOCK_R),)
        _store_kv_int4_fused_rht_kernel[grid](
            k,
            v,
            loc,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            m_even,
            m_odd,
            total_rows,
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            HALF_D=head_dim // 2,
            BLOCK_R=_STORE_RHT_BLOCK_R,
            num_warps=4,
            num_stages=1,
        )
        return

    if use_rht:
        k = apply_rht(k).to(k.dtype)
        v = apply_rht(v).to(v.dtype)

    _store_kv_int4_kernel[(num_tokens, num_heads)](
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
        PACKED_BLOCK=triton.next_power_of_2(max(head_dim, head_dim_v) // 2),
        num_warps=4,
    )


@triton.jit
def _dequant_int4_kernel(
    src_ptr,
    scale_ptr,
    out_ptr,
    slots_ptr,
    num_rows,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_D: tl.constexpr,
    BLOCK_HALF_D: tl.constexpr,
    ROWS: tl.constexpr,
    GATHER: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    row_mask = rows < num_rows
    if GATHER:
        slot = tl.load(slots_ptr + rows // NUM_HEADS, mask=row_mask, other=0)
        src_rows = slot.to(tl.int64) * NUM_HEADS + rows % NUM_HEADS
    else:
        src_rows = rows

    offs_h = tl.arange(0, BLOCK_HALF_D)
    half_mask = row_mask[:, None] & (offs_h < HALF_D)[None, :]
    packed = tl.load(
        src_ptr + src_rows[:, None] * HALF_D + offs_h[None, :],
        mask=half_mask,
        other=0,
    )
    s_raw = tl.load(scale_ptr + src_rows, mask=row_mask, other=0.0)
    s_bits = s_raw.to(tl.int32, bitcast=True)
    zp = (s_bits & 0xF).to(tl.float32)
    scale = (s_bits & -16).to(tl.float32, bitcast=True)

    lo = (packed & 0xF).to(tl.float32)
    hi = ((packed >> 4) & 0xF).to(tl.float32)
    y_lo = (lo - zp[:, None]) * scale[:, None]
    y_hi = (hi - zp[:, None]) * scale[:, None]

    out_base = out_ptr + rows[:, None] * HEAD_DIM
    tl.store(
        out_base + (offs_h * 2)[None, :],
        y_lo.to(out_ptr.dtype.element_ty),
        mask=half_mask,
    )
    tl.store(
        out_base + (offs_h * 2 + 1)[None, :],
        y_hi.to(out_ptr.dtype.element_ty),
        mask=half_mask,
    )


def dequant_int4_per_token_head(
    *,
    data: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    slots: Optional[torch.Tensor] = None,
    use_rht: bool = True,
) -> torch.Tensor:
    """Dequantize packed INT4 ``data`` ``[slots, heads, dim // 2]`` with ``scale`` ``[slots, heads]``.

    When ``use_rht=True`` (default), applies the inverse RHT so the result is in
    the original unrotated basis (matching plain attention reads). When
    ``use_rht=False``, returns values directly in the stored basis.
    """
    assert data.is_contiguous() and scale.is_contiguous()
    assert data.dtype == torch.uint8 and scale.dtype == torch.float32
    assert data.shape[:-1] == scale.shape
    num_heads, half_d = data.shape[-2], data.shape[-1]
    head_dim = half_d * 2
    out_slots = data.shape[0] if slots is None else slots.numel()
    out_shape = (out_slots, num_heads, head_dim)
    if out_slots == 0 or num_heads == 0:
        return torch.empty(out_shape, dtype=dtype, device=data.device)

    # Keep fp32 intermediate before inverse RHT so the rotation matches kvq_ref.py.
    deq_dtype = torch.float32 if use_rht else dtype
    out = torch.empty(out_shape, dtype=deq_dtype, device=data.device)
    num_rows = out_slots * num_heads
    _dequant_int4_kernel[(triton.cdiv(num_rows, _DEQUANT_ROWS),)](
        data,
        scale,
        out,
        slots,
        num_rows,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        HALF_D=half_d,
        BLOCK_HALF_D=triton.next_power_of_2(half_d),
        ROWS=_DEQUANT_ROWS,
        GATHER=slots is not None,
    )
    if use_rht:
        return apply_rht(out, inverse=True, out_dtype=dtype)
    return out
