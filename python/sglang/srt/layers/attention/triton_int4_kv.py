"""Triton-backend fused reads of the int4_per_token_head KV cache.

Both decode and extend (regular 2-stage and deterministic 1-stage unified) run
fused Triton kernels straight on the pool's packed uint8 data and fp32 scales
(with embedded 4-bit zero-points), without materializing any BF16 copy of the
cached prefix.
"""

from typing import Optional

import torch

from sglang.kernels.ops.attention.decode_attention_int4 import (
    decode_attention_fwd_int4,
)
from sglang.kernels.ops.attention.extend_attention_int4 import (
    extend_attention_fwd_int4,
    extend_attention_fwd_unified_int4,
)
from sglang.kernels.ops.kvcache.int4_per_token_head import (
    can_fuse_rht,
    get_rht_matrix,
    get_rht_signs,
    get_rht_split_matrices,
    rht_block_size,
)
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    KVCacheAttentionAccessKind,
)


def uses_fused_int4_kv(
    *, quant_method, phase: str, page_size: int, dcp_size: int
) -> bool:
    """Whether this pool's recipe declares a fused INT4 Triton read for ``phase``."""
    if quant_method is None:
        return False
    access = quant_method.resolve_attention_access(phase, "triton")
    if (
        access is None
        or access.kind != KVCacheAttentionAccessKind.FUSED_DEQUANT
        or access.scale_recipe != "int4_per_token_head"
    ):
        return False
    # The kernels address KV by slot and have no DCP merge.
    if page_size != 1 or dcp_size > 1:
        raise NotImplementedError(
            f"{quant_method.name} on the Triton backend needs page_size 1 and "
            f"no decode context parallelism; got page_size={page_size}, "
            f"dcp_size={dcp_size}."
        )
    return True


def warm_int4_rht_caches(
    *, head_dim: int, v_head_dim: int, device: torch.device
) -> None:
    """Pre-populate cached RHT matrices on ``device`` before CUDA graph capture."""
    dev = torch.device(device)
    for dim in {head_dim, v_head_dim}:
        if dim <= 0 or dim % 2 != 0:
            continue
        b = rht_block_size(dim)
        get_rht_signs(b, dev)
        if can_fuse_rht(dim):
            for dtype in (torch.bfloat16, torch.float16):
                get_rht_split_matrices(dim, dtype, dev)
        elif dev.type == "cuda" and 16 <= b <= 128:
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                get_rht_matrix(b, dtype, dev)


def _check_supported_layer(layer, *, phase: str, score_mod=None) -> None:
    if layer.xai_temperature_len > 0 or layer.k_scale is not None:
        raise NotImplementedError(
            f"int4_per_token_head {phase} supports neither xai temperature "
            "nor checkpoint KV scales."
        )
    if score_mod is not None:
        raise NotImplementedError(
            f"int4_per_token_head {phase} does not support score_mod."
        )


def forward_decode_int4(
    *,
    pool,
    layer,
    q: torch.Tensor,
    o: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    num_kv_splits: torch.Tensor,
    max_kv_splits: int,
    logit_cap: float,
    sinks: Optional[torch.Tensor],
) -> None:
    _check_supported_layer(layer, phase="decode")
    k_buffer, v_buffer, k_scale, v_scale = pool.get_raw_kv_buffer(layer.layer_id)
    decode_attention_fwd_int4(
        q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        k_scale_buffer=k_scale,
        v_scale_buffer=v_scale,
        o=o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        attn_logits=attn_logits,
        attn_lse=attn_lse,
        num_kv_splits=num_kv_splits,
        max_kv_splits=max_kv_splits,
        sm_scale=layer.scaling,
        logit_cap=logit_cap,
        sinks=sinks,
    )


def forward_extend_int4(
    *,
    pool,
    layer,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    custom_mask: Optional[torch.Tensor],
    is_causal: bool,
    mask_indptr: Optional[torch.Tensor],
    max_len_extend: int,
    logit_cap: float,
    sliding_window_size: int,
    sinks: Optional[torch.Tensor],
    window_kv_offsets: Optional[torch.Tensor],
    score_mod=None,
) -> None:
    _check_supported_layer(layer, phase="extend", score_mod=score_mod)
    k_buffer, v_buffer, k_scale, v_scale = pool.get_raw_kv_buffer(layer.layer_id)
    extend_attention_fwd_int4(
        q_extend=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
        k_extend=k.contiguous(),
        v_extend=v.contiguous(),
        o_extend=o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        k_scale_buffer=k_scale,
        v_scale_buffer=v_scale,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        custom_mask=custom_mask,
        is_causal=is_causal,
        mask_indptr=mask_indptr,
        max_len_extend=max_len_extend,
        sm_scale=layer.scaling,
        logit_cap=logit_cap,
        sliding_window_size=sliding_window_size,
        sinks=sinks,
        window_kv_offsets=window_kv_offsets,
    )


def forward_extend_unified_int4(
    *,
    pool,
    layer,
    q: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_len_extend: int,
    custom_mask: Optional[torch.Tensor],
    mask_indptr: Optional[torch.Tensor],
    logit_cap: float,
    is_causal: bool,
    sliding_window_size: int,
    sinks: Optional[torch.Tensor],
    window_start_pos: Optional[torch.Tensor],
    score_mod=None,
) -> None:
    _check_supported_layer(layer, phase="unified extend", score_mod=score_mod)
    k_buffer, v_buffer, k_scale, v_scale = pool.get_raw_kv_buffer(layer.layer_id)
    extend_attention_fwd_unified_int4(
        q=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
        o=o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        k_scale_buffer=k_scale,
        v_scale_buffer=v_scale,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        prefix_lens=prefix_lens,
        max_len_extend=max_len_extend,
        custom_mask=custom_mask,
        mask_indptr=mask_indptr,
        sm_scale=layer.scaling,
        logit_cap=logit_cap,
        is_causal=is_causal,
        sliding_window_size=sliding_window_size,
        sinks=sinks,
        window_start_pos=window_start_pos,
    )
