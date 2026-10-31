"""Triton-backend reads of the int8_per_token_head KV cache.

Decode runs the fused INT8 kernel on the pool's raw int8 data and scales.
Extend still runs the BF16 kernel, but on only the prefix rows it reads,
dequantized per call, instead of a dequantized copy of the whole layer.
"""

from typing import Optional

import torch

from sglang.kernels.ops.attention.decode_attention_int8 import (
    decode_attention_fwd_int8,
)
from sglang.kernels.ops.kvcache.int8_per_token_head import (
    dequant_int8_per_token_head,
)
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    KVCacheAttentionAccessKind,
)


def uses_fused_int8_decode(*, quant_method, page_size: int, dcp_size: int) -> bool:
    """Whether this pool's recipe declares a fused decode read for Triton."""
    if quant_method is None:
        return False
    access = quant_method.resolve_attention_access("decode", "triton")
    if access is None or access.kind != KVCacheAttentionAccessKind.FUSED_DEQUANT:
        return False
    # The kernel addresses KV by slot and has no DCP merge.
    if page_size != 1 or dcp_size > 1:
        raise NotImplementedError(
            f"{quant_method.name} on the Triton backend needs page_size 1 and "
            f"no decode context parallelism; got page_size={page_size}, "
            f"dcp_size={dcp_size}."
        )
    return True


def forward_decode_int8(
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
    if layer.xai_temperature_len > 0 or layer.k_scale is not None:
        raise NotImplementedError(
            "int8_per_token_head decode supports neither xai temperature "
            "nor checkpoint KV scales."
        )
    k_buffer, v_buffer, k_scale, v_scale = pool.get_raw_kv_buffer(layer.layer_id)
    decode_attention_fwd_int8(
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


def dequantized_prefix_kv(
    *, pool, layer_id: int, kv_indices: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prefix K/V rows in ``kv_indices`` order, and indices into those rows.

    ``kv_indices`` must hold exactly the prefix slots (no padding tail).
    """
    k_buffer, v_buffer, k_scale, v_scale = pool.get_raw_kv_buffer(layer_id)
    num_rows = kv_indices.numel()
    # An empty prefix still needs a valid buffer pointer; read padding slot 0.
    slots = kv_indices if num_rows else kv_indices.new_zeros(1)
    k_rows = dequant_int8_per_token_head(
        data=k_buffer, scale=k_scale, dtype=dtype, slots=slots
    )
    v_rows = dequant_int8_per_token_head(
        data=v_buffer, scale=v_scale, dtype=dtype, slots=slots
    )
    rows = torch.arange(num_rows, dtype=kv_indices.dtype, device=kv_indices.device)
    return k_rows, v_rows, rows
