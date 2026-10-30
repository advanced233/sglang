"""INT8 per-token-head split-KV decode against fp32 attention over the dequantized cache."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.decode_attention_int8 import (
    CONVERSIONS,
    decode_attention_fwd_int8,
)
from sglang.kernels.ops.kvcache.int8_per_token_head import (
    store_int8_per_token_head_kv,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
MAX_KV_SPLITS = 8
# Odd lengths leave partial tiles and empty splits; the split counts differ per request.
SEQ_LENS = [1, 37, 300, 1031]
KV_SPLITS = [1, 3, 8, 8]


def _int8_cache(*, num_slots, num_kv_heads, head_dim):
    data = torch.zeros(
        (num_slots, num_kv_heads, head_dim), dtype=torch.int8, device=DEVICE
    )
    scale = torch.zeros((num_slots, num_kv_heads), dtype=torch.float32, device=DEVICE)
    return data, scale


def _reference(*, q, k, v, k_scale, v_scale, sm_scale, logit_cap):
    group = q.shape[1] // k.shape[1]
    outs, start = [], 0
    for b, n in enumerate(SEQ_LENS):
        rows = slice(start, start + n)
        kd = (k[rows].float() * k_scale[rows, :, None]).repeat_interleave(group, 1)
        vd = (v[rows].float() * v_scale[rows, :, None]).repeat_interleave(group, 1)
        s = torch.einsum("hd,nhd->hn", q[b].float(), kd) * sm_scale
        if logit_cap > 0:
            s = logit_cap * torch.tanh(s / logit_cap)
        outs.append(torch.einsum("hn,nhd->hd", s.softmax(dim=-1), vd))
        start += n
    return torch.stack(outs)


@pytest.mark.parametrize("convert", CONVERSIONS)
@pytest.mark.parametrize("block_n", [32, 64])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim,logit_cap",
    [(32, 8, 128, 0.0), (16, 2, 64, 0.0), (8, 8, 128, 30.0)],
    ids=["gqa4", "gqa8_d64", "mha_logit_cap"],
)
def test_matches_reference(
    convert, block_n, num_q_heads, num_kv_heads, head_dim, logit_cap
):
    torch.manual_seed(0)
    total, batch = sum(SEQ_LENS), len(SEQ_LENS)
    kv_shape = (total, num_kv_heads, head_dim)
    k = 2 * torch.randn(kv_shape, dtype=torch.bfloat16, device=DEVICE)
    v = 2 * torch.randn(kv_shape, dtype=torch.bfloat16, device=DEVICE)
    q = torch.randn((batch, num_q_heads, head_dim), dtype=torch.bfloat16, device=DEVICE)
    # Tokens sit in scattered slots, as in a pool that has been reused.
    num_slots = total + 65
    slots = torch.randperm(num_slots - 1, device=DEVICE)[:total] + 1
    shape = dict(num_slots=num_slots, num_kv_heads=num_kv_heads, head_dim=head_dim)
    k_cache, k_scale = _int8_cache(**shape)
    v_cache, v_scale = _int8_cache(**shape)
    store_int8_per_token_head_kv(
        k=k,
        v=v,
        loc=slots,
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
    kv_indptr[1:] = torch.tensor(SEQ_LENS, device=DEVICE).cumsum(0)
    o = torch.empty_like(q)
    sm_scale = head_dim**-0.5

    decode_attention_fwd_int8(
        q=q,
        k_buffer=k_cache,
        v_buffer=v_cache,
        k_scale_buffer=k_scale,
        v_scale_buffer=v_scale,
        o=o,
        kv_indptr=kv_indptr,
        kv_indices=slots,
        attn_logits=torch.empty(
            (batch, num_q_heads, MAX_KV_SPLITS, head_dim),
            dtype=torch.float32,
            device=DEVICE,
        ),
        attn_lse=torch.empty(
            (batch, num_q_heads, MAX_KV_SPLITS), dtype=torch.float32, device=DEVICE
        ),
        num_kv_splits=torch.tensor(KV_SPLITS, dtype=torch.int32, device=DEVICE),
        max_kv_splits=MAX_KV_SPLITS,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
        block_n=block_n,
        convert=convert,
    )

    ref = _reference(
        q=q,
        k=k_cache[slots],
        v=v_cache[slots],
        k_scale=k_scale[slots],
        v_scale=v_scale[slots],
        sm_scale=sm_scale,
        logit_cap=logit_cap,
    )
    # fp16 P plus the bf16 output; a scale or mask bug is off by O(1).
    torch.testing.assert_close(o.float(), ref, atol=5e-3, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
