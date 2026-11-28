"""INT4 per-token-head split-KV decode (with RHT) against fp32 reference over dequantized cache."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.decode_attention_int4 import (
    CONVERSIONS,
    decode_attention_fwd_int4,
)
from sglang.kernels.ops.kvcache.int4_per_token_head import (
    dequant_int4_per_token_head,
    store_int4_per_token_head_kv,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
MAX_KV_SPLITS = 8
# Odd lengths leave partial tiles and empty splits; split counts differ per request.
SEQ_LENS = [1, 37, 300, 1031]
KV_SPLITS = [1, 3, 8, 8]


def _int4_cache(*, num_slots, num_kv_heads, head_dim):
    data = torch.zeros(
        (num_slots, num_kv_heads, head_dim // 2), dtype=torch.uint8, device=DEVICE
    )
    scale = torch.zeros((num_slots, num_kv_heads), dtype=torch.float32, device=DEVICE)
    return data, scale


def _reference(*, q, k_dq, v_dq, sm_scale, logit_cap, sinks=None):
    group = q.shape[1] // k_dq.shape[1]
    outs, start = [], 0
    for b, n in enumerate(SEQ_LENS):
        rows = slice(start, start + n)
        kd = k_dq[rows].repeat_interleave(group, 1)
        vd = v_dq[rows].repeat_interleave(group, 1)
        s = torch.einsum("hd,nhd->hn", q[b].float(), kd) * sm_scale
        if logit_cap > 0:
            s = logit_cap * torch.tanh(s / logit_cap)
        if sinks is not None:
            s_max = torch.maximum(s.max(dim=-1, keepdim=True).values, sinks[:, None])
            exp_s = torch.exp(s - s_max)
            exp_sink = torch.exp(sinks - s_max.squeeze(-1))
            p = exp_s / (exp_s.sum(dim=-1, keepdim=True) + exp_sink[:, None])
        else:
            p = s.softmax(dim=-1)
        outs.append(torch.einsum("hn,nhd->hd", p, vd))
        start += n
    return torch.stack(outs)


@pytest.mark.parametrize("convert", CONVERSIONS)
@pytest.mark.parametrize("use_rht", [True, False], ids=["rht", "norht"])
@pytest.mark.parametrize("block_n", [32, None])
@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,head_dim,logit_cap,with_sinks",
    [
        (32, 8, 128, 0.0, False),
        (16, 2, 64, 0.0, True),
        (8, 8, 128, 30.0, False),
        (8, 2, 96, 0.0, False),
    ],
    ids=["gqa4_d128", "gqa8_d64_sinks", "mha_d128_cap", "gqa4_d96"],
)
def test_matches_reference(
    convert,
    use_rht,
    block_n,
    num_q_heads,
    num_kv_heads,
    head_dim,
    logit_cap,
    with_sinks,
):
    torch.manual_seed(0)
    total, batch = sum(SEQ_LENS), len(SEQ_LENS)
    kv_shape = (total, num_kv_heads, head_dim)
    k = 2 * torch.randn(kv_shape, dtype=torch.bfloat16, device=DEVICE)
    v = 2 * torch.randn(kv_shape, dtype=torch.bfloat16, device=DEVICE)
    q = torch.randn((batch, num_q_heads, head_dim), dtype=torch.bfloat16, device=DEVICE)
    sinks = (
        torch.randn((num_q_heads,), dtype=torch.float32, device=DEVICE)
        if with_sinks
        else None
    )

    num_slots = total + 65
    slots = torch.randperm(num_slots - 1, device=DEVICE)[:total] + 1
    shape = dict(num_slots=num_slots, num_kv_heads=num_kv_heads, head_dim=head_dim)
    k_cache, k_scale = _int4_cache(**shape)
    v_cache, v_scale = _int4_cache(**shape)
    store_int4_per_token_head_kv(
        k=k,
        v=v,
        loc=slots,
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        use_rht=use_rht,
    )
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=DEVICE)
    kv_indptr[1:] = torch.tensor(SEQ_LENS, device=DEVICE).cumsum(0)
    o = torch.empty_like(q)
    sm_scale = head_dim**-0.5

    decode_attention_fwd_int4(
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
        sinks=sinks,
        use_rht=use_rht,
        block_n=block_n,
        convert=convert,
    )

    k_dq = dequant_int4_per_token_head(
        data=k_cache, scale=k_scale, dtype=torch.float32, slots=slots, use_rht=use_rht
    )
    v_dq = dequant_int4_per_token_head(
        data=v_cache, scale=v_scale, dtype=torch.float32, slots=slots, use_rht=use_rht
    )
    ref = _reference(
        q=q,
        k_dq=k_dq,
        v_dq=v_dq,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
        sinks=sinks,
    )
    # fp16 Q/P + bf16 output; a zero-point, scale, nibble, or RHT bug is off by O(1).
    torch.testing.assert_close(o.float(), ref, atol=8e-3, rtol=1.5e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
