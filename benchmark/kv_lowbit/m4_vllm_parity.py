"""Cross-engine parity check: SGLang int4_per_token_head vs vLLM int4_per_token_head.

Verifies on the same GPU inputs:
1. Raw INT4 quantize + pack + scale/zp steganography (post-RHT) is 100% byte-identical.
2. Full RHT + quantize + pack write path aligns with vLLM (seed 0x9E3779B9).
3. Fused INT4 decode attention output aligns with vLLM's TRITON_ATTN INT4 path.

Usage (from the SGLang env):
  python benchmark/kv_lowbit/m4_vllm_parity.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _child_env(*, env: str, gpu: int) -> dict[str, str]:
    child = dict(
        os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID"
    )
    if env == "vllm":
        child["LD_LIBRARY_PATH"] = ":".join(
            p
            for p in child.get("LD_LIBRARY_PATH", "").split(":")
            if p and "compat" not in p
        )
    return child


def _run_sglang_dump(args) -> None:
    import torch

    from sglang.kernels.ops.attention.decode_attention_int4 import (
        decode_attention_fwd_int4,
    )
    from sglang.kernels.ops.kvcache.int4_per_token_head import (
        store_int4_per_token_head_kv,
    )

    torch.manual_seed(0)
    batch, seq_len, num_q_heads, num_kv_heads, head_dim = 4, 256, 32, 8, 128
    total = batch * seq_len
    q = torch.randn((batch, num_q_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    k = 2.0 * torch.randn(
        (total, num_kv_heads, head_dim), dtype=torch.bfloat16, device="cuda"
    )
    v = 2.0 * torch.randn(
        (total, num_kv_heads, head_dim), dtype=torch.bfloat16, device="cuda"
    )
    loc = torch.arange(total, dtype=torch.int64, device="cuda")

    k_buf = torch.zeros(
        (total, num_kv_heads, head_dim // 2), dtype=torch.uint8, device="cuda"
    )
    v_buf = torch.zeros_like(k_buf)
    k_scale = torch.zeros((total, num_kv_heads), dtype=torch.float32, device="cuda")
    v_scale = torch.zeros_like(k_scale)
    store_int4_per_token_head_kv(
        k=k,
        v=v,
        loc=loc,
        k_cache=k_buf,
        v_cache=v_buf,
        k_scale=k_scale,
        v_scale=v_scale,
        use_rht=True,
    )

    # Also store without RHT to test pure quantization + packing + steganography byte parity.
    k_buf_norht = torch.zeros_like(k_buf)
    v_buf_norht = torch.zeros_like(v_buf)
    k_scale_norht = torch.zeros_like(k_scale)
    v_scale_norht = torch.zeros_like(v_scale)
    store_int4_per_token_head_kv(
        k=k,
        v=v,
        loc=loc,
        k_cache=k_buf_norht,
        v_cache=v_buf_norht,
        k_scale=k_scale_norht,
        v_scale=v_scale_norht,
        use_rht=False,
    )

    kv_indptr = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * seq_len
    o = torch.empty_like(q)
    max_splits = 4
    decode_attention_fwd_int4(
        q=q,
        k_buffer=k_buf,
        v_buffer=v_buf,
        k_scale_buffer=k_scale,
        v_scale_buffer=v_scale,
        o=o,
        kv_indptr=kv_indptr,
        kv_indices=loc.to(torch.int32),
        attn_logits=torch.empty(
            (batch, num_q_heads, max_splits, head_dim),
            dtype=torch.float32,
            device="cuda",
        ),
        attn_lse=torch.empty(
            (batch, num_q_heads, max_splits), dtype=torch.float32, device="cuda"
        ),
        num_kv_splits=torch.full(
            (batch,), max_splits, dtype=torch.int32, device="cuda"
        ),
        max_kv_splits=max_splits,
        sm_scale=head_dim**-0.5,
        use_rht=True,
    )
    torch.cuda.synchronize()
    torch.save(
        {
            "q": q.cpu(),
            "k": k.cpu(),
            "v": v.cpu(),
            "k_buf": k_buf.cpu(),
            "v_buf": v_buf.cpu(),
            "k_scale": k_scale.cpu(),
            "v_scale": v_scale.cpu(),
            "k_buf_norht": k_buf_norht.cpu(),
            "v_buf_norht": v_buf_norht.cpu(),
            "k_scale_norht": k_scale_norht.cpu(),
            "v_scale_norht": v_scale_norht.cpu(),
            "o": o.cpu(),
            "batch": batch,
            "seq_len": seq_len,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        },
        args.dump,
    )


def _run_vllm_compare(args) -> None:
    import torch
    import vllm.v1.attention.ops.int4_per_token_head as vi4
    import vllm.v1.attention.ops.triton_unified_attention as tua
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_head_quant,
    )
    from vllm.v1.kv_cache_interface import KVQuantMode

    d = torch.load(args.dump, weights_only=True)
    batch, seq_len = d["batch"], d["seq_len"]
    num_kv_heads, head_dim = d["num_kv_heads"], d["head_dim"]
    total = batch * seq_len
    block_size = 16
    blocks_per_seq = seq_len // block_size
    num_blocks = batch * blocks_per_seq
    q = d["q"].cuda()
    k = d["k"].cuda()
    v = d["v"].cuda()
    slot_mapping = torch.arange(total, dtype=torch.int64, device="cuda")

    def alloc_vllm_sep():
        shape = (num_blocks, block_size, num_kv_heads)
        kc = torch.zeros((*shape, head_dim // 2), dtype=torch.uint8, device="cuda")
        vc = torch.zeros_like(kc)
        ks = torch.zeros(shape, dtype=torch.float32, device="cuda")
        vs = torch.zeros_like(ks)
        return kc, vc, ks, vs

    # 1. Pure quant+pack parity (calling vLLM's _reshape_cache_int4_kernel directly without RHT).
    kc0, vc0, ks0, vs0 = alloc_vllm_sep()
    vi4._run_reshape_kernel(
        vi4._reshape_cache_int4_kernel,
        key=k,
        value=v,
        key_cache=kc0,
        value_cache=vc0,
        k_scale_cache=ks0,
        v_scale_cache=vs0,
        slot_mapping=slot_mapping,
        packing_factor=2,
    )
    kc0_flat = kc0.reshape(total, num_kv_heads, head_dim // 2).cpu()
    vc0_flat = vc0.reshape(total, num_kv_heads, head_dim // 2).cpu()
    ks0_flat = ks0.reshape(total, num_kv_heads).cpu()
    vs0_flat = vs0.reshape(total, num_kv_heads).cpu()
    norht_exact = {
        "k_bytes_exact": bool(torch.equal(kc0_flat, d["k_buf_norht"])),
        "v_bytes_exact": bool(torch.equal(vc0_flat, d["v_buf_norht"])),
        "k_scale_bits_exact": bool(
            torch.equal(
                ks0_flat.view(torch.int32), d["k_scale_norht"].view(torch.int32)
            )
        ),
        "v_scale_bits_exact": bool(
            torch.equal(
                vs0_flat.view(torch.int32), d["v_scale_norht"].view(torch.int32)
            )
        ),
    }

    # 2. Full RHT + quant + pack parity (both default hadacore/triton and triton MMA Tier 2).
    def eval_rht_mode(force_triton_rht: bool):
        orig_flag = vi4._HADACORE_AVAILABLE
        if force_triton_rht:
            vi4._HADACORE_AVAILABLE = False
        try:
            kc, vc, ks, vs = alloc_vllm_sep()
            triton_reshape_and_cache_flash_per_token_head_quant(
                key=k,
                value=v,
                key_cache=kc,
                value_cache=vc,
                k_scale_cache=ks,
                v_scale_cache=vs,
                slot_mapping=slot_mapping,
                kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
            )
            kc_f = kc.reshape(total, num_kv_heads, head_dim // 2).cpu()
            vc_f = vc.reshape(total, num_kv_heads, head_dim // 2).cpu()
            ks_f = ks.reshape(total, num_kv_heads).cpu()
            vs_f = vs.reshape(total, num_kv_heads).cpu()

            block_table = torch.arange(
                num_blocks, dtype=torch.int32, device="cuda"
            ).view(batch, blocks_per_seq)
            cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
            seqused_k = torch.full((batch,), seq_len, dtype=torch.int32, device="cuda")
            out_vllm = torch.empty_like(q)
            tua.unified_attention(
                q=q,
                k=kc,
                v=vc,
                out=out_vllm,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=1,
                seqused_k=seqused_k,
                max_seqlen_k=seq_len,
                softmax_scale=head_dim**-0.5,
                causal=True,
                window_size=(-1, -1),
                block_table=block_table,
                softcap=0.0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                k_scale_cache=ks,
                v_scale_cache=vs,
                kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
            )
            torch.cuda.synchronize()
            o_vllm = out_vllm.cpu().float()
            o_sgl = d["o"].float()
            cos = torch.nn.functional.cosine_similarity(
                o_vllm.flatten(), o_sgl.flatten(), dim=0
            ).item()
            ks_v_val = (ks_f.view(torch.int32) & -16).view(torch.float32)
            ks_s_val = (d["k_scale"].view(torch.int32) & -16).view(torch.float32)
            ks_max_rel = (
                ((ks_v_val - ks_s_val).abs() / ks_s_val.clamp_min(1e-6)).max().item()
            )
            return {
                "vllm_hadacore": bool(vi4._hadacore_available()),
                "k_byte_match_rate": (kc_f == d["k_buf"]).float().mean().item(),
                "v_byte_match_rate": (vc_f == d["v_buf"]).float().mean().item(),
                "k_scale_bit_match_rate": (
                    ks_f.view(torch.int32) == d["k_scale"].view(torch.int32)
                )
                .float()
                .mean()
                .item(),
                "v_scale_bit_match_rate": (
                    vs_f.view(torch.int32) == d["v_scale"].view(torch.int32)
                )
                .float()
                .mean()
                .item(),
                "k_scale_max_rel_diff": ks_max_rel,
                "k_max_nibble_diff": max(
                    ((kc_f & 0xF).int() - (d["k_buf"] & 0xF).int()).abs().max().item(),
                    (((kc_f >> 4) & 0xF).int() - ((d["k_buf"] >> 4) & 0xF).int())
                    .abs()
                    .max()
                    .item(),
                ),
                "attn_cos_sim": cos,
                "attn_max_abs_diff": (o_vllm - o_sgl).abs().max().item(),
            }
        finally:
            vi4._HADACORE_AVAILABLE = orig_flag

    rht_default = eval_rht_mode(force_triton_rht=False)
    rht_triton = eval_rht_mode(force_triton_rht=True)
    # Exact bit-parity holds for no-RHT and vLLM's Triton-MMA RHT (same fp32 Hadamard
    # accumulation before bf16 cast). vLLM's default hadacore path rescales by 1/sqrt(d)
    # inside bf16 and multiplies by sqrt(d) outside, introducing ~1 bf16 ULP (~0.5%) diff.
    parity_ok = (
        all(norht_exact.values())
        and rht_triton["k_byte_match_rate"] == 1.0
        and rht_triton["v_byte_match_rate"] == 1.0
        and rht_triton["k_scale_bit_match_rate"] == 1.0
        and rht_triton["v_scale_bit_match_rate"] == 1.0
        and rht_triton["k_max_nibble_diff"] == 0
        and rht_triton["attn_cos_sim"] >= 0.9999
        and rht_default["k_byte_match_rate"] >= 0.95
        and rht_default["v_byte_match_rate"] >= 0.95
        and rht_default["attn_cos_sim"] >= 0.995
    )
    res = {
        "parity_ok": parity_ok,
        "norht_exact": norht_exact,
        "rht_vllm_default": rht_default,
        "rht_vllm_triton_mma": rht_triton,
    }
    print(json.dumps(res, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--vllm-python",
        default=os.path.expandvars("$KVQ_WORK/envs/vllm/bin/python"),
    )
    p.add_argument(
        "--mode", choices=("orchestrate", "sglang", "vllm"), default="orchestrate"
    )
    p.add_argument("--dump", type=Path, default=None)
    args = p.parse_args()

    if args.mode == "sglang":
        _run_sglang_dump(args)
        return
    if args.mode == "vllm":
        _run_vllm_compare(args)
        return

    if not Path(args.vllm_python).exists():
        raise SystemExit(f"vLLM python not found: {args.vllm_python}")
    with tempfile.TemporaryDirectory(prefix="m4_vllm_parity_") as td:
        dump = Path(td) / "sglang_int4.pt"
        subprocess.run(
            [sys.executable, __file__, "--mode", "sglang", "--dump", str(dump)],
            env=_child_env(env="sglang", gpu=args.gpu),
            check=True,
        )
        subprocess.run(
            [args.vllm_python, __file__, "--mode", "vllm", "--dump", str(dump)],
            env=_child_env(env="vllm", gpu=args.gpu),
            check=True,
        )


if __name__ == "__main__":
    main()
