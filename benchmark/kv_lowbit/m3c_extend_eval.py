"""M3c: fused INT8 extend kernel microbenchmark and offline KL consistency eval.

Two subcommands (and ``report`` to rebuild ``summary.md`` from saved JSONL):

  kernel  Cold-L2 microbenchmark on Qwen3-8B head geometry (32 Q / 8 KV / D=128)
          comparing five extend implementations on identical inputs:
            - bf16-regular        SGLang BF16 2-stage extend
            - bf16-unified        SGLang BF16 1-stage unified extend
            - int8-m3b-gather     M3b prefix gather-dequant + BF16 extend
            - int8-fused-regular  M3c fused INT8 2-stage extend (chunk_bf16)
            - int8-fused-unified  M3c fused INT8 1-stage unified extend (chunk_q)
          Reports median latency, speedup vs BF16 and M3b, peak temporary memory
          per layer call, and numerical accuracy vs fp32 references.

  kl      Offline server-level KL consistency and numerical-semantics check on
          local docs/docs/**/*.mdx prompts (HF_HUB_OFFLINE=1 safe). Compares:
            - int8-chunk-bf16     default M3c fused 2-stage extend
            - int8-chunk-q        SGLANG_ENABLE_TRITON_INT8_UNIFIED_EXTEND=1
            - int8-det            int8_per_token_head + --enable-deterministic-inference
            - bf16-det            BF16 + --enable-deterministic-inference
          Runs four checks per config:
            - match                    prefill vs decode logprobs
            - prefill_cache_hit        radix prefix-hit decode vs fresh prefill
            - decode_cache_hit         two-turn decode-cache-hit vs fresh prefill
            - prefill_cached_vs_fresh  suffix prefill logprobs with vs without
                                       radix prefix cache (directly separates
                                       chunk_bf16 from chunk_q)

Example:
  python benchmark/kv_lowbit/m3c_extend_eval.py kernel --gpu 0 \\
      --out "$KVQ_WORK/outputs/m3c/$TS/kernel"
  python benchmark/kv_lowbit/m3c_extend_eval.py kl --model "$QWEN3_8B" --gpu 0 \\
      --out "$KVQ_WORK/outputs/m3c/$TS/kl"
"""

import argparse
import json
import os
import statistics
import sys
import time
import traceback
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m0_bench  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# Qwen3-8B attention geometry (matches m3_kernel_diag.py).
DEFAULT_SHAPES = {
    # Radix prefix hit: long cached prefix, moderate continuation chunk.
    "prefix-hit": dict(batch=4, prefix=8192, extend=512),
    # Chunked-prefill late chunk of a 16k prompt (source of M3b's +4% TTFT).
    "late-chunk": dict(batch=1, prefix=14336, extend=2048),
    # Fresh prefill with no prefix: isolates Stage 2 (chunk_bf16 vs chunk_q).
    "fresh-prefill": dict(batch=4, prefix=0, extend=4096),
}

KERNEL_IMPLS = (
    "bf16-regular",
    "bf16-unified",
    "int8-m3b-gather",
    "int8-fused-regular",
    "int8-fused-unified",
)

KL_CONFIGS = {
    "int8-chunk-bf16": (
        [
            "--attention-backend",
            "triton",
            "--kv-cache-dtype",
            "int8_per_token_head",
            "--sampling-backend",
            "pytorch",
        ],
        {},
    ),
    "int8-chunk-q": (
        [
            "--attention-backend",
            "triton",
            "--kv-cache-dtype",
            "int8_per_token_head",
            "--sampling-backend",
            "pytorch",
        ],
        {"SGLANG_ENABLE_TRITON_INT8_UNIFIED_EXTEND": "1"},
    ),
    "int8-det": (
        [
            "--attention-backend",
            "triton",
            "--kv-cache-dtype",
            "int8_per_token_head",
            "--sampling-backend",
            "pytorch",
            "--enable-deterministic-inference",
        ],
        {},
    ),
    "bf16-det": (
        [
            "--attention-backend",
            "triton",
            "--sampling-backend",
            "pytorch",
            "--enable-deterministic-inference",
        ],
        {},
    ),
}


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pk = sub.add_parser("kernel", help="Cold-L2 extend kernel microbenchmark")
    pk.add_argument("--gpu", type=int, required=True)
    pk.add_argument("--out", type=Path, required=True)
    pk.add_argument("--shapes", default=",".join(DEFAULT_SHAPES))
    pk.add_argument("--impls", default=",".join(KERNEL_IMPLS))
    pk.add_argument("--num-q-heads", type=int, default=32)
    pk.add_argument("--num-kv-heads", type=int, default=8)
    pk.add_argument("--head-dim", type=int, default=128)
    pk.add_argument("--warmup-ms", type=int, default=100)
    pk.add_argument("--rep-ms", type=int, default=400)
    pk.add_argument("--seed", type=int, default=0)

    pl = sub.add_parser("kl", help="Offline server-level KL consistency eval")
    pl.add_argument("--model", required=True)
    pl.add_argument("--gpu", type=int, required=True)
    pl.add_argument("--out", type=Path, required=True)
    pl.add_argument("--configs", default=",".join(KL_CONFIGS))
    pl.add_argument("--samples", type=int, default=32)
    pl.add_argument("--prompt-len", type=int, default=1024)
    pl.add_argument("--max-new-tokens", type=int, default=512)
    pl.add_argument("--corpus-glob", default="docs/docs/**/*.mdx")
    pl.add_argument("--ready-timeout", type=int, default=1200)
    pl.add_argument("--gpu-wait-timeout", type=int, default=3600)

    pr = sub.add_parser("report", help="Rebuild summary.md from saved JSONL files")
    pr.add_argument("--out", type=Path, required=True)
    return p.parse_args()


# ----------------------------------------------------------------------------
# Subcommand: kernel
# ----------------------------------------------------------------------------


def _chunked_reference_extend(
    *, q, k_all, v_all, batch, prefix, extend, num_q_heads, num_kv_heads
):
    """Exact fp32 causal extend reference computed in Q chunks to bound memory."""
    import torch

    head_dim = q.shape[-1]
    group = num_q_heads // num_kv_heads
    sm_scale = head_dim**-0.5
    total_kv = prefix + extend
    q_4d = q.float().view(batch, extend, num_kv_heads, group, head_dim)
    k_3d = k_all.float().view(batch, total_kv, num_kv_heads, head_dim)
    v_3d = v_all.float().view(batch, total_kv, num_kv_heads, head_dim)
    out = torch.empty_like(q_4d)
    k_pos = torch.arange(total_kv, device=q.device)
    chunk_q = 256
    for b in range(batch):
        kb = k_3d[b]  # [S, H_kv, D]
        vb = v_3d[b]
        for q0 in range(0, extend, chunk_q):
            q1 = min(q0 + chunk_q, extend)
            qb = q_4d[b, q0:q1]  # [M, H_kv, G, D]
            scores = torch.einsum("mhgd,shd->mghs", qb, kb) * sm_scale
            q_pos = prefix + torch.arange(q0, q1, device=q.device)
            causal = k_pos[None, :] <= q_pos[:, None]
            scores = scores.masked_fill(~causal[:, None, None, :], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[b, q0:q1] = torch.einsum("mghs,shd->mhgd", probs, vb)
    return out.reshape(batch * extend, num_q_heads, head_dim)


def _cosine_and_max_err(a, b):
    import torch

    af, bf = a.float().flatten(), b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(af[None], bf[None]).item()
    max_abs = (af - bf).abs().max().item()
    return cos, max_abs


def _run_kernel_bench(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    import triton

    from sglang.kernels.ops.attention.extend_attention import (
        extend_attention_fwd,
        extend_attention_fwd_unified,
    )
    from sglang.kernels.ops.attention.extend_attention_int8 import (
        extend_attention_fwd_int8,
        extend_attention_fwd_unified_int8,
    )
    from sglang.kernels.ops.kvcache.int8_per_token_head import (
        dequant_int8_per_token_head,
        store_int8_per_token_head_kv,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out / "kernel_results.jsonl"
    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    impls = [x.strip() for x in args.impls.split(",") if x.strip()]
    for s in shapes:
        if s not in DEFAULT_SHAPES:
            raise SystemExit(f"unknown shape {s!r}; choose from {list(DEFAULT_SHAPES)}")
    for impl in impls:
        if impl not in KERNEL_IMPLS:
            raise SystemExit(f"unknown impl {impl!r}; choose from {list(KERNEL_IMPLS)}")

    dev = torch.device("cuda")
    hq, hkv, d = args.num_q_heads, args.num_kv_heads, args.head_dim
    sm_scale = d**-0.5

    with open(jsonl_path, "w") as out_f:
        for shape_name in shapes:
            cfg = DEFAULT_SHAPES[shape_name]
            batch, prefix, extend = cfg["batch"], cfg["prefix"], cfg["extend"]
            total_kv = prefix + extend
            gen = torch.Generator(device=dev).manual_seed(args.seed)

            q = torch.randn(
                batch * extend, hq, d, generator=gen, device=dev, dtype=torch.bfloat16
            )
            k_all = torch.randn(
                batch * total_kv,
                hkv,
                d,
                generator=gen,
                device=dev,
                dtype=torch.bfloat16,
            )
            v_all = torch.randn(
                batch * total_kv,
                hkv,
                d,
                generator=gen,
                device=dev,
                dtype=torch.bfloat16,
            )

            # Split into prefix and extend slices per sequence.
            k_4d = k_all.view(batch, total_kv, hkv, d)
            v_4d = v_all.view(batch, total_kv, hkv, d)
            k_extend = k_4d[:, prefix:].reshape(batch * extend, hkv, d).contiguous()
            v_extend = v_4d[:, prefix:].reshape(batch * extend, hkv, d).contiguous()

            # Pool with padding slot 0; store full sequence (prefix + extend) in pool.
            pool_slots = batch * total_kv + 1
            unified_kv_indices = torch.arange(
                1, pool_slots, dtype=torch.int64, device=dev
            )
            unified_kv_indptr = (
                torch.arange(batch + 1, dtype=torch.int32, device=dev) * total_kv
            )
            qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=dev) * extend
            prefix_lens = torch.full((batch,), prefix, dtype=torch.int32, device=dev)

            prefix_kv_indptr = (
                torch.arange(batch + 1, dtype=torch.int32, device=dev) * prefix
            )
            if prefix > 0:
                prefix_kv_indices = (
                    unified_kv_indices.view(batch, total_kv)[:, :prefix]
                    .reshape(-1)
                    .contiguous()
                )
            else:
                prefix_kv_indices = torch.empty(0, dtype=torch.int64, device=dev)

            # BF16 pool
            k_buf_bf16 = torch.zeros(
                pool_slots, hkv, d, dtype=torch.bfloat16, device=dev
            )
            v_buf_bf16 = torch.zeros_like(k_buf_bf16)
            k_buf_bf16[1:] = k_all
            v_buf_bf16[1:] = v_all

            # INT8 pool via M2 write kernel
            k_buf_i8 = torch.zeros(pool_slots, hkv, d, dtype=torch.int8, device=dev)
            v_buf_i8 = torch.zeros_like(k_buf_i8)
            k_scale = torch.zeros(pool_slots, hkv, dtype=torch.float32, device=dev)
            v_scale = torch.zeros_like(k_scale)
            store_int8_per_token_head_kv(
                k=k_all,
                v=v_all,
                loc=unified_kv_indices,
                k_cache=k_buf_i8,
                v_cache=v_buf_i8,
                k_scale=k_scale,
                v_scale=v_scale,
            )

            # fp32 references:
            # 1) ref_bf16: exact BF16 K/V
            # 2) ref_chunk_bf16: dequantized INT8 prefix + BF16 chunk
            # 3) ref_chunk_q: dequantized INT8 prefix + dequantized INT8 chunk
            ref_bf16 = _chunked_reference_extend(
                q=q,
                k_all=k_all,
                v_all=v_all,
                batch=batch,
                prefix=prefix,
                extend=extend,
                num_q_heads=hq,
                num_kv_heads=hkv,
            )
            k_deq_all = (
                k_buf_i8[1:].float() * k_scale[1:, :, None]
            ).to(torch.bfloat16)
            v_deq_all = (
                v_buf_i8[1:].float() * v_scale[1:, :, None]
            ).to(torch.bfloat16)
            ref_chunk_q = _chunked_reference_extend(
                q=q,
                k_all=k_deq_all,
                v_all=v_deq_all,
                batch=batch,
                prefix=prefix,
                extend=extend,
                num_q_heads=hq,
                num_kv_heads=hkv,
            )
            k_mix = k_4d.clone()
            v_mix = v_4d.clone()
            if prefix > 0:
                k_mix[:, :prefix] = k_deq_all.view(batch, total_kv, hkv, d)[:, :prefix]
                v_mix[:, :prefix] = v_deq_all.view(batch, total_kv, hkv, d)[:, :prefix]
            ref_chunk_bf16 = _chunked_reference_extend(
                q=q,
                k_all=k_mix.reshape(batch * total_kv, hkv, d),
                v_all=v_mix.reshape(batch * total_kv, hkv, d),
                batch=batch,
                prefix=prefix,
                extend=extend,
                num_q_heads=hq,
                num_kv_heads=hkv,
            )
            del k_deq_all, v_deq_all, k_mix, v_mix
            torch.cuda.empty_cache()

            for impl in impls:
                out = torch.empty_like(q)

                if impl == "bf16-regular":
                    ref_exact = ref_bf16

                    def op():
                        extend_attention_fwd(
                            q,
                            k_extend,
                            v_extend,
                            out,
                            k_buf_bf16,
                            v_buf_bf16,
                            qo_indptr,
                            prefix_kv_indptr,
                            prefix_kv_indices,
                            None,
                            True,
                            None,
                            extend,
                            1.0,
                            1.0,
                            sm_scale=sm_scale,
                        )

                elif impl == "bf16-unified":
                    ref_exact = ref_bf16

                    def op():
                        extend_attention_fwd_unified(
                            q,
                            out,
                            k_buf_bf16,
                            v_buf_bf16,
                            1.0,
                            1.0,
                            qo_indptr,
                            unified_kv_indptr,
                            unified_kv_indices,
                            prefix_lens,
                            extend,
                            sm_scale=sm_scale,
                            is_causal=True,
                        )

                elif impl == "int8-m3b-gather":
                    ref_exact = ref_chunk_bf16

                    def op():
                        if prefix > 0:
                            k_pref = dequant_int8_per_token_head(
                                data=k_buf_i8,
                                scale=k_scale,
                                dtype=torch.bfloat16,
                                slots=prefix_kv_indices,
                            )
                            v_pref = dequant_int8_per_token_head(
                                data=v_buf_i8,
                                scale=v_scale,
                                dtype=torch.bfloat16,
                                slots=prefix_kv_indices,
                            )
                            idx = torch.arange(
                                prefix_kv_indices.numel(),
                                dtype=prefix_kv_indices.dtype,
                                device=dev,
                            )
                        else:
                            k_pref = k_buf_bf16[:1]
                            v_pref = v_buf_bf16[:1]
                            idx = prefix_kv_indices
                        extend_attention_fwd(
                            q,
                            k_extend,
                            v_extend,
                            out,
                            k_pref,
                            v_pref,
                            qo_indptr,
                            prefix_kv_indptr,
                            idx,
                            None,
                            True,
                            None,
                            extend,
                            1.0,
                            1.0,
                            sm_scale=sm_scale,
                        )

                elif impl == "int8-fused-regular":
                    ref_exact = ref_chunk_bf16

                    def op():
                        extend_attention_fwd_int8(
                            q_extend=q,
                            k_extend=k_extend,
                            v_extend=v_extend,
                            o_extend=out,
                            k_buffer=k_buf_i8,
                            v_buffer=v_buf_i8,
                            k_scale_buffer=k_scale,
                            v_scale_buffer=v_scale,
                            qo_indptr=qo_indptr,
                            kv_indptr=prefix_kv_indptr,
                            kv_indices=prefix_kv_indices,
                            custom_mask=None,
                            is_causal=True,
                            mask_indptr=None,
                            max_len_extend=extend,
                            sm_scale=sm_scale,
                        )

                elif impl == "int8-fused-unified":
                    ref_exact = ref_chunk_q

                    def op():
                        extend_attention_fwd_unified_int8(
                            q=q,
                            o=out,
                            k_buffer=k_buf_i8,
                            v_buffer=v_buf_i8,
                            k_scale_buffer=k_scale,
                            v_scale_buffer=v_scale,
                            qo_indptr=qo_indptr,
                            kv_indptr=unified_kv_indptr,
                            kv_indices=unified_kv_indices,
                            prefix_lens=prefix_lens,
                            max_len_extend=extend,
                            sm_scale=sm_scale,
                            is_causal=True,
                        )

                # Warmup (compile Triton kernels before measuring memory or time).
                op()
                torch.cuda.synchronize()
                cos_bf16, max_bf16 = _cosine_and_max_err(out, ref_bf16)
                cos_exact, max_exact = _cosine_and_max_err(out, ref_exact)

                # Measure peak temporary GPU memory allocated during one eager call.
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(dev)
                base_bytes = torch.cuda.memory_allocated(dev)
                op()
                torch.cuda.synchronize()
                peak_extra_mb = max(
                    0.0, (torch.cuda.max_memory_allocated(dev) - base_bytes) / 1e6
                )

                # Cold-L2 timing via triton.testing.do_bench (flushes 256 MB L2).
                use_graph = False
                fn_to_bench = op
                if impl != "int8-m3b-gather":
                    try:
                        op()
                        torch.cuda.synchronize()
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            op()
                        torch.cuda.synchronize()
                        fn_to_bench = g.replay
                        use_graph = True
                    except Exception:
                        fn_to_bench = op

                p20, med, p80 = triton.testing.do_bench(
                    fn_to_bench,
                    warmup=args.warmup_ms,
                    rep=args.rep_ms,
                    quantiles=[0.2, 0.5, 0.8],
                )
                row = {
                    "shape": shape_name,
                    "batch": batch,
                    "prefix": prefix,
                    "extend": extend,
                    "impl": impl,
                    "median_ms": round(med, 4),
                    "p20_ms": round(p20, 4),
                    "p80_ms": round(p80, 4),
                    "temp_mb": round(peak_extra_mb, 2),
                    "cos_vs_bf16": round(cos_bf16, 7),
                    "max_abs_vs_bf16": round(max_bf16, 5),
                    "cos_vs_exact": round(cos_exact, 7),
                    "max_abs_vs_exact": round(max_exact, 5),
                    "cuda_graph": use_graph,
                }
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
                print(json.dumps(row), flush=True)

            del (
                q,
                k_all,
                v_all,
                k_extend,
                v_extend,
                k_buf_bf16,
                v_buf_bf16,
                k_buf_i8,
                v_buf_i8,
                k_scale,
                v_scale,
                ref_bf16,
                ref_chunk_bf16,
                ref_chunk_q,
            )
            torch.cuda.empty_cache()

    _write_report(args.out)


# ----------------------------------------------------------------------------
# Subcommand: kl (offline server-level consistency)
# ----------------------------------------------------------------------------


def _load_offline_prompts(*, model_path, corpus_glob, prompt_len, count):
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    tokenizer = get_tokenizer(model_path, trust_remote_code=False)
    files = sorted(REPO_ROOT.glob(corpus_glob))
    text = "\n\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in files)
    ids = tokenizer.encode(text, add_special_tokens=False)
    need = prompt_len * count
    if len(ids) < need:
        reps = (need + len(ids) - 1) // len(ids)
        ids = (ids * reps)[:need]
    stride = len(ids) // count
    prompts = [ids[i * stride : i * stride + prompt_len] for i in range(count)]
    comma_ids = tokenizer.encode(",", add_special_tokens=False)
    return prompts, comma_ids, {"files": len(files), "corpus_tokens": len(ids)}


def _http_post_json(url, payload, timeout=1800):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _flush_cache(base_url, timeout_s=30):
    req = urllib.request.Request(
        f"{base_url}/flush_cache?timeout={timeout_s}", method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout_s + 10) as resp:
        resp.read()


def _generate(
    base_url,
    input_ids,
    *,
    max_new_tokens,
    return_logprob=False,
    logprob_start_len=-1,
):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    if return_logprob:
        payload.update(
            {
                "return_logprob": True,
                "return_text_in_logprobs": False,
                "logprob_start_len": logprob_start_len,
            }
        )
    return _http_post_json(base_url + "/generate", payload)


def _kl_stats(a_lists, b_lists):
    """Schulman k3 KL estimator (same as kl_test_utils.compare_kl_divergence)."""
    import numpy as np

    per_sample_kl = []
    all_kl = []
    all_abs = []
    for a, b in zip(a_lists, b_lists):
        a_arr = np.asarray(a, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)
        assert a_arr.shape == b_arr.shape and a_arr.size > 0, (a_arr.shape, b_arr.shape)
        logr = a_arr - b_arr
        kl = (np.exp(logr) - 1.0) - logr
        per_sample_kl.append(float(np.mean(kl)))
        all_kl.append(kl)
        all_abs.append(np.abs(logr))
    flat_kl = np.concatenate(all_kl)
    flat_abs = np.concatenate(all_abs)
    return {
        "samples": len(per_sample_kl),
        "tokens": int(flat_kl.size),
        "avg_kl_div": float(np.mean(per_sample_kl)),
        "p99_kl": float(np.quantile(flat_kl, 0.99)),
        "max_kl": float(np.max(flat_kl)),
        "max_abs_logp_diff": float(np.max(flat_abs)),
        "frac_kl_gt_1e4": float(np.mean(flat_kl > 1e-4)),
        "frac_kl_gt_1e2": float(np.mean(flat_kl > 1e-2)),
    }


def _extract_output_lps(result):
    return [x[0] for x in result["meta_info"]["output_token_logprobs"]]


def _get_prefill_lps_for_outputs(base_url, full_ids, output_lps):
    _flush_cache(base_url)
    t0 = time.time()
    results = _generate(
        base_url,
        full_ids,
        max_new_tokens=0,
        return_logprob=True,
        logprob_start_len=0,
    )
    elapsed_s = time.time() - t0
    input_lps = []
    for i, res in enumerate(results):
        lp = [x[0] for x in res["meta_info"]["input_token_logprobs"]]
        input_lps.append(lp[-len(output_lps[i]) :])
    return input_lps, elapsed_s


def _run_kl_checks(*, base_url, prompts, comma_ids, max_new_tokens):
    rows = []

    # 1. match: fresh decode vs fresh prefill
    _flush_cache(base_url)
    t0 = time.time()
    gen_res = _generate(
        base_url, prompts, max_new_tokens=max_new_tokens, return_logprob=True
    )
    gen_s = time.time() - t0
    full_ids = [prompts[i] + gen_res[i]["output_ids"] for i in range(len(prompts))]
    out_lps = [_extract_output_lps(r) for r in gen_res]
    in_lps, pref_s = _get_prefill_lps_for_outputs(base_url, full_ids, out_lps)
    rows.append(
        {
            "check": "match",
            "hit_samples": len(prompts),
            "total_samples": len(prompts),
            "gen_s": round(gen_s, 2),
            "prefill_s": round(pref_s, 2),
            **_kl_stats(in_lps, out_lps),
        }
    )

    # 2. prefill_cache_hit: cached prompt decode vs fresh prefill
    _flush_cache(base_url)
    _generate(base_url, prompts, max_new_tokens=0)
    t0 = time.time()
    hit_res = _generate(
        base_url, prompts, max_new_tokens=max_new_tokens, return_logprob=True
    )
    gen_s = time.time() - t0
    full_ids, out_lps = [], []
    for i, r in enumerate(hit_res):
        if r["meta_info"].get("cached_tokens", 0) > 0:
            full_ids.append(prompts[i] + r["output_ids"])
            out_lps.append(_extract_output_lps(r))
    if full_ids:
        in_lps, pref_s = _get_prefill_lps_for_outputs(base_url, full_ids, out_lps)
        rows.append(
            {
                "check": "prefill_cache_hit",
                "hit_samples": len(full_ids),
                "total_samples": len(prompts),
                "gen_s": round(gen_s, 2),
                "prefill_s": round(pref_s, 2),
                **_kl_stats(in_lps, out_lps),
            }
        )
    else:
        rows.append(
            {
                "check": "prefill_cache_hit",
                "hit_samples": 0,
                "total_samples": len(prompts),
                "error": "no prefill cache hits",
            }
        )

    # 3. decode_cache_hit: turn-1 prefill+decode -> turn-2 hits decoded tokens
    _flush_cache(base_url)
    t1_res = _generate(
        base_url, prompts, max_new_tokens=max_new_tokens, return_logprob=True
    )
    t2_prompts = [
        prompts[i] + t1_res[i]["output_ids"] + comma_ids for i in range(len(prompts))
    ]
    t0 = time.time()
    t2_res = _generate(
        base_url, t2_prompts, max_new_tokens=max_new_tokens, return_logprob=True
    )
    gen_s = time.time() - t0
    full_ids, out_lps = [], []
    for i, r in enumerate(t2_res):
        if r["meta_info"].get("cached_tokens", 0) > len(prompts[i]) + 1:
            full_ids.append(t2_prompts[i] + r["output_ids"])
            out_lps.append(_extract_output_lps(r))
    if full_ids:
        in_lps, pref_s = _get_prefill_lps_for_outputs(base_url, full_ids, out_lps)
        rows.append(
            {
                "check": "decode_cache_hit",
                "hit_samples": len(full_ids),
                "total_samples": len(prompts),
                "gen_s": round(gen_s, 2),
                "prefill_s": round(pref_s, 2),
                **_kl_stats(in_lps, out_lps),
            }
        )
    else:
        rows.append(
            {
                "check": "decode_cache_hit",
                "hit_samples": 0,
                "total_samples": len(prompts),
                "error": "no decode cache hits",
            }
        )

    # 4. prefill_cached_vs_fresh: directly compare suffix prefill logprobs with
    #    prefix cached in KV vs fresh prefill of the whole prompt. Setting
    #    logprob_start_len=split keeps the radix prefix hit on [0, split) intact.
    split = len(prompts[0]) // 2
    prefixes = [p[:split] for p in prompts]

    _flush_cache(base_url)
    t0 = time.time()
    fresh_res = _generate(
        base_url,
        prompts,
        max_new_tokens=0,
        return_logprob=True,
        logprob_start_len=split,
    )
    fresh_s = time.time() - t0

    _flush_cache(base_url)
    _generate(base_url, prefixes, max_new_tokens=0)
    t0 = time.time()
    cached_res = _generate(
        base_url,
        prompts,
        max_new_tokens=0,
        return_logprob=True,
        logprob_start_len=split,
    )
    cached_s = time.time() - t0

    fresh_lps, cached_lps = [], []
    for f_r, c_r in zip(fresh_res, cached_res):
        if c_r["meta_info"].get("cached_tokens", 0) == 0:
            continue
        f_seq = [
            x[0] for x in f_r["meta_info"]["input_token_logprobs"] if x[0] is not None
        ]
        c_seq = [
            x[0] for x in c_r["meta_info"]["input_token_logprobs"] if x[0] is not None
        ]
        n = min(len(f_seq), len(c_seq))
        fresh_lps.append(f_seq[-n:])
        cached_lps.append(c_seq[-n:])
    if fresh_lps:
        rows.append(
            {
                "check": "prefill_cached_vs_fresh",
                "hit_samples": len(fresh_lps),
                "total_samples": len(prompts),
                "gen_s": round(cached_s, 2),
                "prefill_s": round(fresh_s, 2),
                **_kl_stats(fresh_lps, cached_lps),
            }
        )
    else:
        rows.append(
            {
                "check": "prefill_cached_vs_fresh",
                "hit_samples": 0,
                "total_samples": len(prompts),
                "error": "no prefix cache hits",
            }
        )

    return rows


def _run_kl_eval(args):
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "servers").mkdir(exist_ok=True)
    (args.out / "telemetry").mkdir(exist_ok=True)
    jsonl_path = args.out / "kl_results.jsonl"

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in KL_CONFIGS:
            raise SystemExit(f"unknown config {c!r}; choose from {list(KL_CONFIGS)}")

    prompts, comma_ids, corpus_info = _load_offline_prompts(
        model_path=args.model,
        corpus_glob=args.corpus_glob,
        prompt_len=args.prompt_len,
        count=args.samples,
    )
    print(
        f"Loaded {len(prompts)} prompts of {args.prompt_len} tokens from "
        f"{corpus_info['files']} files ({corpus_info['corpus_tokens']} tokens).",
        flush=True,
    )

    gpu_uuid = m0_bench._gpu_query(args.gpu, "uuid")[0]

    with open(jsonl_path, "w") as out_f:
        for name in configs:
            extra_args, extra_env = KL_CONFIGS[name]
            m0_bench._wait_gpu_free(
                gpu=args.gpu, gpu_uuid=gpu_uuid, timeout_s=args.gpu_wait_timeout
            )
            port = m0_bench._free_port()
            cmd = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                args.model,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--random-seed",
                "42",
                *extra_args,
            ]
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(args.gpu),
                CUDA_DEVICE_ORDER="PCI_BUS_ID",
                **extra_env,
            )
            telemetry = m0_bench.Telemetry(
                gpu=args.gpu, path=args.out / "telemetry" / f"{name}.csv"
            )
            srv = m0_bench.Server(
                cmd=cmd, env=env, log_path=args.out / "servers" / f"{name}.log"
            )
            t0 = time.time()
            try:
                ready = srv.wait_ready(port=port, timeout_s=args.ready_timeout)
                info = m0_bench._parse_server_log(
                    engine="sglang", text=srv.log_text()
                )
                srv_row = {
                    "kind": "server",
                    "config": name,
                    "ready": ready,
                    "ready_s": round(time.time() - t0, 1),
                    "extra_args": extra_args,
                    "extra_env": extra_env,
                    **info,
                    "error_tail": None if ready else srv.log_text()[-3000:],
                }
                out_f.write(json.dumps(srv_row) + "\n")
                out_f.flush()
                print(json.dumps(srv_row), flush=True)
                if not ready:
                    continue

                base_url = f"http://127.0.0.1:{port}"
                try:
                    check_rows = _run_kl_checks(
                        base_url=base_url,
                        prompts=prompts,
                        comma_ids=comma_ids,
                        max_new_tokens=args.max_new_tokens,
                    )
                    for r in check_rows:
                        row = {"kind": "kl", "config": name, **r}
                        out_f.write(json.dumps(row) + "\n")
                        out_f.flush()
                        print(json.dumps(row), flush=True)
                except Exception:
                    err_row = {
                        "kind": "kl",
                        "config": name,
                        "check": "exception",
                        "error": traceback.format_exc()[-2000:],
                    }
                    out_f.write(json.dumps(err_row) + "\n")
                    out_f.flush()
                    print(json.dumps(err_row), flush=True)
            finally:
                srv.stop()
                telemetry.stop()

    _write_report(args.out)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------


def _fmt_sci(val):
    if val is None:
        return "-"
    if val == 0.0:
        return "0"
    if abs(val) < 1e-3 or abs(val) >= 1e3:
        return f"{val:.3e}"
    return f"{val:.6f}"


def _write_report(out_dir: Path):
    lines = ["# M3c Extend Kernel & KL Consistency Summary\n"]

    k_path = out_dir / "kernel_results.jsonl"
    if k_path.exists():
        rows = [json.loads(x) for x in k_path.read_text().splitlines() if x.strip()]
        by_shape = {}
        for r in rows:
            by_shape.setdefault(r["shape"], []).append(r)

        lines.append("## Kernel Microbenchmark (cold L2, 32 Q / 8 KV / D=128)\n")
        lines.append(
            "| Shape (B × prefix + extend) | Impl | Median ms | vs BF16 | vs M3b | "
            "Temp MB | Cos vs BF16 | Max abs vs exact |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for shape, srows in by_shape.items():
            ref_bf16_ms = next(
                (r["median_ms"] for r in srows if r["impl"] == "bf16-regular"), None
            )
            ref_m3b_ms = next(
                (r["median_ms"] for r in srows if r["impl"] == "int8-m3b-gather"), None
            )
            for r in srows:
                sp_bf16 = (
                    f"{ref_bf16_ms / r['median_ms']:.2f}×"
                    if ref_bf16_ms and r["median_ms"] > 0
                    else "-"
                )
                sp_m3b = (
                    f"{ref_m3b_ms / r['median_ms']:.2f}×"
                    if ref_m3b_ms and r["median_ms"] > 0
                    else "-"
                )
                shape_lbl = f"{shape} ({r['batch']}×{r['prefix']}+{r['extend']})"
                lines.append(
                    f"| {shape_lbl} | `{r['impl']}` | {r['median_ms']:.3f} | "
                    f"{sp_bf16} | {sp_m3b} | {r['temp_mb']:.1f} | "
                    f"{r['cos_vs_bf16']:.6f} | {r['max_abs_vs_exact']:.5f} |"
                )
        lines.append("")

    kl_path = out_dir / "kl_results.jsonl"
    if kl_path.exists():
        rows = [json.loads(x) for x in kl_path.read_text().splitlines() if x.strip()]
        srv_rows = [r for r in rows if r.get("kind") == "server"]
        kl_rows = [r for r in rows if r.get("kind") == "kl"]

        if srv_rows:
            lines.append("## Server Startup & KV Capacity\n")
            lines.append("| Config | Ready | Ready s | KV Capacity (tokens) | mem_frac |")
            lines.append("|---|---:|---:|---:|---:|")
            for r in srv_rows:
                lines.append(
                    f"| `{r['config']}` | {r['ready']} | {r.get('ready_s', '-')} | "
                    f"{r.get('capacity_tokens', '-')} | {r.get('mem_fraction_static', '-')} |"
                )
            lines.append("")

        if kl_rows:
            lines.append("## Offline KL Consistency Checks\n")
            lines.append(
                "| Config | Check | Hits | Avg KL | p99 KL | Max KL | "
                "Max |Δlogp| | >1e-4 | >1e-2 |"
            )
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
            for r in kl_rows:
                if "error" in r:
                    lines.append(
                        f"| `{r['config']}` | `{r['check']}` | "
                        f"{r.get('hit_samples', 0)}/{r.get('total_samples', 0)} | "
                        f"ERR: {r['error'][:80]} | - | - | - | - | - |"
                    )
                else:
                    lines.append(
                        f"| `{r['config']}` | `{r['check']}` | "
                        f"{r['hit_samples']}/{r['total_samples']} | "
                        f"{_fmt_sci(r['avg_kl_div'])} | {_fmt_sci(r['p99_kl'])} | "
                        f"{_fmt_sci(r['max_kl'])} | {_fmt_sci(r['max_abs_logp_diff'])} | "
                        f"{100 * r['frac_kl_gt_1e4']:.2f}% | "
                        f"{100 * r['frac_kl_gt_1e2']:.2f}% |"
                    )
            lines.append("")

    summary = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(summary)
    print("\n" + summary, flush=True)


def main():
    args = _parse_args()
    if args.cmd == "kernel":
        _run_kernel_bench(args)
    elif args.cmd == "kl":
        _run_kl_eval(args)
    elif args.cmd == "report":
        _write_report(args.out)


if __name__ == "__main__":
    main()
