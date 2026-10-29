"""M1: offline KV-cache quantization error (pure PyTorch, no engine changes).

Runs a Hugging Face model on windows of real text. For each quantizer in
``kvq_ref.SCHEMES``, the prefix K/V in the cache are replaced by their
quantize-dequantize values and the continuation is scored in one forward
chunk under two semantics:

  chunk_bf16  the continuation's own K/V stay BF16 (SGLang's default extend path)
  chunk_q     the continuation's K/V are quantized before attention reads them
              (vLLM, SGLang's deterministic unified kernel, and every decode step)

Per scheme it reports KL(ref || quantized) of the next-token distribution,
top-1 agreement and the NLL change on the real continuation tokens, plus
per-layer K/V reconstruction error. It also reports K/V channel-outlier ratios
before and after the RHT, and an optional one-layer-at-a-time sensitivity scan.

Example:
  python benchmark/kv_lowbit/m1_offline_error.py --model "$QWEN3_8B" \
      --label qwen3-8b --out "$KVQ_WORK/outputs/m1/$TS"
"""

import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvq_ref  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="HF repo id or local directory")
    p.add_argument("--label", required=True, help="Short name for output files")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--prefix-lens", default="4096,16384")
    p.add_argument("--cont-len", type=int, default=128)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--schemes", default=",".join(kvq_ref.SCHEMES))
    p.add_argument(
        "--per-layer-scheme",
        default="int4_pth_rht",
        help="Scheme for the one-layer-at-a-time scan at the shortest prefix; "
        "empty string skips the scan",
    )
    p.add_argument("--per-layer-samples", type=int, default=2)
    p.add_argument(
        "--corpus-glob",
        default="docs/docs/**/*.mdx",
        help="Text files relative to the repo root, read in sorted order",
    )
    return p.parse_args()


def _load_windows(*, tokenizer, corpus_glob, window_len, count):
    files = sorted(REPO_ROOT.glob(corpus_glob))
    text = "\n\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in files)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) < window_len * count:
        raise SystemExit(
            f"Corpus has {len(ids)} tokens, need {window_len * count}; "
            "lower --samples or pass a larger --corpus-glob"
        )
    # Spread the windows over the corpus instead of taking adjacent files.
    stride = len(ids) // count
    windows = [
        torch.tensor(ids[i * stride : i * stride + window_len]) for i in range(count)
    ]
    return windows, {"files": len(files), "tokens": len(ids)}


def _kv_pairs(cache):
    # transformers >= 4.56 cache layout: one layer object per decoder layer,
    # keys/values shaped [batch, kv_heads, tokens, head_dim], post-RoPE.
    return [(layer.keys, layer.values) for layer in cache.layers]


def _quantize_cache_(cache, *, qfn, layers=None):
    for i, (k, v) in enumerate(_kv_pairs(cache)):
        if layers is None or i in layers:
            k.copy_(qfn(k))
            v.copy_(qfn(v))


def _quantize_on_write(cache, *, qfn):
    update = cache.update

    def quantized_update(key_states, value_states, *args, **kwargs):
        return update(qfn(key_states), qfn(value_states), *args, **kwargs)

    cache.update = quantized_update


@torch.no_grad()
def _prefill(model, prefix):
    out = model(input_ids=prefix[None], use_cache=True, logits_to_keep=1)
    return out.past_key_values


@torch.no_grad()
def _score(model, cache, cont):
    """Log-probs [cont_len, vocab]; appends the continuation to ``cache``."""
    out = model(input_ids=cont[None], past_key_values=cache, use_cache=True)
    return out.logits[0].float().log_softmax(-1)


def _compare(*, ref_logp, q_logp, cont):
    kl = (ref_logp.exp() * (ref_logp - q_logp)).sum(-1)
    top1 = (ref_logp.argmax(-1) == q_logp.argmax(-1)).float()
    target = cont[1:, None]
    nll_ref = -ref_logp[:-1].gather(-1, target).squeeze(-1)
    nll_q = -q_logp[:-1].gather(-1, target).squeeze(-1)
    return {
        "kl": kl.mean().item(),
        "kl_max": kl.max().item(),
        "top1": top1.mean().item(),
        "nll_ref": nll_ref.mean().item(),
        "dnll": (nll_q - nll_ref).mean().item(),
    }


def _rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _kv_errors(ref_cache, q_cache):
    return [
        (_rel_err(k1, k0), _rel_err(v1, v0))
        for (k0, v0), (k1, v1) in zip(_kv_pairs(ref_cache), _kv_pairs(q_cache))
    ]


def _outlier_stats(cache):
    stats = {"k": [], "k_rht": [], "v": []}
    for k, v in _kv_pairs(cache):
        stats["k"].append(kvq_ref.channel_outlier_ratio(k[0]).mean().item())
        stats["k_rht"].append(
            kvq_ref.channel_outlier_ratio(kvq_ref.rht(k[0])).mean().item()
        )
        stats["v"].append(kvq_ref.channel_outlier_ratio(v[0]).mean().item())
    return stats


def _mean(xs):
    return sum(xs) / len(xs)


def _aggregate(per_sample):
    """Average per-sample metric dicts and per-layer (k_err, v_err) lists."""
    out = {}
    for mode in ("chunk_bf16", "chunk_q"):
        rows = per_sample[mode]
        out[mode] = {key: _mean([r[key] for r in rows]) for key in rows[0]}
    layers = list(zip(*per_sample["kv_err"]))
    out["k_err_per_layer"] = [_mean([s[0] for s in layer]) for layer in layers]
    out["v_err_per_layer"] = [_mean([s[1] for s in layer]) for layer in layers]
    out["k_err"] = _mean(out["k_err_per_layer"])
    out["v_err"] = _mean(out["v_err_per_layer"])
    return out


def _run_prefix_len(*, model, windows, prefix_len, cont_len, schemes):
    per_sample = {s: {"chunk_bf16": [], "chunk_q": [], "kv_err": []} for s in schemes}
    outliers = None
    for idx, window in enumerate(windows):
        t0 = time.time()
        prefix = window[:prefix_len].cuda()
        cont = window[prefix_len : prefix_len + cont_len].cuda()
        ref_cache = _prefill(model, prefix)
        if outliers is None:
            outliers = _outlier_stats(ref_cache)
        ref_logp = _score(model, copy.deepcopy(ref_cache), cont)
        for name in schemes:
            qfn = kvq_ref.SCHEMES[name]
            q_cache = copy.deepcopy(ref_cache)
            _quantize_cache_(q_cache, qfn=qfn)
            per_sample[name]["kv_err"].append(_kv_errors(ref_cache, q_cache))
            q_logp = _score(model, copy.deepcopy(q_cache), cont)
            per_sample[name]["chunk_bf16"].append(
                _compare(ref_logp=ref_logp, q_logp=q_logp, cont=cont)
            )
            _quantize_on_write(q_cache, qfn=qfn)
            q_logp = _score(model, q_cache, cont)
            per_sample[name]["chunk_q"].append(
                _compare(ref_logp=ref_logp, q_logp=q_logp, cont=cont)
            )
        del ref_cache
        print(
            f"  prefix {prefix_len} sample {idx}: {time.time() - t0:.1f}s", flush=True
        )
    return {s: _aggregate(v) for s, v in per_sample.items()}, outliers


def _per_layer_scan(*, model, windows, prefix_len, cont_len, qfn):
    num_layers = model.config.num_hidden_layers
    kl = [0.0] * num_layers
    for window in windows:
        prefix = window[:prefix_len].cuda()
        cont = window[prefix_len : prefix_len + cont_len].cuda()
        ref_cache = _prefill(model, prefix)
        ref_logp = _score(model, copy.deepcopy(ref_cache), cont)
        for layer in range(num_layers):
            q_cache = copy.deepcopy(ref_cache)
            _quantize_cache_(q_cache, qfn=qfn, layers={layer})
            q_logp = _score(model, q_cache, cont)
            kl[layer] += _compare(ref_logp=ref_logp, q_logp=q_logp, cont=cont)["kl"]
        del ref_cache
    return [x / len(windows) for x in kl]


def _env_info(model_name):
    import transformers

    return {
        "model": model_name,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
    }


def _fmt(x, digits=4):
    return f"{x:.{digits}g}"


def _render_markdown(result):
    lines = [f"# M1 offline KV quantization error: {result['label']}", ""]
    env = result["env"]
    lines.append(
        f"- model `{env['model']}` | GPU {env['gpu']} | torch {env['torch']} "
        f"(CUDA {env['cuda']}) | transformers {env['transformers']}"
    )
    lines.append(
        f"- corpus: {result['corpus']['files']} files, {result['corpus']['tokens']} "
        f"tokens | {result['samples']} windows per prefix length | "
        f"continuation {result['cont_len']} tokens"
    )
    lines.append("- KL and dNLL are means over continuation positions and windows.")
    for prefix_len, per_scheme in result["by_prefix"].items():
        lines += ["", f"## Prefix {prefix_len} tokens", ""]
        lines.append(
            "| scheme | KL chunk_bf16 | KL chunk_q | top-1 chunk_q | dNLL chunk_q "
            "| K rel err | V rel err |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, r in per_scheme.items():
            lines.append(
                f"| {name} | {_fmt(r['chunk_bf16']['kl'])} | {_fmt(r['chunk_q']['kl'])} "
                f"| {r['chunk_q']['top1']:.4f} | {_fmt(r['chunk_q']['dnll'])} "
                f"| {_fmt(r['k_err'])} | {_fmt(r['v_err'])} |"
            )
    lines += ["", "## Channel-outlier ratio (max/median of per-channel absmax)", ""]
    lines.append("| prefix | K mean | K max (layer) | K after RHT mean | V mean |")
    lines.append("|---:|---:|---:|---:|---:|")
    for prefix_len, o in result["outliers"].items():
        kmax = max(o["k"])
        lines.append(
            f"| {prefix_len} | {_mean(o['k']):.2f} | {kmax:.2f} ({o['k'].index(kmax)}) "
            f"| {_mean(o['k_rht']):.2f} | {_mean(o['v']):.2f} |"
        )
    scan = result.get("per_layer_scan")
    if scan:
        ranked = sorted(enumerate(scan["kl"]), key=lambda t: -t[1])[:8]
        lines += [
            "",
            f"## One-layer scan ({scan['scheme']}, prefix {scan['prefix_len']}, "
            f"{scan['samples']} windows, chunk_bf16)",
            "",
            f"Mean KL over layers {_fmt(_mean(scan['kl']))}; top layers: "
            + ", ".join(f"L{i} {_fmt(v)}" for i, v in ranked),
        ]
    return "\n".join(lines) + "\n"


def main():
    args = _parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prefix_lens = [int(x) for x in args.prefix_lens.split(",")]
    schemes = [s for s in args.schemes.split(",") if s]
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    ).eval()
    windows, corpus = _load_windows(
        tokenizer=tokenizer,
        corpus_glob=args.corpus_glob,
        window_len=max(prefix_lens) + args.cont_len,
        count=args.samples,
    )
    result = {
        "label": args.label,
        "env": _env_info(args.model),
        "corpus": corpus,
        "samples": args.samples,
        "cont_len": args.cont_len,
        "by_prefix": {},
        "outliers": {},
    }
    for prefix_len in prefix_lens:
        print(f"prefix {prefix_len}", flush=True)
        by_scheme, outliers = _run_prefix_len(
            model=model,
            windows=windows,
            prefix_len=prefix_len,
            cont_len=args.cont_len,
            schemes=schemes,
        )
        result["by_prefix"][str(prefix_len)] = by_scheme
        result["outliers"][str(prefix_len)] = outliers
    if args.per_layer_scheme:
        print(f"per-layer scan {args.per_layer_scheme}", flush=True)
        result["per_layer_scan"] = {
            "scheme": args.per_layer_scheme,
            "prefix_len": min(prefix_lens),
            "samples": args.per_layer_samples,
            "kl": _per_layer_scan(
                model=model,
                windows=windows[: args.per_layer_samples],
                prefix_len=min(prefix_lens),
                cont_len=args.cont_len,
                qfn=kvq_ref.SCHEMES[args.per_layer_scheme],
            ),
        }
    (args.out / f"m1_{args.label}.json").write_text(json.dumps(result, indent=1))
    (args.out / f"m1_{args.label}.md").write_text(_render_markdown(result))
    print(_render_markdown(result))


if __name__ == "__main__":
    main()
