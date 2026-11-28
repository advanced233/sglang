"""M4/M5 offline long-context retrieval eval (16k / 32k S-NIAH, MK-NIAH, MV-NIAH).

Evaluates long-context KV cache quantization fidelity completely offline
(HF_HUB_OFFLINE=1 safe) using real natural-language technical documentation from
the repository (`docs/docs/**/*.mdx` / `*.md`) as the haystack, quantized to exact
token lengths with the target model's tokenizer.

Three RULER-style retrieval tasks per context length (deterministic greedy decoding,
Qwen3 non-thinking mode `enable_thinking=false`):
  - s_niah   Single 7-digit passkey inserted across 5 depths (10%, 30%, 50%, 70%, 90%).
  - mk_niah  4 distinct tagged 7-digit codes across the document; query 1 target tag.
  - mv_niah  4 distinct 5-digit values under the same vault tag across 4 depths; list all 4.

Usage:
  python benchmark/kv_lowbit/m4_longctx_eval.py --model "$QWEN3_8B" --gpu 0 \\
      --configs bf16-triton,int4-triton --lengths 16384,32768 \\
      --out "$KVQ_WORK/outputs/m4/$TS/longctx"
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m0_bench  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIGS: dict[str, tuple[list[str], dict[str, str]]] = {
    "bf16-triton": (
        [
            "--attention-backend",
            "triton",
            "--sampling-backend",
            "pytorch",
        ],
        {},
    ),
    "int8-triton": (
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
    "int4-triton": (
        [
            "--attention-backend",
            "triton",
            "--kv-cache-dtype",
            "int4_per_token_head",
            "--sampling-backend",
            "pytorch",
        ],
        {},
    ),
    "int4-unified": (
        [
            "--attention-backend",
            "triton",
            "--kv-cache-dtype",
            "int4_per_token_head",
            "--sampling-backend",
            "pytorch",
        ],
        {"SGLANG_ENABLE_TRITON_INT4_UNIFIED_EXTEND": "1"},
    ),
}

S_NIAH_DEPTHS = (0.10, 0.30, 0.50, 0.70, 0.90)
MK_DEPTHS = (0.15, 0.38, 0.62, 0.85)
MV_DEPTHS = (0.12, 0.37, 0.63, 0.88)
TAG_PREFIXES = ("ALPHA", "DELTA", "OMEGA", "SIGMA", "ORION", "NOVA", "ATLAS", "LYRA")


def _load_corpus_tokens(model_path: str) -> tuple[object, list[int], int]:
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    tokenizer = get_tokenizer(model_path, trust_remote_code=False)
    docs_root = REPO_ROOT / "docs" / "docs"
    files = sorted(list(docs_root.rglob("*.mdx")) + list(docs_root.rglob("*.md")))
    if not files:
        raise RuntimeError(f"No markdown files found under {docs_root}")

    parts = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) >= 200:
            parts.append(text)
    corpus = "\n\n---\n\n".join(parts)
    tokens = tokenizer.encode(corpus, add_special_tokens=False)
    if len(tokens) < 65536:
        raise RuntimeError(f"Offline corpus too short: {len(tokens)} tokens")
    return tokenizer, tokens, len(files)


def _slice_haystack(tokens: list[int], start: int, length: int) -> list[int]:
    n = len(tokens)
    out = []
    pos = start % n
    rem = length
    while rem > 0:
        take = min(rem, n - pos)
        out.extend(tokens[pos : pos + take])
        pos = (pos + take) % n
        rem -= take
    return out


def _insert_needles(
    tokenizer, haystack_tokens: list[int], placements: list[tuple[float, str]]
) -> str:
    """Insert needle text strings at fractional token depths in ascending order."""
    ordered = sorted(placements, key=lambda x: x[0])
    total = len(haystack_tokens)
    pieces = []
    prev = 0
    for depth, needle in ordered:
        idx = max(prev, min(total, int(round(total * depth))))
        if idx > prev:
            pieces.append(
                tokenizer.decode(haystack_tokens[prev:idx], skip_special_tokens=True)
            )
        pieces.append(needle)
        prev = idx
    if prev < total:
        pieces.append(
            tokenizer.decode(haystack_tokens[prev:], skip_special_tokens=True)
        )
    return "".join(pieces)


def _build_tasks(
    *,
    tokenizer,
    corpus_tokens: list[int],
    lengths: list[int],
    s_per_depth: int,
    mk_count: int,
    mv_count: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    tasks = []
    # Stride offsets across the ~940k-token corpus so every sample uses a distinct
    # document slice (no accidental prefix-cache sharing between tasks).
    stride = 17911
    sample_idx = 0

    for target_len in lengths:
        # Reserve ~256 tokens for system/chat template, needles, and question.
        haystack_len = max(512, target_len - 256)

        # 1. S-NIAH: single 7-digit passkey at 5 depths.
        for depth in S_NIAH_DEPTHS:
            for rep in range(s_per_depth):
                code = str(rng.randint(1000000, 9999999))
                needle = (
                    f"\n\n[ARCHIVE RECORD] The special magic passkey for "
                    f"verification is {code}.\n\n"
                )
                hs = _slice_haystack(corpus_tokens, sample_idx * stride, haystack_len)
                doc = _insert_needles(tokenizer, hs, [(depth, needle)])
                prompt = (
                    "Read the following reference document carefully and answer the "
                    "question at the end.\n\n"
                    f"{doc}\n\n"
                    "Question: What is the special magic passkey for verification "
                    "mentioned in the document above? Reply with the exact 7-digit "
                    "number only."
                )
                tasks.append(
                    {
                        "id": f"L{target_len}_sniah_d{int(depth * 100)}_{rep}",
                        "task": "s_niah",
                        "target_len": target_len,
                        "depth": depth,
                        "prompt": prompt,
                        "expected": [code],
                        "distractors": [],
                    }
                )
                sample_idx += 1

        # 2. MK-NIAH: 4 distinct tagged keys across 4 depths; query 1 target key.
        for i in range(mk_count):
            tag_num = 10 + (i * 7) % 89
            tags = [f"{TAG_PREFIXES[k]}-{tag_num + k}" for k in range(4)]
            codes = []
            while len(codes) < 4:
                c = str(rng.randint(1000000, 9999999))
                if c not in codes:
                    codes.append(c)
            placements = []
            for d_pos, tag, code in zip(MK_DEPTHS, tags, codes):
                placements.append(
                    (
                        d_pos,
                        f"\n\n[REGISTRY ENTRY] The secret code assigned to "
                        f"identifier {tag} is {code}.\n\n",
                    )
                )
            query_idx = i % 4
            target_tag = tags[query_idx]
            target_code = codes[query_idx]
            distractors = [c for j, c in enumerate(codes) if j != query_idx]
            hs = _slice_haystack(corpus_tokens, sample_idx * stride, haystack_len)
            doc = _insert_needles(tokenizer, hs, placements)
            prompt = (
                "Read the following reference document carefully and answer the "
                "question at the end.\n\n"
                f"{doc}\n\n"
                f"Question: What is the secret code assigned to identifier "
                f"{target_tag} in the document above? Reply with the exact 7-digit "
                f"number only."
            )
            tasks.append(
                {
                    "id": f"L{target_len}_mkniah_{i}",
                    "task": "mk_niah",
                    "target_len": target_len,
                    "depth": MK_DEPTHS[query_idx],
                    "prompt": prompt,
                    "expected": [target_code],
                    "distractors": distractors,
                }
            )
            sample_idx += 1

        # 3. MV-NIAH: 4 values for one vault key across 4 depths; retrieve all 4.
        for i in range(mv_count):
            vault_tag = f"{TAG_PREFIXES[(4 + i) % len(TAG_PREFIXES)]}-{20 + i * 11}"
            vals = []
            while len(vals) < 4:
                v = str(rng.randint(10000, 99999))
                if v not in vals:
                    vals.append(v)
            placements = [
                (
                    d_pos,
                    f"\n\n[AUDIT LOG] Entry #{k + 1} for vault key {vault_tag} "
                    f"records value {val}.\n\n",
                )
                for k, (d_pos, val) in enumerate(zip(MV_DEPTHS, vals))
            ]
            hs = _slice_haystack(corpus_tokens, sample_idx * stride, haystack_len)
            doc = _insert_needles(tokenizer, hs, placements)
            prompt = (
                "Read the following reference document carefully and answer the "
                "question at the end.\n\n"
                f"{doc}\n\n"
                f"Question: List all 4 five-digit values recorded for vault key "
                f"{vault_tag} in the document above, separated by commas, in order "
                f"from Entry #1 to Entry #4. Reply with the numbers only."
            )
            tasks.append(
                {
                    "id": f"L{target_len}_mvniah_{i}",
                    "task": "mv_niah",
                    "target_len": target_len,
                    "depth": -1.0,
                    "prompt": prompt,
                    "expected": vals,
                    "distractors": [],
                }
            )
            sample_idx += 1

    return tasks


def _score_response(task: dict, text: str) -> tuple[bool, float]:
    expected = task["expected"]
    distractors = task["distractors"]
    if task["task"] in ("s_niah", "mk_niah"):
        target = expected[0]
        normalized = text.replace(",", "")
        found_target = (target in text) or (target in normalized)
        has_distractor = any((d in text) or (d in normalized) for d in distractors)
        ok = found_target and not has_distractor
        return ok, (1.0 if ok else 0.0)
    # mv_niah: count how many of the 4 expected 5-digit numbers appear.
    hits = sum(1 for v in expected if v in text)
    recall = hits / len(expected)
    return hits == len(expected), recall


def _query_one(*, port: int, model: str, task: dict, timeout_s: int) -> dict:
    body = {
        "model": model,
        "max_tokens": 64,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": task["prompt"]}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read())
        content = payload["choices"][0]["message"]["content"] or ""
        usage = payload.get("usage") or {}
        exact, recall = _score_response(task, content)
        return {
            "id": task["id"],
            "task": task["task"],
            "target_len": task["target_len"],
            "depth": task["depth"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "exact": exact,
            "recall": recall,
            "expected": task["expected"],
            "response": re.sub(r"\s+", " ", content).strip()[:160],
            "latency_s": round(time.time() - t0, 2),
            "status": "ok",
        }
    except Exception as exc:
        return {
            "id": task["id"],
            "task": task["task"],
            "target_len": task["target_len"],
            "depth": task["depth"],
            "exact": False,
            "recall": 0.0,
            "expected": task["expected"],
            "response": "",
            "latency_s": round(time.time() - t0, 2),
            "status": f"error: {type(exc).__name__}: {exc}"[:200],
        }


def _write_summary(out_dir: Path) -> None:
    path = out_dir / "niah_results.jsonl"
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    srv_rows = [r for r in rows if r.get("kind") == "server"]
    sample_rows = [r for r in rows if r.get("kind") == "sample"]

    lines = ["# M4 Long-Context Retrieval Eval (Offline S-NIAH / MK-NIAH / MV-NIAH)\n"]
    if srv_rows:
        lines.append("## Servers\n")
        lines.append("| Config | Ready | Ready s | KV Capacity (tokens) | mem_frac |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in srv_rows:
            lines.append(
                f"| `{r['config']}` | {r['ready']} | {r.get('ready_s', '-')} | "
                f"{r.get('capacity_tokens', '-')} | {r.get('mem_fraction_static', '-')} |"
            )
        lines.append("")

    if sample_rows:
        configs = list(dict.fromkeys(r["config"] for r in sample_rows))
        lengths = sorted({r["target_len"] for r in sample_rows})

        lines.append("## Accuracy by Context Length & Task\n")
        lines.append(
            "| Config | Length | Avg Prompt Tok | S-NIAH | MK-NIAH | "
            "MV-NIAH (Exact) | MV-NIAH (Recall) | Overall Exact | Errors |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for cfg in configs:
            for length in lengths:
                sub = [
                    r
                    for r in sample_rows
                    if r["config"] == cfg and r["target_len"] == length
                ]
                if not sub:
                    continue
                ptoks = [r["prompt_tokens"] for r in sub if r.get("prompt_tokens")]
                avg_ptok = f"{sum(ptoks) / len(ptoks):.0f}" if ptoks else "-"

                def _acc(tname, field="exact"):
                    xs = [r for r in sub if r["task"] == tname]
                    if not xs:
                        return "-"
                    if field == "exact":
                        hits = sum(1 for r in xs if r["exact"])
                        return f"{100.0 * hits / len(xs):.1f}% ({hits}/{len(xs)})"
                    avg = sum(r["recall"] for r in xs) / len(xs)
                    return f"{100.0 * avg:.1f}%"

                total_hits = sum(1 for r in sub if r["exact"])
                errs = sum(1 for r in sub if r.get("status") != "ok")
                lines.append(
                    f"| `{cfg}` | {length} | {avg_ptok} | {_acc('s_niah')} | "
                    f"{_acc('mk_niah')} | {_acc('mv_niah', 'exact')} | "
                    f"{_acc('mv_niah', 'recall')} | "
                    f"{100.0 * total_hits / len(sub):.1f}% ({total_hits}/{len(sub)}) | "
                    f"{errs} |"
                )
        lines.append("")

        lines.append("## S-NIAH Accuracy by Depth\n")
        depth_hdr = " | ".join(f"{int(d * 100)}%" for d in S_NIAH_DEPTHS)
        lines.append(f"| Config | Length | {depth_hdr} |")
        lines.append("|---|---:|" + "---:|" * len(S_NIAH_DEPTHS))
        for cfg in configs:
            for length in lengths:
                sub = [
                    r
                    for r in sample_rows
                    if r["config"] == cfg
                    and r["target_len"] == length
                    and r["task"] == "s_niah"
                ]
                if not sub:
                    continue
                cells = []
                for d in S_NIAH_DEPTHS:
                    ds = [r for r in sub if abs(r["depth"] - d) < 1e-6]
                    if not ds:
                        cells.append("-")
                    else:
                        h = sum(1 for r in ds if r["exact"])
                        cells.append(f"{100.0 * h / len(ds):.0f}% ({h}/{len(ds)})")
                lines.append(f"| `{cfg}` | {length} | " + " | ".join(cells) + " |")
        lines.append("")

    summary = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(summary)
    print("\n" + summary, flush=True)


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--configs", default="bf16-triton,int4-triton")
    p.add_argument("--lengths", default="16384,32768")
    p.add_argument("--s-per-depth", type=int, default=2)
    p.add_argument("--mk-count", type=int, default=10)
    p.add_argument("--mv-count", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--ready-timeout", type=int, default=1200)
    p.add_argument("--request-timeout", type=int, default=300)
    p.add_argument("--gpu-wait-timeout", type=int, default=3600)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "servers").mkdir(exist_ok=True)
    (args.out / "telemetry").mkdir(exist_ok=True)

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config {c!r}; choose from {list(CONFIGS)}")
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]

    tokenizer, corpus_tokens, n_files = _load_corpus_tokens(args.model)
    tasks = _build_tasks(
        tokenizer=tokenizer,
        corpus_tokens=corpus_tokens,
        lengths=lengths,
        s_per_depth=args.s_per_depth,
        mk_count=args.mk_count,
        mv_count=args.mv_count,
        seed=args.seed,
    )
    print(
        f"Built {len(tasks)} long-context tasks across lengths={lengths} from "
        f"{n_files} docs ({len(corpus_tokens)} tokens).",
        flush=True,
    )

    gpu_uuid = m0_bench._gpu_query(args.gpu, "uuid")[0]
    jsonl_path = args.out / "niah_results.jsonl"

    with open(jsonl_path, "w") as out_f:
        for name in configs:
            extra_args, extra_env = CONFIGS[name]
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
                info = m0_bench._parse_server_log(engine="sglang", text=srv.log_text())
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

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=args.concurrency
                ) as pool:
                    futs = [
                        pool.submit(
                            _query_one,
                            port=port,
                            model=args.model,
                            task=t,
                            timeout_s=args.request_timeout,
                        )
                        for t in tasks
                    ]
                    for fut in futs:
                        res = {"kind": "sample", "config": name, **fut.result()}
                        out_f.write(json.dumps(res) + "\n")
                        out_f.flush()
                        print(json.dumps(res), flush=True)
            finally:
                srv.stop()
                telemetry.stop()

    _write_summary(args.out)


if __name__ == "__main__":
    main()
