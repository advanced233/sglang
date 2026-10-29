"""M2 smoke: the int8_per_token_head recipe on a real server, next to BF16.

For each config (BF16 and INT8, both on the Triton backend with the same
--max-total-tokens) starts an SGLang server, checks the KV bytes the pool
allocated, sends one short request, runs a decode probe (1k input, 256 output,
concurrency 1) and GSM8K through sgl-eval. The pool size is capped because the
non-fused INT8 read dequantizes a whole layer of the pool into BF16 per call;
at the default memory fraction that copy does not fit (plan.md M2).

Outputs under --out: runs.jsonl, summary.md, servers/*.log, telemetry/*.csv,
eval/*.log, bench/*.jsonl.

Example:
  python benchmark/kv_lowbit/m2_smoke.py --model "$QWEN3_8B" --gpu 0 \\
      --out "$KVQ_WORK/outputs/m2/$TS"
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m0_bench import (  # noqa: E402
    Server,
    Telemetry,
    _free_port,
    _gpu_query,
    _last,
    _server_env,
    _wait_gpu_free,
)

CONFIGS = {
    "bf16-triton": [],
    "int8-triton": ["--kv-cache-dtype", "int8_per_token_head"],
}
# Qwen3 non-thinking sampling recommendation; sgl-eval 0.1.3 has no --top-k flag.
DEFAULT_EVAL_ARGS = (
    "--chat-template-kwarg enable_thinking=false --temperature 0.7 --top-p 0.8"
)


def _record(out, row):
    row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(out / "runs.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def _server_cmd(*, args, config, port):
    return [
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
        "--attention-backend",
        "triton",
        # FlashInfer's sampling kernels JIT-compile against curand.h, which the
        # env's nvcc package does not ship; sgl-eval samples with top_p.
        "--sampling-backend",
        "pytorch",
        "--max-total-tokens",
        str(args.max_total_tokens),
        *CONFIGS[config],
    ]


def _pool_info(log_text, num_layers):
    """Allocated KV bytes per token per layer, from the pool's allocation log."""
    k_gb = _last(r"K size: ([0-9.]+) GB", log_text, float)
    v_gb = _last(r"V size: ([0-9.]+) GB", log_text, float)
    tokens = _last(r"#tokens: (\d+), K size", log_text, int)
    info = {
        "capacity_tokens": _last(r"max_total_num_tokens=(\d+)", log_text, int),
        "kv_dtype_logged": _last(r"KV Cache is allocated\. dtype: (\S+),", log_text),
        "k_gb": k_gb,
        "v_gb": v_gb,
    }
    if k_gb and v_gb and tokens and num_layers:
        info["kv_bytes_per_token_layer"] = round(
            (k_gb + v_gb) * 2**30 / tokens / num_layers
        )
    return info


def _ask(port, model):
    body = {
        "model": model,
        "max_tokens": 64,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": "What is 17 * 23? Answer with the number only.",
            }
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except OSError as e:
        return f"request failed: {e}"


def _decode_probe(*, args, port, tag):
    path = args.out / "bench" / f"{tag}.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        args.model,
        "--dataset-name",
        "random-ids",
        "--tokenize-prompt",
        "--random-input-len",
        "1024",
        "--random-output-len",
        "256",
        "--random-range-ratio",
        "1",
        "--num-prompts",
        "8",
        "--max-concurrency",
        "1",
        "--seed",
        "1",
        "--output-file",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if res.returncode != 0 or not path.exists():
        return {"probe_status": "failed", "probe_error": res.stderr[-2000:]}
    rec = json.loads(path.read_text().splitlines()[-1])
    return {
        "probe_status": "ok",
        "probe_median_itl_ms": rec.get("median_itl_ms"),
        "probe_median_tpot_ms": rec.get("median_tpot_ms"),
    }


def _gsm8k(*, args, port, tag):
    log = args.out / "eval" / f"{tag}.log"
    cmd = [
        args.sgl_eval,
        "run",
        "gsm8k",
        "--base-url",
        f"http://127.0.0.1:{port}/v1",
        *shlex.split(args.eval_args),
    ]
    if args.num_examples:
        cmd += ["--num-examples", str(args.num_examples)]
    with open(log, "w") as f:
        f.write(shlex.join(cmd) + "\n")
        f.flush()
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=7200)
    text = log.read_text(errors="ignore")
    generation = re.findall(r"generation: (\{.*\})", text)
    return {
        "eval_status": "ok" if res.returncode == 0 else f"exit {res.returncode}",
        "gsm8k_score": _last(r"\* score\s*=\s*([0-9.]+)%", text, float),
        "gsm8k_examples": _last(r"(\d+) examples", text, int),
        "eval_generation": generation[-1] if generation else None,
    }


def run_config(args, *, config, gpu_uuid, num_layers):
    port = _free_port()
    _wait_gpu_free(gpu=args.gpu, gpu_uuid=gpu_uuid, timeout_s=3600)
    telemetry = Telemetry(gpu=args.gpu, path=args.out / "telemetry" / f"{config}.csv")
    srv = Server(
        cmd=_server_cmd(args=args, config=config, port=port),
        env=_server_env(
            engine="sglang", gpu=args.gpu, profile_dir=args.out / "profiles"
        ),
        log_path=args.out / "servers" / f"{config}.log",
    )
    row = {"config": config, "max_total_tokens": args.max_total_tokens}
    try:
        t0 = time.time()
        row["ready"] = srv.wait_ready(port=port, timeout_s=args.ready_timeout)
        row["ready_s"] = round(time.time() - t0)
        row.update(_pool_info(srv.log_text(), num_layers))
        if not row["ready"]:
            row["error_tail"] = srv.log_text()[-3000:]
            return row
        row["answer"] = _ask(port, args.model)
        row.update(_decode_probe(args=args, port=port, tag=config))
        if not args.skip_eval:
            row.update(_gsm8k(args=args, port=port, tag=config))
    finally:
        srv.stop()
        telemetry.stop()
        _record(args.out, row)
    return row


def write_summary(out):
    rows = [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines()]
    lines = [
        "# M2 smoke: int8_per_token_head vs BF16 (Triton backend)",
        "",
        "| config | ready | KV dtype | KV B/token/layer | answer | probe ITL ms "
        "| GSM8K % | examples | eval |",
        "|---|---|---|---:|---|---:|---:|---:|---|",
    ]
    for r in rows:
        answer = str(r.get("answer", "-")).replace("\n", " ").replace("|", "/")[:40]
        lines.append(
            f"| {r['config']} | {r.get('ready')} | {r.get('kv_dtype_logged')} "
            f"| {r.get('kv_bytes_per_token_layer', '-')} | {answer} "
            f"| {r.get('probe_median_itl_ms', '-')} | {r.get('gsm8k_score', '-')} "
            f"| {r.get('gsm8k_examples', '-')} | {r.get('eval_status', '-')} |"
        )
    gens = {r.get("eval_generation") for r in rows if r.get("eval_generation")}
    lines += ["", *(f"- sgl-eval generation: `{g}`" for g in sorted(gens))]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _num_layers(model):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model).num_hidden_layers


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--gpu", type=int, required=True, help="nvidia-smi index")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--max-total-tokens", type=int, default=100000)
    p.add_argument("--expect-gpu", default="A100-PCIE-40GB")
    p.add_argument("--ready-timeout", type=int, default=1200)
    p.add_argument(
        "--sgl-eval",
        default=str(Path(os.environ.get("KVQ_WORK", "")) / "envs/sgleval/bin/sgl-eval"),
    )
    p.add_argument("--eval-args", default=DEFAULT_EVAL_ARGS)
    p.add_argument("--num-examples", type=int, default=0, help="0 = full test set")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    for sub in ("servers", "telemetry", "eval", "bench", "profiles"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    name, gpu_uuid = _gpu_query(args.gpu, "name,uuid")
    if args.expect_gpu not in name:
        raise SystemExit(f"GPU {args.gpu} is {name!r}, expected {args.expect_gpu!r}")
    if not args.skip_eval and not Path(args.sgl_eval).exists():
        raise SystemExit(f"sgl-eval not found: {args.sgl_eval}")
    num_layers = _num_layers(args.model)
    for config in args.configs.split(","):
        run_config(args, config=config, gpu_uuid=gpu_uuid, num_layers=num_layers)
    write_summary(args.out)


if __name__ == "__main__":
    main()
