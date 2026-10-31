"""M0: serving probes for the low-bit KV cache project.

Stage 0 measures only what the kernel work needs to know; the full curves for the
final tables are measured in M5 on the final code.

Groups (run in the order given by --groups):
  main  each SGLang config at a 16k input, concurrency 6 and its max; on the
        BF16-triton and fp4_mx_block16 servers also a decode probe (1k input,
        concurrency 1) and, on BF16-triton, a torch-profiler decode capture
  pool  one more fp4_mx_block16 server with a 40k-token pool, same decode probe,
        so the two FP4 points give the TPOT-vs-pool-size slope
  vllm  vLLM BF16 / int8_per_token_head / int4_per_token_head on TRITON_ATTN,
        same 16k points as main
  m3b   SGLang BF16 and int8_per_token_head on Triton (M3b fused decode), same
        sweep plus the decode probe on both

Every server runs on one GPU in its own process group, listens on 127.0.0.1 on a
free port, and is stopped by signalling that group only. GPU telemetry
(temperature, clocks, power, clock-event reasons) is sampled every second while
a server is up. Outputs under --out: runs.jsonl (one line per benchmark),
summary.md, env.json, servers/*.log, telemetry/*.csv, bench/*.jsonl, profiles/.

Example:
  python benchmark/kv_lowbit/m0_bench.py --model "$QWEN3_8B" --gpu 0 \
      --out "$KVQ_WORK/outputs/m0/$TS" --groups main,pool,vllm
"""

import argparse
import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m0_report  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_SCRIPT = (
    REPO_ROOT
    / ".claude/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py"
)

CONFIGS = {
    "sglang-bf16-flashinfer": ("sglang", []),
    "sglang-bf16-triton": ("sglang", ["--attention-backend", "triton"]),
    # Triton on sm80 has no fp8e4nv, so e5m2 is the FP8 baseline on A100.
    "sglang-fp8e5m2-triton": (
        "sglang",
        ["--attention-backend", "triton", "--kv-cache-dtype", "fp8_e5m2"],
    ),
    # 0.7: at the default 0.823 the whole-layer BF16 dequant OOMs during
    # CUDA graph capture (results/a100-40g/20261028-150827).
    "sglang-fp4mx-triton": (
        "sglang",
        [
            "--attention-backend",
            "triton",
            "--kv-cache-dtype",
            "fp4_mx_block16",
            "--mem-fraction-static",
            "0.7",
        ],
    ),
    "sglang-int8-triton": (
        "sglang",
        ["--attention-backend", "triton", "--kv-cache-dtype", "int8_per_token_head"],
    ),
    "vllm-bf16-triton": ("vllm", []),
    "vllm-int8-triton": ("vllm", ["--kv-cache-dtype", "int8_per_token_head"]),
    "vllm-int4-triton": ("vllm", ["--kv-cache-dtype", "int4_per_token_head"]),
}
GROUP_CONFIGS = {
    "main": [
        "sglang-bf16-flashinfer",
        "sglang-bf16-triton",
        "sglang-fp8e5m2-triton",
        "sglang-fp4mx-triton",
    ],
    "vllm": ["vllm-bf16-triton", "vllm-int8-triton", "vllm-int4-triton"],
    # M3b: fused INT8 decode against BF16, both on Triton.
    "m3b": ["sglang-bf16-triton", "sglang-int8-triton"],
}
# Main-group servers that also run the decode probe at their default pool size.
DECODE_PROBE_CONFIGS = (
    "sglang-bf16-triton",
    "sglang-fp4mx-triton",
    "sglang-int8-triton",
)
# Extra servers for the pool group. Two FP4 points fix the slope of the
# whole-layer dequant cost; BF16 reads only the used tokens, so one point does.
EXTRA_POOL_SIZES = {"sglang-fp4mx-triton": [40000]}
# The FP4 profile is deferred to M6, which needs before/after captures anyway.
PROFILE_CONFIGS = ("sglang-bf16-triton",)
# nvidia-smi clock-event reason bits.
THERMAL_BITS = 0x20 | 0x40
POWER_CAP_BIT = 0x4
HW_SLOWDOWN_BITS = 0x8 | 0x80
TELEMETRY_FIELDS = (
    "timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,"
    "utilization.gpu,memory.used,{reasons}"
)


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument(
        "--gpu", type=int, required=True, help="nvidia-smi index (PCI order)"
    )
    p.add_argument("--expect-gpu", default="A100-PCIE-40GB")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--groups", default="main,pool,vllm")
    p.add_argument("--configs", default="", help="Subset of config names; empty = all")
    p.add_argument("--input-lens", default="16384")
    p.add_argument("--output-len", type=int, default=256)
    # 6: the largest concurrency every config fits at a 16k input (SGLang BF16
    # holds ~121k tokens, i.e. 7 requests); "max" is each config's own limit.
    p.add_argument("--concurrency", default="6,max")
    p.add_argument("--no-profile", action="store_true")
    # SGLang captures decode CUDA graphs up to batch 32 on this GPU; larger
    # batches would fall back to eager decode and not be comparable.
    p.add_argument("--max-concurrency-cap", type=int, default=32)
    p.add_argument(
        "--vllm-bin", default=os.path.expandvars("$KVQ_WORK/envs/vllm/bin/vllm")
    )
    p.add_argument("--ready-timeout", type=int, default=1200)
    p.add_argument("--gpu-wait-timeout", type=int, default=3600)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    return p.parse_args()


# ----------------------------------------------------------------------------
# GPU helpers
# ----------------------------------------------------------------------------


def _nvsmi(args):
    out = subprocess.run(
        ["nvidia-smi", *args], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _gpu_query(gpu, fields):
    line = _nvsmi(
        [f"--query-gpu={fields}", "--format=csv,noheader,nounits", "-i", str(gpu)]
    )
    return [x.strip() for x in line.split(",")]


def _compute_pids(gpu_uuid):
    lines = _nvsmi(["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])
    pids = set()
    for line in lines.splitlines():
        uuid, _, pid = line.partition(",")
        if uuid.strip() == gpu_uuid:
            pids.add(int(pid))
    return pids


def _wait_gpu_free(*, gpu, gpu_uuid, timeout_s):
    start = time.time()
    while True:
        pids = _compute_pids(gpu_uuid)
        used_mib = float(_gpu_query(gpu, "memory.used")[0])
        if not pids and used_mib < 1024:
            return
        waited = time.time() - start
        if waited > timeout_s:
            raise SystemExit(f"GPU {gpu} still busy after {timeout_s}s: pids {pids}")
        print(f"GPU {gpu} busy (pids {pids}, {used_mib} MiB); waiting", flush=True)
        # Short polls first: our previous server may still be releasing memory.
        time.sleep(10 if waited < 120 else 600)


def _reasons_field(gpu):
    for name in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
        try:
            _gpu_query(gpu, name)
            return name
        except subprocess.CalledProcessError:
            continue
    raise SystemExit(
        "nvidia-smi supports neither clocks_event_reasons nor clocks_throttle_reasons"
    )


class Telemetry:
    """Samples GPU state into a CSV every interval_ms while it runs."""

    def __init__(self, *, gpu, path, interval_ms=1000):
        self._fields = TELEMETRY_FIELDS.format(reasons=_reasons_field(gpu))
        self._path = path
        self._file = open(path, "w")
        self._file.write(self._fields + "\n")
        self._file.flush()
        # stdbuf: nvidia-smi block-buffers when writing to a file, which would hide
        # the latest samples from summary() until the process exits.
        line_buffered = ["stdbuf", "-oL"] if shutil.which("stdbuf") else []
        self._proc = subprocess.Popen(
            [
                *line_buffered,
                "nvidia-smi",
                f"--query-gpu={self._fields}",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu),
                "-lms",
                str(interval_ms),
            ],
            stdout=self._file,
            stderr=subprocess.DEVNULL,
        )

    def stop(self):
        self._proc.terminate()
        self._proc.wait(timeout=10)
        self._file.close()

    def summary(self, t_start, t_end):
        self._file.flush()
        return summarize_telemetry(self._path, t_start, t_end)


def summarize_telemetry(path, t_start, t_end):
    temps, sm, power, reasons = [], [], [], []
    for line in Path(path).read_text().splitlines()[1:]:
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 8:
            continue
        try:
            stamp, _, frac = cols[0].partition(".")
            ts = time.mktime(time.strptime(stamp, "%Y/%m/%d %H:%M:%S"))
            ts += float(f"0.{frac}") if frac else 0.0
            if not t_start <= ts <= t_end:
                continue
            temps.append(float(cols[1]))
            sm.append(float(cols[2]))
            power.append(float(cols[4]))
            reasons.append(int(cols[7], 16))
        except ValueError:
            continue
    if not temps:
        return {"samples": 0}
    n = len(reasons)
    return {
        "samples": n,
        "temp_max": max(temps),
        "temp_mean": round(statistics.fmean(temps), 1),
        "sm_clock_min": min(sm),
        "sm_clock_median": statistics.median(sm),
        "power_max": max(power),
        "thermal_frac": sum(bool(r & THERMAL_BITS) for r in reasons) / n,
        "power_cap_frac": sum(bool(r & POWER_CAP_BIT) for r in reasons) / n,
        "hw_slowdown_frac": sum(bool(r & HW_SLOWDOWN_BITS) for r in reasons) / n,
    }


# ----------------------------------------------------------------------------
# Servers
# ----------------------------------------------------------------------------


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _server_command(*, config, model, port, extra_args, vllm_bin):
    engine, args = CONFIGS[config]
    if engine == "sglang":
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--random-seed",
            "42",
        ]
    else:
        cmd = [
            vllm_bin,
            "serve",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--attention-backend",
            "TRITON_ATTN",
            "--seed",
            "42",
        ]
    return cmd + args + extra_args


def _server_env(*, engine, gpu, profile_dir):
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID"
    )
    if engine == "sglang":
        env["SGLANG_TORCH_PROFILER_DIR"] = str(profile_dir)
    else:
        # The FlashInfer sampler failed in the env smoke; sampling is not under test.
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        # The CUDA 13 forward-compat libcuda belongs to the SGLang env only.
        env["LD_LIBRARY_PATH"] = ":".join(
            p
            for p in env.get("LD_LIBRARY_PATH", "").split(":")
            if p and "compat" not in p
        )
    return env


class Server:
    """One inference server in its own session; stop() signals only that group."""

    def __init__(self, *, cmd, env, log_path):
        self._log = open(log_path, "w")
        self._log_path = log_path
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    def wait_ready(self, *, port, timeout_s):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=5
                ):
                    return True
            except OSError:
                time.sleep(5)
        return False

    def processes(self):
        import psutil

        try:
            root = psutil.Process(self.proc.pid)
            return [root, *root.children(recursive=True)]
        except psutil.NoSuchProcess:
            return []

    def stop(self):
        import psutil

        # psutil.Process objects refuse to signal a PID that has been reused.
        members = self.processes()
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        for p in members:
            try:
                p.kill()
            except psutil.Error:
                pass
        self._log.close()

    def log_text(self):
        return Path(self._log_path).read_text(errors="ignore")


def _last(pattern, text, cast=str):
    matches = re.findall(pattern, text)
    return cast(matches[-1].replace(",", "")) if matches else None


def _parse_server_log(*, engine, text):
    if engine == "sglang":
        return {
            "capacity_tokens": _last(r"max_total_num_tokens=(\d+)", text, int),
            "default_backend": _last(r"Use (\S+) backend by default", text),
            "mem_fraction_static": _last(
                r"'mem_fraction_static': ([0-9.]+)", text, float
            ),
            "decode_graph_max_bs": _last(
                r"'decode': \{'backend': '\w+', 'max_bs': (\d+)", text, int
            ),
        }
    return {
        "capacity_tokens": _last(r"GPU KV cache size: ([\d,]+) tokens", text, int),
        "gpu_memory_utilization": _last(
            r"current --gpu-memory-utilization=([0-9.]+)", text, float
        ),
    }


# ----------------------------------------------------------------------------
# Session: one output directory, one GPU, a sequence of servers and benchmarks
# ----------------------------------------------------------------------------

_BENCH_KEYS = (
    "completed",
    "duration",
    "output_throughput",
    "input_throughput",
    "request_throughput",
    "median_tpot_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "median_ttft_ms",
    "median_itl_ms",
    "max_concurrent_requests",
    "total_input_tokens",
    "total_output_tokens",
)


class Session:
    def __init__(self, args):
        self.args = args
        self.out = args.out
        for sub in ("servers", "telemetry", "bench", "profiles"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.gpu_uuid = None if args.dry_run else _gpu_query(args.gpu, "uuid")[0]

    def record(self, row):
        row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.out / "runs.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    @contextlib.contextmanager
    def server(self, *, config, extra_args, tag):
        engine = CONFIGS[config][0]
        port = _free_port()
        cmd = _server_command(
            config=config,
            model=self.args.model,
            port=port,
            extra_args=extra_args,
            vllm_bin=self.args.vllm_bin,
        )
        if self.args.dry_run:
            print(f"[server {tag}] {shlex.join(cmd)}")
            yield {
                "engine": engine,
                "port": port,
                "server": None,
                "telemetry": None,
                "info": {"capacity_tokens": 300000},
            }
            return
        _wait_gpu_free(
            gpu=self.args.gpu,
            gpu_uuid=self.gpu_uuid,
            timeout_s=self.args.gpu_wait_timeout,
        )
        telemetry = Telemetry(
            gpu=self.args.gpu, path=self.out / "telemetry" / f"{tag}.csv"
        )
        env = _server_env(
            engine=engine, gpu=self.args.gpu, profile_dir=self.out / "profiles" / tag
        )
        srv = Server(cmd=cmd, env=env, log_path=self.out / "servers" / f"{tag}.log")
        t0 = time.time()
        try:
            ready = srv.wait_ready(port=port, timeout_s=self.args.ready_timeout)
            info = _parse_server_log(engine=engine, text=srv.log_text())
            self.record(
                {
                    "group": "server",
                    "tag": tag,
                    "config": config,
                    "engine": engine,
                    "config_args": CONFIGS[config][1],
                    "extra_args": extra_args,
                    "ready": ready,
                    "ready_s": round(time.time() - t0),
                    **info,
                    "error_tail": None if ready else srv.log_text()[-3000:],
                }
            )
            yield (
                {
                    "engine": engine,
                    "port": port,
                    "server": srv,
                    "telemetry": telemetry,
                    "info": info,
                }
                if ready
                else None
            )
        finally:
            srv.stop()
            telemetry.stop()

    def _bench_once(self, *, ctx, tag, input_len, conc, num_prompts, seed):
        jsonl = self.out / "bench" / f"{tag}.jsonl"
        cmd = [
            sys.executable,
            "-m",
            "sglang.bench_serving",
            "--backend",
            ctx["engine"],
            "--host",
            "127.0.0.1",
            "--port",
            str(ctx["port"]),
            "--model",
            self.args.model,
            "--dataset-name",
            "random-ids",
            "--random-input-len",
            str(input_len),
            "--random-output-len",
            str(self.args.output_len),
            "--random-range-ratio",
            "1",
            "--num-prompts",
            str(num_prompts),
            "--max-concurrency",
            str(conc),
            "--seed",
            str(seed),
            "--output-file",
            str(jsonl),
            "--disable-tqdm",
        ]
        if ctx["engine"] == "sglang":
            # Exact input lengths; the vLLM backend sends re-tokenized text instead.
            cmd.append("--tokenize-prompt")
        if self.args.dry_run:
            print(f"  [bench] {shlex.join(cmd)}")
            return {}, 0.0, 0.0
        before = len(jsonl.read_text().splitlines()) if jsonl.exists() else 0
        t0 = time.time()
        with open(self.out / "bench" / f"{tag}.log", "a") as log:
            proc = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, timeout=3600
            )
        t1 = time.time()
        lines = jsonl.read_text().splitlines() if jsonl.exists() else []
        if proc.returncode != 0 or len(lines) <= before:
            return None, t0, t1
        return json.loads(lines[-1]), t0, t1

    def bench_and_record(
        self, *, ctx, group, config, tag, input_len, conc, num_prompts, seed, extra=None
    ):
        row = {
            "group": group,
            "config": config,
            "tag": tag,
            "input_len": input_len,
            "output_len": self.args.output_len,
            "concurrency": conc,
            "num_prompts": num_prompts,
            "seed": seed,
            "capacity_tokens": ctx["info"].get("capacity_tokens"),
            **(extra or {}),
        }
        for attempt in range(2):
            result, t0, t1 = self._bench_once(
                ctx=ctx,
                tag=tag,
                input_len=input_len,
                conc=conc,
                num_prompts=num_prompts,
                seed=seed,
            )
            if self.args.dry_run:
                return
            own = {p.pid for p in ctx["server"].processes()}
            foreign = sorted(_compute_pids(self.gpu_uuid) - own)
            if not foreign:
                break
        row.update(
            status="ok" if result else "bench_failed",
            retried=attempt > 0,
            foreign_pids=foreign,
            # 1 s of slack so even a short run gets at least one 1 Hz sample.
            telemetry=ctx["telemetry"].summary(t0 - 1, t1 + 1),
            **{k: (result or {}).get(k) for k in _BENCH_KEYS},
        )
        self.record(row)


# ----------------------------------------------------------------------------
# Groups
# ----------------------------------------------------------------------------


def _concurrencies(*, spec, capacity, input_len, output_len, cap):
    fit = capacity // (input_len + output_len) if capacity else cap
    max_conc = min(cap, fit)
    picked = set()
    for tok in spec.split(","):
        conc = max_conc if tok == "max" else int(tok)
        if 1 <= conc <= max_conc:
            picked.add(conc)
    return sorted(picked)


def _num_prompts(conc):
    # Two waves at high concurrency; at least 8 requests for a stable median.
    return min(max(8, 2 * conc), 128)


def _seed(input_len, conc):
    # Unique per (input_len, conc) so no run reuses another run's prompts from the
    # radix cache, and identical across configs so every config sees the same prompts.
    return 1000 + 100 * (input_len // 1024) + conc


def run_sweep(session, *, group, config):
    with session.server(config=config, extra_args=[], tag=config) as ctx:
        if ctx is None:
            return
        _sweep(session, ctx=ctx, group=group, config=config)
        if group not in ("main", "m3b"):
            return
        if config in DECODE_PROBE_CONFIGS:
            _decode_probe(
                session,
                ctx=ctx,
                config=config,
                tag=config,
                pool_tokens=ctx["info"].get("capacity_tokens"),
            )
        if (
            group == "main"
            and config in PROFILE_CONFIGS
            and not session.args.no_profile
        ):
            _profile(session, ctx=ctx, config=config, tag=f"{config}-profile")


def _sweep(session, *, ctx, group, config):
    args = session.args
    for input_len in [int(x) for x in args.input_lens.split(",")]:
        concs = _concurrencies(
            spec=args.concurrency,
            capacity=ctx["info"].get("capacity_tokens"),
            input_len=input_len,
            output_len=args.output_len,
            cap=args.max_concurrency_cap,
        )
        if not concs:
            session.record(
                {
                    "group": group,
                    "config": config,
                    "input_len": input_len,
                    "status": "skipped",
                    "reason": "one request does not fit",
                }
            )
            continue
        for conc in concs:
            session.bench_and_record(
                ctx=ctx,
                group=group,
                config=config,
                tag=config,
                input_len=input_len,
                conc=conc,
                num_prompts=_num_prompts(conc),
                seed=_seed(input_len, conc),
            )


def _decode_probe(session, *, ctx, config, tag, pool_tokens):
    """Decode TPOT at a short context, where pool-size-dependent costs dominate."""
    session.bench_and_record(
        ctx=ctx,
        group="pool",
        config=config,
        tag=tag,
        input_len=1024,
        conc=1,
        num_prompts=8,
        seed=7,
        extra={"pool_tokens": pool_tokens},
    )


def run_pool(session, *, configs):
    for config, sizes in EXTRA_POOL_SIZES.items():
        if config not in configs:
            continue
        for size in sizes:
            tag = f"{config}-pool{size}"
            with session.server(
                config=config, extra_args=["--max-total-tokens", str(size)], tag=tag
            ) as ctx:
                if ctx is not None:
                    _decode_probe(
                        session, ctx=ctx, config=config, tag=tag, pool_tokens=size
                    )


def _profile(session, *, ctx, config, tag):
    trace_dir = session.out / "profiles" / tag
    # Decode at a 16k context with 6 requests fits the BF16 pool (~121k tokens).
    cmd = [
        sys.executable,
        str(PROFILE_SCRIPT),
        "--framework",
        "sglang",
        "--url",
        f"http://127.0.0.1:{ctx['port']}",
        "--output-dir",
        str(trace_dir),
        "--profile-workload",
        "decode",
        "--decode-input-len",
        "16384",
        "--decode-output-len",
        "256",
        "--probe-requests",
        "6",
        "--num-steps",
        "5",
        "--warmup-steps",
        "10",
    ]
    if session.args.dry_run:
        print(f"  [profile] {shlex.join(cmd)}")
        return
    table = session.out / "profiles" / f"{tag}_decode_tables.txt"
    with open(table, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
    session.record(
        {
            "group": "profile",
            "config": config,
            "tag": tag,
            "status": "ok" if proc.returncode == 0 else "profile_failed",
            "tables": table.name,
        }
    )


# ----------------------------------------------------------------------------
# Environment record and entry point
# ----------------------------------------------------------------------------


def _run_text(cmd):
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable: {e}"


def write_env(session):
    from importlib import metadata

    args = session.args
    name, driver, power, sm_max, mem_max = _gpu_query(
        args.gpu, "name,driver_version,power.limit,clocks.max.sm,clocks.max.mem"
    )
    versions = {}
    for pkg in (
        "torch",
        "triton",
        "flashinfer-python",
        "sglang-kernel",
        "transformers",
    ):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    vllm_python = str(Path(args.vllm_bin).parent / "python")
    env = {
        "gpu_index": args.gpu,
        "gpu_name": name,
        "driver": driver,
        "power_limit_w": power,
        "sm_clock_max": sm_max,
        "mem_clock_max": mem_max,
        "sglang_commit": _run_text(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "torch_cuda": _run_text(
            [sys.executable, "-c", "import torch; print(torch.version.cuda)"]
        ),
        "versions": versions,
        "vllm": _run_text(
            [
                vllm_python,
                "-c",
                "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda)",
            ]
        ),
        "cuda_forward_compat": "compat" in os.environ.get("LD_LIBRARY_PATH", ""),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (session.out / "env.json").write_text(json.dumps(env, indent=1))
    if args.expect_gpu not in name:
        raise SystemExit(f"GPU {args.gpu} is {name!r}, expected {args.expect_gpu!r}")


def main():
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        m0_report.write_summary(args.out)
        return
    session = Session(args)
    if not args.dry_run:
        write_env(session)
    selected = set(args.configs.split(",")) if args.configs else set(CONFIGS)
    for group in args.groups.split(","):
        if group in GROUP_CONFIGS:
            for config in GROUP_CONFIGS[group]:
                if config in selected:
                    run_sweep(session, group=group, config=config)
        elif group == "pool":
            run_pool(session, configs=selected)
        else:
            raise SystemExit(f"unknown group {group!r}")
    if not args.dry_run:
        m0_report.write_summary(args.out)


if __name__ == "__main__":
    main()
