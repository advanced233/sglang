"""M3.0: decode-attention kernel diagnosis on one fixed decode shape.

Times SGLang's Triton decode kernel (BF16, fp8_e5m2) and vLLM's TRITON_ATTN
kernel (BF16, INT8 / INT4 per-token-head) on identical inputs, plus vLLM
ablations for the stage-0 hypotheses in plan.md 3.5 (decode tile size; scales
inline in the page vs in separate aligned buffers). Then profiles one call of
each config with Nsight Compute.

The default shape is the stage-0 serving point: 6 sequences x 16k context with
Qwen3-8B attention (32 Q heads, 8 KV heads, head_dim 128). Each op is captured
in a CUDA graph and timed by triton.testing.do_bench, which zeroes a 256 MB
buffer before every call, so L2 is cold; the KV working set (>= 107 MB) is
larger than L2 anyway.

Subcommands:
  run     time and profile every config; start it from the SGLang env
  bench   time the configs of one environment ("sglang" or "vllm")
  launch  run one config's op once between cudaProfilerStart/Stop, for ncu
  report  rebuild summary.md from the files under --out

Example:
  python benchmark/kv_lowbit/m3_kernel_diag.py run --gpu 0 \\
      --out "$KVQ_WORK/outputs/m3diag/$TS"
"""

import argparse
import contextlib
import csv
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import msgspec

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parents[2]
# vLLM TRITON_ATTN launch parameters for this model: the split-KV (3D) decode
# threshold is the CUDA graph capture size closest to 128 // 8 KV heads.
VLLM_BLOCK_SIZE = 16
VLLM_SEQ_THRESHOLD_3D = 16
VLLM_NUM_SEGMENTS = 16
# SGLang's --triton-attention-num-kv-splits default.
SGLANG_MAX_KV_SPLITS = 8
# A100-PCIE-40GB: HBM2 at 1215 MHz on a 5120-bit bus.
DEFAULT_PEAK_GBPS = 1555.0


class Config(msgspec.Struct, frozen=True, kw_only=True):
    impl: str  # "sglang", "vllm", or "ref"
    kv: str  # "bf16", "fp8_e5m2", "int8", "int4"
    # vLLM only. "inline" is vLLM's page layout, [K | K scale | V | V scale] per
    # (token, head); "sep" keeps data and fp32 scales in separate aligned buffers.
    layout: str = "inline"
    # vLLM only: decode TILE_SIZE override; 0 keeps vLLM's choice (16 for a bf16 Q).
    decode_tile: int = 0

    @property
    def env(self):
        return "vllm" if self.impl == "vllm" else "sglang"


CONFIGS = {
    # Practical read-bandwidth ceiling: one reduction over the BF16 K and V bytes.
    "ref-sum": Config(impl="ref", kv="bf16"),
    "sglang-bf16": Config(impl="sglang", kv="bf16"),
    "sglang-fp8e5m2": Config(impl="sglang", kv="fp8_e5m2"),
    "vllm-bf16": Config(impl="vllm", kv="bf16"),
    "vllm-bf16-t32": Config(impl="vllm", kv="bf16", decode_tile=32),
    "vllm-int8": Config(impl="vllm", kv="int8"),
    "vllm-int8-t32": Config(impl="vllm", kv="int8", decode_tile=32),
    "vllm-int8-t64": Config(impl="vllm", kv="int8", decode_tile=64),
    "vllm-int8-sep": Config(impl="vllm", kv="int8", layout="sep"),
    "vllm-int8-sep-t64": Config(impl="vllm", kv="int8", layout="sep", decode_tile=64),
    "vllm-int4": Config(impl="vllm", kv="int4"),
    "vllm-int4-t64": Config(impl="vllm", kv="int4", decode_tile=64),
}

# The kernel that streams the KV cache, per implementation.
MAIN_KERNEL = {
    "ref": "reduce_kernel",
    "sglang": "_fwd_grouped_kernel_stage1",
    "vllm": "kernel_unified_attention",
    "vllm-int4": "_attn_packed",
}


class Shape(msgspec.Struct, frozen=True, kw_only=True):
    batch: int
    seq_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int

    @property
    def num_tokens(self):
        return self.batch * self.seq_len


def kv_bytes(*, cfg, shape):
    """KV bytes one decode step reads, as stored (data plus scales)."""
    data = {
        "bf16": 2 * shape.head_dim,
        "fp8_e5m2": shape.head_dim,
        "int8": shape.head_dim,
        "int4": shape.head_dim // 2,
    }[cfg.kv]
    scale = 4 if cfg.kv in ("int8", "int4") else 0
    return 2 * (data + scale) * shape.num_kv_heads * shape.num_tokens


def _main_kernel(cfg):
    if cfg.impl == "vllm" and cfg.kv == "int4":
        return MAIN_KERNEL["vllm-int4"]
    return MAIN_KERNEL[cfg.impl]


# ----------------------------------------------------------------------------
# Ops (run inside the SGLang or vLLM env)
# ----------------------------------------------------------------------------


def make_inputs(shape, seed=0):
    import torch

    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*size):
        return torch.randn(*size, generator=gen, device="cuda", dtype=torch.bfloat16)

    q = randn(shape.batch, shape.num_q_heads, shape.head_dim)
    k = randn(shape.num_tokens, shape.num_kv_heads, shape.head_dim)
    v = randn(shape.num_tokens, shape.num_kv_heads, shape.head_dim)
    return q, k, v


def reference_attention(*, q, k, v, shape):
    import torch

    group = shape.num_q_heads // shape.num_kv_heads
    kv_shape = (shape.batch, shape.seq_len, shape.num_kv_heads, shape.head_dim)
    qf = q.float().view(shape.batch, shape.num_kv_heads, group, shape.head_dim)
    scores = torch.einsum("bhgd,bshd->bhgs", qf, k.float().view(kv_shape))
    probs = (scores * shape.head_dim**-0.5).softmax(dim=-1)
    out = torch.einsum("bhgs,bshd->bhgd", probs, v.float().view(kv_shape))
    return out.reshape(shape.batch, shape.num_q_heads, shape.head_dim)


def build_op(*, cfg, shape, q, k, v):
    """Returns (op, out, info): op() writes the attention output into out."""
    if cfg.impl == "ref":
        return _build_ref(k=k, v=v)
    if cfg.impl == "sglang":
        return _build_sglang(cfg=cfg, shape=shape, q=q, k=k, v=v)
    return _build_vllm(cfg=cfg, shape=shape, q=q, k=k, v=v)


def _build_ref(*, k, v):
    import torch

    kv = torch.cat([k.flatten(), v.flatten()])
    total = torch.empty((), dtype=torch.float32, device=kv.device)

    def op():
        torch.sum(kv, dim=0, dtype=torch.float32, out=total)

    return op, None, {}


def _build_sglang(*, cfg, shape, q, k, v):
    import torch

    from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd
    from sglang.kernels.ops.attention.metadata import get_num_kv_splits_triton

    dev = q.device
    dtype = torch.bfloat16 if cfg.kv == "bf16" else torch.float8_e5m2
    # Slot 0 is the pool's padding slot; requests own contiguous slots after it.
    k_buf = torch.zeros(
        (shape.num_tokens + 1, shape.num_kv_heads, shape.head_dim),
        dtype=dtype,
        device=dev,
    )
    v_buf = torch.zeros_like(k_buf)
    k_buf[1:] = k.to(dtype)
    v_buf[1:] = v.to(dtype)
    kv_indptr = torch.arange(shape.batch + 1, dtype=torch.int32, device=dev)
    kv_indptr *= shape.seq_len
    kv_indices = torch.arange(1, shape.num_tokens + 1, dtype=torch.int64, device=dev)
    seq_lens = torch.full((shape.batch,), shape.seq_len, dtype=torch.int32, device=dev)
    num_kv_splits = torch.empty(shape.batch, dtype=torch.int32, device=dev)
    # Same split heuristic as TritonAttnBackend.get_num_kv_splits.
    get_num_kv_splits_triton[(1,)](
        num_kv_splits_ptr=num_kv_splits,
        seq_lens_ptr=seq_lens,
        num_seq=shape.batch,
        num_group=1,
        num_head=shape.num_q_heads,
        num_kv_head=shape.num_kv_heads,
        max_kv_splits=SGLANG_MAX_KV_SPLITS,
        device_core_count=torch.cuda.get_device_properties(dev).multi_processor_count,
        MAX_NUM_SEQ=256,
    )
    attn_logits = torch.empty(
        (shape.batch, shape.num_q_heads, SGLANG_MAX_KV_SPLITS, shape.head_dim),
        dtype=torch.float32,
        device=dev,
    )
    attn_lse = torch.empty(
        (shape.batch, shape.num_q_heads, SGLANG_MAX_KV_SPLITS),
        dtype=torch.float32,
        device=dev,
    )
    out = torch.empty_like(q)

    def op():
        decode_attention_fwd(
            q=q,
            k_buffer=k_buf,
            v_buffer=v_buf,
            o=out,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            attn_logits=attn_logits,
            attn_lse=attn_lse,
            num_kv_splits=num_kv_splits,
            max_kv_splits=SGLANG_MAX_KV_SPLITS,
            sm_scale=shape.head_dim**-0.5,
            k_scale=1.0,
            v_scale=1.0,
        )

    splits = sorted(set(num_kv_splits.tolist()))
    return op, out, {"kv_splits": splits, "block_n": 32}


def _vllm_caches(*, cfg, num_blocks, shape):
    """vLLM's per-layer views in its default LBNHC layout: [block, slot, head, C]."""
    import torch

    size = (num_blocks, VLLM_BLOCK_SIZE, shape.num_kv_heads)
    d = shape.head_dim
    if cfg.kv == "bf16":
        raw = torch.zeros((*size, 2 * d), dtype=torch.bfloat16, device="cuda")
        return raw[..., :d], raw[..., d:], None, None
    dtype = torch.int8 if cfg.kv == "int8" else torch.uint8
    data = d if cfg.kv == "int8" else d // 2
    if cfg.layout == "sep":
        k_cache = torch.zeros((*size, data), dtype=dtype, device="cuda")
        k_scale = torch.ones(size, dtype=torch.float32, device="cuda")
        return k_cache, torch.zeros_like(k_cache), k_scale, torch.ones_like(k_scale)
    # One fp32 scale sits in 4 bytes of padding after each head's K and V, as in
    # TritonAttentionImpl._ensure_scale_caches.
    padded = data + 4
    raw = torch.zeros((*size, 2 * padded), dtype=dtype, device="cuda")
    raw_f32 = raw.view(torch.float32)
    return (
        raw[..., :padded],
        raw[..., padded:],
        raw_f32[..., data // 4],
        raw_f32[..., (padded + data) // 4],
    )


def _build_vllm(*, cfg, shape, q, k, v):
    import torch
    import vllm.v1.attention.ops.triton_unified_attention as tua
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_head_quant,
    )
    from vllm.v1.kv_cache_interface import KVQuantMode

    dev = q.device
    blocks_per_seq = -(-shape.seq_len // VLLM_BLOCK_SIZE)
    num_blocks = shape.batch * blocks_per_seq
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=dev).view(
        shape.batch, blocks_per_seq
    )
    slots = block_table[:, :, None] * VLLM_BLOCK_SIZE + torch.arange(
        VLLM_BLOCK_SIZE, device=dev
    )
    slot_mapping = slots.view(shape.batch, -1)[:, : shape.seq_len].reshape(-1).long()
    k_cache, v_cache, k_scale, v_scale = _vllm_caches(
        cfg=cfg, num_blocks=num_blocks, shape=shape
    )
    mode = {
        "bf16": KVQuantMode.NONE,
        "int8": KVQuantMode.INT8_PER_TOKEN_HEAD,
        "int4": KVQuantMode.INT4_PER_TOKEN_HEAD,
    }[cfg.kv]
    if cfg.kv == "bf16":
        blk, off = slot_mapping // VLLM_BLOCK_SIZE, slot_mapping % VLLM_BLOCK_SIZE
        k_cache[blk, off] = k
        v_cache[blk, off] = v
    else:
        # vLLM's own write path, including the RHT for INT4.
        triton_reshape_and_cache_flash_per_token_head_quant(
            key=k,
            value=v,
            key_cache=k_cache,
            value_cache=v_cache,
            k_scale_cache=k_scale,
            v_scale_cache=v_scale,
            slot_mapping=slot_mapping,
            kv_quant_mode=mode,
        )
    padded_d = 1 << (shape.head_dim - 1).bit_length()
    segm = (VLLM_SEQ_THRESHOLD_3D, shape.num_q_heads, VLLM_NUM_SEGMENTS)
    segm_output = torch.empty((*segm, padded_d), dtype=torch.float32, device=dev)
    segm_max = torch.empty(segm, dtype=torch.float32, device=dev)
    segm_expsum = torch.empty(segm, dtype=torch.float32, device=dev)
    cu_seqlens_q = torch.arange(shape.batch + 1, dtype=torch.int32, device=dev)
    seqused_k = torch.full((shape.batch,), shape.seq_len, dtype=torch.int32, device=dev)
    out = torch.empty_like(q)

    def op():
        tua.unified_attention(
            q=q,
            k=k_cache,
            v=v_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=seqused_k,
            max_seqlen_k=shape.seq_len,
            softmax_scale=shape.head_dim**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            seq_threshold_3D=VLLM_SEQ_THRESHOLD_3D,
            num_par_softmax_segments=VLLM_NUM_SEGMENTS,
            softmax_segm_output=segm_output,
            softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_expsum,
            kv_quant_mode=mode,
            k_scale_cache=k_scale,
            v_scale_cache=v_scale,
        )

    tile = tua._get_tile_size(
        head_size=shape.head_dim,
        sliding_window=0,
        element_size=q.element_size(),
        is_prefill=False,
    )
    path = "3d" if shape.batch <= VLLM_SEQ_THRESHOLD_3D else "2d"
    return op, out, {"decode_tile_used": tile, "vllm_path": path}


@contextlib.contextmanager
def vllm_decode_tile(tile):
    """Overrides vLLM's decode TILE_SIZE while launching (and graph-capturing)."""
    if not tile:
        yield
        return
    import vllm.v1.attention.ops.triton_unified_attention as tua

    original = tua._get_tile_size

    def patched(head_size, sliding_window, element_size, is_prefill):
        if is_prefill:
            return original(head_size, sliding_window, element_size, is_prefill)
        return tile

    # The INT4 launcher imports _get_tile_size from this module at call time.
    tua._get_tile_size = patched
    try:
        yield
    finally:
        tua._get_tile_size = original


def _accuracy(*, out, ref):
    import torch

    o, r = out.float().flatten(), ref.flatten()
    cos = torch.nn.functional.cosine_similarity(o, r, dim=0).item()
    rel = ((o - r).norm() / r.norm()).item()
    return {"cos": round(cos, 6), "rel_err": round(rel, 5)}


def _capture(op):
    import torch

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            op()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op()
    torch.cuda.synchronize()
    return graph


def _time_graph(graph):
    import triton

    # do_bench zeroes a 256 MB buffer before every call: cold L2.
    times = triton.testing.do_bench(
        graph.replay, warmup=100, rep=1000, return_mode="all"
    )
    return {
        "time_us_median": round(1e3 * statistics.median(times), 2),
        "time_us_min": round(1e3 * min(times), 2),
        "samples": len(times),
    }


def _bench_one(*, name, shape, q, k, v, ref):
    import torch

    cfg = CONFIGS[name]
    row = {
        "config": name,
        **msgspec.structs.asdict(cfg),
        "kv_bytes": kv_bytes(cfg=cfg, shape=shape),
    }
    try:
        with vllm_decode_tile(cfg.decode_tile):
            op, out, info = build_op(cfg=cfg, shape=shape, q=q, k=k, v=v)
            op()
            torch.cuda.synchronize()
            row.update(info)
            if out is not None:
                row.update(_accuracy(out=out, ref=ref))
            graph = _capture(op)
        row["t_start"] = time.time()
        row.update(_time_graph(graph))
        row["t_end"] = time.time()
        row["status"] = "ok"
    except Exception as e:  # one failing config must not stop the others
        traceback.print_exc()
        row.update(status="failed", error=f"{type(e).__name__}: {e}"[:500])
    return row


def _env_versions():
    from importlib import metadata

    import torch

    versions = {"python": sys.executable, "torch_cuda": torch.version.cuda}
    for pkg in ("torch", "triton", "vllm", "sglang"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    versions["device"] = torch.cuda.get_device_name()
    versions["sm_count"] = torch.cuda.get_device_properties(0).multi_processor_count
    return versions


def bench(args):
    import torch

    shape = shape_from_args(args)
    names = [n for n in selected_configs(args) if CONFIGS[n].env == args.env]
    versions = _env_versions()
    if args.expect_gpu not in versions["device"]:
        raise SystemExit(
            f"device is {versions['device']!r}, expected {args.expect_gpu!r}"
        )
    (args.out / f"env_{args.env}.json").write_text(json.dumps(versions, indent=1))
    q, k, v = make_inputs(shape)
    ref = reference_attention(q=q, k=k, v=v, shape=shape)
    torch.cuda.empty_cache()
    with open(args.out / f"bench_{args.env}.jsonl", "a") as f:
        for name in names:
            row = _bench_one(name=name, shape=shape, q=q, k=k, v=v, ref=ref)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps(row), flush=True)


def launch(args):
    import torch

    shape = shape_from_args(args)
    cfg = CONFIGS[args.config]
    q, k, v = make_inputs(shape)
    with vllm_decode_tile(cfg.decode_tile):
        op, _, _ = build_op(cfg=cfg, shape=shape, q=q, k=k, v=v)
        op()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        op()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()


# ----------------------------------------------------------------------------
# Orchestration (run from the SGLang env)
# ----------------------------------------------------------------------------


def _child_env(*, env, gpu):
    child = dict(
        os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID"
    )
    if env == "vllm":
        # The CUDA 13 forward-compat libcuda belongs to the SGLang env only.
        child["LD_LIBRARY_PATH"] = ":".join(
            p
            for p in child.get("LD_LIBRARY_PATH", "").split(":")
            if p and "compat" not in p
        )
    return child


def _python(*, args, env):
    return sys.executable if env == "sglang" else args.vllm_python


def _passthrough_args(args):
    flags = {
        "--batch": args.batch,
        "--seq-len": args.seq_len,
        "--num-q-heads": args.num_q_heads,
        "--num-kv-heads": args.num_kv_heads,
        "--head-dim": args.head_dim,
        "--expect-gpu": args.expect_gpu,
    }
    return [x for flag, value in flags.items() for x in (flag, str(value))]


def _note(out, text):
    print(text, flush=True)
    with open(out / "run_notes.txt", "a") as f:
        f.write(text + "\n")


def _run_bench(args, *, env, gpu_uuid):
    from m0_bench import _compute_pids

    names = [n for n in selected_configs(args) if CONFIGS[n].env == env]
    if not names:
        return
    cmd = [
        _python(args=args, env=env),
        __file__,
        "bench",
        "--env",
        env,
        "--out",
        str(args.out),
        "--configs",
        ",".join(names),
        *_passthrough_args(args),
    ]
    before = _compute_pids(gpu_uuid)
    with open(args.out / f"bench_{env}.log", "w") as log:
        rc = subprocess.run(
            cmd, env=_child_env(env=env, gpu=args.gpu), stdout=log, stderr=log
        ).returncode
    foreign = before | _compute_pids(gpu_uuid)
    if rc != 0:
        _note(args.out, f"bench {env}: exit code {rc}, see bench_{env}.log")
    if foreign:
        _note(args.out, f"bench {env}: other processes on the GPU: {sorted(foreign)}")


def _ncu_export(*, ncu, report, extra, path):
    res = subprocess.run(
        [ncu, "--import", str(report), *extra], capture_output=True, text=True
    )
    path.write_text(res.stdout if res.returncode == 0 else res.stdout + res.stderr)
    return res.returncode == 0


def _run_ncu(args, *, name):
    cfg = CONFIGS[name]
    ncu_dir = args.out / "ncu"
    ncu_dir.mkdir(exist_ok=True)
    report = ncu_dir / f"{name}.ncu-rep"
    cmd = [
        args.ncu,
        "--profile-from-start",
        "off",
        "--set",
        "full",
        "-f",
        "-o",
        str(report.with_suffix("")),
        _python(args=args, env=cfg.env),
        __file__,
        "launch",
        "--config",
        name,
        *_passthrough_args(args),
    ]
    with open(ncu_dir / f"{name}.log", "w") as log:
        rc = subprocess.run(
            cmd,
            env=_child_env(env=cfg.env, gpu=args.gpu),
            stdout=log,
            stderr=log,
            timeout=1800,
        ).returncode
    if rc != 0 or not report.exists():
        _note(args.out, f"ncu {name}: exit code {rc}, see ncu/{name}.log")
        return
    units = ["--csv", "--print-units", "base"]
    _ncu_export(
        ncu=args.ncu,
        report=report,
        extra=["--page", "details", *units],
        path=ncu_dir / f"{name}_details.csv",
    )
    _ncu_export(
        ncu=args.ncu,
        report=report,
        extra=["--page", "raw", *units],
        path=ncu_dir / f"{name}_raw.csv",
    )
    ok = _ncu_export(
        ncu=args.ncu,
        report=report,
        extra=[
            "--page",
            "source",
            "--csv",
            "--print-source",
            "sass",
            "--kernel-name",
            f"regex:{_main_kernel(cfg)}",
        ],
        path=ncu_dir / f"{name}_sass.csv",
    )
    if not ok:
        _note(args.out, f"ncu {name}: SASS export failed, see ncu/{name}_sass.csv")


def _write_run_env(args, *, gpu_name):
    from m0_bench import _gpu_query, _run_text

    driver, power, sm_max, mem_max = _gpu_query(
        args.gpu, "driver_version,power.limit,clocks.max.sm,clocks.max.mem"
    )
    env = {
        "gpu_index": args.gpu,
        "gpu_name": gpu_name,
        "driver": driver,
        "power_limit_w": power,
        "sm_clock_max": sm_max,
        "mem_clock_max": mem_max,
        "sglang_commit": _run_text(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "ncu": (_run_text([args.ncu, "--version"]).splitlines() or ["?"])[-1],
        "cuda_forward_compat": "compat" in os.environ.get("LD_LIBRARY_PATH", ""),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (args.out / "env.json").write_text(json.dumps(env, indent=1))


def run(args):
    from m0_bench import Telemetry, _gpu_query, _wait_gpu_free

    args.out.mkdir(parents=True, exist_ok=True)
    gpu_name, gpu_uuid = _gpu_query(args.gpu, "name,uuid")
    if args.expect_gpu not in gpu_name:
        raise SystemExit(
            f"GPU {args.gpu} is {gpu_name!r}, expected {args.expect_gpu!r}"
        )
    if not Path(args.vllm_python).exists():
        raise SystemExit(f"vLLM python not found: {args.vllm_python}")
    _write_run_env(args, gpu_name=gpu_name)
    _wait_gpu_free(gpu=args.gpu, gpu_uuid=gpu_uuid, timeout_s=3600)
    telemetry = Telemetry(
        gpu=args.gpu, path=args.out / "telemetry.csv", interval_ms=200
    )
    try:
        for env in ("sglang", "vllm"):
            _run_bench(args, env=env, gpu_uuid=gpu_uuid)
    finally:
        telemetry.stop()
    if not args.skip_ncu:
        for name in selected_configs(args):
            _run_ncu(args, name=name)
    write_report(args.out, peak_gbps=args.peak_gbps)


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

_TIME_US = {"nsecond": 1e-3, "usecond": 1.0, "msecond": 1e3, "second": 1e6}
_DETAIL_METRICS = {
    "Duration": "duration",
    "DRAM Throughput": "dram_pct",
    "Compute (SM) Throughput": "sm_pct",
    "Achieved Occupancy": "occupancy_pct",
    "Registers Per Thread": "regs",
    "Grid Size": "grid",
    "Block Size": "block",
    "Warp Cycles Per Issued Instruction": "cycles_per_issue",
    "SM Frequency": "sm_freq",
    "L2 Hit Rate": "l2_hit_pct",
}
_STALL = re.compile(
    r"smsp__average_warps_issue_stalled_(\w+?)_per_issue_active\.ratio$"
)


def _float(text):
    try:
        return float(str(text).replace(",", ""))
    except ValueError:
        return None


def _csv_rows(path):
    """Rows of an ncu CSV export, skipping any ==PROF== lines before the header."""
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text().splitlines() if ln.startswith('"')]
    return list(csv.reader(io.StringIO("\n".join(lines))))


def _short_kernel(name):
    name = name.removeprefix("void ")
    return name.split("(")[0].split("<")[0][-48:]


def _ncu_details(path):
    rows = _csv_rows(path)
    if not rows:
        return {}
    col = {h: i for i, h in enumerate(rows[0])}
    needed = ("ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value")
    if any(c not in col for c in needed):
        return {}
    kernels = {}
    for r in rows[1:]:
        kid = r[col["ID"]]
        k = kernels.setdefault(kid, {"name": _short_kernel(r[col["Kernel Name"]])})
        key = _DETAIL_METRICS.get(r[col["Metric Name"]])
        if key is None or key in k:
            continue
        value, unit = _float(r[col["Metric Value"]]), r[col["Metric Unit"]]
        if key == "duration" and value is not None:
            value *= _TIME_US.get(unit, float("nan"))
        k[key] = value if value is not None else r[col["Metric Value"]]
    return kernels


def _ncu_raw(path):
    rows = _csv_rows(path)
    if len(rows) < 3:
        return {}
    head = rows[0]
    out = {}
    # Row 1 holds the units; one row per kernel follows.
    for r in rows[2:]:
        rec = dict(zip(head, r))
        stalls = {}
        for metric, value in rec.items():
            m = _STALL.match(metric)
            if m and _float(value):
                stalls[m.group(1)] = _float(value)
        sectors = _float(rec.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"))
        requests = _float(rec.get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"))
        out[rec.get("ID")] = {
            "dram_read_mb": (_float(rec.get("dram__bytes_read.sum")) or 0) / 1e6,
            "sectors_per_req": sectors / requests if sectors and requests else None,
            "stalls": dict(sorted(stalls.items(), key=lambda x: -x[1])[:4]),
        }
    return out


def _sass_mix(path):
    """Warp-level executed instructions per opcode for the main kernel."""
    rows = _csv_rows(path)
    # A "Kernel Name" line precedes the column header.
    starts = [i for i, r in enumerate(rows) if r[:2] == ["Address", "Source"]]
    if not starts:
        return {}
    head = rows[starts[0]]
    if "Instructions Executed" not in head:
        return {}
    i_src, i_exec = head.index("Source"), head.index("Instructions Executed")
    mix = {}
    for r in rows[starts[0] + 1 :]:
        if len(r) <= max(i_src, i_exec) or r[i_src] == "Source":
            continue
        tokens = [t for t in r[i_src].split() if not t.startswith("@")]
        count = _float(r[i_exec])
        if tokens and count:
            mix[tokens[0]] = mix.get(tokens[0], 0) + count
    return mix


def _bench_rows(out):
    rows = []
    for env in ("sglang", "vllm"):
        path = out / f"bench_{env}.jsonl"
        if path.exists():
            rows += [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    order = list(CONFIGS)
    return sorted(rows, key=lambda r: order.index(r["config"]))


def _telemetry_cells(out, row):
    from m0_bench import summarize_telemetry

    path = out / "telemetry.csv"
    if not path.exists() or "t_start" not in row:
        return "- | -"
    t = summarize_telemetry(path, row["t_start"] - 0.2, row["t_end"] + 0.2)
    if not t["samples"]:
        return "- | -"
    return f"{t['sm_clock_median']:.0f} | {100 * t['power_cap_frac']:.0f}%"


def _timing_table(out, rows, peak_gbps):
    ok = {r["config"]: r for r in rows if r.get("status") == "ok"}
    ref = ok.get("ref-sum")
    base = ok.get("sglang-bf16")
    lines = [
        "| config | KV MB | op us (median) | GB/s | % of peak | % of ref-sum "
        "| speed vs sglang-bf16 | cos vs fp32 | SM MHz | power cap | notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    ref_gbps = _gbps(ref) if ref else None
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"| {r['config']} | | | | | | | | | | {r.get('error', '')} |")
            continue
        gbps = _gbps(r)
        vs_ref = f"{100 * gbps / ref_gbps:.0f}%" if ref_gbps else "-"
        vs_base = (
            f"{base['time_us_median'] / r['time_us_median']:.2f}x" if base else "-"
        )
        notes = [
            f"{k}={r[k]}"
            for k in ("kv_splits", "decode_tile_used", "vllm_path", "layout")
            if k in r and (k != "layout" or r["impl"] == "vllm")
        ]
        lines.append(
            f"| {r['config']} | {r['kv_bytes'] / 1e6:.0f} | {r['time_us_median']:.1f} "
            f"| {gbps:.0f} | {100 * gbps / peak_gbps:.0f}% | {vs_ref} | {vs_base} "
            f"| {r.get('cos', '-')} | {_telemetry_cells(out, r)} | {', '.join(notes)} |"
        )
    return lines


def _gbps(row):
    return row["kv_bytes"] / row["time_us_median"] / 1e3


def _ncu_tables(out):
    ncu_dir = out / "ncu"
    kernel_lines = [
        "| config | kernel | grid | us | DRAM % | DRAM read MB | L2 hit % | SM % "
        "| occupancy % | regs | sectors/ld req | top stalls (cycles per issue) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    mix_lines = []
    for name, cfg in CONFIGS.items():
        details = _ncu_details(ncu_dir / f"{name}_details.csv")
        if not details:
            continue
        raw = _ncu_raw(ncu_dir / f"{name}_raw.csv")
        for kid, k in details.items():
            x = raw.get(kid, {})
            stalls = ", ".join(f"{s} {v:.1f}" for s, v in x.get("stalls", {}).items())
            spr = x.get("sectors_per_req")
            kernel_lines.append(
                f"| {name} | {k['name']} | {_fmt(k.get('grid'), '{:.0f}')} "
                f"| {_fmt(k.get('duration'), '{:.1f}')} | {_fmt(k.get('dram_pct'), '{:.0f}')} "
                f"| {_fmt(x.get('dram_read_mb'), '{:.0f}')} | {_fmt(k.get('l2_hit_pct'), '{:.0f}')} "
                f"| {_fmt(k.get('sm_pct'), '{:.0f}')} | {_fmt(k.get('occupancy_pct'), '{:.0f}')} "
                f"| {_fmt(k.get('regs'), '{:.0f}')} | {_fmt(spr, '{:.1f}')} | {stalls} |"
            )
        mix = _sass_mix(ncu_dir / f"{name}_sass.csv")
        if mix:
            total = sum(mix.values())
            loads = {op: n for op, n in mix.items() if op.startswith(("LDG", "LDGSTS"))}
            top = sorted(mix.items(), key=lambda x: -x[1])[:10]
            mix_lines.append(
                f"| {name} | {_main_kernel(cfg)} | {total:,.0f} "
                f"| {', '.join(f'{op} {n:,.0f}' for op, n in sorted(loads.items()))} "
                f"| {', '.join(f'{op} {100 * n / total:.0f}%' for op, n in top)} |"
            )
    if mix_lines:
        mix_lines = [
            "| config | kernel | warp instructions | global loads by opcode "
            "| top opcodes (share of executed) |",
            "|---|---|---:|---|---|",
            *mix_lines,
        ]
    return kernel_lines, mix_lines


def _fmt(x, spec):
    if x is None:
        return "-"
    return spec.format(x) if isinstance(x, (int, float)) else str(x)


def write_report(out, *, peak_gbps):
    rows = _bench_rows(out)
    env = (
        json.loads((out / "env.json").read_text())
        if (out / "env.json").exists()
        else {}
    )
    lines = [
        "# M3.0 decode-attention kernel diagnosis",
        "",
        f"- GPU {env.get('gpu_name', '?')}, driver {env.get('driver', '?')}, "
        f"power limit {env.get('power_limit_w', '?')} W; peak used for %: {peak_gbps:.0f} GB/s",
        f"- SGLang `{str(env.get('sglang_commit', '?'))[:12]}`; ncu {env.get('ncu', '?')}",
    ]
    for e in ("sglang", "vllm"):
        p = out / f"env_{e}.json"
        if p.exists():
            v = json.loads(p.read_text())
            lines.append(
                f"- {e} env: torch {v['torch']} (CUDA {v['torch_cuda']}), triton {v['triton']}, "
                f"vllm {v['vllm']}, sglang {v['sglang']}"
            )
    a = env.get("args", {})
    lines += [
        f"- Shape: {a.get('batch', '?')} seqs x {a.get('seq_len', '?')} tokens, "
        f"Q/KV heads {a.get('num_q_heads', '?')}/{a.get('num_kv_heads', '?')}, "
        f"head_dim {a.get('head_dim', '?')}",
        "- Timing: CUDA graph replay of the whole op (all kernels), triton do_bench with "
        "a cold L2; GB/s counts the KV bytes as stored (data plus scales).",
        "",
        "## Timing",
        "",
        *_timing_table(out, rows, peak_gbps),
        "",
    ]
    kernel_lines, mix_lines = _ncu_tables(out)
    if len(kernel_lines) > 2:
        lines += [
            "## Nsight Compute (one call, --set full, clocks locked to base)",
            "",
            *kernel_lines,
            "",
        ]
    if mix_lines:
        lines += ["## Main-kernel instruction mix (SASS)", "", *mix_lines, ""]
    notes = out / "run_notes.txt"
    if notes.exists():
        lines += [
            "## Run notes",
            "",
            *[f"- {x}" for x in notes.read_text().splitlines()],
            "",
        ]
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print(text)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def shape_from_args(args):
    return Shape(
        batch=args.batch,
        seq_len=args.seq_len,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
    )


def selected_configs(args):
    if not args.configs:
        return list(CONFIGS)
    names = args.configs.split(",")
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; choose from {list(CONFIGS)}")
    return names


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("command", choices=["run", "bench", "launch", "report"])
    p.add_argument("--out", type=Path)
    p.add_argument("--gpu", type=int, help="nvidia-smi index (PCI order); run only")
    p.add_argument("--env", choices=["sglang", "vllm"], help="bench only")
    p.add_argument("--config", choices=list(CONFIGS), help="launch only")
    p.add_argument("--configs", default="", help="comma-separated subset")
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=16384)
    p.add_argument("--num-q-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--expect-gpu", default="A100-PCIE-40GB")
    p.add_argument("--peak-gbps", type=float, default=DEFAULT_PEAK_GBPS)
    p.add_argument(
        "--vllm-python",
        default=str(Path(os.environ.get("KVQ_WORK", "")) / "envs/vllm/bin/python"),
    )
    p.add_argument("--ncu", default="ncu")
    p.add_argument("--skip-ncu", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    if args.command == "launch":
        launch(args)
        return
    if args.out is None:
        raise SystemExit("--out is required")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        if args.gpu is None:
            raise SystemExit("run needs --gpu")
        run(args)
    elif args.command == "bench":
        bench(args)
    else:
        write_report(args.out, peak_gbps=args.peak_gbps)


if __name__ == "__main__":
    main()
