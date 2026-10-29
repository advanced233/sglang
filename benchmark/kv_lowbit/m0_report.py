"""Render summary.md for an M0 output directory from runs.jsonl and env.json."""

import json
from pathlib import Path


def _rows(out_dir):
    path = Path(out_dir) / "runs.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _num(x, fmt="{:.1f}"):
    return "-" if x is None else fmt.format(x)


def _pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def _env_lines(out_dir):
    path = Path(out_dir) / "env.json"
    if not path.exists():
        return []
    env = json.loads(path.read_text())
    v = env["versions"]
    return [
        f"- GPU {env['gpu_name']} (driver {env['driver']}, power limit {env['power_limit_w']} W, "
        f"max SM {env['sm_clock_max']} MHz); CUDA forward compat: {env['cuda_forward_compat']}",
        f"- SGLang `{env['sglang_commit'][:12]}`, torch {v['torch']} (CUDA {env['torch_cuda']}), "
        f"triton {v['triton']}, flashinfer {v['flashinfer-python']}, "
        f"sglang-kernel {v['sglang-kernel']}",
        f"- vLLM env: {env['vllm']}",
    ]


def _flag(args, name):
    return args[args.index(name) + 1] if name in args else None


def _server_table(rows):
    lines = [
        "| server | ready | load s | KV dtype | KV tokens | backend | mem fraction / util "
        "| decode graph max bs |",
        "|---|---|---:|---|---:|---|---:|---:|",
    ]
    for r in rows:
        args = r["config_args"] + r["extra_args"]
        if r["engine"] == "vllm":
            backend = "TRITON_ATTN"
        else:
            backend = (
                _flag(args, "--attention-backend") or r.get("default_backend") or "-"
            )
        mem = r.get("mem_fraction_static") or r.get("gpu_memory_utilization")
        lines.append(
            f"| {r['tag']} | {'yes' if r['ready'] else '**no**'} | {r['ready_s']} "
            f"| {_flag(args, '--kv-cache-dtype') or 'auto'} "
            f"| {_num(r.get('capacity_tokens'), '{:,}')} | {backend} | {_num(mem, '{}')} "
            f"| {_num(r.get('decode_graph_max_bs'), '{}')} |"
        )
    return lines


def _telemetry_cells(r):
    t = r.get("telemetry") or {}
    return (
        f"{_num(t.get('temp_max'), '{:.0f}')} | {_num(t.get('sm_clock_min'), '{:.0f}')} "
        f"| {_pct(t.get('thermal_frac'))} | {_pct(t.get('power_cap_frac'))}"
    )


def _sweep_table(rows):
    lines = [
        "| config | input | conc | prompts | out tok/s | median TPOT ms | median TTFT ms "
        "| max temp C | min SM MHz | thermal | power cap | note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.get("status") == "skipped":
            lines.append(
                f"| {r['config']} | {r['input_len']} | - | - | - | - | - | - | - | - | - "
                f"| skipped: {r['reason']} |"
            )
            continue
        note = [] if r["status"] == "ok" else [r["status"]]
        if r.get("foreign_pids"):
            note.append(f"foreign pids {r['foreign_pids']}")
        if r.get("retried"):
            note.append("retried")
        lines.append(
            f"| {r['config']} | {r['input_len']} | {r['concurrency']} | {r['num_prompts']} "
            f"| {_num(r.get('output_throughput'))} | {_num(r.get('median_tpot_ms'), '{:.2f}')} "
            f"| {_num(r.get('median_ttft_ms'), '{:.0f}')} | {_telemetry_cells(r)} "
            f"| {', '.join(note)} |"
        )
    return lines


def _pool_table(rows):
    lines = [
        "| config | pool tokens | median TPOT ms | out tok/s | max temp C | min SM MHz | thermal | power cap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['pool_tokens']:,} | {_num(r.get('median_tpot_ms'), '{:.2f}')} "
            f"| {_num(r.get('output_throughput'))} | {_telemetry_cells(r)} |"
        )
    return lines


def render(out_dir):
    rows = _rows(out_dir)
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    lines = ["# M0 serving baselines", "", *_env_lines(out_dir), ""]
    sections = [
        ("server", "Servers", _server_table),
        ("main", "SGLang sweep", _sweep_table),
        ("vllm", "vLLM sweep (text prompts, input lengths approximate)", _sweep_table),
        (
            "pool",
            "Decode TPOT vs pool size (input 1024, output 256, concurrency 1)",
            _pool_table,
        ),
    ]
    for group, title, table in sections:
        if group in by_group:
            lines += [f"## {title}", "", *table(by_group[group]), ""]
    if "profile" in by_group:
        lines += ["## Decode profiles (16k context, 6 requests)", ""]
        lines += [
            f"- {r['tag']}: {r['status']}, tables in `profiles/{r['tables']}`"
            for r in by_group["profile"]
        ]
    return "\n".join(lines) + "\n"


def write_summary(out_dir):
    text = render(out_dir)
    (Path(out_dir) / "summary.md").write_text(text)
    print(text)
