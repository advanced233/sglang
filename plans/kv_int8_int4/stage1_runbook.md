# 阶段 1 实验手册：kernel 诊断（M3.0）与 INT8 配方冒烟（M2）

本手册给在 A100 服务器上执行实验的本地 agent 使用，对应 [`plan.md`](plan.md) 的 M3.0 和 M2。共享服务器规则沿用 [`env_setup.md`](env_setup.md) 第 0 节，结果按 `env_setup.md` 步骤 10 脱敏后推送。

一次 GPU 会话做三件事，按顺序执行：
1. **M2 测试**（约 3 分钟）：INT8 配方的单测和 kernel 测试。
2. **M3.0 kernel 诊断**（约 15 分钟）：在同一个 decode 形状上对比 SGLang 和 vLLM 的 attention kernel，并用 ncu 各采一次。
3. **M2 冒烟**（约 25 分钟）：BF16 和 INT8 各起一次服务，看 KV 字节数、单步时间和 GSM8K。

合计约 45 分钟（粗估）。三项互相独立：某一项失败，记录后继续下一项。只有一个例外：第 1 步的 GPU 测试失败时，跳过第 3 步（服务的结果会没有意义），第 2 步照常执行。

用到的代码：
- `benchmark/kv_lowbit/m3_kernel_diag.py`：M3.0 驱动。自己在 SGLang 和 vLLM 两个环境里分别计时，再逐个配置调用 `ncu`，最后写 `summary.md`。
- `benchmark/kv_lowbit/m2_smoke.py`：M2 冒烟驱动。自己启停服务、调用 `bench_serving` 和 sgl-eval，最后写 `summary.md`。
- INT8 配方本身：`python/sglang/kernels/ops/kvcache/int8_per_token_head.py`、`python/sglang/srt/layers/quantization/fp4_kv_cache_quant_method.py` 里的 `Int8PerTokenHeadKVCacheMethod`，以及 `--kv-cache-dtype int8_per_token_head` 的相关接线。

**不要修改这些代码。** 报错的 Traceback 指向上面这些文件时，说明是代码本身的 bug：记录完整报错，停止这一项，在回报里说明，其余项照常执行。

## 0. 每次开始前

```bash
source "$KVQ_WORK/env.sh" && kvq_env sglang
cd "$KVQ_WORK/src/sglang" && git pull --rebase origin kv-int8-int4 && git log --oneline -1
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
GPU=0                          # nvidia-smi 里 A100-PCIE-40GB 的 index（PCI 顺序）
TS=$(date +%Y%m%d-%H%M%S)      # 这一轮所有输出共用这个时间戳
which ncu && ncu --version | tail -1
```

- 代码是 editable 安装，`git pull` 之后不用重装。
- `ncu` 用环境配置时确认过的系统版本（2024.1.0）。`which ncu` 找不到时，第 2 步加 `--ncu <ncu 的绝对路径>`。

## 1. M2 测试（约 3 分钟）

```bash
python -m pytest -q test/registered/unit/layers/quantization/test_fp4_kv_cache_quant_method.py \
  test/registered/unit/server_args/test_resolution_hook_registry.py \
  2>&1 | tee "$KVQ_WORK/logs/stage1_cpu_tests_$TS.log"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU python -m pytest -q \
  test/registered/kernels/ops/kvcache/test_int8_per_token_head.py \
  test/registered/unit/mem_cache/test_int8_kv_pool.py \
  2>&1 | tee "$KVQ_WORK/logs/stage1_gpu_tests_$TS.log"
```

- 两条命令都应该全部通过。第一条不用 GPU。
- 第二条有失败时：记下失败的用例和报错，**跳过第 3 步**，第 2 步照常执行。

## 2. M3.0 kernel 诊断（一条命令，约 15 分钟）

在 SGLang 环境里启动；脚本自己用 `$KVQ_WORK/envs/vllm/bin/python` 跑 vLLM 的部分。

```bash
python benchmark/kv_lowbit/m3_kernel_diag.py run --gpu $GPU \
  --out "$KVQ_WORK/outputs/m3diag/$TS" 2>&1 | tee "$KVQ_WORK/logs/m3diag_$TS.log"
```

脚本做的事：
- **形状**：6 条序列 × 16k 上下文，32 个 Q head、8 个 KV head、head_dim 128，和阶段 0 的服务测量点一致。
- **12 个配置**：

  | 配置 | 内容 |
  |---|---|
  | `ref-sum` | 对和 BF16 KV 一样大的数据做一次求和，作为这张卡实际能达到的读带宽 |
  | `sglang-bf16`、`sglang-fp8e5m2` | SGLang Triton 后端的 decode kernel |
  | `vllm-bf16`、`vllm-int8`、`vllm-int4` | vLLM `TRITON_ATTN` 的 decode kernel，参数与服务时相同 |
  | `vllm-bf16-t32`、`vllm-int8-t32`、`vllm-int8-t64`、`vllm-int4-t64` | 消融：把 vLLM 的 decode tile 从 16 改成 32 / 64 个 token |
  | `vllm-int8-sep`、`vllm-int8-sep-t64` | 消融：scale 不内联在 page 里，而是放进单独的对齐 buffer |

- **计时**：每个配置先检查输出和 fp32 参考结果的余弦相似度，再用 CUDA graph 重放、冷 L2 计时。两个环境依次计时，期间每 200 ms 记录一次 GPU 时钟和功耗。
- **ncu**：每个配置单独起一个进程，用 `--set full` 采一次完整调用（包括 split-KV 的归约和 INT4 的 RHT）。每个配置约 1 分钟。
- **结果**：`summary.md`（计时表、ncu 指标表、主 kernel 的 SASS 指令分布），以及 `bench_*.jsonl`、`ncu/*_details.csv`、`ncu/*_raw.csv`、`ncu/*_sass.csv`、`ncu/*.log`。

预期之内、不用处理的现象：
- `cos vs fp32` 一列：BF16 接近 1；`sglang-fp8e5m2` 和 `vllm-int4` 明显低一些。这一列只用来确认调用参数没错。
- ncu 的耗时比计时表长：ncu 把时钟锁在基础频率。
- `run_notes.txt` 里如果出现 `SASS export failed`，说明这个 ncu 版本导不出 SASS 页面。其余表不受影响，照常推送。

某个配置失败时（`bench_*.log` 里有 Traceback，或 `run_notes.txt` 里有 `exit code`），用 `--configs` 只补跑失败的配置，**换一个新的输出目录**，最后两个目录都推送：

```bash
python benchmark/kv_lowbit/m3_kernel_diag.py run --gpu $GPU --configs vllm-int4,vllm-int4-t64 \
  --out "$KVQ_WORK/outputs/m3diag/$TS-retry" 2>&1 | tee "$KVQ_WORK/logs/m3diag_${TS}_retry.log"
```

## 3. M2 冒烟（一条命令，约 25 分钟）

先在 sgl-eval 环境里核对采样参数的写法：

```bash
kvq_env sgleval && sgl-eval run --help | grep -E -- "--(temperature|top-p|top-k|chat-template-kwarg|num-examples)" ; kvq_env sglang
```

脚本默认传给 sgl-eval 的参数是 `--chat-template-kwarg enable_thinking=false --temperature 0.7 --top-p 0.8 --top-k 20`（Qwen3 非思考模式的推荐值）。如果 `--help` 里的写法不同，用 `--eval-args "<改好的参数>"` 传入，并在回报里写明。

```bash
python benchmark/kv_lowbit/m2_smoke.py --model "$QWEN3_8B" --gpu $GPU \
  --out "$KVQ_WORK/outputs/m2/$TS" 2>&1 | tee "$KVQ_WORK/logs/m2_$TS.log"
```

脚本做的事，对 BF16 和 INT8 各一遍：
- 起服务：`--attention-backend triton --max-total-tokens 100000`，INT8 另加 `--kv-cache-dtype int8_per_token_head`。
  - 池子限制在 10 万 token，是因为 INT8 现在的读取方式会把一整层的池子反量化成 BF16；按默认显存比例分配的池子，这份临时拷贝放不下（`plan.md` M2）。
- 从服务日志读出 KV 实际分配的字节数，换算成"每 token 每层字节数"。预期 BF16 约 4096，INT8 约 2112。
- 发一条短请求（17 × 23），答案应为 391。
- decode 探测：1k 输入、256 输出、并发 1、8 个请求，记录 ITL 中位数。
- GSM8K 全集（1319 题），sgl-eval 在自己的环境里运行。

结果：`summary.md`、`runs.jsonl`、`servers/*.log`、`eval/*.log`、`bench/*.jsonl`、`telemetry/*.csv`。

预期之内、不用处理的现象：
- INT8 的 decode 探测比 BF16 慢：每一步都要整层反量化，这是 M3 要解决的问题。
- 两个配置的 GSM8K 分数差 1–2 分以内都属于采样噪声。

## 4. 自查（只看是否跑完，不判断结果好坏）

- 第 1 步两条命令的通过 / 失败数。
- M3.0 的 `summary.md`：计时表 12 行是否都有数字；ncu 表是否每个配置都有；有没有 `run_notes.txt`。
- M2 的 `summary.md`：两行的 ready 是否都是 True；GSM8K 是否都跑完（examples 为 1319）。
- M3.0 计时期间同卡有没有出现别的进程（写在 `run_notes.txt`）。

## 5. 推送

按 `env_setup.md` 步骤 10 执行，结果目录的内容换成下面这些：

```bash
source "$KVQ_WORK/env.sh" && cd "$KVQ_WORK/src/sglang"
git pull --rebase origin kv-int8-int4
DEST="plans/kv_int8_int4/results/a100-40g/$TS"
mkdir -p "$DEST/logs"
for d in "$KVQ_WORK"/outputs/m3diag/$TS*; do
  cp -r "$d" "$DEST/m3diag$(basename "$d" | sed "s/^$TS//")"
done
find "$DEST" -name "*.ncu-rep" -delete              # ncu 报告只留在本地
[ -d "$KVQ_WORK/outputs/m2/$TS" ] && cp -r "$KVQ_WORK/outputs/m2/$TS" "$DEST/m2"
rm -rf "$DEST/m2/profiles"
cp "$KVQ_WORK"/logs/stage1_*_$TS.log "$KVQ_WORK"/logs/m3diag_$TS*.log "$KVQ_WORK"/logs/m2_$TS.log "$DEST/logs/" 2>/dev/null
du -sh "$DEST"; find "$DEST" -type f -size +5M
```

- 超过 5 MB 的 `.log` 按步骤 10 截断。超过 5 MB 的 `.csv`（可能是 `ncu/*_sass.csv` 或 `ncu/*_raw.csv`）不要截断：超过 20 MB 的不推送，在 README 里写明文件名和大小。
- ncu 的 CSV 里有主机名和进程路径，步骤 10.2 的脱敏会处理，推送前的检查照常执行。

再写一个 `$DEST/README.md`，内容只写事实：
- 三步各自是否跑完，总耗时；
- 第 1 步的通过 / 失败数，失败用例的名字；
- 失败项、补跑项和原因；
- 第 3 步实际使用的 sgl-eval 参数；
- 自查中发现的异常。

然后按步骤 10.2 脱敏、检查，按 10.3 提交，提交信息写 `results: A100-40G stage 1 $TS`。
