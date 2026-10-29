# 阶段 0 实验手册：A100-40G 上的服务基线与离线误差

本手册给在 A100 服务器上执行实验的本地 agent 使用，对应 [`plan.md`](plan.md) 的 M0 和 M1。共享服务器规则沿用 [`env_setup.md`](env_setup.md) 第 0 节，结果按 `env_setup.md` 步骤 10 脱敏后推送。

这是写进简历的项目，实验只做到能支撑结论为止：
- M0 一条命令、M1 两条命令，脚本自己管理服务的启停。
- 不单独做冒烟和预热，第一组正式运行就是冒烟。
- 不单独测热降频：M0 运行期间脚本会每秒自动记录 GPU 温度、时钟和降频原因。
- 合计约 3 小时（粗估）。

用到的代码（都在 `benchmark/kv_lowbit/`）：
- `m0_bench.py`：M0 驱动。按组依次启动服务、压测、停止服务，写出 `runs.jsonl`、`summary.md`、服务日志、遥测 CSV 和压测原始结果。
- `m0_report.py`：从 `runs.jsonl` 生成 `summary.md`，由 `m0_bench.py` 自动调用。
- `m1_offline_error.py`：M1 离线误差，用 HF transformers 直接跑模型，不启动服务。
- `kvq_ref.py`：M1 用的参考量化器。

**不要修改这些代码。** 如果报错的 Traceback 指向 `benchmark/kv_lowbit/` 里的文件，说明是脚本本身的 bug：记录完整报错，停止这一项，在回报里说明，其余项照常执行。

## 0. 每次开始前

```bash
source "$KVQ_WORK/env.sh" && kvq_env sglang
cd "$KVQ_WORK/src/sglang" && git pull --rebase origin kv-int8-int4 && git log --oneline -1
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
GPU=0                          # nvidia-smi 里 A100-PCIE-40GB 的 index（PCI 顺序）
TS=$(date +%Y%m%d-%H%M%S)      # 这一轮所有输出共用这个时间戳
```

- `--gpu` 用的是 `nvidia-smi` 的 index。`m0_bench.py` 会给服务进程设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 和 `CUDA_VISIBLE_DEVICES`，并核对型号是不是 `A100-PCIE-40GB`，型号不符就直接退出。
- 这台机器还有两张 A100 80GB，不要用。

## 1. dry-run（1 分钟）

```bash
python benchmark/kv_lowbit/m0_bench.py --model "$QWEN3_8B" --gpu $GPU \
  --out "$KVQ_WORK/outputs/m0/dryrun-$TS" --dry-run | head -40
```

只打印将要执行的服务和压测命令，不占 GPU。确认模型路径和 vLLM 的路径（默认 `$KVQ_WORK/envs/vllm/bin/vllm`）都对。dry-run 里的并发数是按一个假定容量算的，正式运行时按实际容量算。

## 2. M0（一条命令，约 2–2.5 小时）

```bash
python benchmark/kv_lowbit/m0_bench.py --model "$QWEN3_8B" --gpu $GPU \
  --out "$KVQ_WORK/outputs/m0/$TS" --groups main,pool,profile,vllm \
  2>&1 | tee "$KVQ_WORK/logs/m0_$TS.log"
```

四个组按顺序执行：

| 组 | 内容 | 约耗时 |
|---|---|---|
| `main` | SGLang 四组配置：BF16（flashinfer）、BF16（triton）、`fp8_e5m2`（triton）、`fp4_mx_block16`（triton，显存比例 0.7）。输入 4k / 16k / 32k，输出 256，并发 1、8、32、最大值 | 1 小时 |
| `pool` | 只改池子大小（`--max-total-tokens`）：BF16 测 4 万 / 10 万，FP4 测 4 万 / 10 万 / 20 万 / 30 万；输入 1024，输出 256，并发 1 | 40 分钟 |
| `profile` | BF16 和 FP4 各一次：16k 上下文、6 个请求的 decode，torch profiler 采 5 步，输出按 kernel 拆分的耗时表 | 15 分钟 |
| `vllm` | vLLM 的 BF16 / INT8 / INT4（`TRITON_ATTN`），扫描范围同 `main` | 40 分钟 |

脚本自动处理的事：
- **并发上限**：最大并发 = min(32, 容量 ÷ (输入 + 256))。32 是这张卡上 decode CUDA graph 的上限，见 plan 的 M0。放不下一个请求的组合记为 `skipped`。
- **等 GPU**：每次启动服务前确认 GPU 空闲；被占用就每 10 分钟查一次，最多等 1 小时，仍然被占就退出。
- **只停自己的进程**：服务只监听 `127.0.0.1` 的空闲端口，放在单独的进程组里，停止时只给这个进程组发信号。
- **外部进程**：压测期间同卡出现别的进程，那一次自动重跑一遍；还有就保留标记（`summary.md` 的 note 列）。
- **失败不中断**：某个服务起不来或某次压测失败，记下报错后继续下一项。

预期之内、不用处理的现象：
- `fp4_mx_block16` 明显比 BF16 慢，`pool` 组里它的 TPOT 随池子变大而增加。这正是要测的问题。
- vLLM 组的输入长度不是精确值：vLLM 只能收文本，random-ids 解码成文本后再分词，长度会有偏差。SGLang 组发的是 token id，长度精确。
- `profile` 组的 trace 文件很大，只留在本地；推送的是 `profiles/*_decode_tables.txt`。

中途被打断时，用同一个 `--out` 只补跑缺的组或配置（记录会追加到同一个 `runs.jsonl`），最后重新生成汇总：

```bash
python benchmark/kv_lowbit/m0_bench.py --model "$QWEN3_8B" --gpu $GPU \
  --out "$KVQ_WORK/outputs/m0/$TS" --groups main --configs sglang-fp4mx-triton
python benchmark/kv_lowbit/m0_bench.py --model "$QWEN3_8B" --gpu $GPU \
  --out "$KVQ_WORK/outputs/m0/$TS" --summarize-only
```

## 3. M1（两条命令，约 30–40 分钟）

M1 不启动服务，直接在 GPU 上跑 HF 模型。必须在 M0 结束之后再跑，不要和 M0 抢同一张卡。开跑前用 `nvidia-smi` 确认卡是空的。

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
python benchmark/kv_lowbit/m1_offline_error.py --model "$QWEN3_8B" --label qwen3-8b \
  --out "$KVQ_WORK/outputs/m1/$TS" 2>&1 | tee "$KVQ_WORK/logs/m1_qwen3_$TS.log"
python benchmark/kv_lowbit/m1_offline_error.py --model "$LLAMA31_8B" --label llama31-8b \
  --out "$KVQ_WORK/outputs/m1/$TS" --per-layer-scheme "" 2>&1 | tee "$KVQ_WORK/logs/m1_llama_$TS.log"
unset CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES
```

- 默认参数：前缀 4096 和 16384，后续 128 个 token，每种前缀长度 8 个文本窗口，6 种量化方式。Qwen3 还会做一次"每次只量化一层"的扫描（INT4 + RHT，2 个窗口）。
- 文本来自仓库 `docs/docs` 下的文档，不需要联网。
- 输出 `m1_<label>.json` 和 `m1_<label>.md`。检查 `.md` 开头的 GPU 是不是 `A100-PCIE-40GB`。
- 16k 前缀 OOM 时，改成 `--prefix-lens 4096` 重跑那个模型，并在回报里注明。
- Llama 在环境配置时没有验证过完整加载。加载失败就跳过 Llama，记录报错。

## 4. 自查（只看是否跑完，不判断结果好坏）

- `summary.md` 的 Servers 表：每个服务的 ready 是否都是 yes。
- 有没有 `bench_failed`、`profile_failed`，有没有带 foreign pids 标记的行。
- 遥测列：热降频（thermal）和功耗上限（power cap）是否出现过非 0。
- 两个模型的 M1 `.md` 是否都生成了。

单次失败可以按第 2 节的补跑命令重试一次。重试后仍失败的，记录下来，不再重试。

## 5. 推送

按 `env_setup.md` 步骤 10 执行，只是结果目录的内容换成下面这些：

```bash
source "$KVQ_WORK/env.sh" && cd "$KVQ_WORK/src/sglang"
git pull --rebase origin kv-int8-int4
DEST="plans/kv_int8_int4/results/a100-40g/$TS"
mkdir -p "$DEST/logs"
cp -r "$KVQ_WORK/outputs/m0/$TS" "$DEST/m0"
find "$DEST/m0/profiles" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +   # trace 目录只留在本地
cp -r "$KVQ_WORK/outputs/m1/$TS" "$DEST/m1"
cp "$KVQ_WORK"/logs/m0_$TS.log "$KVQ_WORK"/logs/m1_*_$TS.log "$DEST/logs/"
du -sh "$DEST"; find "$DEST" -type f -size +5M   # 超过 5 MB 的日志按步骤 10 截断
```

再写一个 `$DEST/README.md`，内容只写事实：
- 每个组和每个 M1 模型是否跑完，总耗时；
- 失败项、重试项和原因；
- 自查中发现的异常（例如出现热降频或外部进程）。

然后按步骤 10.2 脱敏、检查，按 10.3 提交，提交信息写 `results: A100-40G stage 0 $TS`。
