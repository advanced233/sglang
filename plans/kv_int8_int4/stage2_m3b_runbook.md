# 阶段 2 第 2 轮：INT8 接入服务（M3b）

对应 [`plan.md`](plan.md) M3 的 M3b：decode 改用融合 INT8 kernel，extend 只反量化读到的前缀行。共享服务器规则、脱敏和推送沿用 [`env_setup.md`](env_setup.md) 第 0 节和步骤 10。合计约 35 分钟。

**不要修改代码。** 报错就记录完整 Traceback，跳过这一项，继续下一项。第 1 节的 GPU 测试有失败时，跳过第 2、3 节。

## 0. 准备

```bash
source "$KVQ_WORK/env.sh" && kvq_env sglang
cd "$KVQ_WORK/src/sglang" && git pull --rebase origin kv-int8-int4 && git log --oneline -1
export HF_HUB_OFFLINE=1
GPU=0                          # A100-PCIE-40GB 的 nvidia-smi index
TS=$(date +%Y%m%d-%H%M%S)
OUT="$KVQ_WORK/outputs/m3b/$TS"
```

## 1. 测试（约 3 分钟）

```bash
python -m pytest -q test/registered/unit/layers/quantization/test_fp4_kv_cache_quant_method.py \
  2>&1 | tee "$KVQ_WORK/logs/m3b_cpu_tests_$TS.log"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU python -m pytest -q \
  test/registered/kernels/ops/kvcache/test_int8_per_token_head.py \
  test/registered/kernels/ops/attention/test_decode_attention_int8.py \
  test/registered/unit/mem_cache/test_int8_kv_pool.py \
  2>&1 | tee "$KVQ_WORK/logs/m3b_gpu_tests_$TS.log"
```

## 2. 冒烟：默认显存比例（一条命令，约 15 分钟）

```bash
python benchmark/kv_lowbit/m2_smoke.py --model "$QWEN3_8B" --gpu $GPU --max-total-tokens 0 \
  --out "$OUT/smoke" 2>&1 | tee "$KVQ_WORK/logs/m3b_smoke_$TS.log"
```

- BF16 和 INT8 各起一次服务，池子大小由默认显存比例决定。
- 记录：KV 容量（token 数）、每 token 每层字节数、17 × 23 的回答、decode 探测（1k 输入、并发 1）、GSM8K 全集。
- INT8 服务起不来（例如 CUDA graph 捕获时 OOM）是需要回报的结果，不是脚本问题：记下服务日志的最后 50 行。

## 3. 服务测速：16k 输入（一条命令，约 15 分钟）

```bash
python benchmark/kv_lowbit/m0_bench.py --model "$QWEN3_8B" --gpu $GPU --groups m3b \
  --input-lens 16384 --output-len 1024 --concurrency 6,max --no-profile \
  --out "$OUT/serve" 2>&1 | tee "$KVQ_WORK/logs/m3b_serve_$TS.log"
```

- BF16 和 INT8 各起一次服务，并发 6 和各自能放下的最大并发各压测一次，再加一次 decode 探测。16k 输入 + 1024 输出时，BF16 最多放下 6 个请求，INT8 约 13 个。
- 结果在 `$OUT/serve/summary.md`：看 `median ITL ms` 和 `out tok/s` 两列。

## 4. 推送

```bash
source "$KVQ_WORK/env.sh" && cd "$KVQ_WORK/src/sglang"
git pull --rebase origin kv-int8-int4
DEST="plans/kv_int8_int4/results/a100-40g/$TS"
mkdir -p "$DEST/logs"
cp -r "$OUT/smoke" "$DEST/smoke"; cp -r "$OUT/serve" "$DEST/serve"
rm -rf "$DEST"/smoke/profiles "$DEST"/serve/profiles
cp "$KVQ_WORK"/logs/m3b_*_$TS.log "$DEST/logs/"
du -sh "$DEST"; find "$DEST" -type f -size +5M
```

- 超过 5 MB 的日志按步骤 10 截断。
- `README.md` 只写事实：各节是否跑完、测试通过 / 失败数、失败项。
- 按步骤 10.2 脱敏、检查，按 10.3 提交，提交信息写 `results: A100-40G stage 2 M3b $TS`。
- 回报：提交号，以及 `smoke/summary.md` 和 `serve/summary.md` 的表格。
