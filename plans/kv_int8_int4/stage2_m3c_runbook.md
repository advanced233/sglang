# 阶段 2 第 3 轮：融合 INT8 extend、确定性模式与 chunk 数值语义（M3c）

对应 [`plan.md`](plan.md) M3 的 M3c：融合 INT8 2-stage extend（`chunk_bf16`）、融合 INT8 1-stage unified extend（`chunk_q` 与 `--enable-deterministic-inference`），以及离线 KL 一致性与 16k 服务 TTFT/ITL 验证。共享服务器规则、脱敏和推送沿用 [`env_setup.md`](env_setup.md) 第 0 节和步骤 10。合计约 35 分钟。

**不要修改代码。** 报错就记录完整 Traceback，跳过这一项，继续下一项。第 1 节的 GPU 测试有失败时，跳过第 2–4 节。

## 0. 准备

注意本轮代码在分支 **`kv-int8-int4-gemini`** 上：

```bash
source "$KVQ_WORK/env.sh" && kvq_env sglang
cd "$KVQ_WORK/src/sglang"
git fetch origin kv-int8-int4-gemini
git checkout -B kv-int8-int4-gemini origin/kv-int8-int4-gemini
git log --oneline -1
export HF_HUB_OFFLINE=1
GPU=0                          # A100-PCIE-40GB 的 nvidia-smi index
TS=$(date +%Y%m%d-%H%M%S)
OUT="$KVQ_WORK/outputs/m3c/$TS"
mkdir -p "$OUT" "$KVQ_WORK/logs"
```

## 1. 测试（约 4 分钟）

```bash
python -m pytest -q test/registered/unit/layers/quantization/test_fp4_kv_cache_quant_method.py \
  2>&1 | tee "$KVQ_WORK/logs/m3c_cpu_tests_$TS.log"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU python -m pytest -q \
  test/registered/kernels/ops/kvcache/test_int8_per_token_head.py \
  test/registered/kernels/ops/attention/test_decode_attention_int8.py \
  test/registered/kernels/ops/attention/test_extend_attention_int8.py \
  test/registered/unit/mem_cache/test_int8_kv_pool.py \
  2>&1 | tee "$KVQ_WORK/logs/m3c_gpu_tests_$TS.log"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU python -m pytest -q \
  test/registered/attention/test_triton_attention_kernels.py \
  -k test_int8_per_token_head_attention_paths \
  2>&1 | tee "$KVQ_WORK/logs/m3c_triton_attn_test_$TS.log"
```

- `test_triton_attention_kernels.py` 只跑新增的 `test_int8_per_token_head_attention_paths`（原文件里 BF16 `head_dim=80` 的旧用例在 sm80 上有已知基线失败）。
- 新增用例覆盖 GQA grouped decode、MHA decode、regular extend 前缀、unified extend，以及 unified extend 的前缀切分逐位不变性（bit-exact）。

## 2. Extend kernel 微基准（约 5 分钟）

```bash
python benchmark/kv_lowbit/m3c_extend_eval.py kernel --gpu $GPU \
  --out "$OUT/kernel" 2>&1 | tee "$KVQ_WORK/logs/m3c_kernel_$TS.log"
```

- 冷 L2，Qwen3-8B 头结构（32 Q / 8 KV / head_dim 128），测 3 个代表形状：
  - `prefix-hit`（4×8192+512）：radix 前缀命中
  - `late-chunk`（1×14336+2048）：16k chunked prefill 尾块（对应 M3b 的 +4% TTFT）
  - `fresh-prefill`（4×0+4096）：无前缀 prefill（对比 `chunk_bf16` 与 `chunk_q` 的 Stage 2 开销）
- 同一份输入对比 5 个实现：`bf16-regular`、`bf16-unified`、`int8-m3b-gather`、`int8-fused-regular`、`int8-fused-unified`，同时记录临时显存（MB）和相对 fp32 参考的精度。
- 结果在 `$OUT/kernel/summary.md`。

## 3. 离线 KL 一致性与确定性模式评测（约 15 分钟）

```bash
python benchmark/kv_lowbit/m3c_extend_eval.py kl --model "$QWEN3_8B" --gpu $GPU \
  --out "$OUT/kl" 2>&1 | tee "$KVQ_WORK/logs/m3c_kl_$TS.log"
```

- 完全离线（使用仓库内 `docs/docs/**/*.mdx` 分词生成的 32 条 1024-token prompt，每条生成 512 token，配合 `--sampling-backend pytorch`）。
- 依次启动 4 个服务并记录启动耗时与默认显存比例下的 KV 容量：
  - `int8-chunk-bf16`（M3c 默认，2-stage 融合 extend）
  - `int8-chunk-q`（`SGLANG_ENABLE_TRITON_INT8_UNIFIED_EXTEND=1`，1-stage 融合 unified extend）
  - `int8-det`（`int8_per_token_head` + `--enable-deterministic-inference`）
  - `bf16-det`（BF16 + `--enable-deterministic-inference`，基线对照）
- 每个服务跑 4 项 KL 检验：`match`、`prefill_cache_hit`、`decode_cache_hit`、`prefill_cached_vs_fresh`（直接分离 `chunk_bf16` 与 `chunk_q` 在前缀命中 vs 重新 prefill 时的语义差异）。
- 结果在 `$OUT/kl/summary.md`。

## 4. 16k 服务测速回归验证（约 8 分钟）

```bash
python benchmark/kv_lowbit/m0_bench.py --model "$QWEN3_8B" --gpu $GPU --groups m3b \
  --configs sglang-int8-triton --input-lens 16384 --output-len 1024 \
  --concurrency 6,max --no-profile --out "$OUT/serve" \
  2>&1 | tee "$KVQ_WORK/logs/m3c_serve_$TS.log"
```

- 只测默认配置 `sglang-int8-triton`（M3b 基线：TTFT 5868 ms、ITL 19.67 ms、并发 6 输出 187.2 tok/s、最大并发 13 输出 234.7 tok/s；BF16 TTFT 5620 ms）。
- 验收：TTFT 下降（消除 M3b 前缀反量化开销），ITL 和最大并发吞吐无回退，默认显存比例下正常启动。
- 结果在 `$OUT/serve/summary.md`。

## 5. 推送

```bash
source "$KVQ_WORK/env.sh" && cd "$KVQ_WORK/src/sglang"
git pull --rebase origin kv-int8-int4-gemini
DEST="plans/kv_int8_int4/results/a100-40g/$TS"
mkdir -p "$DEST/logs"
cp -r "$OUT/kernel" "$DEST/kernel"
cp -r "$OUT/kl" "$DEST/kl"
cp -r "$OUT/serve" "$DEST/serve"
rm -rf "$DEST"/serve/profiles
cp "$KVQ_WORK"/logs/m3c_*_$TS.log "$DEST/logs/"
du -sh "$DEST"; find "$DEST" -type f -size +5M
```

- 超过 5 MB 的日志按 [`env_setup.md`](env_setup.md) 步骤 10 截断。
- `README.md` 只写事实：各节是否跑完、测试通过 / 失败数、失败项。
- 按步骤 10.2 脱敏、检查，按 10.3 提交并推送到 **`origin kv-int8-int4-gemini`**，提交信息写 `results: A100-40G stage 2 M3c $TS`。
- 回报：提交号，以及 `kernel/summary.md`、`kl/summary.md`、`serve/summary.md` 的表格。
