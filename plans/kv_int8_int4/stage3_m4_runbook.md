# 阶段 3：INT4 per-token-head + RHT 实现、调优与验收（M4）

对应 [`plan.md`](plan.md) M4：
- M4a：INT4 非对称量化（4-bit zero-point 嵌入 fp32 scale 尾数低 4 位、scale 分离存储）+ RHT（种子 `0x9E3779B9`）写入/反量化 kernel、配方注册与单测；
- M4b：基于 `FUSED_DEQUANT` 的融合 INT4 decode / 2-stage extend / 1-stage unified extend Triton kernel（PTX nibble 解包 + Q/O 融合 RHT）并接入 Triton 后端；
- M4c（本轮）：Stage 1 寄存器瘦身 + `BLOCK_N_PREFIX=64` 2 级流水线调优、decode 默认 tile 更新为 `(256, 2)`，以及 M4 全量验收（kernel 复测 + 16k 服务复测 + GSM8K 全集 + 16k/32k 长上下文检索 + 离线 KL 一致性）。

共享服务器规则、脱敏和推送沿用 [`env_setup.md`](env_setup.md) 第 0 节和步骤 10。**不要修改代码**，报错记录完整 Traceback 并继续下一项；第 1 节 GPU 单测有失败时跳过后续各节。

## 0. 准备

```bash
source "$KVQ_WORK/env.sh" && kvq_env sglang
cd "$KVQ_WORK/src/sglang"
git fetch origin kv-int8-int4-gemini
git checkout -B kv-int8-int4-gemini origin/kv-int8-int4-gemini
git log --oneline -1
export HF_HUB_OFFLINE=1
GPU=0
TS=$(date +%Y%m%d-%H%M%S)
OUT="$KVQ_WORK/outputs/m4/$TS"
mkdir -p "$OUT" "$KVQ_WORK/logs"
```

## 1. 回归单测（约 2 分钟）

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU python -m pytest -q \
  test/registered/kernels/ops/attention/test_decode_attention_int4.py \
  test/registered/kernels/ops/attention/test_extend_attention_int4.py \
  2>&1 | tee "$KVQ_WORK/logs/m4_gpu_tests_$TS.log"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU python -m pytest -q \
  test/registered/attention/test_triton_attention_kernels.py \
  -k test_int4_per_token_head_attention_paths \
  2>&1 | tee "$KVQ_WORK/logs/m4_triton_attn_test_$TS.log"
```

## 2. Decode & Extend 冷 L2 微基准（约 6 分钟）

```bash
python benchmark/kv_lowbit/m3_kernel_diag.py run --gpu $GPU --skip-ncu \
  --configs sglang-bf16,sglang-int8,sglang-int4,sglang-int4@magic-n128-w4-s3-k8,sglang-int4@magic-n256-w4-s2-k8,sglang-int4@magic-n256-w4-s3-k8,sglang-int4@magic-n256-w8-s2-k8 \
  --out "$OUT/diag" 2>&1 | tee "$KVQ_WORK/logs/m4_diag_$TS.log"

python benchmark/kv_lowbit/m3c_extend_eval.py kernel --gpu $GPU \
  --impls bf16-regular,int8-fused-regular,int4-fused-regular,int4-fused-unified \
  --sweep --sweep-shapes late-chunk,prefix-hit \
  --out "$OUT/extend_kernel" 2>&1 | tee "$KVQ_WORK/logs/m4_extend_kernel_$TS.log"
```

关注：
- `diag/summary.md`：寄存器瘦身 + 默认 `(256, 2)` 后的 decode 耗时（对照上轮 146.4 / 141.3 us）。
- `extend_kernel/summary.md`：`int4-fused-regular` 在 `late-chunk` / `prefix-hit` 上相对 `int8-fused-regular` 和 `bf16-regular` 的速度与 tile sweep 最优解。

## 3. 16k 端到端服务复测 + 1k decode probe（约 10 分钟）

```bash
python benchmark/kv_lowbit/m0_bench.py --model "$QWEN3_8B" --gpu $GPU --groups m4 \
  --configs sglang-int4-triton --input-lens 16384 --output-len 1024 \
  --concurrency 6,max --no-profile --out "$OUT/serve" \
  2>&1 | tee "$KVQ_WORK/logs/m4_serve_$TS.log"
```

关注（对照上轮 TTFT 7955 ms、conc=6 ITL 18.16 ms / 输出 185.2 tok/s、conc=26 输出 250.7 tok/s）：
- `serve/summary.md` 现已直接渲染 M4 表格及 1k 并发 1 decode probe（对标 BF16 13.25 ms / INT8 13.09 ms）；
- TTFT 下降幅度及并发 6 / 26 端到端输出吞吐提升。

## 4. M4 精度验收：GSM8K 全集 + 16k/32k 长上下文 + 离线 KL（约 25 分钟）

```bash
# 4a. GSM8K 全集（1319 题，与 M3b BF16 91.96% / INT8 92.87% 同 harness、同参数）
python benchmark/kv_lowbit/m2_smoke.py --model "$QWEN3_8B" --gpu $GPU \
  --configs int4-triton --max-total-tokens 0 \
  --out "$OUT/gsm8k" 2>&1 | tee "$KVQ_WORK/logs/m4_gsm8k_$TS.log"

# 4b. 离线长上下文检索（16k & 32k S-NIAH / MK-NIAH / MV-NIAH，BF16 vs INT4）
python benchmark/kv_lowbit/m4_longctx_eval.py --model "$QWEN3_8B" --gpu $GPU \
  --configs bf16-triton,int4-triton --lengths 16384,32768 \
  --out "$OUT/longctx" 2>&1 | tee "$KVQ_WORK/logs/m4_longctx_$TS.log"

# 4c. 离线 KL 一致性与确定性模式（int4-chunk-bf16 / int4-chunk-q / int4-det）
python benchmark/kv_lowbit/m3c_extend_eval.py kl --model "$QWEN3_8B" --gpu $GPU \
  --configs int4-chunk-bf16,int4-chunk-q,int4-det \
  --out "$OUT/kl" 2>&1 | tee "$KVQ_WORK/logs/m4_kl_$TS.log"
```

## 5. 推送与回报

将 `$OUT`（去掉可能超过 5 MB 的大文件或按步骤 10 截断）与 `$KVQ_WORK/logs/m4_*_$TS.log` 复制到 `plans/kv_int8_int4/results/a100-40g/$TS/`，脱敏检查后提交并推送到 `origin kv-int8-int4-gemini`，回报：
1. 回归单测通过数；
2. `diag/summary.md`、`extend_kernel/summary.md`（主表 + sweep 前三名）；
3. `serve/summary.md`（16k conc=6 / max + 1k probe）；
4. `gsm8k/summary.md`、`longctx/summary.md`、`kl/summary.md` 表格。
