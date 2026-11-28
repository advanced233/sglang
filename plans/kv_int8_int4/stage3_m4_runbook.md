# 阶段 3：INT4 per-token-head + RHT 实现与融合 kernel 实测（M4）

对应 [`plan.md`](plan.md) M4：
- M4a：INT4 非对称量化（4-bit zero-point 嵌入 fp32 scale 尾数低 4 位、scale 分离存储）+ RHT（种子 `0x9E3779B9`）写入/反量化 kernel、配方注册与单测；
- M4b：基于 `FUSED_DEQUANT` 的融合 INT4 decode / 2-stage extend / 1-stage unified extend Triton kernel（PTX nibble 解包 + Q/O 融合 RHT）并接入 Triton 后端。

共享服务器规则、脱敏和推送沿用 [`env_setup.md`](env_setup.md) 第 0 节和步骤 10。**不要修改代码**，报错记录完整 Traceback 并继续下一项；第 1 节 GPU 单测有失败时跳过第 3–4 节。

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

## 1. 单测与 vLLM 逐字节对照（约 4 分钟）

```bash
python -m pytest -q \
  test/registered/unit/layers/quantization/test_fp4_kv_cache_quant_method.py \
  2>&1 | tee "$KVQ_WORK/logs/m4_cpu_tests_$TS.log"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU python -m pytest -q \
  test/registered/kernels/ops/kvcache/test_int4_per_token_head.py \
  test/registered/kernels/ops/attention/test_decode_attention_int4.py \
  test/registered/kernels/ops/attention/test_extend_attention_int4.py \
  test/registered/unit/mem_cache/test_int4_kv_pool.py \
  2>&1 | tee "$KVQ_WORK/logs/m4_gpu_tests_$TS.log"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU python -m pytest -q \
  test/registered/attention/test_triton_attention_kernels.py \
  -k test_int4_per_token_head_attention_paths \
  2>&1 | tee "$KVQ_WORK/logs/m4_triton_attn_test_$TS.log"

python benchmark/kv_lowbit/m4_vllm_parity.py --gpu $GPU \
  2>&1 | tee "$OUT/vllm_parity.json"
```

## 2. Decode kernel 冷 L2 测速（与 BF16 / INT8 / vLLM INT4 对照，约 4 分钟）

```bash
python benchmark/kv_lowbit/m3_kernel_diag.py run --gpu $GPU --skip-ncu \
  --configs ref-sum,sglang-bf16,sglang-int8,sglang-int4,sglang-int4@magic-n64-w4-s3-k8,sglang-int4@magic-n128-w4-s3-k8,sglang-int4@magic-n128-w4-s3-k16,sglang-int4@magic-n256-w4-s2-k8,sglang-int4@cast-n128-w4-s3-k8,vllm-int4,vllm-int4-t64 \
  --out "$OUT/diag" 2>&1 | tee "$KVQ_WORK/logs/m4_diag_$TS.log"
```

## 3. 16k 端到端服务测速与容量验收（约 10 分钟）

```bash
python benchmark/kv_lowbit/m0_bench.py --model "$QWEN3_8B" --gpu $GPU --groups m4 \
  --configs sglang-int4-triton --input-lens 16384 --output-len 1024 \
  --concurrency 6,max --no-profile --out "$OUT/serve" \
  2>&1 | tee "$KVQ_WORK/logs/m4_serve_$TS.log"
```

验收关注：默认 `--mem-fraction-static` 下正常启动无 OOM、KV 容量约为 BF16 的 3.7×（~45 万 token）、并发 6 ITL 与最大并发输出吞吐。

## 4. 推送与回报

将 `$OUT` 与 `$KVQ_WORK/logs/m4_*_$TS.log` 复制到 `plans/kv_int8_int4/results/a100-40g/$TS/`，脱敏检查后提交并推送到 `origin kv-int8-int4-gemini`，回报：
1. 各组单测通过数与 `vllm_parity.json`；
2. `diag/summary.md` 表格；
3. `serve/summary.md` 表格（容量、TTFT、ITL、最大并发吞吐）。
