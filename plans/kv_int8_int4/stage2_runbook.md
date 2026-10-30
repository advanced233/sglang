# 阶段 2 第 1 轮：INT8 decode kernel（M3a）

对应 [`plan.md`](plan.md) M3 的 M3a：新写的 INT8 decode kernel（`python/sglang/kernels/ops/attention/decode_attention_int8.py`）还没接进服务，只做 kernel 测试和微基准。共享服务器规则、脱敏和推送沿用 [`env_setup.md`](env_setup.md) 第 0 节和步骤 10。合计约 25 分钟。

**不要修改代码。** 报错就记录完整 Traceback，跳过这一项，继续下一项。

## 0. 准备

```bash
source "$KVQ_WORK/env.sh" && kvq_env sglang
cd "$KVQ_WORK/src/sglang" && git pull --rebase origin kv-int8-int4 && git log --oneline -1
export HF_HUB_OFFLINE=1
GPU=0                          # A100-PCIE-40GB 的 nvidia-smi index
TS=$(date +%Y%m%d-%H%M%S)
```

## 1. kernel 测试（约 2 分钟）

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU python -m pytest -q \
  test/registered/kernels/ops/attention/test_decode_attention_int8.py \
  2>&1 | tee "$KVQ_WORK/logs/stage2_tests_$TS.log"
```

12 个用例。有失败也继续第 2 步，在回报里列出失败用例的名字。

## 2. 诊断与参数扫描（一条命令，约 20 分钟）

```bash
python benchmark/kv_lowbit/m3_kernel_diag.py run --gpu $GPU --sweep \
  --configs ref-sum,sglang-bf16,sglang-fp8e5m2,sglang-int8,vllm-int8-sep-t64 \
  --out "$KVQ_WORK/outputs/m3diag/$TS" 2>&1 | tee "$KVQ_WORK/logs/m3diag_$TS.log"
```

- 形状和阶段 1 相同。除了 5 个配置，还会把 INT8 kernel 的 48 种参数组合各计时一次，记录在 `sweep_sglang.jsonl`。
- 然后对 5 个配置和每种转换方式最快的那个组合各跑一次 ncu。
- 某个组合编译或运行失败，只会在 `sweep_sglang.jsonl` 里记一行 `failed`，不影响其余组合。

## 3. 推送

```bash
source "$KVQ_WORK/env.sh" && cd "$KVQ_WORK/src/sglang"
git pull --rebase origin kv-int8-int4
DEST="plans/kv_int8_int4/results/a100-40g/$TS"
mkdir -p "$DEST/logs"
cp -r "$KVQ_WORK/outputs/m3diag/$TS" "$DEST/m3diag"
find "$DEST" -name "*.ncu-rep" -delete
cp "$KVQ_WORK"/logs/stage2_tests_$TS.log "$KVQ_WORK"/logs/m3diag_$TS.log "$DEST/logs/"
du -sh "$DEST"; find "$DEST" -type f -size +20M
```

- 超过 20 MB 的文件不推送，在 README 里注明。
- `README.md` 只写事实：两步是否跑完、测试通过 / 失败数、失败项。
- 按步骤 10.2 脱敏、检查，按 10.3 提交，提交信息写 `results: A100-40G stage 2 M3a $TS`。
- 回报：提交号，以及 `m3diag/summary.md` 里 Timing 和 SGLang INT8 decode sweep 两张表。
