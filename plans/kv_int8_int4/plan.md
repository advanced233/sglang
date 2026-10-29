# SGLang 低比特 KV Cache（INT8 / INT4 per-token-head）：实现计划

本文档记录在 SGLang 上实现 INT8 / INT4 per-token-head KV cache 的计划内容与实现顺序：参照 vLLM 的设计实现两个新的 KV 配方，在 Triton attention 后端里融合反量化，并在单卡 A100-40G 上与 vLLM 的同 dtype 实现对照。

文中性能数字均为按字节数的粗估，只用于定预期、排优先级，实测后逐项替换。

**项目原则**：这是写进简历的项目，不是论文，也不是工业生产项目。每条简历结论只用能支撑它的最小实验：一个代表性配置，每个配置跑一次；只有结果出现意外时才补实验。上游合入是加分项，不是项目成立的前提：fork 上的实现、测试和数据本身就要能支撑简历结论。

本文件只放在 fork 的 `kv-int8-int4` 分支上，向上游提 PR 时不带它。

- 基线代码：[advanced233/sglang](https://github.com/advanced233/sglang) `main`，commit `a1966e1`（2026-10-27）
- 参考实现：
  - vLLM `9bc1c8b`（2026-10-28）：`int8_per_token_head` / `int4_per_token_head`（RFC vllm-project/vllm#37319，INT8 合入于 vllm-project/vllm#38378）。Triton 实现，只在 `TRITON_ATTN` 后端上声明支持。
  - LMDeploy TurboMind 的 INT4 / INT8 KV：CUDA 实现，文档写明支持 sm80 / sm86。只作设计参考，不做对照实验。
- 硬件：
  - 1×A100-40G（稳定可用，所有主实验都在这张卡上）：HBM 1555 GB/s，BF16 稠密 312 TFLOPS。实测型号是 A100-PCIE-40GB，功率上限 250 W。驱动是 550，SGLang（CUDA 13）靠环境内的前向兼容库运行（见 `results/a100-40g/20261028-150827/`）
  - 1×A100-80G（共享）：只用于可选项
- 模型：
  - Qwen3-8B：主力。36 层，32 个 Q head、8 个 KV head，head_dim 128，有 QK-norm。
  - Llama-3.1-8B-Instruct：交叉验证。它是 SGLang 的 `DEFAULT_MODEL_NAME_FOR_TEST`，而且没有 QK-norm，用来对照 K 的 outlier 分布。

路径简写：`srt/` 指 `python/sglang/srt/`，`kops/` 指 `python/sglang/kernels/ops/`，`vllm:` 指 vLLM 仓库根目录。

运行环境的配置步骤见 [`env_setup.md`](env_setup.md)。

## 1. 目标与边界

目标：

1. 为 SGLang 新增 `int8_per_token_head`、`int4_per_token_head` 两个 KV cache 配方，量化语义对齐 vLLM。
2. 在 SGLang Triton 后端的 decode / extend kernel 中融合 per-token 反量化，让低比特 KV 的 decode 收益接近带宽上限；顺带把现有的 `fp4_mx_block16` 接到同一条融合路径上。
3. 在单卡 A100-40G 上给出容量、吞吐、精度三张表，与 vLLM 的同 dtype 实现对照；用 roofline 模型解释加速比随"并发 × 上下文长度"的变化。

不做：

- FP8 Tensor Core 计算（Ampere 没有），以及 Q 的量化：attention 的矩阵乘仍在 BF16 / FP32 下进行。
- flashinfer、fa3 等其他后端的融合：和 vLLM 一样，先只声明支持 Triton 后端。
- MLA 模型（`MLATokenToKVPool`）、混合线性注意力模型、SWA 模型的 KV 池。
- 多卡、PD 分离、HiCache 的 L3 存储后端。
- KIVI 式 per-channel K 量化：只作为可选项 M8。

## 2. 现状核对

| 能力 | 现状 | 位置 | 结论 |
|---|---|---|---|
| `--kv-cache-dtype` 选项 | 只有 `auto / fp8_e5m2 / fp8_e4m3 / mxfp8 / bf16 / bfloat16 / nvfp4 / fp4_mx_block16`，没有 INT8 / INT4 | `srt/arg_groups/fields/model.py:209-218` | 新增两个选项 |
| dtype 字符串解析 | 只认上面这些值，其余直接报错 | `srt/mem_cache/kv_cache_dtype.py:49-90` | 加分支：INT8 → `torch.int8`，INT4 → `torch.uint8` |
| KV 量化配方框架 | 有。基类 `KVCacheQuantMethodBase`，配方注册表 `KV_CACHE_QUANT_REGISTRY`，"每个后端、每个阶段怎么读 KV"的注册表 `KV_CACHE_ATTENTION_ACCESS_REGISTRY` | `srt/layers/quantization/fp4_kv_cache_quant_method.py:110,994,1016` | 新配方挂在这里 |
| 配方的构建入口 | 只有当存储 dtype 是 FP4 时才会构建配方 | `srt/mem_cache/kv_cache_configurator.py:436-450`、`srt/model_executor/pool_configurator.py:500-525` | 门槛放宽为"注册表里有这个名字" |
| 读取访问方式 | 只有 `PLAIN`、`DEQUANT_WORKSPACE`、`NATIVE_FP4` 三种 | `fp4_kv_cache_quant_method.py:53-61` | 融合路径需要新增一种，属于 API 改动，要先和维护者商量（M3） |
| KV 池 | `MHATokenToKVPool` 已经支持：由配方分配 buffer（含 scale buffer），写入时调用 `quantize_and_store` | `srt/mem_cache/memory_pool.py:2184,2455,2891,3046-3065` | 直接复用 |
| 非融合读取 | 配方声明"PLAIN + 存储 dtype"时，`_get_key_buffer` / `_get_value_buffer` 会对**整层** buffer 调用 `dequantize_kv_tensor` | `memory_pool.py:2836-2874` | M2 先走这条路径对齐数值；它也是"优化前"的基线 |
| 现有 FP4 KV 在 Triton 后端 | `fp4_mx_block16` 在 triton / torch_native / flex_attention 上走的就是上一行的整层反量化 | `fp4_kv_cache_quant_method.py:932-935` | 按代码推断很慢（见 3.3），M0 先实测 |
| Triton decode kernel | 只支持标量 `k_scale` / `v_scale`（FP8 per-tensor），K 的 scale 折进 `sm_scale` | `kops/attention/decode_attention.py:235`（MHA）、`:530`（GQA grouped）；K 加载 `:684`、`qk *= sm_scale_withk` `:708`、V 累加 `:744-762`；入口 `:1191` | M3 加 per-token scale |
| Triton extend kernel | 前缀从 K/V buffer 读（标量 scale）；当前 chunk 直接用 BF16 的 `k_extend` / `v_extend` | `kops/attention/extend_attention.py:375`；前缀 `:605-713`；当前 chunk `:792,856`；入口 `:916` | M3；当前 chunk 的数值语义见 M3 第 4 点 |
| 确定性模式的 unified extend | 前缀和当前 chunk 都从 cache 读，语义与 vLLM 相同 | `extend_attention.py:1172,1508`；`srt/layers/attention/triton_backend.py:1797-1799,2141` | M3 一并支持 |
| A100 上的默认后端 | MHA 模型在非 Hopper / Blackwell 上默认用 flashinfer | `srt/arg_groups/model_override_base.py:277-330` | 低比特 KV 要求 `--attention-backend triton`；BF16 基线 flashinfer 和 triton 两种都报 |
| FP4 KV 兼容性检查 | 有 `handle_kv4_compatibility` | `srt/arg_groups/kv_cache_hook.py:112` | 照着它给 INT8 / INT4 加检查 |
| HiCache | host 传输遇到非 `MXFP8_INTERLEAVED` 布局的 scale 会直接报错，而 per-token scale 用的正是 `TOKEN` 布局 | `srt/mem_cache/pool_host/mha.py:161-169` | 第一版声明不支持（可选 M7） |
| 已有测试 | 配方单测、Triton attention kernel 测试 | `test/registered/unit/layers/quantization/test_fp4_kv_cache_quant_method.py`、`test/registered/attention/test_triton_attention_kernels.py` | 在这两个文件上扩展 |

> **环境实测补充**（A100-PCIE-40GB，`results/a100-40g/20261028-150827/`）：
> - `fp8_e4m3` 在 Triton 后端起不来：sm80 上的 Triton 不支持 `fp8e4nv`，只支持 `fp8e4b15` 和 `fp8e5`。FP8 基线改用 `fp8_e5m2`，M0 确认能否启动。
> - `test_triton_attention_kernels.py` 在基线代码上就有 1 个失败：`test_extend_attention_unified_vs_regular` 的 head_dim=80 子用例，最大差值 0.449。sm80 上这个形状用的分块是 BLOCK_M=128、BLOCK_N=128，H100 和 sm86 都不用这组。本项目的两个模型 head_dim 都是 128，不受影响；但 M3 要改 unified kernel，这个失败要记作基线状态，改完后不能新增失败。
> - `bench_serving` 的 `random` 数据集要下载 ShareGPT，离线时会失败；改用 `random-ids`。

vLLM 参考实现的位置：

| 内容 | 位置 |
|---|---|
| dtype 与量化模式 | `vllm:vllm/v1/kv_cache_interface.py:40-180`（`KVQuantMode`） |
| scale 内联在 page 里：每个 (head, slot) 存 `[K | K_scale | V | V_scale]`，scale 为 fp32 | `vllm:vllm/v1/attention/backends/triton_attn.py:308-320`（`customize_spec`）、`:414-476`（scale view） |
| INT8 / FP8 写入 kernel：每个程序处理一个 (token, head)，scale = absmax / 127 | `vllm:vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:169,286` |
| INT4 整套：写入 kernel `:65`、split-dot attention `:320`、入口 `:856,882`、RHT `:1146` | `vllm:vllm/v1/attention/ops/int4_per_token_head.py` |
| INT4 的 RHT 是独立的 fp32 计算：K/V 写入前 `:867-868`，Q `:918`，输出逆变换 `:950` | 同上 |
| 测试 | `vllm:tests/quantization/test_per_token_kv_cache.py`（738 行） |

## 3. 关键数字（粗估）

### 3.1 每 token 的 KV 字节与容量（Qwen3-8B）

可用于 KV 的显存按约 18 GiB 估算（权重约 16.4 GB），实际以启动日志里的 `max_total_num_tokens` 为准。

| KV dtype | 每 (层, KV head) 的 K+V 字节 | 每 token（36 层 × 8 head） | 相对 BF16 的容量 | A100-40G 约可容纳 |
|---|---:|---:|---:|---:|
| BF16 | 512 | 144 KiB | 1× | ~13 万 tokens |
| FP8（per-tensor，A100 上用 e5m2） | 256 | 72 KiB | 2.0× | ~26 万 |
| INT8 per-token-head | 264 = 2 × (128 + 4) | 74.3 KiB | 1.94× | ~25 万 |
| INT4 per-token-head | 136 = 2 × (64 + 4) | 38.3 KiB | 3.76× | ~49 万 |
| fp4_mx_block16 | 144 = 2 × (64 + 8) | 40.5 KiB | 3.56× | ~47 万 |

可选的容量卖点：Qwen3-14B（40 层、8 个 KV head）BF16 权重约 29.6 GB，40G 卡上 KV 只剩约 6 GiB，约 4 万 tokens；INT4 后约 15 万。

> **环境冒烟实测**（Qwen3-8B，A100-PCIE-40GB）：
> - SGLang BF16（默认 `--mem-fraction-static 0.823`）：`max_total_num_tokens=121576`，与上表的约 13 万相符。
> - SGLang `fp4_mx_block16`：默认比例下分配了 432,270 tokens（K、V 各 8.35 GB），但 CUDA graph 捕获时 OOM，原因见 3.3；改成 0.7 后能运行，容量降到 308,468。
> - vLLM（Triton 后端）的 BF16 / INT8 / INT4：138,560 / 268,736 / 521,344 tokens，比值 1.94× / 3.76×，与上表的字节计算完全一致。

### 3.2 decode 单步访存与加速上限

decode 是访存瓶颈：每步要读一遍权重，再读一遍所有请求的 KV。单步时间 ≈（权重字节 + Σ上下文长度 × 每 token KV 字节）/ 带宽。下表按 1.555 TB/s 的理论带宽计算，没有计入 kernel 的实际效率。

| 场景 | 总上下文 | BF16 | INT8 | INT4 |
|---|---:|---:|---:|---:|
| 8 并发 × 4k | 3.3 万 | 13.7 ms | 12.2 ms（1.12×） | 11.4 ms（1.20×） |
| 32 并发 × 4k | 13.1 万 | 23.0 ms | 17.0 ms（1.35×） | 13.8 ms（1.66×） |
| 各自把 KV 占满（4k 上下文） | BF16 32 / INT8 62 / INT4 120 并发 | 23.0 ms，约 1.4k tok/s | 23.0 ms，约 2.7k tok/s（1.94×） | 22.9 ms，约 5.2k tok/s（3.77×） |

- 上下文短、并发低时，读权重占主导，KV 量化几乎不提速，只省显存。
- 同样的并发下，KV 越大，提速越明显。
- KV 占满时，每步读的 KV 字节数大致相同，所以吞吐的提升约等于容量的提升。前提是 kernel 跑到带宽上限，而且计算没有成为瓶颈：120 并发时 GEMM 约 2 TFLOP，在 312 TFLOPS 下约 6 ms，仍然是访存瓶颈。

### 3.3 现有 `fp4_mx_block16` 整层反量化的代价（按代码推断，未实测）

`_get_key_buffer` 每次都把整层 buffer 反量化成 BF16，处理量按池子容量算，和实际用到多少 token 无关。按约 47 万 tokens 的池子估算：

- 每层：读 FP4 + scale 约 0.54 GB，写 BF16 约 1.93 GB。
- 每次 forward：36 层合计约 89 GB，按理论带宽至少多出约 57 ms，与 batch 大小和上下文长度无关。
- 显存：同一时刻 K、V 两份整层 BF16 临时张量约 1.9 GB，吃掉一部分省下的显存。

如果实测成立，这就是"优化前"的数据；把它接到融合路径（M6）本身就是一个能独立提交的上游修复。

> **显存这一半已经被环境冒烟证实**：`fp4_mx_block16` 在默认显存比例下，decode 的 CUDA graph 捕获时 OOM，那次申请的是 846 MiB。这正好是一层 K 在整个池子（432,271 个槽位 × 8 head × 128 维 × 2 字节 ≈ 844 MiB）上反量化成 BF16 的大小。后果是必须把 `--mem-fraction-static` 降到 0.7 才能运行，容量从 43.2 万降到 30.8 万（−29%）。非融合路径不光慢，还把省下来的显存吃掉了一部分。时间上的代价仍由 M0 测量。

### 3.4 由此得到的预期

- 容量：INT8 约 1.9×，INT4 约 3.7×（扣除固定开销后会略低）。
- 吞吐：KV 占满时提升接近容量比；短上下文、低并发时基本不变。
- 精度：INT8 应与 BF16 基本持平；INT4 会有可测的损失，RHT 是关键，M1 先离线量化这一点。
- `fp4_mx_block16` 接到融合路径后，应从"比 BF16 慢很多"变成"接近 INT4"。

## 4. 实现内容

每个模块按"做什么 / 涉及位置 / 验收 / 测试"组织。测试遵循仓库自己的规则：

- 是否值得加一个测试：`.claude/rules/unit-test-admission.md`。每个测试都要回答"哪种未来的改动会让它变红"。
- 放在哪、怎么注册 CI：`.claude/skills/write-sglang-test/SKILL.md` 和 `.claude/skills/kernel-organization/SKILL.md`。
  - 纯 CPU 单测放 `test/registered/unit/` 下，镜像 `srt/` 的目录。
  - kernel 数值测试放 `test/registered/kernels/ops/<group>/`。
  - 新用例默认加进同一子系统的已有文件。
- kernel 归属：读写 KV cache 内容的 op 归 `kops/kvcache/`；attention kernel 内部的融合反量化留在 `kops/attention/`。
- kernel 计时：`.claude/rules/kernel-benchmark.md`，要在冷 L2 下测。
- 精度评测：`.claude/rules/prefer-sgl-eval.md`，用 `sgl-eval`（`python/pyproject.toml` 里锁定为 0.1.3）。

### M0 测量基建与基线

- **环境记录**：GPU 型号与时钟、驱动、CUDA、torch、triton、flashinfer 的版本，SGLang 和 vLLM 的 commit。
- **基线配置**：Qwen3-8B，分别测：
  - BF16：flashinfer（默认后端）和 triton 各一组；
  - FP8：`fp8_e5m2`（triton；`fp8_e4m3` 在 sm80 的 Triton 上不支持）；
  - `fp4_mx_block16`（triton，需要 `--mem-fraction-static 0.7`，见 3.3）。
  - 某个后端起不来，就记下报错，不硬凑。
- **指标**：
  - 容量：启动日志的 `max_total_num_tokens`。
  - 吞吐：`python -m sglang.bench_serving --dataset-name random-ids --random-input-len 4096 --random-output-len 256 --random-range-ratio 1 --max-concurrency N`，记录输出吞吐和 TPOT 中位数。`random` 数据集要下载 ShareGPT，离线时用不了。
  - 拆分：一次 decode step 的 nsys 记录，按 kernel 拆耗时。
- **验证 3.3**：固定 batch=1、短上下文，只改 `--max-total-tokens`（池子大小），看 `fp4_mx_block16` 的 decode 单步时间是否随池子大小线性增长，而 BF16 不变。
- **产出**：`benchmark/kv_lowbit/` 下的启动和压测脚本，以及结果表模板。

### M1 离线误差研究（纯 PyTorch，便宜）

- 用 Qwen3-8B 和 Llama-3.1-8B 跑几条真实 prompt，抓下每层写进 cache 的 K/V（RoPE 之后）。
- 对比几种量化：FP8 per-tensor（e5m2 和 e4m3 都算，离线误差不受 Triton 限制）、INT8 per-token-head、不带 RHT 的 INT4、带 RHT 的 INT4、`fp4_mx_block16`。
- 指标：
  - 每层 attention 输出的相对误差；
  - 末层 logits 相对 BF16 的 KL。
- 要回答的问题：
  1. INT4 不带 RHT 能不能用；
  2. RHT 带来多少收益；
  3. 有 QK-norm 的 Qwen3 和没有 QK-norm 的 Llama，K 的 outlier 是否不同。
- 验收：一张表，结论用于 M4 的取舍。不改引擎，任何 GPU 上都能跑。

### M2 INT8 配方跑通（先不融合）

- **选项与解析**：
  - `srt/arg_groups/fields/model.py:209` 加 `int8_per_token_head`；
  - `srt/mem_cache/kv_cache_dtype.py` 解析为 `torch.int8`。
- **配方**：新建 `Int8PerTokenHeadKVCacheMethod`，注册进 `KV_CACHE_QUANT_REGISTRY`。
  - `create_buffers`：K/V 为 int8 `[容量, head, dim]`，scale 为 fp32 `[容量, head]`。
  - `quantize_and_store`：调用写入 kernel。
  - `dequantize_kv_tensor`：给非融合读取用。
  - `compute_cell_size`：按 3.1 的字节数计算。
- **写入 kernel**：移植 vLLM 的 `_reshape_cache_per_token_head`，放在 `kops/kvcache/` 并注册 `KernelSpec`。
  - SGLang 按 token 寻址，写入位置 `loc` 直接就是行号，比 vLLM 的 block / slot 换算简单。
  - 要保留 vLLM 的两个细节：round-half-away-from-zero 取整，以及 scale 下限 `1e-6`。
- **读取**：先声明"PLAIN + 存储 dtype"，走现成的整层反量化，Triton 后端立刻能跑。
- **构建入口**：放宽 `kv_cache_configurator.py:436` 和 `pool_configurator.py:500` 的 FP4 门槛。
- **兼容性检查**：仿照 `handle_kv4_compatibility`，要求 prefill / decode 都用 triton，并拒绝 MLA、HiCache、unified memory 等组合。
- **验收**：
  - 服务能启动；
  - `max_total_num_tokens` 约为 BF16 的 1.9 倍；
  - GSM8K 与 BF16 的差在 1 分左右。
- **测试**：
  - 单测：配方注册的完整性、`compute_cell_size` 的字节数（属于"关键路径记账"类），放进 `test/registered/unit/layers/quantization/test_fp4_kv_cache_quant_method.py`；
  - kernel 测试：写入 kernel 对照 torch 参考的往返误差上界，以及 `loc` 为填充槽时的处理，放在 `test/registered/kernels/ops/kvcache/`。

### M3 融合 per-token 反量化（核心）

1. **新的访问方式**：在 `KVCacheAttentionAccessKind` 里新增一种（名字和形状要和维护者商量），让池子把原始的 int8 buffer 和 scale buffer 直接交给后端，不再走整层反量化。
   - 这是 API 改动。实现之前先在上游开一个 RFC issue，说明动机：Ampere 上没有低比特 KV，而 `fp4_mx_block16` 在 Triton 后端上是整层反量化。
2. **decode**：改 `_fwd_grouped_kernel_stage1`（`:530`）和 `_fwd_kernel_stage1`（`:235`）。
   - K tile 按 int8 读入后转成 Q 的 dtype 再做 dot，然后乘上这一列 token 的 K scale：在 `:708` 处与 `sm_scale` 合并成 `qk *= sm_scale * ks[None, :]`。
   - V 的 scale 先乘进 P，再做 `:760` 处的 dot。这和 vLLM 的做法一致。
3. **extend**：
   - `_fwd_kernel` 的前缀阶段（`:605-713`）做同样的改动；当前 chunk 阶段不变。
   - 确定性模式下的 `_fwd_kernel_unified` 也一并改掉。
4. **当前 chunk 的数值语义**（要明确做出决定）：
   - SGLang 的默认 extend 对当前 chunk 直接用 BF16 的 `k_extend` / `v_extend`，只有前缀从 cache 读。后果是：同一个 prompt，前缀命中 radix cache 和重新 prefill 时，数值不一样；prefill 和 decode 的 logprob 也会不一致。
   - vLLM 和 SGLang 的确定性模式（unified kernel）都是先写 cache 再从 cache 读，没有这两个问题，但当前 chunk 也要多做一次反量化。
   - 两种语义都实现。用精度、速度，以及 `kl-consistency-test` skill 里的三个 helper 对比后，再决定默认用哪种。按设计，默认语义下 `prefill_cache_hit` 的 KL 不为零，文档里要写明这一点。
5. **CUDA graph**：scale buffer 的地址是固定的，确认 decode 的图捕获正常。
- **验收**：
  - kernel 输出与"先反量化再用 fp32 计算"的参考结果一致，误差在容差内；
  - decode kernel 的 microbench（冷 L2）报告带宽利用率；
  - 端到端 decode 单步时间分别与 M2（非融合）和 BF16-triton 对比。
- **测试**：给 `test/registered/attention/test_triton_attention_kernels.py` 加 int8 per-token scale 的用例，覆盖 GQA grouped、MHA、extend 前缀、unified 四条路径。

### M4 INT4（RHT）

- **移植 vLLM 的 INT4**：
  - 写入：非对称量化，scale = (max − min) / 15，4-bit zero-point 藏在 fp32 scale 尾数的低 4 位；每字节两个 nibble。
  - RHT：固定种子生成的 ±1 符号向量，加上 Walsh-Hadamard 变换，范数为 √b。
  - decode：split-dot，Q 分成奇偶两路；K 的 zero-point 用 `zp × sum(Q)` 修正，V 的 zero-point 单独减去。
- **RHT 在 SGLang 里的位置**：
  - K/V 写入前做 RHT；Q 在进 kernel 前做 RHT；输出做逆变换。
  - extend 的当前 chunk 也要把 `k_extend` / `v_extend` 旋转到同一空间：RHT 是正交变换（差一个常数倍），在旋转空间里算 attention 是等价的。
- **RHT 的开销**：vLLM 里 RHT 是几次独立的 fp32 计算（见第 2 节）。先测它占单步时间的比例，超过几个百分点再考虑融进写入 kernel、Q 的加载和输出的 epilogue。这一项可能反过来给 vLLM 提 PR。
- **与 vLLM 逐字节对齐**：沿用 vLLM 的 RHT 符号种子（`0x9E3779B9`），同一份输入经过两边的写入 kernel，cache 内容应当逐字节相同。这是跨实现的强正确性检查。
- **验收**：
  - 容量约为 BF16 的 3.7 倍；
  - 精度表（GSM8K、长上下文）；
  - decode 速度。
- **测试**：
  - zero-point 藏进尾数的编解码边界（"推导性质"类）；
  - 写入 kernel 与 vLLM 逐字节一致；
  - attention 结果与参考实现一致。

### M5 评测与 vLLM 对照

- 同一张卡、同一个模型、同样的输入输出长度和并发数。vLLM 用 `--kv-cache-dtype int8_per_token_head / int4_per_token_head`，后端只能是 `TRITON_ATTN`。
- **主表**：每行在 4k / 16k / 32k 上下文下各跑一次。

| 配置 | 用途 |
|---|---|
| BF16，flashinfer | SGLang 的默认配置 |
| BF16，triton | 同后端基线 |
| FP8 `fp8_e5m2` | 现有的 8 bit 方案（A100 的 Triton 不支持 e4m3） |
| INT8 per-token-head（M3） | 本项目 |
| INT4 per-token-head（M4） | 本项目 |
| `fp4_mx_block16`：整层反量化（M0）与融合（M6） | 修复前后对比 |
| vLLM INT8 / INT4（Triton） | 参考实现 |

- **指标**：容量、吞吐-并发曲线、TPOT 中位数；精度用 GSM8K（sgl-eval，两边的服务都走 OpenAI 兼容接口）加长上下文（NIAH 或 RULER 子集）。

### M6 现有 `fp4_mx_block16` 接入融合路径（加分）

- 在 M3 的同一组 kernel 里加上 block-16 FP4 的融合反量化：E2M1 数值查表，每 16 个元素一个 scale。
- 与 M0 的整层反量化对比单步时间和峰值显存。
- 改动独立、规模最小，适合作为第一个上游 PR。

### M7（可选）HiCache 支持 `TOKEN` 布局的 scale

- 让 host 池能连同 per-token scale 一起搬运（`srt/mem_cache/pool_host/mha.py:161-169`），实现可以参照 `srt/mem_cache/pool_host/mha_mxfp8.py`。

### M8（可选）KIVI 式 per-channel K

- 只有在时间宽裕时才做。per-channel 需要凑满一组 token 才能量化，未满的部分要留 BF16 尾块；SGLang 默认 page_size=1，实现成本高。大概率跳过。

### M9 性能分析

- 用 nsys / ncu 测 decode kernel 的实际带宽利用率，把单步时间代入 3.2 的模型，与实测对照。
- 解释和理论值的差距来自哪里：scale 的加载、int → bf16 转换、INT4 的解包、RHT 的额外计算。
- 汇总 M1 的结论：INT4 在 Qwen3 和 Llama 上的误差差异，以及 RHT 的作用。

## 5. 实现顺序

```text
第 0 步  在 sgl-project/sglang 搜进行中的 PR / issue（int8 kv、per_token_head、int4 kv、kv cache quant）
   │
阶段 0  M0 基线（含整层反量化的实测）    A100-40G
        M1 离线误差                      任意 GPU
   │
阶段 1  M2 INT8 配方跑通（非融合）        A100-40G   ← 打通 dtype、配方、池子、写入 kernel
   │    （同时在上游开 RFC issue，讨论新的访问方式）
阶段 2  M3 融合 kernel                   A100-40G   ← 工作量最大，也是项目核心
   │
阶段 3  M4 INT4                          A100-40G
   │
阶段 4  M5 评测与 vLLM 对照、M9 分析      A100-40G
   │
阶段 5  M6 → M7 → M8，按时间决定
```

排序理由：

1. **先测基线和离线误差**：不改引擎就能得到"优化前"的数据，同时知道 INT4 是否需要 RHT。
2. **INT8 先走非融合路径**：先把 dtype、配方、池子、写入 kernel 打通，数值对齐了再碰 kernel，出问题时容易定位。
3. **RFC 与 M2 并行**：新的访问方式是 API 改动，等维护者回复的同时可以继续写 M2。
4. **INT4 放在融合之后**：INT4 的读取只有融合路径才有意义，它依赖 M3 的 kernel 结构。
5. **M6 放在最后，但最适合先提上游**：它只依赖 M3，改动最小，可以作为第一个独立 PR。

上游 PR 的拆分建议：(a) M6，`fp4_mx_block16` 的融合读取；(b) 新访问方式 + INT8；(c) INT4。

## 6. 实验协议

- **固定参数**：
  - 模型 Qwen3-8B，BF16 权重。
  - 所有低比特 KV 配置都用 `--attention-backend triton`。
  - 固定一个 `--mem-fraction-static`，所有配置相同。
  - CUDA graph 保持默认开启。
- **速度**：
  - `bench_serving` 的 random 数据集，输入输出长度固定（`--random-range-ratio 1`）。
  - 输入 4k / 16k / 32k，输出 256；并发取 1、8、32 和该配置能放下的最大值。
  - 每组跑一次，报告输出吞吐和 TPOT 中位数。
  - 同卡上出现其他进程的运行作废重跑。
- **温度**：Wan 项目在 A100 上实测过持续负载下的热降频。参与速度对比的几组，开跑前记录 GPU 温度和 SM 时钟，在相近的温度下测。
- **精度**：
  - GSM8K 用 sgl-eval 跑完整的测试集，Qwen3 用非思考模式，采样参数用模型推荐值，所有配置相同，各跑一次。
  - 长上下文：16k / 32k 下的 NIAH，或 RULER 子集。开工时先确认 sgl-eval 有没有长上下文任务；没有的话，按 `prefer-sgl-eval.md` 的例外条款用单独的脚本。
- **偏离度与质量分开报告**：
  - 偏离度：相对 BF16 的 logits KL、greedy 解码的一致率。
  - 质量：任务准确率。
  - 两类指标不能混用：偏离大不等于质量差。
- **记录**：每组结果都附带环境、命令、SGLang / vLLM 的 commit。

## 7. 风险与应对

| 风险 | 应对 |
|---|---|
| 上游已经有人在做，或者维护者不接受新的访问方式 | 第 0 步先搜；开 RFC issue；拆成小 PR，先提 M6。fork 上的实现和数据本身就能支撑简历 |
| Triton kernel 达不到带宽上限（scale 加载、int → bf16 转换） | 报告实测带宽利用率，用 ncu 分析原因；容量收益不受影响 |
| INT4 在某些层精度损失大 | M1 先离线量化误差；必要时按层回退到 INT8 或 BF16 |
| 当前 chunk 用 BF16 导致命中 cache 与否结果不同 | 两种语义都实现，量化对比后再定默认值，并写进文档 |
| flashinfer BF16 基线比 triton 快很多，"低比特 triton vs 默认 flashinfer"不占优 | 如实报告；结论分成"同后端对比"和"对默认配置对比"两种 |
| A100 热降频 | 按第 6 节的温度协议测 |
| SGLang 迭代快 | 实验 pin 在 `a1966e1`；提 PR 前再 rebase |
| 共享的 A100-80G 拿不到 | 主实验只依赖 40G 卡 |

## 8. 简历要点模板（实测后填数）

> **SGLang 低比特 KV Cache：Ampere 上的 INT8 / INT4 per-token-head 量化与融合反量化**
>
> - **实现**：参照 vLLM 的设计，为 SGLang 新增 INT8 / INT4 per-token-head KV cache 配方。在 Triton decode / extend kernel 中融合 per-token 反量化；INT4 采用随机 Hadamard 变换加非对称量化，4-bit zero-point 藏在 fp32 scale 的尾数里，与 vLLM 的写入结果逐字节一致。
> - **修复**：定位 SGLang 现有 FP4 KV 在 Triton 后端上每层整层反量化的问题，接入融合路径后，decode 单步从 x ms 降到 y ms，峰值显存减少 z GB。
> - **效果**：单卡 A100-40G 上，Qwen3-8B 的 KV 容量从约 13 万提升到 x 万 tokens，4k 上下文下 decode 吞吐提升 x×；GSM8K BF16 x vs INT8 y vs INT4 z，32k NIAH ……；与 vLLM 的同 dtype 实现相比，吞吐为其 x%。
> - **分析**：用 roofline 模型（单步字节数 = 权重 + KV）解释加速比随"并发 × 上下文长度"的变化；kernel 带宽利用率 x%；对比 Qwen3（有 QK-norm）和 Llama-3.1 在 INT4 下的误差差异，以及 RHT 的作用。
> - （如果有）向 SGLang 提交并合入 N 个 PR：……
