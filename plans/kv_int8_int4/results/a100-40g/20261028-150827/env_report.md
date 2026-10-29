# A100-40G 环境配置结果

- 运行开始时间：2026-10-28 15:08:27（Asia/Shanghai，依据首次机器探查时间）。工作目录使用 `$KVQ_WORK`，所在文件系统起始可用空间约 2.3 TB。
- 设备：NVIDIA A100-PCIE-40GB；驱动 550.127.05，最大 SM/显存时钟 1410/1215 MHz，功率上限 250 W；开始时温度 33°C、显存占用 15 MiB。Slurm 客户端存在但配置缺失，未发现可用调度器。
- SGLang：分支 kv-int8-int4，配置与自检使用的提交为 `110f0e1f6a091fd374c7e2546b919b37a3aaeff6`。Python 3.12，editable 版本 0.0.0.dev0（源码快照构建的 fallback 版本）；torch 2.14.1+cu130、Triton 3.8.0、transformers 5.19.0、flashinfer 0.7.2、sglang-kernel 0.4.10，nvcc 13.0.88。驱动低于 580，环境内启用 CUDA 13 前向兼容库 580.95.05，GPU 张量计算和 torch.compile 均通过。
- vLLM：0.32.0 官方 cu129 wheel、torch 2.13.0+cu129，CUDA 可用；`int8_per_token_head` 与 `int4_per_token_head` 均存在。FlashInfer 0.7.0.post1 cubin 经 SHA-256 校验并安装，sm80 JIT 缓存以 minimal 模式安装；`vllm download-kernels` 的 all 模式未完成，因为它还下载其他架构缓存。
- sgl-eval：0.1.3，独立环境，`--help` 通过。
- 工具：Nsight Systems 2023.4.4、Nsight Compute 2024.1.0；`RmProfilingAdminOnly=0`，ncu 实测可采集 GPU 性能计数器。`perf_event_paranoid=4`，CPU 采样受限。
- 模型：Qwen3-8B 复用本地五分片权重并通过推理。Llama-3.1-8B-Instruct 的 Hugging Face 访问被拒，改从 ModelScope 下载四个 safetensors 分片；配置与 tokenizer 离线加载、分片头读取通过，完整模型加载未测试。
- 仓库测试：KV 量化 CPU 测试 17 passed；Triton attention GPU 测试 11 passed、1 failed、3 skipped。失败项 `test_extend_attention_unified_vs_regular`（B=8, N_CTX=256, H_Q=64, H_KV=8, D=80），最大差值 0.44921875。
- SGLang 冒烟：默认 FlashInfer 和 Triton BF16 均成功回答 391，`max_total_num_tokens=121576`。FP8 e4m3 在 A100 编译时报告不支持 fp8e4nv。FP4 默认显存比例下 CUDA 图捕获 OOM；`--mem-fraction-static 0.7` 重试成功、回答 391，`max_total_num_tokens=308468`。首次默认配置曾因链接器找不到 libcuda 失败，设置兼容库的链接搜索路径后通过。
- 工具验证：GSM8K 20 题 20/20，非思考模式；仅为小样本自检。原 random 压测因离线缺少 ShareGPT 文件失败；改为 random-ids 后 32 请求全部成功，输出吞吐 402.45 tok/s，中位 TPOT 14.85 ms；数字仅作工具验证。
- vLLM 冒烟：禁用 FlashInfer 采样器并使用 Triton attention 时，BF16、INT8、INT4 均回答 391，KV 容量依次为 138560、268736、521344 tokens。安装 cubin 后默认 FlashInfer 采样器的 BF16 重试成功、回答 391，容量 145808 tokens。容量受可用显存及运行设置影响；安装 cubin 后未重复 INT8/INT4 默认采样器测试。
- 测试服务均已停止，40GB A100 显存恢复至约 15 MiB。Git 初次浅克隆两次中断；之后使用 blob 过滤浅克隆，并从同提交的源码快照填充对象，工作区干净。
