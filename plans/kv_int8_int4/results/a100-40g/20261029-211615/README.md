# 阶段 1 运行记录

- 测试提交：`c1ca3b0c5c8c`；本轮开始 2026-10-29 21:16:15，结果整理于约 21:40（Asia/Shanghai），总耗时约 24 分钟。
- 第 1 节 CPU 测试 29 passed、GPU 测试 7 passed，均无失败。此前一次测试因系统磁盘 I/O 等待中断，未得到结果；本轮在存储恢复后重新执行。
- 第 2 节完成，12 个配置均有计时与 ncu 指标；未发现 run_notes 或配置失败，不做补跑。
- 第 3 节两组服务均就绪，回答 391，KV 字节数分别为 4098 和 2112 B/token/layer。两组 decode 探测因离线缺少 ShareGPT 数据文件失败；两组服务在评测期间因 FlashInfer JIT 编译缺少 `curand.h` 退出，GSM8K 未完成。停止了各自失效服务上的评测重试，退出状态 -15；不改源码、不重试。
- 运行前检查 `sgl-eval run --help`：支持 `--temperature`、`--top-p`、`--chat-template-kwarg`，不支持 `--top-k`。实际 eval 参数：`--chat-template-kwarg enable_thinking=false --temperature 0.7 --top-p 0.8`；top-k 未显式设置。
- ncu 二进制报告与 M2 profile 目录未上传；无超过 20 MB 的 CSV。测试结束后 GPU 0 无计算进程。
