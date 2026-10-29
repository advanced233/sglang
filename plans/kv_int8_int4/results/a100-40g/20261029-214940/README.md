# 阶段 1 M2 补跑

- 本次是 `20261029-211615` 的 M2 冒烟补跑；测试代码提交 `0ee8ede`。开始时间 2026-10-29 21:49:40，结束时间 21:59:57（Asia/Shanghai），约 10 分 17 秒。
- 只运行第 3 节。BF16 和 INT8 服务均就绪，短请求均回答 391；两组 decode 探测均成功，GSM8K 均完成 1319 题。
- 运行前核对 `sgl-eval run --help`：使用 `--chat-template-kwarg enable_thinking=false --temperature 0.7 --top-p 0.8`；该版本不支持 `--top-k`，未显式设置。
- 无失败项或重试项。M2 profile 目录仅保留本地；运行结束后 GPU 0 无计算进程。
