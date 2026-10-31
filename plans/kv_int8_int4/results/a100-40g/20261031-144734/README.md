# 阶段 2 M3b 运行记录

- 代码提交 `9ec83af`；第 0、1、2、3 节执行完成。
- CPU 测试 19 passed，GPU 测试 21 passed；无失败项。
- BF16 和 INT8 冒烟均成功，GSM8K 各完成 1319 题；服务测速各项状态均为 ok，无失败项。
- `serve/summary.md` 的 decode 表标题写 output 256，但实际命令及 runs.jsonl 记录的输出长度为 1024；原始汇总未修改。
- profile 目录未上传；测试结束后 GPU 0 无计算进程。
