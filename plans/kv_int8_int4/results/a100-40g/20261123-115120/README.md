# INT8 decode 消融实验

- 测试代码提交 `9c86d5c`；A100-PCIE-40GB，6×16384 上下文，32 Q / 8 KV heads，head dimension 128，最大 KV splits 8，池大小 200000。
- 8 组冷 L2 计时和精度指标输出完成；基准退出码 0。两组 ncu `--set full` 采集及 details、raw、SASS 导出退出码均为 0。
- ncu 二进制报告只保留在本地。两组 stage-1 均记录 255 registers/thread，local load/store requests 均为 0；这些是本次 ncu 计数，不能单独证明所有编译器 spill 情况。
