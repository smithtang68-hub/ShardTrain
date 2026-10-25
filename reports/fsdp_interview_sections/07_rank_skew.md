# 实验7：Rank skew

## 背景

相同大小的参数和梯度 shard 只能约束存储量，不能保证各 rank 同时提交和执行 collective。本实验在固定通信位置向一个 rank 注入延迟，检验长 NCCL 驻留是否可以由等待晚到 peer 产生，以及 CPU 延迟在什么条件下会传递到 GPU 时间线。

目标 collective 的 GPU 开始时间同时受 API 提交和前序工作约束。若目标 rank 在注入位置之后原有的队列余量为 s，新增 CPU 延迟为 δ，在前序工作不变的近似下，目标 kernel 的开始时间增量应为：

```text
ΔGPU_start ≈ max(0, δ - s)
```

因此，2ms 或10ms 的 CPU 延迟不必等量变成 GPU 开始偏斜；只有延迟超过可隐藏余量，目标 rank 才会更晚进入 collective。这里不预设 s 的具体数值，需结合目标通信之前的时间线判断。

对于同一 collective，早到 rank 可以先启动 NCCL kernel 并等待 peer。若两者接近同时结束，驻留时间差应近似等于开始偏斜：

```text
duration_early - duration_late ≈ start_late - start_early
```

这给出可检查的预期：较大到达偏斜应使早到 rank 驻留更长，而不要求晚到 rank 的实际通信阶段同比变慢。额外 GPU 工作放在提交流上时，也可能消耗队列余量；完整 step 还受其他 rank 和其他阶段约束，不预设其随注入时长严格单调增加。

## 实验

使用完整 Qwen2.5-1.5B、序列长度 512、每小批量 4 条序列、累积 2 个小批量更新一次，FSDP degree 2、TP degree 2、SP 开启。目标固定为 rank0、unit1、前向参数 AG、首 microstep 的首候选，避免共享输出头二次调用重复注入。条件为 CPU 0/2/10/50ms，以及在实际提交流执行 64 次预分配 FP16`2048×2048×2048` GEMM；训练 Tensor 与通信数学不变。

15 个正式运行、450 次注入命中与固定输入核对通过；每条件三个独立进程，关闭 profiler。五次短采集的通信组、序号、形状及字节数序列核对一致，目标通信组为 `(0,2)`，以下到达时间均比较这两个 rank。

| 目标 collective 的短时间线 | API 提交偏斜（ms） | GPU 开始偏斜（ms） |
|---|---:|---:|
| CPU2ms | 2.243 | −0.005 |
| CPU10ms | 12.483 | −0.041 |
| CPU50ms | 50.306 | 29.533 |

差值方向为 rank0−rank2。前两次采集中，目标 API 发出时前序 AG 仍在执行，CPU 偏斜被队列隐藏。CPU50ms 中 rank2 早到并驻留 32.320ms，rank0 驻留 2.782ms，两者结束差约 6μs。GPU 工作条件同样观察到早到 rank 等待，该条件的实际延迟同时包含 GPU 工作和 CPU 提交成本。

完整 step 未随 CPU 等待时长单调增长，独立运行也存在波动。因此机制结论来自每条件一次短采集中确认的队列与等待关系，正式吞吐则使用关闭 profiler 的独立运行。

## 结论

受控注入显示：较小的 CPU 提交延迟可被前序队列隐藏，较大的到达偏斜会拉长早到 rank 的 NCCL 驻留。

- CPU2ms、CPU10ms 的 API 提交偏斜分别为 2.243、12.483ms，GPU 开始偏斜却接近零，说明前序 GPU 队列吸收了这两次 CPU 延迟。
- CPU50ms 中，早到 rank 驻留 32.320ms、晚到 rank 驻留 2.782ms，差值 29.538ms 与 29.533ms 的开始偏斜接近，两者结束仅差约6μs。长驻留主要来自等待晚到 peer；分析该类通信耗时时，应联合各 rank 的到达和完成时刻。

[延迟注入实现](../../experiments/fsdp_interview/cases/q12_rank_skew.py) · [实验3：通信并发](03_communication_overlap.md)
