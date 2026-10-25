# 实验6：原生FSDP调度对照

## 背景

官方 FSDP1 与 FSDP2 都支持参数分片和异步预取，但参数组织、packing 与 buffer 生命周期不同。本实验在相同数学工作量和同步策略下比较两条路径，检验这些执行差别是否表现为完整 step 与显存差异。

若模型、输入、精度、参数保留和梯度同步一致，两条路径应得到相同的逻辑计算、梯度平均和参数更新。FSDP1 的 flat 参数与 FSDP2 的参数组会改变实际 buffer 组织及调用方式，但不能以减少模型工作量解释性能差异。

显存预期需要分开描述：持久状态量相同，并不意味着峰值和 reserved 必须相同。峰值还受同时存活的激活、FullParam、通信及临时 buffer 影响：

```text
M_peak_allocated = M_persistent
                 + max_t(M_activation(t) + M_fullparam(t)
                       + M_communication(t) + M_temporary(t))
```

reserved 另外受到 allocator cache 与分配策略影响。两边都具有预取和异步管理，只能预期其具备隐藏通信的机会，不能据 API 代际先判定谁更快或预先给出步时差。

本实验要回答的是：对齐策略后，数值是否一致；完整 step 和峰值显存有多大差异；现有 trace 能解释到哪一层。

## 实验

模型为 8 层、H512、8 个注意力头、FFN2048、vocab4096，共 29414400 参数。四卡 FSDP，TP/SP=1，S512、microbatch 4、accumulation 2，合成输入驻留 GPU。参数、计算、AG、梯度归约和 AdamW state 均为 FP32，使用 math attention。

两边都保留根模块完整参数、在子模块前向后 reshard；首 microstep 不同步梯度，最后一次同步。FSDP1 使用 `no_sync` 和 BACKWARD_PRE；FSDP2 使用公开梯度同步 API 与显式相邻预取。每条路径使用三个独立进程，每次 10 warmup＋30 measured steps，正式计时关闭 profiler。

独立 dense reference 检查输出、loss、梯度和更新；另外采集两组四 rank trace 与 16 份显存 snapshot，观察 stream、通信和 allocator 状态。

| 官方路径 | Mean step（ms） | 峰值 allocated（GiB） | 峰值 reserved（GiB） |
|---|---:|---:|---:|
| FSDP1 | 105.655 | 1.112958 | 1.310547 |
| FSDP2 | 151.059 | 1.113107 | 1.216797 |

两边每 rank 的 203 项 dense reference 检查全部通过，包含两个 microstep 输出、平均 loss、100 个 gradient 和 100 个 AdamW 更新，符合对齐工作量后的数学预期。

正式步时在本配置中由 FSDP1 更短；allocated 峰值接近，reserved 则由 FSDP2 更低。结果说明 API 更新不保证该负载下更快，也说明 reserved 的差额不能直接解释为模型持久状态减少。速度排序是本次测量所得，机制预期本身没有指定该排序。

## 结论

本次中型 FP32、FSDP-only 配置下，FSDP1 的平均 step 为 105.655ms，短于 FSDP2 的 151.059ms；FSDP2 的 reserved 峰值更低。所以FSDP2可以认为是用部分效率换取自定义泛化性。

- 两者 allocated 峰值接近，分别为 1.112958 与 1.113107GiB；reserved 峰值分别为 1.310547 与 1.216797GiB。该配置下，FSDP1训练步更短，FSDP2 allocator预留更少。
- 两边每 rank 的 203 项 dense reference 检查均通过，模型工作量、梯度同步和更新语义已对齐。这组结果提供了同条件下的速度与显存取舍。

[独立性能测试实现](../../experiments/fsdp_interview/cases/q10_fsdp_only_benchmark.py) · [专项总述](../fsdp_interview_experiments.md)
