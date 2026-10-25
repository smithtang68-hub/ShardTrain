# 实现与实验范围

ShardTrain 面向 Qwen2.5 的单机四 GPU 研究场景。完整模型实测覆盖 F2T2/SP 和 F4T1；当前不支持任意 PyTorch Module 的透明包装，也没有八 GPU、多机、长期预训练收敛或模型质量结论。

## 数值与模型能力

FP32 参考、小规模梯度/更新比较、AMP skip/recovery、参数 ownership 与完整模型短程训练分别验证不同层面的行为。低精度并行归约改变求值顺序，短程数值检查不构成完整 1.5B 模型的长期 dense 轨迹等价证明。

模型适配依赖静态 unit 顺序和明确的参数消费者。任意共享图、动态分支、unused branches、重复 backward、retain_graph 与高阶导数不由已测路径直接推广。跨卡数恢复实验覆盖 model state，未证明 optimizer 续训恢复。

V100 无原生 BF16 支持。当前 Linear/MLP/head 使用 FP16，math-SDPA BMM、master parameters、参数 AG、梯度 RS 和 Adam state 使用 FP32。实测效率不外推到 Ampere/Hopper 或其他 attention backend。

## Autograd、Storage 与内存

Qwen 参数消费者不保存前向 FullWeight，反向读取当前 generation 的权重；前向 Storage 释放还依赖 compute_weight 清空、lease 退休和内部引用断开。compact local activation 仍按需保留到反向。最后 unit、tied owner 或 retain 策略可保留完整参数。

通用 ParameterRMSNorm 直接保存输入，对任意小 view 尚无统一 compact 保证；QwenNorm 的 compact 行为不能推广到所有 generic consumer。引用释放之后仍需 GPU 使用结束才可安全复用。reserved 包含 allocator cache，不能据其未下降判断 Storage 引用未释放。

端点 snapshot 和短窗口无净增长不能证明长期无泄漏或全部峰值分配来源。

## 性能与通信分析

原生基线使用官方 fully_shard/DTensor，但没有搜索其全部最优配置。两条路径的分桶、分片布局、归约顺序和 optimizer 执行存在差异；同 payload/GEMM 工作量不等于逐 kernel 同构。性能提升只针对报告中的设备、负载和配置。

每组只有 3 次独立进程运行，同一次运行内的 step 相关。吞吐计时覆盖完整训练 step，输入已预缓存于 GPU，因此不代表数据读取和 H2D 在内的完整数据管道吞吐。

NCCL 驻留包含 peer 等待；驻留交集不等于 wire overlap。当前短 trace 不足以恢复完整 CUDA Event DAG，或证明所有层都实现同层三重计算重叠。自然 rank 提交偏斜及个别慢 step 仍需更有针对性的采集。

不同设备拓扑的实验分别统计。物理 GPU 映射可能把 FSDP 从 NVLink 改为 PHB 链路；单 rank 的长 AG 也可能来自等待其他 rank，不能单凭其长度推断有效带宽。

[主报告](../reports/shardtrain_final_project_report.md) · [测量方法](benchmark_methodology.md) · [报告索引](../reports/README.md)
