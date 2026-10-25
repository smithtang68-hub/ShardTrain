# ShardTrain

**自研 FSDP × TP/SP 训练 Runtime，研究参数生命周期、异步通信调度与端到端并行策略。**

ShardTrain 基于 PyTorch Tensor、Autograd 和 NCCL 实现参数分片、按需物化及梯度回传。它支持 FSDP-only 与 FSDP × Tensor/Sequence Parallelism（TP/SP），已在单机四张 V100 上运行完整 28 层 Qwen2.5-1.5B。自研路径使用普通 Tensor；官方 FSDP2 的 `fully_shard` 和 DTensor 只用于独立对照。

## 核心工作

- **自主 FSDP 参数生命周期与 Autograd Bridge**：optimizer 绑定稳定 FP32 Shard Leaf；AllGather（AG）生成临时 FullParam，经 `UnitMaterialize` 接入 Autograd，按策略 Reshard、反向按需 rematerialization，再由 ReduceScatter（RS）将梯度送回持久 shard。
- **异步通信依赖与资源生命周期**：多 communicator 与参数 prefetch 按消费者依赖推进，通过必要的跨流 Event 表达完成依赖；ticket/lease 管理 GPU 完成、最后读取与 buffer 复用。
- **端到端并行策略与性能分析**：以自研 / 原生 FSDP2 × F4T1 / F2T2-SP 的 2×2 对照评估完整训练 step，并结合 topology 与 rank skew 分析通信驻留和性能差异。

TP/SP 的 Column/Row Linear、自定义伴随操作以及 QKV、Gate/Up 共享 SP activation AG，是前两项在混合并行路径上的实现基础：保存紧凑局部激活、反向重新聚合，并维持参数与激活的通信语义。

## 四卡实测

固定 matched 对照：4×V100-SXM2 32GB、完整随机初始化 Qwen2.5-1.5B、S512、梯度累积 A2、全局 batch 16；每组 3 次独立运行，各 10 步预热、30 步测量，关闭 profiler。主要 GEMM 为 FP16，math attention、参数及梯度通信为 FP32。

| 引擎 / 并行策略 | Mean step（ms） | 有效 labels/s |
|---|---:|---:|
| ShardTrain F4T1 | 1249.556 | 6543.13 |
| Native FSDP2 F4T1 | 1336.181 | 6118.93 |
| ShardTrain F2T2/SP | 1522.428 | 5370.37 |
| Native FSDP2 F2T2/SP | 1981.821 | 4125.50 |

同布局下，自研吞吐分别高 **6.933% / 30.175%**。另一组同机调度消融中，保留通信并发比串行化、分类隔离高 **9.980% / 7.179%**。这是指定负载与匹配路径的结果；原生最优配置尚未搜索。

## 运行与复现

安装和四卡运行见[复现手册](docs/reproduction_4gpu.md)。[逐 run CSV](docs/release/p1_p2_runs.csv)与[逐 step CSV](docs/release/p1_p2_measured_steps.csv)保留性能测量，统计公式见[测量方法](docs/benchmark_methodology.md)。原始 trace、memory snapshot 和输入数据保留在本地，不随仓库发布。

## 边界与文档

当前验证限单机四卡和指定模型路径，覆盖小规模模型的数值检查、参数与通信 buffer 生命周期，以及完整模型的容量和短程训练。八卡、多机、长期收敛与任意 Module 透明包装属于后续工作。

[主技术报告](reports/shardtrain_final_project_report.md) · [实现架构](docs/architecture.md) · [实现范围](docs/known_limitations.md) · [实验目录](docs/experiment_catalog.md) · [报告索引](reports/README.md)
