# ShardTrain：FSDP 参数生命周期、异步调度与四卡性能评估

ShardTrain 是使用普通 PyTorch Tensor、自定义 Autograd 和 NCCL 自主实现的研究型 FSDP 训练 Runtime，支持 FSDP-only 与 FSDP × TP/SP，在四张 V100 上运行完整 28 层 Qwen2.5-1.5B。工作重点是 FSDP 参数生命周期与 Autograd Bridge、异步通信依赖与资源生命周期管理，以及端到端并行策略和性能分析。TP/SP 算子及共享激活 AG 为前两项提供计算与通信基础。官方 FSDP2 仅作为独立对照，不是自研路径的实现基础。

## 一、FSDP 参数生命周期与 Autograd Bridge

### 稳定 Shard Leaf、临时 FullParam 与 Reshard

普通算子的 Autograd 可能通过 SavedVariable 保存完整权重或其 view，即使模块已清空参数字段，底层 Storage 仍有引用。把 optimizer 绑定到每次新物化的参数，还会破坏参数身份与状态的一致性。

我们把稳定的 FP32 Shard Leaf 作为唯一优化器参数。AllGather（AG）生成临时 FullParam，`UnitMaterialize` 用一个多输出 Autograd Bridge 连接 shard 与各参数算子。消费者的自定义反向保存必要的局部激活与 provider，从当前物化版本重新取得权重；前向结束后按策略 Reshard，清空临时参数槽、释放 lease 并断开物化存储引用。`PreBackward` 在参数反向计算前重新物化，Bridge 收齐参数梯度后通过 ReduceScatter（RS）回传到原 Shard Leaf，由 Autograd 累积到持久 `.grad`，供 optimizer 更新。

这个边界同时固定了 optimizer 身份和 full-parameter 的生存期。释放仍需满足 GPU 最后读取完成，紧凑激活也会保留到反向；最后一个 unit、共享 embedding/head 等按策略保留参数。相关机制分别在 [bridge](../src/sharded_runtime/runtime/bridge.py)、[materializer](../src/sharded_runtime/runtime/materializer.py) 与 [Qwen 参数消费者](../src/sharded_runtime/runtime/qwen2_operators.py) 中实现。

### TP/SP：参数消费者与通信基础

TP 的 Column/Row Parallel Linear 消费 TP-local 参数，SP 管理激活的序列分片及对应 AG/RS。它们的自定义反向需要与参数重新物化、梯度回传和 buffer 生命周期配合，是 FSDP × TP/SP 训练路径的实现基础。

embedding与Norm采用TP复制、FSDP分片的组合：F2T2中，它们在TP轴保持复制，但每份仍沿两路FSDP切分；F4T1没有TP复制，直接沿四路FSDP切分。两种布局调用同一套通用分片流程，由FSDP degree决定分片数，没有额外的FSDP-only专用拆分实现。

QKV 与 Gate/Up 分别共享输入 activation AG。前向保存 compact local activation，反向按需 regather 计算参数梯度，各投影的输入梯度汇合后再归约；这些消费者不持有前向 FullParam。实现见 [Column/Row Parallel](../src/sharded_runtime/parallel/)、[共享投影](../src/sharded_runtime/parallel/shared_projection.py)和 [Qwen 参数消费者](../src/sharded_runtime/runtime/qwen2_operators.py)。

## 二、异步通信依赖与资源生命周期管理

主机按顺序提交、GPU 开始执行和结果可消费，是三个不同的边界。若用“前一次通信完成”统一约束所有后续工作，参数预取会反过来阻塞无关的梯度归约；若只增加 stream 而不管理 buffer，则可能提前释放仍在读取的数据。

调度器只在真实消费者处等待 producer 的 CUDA Event。前向先提交当前共享 SP AG，再提交下一 unit 的参数 Prefetch，参数 AG 完成约束落在目标 unit 的 copy-out。反向在 owner U 的梯度 RS 已提交、尚未消费结果时，为即将计算的 U−1 提前提交目标 U−2 的参数 AG；Norm 归约和当前反向计算沿自己的就绪路径继续。main/norm 梯度分桶和按通信角色拆分的多个 Communicator，避免无关操作共用一条全局完成链。

Ticket 追踪通信的 producer、就绪与消费；Lease 管理 buffer 的所有权、最后读取和退休。CUDA Event 传递必要的跨流依赖，`record_stream` 保护 allocator 的异步使用，leaf 的 post-accumulate Event 标识持久 `.grad` 的最后 writer。CPU 提交返回、GPU 工作完成、Python 引用释放与存储可安全复用分别检查。

这使通信有机会与计算并发，实际收益还取决于对端是否及时提交及硬件链路。实现见 [engine](../src/sharded_runtime/runtime/engine.py)、[通信协调器](../src/sharded_runtime/runtime/comm.py) 与 [NCCL 流绑定](../src/sharded_runtime/distributed/nccl_ordering.py)；[架构文档](../docs/architecture.md)解释状态、依赖和所有权细节。

### 调度消融：保留并发相对串行的吞吐收益

实验固定同机四卡、完整随机初始化 Qwen2.5-1.5B、S512、A2、global batch 16 的自研 F2T2/SP。三个策略执行相同计算与通信，并共用通信顺序、参数首次读取和buffer复用检查：C0_guarded保留通信重叠，C1串行训练通信，C2隔离FSDP与SP/Norm通信。每策略3个独立进程，各预热10步、测量30步，正式计时关闭profiler。详细配置见[调度报告](q5_new_topology_m0_revalidation.md)。

| 策略 | Mean step（ms） | 有效 labels/s |
|---|---:|---:|
| C0_guarded：保留通信并发 | 1518.525 | 5384.17 |
| C1：串行训练通信 | 1670.076 | 4895.59 |
| C2：隔离 FSDP 与 SP/Norm | 1627.535 | 5023.55 |

保留并发相对C1/C2的吞吐高 **9.980% / 7.179%**。时间线中，串行策略最多同时运行1个NCCL kernel，保留重叠时最多3个；参数AG与SP AG、梯度RS均出现重叠。正式训练计时说明，按真实依赖调度通信能够缩短完整step。逐次运行和时间线分析见[调度报告](q5_new_topology_m0_revalidation.md)。

## 三、端到端并行策略与性能分析

### 自研与原生 FSDP2 的 2×2 对照

统一条件为同机 4×V100-SXM2 32GB、完整随机初始化的 28 层 Qwen2.5-1.5B、S512、A2、global batch 16。主要 GEMM 使用 FP16，math-SDPA BMM、master、参数 AG、梯度 RS 和 Adam 状态使用 FP32。每组 3 个独立进程，各 10 步预热、30 步测量；正式计时关闭 profiler，输入预缓存，完整 step 取四 rank 设备完成时间的最大值。

F4T1 每副本 microbatch 为 2，F2T2/SP 为 4，两者每微步均同步 FSDP 梯度。总训练工作量匹配，但局部矩阵形状、微步样本分组随策略变化，因此下面比较的是整套策略，而不是单一通信变量。

| 引擎 / 策略 | Mean step（ms） | 有效 labels/s |
|---|---:|---:|
| ShardTrain F4T1 | 1249.556 | 6543.13 |
| Native FSDP2 F4T1 | 1336.181 | 6118.93 |
| ShardTrain F2T2/SP | 1522.428 | 5370.37 |
| Native FSDP2 F2T2/SP | 1981.821 | 4125.50 |

自研FSDP-only相对自研Hybrid的吞吐高 **21.838%**；同布局下，自研F4T1/Hybrid相对原生分别高 **6.933% / 30.175%**。这说明混合并行在当前工作环境和拓扑下有额外代价。我们认为，FSDP-only省去了逐层SP激活通信和TP梯度归约，且通用FSDP分片从两路扩展为四路，降低了每卡embedding与Norm状态占用；Hybrid的TP轴包含NV1/PHB链路，并改变了局部矩阵形状。这些因素共同影响完整训练step。配置、通信次数及状态显存计算见[并行策略报告](fsdp4_vs_hybrid_2x2_native_4gpu.md)。

### 拓扑映射：物理放置会改变性能

独立的物理映射实验中，M0的两组FSDP链路均为NV2，swap23后变为PHB+NV1。自研Hybrid的step从1512.317增至2660.513ms（+75.92%），原生从1833.116增至3161.993ms（+72.49%）。独立链路测试也呈现相同方向，说明FSDP组的物理放置直接影响训练效率。该实验与上面的策略对照分开运行和统计；详见[物理放置研究](topology_mapping_study.md)。

### Rank Skew：分片均衡不等于执行均衡

参数和梯度shard字节相等，并不保证同组rank同时提交通信。本批基准策略的同一个decoder AG中，早到rank的kernel持续52.913ms，对端仅1.555ms，两者结束相差约0.028ms。跨rank提交偏斜会增加早到方的等待，因此需要结合整组rank分析通信时长。

另一个旧拓扑批次对指定rank注入CPU等待与独立GPU工作：小延迟可被前序队列隐藏，较大延迟使早到方等待更久。该控制实验解释了偏斜传播过程，详见[Rank Skew实验](fsdp_interview_sections/07_rank_skew.md)与[本批通信观测](q5_new_topology_m0_revalidation.md)。

## 正确性检查与复现

已覆盖小模型 FP32 独立对照、FP16 same-start、AMP overflow 跳步与恢复，以及完整模型容量和短程有限更新。当前验证范围为单机四卡和指定模型路径。配置与执行方法见复现手册，性能统计可由公开 CSV 复算。

[Reproduce：复现手册](../docs/reproduction_4gpu.md) · [Evidence：实验与证据目录](../docs/experiment_catalog.md) · [Limitations：完整限制](../docs/known_limitations.md)
