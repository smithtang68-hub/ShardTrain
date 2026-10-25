# ShardTrain 运行时架构

ShardTrain 把模型计算、参数存储和异步通信分开管理。Trainer 只通过公开接口驱动更新；runtime 拥有持久 shard、临时完整参数、梯度与通信 buffer 的生命周期。当前 Qwen 适配器与早期标准 Block 共用这些机制。设计动机和实测结果见[主报告](../reports/shardtrain_final_project_report.md)，本文解释实现依赖。

架构围绕三个重点展开：自主 FSDP 参数生命周期与 Autograd Bridge，负责稳定 Shard Leaf、临时 FullParam、Reshard、反向 rematerialization 和梯度回传；异步通信依赖与资源生命周期，负责多 communicator、prefetch、必要 Event 以及 ticket/lease；端到端并行策略与性能分析，连接自研 / 原生 FSDP2 × F4T1 / F2T2-SP 的 2×2 对照及 topology、rank skew 证据。TP/SP 的 Column/Row 算子和共享激活 AG 是前两项的实现基础。

## 模块与依赖

| 层 | 职责 | 源码入口 |
|---|---|---|
| 配置与分布式上下文 | 验证 mesh，按确定顺序创建有序 process groups，绑定 device | [context](../src/sharded_runtime/distributed/context.py)、[topology](../src/sharded_runtime/distributed/topology.py) |
| 模型与并行算子 | Qwen 数学、TP 切分、SP 激活布局、共享投影 | [Qwen 模型](../src/sharded_runtime/models/qwen2.py)、[parallel](../src/sharded_runtime/parallel/) |
| 注册与物化 | TP-local 参数到 FSDP shard 的布局、tied owner、packed AG 和 copy-out | [参数布局](../src/sharded_runtime/runtime/layout.py)、[materializer](../src/sharded_runtime/runtime/materializer.py) |
| Autograd 与生命周期 | 稳定 shard leaf、多输出 bridge、反向 rematerialization、梯度派发 | [bridge](../src/sharded_runtime/runtime/bridge.py)、[engine](../src/sharded_runtime/runtime/engine.py) |
| 通信协调 | producer/consumer Event、ticket/lease、安全回收与物理流绑定 | [comm](../src/sharded_runtime/runtime/comm.py)、[NCCL ordering](../src/sharded_runtime/distributed/nccl_ordering.py) |
| 训练 | AMP、梯度累积、全组 skip、AdamW、完整 step 计时 | [trainer](../src/sharded_runtime/training/qwen2_trainer.py)、[benchmark](../src/sharded_runtime/training/qwen2_benchmark.py) |

逻辑 rank 为 `f * tp_degree + t`。TP/SP group 固定 f、改变 t；FSDP group 固定 t、改变 f。SP 共用 TP 的逻辑成员，不增加 world size。不同通信角色可用成员相同的独立 communicator；成员相同不等于 stream 与完成链相同。

## 持久参数与临时物化

参数先做 TP 切分，再对 TP-local tensor 做 flat FSDP 分片。Registry 保存逻辑形状、padding、owner 与稳定的 FP32 local shard；optimizer 只持有这些 leaf。临时完整参数不会替换优化器的对象。

每个 unit 的一次使用依次经历：

```text
RESHARDED -> MATERIALIZING -> copy-out ready -> FWD/BWD ACTIVE
          -> release（或策略性 retain）-> RESHARDED
反向：full parameter gradients -> Norm/main RS -> shard gradients -> optimizer
```

这里只展示职责边界，实际状态名及合法转换由 engine 检查。AG 输出先按通信布局收集，再在 compute 流 copy-out 成消费者所需的独立 tensor；目标 unit 只在消费时等待自己的 producer。详见[参数合同](parameter_autograd_contract.md)和[copy-out 布局](fsdp_copy_out_layout.md)。

## Autograd 如何避免完整权重驻留

`UnitMaterialize` 是从持久 shard 到多份临时参数的一个多输出 Function，其 backward 等各输出梯度就绪后统一回传。`PreBackward` 在参数 DGrad 之前恢复当前 unit 的物化。Qwen 自定义算子保存 compact input、形状及参数 provider，反向从当前 generation 取权重，而不将前向 FullWeight 放入 `save_for_backward`。

provider/handle/unit 本身仍是强引用。因此释放不仅依靠 saved-tensor 选择，还依靠 `compute_weight = None`、物化 lease 退休及内部字典清空，断开指向前向 Storage 的链。激活保存到反向，最后 unit/tied owner 或 retain 配置可保留完整参数；不能声称任何时刻都只有分片。通用 ParameterRMSNorm 对任意小 view 的 compact 边界见[实现范围](known_limitations.md#autogradstorage-与内存)。

## TP/SP 实现基础：通信与伴随操作

Column/Row Linear 的自定义反向分别处理输入梯度聚合、参数局部梯度及序列分片。QKV 和 Gate/Up 共享一次完整输入，由各投影消费；保存局部紧凑输入、反向 regather，然后合并输入梯度。ReduceScatter 与 AllGather 在前反向中按真实布局和归约语义配对，复制参数的梯度再做 TP SUM，不能用普通 concat/chunk 替代通信语义。

标准 Block 与 Qwen 适配器各有明确算子合同，详见[TP/SP 合同](tp_sp_contracts.md)和 [Qwen 消费者](../src/sharded_runtime/runtime/qwen2_operators.py)。共享 AG 的历史 5→2 计数只属于标准 Block 的前向共享激活范围。

## 预取、物理流与 buffer 所有权

前向当前 SP AG 提交后，发起下一个 unit 的参数 AG，再在当前真实消费者处消费 SP 结果。预取参数完成只约束目标 copy-out，不建立“无关 RS 等参数 AG”的全局边。

反向 main/norm 梯度分桶：先提交小 Norm RS 与大 main RS，再在消费这些结果前做 gradient-dispatch lookahead。owner U 已提交 RS 时，current U−1 尚未进入，runtime 为 target U−2 提交参数预取；Norm SUM 可沿 compute 分支运行，当前反向计算按数据就绪推进。边界 unit 与 tied owner 有单独状态检查。[materializer.reduce](../src/sharded_runtime/runtime/materializer.py)和 [dispatch_backward_prefetch](../src/sharded_runtime/runtime/engine.py)是对应入口。

五个逻辑 channel 为 compute、FSDP AG/RS、SP AG/RS；实际执行通过 NCCL 流绑定实现并核验，不由 stream 数量推断 overlap。物理流验收限定于已资格验证的 PyTorch 2.9.1 环境，不外推其他后端或版本。通信 ticket 标识 producer、就绪与消费；lease 标识 buffer 的最后读取和回收。CUDA Event 负责跨流数据依赖，`record_stream` 保护 allocator 生命周期。Autograd 的 post-accumulate Event 才是持久 `.grad` 最后 writer 的边界。详见[运行时生命周期](runtime_lifecycle.md)。

同机受控调度消融中，保留通信并发相对通信串行吞吐高 **9.980%**；新增 Event wait 的提交与调度成本也包含在差额中。该结果支持当前负载的端到端调度选择，不能全部解释为线路同时传输的节省；条件与另一个隔离对照见[调度研究](../reports/q5_new_topology_m0_revalidation.md)。

## 端到端并行策略与性能

训练层把同一模型和全局 batch 映射到 F4T1 或 F2T2/SP，与独立官方 FSDP2 路径组成 2×2 对照。完整 step 包含前反向、梯度收尾、AMP、AdamW 和设备完成，取四 rank 最大完成时间；[并行策略研究](../reports/fsdp4_vs_hybrid_2x2_native_4gpu.md)记录匹配条件、吞吐、显存和通信调用。它评估指定负载下的整体策略，原生最优配置尚未搜索。

topology 决定逻辑通信组的物理链路放置；rank skew 则会让早到 rank 的 NCCL 驻留包含等待 peer。两者需要结合跨 rank trace 与完整 step 分析，不能由单 rank 的长 kernel 或驻留交集直接推断带宽和收益。[物理放置研究](../reports/topology_mapping_study.md)、[Rank Skew 注入实验](../reports/fsdp_interview_sections/07_rank_skew.md)与[本批调度研究](../reports/q5_new_topology_m0_revalidation.md)保留各自运行批次和归因边界；自然提交偏斜的根因仍未闭合。

## 证据与实现范围

正式性能来自关闭 profiler 的完整 step；短 trace 用于核验实际操作归属与 NCCL 驻留。驻留交集含 peer 等待，不能直接证明 wire overlap 或全部 GEMM 被掩盖。实现与实验范围见[限制说明](known_limitations.md)。

官方 FSDP2/DTensor 仅属于[独立原生基线](qwen25_precision_and_native_comparison.md)。
