# 项目实现与实验总结

ShardTrain面向单机四卡训练，自主实现FSDP参数管理与Autograd连接，并支持FSDP-only及FSDP×TP/SP。项目围绕参数生命周期、异步依赖与资源管理、端到端策略性能三个重点展开。

## 主要实现与实验结果

| 工作 | 实现与结果 | 深入阅读 |
|---|---|---|
| FSDP参数生命周期与Autograd Bridge | 稳定Shard Leaf绑定optimizer；临时FullParam按需物化、用后Reshard，反向重新物化并将梯度送回原Shard Leaf | [主报告](shardtrain_final_project_report.md)、[参数契约](../docs/parameter_autograd_contract.md) |
| 异步通信依赖与资源管理 | 多Communicator、前后向参数预取、必要Event依赖、Ticket/Lease管理通信buffer；相对全部通信串行吞吐提升9.980% | [调度实验](q5_new_topology_m0_revalidation.md) |
| 端到端并行策略与性能 | 完成自研/原生FSDP2×FSDP-only/Hybrid的2×2对照，并结合物理拓扑和跨rank提交偏斜解释性能 | [策略对照](fsdp4_vs_hybrid_2x2_native_4gpu.md)、[拓扑研究](topology_mapping_study.md) |
| TP/SP实现基础 | Column/Row Linear、QKV/Gate-Up共享激活AG、紧凑局部激活与反向重新聚合，支撑参数与通信生命周期 | [架构](../docs/architecture.md) |
| 完整模型集成与运行诊断 | 完整28层Qwen2.5-1.5B、共享embedding/head、参考数值验证、精度匹配、CPU提交优化与工作量选择 | [Qwen整合报告](qwen_integration_and_diagnostics.md) |

## 四卡性能结果

在完整Qwen2.5-1.5B、S512、梯度累积2和全局批量16的策略对照中，自研FSDP-only与Hybrid平均步时分别为1249.556ms、1522.428ms，原生对应为1336.181ms、1981.821ms。自研在两个同布局对照中的吞吐分别高6.933%、30.175%，两种引擎都是FSDP-only更快。

固定Hybrid的调度实验中，保留通信重叠相对全部串行、按通信类型隔离的吞吐分别高9.980%、7.179%。拓扑映射实验进一步显示，让FSDP组使用两条NV2连接有利于参数聚合和梯度归约。

## 报告与复现入口

[主技术报告](shardtrain_final_project_report.md)说明设计与性能分析，[FSDP专项实验](fsdp_interview_experiments.md)覆盖内存、通信、参数表示和计算量核对，[报告索引](README.md)提供完整导航，[复现手册](../docs/reproduction_4gpu.md)提供公开统计复算与四卡运行步骤。
