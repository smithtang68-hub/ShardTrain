# 四卡实验汇总与阅读入口

本页汇总四卡实验。[主技术报告](shardtrain_final_project_report.md)解释参数生命周期、异步调度和性能分析；下面两份性能报告提供具体配置、统计与结论。

FSDP参数生命周期与Autograd Bridge先读主报告；异步通信重叠见[调度实验](q5_new_topology_m0_revalidation.md)，端到端策略见[两引擎×两策略对照](fsdp4_vs_hybrid_2x2_native_4gpu.md)。补充案例包括[物理放置](topology_mapping_study.md)与[Rank Skew](fsdp_interview_sections/07_rank_skew.md)。TP/SP Column/Row与共享AG是运行这些路径的基础。

| 研究内容 | 结果 | 深入阅读 |
|---|---|---|
| 并行策略与引擎对照 | 自研在两个同布局对照中均更快；两引擎都是F4T1快于Hybrid | [并行策略对照](fsdp4_vs_hybrid_2x2_native_4gpu.md) |
| 通信重叠收益 | 相对全部通信串行吞吐提高9.980%，相对分类隔离提高7.179% | [调度消融](q5_new_topology_m0_revalidation.md) |

策略对照为 4组×3fresh run，调度消融为 3组×3fresh run；每 run 10 warmup+30 measured，共 **21 次独立运行、630 个测量 step**。正式吞吐使用四 rank 最大完整 step，profiling 样本单列。跨并行策略微批形状不同。

并行策略与调度实验分别运行和统计；[物理放置实验](topology_mapping_study.md)则固定Hybrid布局比较GPU映射，结果单独列出。

公开的[逐run CSV](../docs/release/p1_p2_runs.csv)和[逐step CSV](../docs/release/p1_p2_measured_steps.csv)支持统计复算。

[复现](../docs/reproduction_4gpu.md) · [完整限制](../docs/known_limitations.md)
