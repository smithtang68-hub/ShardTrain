# 实验目录

| 实验 | 配置与比较 | 报告 |
|---|---|---|
| 四卡并行策略 | 自研/原生 FSDP2 × F4T1/F2T2-SP；S512、A2、global batch 16 | [策略对照](../reports/fsdp4_vs_hybrid_2x2_native_4gpu.md) |
| 通信调度 | 固定 Hybrid，比较重叠、串行和分类隔离 | [调度对照](../reports/q5_new_topology_m0_revalidation.md) |
| GPU物理映射 | 相同逻辑布局、不同物理链路放置 | [拓扑研究](../reports/topology_mapping_study.md) |
| FSDP专项1—10 | Pinned copy/allocator、通信、参数表示、原生调度、rank skew、GEMM | [方案与结果总览](../reports/fsdp_interview_experiments.md) |
| Qwen运行诊断 | 模型适配、stream/prefetch、AG驻留、CPU提交、负载扩展 | [Qwen综合分析](../reports/qwen_integration_and_diagnostics.md) |

正式性能使用 profiler-off 独立进程，机制分析使用另外采集的短 trace。完整结果见[报告索引](../reports/README.md)，计量方法见[性能测量方法](benchmark_methodology.md)。
