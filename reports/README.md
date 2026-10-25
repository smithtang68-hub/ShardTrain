# 报告索引

[主技术报告](shardtrain_final_project_report.md)围绕FSDP参数生命周期、异步通信与资源管理、端到端并行策略展开。

## 项目与四卡实验

| 报告 | 内容 |
|---|---|
| [主技术报告](shardtrain_final_project_report.md) | 参数生命周期、Autograd Bridge、异步调度和策略性能 |
| [项目总结](shardtrain_project_summary.md) | 主要实现与四卡结果概览 |
| [自研/原生 × FSDP-only/Hybrid](fsdp4_vs_hybrid_2x2_native_4gpu.md) | 四组配置的耗时、吞吐、归约和状态显存 |
| [通信调度对照](q5_new_topology_m0_revalidation.md) | 通信重叠、串行、分类隔离及9.980%吞吐收益 |
| [四卡实验汇总](shardtrain_final_p0_p1_p2_summary.md) | 正式结果和公开CSV入口 |
| [四卡运行验证](shardtrain_4gpu_finalization_report.md) | 参考正确性、完整模型容量、参数与通信检查 |
| [GPU物理映射](topology_mapping_study.md) | FSDP链路放置对两种引擎的性能影响 |
| [Qwen综合分析](qwen_integration_and_diagnostics.md) | 模型适配、stream/prefetch、AG、CPU提交和负载扩展 |

## FSDP专项：一份总述与十份实验

[专项总述](fsdp_interview_experiments.md)集中给出环境与拓扑、S512/microbatch及容量选择、控制变量、计时与profiling方法，以及实验1—10的结果总览。

| 实验 | 报告 |
|---|---|
| 1 | [Pinned transfer](fsdp_interview_sections/01_pinned_transfer.md) |
| 2 | [Pinned allocator](fsdp_interview_sections/02_pinned_allocator.md) |
| 3 | [通信重叠](fsdp_interview_sections/03_communication_overlap.md) |
| 4 | [通信次数与payload](fsdp_interview_sections/04_communication_volume.md) |
| 5 | [参数表示与model state恢复](fsdp_interview_sections/05_parameter_representation.md) |
| 6 | [原生FSDP调度](fsdp_interview_sections/06_native_fsdp_scheduling.md) |
| 7 | [Rank skew](fsdp_interview_sections/07_rank_skew.md) |
| 8 | [本地GEMM](fsdp_interview_sections/08_gemm_profile.md) |
| 9 | [并发与串行计算量](fsdp_interview_sections/09_gemm_concurrency_comparison.md) |
| 10 | [自研与原生计算量](fsdp_interview_sections/10_gemm_native_comparison.md) |

## 性能数据

[逐run CSV](../docs/release/p1_p2_runs.csv) · [逐step CSV](../docs/release/p1_p2_measured_steps.csv) · [测量方法](../docs/benchmark_methodology.md) · [运行与复现](../docs/reproduction_4gpu.md)

报告正文与图表呈现实验结果，CSV保留测量数值。原始trace、memory snapshot与本地运行记录不随仓库发布。
