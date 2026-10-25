# FSDP 专项实验

本目录覆盖 pinned memory、通信调度、通信量、参数表示、官方 FSDP 调度、Rank Skew 和实际 GEMM 工作量。实验 1—10 的方案与结果总览见[专项总报告](../../reports/fsdp_interview_experiments.md)，各实验的详细分析从该报告进入。

## 主负载与计时

完整随机初始化 Qwen2.5-1.5B：28 层、H1536、Q12/KV2、FFN8960、词表 151936、共享 embedding/head。采用 S512、每个 FSDP 副本每微步 microbatch 4、梯度累积 2、F2T2/SP，前后向参数预取与异步梯度 RS 均开启。

自研使用自己的 Tensor 分片实现，原生调用官方 `fully_shard`。双方线性计算为 FP16，master 参数、AG/RS、optimizer 和 math-attention BMM 为 FP32。全局每步 16 条序列、8192 个输入位置、8176 个有效 next-token 标签，TP/SP 不重复计数。

主负载由 WikiText-103 train 文本连续打包成 1024 个 S512 block。每次计时 10 步预热、30 步测量，消费 640 个 block，GPU 输入预缓存位于计时外；3 个独立进程按旋转顺序执行。profiling 使用独立进程和短采集窗口，其时间不混入正式吞吐。

## 源码与接口

| 用途 | 源码 |
|---|---|
| 环境与拓扑 | [scripts/env_snapshot.py](scripts/env_snapshot.py) |
| 文本输入准备 | [scripts/prepare_data_rows.py](scripts/prepare_data_rows.py) |
| 主负载计时、profiling、延迟注入与原生对照 | [scripts/run_matrix.py](scripts/run_matrix.py) |
| pinned memory、状态恢复与矩阵形状诊断 | [scripts/run_auxiliary.py](scripts/run_auxiliary.py) |
| 实际通信量 | [cases/q06_comm_volume.py](cases/q06_comm_volume.py) |
| 参数表示 | [cases/q09_param_representation.py](cases/q09_param_representation.py) |
| 时间线关联 | [scripts/parse_trace_indexed.py](scripts/parse_trace_indexed.py) |
| GEMM 分析 | [scripts/analyze_gemm_trace.py](scripts/analyze_gemm_trace.py)、[scripts/summarize_gemm_shapes.py](scripts/summarize_gemm_shapes.py) |

在仓库根目录、安装依赖后查看实际参数：

```bash
python experiments/fsdp_interview/scripts/run_matrix.py --help
python experiments/fsdp_interview/scripts/run_auxiliary.py --help
```

脚本用于 Linux GPU 实验；执行前应检查源码中的当前运行目录、Python、Nsight Systems 与输入路径。输出保存在运行目录，本地保留。

## 分析方法

通信角色依据 NVTX、同线程 CUDA API 和对应 GPU kernel 关联，跨 rank 使用同组成员与 collective sequence 对齐。Python stream 指针与 Nsight stream 编号不属于同一命名空间，不能直接按整数对应。

时间线区间交集表示同时驻留，完整训练收益由关闭 profiler 的受控计时确定。所用 V100/Nsight Systems 组合未提供 SM Active/DRAM counter，单凭 kernel 时长不能确证 SM/HBM 争用。参数表示与跨卡数恢复使用小模型，原生调度对照使用共同的 8 层 FP32 FSDP-only 模型，均与完整 Qwen 主负载分开分析。
