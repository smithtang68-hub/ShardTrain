# 四卡运行与物理拓扑对照

本实验固定逻辑 mesh `[[0,1],[2,3]]`，比较不同 GPU 物理映射下的正确性、容量、通信和完整训练性能。M0 使用物理顺序 ABCD，M1 交换最后两张卡，使用 ABDC；逻辑 FSDP 组为 02/13，TP/SP 组为 01/23。

实测 M0 的两组 FSDP 链路均为 NV2，M1 为 PHB/NV1。实际运行前需要重新检查设备与拓扑。

## 工作量与控制方法

- 完整随机初始化 Qwen2.5-1.5B，28 层，F2T2/SP，S512，每个 FSDP 副本 microbatch 4，梯度累积 2。
- 每个 optimizer step 处理 16 条序列、8192 个输入位置、8176 个有效 next-token 标签；TP/SP 不重复计数。
- 主要线性 GEMM 使用 FP16，math-attention BMM、master 参数、参数 AG、梯度 RS 和 Adam 状态使用 FP32。
- 自研与官方 FSDP2 分别在 M0/M1 上运行；前后向参数预取、异步梯度 RS 均开启。
- 每组 3 个独立进程，各 10 步预热、30 步测量。关闭 profiler 和逐操作记录，完整训练 step 取四个 rank 的最大完成时间。
- 独立 nccl-tests 使用实际消息大小与有序 GPU 配对；profiling 单独运行，采集 CUDA、必要 NVTX 与显存端点状态，不混入正式吞吐统计。

## 源码与接口

| 用途 | 源码 |
|---|---|
| 运行阶段与参数 | [cli.py](cli.py) |
| 物理映射 | [placement.py](placement.py) |
| 数值与生命周期检查 | [correctness.py](correctness.py)、[mapping_worker.py](mapping_worker.py)、[amp_worker.py](amp_worker.py) |
| 完整模型训练 | [train.py](train.py) |
| 独立通信对照 | [reference.py](reference.py)、[build_reference.py](build_reference.py) |
| 统计与绘图 | [analyze.py](analyze.py) |

在仓库根目录、安装依赖后查看实际参数：

```bash
python -m experiments.finalization_4gpu.cli --help
```

接口包含环境检查、正确性、容量、正式计时、profiling、独立通信对照和结果分析。执行 GPU 阶段前需配置当前设备映射、输入、工具路径和运行目录。输出保存在运行目录，本地保留。

结果见[四卡运行报告](../../reports/shardtrain_4gpu_finalization_report.md)和[拓扑映射研究](../../reports/topology_mapping_study.md)。
