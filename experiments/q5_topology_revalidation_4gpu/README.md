# 通信重叠调度消融

本实验固定 ShardTrain Hybrid F2T2/SP，比较三种额外通信等待策略。实测物理映射的两组 FSDP 链路均为 NV2，TP 两组为 NV1/PHB；实际运行前需重新检查链路条件。

| 策略 | 等待关系 |
|---|---|
| C0_guarded | 保留真实数据依赖，不额外串行无关通信 |
| C1 | 每项训练通信等待此前最近提交的训练通信完成 |
| C2 | FSDP 与 SP/Norm 及复制梯度通信互相隔离，FSDP 参数 AG 与梯度 RS 之间仍允许并行 |

三种策略采用相同的通信顺序检查和元数据处理，使用 CUDA Event 在实际提交流上添加等待，不改变通信张量和计算工作量。

## 工作量与控制方法

完整随机初始化的 28 层 Qwen2.5-1.5B，S512，每个 FSDP 副本 microbatch 4，梯度累积 2，全局每步 16 条序列、8176 个有效标签。前后向参数预取与异步梯度 RS 均开启。主要线性 GEMM 为 FP16，math-attention BMM、master 参数、参数 AG、梯度 RS 和 Adam 状态为 FP32。

每种策略 3 个独立进程，每次 10 步预热、30 步测量，共 270 个测量 step，按轮换顺序运行。正式计时关闭 profiler 和逐操作记录，保留共同的运行时检查；完整 step 包含前反向、梯度处理、AMP、AdamW 与设备完成，取四个 rank 最大值。

C0_guarded 与 C1 分别使用独立进程进行 10 步预热、1 步采集，记录 CUDA、必要 NVTX 和显存端点状态。时间线用于分析通信角色、跨 rank 到达偏斜与 kernel 驻留交集，正式吞吐采用关闭 profiler 的计时。C2 的策略定义允许 AG/RS 并发，但该定义本身不证明实际重叠；kernel 区间交集也不等于实际链路传输同时进行。

## 源码与接口

| 用途 | 源码 |
|---|---|
| 运行阶段与参数 | [cli.py](cli.py) |
| 三种策略及配置 | [contracts.py](contracts.py) |
| 数值与通信检查 | [gate_worker.py](gate_worker.py) |
| 完整模型训练 | [train.py](train.py) |
| 时间线与跨 rank 分析 | [mechanism.py](mechanism.py) |
| 正式计时统计 | [analysis.py](analysis.py) |

在仓库根目录、安装依赖后查看实际参数：

```bash
python -m experiments.q5_topology_revalidation_4gpu.cli --help
```

接口包含阶段计划、参考检查、容量、正式计时、profiling 和分析。执行前需配置当前设备映射、输入、工具路径与运行目录。输出保存在运行目录，本地保留。

结果见[通信重叠与调度收益报告](../../reports/q5_new_topology_m0_revalidation.md)。
