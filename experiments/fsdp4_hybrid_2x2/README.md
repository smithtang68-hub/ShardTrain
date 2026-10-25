# 自研与原生 FSDP2 的两引擎、两策略对照

本实验比较 ShardTrain 与官方 PyTorch FSDP2，在 FSDP-only 和 FSDP × TP/SP 两种布局下的完整训练性能。

| 分组 | 引擎 | 布局 | 每副本 microbatch / 梯度累积 |
|---|---|---|---|
| S-H | ShardTrain | F2T2/SP | 4 / 2 |
| N-H | 官方 `fully_shard` + DTensor TP/SP | F2T2/SP | 4 / 2 |
| S-F | ShardTrain | F4T1，无 SP | 2 / 2 |
| N-F | 官方 `fully_shard`，不启用 TP/SP | F4T1，无 SP | 2 / 2 |

## 工作量与控制方法

完整随机初始化的 28 层 Qwen2.5-1.5B，S512，全局每步 16 条序列、8192 个输入位置、8176 个有效标签。双方使用相同的 WikiText token 输入和全局初始参数；主要线性 GEMM 为 FP16，math-attention BMM、master、参数 AG、梯度 RS 和 Adam 状态为 FP32。AdamW learning rate 为 1e-5，`foreach=False`，每个 microbatch 都同步 FSDP 梯度。

每组运行 3 个独立进程，各 10 步预热、30 步测量，共 360 个测量 step。输入预缓存与初始化检查在计时外完成。正式计时关闭 profiler 和逐操作记录，包含前反向、梯度处理、AMP、AdamW 与设备完成，取四个 rank 的完整 step 最大值。

同布局下，自研与原生匹配微步输入及局部矩阵工作量。跨布局保持每步全局样本集合相同，但 microbatch 分组、矩阵形状和浮点累积次序不同，因此比较的是整体并行策略。小模型用于独立数值对照；完整模型检查容量、数值有限性和实际更新，不能等同于完整模型逐元素参考验证。

## 源码与接口

| 用途 | 源码 |
|---|---|
| 四组配置 | [configs](configs/) |
| 运行阶段与参数 | [cli.py](cli.py) |
| 四组训练 | [worker.py](worker.py) |
| 数值检查 | [gate_worker.py](gate_worker.py) |
| 统计分析 | [analyze.py](analyze.py) |
| 官方原生模型构建 | [qwen2_native_baseline.py](../../src/sharded_runtime/training/qwen2_native_baseline.py) |

在仓库根目录、安装依赖后查看实际参数：

```bash
python -m experiments.fsdp4_hybrid_2x2.cli --help
```

接口包含配置准备、正确性、容量、正式计时和分析。执行前需配置当前设备映射、输入与运行目录。输出保存在运行目录，本地保留。

结果见[两引擎、两策略对照报告](../../reports/fsdp4_vs_hybrid_2x2_native_4gpu.md)。
