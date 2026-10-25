# 四卡 FSDP-only 与 Hybrid 对照

## 背景

这组实验要区分两个问题：固定引擎时，F4T1省去逐层TP/SP通信，能否抵消更大的FSDP通信量；固定布局时，两条真实引擎在相同工作量下的完整step是否存在稳定差异。Hybrid改变局部矩阵形状和通信位置，因此不能仅凭传输字节更少就预判更快，也不预设自研一定胜过原生。

基础状态显存则可以独立计算。设全局唯一参数为P，其中在TP维复制的embedding与Norm共R个参数；parameter、gradient、Adam m/v均为FP32，则每rank预期为：

```text
State_F4T1   = 16 * P/4
State_Hybrid = 16 * ((P-R)/4 + R/2)
Delta_state  = 4R = 933,844,992 B = 0.869711 GiB
```

实验据此核对状态分配，再测量四组完整step。若F4T1传输量更大但步时更短，支持“通信位置、频率和计算布局共同决定本机策略”的解释；allocated峰值还包含激活与临时buffer，不要求其差值恰好等于基础状态差。

## 实验

**配置与检查**

同机 4×V100-SXM2 32GB，完整随机初始化的 28 层 Qwen2.5-1.5B、1,543,714,304 个唯一参数，S512、A2、每步 16 条序列/8176 个有效标签。Hybrid 为 F2T2/SP、每 F 副本 MB4；F4T1 关闭 SP、每副本 MB2。每个 microbatch 均同步梯度。Linear/MLP/head 主要 GEMM 为 FP16，math-attention BMM、参数 AG（AllGather）、梯度 RS（ReduceScatter）及 Adam 状态为 FP32。

双方使用同一 WikiText token 缓存和实际重构一致的全局初始参数；AdamW 为 lr1e−5、betas=(0.9, 0.999)、eps=1e−8、weight_decay=0.01、foreach=False。跨策略每步 block 集合相同，但微步子集、矩阵形状和浮点累积次序不同，因此比较的是整体策略。环境为 PyTorch 2.9.1+cu126、NCCL 2.27.5；M0 的 Hybrid FSDP 两组为 NV2，TP 两组为 NV1/PHB。

原生 FSDP2 使用 `precision_policy="matched"`，即 `MixedPrecisionPolicy(param_dtype=None, reduce_dtype=torch.float32)`：常驻参数、参数聚合和梯度归约均为 FP32，Linear/MLP/head 通过 autocast 使用 FP16。28 个 decoder 层分别调用 `fully_shard(..., reshard_after_forward=True)`；共享 embedding/head 的根模块与 final norm 分别成组，均设置 `reshard_after_forward=False`，保留前向已聚合的参数供反向使用。显式前向和反向预取均开启，每个 microbatch 都执行梯度归约。F4T1 使用四卡 FSDP mesh 并跳过 TP/SP；Hybrid 先在两卡 TP mesh 上设置列/行切分和 Norm 的序列并行，再在两卡 FSDP mesh 上切分参数。具体实现见[原生配置代码](../src/sharded_runtime/training/qwen2_native_baseline.py)及[本轮配置](../experiments/fsdp4_hybrid_2x2/configs/N-H.yaml)。

8组小规模参考模型检查与四组完整模型容量检查通过；正式训练中所有rank的loss和梯度保持有限，没有触发溢出跳步，实际338个Adam参数状态均更新40次。小规模FP16检查覆盖相同起点更新、溢出跳步和恢复；完整模型检查覆盖容量、数值有限性与实际优化器更新。

**性能与显存**

四组各 3 次独立进程、10 warmup+30 measured；共 360 个测量 step，profiler 和逐操作 recorder 关闭。计时包括 A2 前反向、梯度收尾、AMP、AdamW 和设备完成，取四 rank 完整 step 最大值。

| 引擎 / 策略 | Mean ms | P50 / P95 ms | run mean SD ms | labels/s |
|---|---:|---:|---:|---:|
| ShardTrain F4T1 | 1249.556 | 1249.529 / 1252.739 | .062 | 6543.13 |
| Native FSDP2 F4T1 | 1336.181 | 1331.298 / 1364.469 | 8.441 | 6118.93 |
| ShardTrain Hybrid | 1522.428 | 1518.862 / 1525.954 | 5.374 | 5370.37 |
| Native FSDP2 Hybrid | 1981.821 | 1966.137 / 2160.042 | 47.707 | 4125.50 |

SD 是三个独立进程均值的样本标准差；P50/P95 是每组 90 个相关 step 的描述分位数。自研 F4T1 比自研 Hybrid 吞吐高 21.838%，原生相应高 48.320%；跨引擎比较只用同布局作为分母。吞吐提升为 `baseline_ms / candidate_ms − 1`，不是耗时降低百分比。

| 引擎 / 策略 | peak allocated GiB | peak reserved GiB |
|---|---:|---:|
| ShardTrain F4T1 | 12.77312 | 21.51758 |
| Native FSDP2 F4T1 | 15.86071 | 18.60547 |
| ShardTrain Hybrid | 13.80869 | 23.17578 |
| Native FSDP2 Hybrid | 15.42685 | 19.71094 |

两指标均为测量边界 reset 后、四 rank×三 run 最大值。自研 allocated 较低、reserved 较高。

**通信计量**

同布局两引擎实际执行的本地 GEMM，其 `operator/M/N/K/batch/dtype` 及调用次数相同：每 rank 每 step 1182 次 FP16 矩阵调用、336 次 FP32 attention BMM。同布局 FSDP AG/RS 总字节也相同，AG 都为 116 次；自研/原生 RS 为 116/60 次，差别来自梯度分组方式。

每层有两个 RMSNorm。自研将两者的梯度合为一个 Norm 分组，与该层其余参数分别执行 RS；原生把该层所有参数交给一次 `fully_shard(layer)`，因此这两个 Norm 的梯度随整层梯度一起 RS，不产生额外的层内 Norm RS。独立包装的 final norm 仍有自己的 RS。按每步两个 microbatch 计算，自研为 `2×(28×2+1+1)=116` 次，原生为 `2×(28+1+1)=60` 次。

Hybrid 还要合并 TP 两卡对复制参数的梯度贡献：原生先完成 TP AllReduce，再由 FSDP RS 生成本地梯度分片；自研先做 FSDP RS，再对分片做 TP AllReduce。实际每步均有 114 次 Norm TP AllReduce，即 `57 个 Norm×2 个 microbatch`，原生每次 6144 B，自研每次 3072 B。共享 embedding/head 每步另有两次 TP AllReduce，原生每次 933,494,784 B，自研每次 466,747,392 B。上述次数和大小按实际训练中使用的张量形状统计。

**基础状态**

本模型的 embedding 与 head 共享一份 `151936×1536` 权重，共 233,373,696 个参数；28 层各两个 Norm 加上 final norm，共 `57×1536=87,552` 个参数。Hybrid 中这 233,461,248 个参数在 TP 维复制，只沿 FSDP 维二分；其余 1,310,253,056 个参数沿 TP 和 FSDP 两维四分。F4T1 则把全部参数沿 FSDP 维四分。

| 参数类别 | 全局唯一参数数 | F4T1 每 rank 参数数 | Hybrid 每 rank 参数数 |
|---|---:|---:|---:|
| 共享 embedding/head | 233,373,696 | 58,343,424 | 116,686,848 |
| 全部 Norm | 87,552 | 21,888 | 43,776 |
| 其余参数 | 1,310,253,056 | 327,563,264 | 327,563,264 |
| 合计 | 1,543,714,304 | 385,928,576 | 444,293,888 |

参数、梯度、Adam 一阶矩 m 和二阶矩 v 均为 FP32，四份合计每个本地参数 16 B。因此基础状态为 Hybrid 7,108,702,208 B / 6.620495 GiB/rank、F4T1 6,174,857,216 B / 5.750784 GiB/rank，差额为 933,844,992 B / 0.869711 GiB/rank。两引擎的实际状态大小一致。前表的 allocated 峰值还包括激活与临时缓冲，因此与基础状态大小不同。

## 结论

- **本机S512、global batch 16下，两种引擎都以F4T1更快。** 自研F4T1/Hybrid为1249.556/1522.428ms，原生为1336.181/1981.821ms；F4T1的吞吐分别高21.838%和48.320%。
- **同布局下，自研吞吐比匹配的官方FSDP2高6.933%（F4T1）和30.175%（Hybrid）。** 双方实际本地GEMM工作量与FSDP总payload相同，执行差异集中在参数管理、SP聚合、梯度归约和调度路径。
- **F4T1在更大的通信总字节下仍更快。** F4T1的Ring发送模型为26.387GB/rank/step，高于Hybrid的自研12.777GB和原生13.359GB；省去逐层SP通信与复制参数TP归约、改变局部矩阵形状，对完整step具有实际影响。
- **Hybrid为TP复制参数付出额外状态显存。** embedding/head与Norm只沿两路FSDP分片，使每rank参数、梯度与Adam状态合计比F4T1增加0.869711GiB；这项差额与两条引擎的实际状态统计一致。

[逐run统计](../docs/release/p1_p2_runs.csv) · [逐step统计](../docs/release/p1_p2_measured_steps.csv) · [运行与复现](../docs/reproduction_4gpu.md)
