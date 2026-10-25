# 性能测量方法

## 模型与工作量

完整随机初始化 Qwen2.5-1.5B，28 层、1,543,714,304 个 unique 参数。正式四卡策略对照采用 packed token 输入，S512、梯度累积 A2、global batch 16，每个 optimizer step 包含 8,192 个 input positions 和 8,176 个有效 next-token labels。

F4T1 的 local microbatch 为 2；F2T2/SP 的 local microbatch 为 4。两种策略每步处理相同数量的序列，但局部 GEMM 形状、microstep 子集和 collective 随策略改变。该对照评估整套并行策略。同布局的自研与原生使用相同输入划分。

Linear/MLP/head 使用 FP16；math-SDPA BMM、master parameters、参数 AG、梯度 RS 和 Adam state 使用 FP32。AdamW 为 lr=1e-5、foreach=False。每个 microstep 都同步 FSDP 梯度，两条路径使用相同设置。

## 计时与统计

正式计时关闭 profiler，输入预缓存于 GPU。计时包含前向、反向、梯度收尾、AMP、AdamW 和设备完成；每个 step 取四个 rank elapsed 的最大值。数据准备和 H2D 不在该窗口内。

每组 3 次独立进程运行，每次 10 warmup + 30 measured steps。策略对照共 12 次运行，调度对照共 9 次运行，合计 630 个正式测量 step。warmup 和 profiling 样本不进入正式均值，所有测量样本均保留。

```text
group_mean_ms = mean(all measured rank-max step times in the group)
effective_labels/s = 8176 * 1000 / group_mean_ms
input_positions/s = 8192 * 1000 / group_mean_ms
throughput_gain = baseline_mean_ms / candidate_mean_ms - 1
time_reduction = 1 - candidate_mean_ms / baseline_mean_ms
```

每次运行的测量步数相同，因此 pooled mean 等于三个 run mean 的等权均值。P50/P95 使用排序后 `(n-1)q` 的线性插值。run-mean SD 使用三个独立 run mean 的样本标准差；同一次运行内的 30 个 step 不视为 30 次独立实验。

公开的[逐 step CSV](release/p1_p2_measured_steps.csv)和[逐 run CSV](release/p1_p2_runs.csv)保留实际时间与显存数值。两组实验分别汇总，不合并为一个基线。

## 原生对照与调度消融

原生使用官方 `torch.distributed.fsdp.fully_shard`、DTensor 与 TP/SP 接口；自研使用普通 Tensor、自定义 Autograd Bridge 和参数生命周期。两条路径核对全局初态、本地 GEMM shape/dtype/count 和通信 payload，但分桶与浮点求值顺序允许不同。

同策略下每步参数 AG 均为 116 次；自研/原生梯度 RS 为 116/60 次，差别来自 main/norm 分桶与整层分桶，总 FSDP payload 相同。该结果不代表原生所有配置的性能上限。

调度消融固定 Hybrid 布局与输入，比较保留通信重叠、全部训练通信串行、FSDP与非FSDP通信分类隔离。等待通过 GPU Event 实现；增加等待本身的提交成本也计入结果。正式统计与独立短 trace 分开，trace 用于观察操作所属 stream、排队和 NCCL 驻留交集。

## 显存与 timeline

allocated/reserved 报告各 rank、各 run 中的峰值，GiB=2^30 bytes。reserved 包含 allocator cache，NCCL/CUDA 直接分配不一定包含在该统计中。NVML 采样和端点 snapshot 用于补充观察，不能由两个端点重建完整分配时间线。

NCCL kernel 驻留可包含等待 peer。驻留交集不直接等于 wire 传输交集，也不等于同层 GEMM 被完整掩盖。分析需联合 rank、CPU 提交、实际 stream 与完整 step。

[策略对照](../reports/fsdp4_vs_hybrid_2x2_native_4gpu.md) · [调度对照](../reports/q5_new_topology_m0_revalidation.md) · [FSDP专项](../reports/fsdp_interview_experiments.md)
