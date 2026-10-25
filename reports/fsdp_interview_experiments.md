# 四卡 FSDP 专项实验方案与结果

本组实验以完整训练步检验异步内存与通信的收益，独立核对分片通信次数和数据量，并比较官方 FSDP1/FSDP2 的参数表示与调度。实验1—10的方案、共同环境和结果总览集中于本页，各实验报告按研究问题、理论或机制预期、控制实验、观测与结论组织。通信量、矩阵调用次数和FLOPs可由结构独立推导；调度、分配器和实现之间的速度比较使用可检验假设，不预设加速百分比。

## 环境、拓扑与训练负载

| 项目 | 配置 |
|---|---|
| GPU | 4 × V100 级 Tesla PG503-216，SM70，32GB |
| 软件 | Python 3.12.7，PyTorch 2.9.1+cu126，NCCL 2.27.5，Nsight Systems 2024.6.1.90 |
| 主模型 | 完整随机初始化 Qwen2.5-1.5B，28层 |
| 并行布局 | FSDP degree 2，TP degree 2，SP 开启 |
| 进程组 | TP/SP：`(0,1)/(2,3)`；FSDP：`(0,2)/(1,3)` |
| FSDP 链路 | 两组分别跨 NV1、PHB，链路不对称 |
| 训练输入 | 固定 WikiText-103，连续打包为 1024 × 512 token blocks |
| 每次更新 | S512，microbatch 4，accumulation 2；16条不同序列、8192个输入位置、8176个有效 next-token labels |
| 精度 | FP16 linear；FP32 master 参数、参数 AG、梯度 RS、AdamW 状态与 math attention BMM |

TP/SP rank 不重复计入样本数。输入预先缓存到 GPU，数据读取和准备不计入训练步时。ShardTrain 管理普通 Tensor 分片和自定义 autograd；原生对照使用官方 `fully_shard`、DTensor 与 TP/SP API。自研 `forward_prefetch`、`backward_prefetch`、`async_grad_rs` 均开启，原生调度由 FSDP2 管理。

选择 S512 / microbatch 4 / accumulation 2，是为了在完整模型上形成持续计算与通信负载，同时保留重复实验的显存余量。两条路径先各执行10步预热、10步测量的容量预检，物理空闲端点均超过10GiB，因此没有继续逼近显存上限。显存分别统计 allocated、reserved 与设备物理空闲。以下结果属于上述不对称拓扑，不与后续其他拓扑的测量混合。

## 控制变量与计量方法

| 实验 | 主要改变量 | 对照与测量 |
|---|---|---|
| [1：Pinned transfer](fsdp_interview_sections/01_pinned_transfer.md) | pinned/pageable、阻塞/异步复制、独立 stream、缓冲池 | 单卡4/16/64/128MiB，保持 H2D 和 GEMM 工作量相同 |
| [2：Pinned allocator](fsdp_interview_sections/02_pinned_allocator.md) | HostAlloc/HostRegister、缓存命中 | 核实实际 CUDA API，分开测量 CPU copy、驱动注册及总 allocation 成本 |
| [3：通信并发](fsdp_interview_sections/03_communication_overlap.md) | 只增加 CUDA Event 等待依赖 | 数学通信、计算、输入及存储生命周期不变，比较完整 step |
| [4：通信数据量](fsdp_interview_sections/04_communication_volume.md) | 独立布局预测与实际执行记录 | 核对通信次数、字节数及参数首次使用 |
| [5：参数表示](fsdp_interview_sections/05_parameter_representation.md) | 官方 FSDP1 两种原参数模式、FSDP2 | 独立 dense reference、冻结参数、optimizer 分组及模型 checkpoint 4↔2 恢复 |
| [6：原生调度](fsdp_interview_sections/06_native_fsdp_scheduling.md) | 官方 FSDP1/FSDP2 | 同一中型 FP32 FSDP-only 模型，匹配参数保留和梯度同步 |
| [7：Rank skew](fsdp_interview_sections/07_rank_skew.md) | 指定 rank 的 CPU 等待或 GPU 工作 | 固定目标 collective，比较 API 提交、GPU 开始与结束时间 |
| [8：GEMM profile](fsdp_interview_sections/08_gemm_profile.md) | 观测自研路径的实际本地矩阵操作 | 关联 shape、dtype、stride、源码阶段和 GPU kernel |
| [9：并发与串行计算对照](fsdp_interview_sections/09_gemm_concurrency_comparison.md) | 自研并发/串行调度 | 比较逐调用矩阵工作量与 kernel 类型、次数 |
| [10：自研与原生计算对照](fsdp_interview_sections/10_gemm_native_comparison.md) | 自研并发、自研串行、原生 FSDP2 | 只统计实际本地矩阵操作，避免 DTensor 全局元信息重复计数 |

正式性能每组使用三个独立进程，各10步预热、30步测量，轮换执行顺序。完整 step 采用四 rank 耗时最大值，关闭 profiler 和逐操作记录。实验1、2使用单卡复制微基准，实验5使用小型 FP32 模型，实验6使用中型 FP32 模型，其余实验采用表中的完整模型负载。

正确性检查与性能计时分开：小模型对照独立 Hugging Face dense reference，检查 loss、gradient、parameter update 与 AMP overflow 处理；完整模型检查训练更新的有限值和 overflow。调度实验新增的跨通信域依赖只指向此前已提交操作，并核对各 rank 的共同提交顺序。

Profiling 使用独立进程。NVTX 范围、同线程 CUDA API 和同进程 correlation ID 用于关联通信角色与 GPU kernel；跨 rank 时间比较使用同一份 Nsight Systems 采集。NCCL kernel 驻留可能包含 peer 等待，不能直接等同于链路传输时间；overlap 是否有收益由无 profiler 的完整 step 对照判断。显存 snapshot 用于 GPU allocator 分析，不用于推算主机 pinned memory 驻留量。

## 完整模型的正式性能

以下每组包含三个独立进程、共90个测量 step：

| 路径或策略 | Mean step（ms） | 有效 labels/s |
|---|---:|---:|
| ShardTrain C0_fast | 2717.32 | 3008.84 |
| Native FSDP2，匹配配置 | 3272.89 | 2498.10 |
| C0_guarded：相同记录与检查，无新增等待 | 2719.59 | 3006.34 |
| C1：所有训练通信串行 | 3356.50 | 2435.87 |
| C2：FSDP 与非 FSDP 通信隔离 | 3253.50 | 2512.99 |
| C3：进一步隔离 FSDP 与主计算 stream | 3469.64 | 2356.44 |

C0_guarded 比 C1、C2、C3 都快，时间线确认了相应调度变化。完整 step 差值体现这组调度改动的整体影响，包括并发机会、等待和 API 成本。Native 结果对应本页固定配置。

## 实验1—10结果总览

| 实验 | 理论或机制预期 | 观察与判断 |
|---|---|---|
| 1 | 预先pinned可减少主机阻塞；独立stream只有覆盖了复制时间才减少总完成时间 | 128MiB CPU调用P50从12.355降至0.083ms；相同H2D＋GEMM工作量完成时间下降14.47%，支持这组条件下的重叠收益 |
| 2 | 缓存命中省去注册/分配，但pin_memory仍需CPU copy；首次分配成本不能只按单个CUDA API估算 | 实际API确认HostAlloc/HostRegister与缓存复用，缓存命中仍保留CPU copy，符合预期 |
| 3 | 独立通信若暴露在关键路径，保留并发应优于新增串行等待；资源竞争也可能抵消收益 | 并发完整step为2719.59ms，优于三种串行或隔离策略，时间线与性能共同支持假设 |
| 4 | retain省去部分反向AG，分桶增加RS次数而不增加总梯度payload | 预测与实测AG/RS均116次，细化FSDP Ring发送模型9,729,552,384 B；2400项unit及2400项首次使用检查吻合 |
| 5 | 参数表示可影响冻结和分组能力，但受支持配置应保持逻辑模型及恢复结果一致 | 八个可训练组合通过dense reference，三种官方路径的模型checkpoint 4↔2恢复通过 |
| 6 | 同工作量下数值应匹配，时间和显存取决于具体调度，不能由版本名预判赢家 | FSDP1/FSDP2平均105.655/151.059ms、reserved峰值1.310547/1.216797GiB，速度与显存各有取舍 |
| 7 | 队列余量可隐藏小提交延迟；超过余量后，晚到peer应增加早到rank的驻留 | 小skew被隐藏，大skew增加等待，与跨rank到达及驻留关系一致 |
| 8 | 层数、分组和反向结构预计四rank有4728次线性矩阵调用与1344次BMM | 实测共6072次，次数、dtype与结构推导一致 |
| 9 | 只改等待策略不应改变数学工作量 | 并发与串行逐调用shape、dtype、stride及kernel类型/次数相同，支持同工作量对照 |
| 10 | 同模型、布局和精度下，两引擎实际本地矩阵工作量应一致 | 三条路径均匹配独立结构基准，每rank为19508546764800 FLOPs |

各实验的完整条件、结果与适用范围见上表链接。[实验实现与复现入口](../experiments/fsdp_interview/README.md)。
