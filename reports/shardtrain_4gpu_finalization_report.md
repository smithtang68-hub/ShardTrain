# 四卡运行验证与实验总结

**本实验完成四卡运行验证、性能对照和通信分析。** 本页汇总拓扑映射实验的配置与检查结果；后续两引擎、两策略的结果见[并行策略对照](fsdp4_vs_hybrid_2x2_native_4gpu.md)。

## 固定负载与验收范围

4×V100-SXM2 32GB、PyTorch2.9.1+cu126、NCCL2.27.5；完整随机 28 层 Qwen2.5-1.5B，F2T2/SP，S512/MB4每F副本/A2，每 step 16 序列/8176 有效标签。FP16 主要 GEMM、FP32 math-attention/参数聚合/梯度归约，AdamW lr1e−5、foreach=False。自研开启前向参数预取、反向参数预取和异步梯度RS，按数据依赖在CUDA流之间传递完成事件，原生为官方 `fully_shard`+DTensor 的 matched 路径。

| 检查项目 | 完成情况 | 检查内容 |
|---|---|---|
| 小规模模型正确性检查 | 12组配置、48组rank检查通过 | FP32、AMP same-start/skip/recovery、映射与生命周期各自范围 |
| 完整模型容量 | 自研/原生×M0/M1 四组通过 | 完整28层初始化、两微步梯度累积、训练输入与参数读取顺序 |
| 正式性能 | 四组各3进程、10+30步，共360测量step | profiler-off、max-rank 完整 step，详见[放置研究](topology_mapping_study.md) |
| 通信参考 | 21条件×3进程，63次 nccl-tests，wrong=0 | 按实际消息大小与GPU配对进行独立通信测试 |

M0的两组FSDP连接均为NV2。交换rank2/3后，FSDP连接变差，自研和原生step都明显变慢。精确时长、SD/P50/P95、allocated/reserved和链路测试结果统一放在[物理放置研究](topology_mapping_study.md)。

## 参数状态和异步执行的实证

每rank持久参数shard为1,777,175,552B；参数、梯度和Adam m/v合计6.620495GiB。embedding与Norm在TP轴复制后沿两路FSDP切分，其状态占用高于四路FSDP布局，具体分配见[策略对照](fsdp4_vs_hybrid_2x2_native_4gpu.md)。临时完整参数、激活、通信缓冲和分配器缓存另计。配对显存快照中，正式测量窗口首尾的活跃显存无净增长。

两种物理映射的时间线中，所分析的3656次集合通信均关联到实际kernel，其中3632项训练通信的顺序和数据量匹配。前向参数AG与SP AG重叠的边界分别为172/224、224/224，反向三路通信重叠均为200/200。结合两种映射的步时可见，链路速度与跨rank等待同样影响整体性能，优化需要同时考虑拓扑和通信重叠。

## 实验设置与复现

训练实现、环境与执行方法见[复现手册](../docs/reproduction_4gpu.md)，参数与通信机制见[运行时架构](../docs/architecture.md)。
