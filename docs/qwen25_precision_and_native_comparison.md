# Qwen精度与原生FSDP2对照

自研路径使用普通 Tensor、独立参数分片、Autograd Bridge 和通信生命周期；官方 FSDP2 的 fully_shard、DTensor 仅存在于独立原生基线。两者共用模型工作量和训练设置，底层参数管理各自实现。

V100 实测使用 FP16 Linear/MLP/head，FP32 math attention、master parameters、参数 AG、梯度 RS 和 Adam state。相同 dtype 配置不代表每个内部算子都以同一精度执行，因此使用实际 GEMM 的 shape/dtype/count 核对计算工作量。

四卡正式比较包含两条引擎与 F4T1/F2T2-SP 两种策略。同策略使用相同输入划分、初态、optimizer 和每 microstep 梯度同步。原生层级分桶与自研 main/norm 分桶的调用次数不同，但核对总通信 payload。

基线没有搜索全部原生配置，也不将自研改写为 FSDP2。[策略报告](../reports/fsdp4_vs_hybrid_2x2_native_4gpu.md)列出实际配置和性能，[计算量实验](../reports/fsdp_interview_sections/10_gemm_native_comparison.md)核对真实本地矩阵工作量。
