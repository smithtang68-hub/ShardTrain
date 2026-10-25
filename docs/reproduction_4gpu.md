# 安装、四卡运行与 profiling

## 环境与安装

实测环境为 Linux、4×V100-SXM2 32GB、PyTorch 2.9.1、Transformers 4.57.3。V100 使用 FP16 线性计算；math attention 与参数/梯度通信为 FP32。GPU 训练使用 NCCL。

```bash
python -m pip install -e ".[dev,qwen]"
nvidia-smi
nvidia-smi topo -m
```

运行前核对驱动与 CUDA/PyTorch 兼容性、四张 GPU 的可见顺序、实际互连及空闲显存。不同拓扑可能改变 FSDP/TP 的最佳 rank 映射。[拓扑研究](../reports/topology_mapping_study.md)说明了实测影响。

## 输入与配置

正式四卡策略对照使用完整 28 层 Qwen2.5-1.5B、S512、A2、global batch 16。Hybrid 为 F2T2/SP、local microbatch 4；FSDP-only 为 F4T1、local microbatch 2。参数随机初始化，输入为预先 packed 的真实 token。输入数据需自行准备；使用不同数据可重跑该配置，但不会逐步复现原有数值轨迹。

[四组实验配置](../experiments/fsdp4_hybrid_2x2/configs/)分别给出自研/原生及两种策略。模型配置使用仓库中的 YAML；读取外部 Hugging Face checkpoint 时仍支持其标准配置格式。

GPU 主机应先运行项目的正确性检查和短容量试跑，再做正式计时。容量选择留出通信临时 buffer 和 allocator cache 余量，方法见[FSDP专项总述](../reports/fsdp_interview_experiments.md)。

## 训练入口

以下命令使用同一 Hybrid 工作量和 GPU-resident synthetic tokens，提供两条引擎的直接运行入口；输入形式与报告中的真实 packed tokens 分开。模型和训练设置以配置文件为准。

```bash
torchrun --standalone --nproc-per-node=4 -m sharded_runtime.training.qwen2_benchmark \
  --config experiments/fsdp4_hybrid_2x2/configs/S-H.yaml \
  --warmup 10 --profile-mode off --runtime-events off \
  debug.output_dir=runs/shardtrain_hybrid

torchrun --standalone --nproc-per-node=4 scripts/benchmark_qwen2_native.py \
  --config experiments/fsdp4_hybrid_2x2/configs/N-H.yaml \
  --warmup 10 --profile-mode off \
  debug.output_dir=runs/native_hybrid
```

两个入口的完整选项可通过 `--help` 查看。真实 packed-token 的实验入口为 [train_workload](../experiments/fsdp_interview/scripts/train_workload.py) 与[四组策略 worker](../experiments/fsdp4_hybrid_2x2/worker.py)，输入由 [prepare_data](../experiments/fsdp_interview/scripts/prepare_data.py) 准备。原生与自研应固定相同输入、精度、全局 batch、optimizer、预热和测量步数。正式对照需各启 3 次独立进程，不能将单次最快结果当作结论。

## profiling

将 profiling 与正式计时分开运行。PyTorch profiler 使用 `--profile-mode torch`；Nsight Systems 使用 `--profile-mode nsight` 配合 `nsys profile`，保留 CUDA 和 NVTX。只捕获预热后的短窗口，避免完整训练 trace 带来大量 I/O 与观测开销。

先记录实际拓扑与每个 rank 对应的 GPU，再按 NVTX/collective 元数据关联 API 与 kernel。stream 编号只对该进程、该 CUDA context 有效。内存分析同时观察 allocated、reserved、峰值与 snapshot；NCCL kernel 驻留时间不能直接当作链路传输时间。

## 结果阅读

[测量方法](benchmark_methodology.md)说明完整 step、rank-max、吞吐和显存口径。[逐 run CSV](release/p1_p2_runs.csv)与[逐 step CSV](release/p1_p2_measured_steps.csv)保留公开性能统计。原始 trace、snapshot、输入数据与本地运行输出不随仓库发布。
