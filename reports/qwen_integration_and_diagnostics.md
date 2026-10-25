# Qwen2.5 集成与运行诊断

## 背景

本报告说明 Qwen2.5-1.5B 的模型适配、数值对照、通信依赖与完整训练性能。整体结论见[主技术报告](shardtrain_final_project_report.md)。

**通信依赖**

下一 unit 的参数 AG 只约束目标 unit 的 copy-out。当前 SP ReduceScatter（RS）与 residual 沿各自数据依赖执行，可在该参数 AG 结束前推进；关闭预取时不提前提交参数 AG。

**参数 AG**

同名AG出现时长差异时，需要区分三种解释。若主要由payload决定，在相同group/dtype下较大的消息应有更长的执行窗口；若主要是同stream排队，差异应表现为API提交到kernel开始的间隔；若是peer晚到，同组早到rank会长驻留而多个rank接近同时结束。三者对应不同观测，不能只凭一个kernel的长度判定“预取变慢”。

**CPU 提交**

这里要解释的是固定输入形状下，步时为何随运行推进而增长。若计算工作量增加，应看到矩阵形状、次数或设备计算时间相应改变；若CPU提交路径扫描全部历史ticket，其成本会随历史数量增长，即使每步数学工作量不变；若同步日志阻塞，应看到write停顿与GPU无新工作提交的空档对齐。增大S/microbatch可以提高每步计算密度，却不能单独解释固定工作量中的逐步退化。

## 实验

**模型与参考**

模型支持 QKV bias、GQA、RoPE、mask、embedding、final norm、LM head 与 next-token loss。模型有 28 层、1,543,714,304 个唯一参数、338 个 Parameter，直接初始化最终 TP/FSDP shard；embedding/head 真实共享，由唯一 owner 合并梯度、归约和更新。

QKV 与 Gate/Up 各共享一次 SP activation AllGather（AG）。自定义 Autograd 保存紧凑 activation，反向读取当前物化权重。训练支持 safetensors 严格加载、FP32 master、SGD/AdamW、GradScaler 和全 rank overflow 一致跳步与恢复。

实现见[分片适配](../src/sharded_runtime/runtime/qwen2_adapter.py)与[参数消费者](../src/sharded_runtime/runtime/qwen2_operators.py)。

独立参考使用 Transformers 4.57.3，检查全局 mask、跨 SP label 和有效 token 分母。

| 检查 | 已观测结果 |
|---|---|
| CPU/Gloo 分片与三步 SGD | 8 cases、16,092 次 Tensor 对照通过，最大误差 4.7684e-7 |
| 四卡 FP32 独立 HF 训练 | 29 cases、75,384 次 logits/loss、梯度和更新比较通过，最大误差 4.7684e-7 |
| FP16 每步同起点参考 | SGD/AdamW、两策略共 4 cases，五步含一次 overflow、四次更新通过；实际梯度下 3,200 次 leaf 更新与 optimizer 状态逐位重放一致 |
| 官方 1.5B 权重 | 四卡 logits/loss 最大误差 1.0204e-4 / 2.8610e-6，检查前向输出与loss |

**通信调度**

8,896/8,896 次通信均映射到指定 CUDA 流。仅把工作分配到不同 CUDA 流仍不充分：复用 communicator 会使“上一层 FSDP RS → norm TP SUM → 当前 SP AG → GEMM”形成串行链。

FSDP AG/梯度 RS、TP activation/norm SUM 使用独立通信域。前向连续提交 SP AG 和下一参数 AG；反向先分发 norm RS、主矩阵 RS 和下一参数 AG，再消费 norm bucket。writer event、dispatch token 和 lease generation 保留真实就绪关系。

完整模型、F2T2/SP、S128/B1/A1、FP16 AdamW、reshard、异步功能全开时，前向双 AG 重叠为 126/216 边界；反向 RS＋norm SUM＋AG 本 rank 驻留交集为 200/200，参与者同时驻留为 150/200，RS＋GEMM＋AG 为 74/200。

**参数 AG**

rank1 第二个预热步骤中，U0 embedding/head 按需 AG 的 FP32 输入为 **445.125MiB**、驻留 **146.169ms**；首 decoder U1 的预取输入为 **44.633MiB**、驻留 **14.523ms**，载荷与驻留均相差约十倍。

**精度对照**

线性反向 dX/dW 使用与前向一致的有效 FP16，再转为 FP32 master 梯度。S128/B1、global batch2 的 ABBA 对照每轮预热 5 步、测量 20 步，关闭采集，自研/原生为 **2.069353/1.736943 秒**，自研步时高 19.14%；两轮后五步相对前五步增长 51.77%/57.16%。这一小负载下存在明显的步时增长，需进一步检查 CPU 提交和调度开销。

**CPU 提交**

同步逐事件记录的诊断路径每 rank 每步写入 10,693 条事件；一次 **21.845528ms** write 停顿与 GPU 提交空档重合。ticket 持续追加、retire/finalize 扫描全历史，也会使 CPU 工作量随步骤增长。

训练路径使用活动 ticket/lease 索引并及时退休完成项，合并标量读回，将批量诊断写出放在计时窗外。与逐项读回相比，DToH 从每步 345 次/2737B 降至 **4 次/40B**；338 个 Adam step scalar 标注未对应 CUDA 拷贝。

**工作量与正式对照**

S128/B1 两步带采集的诊断实验为 **3.522854 秒/步、MFU 0.135575%**；该配置的线性反向使用 FP32，日志与采集增加提交成本。它与下面的正式计时存在精度、负载和采集条件差异。

容量筛选要求至少 2GiB 物理空闲、allocated≤28GiB、reserved≤29GiB。S512/B10 两次复核优于已测且满足余量的 S1024/B4、S2048/B2。正式模型随机初始化，F2T2/SP、每副本 B10、A1、global batch20，每步 10,240 个输入位置、10,220 个有效 label；GPU 常驻合成输入、AdamW lr1e-5。

同机四张 PG503-216/SM70，Torch 2.9.1+cu126、NCCL 2.27.5；TP [0,1]/[2,3] 为 NV1/PHB，FSDP [0,2]/[1,3] 为 NV2/PHB。双方线性前反向 FP16，math-attention BMM、master、参数 AG、梯度 RS、Adam 状态 FP32。原生独立使用官方 fully_shard、TP/SP、DTensor；自研使用自己的 Autograd 和调度。

ABBA 四个新进程，每轮 5 步预热、40 步测量，每引擎共 80 个全局步骤；关闭 profiler/history/事件诊断，完整 step 包含末尾 CUDA 同步，取四 rank 最大 wall。

| 指标 | ShardTrain | 原生 FSDP2＋TP/SP |
|---|---:|---:|
| 平均步时 | **2.0806 秒** | **2.2387 秒** |
| 输入 tokens/s | 4921.6 | 4574.0 |
| 模型 MFU | **9.38%** | **8.71%** |
| peak allocated | 21.18GiB | 23.12GiB |
| peak reserved | 28.68GiB | 28.42GiB |

双方各两份采集分别观察 step5/6 与 step35/36，来自不同新进程。每份 6,072 个矩阵 kernel 确认线性前反向 FP16、attention BMM FP32；32 份显存 snapshot 端点未显示持续增长。同 S512/B10 下小模型 HF loss/gradient/update 对照通过。

## 结论

- **独立通信域让参数预取与激活、梯度通信实际交错。** 8,896次通信全部落在指定CUDA流；完整模型S128/B1中，前向双AG重叠为126/216边界，反向RS＋Norm归约＋AG驻留交集为200/200。
- **这组长AG同时包含载荷差异与同流排队。** embedding/head输入445.125MiB、驻留146.169ms，decoder预取输入44.633MiB、驻留14.523ms；decoder另有143.110ms发起延迟，对应等待前序AG和自身copy-in。
- **主机记录与标量读取是可见的提交成本。** 一次21.846ms同步write与GPU提交空档重合；活动ticket/lease管理与批量标量读取后，每步DToH由345次/2737B降至4次/40B。
- **完整28层模型在S512/B10正式同机对照中，自研吞吐高7.60%。** 自研/原生平均step为2.0806/2.2387秒，模型MFU为9.38%/8.71%；较大的局部工作量下，实际完整step呈现稳定收益。
