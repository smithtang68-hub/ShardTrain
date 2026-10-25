# 实验8：本地GEMM分析

## 背景

配置中的 dtype 不能直接说明实际矩阵计算的精度和工作量。为了判断后续并发、串行及原生比较是否在做相同计算，本实验先从模型结构推导应有的调用次数、局部形状和数学 FLOPs，再核对实际操作数及 GPU 执行。

本实验的局部 microbatch B=4、S=512、hidden size H=1536、MLP intermediate size I=8960、总 KV hidden size Hkv=256、词表 V=151936；decoder 层数 L=28、TP degree T=2、每 optimizer step 的 microstep 数 A=2，四个 rank。

每层每 microstep 有 Q/K/V、attention output、Gate/Up/Down 共7个线性投影，此外每 microstep 有1次 LM head。普通线性反向分别计算 dX、dW，对应两次 GEMM；embedding lookup 本身不增加 GEMM。无 activation recomputation 时，每 rank 每 optimizer step 应有：

```text
linear_calls = 3 × A × (7L + 1) = 1182
attention_bmm_calls = 6 × A × L = 336
total_calls = 1518
```

attention 的6次 BMM 来自前向 QKᵀ、PV 共2次及反向4次。因此四 rank 预期为4728次线性 GEMM、1344次 BMM，共6072次；按前向/反向分开则为2024/4048次。这些数量由模型和反向结构给出，不包含 bias、Norm、softmax 等非矩阵操作。

SP在decoder投影前收集完整序列，Q projection 应为 `M=B×S=2048、N=H/T=768、K=H=1536`。每 rank 有6个query heads，attention BMM 的一个前向形状应为 `Bbatch=4×6=24、M=N=512、K=128`。LM head消费本地sequence shard，应为 `M=B×S/T=1024、N=151936、K=1536`。

一次 GEMM 的数学乘加工作量按 `2MNK`，batched GEMM 按 `2BbatchMNK` 计算。将上述局部形状按层、microstep 和前反向展开，预期每 rank 每 optimizer step：

```text
decoder_linear_FLOPs = 6AL(BS/T) × (2H² + 2H×Hkv + 3H×I)
                    = 16099684909056
attention_BMM_FLOPs = 12AL × B×S²×H/T
                    = 541165879296
LM_head_FLOPs       = 6A(BS/T) × H×V
                    = 2867695976448
total_GEMM_FLOPs    = 19508546764800
```

attention BMM 应占这一矩阵工作量的约2.774%。当前路径对线性计算使用 FP16 autocast、对 attention 选择 math backend，因此待核对的精度预期为 FP16 mm/addmm 输入输出与 FP32 attention BMM。

## 实验

分析自研并发路径 C0：四 rank、完整 28 层 Qwen2.5-1.5B、序列长度 512、每小批量 4 条序列、累积 2 个小批量更新一次，FSDP degree 2、TP degree 2、SP 开启、三个运行时异步开关全开，采集第 10 个 optimizer step。采集器直接读取实际本地 ATen 操作数，再经 NVTX→同线程 API→同进程 correlation 对应 GPU kernel。

| 四 rank 汇总 | 结构预期 | 实测 |
|---|---:|---:|
| 真实本地矩阵调用 | 6072 | 6072，全部精确匹配 |
| FP16 mm/addmm | 4728 | 4728 |
| FP32 bmm | 1344 | 1344 |
| 前向 / 反向调用 | 2024 / 4048 | 2024 / 4048，未标注0 |

Q projection、attention BMM和LM head的实际局部形状分别为 M2048/N768/K1536、B24/M512/N512/K128、M1024/N151936/K1536，与上述推导一致。按实际局部操作数累加，每 rank 为19508546764800 FLOPs，FP32 BMM占2.774%，也与结构核算一致。

本次另有448条split-K reduction，作为辅助kernel单列，同一GEMM的数学FLOPs只计一次。正式MFU与框架性能使用关闭形状采集器、TorchDispatch和NVTX采集功能的运行。

[采集器](../../experiments/fsdp_interview/cases/gemm_shapes.py) · [精确关联分析](../../experiments/fsdp_interview/scripts/analyze_gemm_trace.py)

## 结论

- 自研每个optimizer step在四rank上执行6072次矩阵调用，与结构推导一致：4728次FP16 mm/addmm、1344次FP32 bmm。
- 每rank矩阵工作量为19508546764800 FLOPs，FP32 attention BMM占2.774%。该负载的实际精度构成为FP16线性计算与FP32 attention BMM，可作为[并发、串行和原生对照](10_gemm_native_comparison.md)的统一计算基准。
- SP聚合后的decoder Q projection使用M2048，LM head使用本地M1024；按实际local operands计量能够正确反映两种位置的计算量。
