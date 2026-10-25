# 实验10：自研与原生的GEMM对照

## 背景

自研比原生快，可能来自调度与参数管理，也可能来自两条路径实际计算了不同的任务。本实验把自研并发C0、自研串行C1与Native FSDP2放到同一局部矩阵计量下，检验工作量与计算精度是否匹配。

三条路径使用同一模型、输入形状、TP/SP布局与计算精度。FSDP参数表示和分桶方式不同，不应减少QKV、MLP、attention和LM head所需的矩阵运算。因此预期每rank的实际本地调用特征组合及次数相同，而不要求三条Autograd路径按相同先后顺序提交。

[实验8](08_gemm_profile.md)从模型结构推导出每rank应有1182次线性GEMM、336次BMM、共1518次矩阵调用，数学工作量为19508546764800 FLOPs；四rank合计6072次。Native也应满足这组基准。

判断对象必须是local operands。DTensor的全局元信息描述逻辑计算，若与其底层本地执行同时相加，会重复计量，可能错误地得出Native做了更多计算。具体kernel驻留时间、参数通信次数与分桶不要求相同。

## 实验

FSDP 专项实验分析上述三种策略的计算，配置为完整 28 层 Qwen2.5-1.5B、序列长度 512、每小批量 4 条序列、累积 2 个小批量更新一次，FSDP degree 2、TP degree 2、SP 开启。三方实际线性输入/输出为FP16，math attention BMM 为FP32。

| 四 rank 汇总 | 结构或对照预期 | 自研 C0 / C1 | Native |
|---|---:|---:|---:|
| 本地矩阵调用 | 各6072 | 各6072，全部匹配 | 6072，全部匹配 |
| FP16 mm/addmm / FP32 bmm | 各4728 / 1344 | 各4728 / 1344 | 4728 / 1344 |
| 本地操作特征组合及次数 | 三者相同 | 相同 | 相同 |
| 不计入执行量的全局 DTensor 元信息 | 不预设条数 | 0 | 4728 |
| 缺少源码阶段标记的调用 | 单独记录 | 0 | 2024 |

逐 rank 的操作类型、M/N/K/B、数据类型、输入输出 stride 的组合及次数，以及 GPU kernel 名称及次数均相同。自研 C0/C1 调用顺序也相同；Native 顺序不同，因此按实际操作特征匹配。

按实际本地 `2MNK` 或 `2BbatchMNK` 累加，每rank三方均为19508546764800 FLOPs；FP32 BMM占数学工作量2.774%，与结构核算一致。448条额外split-K reduction单列，GEMM数学FLOPs只计一次。这里统计的是操作数对应的矩阵工作量，训练MFU仍使用统一模型公式。

[正式吞吐](../fsdp_interview_experiments.md) · [精度核对](08_gemm_profile.md) · [形状汇总实现](../../experiments/fsdp_interview/scripts/summarize_gemm_shapes.py)

## 结论

- 自研并发、自研串行与原生三条路径的本地矩阵工作量一致：每个optimizer step的四rank汇总均为6072次调用，其中FP16 mm/addmm为4728次、FP32 bmm为1344次。
- 三方每rank均为19508546764800 FLOPs，FP32 BMM占2.774%；实际operator、M/N/K/B、dtype、stride组合及次数匹配。
- 三方性能对照具有相同的本地矩阵计算基准，分析框架速度差时可以聚焦参数管理、通信调度和CPU提交的执行差异。
