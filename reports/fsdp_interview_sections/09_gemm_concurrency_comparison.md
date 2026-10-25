# 实验9：并发与串行的GEMM对照

## 背景

通信并发是否更快，必须先排除计算被省略或精度改变的可能。本实验比较自研并发 C0与串行 C1的实际矩阵工作量，检验性能差异是否发生在相同计算任务上。

C1在C0的计算与通信图上增加Event等待，不应改变任何矩阵的操作数形状、dtype或数学调用次数。对每个rank，令矩阵调用特征包含operator、M/N/K/B、输入输出形状、dtype和stride，则必要预期为：

```text
multiset(signatures_C0) = multiset(signatures_C1)
sum(FLOPs_C0) = sum(FLOPs_C1)
```

由[实验8的结构推导](08_gemm_profile.md)，两者每rank应各有1182次线性GEMM、336次BMM，矩阵工作量均为19508546764800 FLOPs。任何缺失调用或低精度替换都会使这一预期不成立。

计算量相同并不要求两次GPU驻留时间相同；并发通信可能竞争资源，也可能使kernel排队不同。具体kernel算法和调用顺序是进一步观察项，不把它们相同当作数学等价的必要条件。

## 实验

两条路径均为完整 28 层 Qwen2.5-1.5B、序列长度 512、每小批量 4 条序列、累积 2 个小批量更新一次，四 rank、FSDP degree 2、TP degree 2、SP 开启。参数和 AG/RS 为FP32，线性 autocast 为FP16，math attention BMM 为 FP32。C1 通过 CUDA Event 添加真实串行等待，时间线确认了相应调度。

| 四 rank 汇总 | 结构或对照预期 | 自研并发 C0 | 自研串行 C1 |
|---|---:|---:|---:|
| 实际本地矩阵调用 | 各6072 | 6072 | 6072 |
| FP16 mm/addmm | 各4728 | 4728 | 4728 |
| FP32 bmm | 各1344 | 1344 | 1344 |
| 未匹配调用 | 0 | 0 | 0 |
| shape / dtype / stride组合与次数 | 两者相同 | 完全相同 | 完全相同 |

两次采集的阶段来自源码NVTX标记。结果不仅满足组合与次数相同，逐rank、逐调用序号的元信息也相同，GPU kernel名称及次数相同；每rank数学FLOPs均与结构推导一致。逐rank比较匹配矩阵操作的kernel区间并集之和，未观察到C0全部GEMM普遍大幅变慢。具体形状的耗时差异仍需结合阶段、rank和对应通信区间分析。

同一GEMM的split-K辅助kernel单列，数学工作量只计一次。

[专项总述](../fsdp_interview_experiments.md) · [原生对照](10_gemm_native_comparison.md) · [形状汇总工具](../../experiments/fsdp_interview/scripts/summarize_gemm_shapes.py)

## 结论

- C0与C1每个optimizer step的四rank汇总均为6072次矩阵调用，逐rank、逐调用的shape、dtype、stride相同，GPU kernel名称及次数也相同。
- 两者每rank矩阵工作量均为19508546764800 FLOPs，计算任务与操作数精度保持一致，为[实验3的调度吞吐对照](03_communication_overlap.md)提供了计算一致性依据。
