# TP/SP 数学与布局

输入与输出采用 [B,S,H]，sequence 维为 dim=1。world size=fsdp_degree*tp_degree；SP 与 TP 共用逻辑成员，不额外增加进程。TP-local 参数再进入 FSDP 分片。

Column Linear 的 PyTorch weight [out,in] 沿 dim=0 切分，Row Linear 沿 dim=1 切分。Q/K/V 与 Gate/Up 使用 Column，attention output 与 MLP Down 使用 Row；Norm及未做TP切分的参数保持TP复制。复制不等于不做FSDP分片。

## 通信与 Autograd

| 操作 | Forward | Backward |
|---|---|---|
| TP copy | identity | AllReduce SUM |
| TP reduce | AllReduce SUM | identity |
| SP gather | sequence AllGather | sequence ReduceScatter SUM |
| SP reduce-scatter | sequence ReduceScatter SUM | sequence AllGather |
| Column，SP off | copy + local Linear | 合并各feature分支的输入梯度 |
| Column，SP on | SP gather + local Linear | 返回sequence-sharded输入梯度 |
| Row，SP off | local Linear + TP reduce | 使用同一replicated输出的adjoint |
| Row，SP on | local Linear + SP reduce-scatter | gather输出梯度后计算本地dW/dX |

TP reduce 的 backward identity 对应同一个 replicated 输出的数学 adjoint，不能再次 SUM 而重复计算 loss。Gloo 使用 AllReduce+slice 实现相同数学语义，NCCL 使用真实 ReduceScatter。

QKV与Gate/Up共用组级SP输入AG。Qwen消费者将共享完整输入用于Norm和局部投影；反向保存compact local activation并regather，合并各分支dX。参数梯度通过UnitMaterialize返回稳定shard leaf。

## 梯度与累积

Loss按逻辑token/batch和accumulation缩放，避免重复累计TP副本。Norm等复制参数的梯度按其实际输入分片和归约语义同步；TP切分的投影权重不额外做重复平均。FSDP梯度归约由materializer管理，不能与TP SUM混为一次操作。

标准Block与Qwen适配器的具体消费者和支持范围分别由实现检查。[共享投影](../src/sharded_runtime/parallel/shared_projection.py)、[Column/Row Linear](../src/sharded_runtime/parallel/tp_linear.py)与[Qwen消费者](../src/sharded_runtime/runtime/qwen2_operators.py)给出实际代码。
