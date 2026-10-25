# Unit、Ticket 与 Buffer 生命周期

## Unit 与 generation

每个 unit 管理持久 shard、临时 FullParam 与一次物化的 generation。前向依次进入物化、消费和 reshard/retain；反向在需要时重新物化，随后执行梯度 RS 和 shard gradient 提交。retained unit 不重复做对应的反向 AG。

预取只提前发起目标 unit 的参数 AG，目标消费者在 copy-out 前等待该 AG。无关的当前 RS 不继承预取 AG 完成依赖。已有合法 lease 时，重复物化请求幂等处理；stale generation、提前使用和缺失梯度由状态检查拒绝。

## Stream 与依赖

逻辑 channel 包括 compute、FSDP AG/RS 和 SP AG/RS。NCCL 通信在绑定的实际 CUDA stream 上提交，具体 stream 编号随进程和 context 变化。依赖只来自数据 producer、communicator 顺序和 buffer 最后使用，不以全局等待连接所有通信。

前向在当前 SP AG 提交后发起下一个 unit 的参数预取。反向在 owner unit 提交梯度 RS 后，通过 lookahead 为后续消费者准备参数；Norm归约和当前反向按数据就绪推进。stream_ordered 与 conservative 策略的等待约束不同，但均保留数据与生命周期要求。

## Ticket 与 lease

Ticket 关联一次通信、producer-ready Event、消费者和所需 buffer。消费者在真实 stream 上接入必要完成边。lease 持有 tensor，直到最后读取完成；退休时清理引用，并通过 Event/record_stream 保护 allocator 安全复用。

Python 引用断开、GPU 使用结束和 allocator reserved 下降是三个不同边界。Autograd 的 post-accumulate Event 表示持久 .grad 的最后 writer 完成，optimizer 必须在此之后运行。

正常 step 收尾要求活动 lease 清空、稳定 shard 身份保持、梯度就绪且没有未完成的消费。异常进入失败状态，不尝试凭空恢复已分歧的 collective 顺序。

[通信实现](../src/sharded_runtime/runtime/comm.py) · [生命周期实现](../src/sharded_runtime/runtime/engine.py) · [调度实验](../reports/q5_new_topology_m0_revalidation.md)
