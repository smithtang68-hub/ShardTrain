# 参数与 Autograd 生命周期

## 持久 shard 与临时 FullParam

参数先按 TP 布局划分，再对 TP-local tensor 做 flat FSDP 分片。每个 local shard 是稳定的 leaf nn.Parameter，optimizer 只绑定这些对象。参数物化生成临时 FullParam，不交换 optimizer 的参数身份，也不通过 .data 替换 leaf。

Copy-in 使用 no_grad staging；UnitMaterialize.apply 把持久 shards 连接到多输出 FullParams。Bridge 在输出梯度就绪后回传 shard gradients，由 Autograd AccumulateGrad 写入 local_shard.grad。通信布局、padding 和归约次数由 materializer 管理。

## 前向释放与反向物化

Qwen 自定义消费者保存 compact local activation 和参数 provider，不把前向 FullWeight 放进 save_for_backward。前向结束清空 compute_weight；reshard 退休物化 lease 并清理指向临时 Storage 的内部引用。provider/handle/unit 仍是强引用，必须同时满足这些释放条件。

PreBackward 在反向消费者读取参数前恢复当前 unit 的完整权重。retained unit 复用合法 generation，其他 unit 重新物化；消费者拒绝 stale generation。QKV 与 Gate/Up 的共享输入在反向重新聚合，多个投影输入梯度合并后返回。

## 梯度完成与 optimizer

FSDP 梯度按 main/norm 分桶执行 RS，TP 复制参数按语义做 SUM。持久 leaf 的 post-accumulate hook 在最后 writer 后记录 Event；finalize_backward 等待必要完成边，再允许 optimizer 使用 .grad。Work.wait、通信入队与最终 leaf writer 完成是不同边界。

最后 unit、tied embedding/head 或 retain 策略可保留完整参数。一般 consumer 的小 view、重复 backward 与高阶导数限制见[实现范围](known_limitations.md)。

[Bridge](../src/sharded_runtime/runtime/bridge.py) · [Engine](../src/sharded_runtime/runtime/engine.py) · [Materializer](../src/sharded_runtime/runtime/materializer.py)
