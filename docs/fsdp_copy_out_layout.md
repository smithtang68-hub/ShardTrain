# Packed FSDP AG、Copy-out 与梯度 RS

每个参数的 TP-local 元素数为 n，FSDP degree 为 F，shard 长度 q=ceil(n/F)，padding=F*q-n。多参数打包时，offset[j]=sum(q[:j])、L=sum(q)、AGStorage 长度 M=F*L。FSDP local rank r 的参数 j 位于 r*L+offset[j]。

AG 输出为 rank-major。Copy-out 独立分配每个参数所需的 n 个连续元素，从各 rank 的 slot 提取 valid=max(0,min(q,n-r*q)) 个元素到参数区间 r*q，跳过 padding。独立 FullParam 不与 AGStorage 或其他参数共用 Storage。

转换期间的应用 buffer 约为 (M+sum(n))*sizeof(dtype)，释放 AGStorage 后消费者仍可使用 FullParam。最后读取和 allocator 安全复用由 Event/record_stream 管理；allocator reserved 不要求立即下降。

梯度按同一 destination-major 布局打包，padding 填零；RSOutput 长 L，与较大的 RSInput 分离。FSDP SUM/F 只执行一次，TP复制参数按自身语义做进一步归约。GradShard 可 view RSOutput，不能因小 view 保留不再需要的大 RSInput。

布局、padding、copy-out 和归约实现见[materializer](../src/sharded_runtime/runtime/materializer.py)与[parameter layout](../src/sharded_runtime/runtime/layout.py)。Gloo 的 AllReduce+slice 是数学替代路径，实际 NCCL 使用 RS，二者的通信成本不同。
