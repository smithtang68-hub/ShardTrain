# 实验4：通信次数与数据量

## 背景

常用的FSDP估算是每个microstep“两次参数AG、一次梯度RS”。本实验先由参数形状和生命周期推导次数与payload，再用实际通信记录核对：共享参数、前向retain、梯度分桶和逐microstep同步，是否足以解释运行结果与简单模型的差别。

固定完整28层Qwen2.5-1.5B，FSDP degree F=2、TP degree T=2、SP开启；S512、local microbatch B=4、梯度累积 A=2。以下均按**每rank、每optimizer step**统计，不把四个rank重复相加。

设参数j在TP切分后的元素数为n_j，每元素占d_j字节。FSDP通信包含padding，因此一个unit的完整payload为：

```text
q_j = ceil(n_j / F)
W_u = sum_j(F * q_j * d_j)
W   = sum_u(W_u)
rho = (F - 1) / F
```

AG以完整output、RS以完整input定义payload。理想Ring中，每rank单向发送量分别为rho乘该payload；AllReduce为2*rho乘input。这个模型不包含协议开销，也不统计收发之和、HBM访问或全GPU总量。

若所有unit前向后都reshard，反向重新AG，且梯度payload G=W，则FSDP基线为：

```text
V_simple = A * rho * (2W + G)
         = 3W                         # F=2、A=2、G=W
```

本模型TP-local完整参数为W=3,554,351,104 B，因此简单模型预测10,663,053,312 B。这里的W是TP切分后、FSDP聚合前的参数量；embedding与Norm在TP维复制，不能将全模型参数一律除以T。

这条路径有30个参数unit。投影权重按TP切分；共享embedding/head和Norm在TP维复制，再沿FSDP维分片。参数及梯度通信均为FP32，所有参数的FSDP padding在该配置下恰好为零。

由H=1536、MLP宽度I=8960、KV总宽度C=256、词表V=151936，可直接计算每层的TP-local参数数；投影矩阵、QKV bias与两个Norm分别贡献：

```text
n_block = (2H^2 + 2HC + 3HI)/T + (H + 2C)/T + 2H
        = 23,400,448
W_block = 4*n_block; W_embedding = 4*V*H; W_finalnorm = 4*H
```

| unit类别 | 数量 | 每unit完整payload（B） | 每microstep AG次数 | 每microstep RS次数 |
|---|---:|---:|---:|---:|
| 共享embedding/head | 1 | 933,494,784 | 1 | 1 |
| Decoder block | 28 | 93,601,792 | 2 | 2 |
| Final norm | 1 | 6,144 | 1 | 1 |

共享embedding/head和final norm保留前向完整参数，不需要对应的反向AG。每个decoder将main与Norm梯度分成两个RS桶，但两个桶的input之和仍等于该层完整梯度。每个microstep实际同步梯度，累积A=2不会把RS压缩为每optimizer step一次。

令r_u为前向retain指示量，k_u为该unit的梯度桶数，则预期为：

```text
N_AG = A * sum_u(2 - r_u)             = 2 * (28*2 + 1 + 1) = 116
N_RS = A * sum_u(k_u)                 = 2 * (28*2 + 1 + 1) = 116
V_AG = A * rho * sum_u((2-r_u) * W_u) = 6,175,201,280 B
V_RS = A * rho * W                   = 3,554,351,104 B
```

两个retained unit合计R=933,500,928 B。因此细化后的FSDP发送模型为：

```text
V_FSDP = A * rho * (3W - R)
       = 9,729,552,384 B
```

相对简单模型恰好少933,500,928 B；减少来自省下的反向AG。分桶则使RS从“每unit一次”的60次变成116次，**增加调用次数，不增加梯度payload总量**。这是实验应分别验证的两个预期。

每层每microstep，QKV和Gate/Up的共享输入各做一次前向AG；两个Row算子的输出梯度各做一次反向AG，共享输入的WGrad再各regather一次。因此SP AG共6次。两个Row前向各一次RS，两个共享投影反向各一次RS，因此SP RS共4次。

令L=28，D=B*S*H=4*512*1536=3,145,728，完整激活X16=2D=6,291,456 B、X32=4D=12,582,912 B。SP degree为T=2：

```text
N_SP_AG = 6 * A * L = 336
N_SP_RS = 4 * A * L = 224
V_SP_AG = A * L * ((T-1)/T) * (6*X16)          = 1,056,964,608 B
V_SP_RS = A * L * ((T-1)/T) * (2*X16 + 2*X32) = 1,056,964,608 B
```

AG操作数均为FP16；RS包含两次FP16前向输出和两次FP32反向输入梯度。因此AG/RS次数不同、模型发送量相同，并不矛盾。反向输入梯度恢复为共享投影的原输入精度，不能只按GEMM的FP16精度估算全部激活通信。

复制参数在FSDP RS后做TP AllReduce。57个Norm每microstep各一次，预计A*(2L+1)=114次，每次input为FSDP分片后的3072 B；共享embedding/head预计A=2次，每次466,747,392 B。T=2时Ring系数2*(T-1)/T=1，模型发送量分别为350,208 B和933,494,784 B。

## 实验

记录四个rank各10个测量optimizer step，逐次读取实际group、dtype、input/output shape与字节数。FSDP预测只使用注册参数布局、retain策略和分桶方式，不从已观测的通信次数反推；首次使用通过copy-out关联到对应的AG。

| 通信角色 | 预测次数 | 实际次数 | 预测Ring发送量（B） | 实测payload代入模型（B） |
|---|---:|---:|---:|---:|
| FSDP参数AG | 116 | 116 | 6,175,201,280 | 6,175,201,280 |
| FSDP梯度RS | 116 | 116 | 3,554,351,104 | 3,554,351,104 |
| SP激活AG | 336 | 336 | 1,056,964,608 | 1,056,964,608 |
| SP激活RS | 224 | 224 | 1,056,964,608 | 1,056,964,608 |
| TP Norm梯度AllReduce | 114 | 114 | 350,208 | 350,208 |
| TP embedding梯度AllReduce | 2 | 2 | 933,494,784 | 933,494,784 |

实测FSDP AG完整output合计12,350,402,560 B、RS完整input合计7,108,702,208 B，分别乘rho=1/2得到表中数值；SP AG完整output与RS完整input均为2,113,929,216 B。

每rank每step共908次通信，四rank各10步应有908*4*10=36,320个事件，与记录一致。逐unit的FSDP布局、次数和字节检查共4*10*A*30=2400项，首次使用另有2400项，全部符合预测。SP/TP行由上述层内公式与实际inventory另外核对，不混称为这2400项FSDP检查。

预取还要求区分发起阶段和首次消费阶段：按消费语义每step应为前向60次、反向56次参数AG；共享head边界会提前发起unit1供反向使用的AG，实际发起标签为前向62次、反向54次。两种划分都合计116次，对应生产者与首次copy-out的关联也吻合，没有额外增加AG。

## 结论

**参数retain减少了AG数据量，梯度分桶增加了RS调用次数；按这两项修正后，FSDP通信预测与运行记录一致。**

- 共享embedding/head与final norm保留完整参数，使每rank每步的FSDP Ring发送模型从10,663,053,312降至9,729,552,384 B，恰好减少933,500,928 B。
- 每个decoder分为main与Norm两个梯度桶，使RS由单桶模型的60次增至116次，完整梯度payload总量保持不变；参数AG同样为116次。
- SP每步336次AG、224次RS。AG均为FP16，RS包含FP16前向与FP32反向，两者的Ring发送模型均为1,056,964,608 B，解释了次数不同而数据量相同的结果。

通信成本应分别统计调用次数、payload与实际dtype，并将retain和分桶方式纳入参数生命周期模型。

[独立通信量核对实现](../../experiments/fsdp_interview/cases/q06_comm_volume.py) · [实验方案与总览](../fsdp_interview_experiments.md)
