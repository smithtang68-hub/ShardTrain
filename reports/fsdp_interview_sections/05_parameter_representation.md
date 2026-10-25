# 实验5：参数表示与模型状态恢复

## 背景

FlatParameter、原 Parameter 视图和 DTensor 暴露了不同的参数管理接口。本实验关注这种差异是否改变冻结参数、optimizer 分组和 model state 恢复的使用约束，以及满足各自约束后是否仍得到正确的训练结果。

四卡数据并行中，各 rank 使用不同输入。独立 dense reference 先计算各自梯度，再取平均；支持配置下，分片结果应与该参考在原容差内一致：

```text
g_reference = (1 / F) * sum_r g_r
P_after_step = AdamW(P_before_step, g_reference)
```

冻结参数应不进入 optimizer，训练后权重不变；多个 optimizer group 应让指定参数获得各自的 weight decay。参数表示决定这些设置是否能直接表达：

| 官方路径 | 依据参数表示提出的预期 |
|---|---|
| FSDP1 `use_orig_params=False` | 一个 flat leaf 不能在同一次封装中混合不同 requires_grad；也不能把同一个 leaf 同时放入不同 weight-decay 组，需要按参数属性拆分封装 |
| FSDP1 `use_orig_params=True` | 原 Parameter 身份保留，可按原参数设置冻结与分组；分片视图形状不要求与全局形状相同 |
| FSDP2 `fully_shard` | 按原参数名表达全局/本地形状及 placement，可检查冻结与多组 optimizer 的语义 |

对于 FULL model state，保存的是逻辑全局参数，恢复时应能重新切分到不同 F：

```text
P_restored = P_saved
f(x; P_restored) = f(x; P_saved)
```

因此 4→2 与 2→4 应恢复相同参数和前向输出，而不要求每个 rank 的 local shard 形状或内容相同。这一预期不包括 optimizer state 续训恢复。

## 实验

四卡 FP32 小模型包含 1104 个参数。训练检查覆盖输出、loss、gradient 和一步 AdamW 更新，另检查参数表示、冻结和多组 weight decay。FULL model checkpoint 在 4→2、2→4 的独立进程中恢复，并与独立 CPU 参考副本比较。

| 官方路径 | 实际观察与核对 |
|---|---|
| FSDP1 `use_orig_params=False` | 单根模块暴露一个 flat leaf；同组混合冻结/可训练参数被拒绝，按参数属性分别封装后可设置不同 weight-decay 组 |
| FSDP1 `use_orig_params=True` | 保留原 Parameter 对象及分组能力；本地视图可为一维或空，内部仍使用 flat storage |
| FSDP2 `fully_shard` | 按参数名暴露 DTensor 信息，本次冻结和多个 optimizer group 检查通过 |

八个可训练组合、共 32 个 rank-case 通过独立 dense reference；另四个构造拒绝与混合冻结限制一致。冻结参数不进入 optimizer、权重逐 bit 不变。FSDP1 的 full-grad 观察上下文可能临时暴露全零梯度，退出后恢复 None，不能把该临时视图误判为冻结参数参与了更新。

三条路径的 FULL model checkpoint 均完成双向跨卡数恢复，90 项参数零容差检查和 18 项 forward 检查通过。检查器使用独立参考与加载数据副本，并验证其能够识别非零误差及意外共享参考数据的情况。

## 结论

三种官方参数表示在各自支持的配置下得到一致的训练结果，并完成了小模型 FULL model state 的跨卡数恢复。

- 八个可训练组合、32 个 rank-case 通过独立 dense reference；冻结参数权重逐 bit 不变，按设定的 optimizer 分组完成更新。
- FSDP1 flat leaf 需要按冻结属性和 weight decay 拆分封装；`use_orig_params=True` 与 FSDP2 保留了按原参数表达这些设置的能力。选择参数表示会改变模型封装和 optimizer 配置方式。
- 三条路径均完成 4→2、2→4 恢复，90 项参数零容差检查和 18 项 forward 检查通过。FULL model state 可以重新切分到新的 world size，保持逻辑参数和前向结果。

[独立官方 API 实验](../../experiments/fsdp_interview/cases/q09_param_representation.py) · [模型恢复检查器](../../experiments/fsdp_interview/scripts/audit_q09_roundtrip.py)
