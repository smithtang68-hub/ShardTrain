# FP16 数值诊断

本目录使用 4 层、hidden size 32、39,456 个参数的小模型，比较分布式执行与独立 Hugging Face 参考的梯度、参数更新和 AdamW 状态。它用于研究 FP16 舍入及优化器状态的传播，不用于性能测量。

## 对照方法

- 四卡 F2T2/SP，前后向参数预取与异步梯度 RS 均开启，seed 313，AdamW learning rate 0.001，初始 scale 128、growth interval 2。
- `same_start` 每步从同一起点比较；`independent` 让两条轨迹各自更新。两种模式分别运行，判断范围不同。
- 独立 Hugging Face 算术对照不调用自研模型或分布式通信，用来观察分区计算与浮点运算次序的影响。
- 采集目标参数的归约前后梯度、unscale、Adam 状态和计算视图，支持实际梯度重放与误差分布分析。

原有比较容差保持不变。有限数值和短程更新不代表长期收敛，单步同起点比较也不能替代独立轨迹比较。诊断采集包含同步和 CPU 拷贝，其耗时不能与关闭 profiler 的完整训练计时比较。

## 源码与接口

| 用途 | 源码 |
|---|---|
| 四卡轨迹采集 | [capture_trajectory.py](capture_trajectory.py) |
| 独立参考算术对照 | [pure_hf_control.py](pure_hf_control.py) |
| 轨迹分布与趋势 | [compare_trajectories.py](compare_trajectories.py) |
| 实际状态重放 | [replay_adamw.py](replay_adamw.py) |

在仓库根目录、安装依赖后查看实际参数：

```bash
python -m experiments.fp16_numerics.capture_trajectory --help
python -m experiments.fp16_numerics.pure_hf_control --help
python -m experiments.fp16_numerics.compare_trajectories --help
```

四卡采集需要当前 GPU 映射与输出目录，参考算术对照可独立指定设备。输出保存在运行目录，本地保留。

训练精度与参考检查的整体说明见[Qwen 集成与诊断](../../reports/qwen_integration_and_diagnostics.md)。
