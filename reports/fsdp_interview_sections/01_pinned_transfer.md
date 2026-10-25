# 实验1：Pinned transfer与复制计算重叠

## 背景

训练输入传输有两个不同目标：让 CPU 尽早继续提交工作，以及让同一批传输与计算更早完成。仅将 `non_blocking` 打开，不能同时保证这两个收益。本实验将 CPU 调用返回、GPU 传输完成和 H2D＋GEMM 总完成时间分开测量。

对于提前准备好的 pinned 源，阻塞复制的 CPU 调用包含传输等待；异步复制主要完成入队。因此预期 CPU 调用明显缩短，但同样的字节仍需 DMA，单独一次传输的完成时间应接近：

```text
T_cpu_blocking ≈ T_submit + T_wait_dma
T_cpu_async ≈ T_submit
T_copy_complete_async ≈ T_copy_complete_blocking
```

如果每批先执行 `pageable.pin_memory()`，总路径还增加分配及 CPU copy：

```text
T_temporary_pin_path = T_pin_allocation + T_cpu_copy + T_h2d
```

所以临时 pin 不一定比直接传输更快。提前 pinned 才能将这部分准备成本移出当前传输窗口。

对彼此无数据依赖的 H2D 和 GEMM，串行 stream 要依次完成两项工作；独立 stream 才有重叠机会。忽略调度开销和资源争用时：

```text
T_serial_ideal = T_h2d + T_gemm
T_parallel_ideal = max(T_h2d, T_gemm)
maximum_hidden_time = min(T_h2d, T_gemm)
```

实际收益还受带宽、GPU 资源和提交延迟影响，不能由“使用两个 stream”直接算出固定加速比。可检验的预期是：CPU 提交等待下降；独立工作出现真实区间交集时，相同总工作量可能更快完成。

## 实验

实际设备为 V100 级 SM70，PyTorch 2.9.1+cu126。三个独立进程扫描 4/16/64/128MiB 连续 uint8，每组分别预热、重复计时。比较提前 pinned 的阻塞/异步 H2D、临时 pin 后 H2D，以及相同 H2D＋八次 GEMM 的串行/独立 stream 两种执行方式。

正式计时关闭 profiler。内容核对在计时外执行；独立短 trace 观察复制与计算的区间交集。

128MiB 的完成时间中位数如下：

| 工作 | 完成时间 P50（ms） | 与预期的对应 |
|---|---:|---|
| 提前 pinned，阻塞 H2D | 12.473 | 传输完成时间基线 |
| 提前 pinned，异步 H2D | 12.412 | 与阻塞传输接近，没有因异步提交而消除传输本身 |
| 临时 pin 后 H2D | 38.981 | 当前窗口内的 pin 与 CPU copy 增加了总成本 |
| H2D＋八次 GEMM，串行 | 13.424 | 相同总工作量的串行基线 |
| 相同 H2D＋GEMM，独立 stream | 11.481 | 总完成时间下降 14.47% |

阻塞改异步后，CPU 复制调用 P50 从 12.355 降到 0.083ms，而传输完成时间仍约 12.4ms，支持“减少 CPU 等待，而非提高单次 DMA 速度”的预期。独立 stream 的短 trace 三个测量样本中，复制与计算区间交集约 1.952ms，串行案例为零；这与 profiler-off 总完成时间下降相互印证。

并行案例的11.481ms甚至低于另一项单独H2D的12.412ms，说明不能把不同案例测得的分项时间当作不变常数，直接代入理想模型。这里的14.47%来自相同H2D＋GEMM总工作量的两种调度对照，而不是由单独传输时间推算。独立进程间的资源与提交状态会影响具体时间，因此也不将1.943ms的完成时间差直接等同于另一份trace的1.952ms交集，不声称实测达到理想 `max(T_h2d,T_gemm)` 下界。

## 结论

**提前pinned配合异步H2D减少了CPU提交等待；独立stream进一步缩短了相同复制与计算工作量的总完成时间。**

- 128MiB阻塞改异步后，CPU调用从12.355降至0.083ms；GPU传输完成时间仍约12.4ms。收益来自CPU可以继续提交工作。
- 相同H2D＋八次GEMM从串行改为独立stream，总完成时间由13.424降至11.481ms，下降14.47%。
- 临时pin后H2D需要38.981ms，提前pinned的异步H2D为12.412ms。提前准备pinned buffer能把分配和CPU copy移出当前传输窗口。

实验通过保留源/目标引用、等待完成Event及登记 `record_stream` 管理复用，所有内容核对通过，复制与计算可在数据正确的条件下重叠。

[实验代码](../../experiments/fsdp_interview/cases/q01_q03_pinned_memory.py) · [实验2：分配器机制](02_pinned_allocator.md)
