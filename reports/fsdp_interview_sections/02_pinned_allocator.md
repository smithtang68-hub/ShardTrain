# 实验2：Pinned allocator

## 背景

[实验1](01_pinned_transfer.md)发现临时 pin 的总成本较高。本实验进一步区分两件事：allocator 是否复用了 pinned storage，以及 pageable 数据是否仍要复制。缓存命中只能回答前者，不能直接说明 `pin_memory()` 已变为零成本。

对普通 pageable Tensor，`pin_memory()` 创建 pinned storage 并复制原内容。将耗时拆开：

```text
T_pin_cold = T_allocate_or_register + T_cpu_copy + T_allocator_overhead
T_pin_cached = T_cache_lookup + T_cpu_copy + T_allocator_overhead
```

若释放后的块可以复用，同尺寸连续分配操作应集中在首次，之后可以为零；CPU copy 的字节数仍等于输入大小，因此 API 耗时不会随分配次数归零。对于已经 pinned 的 Tensor，则可以直接返回，无需重复复制。

分配路径还有第二项可检验预期：默认配置应走 HostAlloc；启用 HostRegister 配置后，实际 CUDA API 应切换到 HostRegister。后者的完整分配过程包含 malloc、页面预触碰/mapping 和驱动注册，首次使用还可能包含初始化成本：

```text
T_register_allocation = T_malloc + T_page_preparation + T_host_register
                     + T_first_use_setup + T_other_overhead
```

## 实验

环境为 PyTorch 2.9.1+cu126、V100 级 SM70。实验沿 Tensor→HostAllocator→缓存 slow path→CUDA Host API 核对实现，并保留三次独立进程的复制测试。每进程、每尺寸连续执行三轮临时 pin，观察新分配增量和 CPU 调用时间。

HostRegister 对照在新进程启动前将 `PYTORCH_ALLOC_CONF` 与 `PYTORCH_CUDA_ALLOC_CONF` 均设为 `pinned_use_cuda_host_register:True,pinned_num_register_threads:4`，再以实际 API 判断配置是否生效。

| 128MiB 检查 | 机制预期 | 实测结果 |
|---|---|---|
| 连续三轮临时 pin | 首次分配后可命中缓存 | slow-path 新分配增量为 1/0/0 |
| 缓存命中后的临时 pin | 仍需 CPU copy | CPU pin P50 仍为 27.784ms |
| 默认配置 | 实际调用 HostAlloc | 采集确认 HostAlloc |
| 注册配置 | 切换到 HostRegister，总成本包含其他阶段 | 主注册 API 12.375ms，总 CPU allocation 151.903ms |

## 结论

**Pinned allocator 复用的是内存块；每次将 pageable Tensor 转为 pinned Tensor，仍需复制输入数据。**

- **缓存确实减少了分配。** 128MiB 连续三轮的新分配次数为1/0/0；缓存命中后的 `pin_memory()` CPU调用仍为27.784ms，内容复制仍占用准备时间。
- **分配路径可以切换，完整分配成本包含多个阶段。** 默认调用HostAlloc，启用配置后调用HostRegister；首次注册API为12.375ms，包含页面准备、初始化等阶段的CPU allocation共151.903ms。

因此，训练输入应提前准备并复用pinned buffer，把分配和数据复制安排在数据准备阶段，以完整准备耗时衡量优化效果。

[实验代码](../../experiments/fsdp_interview/cases/q01_q03_pinned_memory.py)
