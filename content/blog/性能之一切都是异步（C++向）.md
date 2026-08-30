---
title: '性能之都是异步（C++）'
description: '一切都是传输，一切都是异步'
pubDate: '2026-08-30'
heroImage: '/img/2.png'
tags:
  - cpp
  - infra
---

# 性能之一切都是传输，一切都是异步（C++向）
 从底层向上，进行高性能cpp的逻辑梳理，后续大概还会延伸至 Cuda相关内容

Ps: 个人梳理，可能会有点列概念的感觉...

### 硬件感知

现代CPU的速度远超内存，因此在性能瓶颈位于 CPU从内存中获取数据时，就可以考虑到L1/L2/L3缓存，缓存行（Cache Line，通常为 64字节）是CPU与内存交换数据的最小单位，在多个线程频繁访问同一Cache Line的不同变量，会导致其他的核心缓存失效，严重影响性能；可以通过 `alignas(std::hardware_destructive_interference_size)` 强制对齐，消除**伪共享**



CPU访问对齐数据时，可以在单个内存事务中完成。未对其的数据可能需要多次访问，使用alignas or aligned_alloc 可以控制对齐方式；同时这也是 SIMD指令（like SSE/AVX）的强制要求



在CPU的流水线技术下，遇到错误的分支预测会清空流水线，因此在Hot Path下，应尽量使用 `if constexpr` 将分支判断移到编译期，避免运行时的分支预测开销



LSU与Cache控制器，LSU将数据从缓存加载到CPU寄存器，通常包含 地址生成单元（AGU），用于计算访存指令的物理地址；为了隐藏访存延迟，LSU内部会有 预取器提前拉取数据，同时还有 加载/存储队列，用于缓冲和重排序乱序执行的访存请求。Cache控制器，完成主存地址到Cache地址的转换，LSU负责发出读写请求，Cache控制器管理缓存命中、替换和与主存的交互

### 内存管理与分配

内存分配核心思想是减少系统调用，避免内存碎片，提高缓存命中率



通过右值引用（&&）和 `std::move` 转移资源所有权，避免深拷贝带来的巨大开销；简单解释一下，深拷贝需要进行整块内存数据的复制（memcopy），但采用 右值引用就只是将临时的堆内存拿出来继续使用；而 move 不宜用任何数据，不生成机器码，本质就是一个Cast（强制类型转换），编译器标记（左值标记为右值） -> 触发移动构造（调用移动构造函数）-> 指针窃取（将对方的堆指针拿过来，拿走大小元素，把对方的指针置空，把对方的大小置零）



采用内存池，频繁的 new/delete 引发系统调用开销，堆内存碎片化，多线程下锁的竞争；内存池可以提前创建一批内存，采用 FreeList管理，适用于 对象大小固定或者变化小的场景



采用缓存友好的数据结构，AoS(结构体数组)与SoA(数组结构体)，后者充分利用CPU的预取机制，对于需要向量化计算的场景，需要连续同类型的数据因此采用SoA，更可以适用于SIMD



伪共享与内存对齐；多线程并发修改在同一个Cache Line的不同变量，由于缓存一致性协议（like MESI），当改变了一个变量后，整个CPU都需要主从加载整个Cache Line，就会导致频繁的 缓存行（Cache Line）无效化和回写，通过 alignas(64)，或者加入 padding，将两个变量隔开64字节（Cache Line的大小）就可以解决了



### 联调通信

多路CPU服务器上，NUMA(非统一内存访问)架构，采用 numactl或系统API进行内存绑定，确保线程访问本地内存，避免跨节点内存访问带来的巨大延迟；



后续CPU与GPU进行协同

CPU是延迟导向的，负责长处理复杂的逻辑控制，GPU是吞吐量导向的，复杂大量的数据密集型并行计算，部分trick举例

1. 页锁定内存（Pinned Memory）普通的  malloc  分配的内存是“分页内存（Pageable Memory）”，它可能会被操作系统随时交换到磁盘。如果直接对这种内存使用  cudaMemcpyAsync ，CUDA 驱动为了保证数据安全，会在底层偷偷将其转换为同步操作。必须使用  cudaHostAlloc  分配页锁定内存，才能触发 DMA（直接内存访问）引擎进行真正的零阻塞传输。

2. Stream 内部的严格保序（In-Order Execution）CUDA Stream 内部的指令是严格顺序执行的。在上述代码中，同一个 Stream 内的  Memcpy(H2D) -> Kernel -> Memcpy(D2H)  绝对不会乱序。这意味着你不需要手动写任何 Event 或 Barrier 来同步这三步，硬件自动帮你保证了数据依赖的正确性。

3. Stream 之间的并发（Out-of-Order Execution）不同 Stream 之间的任务是并发的。当 Stream 0 的 GPU 正在执行  Kernel  计算时，DMA 引擎可以同时在 Stream 1 中执行  Memcpy(H2D) 。这就实现了计算与通信的重叠（Overlap）。

4. 任务分块（Chunking / Tiling）如果整个 Tensor 只有 1 个 Stream 来处理，GPU 算完最后一块时，CPU 才开始传下一批，重叠效果极差。将大 Tensor 切分为 N 块（如代码中的  num_chunks ），并在多个 Stream 之间轮询（Round-Robin）分发，能让 GPU 的计算单元和 PCIe 的 DMA 引擎始终处于“满载”状态

	

### 一切皆异步

如果采用同步模型，系统就会变成“单行道”：CPU 传数据 -> 阻塞等 GPU -> GPU 算完 -> 阻塞等 CPU 采样 -> CPU 传下一批... 这种模式下，CPU 和 GPU 永远有一方阻塞。



异步的核心目的，就是实现“流水线重叠（Pipeline Overlap）”：

- 计算与传输重叠：当 GPU 正在算第 N 层时，DMA 引擎正在把第 N+1 层需要的数据搬进显存。
- CPU 与 GPU 重叠：当 GPU 在算第 N 个 Token 时，CPU 正在异步处理第 N-1 个 Token 的解码（Detokenize）和停止条件判断，同时还在处理下一个 HTTP 请求的 Tokenizer。
- 多流并发：通过多个 CUDA Stream，把大任务切碎，让 GPU 的几千个核心和 PCIe 总线永远处于“满载”状态。



返回单CPP操作CPU同理，CPU计算与从缓存中 从内存中获取数据，对于内存详细的管理，伪共享受缓存一致性影响回写无效化，其实异步和传输一直都在程序设计的最本质，但或许我们需要根据自己真实的需求，来对齐到底层的粒度



在真实的场景下，这个逻辑依旧可用

- vLLM 的异步输出处理（Async Output Processing）：

GPU 算出 Token 后，必须等 CPU 判定“是否遇到停止符（如 ）”才能继续。这导致 GPU 被迫停下来等 CPU。新版 vLLM 的做法是：CPU 判定第 N 个 Token 的同时，GPU 假设它没停止，直接继续算第 N+1 个 Token。用极小的“无效计算”代价，换来了 GPU 的绝对不空闲

- BladeLLM 的纯异步架构（TAG）：

BladeLLM 甚至把调度器（Scheduler）和模型执行器（Model Runner）彻底解耦。它们之间通过共享内存和 Unix Domain Socket 进行异步消息传递。整个系统没有任何全局同步点（Barrier），CPU 协程和 GPU Worker 像齿轮一样完美咬合，实现了真正的“零等待”



Zero-Copy & 统一内存：

既然传输是瓶颈，那能不能不传？现在有不少相关的trick正在实现

- Pinned Memory（页锁定内存）：让 CPU 内存不被操作系统换页，DMA 引擎可以直接以最高速度搬运。
- CUDA Unified Memory（统一内存）：通过底层硬件（如 NVLink 或 PCIe P2P），让 CPU 和 GPU 共享同一个虚拟地址空间。代码里不需要写 cudaMemcpy，硬件会自动在后台按需迁移数据（Data Migration）。



### GPU内存的tile，warp调度

TODO