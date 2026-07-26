# 大模型推理服务系统：从请求入口到 GPU 内核的完整链路

> 本文基于大模型服务系统的脉络，结合 Orca、vLLM、SGLang、TGI、DistServe / Splitwise 等经典文章与项目，系统梳理生成式大模型推理服务（LLM Serving）的核心架构。重点放在**服务系统**本身——如何管理请求、会话、KV Cache、调度批次与分布式 Worker——而非单纯的算子级推理加速。
>
> **关于源码**：正文中的代码块为便于理解的**简化示意**；真实项目源码片段请见文末「源码附录」，其中引用了 vLLM V1 与 SGLang 主干代码。

---

## Why LLM-Inference

训练一个大模型只发生一次，但部署后它要持续响应成千上万用户的请求。与训练不同，推理服务面临的是**不确定到达的流式请求、变化的提示长度、多轮会话状态、流式/非流式输出、身份认证与限流**等工程问题。

一个推理服务系统的核心目标可以概括为：

- **低延迟**：首 token 时间（TTFT）和相邻 token 间隔（ITL/TPOT）满足业务 SLO。
- **高吞吐**：在同等 GPU 资源下服务更多并发请求。
- **高资源利用率**：让昂贵的 GPU 计算单元和 HBM 尽量被有效计算填满。
- **稳定性与可扩展性**：支持多卡、多节点、故障隔离、动态扩缩容。

这些目标很难靠单个快速算子实现，而需要一套**从 API 入口到 GPU 内核的完整系统**

---

## 推理系统分层

从系统架构看，一次用户请求大致经过以下层次：

```
用户请求
  │
  ▼
服务层：API 终端 / 请求队列 / 会话管理 / 认证分流
  │
  ▼
调度层：请求池 / 批次构建 / KV Cache 分配 / 完成判定
  │
  ▼
模型层：大模型 forward / GPU Worker / 张量并行 / 流水线并行
  │
  ▼
张量层：算子运行时 / 算子库 / 通信库（NCCL）/ CUDA Kernel
  │
  ▼
底层硬件：GPU / HBM / NVLink / 网络
```

下面各节分别对应服务层、调度层、模型层，其中 KV Cache 与会话管理贯穿服务层与调度层，是推理服务系统的“状态核心”。

---

## 会话管理与 KV Cache：LLM-Serving的状态核心

### 会话（Session）带来的产品形态

大模型对话不是无状态的一次性调用。用户需要：

- 创建 / 切换 / 删除会话；
- 多轮对话保留上下文；
- 修改历史对话后重新推理；
- 在一条历史上分叉出多个新会话；
- 随时中断或继续流式输出。

这些需求都意味着服务系统必须维护**每个对话的历史状态**，而 Transformer 自注意力的历史状态最自然的载体就是 **KV Cache**。

### KV Cache 的三种组织思路

KV Cache 是 Transformer 解码阶段避免重复计算历史 token 的关键数据结构。如何管理它，直接决定了一个服务系统能支持多少并发、多长上下文、多复杂的会话语义，常见有三种思路如下所示：

#### 思路 A：每会话独立缓存（复制式）

- 每个新会话创建一份完整 KV Cache；
- 修改历史时，复制未修改的前缀；
- 分叉会话时，拷贝公共前缀。

**优点**：实现简单，100% 命中已计算结果。
**缺点**：KV Cache 不能无限分配，复制带来额外内存与写开销，长上下文下 quickly OOM。

#### 思路 B：固定大小的 KV Cache 池 + 前缀匹配

- 系统内只分配固定数量的 KV Cache；
- 通过前缀匹配选出最合适的缓存，未命中部分重新计算。

**优点**：内存上限可控。
**缺点**：若每个槽位都按最大长度预分配，大部分情况用不满，造成大量空间浪费；总会话数受池大小限制。

#### 思路 C：按需动态分配的块式缓存（PagedAttention / RadixAttention）

这是当前工业界与开源项目最主流的做法。核心思想是把操作系统中的**虚拟内存 + 分页**机制搬到 GPU HBM 里：

- 将 KV Cache 切分为固定大小的 **block**（例如 16 个 token）；
- 每个序列维护一张**逻辑 block table**，映射到物理上不连续的 block；
- 物理 block 按需分配，不用为未生成的 token 预留空间；
- 多个序列共享同一物理 block，并通过**引用计数**管理生命周期；
- 写时复制（Copy-on-Write）保证分叉、前缀共享时的正确性。

Block 1 被多个会话共享时引用计数增加；当某个会话修改其历史时，只需复制被修改的 block，其余 block 继续共享。

### vLLM：PagedAttention 的工程化实现

vLLM（SOSP 2023）把上述块式管理落到了一个完整的生产级服务框架中。其关键贡献《Efficient Memory Management for Large Language Model Serving with PagedAttention》指出：传统框架为每个请求按最大长度连续预分配 KV Cache，导致 **60%–80% 的内存被浪费**；PagedAttention 通过非连续块分配，将浪费降到约 **4%**。

vLLM 的 KV Cache 管理包含：

- **Block Manager**：维护全局物理 block 与每序列逻辑 block table；
- **按需分配**：序列每生成若干 token 才申请新 block；
- **Prefix Caching**：缓存公共系统提示、多轮对话前缀等，提高命中；
- **Swap / Recompute**：内存不足时，可将 block 换出到 CPU，或在重新调度时重算。

这让 vLLM 在同样显存下能容纳 2–4 倍并发序列，成为现代 LLM 服务系统的事实基准之一

代码如下（逻辑block table、物理block 按需分配、引用计数、copy-on-write）（简化版）

```python
from typing import List, Dict, Optional
from copy import deepcopy

BLOCK_SIZE = 16  # 每个 block 存放 16 个 token 的 K/V

class Block:
    """物理 KV Cache 块。"""
    _id_counter = 0
    def __init__(self):
        self.id = Block._id_counter
        Block._id_counter += 1
        self.ref_count = 0          # 引用计数
        self.tokens: List[int] = [] # 该 block 已写入的 token id（示意）

class BlockSpaceManager:
    """简化版 vLLM BlockSpaceManager：逻辑 block table → 物理 block。"""
    def __init__(self, num_gpu_blocks: int):
        self.free_blocks = [Block() for _ in range(num_gpu_blocks)]
        self.block_tables: Dict[int, List[Block]] = {}  # seq_id -> 物理 block 列表

    def allocate(self, seq_id: int, num_blocks: int) -> bool:
        if len(self.free_blocks) < num_blocks:
            return False
        blocks = [self.free_blocks.pop() for _ in range(num_blocks)]
        for b in blocks:
            b.ref_count += 1
        self.block_tables[seq_id] = blocks
        return True

    def append_slot(self, seq_id: int) -> Optional[Block]:
        """为序列新增一个 token 分配 slot；跨 block 边界时申请新物理 block。"""
        blocks = self.block_tables[seq_id]
        last = blocks[-1]
        if len(last.tokens) < BLOCK_SIZE:
            return last
        if not self.free_blocks:
            return None
        new_block = self.free_blocks.pop()
        new_block.ref_count += 1
        blocks.append(new_block)
        return new_block

    def fork(self, parent_id: int, child_id: int):
        """分叉会话：子序列共享父序列的物理 block，引用计数 +1。"""
        parent_blocks = self.block_tables[parent_id]
        for b in parent_blocks:
            b.ref_count += 1
        self.block_tables[child_id] = list(parent_blocks)

    def write(self, seq_id: int, token_id: int):
        """写时复制：若 block 被多个序列共享，先复制再修改。"""
        blocks = self.block_tables[seq_id]
        last = blocks[-1]
        if last.ref_count > 1:
            # copy-on-write
            new_block = self.free_blocks.pop()
            new_block.tokens = deepcopy(last.tokens)
            new_block.ref_count = 1
            last.ref_count -= 1
            blocks[-1] = new_block
            last = new_block
        last.tokens.append(token_id)

    def free(self, seq_id: int):
        for b in self.block_tables.get(seq_id, []):
            b.ref_count -= 1
            if b.ref_count == 0:
                self.free_blocks.append(b)
        self.block_tables.pop(seq_id, None)
```

### SGLang：RadixAttention 与 Cache-Aware Scheduling

SGLang 的服务运行时（SGLang Runtime, SRT）提出了 **RadixAttention**，用 **Radix Tree** 自动复用跨请求的 KV Cache 前缀。

与 vLLM 的 PagedAttention 相比：

| 特性     | vLLM PagedAttention      | SGLang RadixAttention                    |
| -------- | ------------------------ | ---------------------------------------- |
| 基本单元 | 固定大小 block           | 固定大小 block                           |
| 共享索引 | block table + hash       | Radix Tree（前缀树）                     |
| 共享范围 | 序列内/同 batch/前缀缓存 | 自动跨请求、多轮、分支、few-shot         |
| 调度配合 | FCFS + prefix caching    | Cache-Aware Scheduling（优先命中长前缀） |

SGLang 的调度器被描述为“the brain of SGLang’s serving system”。它维护 `running_batch` 与 `waiting_queue`，并在调度时优先选择能与 Radix Tree 中已有缓存节点形成更长前缀匹配的请求，近似于对 radix tree 做深度优先遍历。这种 **cache-aware scheduling** 能显著降低重复前缀的计算量，特别适合：

- 长系统提示（system prompt）的多用户共享；
- few-shot 示例重复利用；
- 多轮对话中用户反复回到 earlier context；
- Agent / 工具调用中的分支推理（fork / join）。

下面给出一段**简化版 RadixCache** 示意代码，展示 SGLang 如何用 Radix Tree 做前缀匹配与复用。真实实现还会处理 LRU 淘汰、block 粒度的映射、与 Scheduler 的 cache-aware 配合等：

代码如下（Radix Tree前缀匹配与复用）

```python
from typing import List, Tuple, Optional

class TreeNode:
    def __init__(self):
        self.children: dict = {}      # token_id -> TreeNode
        self.value = None             # 关联的 KV block 列表（示意）
        self.ref_count = 0            # 被多少活跃序列引用

class RadixCache:
    """简化版 SGLang RadixCache：用 Radix Tree 索引可复用的 KV 前缀。"""
    def __init__(self):
        self.root = TreeNode()

    def _match_prefix(self, node: TreeNode, key: List[int]) -> Tuple[List[int], TreeNode]:
        """沿树匹配 key 的最长前缀，返回 (matched_prefix, last_matched_node)。"""
        matched = []
        while key:
            token = key[0]
            if token not in node.children:
                break
            child = node.children[token]
            # 示意：这里假设每个节点只存一个 token；真实实现会按 block 或公共段压缩
            matched.append(token)
            node = child
            key = key[1:]
        return matched, node

    def insert(self, key: List[int], value):
        """将新序列的 KV Cache 前缀插入 radix tree。"""
        matched, node = self._match_prefix(self.root, key)
        remaining = key[len(matched):]
        for token in remaining:
            node.children[token] = TreeNode()
            node = node.children[token]
        node.value = value
        node.ref_count += 1
        return matched  # 返回命中的前缀长度

    def match(self, key: List[int]) -> Tuple[Optional, int]:
        """查询 key 的最长可复用前缀。"""
        matched, node = self._match_prefix(self.root, key)
        return node.value, len(matched)

    def pretty_print(self, node=None, indent=0):
        node = node or self.root
        for token, child in node.children.items():
            marker = "*" if child.value else ""
            print("  " * indent + f"token {token} ref={child.ref_count}{marker}")
            self.pretty_print(child, indent + 1)

# 用法示例
cache = RadixCache()
cache.insert([1, 2, 3, 4], "kv_blocks_A")
cache.insert([1, 2, 3, 5], "kv_blocks_B")  # 共享前缀 [1,2,3]
val, match_len = cache.match([1, 2, 3, 5, 6])
print(f"命中前缀长度: {match_len}, value={val}")  # 命中 4 个 token
```

、、、

```python
def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
    """Find the longest cached prefix of ``key`` in the radix tree."""
    key = params.key
    key, _ = key.maybe_to_bigram_view(self.is_eagle)

    if self.disable or len(key) == 0:
        return self._empty_match_result

    key = key.page_aligned(self.page_size)
    if len(key) == 0:
        return self._empty_match_result

    value, last_node = self._match_prefix_helper(self.root_node, key)
    if value:
        value = torch.cat(value)
    else:
        value = self._empty_match_result.device_indices
    return MatchResult(
        device_indices=value,
        last_device_node=last_node,
        last_host_node=last_node,
        best_match_node=last_node,
    )


def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
    access_time = time.monotonic()
    node.last_access_time = access_time

    child_key = key.child_key(self.page_size)
    value = []
    while len(key) > 0 and child_key in node.children.keys():
        child = node.children[child_key]
        child.last_access_time = access_time
        prefix_len = child.key.match(key, page_size=self.page_size)
        if prefix_len < len(child.key):
            # 命中点落在某个 node 中间：分裂节点，得到精确边界
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            value.append(child.value)
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)
    return value, node
```

### KV Cache 管理决定服务上限

会话管理本质上就是 KV Cache 的生命周期管理。从“每会话独立”到“固定池”再到“块式动态分配 + 引用计数 + 前缀共享”，每一次演进都在用更复杂的索引结构换取更高的内存复用率。vLLM 与 SGLang 分别代表了 PagedAttention 与 RadixAttention 两种工程路线，但目标一致：**让服务系统在有限 HBM 内支持更多、更长、更复杂的会话**。

---

## 服务层：统一 API、异步 I/O 与请求队列

### 非流式与流式输出

服务层的首要职责是把大模型能力封装成用户可消费的 API，最常见的是兼容 OpenAI 的 `/v1/chat/completions`。

- **非流式（non-streaming）**：模型完整生成全部内容后一次性返回。实现简单，但用户需要等待整个生成完成，交互体验差。
- **流式（streaming）**：每生成一个 token 就通过 SSE 或 chunked response 推送给用户，用户可以“边听边想”，也可以随时中断。、类比为“从网络下载文件，用户可随时中断”。

流式输出对服务层提出了更高要求：服务端必须维护每个连接的 I/O 管线，处理客户端断联、超时、取消请求，并把这些状态同步给调度层。

### 服务终端应当是轻量的

**服务终端和大模型计算应该设计成异步的**。终端层只负责：

- 接收并解析请求（模型名称、请求内容、采样参数、工具调用配置等）；
- 身份认证、限流、分流；
- 把请求放入请求队列；
- 维护 I/O 管线（流式输出、断联处理）。

真正的计算应该下沉到调度层与模型层。这样终端层可以水平扩展，而不会因为单个请求的 forward 计算阻塞其他连接。

下面是一段**最小异步服务层示意**：用 FastAPI + asyncio Queue 演示“终端只负责接请求、丢队列、返回流式结果”，模型计算在后台独立循环中处理。

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

app = FastAPI()
request_queue: asyncio.Queue = asyncio.Queue()

@dataclass
class Job:
    job_id: str
    prompt: str
    response_queue: asyncio.Queue  # 每个 job 独立的输出通道

async def model_worker():
    """后台模型 Worker：不断从队列取 job，逐个 token 模拟生成。"""
    while True:
        job = await request_queue.get()
        # 真实系统中这里调用调度器 + model runner
        for token in f"echo: {job.prompt}".split():
            await asyncio.sleep(0.05)  # 模拟推理延迟
            await job.response_queue.put(token + " ")
        await job.response_queue.put(None)  # None 表示生成结束

async def token_stream(job: Job) -> AsyncIterator[str]:
    while True:
        token = await job.response_queue.get()
        if token is None:
            break
        yield token

@app.post("/chat")
async def chat(prompt: str):
    job = Job(job_id="abc", prompt=prompt, response_queue=asyncio.Queue())
    await request_queue.put(job)
    return StreamingResponse(token_stream(job), media_type="text/plain")

# 启动：uvicorn.run(app) 并在外部启动 model_worker()
```

这段代码的核心思想是：**HTTP handler 不阻塞在模型计算上**，而是通过 `asyncio.Queue` 把 job 投递给后台模型 Worker，每个 job 再通过自己的 `response_queue` 把流式结果传回客户端。

### 两种典型架构：TGI 的 Router-Worker 分离 vs vLLM 的 API Server-Scheduler

**Hugging Face Text Generation Inference (TGI)** 采用明确的三组件架构：

1. **Router（Rust）**：HTTP/gRPC 入口，负责接收请求、维护队列、动态组 batch、暴露 API；
2. **Launcher**：启动模型 server 并协调参数；
3. **Model Server（Python）**：加载模型、执行 forward、管理 KV Cache，多个 shard 通过 NCCL 同步。

Router 与 Model Server 可以不在同一台机器，便于独立扩缩容。

**vLLM** 的架构则更像一个**集中式调度器**：

- `LLMEngine` 中有一个 **Scheduler**；
- Scheduler 维护 waiting / running / swapped 三类队列；
- 每个迭代决定哪些请求入 batch、哪些完成、哪些需要抢占或换出；
- 多个 GPU Worker 执行 Scheduler 分配好的 forward。

两种架构都体现了“**入口轻量、计算下沉、队列解耦**”的设计哲学。

---

## 调度层：迭代级调度与连续批处理

### 从静态批处理到连续批处理

最早的大模型服务采用**静态批处理**：先把一批请求凑齐，再一起 forward，等最慢的请求生成完才能释放 batch。这导致：

- 短请求被长请求拖慢；
- batch 大小随请求完成而缩水，GPU 利用率下降；
- 无法处理请求动态到达。

**Orca**（OSDI 2022，《Orca: A Distributed Serving System for Transformer-Based Generative Models》）首次系统性地提出了 **iteration-level scheduling**，也就是今天常说的 **continuous batching（连续批处理）**：

- 调度粒度是**单个解码迭代**，而不是整个请求；
- 每个迭代结束后，已完成的请求立即离开 batch，新到达的请求可以立刻加入；
- 同一 batch 内可以混合不同阶段的请求（prefill + decode）。

相比静态批处理，Orca 在实验中取得了最高 **36× 的吞吐提升**，奠定了现代 LLM 服务系统调度层的基础。

为了直观理解连续批处理，下面是一段**最小可运行模拟器**。它没有真实模型，但展示了 `waiting / running / completed` 三个队列如何在每个迭代中动态变化：新请求加入、已完成请求离开、长请求继续运行。

```python
import random
from collections import deque
from dataclasses import dataclass

@dataclass
class Request:
    req_id: int
    prompt_len: int          # prefill 长度
    remaining_decode: int    # 还需要生成的 token 数
    state: str = "waiting"   # waiting / running / completed

def simulate_continuous_batching(num_iters=12, max_batch=4):
    waiting = deque()
    running = []
    completed = []
    rng = random.Random(42)

    # 模拟请求动态到达
    def maybe_arrive(t):
        if rng.random() < 0.5:
            waiting.append(Request(
                req_id=t,
                prompt_len=rng.randint(2, 6),
                remaining_decode=rng.randint(2, 8)
            ))

    for it in range(num_iters):
        maybe_arrive(it)

        # 1. 把 waiting 中请求加入 running（受 batch 大小限制）
        while len(running) < max_batch and waiting:
            req = waiting.popleft()
            req.state = "running"
            # 第一次调度：先 prefill，这里简化为直接扣减 decode
            running.append(req)

        # 2. 执行一次 forward：每个 running 请求生成一个 token
        still_running = []
        for req in running:
            req.remaining_decode -= 1
            if req.remaining_decode <= 0:
                req.state = "completed"
                completed.append(req)
            else:
                still_running.append(req)
        running = still_running

        print(f"iter {it:2d}: waiting={len(waiting):2d}  "
              f"running={len(running):2d}  completed={len(completed):2d}")

simulate_continuous_batching()
```

运行结果类似：

```text
iter  0: waiting= 0  running= 1  completed= 0
iter  1: waiting= 1  running= 2  completed= 0
iter  2: waiting= 2  running= 3  completed= 0
...
iter 10: waiting= 3  running= 3  completed= 6
iter 11: waiting= 4  running= 3  completed= 7
```

每个迭代结束后，batch 的组成都在变化：没有请求会被一个长请求“绑架”到结束。

### 三个 batching 问题

连续批处理有必须解决的三个问题：

#### 问题 1：短请求需要等待长请求

在静态 batch 中，一个长生成请求会阻塞同 batch 的短请求。连续批处理通过“每迭代重新组 batch”解决：短请求生成 `<end>` 后立即返回，长请求继续留在 running batch 中。

#### 问题 2：短输入需要 padding

不同请求的输入长度不同。为凑成矩形张量，往往要给短输入补 padding。连续批处理配合**动态 padding / padding-free attention**（如 Flash Attention、PagedAttention Kernel）可以显著减少无效计算。

#### 问题 3：Prefill 与 Decode 同时出现

- **Prefill（提示编码）**：一次性处理整个 prompt，计算密集（compute-bound），token 数多；
- **Decode（逐 token 生成）**：每次只处理一个新 token，访存密集（memory-bound），反复读取 KV Cache。

两者对 GPU 的压力完全不同。、可以跨请求批量化的计算包括 MLP、QKV Linear、RMS Norm、RoPE；而连接 KV Cache 与 Self Attention 需要按请求单独处理（因为每个序列的 KV Cache 位置不同）。

### vLLM 的调度策略

vLLM 的 Scheduler 在每个迭代做以下判断：

1. 从 waiting queue 中按 FCFS 取请求；
2. 检查当前 GPU 上是否有足够的 KV Cache block；
3. 如果可以，将请求加入 running batch；
4. 执行一次 forward；
5. 判断哪些请求完成、哪些需要 preempt/swap。

随着版本演进，vLLM 还加入了：

- **Chunked Prefill**：把长 prompt 的 prefill 切成多个小块，与 decode 请求混合调度，避免长 prefill 独占 GPU 导致短 decode 等待；
- **Prefix Caching**：调度时优先利用已缓存的公共前缀；
- **Multi-Step Scheduling**：一次调度连续执行多步 decode，减少 CPU 调度开销。

下面是一段**简化版 vLLM Scheduler** 示意代码， `waiting / running / swapped` 三队列在每个调度步如何流转

```python
from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class Sequence:
    seq_id: int
    status: str = "waiting"          # waiting / running / swapped
    prompt_tokens: List[int] = field(default_factory=list)
    output_tokens: List[int] = field(default_factory=list)
    logical_blocks: List[int] = field(default_factory=list)  # 逻辑 block id
    blocks_to_copy: List[tuple] = field(default_factory=list)

class Scheduler:
    """简化版 vLLM Scheduler。真实实现还包含 prefix caching、
    chunked prefill、preemption 策略、speculative decoding 等。"""
    def __init__(self, block_manager, max_num_seqs=8, max_model_len=4096):
        self.block_manager = block_manager
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len

        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []
        self.swapped: List[Sequence] = []

    def add_request(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> dict:
        """每个推理迭代调用一次，返回本轮需要 forward 的序列集合。"""
        # 1. 先把 swapped 中能够重新获得 block 的请求加回 running
        new_swapped = []
        for seq in self.swapped:
            if self.block_manager.can_allocate(seq):
                self.block_manager.allocate(seq)
                seq.status = "running"
                self.running.append(seq)
            else:
                new_swapped.append(seq)
        self.swapped = new_swapped

        # 2. running 中已完成或需要抢占的请求处理
        still_running = []
        for seq in self.running:
            if seq_is_done(seq):
                self.block_manager.free(seq)
                seq.status = "finished"
            elif not self.block_manager.can_append_slot(seq):
                # KV Cache 不足，抢占：这里简化为 swap 到 CPU
                seq.status = "swapped"
                self.swapped.append(seq)
            else:
                still_running.append(seq)
        self.running = still_running

        # 3. 从 waiting 中按 FCFS 取请求加入 running
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq):
                break
            self.waiting.pop(0)
            self.block_manager.allocate(seq)
            seq.status = "running"
            self.running.append(seq)

        # 4. 构造调度输出给 model runner
        return {
            "num_prefill_seqs": sum(1 for s in self.running if not s.output_tokens),
            "num_decode_seqs": sum(1 for s in self.running if s.output_tokens),
            "running": self.running,
            "swapped": self.swapped,
        }

def seq_is_done(seq: Sequence) -> bool:
    # 示意：达到最大长度或生成结束符
    return len(seq.output_tokens) >= 20 or (seq.output_tokens and seq.output_tokens[-1] == 2)
```

关键语义：

- `schedule()` 在每个 forward 前被调用一次；
- 它同时处理 `swapped` 恢复、`running` 完成/抢占、`waiting` 准入；
- 与 `BlockSpaceManager` 配合，KV Cache 不足时把低优先级序列换出，而不是直接 OOM。

### SGLang 的 Cache-Aware 与 Zero-Overhead 调度

SGLang 的调度器在 Orca / vLLM 的基础上做了两方面的特色优化：

1. **Cache-Aware Scheduling**：调度时不仅看请求到达顺序，还看该请求能与 Radix Tree 中已有缓存节点匹配多长前缀，优先调度“缓存命中率高”的请求，最大化 KV Cache 复用。
2. **Zero-Overhead CPU Scheduling**：通过 `OverlapThread` 将 CPU 端的调度、tokenization、detokenization 与 GPU 计算重叠，使 GPU 利用率达到 95% 以上。

SGLang 的请求流也体现了服务系统的典型路径：

```
用户请求
  → TokenizerManager（分词）
  → Scheduler（入 waiting queue 或 running batch）
  → TpModelWorker / ModelRunner（forward）
  → 采样
  → DetokenizerManager（回退为文本）
  → 返回客户端
```

---

## Prefill vs Decode：阶段特性、Chunked Prefill 与 PD 分离

### 两个阶段的资源特征

、 prefill 与 decode 的资源需求：

| 阶段    | 计算特征                  | 瓶颈                    | 出现频率                | 优化目标                  |
| ------- | ------------------------- | ----------------------- | ----------------------- | ------------------------- |
| Prefill | 计算密集（compute-bound） | 矩阵乘法算力            | 较低（每个请求一次）    | 降低 TTFT                 |
| Decode  | 访存密集（memory-bound）  | KV Cache 带宽、HBM 容量 | 极高（每个 token 一次） | 降低 TPOT / ITL，提高吞吐 |

因为两者特征相反，把它们简单混在一起调度会出现“prefill 拖慢 decode”或“decode 让 prefill 算力吃不饱”的问题。

### Batch 的三种形态

现代服务框架通常把 batch 分为三类：

1. **Prefill-Decode 混合**：batch 内请求长度不固定，既有 prefill 也有 decode。灵活性高，但负载不均衡。
2. **Chunked Prefill**：把超长 prefill 切成若干短 prefill chunk，与 decode 混合调度，使硬件算力使用更均衡。vLLM V1 已默认启用。
3. **Decode Only**：batch 内每个请求长度都为 1，只包含 decode。计算行为固定，非常适合 CUDA Graph 优化，实际出现频率很高。

下面用一段**简化代码**展示 Chunked Prefill 如何将长 prompt 切分为固定大小的 chunk，并与 decode 请求一起组成一个 batch。真实框架中，这一逻辑位于 Scheduler 内部，并需要与 PagedAttention Kernel 配合处理不连续的 KV block：

```python
from typing import List
from dataclasses import dataclass

CHUNK_SIZE = 512  # 每个 prefill chunk 最多处理多少 token

@dataclass
class Req:
    req_id: int
    prompt: List[int]        # 待 prefill 的 token
    generated: List[int]     # 已生成的 token
    prefill_offset: int = 0  # 下一次 chunk 从 prompt 的哪个位置开始

    def is_prefilling(self) -> bool:
        return self.prefill_offset < len(self.prompt)

    def next_prefill_chunk(self) -> List[int]:
        end = min(self.prefill_offset + CHUNK_SIZE, len(self.prompt))
        chunk = self.prompt[self.prefill_offset:end]
        self.prefill_offset = end
        return chunk

def build_mixed_batch(waiting: List[Req], running: List[Req], max_tokens: int):
    """把 decode 请求和 prefill chunk 混合成一个 batch，控制总 token 数。"""
    batch = []
    token_budget = max_tokens

    # 1. 优先保留/加入 decode 请求（每个只需 1 个新 token）
    for req in running:
        if not req.is_prefilling() and token_budget >= 1:
            batch.append(("decode", req, [req.generated[-1]]))
            token_budget -= 1

    # 2. 剩余预算用于 prefill chunk
    for req in waiting:
        if req.is_prefilling():
            chunk = req.next_prefill_chunk()
            if len(chunk) <= token_budget:
                batch.append(("prefill", req, chunk))
                token_budget -= len(chunk)
            else:
                # 预算不够：回退 offset，等下一轮
                req.prefill_offset -= len(chunk)
                break
    return batch

# 示例：一个长 prompt 请求和两个 decode 请求组成 batch
req_long = Req(1, prompt=list(range(900)), generated=[])
req_d1 = Req(2, prompt=[], generated=[100, 101])
req_d2 = Req(3, prompt=[], generated=[200])

batch = build_mixed_batch([req_long], [req_d1, req_d2], max_tokens=1024)
for kind, req, tokens in batch:
    print(f"{kind:7s} req={req.req_id} tokens={len(tokens)}")
```

输出类似：

```text
decode  req=2 tokens=1
decode  req=3 tokens=1
prefill req=1 tokens=512
```

长 prompt 不会一次性占满 GPU，而是分 chunk 与 decode 交替执行，既保证 decode 不被长时间阻塞，又充分利用 prefill 的算力密集特性。

### PD 分离：把两个阶段放到不同硬件

当系统规模扩大到多 GPU / 多节点时，一种更激进的思路是 **Prefill-Decode Disaggregation（PD 分离）**：把 prefill 和 decode 放到不同的 GPU 池甚至不同机型上，各自用最适合的并行策略和延迟目标。

代表性工作：

- **DistServe**（OSDI 2024）：提出按 goodput（同时满足 TTFT 与 TPOT SLO 的请求比例）优化，将 prefill/decode 物理分离，通过 KV Cache “pull” 从 prefill 实例转移到 decode 实例。在同节点 NVLink 下转移开销 <0.1%，可实现 **7.4× goodput 提升** 或 **12.6× 更紧的 SLO**。
- **Splitwise**（ISCA 2024）：研究阶段拆分后的同构/异构硬件选择，包括功耗受限的 decode 集群，可在 iso-cost 下提升 1.4× 吞吐、iso-power 下提升 2.35× 吞吐。
- **Mooncake**（Moonshot AI）：以 KV Cache 为中心的分离式架构，用 CPU/DRAM/SSD 作为共享 KV Cache 池，支撑 Kimi 生产环境。
- **NVIDIA Dynamo**（2025）：开源分布式服务框架，内置 disaggregated serving 与 NIXL 快速 KV 传输。

PD 分离的核心权衡是 **KV Cache 传输开销**。当模型采用 MLA 等紧凑 KV 表示时，传输成本大幅下降，PD 分离的优势会进一步放大。

---

## 模型层与张量层：Worker、分布式执行与通信

### Worker 的组织方式

模型层可以被描述为“多个不断循环的线程或进程”：

- 每个 Worker 等待调度层准备好输入；
- 执行一次 forward；
- 将结果写回对应请求；
- 把请求返回请求池，更新状态。

在分布式推理中，一般**每个 GPU 实例对应一个线程/进程**，避免重复切换设备带来的开销。vLLM 的 `Worker` / `ModelRunner`、SGLang 的 `TpModelWorker` / `ModelRunner`、TGI 的 Python Model Server 都是这一抽象的不同实现。

### 并行策略

当单卡放不下模型时，模型层需要引入并行：

- **张量并行（Tensor Parallelism, TP）**：把单层权重切分到多张卡，每张卡算一部分，结果通过 NCCL all-reduce / all-gather 合并。适合降低单卡显存、减少 decode 阶段的 HBM 压力。
- **流水线并行（Pipeline Parallelism, PP）**：把模型按层切分到多张卡，数据像流水线一样传递。适合超大规模模型跨节点部署，但会带来 bubble。
- **数据并行（Data Parallelism, DP）**：同一模型复制到多组卡，各自服务不同请求。吞吐扩展最自然，但要求单卡能放下模型。

实际生产系统往往是 TP + PP + DP 的组合，并由调度层统一决定请求进入哪一组副本。

### CUDA Graph 与通信库

Decode Only batch 的计算图非常固定，非常适合 **CUDA Graph** 捕获：把 kernel 启动开销降到几乎为零，对小 batch、短序列的收益尤其明显。

底层通信则几乎都由 **NCCL**（NVIDIA Collective Communications Library）或对应硬件的通信库完成。调度层构建的 batch 在 Worker 内部被拆分为正确的 tensor shape，再经由 NCCL 在 TP/PP 组内同步。

---

完整链路：

```
用户 → 终端 → 请求池 → 调度器 → 批次 → 大模型
                ↑           ↓
            KVCache 池 ←  结果回写
```

现代 LLM 服务系统的所有核心问题：

1. **状态管理**：如何用块式、共享、引用计数的 KV Cache 支撑多轮、分叉、修改历史的会话？
2. **请求接入**：如何通过异步 API、流式输出、轻量终端解耦 I/O 与计算？
3. **调度策略**：如何用连续批处理、cache-aware scheduling、chunked prefill、PD 分离平衡延迟与吞吐？
4. **分布式执行**：如何通过 TP/PP/DP/CUDA Graph/NCCL 把单模型扩展到多卡多节点？

从 Orca 的 iteration-level scheduling，到 vLLM 的 PagedAttention，到 SGLang 的 RadixAttention，再到 DistServe / Splitwise 的 PD 分离，这些经典工作共同勾勒出一条主线：**大模型推理服务正在从“把模型跑起来”进化为“在有限资源下高质量地服务大量动态请求”**。

### 未来趋势

- **更长上下文**：KV Cache 容量与管理复杂度将持续膨胀，压缩（MLA、量化）、换出、分层存储会更重要。
- **Agent 与多轮工具调用**：分支、fork/join、状态共享会进一步放大 RadixAttention 这类前缀复用机制的价值。
- **多模态服务**：图像、音频、视频 token 的加入会让 batching、padding、KV Cache 结构都发生变化。
- **异构与边缘部署**：PD 分离、模型切分、动态调度将不仅发生在数据中心 GPU 之间，也会发生在 CPU/GPU/NPU/边缘设备的混合环境中。

---

## 源码附录：真实项目中的关键片段

> 以下代码片段均来自 vLLM 与 SGLang 官方仓库的最新主干（vLLM `v1/core/sched/scheduler.py`、`v1/core/kv_cache_manager.py`；SGLang `srt/managers/scheduler.py`、`srt/managers/schedule_policy.py`、`srt/mem_cache/radix_cache.py`）。为便于阅读，对注释与空行做了少量精简，但保留了核心逻辑与类/方法签名。

### A. vLLM V1 Scheduler：迭代级调度与抢占

`vllm/v1/core/sched/scheduler.py` 中的 `Scheduler.schedule()` 是 vLLM 服务系统的核心。它首先调度 `running` 队列，处理完成/抢占；然后调度 `waiting` 队列，利用前缀缓存减少重复计算。

```python
def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
    self.current_step += 1

    scheduled_new_reqs: list[Request] = []
    scheduled_resumed_reqs: list[Request] = []
    scheduled_running_reqs: list[Request] = []
    preempted_reqs: list[Request] = []

    req_to_new_blocks: dict[str, KVCacheBlocks] = {}
    num_scheduled_tokens: dict[str, int] = {}
    token_budget = self.max_num_scheduled_tokens

    self.kv_cache_manager.new_step_starts()

    # 1) 先调度 RUNNING 请求（保留已运行序列的连续性）
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]

        num_new_tokens = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.scheduler_config.long_prefill_token_threshold
        num_new_tokens = min(num_new_tokens, token_budget)

        # 为新 token 分配 KV block；不足则抢占低优先级请求
        while True:
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_lookahead_tokens=self.num_lookahead_tokens,
            )
            if new_blocks is not None:
                break

            if self.policy == SchedulingPolicy.PRIORITY:
                preempted_req = max(
                    self.running,
                    key=lambda r: (r.priority, r.arrival_time),
                )
                self.running.remove(preempted_req)
                # 回滚已记录的 token 预算
                if preempted_req in scheduled_running_reqs:
                    scheduled_running_reqs.remove(preempted_req)
                    preempted_req_id = preempted_req.request_id
                    token_budget += num_scheduled_tokens.pop(preempted_req_id)
                    req_to_new_blocks.pop(preempted_req_id)
                    req_index -= 1
            else:
                preempted_req = self.running.pop()

            self._preempt_request(preempted_req, scheduled_timestamp)
            preempted_reqs.append(preempted_req)
            if preempted_req == request:
                break

        if new_blocks is None:
            break

        scheduled_running_reqs.append(request)
        request_id = request.request_id
        req_to_new_blocks[request_id] = new_blocks
        num_scheduled_tokens[request_id] = num_new_tokens
        token_budget -= num_new_tokens
        req_index += 1

    # 2) 再调度 WAITING 请求（新请求加入）
    if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
        while (self.waiting or self.skipped_waiting) and token_budget > 0:
            num_running = len(self.running) + self.num_waiting_for_streaming_input
            if num_running >= self.max_num_running_reqs:
                break

            request_queue = self._select_waiting_queue_for_scheduling()
            request = request_queue.peek_request()

            # 利用前缀缓存：先查询已缓存 token 数
            if request.num_computed_tokens == 0:
                (
                    new_computed_blocks,
                    num_new_local_computed_tokens,
                    request.shared_prefix_boundary,
                ) = self.kv_cache_manager.get_computed_blocks(request)
                num_computed_tokens = num_new_local_computed_tokens
            else:
                new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                num_new_local_computed_tokens = 0
                num_computed_tokens = request.num_computed_tokens

            # 计算还需要计算的 token 数（chunked prefill 在这里切块）
            num_new_tokens = request.num_tokens - num_computed_tokens
            threshold = self.scheduler_config.long_prefill_token_threshold
            if 0 < threshold < num_new_tokens:
                num_new_tokens = threshold

            if (
                not self.scheduler_config.enable_chunked_prefill
                and num_new_tokens > token_budget
            ):
                break
            num_new_tokens = min(num_new_tokens, token_budget)

            # 分配 KV block
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
                num_new_computed_tokens=num_new_local_computed_tokens,
                new_computed_blocks=new_computed_blocks,
            )
            if new_blocks is None:
                break

            request_queue.pop_request()
            self._allocate_and_set_running(request)
            scheduled_new_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens

    # 3) 构造 SchedulerOutput 给 Model Runner
    return SchedulerOutput(
        scheduled_new_reqs=scheduled_new_reqs,
        scheduled_resumed_reqs=scheduled_resumed_reqs,
        scheduled_running_reqs=scheduled_running_reqs,
        preempted_reqs=preempted_reqs,
        num_scheduled_tokens=num_scheduled_tokens,
        req_to_new_blocks=req_to_new_blocks,
        # ... 其余字段省略
    )
```

### B. vLLM V1 KVCacheManager：前缀缓存与按需分配

`vllm/v1/core/kv_cache_manager.py` 中的 `get_computed_blocks` 与 `allocate_slots` 体现了 PagedAttention 的块式管理：

```python
def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int, int]:
    if not self.prefix_cache_lookup_enabled(request):
        return self.empty_kv_cache_blocks, 0, 0

    # 最长可命中长度限制为 prompt_length - 1，因为最后一个 token 需要重算 logits
    max_cache_hit_length = request.num_tokens - 1
    computed_blocks, num_new_computed_tokens, num_uncached = (
        self.coordinator.find_longest_cache_hit(
            request.block_hashes, max_cache_hit_length
        )
    )
    shared_prefix_boundary = (
        num_new_computed_tokens + num_uncached if num_uncached else 0
    )
    blocks = self.create_kv_cache_blocks(computed_blocks)
    return blocks, num_new_computed_tokens, shared_prefix_boundary


def allocate_slots(
    self,
    request: Request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: KVCacheBlocks | None = None,
    num_lookahead_tokens: int = 0,
) -> KVCacheBlocks | None:
    # 计算各类 token 边界：已计算、新命中、外部缓存、待计算
    num_local_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
    total_computed_tokens = min(
        num_local_computed_tokens + num_external_computed_tokens,
        self.max_model_len,
    )

    # 对 waiting/preempted 请求保留 watermark，避免频繁抢占
    watermark_blocks = 0
    if request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
        watermark_blocks = self.watermark_blocks

    num_tokens_need_slot = min(
        total_computed_tokens + num_new_tokens + num_lookahead_tokens,
        self.max_model_len,
    )

    # 计算还需分配多少物理 block
    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=num_tokens_need_slot,
        new_computed_blocks=new_computed_block_list,
        num_encoder_tokens=num_encoder_tokens,
        total_computed_tokens=num_local_computed_tokens + num_external_computed_tokens,
        num_local_computed_tokens=num_local_computed_tokens,
        num_tokens_main_model=total_computed_tokens + num_new_tokens,
    )

    available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
    required_blocks = num_blocks_to_allocate + watermark_blocks
    if required_blocks > available_blocks:
        return None  # 显存不足，调度器会触发抢占或延迟

    # 真正分配物理 block
    new_blocks = self.coordinator.allocate_new_blocks(
        request.request_id,
        num_tokens_need_slot,
        num_tokens_main_model,
        num_encoder_tokens,
    )

    # 将已确认 token 对应的 block 加入前缀缓存
    num_tokens_to_cache = min(total_computed_tokens + num_new_tokens, request.num_tokens)
    self.coordinator.cache_blocks(request, num_tokens_to_cache)

    return self.create_kv_cache_blocks(new_blocks)
```

### C. SGLang Scheduler：事件循环与批次规划

`python/sglang/srt/managers/scheduler.py` 中的事件循环展示了 SGLang 服务系统的主循环结构：

```python
def event_loop_normal(self):
    """A normal scheduler loop."""
    while True:
        if self.gracefully_exit:
            break

        # 1. 接收前端请求
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)
        if self._engine_paused:
            continue

        # 2. 规划下一个 batch
        plan = self.get_next_batch_to_run(
            running_batch=self.running_batch, last_batch=self.last_batch
        )
        self.running_batch = plan.running_batch
        batch = plan.batch_to_run
        self.cur_batch_for_debug = batch

        # 3. 执行 forward 并处理结果
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
        else:
            self.on_idle()

        self.last_batch = batch
```

在 `_get_new_batch_prefill_raw` 中，SGLang 使用 `PrefillAdder` 逐条评估 waiting 队列中的请求，并综合考虑 radix cache 前缀命中、chunked prefill、token 预算：

```python
def _get_new_batch_prefill_raw(self, ...):
    # ... 前置检查省略 ...

    # 按调度策略计算 waiting 队列优先级
    self.policy.calc_priority(self.waiting_queue, running_batch)

    adder = PrefillAdder(
        self.page_size,
        self.tree_cache,          # RadixCache
        self.token_to_kv_pool_allocator,
        running_batch,
        self.new_token_ratio_tracker.current,
        self.max_prefill_tokens,
        chunked_prefill_size,
        running_bs if self.is_mixed_chunk else 0,
        # ... 其他参数
    )

    # 逐个尝试把 waiting 请求加入本 batch
    for req in self.waiting_queue:
        running_bs = len(running_batch.reqs)
        if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
            running_batch.batch_is_full = True
        if running_batch.batch_is_full:
            break

        # 调用 PrefillAdder 评估：是否可加入、是否触发 chunked prefill
        res = adder.add_one_req(req, self.chunked_req is not None, None)
        if res == AddReqResult.FIT_PARTIAL:
            # 部分 fit，生成 chunked request 留到下一轮
            self.chunked_req = req
            break
```

### D. SGLang RadixCache：Radix Tree 前缀匹配

`python/sglang/srt/mem_cache/radix_cache.py` 中，`match_prefix` 与 `_match_prefix_helper` 是 RadixAttention 的核心：

```python
def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
    """Find the longest cached prefix of ``key`` in the radix tree."""
    key = params.key
    key, _ = key.maybe_to_bigram_view(self.is_eagle)

    if self.disable or len(key) == 0:
        return self._empty_match_result

    key = key.page_aligned(self.page_size)
    if len(key) == 0:
        return self._empty_match_result

    value, last_node = self._match_prefix_helper(self.root_node, key)
    if value:
        value = torch.cat(value)
    else:
        value = self._empty_match_result.device_indices
    return MatchResult(
        device_indices=value,
        last_device_node=last_node,
        last_host_node=last_node,
        best_match_node=last_node,
    )


def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
    access_time = time.monotonic()
    node.last_access_time = access_time

    child_key = key.child_key(self.page_size)
    value = []
    while len(key) > 0 and child_key in node.children.keys():
        child = node.children[child_key]
        child.last_access_time = access_time
        prefix_len = child.key.match(key, page_size=self.page_size)
        if prefix_len < len(child.key):
            # 命中点落在某个 node 中间：分裂节点，得到精确边界
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            value.append(child.value)
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = key.child_key(self.page_size)
    return value, node
```

### E. SGLang PrefillAdder：Cache-Aware 准入

`python/sglang/srt/managers/schedule_policy.py` 中的 `PrefillAdder.add_one_req` 展示了 SGLang 如何基于 `prefix_indices`（来自 RadixCache 的命中长度）做 cache-aware 准入：

```python
def add_one_req(self, req: Req, has_chunked_req: bool, truncation_align_size):
    # ... 前置检查省略 ...

    max_new = min(
        max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
        CLIP_MAX_NEW_TOKENS,
    )
    # 真正需要 prefill 的长度 = 总长度 - radix cache 命中长度
    cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
    total_tokens = cand_extend_input_len + max_new + self.page_size

    # 根据 host_hit_length 和 page_size 调整实际输入 token 数
    real_input_tokens = cand_extend_input_len - req.host_hit_length
    real_input_tokens = self.ceil_paged_tokens(real_input_tokens)
    prefix_len = len(req.prefix_indices)

    if total_tokens >= self.rem_total_tokens:
        return AddReqResult.NO_TOKEN

    # ... SWA/LoRA/CP 等特殊路径省略 ...

    if (
        self.rem_chunk_tokens is None  # chunked prefill 关闭
        or cand_extend_input_len <= self.rem_chunk_tokens  # 可一次性 fit
    ):
        # 非 chunked：本迭代完成整个 prefill
        req.set_extend_range(
            len(req.prefix_indices), len(req.full_untruncated_fill_ids)
        )
        self.can_run_list.append(req)
        self._update_prefill_budget(
            0,
            req.extend_range.length,
            min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS),
            req.retracted_stain,
        )
    else:
        # Chunked prefill：只取一个 chunk
        trunc_len = self.rem_chunk_tokens
        req.set_extend_range(
            len(req.prefix_indices), len(req.prefix_indices) + trunc_len
        )
        self.can_run_list.append(req)
        self.new_chunked_req = req
        self._update_prefill_budget(0, trunc_len, 0, req.retracted_stain)

    return self.budget_state()
```

这些真实代码与正文中的简化示意相互印证：vLLM 通过 `KVCacheManager` 与 `Scheduler` 配合实现 PagedAttention + continuous batching；SGLang 则通过 `RadixCache` 与 `PrefillAdder` 配合实现 cache-aware scheduling + chunked prefill。

---

## 参考与延伸阅读

### 开源项目与文档

- **vLLM**: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) | [Architecture Docs](https://www.mintlify.com/vllm-project/vllm/concepts/architecture) | [Optimization Docs](https://docs.vllm.ai/en/latest/performance/optimization.html)
- **SGLang**: [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) | [Architecture Overview](https://sgl-project-sglang-93.mintlify.app/developer/architecture-overview) | [System Architecture](https://sgl-project-sglang-93.mintlify.app/concepts/architecture)
- **TGI**: [https://github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) | [Architecture Docs](https://hugging-face.cn/docs/text-generation-inference/architecture)
- **NVIDIA Dynamo**: 开源分布式服务框架，支持 PD 分离与 KV 传输。
- **Mooncake**: Moonshot AI 的 KV-Centric 分离式服务系统。

*本文重点讨论 LLM 推理的“服务系统”层面——请求如何被接收、调度、复用状态并分布式执行。若读者关注的是算子级推理加速（如 Flash Attention、量化、投机采样等），可另行展开。*
