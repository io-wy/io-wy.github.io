---
title: 'Go Internals'
description: 'Go GMP scheduling, memory management and garbage collection'
pubDate: '2026-02-22'
heroImage: '/img/5.png'
pinned: true
tags:
  - backend
  - golang
---

用 AI 复习一下 go，然后顺手整理成文章和大家分享一下~

## GMP

GMP = goroutine + machine + processor

基本认知：**G 是 goroutine，M 是 OS 线程，P 是执行 Go 代码需要的上下文（本地队列、缓存等）。M 必须拿到 P 才能跑 G**

P 自己会维护本地队列，然后还有全局队列，每个 P 维护的本地队列里面携带着 G，然后跟 M 结合就可以工作了。

### 队列与偷取

- 每个 P 有本地队列，另外有全局队列
- 优先本地取任务，本地空时先全局拿，再从别的 P 偷一半（work stealing）

### 阻塞与 handoff

G 发生 syscall 长阻塞时，M 可能让出 P 给别的 M，避免 CPU 闲置，等 M 做完之后还会分配空闲的 P 给 M，不然就新创建一个

### 具体细节

**g0**

g0 是一类特殊的 goroutine，负责 g 之间的切换调度，一个 m 有一个 g0。

**G 的状态机**

| 状态 | 含义 |
|------|------|
| _Grunnable | 可运行，已在某个队列等待 |
| _Grunning | 正在 CPU 上执行 |
| _Gwaiting | 等待事件（锁、channel、条件）被 park |
| _Gsyscall | 执行系统调用中 |
| _Gdead | 结束或未复用 |
| _Gpreempted | 被抢占 |

**调度主循环**

1. g0 调 `schedule()`
2. `schedule()` 调 `findRunnable()` 找下一个 G
3. 找到后 `execute(gp, ...)`
4. `execute` 最后 `gogo(&gp.sched)` 切到业务 G
5. 业务 G 因让出/阻塞/结束再 `mcall(...)` 切回 g0
6. g0 继续下一轮 `schedule()`

goroutine 切换由 g0 完成：schedule -> findRunnable -> execute -> gogo，业务 g 通过 mcall 交还控制权。

**四类调度事件**

1. **主动让出**：`runtime.Gosched`
   - 当前 G 从 running 变 runnable，回队列等下次调度
2. **被动阻塞**：gopark/park_m
   - 如锁、channel 等，running -> waiting，等 goready 唤醒
3. **正常结束**：goexit0
   - running -> dead，调度器继续找下一个
4. **抢占调度**：sysmon/retake
   - 用于防止 syscall 长时间占着资源导致系统吞吐下降

**Syscall**

1. **进入 syscall（reentersyscall）**
   - G 状态改为 _Gsyscall
   - P 状态改为 _Psyscall
   - 当前 M 与 P 脱钩（M 进内核态可能长时间不可用）
   - 记录 oldp，返回用户态时优先尝试拿回

2. **syscall 期间的全局监控（sysmon -> retake）**
   - 监控线程观察各 P 状态
   - 如果某 P syscall 太久且系统有调度压力，会把该 P 夺回（设 idle 并 handoffp）

3. **退出 syscall（exitsyscall）**
   - 快路径：oldP 还可用，直接恢复 running 继续执行
   - 慢路径：拿不到 P，就把自己放回全局 runnable 队列，M 休眠/再调度

> "syscall 时 M 可能卡住，但 P 会被回收再利用，保证整体并行度不被拖死。"

### 小结

- **G 是 goroutine（任务），M 是内核线程（执行体），P 是调度上下文（本地队列+资源）。M 必须绑定 P 才能执行 G，P 的数量由 GOMAXPROCS 决定并行上限。**
- **调度取任务顺序**：runnext -> 本地 runq -> 全局 runq -> netpoll -> work stealing
- **本地队列优先**是为了减少全局锁竞争；**work stealing** 是为了均衡负载
- **goroutine 切换**由 g0 完成：schedule -> findRunnable -> execute -> gogo，业务 G 通过 mcall 交还控制权
- **阻塞分两类**：用户态阻塞（锁/channel）走 gopark/goready；系统调用阻塞走 entersyscall/exitsyscall

### Code

```go
package main

import (
	"fmt"
	"runtime"
	"time"
)

var sink uint64

func busy(id int, done chan<- struct{}) {
	for i := 0; i < 200_000_000; i++ {
		sink += uint64(i)
	}
	fmt.Println("worker done:", id)
	done <- struct{}{}
}

func main() {
	runtime.GOMAXPROCS(1) // 强行单核，便于观察调度
	done := make(chan struct{}, 2)

	go busy(1, done)
	go busy(2, done)

	ticker := time.NewTicker(300 * time.Millisecond)
	defer ticker.Stop()

loop:
	for {
		select {
		case <-done:
			if len(done) == 1 {
			}
		case <-ticker.C:
			fmt.Println("tick...")
		}
		if len(done) == 2 {
			break loop
		}
	}
	fmt.Println("all done")
}
```

运行：

```bash
GOMAXPROCS=1 GODEBUG=schedtrace=1000,scheddetail=1 go run main.go
```

---

## 内存

### 运行时

靠 runtime 分层缓存保证分配足够快，锁足够少。

**基本对象**

- **page**：Go 堆的最小存储页，通常是 8KB（Go runtime 视角）
- **mspan**：由多个 page 组成，是"最小管理单元"；内部切成固定大小对象槽位

**三层缓存角色**

- **mcache**：每个 P 私有缓存，分配小对象几乎无锁，最快
- **mcentral**：按大小类别聚合 mspan，有锁，但锁粒度比全局小
- **mheap**：全局堆，最终内存来源；不够时再向 OS 申请（如 mmap）

**一次分配**

代码里的 new/make 先进入 mallocgc，先判断对象大小和是否含有指针；小对象就优先从 P 的 mcache 对应的 mspan 拿空槽；mcache 当前的 span 满了，就从 mcentral refill（拿一个）span，如果 mcentral 也没有可用 span，就让 mheap 用空闲 page 组一个新 span；如果 mheap 空闲页不足，再向 OS 申请。

---

## 垃圾回收

垃圾回收更新挺快的，所以就先说经典的三色标记然后再说更新吧。

- **白色**：当前确定未存活
- **灰色**：确认自己存活，但是指向的对象还没扫完
- **黑色**：自己和下游都扫完了（都活下来啦！）

**标记规则**：从灰对象出发，将其所指向的对象都置灰。所有指向对象都置灰后，当前灰对象置黑。

然后就会出点两类问题：
1. **多标/浮动垃圾**：用户协程和 GC 协程**并发**执行，部分垃圾对象被误标记从而导致 GC 未按时将其回收
2. **漏标**：初始时刻对象 B 持有对象 C 的引用，GC 把 A 置黑，此时 B 还是灰色，未扫描，用户协程建立 A 指向 C，然后 B 删除对 C 的指向，GC 协程才开始执行对 B 的扫描，由于 B 无法达到 C，但是 A 已经扫描过了，因此 C 是**不可达对象**，就会被**误删**

**屏障机制**

1. **插入写屏障**（Dijkstra 思路）：在"建立新引用"前，把目标对象先置灰，阻止黑->白裸连
2. **删除写屏障**（Yuasa 思路）：在"删除旧引用"前，把旧目标先置灰，避免最后一条边被并发删掉后漏标
3. **混合写屏障**（Go 实战方案）：把上面两种效果结合，核心目标是：并发下既不漏标，又尽量减少 STW 成本

**在混合屏障机制下**：
1. 堆对象写指针时走屏障
2. 栈本身不做同等粒度的逐写屏障（成本太高）
3. 运行时通过栈扫描阶段配合屏障来兜底正确性
4. 所以不要說"Go 完全无 STW"，而是"STW 很短，主要用于阶段切换和根处理"