---
title: 'Kubernetes Operator 开发：从 CRD 到 controller-runtime 实战'
description: '深入 Operator 架构，理解 client-go 的 Informer/Workqueue 机制，再到 controller-runtime 的 Reconcile 模式'
pubDate: '2026-07-16'
heroImage: '/img/2.png'
tags:
  - kubernetes
  - operator
  - controller
  - client-go
  - controller-runtime
---

Operator 不是魔法，它是 Kubernetes 既有机制的一种组合范式——用 **CRD** 把领域语义带进 K8s API，用 **控制器** 把 spec 落地为实际资源。从 `client-go` 手写控制器，到 `controller-runtime` 只关心 Reconcile，再到 Kubebuilder 一键生成脚手架，每一层解决的是不同的抽象级别问题。

---

## What is an Operator

Operator 的本质：

* **CRD** 把应用领域的语义建模成 Kubernetes API（比如 `ModelClaim`、`PodAutoscaler`）；
* **控制器（Controller）** 持续观察（Reconcile Loop）这些自定义资源，并把期望状态（Spec）同步到实际状态（Status）；
* 控制器可以基于 `client-go` 手写，也可以基于 `controller-runtime` / Kubebuilder 工程化生成。

为什么要用 Operator？因为有些东西用 Deployment + ConfigMap 叠不出来——管理有状态应用、做有感知的扩缩容、编排复杂工作流，这些场景天然需要自定义控制面逻辑。

---

## Architecture Stack

Operator 的技术栈从底往上可以分为五层：

| 层次 | 组件 | 职责 |
|------|------|------|
| **Kubernetes API** | kube-apiserver + etcd | 资源存储、认证鉴权、Watch 推送 |
| **CRD 体系** | `apiextensions.k8s.io/v1` | 自定义资源定义、Schema 校验、版本转换 |
| **client-go** | Reflector + DeltaFIFO + Indexer + Workqueue | List/Watch 本地缓存、事件去重、限速重试 |
| **controller-runtime** | Manager + Builder + Reconciler | 抽象工作队列、封装缓存、提供声明式 Watch |
| **脚手架** | Kubebuilder / Operator SDK | 代码生成、RBAC/CRD/Webhook 集成 |

大多数生产级 Operator（比如 AIBrix、KServe、Argo）都跑在第 4 层以上——用 `controller-runtime` 组织控制器，用 Kubebuilder 管理项目骨架，但你如果不懂第 3 层 client-go 的原理，遇到诡异问题时很难定位根因。

---

## CRD

### CRD & CR

**CustomResourceDefinition（CRD）** 告诉 API Server：

* 这个自定义资源的 GVK 是什么；
* 有哪些字段、字段类型是什么；
* 支持哪些 API 版本、版本间如何转换；
* 是否需要额外的校验 / 默认值 / 转换 Webhook。

CRD 创建后，用户就可以创建它的实例——**Custom Resource（CR）**。API Server 会把它像 Pod 一样持久化到 etcd，并对外提供 RESTful 接口。

### Example

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: crontabs.stable.example.com
spec:
  group: stable.example.com
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                cronSpec:
                  type: string
                image:
                  type: string
                replicas:
                  type: integer
                  minimum: 1
                  maximum: 10
  scope: Namespaced
  names:
    plural: crontabs
    singular: crontab
    kind: CronTab
    shortNames:
      - ct
```

* `served: true` —— 该版本可通过 API 访问；
* `storage: true` —— etcd 存储的是这个版本（一个 CRD 只能有一个 storage 版本）；
* 多版本时通过 **Conversion Webhook** 做互转。

### GVK + Spec + Status

每个 K8s 对象的结构分三块：

| 字段 | 含义 |
|------|------|
| `metadata` | 名字、命名空间、Label、OwnerReference、Finalizer |
| `spec` | **期望状态**，用户想让它变成什么样 |
| `status` | **实际状态**，控制器汇报当前是什么样子 |

控制器的核心任务，就是不断把 `spec` 翻译成集群里的一群子资源（Pod、Service、PVC 等），并把结果写回 `status`。

---

## client-go Internals

### Why Informer

如果控制器反复直接 `List` API Server，两个问题是致命的：

1. **性能爆炸**——控制器越多、资源越多，API Server 压力越大；
2. **事件丢失**——两次 List 之间发生的变化无法被感知。

`client-go` 的 **Informer** 用 `List + Watch` 解决这两个问题：先全量列出，再长连接监听增量事件，把事件缓存在本地。**Workqueue** 把事件转化成"待处理任务"，管理去重、限速、重试。

### Data Flow

```text
┌─────────────┐     List / Watch      ┌──────────┐
│  API Server │ ←──────────────────→ │ Reflector │
└─────────────┘                       └────┬─────┘
                                           │ Add/Update/Delete/Replace/Sync
                                           ▼
                                    ┌─────────────┐
                                    │  DeltaFIFO  │  ← 每个 key 挂一串 Deltas
                                    └──────┬──────┘
                                           │ Pop()
                                           ▼
                              ┌────────────────────────┐
                              │ sharedProcessor /       │
                              │ HandleDeltas            │
                              └───────┬────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   ┌─────────────┐           ┌─────────────────┐          ┌─────────────┐
   │ Indexer/    │           │ Resource Event  │          │  Workqueue  │
   │ ThreadSafe  │           │ Handlers        │          │ (rate-lim)  │
   │ Store       │           │ (OnAdd/Update/  │          │             │
   │ (lister)    │           │  Delete)        │          │             │
   └─────────────┘           └─────────────────┘          └──────┬──────┘
                                                                 │ Get()
                                                                 ▼
                                                          ┌─────────────┐
                                                          │   Worker    │
                                                          │ syncHandler │
                                                          └─────────────┘
```

### Core Components

**Reflector**（`tools/cache/reflector.go`）
只做一件事：对某个 GVK 执行 **ListAndWatch**。List 拿到全量快照 → `Replace`，Watch 监听增量事件 → `Add/Update/Delete`，Resync 周期塞回 `Sync` 类型的 Delta。通过 `resourceVersion` 保证 Watch 断连后能续上。

**DeltaFIFO**（`tools/cache/delta_fifo.go`）
**按 key 聚合 Delta 列表** 的队列：

```go
type Delta struct {
    Type   DeltaType   // Added, Updated, Deleted, Replaced, Sync
    Object interface{}
}
```

每个对象 key 挂一串 Delta，Pop 时一次性弹出，好处是去重 + 可见完整变更历史 + 支持 Sync。内部用 `sync.Cond` 做阻塞 Pop。

**Indexer / ThreadSafeStore**
线程安全的本地缓存。默认 key = `"namespace/name"`。控制器通过 **Lister** 读取，O(1) 且不经过网络：

```go
foo, err := c.lister.Foos(namespace).Get(name)
```

缓存是过期的（stale），所以 Reconcile 必须是 level-triggered。

**Workqueue**（`util/workqueue/`）
把"对象发生了变化"转化为"请处理这个 key"：

| 能力 | 说明 |
|------|------|
| 去重 | 同一 key 多次 Add，队列里只保留一个 |
| 延迟 | `AddAfter(key, duration)` |
| 限速 | `AddRateLimited(key)`，支持指数退避 |
| 重试 | 出错后重新入队，`Forget(key)` 成功后清除 |
| Done 语义 | `Get()` 必须配 `Done(key)` |

### Event Handler

Informer 的 handler 不应该做重活，只做一件事：**把 key 塞进 workqueue**。

```go
informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
    AddFunc: func(obj interface{}) {
        key, err := cache.MetaNamespaceKeyFunc(obj)
        if err == nil {
            queue.Add(key)
        }
    },
    UpdateFunc: func(oldObj, newObj interface{}) {
        key, err := cache.MetaNamespaceKeyFunc(newObj)
        if err == nil {
            queue.Add(key)
        }
    },
    DeleteFunc: func(obj interface{}) {
        key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
        if err == nil {
            queue.Add(key)
        }
    },
})
```

注意 `DeleteFunc` 用 `DeletionHandlingMetaNamespaceKeyFunc`，因为删除事件可能只是一个 `DeletedFinalStateUnknown` tombstone。

---

## Minimal Controller

### Struct & Init

```go
type Controller struct {
    clientset   kubernetes.Interface
    informer    informers.FooInformer
    lister      listers.FooLister
    queue       workqueue.RateLimitingInterface
    syncHandler func(key string) error
}

func NewController(
    clientset kubernetes.Interface,
    informerFactory informers.SharedInformerFactory,
) *Controller {
    fooInformer := informerFactory.Stable().V1().Foos()

    c := &Controller{
        clientset: clientset,
        informer:  fooInformer,
        lister:    fooInformer.Lister(),
        queue: workqueue.NewNamedRateLimitingQueue(
            workqueue.NewItemExponentialFailureRateLimiter(5*time.Second, 5*time.Minute),
            "foos",
        ),
    }

    c.syncHandler = c.processNextItem

    fooInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
        AddFunc:    c.enqueue,
        UpdateFunc: func(_, new interface{}) { c.enqueue(new) },
        DeleteFunc: c.enqueueDeletion,
    })

    return c
}
```

### Worker Loop

```go
func (c *Controller) Run(workers int, stopCh <-chan struct{}) error {
    defer c.queue.ShuttingDown()

    if !cache.WaitForCacheSync(stopCh, c.informer.Informer().HasSynced) {
        return fmt.Errorf("failed to sync caches")
    }

    for i := 0; i < workers; i++ {
        go wait.Until(c.runWorker, time.Second, stopCh)
    }

    <-stopCh
    return nil
}

func (c *Controller) runWorker() {
    for c.processNextWorkItem() {
    }
}

func (c *Controller) processNextWorkItem() bool {
    key, quit := c.queue.Get()
    if quit {
        return false
    }
    defer c.queue.Done(key)

    err := c.syncHandler(key.(string))
    if err == nil {
        c.queue.Forget(key)
        return true
    }

    c.queue.AddRateLimited(key)
    return true
}
```

### syncHandler

```go
func (c *Controller) processNextItem(key string) error {
    ns, name, err := cache.SplitMetaNamespaceKey(key)
    if err != nil {
        return err
    }

    foo, err := c.lister.Foos(ns).Get(name)
    if err != nil {
        if errors.IsNotFound(err) {
            return nil // 对象已被删除，做清理
        }
        return err
    }

    // 幂等地创建/更新派生资源
    if err := c.reconcileFoo(foo); err != nil {
        return err
    }

    // 写回 status
    fooCopy := foo.DeepCopy()
    fooCopy.Status.AvailableReplicas = ...
    _, err = c.clientset.StableV1().Foos(ns).UpdateStatus(
        context.TODO(), fooCopy, metav1.UpdateOptions{})
    return err
}
```

关键设计要点：
* `workers` 决定并发度，所有 worker 从同一个 queue 取 key；
* `AddRateLimited` 在出错时按指数退避重试；
* `Forget(key)` 成功后清除重试状态，防止内存泄漏；
* 关闭时先 `close(stopCh)`，等 informer 和 worker 退出，再 `ShuttingDown()`。

---

## controller-runtime

手写 client-go 控制器虽然能学到原理，但重复代码太多了——Scheme 注册、Cache 管理、Watch 注册、Workqueue 限速、Leader Election、Metrics、优雅关闭……`controller-runtime` 把这些标准化了。

### Manager

```go
mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
    Scheme:             scheme,
    MetricsBindAddress: ":8080",
    LeaderElection:     true,
    LeaderElectionID:   "example-operator-lock",
})
```

Manager 统一提供：Client（typed client）、Cache（共享缓存）、EventRecorder、Leader Election、Runnable 生命周期。

### Builder

```go
func (r *FooReconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&stablev1.Foo{}).                          // 主资源
        Owns(&corev1.Pod{}).                           // 子资源变化触发 Reconcile
        Watches(
            &source.Kind{Type: &corev1.ConfigMap{}},
            handler.EnqueueRequestsFromMapFunc(r.mapConfigMapToFoo),
        ).
        Complete(r)
}
```

### Event Pipeline

```text
API Server
    ↓ (Watch)
Source (source.Kind 等)
    ↓
Predicate (过滤事件，如 generation 没变就跳过)
    ↓
EventHandler (转成 reconcile.Request)
    ↓
WorkQueue
    ↓
Controller worker
    ↓
Reconciler.Reconcile(ctx, req)
```

| 组件 | 作用 |
|------|------|
| `source.Source` | 产生原始事件，常见 `source.Kind` |
| `predicate.Predicate` | 过滤事件，减少无效 Reconcile |
| `handler.EventHandler` | 把事件映射成 `reconcile.Request{NamespacedName}` |
| `WorkQueue` | 内部仍是 client-go 的 rate-limiting queue |
| `Reconciler` | 你写的业务逻辑 |

### Reconcile

```go
func (r *FooReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    var foo stablev1.Foo
    if err := r.Get(ctx, req.NamespacedName, &foo); err != nil {
        if apierrors.IsNotFound(err) {
            return ctrl.Result{}, nil
        }
        return ctrl.Result{}, err
    }

    // 幂等地同步派生资源
    // ...

    if err != nil {
        return ctrl.Result{RequeueAfter: 30 * time.Second}, err
    }

    return ctrl.Result{}, nil
}
```

* `Reconcile` 只拿一个 `Request`（namespace/name），天然 **level-triggered**；
* 不要依赖事件顺序——事件可能被合并、重排、重试；
* 对象不存在就返回 `nil`，不要报错；
* `RequeueAfter` 和返回 `err` 都会重新入队，前者不触发指数退避。

### Owns & Finalizer

* **Owns**：控制器创建子资源时设 `ownerReference`（controller=true），子资源变化自动通过 `EnqueueRequestForOwner` 触发父 Reconcile，同时子资源随父删除而 GC；
* **Finalizer**：CR 删除前执行清理。流程：设置 `deletionTimestamp` → 控制器发现 finalizer 还在 → 执行清理 → 移除 finalizer → GC 真正删除对象；
* **Predicate + GenerationChangedPredicate**：避免 status 更新触发新一轮 Reconcile 的无限循环。

### Mapping to client-go

```text
controller-runtime          client-go
─────────────────────────────────────────────
Manager                     生命周期 + shared components
Cache                       SharedInformer + Indexer
Client                      RESTClient / dynamic client
Controller                  Informer + Workqueue + workers
Reconciler                  syncHandler
Builder                     声明式 Watch 注册
```

---

## Case

### AIBrix: LLM Inference Operator

[AIBrix](https://github.com/vllm-project/aibrix) 是 vLLM 社区开源的 LLM 推理基础设施项目，用 Kubebuilder v4 构建，包含 10 个 CRD 和对应控制器。拿它的 **ModelClaim** 控制器来串一遍上面的概念。

**场景**：ModelClaim 声明"这个模型应该被下载并激活到某个 GPU Pod 上"。

```go
type ModelClaimSpec struct {
    ModelName    *string                      // 模型标识
    PodSelector  *metav1.LabelSelector         // 选哪个 GPU Pod 池
    ArtifactURL  string                        // 模型权重地址（s3://, gcs://, huggingface://）
    Engine       string                        // vllm / sglang
    Replicas     *int32                        // 引擎进程数
    EngineConfig *ModelClaimEngineConfig       // 引擎启动参数
}
```

**Reconciler 结构**——内嵌 `client.Client`，直接调用 `r.Get`、`r.Update`，这是 controller-runtime 的标准注入风格：

```go
type ModelClaimReconciler struct {
    client.Client
    Scheme        *runtime.Scheme
    Recorder      record.EventRecorder
    Runtime       RuntimeClient            // 驱动 sidecar 做 activate/deactivate
    Locality      LocalityProvider         // 节点亲和打分
    SnapshotCache *runtimeSnapshotCache    // 运行时观测缓存
}
```

**注册流程**——`For` 主资源 + `Owns` 子资源 + Predicate 过滤：

```go
func Add(mgr manager.Manager, _ config.RuntimeConfig) error {
    r := &ModelClaimReconciler{
        Client:   mgr.GetClient(),
        Scheme:   mgr.GetScheme(),
        Recorder: mgr.GetEventRecorderFor(controllerName),
        Runtime:  NewRuntimeClient(),
        Locality: uniformLocality{},
        SnapshotCache: newRuntimeSnapshotCache(
            defaultRuntimeSnapshotTTL, time.Now,
        ),
    }

    return ctrl.NewControllerManagedBy(mgr).
        Named(controllerName).
        For(&modelv1alpha1.ModelClaim{},
            builder.WithPredicates(predicate.Or(
                predicate.GenerationChangedPredicate{},
                predicate.AnnotationChangedPredicate{},
            ))).
        Owns(&corev1.Pod{}).
        Complete(r)
}
```

**状态机**：`Pending → Scheduling → Loading → Activating → Active`。Reconcile 每次读到当前 phase，做对应的驱动动作。这是 level-triggered 的典型体现——不关心上一个事件是什么，只看当前状态该做什么。

---

### KServe: ML Inference Platform

[KServe](https://github.com/kserve/kserve) 是 Kubeflow 生态的模型推理平台，核心 CRD 是 **InferenceService**。它的架构展示了一个更复杂的 Operator 设计模式——**主控制器委托给子 Reconciler**。

**CRD 结构**：InferenceService 的 spec 分三个组件：

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
spec:
  predictor:      # 核心推理组件（必填）
    model:
      modelFormat: sklearn        # 或 pytorch, huggingface, vllm
      storageUri: s3://my-bucket/model.pkl
      runtime: my-runtime         # 可选，指定 ServingRuntime
  transformer:    # 可选：预处理/后处理
  explainer:      # 可选：模型解释（LIME / SHAP）
```

**Reconciler 分层——从 CR 到真实资源**：

```text
InferenceServiceReconciler
    ├─ ComponentReconciler (Predictor / Transformer / Explainer)
    │     └─ 组装 PodSpec + 选择 ServingRuntime
    ├─ WorkloadReconciler
    │     ├─ KsvcReconciler  (Serverless 模式 → Knative Service)
    │     └─ DeploymentReconciler (Standard 模式 → Deployment)
    └─ IngressReconciler
          └─ Istio VirtualService / Gateway API HTTPRoute
```

这三层解耦很有代表性：

| 层次 | 职责 | 你写的代码 |
|------|------|-----------|
| **Component** | 把 InferenceService 的规格翻译为每个组件的 PodSpec | 匹配 runtime、注入存储初始化 sidecar |
| **Workload** | 把 PodSpec 落地为 K8s 资源（Knative Service 或 Deployment） | 调用 K8s API 创建/更新 |
| **Ingress** | 把流量路由规则写入网络层 | 写 Istio 或 Gateway API |

每个子 Reconciler 都可以独立测试，`mainReconciler.Reconcile()` 只负责协调它们的执行顺序和状态聚合。

**ServingRuntime 抽象**：KServe 用 `ServingRuntime` CRD 把框架配置从控制器逻辑里解耦出来。控制器不硬编码各个框架的 container 配置，而是根据 `modelFormat` 自动匹配已有的 ServingRuntime：

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
spec:
  supportedModelFormats:
    - name: sklearn
      version: "1"
      autoSelect: true
  containers:
    - image: kserve/sklearnserver:v0.14
      args:
        - --model_dir=/mnt/models
```

这意味着加一个新框架只需要 apply 一个新的 ServingRuntime CR，不需要改控制器代码——这是 CRD + controller-runtime 的另一种灵活用法：**用 CRD 描述框架配置，用控制器编排调度逻辑**。

**三种部署模式对比**——RawDeployment vs Serverless vs ModelMesh，对应不同的 `WorkloadReconciler` 实现：

| 模式 | 底层资源 | 缩到零 | 金丝雀发布 | 适用场景 |
|------|---------|--------|-----------|---------|
| RawDeployment | Deployment + HPA | ❌ | ❌ | 常驻高吞吐 |
| Serverless | Knative Service + KPA | ✅ | ✅ | 动态负载、节省资源 |
| ModelMesh | 共享模型服务器 | 有条件 | ❌ | 多模型高密度部署 |

模式选择是控制器层面的策略——Reconciler 根据 annotation 决定走 `KsvcReconciler` 还是 `DeploymentReconciler`，体现了 controller-runtime 的灵活性：底层还是 `For + Owns + Watches`，但业务编排可以做任意复杂。

---

## When to Use What

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| 快速原型 / 教学示例 | client-go 手写 | 理解底层原理 |
| 生产级 Operator | controller-runtime + Kubebuilder | 成熟框架、社区标准 |
| 简单 CRD + 无控制器 | 纯 CRD + Webhook | 不需要 Reconcile 逻辑时 |
| 复杂编排（多 CRD 联动） | controller-runtime + 单独 Manager | 可维护性更强 |

**适合用 Operator 的场景**：

* 需要自定义扩缩容策略（如 LLM 推理的 Request-based autoscaler）；
* 管理有状态分布式系统（如 RayCluster、MySQL）；
* 需要把外部资源（云存储、GPU 设备）映射为 Kubernetes API；
* 需要编排带依赖的多步工作流。

**不太适合的场景**：

* 纯粹配置管理——ConfigMap + Deployment 能解决的别上 Operator；
* 脚本级运维任务——Operator 的生产成本和维护开销不大。

---

## References

* [kubernetes/sample-controller](https://github.com/kubernetes/sample-controller/blob/master/docs/controller-client-go.md) — 官方 client-go 控制器示例
* [client-go tools/cache](https://pkg.go.dev/k8s.io/client-go/tools/cache) — Reflector、DeltaFIFO、Indexer API
* [client-go util/workqueue](https://pkg.go.dev/k8s.io/client-go/util/workqueue) — Workqueue 与限速器
* [controller-runtime 源码分析](https://blog.huweihuang.com/k8s-source-code-analysis/kube-controller-manager/controller-runtime/) — Builder 与 Controller 启动流程
* [AIBrix: vLLM 推理基础设施](https://github.com/vllm-project/aibrix) — 文中实战参考的 LLM 推理 Operator
* [KServe: 模型推理平台](https://github.com/kserve/kserve) — 文中实战参考的模型推理 Operator
* [Kubebuilder 官方文档](https://book.kubebuilder.io/) — 脚手架使用指南
