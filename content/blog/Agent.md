---
title: 'Agent'
description: 'just agent'
pubDate: '2026-03-28'
heroImage: '/img/12.png'
pinned: true
tags:
  - agent
  - llm
---

# Agent 



持续更新 ing......



## agent tool

以 Claude Code的 BashTool 和 FileWriteTool 

**FileWriteTool**

* 全量写入：Model 发送的 `content` 中的换行符是按原文写入的（LF硬编码）；
* 强制先读后写： 模型不能写入从未读取过的文件
* Staleness 检测（并发安全）
* Sidecar 操作： Skill自动发现；LSP通知；文件历史追踪；git diff计算

**BashTool**

**权限系统**

* tree-sitter提供准确AST
* 环境变量和wrapper剥离
* 规则匹配系统：Exact匹配；Prefix匹配；wildcard（通配符匹配）
* 安全分类器：Prompt 级别的规则（在 settings.json 中配置）； Deny/Ask 分类器：并行检查（Haiku 调用）
* 其他安全检查：Path Constraints：路径约束检查；命令注入检查；Read-only 检测

**执行机制**

shell 输出
  → Claude Code Hints 提取（<claude-code-hint /> 标签 → 业务逻辑, 从 stdout 剥离）
  → Image 检测 + 尺寸裁剪
  → 大输出持久化（>64MB 截断）
  → 沙箱违规标注（SandboxManager.annotateStderrWithSandboxFailures）
  → 语义退出码解释（interpretCommandResult）
  → 空行清理
  → 返回 Out 对象

### build tool

**多层防御**

- 每层安全检测应该独立且正交（相互覆盖不同的攻击面）
- 落盘的数据要有校验（WriteTool 的 mtime + content 双重校验）

**严格“校验->权限->执行“**

- 三阶段必须严格分离，不能混合
- validateInput 失败 → 不走 permission（节省用户确认时间）
- checkPermissions 失败 → 不走 call（防止越权）
- 每个阶段的错误返回不同的 errorCode（便于前端区分处理）

**异步进度报告**

使用 AsyncGenerator 实现进度流

**安全env var 白名单**





## agent 推理



agent is loop ( LLM教派 )

agent 推理可以简单理解为 基于目前情况进行下一步决策

因此和 LLM 推理不同的是，agent 推理拆成了更细粒度的内容，但这句话可能有些偏颇，一定程度上，我还是认为 LLM 发展过程中，agent推理是自然而然地可以被学会，因此在这里就简单的谈了相关内容



state =  用户目标 + 当前上下文（内部记忆）+ 外部记忆 + 工具返回结果（应该放在 context 比较舒服）+ 系统约束

agent 推理的第一步不是“想”，而是“把什么信息视作当前世界状态”



解释几个名词

**reasoning**

模型在当前上下文里做中间推断，这里又能引出一个新的名词**CoT**（chain of thought），把部分中间推断用文本写出来的 prompting 方式，因此 CoT 其实就是一种外显的 reasoning

* agent 可以没有 显示reasoning（CoT），但依然还是可以做复杂决策
* agent reasoning 是在环境里反复 观察 决定 执行 修正（应该不是什么重要的概念）



**ReAct**

Thought -> Action -> Observation -> Thought -> Action -> Observation ...

把两件事结合起来了，reasoning 和 acting，明显局限就是

* 只有一条 greedy 轨迹
* 控制流在 prompt 里面，审计困难（prompt注入容易打）
* 一旦 observation 带偏 bug多



**ToT/LATS**

 因此很多人认为 agent 应该是 搜索，而不是 单链思考

Tree of Thoughts （ToT）:  把“thought”当成树节点，做分支、评估、剪枝、回溯 （langgraph？bushi）

Tree Search for LM Agents（LATS）：不只在文本 thought 上搜索，而是在真实环境轨迹上做 tree search

非常 trivial 的逻辑，搜索：

生成多个候选 -> 评估各个候选 -> 先拓展有价值的branch -> 不行就回退 -> 重复直到找到高质量路径



## Agent Build

**工具层**

* 原子性
* 参数校验，授权，执行
* 资源管理：超时，中断，后台
* 异步报告
* 工具副作用

**安全层**

* 多层防御深度
* 最外层输出归一化
* 黑白名单
* 协作并发安全

**Agent推理**

* 工具选择：决策规则，搜索用glob/grep；读文件用read；独立判断用subagent
* 任务分解：意图识别，并行/串行，依赖关系
* 中断回复：fallback 机制

**SubAgent机制**

* context隔离
* 写文件，传摘要
* 后台执行/并行/队间通信

**上下文管理**

参考之前的文章

**Memory系统**

持久化，跨会话记忆

- 用户偏好、项目上下文、反馈历史
- 分层级：会话内（transient）→ 项目级（project memory）→ 团队级（team memory）

**人类接口**

权限

- **Deny > Ask > Allow** 的优先级顺序
- **自动分类器**在后台运行（高置信度 allow 时自动批准，用户无感知）
- **权限规则持久化**（用户批准一次后不再问）
- **权限规则类型**：allow / ask / deny，每种可配置 exact / prefix / wildcard 匹配

进度可视化

- 已经做了什么（进度条、实时输出）
- 还要多久（超时时间、已完成步骤）

工具异步报告

结果透明度

- FileWriteTool 显示 diff patch（新增/删除/修改的行数）
- BashTool 显示 stdout + stderr
- 子代理完成时返回摘要

反馈闭环

- 逐条评估的反馈文本框
- 自动保存（每 800ms）
- Submit All Reviews 一键汇总
- Previous feedback 对比

副作用提示

- BashTool 的 `--simulatedSedEdit` 在用户预览后才执行
- 破坏性操作在 UI 中有醒目标识
- 后台执行的任务在完成后通知

调式能力

- 完整 transcript 记录
- 工具调用链可追溯
- 成本/耗时/Token 消耗透明
- 可复现的 seed 固定

## Skill

**召回率 skill质量评估**



**典型的Skill目录结构**

```
skill-name/
├── SKILL.md           # 必需 — 触发元数据 + 核心工作流
├── references/        # 可选 — 参考材料，按需加载
├── examples/          # 可选 — 可直接运行的代码示例
└── scripts/           # 可选 — 可执行的工具脚本
```

**触发模式**

| 模式            | 信号形式                                                | 适用场景                             |
| --------------- | ------------------------------------------------------- | ------------------------------------ |
| **Reactive**    | "This skill should be used when user says/asks..."      | 用户明确请求时触发（占 80%+）        |
| **Enforcement** | "You MUST use this before [class of work]..."           | 在某类工作前强制 Claude 执行检查清单 |
| **Proactive**   | "Automatically triggers when [observable condition]..." | 基于上下文自动触发                   |

### **skill-creator**

比较重要的点：**意图捕获 运行测试 描述优化**

**Phase 0：意图捕获**

- 这个 Skill 要让 Claude 做什么？
- 什么时候触发？（用户的什么话/上下文）
- 输出格式是什么？
- 需要测试用例吗？

Phase 1：调研与设计

- 主动追问边缘情况、输入/输出格式、示例文件、成功标准、依赖项
- 可用 MCP 做并行搜索（查文档、找类似 Skill、搜最佳实践）

Phase 2：编写 SKILL.md

description 要写得"pushy"——默认倾向是 undertrigger（Claude 偏向不触发 Skill），所以 description 必须主动推一把

- **祈使句** — "Use this skill when..." 而非 "This skill does..."
- **解释 why** — 告诉 Claude 为什么某件事重要，LLM 的理解能力远超字面遵循
- **ALWAYS/NEVER 是黄旗** — 如果发现自己在用全大写约束，停下来转成解释原因
- **包含示例** — Input/Output 模式让 Claude 快速理解预期产出
- **引用文件时说明加载时机** — 不只是列文件名，要写"Load when..



**Phase 3：运行测试**（claude -p 子进程）

**Step 1：并行 Spawn 子代理**

每个测试用例 spawn 两个子代理（**必须在同一轮**）：

- With-skill 运行 → 给子代理 Skill 路径 + 测试 prompt
- Baseline 运行 → 创建新 Skill 时无 Skill；改进旧 Skill 时用快照

目录结构：

```
workspace/iteration-N/eval-ID-descriptive-name/
├── with_skill/outputs/
└── without_skill/outputs/
```

**Step 2：运行期间起草断言**

等待不是选项——利用这段时间写断言。好的断言是**客观可验证且有描述性名称**的。主观 Skill（写作风格、设计质量）不走断言，走定性评估。

**Step 3：捕获 Timing 数据**

子代理任务完成时通知包含 `total_tokens` 和 `duration_ms`。**立即保存**——这是唯一的机会。

**Step 4：评分 + 聚合 + 分析 + 审查**

- **评分**：grader 子代理逐条评估断言（通过/失败 + 证据引用）。字段名必须是 `text`/`passed`/`evidence`——viewer 硬依赖这些名字
- **聚合**：`aggregate_benchmark.py` 产出 pass_rate/time/tokens 的 mean ± stddev + delta
- **分析师 pass**：找聚合指标隐藏的模式——总是通过的断言（非区分性）、高方差的 eval（可能 flaky）
- **启动审查器**：`generate_review.py` 启动 HTTP server + SPA viewer（端口 3117）

Viewer 有两个标签页：

- **Outputs**：逐条显示 prompt + 输出文件 + 评分 + 反馈框
- **Benchmark**：统计汇总 + per-eval 详细结果

用户在每个 eval 的文本框中写反馈，全部完成后点 "Submit All Reviews"。

**Step 5：读反馈**

空反馈 = 用户认为没问题。关注有具体抱怨的 eval。改进完成记得 kill viewer 进程。



**Phase 4：改进循环**

**四条改进原则**：

1. **从反馈中归纳** — 别 overfit 到几个测试用例。Skill 会被用百万次
2. **保持 Prompt 精简** — 读 transcript（不是只看最终输出），去掉不创造价值的部分
3. **解释 Why** — LLM 理解原理后能超越字面指令
4. **提取重复工作为脚本** — 如果三个测试用例的子代理都写了 `create_docx.py` → 打包到 `scripts/`

迭代循环：改进 → 重新跑测试 → 启动 reviewer（`--previous-workspace`）→ 等反馈 → 再改

停止条件：用户满意 / 所有反馈为空 / 没有有意义的进展。

**Phase 5：盲比较（高级，可选）**

当用户问"新版本真的更好吗？"时启用。使用两个独立子代理：

- **Comparator**：盲看两个输出 A/B，不知对应哪个版本。建立评分 rubrics（content 维度 + structure 维度各 1-5），计算 1-10 总分
- **Analyzer**：揭示赢家输家，分析原因，输出改进建议（优先级 + 类别 + 预期影响）

大多数用户不需要这一步——人工审查循环通常足够了。

**Phase 6：描述优化（Trigger 精度优化）**

当 Skill 功能已定但触发不准时启用。这是 Trigger 层面的工程优化，独立于内容改进。

**Step 1 — 生成 20 个触发评估查询**

8-10 条 should-trigger + 8-10 条 should-not-trigger。

关键：**查询必须真实**。不是 "Format this data" 而是：

```
"ok so my boss just sent me this xlsx file (its in my downloads, called something
like 'Q4 sales final FINAL v2.xlsx') and she wants me to add a column..."
```

反例要有挑战性。"写个斐波那契函数" 作为 PDF Skill 的反例太容易了，不考验任何东西。

**Step 2 — 用户审查**

通过 `eval_review.html` 模板在浏览器中让用户编辑、开关、增删。

**Step 3 — 运行优化循环**

`run_loop.py` 执行完整的优化过程：

1. 分层 60/40 train/test 分割（seed=42 固定）
2. 评估当前描述（每条跑 3 次，多数投票）
3. 调用 `improve_description.py` 生成新描述
4. 用新描述重新评估全部
5. 最多 5 次迭代
6. 按 **test_passed** 选最优（防止过拟合）
7. 输出 HTML 报告

**Step 4 — 应用**

将 `best_description` 写入 SKILL.md frontmatter。

**Phase 7：打包**

```bash
python -m scripts.package_skill <skill-path>
```

→ 产出 `.skill` 文件（ZIP），可分发安装。

### skill-analysis

参考skill-creator 额外增加了

recall评分；effectiveness评分；跨技能冲突检查；演化规则；

**Recall 评分细则**

格式检查（强制项）：

- 正确的信号形式（Reactive/Enforcement/Proactive）
- 无寄生第二人称（"you should" 混入 Reactive description）
- 不模糊（无 "provides guidance", "helps with"）
- 不冗余（不只复述 Skill 名字）

强制项失败 → **Blocker**，不通过。

量化分析（Reactive 模式）：

- 短语计数（4-8 个最理想）
- 词汇多样性（动词同义词、生命周期覆盖、场景触发）
- 领域锚定（专有名词）
- 自然语言对齐（用户口吻加分）

**Effectiveness 评分细则**

四维评分，各 0-25：

| 维度               | 评分方式                                                     |
| ------------------ | ------------------------------------------------------------ |
| **工作流可操作性** | 每步 0-2 分（0=模糊, 1=具体但不完整, 2=完整可行含工具+验证） |
| **渐进式披露质量** | 文件存在且被引用 +3，有"load when"条件 +4，有深度 +3         |
| **内容-触发对齐**  | description 承诺的每个场景 body 是否覆盖（主缺口 -15, 部分 -7） |
| **内容完整度**     | 7 元素各 0-2：目的/触发/工作流/资源/最佳实践/常见错误/验证   |

**跨技能冲突检查**

扫描同一目录下的 Skill，对比触发短语重叠：

```
Skill A: "create a hook", "add a hook", "debug a hook"
Skill B: "hook review", "check hooks", "debug hooks"
Overlap: "hook" + "debug"
```

解决策略：缩小实体、缩小动词、移动触发词到正确 Skill、合并 Skill、保持独立加 disambiguating context。

**演化**

1. **场景发现** — 通过对话绘制使用地图（Hit/Miss/False Positive/Blind Spot）
2. **运行诊断** — Phase 0-4 分析
3. **用户讨论** — P0 直接修复，P1 给选项，P2/P3 确认
4. **应用变更并验证** — 每步 quick verify + 全部完成后 full verify + user test
5. **记录演化日志** — before/after 分数、change set、user test 结果、下次 review 日期
