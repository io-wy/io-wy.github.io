---
title: 'Agent Parallel'
description: 'agent parallel then llm'
pubDate: '2026-03-28'
heroImage: '/img/12.png'
pinned: true
tags:
  - agent
  - llm
---

# Agent 平行



持续更新 ing......

先解释标题，我们简单称呼 agent 平行面为与 LLM 发展同向的内容，很多人也称呼为 **agent 补偿性工程**（例如 skill，mcp之类的工作）；于此相对的是 agent 垂直面，便是 LLM 发展的过程中无法触及的内容，算是 **agent 系统性工程**，（agent harness大多也是这个意思）（例如 状态管理 fallback机制）



## skill



此处探讨更多的是skill召回率的问题，没有 embedding / vector search，也没有 BM25 / TF-IDf 关键词匹配；完全依赖于 LLM 从 system prompt 的 skill 列表中 ”看到并理解“



暂时不讨论 tricks 相关的问题，skill的实现大多是从 sessionstart hook 讲 skill 描述列表注入 system prompt，之后由 LLM 根据 description + triggers 字段 自主匹配； 匹配成功后 resolveSkillPath（）解析路径，读取skill.md和reference文件夹的内容



召回率的核心瓶颈只有 description 字段，这里就简单介绍 tricks

* 同义词，中英触发词
* 添加 triggers 字段，参考 code-review 的 metadata.triggers
* 提高准确率，类似对比学习的，提供反例
* 保持在400词左右，过短：召回率低，过长：prompt负担大

这里是code-review的description部分



```markdown
name: code-reviewer
description: Analyzes code diffs and files to identify bugs, security vulnerabilities (SQL injection, XSS, insecure deserialization), code smells, N+1 queries, naming issues, and architectural concerns, then produces a structured review report with prioritized, actionable feedback. Use when reviewing pull requests, conducting code quality audits, identifying refactoring opportunities, or checking for security issues. Invoke for PR reviews, code quality checks, refactoring suggestions, review code, code quality. Complements specialized skills (security-reviewer, test-master) by providing broad-scope review across correctness, performance, maintainability, and test coverage in a single pass.
license: MIT
allowed-tools: Read, Grep, Glob
metadata:
  author: https://github.com/Jeffallan
  version: "1.1.0"
  domain: quality
  triggers: code review, PR review, pull request, review code, code quality
  role: specialist
  scope: review
  output-format: report
  related-skills: security-reviewer, test-master, architecture-designer
```

加点中文出发词，精简一下效果更好



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



## tool



mcp vs cli 会在此处提到





