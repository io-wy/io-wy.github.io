---
title: 'OpenCode Overview'
description: 'Opensource Coding-Agent'
pubDate: '2026-02-13'
heroImage: '/img/19.png'
tags:
  - tools
  - ai
---

# OpenCode Overview

> 详细内容打算放在知乎上

---

## 第一部分：项目架构概览

### 1.1 项目定位与技术栈

**OpenCode** 是一个 AI 驱动的开发工具，定位为 Anthropic 官方的 CLI 工具。从源码分析来看，这是一个**高度工程化的 Agent 框架**，而不仅仅是简单的 AI 工具。

#### 核心技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| **Bun** | 1.3.8 | 高性能 JavaScript 运行时 |
| **TypeScript** | 5.8.2 | 强类型保证 |
| **Turbo** | 2.5.6 | Monorepo 构建系统 |
| **Vercel AI SDK** | 5.0.124 | 多模型统一接口 |
| **Solid.js** | 1.9.10 | 响应式 UI 框架 |
| **Tauri** | - | 跨平台桌面应用 |
| **Drizzle ORM** | - | 类型安全的数据库 ORM |
| **Hono** | 4.10.7 | 轻量级 Web 框架 |

#### 技术选型亮点

1. **Bun 作为运行时**：相比 Node.js，Bun 提供更快的启动速度和更好的性能
2. **Vercel AI SDK**：统一的多模型接口，支持 20+ AI 提供商
3. **Solid.js**：比 React 更轻量，性能更好的响应式框架
4. **Turbo**：高效的 Monorepo 构建工具，支持增量构建和缓存

### 1.2 Monorepo 架构设计

OpenCode 采用了**精心设计的 Monorepo 结构**，体现了模块化和关注点分离的原则：

```
opencode/
├── packages/
│   ├── opencode/          # 核心 CLI（主包）- 18,000+ 行代码
│   │   ├── src/
│   │   │   ├── agent/     # Agent 系统
│   │   │   ├── tool/      # Tool 系统（24个核心工具）
│   │   │   ├── skill/     # Skill 系统
│   │   │   ├── permission/# 权限系统
│   │   │   ├── provider/  # AI Provider 抽象层
│   │   │   ├── session/   # 会话管理
│   │   │   ├── mcp/       # Model Context Protocol
│   │   │   ├── cli/       # CLI 命令（18个命令）
│   │   │   └── ...
│   │   └── package.json
│   │
│   ├── console/           # 控制台服务
│   │   ├── app/          # 控制台前端
│   │   ├── core/         # 核心逻辑和数据库
│   │   ├── function/     # 函数处理
│   │   ├── mail/         # 邮件服务
│   │   └── resource/     # 资源管理
│   │
│   ├── desktop/          # Tauri 桌面应用
│   ├── web/              # Web 服务
│   ├── sdk/              # JavaScript SDK
│   ├── ui/               # Solid.js 组件库
│   ├── util/             # 共享工具库
│   ├── plugin/           # 插件系统
│   └── enterprise/       # 企业版功能
│
├── .opencode/            # Agent、Skill、Tool 配置
│   ├── agent/           # Agent 定义文件
│   ├── skill/           # Skill 定义文件
│   ├── tool/            # Tool 实现文件
│   └── opencode.jsonc   # 主配置文件
│
└── AGENTS.md            # Agent 开发规范
```

#### 架构设计亮点

1. **清晰的边界划分**
   - CLI、Web、Desktop 各自独立但共享核心逻辑
   - 每个包都有明确的职责和边界

2. **SDK 优先设计**
   - 提供 `@opencode-ai/sdk` 供外部集成
   - 核心功能通过 SDK 暴露，CLI 只是一个客户端

3. **插件化架构**
   - 支持通过插件扩展功能
   - 插件可以提供自定义 Tool、Auth 等

4. **配置驱动**
   - `.opencode/` 目录存放所有配置
   - 支持项目级和全局级配置

### 1.3 核心模块概览

#### 1.3.1 Agent 系统（`packages/opencode/src/agent/`）

- **职责**：管理不同类型的 AI Agent
- **核心文件**：`agent.ts`（339 行）
- **内置 Agent**：7 种（build、plan、general、explore、compaction、title、summary）
- **特性**：
  - 声明式配置
  - 权限系统集成
  - 动态 Agent 生成

#### 1.3.2 Tool 系统（`packages/opencode/src/tool/`）

- **职责**：提供 Agent 可调用的工具
- **核心文件**：`tool.ts`（90 行）、`registry.ts`（161 行）
- **工具数量**：24 个核心工具 + 自定义工具
- **特性**：
  - 类型安全的参数验证
  - 自动输出截断
  - 插件化扩展

#### 1.3.3 权限系统（`packages/opencode/src/permission/`）

- **职责**：控制 Agent 的操作权限
- **核心文件**：`next.ts`（281 行）
- **特性**：
  - 规则引擎
  - 通配符匹配
  - 三种动作：allow、deny、ask

#### 1.3.4 Provider 系统（`packages/opencode/src/provider/`）

- **职责**：统一多个 AI 提供商的接口
- **核心文件**：`provider.ts`（1263 行）
- **支持的提供商**：20+ 个（OpenAI、Anthropic、Google、AWS、Azure 等）
- **特性**：
  - 统一抽象层
  - 自动模型发现
  - 成本计算

#### 1.3.5 Skill 系统（`packages/opencode/src/skill/`）

- **职责**：可复用的 Agent 技能
- **核心文件**：`skill.ts`（189 行）
- **特性**：
  - Markdown 格式定义
  - 多级目录扫描
  - URL 远程加载

#### 1.3.6 会话管理（`packages/opencode/src/session/`）

- **职责**：管理用户与 Agent 的对话会话
- **核心文件**：`message-v2.ts`（803 行）
- **特性**：
  - 消息持久化
  - 上下文压缩
  - 多模态支持

### 1.4 构建与开发流程

#### 开发命令

```bash
# 开发模式
bun run dev

# 桌面应用开发
bun run dev:desktop

# Web 应用开发
bun run dev:web

# 类型检查
bun run typecheck

# 构建所有包
bun run build
```

#### Turbo 任务配置

```json
{
  "tasks": {
    "typecheck": {
      "dependsOn": ["^typecheck"]
    },
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**"]
    },
    "test": {
      "dependsOn": ["^build"]
    }
  }
}
```

### 1.5 架构设计模式总结

OpenCode 采用了多种先进的架构设计模式：

1. **Monorepo 模式**：统一管理多个相关包
2. **插件化架构**：通过插件扩展功能
3. **适配器模式**：Provider 系统统一多个 AI 提供商
4. **策略模式**：Agent 系统支持不同的执行策略
5. **责任链模式**：权限系统的规则链
6. **工厂模式**：Tool 和 Agent 的动态创建
7. **观察者模式**：事件总线系统


