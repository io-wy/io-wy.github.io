---
title: typescript相对完备
date: 2026-02-14T10:54:27.000Z
tags: [ai]
category: 自用
comments: true
draft: false
---
> 基于TypeScript 5.x最新特性，结合OpenCode实战代码示例

## 目录

- [第一部分：TypeScript基础类型系统](#第一部分typescript基础类型系统)
- [第二部分：接口与类型别名](#第二部分接口与类型别名)
- [第三部分：函数类型](#第三部分函数类型)
- [第四部分：泛型系统](#第四部分泛型系统)
- [第五部分：高级类型](#第五部分高级类型)
- [第六部分：条件类型](#第六部分条件类型)
- [第七部分：映射类型](#第七部分映射类型)
- [第八部分：模板字面量类型](#第八部分模板字面量类型)
- [第九部分：类与装饰器](#第九部分类与装饰器)
- [第十部分：命名空间与模块](#第十部分命名空间与模块)
- [第十一部分：实战技巧](#第十一部分实战技巧)

---

## 前言

根据[GitHub 2025年度报告](https://www.programming-helper.com/tech/typescript-2026-number-one-github-ai-typed-languages-python)，TypeScript已经超越Python和JavaScript，成为GitHub上使用最多的编程语言。[TypeScript在2025年已从&#34;可选项&#34;变为&#34;必需品&#34;](https://jeffbruchado.com.br/en/blog/typescript-trends-2025-essential-modern-development)。

本教程将带你从零基础到能够编写OpenCode级别的复杂TypeScript代码。

---

## 第一部分：TypeScript基础类型系统

### 1.1 什么是TypeScript？

**TypeScript** 是JavaScript的超集（superset），为JavaScript添加了**静态类型检查**。

**核心概念：**

- **超集（Superset）**：所有JavaScript代码都是有效的TypeScript代码
- **静态类型检查（Static Type Checking）**：在编译时检查类型错误，而不是运行时
- **类型注解（Type Annotation）**：用冒号 `:` 显式声明变量的类型
- **类型推断（Type Inference）**：TypeScript自动推断变量类型
- **编译时（Compile-time）**：代码转换为JavaScript的阶段
- **运行时（Runtime）**：代码实际执行的阶段

**为什么使用TypeScript？**

1. **提前发现错误**：在编译时而非运行时发现bug
2. **更好的IDE支持**：自动补全、重构、跳转定义
3. **代码可维护性**：类型即文档，易于理解代码意图
4. **团队协作**：类型约束减少沟通成本

### 1.2 原始类型（Primitive Types）

TypeScript支持JavaScript的所有原始类型，并添加了类型注解。

#### 1.2.1 string - 字符串类型

```typescript
// 基本字符串
let name: string = "opencode"
let greeting: string = 'hello'

// 模板字符串（Template Literal）
let message: string = `Hello, ${name}!`

// 多行字符串
let multiline: string = `
  This is a
  multiline string
`
```

**关键术语：**

- **模板字符串（Template Literal）**：使用反引号 `` ` `` 和 `${}` 插入表达式

#### 1.2.2 number - 数字类型

```typescript
// TypeScript中所有数字都是浮点数
let age: number = 42
let price: number = 99.99

// 不同进制的数字
let hex: number = 0xf00d      // 十六进制
let binary: number = 0b1010   // 二进制
let octal: number = 0o744     // 八进制

// 特殊数值
let notANumber: number = NaN
let infinity: number = Infinity
```

**关键术语：**

- **浮点数（Floating-point）**：可以表示小数的数字类型
- **进制（Radix）**：数字的表示基数（2进制、8进制、10进制、16进制）

#### 1.2.3 boolean - 布尔类型

```typescript
let isActive: boolean = true
let isCompleted: boolean = false

// 布尔值通常来自比较运算
let isGreater: boolean = 10 > 5  // true
let isEqual: boolean = "a" === "b"  // false
```

#### 1.2.4 null 和 undefined

```typescript
let nothing: null = null
let notDefined: undefined = undefined

// 重要概念：strictNullChecks
// 当tsconfig.json中启用strictNullChecks时：
// null和undefined是独立类型，不能赋值给其他类型

let value: string = "hello"
// value = null  // 错误！（strictNullChecks开启时）
// value = undefined  // 错误！（strictNullChecks开启时）

// 如果需要允许null或undefined，使用联合类型
let nullable: string | null = "hello"
nullable = null  // 正确

let optional: string | undefined = "hello"
optional = undefined  // 正确
```

**关键术语：**

- **strictNullChecks**：TypeScript编译选项，启用后null和undefined不能赋值给其他类型
- **联合类型（Union Type）**：使用 `|` 表示值可以是多种类型之一

#### 1.2.5 symbol - 唯一标识符

```typescript
// Symbol创建唯一的标识符
let sym1: symbol = Symbol("key")
let sym2: symbol = Symbol("key")

// 即使描述相同，每个symbol都是唯一的
console.log(sym1 === sym2)  // false

// Symbol常用于对象的唯一属性键
const uniqueKey = Symbol("id")
let obj = {
  [uniqueKey]: 12345
}
```

**关键术语：**

- **Symbol**：ES6引入的原始类型，用于创建唯一标识符

#### 1.2.6 bigint - 大整数

```typescript
// bigint用于表示超过Number.MAX_SAFE_INTEGER的整数
let big: bigint = 100n  // 注意：数字后面加n
let huge: bigint = BigInt("9007199254740991")

// Number.MAX_SAFE_INTEGER = 9007199254740991
// 超过这个值的整数运算可能不精确

// bigint和number不能混合运算
let num: number = 100
let bigNum: bigint = 100n
// let result = num + bigNum  // 错误！
```

**关键术语：**

- **MAX_SAFE_INTEGER**：JavaScript中可以安全表示的最大整数
- **bigint字面量**：数字后加 `n` 表示bigint类型

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/session/index.ts
export namespace Session {
  export const Info = z.object({
    id: Identifier.schema("session"),  // string类型
    slug: z.string(),
    projectID: z.string(),
    time: z.object({
      created: z.number(),    // 时间戳用number
      updated: z.number(),
      archived: z.number().optional(),
    }),
  })
}
```

### 1.3 数组类型（Array Types）

数组是存储多个相同类型值的集合。

#### 1.3.1 基本数组声明

```typescript
// 方式1：类型[]（推荐）
let numbers: number[] = [1, 2, 3, 4, 5]
let strings: string[] = ["a", "b", "c"]
let booleans: boolean[] = [true, false, true]

// 方式2：Array<类型>（泛型写法）
let numbers2: Array<number> = [1, 2, 3]
let strings2: Array<string> = ["a", "b", "c"]

// 两种写法完全等价，推荐使用第一种（更简洁）
```

**关键术语：**

- **泛型（Generic）**：使用 `<>` 表示的类型参数，后面会详细讲解

#### 1.3.2 只读数组（Readonly Array）

```typescript
// readonly修饰符防止数组被修改
let readonlyArr: readonly number[] = [1, 2, 3]
let readonlyArr2: ReadonlyArray<number> = [1, 2, 3]

// 不能修改只读数组
// readonlyArr.push(4)     // 错误！
// readonlyArr[0] = 10     // 错误！
// readonlyArr.pop()       // 错误！

// 但可以读取
console.log(readonlyArr[0])  // 正确
console.log(readonlyArr.length)  // 正确

// 可以通过解构创建新数组
let newArr = [...readonlyArr, 4]  // 正确
```

**关键术语：**

- **readonly修饰符**：防止属性或数组被修改
- **不可变性（Immutability）**：数据一旦创建就不能修改

#### 1.3.3 多维数组

```typescript
// 二维数组
let matrix: number[][] = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]

// 三维数组
let cube: number[][][] = [
  [[1, 2], [3, 4]],
  [[5, 6], [7, 8]]
]

// 访问元素
let element = matrix[0][1]  // 2
```

#### 1.3.4 联合类型数组

```typescript
// 数组元素可以是多种类型
let mixed: (string | number)[] = [1, "two", 3, "four"]

// 注意括号的位置
let arr1: string | number[]  // string 或 number数组
let arr2: (string | number)[]  // (string或number)的数组

// 复杂的联合类型数组
type Status = "pending" | "completed"
let statuses: Status[] = ["pending", "completed"]
```

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/session/message-v2.ts
export namespace MessageV2 {
  // parts是一个Part类型的数组
  export interface WithParts {
    parts: Part[]  // Part[]表示Part类型的数组
  }

  // 过滤压缩的消息
  export async function filterCompacted(
    messages: WithParts[]  // WithParts类型的数组
  ): Promise<WithParts[]> {
    return messages.filter(m => !m.parts.some(p => p.type === "compaction"))
  }
}
```

### 1.4 元组类型（Tuple Types）

**元组（Tuple）** 是固定长度和类型的数组，每个位置的类型都是确定的。

#### 1.4.1 基本元组

```typescript
// 元组：[类型1, 类型2, ...]
let tuple: [string, number] = ["age", 42]

// 访问元素时，TypeScript知道每个位置的类型
let name: string = tuple[0]  // string
let age: number = tuple[1]   // number

// 错误的赋值
// tuple = [42, "age"]  // 错误！类型顺序不对
// tuple = ["age"]      // 错误！长度不对
```

**关键术语：**

- **元组（Tuple）**：固定长度和类型的数组
- **位置类型（Positional Type）**：每个位置有确定的类型

#### 1.4.2 可选元素

```typescript
// 使用 ? 表示可选元素
let tuple1: [string, number?] = ["name"]  // 第二个元素可选
let tuple2: [string, number?] = ["name", 30]

// 可选元素必须在必需元素之后
// let invalid: [string?, number] = [undefined, 42]  // 错误！
```

#### 1.4.3 剩余元素（Rest Elements）

```typescript
// 使用 ...类型[] 表示剩余元素
let tuple1: [string, ...number[]] = ["items", 1, 2, 3, 4, 5]
let tuple2: [string, ...number[]] = ["items"]  // 剩余元素可以为空

// 剩余元素必须在最后
// let invalid: [...number[], string] = [1, 2, "end"]  // 错误！

// 复杂的剩余元素
let tuple3: [boolean, ...string[], number] = [true, "a", "b", "c", 42]
```

#### 1.4.4 命名元组（Labeled Tuples）

TypeScript 4.0+ 支持给元组元素命名，提高可读性。

```typescript
// 命名元组：[名称: 类型, ...]
type Range = [start: number, end: number]
type Point = [x: number, y: number, z?: number]
type RGB = [red: number, green: number, blue: number]

let range: Range = [0, 100]
let point: Point = [10, 20]
let color: RGB = [255, 128, 0]

// 名称只是提示，不影响使用
let start = range[0]  // 仍然通过索引访问
```

**关键术语：**

- **命名元组（Labeled Tuple）**：为元组元素添加名称标签，提高代码可读性

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/session/message-v2.ts
export namespace MessageV2 {
  // 时间范围可以用元组表示
  export interface Part {
    time?: {
      start: number
      end?: number
    }
  }

  // 可以简化为命名元组
  type TimeRange = [start: number, end?: number]

  // 使用示例
  let executionTime: TimeRange = [Date.now()]
  let completedTime: TimeRange = [Date.now(), Date.now() + 1000]
}
```

### 1.5 对象类型（Object Types）

对象是TypeScript中最常用的复合类型。

#### 1.5.1 内联对象类型

```typescript
// 直接在变量声明中定义对象类型
let person: { name: string; age: number } = {
  name: "Alice",
  age: 30
}

// 多行写法（推荐）
let user: {
  name: string
  age: number
  email: string
} = {
  name: "Bob",
  age: 25,
  email: "bob@example.com"
}
```

#### 1.5.2 可选属性（Optional Properties）

```typescript
// 使用 ? 表示属性可选
let user: {
  name: string
  age?: number      // 可选属性
  email?: string
} = {
  name: "Bob"  // age和email可以不提供
}

// 访问可选属性时需要检查
if (user.age !== undefined) {
  console.log(user.age)
}

// 或使用可选链（Optional Chaining）
console.log(user.age?.toFixed(2))
```

**关键术语：**

- **可选属性（Optional Property）**：使用 `?` 标记，可以不提供
- **可选链（Optional Chaining）**：使用 `?.` 安全访问可能不存在的属性

#### 1.5.3 只读属性（Readonly Properties）

```typescript
// 使用 readonly 防止属性被修改
let config: {
  readonly apiKey: string
  timeout: number
} = {
  apiKey: "secret-key",
  timeout: 5000
}

// config.apiKey = "new-key"  // 错误！只读属性不能修改
config.timeout = 10000        // 正确

// readonly只在编译时检查，运行时仍可修改
// 这是TypeScript的类型系统限制
```

**关键术语：**

- **只读属性（Readonly Property）**：使用 `readonly` 标记，不能修改
- **编译时检查（Compile-time Check）**：只在TypeScript编译阶段检查

#### 1.5.4 索引签名（Index Signatures）

当不知道所有属性名时，使用索引签名。

```typescript
// 基本索引签名
let scores: { [studentName: string]: number } = {
  "Alice": 95,
  "Bob": 87,
  "Charlie": 92
}

// 可以添加任意string类型的键
scores["David"] = 88
scores["Eve"] = 91

// 数字索引签名
let items: { [index: number]: string } = {
  0: "first",
  1: "second",
  2: "third"
}

// 混合使用固定属性和索引签名
let config: {
  name: string           // 固定属性
  [key: string]: any    // 索引签名
} = {
  name: "MyConfig",
  timeout: 5000,
  retries: 3
}
```

**关键术语：**

- **索引签名（Index Signature）**：使用 `[key: type]: valueType` 定义动态属性
- **动态属性（Dynamic Property）**：属性名在编译时未知

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/session/index.ts
export namespace Session {
  export const Info = z.object({
    id: Identifier.schema("session"),
    slug: z.string(),
    projectID: z.string(),
    directory: z.string(),
    parentID: Identifier.schema("session").optional(),  // 可选
    time: z.object({
      created: z.number(),
      updated: z.number(),
      compacting: z.number().optional(),  // 可选
      archived: z.number().optional(),    // 可选
    }),
  })
}

// 来自 packages/opencode/src/tool/tool.ts
export namespace Tool {
  // 工具的元数据使用索引签名
  export type Metadata = {
    [key: string]: any
  }
}
```

---

## 第二部分：接口与类型别名

### 2.1 类型别名（Type Alias）

**类型别名（Type Alias）** 使用 `type` 关键字为类型创建新名称。

#### 2.1.1 基本类型别名

```typescript
// 为原始类型创建别名
type ID = string | number
type Status = "active" | "inactive" | "pending"
type Count = number

// 使用类型别名
let userId: ID = "user-123"
let accountId: ID = 12345
let status: Status = "active"
```

**关键术语：**

- **类型别名（Type Alias）**：使用 `type` 关键字为类型命名
- **联合类型（Union Type）**：使用 `|` 表示多个类型之一

#### 2.1.2 对象类型别名

```typescript
// 为对象类型创建别名
type Point = {
  x: number
  y: number
}

type User = {
  id: ID
  name: string
  status: Status
  createdAt: number
}

// 使用类型别名
let point: Point = { x: 10, y: 20 }
let user: User = {
  id: "123",
  name: "Alice",
  status: "active",
  createdAt: Date.now()
}
```

#### 2.1.3 函数类型别名

```typescript
// 为函数类型创建别名
type Handler = (input: string) => void
type Callback = (error: Error | null, result?: any) => void
type Predicate<T> = (value: T) => boolean

// 使用函数类型别名
const handleInput: Handler = (input) => {
  console.log(input)
}

const isPositive: Predicate<number> = (value) => {
  return value > 0
}
```

**关键术语：**

- **函数类型（Function Type）**：描述函数的参数和返回值类型
- **泛型类型别名（Generic Type Alias）**：带类型参数的类型别名

#### 2.1.4 泛型类型别名

```typescript
// 泛型类型别名
type Container<T> = {
  value: T
}

type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E }

// 使用泛型类型别名
let stringContainer: Container<string> = { value: "hello" }
let numberContainer: Container<number> = { value: 42 }

let successResult: Result<string> = {
  success: true,
  data: "Success!"
}

let errorResult: Result<string> = {
  success: false,
  error: new Error("Failed")
}
```

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/id/id.ts
export namespace Identifier {
  export type ID = string

  export function schema(prefix: string): z.ZodString {
    return z.string()
  }
}

// 来自 packages/opencode/src/session/index.ts
export namespace Session {
  // 类型别名简化复杂类型
  export type Info = z.output<typeof Info>
  export type ShareInfo = z.output<typeof ShareInfo>
}

// 来自 packages/opencode/src/provider/provider.ts
export namespace Provider {
  export type Model = {
    id: string
    providerID: string
    contextWindow: number
    maxOutput: number
    capabilities: {
      streaming: boolean
      tools: boolean
      vision: boolean
    }
  }
}
```

### 2.2 接口（Interface）

**接口（Interface）** 定义对象的结构，描述对象应该有哪些属性和方法。

#### 2.2.1 接口基础

```typescript
// 基本接口定义
interface User {
  id: string
  name: string
  email: string
}

// 使用接口
let user: User = {
  id: "123",
  name: "Alice",
  email: "alice@example.com"
}

// 接口可以描述任何对象结构
interface Point {
  x: number
  y: number
}

interface Rectangle {
  topLeft: Point
  bottomRight: Point
}
```

**关键术语：**

- **接口（Interface）**：使用 `interface` 关键字定义对象的形状
- **结构类型（Structural Typing）**：TypeScript基于结构而非名称进行类型检查

#### 2.2.2 可选属性和只读属性

```typescript
// 可选属性
interface Config {
  apiKey: string
  timeout?: number     // 可选
  retries?: number     // 可选
  debug?: boolean      // 可选
}

let config1: Config = { apiKey: "key" }  // 正确
let config2: Config = {
  apiKey: "key",
  timeout: 5000,
  debug: true
}  // 正确

// 只读属性
interface Point {
  readonly x: number
  readonly y: number
}

let point: Point = { x: 10, y: 20 }
// point.x = 30  // 错误！只读属性不能修改

// 混合使用
interface User {
  readonly id: string      // 只读
  name: string             // 可修改
  email?: string           // 可选
  readonly createdAt: number  // 只读
}
```

#### 2.2.3 函数类型接口

```typescript
// 接口可以描述函数类型
interface SearchFunc {
  (query: string, limit: number): Promise<string[]>
}

// 实现接口
const search: SearchFunc = async (query, limit) => {
  // 实现搜索逻辑
  return []
}

// 带属性的函数接口
interface Counter {
  (start: number): string  // 函数签名
  interval: number         // 属性
  reset(): void           // 方法
}

function createCounter(): Counter {
  const counter = ((start: number) => {
    return `Count: ${start}`
  }) as Counter

  counter.interval = 1000
  counter.reset = () => {
    console.log("Reset")
  }

  return counter
}
```

**关键术语：**

- **调用签名（Call Signature）**：描述函数如何被调用
- **混合类型（Hybrid Type）**：既是函数又有属性的类型

#### 2.2.4 索引签名

```typescript
// 字符串索引签名
interface StringMap {
  [key: string]: string
}

let map: StringMap = {
  name: "Alice",
  city: "NYC"
}

map["country"] = "USA"  // 可以添加任意string键

// 数字索引签名
interface NumberArray {
  [index: number]: string
}

let arr: NumberArray = ["a", "b", "c"]

// 混合固定属性和索引签名
interface Dictionary {
  length: number           // 固定属性
  [key: string]: any      // 索引签名
}

// 注意：固定属性的类型必须是索引签名类型的子类型
interface ValidDict {
  length: number
  [key: string]: number | string  // length的number类型是这个联合类型的子类型
}
```

**关键术语：**

- **索引签名（Index Signature）**：允许对象有任意数量的属性
- **字符串索引（String Index）**：使用字符串作为键
- **数字索引（Number Index）**：使用数字作为键

#### 2.2.5 接口继承

```typescript
// 单继承
interface Animal {
  name: string
  age: number
}

interface Dog extends Animal {
  breed: string
  bark(): void
}

let dog: Dog = {
  name: "Buddy",
  age: 3,
  breed: "Golden Retriever",
  bark() {
    console.log("Woof!")
  }
}

// 多重继承
interface Flyable {
  fly(): void
  altitude: number
}

interface Swimmable {
  swim(): void
  depth: number
}

interface Duck extends Flyable, Swimmable {
  quack(): void
}

let duck: Duck = {
  fly() { console.log("Flying") },
  altitude: 100,
  swim() { console.log("Swimming") },
  depth: 5,
  quack() { console.log("Quack!") }
}
```

**关键术语：**

- **继承（Inheritance）**：使用 `extends` 关键字继承接口
- **多重继承（Multiple Inheritance）**：一个接口可以继承多个接口

#### 2.2.6 接口合并（Declaration Merging）

TypeScript的一个独特特性：同名接口会自动合并。

```typescript
// 第一次声明
interface Window {
  title: string
}

// 第二次声明
interface Window {
  width: number
  height: number
}

// 自动合并为：
// interface Window {
//   title: string
//   width: number
//   height: number
// }

let window: Window = {
  title: "My Window",
  width: 800,
  height: 600
}

// 这个特性常用于扩展第三方库的类型
declare global {
  interface Window {
    myCustomProperty: string
  }
}
```

**关键术语：**

- **声明合并（Declaration Merging）**：同名声明自动合并
- **全局扩展（Global Augmentation）**：扩展全局类型

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/session/message-v2.ts
export namespace MessageV2 {
  // 基础消息接口
  export interface Info {
    id: string
    sessionID: string
    role: "user" | "assistant" | "system"
    time: {
      created: number
      updated: number
    }
  }

  // 用户消息接口继承基础接口
  export interface User extends Info {
    role: "user"
    variant?: string
    system?: string
    tools?: Record<string, boolean>
  }

  // 助手消息接口继承基础接口
  export interface Assistant extends Info {
    role: "assistant"
    agent: string
    modelID: string
    providerID: string
    finish?: string
  }

  // 带parts的消息接口
  export interface WithParts {
    parts: Part[]
  }
}
```

### 2.3 Type vs Interface

**何时使用Type，何时使用Interface？**

根据[TypeScript官方文档和社区最佳实践](https://www.rst.software/blog/advanced-typing-in-typescript-with-generics)：

```typescript
// ============ Interface的优势 ============

// 1. 声明合并（只有interface支持）
interface User {
  name: string
}

interface User {
  age: number
}
// 自动合并

// 2. 继承语法更清晰
interface Animal {
  name: string
}

interface Dog extends Animal {
  breed: string
}

// 3. 性能：interface在大型项目中编译更快

// ============ Type的优势 ============

// 1. 联合类型（interface不支持）
type Status = "pending" | "completed" | "failed"
type ID = string | number

// 2. 交叉类型
type Admin = User & { permissions: string[] }

// 3. 映射类型
type Readonly<T> = {
  readonly [P in keyof T]: T[P]
}

// 4. 条件类型
type NonNullable<T> = T extends null | undefined ? never : T

// 5. 元组类型
type Point = [number, number]

// 6. 原始类型别名
type Name = string
```

**推荐规则：**

1. **对象形状**：优先使用 `interface`
2. **联合类型、交叉类型**：使用 `type`
3. **需要声明合并**：使用 `interface`
4. **复杂类型操作**：使用 `type`
5. **公共API**：优先使用 `interface`（更容易扩展）

**OpenCode的实践：**

```typescript
// OpenCode混合使用两者

// 使用interface定义对象结构
export interface SessionInfo {
  id: string
  title: string
}

// 使用type定义联合类型和类型别名
export type SessionID = string
export type Status = "idle" | "busy"
export type Part = TextPart | ToolPart | FilePart
```

---

## 第三部分：函数类型

### 3.1 函数类型基础

TypeScript为函数提供了强大的类型系统。

#### 3.1.1 函数声明和表达式

```typescript
// 函数声明
function add(a: number, b: number): number {
  return a + b
}

// 函数表达式
const multiply = (a: number, b: number): number => {
  return a * b
}

// 简写形式
const subtract = (a: number, b: number): number => a - b

// 完整的函数类型注解
const divide: (a: number, b: number) => number = (a, b) => {
  return a / b
}
```

**关键术语：**

- **函数声明（Function Declaration）**：使用 `function` 关键字
- **函数表达式（Function Expression）**：将函数赋值给变量
- **箭头函数（Arrow Function）**：使用 `=>` 的简洁语法

#### 3.1.2 可选参数和默认参数

```typescript
// 可选参数（使用?）
function buildName(firstName: string, lastName?: string): string {
  if (lastName) {
    return `${firstName} ${lastName}`
  }
  return firstName
}

buildName("Alice")           // 正确
buildName("Alice", "Smith")  // 正确

// 默认参数
function greet(name: string, greeting: string = "Hello"): string {
  return `${greeting}, ${name}!`
}

greet("Alice")              // "Hello, Alice!"
greet("Alice", "Hi")        // "Hi, Alice!"

// 注意：可选参数必须在必需参数之后
// function invalid(a?: string, b: string) {}  // 错误！

// 但默认参数可以在任何位置
function valid(a: string = "default", b: string) {
  return `${a} ${b}`
}
```

**关键术语：**

- **可选参数（Optional Parameter）**：使用 `?` 标记，可以不传
- **默认参数（Default Parameter）**：提供默认值

#### 3.1.3 剩余参数（Rest Parameters）

```typescript
// 剩余参数收集所有额外参数到数组中
function sum(...numbers: number[]): number {
  return numbers.reduce((total, n) => total + n, 0)
}

sum(1, 2, 3)        // 6
sum(1, 2, 3, 4, 5)  // 15

// 剩余参数必须是最后一个参数
function buildArray<T>(first: T, ...rest: T[]): T[] {
  return [first, ...rest]
}

buildArray(1, 2, 3, 4)  // [1, 2, 3, 4]

// 剩余参数的类型可以是元组
function logInfo(message: string, ...details: [number, boolean]): void {
  console.log(message, details[0], details[1])
}

logInfo("Status", 200, true)
```

**关键术语：**

- **剩余参数（Rest Parameters）**：使用 `...` 收集多个参数
- **展开运算符（Spread Operator）**：使用 `...` 展开数组

#### 3.1.4 函数重载（Function Overloads）

函数重载允许一个函数根据不同的参数类型有不同的行为。

```typescript
// 重载签名（Overload Signatures）
function process(input: string): string
function process(input: number): number
function process(input: string[]): string[]

// 实现签名（Implementation Signature）
function process(input: string | number | string[]): string | number | string[] {
  if (typeof input === "string") {
    return input.toUpperCase()
  } else if (typeof input === "number") {
    return input * 2
  } else {
    return input.map(s => s.toUpperCase())
  }
}

// 调用时TypeScript知道返回类型
let str: string = process("hello")      // string
let num: number = process(42)           // number
let arr: string[] = process(["a", "b"]) // string[]
```

**关键概念：**

- **重载签名**：描述函数的不同调用方式
- **实现签名**：实际的函数实现，必须兼容所有重载签名
- 调用者只能看到重载签名，看不到实现签名

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/session/index.ts
export namespace Session {
  // 函数重载：根据参数返回不同类型
  export function get(id: string): Promise<Info>
  export function get(id: string, throwIfNotFound: false): Promise<Info | undefined>
  export function get(id: string, throwIfNotFound?: boolean): Promise<Info | undefined> {
    // 实现...
  }
}
```

#### 3.1.5 this参数

TypeScript允许显式声明函数中 `this` 的类型。

```typescript
interface Database {
  query(sql: string): void
}

// 声明this的类型
function executeQuery(this: Database, sql: string) {
  this.query(sql)  // TypeScript知道this是Database类型
}

const db: Database = {
  query(sql: string) {
    console.log(`Executing: ${sql}`)
  }
}

// 使用call/apply/bind调用
executeQuery.call(db, "SELECT * FROM users")

// 箭头函数不能声明this类型
// const invalid = (this: Database) => {}  // 错误！
```

**关键术语：**

- **this参数**：函数的第一个参数，名为 `this`，用于声明this的类型
- **不会出现在参数列表中**：调用时不需要传递this参数

#### 3.1.6 void 和 never 返回类型

```typescript
// void：函数没有返回值
function log(message: string): void {
  console.log(message)
  // 可以有空return或return undefined
}

// void类型的函数可以返回undefined
function doSomething(): void {
  return undefined  // 正确
}

// never：函数永远不会返回
function throwError(message: string): never {
  throw new Error(message)
  // 永远不会到达这里
}

function infiniteLoop(): never {
  while (true) {
    // 无限循环
  }
}

// never用于穷尽检查
type Shape = Circle | Square

function getArea(shape: Shape): number {
  switch (shape.kind) {
    case "circle":
      return Math.PI * shape.radius ** 2
    case "square":
      return shape.size ** 2
    default:
      // 如果所有case都覆盖了，这里的shape类型是never
      const _exhaustive: never = shape
      throw new Error("Unknown shape")
  }
}
```

**关键术语：**

- **void**：表示没有返回值
- **never**：表示永远不会返回（抛出异常或无限循环）
- **穷尽检查（Exhaustiveness Checking）**：使用never确保处理了所有情况

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/util/log.ts
export namespace Log {
  export function info(message: string, data?: any): void {
    console.log(message, data)
  }

  export function error(message: string, error?: Error): void {
    console.error(message, error)
  }
}

// 来自 packages/opencode/src/session/prompt.ts
export namespace SessionPrompt {
  export function cancel(sessionID: string): void {
    const match = state()[sessionID]
    if (match) {
      match.abort.abort()
      delete state()[sessionID]
    }
  }
}
```

---

## 第四部分：泛型系统（Generics）

泛型是TypeScript最强大的特性之一，允许编写可重用的、类型安全的代码。根据[最新的TypeScript高级特性指南](https://leapcell.io/blog/mastering-typescript-generics-conditions-mappings-and-inference)，泛型是构建复杂类型系统的基础。

### 4.1 泛型基础

**关键概念：泛型（Generic）** 是类型的参数化，就像函数参数一样，但操作的是类型而不是值。

#### 4.1.1 泛型函数

```typescript
// 不使用泛型的问题
function identityNumber(value: number): number {
  return value
}

function identityString(value: string): string {
  return value
}

// 使用any失去类型安全
function identityAny(value: any): any {
  return value
}

// 使用泛型：类型安全且可重用
function identity<T>(value: T): T {
  return value
}

// 使用时可以显式指定类型
let num = identity<number>(42)        // num: number
let str = identity<string>("hello")   // str: string

// 或让TypeScript自动推断
let num2 = identity(42)      // TypeScript推断T为number
let str2 = identity("hello") // TypeScript推断T为string
```

**关键术语：**

- **类型参数（Type Parameter）**：使用 `<T>` 声明的类型变量
- **类型实参（Type Argument）**：调用时传入的具体类型
- **类型推断（Type Inference）**：TypeScript自动推断类型参数

#### 4.1.2 多个类型参数

```typescript
// 多个类型参数
function pair<T, U>(first: T, second: U): [T, U] {
  return [first, second]
}

let p1 = pair<string, number>("age", 30)  // [string, number]
let p2 = pair("name", true)               // 推断为[string, boolean]

// 实际应用：键值对
function createMap<K, V>(key: K, value: V): Map<K, V> {
  const map = new Map<K, V>()
  map.set(key, value)
  return map
}

let stringToNumber = createMap("count", 42)
let numberToString = createMap(1, "one")
```

#### 4.1.3 泛型数组

```typescript
// 泛型数组函数
function first<T>(arr: T[]): T | undefined {
  return arr[0]
}

function last<T>(arr: T[]): T | undefined {
  return arr[arr.length - 1]
}

// 使用
let firstNum = first([1, 2, 3])      // number | undefined
let firstStr = first(["a", "b"])     // string | undefined

// 数组操作
function map<T, U>(arr: T[], fn: (item: T) => U): U[] {
  return arr.map(fn)
}

let numbers = [1, 2, 3]
let strings = map(numbers, n => n.toString())  // string[]

// 过滤
function filter<T>(arr: T[], predicate: (item: T) => boolean): T[] {
  return arr.filter(predicate)
}

let evens = filter([1, 2, 3, 4], n => n % 2 === 0)  // number[]
```

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/session/message-v2.ts
export namespace MessageV2 {
  // 泛型函数：过滤特定类型的part
  export function filterParts<T extends Part["type"]>(
    parts: Part[],
    type: T
  ): Extract<Part, { type: T }>[] {
    return parts.filter(p => p.type === type) as any
  }

  // 使用
  const textParts = filterParts(message.parts, "text")     // TextPart[]
  const toolParts = filterParts(message.parts, "tool")     // ToolPart[]
}
```

### 4.2 泛型约束（Generic Constraints）

泛型约束限制类型参数必须满足某些条件。

#### 4.2.1 extends约束

```typescript
// 约束类型参数必须有length属性
interface HasLength {
  length: number
}

function logLength<T extends HasLength>(item: T): void {
  console.log(item.length)
}

logLength("hello")           // 正确：string有length
logLength([1, 2, 3])         // 正确：array有length
logLength({ length: 10 })    // 正确：对象有length
// logLength(42)             // 错误：number没有length

// 约束为特定类型
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key]
}

let person = { name: "Alice", age: 30 }
let name = getProperty(person, "name")  // string
let age = getProperty(person, "age")    // number
// getProperty(person, "invalid")       // 错误：不是person的键
```

**关键术语：**

- **泛型约束（Generic Constraint）**：使用 `extends` 限制类型参数
- **keyof操作符**：获取对象类型的所有键的联合类型

#### 4.2.2 多重约束

```typescript
// 使用交叉类型实现多重约束
interface Nameable {
  name: string
}

interface Ageable {
  age: number
}

function printInfo<T extends Nameable & Ageable>(obj: T): void {
  console.log(`${obj.name} is ${obj.age} years old`)
}

printInfo({ name: "Alice", age: 30 })  // 正确
// printInfo({ name: "Bob" })          // 错误：缺少age
```

#### 4.2.3 使用类型参数约束

```typescript
// 一个类型参数约束另一个类型参数
function copyFields<T extends U, U>(target: T, source: U): T {
  for (let key in source) {
    ;(target as any)[key] = source[key]
  }
  return target
}

// T必须是U的子类型
let x = { a: 1, b: 2, c: 3 }
let y = { a: 10, b: 20 }
copyFields(x, y)  // 正确：x有y的所有属性
```

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/tool/tool.ts
export namespace Tool {
  export interface Info {
    name: string
    parameters: any
    execute: (input: any) => any
  }

  // 泛型约束：T必须是Tool.Info的子类型
  export type InferParameters<T extends Info> =
    T extends { parameters: infer P } ? P : never

  export type InferResult<T extends Info> =
    T extends { execute: (...args: any[]) => infer R } ? R : never
}

// 使用
type BashParams = Tool.InferParameters<typeof BashTool>
type BashResult = Tool.InferResult<typeof BashTool>
```

### 4.3 泛型接口和类型

#### 4.3.1 泛型接口

```typescript
// 基本泛型接口
interface Box<T> {
  value: T
}

let stringBox: Box<string> = { value: "hello" }
let numberBox: Box<number> = { value: 42 }

// 泛型函数接口
interface Transformer<T, U> {
  (input: T): U
}

let toString: Transformer<number, string> = (n) => n.toString()
let toNumber: Transformer<string, number> = (s) => parseInt(s)

// 复杂的泛型接口
interface Repository<T> {
  items: T[]
  add(item: T): void
  remove(id: string): void
  find(predicate: (item: T) => boolean): T | undefined
}

class UserRepository implements Repository<User> {
  items: User[] = []

  add(user: User): void {
    this.items.push(user)
  }

  remove(id: string): void {
    this.items = this.items.filter(u => u.id !== id)
  }

  find(predicate: (user: User) => boolean): User | undefined {
    return this.items.find(predicate)
  }
}
```

#### 4.3.2 泛型类型别名

```typescript
// 基本泛型类型别名
type Container<T> = {
  value: T
  getValue(): T
  setValue(value: T): void
}

// 泛型联合类型
type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E }

function divide(a: number, b: number): Result<number> {
  if (b === 0) {
    return { success: false, error: new Error("Division by zero") }
  }
  return { success: true, data: a / b }
}

// 使用Result
let result = divide(10, 2)
if (result.success) {
  console.log(result.data)  // TypeScript知道这里有data
} else {
  console.log(result.error) // TypeScript知道这里有error
}

// 泛型Promise类型
type AsyncResult<T> = Promise<Result<T>>

async function fetchUser(id: string): AsyncResult<User> {
  try {
    const user = await api.getUser(id)
    return { success: true, data: user }
  } catch (error) {
    return { success: false, error: error as Error }
  }
}
```

**关键术语：**

- **默认类型参数（Default Type Parameter）**：使用 `=` 提供默认类型
- **可辨识联合（Discriminated Union）**：使用共同属性区分联合类型

#### 4.3.3 泛型类

```typescript
// 基本泛型类
class Stack<T> {
  private items: T[] = []

  push(item: T): void {
    this.items.push(item)
  }

  pop(): T | undefined {
    return this.items.pop()
  }

  peek(): T | undefined {
    return this.items[this.items.length - 1]
  }

  get size(): number {
    return this.items.length
  }
}

let numberStack = new Stack<number>()
numberStack.push(1)
numberStack.push(2)
let top = numberStack.pop()  // number | undefined

// 泛型类继承
class NamedStack<T> extends Stack<T> {
  constructor(public name: string) {
    super()
  }
}

let myStack = new NamedStack<string>("MyStack")
```

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/storage/storage.ts
export namespace Storage {
  // 泛型存储接口
  export interface Store<T> {
    get(key: string): Promise<T | undefined>
    set(key: string, value: T): Promise<void>
    delete(key: string): Promise<void>
    list(): Promise<T[]>
  }

  // 实现泛型存储
  export class FileStore<T> implements Store<T> {
    constructor(private directory: string) {}

    async get(key: string): Promise<T | undefined> {
      // 实现...
    }

    async set(key: string, value: T): Promise<void> {
      // 实现...
    }

    async delete(key: string): Promise<void> {
      // 实现...
    }

    async list(): Promise<T[]> {
      // 实现...
    }
  }
}
```

### 4.4 默认类型参数

```typescript
// 基本默认类型参数
interface Container<T = string> {
  value: T
}

let container1: Container = { value: "hello" }  // T默认为string
let container2: Container<number> = { value: 42 }

// 多个默认类型参数
type Result<T = any, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E }

let result1: Result = { success: true, data: "anything" }
let result2: Result<number> = { success: true, data: 42 }
let result3: Result<number, string> = {
  success: false,
  error: "Custom error"
}

// 默认类型参数的约束
interface Response<T = unknown, E extends Error = Error> {
  data?: T
  error?: E
}
```

**关键术语：**

- **默认类型参数（Default Type Parameter）**：使用 `=` 指定默认类型
- **类型参数顺序**：有默认值的类型参数必须在没有默认值的之后

### 4.5 泛型工具函数

```typescript
// 深度克隆
function deepClone<T>(obj: T): T {
  return JSON.parse(JSON.stringify(obj))
}

// 合并对象
function merge<T, U>(obj1: T, obj2: U): T & U {
  return { ...obj1, ...obj2 }
}

// 选择属性
function pick<T, K extends keyof T>(obj: T, keys: K[]): Pick<T, K> {
  const result = {} as Pick<T, K>
  for (const key of keys) {
    result[key] = obj[key]
  }
  return result
}

// 使用
let person = { name: "Alice", age: 30, email: "alice@example.com" }
let nameAndAge = pick(person, ["name", "age"])  // { name: string; age: number }

// 省略属性
function omit<T, K extends keyof T>(obj: T, keys: K[]): Omit<T, K> {
  const result = { ...obj }
  for (const key of keys) {
    delete result[key]
  }
  return result
}

let withoutEmail = omit(person, ["email"])  // { name: string; age: number }
```

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/util/fn.ts
export function fn<I, O>(
  schema: z.ZodType<I>,
  handler: (input: I) => O
): (input: unknown) => O {
  return (input: unknown) => {
    const validated = schema.parse(input)
    return handler(validated)
  }
}

// 使用
const createUser = fn(
  z.object({
    name: z.string(),
    email: z.string().email()
  }),
  (input) => {
    // input的类型自动推断为{ name: string; email: string }
    return { id: generateId(), ...input }
  }
)
```

---

## 第五部分：高级类型

根据[TypeScript高级类型指南](https://www.rst.software/blog/advanced-typing-in-typescript-with-generics)，高级类型是构建复杂类型系统的关键。

### 5.1 联合类型和交叉类型

#### 5.1.1 联合类型（Union Types）

**联合类型（Union Type）** 表示值可以是多种类型之一，使用 `|` 连接。

```typescript
// 基本联合类型
type StringOrNumber = string | number

let value: StringOrNumber = "hello"
value = 42  // 正确

// 字面量联合类型
type Status = "pending" | "running" | "completed" | "failed"
type HttpMethod = "GET" | "POST" | "PUT" | "DELETE"

let status: Status = "pending"
// status = "unknown"  // 错误！

// 对象联合类型
type Success = { success: true; data: string }
type Failure = { success: false; error: Error }
type Result = Success | Failure

function handleResult(result: Result) {
  if (result.success) {
    console.log(result.data)  // TypeScript知道这里有data
  } else {
    console.log(result.error) // TypeScript知道这里有error
  }
}
```

**关键术语：**

- **联合类型（Union Type）**：使用 `|` 表示多个类型之一
- **类型收窄（Type Narrowing）**：通过条件判断缩小类型范围

#### 5.1.2 可辨识联合（Discriminated Unions）

**可辨识联合** 是一种特殊的联合类型，每个成员都有一个共同的字面量属性（称为"判别式"）。

```typescript
// 可辨识联合的标准模式
interface Circle {
  kind: "circle"  // 判别式
  radius: number
}

interface Square {
  kind: "square"  // 判别式
  size: number
}

interface Triangle {
  kind: "triangle"  // 判别式
  base: number
  height: number
}

type Shape = Circle | Square | Triangle

// TypeScript可以根据kind自动收窄类型
function getArea(shape: Shape): number {
  switch (shape.kind) {
    case "circle":
      return Math.PI * shape.radius ** 2  // shape是Circle
    case "square":
      return shape.size ** 2  // shape是Square
    case "triangle":
      return (shape.base * shape.height) / 2  // shape是Triangle
    default:
      // 穷尽检查：如果所有case都覆盖了，这里的shape是never
      const _exhaustive: never = shape
      throw new Error("Unknown shape")
  }
}
```

**关键术语：**

- **可辨识联合（Discriminated Union）**：也叫"标签联合"或"代数数据类型"
- **判别式（Discriminant）**：用于区分联合成员的共同属性
- **穷尽检查（Exhaustiveness Check）**：确保处理了所有可能的情况

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/session/message-v2.ts
export namespace MessageV2 {
  // 可辨识联合：使用type作为判别式
  export type Part =
    | { type: "text"; text: string; time?: TimeInfo }
    | { type: "tool"; tool: string; state: ToolState }
    | { type: "file"; url: string; filename: string }
    | { type: "reasoning"; text: string; time?: TimeInfo }
    | { type: "compaction"; summary: string }

  // 根据type自动收窄类型
  function processPart(part: Part) {
    switch (part.type) {
      case "text":
        console.log(part.text)  // TypeScript知道有text属性
        break
      case "tool":
        console.log(part.tool, part.state)  // TypeScript知道有tool和state
        break
      case "file":
        console.log(part.url, part.filename)  // TypeScript知道有url和filename
        break
    }
  }
}
```

#### 5.1.3 交叉类型（Intersection Types）

**交叉类型（Intersection Type）** 将多个类型合并为一个，使用 `&` 连接。

```typescript
// 基本交叉类型
interface Timestamped {
  timestamp: number
}

interface Tagged {
  tags: string[]
}

type TimestampedAndTagged = Timestamped & Tagged

let item: TimestampedAndTagged = {
  timestamp: Date.now(),
  tags: ["important", "urgent"]
}

// 扩展类型
interface User {
  id: string
  name: string
}

type Admin = User & {
  permissions: string[]
  role: "admin"
}

let admin: Admin = {
  id: "123",
  name: "Alice",
  permissions: ["read", "write", "delete"],
  role: "admin"
}

// 混合（Mixin）模式
function extend<T, U>(first: T, second: U): T & U {
  return { ...first, ...second }
}

let person = { name: "Bob" }
let employee = { employeeId: "E123" }
let personEmployee = extend(person, employee)
// personEmployee: { name: string } & { employeeId: string }
```

**关键术语：**

- **交叉类型（Intersection Type）**：使用 `&` 合并多个类型
- **混合（Mixin）**：组合多个对象的属性

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/session/message-v2.ts
export namespace MessageV2 {
  // 基础接口
  export interface Info {
    id: string
    sessionID: string
    role: string
  }

  // 使用交叉类型扩展
  export type User = Info & {
    role: "user"
    variant?: string
    tools?: Record<string, boolean>
  }

  export type Assistant = Info & {
    role: "assistant"
    agent: string
    modelID: string
  }
}
```

### 5.2 类型守卫（Type Guards）

类型守卫是运行时检查，帮助TypeScript收窄类型。

#### 5.2.1 typeof 类型守卫

```typescript
function padLeft(value: string, padding: string | number) {
  if (typeof padding === "number") {
    // 这里padding是number
    return " ".repeat(padding) + value
  }
  // 这里padding是string
  return padding + value
}

// typeof可以检查的类型
function checkType(value: unknown) {
  if (typeof value === "string") {
    console.log(value.toUpperCase())
  } else if (typeof value === "number") {
    console.log(value.toFixed(2))
  } else if (typeof value === "boolean") {
    console.log(value ? "yes" : "no")
  } else if (typeof value === "function") {
    value()
  } else if (typeof value === "object") {
    // 注意：null的typeof也是"object"
    if (value === null) {
      console.log("null")
    } else {
      console.log("object")
    }
  }
}
```

**关键术语：**

- **typeof操作符**：JavaScript运行时类型检查
- **类型收窄（Type Narrowing）**：根据检查结果缩小类型范围

#### 5.2.2 instanceof 类型守卫

```typescript
class Bird {
  fly() {
    console.log("Flying")
  }
  layEggs() {
    console.log("Laying eggs")
  }
}

class Fish {
  swim() {
    console.log("Swimming")
  }
  layEggs() {
    console.log("Laying eggs")
  }
}

function move(animal: Bird | Fish) {
  if (animal instanceof Bird) {
    animal.fly()  // TypeScript知道这是Bird
  } else {
    animal.swim()  // TypeScript知道这是Fish
  }

  // 两者都有的方法可以直接调用
  animal.layEggs()
}

// instanceof也可以用于内置类型
function processValue(value: Date | string) {
  if (value instanceof Date) {
    console.log(value.toISOString())
  } else {
    console.log(value.toUpperCase())
  }
}
```

**关键术语：**

- **instanceof操作符**：检查对象是否是某个类的实例
- **原型链检查**：instanceof沿着原型链检查

#### 5.2.3 in 类型守卫

```typescript
interface Car {
  drive(): void
}

interface Boat {
  sail(): void
}

function operate(vehicle: Car | Boat) {
  if ("drive" in vehicle) {
    vehicle.drive()  // TypeScript知道这是Car
  } else {
    vehicle.sail()  // TypeScript知道这是Boat
  }
}

// in也可以检查可选属性
interface Person {
  name: string
  age?: number
}

function printAge(person: Person) {
  if ("age" in person && person.age !== undefined) {
    console.log(`Age: ${person.age}`)
  }
}
```

**关键术语：**

- **in操作符**：检查对象是否有某个属性
- **属性检查**：包括自有属性和继承属性

#### 5.2.4 自定义类型守卫

使用 `is` 关键字创建自定义类型守卫函数。

```typescript
// 基本自定义类型守卫
function isString(value: unknown): value is string {
  return typeof value === "string"
}

function isNumber(value: unknown): value is number {
  return typeof value === "number"
}

function processValue(value: unknown) {
  if (isString(value)) {
    console.log(value.toUpperCase())  // TypeScript知道这是string
  } else if (isNumber(value)) {
    console.log(value.toFixed(2))  // TypeScript知道这是number
  }
}

// 复杂的自定义类型守卫
interface User {
  id: string
  name: string
  email: string
}

function isUser(obj: any): obj is User {
  return (
    typeof obj === "object" &&
    obj !== null &&
    typeof obj.id === "string" &&
    typeof obj.name === "string" &&
    typeof obj.email === "string"
  )
}

function greetUser(data: unknown) {
  if (isUser(data)) {
    console.log(`Hello, ${data.name}!`)  // TypeScript知道data是User
  }
}
```

**关键术语：**

- **类型谓词（Type Predicate）**：`value is Type` 语法
- **用户定义类型守卫（User-Defined Type Guard）**：自定义的类型检查函数

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/session/message-v2.ts
export namespace MessageV2 {
  // 自定义类型守卫
  export function isToolPart(part: Part): part is ToolPart {
    return part.type === "tool"
  }

  export function isTextPart(part: Part): part is TextPart {
    return part.type === "text"
  }

  export function isFilePart(part: Part): part is FilePart {
    return part.type === "file"
  }

  // 使用类型守卫
  function processParts(parts: Part[]) {
    const toolParts = parts.filter(isToolPart)  // ToolPart[]
    const textParts = parts.filter(isTextPart)  // TextPart[]
  }
}
```

#### 5.2.5 断言函数（Assertion Functions）

断言函数在条件不满足时抛出错误，TypeScript会据此收窄类型。

```typescript
// 基本断言函数
function assert(condition: any, message?: string): asserts condition {
  if (!condition) {
    throw new Error(message || "Assertion failed")
  }
}

function processValue(value: unknown) {
  assert(typeof value === "string", "Value must be a string")
  // 这里TypeScript知道value是string
  console.log(value.toUpperCase())
}

// 类型断言函数
function assertIsString(value: unknown): asserts value is string {
  if (typeof value !== "string") {
    throw new Error("Not a string")
  }
}

function assertIsNumber(value: unknown): asserts value is number {
  if (typeof value !== "number") {
    throw new Error("Not a number")
  }
}

function calculate(input: unknown) {
  assertIsNumber(input)
  // 这里TypeScript知道input是number
  return input * 2
}

// 断言对象属性
function assertHasProperty<T, K extends string>(
  obj: T,
  key: K
): asserts obj is T & Record<K, unknown> {
  if (!(key in obj)) {
    throw new Error(`Missing property: ${key}`)
  }
}

function processObject(obj: object) {
  assertHasProperty(obj, "name")
  // 这里TypeScript知道obj有name属性
  console.log(obj.name)
}
```

**关键术语：**

- **断言函数（Assertion Function）**：使用 `asserts` 关键字
- **断言签名（Assertion Signature）**：`asserts condition` 或 `asserts value is Type`

### 5.3 类型断言（Type Assertions）

类型断言告诉TypeScript"相信我，我知道这是什么类型"。

#### 5.3.1 as 语法

```typescript
// 基本类型断言
let value: unknown = "hello"
let length = (value as string).length

// 从更宽泛的类型断言到更具体的类型
let input: any = document.getElementById("input")
let inputElement = input as HTMLInputElement
inputElement.value = "Hello"

// 断言为联合类型的某个成员
type Status = "pending" | "completed"
let status: string = "pending"
let typedStatus = status as Status

// 常量断言
let obj = { name: "Alice" } as const
// obj的类型是 { readonly name: "Alice" }
// obj.name = "Bob"  // 错误！

let arr = [1, 2, 3] as const
// arr的类型是 readonly [1, 2, 3]
// arr.push(4)  // 错误！
```

**关键术语：**

- **类型断言（Type Assertion）**：使用 `as` 语法
- **常量断言（Const Assertion）**：使用 `as const` 创建只读字面量类型

#### 5.3.2 非空断言

```typescript
// 非空断言操作符 !
function getValue(): string | null {
  return "hello"
}

let value = getValue()
// let length = value.length  // 错误：value可能是null

let length = value!.length  // 告诉TypeScript这不是null

// 可选链 vs 非空断言
interface User {
  name: string
  address?: {
    city: string
  }
}

let user: User = { name: "Alice" }

// 可选链：安全访问
let city1 = user.address?.city  // string | undefined

// 非空断言：假设address存在
let city2 = user.address!.city  // string（但运行时可能报错）
```

**关键术语：**

- **非空断言（Non-null Assertion）**：使用 `!` 操作符
- **可选链（Optional Chaining）**：使用 `?.` 安全访问

#### 5.3.3 双重断言

有时需要先断言为 `unknown`，再断言为目标类型。

```typescript
// 不兼容的类型断言需要双重断言
let value = "hello" as unknown as number  // 不推荐，但有时必要

// 实际应用：处理第三方库的类型问题
interface OldAPI {
  getData(): any
}

interface NewAPI {
  getData(): { id: string; name: string }
}

let api: OldAPI = getAPI()
let data = api.getData() as unknown as NewAPI["getData"]
```

**注意：** 类型断言不会改变运行时的值，只是告诉TypeScript如何理解类型。滥用类型断言会失去类型安全性。

---

## 第六部分：条件类型（Conditional Types）

条件类型是TypeScript最强大的特性之一，允许根据条件选择类型。根据[TypeScript高级特性指南](https://leapcell.io/blog/mastering-typescript-generics-conditions-mappings-and-inference)，条件类型是构建复杂类型系统的核心。

### 6.1 条件类型基础

**条件类型（Conditional Type）** 的语法类似于JavaScript的三元运算符：`T extends U ? X : Y`

#### 6.1.1 基本条件类型

```typescript
// 基本语法：T extends U ? X : Y
// 如果T可以赋值给U，则类型为X，否则为Y

type IsString<T> = T extends string ? true : false

type A = IsString<string>   // true
type B = IsString<number>   // false
type C = IsString<"hello">  // true（字面量类型是string的子类型）

// 实际应用：根据类型返回不同类型
type TypeName<T> =
  T extends string ? "string" :
  T extends number ? "number" :
  T extends boolean ? "boolean" :
  T extends undefined ? "undefined" :
  T extends Function ? "function" :
  "object"

type T0 = TypeName<string>    // "string"
type T1 = TypeName<"a">       // "string"
type T2 = TypeName<true>      // "boolean"
type T3 = TypeName<() => void> // "function"
type T4 = TypeName<string[]>  // "object"
```

**关键术语：**

- **条件类型（Conditional Type）**：根据类型关系选择类型
- **extends关键字**：在条件类型中表示"可赋值给"或"是子类型"
- **类型分支（Type Branch）**：条件为真或假时的类型

#### 6.1.2 嵌套条件类型

```typescript
// 多层嵌套的条件类型
type DeepReadonly<T> =
  T extends (infer U)[] ? ReadonlyArray<DeepReadonly<U>> :
  T extends object ? { readonly [K in keyof T]: DeepReadonly<T[K]> } :
  T

// 使用
type Person = {
  name: string
  age: number
  hobbies: string[]
  address: {
    city: string
    country: string
  }
}

type ReadonlyPerson = DeepReadonly<Person>
// {
//   readonly name: string
//   readonly age: number
//   readonly hobbies: ReadonlyArray<string>
//   readonly address: {
//     readonly city: string
//     readonly country: string
//   }
// }
```

### 6.2 分布式条件类型

当条件类型作用于联合类型时，会自动分布到每个成员上。

```typescript
// 分布式条件类型
type ToArray<T> = T extends any ? T[] : never

// 当T是联合类型时，条件类型会分布
type Result = ToArray<string | number>
// 等价于：ToArray<string> | ToArray<number>
// 结果：string[] | number[]

// 对比：非分布式
type ToArrayNonDist<T> = [T] extends [any] ? T[] : never
type Result2 = ToArrayNonDist<string | number>
// 结果：(string | number)[]

// 实际应用：过滤联合类型
type Exclude<T, U> = T extends U ? never : T
type Extract<T, U> = T extends U ? T : never

type T0 = Exclude<"a" | "b" | "c", "a">  // "b" | "c"
type T1 = Extract<"a" | "b" | "c", "a" | "f">  // "a"
```

**关键术语：**

- **分布式条件类型（Distributive Conditional Type）**：自动分布到联合类型的每个成员
- **裸类型参数（Naked Type Parameter）**：直接使用的类型参数（如 `T extends U`）会触发分布
- **包装类型参数（Wrapped Type Parameter）**：被包装的类型参数（如 `[T] extends [U]`）不会分布

### 6.3 infer 关键字

**infer** 关键字用于在条件类型中推断类型。

#### 6.3.1 推断函数返回类型

```typescript
// 推断函数返回类型
type ReturnType<T> = T extends (...args: any[]) => infer R ? R : never

type Func1 = () => string
type Func2 = (x: number) => number
type Func3 = (a: string, b: number) => boolean

type R1 = ReturnType<Func1>  // string
type R2 = ReturnType<Func2>  // number
type R3 = ReturnType<Func3>  // boolean

// 推断函数参数类型
type Parameters<T> = T extends (...args: infer P) => any ? P : never

type P1 = Parameters<Func1>  // []
type P2 = Parameters<Func2>  // [x: number]
type P3 = Parameters<Func3>  // [a: string, b: number]
```

**关键术语：**

- **infer关键字**：在条件类型中声明一个待推断的类型变量
- **类型推断（Type Inference）**：TypeScript自动推断类型

#### 6.3.2 推断Promise类型

```typescript
// 推断Promise的值类型
type Awaited<T> = T extends Promise<infer U> ? U : T

type P1 = Awaited<Promise<string>>  // string
type P2 = Awaited<number>           // number

// 递归推断嵌套Promise
type DeepAwaited<T> =
  T extends Promise<infer U>
    ? DeepAwaited<U>  // 递归展开
    : T

type P3 = DeepAwaited<Promise<Promise<string>>>  // string
type P4 = DeepAwaited<Promise<Promise<Promise<number>>>>  // number
```

#### 6.3.3 推断数组元素类型

```typescript
// 推断数组元素类型
type ElementType<T> = T extends (infer E)[] ? E : T

type E1 = ElementType<string[]>  // string
type E2 = ElementType<number[]>  // number
type E3 = ElementType<string>    // string（不是数组）

// 推断元组的第一个元素
type First<T> = T extends [infer F, ...any[]] ? F : never

type F1 = First<[string, number, boolean]>  // string
type F2 = First<[number]>                   // number
type F3 = First<[]>                         // never

// 推断元组的最后一个元素
type Last<T> = T extends [...any[], infer L] ? L : never

type L1 = Last<[string, number, boolean]>  // boolean
type L2 = Last<[number]>                   // number
```

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/tool/tool.ts
export namespace Tool {
  export interface Info {
    name: string
    parameters: any
    execute: (input: any) => any
  }

  // 使用infer推断参数类型
  export type InferParameters<T extends Info> =
    T extends { parameters: infer P } ? P : never

  // 使用infer推断返回类型
  export type InferResult<T extends Info> =
    T extends { execute: (...args: any[]) => infer R } ? R : never

  // 推断异步返回类型
  export type InferAsyncResult<T extends Info> =
    T extends { execute: (...args: any[]) => Promise<infer R> } ? R : never
}

// 使用
type BashParams = Tool.InferParameters<typeof BashTool>
type BashResult = Tool.InferResult<typeof BashTool>
```

### 6.4 条件类型的实用模式

#### 6.4.1 NonNullable

```typescript
// 移除null和undefined
type NonNullable<T> = T extends null | undefined ? never : T

type T0 = NonNullable<string | null | undefined>  // string
type T1 = NonNullable<string[] | null>            // string[]
```

#### 6.4.2 提取函数类型

```typescript
// 提取对象中的函数类型
type FunctionPropertyNames<T> = {
  [K in keyof T]: T[K] extends Function ? K : never
}[keyof T]

type FunctionProperties<T> = Pick<T, FunctionPropertyNames<T>>

interface Person {
  name: string
  age: number
  greet(): void
  walk(): void
}

type PersonFunctions = FunctionPropertyNames<Person>  // "greet" | "walk"
type PersonMethods = FunctionProperties<Person>
// {
//   greet(): void
//   walk(): void
// }
```

#### 6.4.3 深度Partial

```typescript
// 递归地将所有属性变为可选
type DeepPartial<T> = T extends object ? {
  [P in keyof T]?: DeepPartial<T[P]>
} : T

interface Config {
  server: {
    host: string
    port: number
    ssl: {
      enabled: boolean
      cert: string
    }
  }
  database: {
    host: string
    port: number
  }
}

type PartialConfig = DeepPartial<Config>
// {
//   server?: {
//     host?: string
//     port?: number
//     ssl?: {
//       enabled?: boolean
//       cert?: string
//     }
//   }
//   database?: {
//     host?: string
//     port?: number
//   }
// }
```

#### 6.4.4 联合类型转交叉类型

```typescript
// 将联合类型转换为交叉类型
type UnionToIntersection<U> =
  (U extends any ? (k: U) => void : never) extends ((k: infer I) => void) ? I : never

type Union = { a: string } | { b: number }
type Intersection = UnionToIntersection<Union>
// { a: string } & { b: number }
```

**关键术语：**

- **类型体操（Type Gymnastics）**：使用高级类型特性构建复杂类型
- **递归类型（Recursive Type）**：类型定义中引用自身

---

## 第七部分：映射类型（Mapped Types）

映射类型允许从现有类型创建新类型，通过转换属性。根据[TypeScript映射类型指南](https://medium.com/@ignatovich.dm/typescript-and-advanced-type-manipulation-mapped-types-template-literal-types-and-more-cc22533bc15d)，映射类型是类型转换的核心工具。

### 7.1 映射类型基础

#### 7.1.1 基本映射类型

```typescript
// 基本语法：{ [P in K]: T }
// 遍历K中的每个键P，将其映射为类型T

type Readonly<T> = {
  readonly [P in keyof T]: T[P]
}

type Partial<T> = {
  [P in keyof T]?: T[P]
}

type Required<T> = {
  [P in keyof T]-?: T[P]  // -? 移除可选修饰符
}

// 使用
interface User {
  name: string
  age: number
  email?: string
}

type ReadonlyUser = Readonly<User>
// {
//   readonly name: string
//   readonly age: number
//   readonly email?: string
// }

type PartialUser = Partial<User>
// {
//   name?: string
//   age?: number
//   email?: string
// }

type RequiredUser = Required<User>
// {
//   name: string
//   age: number
//   email: string  // 不再可选
// }
```

**关键术语：**

- **映射类型（Mapped Type）**：使用 `in` 关键字遍历键
- **keyof操作符**：获取对象类型的所有键
- **修饰符（Modifier）**：`readonly`、`?`、`-readonly`、`-?`

#### 7.1.2 修饰符操作

```typescript
// 添加readonly
type AddReadonly<T> = {
  readonly [P in keyof T]: T[P]
}

// 移除readonly
type RemoveReadonly<T> = {
  -readonly [P in keyof T]: T[P]
}

// 添加可选
type AddOptional<T> = {
  [P in keyof T]?: T[P]
}

// 移除可选
type RemoveOptional<T> = {
  [P in keyof T]-?: T[P]
}

// 组合：移除readonly和可选
type Mutable<T> = {
  -readonly [P in keyof T]-?: T[P]
}
```

### 7.2 键重映射（Key Remapping）

TypeScript 4.1+ 支持在映射类型中重映射键。

#### 7.2.1 as子句

```typescript
// 使用as重映射键
type Getters<T> = {
  [P in keyof T as `get${Capitalize<string & P>}`]: () => T[P]
}

interface Person {
  name: string
  age: number
  email: string
}

type PersonGetters = Getters<Person>
// {
//   getName: () => string
//   getAge: () => number
//   getEmail: () => string
// }

// Setters
type Setters<T> = {
  [P in keyof T as `set${Capitalize<string & P>}`]: (value: T[P]) => void
}

type PersonSetters = Setters<Person>
// {
//   setName: (value: string) => void
//   setAge: (value: number) => void
//   setEmail: (value: string) => void
// }
```

**关键术语：**

- **键重映射（Key Remapping）**：使用 `as` 子句转换键名
- **Capitalize**：内置类型，将字符串首字母大写

#### 7.2.2 过滤属性

```typescript
// 过滤掉特定类型的属性
type RemoveKind<T> = {
  [K in keyof T as Exclude<K, "kind">]: T[K]
}

interface Circle {
  kind: "circle"
  radius: number
}

type CircleWithoutKind = RemoveKind<Circle>
// { radius: number }

// 只保留函数属性
type PickFunctions<T> = {
  [K in keyof T as T[K] extends Function ? K : never]: T[K]
}

interface Mixed {
  name: string
  age: number
  greet(): void
  walk(): void
}

type OnlyFunctions = PickFunctions<Mixed>
// {
//   greet(): void
//   walk(): void
// }
```

### 7.3 内置映射类型

TypeScript提供了许多内置的映射类型工具。

#### 7.3.1 Pick 和 Omit

```typescript
// Pick：选择属性
type Pick<T, K extends keyof T> = {
  [P in K]: T[P]
}

// Omit：排除属性
type Omit<T, K extends keyof any> = Pick<T, Exclude<keyof T, K>>

interface Todo {
  title: string
  description: string
  completed: boolean
  createdAt: number
}

type TodoPreview = Pick<Todo, "title" | "completed">
// {
//   title: string
//   completed: boolean
// }

type TodoInfo = Omit<Todo, "completed" | "createdAt">
// {
//   title: string
//   description: string
// }
```

#### 7.3.2 Record

```typescript
// Record：创建具有指定键和值类型的对象
type Record<K extends keyof any, T> = {
  [P in K]: T
}

type PageInfo = Record<"home" | "about" | "contact", { title: string; url: string }>
// {
//   home: { title: string; url: string }
//   about: { title: string; url: string }
//   contact: { title: string; url: string }
// }

// 实际应用：状态映射
type Status = "idle" | "loading" | "success" | "error"
type StatusMessages = Record<Status, string>

const messages: StatusMessages = {
  idle: "Ready",
  loading: "Loading...",
  success: "Success!",
  error: "Error occurred"
}
```

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/tool/tool.ts
export namespace Tool {
  // 工具注册表使用Record
  export type Registry = Record<string, Info>

  // 工具元数据映射
  export type Metadata = Record<string, any>
}

// 来自 packages/opencode/src/session/status.ts
export namespace SessionStatus {
  // 会话状态映射
  export type StatusMap = Record<string, Info>
}
```

### 7.4 高级映射模式

#### 7.4.1 深度映射

```typescript
// 递归地应用映射
type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object
    ? DeepReadonly<T[P]>
    : T[P]
}

type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object
    ? DeepPartial<T[P]>
    : T[P]
}

interface NestedConfig {
  server: {
    host: string
    port: number
    ssl: {
      enabled: boolean
      cert: string
    }
  }
}

type ReadonlyConfig = DeepReadonly<NestedConfig>
// {
//   readonly server: {
//     readonly host: string
//     readonly port: number
//     readonly ssl: {
//       readonly enabled: boolean
//       readonly cert: string
//     }
//   }
// }
```

#### 7.4.2 条件映射

```typescript
// 根据条件转换属性类型
type Nullable<T> = {
  [P in keyof T]: T[P] | null
}

type Promisify<T> = {
  [P in keyof T]: T[P] extends Function
    ? (...args: Parameters<T[P]>) => Promise<ReturnType<T[P]>>
    : Promise<T[P]>
}

interface API {
  getUser(id: string): User
  deleteUser(id: string): boolean
  config: Config
}

type AsyncAPI = Promisify<API>
// {
//   getUser: (id: string) => Promise<User>
//   deleteUser: (id: string) => Promise<boolean>
//   config: Promise<Config>
// }
```

---

## 第八部分：模板字面量类型（Template Literal Types）

模板字面量类型是TypeScript 4.1引入的强大特性，允许在类型层面操作字符串。根据[模板字面量类型指南](https://krython.com/tutorial/typescript/template-literal-types-string-manipulation-at-type-level)，这是构建类型安全API的关键工具。

### 8.1 基本模板字面量类型

```typescript
// 基本语法：使用反引号和${}
type World = "world"
type Greeting = `hello ${World}`  // "hello world"

// 与联合类型结合
type EmailLocaleIDs = "welcome_email" | "email_heading"
type FooterLocaleIDs = "footer_title" | "footer_sendoff"

type AllLocaleIDs = `${EmailLocaleIDs | FooterLocaleIDs}_id`
// "welcome_email_id" | "email_heading_id" | "footer_title_id" | "footer_sendoff_id"

// 多个联合类型的笛卡尔积
type Color = "red" | "blue"
type Size = "small" | "large"
type Style = `${Color}-${Size}`
// "red-small" | "red-large" | "blue-small" | "blue-large"
```

**关键术语：**

- **模板字面量类型（Template Literal Type）**：在类型层面操作字符串
- **笛卡尔积（Cartesian Product）**：多个联合类型的所有组合

### 8.2 内置字符串操作类型

TypeScript提供了四个内置的字符串操作类型：

```typescript
// Uppercase：转大写
type Loud = Uppercase<"hello">  // "HELLO"
type LoudGreeting = Uppercase<"hello world">  // "HELLO WORLD"

// Lowercase：转小写
type Quiet = Lowercase<"HELLO">  // "hello"
type QuietGreeting = Lowercase<"HELLO WORLD">  // "hello world"

// Capitalize：首字母大写
type Capitalized = Capitalize<"hello">  // "Hello"
type CapitalizedGreeting = Capitalize<"hello world">  // "Hello world"

// Uncapitalize：首字母小写
type Uncapitalized = Uncapitalize<"Hello">  // "hello"
type UncapitalizedGreeting = Uncapitalize<"Hello World">  // "hello World"

// 组合使用
type MakeGetter<T extends string> = `get${Capitalize<T>}`
type UserGetter = MakeGetter<"name">  // "getName"
type AgeGetter = MakeGetter<"age">    // "getAge"
```

**关键术语：**

- **Uppercase/Lowercase/Capitalize/Uncapitalize**：内置字符串转换类型
- **intrinsic类型**：由TypeScript编译器内部实现的特殊类型

### 8.3 实际应用：事件系统

```typescript
// 类型安全的事件系统
type PropEventSource<T> = {
  on<K extends string & keyof T>(
    eventName: `${K}Changed`,
    callback: (newValue: T[K]) => void
  ): void

  off<K extends string & keyof T>(
    eventName: `${K}Changed`
  ): void
}

declare function makeWatchedObject<T>(obj: T): T & PropEventSource<T>

const person = makeWatchedObject({
  firstName: "Alice",
  age: 30
})

// 类型安全：只能监听存在的属性
person.on("firstNameChanged", (newName) => {
  console.log(newName.toUpperCase())  // newName是string
})

person.on("ageChanged", (newAge) => {
  console.log(newAge.toFixed(2))  // newAge是number
})

// 错误：属性不存在
// person.on("firstNameChange", () => {})  // 错误！
// person.on("emailChanged", () => {})     // 错误！
```

### 8.4 Getter/Setter类型生成

```typescript
// 自动生成Getter类型
type Getters<T> = {
  [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K]
}

// 自动生成Setter类型
type Setters<T> = {
  [K in keyof T as `set${Capitalize<string & K>}`]: (value: T[K]) => void
}

interface Person {
  name: string
  age: number
  email: string
}

type PersonGetters = Getters<Person>
// {
//   getName: () => string
//   getAge: () => number
//   getEmail: () => string
// }

type PersonSetters = Setters<Person>
// {
//   setName: (value: string) => void
//   setAge: (value: number) => void
//   setEmail: (value: string) => void
// }

// 组合Getters和Setters
type PersonAccessors = PersonGetters & PersonSetters
```

**OpenCode实际应用：**

```typescript
// 来自 packages/opencode/src/tool/tool.ts
export namespace Tool {
  // 工具名称前缀
  type ToolPrefix = "mcp__" | "plugin__" | "builtin__"
  type ToolName = "read" | "write" | "bash" | "edit"

  // 生成带前缀的工具名
  type PrefixedTools = `${ToolPrefix}${ToolName}`
  // "mcp__read" | "mcp__write" | "mcp__bash" | "mcp__edit" |
  // "plugin__read" | "plugin__write" | ...
}

// 来自 packages/opencode/src/session/message-v2.ts
export namespace MessageV2 {
  // 事件名称生成
  type PartType = "text" | "tool" | "file"
  type PartEvent = `part.${PartType}.${string}`
  // "part.text.created" | "part.tool.started" | "part.file.uploaded" | ...
}
```

### 8.5 路由类型生成

```typescript
// REST API路由类型
type HTTPMethod = "GET" | "POST" | "PUT" | "DELETE"
type Resource = "users" | "posts" | "comments"

type APIRoute = `/${Resource}` | `/${Resource}/:id`
// "/users" | "/users/:id" | "/posts" | "/posts/:id" | "/comments" | "/comments/:id"

type APIEndpoint = `${HTTPMethod} ${APIRoute}`
// "GET /users" | "POST /users" | "GET /users/:id" | ...

// 更复杂的路由系统
type RouteParams<T extends string> =
  T extends `${infer _Start}:${infer Param}/${infer Rest}`
    ? { [K in Param | keyof RouteParams<Rest>]: string }
    : T extends `${infer _Start}:${infer Param}`
      ? { [K in Param]: string }
      : {}

type UserRoute = "/users/:userId/posts/:postId"
type Params = RouteParams<UserRoute>
// { userId: string; postId: string }
```

### 8.6 CSS类型生成

```typescript
// CSS属性类型
type CSSUnit = "px" | "em" | "rem" | "%"
type CSSValue<T extends number> = `${T}${CSSUnit}`

type Width = CSSValue<100>  // "100px" | "100em" | "100rem" | "100%"

// Tailwind风格的类名
type Size = "sm" | "md" | "lg" | "xl"
type Color = "red" | "blue" | "green"
type Shade = "100" | "200" | "300" | "400" | "500"

type TextSize = `text-${Size}`
type BgColor = `bg-${Color}-${Shade}`
type ClassName = TextSize | BgColor
// "text-sm" | "text-md" | ... | "bg-red-100" | "bg-red-200" | ...
```

---

## 总结与最佳实践

### TypeScript学习路径

根据本教程的内容，推荐的学习路径：

- 掌握基本类型系统
- 理解接口和类型别名
- 学会函数类型定义
- 深入理解泛型
- 掌握联合类型和交叉类型
- 学会使用类型守卫
- 掌握条件类型和infer
- 理解映射类型
- 学会模板字面量类型
- 阅读优秀开源项目（如OpenCode）
- 编写类型安全的代码
- 构建自己的类型工具库

### OpenCode中的TypeScript模式

通过分析OpenCode源码，我们可以学到以下模式：

#### 1. 命名空间组织代码

```typescript
// 使用命名空间组织相关类型和函数
export namespace Tool {
  export interface Info { /* ... */ }
  export type Registry = Record<string, Info>
  export function get(name: string): Info | undefined { /* ... */ }
}

export namespace Session {
  export interface Info { /* ... */ }
  export type Status = "idle" | "busy"
  export async function create(): Promise<Info> { /* ... */ }
}
```

#### 2. 可辨识联合类型

```typescript
// 使用type字段作为判别式
export type Part =
  | { type: "text"; text: string }
  | { type: "tool"; tool: string; state: ToolState }
  | { type: "file"; url: string; filename: string }

// TypeScript自动收窄类型
function processPart(part: Part) {
  switch (part.type) {
    case "text":
      return part.text  // TypeScript知道有text属性
    case "tool":
      return part.tool  // TypeScript知道有tool属性
  }
}
```

#### 3. 泛型约束和推断

```typescript
// 使用泛型约束和infer推断类型
export type InferParameters<T extends Info> =
  T extends { parameters: infer P } ? P : never

export type InferResult<T extends Info> =
  T extends { execute: (...args: any[]) => infer R } ? R : never
```

#### 4. 类型安全的配置

```typescript
// 使用Zod进行运行时验证和类型推断
export const Info = z.object({
  id: z.string(),
  name: z.string(),
  status: z.enum(["idle", "busy"])
})

export type Info = z.output<typeof Info>
```

### 常见陷阱和解决方案

1. **过度使用any**

   - ❌ 错误：`let data: any = fetchData()`
   - ✅ 正确：`let data: unknown = fetchData()` 然后进行类型检查
2. **忽略null/undefined**

   - ❌ 错误：`user.address.city`
   - ✅ 正确：`user.address?.city` 或先检查
3. **类型断言滥用**

   - ❌ 错误：`data as User` 不检查
   - ✅ 正确：使用类型守卫验证后再使用
4. **复杂类型难以理解**

   - ❌ 错误：一行写完所有逻辑
   - ✅ 正确：拆分成多个中间类型，添加注释

### 推荐资源

基于本教程引用的资料：

1. **官方文档**

   - [TypeScript官方手册](https://www.typescriptlang.org/docs/)
   - [TypeScript Playground](https://www.typescriptlang.org/play)
2. **进阶学习**

   - [TypeScript高级特性指南](https://leapcell.io/blog/mastering-typescript-generics-conditions-mappings-and-inference)
   - [映射类型和模板字面量](https://medium.com/@ignatovich.dm/typescript-and-advanced-type-manipulation-mapped-types-template-literal-types-and-more-cc22533bc15d)
   - [TypeScript 2025新特性](https://www.codertrove.com/articles/typescript-2025-whats-new)
3. **实战项目**

   - [OpenCode源码](https://github.com/anthropics/opencode) - 学习世界级TypeScript代码
   - [Total TypeScript](https://www.totaltypescript.com/) - 系统化学习

### 结语

TypeScript已经成为现代前端开发的标准。根据[GitHub 2025报告](https://www.programming-helper.com/tech/typescript-2026-number-one-github-ai-typed-languages-python)，TypeScript已经超越Python和JavaScript，成为最受欢迎的编程语言。

掌握TypeScript不仅能提高代码质量，还能：

- **提前发现错误**：在编译时而非运行时
- **提升开发效率**：更好的IDE支持和自动补全
- **改善代码可维护性**：类型即文档
- **增强团队协作**：类型约束减少沟通成本

希望这份教程能帮助你从入门到精通TypeScript，编写出像OpenCode一样优秀的代码！

---

**Sources:**

- [TypeScript in 2025: New Features, ESM &amp; AI-Driven Dev](https://www.codertrove.com/articles/typescript-2025-whats-new)
- [TypeScript 2026: How It Became #1 on GitHub](https://www.programming-helper.com/tech/typescript-2026-number-one-github-ai-typed-languages-python)
- [Mastering TypeScript Generics](https://leapcell.io/blog/mastering-typescript-generics-conditions-mappings-and-inference)
- [Advanced Typing in TypeScript with Generics](https://www.rst.software/blog/advanced-typing-in-typescript-with-generics)
- [TypeScript and Advanced Type Manipulation](https://medium.com/@ignatovich.dm/typescript-and-advanced-type-manipulation-mapped-types-template-literal-types-and-more-cc22533bc15d)
- [Template Literal Types Tutorial](https://krython.com/tutorial/typescript/template-literal-types-string-manipulation-at-type-level)
