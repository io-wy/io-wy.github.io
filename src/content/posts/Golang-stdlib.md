---
title: golang-stdlib
date: 2026-02-22T10:54:27.000Z
tags: [backend]
category: 自用
comments: true
draft: false

---

有点晚了，没有太多想法，就多写点东西吧

## 错误处理

go的错误处理是一种哲学，error is value！

**调用下游服务 → error 包装**

err的本质

```go
type error interface {
    Error() string
}
```

大多设计一下场景的api都会有err的判断（虽然大多时候我会交给lsp和编译器来提醒我QAQ）

| 场景       | 是否应该返回 error |
| ---------- | ------------------ |
| 文件操作   | 必须               |
| 网络请求   | 必须               |
| JSON 解析  | 必须               |
| 数据库查询 | 必须               |
| 类型转换   | 必须               |
| IO 读写    | 必须               |

## defer

在golang中，defer可以给编程的过程带来很大的便利，一些操作可以使用defer来防止不及时关闭，例如互斥锁，文件，数据库连接等

**defer：函数返回前执行（延迟执行）常用于资源释放**

以文件操作作为例子

```go
func main(){
	file, err := os.Open("test.txt")
	if err := nil{
		log.Println(err)
		return
	}
	defer file.Close()
	//...
}
```

然而，file.Close()是有可能返回err的，但是我们并没有处理，虽然我们把file.Close()放到最后执行，也不在乎他是不是返回错误，但是重要的不是错误本身，而是**file没有按照我们的想法关闭**，但是由于写在defer里面，我们无法处理错误；

在特殊情况我们想要进行处理，应该这么写defer

```golang
 func fileClose() (err error) {
      file, err := os.Open("test.txt")
     if err != nil{
         return//处理的是打开的错误
     }
      defer func() {
          closeErr := file.Close()
          if err == nil {  // 只有原函数没错误时才记录 close 错误
              err = closeErr//处理的是关闭的错误
          }
      }()
      return
 }

```

首先我们需要理解一下闭包：函数+它引用的外部变量，这个匿名函数func()使用到了file和err，这两个都来自外层函数fileClose，这里的err是命名返回值，它在整个函数栈帧里是同一个变量，defer执行时修改的是同一个变量；完整逻辑如下

**关闭文件 -> 如果原函数没有错误，才把close的错误赋值给err**

这样可以达到目的，如果os.Open已经报错，那file可能是nil，那就不会覆盖原始错误（比如不存在test.txt）

### 闭包

闭包很值得在意的是，函数会携带状态

```go
func counter() func() int {
    count := 0

    return func() int {
        count++
        return count
    }
}
```

使用

```go
c := counter()

fmt.Println(c()) // 1
fmt.Println(c()) // 2
fmt.Println(c()) // 3
```

闭包拿到的是count的地址，而不是值，因此每次都会读取这个地址的值，继续增加，对于编译器来说，会把闭包变成这样

```go
type closure struct {
    x *int
}//闭包用到的原函数的地址
func (c *closure) call() {
    fmt.Println(*c.x)
}//匿名函数
```

当闭包捕获变量的时候，这个变量通常会逃逸到堆上，因为函数返回变量后还要活着，如果还在栈上的话就会回收了，这里我们还能考虑到一个大坑

```go
func main() {
    for i := 0; i < 3; i++ {
        go func() {
            fmt.Println(i)
        }()
    }
}
```

以为输出012，实际上都是3，因为并发执行的匿名函数是以闭包的方式操作的，所以里面的go func操作的其实是同一个值，当然每次打印出来都一样啦

如果要写的话可以这么做

```go
for i := 0; i < 3; i++ {
    go func(i int) {
        fmt.Println(i)
    }(i)//这是是调用，立即执行函数（相当于创建了一个新的i）
}
```

#### 闭包工程应用

构造私有状态

```go
func NewUser(name string) func() string {
    return func() string {
        return name
    }
}
```

Web框架的中间件

```go
func logger(next Handler) Handler {
    return func(ctx Context) {
        fmt.Println("before")
        next(ctx)
        fmt.Println("after")
    }
}
```

上面提到的延迟执行

```go
defer func() {
    fmt.Println(x)
}()
```

函数工厂

```go
func multiply(factor int) func(int) int {
    return func(x int) int {
        return x * factor
    }
}
```

本质上闭包就是让函数拥有状态（最呆的理解方式我是直接当作地址，指针，堆栈来接受这个逻辑的）

## 文件读取

**文件名直接读取**

```go
func fileOne() {
	content, err := os.ReadFile("test.txt")
	if err != nil {
		panic(err)
	}
	fmt.Println(string(content))
}
```

**先创建文件句柄再读取文件**

```go
func fileThree() {
	file, err := os.Open("test.txt")
	if err != nil {
		panic(err)
	}
	defer file.Close()
 
	content, err := ioutil.ReadAll(file)
	fmt.Println(string(content))
}
```

以只读的方式打开文件

**按行读取文件**

主要操作使用ioutil库，其中两个方法可以实现按行读取，但是要注意，真的按行读取时根据\n来区分每一行的，如果是没有分行的大文件，就不能使用按行读取了( **注意trim**！)

```go
func fileSix() {
	// 创建文件句柄
	fi, err := os.Open("test.txt")
	if err != nil {
		panic(err)
	}
	defer fi.Close()
 
	// 创建reader
	r := bufio.NewReader(fi)
 
	for {
		line, err := r.ReadString('\n')
		line = strings.TrimSpace(line)//注意trim
		if err != nil && err != io.EOF {
			panic(err)
		}
		if err == io.EOF {
			break
		}
		fmt.Println(line)
	}
}
```

**按字节读取**

对于不分行的大文件，只能按字节来读取整个文件（使用os库）（也可以使用syscall库）

```go
func fileSeven() {
	// 创建文件句柄
	fi, err := os.Open("test.txt")
	if err != nil {
		panic(err)
	}
	defer fi.Close()
 
	// 创建reader
	r := bufio.NewReader(fi)
 
	// 每次读取1024个字节
	buf := make([]byte, 1024)
	for {
		n, err := r.Read(buf)
		if err != nil && err != io.EOF {
			panic(err)
		}
		if n == 0 {
			break
		}
		fmt.Println(string(buf[:n]))
	}
}
```

总结一下

```go
  // 1. 小文件直接读完
    data, err := os.ReadFile("a.txt")
    fmt.Println(string(data))

  // 2. 逐行读（推荐）
    file, _ := os.Open("a.txt")
    defer file.Close()

    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        fmt.Println(scanner.Text())
    }

  // 3. 大文件流式处理
    reader := bufio.NewReader(file)
    buf := make([]byte, 1024)

    for {
        n, err := reader.Read(buf)
        if err == io.EOF {
            break
        }
        fmt.Println(string(buf[:n]))
    }
```

完整的结构是：打开文件（os） ->  读写（file/string）-> 字符处理（string） -> 关闭文件

常用api

```go
os.Open
os.Create
os.OpenFile
os.ReadFile
os.WriteFile
file.Read
file.Write
bufio.NewScanner
scanner.Scan
strings.Split
strings.TrimSpace
```

## 命令行交互

这块很简单，快速结束

```go
func main() {
	wordPtr := flag.String("word", "Jude", "a string")
	numPtr := flag.Int("numb", 42, "an int")
	boolPtr := flag.Bool("fork", false, "a bool")
 
	var svar string
	flag.StringVar(&svar, "svar", "bar", "a string var")
 
	flag.Parse()
 
	fmt.Println("word: ", *wordPtr)
	fmt.Println("numb: ", *numPtr)
	fmt.Println("fork: ", *boolPtr)
	fmt.Println("svar: ", svar)
	fmt.Println("tail: ", flag.Args())
}
```

终端跑一下demo

```bash
$ go run main.go -word=opt -numb=7 -fork -svar=flag
 
word:  opt
numb:  7
fork:  true
svar:  flag
tail:  []
```

被忽略的是flag就是默认值

## 类型转换

`strconv` = string convert

**string → int**

```go
i, err := strconv.Atoi("123")
```

等价于：

```go
i, err := strconv.ParseInt("123", 10, 0)
```

**ParseInt 原型**

```go
func ParseInt(s string, base int, bitSize int) (i int64, err error)
```

参数说明：

- base：进制（2~36）
- bitSize：8 / 16 / 32 / 64

例子：

```go
i, err := strconv.ParseInt("FF", 16, 64)
```

**int/float/bool → string**

```
s := strconv.Itoa(123)
s := strconv.FormatInt(int64(123), 10)

f, err := strconv.ParseFloat("3.14", 64)
s := strconv.FormatFloat(f, 'f', 2, 64)

b, err := strconv.ParseBool("true")
```

## 字符串

由于string**底层**是

```go
type string struct {
    ptr *byte
    len int
}
```

只读字节数组，因此string是**不可变**的

string作为utf8因此在**遍历**上有写小问题

```go
s := "你好"
fmt.Println(len(s))//输出为6
//因此遍历需要采用int32（rune来做）
for i, r := range s {
    fmt.Println(i, r)
}
```

 **string 和 []byte 转换**

```go
b := []byte(s)
s2 := string(b)
```

这会发生拷贝！！注意一下

**常用函数**

```go
strings.Split
strings.Join
strings.Trim
strings.Contains
strings.HasPrefix
strings.Builder
```

高性能拼接（比直接`+`有用）

```go
var b strings.Builder
b.WriteString("hello")
b.WriteString(" world")
s := b.String()
```













