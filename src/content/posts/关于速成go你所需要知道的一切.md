---
title: 关于速成go你所需要知道的一切
date: 2025-11-25T10:54:27.000Z
tags: [后端]
category: 自用
comments: true
draft: false
---

### Step 0 开发环境配置

自行处理，懒得截图

### Step 1 语言入门

```go
package main
import "fmt"

func main(){fmt.Println("Hello Golang")}
```

#### Go的关键字和保留字

关键字

```
    break        default      func         interface    select
    case         defer        go           map          struct
    chan         else         goto         package      switch
    const        fallthrough  if           range        type
    continue     for          import       return       var
```

保留字

```
    Constants:    true  false  iota  nil

    Types:    int  int8  int16  int32  int64
              uint  uint8  uint16  uint32  uint64  uintptr
              float32  float64  complex128  complex64
              bool  byte  rune  string  error

    Functions:   make  len  cap  new  append  copy  close  delete
                 complex  real  imag
                 panic  recover
```

#### 声明

go是静态类型语言，有固定的数据类型，有如下四种声明方式

```go
    var（声明变量）, const（声明常量）, type（声明类型） ,func（声明函数）。
```

##### 变量

声明变量有简短模式，在函数内部使用

```go
package main
import "fmt"
func main() {
  x := 10 // 使用 := 进行定义并初始化
  fmt.Println(x) // 输出语句 10
```

：=自行推导变量类型，因此不需要提供数据类型

多变量可以如下操作

```go
x,y = 20,10
x, y = y+3,x+3
```

##### 常量

常量是编译期就以哦经确定的字符串，数字或bool，编译器会进行初始化推断

```go
const x,y int = 20,10
const a,b string= "hello","world"
const(
	a,b string = "hello","world"
	x,y = 10,20
)
```

##### 运算 函数 流程控制

和其他语言基本一样，看代码就知道

```go
func add(a int,b int){return a+b}
func add_self(a int){return a++}
func zero_judge(a int)(result string){
	if a == 0{
		return "是0""
	}else x>0 {
		return "正数"
	}else{return "负数"}
}
func main(){
    for i:=0;i<5;i++{
        if i==4 {
            continue
        }else if i==5{
            break
        }
        fmt.Println(i)
	}
```

###### 流程控制

#### 类型

分为值类型和引用类型 ,基本类型，数组，字符串基本都是属于值类型，引用类型主要包括，slice，map，channel， interface，function

本质上是，引用类型不直接存储数据本身，而是存储指向数据的指针

- 切片（Slices）：切片是对数组的封装，提供了一个灵活、动态的视图。当修改切片中的元素时，实际上是在修改底层数组的相应元素。
- 映射（Maps）：映射是一种存储键值对的集合。将映射传递给一个函数或者赋值给另一个变量时，任何对映射的修改都会反映在所有引用了这个映射的地方。
- 通道（Channels）：通道用于在不同的 goroutine 之间传递消息。通道本质上是引用类型，当复制或传递它们时，实际上传递的是对通道数据结构的引用。
- 接口（Interfaces）：接口类型是一种抽象类型，定义了一组方法，但不会实现这些方法。接口内部存储的是指向实现了接口方法的值的指针和指向该类型信息的指针。
- 函数（Functions）：在 Go 中，函数也是一种引用类型。当把一个函数赋给另一个变量时，实际上是在复制一个指向该函数的引用。

稍微提一下，和C语言一样，值传入函数是fork了一个副本所以不会改变原来的量，但是引用类型是会的

##### slice

```go
package main
import "fmt"
func main(){
    //这是切片
    a:=make([]int,5)//slice初始化方式
    for i:=0;i<5,i++{
        a = append(a,i)
    }
    fmt.Println(a) // [0 0 0 0 0 0 1 2 3 4]
    //这是数组
    var a []int
    for i := 0; i < 5; i++ {
       a = append(a, i)
    }
    fmt.Println(a) // [0 1 2 3 4]
}
```

slice是把数组的内存截除make第二个参数“5”

##### map

```go
package main
import "fmt"
func main() {
   // 定义 变量strMap
   var strMap map[int]string
   // 进行初始化
   strMap = make(map[int]string)
   // 给map 赋值
   for i := 0; i < 5; i++ {
      strMap[i]  = "迈莫coding"
   }
   // 打印出map值
   for k, v := range strMap{
      fmt.Println(k, ":", v)
   }
  // 打印出map 长度
  fmt.Println(len(strMap))
```

map和py种的map非常接近

##### sturct

有点类似java中的类，struct把多个不同类型的字段打包整合成一个符合类型，字段大多是基本数据类型

```go
type user struct{
	name string
	age int
}
//有点像类里面的成员函数，这里是结构体的user的read方法
func (u *user) Read string{
    return fmt.Sprintf("%s已经 %d岁了"，u.name,u.age)
}
func main(){
    u :=&user{
        name :"io",
        age =0,
    }
    fmt.Println(u.name,"已经"，u.age,"岁了")
    fmt.Println(u.Read())
}
```

##### interface

接口是为多个struct or 模板做好了一个统一的规定范式，需要遵守的，对可维护性非常有帮助

你创建符合接口的东西之后，你可以添加其他相关的字段/函数，但是不能不是下你接口的东西

```go
package main

import (
        "fmt"
        "math"
)
// 定义接口
type Shape interface {
        Area() float64
        Perimeter() float64
}
// 定义一个结构体
type Circle struct {
        Radius float64
}

// Circle 实现 Shape 接口
func (c Circle) Area() float64 {
        return math.Pi * c.Radius * c.Radius
}
func (c Circle) Perimeter() float64 {
        return 2 * math.Pi * c.Radius
}
func main() {
        c := Circle{Radius: 5}
        var s Shape = c // 接口变量可以存储实现了接口的类型
        fmt.Println("Area:", s.Area())
        fmt.Println("Perimeter:", s.Perimeter())
}
```

### Step2 快速进入真实场景

#### 并发编程

#### http 网络编程

http服务端

```go
package main
import(
	"fmt"
    "io"
	"net/http"
)
func main(){
    resp,_ = http.Get("http://127.0.0.1:8000/go")
    defer resp.Body.Close()
    fmt.Println(resp.Status)
    fmt.Println(resp.Header)

    buf :=make([]byte,1024)
    for{
        n,err : = resp.Body.Read(buf)
        if err : = nil && err != io.EOF{
            fmt.Println(err)
            return
        } else{
            fmt.Println("读取完了")
            res := string(buf[:n])
            fmt.Println(res)
            break
        }
    }
}
```

#### websocket

### Step3 CRUD需要

#### web框架

##### gin

![image-20251129201904854](D:%5C0%20program%5Cblog%5Cio-wy.github.io%5Csrc%5Ccontent%5Cposts%5Cimage-20251129201904854.png)

优势说完了，开始说怎么用

###### Quick Start

```go
package main
import(
	"net/http"
	"github.com/gin-gonic/gin"
)
func main(){
	r:=gin.Default()
	r.GET("/ping",func(c *gin.Context){
		c.JSON(http.StatusOK,gin.H{
			"message":"pong",
		})
	})
	r.Run()
}//默认监听localhost:8000
```

###### Build Your Program

![image-20251129202747594](D:%5C0%20program%5Cblog%5Cio-wy.github.io%5Csrc%5Ccontent%5Cposts%5Cimage-20251129202747594.png)

然后这个时候你就应该找到一个项目，开始copy on writing

##### kratos

##### go-zero

#### mq（消息队列）

#### redis

#### gorm（数据库）

**gorm**可以直接理解成go写的orm，orm定义如下：Object-Relationl Mapping， 它的作用是映射数据库和对象之间的关系，方便我们在实现数据库操作的时候不用去写复杂的sql语句，把对数据库的操作上升到对于对象的操作。

##### 上手gorm

go.mod添加

```go
require{
	github.com/jinzhu/gorm v1.9.12
}
```

##### 创建DB连接，然后开始CRUD

```go
package main
import(
	"github.com/jinzhu/gorm"
	_"github.com/jinzhu/gorm/dialects/mysql"
)
func main(){
	var err error
	db,connErr :=gorm.Open("mysql",xxxx)
    if connErr != nil {
        panic("failed to connect database")
    }
    defer db.Close()
  db.SingularTable(true)
}
```

gorm支持很多数据库，比如说PostgreSQL，MYSQL等

创建映射表的struct

```go
CREATE TABLE `test` (
  `id` bigint(20) NOT NULL,
  `name` varchar(5) DEFAULT NULL,
  `age` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
type Test struct {
    ID   int64  `gorm:"type:bigint(20);column:id;primary_key"`
    Name string `gorm:"type:varchar(5);column:name"`
    Age  int    `gorm:"type:int(11);column:age"`
}
```

然后就是增删改查

```go
package main

import (
	"fmt"
	"github.com/jinzhu/gorm"
	_ "github.com/jinzhu/gorm/dialects/mysql"
)

// UserInfo 用户信息
type UserInfo struct {
	ID uint
	Name string
	Gender string
	Hobby string
}
func main() {
	db, err := gorm.Open("mysql", "root:root1234@(127.0.0.1:13306)/db1?charset=utf8mb4&parseTime=True&loc=Local")
	if err!= nil{
		panic(err)
	}
	defer db.Close()
	// 自动迁移
	db.AutoMigrate(&UserInfo{})
	u1 := UserInfo{1, "七米", "男", "篮球"}
	u2 := UserInfo{2, "沙河娜扎", "女", "足球"}
	// 创建记录
	db.Create(&u1)
	db.Create(&u2)
	// 查询
	var u = new(UserInfo)
	db.First(u)
	fmt.Printf("%#v\n", u)
	var uu UserInfo
	db.Find(&uu, "hobby=?", "足球")
	fmt.Printf("%#v\n", uu)
	// 更新
	db.Model(&u).Update("hobby", "双色球")
	// 删除
	db.Delete(&u)
}
```

### Step4 what do u need （进阶）

##### GMP

##### 并发底层实现（golang）
