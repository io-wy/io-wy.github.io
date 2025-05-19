---
title: C++系统编程（一）
date: 2025-05-19T10:54:27.000Z
tags: [C++]
category: 后端
comments: true
draft: false
---

## C++系统编程（一）

​ 给自己挖坑来了，相对于其他的语言，c++更接近系统，作为一个清高的理想主义者，不愿意变成“别人”眼中的码农，至少兼修一部分engineer，虽然这段时间也挺迷茫的，我的ai还能跟着世界走多远呢...

### 建立基本概念

​ 首先系统，自然是我们所知道的操作系统，基本的了解情况就是到，系统编程就是应用层和系统层面进行交互，某种程度上来说，事得到更多的权限来做更多的事

​ 既然更贴近系统，那自然复杂性肯定是远远增加的。

​ 掌握的技术分块：文件系统，内存管理，线程管理，系统调用，文件IO...

### 多线程的操作

​ 相对于文件内存IO等在操作系统也会学到的知识，个人认为还是多线程更贴近主题，因此就从这开始了

#### 基本概念

并发：多个任务交替执行，宏观上表现出同时进行的效果

并行：多个任务在不同处理器同时执行

#### 线程创建

```c++
#include<thread>
std::thread sth(func,*args);//func为可调用的对象，*arg是func的参数
```

std::thread是关键字，定义线程的，sth是线程名

##### func引申

函数指针

```cpp
void print(int num){cout<<num<<endl;}
int main()
{	thread thread1(print,1);//这里&print也是可以的
	thread1.join()；//线程创建了，输出1
	return 0;
}
```

lambda表达式

```cpp
int main(){
	thread thread2([](int num)){cout<<num<<endl;},1);
    thread2.join();
    return 0;
}
```

这里分析一下lambda吧，[]可以假装看成函数名，num为函数应该有的参数，{}里面就是函数主体了，后面的1是线程写法里面传入func的arg

非静态成员函数

```cpp
class print{
	void print1(int num){cout<<num<<endl;}
}
int main(){
	print a;//创建类对象
	thread thread3(&print::print1,&a,1);
	thread3.join();
	return 0;
}
```

值得思考的为什么是&print::print；后面域的符号就是调用成员函数，&a为什么需要引用呢；有点C基础的味道了

静态成员函数
