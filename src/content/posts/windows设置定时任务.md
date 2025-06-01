---
title: windows设置定时任务
date: 2025-06-02T10:54:27.000Z
tags: [脚本]
category: 教程
comments: true
draft: false
---

## windows设置定时任务

经常爬虫完获取资讯，或者自动登录校园网，手动操作实在麻烦，因此这里设置定时任务的操作就不得不掌握了

### 设定

此处以登录校园网为例

```powershell
schtasks /create /tn "MyPythonTask" /tr "python C:\scripts\myscript.py" /sc once /st HH:mm
```

这是模板样例

```powershell
schtasks /create /tn "Net_LoginTask" /tr "python C:\Users\xxx\Desktop\temp\Net-login-main\Net-login-main\Net-login-CN.py" /sc daily /st 07:05
```

这是我自己的操作

### 检查

```powershell
schtasks /query /tn "Net_LoginTask"
```

### 删除

后来我们都不需要了

```powershell
schtasks /delete /tn "Net_LoginTask" /f
```
