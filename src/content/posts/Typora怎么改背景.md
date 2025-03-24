---
title: typora改背景
date: 2025-03-23T10:54:27.000Z
tags: [前端]
category: 自用
comments: true
draft: false
---

## Typora怎么改背景

我忍这个无聊的背景很久了

首先打开左上角文件，偏好设置，选择外观，打开主题文件夹，然后你就到了主题文件夹

在文件资源管理器，如果你用的是github主题，那你就去改github.css就好，其他主题同理

然后用下面的代码

```css
#write {
  background-image: url(./image/1.png);
  height: 100%;
  width: 100%;
  overflow: scroll;
  background-size: cover;
}
```

background-image的url如果不会相对路径用绝对路径就好，用上面这个，需要你在github.css的同级目录下新建一个image的文件夹，然后把你喜欢的图片放进去

修改css文件是一个让人开心的事，因为改完真的会让人心情愉悦~

重新打开typora就好了

哦对了，如果需要不收费的typora以前的版本.exe可以找我哦，不过应该没人需要吧~~~
