---
title: python脚本写法
date: 2025-03-11T10:54:27.000Z
tags: [脚本]
category: 自用
comments: true
draft: false
---

## python脚本写法

这里并不注重于一些比较专门的用法，大部分介绍的是这些用法的前置知识，当然会有所提及

是我的python基础太差了才需要这篇blog的

### what you need

- 大佬不必看
- 你需要有浏览器引擎
- 了解命令行有python环境(Anaconda)
- 有偷懒的需求

### 字符串操作

当时逃课了，自己又懒得看QAQ

### argparse

首先当然是命令行怎么玩

```python
import argparse
def main(args):
    pass
if __name__="__main__":
    paser=argparse.ArgumentParser()#可以加个description
    parser.add_argument('-d','--directory',help='输入目录')
    parser.add_argument('-n','--number',help='输入数字')#require=True,type=int这些条件都可以假如
    args=parser.parse_args()
#运行就是python script.py -d <directory_path> -n <number>
```

因为是自己用，所以参数的限泛用性可以比较低

### 拋异常

```python
    try:
        main(args)
    except Exception as e:
        print(f"错误发生: {str(e)}")
        sys.exit(1)
```

懒得多写，反正给个报错退出就好

### 文件操作

#### os库

```python
import os

# 路径操作
current_dir = os.getcwd()  # 获取当前目录
os.chdir('/path')          # 切换目录

# 文件操作
os.rename('old.txt', 'new.txt')  # 重命名
os.remove('file.txt')            # 删除文件
os.stat('file.txt')              # 获取文件属性

# 目录操作
os.makedirs('dir1/dir2', exist_ok=True)  # 递归创建目录
os.listdir('.')                          # 列出目录内容

# 路径判断
os.path.exists('/path')        # 路径是否存在
os.path.isfile('file.txt')     # 是否是文件
os.path.isdir('directory')     # 是否是目录
#路径生成
os.path.join('des','name')
```

#### pathlib库

```python
from pathlib import Path

input_dir = Path(args.directory)
if not input_dir.is_dir():
    raise ValueError("无效目录路径")

# 获取所有.py文件
files = list(input_dir.glob("*.py"))
```

#### shutil库

```python
import shutil

# 文件复制
shutil.copy('src.txt', 'dst.txt')     # 复制文件
shutil.copy2('src.txt', 'dst.txt')    # 保留元数据

# 目录操作
shutil.copytree('src_dir', 'dst_dir')  # 递归复制目录
shutil.rmtree('directory')             # 递归删除目录
shutil.move('data.csv','C:/数据分析')
# 压缩打包
shutil.make_archive('backup', 'zip', 'src_dir')
```

#### 批量化处理

遍历！！！批量化处理的关键

```python
# 使用os.walk遍历目录树
for root, dirs, files in os.walk('.'):
    for file in files:
        path = os.path.join(root, file)
        if(path.endwith('.csv'))#筛选
        	print(path)

import glob
pdf_files = glob.glob('**/*.pdf', recursive=True)
```

```python
path = '/home/user/docs/report.pdf'

os.path.basename(path)   # 'report.pdf'
os.path.dirname(path)    # '/home/user/docs'
os.path.splitext(path)   #('/home/user/docs/report', '.pdf')
os.path.join('dir', 'sub', 'file.txt')  # 'dir/sub/file.txt'
```

```python
# 批量重命名示例
for filename in os.listdir('.'):
    if filename.endswith('.jpg'):
        new_name = filename.replace(' ', '_').lower()
        os.rename(filename, new_name)

# 使用正则表达式处理文件名
import re

pattern = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
for filename in os.listdir('.'):
    match = pattern.search(filename)
    if match:
        new_name = f"{match.group(1)}{match.group(2)}{match.group(3)}.txt"
        os.rename(filename, new_name)
```

### 舒适区操作

#### tqdm

```python
from tqdm import tqdm

with Pool(args.num_processes) as pool:
    results = list(tqdm(pool.imap(safe_run, files), total=len(files)))
```

#### 正则表达式

我不会

#### Selenium

获取html

```python
from selenuim import webdriver

brower=webdriver.Edge()
brower.get('https://www.taobao.com')
print(browser.page_source)#获取html
browser.close()
```

查找单个元素

```python
input_first = browser.find_element_by_id('q')
input_second = browser.find_element_by_css_selector('#q')
input_third = browser.find_element_by_xpath('//*[@id="q"]')
print(input_first,input_second,input_third)
browser.close()
```

元素获取

```python
find_element_by_name
find_element_by_xpath
find_element_by_link_text
find_element_by_partial_link_text
find_element_by_tag_name
find_element_by_class_name
find_element_by_css_selector
```

```python
#基本定位方法
# ID
element = driver.find_element(By.ID, 'element-id')
# 类名
element = driver.find_element(By.CLASS_NAME, 'class-name')
# 标签名
element = driver.find_element(By.TAG_NAME, 'div')
# 名称
element = driver.find_element(By.NAME, 'username')
# 链接文本
element = driver.find_element(By.LINK_TEXT, 'Click Here')
# 部分链接文本
element = driver.find_element(By.PARTIAL_LINK_TEXT, 'Click')
#复制
# 单个元素
element = driver.find_element(By.CSS_SELECTOR, 'div.main > input[type="text"]')
# 多个元素
elements = driver.find_elements(By.CSS_SELECTOR, '.item')
#XPath定位
# 绝对路径
element = driver.find_element(By.XPATH, '/html/body/div[1]/form/input')
# 相对路径
element = driver.find_element(By.XPATH, '//input[@name="username"]')
# 包含文本
element = driver.find_element(By.XPATH, '//a[contains(text(), "Login")]')
```

模拟用户操作

```python
# 点击
element.click()
# 输入文本
element.send_keys('text')
# 清除内容
element.clear()
# 获取文本
text = element.text
# 获取属性
value = element.get_attribute('value')
# 鼠标悬停
from selenium.webdriver.common.action_chains import ActionChains
actions = ActionChains(driver)
actions.move_to_element(element).perform()
# 拖放操作
source = driver.find_element(By.ID, 'source')
target = driver.find_element(By.ID, 'target')
actions.drag_and_drop(source, target).perform()
# 键盘操作
from selenium.webdriver.common.keys import Keys
element.send_keys(Keys.CONTROL + 'a')
element.send_keys(Keys.BACKSPACE)
```

之后再写吧，困了

#### Scrapy

```shell
scrapy startproject mySpider
```

然后就会有一个文件夹

```shell
mySpider/
    scrapy.cfg
    mySpider/
        __init__.py
        items.py
        pipelines.py
        settings.py
        spiders/
            __init__.py
            ...
```

在spiders自己创建一个python文件，这里是zhihu_spider.py

```python
import scrapy
class zhihu_spider(scrapy.Spider):
	name='zhihu_spdier'
    allowed_domains=['xxx.com']#域名
    start_urls=['https://www.xxx.com']
    def parse(self,response):
        for xx in response.css("xxx")#提取每个xxx的信息
        	yield{
                'xxx':xx.css('xxx').get()#xxx的具体信息，用css来解码
            }
```

配置，在settings.py

```python
# 启用日志记录
LOG_ENABLED = True
LOG_LEVEL = 'INFO'

# 设置下载延迟，避免对服务器造成过大压力
DOWNLOAD_DELAY = 1

# 启用User-Agent池
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
```

然后就可以运行了

```shell
scrapy crawl zhihu_spider
```

后续一般会把需要的数据导入到csv里面只需要在yield位置修改为pandas或者其他的数据处理的库放入新建的文件就好，创建文件等步骤上文有提及

#### Playwright

之后再学，累了

下面的链接可以练练手

[20个Python 非常实用的自动化脚本 - 虾米哟 - 博客园](https://www.cnblogs.com/zlibraryxiayu/p/18763505)
