# 京东评论爬虫及数据转换项目

本项目用于从京东指定商品页面爬取用户评论数据，并将数据保存为 JSON 文件。同时提供独立的 Python 脚本，将 JSON 数据转换为 Excel 文件，方便后续数据分析和处理。

**作者：lalalala**
**北京邮电大学正大杯 2025 参赛使用**

---

## 第一部分：京东评论爬取

### 目录

- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [使用说明](#使用说明)
  - [1. 环境准备](#1-环境准备)
  - [2. 配置爬虫参数](#2-配置爬虫参数)
  - [3. 运行爬虫](#3-运行爬虫)
  - [4. JSON 转 Excel（备用方案）](#4-json-转-excel备用方案)

### 技术栈

- **Python 3.11+**  
  最新版本的 Python 开发和运行环境。

- **Scrapy**  
  高效的网络爬虫框架，用于爬取京东评论数据。  
  [Scrapy 官网](https://scrapy.org/)

- **Pandas**  
  数据处理库，用于读取 JSON 并转换为 Excel。  
  [Pandas 官网](https://pandas.pydata.org/)

- **openpyxl**  
  Excel 文件写入支持库，与 Pandas 配合导出 Excel 文件。  
  [openpyxl PyPI](https://pypi.org/project/openpyxl/)

- **代理**  
  由于存在多次爬取后 ip 被封禁的可能性，本人申请了 12 小时的免费快代理会员使用代理爬取。  
  [快代理官网](https://www.kuaidaili.com/)

### 项目结构

```plaintext
spider
├── json_to_excel.py
├── output.json
├── output.xlsx
├── scrapy.cfg
└── myproject
    ├── __pycache__
    │   ├── __init__.cpython-311.pyc
    │   └── spider_for_jindong.cpython-311.pyc
    ├── spiders
    │   ├── __pycache__
    │   │   └── （此处可能有自动生成的 .pyc 文件）
    │   └── spider_for_jindong.cpython-311.pyc
    ├── __init__.py
    ├── items.py
    ├── middlewares.py
    ├── pipelines.py
    └── settings.py

```

### 使用说明

#### 1. 环境准备

请确保已安装 Python 3.11 或更高版本，建议在虚拟环境中安装项目依赖。

安装主要依赖：

```bash
pip install scrapy pandas openpyxl
```

#### 2. 配置爬虫参数

在 `spider_for_jindong.py` 文件中，根据实际需求调整以下参数：

- **MAX_PAGES**  
  指定要爬取的评论总页数（从 0 开始计数）。

- **COMMENTS_PER_PAGE**  
  每页爬取的评论数量。

- **PRODUCT_ID**  
  待爬取评论的京东商品 ID。

- **PROXY_ADDRESS**  
  如需使用代理，设置代理地址。

#### 3. 运行爬虫

在 spider 文件夹下下执行以下命令（确保爬虫类中的 `name` 属性为 `jingdong_comment_spider`）：

```bash
scrapy crawl jingdong_comment_spider
```

运行后，会生成：

- `output.json`：存储爬取的评论数据（JSON 格式）

#### 4. JSON 转 Excel

在之前的尝试中，直接使用 Scrapy 导出 Excel 数据存在问题，我重新编写了独立脚本将 JSON 数据转换为 Excel 格式。

确保 `output.json` 文件与 `json_to_excel.py` 脚本位于同一目录下，然后执行：

```bash
python json_to_excel.py
```

脚本会：

- 读取 `output.json` 数据
- 清洗文本数据（去除 Excel 不允许的控制字符）
- 生成 `output.xlsx` 文件

### 注意事项

- **代理设置：** 需要确保 `PROXY_ADDRESS` 配置正确且代理服务器正常运行。
- **Excel 导出：** 部分 Excel 导出库可能存在限制或 Bug，建议升级到最新版本；若有问题，可使用转换脚本进行修正。
- **数据清洗：** 转换脚本内置了文本清洗逻辑，可避免因非法字符导致 Excel 写入错误，但这也导致 json 文件和 excel 文件存在不同。

---
