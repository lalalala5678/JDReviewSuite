# -*- coding: utf-8 -*-
# Scrapy settings for myproject project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = "myproject"

SPIDER_MODULES = ["myproject.spiders"]
NEWSPIDER_MODULE = "myproject.spiders"

# 如果想让爬虫遵守 robots.txt，请改为 True；此处为 False 以便爬取更多页面
ROBOTSTXT_OBEY = False

# 日志级别
LOG_LEVEL = 'DEBUG'

# -----------------------------
# scrapy-selenium 相关配置
# -----------------------------
# 1) 启用 scrapy-selenium 的中间件
DOWNLOADER_MIDDLEWARES = {
    'scrapy_selenium.SeleniumMiddleware': 800,
    # 如有其他自定义中间件，请确保顺序合适
}

# 2) 指定使用 Chrome 作为浏览器驱动
SELENIUM_DRIVER_NAME = 'chrome'

# 3) chromedriver 的实际路径，如果已在系统 PATH，可只写 'chromedriver'
#   Windows 示例: r'C:\path\to\chromedriver.exe'
SELENIUM_DRIVER_EXECUTABLE_PATH = r'C:\path\to\chromedriver.exe'

# 4) Chrome 启动参数
SELENIUM_DRIVER_ARGUMENTS = [
    '--headless',            # 无头模式，可注释掉以便调试
    '--no-sandbox',
    # 设置代理，如需去掉代理可删除此行
    '--proxy-server=http://t14073074054095:6arg6ric@o918.kdltpspro.com:15818'
]

# -----------------------------
# 其他常用 Scrapy 配置
# -----------------------------

# 如果需要将数据导出到 JSON 文件
FEEDS = {
    "output.json": {"format": "json", "encoding": "utf-8", "overwrite": True},
}

# 默认编码
FEED_EXPORT_ENCODING = "utf-8"

# 建议使用 AsyncioSelectorReactor（Scrapy 默认）
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

# ============ 下面是一些可选或常用的配置，酌情启用/修改 ============

# 并发请求数量（默认16），可根据需要调整
# CONCURRENT_REQUESTS = 32

# 下载延迟（秒），防止过快访问导致封锁
# DOWNLOAD_DELAY = 1

# 禁用 cookies（默认启用）
# COOKIES_ENABLED = False

# 禁用重试（默认启用），若频繁遇到网络超时可考虑启用
# RETRY_ENABLED = False

# 自定义请求头
# DEFAULT_REQUEST_HEADERS = {
#    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#    "Accept-Language": "en",
# }

# 启用或禁用扩展
# EXTENSIONS = {
#    'scrapy.extensions.telnet.TelnetConsole': None,
# }

# 配置 item pipeline
# ITEM_PIPELINES = {
#    'myproject.pipelines.MyprojectPipeline': 300,
# }

# 启用或禁用 AutoThrottle 扩展
# AUTOTHROTTLE_ENABLED = True
# AUTOTHROTTLE_START_DELAY = 5
# AUTOTHROTTLE_MAX_DELAY = 60
# AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# AUTOTHROTTLE_DEBUG = False

# HTTP 缓存配置（默认禁用）
# HTTPCACHE_ENABLED = True
# HTTPCACHE_EXPIRATION_SECS = 0
# HTTPCACHE_DIR = 'httpcache'
# HTTPCACHE_IGNORE_HTTP_CODES = []
# HTTPCACHE_STORAGE = 'scrapy.extensions.httpcache.FilesystemCacheStorage'
