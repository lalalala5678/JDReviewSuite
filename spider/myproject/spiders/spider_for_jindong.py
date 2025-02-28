# -*- coding: utf-8 -*-
import scrapy
import json
import re

class JdCommentSpider(scrapy.Spider):
    name = 'jingdong_comment_spider'

    # ========== 可配置参数 ==========
    MAX_PAGES = 500                     # 增加爬取评论的总页数（从 0 开始计数）
    COMMENTS_PER_PAGE = 20              # 每页评论数量（前提是接口允许更多数据）
    PRODUCT_ID = "100054492400"         # 待爬取评论的京东商品 ID
    # 代理设置：隧道地址、用户名、密码与 requests 示例一致
    PROXY_ADDRESS = "http://t14073074054095:6arg6ric@o918.kdltpspro.com:15818/"

    # ========== 自定义设置 ==========
    custom_settings = {
        "FEEDS": {
            "output.json": {"format": "json", "encoding": "utf-8", "overwrite": True},
        },
        "ROBOTSTXT_OBEY": False,
        "LOG_LEVEL": "DEBUG",
    }

    def start_requests(self):
        base_url = (
            "https://club.jd.com/comment/productPageComments.action?"
            "callback=fetchJSON_comment98"
        )
        for page_number in range(self.MAX_PAGES):
            url = (
                f"{base_url}&productId={self.PRODUCT_ID}&score=0&sortType=5"
                f"&page={page_number}&pageSize={self.COMMENTS_PER_PAGE}"
                f"&isShadowSku=0&fold=1"
            )
            self.logger.debug(f"请求第 {page_number} 页评论，URL: {url}")
            yield scrapy.Request(
                url=url,
                callback=self.parse_comments,
                meta={'proxy': self.PROXY_ADDRESS}
            )

    def parse_comments(self, response):
        self.logger.debug("开始解析评论数据")
        try:
            jsonp_pattern = r'fetchJSON_comment98\((.*)\)'
            match = re.search(jsonp_pattern, response.text)
            if match:
                json_data = json.loads(match.group(1))
            else:
                self.logger.error("未能匹配到 JSONP 数据")
                return
        except Exception as error:
            self.logger.error(f"解析 JSON 数据时发生异常: {error}")
            return

        comments = json_data.get('comments', [])
        if not comments:
            self.logger.info("当前页无评论数据，停止爬取")
            # 终止爬虫，避免继续无效请求
            self.crawler.engine.close_spider(self, "No more comments")
            return

        for comment in comments:
            item = {
                "comment_id": "jd" + str(comment.get('id', '')),
                "content": comment.get('content', ''),
                "creation_time": comment.get('creationTime', ''),
                "score": comment.get('score', 0)
            }
            score = item["score"]
            # 根据评分划分评论类型：4 分及以上为好评，3 分为中评，其它为差评
            if score >= 4:
                item["comment_type"] = "好评"
            elif score == 3:
                item["comment_type"] = "中评"
            else:
                item["comment_type"] = "差评"

            self.logger.info(f"Yield 评论: {item}")
            yield item
