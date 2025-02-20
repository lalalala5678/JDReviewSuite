# -*- coding: utf-8 -*-
"""
示例脚本：动态生成“滋补汤”遮罩图并绘制词云，
同时包括文本预处理、情感分析（SnowNLP）和 LDA 主题建模示例。
"""

import json
import re
import pandas as pd
import numpy as np
import jieba
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud
from snownlp import SnowNLP
from gensim import corpora, models
from PIL import Image, ImageDraw, ImageFont

#==================== 配置参数 ====================
# 遮罩图片相关
INITIAL_MASK_WIDTH = 1000        # 初始遮罩图片宽度（适当加大）
INITIAL_MASK_HEIGHT = 500        # 初始遮罩图片高度（适当加大）
MASK_BG_COLOR = "white"          # 遮罩图片背景色
MASK_PADDING = 40                # 遮罩图边距（文字外留边距，可再调大）

# 文本绘制相关
FONT_PATH = "C:/Windows/Fonts/STZHONGS.ttf"  # 字体文件路径，请确保存在
FONT_SIZE = 300                  # 使用的字体大小（更大字体，形状更明显）
TEXT_FILL_COLOR = "black"        # 文字填充色
TEXT_STROKE_WIDTH = 8            # 文字描边宽度（数值越大描边越粗）
TEXT_STROKE_FILL = "black"       # 文字描边颜色

# matplotlib中文显示设置
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用黑体显示中文
plt.rcParams["axes.unicode_minus"] = False    # 正常显示负号

#==================== 1. 读取 JSON 数据并构建 DataFrame ====================
json_filepath = "output.json"  # JSON 文件路径
with open(json_filepath, "r", encoding="utf-8") as file:
    json_data = json.load(file)

df_reviews = pd.DataFrame(json_data)

#==================== 2. 数据去重、清洗与分词 ====================
df_reviews.drop_duplicates(subset="content", inplace=True)

chinese_char_pattern = re.compile(r"[^\u4e00-\u9fa5]")
df_reviews["clean_text"] = df_reviews["content"].astype(str).apply(
    lambda x: chinese_char_pattern.sub("", x)
)

def cut_chinese_words(text):
    """使用结巴对中文文本进行分词，并以空格分隔"""
    return " ".join(jieba.cut(text))

df_reviews["clean_text"] = df_reviews["clean_text"].apply(cut_chinese_words)

#==================== 3. 读取停用词表并过滤停用词 ====================
stopwords_filepath = "stoplist.txt"  # 请确保 stoplist.txt 存在
try:
    with open(stopwords_filepath, "r", encoding="utf-8") as file:
        stopwords = [line.strip() for line in file if line.strip()]
except FileNotFoundError:
    stopwords = []

def filter_stopwords(text):
    """过滤文本中的停用词"""
    words = text.split()
    return " ".join([word for word in words if word not in stopwords])

df_reviews["clean_text"] = df_reviews["clean_text"].apply(filter_stopwords)

#==================== 4. 统计词频并输出评论统计信息 ====================
all_words = []
for word_line in df_reviews["clean_text"]:
    all_words.extend(word_line.split())

word_frequency = pd.Series(all_words).value_counts()

print(f"共分析了 {len(df_reviews)} 条评论。")
print(f"去重并分词后，共计 {len(all_words)} 个词。")

#========= 新增：筛选并输出“无意义词”示例（按长度或自定义逻辑） =========
def is_meaningless(word):
    """
    判断词语是否“无意义”的示例逻辑：
    1) 词长小于等于1
    2) 或者其它自定义条件（如出现频次太低、某些特殊符号等）
    """
    return len(word) <= 1

meaningless_words = [w for w in word_frequency.index if is_meaningless(w)]

# 输出无意义词，每个词语换行
if meaningless_words:
    print("\n以下是判定为'无意义'的词语：")
    for w in meaningless_words:
        print(w)

# 从词频中移除无意义词
word_frequency = word_frequency.drop(labels=meaningless_words, errors="ignore")

#==================== 5. 动态生成“滋补汤”遮罩图片 ====================
mask_width = INITIAL_MASK_WIDTH
mask_height = INITIAL_MASK_HEIGHT

font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
mask_text = "滋补汤"

temp_image = Image.new("RGB", (mask_width, mask_height))
temp_draw = ImageDraw.Draw(temp_image)
try:
    text_bbox = temp_draw.textbbox((0, 0), mask_text, font=font, stroke_width=TEXT_STROKE_WIDTH)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
except AttributeError:
    text_width, text_height = temp_draw.textsize(mask_text, font=font)

if text_width + MASK_PADDING > mask_width:
    mask_width = text_width + MASK_PADDING
if text_height + MASK_PADDING > mask_height:
    mask_height = text_height + MASK_PADDING

mask_image_pil = Image.new("RGB", (mask_width, mask_height), color=MASK_BG_COLOR)
draw = ImageDraw.Draw(mask_image_pil)
text_x = (mask_width - text_width) // 2
text_y = (mask_height - text_height) // 2

draw.text(
    (text_x, text_y),
    mask_text,
    fill=TEXT_FILL_COLOR,
    font=font,
    stroke_width=TEXT_STROKE_WIDTH,
    stroke_fill=TEXT_STROKE_FILL
)

mask_image_array = np.array(mask_image_pil)
# 如果效果与预期相反，可尝试反转： mask_image_array = 255 - mask_image_array

#==================== 6. 生成词云 ====================
wordcloud = WordCloud(
    font_path=FONT_PATH,
    background_color="white",
    max_words=1000,
    mask=mask_image_array,
    mode="RGB",
    scale=2  # 放大绘制倍数，可自行调大
)
wordcloud.generate_from_frequencies(word_frequency)

plt.figure(figsize=(12, 8), dpi=300)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("滋补汤形状词云", fontsize=16)

output_image_path = "zibutang_wordcloud.png"
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
print(f"词云图已保存为 {output_image_path}")

plt.show()

#==================== 7. SnowNLP 情感分析 ====================
def calculate_sentiment(text):
    """计算文本情感得分，空文本返回中性 0.5"""
    if not text.strip():
        return 0.5
    s = SnowNLP(text)
    return s.sentiments

df_reviews["sentiment_score"] = df_reviews["clean_text"].apply(calculate_sentiment)
print("\n情感分析示例：")
print(df_reviews[["content", "sentiment_score"]].head(5))

#==================== 8. LDA 主题模型示例 ====================
texts_tokenized = [text.split() for text in df_reviews["clean_text"]]
dictionary = corpora.Dictionary(texts_tokenized)
corpus = [dictionary.doc2bow(text) for text in texts_tokenized]

num_topics = 3
lda_model = models.LdaModel(
    corpus=corpus,
    num_topics=num_topics,
    id2word=dictionary,
    random_state=42
)

print("\nLDA 主题输出:")
topics = lda_model.print_topics(num_words=10)
for topic_idx, topic in topics:
    print(f"主题 {topic_idx}：{topic}")

#==================== 9. 生成评论关联强度网络图 ====================
import networkx as nx

# 获取频率前30的词汇
top_30_words = word_frequency.head(30)

# 创建一个无向图
G = nx.Graph()

# 为每个词语添加节点
for word in top_30_words.index:
    G.add_node(word, size=top_30_words[word])

# 生成词语间的边（依据共现关系）
for idx, row in df_reviews.iterrows():
    words_in_review = set(row["clean_text"].split())
    for word1 in words_in_review:
        for word2 in words_in_review:
            if word1 != word2 and word1 in top_30_words.index and word2 in top_30_words.index:
                if G.has_edge(word1, word2):
                    G[word1][word2]["weight"] += 1
                else:
                    G.add_edge(word1, word2, weight=1)

# 设置节点大小，边的权重对应连接强度
node_sizes = [G.nodes[node]["size"] * 10 for node in G.nodes]

# 画图
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.15, iterations=20)  # 布局
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="lightcoral")
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color="gray")
nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")

plt.title("评论词语关联网络图", fontsize=16)
plt.axis("off")
plt.show()

#==================== 10. 生成评论文本评述表 ====================
# 将词频和词语转换为DataFrame并输出为 Excel 文件
word_frequency_df = top_30_words.reset_index()
word_frequency_df.columns = ['词语', '频率']
excel_output_path = "comment_review.xlsx"
word_frequency_df.to_excel(excel_output_path, index=False)

print(f"\n评论文本评述表已保存为 {excel_output_path}")

print("\n脚本执行完毕。")
