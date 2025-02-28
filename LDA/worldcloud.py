import json
import re
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def parse_topic_string(topic_str):
    """
    将形如 0.034*'效果' + 0.014*'方便' ... 的字符串解析为 {词: 权重, ...}
    """
    parts = topic_str.split('+')
    pattern = r"([\d\.]+)\*'([^']+)'"
    word_freq = {}
    for p in parts:
        p = p.strip()
        match = re.search(pattern, p)
        if match:
            weight = float(match.group(1))
            word = match.group(2)
            word_freq[word] = word_freq.get(word, 0) + weight
    return word_freq

def create_heart_mask(size=500, flip_upside_down=False, invert_color=False):
    """
    使用经典心形方程 (x^2 + y^2 - 1)^3 - x^2*y^3 <= 0 在 size x size 网格中生成二值mask。
    - flip_upside_down: 是否上下翻转心形
    - invert_color:     是否反相颜色（让 inside/outside 反过来）
    """
    x = np.linspace(-1.5, 1.5, size)
    y = np.linspace(-1.5, 1.5, size)
    X, Y = np.meshgrid(x, y)

    # 心形方程: (x^2 + y^2 -1)^3 - x^2*y^3 <= 0 表示心形内部
    equation = (X**2 + Y**2 - 1)**3 - X**2 * (Y**3)

    # 初始化 mask，全为 0（黑色）
    mask = np.zeros((size, size), dtype=np.uint8)
    # 将满足心形内部条件的像素设为 255（白色）
    mask[equation <= 0] = 255

    # 如果需要上下翻转心形
    if flip_upside_down:
        mask = np.flipud(mask)

    # 如果需要反相颜色（通常当你发现“文字跑到外面”时可以反相）
    if invert_color:
        mask = 255 - mask

    return mask

def main():
    # 1. 读入 JSON 并解析「正面评论」部分
    with open('lda_analysis_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    positive_lda = data.get("positive_reviews_lda", {})

    # 2. 收集正面评论里的所有词频
    positive_word_freq = {}
    for topic_name, topic_str in positive_lda.items():
        one_topic_freq = parse_topic_string(topic_str)
        for w, freq in one_topic_freq.items():
            positive_word_freq[w] = positive_word_freq.get(w, 0) + freq

    # 3. 生成「倒过来的」心形掩码
    heart_mask = create_heart_mask(
        size=600,
        flip_upside_down=True,  # 让心形上下翻转
        invert_color=True      # 保持“内部 = 白，外部 = 黑”
    )

    # 4. 根据词频和心形 mask 生成词云
    wc = WordCloud(
        font_path='msyh.ttc',       # 中文字体路径
        background_color='white',   # 背景色
        mask=heart_mask,            # 使用心形遮罩
        max_words=200,              # 最多显示多少词
        max_font_size=120,          # 最大字号
        scale=2,                     # 提高分辨率
        colormap='autumn' 
    ).generate_from_frequencies(positive_word_freq)

    # 5. 显示并保存结果
    plt.figure(figsize=(8, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("正面评论词云 - 倒过来的心形")
    plt.show()

    wc.to_file("positive_reviews_inverted_heart.png")

if __name__ == "__main__":
    main()
