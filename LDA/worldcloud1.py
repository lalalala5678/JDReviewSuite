import json
import re
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 如果要使用形态学操作（可选），需要导入:
from scipy.ndimage import binary_dilation

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

def create_heart_mask(
    size=500, 
    flip_upside_down=False, 
    invert_color=False, 
    broken=False,
    # 以下参数用于控制“上面裂得多、下面裂得少”
    top_crack_ampl=0.3,      # 裂纹在心形上方的最大扰动幅度
    bottom_crack_ampl=0.05,  # 裂纹在心形下方的最小扰动幅度
    freq_low=10,            # 低频正弦波的频率（整体裂缝走向）
    freq_high=40,           # 高频正弦波的频率（细碎锯齿/裂纹细节）
    crack_width=0.03,       # 基础裂缝宽度
    dilation_iter=0         # 是否膨胀裂缝(>0表示进行几次膨胀)
):
    """
    使用经典心形方程 (X^2 + Y^2 - 1)^3 - X^2*(Y^3) <= 0 在 size x size 网格中生成二值 mask。
    - flip_upside_down: 是否上下翻转心形
    - invert_color:     是否反相颜色（让 inside/outside 反过来）
    - broken:           是否生成“破碎心形”
    - top_crack_ampl:   上方裂纹正弦扰动的最大幅度
    - bottom_crack_ampl:下方裂纹正弦扰动的最小幅度
    - freq_low:         低频正弦波频率（决定裂缝整体“弯曲”形态）
    - freq_high:        高频正弦波频率（在裂缝上叠加更细微的锯齿形）
    - crack_width:      判断裂缝的阈值区间
    - dilation_iter:    对裂缝做几次形态学膨胀(可让裂缝边缘更粗)
    """
    # 1. 生成坐标网格
    x = np.linspace(-1.5, 1.5, size)
    y = np.linspace(-1.5, 1.5, size)
    X, Y = np.meshgrid(x, y)

    # 2. 计算心形方程区域 (heart inside)
    equation = (X**2 + Y**2 - 1)**3 - X**2*(Y**3)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[equation <= 0] = 255  # 心形内部用 255 表示

    if broken:
        # 3. 随 Y 变化的裂纹幅度（保证上方裂得多、下方裂得少）
        #    - 当 Y=-1.5 (顶部) 时 amplitude = top_crack_ampl
        #    - 当 Y= 1.5 (底部) 时 amplitude = bottom_crack_ampl
        #    其中 (Y + 1.5) / 3.0 可将 Y 从 [-1.5,1.5] 映射到 [0,1]
        amplitude = top_crack_ampl + (bottom_crack_ampl - top_crack_ampl) * ((Y + 1.5) / 3.0)

        # 4. 先构建低频裂缝
        crack_line = X + amplitude * np.sin(freq_low * Y)

        # 5. 再叠加一个高频小扰动（相当于加粗边缘的“锯齿感”）
        np.random.seed(42)  # 固定随机数种子，若想每次都不一样可删掉此行
        random_phase = 2.0 * np.pi * np.random.rand()
        crack_line += 0.05 * np.sin(freq_high * Y + random_phase)

        # 6. 判断裂缝区域：|crack_line| < crack_width
        crack = (np.abs(crack_line) < crack_width)

        # 7. 将裂缝处的心形像素掏空 (从 255 变成 0)
        mask[crack & (mask == 255)] = 0

        # 8. 如果需要对裂纹做形态学膨胀，让裂纹边缘更明显
        if dilation_iter > 0:
            dilated_crack = binary_dilation(crack, iterations=dilation_iter)
            mask[dilated_crack & (mask == 255)] = 0

    # 上下翻转（若需要心形正立或倒置）
    if flip_upside_down:
        mask = np.flipud(mask)

    # 反相颜色（有些时候需要让文字在“心形内”，则 mask=白，其它=黑）
    if invert_color:
        mask = 255 - mask

    return mask


def main():
    # 1. 读入 JSON 并解析评论（此处示例使用 negative_reviews_lda 键，视情况修改）
    with open('lda_analysis_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    positive_lda = data.get("negative_reviews_lda", {})

    # 2. 汇总所有词频
    positive_word_freq = {}
    for topic_name, topic_str in positive_lda.items():
        one_topic_freq = parse_topic_string(topic_str)
        for w, freq in one_topic_freq.items():
            positive_word_freq[w] = positive_word_freq.get(w, 0) + freq

    # 3. 生成“上方裂多、下方裂少”的破碎心形掩码
    heart_mask = create_heart_mask(
        size=600,
        flip_upside_down=True,      # 可视需要决定是否翻转
        invert_color=True,
        broken=True,
        # 以下参数决定“上面裂得多、下面裂得少”的程度
        top_crack_ampl=0.3,         # 上方的最大裂纹幅度
        bottom_crack_ampl=0.05,     # 下方的最小裂纹幅度
        freq_low=10,                # 低频正弦波频率
        freq_high=40,               # 高频正弦波频率
        crack_width=0.03,           # 裂缝基本宽度
        dilation_iter=1             # 对裂纹膨胀一次，让边缘更明显
    )

    # 4. 用生成的 mask 做词云
    wc = WordCloud(
        font_path='msyh.ttc',       # 字体路径（改为你系统的中文字体）
        background_color='white',
        mask=heart_mask,
        max_words=200,
        max_font_size=120,
        scale=2,
        colormap='winter'
    ).generate_from_frequencies(positive_word_freq)

    # 5. 显示并保存
    plt.figure(figsize=(8, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("上面裂多、下面裂少的破碎心形词云")
    plt.show()

    wc.to_file("broken_heart_more_crack_on_top.png")


if __name__ == "__main__":
    main()
