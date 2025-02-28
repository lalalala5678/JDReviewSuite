import json
import re
import pandas as pd

def parse_topic_string(topic_str):
    """
    将类似 "0.034*'效果' + 0.014*'方便' + 0.008*'实惠'..." 
    的字符串解析为 [(0.034, '效果'), (0.014, '方便'), ...]
    """
    # 先按 + 切分
    parts = topic_str.split('+')
    results = []
    # 用一个正则表达式来捕获: [数字] * '[词]'
    pattern = r"([\d\.]+)\*'([^']+)'"
    for p in parts:
        p = p.strip()  # 去除前后空格
        match = re.search(pattern, p)
        if match:
            weight = float(match.group(1))   # 第一个捕获组是数字
            word = match.group(2)           # 第二个捕获组是单词
            results.append((weight, word))
    return results

def lda_json_to_excel(json_path, excel_path):
    # 1. 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. 获取正面评论和负面评论的 LDA 结果
    positive_lda = data.get("positive_reviews_lda", {})
    negative_lda = data.get("negative_reviews_lda", {})
    
    # 3. 分别构建两个 DataFrame
    #    每个 DataFrame 包含 [Topic, Weight, Word] 列
    #    其中 Topic 为 topic_0, topic_1 ... 方便在表格中区分不同主题
    pos_rows = []
    for topic_name, topic_str in positive_lda.items():
        parsed = parse_topic_string(topic_str)
        for weight, word in parsed:
            pos_rows.append({
                "Topic": topic_name,
                "Weight": weight,
                "Word": word
            })
    df_positive = pd.DataFrame(pos_rows, columns=["Topic", "Weight", "Word"])

    neg_rows = []
    for topic_name, topic_str in negative_lda.items():
        parsed = parse_topic_string(topic_str)
        for weight, word in parsed:
            neg_rows.append({
                "Topic": topic_name,
                "Weight": weight,
                "Word": word
            })
    df_negative = pd.DataFrame(neg_rows, columns=["Topic", "Weight", "Word"])

    # 4. 保存到同一个 Excel 文件中的不同 Sheet
    with pd.ExcelWriter(excel_path) as writer:
        df_positive.to_excel(writer, sheet_name='Positive_LDA', index=False)
        df_negative.to_excel(writer, sheet_name='Negative_LDA', index=False)

if __name__ == "__main__":
    # 输入与输出文件可根据需要自行调整
    json_input = "lda_analysis_results.json"
    excel_output = "lda_analysis_results.xlsx"
    lda_json_to_excel(json_input, excel_output)
    print(f"转换完成！已生成 {excel_output}")
