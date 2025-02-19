#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
import os
import re

def clean_text(text):
    """
    清除字符串中 Excel 不允许的控制字符：
    ASCII 0x00-0x08、0x0B-0x0C、0x0E-0x1F 内的字符
    """
    if isinstance(text, str):
        return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', text)
    return text

def convert_json_to_excel(json_file: str, excel_file: str):
    """
    读取指定 JSON 文件，将其转换为 DataFrame，并清理非法字符，
    最后保存为 Excel 文件。
    """
    if not os.path.exists(json_file):
        print(f"错误：文件 {json_file} 不存在。")
        return

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取 JSON 文件时出错：{e}")
        return

    # 转换为 DataFrame
    df = pd.DataFrame(data)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(clean_text)

    try:
        df.to_excel(excel_file, index=False)
        print(f"ok，{excel_file}")
    except Exception as e:
        print(f"出错：{e}")

if __name__ == '__main__':
    json_filename = "output.json"
    excel_filename = "output.xlsx"
    convert_json_to_excel(json_filename, excel_filename)
