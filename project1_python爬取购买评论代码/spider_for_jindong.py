import requests
import json
import pandas as pd
import time

ids = []
contents = []
times = []

# 定义三个空列表，以便保存数据
for i in range(10):
    Url = f'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=68166363306&score=0&sortType=5&page={i}&pageSize=10&isShadowSku=0&fold=1'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36 Edg/105.0.1343.53',
        'Referer': 'https://item.jd.com/100004770259.html'
    }
    # 模拟浏览器，反爬虫
    data = requests.get(url=Url, headers=headers).text.lstrip('fetchJSON_comment98(').rstrip(');')

    # Debugging step to check the raw data before parsing it
    print(f"第{i + 1}页返回的数据：\n{data}")  # 打印出原始数据
    
    # 检查返回的数据是否为空
    if not data.strip():  # 如果数据为空，跳过当前循环
        print(f"第{i + 1}页没有返回有效数据，跳过...")
        continue

    # 清理数据，确保有效的 JSON 格式
    try:
        data_clean = data[len('fetchJSON_comment98('):-2]  # Strip the outer wrapper more securely
        jsondata = json.loads(data_clean)
    except json.decoder.JSONDecodeError as e:
        print(f"第{i + 1}页解析 JSON 时发生错误: {e}")
        continue

    print(f'正在获取第{i + 1}页...')
    
    # 显示进度
    for x in jsondata['comments']:
        ids.append(x['id'])
        contents.append(x['content'].replace('\n', ' '))  # 将换行符替换
        times.append(x['creationTime'])
    
    time.sleep(3)  # 暂停三秒，避免访问过快被封 IP
    print(f'正在保存为 Excel 文件')

# 将数据保存为 Excel 文件
df = pd.DataFrame({'id': ids, '评论时间': times, '评论内容': contents})
df.to_excel("output.xlsx", sheet_name='Sheet1')

print("保存成功！")
