import json
import jieba
from gensim import corpora
from gensim.models import LdaModel

# 1. 加载数据
with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 将好评和差评分别提取出来
positive_comments = [entry['content'] for entry in data if entry['comment_type'] == '好评']
negative_comments = [entry['content'] for entry in data if entry['comment_type'] == '差评']

# 2. 数据预处理：分词
def jieba_cut(text):
    return list(jieba.cut(text))

positive_texts = [jieba_cut(comment) for comment in positive_comments]
negative_texts = [jieba_cut(comment) for comment in negative_comments]

# 3. 加载停用词
def load_stopwords(file_path='stoplist.txt'):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = set(f.read().splitlines())  # 每一行作为一个停用词
    return stopwords

stopwords = load_stopwords()

# 4. 创建LDA模型：好评7个主题，差评5个主题
def build_corpus(texts):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, corpus

# 好评LDA模型
positive_dict, positive_corpus = build_corpus(positive_texts)
positive_lda = LdaModel(positive_corpus, num_topics=7, id2word=positive_dict, passes=10)

# 差评LDA模型
negative_dict, negative_corpus = build_corpus(negative_texts)
negative_lda = LdaModel(negative_corpus, num_topics=5, id2word=negative_dict, passes=10)

# 5. 输出每个主题的关键词，获取更多关键词（不限制为10个），并剔除停用词
def get_topics_keywords(lda_model, num_topics, stopwords, topn=50):
    topics_keywords = {}
    for i in range(num_topics):
        words = [word for word, _ in lda_model.show_topic(i, topn=topn)]
        # 剔除停用词
        filtered_words = [word for word in words if word not in stopwords]
        topics_keywords[f"Topic_{i+1}"] = filtered_words
    return topics_keywords

# 获取好评和差评的主题关键词，增加关键词数量
positive_topics_keywords = get_topics_keywords(positive_lda, 7, stopwords, topn=50)  # 输出每个主题50个关键词
negative_topics_keywords = get_topics_keywords(negative_lda, 5, stopwords, topn=50)  # 输出每个主题50个关键词

# 6. 将结果保存为JSON文件
output_data = {
    "positive_comments_topics": positive_topics_keywords,
    "negative_comments_topics": negative_topics_keywords
}

with open('lda_topics_keywords.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print("结果已保存为 lda_topics_keywords.json")
