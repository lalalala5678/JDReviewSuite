import json
import jieba
import matplotlib.pyplot as plt
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

# 3. 创建LDA模型
def build_corpus(texts):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return dictionary, corpus

# 4. LDA主题数寻优并输出困惑度图
def compute_perplexity(corpus, dictionary, topic_num):
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=topic_num, passes=10)
    perplexity = lda_model.log_perplexity(corpus)
    return perplexity

# 计算不同主题数的困惑度
def plot_perplexity_vs_topics(texts, title):
    dictionary, corpus = build_corpus(texts)
    perplexities = []
    topic_range = range(2, 11)  # 设置需要评估的主题数范围（2到10）

    for num_topics in topic_range:
        perplexity = compute_perplexity(corpus, dictionary, num_topics)
        perplexities.append(perplexity)

    # 绘制困惑度 vs 主题数图
    plt.plot(topic_range, perplexities)
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')
    plt.title(title)
    plt.show()

# 分别绘制好评和差评的困惑度图
plot_perplexity_vs_topics(positive_texts, 'LDA Perplexity vs. Number of Topics for Positive Comments')
plot_perplexity_vs_topics(negative_texts, 'LDA Perplexity vs. Number of Topics for Negative Comments')
