import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import  jieba

def data_clean(data_set):
    data_set.drop('date', axis=1, inplace=True)

    data_set['category']=data_set['category'].apply(lambda x:x.strip())
    data_set['rank']=data_set['rank'].apply(lambda x:0 if x<=30 else (1 if 30<x<70 else 2))
    data_set.amount=data_set.amount.apply(lambda x:0 if x<1000 else (1 if x<10000 else 2))

    aggrement = {'[都市激战]': 1, '[东方玄幻]': 2, '[异界大陆]': 3, '[都市生活]': 4, '[古典仙侠]': 5, '[游戏生涯]': 6, '[现代修真]': 7, '[奇幻修真]': 8, '[谍战特工]': 9,
                     '[都市异能]': 10, '[洪荒封神]': 11, '[异术超能]': 12, '[校园风云]': 13, '[历史穿越]': 14, '[都市重生]': 15, '[末世危机]': 16, '[现实题材]': 17, '[虚拟网游]': 18, '[商业大亨]': 19, '[架空历史]': 20, '[娱乐明星]': 21, '[电子竞技]': 22, '[军旅生涯]': 23, '[异世争霸]': 24}
    data_set.category=data_set.category.map(aggrement)
    data_set['category']=data_set['category'].apply(lambda x:0 if x<=10 else (1 if 10<x<20 else 2))

    linshi=data_set.status
    data_set.drop('status', axis=1, inplace=True)
    data_set['status']=linshi
    return data_set

def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

def split_word(data_set):
    data_set['bookname_cut'] = data_set.bookname.apply(chinese_word_cut)    #调用结巴分词对中文文本进行分词，然后用空格进行拼接
    data_set['section_cut'] = data_set.section.apply(chinese_word_cut)
    data_set['author_cut'] = data_set.author.apply(chinese_word_cut)

    max_df = 0.8 # 在超过这一比例的文档中出现的关键词（过于平凡），去除掉。
    min_df = 5# 在低于这一数量的文档中出现的关键词（过于独特），去除掉。
    stoplist = stopwords.words('english')   #获取停止词
    vect = CountVectorizer(max_df = max_df,                             #CountVectorizer向量化工具，它依据词语出现频率转化向量。
                           min_df = min_df,
                           token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',    #初始化一个统计词频对象
                           stop_words=frozenset(stoplist))
    #fit_transform()向量转换，使用统计词频模型来生成训练语料的统计词频矩阵。
    #get_feature_names()获取词袋模型中的所有词语
    term_bookname = pd.DataFrame(vect.fit_transform(data_set.bookname_cut).toarray(), columns=vect.get_feature_names())
    term_section = pd.DataFrame(vect.fit_transform(data_set.section_cut).toarray(), columns=vect.get_feature_names())
    term_author = pd.DataFrame(vect.fit_transform(data_set.author_cut).toarray(), columns=vect.get_feature_names())
    # print(term_author)
    # data_set.drop('abstract', axis=1, inplace=True)
    # print(term_bookname.loc[term_bookname['最强']=='2'])
    lines=[]
    lines=data_set.iloc[:,0:2]
    lines=np.hstack((lines,term_bookname.values))
    lines=np.hstack((lines,term_section.values))
    lines=np.hstack((lines,term_author.values))
    lines=np.hstack((lines,pd.DataFrame(data_set.amount)))
    lines = np.hstack((lines, pd.DataFrame(data_set.status)))
    print(lines.shape)
    return lines.tolist()

data_set=pd.read_csv('artical.csv')
data_one=data_clean(data_set)
# print(data_one)
data_two=split_word(data_one)
# print(data_two)
