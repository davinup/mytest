import numpy as np
import  pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import  jieba
import operator as opt


def data_clean(data_set):
    # 处理缺失值
    colu = data_set.columns
    # print(data_set.isnull().sum())
    for cent in colu:
        data_set.dropna(subset=[cent], inplace=True)  # 遍历每一列，找到缺失值，并删除该行
    # 处理重复值
    # print('重复行：', data_set.duplicated().sum())
    data_set.drop_duplicates(inplace=True)

    # 删除无用列
    data_set.drop('_id', axis=1, inplace=True)
    data_set.drop('url', axis=1, inplace=True)

    # 处理异常值
    date_index = data_set.index[data_set['published_date'].str.len() < 10].tolist()  # 获取异常数据的所在行数
    published_date = data_set[data_set['published_date'].str.len() > 10]['published_date']  # 获取本列正常数据
    for i in range(len(date_index)):
        data_set.iloc[date_index[i], 3] = published_date[i]  # 随机给异常值替换本列数据
    data_set.iloc[:, 3]=data_set.published_date.str[0:4]

    # 对出版时间进行转换
    map_dic = {'2014': 1, '2015': 2, '2016': 3, '2017': 4, '2018': 5, '2019': 6}
    data_set['published_date'] = data_set['published_date'].map(map_dic)
    # print(data_set.published_date)
    # for cent in map_dic.keys():
    #     linshi = data_set.index[data_set['published_date'].str.contains(cent)].tolist()
    #     data_set.iloc[linshi, 3] = map_dic[cent]

    aggrement = {'代码审计': 1, '无线安全': 2, '移动安全': 3, 'CTF': 4, 'Web安全': 5, '安全报告': 6, '内网渗透': 7, '系统安全': 8, 'Windows': 9,
                 '工控安全': 10, '安全文献': 11, '专题': 12, 'Linux': 13, '国外资讯': 14, '工具': 15, '独家': 16, '国内资讯': 17, '其他': 18}
    # data_set['tags']=data_set['tags'].map(aggrement)
    data_set.index = range(len(data_set))
    class_label=data_set['tags']
    return data_set,class_label


def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

def split_word(data_set):
    data_set['abstract_cut'] = data_set.abstract.apply(chinese_word_cut)
    data_set['author_cut'] = data_set.author.apply(chinese_word_cut)
    data_set['content_cut'] = data_set.content.apply(chinese_word_cut)
    data_set['title_cut'] = data_set.title.apply(chinese_word_cut)

    max_df = 0.8 # 在超过这一比例的文档中出现的关键词（过于平凡），去除掉。
    min_df = 5 # 在低于这一数量的文档中出现的关键词（过于独特），去除掉。
    stoplist = stopwords.words('english')   #获取停止词
    vect = CountVectorizer(max_df = max_df,
                           min_df = min_df,
                           token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                           stop_words=frozenset(stoplist))
    #向量转换
    term_abstract = pd.DataFrame(vect.fit_transform(data_set.abstract_cut).toarray(), columns=vect.get_feature_names())
    term_author = pd.DataFrame(vect.fit_transform(data_set.author_cut).toarray(), columns=vect.get_feature_names())
    term_content = pd.DataFrame(vect.fit_transform(data_set.content_cut).toarray(), columns=vect.get_feature_names())
    term_title = pd.DataFrame(vect.fit_transform(data_set.title_cut).toarray(), columns=vect.get_feature_names())
    # data_set.drop('abstract', axis=1, inplace=True)
    lines=np.hstack((term_abstract.values,term_author.values))
    lines=np.hstack((lines,term_content.values))
    lines=np.hstack((lines,term_title.values))
    lines=np.hstack((lines,pd.DataFrame(data_set.published_date.values)))
    return lines


def filemat(filename):
    data_set=pd.read_csv(filename)
    result_mat, class_label = data_clean(data_set)
    print(result_mat.shape)
    result_mat=split_word(result_mat)
    return result_mat,class_label

training_mat, training_label=filemat('Nsoadnews.csv')
print(training_label)
