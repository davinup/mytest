from math import log
import operator
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import  jieba
import re
from requests_html import HTMLSession

# =====================================================================================================================
#----------------------数据爬取


#循环获取本页面所有书本信息
def get_text_link_from_sel(results):
    linshi=[]
    try:
        for result in results:
            mytext = result.text
            lines=mytext.split('\n')
            # mylink = list(result.absolute_links)[0]
            # mytext.append(mylink)
            # mylist.append(mytext)
            linshi.append(lines)
        return linshi
    except:
        return None

def data_get():
    print('数据爬取中。。。')
    # fb=open('books.csv','w',encoding='utf-8')
    # df = pd.DataFrame(columns = ['name','author','publisher','date','price','score','abstract','url'])
    session_one = HTMLSession()
    url = 'https://www.17k.com/top/refactor/top100/10_bookshelf/10_bookshelf_top_100_pc.html'
    r = session_one.get(url)
        # print(r.html.absolute_links)
        #获取书本信息所在标记
    sel = 'body > div.Main.Top100 > div.content.TABBOX > div:nth-child(2) > div:nth-child(2) > table'
        #获取对应书的信息
    results = r.html.find(sel)
        #循环获取所有书本信息
    # get_text_link_from_sel(sel)
    result_list=get_text_link_from_sel(results)
    colmuns_book=['rank','category','bookname','section','date','author','status','amount']
    linshi=result_list[0][8:]
    print(result_list)
    book_data=[]
    j=0
    while j<len(linshi):
        book_data.append(linshi[j:j+8])
        j+=8
    # print(book_data)
    df = pd.DataFrame(book_data)
    df.columns = colmuns_book     #给数据加列名
    return df
# df.to_csv('artical.csv', encoding='utf-8', index=False)      #存到对应的csv文件中


# =====================================================================================================================
#----------------------数据清洗
def data_clean(data_set):
    print('数据清洗中。。。')
    colu = data_set.columns         #获取列名
    # print(data_set.isnull().sum())
    for cent in colu:
        data_set.dropna(subset=[cent], inplace=True)  # 遍历每一列，找到缺失值，并删除该行

    # 处理重复值
    # print('重复行：', data_set.duplicated().sum())
    data_set.drop_duplicates(inplace=True)
    # #删除不需要的列
    data_set.drop('date', axis=1, inplace=True)

    #字符串转数字
    data_set['rank']=data_set['rank'].astype('int')
    data_set['amount'] = data_set['amount'].astype('int')

    data_set['category']=data_set['category'].apply(lambda x:x.strip())
    data_set['rank']=data_set['rank'].apply(lambda x:0 if x<=30 else (1 if 30<x<70 else 2))
    data_set.amount=data_set.amount.apply(lambda x:0 if x<1000 else 1)
    # data_set.amount=data_set.amount.apply(lambda x:0 if x<1000 else (1 if x<10000 else 2))


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
    print('数据分词中(结巴分词)。。。')
    data_set['bookname_cut'] = data_set.bookname.apply(chinese_word_cut)    #调用结巴分词对中文文本进行分词，然后用空格进行拼接
    data_set['section_cut'] = data_set.section.apply(chinese_word_cut)
    data_set['author_cut'] = data_set.author.apply(chinese_word_cut)

    max_df = 0.8 # 在超过这一比例的文档中出现的关键词（过于平凡），去除掉。
    min_df = 5 # 在低于这一数量的文档中出现的关键词（过于独特），去除掉。
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
    lines=[]
    lines=data_set.iloc[:,0:2]
    lines=np.hstack((lines,term_bookname.values))
    lines=np.hstack((lines,term_section.values))
    lines=np.hstack((lines,term_author.values))
    lines=np.hstack((lines,pd.DataFrame(data_set.amount)))
    lines = np.hstack((lines, pd.DataFrame(data_set.status)))
    # print(lines)
    return lines.tolist()
# =====================================================================================================================
#----------------------数据分类

"""
函数说明:计算给定数据集的经验熵(香农熵)

Parameters:
    dataSet - 数据集
Returns:
    shannonEnt - 经验熵(香农熵)
"""
def calcShannonEnt(dataSet):
    numEntires = len(dataSet)                        #返回数据集的行数
    labelCounts = {}                                #保存每个标签(Label)出现次数的字典
    for featVec in dataSet:                            #对每组特征向量进行统计
        currentLabel = featVec[-1]                    #提取标签(Label)信息
        if currentLabel not in labelCounts.keys():    #如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1                #Label计数
    shannonEnt = 0.0                                #经验熵(香农熵)
    for key in labelCounts:                            #计算香农熵
        prob = float(labelCounts[key]) / numEntires    #选择该标签(Label)的概率
        shannonEnt -= prob * log(prob, 2)            #利用公式计算
    return shannonEnt                                #返回经验熵(香农熵)

"""
函数说明:创建测试数据集

Parameters:
    无
Returns:
    dataSet - 数据集
    labels - 特征标签
"""
def createDataSet():
    data_set = data_get()
    result_mat = data_clean(data_set)
    result_data = split_word(result_mat)
    print(result_data)
    labels = ['A', 'B', 'C', 'D','E','F','G','H','I','J']        #特征标签
    return result_data, labels                             #返回数据集和分类属性

"""
函数说明:按照给定特征划分数据集

Parameters:
    dataSet - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值
Returns:
    无
"""
def splitDataSet(dataSet, axis, value):
    retDataSet = []                                        #创建返回的数据集列表
    for featVec in dataSet:                             #遍历数据集
        # print(featVec)
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]                #去掉axis特征(不包括axis列)
            reducedFeatVec.extend(featVec[axis+1:])     #将符合条件的添加到返回的数据集(把剩下的列添加进来)
            retDataSet.append(reducedFeatVec)
    return retDataSet                                      #返回划分后的数据集

"""
函数说明:选择最优特征

Parameters:
    dataSet - 数据集
Returns:
    bestFeature - 信息增益最大的(最优)特征的索引值
"""
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1                    #特征数量
    baseEntropy = calcShannonEnt(dataSet)                 #计算数据集的香农熵
    bestInfoGain = 0.0                                  #信息增益
    bestFeature = -1                                    #最优特征的索引值
    for i in range(numFeatures):                         #遍历所有特征
        #获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)                         #创建set集合{},元素不可重复
        newEntropy = 0.0                                  #经验条件熵
        for value in uniqueVals:                         #计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)#subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))           #计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)     #根据公式计算经验条件熵(概率*(-xlogx)+概率*(-ylogy))
        infoGain = baseEntropy - newEntropy                     #信息增益
        # print("第%d个特征的增益为%.3f" % (i, infoGain))            #打印每个特征的信息增益
        if (infoGain > bestInfoGain):                             #计算信息增益
            bestInfoGain = infoGain                             #更新信息增益，找到最大的信息增益
            bestFeature = i                                     #记录信息增益最大的特征的索引值
    return bestFeature                                             #返回信息增益最大的特征的索引值


"""
函数说明:统计classList中出现此处最多的元素(类标签)

Parameters:
    classList - 类标签列表
Returns:
    sortedClassCount[0][0] - 出现此处最多的元素(类标签)
"""
def majorityCnt(classList):
    classCount = {}
    for vote in classList:                                        #统计classList中每个元素出现的次数
        if vote not in classCount.keys():                         #classCount.keys()：classCount的所有的键
            classCount[vote] = 0
        classCount[vote] += 1
    #classCount.items()：返回可遍历的(键, 值) 元组数组,key:排序依据，reverse升降序
    # operator模块提供的itemgetter函数用于获取对象的哪些维的数据
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)        #根据字典的值（不是键）降序排序
    return sortedClassCount[0][0]                                #返回classList中出现次数最多的元素

"""
函数说明:创建决策树

Parameters:
    dataSet - 训练数据集
    labels - 分类属性标签
    featLabels - 存储选择的最优特征标签
Returns:
    myTree - 决策树
"""
def createTree(dataSet, labels, featLabels):

    classList = [example[-1] for example in dataSet]            #取分类标签(是否放贷:yes or no)
    if classList.count(classList[0]) == len(classList):            #如果类别完全相同则停止继续划分(递归出口，如果已经确定结果，不再需要进行分类)
        return classList[0]
    if len(labels) == 0:                                    #遍历完所有特征时返回出现次数最多的类标签(所有特征用完，还没有分好类，投票表决法:少数服从多数)
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)                #选择最优特征
    bestFeatLabel = labels[bestFeat]                            #最优特征的标签
    featLabels.append(bestFeatLabel)
    # print('%d--%d'%(len(labels), bestFeat))
    myTree = {bestFeatLabel:{}}                                  #根据最优特征的标签生成树
    del(labels[bestFeat])                                        #删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet]        #得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)                                #去掉重复的属性值
    for value in uniqueVals:                                    #遍历特征，创建决策树。
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree

"""
函数说明:使用决策树分类

Parameters:
    inputTree - 已经生成的决策树
    featLabels - 存储选择的最优特征标签
    testVec - 测试数据列表，顺序对应最优特征标签
Returns:
    classLabel - 分类结果
"""
def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))#获取决策树结点
    secondDict = inputTree[firstStr]                                                        #获得内部下一个字典
    featIndex = featLabels.index(firstStr)                                                  #从上向下获取树结点的索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':                                     #判断决策树是否遍历完全（遍历完后后变成str类型）
                classLabel = classify(secondDict[key], featLabels, testVec)                     #如果没有分类完成，继续递归向下遍历
            else:
                classLabel = secondDict[key]
    return classLabel

if __name__ == '__main__':
    # dataSet, labels = createDataSet('artical.csv')
    dataSet, labels = createDataSet()
    # print(labels)
    featLabels = []
    print('构建决策树。。。')
    myTree = createTree(dataSet, labels, featLabels)
    # testVec = [0,0,0,0,0,0,1,1]
    testVec = [0, 1, 0, 0, 0, 0, 1, 1]
    print('决策树\n',myTree)
    print(featLabels)
    #测试数据
    print('测试数据：',testVec)
    result = classify(myTree, featLabels, testVec)
    print('测试结果：',result)
    # if result == 'yes':
    #     print('放贷')
    # if result == 'no':
    #     print('不放贷')
