import pandas as pd
import  numpy as np
import  random

#数据归一化
def normData(dataset):
    maxVals = dataset.max(axis=0) #求列的最大值
    minVals = dataset.min(axis=0) #求列的最小值
    ranges = maxVals - minVals
    m=dataset.shape[0]
    normset=dataset-np.tile(minVals,(m,1))
    normset=normset/np.tile(ranges,(m,1))
    # normset = (dataset - minVals) / ranges
    return normset


""""
函数功能:随机切分训练集和测试集参数说明:
dataset :输入的数据集rate:训练集所占比例返回:切分好的训练集和测试集
"""
def randsplit(dataset, rate):
      l = list(dataset. index)  #提取出索引
      random.shuffle(l) #随机打乱索引
      dataset.index = l  #将打乱后的索引重新赋值给原数据集
      n = dataset.shape[0]  #总行数
      m = int(n*rate)  #训练集的数量
      train = dataset.loc[range(m),:]  #提取前m个记录作为训练集
      test = dataset.loc[range(m,n),:]  #剩下的作为测试集
      dataset.index = range (dataset.shape[0]) #更新原数据集的索引
      test.index = range(test.shape[0])  #更新测试集的索引数据分析标
      return train, test


def gnb_classify(train,test):
    labels = train.iloc[:, -1].value_counts().index  # 提取训练集的标签种类
    mean = []  # 存放每 个类别的均值
    std =[] #存放每 个类别的方差
    result = []  # 存放测试集的预测结果
    for i in labels:
        item = train.loc[train.iloc[:, -1] == i, :]  # 分别提取出每一种类别
        m = item.iloc[:,:-1].mean()  # 当前类别的平均值
        s = np.sum((item.iloc[:,:-1] - m) ** 2) / (item.shape[0])  # 当前类别的方差
        mean.append(m)  # 将 当前类别的平均值追加至列表
        std.append(s)  # 将 当前类别的方差追加至列表
    means = pd.DataFrame(mean, index=labels)  # 变成DF格式，索引为类标签
    stds = pd.DataFrame(std, index=labels)  # 变成DF格式， 索引为类标签
    for j in range(test.shape[0]):
        iset = test.iloc[j, :-1].tolist()  # 当前测试实例
        iprob = np.exp(-1*(iset-means)**2/(stds*2))/(np.sqrt(2*np.pi*stds))  # 正态分布公式
        prob = 1  # 初始化当前实例总概率
        col_list = test_set.columns[:-1]  # 获取特征列表
        for k in col_list:  # 遍力每个特征56666
            prob*= iprob[k]  # 特征概率之积即为当前实例概率
            cla = prob.index[np.argmax(prob.values)]  # 返回最大概率的类别
        result.append(cla)
    test['predict'] = result
    acc = (test.iloc[:, -1] == test.iloc[:, -2]).mean()  # 计算预测准碗率
    print(f'模型预测准确率为{acc}')
    return test


data_set=pd.read_csv('KDDTest+.csv',header=None)

df=pd.DataFrame(columns = ['A','B','C','D','E'])
df.iloc[:,0]=data_set.iloc[:,22]
df.iloc[:,1]=data_set.iloc[:,23]
df.iloc[:,2]=data_set.iloc[:,31]
df.iloc[:,3]=data_set.iloc[:,32]
df.iloc[:,-1]=data_set.iloc[:,-1]

df['truedate']=data_set.iloc[:,-2]

train_set,test_set=randsplit(df,0.8)
result_data=gnb_classify(train_set,test_set)
print(result_data.head())


