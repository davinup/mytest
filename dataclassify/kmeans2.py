import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def normData(dataset):
    maxVals = dataset.max(axis=0) #求列的最大值
    minVals = dataset.min(axis=0) #求列的最小值
    ranges = maxVals - minVals
    m=dataset.shape[0]
    # normset=dataset-np.tile(minVals,(m,1))
    # normset=normset/np.tile(ranges,(m,1))
    normset = (dataset - minVals) / ranges
    return normset


# 加载数据
def loadDataSet(fileName):
    df = pd.read_csv(fileName, header=None)
    row = df.shape[0]
    col = df.shape[1]
    orginset=df.values
    aggrement = {'tcp': 1, 'udp': 2, 'icmp': 3, 'REJ': 1, 'RSTO': 2, 'RSTR': 3, 'S0': 4, 'S1': 5, 'S2': 6, 'S3': 7,
                 'SF': 8, 'SH': 9}
    map_dic = {'private': 1, 'ftp_data': 2, 'eco_i': 3, 'telnet': 4, 'http': 5, 'smtp': 6, 'ftp': 7, 'ldap': 8,
               'pop_3': 9, 'courier': 10, 'discard': 11, 'ecr_i': 12, 'imap4': 13, 'domain_u': 14, 'mtp': 15,
               'systat': 16, 'iso_tsap': 17, 'other': 18, 'csnet_ns': 19, 'finger': 20, 'uucp': 21, 'whois': 22,
               'netbios_ns': 23, 'link': 24, 'Z39_50': 25, 'sunrpc': 26, 'auth': 27, 'netbios_dgm': 28, 'uucp_path': 29,
               'vmnet': 30, 'domain': 31, 'name': 32, 'pop_2': 33, 'http_443': 34, 'urp_i': 35, 'login': 36,
               'gopher': 37, 'exec': 38, 'time': 39, 'remote_job': 40, 'klogin': 41}

    map_dic2={'neptune':1,'normal':2,'saint':3,'mscan':4,'guess_passwd':5,'smurf':6,'apache2':7,'satan':8,'buffer_overflow':9,'back':10,'warezmaster':11,'snmpgetattack':12,'processtable':13,'pod':14,'httptunnel':15,'nmap':16,'ps':17,'snmpguess':18,'ipsweep':19,'mailbomb':20,'portsweep':21,'multihop':22,'named':23,'sendmail':14}
    df[1] = df[1].map(aggrement)
    df[2] = df[2].map(map_dic)
    df[3] = df[3].map(aggrement)
    df[41] = df[41].map(map_dic2)
    result_mat = df.values
    # print(result_mat[:,3])
    # np.delete(result_mat, 2, axis=1)
    # result_mat=normData(result_mat)
    return orginset,result_mat


# 欧氏距离计算
def distEclud(x, y):
    return np.sqrt(np.sum(np.power((x - y) , 2)))  # 计算欧氏距离


# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet, k):
    m, n = dataSet.shape
    centroids = np.mat(np.zeros((k, n)))
    for i in range(n):
        minj=min(dataSet[:,i])#统计每一列最小值
        maxj=max(dataSet[:,i])#统计每一列最大值
        rangej=float(maxj-minj)
        array2=minj+rangej*np.random.rand(k,1)#随机范围内的三个数
        centroids[:, i] = np.mat(array2)
    # print(centroids)
    return centroids


# k均值聚类
def KMeans(dataSet, k):
    m = np.shape(dataSet)[0]  # 行的数目
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m, 2)))
    clusterChange = True

    # 第1步 初始化centroids
    centroids = randCent(dataSet, k)
    times=1
    while clusterChange and times<10:
        clusterChange = False

        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = np.inf
            minIndex = -1

            # 遍历所有的质心
            # 第2步 找出最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = distEclud(centroids[j, :], dataSet[i, :])

                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
            clusterAssment[i, :] = minIndex, minDist
        # 第 4 步：更新质心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取簇类所有的点
            if len(pointsInCluster) != 0:#在选出其他分类所对应的数据集时，由于会产生空的数组，传入mean()函数中。
                centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值
        # print('第%d次'%times)
        # showCluster(dataSet, k, centroids, clusterAssment)#画出每次质心变化后的样本分布
    # print("Congratulations,cluster complete!")
        times += 1
    return centroids, clusterAssment


def showCluster(dataSet, k, centroids, clusterAssment):
    m, n = dataSet.shape
    if n != 2:
        print("数据不是二维的")
        return 1

    mark = ['oy', 'ob', 'og', 'ok', 'oc', 'ow', 'om', 'dr', '<r', 'pr']
    if k > len(mark):
        print("k值太大了")
        return 1

    # plt.figure(figsize=(10, 10))
    # 绘制所有的样本
    for i in range(m):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    plt.title('K-means')
    # 绘制质心
    mark = ['Dr', 'Dr', 'Dr', 'Dr', 'Dc', 'Db', 'Db', 'Db', 'Db', 'Db']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1],mark[i],MarkerSize=8)
    print('质心坐标：')
    print(centroids)
    plt.show()


# if __name__ == '__main__':
k = 4
orginset,dataSet = loadDataSet("KDDTest+.csv")
centroids, clusterAssment = KMeans(dataSet, k)
# print(dataSet)
# print(clusterAssment[100:200,:])
for j in range(k):
    print('第%d类：'%(j+1))
    for cent in range(len(clusterAssment)):
        if clusterAssment[cent,0]==j:
            print(orginset[cent])
# print(centroids)
    # showCluster(dataSet, k, centroids, clusterAssment)
