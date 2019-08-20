import numpy as np
import matplotlib.pyplot as plt


# 加载数据
def loadDataSet(fileName):
    fr = open(fileName)
    lines = fr.readlines()
    line_nums = len(lines)
    # data = np.loadtxt(fileName, delimiter='\t')#读取\t分割的txt文件
    result_mat = np.zeros((line_nums, 2))
    for i in range(line_nums):
        line = lines[i].strip()
        item_mat = line.split(',')
        item_mat = item_mat[33:35]
        result_mat[i, :] = item_mat[0: 2]
    # print(result_mat)
    return result_mat


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
        print('第%d次'%times)
        showCluster(dataSet, k, centroids, clusterAssment)#画出每次质心变化后的样本分布
        print("Congratulations,cluster complete!")
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


if __name__ == '__main__':
    k = 4
    dataSet = loadDataSet("KDDTest+.txt")
    centroids, clusterAssment = KMeans(dataSet, k)
    # showCluster(dataSet, k, centroids, clusterAssment)
