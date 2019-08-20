import pandas as pd
import numpy as np
import operator as opt

import numpy as np
import matplotlib.pyplot as plt
import operator as opt

def normData(dataset):
    maxVals = dataset.max(axis=0) #求列的最大值
    minVals = dataset.min(axis=0) #求列的最小值
    ranges = maxVals - minVals
    m=dataset.shape[0]
    normset=dataset-np.tile(minVals,(m,1))
    normset=normset/np.tile(ranges,(m,1))
    # normset = (dataset - minVals) / ranges
    return normset


def classify(dataset, labels, testdata, k):
    dataset = np.tile(dataset, (testdata.shape[0], 1))
    distSquareMat = (dataset - testdata) ** 2  # 计算差值的平方
    distSquareSums = distSquareMat.sum(axis=1)  # 求每一行的差值平方和
    distances = distSquareSums ** 0.5  # 开根号，得出每个样本到测试点的距离
    sortedIndices = distances.argsort()  # array.argsort(),默认axis=0从小到大排序，得到排序后的下标位置
    indices = sortedIndices[:k]  # 取距离最小的k个值对应的小标位置
    labelCount = {}  # 存储每个label的出现次数
    for i in indices:
        label = labels[i]
        labelCount[label] = labelCount.get(label, 0) + 1  # 次数加1,dict.get(k, val)获取字典中k对应的值,没有k,则返回val

    sortedCount = sorted(labelCount.items(), key=opt.itemgetter(1), reverse=True)  # operator.itemgetter(),结合sorted使用,可按不同的区域进行排序
    return sortedCount[0][0]  # 返回最多的一个label


def file2mat(test_filename, para_num):
    """
    将表格存入矩阵，test_filename为表格路径，para_num为存入矩阵的列数
    返回目标矩阵，和矩阵每一行数据的类别
    """
    df = pd.read_csv(test_filename, header=None)
    row = df.shape[0]
    col = df.shape[1]
    aggrement = {'tcp': 1, 'udp': 2, 'icmp': 3, 'R EJ': 1, 'RSTO': 2, 'RSTR': 3, 'S0': 4, 'S1': 5, 'S2': 6, 'S3': 7,
                 'SF': 8, 'SH': 9}
    map_dic = {'private': 1, 'ftp_data': 2, 'eco_i': 3, 'telnet': 4, 'http': 5, 'smtp': 6, 'ftp': 7, 'ldap': 8,
               'pop_3': 9, 'courier': 10, 'discard': 11, 'ecr_i': 12, 'imap4': 13, 'domain_u': 14, 'mtp': 15,
               'systat': 16, 'iso_tsap': 17, 'other': 18, 'csnet_ns': 19, 'finger': 20, 'uucp': 21, 'whois': 22,
               'netbios_ns': 23, 'link': 24, 'Z39_50': 25, 'sunrpc': 26, 'auth': 27, 'netbios_dgm': 28, 'uucp_path': 29,
               'vmnet': 30, 'domain': 31, 'name': 32, 'pop_2': 33, 'http_443': 34, 'urp_i': 35, 'login': 36,
               'gopher': 37, 'exec': 38, 'time': 39, 'remote_job': 40, 'klogin': 41}
    df[1] = df[1].map(aggrement)
    df[2] = df[2].map(map_dic)
    df[3] = df[3].map(aggrement)
    result_mat = df.values
    class_label = result_mat[:, -2]
    data_mat = np.zeros((row, col - 1))
    data_mat[:, :-1] = result_mat[:, :-2]
    data_mat[:, -1] = result_mat[:, -1]
    # fr = open(test_filename)
    # lines = fr.readlines()
    # line_nums = len(lines)
    # result_mat = np.zeros((line_nums, para_num))  # 创建line_nums行，para_num列的矩阵
    # class_label = []
    # for i in range(line_nums):
    #     line = lines[i].strip()
    #     item_mat = line.split(',')
    #     item_mat=item_mat[4:42]
    #     result_mat[i, :] = item_mat[0: para_num]
    #     class_label.append(item_mat[-1])  # 表格中最后一列正常1异常2的分类存入class_label
    # fr.close()
    return data_mat, class_label




def test(training_filename, test_filename):
    print('-----+++++')
    training_mat, training_label = file2mat(training_filename, 36)
    test_mat, test_label = file2mat(test_filename, 36)
    training_mat=normData(training_mat)
    test_mat=normData(test_mat)
    test_size = test_mat.shape[0]
    errorcount=0.0
    for i in range(test_size):
        preresult=classify(test_mat[i],training_label,training_mat,5)
        print('模拟预测值：%s,真实值：%s'%(preresult,test_label[i]))
        if (preresult!=test_label[i]):
            errorcount+=1.0
    errorrate=errorcount/test_size
    print ('\n准确率：%f' % (1 - errorrate))


# if __name__ == '__main__':
test('KDDTrain+.csv', 'KDDTest+.csv')