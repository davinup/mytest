import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def readData(path, name=[]):
    data = pd.read_csv(path, names=name)
    data = (data - data.mean()) / data.std()
    data.insert(0, 'First', 1)
    return data


def costFunction(X, Y, theta):
    inner = np.power(((X * theta.T) - Y.T), 2)
    return np.sum(inner) / (2 * len(X))


def gradientDescent(data, theta, alpha, iterations):
    eachIterationValue = np.zeros((iterations, 1))
    temp = np.matrix(np.zeros(theta.shape))
    X = np.matrix(data.iloc[:, 0:-1].values)
    print(X)
    Y = np.matrix(data.iloc[:, -1].values)
    m = X.shape[0]
    colNum = X.shape[1]
    for i in range(iterations):
        error = (X * theta.T) - Y.T
        for j in range(colNum):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / m) * np.sum(term))
        theta = temp
        eachIterationValue[i, 0] = costFunction(X, Y, theta)
    return theta, eachIterationValue


if __name__ == "__main__":
    data = readData('ex1data2.txt', ['Size', 'Bedrooms', 'Price'])
    # data = (data - data.mean()) / data.std()
    theta = np.matrix(np.array([0, 0, 0]))

    iterations = 1500
    alpha = 0.01

    theta, eachIterationValue = gradientDescent(data, theta, alpha, iterations)

    print(theta)

    plt.plot(np.arange(iterations), eachIterationValue)
    plt.title('CostFunction')
    plt.show()

