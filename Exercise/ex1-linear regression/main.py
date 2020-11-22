import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from sklearn import linear_model

def parse_args():
    parser = argparse.ArgumentParser("fit_a_line")
    parser.add_argument(
        '--train_multiple',
        type=int,
        default=0,
        help="If set, run the task with multiple features.")
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=1000, 
        help="number of epochs.")
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01, 
        help="learning rate.")
    args = parser.parse_args()
    return args

def linearRegression(data):
  addColumn(data, 0, 'ones', 1)

  cols = data.shape[1]
  X = data.iloc[:,0:cols-1]# 除了最后一列
  y = data.iloc[:,cols-1:cols]# 最后一列

  # 转换数据类型，初始化theta
  X = np.matrix(X.values)
  y = np.matrix(y.values)
  theta = np.matrix(np.zeros([1, cols - 1]))
  
  theta, cost = gradientDescent(X, y, theta, args.lr, args.num_epochs)# training
  return theta, cost

def singleLinearRegression():
  path = 'ex1data1.txt'
  data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
  theta, cost = linearRegression(data)

  print("gradinet")
  print(theta)

  x = np.linspace(data.Population.min(), data.Population.max(), 100)# 在区间内生成100个均匀分布的点
  f = theta[0, 0] + (theta[0, 1] * x)# line

  fig, ax = plt.subplots(figsize=(12,8))
  ax.plot(x, f, 'r', label='Prediction')
  ax.scatter(data.Population, data.Profit, label='Traning Data')
  ax.legend(loc='best')
  ax.set_xlabel('Population')
  ax.set_ylabel('Profit')
  ax.set_title('Predicted Profit vs. Population Size')
  plt.show()
  # fig.savefig("fit.jpg")

  # fig, ax = plt.subplots(figsize=(12,8))
  # ax.plot(np.arange(args.num_epochs), cost, 'r')
  # ax.set_xlabel('Iterations')
  # ax.set_ylabel('Cost')
  # ax.set_title('Error vs. Training Epoch')
  # plt.show()
  # fig.savefig("training.jpg")

def multipleLinearRegression():
  path = 'ex1data2.txt'
  data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
  data = (data - data.mean()) / data.std()# 特征归一化
  theta, cost = linearRegression(data)

  fig, ax = plt.subplots(figsize=(12,8))
  ax.plot(np.arange(args.num_epochs), cost, 'r')
  ax.set_xlabel('Iterations')
  ax.set_ylabel('Cost')
  ax.set_title('Error vs. Training Epoch')
  plt.show()

def computeCost(X, y, theta):
  inner = np.power((X * theta.T - y), 2)
  return np.sum(inner) / (2 * len(X))

def gradientDescent(X, y, theta, alpha, iters):
  temp = np.matrix(np.zeros_like(theta, dtype = float))
  # parameters = theta.ravel().shape[1] # ravel会修改原来的值
  parameters = theta.flatten().shape[1] # flatten不会
  cost = np.zeros(iters)

  for iter in range(iters):
    error = (X * theta.T) - y

    for i in range(parameters):
      term = np.multiply(error, X[:,i])# 逗号表示隔开了一个维度，这里取所有行，i列
      temp[0,i] = theta[0,i] - ((alpha / len(X)) * np.sum(term))

    theta = temp
    cost[iter] = computeCost(X, y, theta)

  return theta, cost

def addColumn(data, index, name, num):# 使模型可转换成向量乘法
  data.insert(index, name, num)

def trainLinearBySklearn():
  path = 'ex1data2.txt'
  data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
  data = (data - data.mean()) / data.std()# 特征归一化
  addColumn(data, 0, 'ones', 1)
  cols = data.shape[1]
  X = data.iloc[:,0:cols-1]# 除了最后一列
  y = data.iloc[:,cols-1:cols]# 最后一列
  # 转换数据类型，初始化theta
  X = np.matrix(X.values)
  y = np.matrix(y.values)

  model = linear_model.LinearRegression()
  model.fit(X, y)

  x = np.array(X[:, 1].A1)
  f = model.predict(X).flatten()

  fig, ax = plt.subplots(figsize=(12,8))
  ax.plot(x, f, 'r', label='Prediction')
  ax.scatter(data.Population, data.Profit, label='Traning Data')
  ax.legend(loc=2)
  ax.set_xlabel('Population')
  ax.set_ylabel('Profit')
  ax.set_title('Predicted Profit vs. Population Size')
  plt.show()

def normalEqn():
  path = 'ex1data1.txt'
  data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
  addColumn(data, 0, 'ones', 1)
  cols = data.shape[1]
  X = data.iloc[:,0:cols-1]# 除了最后一列
  y = data.iloc[:,cols-1:cols]# 最后一列
  # 转换数据类型，初始化theta
  X = np.matrix(X.values)
  y = np.matrix(y.values)

  theta = np.linalg.inv(X.T@X)@X.T@y# X.T@X等价于X.T.dot(X)

  print("normal")
  print(theta)

  x = np.linspace(data.Population.min(), data.Population.max(), 100)# 在区间内生成100个均匀分布的点
  f = theta[0, 0] + (theta[1, 0] * x)# line

  fig, ax = plt.subplots(figsize=(12,8))
  ax.plot(x, f, 'r', label='Prediction')
  ax.scatter(data.Population, data.Profit, label='Traning Data')
  ax.legend(loc='best')
  ax.set_xlabel('Population')
  ax.set_ylabel('Profit')
  ax.set_title('Predicted Profit vs. Population Size')
  plt.show()

def main():
  if args.train_multiple == 1:
    multipleLinearRegression()
  else:
    singleLinearRegression()

  normalEqn()

if __name__ == '__main__':
  args = parse_args()
  main()
