import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from scipy import stats

def plot_data(data):
  X = data['X']

  plt.scatter(x[:,0], x[:,1], marker='x', c='b', s=10)
  plt.xlim((0,30))
  plt.ylim((0,30))
  plt.xlabel('Latency (ms)')
  plt.ylabel('Throughput (mb/s)')

def estimate_Gaussian(X):
  mu = X.mean(axis=0)
  sigma2 = X.var(axis=0)
  
  return mu, sigma2

def multivariate_Gaussian(X, mu, sigma2):
  p = np.zeros((X.shape[0], 1))
  n = len(mu)
  if np.ndim(sigma2) == 1:# 原始模型是多元高斯分布的一个特例
    sigma2 = np.diag(sigma2) #对角阵

  for i in range(X.shape[0]):
    p[i] = (2*np.pi)**(-n/2) * np.linalg.det(sigma2)**(-1/2) * np.exp(-0.5*(X[i,:]-mu).T@np.linalg.inv(sigma2)@(X[i,:]-mu))
  return p

def visualize_fit(X, mu, sigma2):
  a = np.linspace(0, 30, 100)
  b = np.linspace(0, 30, 100)
  aa, bb = np.meshgrid(a, b)
  z = multivariate_Gaussian(np.c_[aa.flatten(), bb.flatten()], mu, sigma2)
  z = z.reshape(aa.shape)
  levels = [10**h for h in range(-20,0,3)]
  plt.contour(a, b, z, levels, linewidths=1)
  plt.scatter(X[:,0], X[:,1], marker='x', c='b', s=10)
  plt.xlabel('Latency (ms)')
  plt.ylabel('Throughput (mb/s)')

def select_threshold(pval, yval):
  best_f1 = 0
  best_epsilon = 0

  for epsilon in np.linspace(min(pval), max(pval), 1001):
    y_predict = np.zeros(yval.shape)
    y_predict[pval<epsilon] = 1 #把小于阈值的设为1
    tp = np.sum(y_predict[yval==1]) #真正例
    precision = tp/np.sum(y_predict) #查准率
    recall = tp/np.sum(yval) #查全率
    f1 = (2 * precision * recall) / (precision + recall)

    if f1 > best_f1:
      best_f1 = f1
      best_epsilon = epsilon
  return best_epsilon, best_f1

def detect_2D():
  data = loadmat('data/ex8data1.mat')
  X = data['X']
  Xval = data['Xval']
  yval = data['yval']

  # 假设每个特征服从高斯分布，求高斯分布的参数
  mu, sigma2 = estimate_Gaussian(X)

  #得到概率向量
  p = multivariate_Gaussian(X, mu, sigma2) 

  #验证集的概率向量
  pval = multivariate_Gaussian(Xval, mu, sigma2)

  epsilon, f1 = select_threshold(pval, yval)

  outliers = np.array([X[i] for i in range(len(X)) if p[i] < epsilon]) # 概率小于阈值记为异常点
  plt.figure()
  plt.scatter(outliers[:,0], outliers[:,1], s=100, marker='o', facecolors='none', edgecolors='r', linewidths=2)
  visualize_fit(X, mu, sigma2)
  plt.show()

def detect_nD():
  #导入数据
  data = loadmat('data/ex8data2.mat')
  X = data['X'] #(1000, 11)
  Xval = data['Xval'] #(100, 11)
  yval = data['yval']

  #估计高斯分布的参数
  mu, sigma2 = estimate_Gaussian(X)

  #得到概率向量
  p = multivariate_Gaussian(X, mu, sigma2) 

  #验证集的概率向量
  pval = multivariate_Gaussian(Xval, mu, sigma2)

  #寻找最优阈值
  epsilon, f1 = select_threshold(pval, yval) #(1.377228890761358e-18, 0.6153846153846154)

  #找出异常点
  outliers = np.array([X[i] for i in range(len(X)) if p[i] < epsilon]) # 概率小于阈值记为异常点

  #输出结果
  print('Best epsilon found using cross-validation: %e\n' %epsilon)
  print('Best F1 on Cross Validation Set:  %f\n' %f1)
  print('# Outliers found: %d\n\n' %len(outliers))

def main():
  # detect_2D()
  detect_nD()
    
if __name__ == '__main__':
  main()