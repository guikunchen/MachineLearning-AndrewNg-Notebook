import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm
import argparse

def plot_data(data):
  positive = data[data['y'].isin([1])]
  negative = data[data['y'].isin([0])]

  fig, ax = plt.subplots(figsize=(12,8))
  ax.scatter(positive['X1'], positive['X2'], s=30, marker='x', label='Positive')
  ax.scatter(negative['X1'], negative['X2'], s=30, marker='o', label='Negative')
  ax.legend()
  plt.show()

def train_linear_svm(data):
  svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
  svc.fit(data[['X1', 'X2']], data['y'])
  print(svc.score(data[['X1', 'X2']], data['y']))

  data['SVM 1 Confidence'] = svc.decision_function(data[['X1', 'X2']])
  fig, ax = plt.subplots(figsize=(12,8))
  ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM 1 Confidence'], cmap='seismic')
  ax.set_title('SVM (C=1) Decision Confidence')
  plt.show()

  svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)# overfit
  svc2.fit(data[['X1', 'X2']], data['y'])
  print(svc2.score(data[['X1', 'X2']], data['y']))

  data['SVM 2 Confidence'] = svc2.decision_function(data[['X1', 'X2']])
  fig, ax = plt.subplots(figsize=(12,8))
  ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM 2 Confidence'], cmap='seismic')
  ax.set_title('SVM (C=100) Decision Confidence')
  plt.show()

def train_rbf_svm(data):
  svc = svm.SVC(C=100, gamma=10, probability=True)
  svc.fit(data[['X1', 'X2']], data['y'])
  svc.score(data[['X1', 'X2']], data['y'])

  data['Probability'] = svc.predict_proba(data[['X1', 'X2']])[:,0]
  fig, ax = plt.subplots(figsize=(12,8))
  ax.scatter(data['X1'], data['X2'], s=30, c=data['Probability'], cmap='Reds')
  plt.show()

def gaussian_kernel(x1, x2, sigma):
  return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma ** 2))))

def part1_1():
  raw_data = loadmat('data/ex6data1.mat')
  # raw_data = loadmat('data/ex6data2.mat')
  data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
  data['y'] = raw_data['y']

  plot_data(data)
  train_linear_svm(data)
  train_rbf_svm(data)

def part1_2():
  raw_data = loadmat('data/ex6data3.mat')

  X = raw_data['X']
  Xval = raw_data['Xval']
  y = raw_data['y'].ravel()
  yval = raw_data['yval'].ravel()

  C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
  gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

  best_score = 0
  best_params = {'C': None, 'gamma': None}

  for C in C_values:
    for gamma in gamma_values:
      svc = svm.SVC(C=C, gamma=gamma)
      svc.fit(X, y)
      score = svc.score(Xval, yval)
          
      if score > best_score:
        best_score = score
        best_params['C'] = C
        best_params['gamma'] = gamma

  print("best_score: " + str(best_score) + ", " + str(best_params))

def part2():
  spam_train = loadmat('data/spamTrain.mat')
  spam_test = loadmat('data/spamTest.mat')

  X = spam_train['X']
  Xtest = spam_test['Xtest']
  y = spam_train['y'].ravel()
  ytest = spam_test['ytest'].ravel()

  svc = svm.SVC()
  svc.fit(X, y)
  print('Training accuracy = {0}%'.format(np.round(svc.score(X, y) * 100, 2)))
  print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))



def main():
  # part1_1()
  # part1_2()
  part2()
  
if __name__ == '__main__':
  main()