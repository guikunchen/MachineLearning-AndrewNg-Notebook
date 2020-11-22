import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

import scipy.optimize as opt# find optimum parameters
from sklearn import linear_model# 调用sklearn的线性回归包

def parse_args():
    parser = argparse.ArgumentParser("logistic regression")
    parser.add_argument(
        '--regularized',
        type=int,
        default=1,
        help="If set, run the task with regularization.")
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

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def computeCost(theta, X, y):# 为了配合opt.fmin_tnc，theta必须放前面
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

def gradient(theta, X, y):# 为了配合opt.fmin_tnc，theta必须放前面
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    
    return grad

def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

def extractData(data):
  # set X (training data) and y (target variable)
  cols = data.shape[1]
  X = data.iloc[:,0:cols-1]
  y = data.iloc[:,cols-1:cols]

  # convert to numpy arrays and initalize the parameter array theta
  X = np.array(X.values)
  y = np.array(y.values)
  theta = np.zeros([1, cols - 1])
  
  return X, y, theta

def nonRegularizedLogisticRegression():
  path = 'ex2data1.txt'
  data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

  data.insert(0, 'Ones', 1)# add a ones column - this makes the matrix multiplication work out easier
  X, y, theta = extractData(data)

  result = opt.fmin_tnc(func=computeCost, x0=theta, fprime=gradient, args=(X, y))

  theta_min = np.matrix(result[0])
  predictions = predict(theta_min, X)
  correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
  accuracy = (sum(map(int, correct)) / len(correct) * 100)
  
  print('accuracy of NonReg: {0}'.format(accuracy))
  # showNonRegRes(data, result)

def computeCostReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg

# def gradientReg(theta, X, y, learningRate):
#   # leave j0 alone
#   theta_j1_to_n = theta[:,1:]
#   regularized_theta = (learningRate / len(X)) * theta_j1_to_n

#   # by doing this, no offset is on theta_0
#   regularized_term = np.concatenate([np.zeros([1,1]), regularized_theta], axis=1)

#   return gradient(theta, X, y) + regularized_term

def gradientReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])
    return grad


def regularizedLogisticRegression():
  path =  'ex2data2.txt'
  data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

  # 添加多项式，删除原本的两列
  degree = 6
  x1 = data['Test 1']
  x2 = data['Test 2']

  data.insert(3, 'Ones', 1)

  for i in range(1,degree+1):
    for j in range(0,i+1):
        data['F'+str(i-j)+str(j)]=np.power(x1,i-j)*np.power(x2,j)

  data.drop('Test 1', axis=1, inplace=True)
  data.drop('Test 2', axis=1, inplace=True)

  cols=data.shape[1]
  X = data.iloc[:,1:cols]
  y = data.iloc[:,0:1]

  theta = np.zeros(cols-1)
  X=np.array(X.values)
  y=np.array(y.values)
  learningRate = 1# try different lambda

  result = opt.fmin_tnc(func=computeCostReg, x0=theta, fprime=gradientReg, args=(X, y, learningRate))

  theta_min = np.matrix(result[0])
  predictions = predict(theta_min, X)
  correct = [1 if ((a == b)) else 0 for (a, b) in zip(predictions, y)]
  accuracy = (sum(map(int, correct)) /len(correct))*100

  print('accuracy of Reg: {0}'.format(accuracy))
  # showRegRes(pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted']), result)

def trainLogisticBySklearn():
  path =  'ex2data2.txt'
  data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

  degree = 6
  x1 = data['Test 1']
  x2 = data['Test 2']

  data.insert(3, 'Ones', 1)

  for i in range(1,degree+1):
    for j in range(0,i+1):
        data['F'+str(i-j)+str(j)]=np.power(x1,i-j)*np.power(x2,j)

  data.drop('Test 1', axis=1, inplace=True)
  data.drop('Test 2', axis=1, inplace=True)

  cols=data.shape[1]
  X = data.iloc[:,1:cols]
  y = data.iloc[:,0:1]

  X=np.array(X.values)
  y=np.array(y.values)

  model = linear_model.LogisticRegression(penalty='l2', C=1.0)
  model.fit(X, y.ravel())
  print('accuracy of sklearn lib: ' + str(model.score(X, y)))

def showNonRegData(data):
  positive = data[data['Admitted'].isin([1])]
  negative = data[data['Admitted'].isin([0])]

  fig, ax = plt.subplots(figsize=(12,8))
  ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
  ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
  ax.legend()
  ax.set_xlabel('Exam 1 Score')
  ax.set_ylabel('Exam 2 Score')
  plt.show()

def showNonRegRes(data, result):
  positive = data[data['Admitted'].isin([1])]
  negative = data[data['Admitted'].isin([0])]
  plotting_x1 =np.linspace(30,100,100)
  plotting_h1 =(-result[0][0]-result[0][1]*plotting_x1)/result[0][2]
  # θ0+θ1x+θ2y=0
  fig,ax= plt.subplots(figsize=(12,8))
  ax.plot(plotting_x1,plotting_h1,'y',label='prediction')
  ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
  ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
  ax.legend()
  ax.set_xlabel('Exam 1 Score')
  ax.set_ylabel('Exam 2 Score')
  plt.show()

def showRegData(data):
  postive2=data_init[data_init['Accepted'].isin([1])]
  negative2=data_init[data_init['Accepted'].isin([0])]

  fig,ax=plt.subplots(figsize=(12,8))
  ax.scatter(postive2['Test 1'],postive2['Test 2'],s=50,c='b',marker='o',label='Accepted')
  ax.scatter(negative2['Test 1'],negative2['Test 2'],s=50,c='r',marker='x',label=' Not Accepted')

def hfunc2(result, x1, x2, degree=6):
  temp = result[0][0]
  place = 0
  for i in range(1, degree+1):
      for j in range(0, i+1):
          temp+= np.power(x1, i-j) * np.power(x2, j) * result[0][place+1]
          place+=1
  return temp

def find_decision_boundary(result):
  t1=np.linspace(-1,1.5,1000)
  t2=np.linspace(-1,1.5,1000)
  cordinates=[(x,y)for x in t1 for y in t2]
  x_cord,y_cord=zip(*cordinates)
  h_val=pd.DataFrame({'x1':x_cord,'x2':y_cord})
  h_val['hval']=hfunc2(result, h_val['x1'], h_val['x2'])

  decison=h_val[np.abs(sigmoid(h_val['hval'])-0.5)<0.01]
  return decison.x1,decison.x2

def showRegRes(data, result):
  postive2=data[data['Accepted'].isin([1])]
  negative2=data[data['Accepted'].isin([0])]
  fig, ax = plt.subplots(figsize=(12,8))
  ax.scatter(postive2['Test 1'], postive2['Test 2'], s=50, c='b', marker='o', label='Accepted')
  ax.scatter(negative2['Test 1'], negative2['Test 2'], s=50, c='r', marker='x', label='Rejected')
  ax.set_xlabel('Test 1 Score')
  ax.set_ylabel('Test 2 Score')

  x, y = find_decision_boundary(result)
  plt.scatter(x, y, c='y', s=10, label='Prediction')
  ax.legend()
  plt.show()


def main():
  if args.regularized == 1:
    regularizedLogisticRegression()
  else:
    nonRegularizedLogisticRegression()

  trainLogisticBySklearn()
  

if __name__ == '__main__':
  args = parse_args()
  main()
