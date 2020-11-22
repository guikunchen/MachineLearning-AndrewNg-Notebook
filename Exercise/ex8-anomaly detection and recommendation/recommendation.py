import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from scipy import stats
from scipy.optimize import minimize

def plot_data(data):
  Y = data['Y']# Y是包含从1到5的等级的（数量的电影x数量的用户）数组。
  R = data['R']# R是包含指示用户是否给电影评分的二进制值的“指示符”数组。

  #可视化评分矩阵，MATLAB中imagesc(A)将矩阵A中的元素数值按大小转化为不同颜色，并在坐标轴对应位置处以这种颜色染色
  plt.figure(figsize=(5,9))
  plt.imshow(Y)
  plt.colorbar() #加颜色条
  plt.ylabel('Movies')
  plt.xlabel('Users')
  plt.show()

  # fig, ax = plt.subplots(figsize=(12,12))
  # ax.imshow(Y)
  # ax.set_xlabel('Users')
  # ax.set_ylabel('Movies')
  # fig.tight_layout()
  # plt.show()

def cost_func(params, Y, R, num_features):
  Y = np.matrix(Y)  # (1682, 943)
  R = np.matrix(R)  # (1682, 943)
  num_movies = Y.shape[0]
  num_users = Y.shape[1]
  
  # reshape the parameter array into parameter matrices
  X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))  # (1682, 10)
  Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))  # (943, 10)
  
  # initializations
  J = 0
  X_grad = np.zeros(X.shape)  # (1682, 10)
  Theta_grad = np.zeros(Theta.shape)  # (943, 10)
  
  # compute the cost
  error = np.multiply((X * Theta.T) - Y, R)  # (1682, 943)
  squared_error = np.power(error, 2)  # (1682, 943)
  J = (1. / 2) * np.sum(squared_error)
  
  # calculate the gradients
  X_grad = error * Theta
  Theta_grad = error.T * X
  
  # unravel the gradient matrices into a single array
  grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))
  
  return J, grad

def regularized_cost_func(params, Y, R, num_features, learning_rate = 0):
  Y = np.matrix(Y)  # (1682, 943)
  R = np.matrix(R)  # (1682, 943)
  num_movies = Y.shape[0]
  num_users = Y.shape[1]
  
  # reshape the parameter array into parameter matrices
  X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))  # (1682, 10)
  Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))  # (943, 10)
  
  # initializations
  J = 0
  X_grad = np.zeros(X.shape)  # (1682, 10)
  Theta_grad = np.zeros(Theta.shape)  # (943, 10)
  
  # compute the cost
  error = np.multiply((X * Theta.T) - Y, R)  # (1682, 943)
  squared_error = np.power(error, 2)  # (1682, 943)
  J = (1. / 2) * np.sum(squared_error)
  
  # add the cost regularization
  J = J + ((learning_rate / 2) * np.sum(np.power(Theta, 2)))
  J = J + ((learning_rate / 2) * np.sum(np.power(X, 2)))
  
  # calculate the gradients with regularization
  X_grad = (error * Theta) + (learning_rate * X)
  Theta_grad = (error.T * X) + (learning_rate * Theta)
  
  # unravel the gradient matrices into a single array
  grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))
  
  return J, grad

def test_cost_func(Y, R, X, Theta, requied_regularzed = True):
  users = 4
  movies = 5
  features = 3

  X_sub = X[:movies, :features]
  Theta_sub = Theta[:users, :features]
  Y_sub = Y[:movies, :users]
  R_sub = R[:movies, :users]

  params = np.concatenate((np.ravel(X_sub), np.ravel(Theta_sub)))

  cur_cost_func = regularized_cost_func if requied_regularzed else cost_func
  args = (Y_sub, R_sub, features, 10) if requied_regularzed else (Y_sub, R_sub, features)

  #计算数值梯度与理论梯度之间的差值
  num_grad = get_numerical_gradient(cur_cost_func, params, args=args)
  _, grad = cur_cost_func(params, *args)
  diff = np.linalg.norm(num_grad-grad) / np.linalg.norm(num_grad+grad)
  print('If your cost function implementation is correct, then \n' 
           'the relative difference will be small (less than 1e-9). \n' 
           'Relative Difference: %g\n'% diff)

def get_movie_dic():
  movie_list = []
  f = open('data/movie_ids.txt', encoding= 'gbk')
  for line in f:
    tokens = line.split(' ')
    tokens[-1] = tokens[-1][:-1]# remove the last character '\n'
    movie_list.append(' '.join(tokens[1:]))
  return movie_list

def get_my_ratings():
  # We can add our own ratings vector to the existing data set to include in the model.
  my_ratings = np.zeros((1682, 1))

  my_ratings[0] = 4
  my_ratings[6] = 3
  my_ratings[11] = 5
  my_ratings[53] = 4
  my_ratings[63] = 5
  my_ratings[65] = 3
  my_ratings[68] = 5
  my_ratings[97] = 2
  my_ratings[182] = 4
  my_ratings[225] = 5
  my_ratings[354] = 5

  return my_ratings

def get_numerical_gradient(J, params, args=()):
  num_grad = np.zeros(params.shape)
  perturb = np.zeros(params.shape)
  e = 1e-4
  for p in range(len(params)):
    perturb[p] = e
    loss1, _ = J(params - perturb, *args)
    loss2, _ = J(params + perturb, *args)
    num_grad[p] = (loss2 - loss1) / (2*e)
    perturb[p] = 0
  return num_grad

def collaborative_filtering(Y, R):
  my_ratings = get_my_ratings()
  # add myself
  Y = np.append(Y, my_ratings, axis=1)
  R = np.append(R, my_ratings != 0, axis=1)# true of false

  num_users = Y.shape[1]# 1682
  num_movies = Y.shape[0]# 944
  num_features = 10
  learning_rate = 10.

  def normalize_ratings(Y, R):
    #为了下面可以直接矩阵相减，将(1682,)reshape成(1682,1)
    mu = (np.sum(Y, axis=1)/np.sum(R, axis=1)).reshape((len(Y),1)) 
    Y_norm = (Y - mu)*R #未评分的依然为0
    return Y_norm, mu

  Ynorm, Ymean = normalize_ratings(Y, R)

  X = np.random.randn(num_movies, num_features)# small random value
  Theta = np.random.randn(num_users, num_features)# 导致结果是不可复现的
  init_params = np.r_[X.flatten(), Theta.flatten()]

  # 应该用Ynorm跑比较合理，最后得出来的值在0-5范围内，如果用Y来跑的话最后再加Ymean评分就会超过5
  # ex8.pdf里面应该就是用Y来进行训练，那normalization都白做了
  min_params = minimize(fun=regularized_cost_func, x0=init_params, args=(Ynorm, R, num_features, learning_rate), 
                method='CG', jac=True, options={'maxiter': 100})

  X = np.matrix(np.reshape(min_params.x[:num_movies * num_features], (num_movies, num_features)))
  Theta = np.matrix(np.reshape(min_params.x[num_movies * num_features:], (num_users, num_features)))

  return X, Theta, Ymean


def main():

  data = loadmat('data/ex8_movies.mat')
  Y = data['Y']# Y是包含从1到5的等级的（数量的电影x数量的用户）数组。
  R = data['R']# R是包含指示用户是否给电影评分的二进制值的“指示符”数组。

  params_data = loadmat('data/ex8_movieParams.mat')
  X = params_data['X']
  Theta = params_data['Theta']

  # plot_data(data)
  # test_cost_func(Y, R, X, Theta)

  movie_list = get_movie_dic()

  X, Theta, Ymean = collaborative_filtering(Y, R)
  predictions = X * Theta.T 
  my_preds = predictions[:, -1] + Ymean# predict my ratings

  idx = np.argsort(my_preds, axis=0)[::-1]# [::-1] means to reverse the order
  print("Top 10 movie predictions:")
  for i in range(10):
    j = int(idx[i])
    print('Predicted rating of {0} for movie {1}.'.format(str(float(my_preds[j])), movie_list[j]))

if __name__ == '__main__':
  main()