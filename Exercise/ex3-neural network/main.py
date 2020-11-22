import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
import argparse

from sklearn.metrics import classification_report# 这个包是评价报告

def parse_args():
  parser = argparse.ArgumentParser("logistic regression")
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

def load_data(path, transpose=True):
  data = sio.loadmat(path)
  y = data.get('y')  # (5000,1)
  y = y.reshape(y.shape[0])  # make it back to column vector (5000,)

  X = data.get('X')  # (5000,400)

  if transpose:
    # for this dataset, you need a transpose to get the orientation right
    X = np.array([im.reshape((20, 20)).T for im in X])# (5000, 20, 20)

    # and I flat the image again to preserve the vector presentation
    X = np.array([im.reshape(400) for im in X])# (5000, 400)

  return X, y

def plot_an_image(X):
#     """
#     image : (400,)
#     """
  pick_one = np.random.randint(0, 5000)
  image = X[pick_one, :]
  fig, ax = plt.subplots(figsize=(1, 1))
  ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)
  plt.xticks(np.array([]))  # just get rid of ticks
  plt.yticks(np.array([]))
  # plt.show()
  # print('this should be {}'.format(y[pick_one]))

def plot_100_image(X):
  """ sample 100 image and show them
  assume the image is square

  X : (5000, 400)
  """
  size = int(np.sqrt(X.shape[1]))

  # sample 100 image, reshape, reorg it
  sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 100*400
  sample_images = X[sample_idx, :]

  fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))

  for r in range(10):
    for c in range(10):
      ax_array[r, c].matshow(sample_images[10 * r + c].reshape((size, size)),
                             cmap=matplotlib.cm.binary)
      plt.xticks(np.array([]))
      plt.yticks(np.array([]))
  # plt.show()

def cost(theta, X, y):
  ''' cost fn is -l(theta) for you to minimize'''
  return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))

def regularized_cost(theta, X, y, l=1):
  '''you don't penalize theta_0'''
  theta_j1_to_n = theta[1:]
  regularized_term = (l / (2 * len(X))) * np.power(theta_j1_to_n, 2).sum()

  return cost(theta, X, y) + regularized_term

def gradient(theta, X, y):
    '''just 1 batch gradient'''
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)

def regularized_gradient(theta, X, y, l=1):
    '''still, leave theta_0 alone'''
    theta_j1_to_n = theta[1:]
    regularized_theta = (l / len(X)) * theta_j1_to_n

    # by doing this, no offset is on theta_0
    regularized_term = np.concatenate([np.array([0]), regularized_theta])

    return gradient(theta, X, y) + regularized_term

def logistic_regression(X, y, l=1):
  """generalized logistic regression
  args:
      X: feature matrix, (m, n+1) # with incercept x0=1
      y: target vector, (m, )
      l: lambda constant for regularization

  return: trained parameters
  """
  # init theta
  theta = np.zeros(X.shape[1])

  # train it
  res = opt.minimize(fun=regularized_cost,
                     x0=theta,
                     args=(X, y, l),
                     method='TNC',
                     jac=regularized_gradient,
                     options={'disp': True})
  # get trained parameters
  final_theta = res.x

  return final_theta

def predict(x, theta):
  prob = sigmoid(x @ theta)
  return (prob >= 0.5).astype(int)

def train_single_model(X, y):
  t0 = logistic_regression(X, y[0])
  print(t0.shape)
  y_pred = predict(X, t0)
  print('Accuracy={}'.format(np.mean(y[0] == y_pred)))

def train_K_model(X, y, raw_y):
  k_theta = np.array([logistic_regression(X, y[k]) for k in range(10)])
  print(k_theta.shape)

  prob_matrix = sigmoid(X @ k_theta.T)# (5000, 10)
  y_pred = np.argmax(prob_matrix, axis=1)# 返回沿轴axis最大值的索引，axis=1代表行
  # https://www.cnblogs.com/rrttp/p/8028421.html

  y_answer = raw_y.copy()# (5000, 1)
  y_answer[y_answer==10] = 0
  print(classification_report(y_answer, y_pred))

def load_weight(path):
  # 给定一个只包括一个隐藏层的网络，那么有两层参数，参数已给出
  data = sio.loadmat(path)
  return data['Theta1'], data['Theta2']

def feed_forward_prediction():
  theta1, theta2 = load_weight('ex3weights.mat')# (25, 401) (10, 26) 26是因为加了个bias
  X, y = load_data('ex3data1.mat',transpose=False)
  X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)# intercept

  a1 = X

  z2 = a1 @ theta1.T # (5000, 401) @ (25,401).T = (5000, 25)
  z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)# intercept
  a2 = sigmoid(z2)

  z3 = a2 @ theta2.T
  a3 = sigmoid(z3)

  y_pred = np.argmax(a3, axis=1) + 1  # numpy is 0 base index, +1 for matlab convention，返回沿轴axis最大值的索引，axis=1代表行
  print(classification_report(y, y_pred))


def main():
  raw_X, raw_y = load_data('ex3data1.mat')
  # add intercept=1 for x0
  X = np.insert(raw_X, 0, values=np.ones(raw_X.shape[0]), axis=1)# (5000, 401)
  
  # y have 10 categories here. 1..10, they represent digit 0 as category 10 because matlab index start at 1
  # I'll ditit 0, index 0 again
  y_matrix = []
  for k in range(1, 11):
      y_matrix.append((raw_y == k).astype(int))# false would be 0
  # last one is k==10, it's digit 0, bring it to the first position
  y_matrix = [y_matrix[-1]] + y_matrix[:-1]
  y = np.array(y_matrix)
  print("y.shape: "+str(y.shape))# (10, 5000)

  # train_single_model(X, y)
  # train_K_model(X, y, raw_y)
  feed_forward_prediction()
  
if __name__ == '__main__':
  args = parse_args()
  main()
