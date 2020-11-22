import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm
import argparse

def plot_data(X):
  fig, ax = plt.subplots(figsize=(12,8))
  ax.scatter(list(X[:, 0]), list(X[:, 1]))
  plt.show()

def pca(X):
  # normalize the features
  X = (X - X.mean()) / X.std()
  
  # compute the covariance matrix
  X = np.matrix(X)
  cov = (X.T * X) / X.shape[0]
  
  # perform SVD
  U, S, V = np.linalg.svd(cov)
  
  return U, S, V

def project_data(X, U, k):
  U_reduced = U[:,:k]
  return np.dot(X, U_reduced)

def recover_data(Z, U, k):
  U_reduced = U[:,:k]
  return np.dot(Z, U_reduced.T)

def train_2D_pca():
  raw_data = loadmat('data/ex7data1.mat')
  X = raw_data['X']

  U, S, V = pca(X)

  Z = project_data(X, U, 1)

  X_recovered = recover_data(Z, U, 1)

def plot_n_image(X, n):
  """ plot first n images
  n has to be a square number
  """
  pic_size = int(np.sqrt(X.shape[1]))
  grid_size = int(np.sqrt(n))

  first_n_images = X[:n, :]

  fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size,
                              sharey=True, sharex=True, figsize=(8, 8))

  for r in range(grid_size):
    for c in range(grid_size):
      ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)))
      plt.xticks(np.array([]))
      plt.yticks(np.array([]))

def train_image_pca():
  faces = loadmat('data/ex7faces.mat')
  X = faces['X']

  fig, ax = plt.subplots(1, 2)
  face = np.reshape(X[3,:], (32, 32))
  ax[0].imshow(face)
  
  U, S, V = pca(X)
  Z = project_data(X, U, 100)

  X_recovered = recover_data(Z, U, 100)
  face = np.reshape(X_recovered[3,:], (32, 32))
  ax[1].imshow(face)

  plt.show()


def main():
  train_2D_pca()
  # train_image_pca()

  
if __name__ == '__main__':
  main()