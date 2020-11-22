import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
import sys
from IPython.display import Image

from sklearn.cluster import KMeans
from skimage import io

def plot_data(data):
  sb.set(context="notebook", style="white")
  sb.lmplot('X1', 'X2', data=data, fit_reg=False)
  plt.show()

def find_closest_centroids(X, centroids):
  m = X.shape[0]
  k = centroids.shape[0]
  idx = np.zeros(m)
  
  for i in range(m):
    min_dist = sys.maxsize
    for j in range(k):
      dist = np.sum((X[i,:] - centroids[j,:]) ** 2)
      if dist < min_dist:
        min_dist = dist
        idx[i] = j
  
  return idx

def compute_centroids(X, idx, k):
  m, n = X.shape
  centroids = np.zeros((k, n))
  
  for i in range(k):
    indices = np.where(idx == i)
    centroids[i,:] = (np.sum(X[indices,:], axis=1) / len(indices[0])).ravel()
  
  return centroids

def init_centroids(X, k):
  m, n = X.shape
  centroids = np.zeros((k, n))
  idx = np.random.randint(0, m, k)
  
  for i in range(k):
    centroids[i,:] = X[idx[i],:]
  
  return centroids

def run_k_means(X, initial_centroids, max_iters):
  m, n = X.shape
  k = initial_centroids.shape[0]
  idx = np.zeros(m)
  centroids = initial_centroids
  
  for i in range(max_iters):
    idx = find_closest_centroids(X, centroids)
    centroids = compute_centroids(X, idx, k)
  
  return idx, centroids

def train_plot_k_means(X, max_iters):
  idx, centroids = run_k_means(X, init_centroids(X, 3), 10)
  cluster1 = X[np.where(idx == 0)[0],:]
  cluster2 = X[np.where(idx == 1)[0],:]
  cluster3 = X[np.where(idx == 2)[0],:]

  fig, ax = plt.subplots(figsize=(12,8))
  ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')
  ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
  ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
  ax.legend()
  plt.show()

def train_2D_k_means():
  raw_data = loadmat('data/ex7data2.mat')
  X = raw_data['X']
  train_plot_k_means(X, 10)

def train_image_k_means():
  # 我们可以使用聚类来找到最具代表性的少数颜色，本来有2^24种颜色，现在只用16种颜色来表示原来的图片
  image_data = loadmat('data/bird_small.mat')
  A = image_data['A']# 128 * 128 * 3
  
  A = A / 255.# normalize value ranges
  X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))# reshape the array
  
  # randomly initialize the centroids
  initial_centroids = init_centroids(X, 16)
  # run the algorithm
  idx, centroids = run_k_means(X, initial_centroids, 10)
  # get the closest centroids one last time
  idx = find_closest_centroids(X, centroids)# idx(m,1) centroids(k,n)
  # map each pixel to the centroid value
  X_recovered = centroids[idx.astype(int),:]

  X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))
  plt.imshow(X_recovered)
  plt.show()

def train_image_k_means_sklearn():
  # cast to float, you need to do this otherwise the color would be weird after clustring
  pic = io.imread('data/bird_small.png') / 255.
  # serialize data
  data = pic.reshape(128*128, 3)
  model = KMeans(n_clusters=16, n_init=100, n_jobs=-1)
  model.fit(data)

  centroids = model.cluster_centers_
  C = model.predict(data)

  compressed_pic = centroids[C].reshape((128,128,3))
  fig, ax = plt.subplots(1, 2)
  ax[0].imshow(pic)
  ax[1].imshow(compressed_pic)
  plt.show()

def main():
  train_image_k_means()

  
if __name__ == '__main__':
  main()