#!/usr/bin/env python
# coding: utf-8

# In[11]:


#Load libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.io import loadmat
from scipy.sparse import csc_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy.linalg import norm
import random
import math
import time


# In[2]:


#load data 200 two-dimensional points, 6 cluster centers
data_dot_200_6 = np.loadtxt(r"C:\Users\HAOHAN\Desktop\优化\data\op_data\200_dot_6.txt")
x_dot_200_6 = data_dot_200_6[0]
y_dot_200_6 = data_dot_200_6[1]
label_dot_200_6 = data_dot_200_6[2]

fig, ax = plt.subplots(dpi=200)
plt.title("Artificial Data Set(200 two-dimensional points, 6 cluster centers)")
scatter = ax.scatter(x_dot_200_6, y_dot_200_6, c=label_dot_200_6, cmap=plt.cm.RdYlBu)
plt.show()


# In[61]:


#load wine data
file=r"C:\Users\HAOHAN\Desktop\优化\data\datasets\datasets\wine\wine_data.mat"
data_wine = loadmat(file,mat_dtype=True)["A"].todense().T


# In[6]:


class UnionFindSet:
    def __init__(self, n):
        self.parent = list(range(n))

    #合并index1和index2所属集合
    def union(self, index1: int, index2: int):
        self.parent[self.find(index2)] = self.find(index1)

    #查找index结点的父结点（含路径压缩）
    def find(self, index: int) -> int:
        if self.parent[index] != index:
            self.parent[index] = self.find(self.parent[index])
        return self.parent[index]


# In[136]:


def f(x,a,lambda1,delta):
    f_1 = np.sum((norm(x[i]-a[i]))**2 for i in range(len(x)))
    x_diff = list()
    f_2 = 0
    for i in range(len(a)):
        # for j in range(i+1,len(a)):
        #   x_diff.append(x[i]-x[j])
        xi_diff = x[i] - x[i+1:]
        xi_diff_norm = norm(xi_diff, ord=2, axis=1)
        mask = xi_diff_norm <= delta
        f_2 += np.sum(xi_diff_norm[mask] ** 2 / (2*delta))
        f_2 += np.sum(xi_diff_norm[~mask] - delta/2)

    #f_2 = phi(x_diff,delta)
    return 1/2*(f_1) + lambda1 * f_2

def g(x,a,lambda1,delta):
    grad_1 = x - a
    grad_2 = np.zeros(x.shape)
    for i in range(x.shape[0]):
        xi_diff = x[i] - x[i+1:]
        xi_diff_norm = norm(xi_diff, ord=2, axis=1)
        mask = xi_diff_norm <= delta
        grad_2[i] += lambda1 * np.sum(xi_diff[mask] / delta, axis=0)
        grad_2[i] += lambda1 * np.sum(xi_diff[~mask] / np.expand_dims(xi_diff_norm[~mask], axis=1), axis=0) 

    return grad_1 + grad_2

def g_MiniBatch(x,a,lambda1,delta,batch_size):
    data_index_array = np.random.randint(0, x.shape[0], batch_size)
    grad_1 = x - a
    grad_2 = np.zeros(x.shape)
    for i in data_index_array:
        xi_diff = x[i] - x[i+1:]
        xi_diff_norm = norm(xi_diff, ord=2, axis=1)
        mask = xi_diff_norm <= delta
        grad_2[i] += lambda1 * np.squeeze(np.asarray(np.sum(xi_diff[mask] / delta, axis=0)))
        grad_2[i] += lambda1 * np.squeeze(np.asarray(np.sum(xi_diff[~mask] / np.expand_dims(xi_diff_norm[~mask], axis=1), axis=0)))

    return grad_1 + grad_2


# In[279]:


def GradientMethod(obj,grad,x,options,a,MiniBatch=False):
    T1 = time.time() #计时
    global lambda1
    global delta
    lambda1 = options['lambda']
    delta = options['delta']
    alpha = options['alpha']
    tol = options['tol']
    isprint = options['isprint']
    max_iter = options["max_iter"]
    if MiniBatch: 
        batch_size = options["batch_size"]
        gradient = grad(x,a,lambda1,delta,batch_size)
    else:
        gradient = grad(x,a,lambda1,delta)
    x_lst = [x]
    gradient_lst = [norm(gradient)]
    k = 0
    while norm(gradient) >= tol and k < max_iter:
        k += 1
        x = x - alpha * gradient / k
        if MiniBatch: gradient = grad(x,a,lambda1,delta,batch_size)
        else: gradient = grad(x,a,lambda1,delta)
        if isprint and (k % 10000 == 1):
            print("Iteration:",k-1,"obj:",obj(x,a,lambda1,delta),"norm_gradient:",norm(gradient), "time:{:.2f}s".format(time.time() - T1))
        x_lst.append(x)
        gradient_lst.append(norm(gradient))
    print("Iteration:",k-1,"obj:",obj(x,a,lambda1,delta),"norm_gradient:",norm(gradient))
    T2 = time.time() #计时
    print('程序运行时间:{:.4f}分' .format((T2 - T1)/60))   
    return x,x_lst,gradient_lst


# In[14]:


def plot_convergence_figure(obj, x_lst, gradient_lst, options, x_init):
    lambda1 = options['lambda']
    alpha = options['alpha']

    assert len(x_lst) == len(gradient_lst)
    iteration_num = len(gradient_lst)
    
    #The norm of the gradient VS #iterations
    plt.figure(dpi = 300, figsize=(6, 6.5))
    ax1 = plt.subplot(2, 1, 1)
    scatter = ax1.scatter(range(iteration_num), gradient_lst, s = 5)
    plt.title("The norm of the gradient VS #Iterations")
    ax1.set_xlabel('#iterations')
    ax1.set_ylabel('gradient norm')
    
    #Relative error VS #Iterations
    ax2 = plt.subplot(2, 1, 2)
    best_function_value = obj(x_lst[-1],x_init,lambda1,delta)
    scatter = ax2.scatter(range(iteration_num), [abs(obj(each,x_init,lambda1,delta) - best_function_value) / (max(1, abs(best_function_value))) for each in x_lst], s = 5)
    plt.title("Relative error VS #Iterations")
    ax2.set_xlabel('#iterations')
    ax2.set_ylabel('relative error')
    
    plt.tight_layout()
    plt.show()


# In[ ]:


data = data_dot_200_6.T[:,:2]
delta_value = 1e-4
solution_list_all, x_list_all, gradient_list_all, function_value_list_all = [], [], [], []
for batch_size in [1, 25, 50, 100, 200]:
    options = {
    "lambda":lambda_val,
    "delta":delta_value,
    "tol":1e-1,
    "isprint":True,
    "alpha": 1,
    "max_iter": 30000,
    "batch_size": batch_size
    }
    print(batch_size)
    solution_list, x_list, gradient_list = GradientMethod(f,g_MiniBatch,data,options,data,MiniBatch=True)
    tmp = []
    for each in x_list:
        tmp.append(f(each,data,lambda_val,delta_value))
    function_value_list_all.append(tmp)
    solution_list_all.append(solution_list)
    x_list_all.append(x_list)
    gradient_list_all.append(gradient_list)


# In[ ]:


plt.figure(dpi = 300)
k_list = [1, 25, 50, 100, 200]
for i in range(len(function_value_list_all)):
    plt.plot([each * 50 for each in range(len(function_value_list_all[i][10:5000:50]))], [np.sqrt(norm(each - min(function_value_list_all[i][10:5000:50])) / max(1, abs(min(function_value_list_all[i][10:5000:50])))) for each in function_value_list_all[i][10:5000:50]], label="k="+str(k_list[i]))
plt.legend()
plt.xlabel("#Iterations")
plt.ylabel("Relative Error")
plt.title("Relative Error VS #Iterations in artifitial dataset")
plt.show()


# In[224]:


X = solution_data_dot_200_6_SGD
#print(X)
uf = UnionFindSet(X.shape[0])
# clustering
tolerance = 0.7 # TBD

for i in range(X.shape[0]):
    for j in range(i+1, X.shape[0]):
        # 两两比较距离
        if norm(X[i] - X[j]) < tolerance:
            uf.union(i, j)
# 聚类结果
label_data_dot_200_6 = list()
for i in range(X.shape[0]):
    label_data_dot_200_6.append(uf.find(i)) # 第i条数据的聚类类别

# Visualization
print(len(set(label_data_dot_200_6)))

#predict data
fig, ax = plt.subplots()
scatter = ax.scatter(x_dot_200_6, y_dot_200_6, c=label_data_dot_200_6, cmap=plt.cm.RdYlBu)
plt.show()


# In[280]:


data = data_wine
delta = delta_value
solution_list_all_wine, x_list_all_wine, gradient_list_all_wine  = [], [], []
for batch_size in [1, 50, 100, 178]:
    options = {
    "lambda":lambda_val,
    "delta":delta_value,
    "tol":1e-1,
    "isprint":True,
    "alpha": 0.1,
    "max_iter": 20000,
    "batch_size": batch_size
    }
    print(batch_size)
    solution_list, x_list, gradient_list = GradientMethod(f,g_MiniBatch,data,options,data,MiniBatch=True)
    solution_list_all_wine.append(solution_list)
    x_list_all_wine.append(x_list)
    gradient_list_all_wine.append(gradient_list)


# In[281]:


function_value_list_all_wine = []
for i in range(len(x_list_all_wine)):
    tmp = []
    for each in x_list_all_wine[i]:
        tmp.append(f(each,data,lambda_val,delta_value))
    function_value_list_all_wine.append(tmp)


# In[285]:


plt.figure(dpi = 300)
k_list = [1, 50, 100, 178]
for i in range(len(function_value_list_all_wine)):
    plt.plot([each * 50 for each in range(len(function_value_list_all_wine[i][:20000:50]))], [np.sqrt(norm(each - min(function_value_list_all_wine[i][:20000:50])) / max(1, abs(min(function_value_list_all_wine[i][:20000:50])))) for each in function_value_list_all_wine[i][:20000:50]], label="k="+str(k_list[i]))
plt.legend()
plt.xlabel("#Iterations")
plt.ylabel("Relative Error")
plt.title("Relative Error VS #Iterations in wine dataset")
plt.show()


# In[289]:


plt.figure(dpi = 300)
k_list = [1, 50, 100, 178]
for i in range(len(function_value_list_all_wine)):
    plt.plot([each * 50 for each in range(len(gradient_list_all_wine[i][:20000:50]))], gradient_list_all_wine[i][:20000:50], label="k="+str(k_list[i]))
plt.legend()
plt.xlabel("#Iterations")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm VS #Iterations in wine dataset")
plt.show()


# In[ ]:




