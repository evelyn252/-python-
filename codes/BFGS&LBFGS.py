import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
from scipy.sparse import csc_matrix
import time

# load data
data_200_dot_3 = np.loadtxt('op_data/200_dot_3.txt')
data_200_dot_6 = np.loadtxt('op_data/200_dot_6.txt')
data_2000_dot_3 = np.loadtxt('op_data/2000_dot_6.txt')
wine_data_mat = loadmat("wine_data.mat",mat_dtype=True)
wine_label_mat = loadmat("wine_label.mat",mat_dtype=True)
wine_data=wine_data_mat['A']
wine_label=wine_label_mat['b']


def function(x):
    '''
    Parameters
    ----------
    x : numpy.ndarray
        shape: (row*col,1)
        DESCRIPTION: The independent variable of function 'f_clust'.
        
    Returns
    -------
    type: float
        DESCRIPTION: The solution of f_clust.
        
    PS: A, row, col, lamd, delta all store as global variables.
    '''
    f1=[np.power(np.linalg.norm(x[i*row:(i+1)*row]-A[:, i].reshape(-1, 1)),2) for i in range(col)]
    f2=0
    for i in range(col-1):
        xi_xj_norm=np.asarray(np.linalg.norm([x[i*row:(i+1)*row]-x[j*row:(j+1)*row] for j in range(i+1,col)],axis=1,ord=2))       
        mask=xi_xj_norm<=delta
        f2+=sum(xi_xj_norm[mask]**2*0.5/delta)
        f2+=sum(xi_xj_norm[~mask]-delta*0.5)
    return 0.5*sum(f1)+lamd*f2


def gradient(x):
    '''

    Parameters
    ----------
    x : numpy.ndarray
        shape: (row*col,1)
        DESCRIPTION: The independent variable of gradient function.
    Returns
    -------
    type: numpy.ndarray
          shape: (row*col,1)
          DESCRIPTION: The solution of gradient function.

    '''

    solution1=np.asarray(np.concatenate(list(map(lambda i:x[i*row:(i+1)*row]-A[:, i].reshape(-1, 1),range(col))),axis=0))
    solution2=np.zeros([row*col,1])
    for i in range(col):
        xi_xj=np.asarray([x[i*row:(i+1)*row]-x[j*row:(j+1)*row] for j in range(col)]).reshape((col,row),order='C')
        m=np.all(np.equal(xi_xj, 0), axis=1) 
        xi_xj=xi_xj[~m] #去掉0项
        xi_xj_norm=np.linalg.norm(xi_xj,axis=1,ord=2)
        mask=xi_xj_norm<=delta
        solution2[i*row:(i+1)*row]+=(np.sum(xi_xj[mask]/delta,axis=0)).reshape(row,1)
        solution2[i*row:(i+1)*row]+=(np.sum(xi_xj[~mask]/np.expand_dims(xi_xj_norm[~mask],axis=1),axis=0)).reshape(row,1)
    return solution1+lamd*solution2


def plot_convergence_figure(obj, x_lst, gradient_lst, x_init):

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
    best_function_value = obj(x_lst[-1])
    scatter = ax2.scatter(range(iteration_num), [abs(obj(each) - best_function_value) / (max(1, abs(best_function_value))) for each in x_lst], s = 5)
    plt.title("Relative error VS #Iterations")
    ax2.set_xlabel('#iterations')
    ax2.set_ylabel('relative error')
    
    plt.tight_layout()
    plt.show()



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


def BFGS(func, grad, x0, options):
    epsilon = options['tol']
    maxiter = options['maxiter']
    k = 0  #number of iterations
    n = np.shape(x0)[0]  
    Bk = np.eye(n) #initial Hessian matrix is I
    alpha = s
    x_lst = [x0]
    norm_grad_lst = [norm(grad(x0))]


    while k < maxiter:
        gk1 = grad(x0)
        if norm(gk1) < epsilon:
            break
        dk = -1.0*np.linalg.solve(Bk,gk1)
        while func(x0 + alpha * dk) > func(x0) + gamma * alpha * np.dot(gk1.T, dk)[0,0]:
            alpha = sigma * alpha 
        
        x1 = x0 + alpha * dk
        x_lst.append(x1)
        norm_gk1 = norm(grad(x1))
        norm_grad_lst.append(norm_gk1)
        print("ITER:{:0>6d}".format(len(x_lst)-1),end='   ')
        print("OBJ:", func(x1),end='   ')
        print("NORM:", norm_gk1,end='\n')

        #BFGS
        sk = x1 - x0
        yk = gk1 - grad(x0)   

        if np.dot(sk.T,yk) > 0:    
            Bs = np.dot(Bk,sk)
            ys = np.dot(yk.T,sk)
            sBs = np.dot(np.dot(sk.T,Bk),sk) 
            Bk = Bk - Bs.reshape((n,1))*Bs/sBs + yk.reshape((n,1))*yk/ys

        k += 1
        x0 = x1
    x = x0
    return x_lst ,norm_grad_lst, x, k


def LBFGS(fun, grad, x0, options):
    epsilon = options['tol']
    maxiter = options['maxiter']
    rho = 0.55
    sigma = 0.4
    
    H0 = np.eye(np.shape(x0)[0])

    x_lst = [x0]
    norm_grad_lst = [norm(grad(x0))]
    
    #s和y用于保存近期m个，这里m取6
    s = []
    y = []
    m = 6
    
    k = 1
    gk = np.mat(grad(x0))#计算梯度
    dk = -H0 * gk
    while (k < maxiter):  
        gk1 = grad(x0)
        if norm(gk1) < epsilon:
            break           
        n = 0
        mk = 0
        gk = np.mat(grad(x0))#计算梯度
        while (n < 20):
            newf = fun(x0 + rho ** n * dk)
            oldf = fun(x0)
            if (newf < oldf + sigma * (rho ** n) * (gk.T * dk)[0, 0]):
                mk = n
                break
            n = n + 1
        
        #LBFGS校正
        x = x0 + rho ** mk * dk
        #print x
        
        #保留m个
        if k > m:
            s.pop(0)
            y.pop(0)
            
        #计算最新的
        sk = x - x0
        yk = grad(x) - gk
        
        s.append(sk)
        y.append(yk)
        
        #two-loop的过程
        t = len(s)
        qk = grad(x)
        a = []
        for i in range(t):
            alpha = (s[t - i - 1].T * qk) / (y[t - i - 1].T * s[t - i - 1])
            qk = qk - alpha[0, 0] * y[t - i - 1]
            a.append(alpha[0, 0])
        r = H0 * qk
            
        for i in range(t):
            beta = (y[i].T * r) / (y[i].T * s[i])
            r = r + s[i] * (a[t - i - 1] - beta[0, 0])
 
            
        if (yk.T * sk > 0):
            dk = -r            
        
        k = k + 1
        x_lst.append(x)
        norm_gk1 = norm(grad(x))
        norm_grad_lst.append(norm_gk1)
        print("ITER:{:0>6d}".format(len(x_lst)-1),end='   ')
        print("OBJ:", fun(x),end='   ')
        print("NORM:", norm_gk1,end='\n')
        x0 = x
        
    x = x0
    return x_lst, norm_grad_lst, x,  k


#global variables
lamd = 0.5
delta = 0.001

#global variables for backtracking
s=1
sigma=0.5
gamma=0.1  


options = {
    "tol": 1e-3,
    "maxiter": 1e5
}

# wine BFGS
A = wine_data
row = A.shape[0]
col = A.shape[1]
x0 = np.concatenate(list(map(lambda i:A.getcol(i).toarray(), range(col))),axis=0)


T1 = time.time()
temp_wine_1 = BFGS(function, gradient, x0, options)
T2 = time.time()
run_time_wine_1 = (T2-T1)/60 

run_time_wine_1

plot_convergence_figure(function, temp_wine_1[0], temp_wine_1[1], x0)

# wine L-BFGS
T1 = time.time()
temp_wine_2 = LBFGS(function, gradient, x0, options)
T2 = time.time()
run_time_wine_2 = (T2-T1)/60 

run_time_wine_2

plot_convergence_figure(function, temp_wine_2[0], temp_wine_2[1], x0)

# data_200_dot_3 BFGS
A = data_200_dot_3[0:2]
x0 = A.reshape([-1,1],order='F')
row = A.shape[0]
col = A.shape[1]

T1 = time.time()
x0=A.reshape([-1,1],order='F')
temp_200_3 = BFGS(function, gradient, x0, options)
T2 = time.time()
run_time_200_3 = (T2-T1)/60 

run_time_200_3

plot_convergence_figure(function, temp_200_3[0], temp_200_3[1], x0)

X = temp_200_3[2].reshape([-1,2], order='F')
#print(X)
uf = UnionFindSet(X.shape[0])
# clustering
tolerance = 3 # TBD

for i in range(X.shape[0]):
    for j in range(i+1, X.shape[0]):
        # 两两比较距离
        if norm(X[i] - X[j]) < tolerance:
            uf.union(i, j)
# 聚类结果
label_result_200_3 = list()
for i in range(X.shape[0]):
    label_result_200_3.append(uf.find(i)) # 第i条数据的聚类类别

#orginal data
x=data_200_dot_3[0, :]
y=data_200_dot_3[1, :]
c=label_result_200_3

fig, ax = plt.subplots()
scatter = ax.scatter(x, y, c=c, cmap=plt.cm.RdYlBu)
plt.show()


#data_200_dot_6 BFGS
A = data_200_dot_6[0:2]
x0 = A.reshape([-1,1],order='F')
x_dot_200_6 = data_200_dot_6[0]
y_dot_200_6 = data_200_dot_6[1]
row = A.shape[0]
col = A.shape[1]

T1 = time.time()
x0 = A.reshape([-1,1],order='F')
temp_200_6 = BFGS(function, gradient, x0, options)
T2 = time.time()
run_time_200_6 = (T2-T1)/60 

run_time_200_6

plot_convergence_figure(function, temp_200_6[0], temp_200_6[1], x0)


X = temp_200_6[2].reshape([-1,2], order='F')
#print(X)
uf = UnionFindSet(X.shape[0])
# clustering
tolerance = 6 # TBD

for i in range(X.shape[0]):
    for j in range(i+1, X.shape[0]):
        # 两两比较距离
        if norm(X[i] - X[j]) < tolerance:
            uf.union(i, j)
# 聚类结果
label_result_200_6 = list()
for i in range(X.shape[0]):
    label_result_200_6.append(uf.find(i)) # 第i条数据的聚类类别

fig, ax = plt.subplots()
scatter = ax.scatter(x_dot_200_6, y_dot_200_6, c=label_result_200_6, cmap=plt.cm.RdYlBu)
plt.show()