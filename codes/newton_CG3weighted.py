# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:49:35 2021

@author: Evelyn
"""
from scipy.io import loadmat
import numpy as np
from scipy.sparse import csc_matrix
import time

def get_weighted_sparse(k):
    for i in range(col-1):
        w=np.array([np.exp(-v*np.power(np.linalg.norm((A[:, i].reshape(-1,1)-A[:, j].reshape(-1,1)).toarray(),ord=2),2)) for j in range(i+1,col)])
        if k>=col-i-1:
            k=col-i-1
        index=np.argpartition(w,-k)[-k:]
        for j in range(k):
            Weight[(i,index[j]+i+1)]=w[index[j]]
            Weight[(index[j]+i+1,i)]=w[index[j]]
    return Weight

def get_weighted(k):
    for i in range(col-1):
        w=np.array([np.exp(-v*np.power(np.linalg.norm((A[:, i].reshape(-1,1)-A[:, j].reshape(-1,1))),2)) for j in range(i+1,col)])
        if k>=col-i-1:
            k=col-i-1
        index=np.argpartition(w,-k)[-k:]
        for j in range(k):
            Weight[(i,index[j]+i+1)]=w[index[j]]
            Weight[(index[j]+i+1,i)]=w[index[j]]
    return Weight

# --------------------my version(slow version)-------------------------------
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
    for i in range(col):
        for j in range(i+1,col):
            n=np.linalg.norm(x[i*row:(i+1)*row]-x[j*row:(j+1)*row])
            f2+=Weight.get((i,j),0)*0.5/delta*np.power(n,2) if n<=delta else Weight.get((i,j),0)*(n-delta*0.5)
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
        for j in range(col):
            n=np.linalg.norm(x[i*row:(i+1)*row]-x[j*row:(j+1)*row])
            solution2[i*row:(i+1)*row]+=Weight.get((i,j),0)*(x[i*row:(i+1)*row]-x[j*row:(j+1)*row])/delta if n<=delta else Weight.get((i,j),0)*(x[i*row:(i+1)*row]-x[j*row:(j+1)*row])/n    
    return solution1+lamd*solution2

def hessian(x):
    '''

    Parameters
    ----------
    x : numpy.ndarray
        shape: (row*col,1)
        DESCRIPTION: The independent variable of hessian funciton
    Returns
    -------
    type: numpy.ndarray
          shape: (row*col,row*col)
          DESCRIPTION: The solution of hessian function.

    '''
    mat=[]
    hub={}
    for i in range(col):
        hubij=np.zeros([row,row])
        for j in range(col):
            n=np.linalg.norm(x[i*row:(i+1)*row]-x[j*row:(j+1)*row])
            hubij+=Weight.get((i,j),0)*np.identity(row)/delta if n<=delta else Weight.get((i,j),0)*np.identity(row)/n-(x[i*row:(i+1)*row]-x[j*row:(j+1)*row])@(x[i*row:(i+1)*row]-x[j*row:(j+1)*row]).T/np.power(n,3)
        hub[i]=hubij
           
    for i in range(col):
        r=[]
        for j in range(col):
            if i==j:
                r.append(np.identity(row)+lamd*hub[i])
            else:
                n=np.linalg.norm(x[i*row:(i+1)*row]-x[j*row:(j+1)*row])
                hubij=Weight.get((i,j),0)*np.identity(row)/delta if n<=delta else Weight.get((i,j),0)*np.identity(row)/n-(x[i*row:(i+1)*row]-x[j*row:(j+1)*row])@(x[i*row:(i+1)*row]-x[j*row:(j+1)*row]).T/np.power(n,3)                
                r.append(-lamd*hubij)
        mat.append(np.concatenate(r,axis=1))
    return np.concatenate(mat,axis=0)

class Newton_CG:
    def __init__(self,func,grad,hess,x0,tol,maxiter):
        self.func=func 
        self.grad=grad
        self.hess=hess
        self.x0=x0    #initial point
        self.x_sequence=[x0]  #sequence of {x_k}
        self.norm_grad_x_sequence=[np.linalg.norm(self.grad(x0))] #sequence of {||grad_x_k||}
        self.cg_max=10  #the max iterations of CG
        self.epsilon=tol #the tolerance of newton-CG
        self.maxiter=maxiter  #the max iteration of newton-CG

    def backtracking(self,x,d):
        alpha=s
        x_1=x+alpha*d
        while self.func(x_1)>self.func(x)+gamma*alpha*np.dot(self.grad(x).T,d)[0,0]:
            alpha=sigma*alpha
            x_1=x+alpha*d
        return alpha 
    
    def cal_dk(self,xk):
        B=self.hess(xk)
        vj=np.zeros([row*col,1])
        rj=self.grad(xk)
        pj=-rj
        cg_tol=min(1,np.power(np.linalg.norm(rj),0.1))*np.linalg.norm(rj)
        dk=-rj
        for j in range(0,self.cg_max):
            judge=(pj.T@B@pj)[0][0]
            if judge<=0:
                dk=vj if j>0 else dk
                break
            norm_rj=np.linalg.norm(rj) #the norm of old rj 
            sigmaj=np.power(norm_rj,2)/judge
            vj+=sigmaj*pj
            rj+=sigmaj*B@pj #new rj
            norm_rj_1=np.linalg.norm(rj) #the norm of new rj
            if norm_rj_1<cg_tol:
                dk=vj 
                break
            betaj_1=np.power(norm_rj_1,2)/np.power(norm_rj,2)
            pj=-rj+betaj_1*pj
        return dk
                    
    
    def Newton_CG(self):
        xk=self.x0
        print("INIT OBJ:",self.func(x0))
        for k in range(self.maxiter):
            dk=self.cal_dk(xk)
            alphak=self.backtracking(xk,dk)
            xk=xk+alphak*dk
            self.x_sequence.append(xk)
            norm_grad_xk=np.linalg.norm(self.grad(xk))
            self.norm_grad_x_sequence.append(norm_grad_xk)
            print("ITER:{:0>6d}".format(len(self.x_sequence)-1),end='   ')
            print("OBJ:",self.func(xk),end='   ')
            print("NORM:",norm_grad_xk,end='\n')
            if norm_grad_xk<=self.epsilon:
                return xk

#读取数据，只用wine数据进行了测试，其他可自行修改
file='.\wine\wine_data.mat'
dataA=loadmat(file,mat_dtype=True)

file='.\wine\wine_label.mat'
datab=loadmat(file,mat_dtype=True)

A=dataA['A']
b=datab['b']

#设置一些全局变量，可自行修改
lamd=0.05
delta=0.001
row=A.shape[0]
col=A.shape[1]
Weight={}
s=1
sigma=0.5
gamma=0.1
v=0.5

T1 = time.time() #计时

# 测试Newton_CG(Weighted 版本)
# ----------------测试wine
x0=np.concatenate(list(map(lambda i:A.getcol(i).toarray(), range(col))),axis=0)
tol=0.001
delta=1e-4
maxiter=10000

Weight=get_weighted_sparse(5)
temp=Newton_CG(function, gradient, hessian, x0, tol, maxiter)
temp.Newton_CG()
T2 = time.time()
print('程序运行时间:{:.2f}分' .format((T2 - T1)/60))   
np.savez('.\\compare\\wine_weight',temp.x_sequence, temp.norm_grad_x_sequence)

# ---------------测试一个自己建的很小的数据集 
# A=np.array([[0.,0.5,1.,0.5,1.,0.5]]*4)
# col=A.shape[1]
# row=A.shape[0]
# Weighted=get_weighted(3)
# x0=A.reshape([-1,1])
# temp=Newton_CG(function, gradient, hessian, x0, tol, maxiter)
# temp.Newton_CG()


