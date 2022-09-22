# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 14:29:25 2021

@author: Evelyn
"""
#如果要运行程序，请把本文件放到解压的数据文件下

from scipy.io import loadmat
import numpy as np
from scipy.sparse import csc_matrix
import time

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
            f2+=0.5/delta*np.power(n,2) if n<=delta else n-delta*0.5
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
            solution2[i*row:(i+1)*row]+=(x[i*row:(i+1)*row]-x[j*row:(j+1)*row])/delta if n<=delta else (x[i*row:(i+1)*row]-x[j*row:(j+1)*row])/n    
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
            hubij+=np.identity(row)/delta if n<=delta else np.identity(row)/n-(x[i*row:(i+1)*row]-x[j*row:(j+1)*row])@(x[i*row:(i+1)*row]-x[j*row:(j+1)*row]).T/np.power(n,3)
        hub[i]=hubij
           
    for i in range(col):
        r=[]
        for j in range(col):
            if i==j:
                r.append(np.identity(row)+lamd*hub[i])
            else:
                n=np.linalg.norm(x[i*row:(i+1)*row]-x[j*row:(j+1)*row])
                hubij=np.identity(row)/delta if n<=delta else np.identity(row)/n-(x[i*row:(i+1)*row]-x[j*row:(j+1)*row])@(x[i*row:(i+1)*row]-x[j*row:(j+1)*row]).T/np.power(n,3)                
                r.append(-lamd*hubij)
        mat.append(np.concatenate(r,axis=1))
    return np.concatenate(mat,axis=0)

# ---------------------------new version(quick)-------------------------------
def function1(x):
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

def gradient1(x):
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


def hessian1(x):
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
        xi_xj=np.asarray([x[i*row:(i+1)*row]-x[j*row:(j+1)*row] for j in range(col)]).reshape((col,row),order='C')
        m=np.all(np.equal(xi_xj, 0), axis=1) 
        xi_xj=xi_xj[~m] #去掉0项
        xi_xj_norm=np.linalg.norm(xi_xj,axis=1,ord=2)
        mask=xi_xj_norm<=delta
        hubij+=np.sum([(np.identity(row)/delta if judge==True else np.identity(row)/xi_xj_norm[j]-xi_xj[j].reshape(row,1)@xi_xj[j].reshape(1,row)/(xi_xj_norm[j]**3)) for j,judge in enumerate(mask)],axis=0)
        hub[i]=hubij
           
    for i in range(col):
        r=[]
        for j in range(col):
            if i==j:
                r.append(np.identity(row)+lamd*hub[i])
            else:
                n=np.linalg.norm(x[i*row:(i+1)*row]-x[j*row:(j+1)*row])
                hubij=np.identity(row)/delta if n<=delta else np.identity(row)/n-(x[i*row:(i+1)*row]-x[j*row:(j+1)*row])@(x[i*row:(i+1)*row]-x[j*row:(j+1)*row]).T/np.power(n,3)                
                r.append(-lamd*hubij)
        mat.append(np.concatenate(r,axis=1))
    return np.concatenate(mat,axis=0)

# 测试以上函数
# print(gradient(x))
# print(hessian(x))
# print(function(x))
# xk=gradient(x)
# B=hessian(x)
# print((xk.T@B@xk)[0][0])

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
        print("INITIAL OBJ:",self.func(x0))
        for k in range(self.maxiter):
            dk=self.cal_dk(xk)
            alphak=self.backtracking(xk,dk)
            xk=xk+alphak*dk
            self.x_sequence.append(xk)
            norm_grad_xk=np.linalg.norm(self.grad(xk))
            self.norm_grad_x_sequence.append(norm_grad_xk)
            print("ITER:{:0>6d}".format(len(self.x_sequence)-1),end='   ')
            # print("OBJ:",self.func(xk),end='   ')
            print("NORM:",norm_grad_xk,end='\n')
            if norm_grad_xk<=self.epsilon:
                return xk

T1 = time.time() #计时

# 测试Newton_CG

#设置一些全局变量，可自行修改
lamd=0.5
delta=0.001
s=1
sigma=0.5
gamma=0.1
maxiter=10000

#稀疏矩阵
file='.\wine\wine_data.mat'
dataA=loadmat(file,mat_dtype=True)
file='.\wine\wine_label.mat'
datab=loadmat(file,mat_dtype=True)
A=dataA['A']
b=datab['b']
row=A.shape[0]
col=A.shape[1]
x0=np.concatenate(list(map(lambda i:A.getcol(i).toarray(), range(col))),axis=0)
tol=0.001
maxiter=10000

# ----------------测试wine
lamda=0.05
delta=1e-4
temp=Newton_CG(function, gradient, hessian, x0, tol, maxiter)
temp.Newton_CG()       
T2 = time.time()
print('程序运行时间:{:.2f}分' .format((T2 - T1)/60))   
np.savez('.\\compare\\wine_no_weight',temp.x_sequence, temp.norm_grad_x_sequence)
  
# ---------------测试一个自己建的很小的数据集      
# A=np.array([[0.,0.5,1.,0.5,1.,0.5]]*4)
# col=A.shape[1]
# row=A.shape[0]
# x0=A.reshape([-1,1],order='F')
# temp=Newton_CG(function, gradient, hessian, x0, tol, maxiter)
# temp.Newton_CG()
        
# --------------测试人工生成数据集
# data=np.loadtxt('.\\op_data\\200_circle_2.txt')
# A=data[0:2]
# x0=A.reshape([-1,1],order='F')
# row=A.shape[0]
# col=A.shape[1]
# cir_200=Newton_CG(function, gradient, hessian, x0, tol, maxiter)
# x_star=cir_200.Newton_CG()
# T2 = time.time()
# print('程序运行时间:{:.2f}分' .format((T2 - T1)/60)) 
# np.savez('cir_200',cir_200.x_sequence,cir_200.norm_grad_x_sequence)

# data=np.loadtxt('.\\op_data\\200_dot_3.txt')
# A=data[0:2]
# x0=A.reshape([-1,1],order='F')
# row=A.shape[0]
# col=A.shape[1]
# dot_200=Newton_CG(function, gradient, hessian, x0, tol, maxiter)
# dot_200.Newton_CG()
# T2 = time.time()
# print('程序运行时间:{:.2f}分' .format((T2 - T1)/60)) 
# np.savez('dot_200',dot_200.x_sequence,dot_200.norm_grad_x_sequence)

# T1 = time.time()
# data=np.loadtxt('.\\op_data\\200_moon_2.txt')
# A=data[0:2]
# x0=A.reshape([-1,1],order='F')
# row=A.shape[0]
# col=A.shape[1]
# moon_200=Newton_CG(function, gradient, hessian, x0, tol, maxiter)
# moon_200.Newton_CG()
# T2 = time.time()
# print('程序运行时间:{:.2f}分' .format((T2 - T1)/60)) 
# np.savez('moon_200',moon_200.x_sequence,moon_200.norm_grad_x_sequence)          

# T1 = time.time()
# data=np.loadtxt('.\\op_data\\200_random_4.txt')
# A=data[0:2]
# x0=A.reshape([-1,1],order='F')
# row=A.shape[0]
# col=A.shape[1]
# random_200=Newton_CG(function, gradient, hessian, x0, tol, maxiter)
# random_200.Newton_CG()
# T2 = time.time()
# print('程序运行时间:{:.2f}分' .format((T2 - T1)/60)) 
# np.savez('random_200',random_200.x_sequence,random_200.norm_grad_x_sequence)          
      

# T1 = time.time() 
# data=np.loadtxt('.\\op_data\\200_dot_6.txt')
# A=data[0:2]
# x0=A.reshape([-1,1],order='F')
# row=A.shape[0]
# col=A.shape[1]
# dot_200_6=Newton_CG(function, gradient, hessian, x0, tol, maxiter)
# dot_200_6.Newton_CG()
# T2 = time.time()
# print('程序运行时间:{:.2f}分' .format((T2 - T1)/60)) 
# np.savez('dot_200_6',dot_200_6.x_sequence, dot_200_6.norm_grad_x_sequence) 

# T1 = time.time() 
# lamda_list=[0.05,0.1,0.2,0.5,0.8,1,3,5,7.5,10.]
# data=np.loadtxt('.\\op_data\\200_dot_6.txt')
# A=data[0:2]
# x0=A.reshape([-1,1],order='F')
# row=A.shape[0]
# col=A.shape[1]
# tol=0.1
# for i in lamda_list:
#     lamd=i    
#     dot_2000=Newton_CG(function, gradient, hessian, x0, tol, maxiter)
#     dot_2000.Newton_CG()
#     np.savez('dot_200_6_'+str(i),dot_2000.x_sequence,dot_2000.norm_grad_x_sequence)
# T2 = time.time()
# print('程序运行时间:{:.2f}分' .format((T2 - T1)/60)) 
# np.savez('dot_2000',dot_2000.x_sequence,dot_2000.norm_grad_x_sequence)

# ------------------------测试真实数据
# file='.\\vowel\\vowel_data.mat'
# dataA=loadmat(file,mat_dtype=True)
# A=dataA['A']
# row=A.shape[0]
# col=A.shape[1]
# x0=np.concatenate(list(map(lambda i:A.getcol(i).toarray(), range(col))),axis=0)
# tol=0.001
# vowel=Newton_CG(function, gradient, hessian, x0, tol, maxiter)
# vowel.Newton_CG()       
# T2 = time.time()
# print('程序运行时间:{:.2f}分' .format((T2 - T1)/60)) 
# np.savez('vowel',vowel.x_sequence, vowel.norm_grad_x_sequence)
