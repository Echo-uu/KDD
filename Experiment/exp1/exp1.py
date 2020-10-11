import math
import random
import numpy as np
import pandas as pd
from sympy import symbols, diff
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt

x, y = load_svmlight_file("F:/data/Experiment/exp1/housing_scale")

X = x.todense()#transfer sparse matrix to dense matrix

#adding a column(1) ahead of data 
one = np.mat(np.ones((506,1)))
X = np.c_[one, X]

#split the dataset
x_train, x_test, y_train, y_test = train_test_split(X , y, test_size = 0.25, shuffle= False)

y_train = (np.mat(y_train)).T
y_test = (np.mat(y_test)).T

#求参数w
def calW(x, y):
    return (x.T * x).I * (x.T * y)

#损失函数
def loss(x, y, w):
    return ((y - x * w).T * (y - x * w)) / 2

#求闭式解
def closeForm():
    w = calW(x_train, y_train)
    loss_train = loss(x_train, y_train, w)
    loss_val = loss(x_test, y_test, w)
    print('loss train: ', loss_train[0,0]/127)
    print('loss val: ', loss_val[0,0]/127, '\n')



#求导
def derivation(x, y, w):
    #设置14个参数匹配13个属性w_i和一个常数量b
    x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14 = symbols('x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14')
    para =np.mat([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14])   
    para = para.T

    #loss函数
    los = loss(x, y, para)
    los = los[0,0]
    #print(loss)

    #对每个属性w_i求偏导后代入随机初始化的每个w_i值求出梯度
    G = diff(los, para).subs(\
        {x1:w[0][0,0],x2:w[1][0,0],x3:w[2][0,0],x4:w[3][0,0],x5:w[4][0,0],\
            x6:w[5][0,0],x7:w[6][0,0],x8:w[7][0,0],x9:w[8][0,0],x10:w[9][0,0],\
                x11:w[10][0,0],x12:w[11][0,0],x13:w[12][0,0],x14:w[13][0,0]})
    return G

def SGD(x, y, t_x, t_y):
    w = np.mat(np.random.rand(14,1))
    
    #设置初始学习率为0.1
    l = 0.1

    ls = np.arange(10)

    print('\t\t*****Stochastic Gradient Descent begin*****')
    for i in range(10):        
        rnd = random.randint(0,400)
        #求导
        G = derivation(x[rnd], y[rnd], w)

        #取梯度G的负方向，记为D。
        D = -G
        #更新模型参数
        w_n = w + l * D 
        #输出loss_train和loss_val的值。
        # print(i+1,'times : ')
        # print('loss_train:',loss(x, y, w_n)[0,0])
        # print('loss_val: ', loss(t_x, t_y, w_n)[0,0])
        ls[i]=(loss(t_x, t_y, w_n)[0,0])/127
        #迭代模型参数，降低学习率
        w = w_n
        l *= (0.95 ** (i+1))
    
    plt.plot(range(10), ls)
    plt.title('loss_val')n
    plt.show()

closeForm()

SGD(x_train, y_train, x_test, y_test)
